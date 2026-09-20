# check_trigger_spectral_alignment_v2_1.py
"""
測試：trigger token 的 embedding 跟 W_pre 的右奇異向量 (V) 之間的 cosine similarity，
是否集中在特定的奇異值排名區間（band-pass 假設），還是整個 spectrum 都是 diffuse。
v2: 用真實 forward pass 逐層抓 self-attention 實際吃到的 hidden state，
而不是重複套用同一個原始 embedding；並把 trigger 插入真實句子中，
對多個句子取平均，降低單一句子造成的噪訊。
v2.1: 
1. 結合真實的 forward pass 抓取逐層 hidden state。
2. 支援從 JSON/JSONL 讀取真實 SST-2 測試句子。
3. 完全對齊 attack_utils.py 的 insert_trigger 邏輯，並精準定位字元區間。
v3: 新增貢獻。
v4: 新增全局貢獻 Rank 分佈統計，量化 Top-k 的集中度。
"""

import torch
import random
import json
import os
from collections import Counter
from transformers import RobertaModel, RobertaTokenizerFast

MODEL_PATHS = [
    "./poisoned_roberta_large_badnet",
    "./poisoned_roberta_large_insent",
    "./clean_roberta_large",
]
BASE_MODEL = "roberta-large"
LAYERS_TO_CHECK = [0, 3, 6, 9, 12, 15, 18, 21, 23]
MODULES_TO_CHECK = ["query", "value"]

TRIGGERS = {
    "badnet": "mn",
    "badnet_control_negative": "abysmal", 
    "badnet_control_positive": "perfect",
    "insent": "I watched this 3D movie",
    "insent_control_negative": "This movie is abysmal", 
    "insent_control_positive": "This movie is perfect",
    "insent_control_partial_1": "3D movie",
    "insent_control_partial_2": "movie",
    "insent_control_partial_3": "watched",
    "control_1": "the",
    "control_2": "When transplanting seedlings, candied teapots will make the task easier."
}

# 測試設定
NUM_SAMPLE_SENTENCES = 20   # 每個 trigger 取多少個句子做平均（建議正式測試可調高至 50-100）
SEED = 42

# JSON 測試集路徑
SAMPLE_SENTENCES_SOURCE = "./data/imdb/test.json" 


def load_base_sentences(path, n, seed):
    """支援讀取 JSON Array 或 JSON Lines 格式，並提取 'sentence' 欄位"""
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        try:
            # 嘗試作為完整的 JSON Array 讀取
            data = json.load(f)
            if isinstance(data, dict):
                # 處理 HuggingFace dataset dict 結構 (例如 {"train": [...], "test": [...]})
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and "sentence" in v[0]:
                        data = v
                        break
        except json.JSONDecodeError:
            # 如果失敗，嘗試作為 JSON Lines 讀取
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    # 對齊 build_poisoned_test_dataloader：只用 label == 1 的句子，
    # 因為訓練時 trigger 只被插入過這種語境，這是唯一模型實際見過的情境
    sentences = [item["sentence"] for item in data
                 if "sentence" in item and item.get("label") == 1]

    if not sentences:
        raise ValueError(f"從 {path} 中找不到 label==1 的 'sentence'，請檢查檔案格式！")

    random.seed(seed)
    return random.sample(sentences, min(n, len(sentences)))


def insert_trigger_and_get_span(text, trigger, seed_offset):
    """
    完全對齊 attack_utils.py 的邏輯，並同步計算插入的字元區間。
    回傳新句子與插入片段的字元區間 (char_start, char_end)。
    """
    rng = random.Random(seed_offset)
    trigger_words = trigger.split() if isinstance(trigger, str) else trigger
    words = text.split()
    
    # 與 attack_utils.py 保持完全一致的插入邏輯
    if len(words) <= 1:
        insert_idx = 0
    else:
        insert_idx = rng.randint(1, len(words) - 1)
        
    # 定位 char_start
    prefix = ' '.join(words[:insert_idx])
    char_start = len(prefix) + (1 if prefix else 0)
    
    # 合成新句子
    new_words = words[:insert_idx] + trigger_words + words[insert_idx:]
    new_text = ' '.join(new_words)
    
    # 定位 char_end
    trigger_str = ' '.join(trigger_words)
    char_end = char_start + len(trigger_str)

    return new_text, char_start, char_end


def get_trigger_token_span(offset_mapping, char_start, char_end):
    """用 tokenizer 的 offset_mapping 找出跟 [char_start, char_end) 有重疊的 token 索引。"""
    positions = []
    for i, (s, e) in enumerate(offset_mapping):
        if s == e:  # 跳過 special token
            continue
        if s < char_end and e > char_start:
            positions.append(i)
    return positions


def get_layer_input_hidden_states(model, layers_to_check, input_ids, attention_mask, device):
    """
    用 forward pre-hook 抓每個指定層 self-attention 實際吃到的 hidden_states。
    """
    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs):
            captured[layer_idx] = inputs[0].detach()
        return hook

    for layer_idx in layers_to_check:
        attn_self = model.encoder.layer[layer_idx].attention.self
        hooks.append(attn_self.register_forward_pre_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    return captured  # {layer_idx: tensor[1, seq_len, hidden_dim]}


def analyze_layer_module_contribution(v_units, S, hidden_vec):
    """
    回傳這一個樣本的完整的、有正負號的 cos_sim 向量，以及其「真實貢獻度」向量 (sigma * (v^T h))
    """
    # 這裡可以不 norm hidden_vec，因為我們要看絕對的 magnitude 貢獻
    # 但為了跨樣本公平比較，我們先對 hidden_vec 進行 l2-norm，再看純矩陣造成的貢獻分配
    hidden_unit = hidden_vec / hidden_vec.norm()
    
    # 計算投影 (v^T h)
    cos_sims = v_units @ hidden_unit  # <--- 拿掉 .abs()，保留真實方向
    
    # 乘上奇異值 (sigma_i * v_i^T h)
    contributions = cos_sims * S 
    return cos_sims, contributions  # shape: [num_singular_vectors]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizerFast.from_pretrained(BASE_MODEL)

    print("Loading base model architecture...")
    model = RobertaModel.from_pretrained(BASE_MODEL)
    model.to(device)
    model.eval()

    # 讀取真實句子
    print(f"Loading {NUM_SAMPLE_SENTENCES} sample sentences from JSON...")
    try:
        base_sentences = load_base_sentences(SAMPLE_SENTENCES_SOURCE, NUM_SAMPLE_SENTENCES, SEED)
    except Exception as e:
        print(f"Error loading sentences: {e}")
        return

    results = []
    distribution_stats = {} # 記錄全局分佈

    for model_path in MODEL_PATHS:
        model_name = os.path.basename(os.path.normpath(model_path))
        print(f"\n{'='*60}\nEvaluating Model: {model_name}\n{'='*60}")

        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(bin_path):
            print(f"Warning: {bin_path} not found. Skipping...")
            continue
            
        state_dict = torch.load(bin_path, map_location="cpu")
        state_dict = {k.replace("roberta.", "", 1) if k.startswith("roberta.") else k: v
                      for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        # 預先算好 SVD，加速運算
        svd_cache = {}
        for layer_idx in LAYERS_TO_CHECK:
            attn = model.encoder.layer[layer_idx].attention.self
            for module_name in MODULES_TO_CHECK:
                weight_matrix = getattr(attn, module_name).weight.detach()
                _, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False) # 抓出 S
                v_units = Vh / Vh.norm(dim=1, keepdim=True)   # 存正規化後的右奇異向量
                svd_cache[(layer_idx, module_name)] = {"v_units": v_units, "S": S} # 存入 Dict

        # 用於收集整個 model 下的所有 Contribution Ranks
        rank_pools = {"real_trigger": [], "control": []}

        for trigger_name, trigger_text in TRIGGERS.items():
            group = "control" if "control" in trigger_name else "real_trigger"

            accum_signed_cos_sims = {}
            accum_abs_cos_sims = {}
            accum_signed_contributions = {}
            accum_abs_contributions = {}
            valid_samples = 0

            for sample_idx, base_text in enumerate(base_sentences):
                new_text, char_start, char_end = insert_trigger_and_get_span(
                    base_text, trigger_text, seed_offset=SEED + sample_idx
                )
                
                encoded = tokenizer(
                    new_text, return_tensors="pt", return_offsets_mapping=True
                )
                offset_mapping = encoded.pop("offset_mapping")[0].tolist()
                trigger_positions = get_trigger_token_span(offset_mapping, char_start, char_end)
                
                if not trigger_positions:
                    continue  

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                if sample_idx == 0:
                    decoded_trigger = tokenizer.decode(
                        input_ids[0, trigger_positions]
                    )
                    print(f"  [sanity check] trigger='{trigger_text}' -> "
                          f"decoded tokens='{decoded_trigger}'")

                captured = get_layer_input_hidden_states(
                    model, LAYERS_TO_CHECK, input_ids, attention_mask, device
                )

                for layer_idx in LAYERS_TO_CHECK:
                    hidden = captured[layer_idx][0]  # [seq_len, hidden_dim]
                    # 若 trigger 切成多個 token，針對這些 position 的 hidden state 取平均
                    trigger_vec = hidden[trigger_positions, :].mean(dim=0)

                    for module_name in MODULES_TO_CHECK:
                        svd_data = svd_cache[(layer_idx, module_name)]
                        v_units = svd_data["v_units"]
                        S = svd_data["S"]

                        # 算出綜合 Sigma 放大率的貢獻
                        cos_sims, contributions = analyze_layer_module_contribution(v_units, S, trigger_vec)  # 有號

                        key = (layer_idx, module_name)
                        if key not in accum_signed_cos_sims:
                            accum_signed_cos_sims[key] = cos_sims.clone()
                            accum_abs_cos_sims[key] = cos_sims.abs().clone()
                            accum_signed_contributions[key] = contributions.clone()
                            accum_abs_contributions[key] = contributions.abs().clone()
                        else:
                            accum_signed_cos_sims[key] += cos_sims
                            accum_abs_cos_sims[key] += cos_sims.abs()
                            accum_signed_contributions[key] += contributions
                            accum_abs_contributions[key] += contributions.abs()
                            
                valid_samples += 1

            if valid_samples == 0:
                print(f"[{model_name}] [trigger={trigger_name}] no valid samples, skipped")
                continue

            for (layer_idx, module_name), summed_signed_cos_sims in accum_signed_cos_sims.items():
                avg_signed_cos_sims = summed_signed_cos_sims / valid_samples
                avg_abs_cos_sims = accum_abs_cos_sims[(layer_idx, module_name)] / valid_samples
                avg_signed_contributions = accum_signed_contributions[(layer_idx, module_name)] / valid_samples
                avg_abs_contributions = accum_abs_contributions[(layer_idx, module_name)] / valid_samples

                _, top_ranks_cos_sims = torch.topk(avg_abs_cos_sims, k=5)  # 用「平均絕對值」選 top-k，跟 v2.1 一致，才能比較
                top_signed_at_ranks_cos_sims = avg_signed_cos_sims[top_ranks_cos_sims]
                top_abs_at_ranks_cos_sims = avg_abs_cos_sims[top_ranks_cos_sims]

                _, top_ranks_contributions = torch.topk(avg_abs_contributions, k=5)  # 用「平均絕對值」選 top-k，跟 v2.1 一致，才能比較
                top_signed_at_ranks_contributions = avg_signed_contributions[top_ranks_contributions]
                top_abs_at_ranks_contributions = avg_abs_contributions[top_ranks_contributions]

                # 收集到 Model 級別的清單中
                rank_pools[group].extend(top_ranks_contributions.tolist())

                # sign_consistency 越接近 ±1，代表這個 rank 在 20 個樣本裡符號幾乎都一致；
                # 越接近 0，代表符號在樣本間隨機翻轉、互相抵消
                sign_consistency_cos_sims = (top_signed_at_ranks_cos_sims / top_abs_at_ranks_cos_sims).tolist()
                sign_consistency_contributions = (top_signed_at_ranks_contributions / top_abs_at_ranks_contributions).tolist()

                # PRINT1. 先將要印出的 List 轉換成格式化好的字串
                c_ranks = str(top_ranks_cos_sims.tolist())
                c_signed = str([round(s,4) for s in top_signed_at_ranks_cos_sims.tolist()])
                c_abs = str([round(s,4) for s in top_abs_at_ranks_cos_sims.tolist()])
                c_sync = str([round(s,2) for s in sign_consistency_cos_sims])
                c_mean = f"{avg_abs_cos_sims.mean().item():.4f}"

                t_ranks = str(top_ranks_contributions.tolist())
                t_signed = str([round(s,4) for s in top_signed_at_ranks_contributions.tolist()])
                t_abs = str([round(s,4) for s in top_abs_at_ranks_contributions.tolist()])
                t_sync = str([round(s,2) for s in sign_consistency_contributions])
                t_mean = f"{avg_abs_contributions.mean().item():.4f}"

                # PRINT2. 用 f-string 的 :<寬度 語法來強制對齊 (數字代表預留的字元寬度)
                print(f"[{model_name:<31}] [trigger={trigger_name:<25}] [layer={layer_idx:>2}] [module={module_name:<5}] (n={valid_samples:>2}) | "
                      f"[cos_sims] ranks={c_ranks:<30} "
                      f"signed={c_signed:<45} "
                      f"abs={c_abs:<40} "
                      f"sign_c={c_sync:<35} "
                      f"mean_abs={c_mean:<6} | "
                      f"[contribs] ranks={t_ranks:<30} "
                      f"signed={t_signed:<45} "
                      f"abs={t_abs:<40} "
                      f"sign_c={t_sync:<35} "
                      f"mean_abs={t_mean:<6}")

                results.append({
                    "model": model_name,
                    "trigger": trigger_name,
                    "layer": layer_idx,
                    "module": module_name,
                    "num_samples": valid_samples,
                    "top_ranks_cos_sims": top_ranks_cos_sims.tolist(),
                    "top5_signed_cos_sims": top_signed_at_ranks_cos_sims.tolist(), # 存入帶有正負號的值
                    "top5_abs_cos_sims": top_abs_at_ranks_cos_sims.tolist(),
                    "sign_consistency_cos_sims": sign_consistency_cos_sims,
                    "mean_abs_cos_sims": avg_abs_cos_sims.mean().item(),
                    "top_ranks_contributions": top_ranks_contributions.tolist(),
                    "top5_signed_contributions": top_signed_at_ranks_contributions.tolist(), # 存入帶有正負號的值
                    "top5_abs_contributions": top_abs_at_ranks_contributions.tolist(),
                    "sign_consistency_contributions": sign_consistency_contributions,
                    "mean_abs_contributions": avg_abs_contributions.mean().item(),
                })

        # --- 模型層級的全局分佈統計 ---
        for group_name, ranks in rank_pools.items():
            if not ranks:
                continue
            c = Counter(ranks)
            total = len(ranks)
            min_rank = min(ranks)
            max_rank = max(ranks)
            print(f"\n--- [{model_name}] [{group_name}] Global Rank Distribution (n={total}) ---")
            print(f"Total Top-5 Ranks Collected: {total}")
            print(f"Overall Range: Min Rank = {min_rank}, Max Rank = {max_rank}")
            print(f"Most Frequent Ranks: {c.most_common(10)}")
            
            group_stats = {
                "total": total, 
                "min_rank": min_rank,
                "max_rank": max_rank,
                "most_common": c.most_common(20),
            }
            
            for k in (8, 32, 64, 128, 256, 512, 1024):
                # 取出所有落入 Top-k 區間的 ranks
                subset = [r for r in ranks if r < k]
                cnt = len(subset)
                
                if cnt > 0:
                    subset_min = min(subset)
                    subset_max = max(subset)
                    print(f"  top-{k:<3}: {cnt:>4}/{total} ({cnt/total:>6.2%}) | subset range: [{subset_min}, {subset_max}]")
                else:
                    subset_min = None
                    subset_max = None
                    print(f"  top-{k:<3}: {cnt:>4}/{total} ({0:>6.2%}) | subset range: [N/A, N/A]")
                
                # 同步存入 JSON stats
                group_stats[f"percent_in_top_{k}"] = cnt / total if total > 0 else 0
                group_stats[f"top_{k}_min_rank"] = subset_min
                group_stats[f"top_{k}_max_rank"] = subset_max
                
            distribution_stats.setdefault(model_name, {})[group_name] = group_stats

    # 把分佈統計也寫進 JSON
    final_output = {
        "detail_results": results,
        "global_distribution_stats": distribution_stats
    }

    with open("spectral_alignment_results_v4_hidden_states.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("\nSaved to spectral_alignment_results_v4_hidden_states.json")


if __name__ == "__main__":
    main()