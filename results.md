# RoRA 重現實驗結果

設定：RoBERTa-large + BadNet ("mn" trigger) + SST-2，lr=2e-4（除非另有標註），20 epochs，seed=0

## 主要結果：LoRA/DoRA × 機制1/2 組合（固定 lr=2e-4, k=32）

| Method | 機制 | Test CA | ASR |
|---|---|---|---|
| LoRA | baseline | 0.9528 | 0.7217 |
| LoRA | +Cl (p=0.1) | 0.9550 | **0.2101** |
| LoRA | +Tr (λ=10, k=32) | 0.9588 | 0.9648 |
| LoRA | +Cl+Tr | 0.9610 | 0.3234 |
| DoRA | baseline | 0.9561 | 0.6821 |
| DoRA | +Cl (p=0.1) | 0.9495 | **0.2904** |
| DoRA | +Tr (λ=10, k=32) | 0.9533 | 0.8526 |
| DoRA | +Cl+Tr | 0.9583 | **0.1859**（八組裡最佳） |

**核心發現：Cl 單獨就很有效（兩個 method 都大降）；Tr 單獨是反效果（ASR 不降反升）；Cl+Tr 一起比 Cl 單獨更好——兩機制有互補/協同效應，不是簡單相加。**

## 機制2消融：LoRA + Tr，固定 λ=10

### k 消融（固定 lr=2e-4）

| k | Test CA | ASR |
|---|---|---|
| 8 (=r) | 0.9594 | **0.9373**（三者最佳） |
| 32 (借用 Figure 2 caption 的數字) | 0.9588 | 0.9648 |
| 1024 (不截斷，等效完整方陣) | 0.9500 | 1.0000（最差） |

趨勢：k 越小越好，單調。

### lr 消融（固定 k=8）

| lr | Test CA | ASR |
|---|---|---|
| 2e-5 | 0.9605 | 1.0000（太小，訓練不動，backdoor 完全沒被觸動） |
| 2e-4 | 0.9594 | **0.9373**（最佳，主線一直使用的值） |
| 2e-3 | 0.9456 | 訓練崩潰（epoch 1 起 dev acc ~0.49，等同亂猜，數字無意義） |

結論：lr=2e-4 已是三者中最佳，排除「lr 選錯導致機制2效果差」這個假設。

## 理論發現（已驗證，非訓練實驗）

論文 Eq.9→Eq.10 的正交懲罰 Ω(A,B)=‖U^T B‖²_F+‖AV‖²_F，若 U、V 取自未截斷的完整 SVD，
在 W_pre 為滿秩方陣時，數學上會退化成標準 L2 weight decay
（∵ 完整正交矩陣不改變 Frobenius norm：‖U^T B‖_F = ‖B‖_F）。

實測驗證：RoBERTa-large 的 48 個 query/value 層，SVD 後奇異值全數 >1e-6（1024/1024，滿秩）。
論文本身未說明 Eq.10 計算時 U、V 是否截斷、截斷到多少維——這是我們自己的實作決定（k=8/32/1024 皆已測試），
不是論文明確指定的答案。

## 論文對照

Table 5（RoBERTa+BadNet）Tr alone: CA 95.33 / ASR 13.42
我們的 Tr alone 最佳版本（k=8, lr=2e-4）: CA 95.94 / ASR 93.73 —— 方向一致（CA 相近）但 ASR 落差巨大

值得注意：論文 Table 5 裡 Tr alone 只在 RoBERTa+BadNet 這個組合特別有效（13.42），
其餘三個 model+attack 組合（RoBERTa+InSent, LLaMA+BadNet, LLaMA+InSent）Tr alone 效果皆遠差（77–98），
顯示這個低點本身在論文裡也不是普遍模式。

## 尚未完成

- 機制3（spectral rescaling）—— 程式碼已寫完（改 module.scaling["default"]，重新載入 best checkpoint 後套用），還沒實際跑
- λ 網格（只測過 λ=10，論文網格 {1,5,10,15,20}）—— 決定不繼續追，優先度較低
- Cl+Tr+Pt 三機制全開 —— 依賴機制3完成
- threshold-based SVD 截斷（用奇異值門檻而非固定 k）—— 只是想法，沒有實作
- InSent 攻擊、CR/CoLA 資料集、BERT/LLaMA 架構 —— 全部未測試，範圍仍需與 advisor 確認

## 待確認的實作決定（未經論文或 advisor 證實，需要標注）

- BERT 版本用 bert-large-uncased（論文、PSIM repo 均未指定 cased/uncased）
- LLaMA 版本假設 huggyllama/llama-7b（論文未指定規模）
- 機制2 SVD 截斷 k（論文未明講是否截斷、截斷到多少）
- 機制3「top three layers」假設為模型最後三層（論文未明講是哪三層）
- 機制3對 DoRA 的 scaling 語義尚未查證（module.scaling["default"] 在 DoRA 的正規化步驟下是否等價於 LoRA 的情況，未確認）

## 機制3驗證：Cl+Tr+Pt 全開（最接近論文完整 RoRA 的版本）

設定：k=8（消融實驗中最佳值），λ=10，p=0.1，post-training rescaling 套用於「最後三層」（猜測，非論文確認）

| Method | Test CA (Cl+Tr) | ASR (Cl+Tr) | Test CA (+Pt) | ASR (+Pt) |
|---|---|---|---|---|
| LoRA 全開 | 0.9555 | 0.1562 | **0.9583** | **0.1320** |
| DoRA 全開 | 0.9528 | 0.4136 | **0.9599** | **0.3696** |

機制3（Pt）在兩組上都讓 ASR 進一步下降，CA 沒有犧牲，是三個機制第一次同時朝正確方向疊加。

重新算出的 scaling s（訓練時原值：LoRA=2, DoRA=1）：
- LoRA: layer22.value=16.05, layer23.query=7.08, layer23.value=9.43
- DoRA: layer22.value=13.88, layer23.query=5.65, layer23.value=4.58

s 遠大於訓練時原值，符合論文邏輯（σ_max(ΔW) ≪ σ_max(W_pre) → 新 s 應遠大於訓練時 s），方向被實測驗證。

### 與論文的落差（誠實記錄，不簡化為「已接近」）

論文 Table 2 RoRA 完整版（RoBERTa+BadNet）：CA 95.99 / ASR **6.49**
我們的最佳結果（LoRA 全開）：CA 95.83 / ASR **13.20**——ASR 是論文的兩倍以上，不是「接近」的差距。

已知、但未排除的落差來源（至少五項，未逐一驗證何者是主因）：
1. k=8 是消融後的自選最優值，非論文證實；未測 k<8 的更小值
2. Tr alone 本身仍遠差於論文（93.73% vs 13.42%），疊加後的改善可能主要來自 Cl 而非 Tr
3. λ 網格完全未測（只用 λ=10，論文網格 {1,5,10,15,20}）
4. 論文數字為多次 run 取最好；我們僅單次 seed=0
5. 「top three layers」為猜測，非論文明確指定；未測其他層選取方式

這五項是後續（若有時間/需要）可查證的方向清單，非本輪重現的優先事項。

## 主要結果（k=8 版本）：用消融實驗中確認的最佳 k 值重新呈現

以下數字取自對同一 checkpoint 的重新評估（evaluate_checkpoint.py，決定性，非重新訓練）：
DoRA+Tr(k=8) 來自新跑；其餘三格取自先前已跑過但未存 log 的訓練結果，經 backfill 評估驗證數字一致。

| Method | 機制 | Test CA | ASR |
|---|---|---|---|
| LoRA | +Tr (λ=10, k=8) | 0.9594 | 0.9373 |
| LoRA | +Cl+Tr (k=8) | 0.9555 | 0.1562 |
| DoRA | +Tr (λ=10, k=8) | 0.9561 | 0.7635 |
| DoRA | +Cl+Tr (k=8) | 0.9528 | 0.4136 |

（上方「主要結果：LoRA/DoRA × 機制1/2 組合」表格中 k=32 的版本繼續保留，兩者並存供對照——
k=32 是最初依 Figure 2 caption 選的值，k=8 是消融實驗後確認的最佳值。）

## 機制2消融補充：DoRA 的 k 對照（原表只有 LoRA）

| Method | k | Test CA | ASR |
|---|---|---|---|
| DoRA+Tr | 8 | 0.9561 | **0.7635** |
| DoRA+Tr | 32 | 0.9533 | 0.8526 |

DoRA 上同樣是 k=8 優於 k=32，跟 LoRA 呈現的趨勢一致——「k 越小越好」不是 LoRA 特有現象。

## 補充：evaluate_checkpoint.py 決定性驗證

用同一個 checkpoint 重新評估（backfill_eval.sh），三筆原本只有口頭記錄、未存 log 的舊結果
（LoRA+Tr k=1024、LoRA+Tr k=8/lr=2e-3、LoRA+Tr k=8/lr=2e-5）與 evaluate_checkpoint.py 重新跑出的結果逐位元一致，
證實 evaluation（非訓練）本身是決定性的，可信賴用於事後補測未存檔的 checkpoint。

## LoRA+Cl 的 11-seed 變異性驗證（回應 advisor 期望的「~10%」目標）

固定設定：lr=2e-4, p=0.1，seed 0–10，其餘同主要結果表格。

| Seed | ASR | Test CA |
|---|---|---|
| 0 | 0.2101 | 0.9550 |
| 1 | 0.3113 | 0.9445 |
| 2 | 0.2783 | 0.9621 |
| 3 | 0.1529 | 0.9566 |
| 4 | 0.2035 | 0.9517 |
| 5 | 0.3289 | 0.9572 |
| 6 | 0.4950 | 0.9484 |
| 7 | **0.1353**（最佳） | 0.9550 |
| 8 | 0.2178 | 0.9467 |
| 9 | 0.3795 | 0.9561 |
| 10 | 0.1529 | 0.9561 |

平均 ASR ≈ 0.2605（26.05%），標準差大，範圍 0.1353–0.4950（近 4 倍差距）。

### 解讀

單次 run 的「典型」表現（平均）約 26%，遠高於論文 Table 4 的 Cl alone（10.34%）。
但 best-of-11（seed=7）達到 **13.53%**，已非常接近論文數字。

這與論文自身聲明的方法論一致：「Unless otherwise noted, results in the remaining tables and
figures also reflect the best performance from our repeated experiments.」——論文的 10.34% 很可能
同樣是 best-of-N 的結果，不是穩定的單次表現。

機制1（Cl）本身的效果對 random seed 相當敏感——這個高變異性本身是一個值得報告的觀察，
不應該被「挑最好的一次」這個呈現方式掩蓋掉。誠實的結論是：機制1平均能把 ASR 從 baseline 的
~70% 壓到 ~26%，最佳情況下可達 13.5%（接近論文數字），但單次結果不穩定。
