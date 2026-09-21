# RoRA 重現實驗結果

設定：RoBERTa-large + BadNet ("mn" trigger) / InSent ("I watched this 3D movie") + SST-2，
lr=2e-4（除非另有標註），20 epochs，seed=0（除多 seed 章節）。
**總覽對照表採用 k=8。此值最初被視為基準，但擴大驗證後發現 k=8 本身在多 seed 下同樣高變異
（見「機制2消融」）——目前沒有任何測過的 k 值是穩定基準，k=8 只是最早、被最多其他消融
共用的參照點，非「已驗證穩定」的選擇。**

Poisoning 驗收（ASR>95% 門檻，論文要求）：
- BadNet: dev clean acc 99.10%, ASR 100.00%
- InSent: dev clean acc 98.80%, ASR 100.00%

---

## Notation

**論文沿用的符號（Proposition 4.2, Section 4.2）：**
- $\rho_{bd}$：backdoor alignment，trigger 方向與 backdoor 決策邊界的對齊程度
- $\rho_{eff}=\rho_{cl}-\rho_{tr}$：effective clean alignment
- $c$（在 $\langle c,\cdot\rangle$ 中）：分類 margin 方向向量，$c=e_y-e_{y_{bd}}$
- $c$（在 $\|\cdot\|_c$ 中）：逐欄位（column-wise）矩陣範數的下標——**與上面的 $c$ 是不同符號，僅巧合共用字母，論文原文如此**
- $r,\alpha$：LoRA rank 與 scaling 超參數；$s=\alpha/r$（LoRA）或 peft 預設值（DoRA，見「待確認的實作決定」）
- $BA=\Delta W$：LoRA 低秩更新量，$B\in\mathbb{R}^{d\times r},A\in\mathbb{R}^{r\times d}$
- $m$：DoRA 的 magnitude vector（可訓練參數，逐 column）
- $W_{pre}$：凍結的預訓練權重矩陣

**本文件縮寫：**
- CA = Clean Accuracy（測試集上的分類準確率，數字皆為百分比）；ASR = Attack Success Rate
- Cl = clean-strengthened regularization（機制1，dropout on $W_{pre}$）
- Tr = trigger-insensitive regularization（機制2，正交懲罰 $\Omega(A,B)$）
- Pt = post-training spectral rescaling（機制3，訓練後重新校準 scaling $s$）
- 三者名稱與縮寫皆沿用論文 Table 4 caption："Cl = clean-strengthened regularization;
  Tr = trigger-insensitive regularization; Pt = Post-training spectral rescaling"

---

## 總覽對照表：我們的結果 vs 論文 Table 2/4/5（RoBERTa/SST-2-only）

論文 InSent 兩格（LoRA+Cl、LoRA+Cl+Tr+Pt）CA/ASR 已用論文 Table 4 截圖核實補齊。
DoRA+Cl+Tr+Pt 兩格已用論文 Table 5（"Performance of integrating RoRA with LoRA variants"，
RoBERTa, SST-2）截圖核實補齊——這是論文自己做過的「RoRA 疊加在 DoRA 上」實驗，
跟我們的 DoRA+Cl+Tr+Pt 是同一個實驗設計，非我們自創的額外對照。

**Cl+Tr+Pt 四格已用修正後的 layer 選取（`[-6:]`，3 個完整 transformer layer）重跑更新。**
**所有含 Tr 的格子（單次 seed=0/若干 seed）代表性有限，見下方「Tr 機制總結」。**

| Method | BadNet CA/ASR (me) | BadNet CA/ASR (paper) | InSent CA/ASR (me) | InSent CA/ASR (paper) |
|---|---|---|---|---|
| LoRA baseline | 95.28 / 72.17 | 95.71 / 99.74 | 95.94 / 75.36 | 95.68 / 87.09 |
| LoRA +Cl | 95.50 / 21.01† | 96.16 / 10.34 | 95.11 / 16.50 | 96.16 / 65.68 |
| LoRA +Tr (k=8)‡ | 95.94 / 93.73 | 95.33 / 13.42 | 95.66 / 88.12 | 95.86 / 93.84 |
| LoRA +Pt alone | 95.17 / 48.29 | 無 | 95.94 / 21.01 | 無 |
| LoRA +Cl+Tr (k=8) | 95.55 / 15.62 | 無 | 95.55 / 17.38 | 無 |
| LoRA +Cl+Pt | 95.50 / 16.50 | 無 | 94.78 / 6.49 | 無 |
| LoRA +Tr+Pt | 95.72 / 78.55 | 無 | 95.88 / 36.08 | 無 |
| LoRA +Cl+Tr+Pt | **95.72 / 11.55** | 95.99 / 6.49 | **95.55 / 6.71** | 95.83 / 17.05 |
| DoRA baseline | 95.61 / 68.21 | 95.61 / 66.23 | 95.44 / 47.52 | 95.99 / 99.34 |
| DoRA +Cl | 94.95 / 29.04 | 無 | 95.33 / 16.72 | 無 |
| DoRA +Tr (k=8)‡ | 95.61 / 76.35 | 無 | 95.77 / 49.50 | 無 |
| DoRA +Pt alone | 95.44 / 46.31 | 無 | 95.39 / 8.47 | 無 |
| DoRA +Cl+Tr (k=8) | 95.28 / 41.36 | 無 | 94.29 / 19.80 | 無 |
| DoRA +Cl+Pt | 95.55 / 19.80 | 無 | 95.22 / 7.37 | 無 |
| DoRA +Tr+Pt | 95.06 / 32.12 | 無 | 95.88 / 14.74 | 無 |
| DoRA +Cl+Tr+Pt | **95.61 / 27.39** | 96.38 / 6.16 | **94.84 / 7.70** | 95.33 / 21.56 |

† 單次 seed=0 結果；11 次獨立訓練均值 26.05，最佳 13.53——單一數字不代表典型表現，見「機制1消融」章節。
‡ 單次 seed=0 結果；k=8 的 6 次獨立訓練均值為 89.8%（範圍 67.3–99.7%）——單次數字同樣不代表典型表現，見「Tr 機制總結」章節。

**觀察 1：** InSent 的 LoRA Cl+Pt（6.49%）幾乎等於完整 Cl+Tr+Pt（6.71%）——加不加 Tr 幾乎沒差。
這直接證實「疊加後的改善可能主要來自 Cl 而非 Tr」這個先前的推測，不再只是推論。
Pt 本身也單獨有效（LoRA InSent baseline 75.36%→Pt alone 21.01%），證實機制3不完全依賴
機制1、2先鋪路，修正先前「機制3依賴機制1、2」的部分判斷。

**觀察 2：** DoRA 側呈現不同模式——Cl+Tr+Pt（7.70%）比 Cl+Pt（7.37%）還差一點，
Tr 在 DoRA 上對完整疊加是輕微拖累，不是中性也不是助力。Tr 對 LoRA/DoRA 的貢獻方向不一致，
記錄但不深究成因。

**觀察 3：** InSent 的 LoRA Cl+Tr+Pt 全開，我們的 CA（95.55）與論文 CA（95.83）已不再逐位元相同
（修正 layer 選取後的新結果）——先前記錄的「逐位元相同」是修正前版本的巧合，已隨數字更新而消失，
不再需要特別說明。

**觀察 4：** DoRA+Cl+Tr+Pt 在 InSent 上，我們的 ASR（7.70）依然低於論文對應數字（21.56）——
修正後這個優勢不減反增，是目前所有對照格子裡唯一一處在「論文有直接測過的精確對照組」上
表現優於論文的案例，且相當穩固。原因未探究，可能候選：論文該格也是單次或少次結果、
我們的 k=8/6-module Pt 選擇剛好在此設定下更有效、或純粹雜訊。不做進一步因果推論。

---

## 核心發現

1. **Baseline 幾乎完全記得 backdoor**——LoRA/DoRA 在兩種攻擊下 ASR 均落在 47–75% 範圍，
   遠高於論文對應 baseline（多數接近 90–100%），但方向一致：不加任何機制，LoRA 無法自動遺忘。
2. **Cl（clean-strengthened regularization）單獨是三個機制裡效果最強、也最穩定的**——兩個
   method、兩種攻擊皆從 baseline 大降至 17–29%，且與論文 Table 4 定性結論（Cl alone 最有效）
   一致。11 次獨立訓練顯示雖有變異（13.5–49.5%），但均值仍明顯優於 baseline，方向穩固。
3. **Tr（trigger-insensitive regularization）作為獨立機制，未能重現論文宣稱的效果，且其
   效果本身呈現接近隨機的高變異。** 完整 k=1–7 掃描無平滑趨勢；k=2（4 次）與 k=8（6 次）
   兩組獨立訓練，ASR 範圍分別為 57–100% 與 67–100%；λ 網格驗證
   （對齊論文 {1,5,10,15,20}）同樣非單調，排除「平均化強度不足」假說；已查證 Eq.9→10→11
   的懲罰方向與符號正確，排除「實作邏輯錯誤」；已證明未截斷時 Eq.10 在滿秩下退化成純 L2，
   且已用 14 個矩陣的 SVD 檢查確認奇異值譜全數平滑滿秩（詳見「Tr 機制總結」）。**進一步用獨立的
   SVD alignment 實驗直接檢驗「是否存在可分離的 trigger 專屬子空間」這個假設本身，結果為否
   （見「機制2延伸：trigger 子空間假設的直接驗證」），且此結論在區分「模型自己的 trigger」與
   「模型從未學過的另一個 trigger」後依然成立——這排除了 Tr 失敗的一個常見候選解釋
   （即「只是還沒找到對的 k」）。**
   **把 Eq.10 重新詮釋為「逼進 $W_{pre}$ 未被使用的子空間」（大 k Tr），實測 k∈{64,128,256,384,512}
   後同樣呈現與 k≤32 時相同的高變異，五個 k 的均值排序（k=128 最低、k=384 最高）之間差距
   不到 0.25，且每個 k 底下的個別訓練結果本身變異幅度就跟這個排序差距同量級（見「衍生假說：
   大 k Tr」）——目前資料不足以判定任何一個大 k 值系統性優於其他。**
   綜合以上，Tr 單獨使用時「看不到穩定成功」——不是零效果（InSent 上與 Cl 疊加時近乎中性，
   BadNet 上小幅正貢獻），但從未穩定重現論文的強力效果。這個結論排除的不是幾類互相獨立的
   可能性，而是同一條推理鏈上的兩端（見「Tr 機制總結」之「已排除的解釋」）：一端是「trigger
   有專屬子空間、只是超參數沒調對」（k、λ 皆已測滿全範圍，含大 k 重新詮釋版本，仍無穩定效果），
   另一端是這個前提本身（SVD alignment 實驗直接顯示不存在可分離的 trigger 子空間）——兩端
   互相印證，指向更根本的問題。
4. **三機制疊加方向正確，Pt 在完整消融下對兩個 method、兩種攻擊皆有實質貢獻**——
   BadNet 上 Cl+Tr+Pt（LoRA 11.55%, DoRA 27.39%）優於 Cl+Tr（15.62%, 41.36%）；
   InSent 上同樣一致（6.71%/7.70% vs 17.38%/19.80%）。Pt 是三機制中唯一在完整 Pt 消融下
   （Pt alone、Cl+Pt、Tr+Pt、Cl+Tr+Pt 四種組合）都穩定產生正貢獻的機制。
5. **數值重現未達成，但方向性重現成立**——多數格子 ASR 高於論文對應值，落差 1.3–8 倍不等；
   DoRA+Cl+Tr+Pt（InSent）為唯一例外，優於論文；已知多項未排除的落差來源（見「待確認事項」）。
6. **機制1（dropout）與 DoRA 的架構存在無法簡單修復的結構性衝突**——DoRA 的 `mag_scale`
   （$m/\|W_{pre}+sBA\|_c$）要求對「整體、一致」的矩陣計算，任何只作用於部分項（例如只對
   $W_{pre}$ 的 output activation 做 dropout）的機制，數學上都無法讓 mag_scale 正確反映
   被 dropout 過的狀態。已用原始碼、觸發次數、輸出數值三重驗證機制1對 DoRA 確實生效，
   但生效方式（只影響 base activation，不影響 mag_scale 本身）與 LoRA 不同、且無法讓兩者
   完全對等，也沒有不犧牲 RoRA 核心設計（選擇性作用於 $W_{pre}$）的修法。見「方法論附註」。

---

## Tr 機制總結：綜合結論（多輪驗證後的最終定案）

**結論：我們實作出的 Tr 機制，作為一個獨立正則化項，在本專案的訓練設定下看不到穩定成功。**

**支持證據（七項，逐一查證過）：**
1. k=1–7 完整掃描：鋸齒狀非單調震盪，無平滑最優值
2. k=2 四次獨立訓練：範圍 0.5732–0.9967
3. k=8 六次獨立訓練：範圍 0.6733–0.9967
4. λ 網格驗證（對齊論文 1/5/10）：全劣於主線設定，彼此非單調
5. Eq.9→10→11 方向與符號已逐字核對論文原文，確認實作邏輯正確——排除「做反方向」的可能
6. 14 個矩陣（跨 layer 0/3/6/9/12/15/18/21/23 的 query/value）SVD 檢查：全數滿秩、
   全數平滑無斷崖——確認「未截斷退化成 L2」這個理論弱點是普遍存在的，非單一矩陣特例；
   同時發現 $\sigma_1/\sigma_2$ 突出程度隨層深度變化（中間層 2–2.8 倍，淺層/深層僅 1.1–1.4 倍），
   代表同一個固定 k 在不同層實際代表的截斷比例不一致，是震盪的一個可能（未證實）成因
7. **獨立的 SVD alignment 實驗（真實 forward hidden state，真實 IMDB label=1 句子，
   badnet/insent-poisoned 與 clean 三模型對照）：real trigger 在 top-8/32/64/128 奇異方向的
   集中度全數低於（不是高於）任意選取的 control 詞，且此現象在從未中毒的 clean 模型中同樣
   存在——直接證據顯示不存在可分離、trigger 專屬的低維子空間。詳見下方獨立章節。**

**已排除的解釋（一條因果鏈，而非五個互相獨立的檢查項）：**

上述證據 5、6 是實作核對，不是「排除的可能性」，僅確認懲罰方向與符號正確、且理論上的滿秩退化
弱點已擴大驗證非單一矩陣特例，作為底下推理鏈的前提，不再單獨列為候選解釋。真正需要排除的
候選是以下這條鏈：

1. 用獨立的 SVD alignment 實驗（14 層 query/value 矩陣，真實 forward hidden state）直接檢驗
   trigger 是否落在可分離子空間——結果：trigger 並未比一般 input 詞更集中在任何子空間
   （own_trigger 甚至比 control 更分散，見「機制2延伸」）。同時發現一個非 trigger 專屬、
   所有輸入共享的性質：任一 input 的表徵，到 k≈64 時，落在前 64 個奇異方向裡的比例平均已達
   「主要 5 個貢獻方向」的九成——這是 $W_{pre}$ 譜結構本身的性質，不是 trigger 的性質。
2. 由 (1) 直接推得：不存在一個 trigger 專屬的方向或子空間，可以被 SVD 找出來單獨 target——
   因為 trigger 的表徵在這個空間裡跟其他一般詞沒有可辨別的差異，沒有東西可以「精準瞄準」。
3. 由 (2) 可推得的兩個子推論，皆已實測驗證：
   - **(3a) 沒有任何 k 可以「擋掉 trigger 方向、同時保留其他有用的 pretrained 方向」**，
     因為根本不存在可分離的界線。k<32 的實驗結果（表現不佳、高變異，見 k 消融章節）
     與此推論一致：不是「還沒找到對的小 k」，是小 k 本身就沒有東西可以精準切開。
   - **(3b) 退一步改用大 k Tr（不試圖 target trigger，改為全面逼近 $W_{pre}$ 未使用的子空間）**，
     實測 k∈{64,128,256,384,512}，僅單獨使用 Tr（不配合 Cl/Pt），仍無法讓 ASR 穩定下降，
     且呈現與 k<32 時相同的高變異（見「衍生假說：大 k Tr」）。
4. 另外測試了 Eq.10 跨層 penalty 的平均化 vs 加總（對齊論文 λ 網格），結果同樣不佳：
   λ 加大（加總形式，對齊論文 1/5/10）效果劣於現行平均化設定，且彼此非單調（見「λ 網格驗證」）。
   目前僅測過對齊論文網格的加大方向，未測過縮小 λ 的方向，這條路徑尚未窮盡。

**仍未定論的部分：** 為何這個在論文裡（至少 RoBERTa+BadNet 這個組合）宣稱有效的機制，
在我們的實作下呈現接近隨機的效果。候選解釋包括：(a) 論文的 13.42% 本身是多次 run 取
最好，其"典型"表現可能同樣不穩定，只是論文未報告變異程度；(b) 截斷維度在不同層應該
不同、但論文與我們的實作都用同一個全域 k，可能是震盪成因之一（未驗證）；(c) Eq.10
這個正則化項本身，即使正確截斷，對訓練動態的影響可能天生就是弱訊號、易被其他梯度分量
淹沒；**(d) Eq.10 想要壓制的「trigger 方向」本身可能不存在於任何固定的低維子空間裡
（現已有直接證據支持這個候選，即上方推理鏈），這代表 Eq.10 的整個設計前提——「trigger 有
專屬方向、可以被 SVD 截斷找出來」——在我們的重現設定下可能就是不成立的，而不只是
「找錯 k」的問題。** 四者皆為推測或部分證實，非完全確認的因果結論。

**與 Cl、Pt 的對比：** Cl（11 次獨立訓練）與此處 Tr（k=2/k=8 共 10 次獨立訓練）都呈現高變異，
但 Cl 的均值（26.1%）遠優於 baseline（~70%），方向穩固；Tr 的均值（k=8: 89.8%）
與 baseline（72.2%）相近甚至更差，沒有展現出穩固的正貢獻方向。這是 Cl 與 Tr 最關鍵的
差異——不是「兩者都不穩定」，是「Cl 不穩定但方向對，Tr 不穩定且方向不明確」。

---

## 機制2延伸：trigger 子空間假設的直接驗證（SVD Alignment 實驗）

### 動機

機制2消融（見下方章節）顯示不管怎麼調 k，Tr 的效果都接近雜訊等級。一個常見的候選解釋是
「Eq.10 的設計本身沒錯，只是我們沒找到 trigger 真正落在的那個子空間維度」。這個實驗直接
檢驗這個假設本身：trigger 的表徵，在 $W_{pre}$ 的奇異值分解下，是否真的集中在某個
可辨識、跟一般文字不同的子空間裡？

### 方法演進（三個版本，記錄修正過程）

1. **v1（初版）**：用 trigger 的原始 embedding（未經任何 transformer layer）對每一層
   query/value 矩陣的右奇異向量算 cosine similarity。**方法論缺陷**：每一層都重複套用
   同一個 embedding，忽略了 self-attention 逐層混合上下文的事實——layer 3 的輸入早已
   不是 layer 0 的原始 embedding。此版本結果（layer0 對照 clean/badnet/insent 三模型）
   顯示 trigger（`mn`/InSent）與多個 control 詞的 mean cosine similarity 幾乎相同
   （皆 ≈0.025），且此現象在 clean 模型中同樣存在。因方法論有缺陷，此版本結果不作為
   最終結論依據，僅記錄修正過程；「the」在 54 個 layer×module 組合中從未出現其他實詞
   共有的 rank-0 凸起，是此版本唯一額外確認過的細節。
2. **v2/v2.1（中間版）**：改用 forward pre-hook 抓 `attention.self` 實際吃到的 hidden
   state（即上一層完整 transformer block 的真實輸出），並將 trigger 插入真實句子中而非
   孤立測試，插入邏輯完全對齊 `attack_utils.insert_trigger`。
3. **v4（最終版，含以下修正，此版本結果為正式採用）**：
   - 加入 $\sigma$ 加權的「真實貢獻度」`contributions = cos_sims * S`（而非單純 cosine
     similarity），修正了「trigger 即使對齊到一個奇異值很小的方向，該方向對輸出幾乎沒有
     影響」這個原始 band-pass 假設的漏洞
   - **資料來源修正**：查證 GitHub 上 `poisoned_pretrain.py` 後確認，backdoor pretraining
     用的語料是 **IMDB**（`dataset_tag` 預設 `"imdb"`），不是 SST-2——SST-2 只在後續
     `variant_finetune.py` 的 LoRA 微調階段使用。base sentence 來源修正為 IMDB
   - **label 篩選修正**：查證同一份 code 確認 train/dev 對 `label==0`（target class）插入
     trigger 且不改標籤，test 則對 `label==1`（non-target class）插入 trigger 來測 ASR——
     這是標準的 target-class poisoning 設計（trigger 學到的捷徑是「看到 trigger→判成
     label 0」），不是隨機或疏漏。因此 base sentence 篩選改為 `label==1`，對齊真正
     ASR 評測時 trigger 被使用的情境
   - **分組統計修正**：初版把所有 trigger（含 control）混在同一個全域分佈裡統計，抹掉了
     「真 trigger vs control」的對比，修正為依 `real_trigger`（`mn`、InSent 句）/`control`
     （`abysmal`、`perfect`、`the`、`movie`、`watched`、`3D movie`、兩句 decoy 句）分組統計
   - 三個模型（poisoned_roberta_large_badnet / poisoned_roberta_large_insent /
     clean_roberta_large）皆測試，clean 模型作為「backdoor 訓練是否是成因」的對照組

### 結果 A：real_trigger vs control（2-way，pool 所有 trigger）

| 模型 | 組別 | n | top-8 | top-32 | top-64 | top-128 | top-256 |
|---|---|---|---|---|---|---|---|
| badnet-poisoned | real_trigger | 180 | 39.44% | 75.00% | 89.44% | 96.67% | 100.00% |
| badnet-poisoned | control | 810 | 51.60% | 84.07% | 93.95% | 98.52% | 100.00% |
| insent-poisoned | real_trigger | 180 | 42.22% | 76.67% | 89.44% | 95.00% | 99.44% |
| insent-poisoned | control | 810 | 49.01% | 83.09% | 92.72% | 97.78% | 99.63% |
| clean（從未中毒） | real_trigger | 180 | 48.33% | 78.33% | 90.00% | 95.56% | 100.00% |
| clean（從未中毒） | control | 810 | 50.62% | 81.23% | 90.99% | 96.54% | 99.26% |

### 結果 B：own_trigger / foreign_trigger / control（3-way 細分，僅中毒模型適用）

同一模型內部細分「自己真正學到的 trigger」vs「模型從未見過的另一種 trigger」vs「一般
control 詞」，比 2-way 版本更嚴謹——避免把 `mn`（BadNet trigger）跟 InSent 句都算進同一個
`real_trigger`，稀釋掉模型自己真正學到的攻擊跟一個不相干 trigger 之間的對比。

| 模型 | 組別 | top-8 |
|---|---|---|
| badnet-poisoned | own_trigger（`mn`） | **36.67%** |
| badnet-poisoned | foreign_trigger（InSent） | 42.22% |
| badnet-poisoned | control | 51.60% |
| insent-poisoned | own_trigger（InSent） | **34.44%** |
| insent-poisoned | foreign_trigger（`mn`） | 50.00% |
| insent-poisoned | control | 49.01% |

**⚠️ 資料完整性提醒**：目前只留存 top-8 這一欄的 3-way 數字；top-32/64/128/256 以及各組
精確 n（原 180 筆如何拆分成 own/foreign 各半）尚未留存，需要重新跑一次才能補完——**在
補完之前，下方結論僅基於 top-8 這一個資料點，不代表已驗證跨所有 k 尺度的完整模式**。

**結論（基於 top-8，暫定）：own_trigger 在兩個中毒模型上的集中度都是三組裡最低的**——
不是次低，是三組（own/foreign/control）裡最分散的一組。這比 2-way 版本的證據更強：即使
排除了「跟另一個不相干 trigger 混在一起」的稀釋效應，模型自己真正學到的攻擊 trigger，
其表徵依然沒有集中到任何可辨識的低維子空間，反而比一個隨機 control 詞更分散。

### 綜合結論（2-way + 3-way 共同支持）

**不存在可分離的 trigger 專屬子空間，且此結論在細分 own/foreign 後依然成立、甚至更清楚。**
不論用哪種分組方式，"trigger"（無論是不是模型自己的）都沒有比 control 詞更集中，方向
一致與预期相反。且「real_trigger 比 control 更分散」的現象在從未中毒過的 clean 模型裡也
存在（2-way 結果），證明這不是 backdoor 訓練造成的，而是這批字詞本身 hidden state 分佈的
內在差異，跟是否為 trigger 無關。

**附帶發現：** 不論哪一組，訊號幾乎全數（top-256 達 99.3–100%，top-128 已達 95–98%）
落在 1024 維裡的前 256 維——1024 維裡有 700 維以上，對任何自然語言輸入幾乎都沒有影響力。
這是所有測過的字詞共享的性質，不是 trigger 專屬的。

**未經統計驗證、保守記錄的觀察：** 2-way 版本中，中毒模型的 real_trigger/control 差距
（badnet 12.2pp、insent 6.8pp）比 clean 模型（2.3pp）更大。這可能暗示 poisoning 訓練
讓 control 詞的集中度相對變高了一些，但這只是單次結果（n=180 vs n=810 不對稱），未經
獨立重複驗證。**此觀察不作為結論，僅記錄供未來若有時間可延伸驗證。**

### 衍生假說：大 k Tr（脫離全域表徵空間，而非狙擊 trigger）

既然沒有 trigger 專屬子空間，但訊號幾乎全部集中在前 256 維以內，Tr 的設計意圖可以重新
詮釋為：不是「精準狙擊 trigger 方向」（已證明不存在這種可分離方向），而是「把 ΔW 逼進
$W_{pre}$ 幾乎沒在用的那 700+ 維空間」。已測試 k∈{64,128,256,384,512}，BadNet/LoRA，
僅單獨使用 Tr（不配合 Cl/Pt），外加一組同等 λ 強度的 L2 weight decay 對照。

訓練腳本未鎖定 cuDNN 決定性（見「方法論附註」），因此下表每一格列出的是「獨立完整跑完
20 epoch 的訓練嘗試」各自的 final ASR，不對應特定 seed 身份：

| 設定 | 各次獨立訓練 final ASR | 均值 |
|---|---|---|
| L2 weight decay（同等強度對照） | 0.9978、0.9945、0.9956 | 0.9960 |
| Tr, k=64 | 0.9912、0.9934、0.7173 | 0.9006 |
| Tr, k=128 | 0.7635、0.4180、0.8185、0.8724 | 0.7181 |
| Tr, k=256 | 0.7140、0.9835 | 0.8488 |
| Tr, k=384 | 0.9978、0.9406、0.9582 | 0.9655 |
| Tr, k=512 | 0.8779、0.8746、0.8570 | 0.8698 |

k=128、k=256 另各有訓練未能完整跑完 20 epoch，不列入上表：k=128 一次在啟動階段即中止
（零 epoch 資料）；k=256 兩次分別在訓練中途停止，其中一次已確認是 GPU 記憶體被另一行程
佔用所致（與 Tr 訓練本身的數值穩定性無關），另一次原因不明。

**論點一：大 k Tr 不是「偽裝的 L2」。** L2 對照組本身幾乎沒有效果（0.996，接近 baseline），
而所有測過的大 k Tr 條件均值都明顯低於 L2。這排除了「大 k 效果只是重新發現 L2 weight
decay 有點用」這個混淆假說——先前已證明未截斷時 Eq.10 退化成 L2，但這裡的效果不能單純
用退化成 L2 解釋。

**論點二：在 k=64/128/256/384/512 這個範圍內，沒有證據支持任何一個 k 系統性優於其他。**
五個 k 的均值排序為 128(0.72) < 256(0.85) < 512(0.87) < 64(0.90) < 384(0.97)，但每個 k
底下的個別訓練結果本身變異都很大——例如 k=64 三次是 0.99/0.99/0.72，k=128 四次是
0.76/0.42/0.82/0.87——變異幅度跟 k≤32 時看到的鋸齒狀高變異是同一種形狀，不是一個乾淨的
U 型或單調趨勢。加上 cuDNN 非決定性讓每個 k 的均值本身會隨著累積更多次獨立訓練持續小幅
波動，現有樣本量（每個 k 僅 2–4 次）太小，不足以把任一 k 判定為最佳。

**目前唯一站得住腳的結論**：大 k Tr 確實比 L2 weight decay 有更強的效果，但哪個 k 最好、
或是否存在真正的最優 k，現有資料不足以回答。

---

## 方法論附註

### evaluate_checkpoint.py 決定性驗證

用同一個 checkpoint 重新評估（backfill_eval.sh），三筆原本只有口頭記錄、未存 log 的舊結果
（LoRA+Tr k=1024、LoRA+Tr k=8/lr=2e-3、LoRA+Tr k=8/lr=2e-5）與 evaluate_checkpoint.py 重新跑出的
結果逐位元一致，證實 evaluation（非訓練）本身是決定性的，可信賴用於事後補測未存檔的 checkpoint。

### 訓練本身的隨機性未完全鎖定（cuDNN 非決定性）

`variant_finetune.py` 對 `torch.manual_seed`/`np.random.seed`/`random.seed` 都設了值，
但沒有設定 `torch.backends.cudnn.deterministic=True`（也沒設 `cudnn.benchmark=False`），
導致同一個 seed 數值在兩次獨立執行中，GPU 上的浮點運算路徑仍可能不同，訓練多個 epoch 後
放大成明顯不同的最終 ASR：k=128 的 seed=2 在兩次執行中分別得到 0.7635 與 0.8724。

**影響範圍：** 本文件所有「N 次獨立訓練」章節（機制1的 11 次、機制2的 k=2 四次／k=8 六次、
大 k Tr 各設定的 2–4 次）裡的每一次，都應理解為一次獨立隨機訓練，不是可重現、可個別追蹤
核對的實驗單位——重跑同一個 seed 編號不保證重現同一個結果。這不影響這些章節已得出的
「效果高變異」這個結論本身（變異本來就存在），但這些變異裡混有 cuDNN 非決定性貢獻的
額外雜訊，不是純粹只來自「不同初始化/不同資料順序」。

**日後改善：** 若之後的實驗需要真正可重現、可核對的單次結果，應在腳本最上方（`import torch`
之前）設定 `os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"`，並在初始化時加上
`torch.backends.cudnn.deterministic=True`、`torch.backends.cudnn.benchmark=False`、
`torch.use_deterministic_algorithms(True)`。此設定可能會降低訓練速度，尚未評估是否值得
在目前的時間壓力下全面套用。

### 機制3新 scaling s 數值（修正後，6-module / 3 完整 transformer layer）

新 $s$ 值遠大於訓練時原值（LoRA 訓練時 $s=\alpha/r=2$；DoRA 訓練時 $s=1$，peft 預設），
符合論文邏輯（$\sigma_{max}(\Delta W)\ll\sigma_{max}(W_{pre})\Rightarrow$ 新 $s$ 應遠大於訓練時 $s$），
方向被實測驗證。以 BadNet LoRA Cl+Tr+Pt 為例：layer21.query=4.96, layer21.value=12.36,
layer22.query=6.47, layer22.value=16.05, layer23.query=7.08, layer23.value=9.43——
多出的 layer21 兩個 module 的 $s$ 值與 layer22/23 同量級，支持「top three layers 確實
橫跨三個完整 transformer layer」這個修正方向是合理的。

### 機制1的 dropout hook 與 DoRA：三重驗證確認生效，但存在結構性限制（已結案，不修）

**驗證過程（三層）：**
1. **原始碼**（peft 0.11.1, `tuners/lora/layer.py`）：`Linear.forward()` 主體先呼叫
   `result = self.base_layer(x)`（真正的 forward 呼叫，觸發 hook），之後對 DoRA 分支呼叫
   `self._apply_dora(x, ...)`，其回傳值以 `result = result + result_dora` 的方式相加，
   不是替換或扣除。`_apply_dora` 內部另外用 `F.linear(x, weight)` 重新計算一份 base term，
   這次計算繞過 hook（直接對 weight tensor 做 `F.linear`，非呼叫 `base_layer.__call__`）。
2. **觸發次數實測**：hook 呼叫次數 = 245（217 train step + 28 eval step）× 48 modules = 11760，
   與印出的計數精確吻合，證實 hook 確實在 DoRA 訓練中被觸發，不是掛假的。
3. **輸出數值實測**：固定 seed 下，有/無 hook 兩次獨立跑的 logits 明顯不同
   （[-1.613, 0.890] vs [-2.579, 2.223]），證實 hook 效果確實傳遞到最終輸出，
   未被內部重新計算完全覆蓋。

**結構性結論（已定案，不修）：** `mag_scale = magnitude / weight_norm` 中的 `weight_norm`
（$\|W_{pre}+sBA\|_c$）由 `get_weight_norm`/`get_delta_weight` 直接讀取原始 `.weight` 張量計算，
不受任何 dropout 影響——這是 peft 架構本身的設計（連 peft 內建的 `lora_dropout` 也同樣不影響
`weight_norm`，非我們實作的不一致）。這代表：DoRA 的 `mag_scale` 要求對整體矩陣做「一致」計算，
任何選擇性、部分作用於 $W_{pre}$ 的機制（包括機制1），在數學上都無法讓 `mag_scale`
正確反映被該機制影響過的狀態。機制1確實對 DoRA 的最終輸出生效（已驗證），但生效路徑只作用於
base activation 這一項，不會、也不可能影響 `mag_scale` 本身——這不是實作疏漏，而是 DoRA
架構本身（`mag_scale` 要求整體一致性 vs RoRA 要求選擇性作用於 $W_{pre}$）的根本張力。

**不存在不犧牲 RoRA 核心設計的修法**：若改成對最終合併輸出做 dropout，`mag_scale` 的一致性
可以維持，但此時已不是「選擇性作用於 pretrained weights」的 RoRA 設計，而是通用的 post-layer
dropout。RoRA 論文完全未討論 DoRA 上的這個組合，我們推測作者可能未處理過這個特定情況，
但無法證實——這不影響本輪已確認的結論（機制1對 DoRA 確實生效，僅結構性地與 LoRA 不同）。
此問題判定為結構性限制，不再繼續深究或嘗試修復。

---

## 機制1消融：LoRA+Cl 的 11 次獨立訓練變異性驗證

固定設定：lr=2e-4, p=0.1，BadNet，seed 0–10。

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

單次「典型」表現（平均）約 26%，遠高於論文 Table 4 的 Cl alone（10.34%）。但最佳一次
（13.53%）已非常接近論文數字。這與論文自身聲明的方法論一致：「Unless otherwise noted,
results in the remaining tables and figures also reflect the best performance from our
repeated experiments.」——論文的 10.34% 很可能同樣是多次取最好的結果，不是穩定的單次表現。

機制1（Cl）本身的效果對訓練隨機性相當敏感——這個高變異性本身是一個值得報告的觀察，
不應該被「挑最好的一次」這個呈現方式掩蓋掉。誠實的結論是：機制1平均能把 ASR 從 baseline 的
~70% 壓到 ~26%，最佳情況下可達 13.5%（接近論文數字），但單次結果不穩定。

InSent 只跑了一次（16.50%），未做多次獨立驗證，沒有理由假設它比 BadNet 更穩定——在做
類似的多次驗證之前，不對 InSent 與 BadNet 之間的 Cl 效果差異做任何因果解釋。

---

## 機制2消融：Tr（正交懲罰），固定 λ=10（除 λ 網格章節外）

### k 消融（固定 lr=2e-4, LoRA, BadNet）—— 完整版，含精細掃描與多次獨立驗證

| Method | k | Test CA | ASR |
|---|---|---|---|
| LoRA+Tr | 1 | 0.9594 | 0.9153 |
| LoRA+Tr | 2（seed=0） | 0.9561 | 0.5732 |
| LoRA+Tr | 3 | 0.9506 | 0.5732 |
| LoRA+Tr | 4 | 0.9572 | 0.8262 |
| LoRA+Tr | 5 | 0.9577 | 0.9758 |
| LoRA+Tr | 6 | 0.9528 | 0.7877 |
| LoRA+Tr | 7 | 0.9610 | 0.8713 |
| LoRA+Tr | 8 (=r), seed=0 | 0.9594 | 0.9373 |
| LoRA+Tr | 32 | 0.9588 | 0.9648 |
| LoRA+Tr | 1024（不截斷，等效完整方陣） | 0.9500 | 1.0000（最差） |

**k=2 的四次獨立訓練（BadNet, LoRA+Tr, k=2, λ=10）：**

| 次 | Test CA | ASR |
|---|---|---|
| 1 | 0.9561 | 0.5732 |
| 2 | 0.9594 | 0.9967 |
| 3 | 0.9605 | 0.9725 |
| 4 | 0.9572 | 0.5787 |
| **均值** | 0.9583 | **0.7803** |

**k=8 的六次獨立訓練（BadNet, LoRA+Tr, k=8, λ=10）：**

| 次 | 最終存檔 epoch | ASR |
|---|---|---|
| 1 | — | 0.9373 |
| 2 | epoch 3 | 0.9967 |
| 3 | epoch 4 | 0.9956 |
| 4 | epoch 9 | 0.6733 |
| 5 | epoch 6 | 0.9483 |
| 6 | epoch 7 | 0.8361 |
| **均值** | — | **0.8979** |

**結論：k=2、k=8 沒有一個能被判定為最佳值，兩者皆為雜訊等級的高變異。** k=2 四次獨立
訓練範圍 0.5732–0.9967，k=8 六次範圍 0.6733–0.9967，均值本身也相近（78.0% vs 89.8%）且
區間有大幅重疊——沒有證據顯示任一測過的 k 值系統性優於另一個，總覽表沿用 k=8 純粹是
歷史慣性（最早使用、被最多其他消融實驗共用），不代表它是已驗證的穩定基準。詳細解讀見
「Tr 機制總結」章節。訓練本身的 cuDNN 非決定性（見方法論附註）意味著這裡的每一次也都是
獨立隨機訓練，不是可重現、可個別追蹤核對的實驗單位。

DoRA 的 k 對照（僅測過 k=8, 32，未做多次獨立驗證，同樣可能受此雜訊問題影響）：

| Method | k | Test CA | ASR |
|---|---|---|---|
| DoRA+Tr | 8 | 0.9561 | 0.7635 |
| DoRA+Tr | 32 | 0.9533 | 0.8526 |

### λ 網格驗證（固定 k=2, LoRA, BadNet, seed=0）—— 假說已推翻

**懷疑：** `penalty / num_penalized_layers`（除以 48）讓有效強度 λ_eff=10/48≈0.21 落在論文
網格 {1,5,10,15,20} 之外，換算回加總形式應為 λ∈{48,240,480,720,960} 才對齊論文的 1/5/10。

**實測（k=2 固定，加總形式）：**

| λ (加總) | 對應論文網格值 | Test CA | ASR |
|---|---|---|---|
| 48 | ~1 | 0.9616 | 0.9461 |
| 240 | ~5 | 0.9533 | 0.7404 |
| 480 | ~10 | 0.9588 | 0.8306 |
| 10（平均化，主線設定） | — | 0.9561 | 0.5732 |

**結論：假說已推翻。** 三個對齊論文網格的 λ 值皆劣於主線設定，且彼此間非單調——但此結論
本身建立在 k=2、單次訓練的設定上，λ 網格本身是否同樣受高變異影響未再驗證。平均化本身不是
造成落差的原因，此點仍然成立（因為對照組本身用的也是同一個 k=2、同一次訓練）。**只測過
對齊論文網格的加大方向（λ 加總 48/240/480 對齊 1/5/10），未測過縮小 λ 的方向，這條路徑
尚未窮盡，不應宣稱「λ 已排除」的範圍包含縮小方向。**

### lr 消融（固定 k=8, LoRA, BadNet）

| lr | Test CA | ASR |
|---|---|---|
| 2e-5 | 0.9605 | 1.0000（太小，訓練不動，backdoor 完全沒被觸動） |
| 2e-4 | 0.9594 | **0.9373**（最佳，主線一直使用的值） |
| 2e-3 | 0.9456 | 訓練崩潰（epoch 1 起 dev acc ~0.49，等同亂猜，數字無意義） |

結論：lr=2e-4 已是三者中最佳，排除「lr 選錯導致機制2效果差」這個假設。

### 理論發現：Eq.10 在滿秩情況下退化成 L2 weight decay（已驗證，非訓練實驗）

論文 Eq.9→Eq.10 的正交懲罰 $\Omega(A,B)=\|U^\top B\|_F^2+\|AV\|_F^2$，若 $U,V$ 取自未截斷的
完整 SVD，在 $W_{pre}$ 為滿秩方陣時，數學上會退化成標準 L2 weight decay
（∵ 完整正交矩陣不改變 Frobenius norm：$\|U^\top B\|_F=\|B\|_F$）。

**實測驗證（已擴大至 14 個矩陣）：** RoBERTa-large 的 layer 0/3/6/9/12/15/18/21/23（query/value）
共 14 個矩陣，SVD 後奇異值全數 >1e-6（1024/1024，滿秩），全數呈現平滑衰減、無斷崖。
$\sigma_1/\sigma_2$ 比值隨層深度變化：中間層（6/9/12/15）達 1.7–2.8，淺層（0/3）與深層（18/21/23）
僅 1.1–1.4——代表最大奇異值的「突出程度」在不同層並不一致，同一個固定 k 在不同層實際
截斷的比例因此也不一致，這是 Tr 震盪的一個可能（未證實）成因。

論文本身未說明 Eq.10 計算時 $U,V$ 是否截斷、截斷到多少維——這是我們自己的實作決定，
不是論文明確指定的答案。

### 論文對照

論文 Table 5（RoBERTa+BadNet）Tr alone: CA 95.33 / ASR 13.42。我們的 Tr alone 結果
（k=8 六次均值 89.8% 與範圍 67.3–99.7%；k=2 四次均值 78.0% 與範圍 57.32%–99.67%）均遠高於
論文數字，且經多次獨立驗證後，此落差無法用「湊到一個更好的 k」來系統性縮小——落差的性質
更接近「Tr 這個機制本身的實作，在我們的訓練設定下就是效果不穩定」，而非「差一個超參數」。

值得注意：論文裡 Tr alone 只在 RoBERTa+BadNet 這個組合特別有效（13.42），其餘三個
model+attack 組合（RoBERTa+InSent, LLaMA+BadNet, LLaMA+InSent）Tr alone 效果皆遠差
（77–98），顯示這個低點本身在論文裡也不是普遍模式。

---

## 機制3在 DoRA 上的理論分析（使用者原創推導，非論文內容）

DoRA forward（理論分析用的簡化 margin 形式，不含 dropout 交互作用——後者見「方法論附註」）：
$M^{DoRA}(s)=\dfrac{m}{\|W_{pre}+s\cdot BA\|_c}\cdot(A_0+s\cdot A_1)$，
其中 $A_0=\langle c,W_{pre}x^{trig}\rangle$，$A_1=\langle c,\Delta Wx^{trig}\rangle$（同 Proposition 4.2 定義）。

**推導 1（已證明）：DoRA 與 LoRA 的 margin 零點相同。**
外層 $m/\|\cdot\|$ 恆正（$m>0$，範數恆正），不影響 margin 正負號，故
$s^\star_{DoRA}=-A_0/A_1=s^\star_{LoRA}$。這代表 Eq.12 公式沿用到 DoRA 上，
在「讓 margin 翻正」這個判準上理論成立，不是誤用。

**推導 2（已證明）：DoRA 的 margin 存在封頂，LoRA 沒有。**
$s\to\infty$ 時，$M^{LoRA}(s)\to\infty$（線性發散）；但 $M^{DoRA}(s)\to\dfrac{m\cdot A_1}{\|BA\|_c}$
（收斂到固定常數）。DoRA 不管 $s$ 調多大，margin 都無法超過這個天花板——這是 LoRA 沒有的限制。

**推導 3（已證明，適用於兩者）：Eq.12 隱含 "wishful thinking" 條件 $\rho_{eff}>\rho_{bd}$。**
代入 $s=\sigma_{pre}/\sigma_\Delta$ 到 $s>s^\star=(\rho_{bd}/\rho_{eff})\cdot(\sigma_{pre}/\sigma_\Delta)$，
兩邊消去 $\sigma_{pre}/\sigma_\Delta$ 得 $\rho_{eff}>\rho_{bd}$。即 Eq.12 這個簡化公式要真正生效，
前提是機制1、2已經把 $\rho_{eff}$ 推得比 $\rho_{bd}$ 大——此條件對 DoRA、LoRA 相同（因零點相同），
機制3依賴機制1、2先鋪路，不是獨立生效的機制。**此推導與 Pt alone 消融結果（機制3單獨也有效）
不衝突——推導講的是「Eq.12 的簡化公式要生效需要 ρ_eff>ρ_bd」，並不排除 baseline 本身
（未經機制1、2強化）在某些情況下就已滿足這個條件。**

---

## 附錄 A：主要結果（k=32 版本，最初依 Figure 2 caption 選值，非消融後最佳值）

| Method | 機制 | Test CA | ASR |
|---|---|---|---|
| LoRA | baseline | 0.9528 | 0.7217 |
| LoRA | +Cl (p=0.1) | 0.9550 | 0.2101 |
| LoRA | +Tr (λ=10, k=32) | 0.9588 | 0.9648 |
| LoRA | +Cl+Tr | 0.9610 | 0.3234 |
| DoRA | baseline | 0.9561 | 0.6821 |
| DoRA | +Cl (p=0.1) | 0.9495 | 0.2904 |
| DoRA | +Tr (λ=10, k=32) | 0.9533 | 0.8526 |
| DoRA | +Cl+Tr | 0.9583 | 0.1859（八組裡最佳） |

---

## 待確認的實作決定（未經論文或 advisor 證實，需要標注）

- BERT 版本用 bert-large-uncased（論文、PSIM repo 均未指定 cased/uncased）
- LLaMA 版本假設 huggyllama/llama-7b（論文未指定規模）
- 機制2 SVD 截斷 k（論文未明講是否截斷、截斷到多少）——已消融 k∈{1,...,8,32,64,128,256,384,512,1024}，
  對 k=2、k=8 皆做過多次獨立驗證，任一單一 k 的表現都無法穩定重現，效果近乎雜訊；大 k
  （64–512）延伸驗證同樣呈現高變異，且五個大 k 的均值彼此差距不到 0.25，個別訓練的變異
  幅度就跟這個差距同量級，**沒有任何一個 k 可判定為最優**；且已用獨立的 SVD alignment
  實驗（含 3-way own/foreign/control 細分）排除「存在固定 trigger 子空間、只是 k 沒找對」
  這個解釋（見「機制2延伸」章節）
- 機制2跨層 penalty 加總 vs 平均（論文 Eq.11 未明講）——已測試對齊論文網格的加總形式，
  結果皆劣於現行平均化設定，假說已推翻，但仍不確定論文原意是哪一種；且只測過加大方向，
  縮小 λ 的方向未測過
- **訓練腳本未鎖定 cuDNN 決定性**（見「方法論附註」）——所有既有多次獨立驗證結果的每一次
  都應理解為獨立隨機嘗試，不是可重現身份；日後若需要可重現的單次結果，需補上
  `cudnn.deterministic=True` 等設定，尚未套用
- 機制3「top three layers」假設為模型最後三層——已修正為 `[-6:]`（3 個完整 transformer
  layer），總覽表已用修正版本重跑更新，此項可視為已解決
- 機制3對 DoRA 的 scaling 語義——已完成理論推導，但推導本身的近似項在 DoRA 封頂結構下的
  誤差程度未量化
- 機制1 dropout hook 對 DoRA 的生效機制與結構性限制——已結案，無不犧牲 RoRA 核心設計的修法
- ASR 僅在 dev acc 刷新時量測，20 epoch 中僅 5–7 次評估，可能錯過更低的中間值
- 論文數字多為多次 run 取最好；我們除 LoRA+Cl（BadNet, 11 次）、LoRA+Tr k=2（4 次）、
  LoRA+Tr k=8（6 次）、大 k Tr（k=64/384/512 各 3 次，k=128 共 4 次，k=256 共 2 次）外，
  其餘均為單次——鑑於多組結果都顯示極高變異性，其餘單次數字的代表性應保守看待

---

## 尚未完成

### 有明確查證路徑（若有餘裕可執行）
- 補完 SVD alignment 實驗 3-way（own/foreign/control）細分的完整 top-32/64/128/256 數字
  及各組精確 n——目前僅記錄了 top-8 這一欄
- 驗證「同一 k 在不同層代表不同截斷比例」是否為震盪真正成因：可對每層各自用不同的 k
  （依該層 σ1/σ2 比值調整），而非全域統一 k，觀察震盪是否減緩——工程量較大，未執行
- 對 λ 網格、k=32/1024 等其餘設定重複多次獨立訓練，確認雜訊程度是否一致；也尚未測過
  縮小 λ 的方向
- k=256 目前僅 2 次獨立結果（0.7140、0.9835，差距極大）、k=128 僅 4 次（0.42–0.87）——
  若有餘裕，可再補幾次獨立訓練縮小均值的不確定範圍
- SVD alignment 實驗中「中毒模型差距比 clean 模型大」的觀察（2-way 版本）——單次結果，
  未經多次獨立驗證，若有餘裕可延伸確認是否為真訊號
- **視時間許可，考慮在訓練腳本加上 `cudnn.deterministic=True` 等設定**，讓未來需要核對的
  單次結果具備真正的可重現性（會犧牲一些訓練速度，尚未評估是否值得現在做）

### 目前沒有已知解法，記錄為限制
- BERT cased/uncased、LLaMA 規模：三處官方來源皆未指定，僅能靠官方 code 釋出或聯繫作者解決
- 機制3「top three layers」指哪三層：論文僅給模糊描述，我們的 6-module 修正是合理延伸，
  非論文確認答案
- σ_max 近似 A_1 在 DoRA 封頂結構下的誤差界：需獨立理論推導
- 機制1 對 DoRA 的結構性限制：已確認無法修復，不再列為待辦
- **Tr 機制本身（含大 k 版本）看不到穩定成功的根本原因：已沿著「trigger 子空間是否存在→
  是否有 k 能精準切開→重新詮釋成大 k 逼近未使用子空間」這條因果鏈逐段排除，且已排除
  超參數（λ 加大方向、lr）選錯的可能性，仍無法定論是論文本身效果不穩定、逐層 k 不一致、
  還是正則化訊號天生微弱**（皆為推測，非確認結論）

### 其他
- threshold-based SVD 截斷——已知奇異值譜平滑無斷崖，且 k 本身已證實效果近乎雜訊，
  預期收益低，不再列為優先
- BERT/LLaMA 架構、CR/CoLA 資料集——全部未測試，風險評估後決定暫不執行