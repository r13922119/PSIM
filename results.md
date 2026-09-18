# RoRA 重現實驗結果

設定：RoBERTa-large + BadNet ("mn" trigger) / InSent ("I watched this 3D movie") + SST-2，
lr=2e-4（除非另有標註），20 epochs，seed=0（除 11-seed 章節）。
**總覽對照表採用 k=8（消融實驗確認之最佳值，非論文明確指定）；k=32 版本見附錄。**
**注意：k 精細掃描（見「機制2消融」）發現 k=2 優於 k=8，總覽表尚未更新為 k=2 版本，見該章節說明。**

Poisoning 驗收（ASR>95% 門檻，論文要求）：
- BadNet: dev clean acc 99.10%, ASR 100.00%
- InSent: dev clean acc 98.80%, ASR 100.00%

---

## 總覽對照表：我們的結果 vs 論文 Table 2/4/5（RoBERTa/SST-2-only）

論文 InSent 兩格（LoRA+Cl、LoRA+Cl+Tr+Pt）CA/ASR 已用論文 Table 4 截圖核實補齊。
DoRA+Cl+Tr+Pt 兩格已用論文 Table 5（"Performance of integrating RoRA with LoRA variants"，
RoBERTa, SST-2）截圖核實補齊——這是論文自己做過的「RoRA 疊加在 DoRA 上」實驗，
跟我們的 DoRA+Cl+Tr+Pt 是同一個實驗設計，非我們自創的額外對照。

| Method | BadNet CA/ASR (me) | BadNet CA/ASR (paper) | InSent CA/ASR (me) | InSent CA/ASR (paper) |
|---|---|---|---|---|
| LoRA baseline | 95.28 / 72.17 | 95.71 / 99.74 | 95.94 / 75.36 | 95.68 / 87.09 |
| LoRA +Cl | 95.50 / 21.01† | 96.16 / 10.34 | 95.11 / 16.50 | 96.16 / 65.68 |
| LoRA +Tr (k=8) | 95.94 / 93.73 | 95.33 / 13.42 | 95.66 / 88.12 | 95.86 / 93.84 |
| LoRA +Cl+Tr (k=8) | 95.55 / 15.62 | 無 | 95.55 / 17.38 | 無 |
| LoRA +Cl+Tr+Pt | 95.83 / 13.20 | 95.99 / 6.49 | 95.83 / 10.67 | 95.83 / 17.05 |
| DoRA baseline | 95.61 / 68.21 | 95.61 / 66.23 | 95.44 / 47.52 | 95.99 / 99.34 |
| DoRA +Cl | 94.95 / 29.04 | 無 | 95.33 / 16.72 | 無 |
| DoRA +Tr (k=8) | 95.61 / 76.35 | 無 | 95.77 / 49.50 | 無 |
| DoRA +Cl+Tr (k=8) | 95.28 / 41.36 | 無 | 94.29 / 19.80 | 無 |
| DoRA +Cl+Tr+Pt | 95.99 / 36.96 | 96.38 / 6.16 | 94.95 / 14.30 | 95.33 / 21.56 |

† 單次 seed=0 結果；11-seed 均值 26.05，最佳（seed=7）13.53——單一數字不代表典型表現，見「機制1消融」章節。

**觀察 1：** InSent 的 LoRA Cl+Tr+Pt 全開，我們的 CA（95.83）與論文 CA（95.83）逐位元相同——
大概率為巧合（兩位小數的重合機率不算低），不視為驗證證據，僅記錄此現象。

**觀察 2：** DoRA+Cl+Tr+Pt 在 InSent 上，我們的 ASR（14.30）低於論文對應數字（21.56）——
這是目前所有對照格子裡，我們唯一一處在「論文有直接測過的精確對照組」上表現優於論文的案例。
原因未探究，可能候選：論文該格也是單次或少次結果、我們的 k=8/Pt「top three layers」等實作
選擇剛好在這個設定下更有效、或純粹雜訊。不做進一步因果推論。

**觀察 3：** 此表使用 k=8（原先消融實驗確認之最佳值）——但見下方「機制2消融」章節，
k 精細掃描發現 k=2 才是真正的最佳值（LoRA+Tr alone: ASR 57.32% vs k=8 的 93.73%）。
此表的 LoRA/DoRA +Tr、+Cl+Tr、+Cl+Tr+Pt 各格若改用 k=2 重跑，數字可能顯著更好——
尚未重跑，總覽表暫不更新，此為已知待辦事項。

---

## 核心發現

1. **Baseline 幾乎完全記得 backdoor**——LoRA/DoRA 在兩種攻擊下 ASR 均落在 47–75% 範圍，
   遠高於論文對應 baseline（多數接近 90–100%），但方向一致：不加任何機制，LoRA 無法自動遺忘。
2. **Cl（clean-strengthened regularization）單獨是三個機制裡效果最強的**——兩個 method、
   兩種攻擊皆從 baseline 大降至 17–29%，且與論文 Table 4 定性結論（Cl alone 最有效）一致。
3. **Tr（trigger-insensitive regularization）單獨效果對 k 高度敏感，且非單調**——k=2 時
   ASR 57.32%（目前最佳），k=1 時 91.53%，k=8 時 93.73%，k=32/1024 更差——存在中間最優值，
   不是「k 越小越好」的單調關係（此結論已由早期消融推翻，見「機制2消融」章節）。
4. **三機制疊加（Cl+Tr+Pt）方向正確，且是唯一能讓 Tr 產生正貢獻的組合**——BadNet 上 Cl+Tr
   明顯優於 Cl 單獨（協同效應），InSent 上則未重現此協同效應（Cl+Tr 略差於 Cl 單獨）。
   注意：此結論基於 k=8，尚未用 k=2 重新驗證協同效應是否依然成立或更強。
5. **數值重現未達成，但方向性重現成立**——所有格子 ASR 高於論文對應值（除 DoRA InSent 全開
   外），落差 1.3–8 倍不等；已知至少五項未排除的落差來源（見「待確認事項」）。

---

## 方法論附註

### evaluate_checkpoint.py 決定性驗證

用同一個 checkpoint 重新評估（backfill_eval.sh），三筆原本只有口頭記錄、未存 log 的舊結果
（LoRA+Tr k=1024、LoRA+Tr k=8/lr=2e-3、LoRA+Tr k=8/lr=2e-5）與 evaluate_checkpoint.py 重新跑出的
結果逐位元一致，證實 evaluation（非訓練）本身是決定性的，可信賴用於事後補測未存檔的 checkpoint。

### 機制3新 scaling s 數值

新 $s$ 值遠大於訓練時原值（LoRA 訓練時 $s=\alpha/r=2$；DoRA 訓練時 $s=1$，peft 預設），
符合論文邏輯（$\sigma_{max}(\Delta W)\ll\sigma_{max}(W_{pre})\Rightarrow$ 新 $s$ 應遠大於訓練時 $s$），
方向被實測驗證：

- BadNet, LoRA: layer22.value=16.05, layer23.query=7.08, layer23.value=9.43
- BadNet, DoRA: layer22.value=13.88, layer23.query=5.65, layer23.value=4.58
- InSent, LoRA: layer22.value=19.61, layer23.query=7.50, layer23.value=8.69
- InSent, DoRA: layer22.value=19.79, layer23.query=10.45, layer23.value=8.74

InSent 上機制3的降幅（LoRA 6.7pp, DoRA 5.5pp）比 BadNet（LoRA 2.4pp, DoRA 4.4pp）更大，
且兩 method 降幅更接近——可能與「機制3 DoRA 理論分析」推導2的封頂效應在此組數據下
影響較小有關，未驗證。

InSent 上 LoRA/DoRA 算出的新 $s$ 值幾乎相同（19.6 vs 19.8, 7.5 vs 10.4, 8.7 vs 8.7）——
這不是理論推導的必然結果（推導只保證公式形式相同，不保證兩個 method 訓練出的
$\sigma_{max}(\Delta W)$ 數值接近），可能暗示兩者訓練出的 $\rho_{eff}$ 相近，
但僅為單組（InSent, seed=0）觀察，未經驗證，不構成規律。

---

## 機制1消融：LoRA+Cl 的 11-seed 變異性驗證（回應 advisor 期望的「~10%」目標）

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

單次 run 的「典型」表現（平均）約 26%，遠高於論文 Table 4 的 Cl alone（10.34%）。
但 best-of-11（seed=7）達到 **13.53%**，已非常接近論文數字。這與論文自身聲明的方法論一致：
「Unless otherwise noted, results in the remaining tables and figures also reflect the best
performance from our repeated experiments.」——論文的 10.34% 很可能同樣是 best-of-N 的結果，
不是穩定的單次表現。

機制1（Cl）本身的效果對 random seed 相當敏感——這個高變異性本身是一個值得報告的觀察，
不應該被「挑最好的一次」這個呈現方式掩蓋掉。誠實的結論是：機制1平均能把 ASR 從 baseline 的
~70% 壓到 ~26%，最佳情況下可達 13.5%（接近論文數字），但單次結果不穩定。

InSent 只跑了單一 seed（16.50%），未做多 seed 驗證，沒有理由假設它比 BadNet 更穩定——
在做類似的多 seed 驗證之前，不對 InSent 與 BadNet 之間的 Cl 效果差異做任何因果解釋。

---

## 機制2消融：Tr（正交懲罰），固定 λ=10

### k 消融（固定 lr=2e-4, LoRA, BadNet）—— 完整版，含精細掃描

| Method | k | Test CA | ASR |
|---|---|---|---|
| LoRA+Tr | 1 | 0.9594 | 0.9153 |
| LoRA+Tr | **2** | 0.9561 | **0.5732**（目前所有 Tr alone 結果中最佳） |
| LoRA+Tr | 3–7 | 尚未測試 | 精細掃描進行中（k_finegrained_sweep.sh） |
| LoRA+Tr | 8 (=r) | 0.9594 | 0.9373 |
| LoRA+Tr | 32（借用 Figure 2 caption 的數字） | 0.9588 | 0.9648 |
| LoRA+Tr | 1024（不截斷，等效完整方陣） | 0.9500 | 1.0000（最差） |

**重要更正：早期僅測 k∈{8,32,1024} 三個稀疏點時，誤判為「k 越小越好，單調遞減」。**
補測 k=1、k=2 後發現非單調——k=2 明顯優於兩側（k=1 與 k=8），顯示存在中間最優值。
仍遠高於論文 13.42%，但方向上有實質進展。k=3–7 精細掃描結果待補。

DoRA 的 k 對照（僅測過 k=8, 32，尚未用 k=1、2 精細掃描驗證是否也存在中間最優值）：

| Method | k | Test CA | ASR |
|---|---|---|---|
| DoRA+Tr | 8 | 0.9561 | 0.7635 |
| DoRA+Tr | 32 | 0.9533 | 0.8526 |

### lr 消融（固定 k=8, LoRA, BadNet）

| lr | Test CA | ASR |
|---|---|---|
| 2e-5 | 0.9605 | 1.0000（太小，訓練不動，backdoor 完全沒被觸動） |
| 2e-4 | 0.9594 | **0.9373**（最佳，主線一直使用的值） |
| 2e-3 | 0.9456 | 訓練崩潰（epoch 1 起 dev acc ~0.49，等同亂猜，數字無意義） |

結論：lr=2e-4 已是三者中最佳，排除「lr 選錯導致機制2效果差」這個假設。
**注意：此 lr 消融是在 k=8 下做的，k=2 是否有不同的最佳 lr，尚未驗證。**

### 理論發現：Eq.10 在滿秩情況下退化成 L2 weight decay（已驗證，非訓練實驗）

論文 Eq.9→Eq.10 的正交懲罰 $\Omega(A,B)=\|U^\top B\|_F^2+\|AV\|_F^2$，若 $U,V$ 取自未截斷的
完整 SVD，在 $W_{pre}$ 為滿秩方陣時，數學上會退化成標準 L2 weight decay
（∵ 完整正交矩陣不改變 Frobenius norm：$\|U^\top B\|_F=\|B\|_F$）。

實測驗證：RoBERTa-large 的 48 個 query/value 層，SVD 後奇異值全數 >1e-6（1024/1024，滿秩）。
論文本身未說明 Eq.10 計算時 $U,V$ 是否截斷、截斷到多少維——這是我們自己的實作決定
（k=1/2/8/32/1024 皆已測試），不是論文明確指定的答案。

### 論文對照

論文 Table 5（RoBERTa+BadNet）Tr alone: CA 95.33 / ASR 13.42。我們目前的 Tr alone 最佳版本
（k=2, lr=2e-4）: CA 95.61 / ASR 57.32——方向一致（CA 相近）但 ASR 仍有落差，
較 k=8 版本（93.73）已大幅縮小落差倍數（從約 7 倍縮小到約 4 倍）。

值得注意：論文裡 Tr alone 只在 RoBERTa+BadNet 這個組合特別有效（13.42），其餘三個
model+attack 組合（RoBERTa+InSent, LLaMA+BadNet, LLaMA+InSent）Tr alone 效果皆遠差
（77–98），顯示這個低點本身在論文裡也不是普遍模式。

---

## 機制3在 DoRA 上的理論分析（使用者原創推導，非論文內容）

DoRA forward：$M^{DoRA}(s)=\dfrac{m}{\|W_{pre}+s\cdot BA\|_c}\cdot(A_0+s\cdot A_1)$，
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
機制3依賴機制1、2先鋪路，不是獨立生效的機制。

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
- 機制2 SVD 截斷 k（論文未明講是否截斷、截斷到多少）——已消融 k∈{1,2,8,32,1024}，
  k=2 目前最佳，但 k=3–7 尚未測完，最終最優值待確認
- 機制3「top three layers」假設為模型最後三層（論文未明講是哪三層）
- 機制3對 DoRA 的 scaling 語義——已完成理論推導（見上方「機制3在 DoRA 上的理論分析」），
  但推導本身的近似項（用 σ_max 代替精確的 trigger-projection A_1）在 DoRA 封頂結構下的
  誤差程度未量化
- 論文數字多為多次 run 取最好；我們除 LoRA+Cl（BadNet）做過 11-seed 掃描外，其餘均為單次 seed=0
- Tr alone 本身仍遠差於論文（k=2 時 57.32% vs 13.42%），疊加後的改善可能主要來自 Cl 而非 Tr，
  尚未做消融拆解驗證此假設；亦尚未用新最佳 k=2 重跑 Cl+Tr、Cl+Tr+Pt 組合

---

## 尚未完成

- k=3–7 精細掃描（k_finegrained_sweep.sh 執行中）—— 確認 k=2 是否為真正最優值
- 用 k=2（而非 k=8）重跑 DoRA+Tr、LoRA/DoRA+Cl+Tr、+Cl+Tr+Pt，更新總覽對照表
- λ 網格（只測過 λ=10，論文網格 {1,5,10,15,20}）—— 決定不繼續追，優先度較低
- threshold-based SVD 截斷（用奇異值門檻而非固定 k）—— 只是想法，沒有實作
- BERT/LLaMA 架構、CR/CoLA 資料集 —— 全部未測試，範圍已與 InSent 部分擴展，
  但仍不含 BERT/LLaMA；LLaMA 因規模/target_modules 命名等多項未驗證假設，風險評估後決定暫不執行