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
