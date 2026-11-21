# W_pre dropout 的理論說法：一層與兩層玩具模型

- 建立：2026-11-17 21:40（台北），Claude，依你 11/17 21:24 的要求。只存在 project，沒有放進 repo，也沒有 push。
- **作者**：全文的推導、數值檢查和實驗設計都是 Claude 的，你還沒讀過。標【你提出】的只有引用你草稿（`claude/talk_1119_material.md` 第一節）的句子。
- **標記**：
  - 【證明】：在寫明的假設下嚴格成立。
  - 【近似推導】：用了 Taylor 展開等近似，近似條件寫在旁邊。
  - 【直覺】：沒有推導。
  - 另外，【資料】是可以從 repo 重算的數字；【程式碼事實】是讀程式確認的。
- **每個結論後面都接一個實驗**，用來在 RoBERTa-large × LoRA 的設定裡證實或否定它（老師的要求第 1 項）。實驗優先只用現有 checkpoint 做評估、不重新訓練；能用 SST-2 train 或 dev 的地方就不用 test。所有用 test 量的數字只給研究者看（`claude/rules.md` 你的規則 2）。
- **理由的來源**：沒有用 RoRA 論文的說法當理由。引用的理論只有 Wager, Wang & Liang (2013) 的二階近似做法（project 裡有 PDF）。
- **和其他文件的關係**：
  - `claude/thesis_narrative_B.md` §9.1 的一、四 (a)、五，已有 squared error 的展開、cross entropy 的二階項和「分類頭」假說；`claude/talk_1119_material.md` 第三節第 1、2 點也是。
  - 本文件把它們寫完整，並更正兩處，見 §2.3 和 §3.5 的「更正」。

---

## 0. 設定與符號

**repo 裡的做法**【程式碼事實：`cl_utils.py`、`variant_finetune.py` 第 365–378 行】
- dropout 是掛在 LoRA 模組的 `base_layer`（凍結的 W_pre）上的 forward hook，所以每個 q／v 模組的輸出是

$$y = \mathrm{Dropout}(W_{\text{pre}}x + b) + \Delta W x, \qquad \Delta W = \tfrac{\alpha}{r}BA .$$

  雜訊只打在凍結的那一支，LoRA 那一支不受影響。
- 只在訓練時開；用 inverted dropout，留下的部分放大 $1/(1-p)$；每個 token 各自抽一次遮罩。
- **逐座標**：遮的是 $W_{\text{pre}}x+b$ 的每個輸出座標，bias 也一起遮。
- **SVD**：遮的是 $c = U^\top(W_{\text{pre}}x)$ 的每個分量；$U$ 是 $W_{\text{pre}}$ 的左奇異向量，bias 不遮。
- 另外兩種一般的 dropout 兩組都有：LoRA 自己的輸入 dropout（0.1），和分類頭裡的 dropout（0.1）。下面的推導只處理凍結那一支的 dropout。

**符號**
- $u(x)$：被遮的那個訊號。逐座標時 $u = W_{\text{pre}}x + b$；SVD 時是 $W_{\text{pre}}x$。
- $\xi$：遮罩造成的雜訊，所以帶雜訊的輸出是 $u + \xi$。
- $\lambda = p/(1-p)$（p = 0.2 時 λ = 0.25）。
- 雜訊的共變異數 $\Sigma(x) = \mathrm{Cov}(\xi \mid x)$：
  - 逐座標：$\Sigma(x) = \lambda\,\mathrm{diag}(u_1^2, \dots, u_{d}^2)$。
  - SVD：$\Sigma(x) = \lambda \sum_r c_r(x)^2\, U_r U_r^\top$。

**性質 (P)**【證明】
- $\mathbb{E}[\xi \mid x] = 0$。
- $\Sigma(x)$ 只由 $x$、$W_{\text{pre}}$、$b$、$p$ 決定，不含任何被訓練的參數（$\Delta W$、分類頭）。
- 證明：照定義，遮罩 $m_j \sim \text{Bernoulli}(1-p)$ 和所有參數獨立；$\mathbb{E}[m_j/(1-p) - 1] = 0$；$\mathrm{Var}[m_j/(1-p)] = \lambda$。

*實驗 0（約十分鐘，GPU）*：在一個 PZ 上開著 hook，同一句話抽 K = 1,000 次遮罩，檢查每個模組輸出的平均是否等於關掉 dropout 的輸出，共變異數是否等於上面的式子。這是程式碼檢查，不是科學問題；放在這裡，是因為下面每一節都靠這個性質。

---

## 1. 一層網路、squared error

**模型**：輸出 $z = u(x) + \xi + \Delta W x$，loss $\ell = \tfrac12\lVert t - z\rVert^2$，$\Delta W$ 被訓練，$W_{\text{pre}}$ 凍結。

**結論 1.1**【證明】期望 loss 只多一個和 ΔW 無關的常數。

$$\mathbb{E}_\xi\,\ell = \tfrac12\lVert t - u - \Delta W x\rVert^2 + \tfrac12\,\mathrm{tr}\,\Sigma(x).$$

- 推導：展開平方；交叉項 $-(t-u-\Delta Wx)^\top \mathbb{E}\xi = 0$；剩下 $\tfrac12\mathbb{E}\lVert\xi\rVert^2 = \tfrac12\mathrm{tr}\,\Sigma(x)$。
- 依 (P)，最後一項不含 ΔW。逐座標時它等於 $\tfrac{\lambda}{2}\lVert W_{\text{pre}}x + b\rVert^2$，和 `thesis_narrative_B.md` §9.1 一的式子相同（那裡沒寫 bias）。

**結論 1.2**【證明】每一步的隨機梯度都不偏。
- 對 ΔW：$\nabla_{\Delta W}\ell = -(t - z)x^\top$。它對 $\xi$ 是仿射的，所以 $\mathbb{E}_\xi\nabla_{\Delta W}\ell$ 等於不加 dropout 的梯度。
- LoRA 的參數化也一樣：
  - $\nabla_B\ell = -\tfrac{\alpha}{r}(t-z)(Ax)^\top$；
  - $\nabla_A\ell = -\tfrac{\alpha}{r}B^\top(t-z)x^\top$；
  - 兩者都對 $\xi$ 仿射。
- LoRA 自己的輸入 dropout 用的是另一個獨立的遮罩；以它為條件，結論不變。
- 所以「對 ΔW 沒有系統性作用」的精確意思是三件事：期望 loss 只差一個常數；期望梯度場完全相同；最小值的位置相同。
- 它**沒有**說：有限步長、帶雜訊的訓練會走出同一條軌跡。dropout 多加的梯度雜訊，共變異數約是 $\Sigma(x)\otimes xx^\top$。

**結論 1.3**【近似推導】用 Adam 時，不偏但更吵的梯度會讓步伐變小。
- Adam 每個座標的步伐約是 $\eta\,\bar g_j/\sqrt{v_j}$，$v_j \approx \mathbb{E}[g_j^2] = \bar g_j^2 + \mathrm{Var}(g_j)$。
- dropout 不改 $\bar g_j$（結論 1.2），只讓 $\mathrm{Var}(g_j)$ 變大，所以步伐縮成約 $\sqrt{(\bar g_j^2 + V_0)/(\bar g_j^2 + V_0 + V_{\text{drop}})}$ 倍。
- 近似條件：$\beta_2$ 的平均窗口比 $\bar g$ 變化的時間尺度長；$\epsilon$ 可忽略。
- 這條路連一層、squared error 都成立；期望值分析看不到它（`talk_1119_material.md` 第三節第 2 點已提過）。
- 預測：加 dropout 時 ΔW 比較小。【資料：`logs/dw_last.txt`】σmax(ΔW)/σmax(W) 的 48 模組平均，普通 LoRA 0.260、逐座標 0.229、SVD 0.235；同 PZ、同 seed 配對，較小 15/15、14/15。和預測相符，但不能分辨這條路和其他會讓 ΔW 變小的機制。

**結論 1.4**【推論，靠資料】光靠「步伐變小」解釋不了 ASR 下降。
- 理由：普通 LoRA 用較小的學習率讓更新變小，後門反而留得更多。
- 【資料：`compare_s5_output.log`】persistence（epoch 15–19 平均的 test ASR，即 log 的診斷欄 `last5`），lr 2e-4 → 1e-4 → 5e-5：
  - badnet_s0：0.399 → 0.743 → 0.910；
  - badnet_s2：0.711 → 0.846 → 0.914；
  - oldpz：0.832 → 0.880 → 0.988。
- 那組是另一批 PZ、seed 3–5、best-dev 規則；而且那個實驗是為正交懲罰預先登記的，拿來推 dropout 是事後的次要解讀。

*實驗 1A（梯度偏差；只評估，約 40 分鐘）*
- **回答什麼**：一層 squared error 的結論（同一層的 ΔW 收到的期望梯度不變），在 RoBERTa 裡偏離多少。
- **做法**：取 15 個普通 LoRA 的 `last/` checkpoint，在 SST-2 train 抽 512 句。對每一層的 LoRA（A、B）和分類頭，算三種梯度：
  - (a) 不開 dropout；
  - (b) 開逐座標 p 0.2，對 K = 32 組遮罩取平均；
  - (c) 和 (b) 一樣，但只在第 ℓ 層開 dropout，看第 ℓ 層自己的 LoRA 和下游的差別。
  - 再把 loss 換成 logits 上的 squared error 重做一次，分開「深度」和「cross entropy」各自的貢獻。
- **讀法**：偏差比 $\lVert\bar g_{\text{drop}} - g_0\rVert/\lVert g_0\rVert$，誤差用遮罩重抽估計的 Monte Carlo 標準誤，超過 3 倍標準誤才算有偏。
  - 只在第 ℓ 層開 dropout 時，同一層的偏差接近 0、下游有偏 → 支持「一層的結論在同一層近似成立，作用在下游」。
  - 同一層就明顯有偏 → 一層的結論不能外推到 RoBERTa。

*實驗 1B（Adam 的步伐；和 1A 同一批前向、反向計算）*
- 由 1A 的每座標 $\bar g$、$\mathrm{Var}(g)$，算結論 1.3 預測的步伐比，和實測的 ΔW 大小比（0.229/0.260 ≈ 0.88、0.235/0.260 ≈ 0.90）對照。
- 接近 → 優化器那條路足以說明 ΔW 為什麼小約一成；差很多 → 還有別的原因。
- 限制：只有最後一個 epoch 的 checkpoint，梯度統計不一定代表整段訓練。

---

## 2. 一層網路、cross entropy

**模型**：logits $z = u(x) + \xi + \Delta W x \in \mathbb{R}^K$，$\ell = A(z) - z_y$，$A(z) = \log\sum_k e^{z_k}$。二元時，margin $m = z_1 - z_0$（label 1 是正面），margin 上的雜訊 $\eta = \xi_1 - \xi_0$。

**結論 2.1**【證明】多出來的項含 ΔW，所以有系統性作用。

$$\mathbb{E}_\xi\,\ell = \ell(\bar z) + R(\bar z), \qquad R(\bar z) = \mathbb{E}\,A(\bar z + \xi) - A(\bar z) \ge 0, \qquad \bar z = u + \Delta W x.$$

- 線性項 $-z_y$ 的期望不變。$A$ 是凸函數，由 Jensen 不等式得 $R \ge 0$。
- 只有當 $\xi$ 幾乎必然落在全 1 向量的方向上時，$R = 0$；$A$ 只沿這個方向是線性的。
- $R$ 依賴 $\bar z$，而 $\bar z$ 含 ΔW，所以一般情況下 dropout 會改變 ΔW 的期望梯度。

**結論 2.2**【近似推導】二階形式。

$$R \approx \tfrac12\,\mathrm{tr}\big(H(\bar z)\,\Sigma(x)\big), \qquad H = \mathrm{diag}(\pi) - \pi\pi^\top,\ \ \pi = \mathrm{softmax}(\bar z).$$

- 二元時：$R \approx \tfrac12\,\sigma'(\bar m)\,v(x)$，其中 $\sigma'(m) = \sigma(m)(1-\sigma(m))$，$v(x) = \mathrm{Var}(\eta)$。逐座標時 $v = \lambda(u_1^2 + u_0^2)$。
- 這和 `thesis_narrative_B.md` §9.1 四 (a) 的式子一致。
- **近似條件**：margin 雜訊的標準差 $\sqrt{v} \lesssim 0.5$。inverted Bernoulli 雜訊是偏斜的，三階項是 $\tfrac16\ell'''(\bar m)\,\kappa_3$，其中 $\ell''' = \sigma'(1-2\sigma)$，$\kappa_3 = -\tfrac{p(1-2p)}{(1-p)^2}\sum_i (w_iu_i)^3$（$w$ 是讀出 margin 的方向：一層時 $w = e_1 - e_0$；§3 的兩層模型裡是分類頭的方向）。
- 數值檢查（§5 A）：
  - 標準差 0.27–0.43 時，二階式和 Monte Carlo 差 0–12%；
  - 0.6–0.9 時差 2–10%；
  - 1.8–2.8 時最多差 71%。
  - 加上三階項沒有一致的改善：標準差 0.9 以下的 6 句裡，2 句變好、3 句變差、1 句差不多。

**結論 2.3**【近似推導】每一句在 margin 上的推力方向推得出來：往「更有把握」推。

$$\partial R/\partial \bar m = \tfrac12\,v(x)\,\sigma'(\bar m)\,(1-2\sigma(\bar m)).$$

- $\bar m > 0$ 時為負，gradient descent 把 $\bar m$ 往上推；$\bar m < 0$ 時往下推。
- 這個推力和 label 無關：判錯的句子會被往錯的那一側推得更遠，和資料項相反。
- 推力在 $|\bar m| = \ln(2+\sqrt3) \approx 1.32$ 時最大，在 $\bar m = 0$ 和 $|\bar m| \to \infty$ 時是 0。
- **更正**【你提出】的「cross entropy 下有作用，但大小和方向推不出來」，以及 `talk_1119_material.md` 第三節第 1 點 Claude 寫的「方向推不出來」，都說得太強：
  - 推得出來的：每一句在 margin 上的推力方向和大小（近似下）。
  - 推不出來的：ΔW 在權重空間的淨改變。它是 $\sum_n c_n\,e\,x_n^\top$（$e$ 是正面減負面的方向），每一項的正負取決於那一句當下的 margin；所以「ΔW 變大還是變小、往哪轉、最後乾淨 margin 變大還是變小」要看資料分布。
  - 例：全部句子都判對時，額外推力和資料項同向；有判錯的句子時，在那些句子上兩者相反。

**結論 2.4**【證明】一層的目標函數碰不到 trigger。
- trigger 在輸入端是 $x + \delta$，微調資料裡從不出現。
- **整個矩陣的 ΔW，用 gradient descent 從 $\Delta W_0$ 出發**：兩個 loss 項的梯度都是 $c(x_n)\,x_n^\top$ 的形式（資料項和 $R$ 都是），所以 $\Delta W - \Delta W_0$ 的每一列都落在 $S_x = \mathrm{span}\{x_n\}$ 裡，而

$$\Delta W\delta = \Delta W_0\delta + (\Delta W - \Delta W_0)\,P_{S_x}\delta .$$

  加不加 dropout 都一樣：訓練只能透過 $\delta$ 落在乾淨輸入張成空間裡的那一部分 $P_{S_x}\delta$ 影響 trigger。
- **LoRA**（$B_0 = 0$，$A_0$ 隨機）：
  - $A - A_0$ 的每一列也落在 $S_x$ 裡；
  - 所以 $\Delta W\delta = \tfrac{\alpha}{r}B\,(A_0\delta + (A-A_0)P_{S_x}\delta)$；
  - 其中 $BA_0\delta$ 是隨機初始化漏進來的部分，目標函數不針對它。
- 【你提出】的「trigger 不在微調資料裡，一層推導本來就碰不到它」在這個意義下成立。
- **Adam** 的逐座標縮放會讓更新離開 $S_x$，這個子空間性質就不再嚴格成立；離開多少沒有推導【直覺】。

*實驗 2A（訓練句的 margin 往哪走；只評估，約 15 分鐘）*
- **回答什麼**：一層 cross entropy 的推力（把訓練句的 |margin| 推大），是否是 RoBERTa 裡的主要效果。
- **做法**：50 個設定（45 個 `last/` + 5 個 PZ），在 SST-2 train 的 5,000 句上算 margin；同 PZ、同 seed 配對比較 dropout 和普通 LoRA 的 median |m|。
- **讀法**：
  - dropout 的 |m| 較大，至少 12/15 對 → 支持結論 2.3 是主要效果。
  - 較小，至少 12/15 對 → 否定，改支持 §3 的「分類頭被懲罰」。
- **門檻來源**：配對符號檢定，15 對中至少 12 對，單尾 p ≈ 0.018（二項分布，H0 機率 0.5）。
- **已知的線索**【資料：`summary_pz.csv`】：test 的乾淨正面句上，dropout 的 median m 較小，30/30 對（plain 9.9–11.1，dropout 7.2–8.5）。這是 test，不是 train；而且是 logit 單位，受尺度影響。

*實驗 2B（trigger 和乾淨輸入的重疊；只評估，約 5 分鐘）*
- **回答什麼**：結論 2.4 的 $P_{S_x}\delta$ 在 RoBERTa 裡有多大。
- **做法**：在每一層 q／v 模組的輸入端，取 SST-2 train 句子的 token 表示，算共變異數 $C_\ell$。實際上「張成空間」幾乎是滿的，所以改量能量：把「mn」插進 SST-2 dev 的正面句（不用 test），量 trigger token 的表示 $\delta_\ell$ 的相對能量 $\delta_\ell^\top C_\ell \delta_\ell/(\lVert\delta_\ell\rVert^2\,\mathrm{tr}C_\ell/d)$，用隨機方向當基準。
- **讀法**：遠低於隨機 → trigger 幾乎不被乾淨資料的梯度碰到，一層推導「碰不到 trigger」在 RoBERTa 也近似成立；接近或高於隨機 → 乾淨訓練本來就會動到 trigger 的方向。

---

## 3. 兩層玩具模型

**模型**（照你的規格）
- 隱藏層 $h = u(x) + \xi + \Delta W x$：凍結的第一層 $W_{\text{pre}}$，加上被訓練的低秩更新 ΔW，中間沒有激活函數。
- 輸出 $z = Vh + c$：被訓練的線性分類頭，loss 是 cross entropy。
- 只在訓練時對 $u$ 做 inverted dropout。
- 二元時，$w = V_1 - V_0$，$\bar m = w^\top(u + \Delta W x) + c_1 - c_0$，$q(x) = \sigma'(\bar m)$。

**後門**
- trigger 在輸入端是 $\delta$，訓練資料裡從不出現。
- 假設 (T)：$\Delta W\delta = 0$，也就是 trigger 只透過 $W_{\text{pre}}$ 進來。整個矩陣的 ΔW 從 0 出發、$\delta \perp S_x$、用 gradient descent 時，這由結論 2.4 保證。
- $b := W_{\text{pre}}\delta$：trigger 在隱藏層的表示。
- 測試時的**拉力** $s = m(x) - m(x+\delta) = -w^\top b$；在線性模型和 (T) 下和 $x$ 無關，$s > 0$ 表示往後門目標（負面）推。
- ASR = 乾淨正面句中 $m(x) \le s$ 的比例。**不受尺度影響的後門分數**是 $s/m$（對應我們量的 median(s)/median(m)）。

**結論 3.1**【證明】logits 上的雜訊是 $V\xi$，共變異數 $V\Sigma(x)V^\top$：依賴分類頭，不依賴 ΔW（由 (P) 直接得到）。

**結論 3.2**【近似推導】兩個被訓練的部分受到的有效正則化不同。

$$\mathbb{E}_\xi\,\ell \approx \ell(\bar m) + R, \qquad R = \tfrac12\,\mathbb{E}_x\big[q(x)\,w^\top\Sigma(x)\,w\big].$$

- **分類頭**：
  - $\partial R/\partial w = Dw + \tfrac12\mathbb{E}_x[q'(\bar m)\,(w^\top\Sigma w)\,\bar h(x)]$，其中 $D := \mathbb{E}_x[q(x)\Sigma(x)]$，第二項落在 $\mathrm{span}\{\bar h(x_n)\}$ 裡。
  - 第一項是依資料加權的 L2：分類頭讀「凍結那一支在不確定的訓練句上能量大」的方向越多，懲罰越大。
  - 逐座標時 $D = \lambda\,\mathrm{diag}(\mathbb{E}[q\,u_i^2])$；SVD 時 $D = \lambda\sum_r \mathbb{E}[q\,c_r^2]\,U_rU_r^\top$。
- **ΔW**：
  - $\partial R/\partial \Delta W = \tfrac12\mathbb{E}_x[q'(\bar m)\,(w^\top\Sigma w)\,w\,x^\top]$。
  - 只有結論 2.3 那種「推向更有把握」的力量，**沒有**收縮 ΔW 的項，因為 ΔW 的輸出從來不在雜訊裡。
- **對照**：如果 dropout 打在整個 $h$ 上（兩支都遮），雜訊會含 $(\Delta Wx)_i^2$，ΔW 的輸出也會被懲罰。現在只遮凍結那一支，所以懲罰只落在「讀凍結輸出的分類頭」上。
- 和 Wager et al. (2013) 的關係：他們的結果是「讀被遮特徵、正在被訓練的權重受到 L2」；這裡讀被遮特徵的是分類頭，所以 L2 落在分類頭，不在 ΔW。
- 近似條件同結論 2.2。

**結論 3.3**【證明】只訓練分類頭、不加 dropout 時，拉力完全不變。
- 假設：ΔW 不被訓練（或它的輸出從來沒有沿 $b$ 的分量），$b \perp u(x_n)$ 對所有訓練句成立，用 gradient descent（或 gradient flow）。
- 那麼 $s(t) = s(0)$。
- 理由：資料項對 $w$ 的梯度落在 $\mathrm{span}\{\bar h(x_n)\} \perp b$。
- 數值檢查（§5 C）：2.42 → 2.42。

**結論 3.4**【證明：恆等式；正負號要多一個假設】ΔW 一起訓練時，拉力會跟著網路的尺度一起長。
- ΔW 的梯度是 $g\,w\,x^\top$，所以 $d\Delta W/dt = -\eta\,w(t)\,G(t)^\top$，$G = \mathbb{E}[g\,x]$，$g = \sigma(m) - y$。ΔW 從 0 出發，且不加 dropout 時：

$$b^\top\Delta W(t) = \eta\int_0^t s(\tau)\,G(\tau)^\top d\tau, \qquad \frac{ds}{dt}\Big|_{\text{資料項}} = \eta^2\int_0^t s(\tau)\,\langle G(\tau), G(t)\rangle\,d\tau .$$

- 只要前後各步的 $G$ 正相關（多數訓練的情況，這是假設），正的拉力就會隨訓練變大：ΔW 往分類頭的方向寫，而分類頭帶有沿 $b$ 的後門分量。
- 數值檢查：不加 dropout，拉力 2.42 → 5.38，同時乾淨 median margin 長到 17.4。
- 所以 ΔW 一起訓練時，要比的是 $s/m$，不是 $s$。

**結論 3.5**【近似推導】凍結那一支的 dropout 會抵消拉力。
- 它在 $ds/dt$ 上多加一項 $\eta\,b^\top D\,w$。
- 若 $b$ 是 $D$ 的特徵向量，特徵值 $d_b = \mathbb{E}_x[q(x)\,\hat b^\top\Sigma(x)\hat b] > 0$，且 $s > 0$，這一項就是 $-\eta\,d_b\,s$：抵消拉力，速度和「凍結那一支在不確定的乾淨訓練句上，沿 trigger 方向的雜訊能量」成正比。
- **只訓練分類頭**：拉力往 0 衰減。數值檢查：p 0.05 時 2.42 → 0.79；二階預測 1.08，低估了衰減，p 越大低估越多。
- **ΔW 一起訓練**：衰減和結論 3.4 的增長互相抵銷；拉力長得比較少，$s/m$ 下降。數值檢查：p = 0、0.05、0.1、0.2 時，$s/m$ = 0.310、0.267、0.229、0.156；二階預測的 $s$ 和實際差不到 0.05。
- 乾淨 margin 也會變小一些：$D$ 也收縮讀乾淨方向的分類頭權重，但有資料項頂著。數值檢查：17.4 → 15.6。
- **更正**：`thesis_narrative_B.md` §9.1 五（Gemini 的敘事版，標為假說）說「分類頭學會把 W_pre 的音量調小」。依這個推導，更精確的說法是：
  - 被調小的是分類頭讀「凍結那一支在訓練時晃得大」的方向，不是 W_pre 本身；
  - 只有當 trigger 的表示也落在那些方向上（下面條件 C2），後門才跟著被打折。
  - 那段最後一句「精確的說法」已經指出這一點；它和本結論相容。

**在什麼條件下，這個正則化會壓低測試時的後門分數**（結論 3.5 的條件）

| 條件 | 內容 | 不成立時 |
|---|---|---|
| C1 | (T)：trigger 只透過 W_pre 進來，$\Delta W\delta \approx 0$ | $w^\top\Delta W\delta$ 會加進拉力，這個正則化管不到 |
| C2 | $d_b > 0$：乾淨訓練句在凍結那一支，沿 trigger 表示 $b$ 的方向有雜訊能量。逐座標：$b$ 所在的座標，乾淨句的凍結輸出夠大。SVD：$b$ 在乾淨輸入有能量的左奇異方向上有分量。$b$ 仍可以和每一個乾淨輸出都正交（§5 C 就是這種情況） | trigger 落在乾淨資料從不激發的座標或方向上，沒有衰減 |
| C3 | $D$ 不把 $b$ 和分類頭反向的方向耦合在一起：$b^\top Dw$ 和 $b^\top w$ 同號（充分條件：$b$ 是 $D$ 的特徵向量） | 衰減可能變慢，甚至反向 |
| C4 | 乾淨訓練句維持不確定（$q > 0$）夠久，累積的 $\int\eta\,d_b\,dt$ 夠大 | 資料很快被配得很有把握時，$q \to 0$，正則化跟著消失 |
| C5 | Taylor 近似成立：margin 雜訊標準差 ≲ 0.5 | 只能靠數值；§5 C 在更大的 p 下方向仍相同，衰減比預測強 |
| C6 | 優化器是 gradient descent | AdamW 的 weight decay 兩組都會讓拉力衰減，組間差別仍是 $D$；Adam 的逐座標縮放破壞結論 3.3 的嚴格不變【直覺】 |

這些條件成立時，下降的是 $s/m$。ASR 會不會跟著降，還要看乾淨 margin 的分布，因為 ASR = $P(m \le s)$。

**結論 3.6**【近似推導，事後】和我們量到的形態一致，但不是獨立的檢驗。
- 一致的地方【資料：`counterfactual_checkpoints.csv`、`summary_pz.csv`】：
  - 乾淨 margin 變小，30/30 對；
  - 拉力相對 margin 變小，$s/m$ 在 5/5 PZ 都下降；
  - 只換入 dropout 的拉力，ASR 降得比實際總降幅還多，30/30 對。
- 但這些推導是看過資料之後才寫的，所以這不算獨立的檢驗。要算檢驗，得用下面實驗 3C 那種事先寫好的預測。

**結論 3.7**【直覺】玩具模型搬到 RoBERTa 時有兩處對不上。
- 玩具的「分類頭」對應的是某個 q／v 輸出下游所有被訓練的參數：後面各層的 LoRA，加上分類頭。雜訊要先穿過凍結的 attention、FFN、LayerNorm（都非線性）才到得了它們。
- trigger 是從最底層的 token 進來的，每一層的 ΔW 都看得到它，所以 C1 不一定成立。

*實驗 3A（交換分類頭與分層交換；只評估，約 15 分鐘）*
- **回答什麼**：拉力的下降跟著「讀雜訊的那一方」走（結論 3.2、3.5），還是跟著同一層的 ΔW 走。
- **做法**：同 PZ、同 seed，把三組（plain、逐座標 0.2、SVD 0.2）的分類頭和 48 個 ΔW 交叉組合成 3 × 3，新增 6 種 × 15 對 = 90 個設定，用 `margin_eval.py` 同樣的方式量 median s、$s/m$ 和 ASR。再做分層版：第 ℓ 層以上的 ΔW 和分類頭用 dropout 那組，以下用 plain，掃 ℓ。
- **讀法**：
  - $s/m$ 主要跟著分類頭和上層走 → 支持。
  - 主要跟著底層的 ΔW 走 → 否定玩具模型的機制，或表示 C1 不成立。
- **門檻**：每個比較至少 12/15 對同向（同上，單尾 p ≈ 0.018）。

*實驗 3B（C1：trigger 走哪一支；只評估，約 5 分鐘）*
- **做法**：在每個 q／v 模組，把插入「mn」前後輸入表示的差 $\Delta h_\ell$ 拆成兩支，算 $\lVert\tfrac{\alpha}{r}BA\,\Delta h_\ell\rVert / \lVert W_{\text{pre}}\Delta h_\ell\rVert$。用 SST-2 dev 的正面句插入「mn」，45 個 `last/`。
- **讀法**：底層就有大比例 → C1 在 RoBERTa 不成立，玩具模型的結論要打折。

*實驗 3C（C2、C3：在分類頭的輸入端量 D 和 $d_b$；只評估，約 45 分鐘）*
- **做法**：把分類頭 `out_proj` 的兩列之差當作 $w$，它的輸入（tanh 之後的特徵）當作 $h$。
  - 在 SST-2 train 抽 500 句，開訓練時的那種 dropout，抽 K = 32 組遮罩，估計特徵的雜訊共變異數，得到 $D_f = \mathbb{E}[q\,\Sigma_f]$。
  - 在 SST-2 dev 的正面句插入「mn」，估計 trigger 方向 $b_f$。
  - 算 $\rho_b = \hat b_f^\top D_f\hat b_f / (\mathrm{tr}D_f/\dim)$，以及 $b_f^\top D_f w$ 的正負號（C3）。
- **事先寫好的預測**：
  - C2：普通 LoRA 和 PZ 上的 $\rho_b$ 明顯大於 0；
  - C3：$b_f^\top D_f w$ 和 $b_f^\top w$ 同號；
  - 跨 PZ：$\rho_b$ 越大，dropout 讓 $s/m$ 降得越多。只有 5 顆 PZ，相關係數很弱，只能當線索。
- 這是對結論 3.5 的獨立檢驗，彌補結論 3.6 是事後的問題。

*實驗 3D（被懲罰的量本身：對雜訊的敏感度；只評估，約 40 分鐘）*
- **回答什麼**：$R$ 懲罰的是 $q\cdot\mathrm{Var}(m)$。如果這個機制在運作，dropout 訓練出的模型，對「評估時打開同樣的 dropout」應該比較不敏感。
- **做法**：50 個設定，SST-2 dev 872 句，各抽 K = 16 組遮罩，量每句的 $\mathrm{sd}(m)/|m|$（不受尺度影響），比較 median。
- **讀法**：dropout 那兩組較小，至少 12/15 對 → 支持；沒有變小 → 二階懲罰在 RoBERTa 裡沒有作用，§3 的機制被否定。

*實驗 3E（結論 3.3、3.4：分類頭讀 trigger 的那一份怎麼變；只評估，和 3C 共用計算）*
- **做法**：用 3C 的特徵：
  - 取乾淨特徵的主成分，張成能量 99% 的子空間；
  - 把 trigger 方向 $b_f$ 拆成子空間內、子空間外兩部分；
  - 比較 PZ 的分類頭 $w_0$ 和微調後的 $w$，各自對子空間外那部分的讀取量；用 $\lVert w\rVert$ 正規化，處理尺度問題。
- **預測**：普通 LoRA 保留較多 $w_0$ 的讀取量；dropout 那兩組保留較少，至少 12/15 對。
- **限制**：特徵本身也隨 ΔW 改變；這個實驗只看分類頭這一層。

---

## 4. 一覽

| 結論 | 層級 | 實驗 | 支持／否定的判準 | 成本 |
|---|---|---|---|---|
| (P) 雜訊不偏，共變異數不含被訓練的參數 | 證明 | 0 | 平均和共變異數對得上 | 約 10 分鐘 |
| 1.1–1.2 squared error 下同一層的期望梯度不變 | 證明 | 1A | 同一層的偏差在 3 倍 Monte Carlo 標準誤以內 | 約 40 分鐘 |
| 1.3 Adam 讓步伐變小 | 近似推導 | 1B | 預測的步伐比 ≈ 實測的 0.88／0.90 | 和 1A 共用 |
| 1.4 步伐變小解釋不了 ASR 下降 | 推論（資料） | 已有：`compare_s5_output.log` | — | 0 |
| 2.1 cross entropy 下有系統性作用 | 證明 | 1A（cross entropy 和 squared error 對照） | cross entropy 的偏差明顯大於 squared error | 和 1A 共用 |
| 2.2–2.3 每一句往「更有把握」推 | 近似推導 | 2A | 訓練句的 \|m\| 變大，至少 12/15 對 | 約 15 分鐘 |
| 2.4 一層碰不到 trigger | 證明（gradient descent） | 2B | trigger 的相對能量遠低於隨機方向 | 約 5 分鐘 |
| 3.1–3.2 懲罰落在分類頭，不在 ΔW | 近似推導 | 3D、3A | 敏感度下降至少 12/15 對；$s/m$ 跟著分類頭走 | 約 40 + 15 分鐘 |
| 3.3–3.4 沒有 dropout 時，子空間外的讀取量保留，並隨尺度長 | 證明（玩具） | 3E | 普通 LoRA 保留較多讀取量 | 和 3C 共用 |
| 3.5 和條件 C1–C3 | 近似推導 | 3B、3C | 底層 ΔW 的比例小；$\rho_b > 0$；同號 | 約 5 + 45 分鐘 |
| 3.6 和量到的形態一致 | 近似推導（事後） | 不是獨立檢驗，看 3C | — | — |
| 3.7 搬到 RoBERTa 的對應 | 直覺 | 3A 的分層版 | 找得到拉力開始跟著換的那一層 | 和 3A 共用 |

**成本的依據**【資料：`logs/margin_eval_20261117_180625.log`】
- 50 個設定，每個 2,730 句（1,821 句乾淨 + 909 句插入 trigger），在一張 RTX 6000 Ada 上約 456 秒，也就是每秒約 300 句前向。
- 需要遮罩重抽的實驗乘上 K；需要反向的，Claude 粗估慢約 3 倍，沒有實測。

---

## 5. 數值檢查（玩具模型，不是證明）

- **設定**：輸入 12 維，乾淨資料只在前 10 維；trigger 是第 11 維。隱藏層 30 維。
  - 乾淨的凍結輸出落在座標 0–19 裡的一個 10 維子空間；trigger 的表示 $b$ 也在座標 0–19，但和那個子空間正交，也就是和每一個乾淨的凍結輸出都正交，最大內積 1e-14。
  - 400 句，標籤由前 10 維的一個線性規則決定。
  - 分類頭的初始值帶有拉力 2.42；ΔW 是整個矩陣，從 0 出發；full-batch gradient descent，每一步重抽遮罩，4,000 步，步長 0.05。
- **A. 二階近似**（Monte Carlo，40 萬組遮罩）：
  - margin 雜訊標準差 0.27–0.43 時，二階式和 Monte Carlo 差 0–12%；
  - 0.6–0.9 時差 2–10%；
  - 1.8–2.8 時最多差 71%：二階式在一句上只有 Monte Carlo 的 29%（0.0018 對 0.0062）。
  - 加上三階項沒有一致的改善：標準差 0.9 以下的 6 句裡，2 句變好、3 句變差、1 句差不多。
- **B. 一層 squared error**：遮罩取平均後的隨機梯度，和不加 dropout 的梯度最大差 2.4e-3；最大的 Monte Carlo 標準誤是 2.5e-3。
- **C. 訓練後的拉力 $s$、乾淨正面句的 median margin、$s/m$**：

| | p | 拉力 $s$（起點 2.42） | 二階預測的 $s$ | 乾淨 median margin | $s/m$ |
|---|---|---|---|---|---|
| 只訓練分類頭 | 0 | 2.42 | 2.42 | 6.75 | 0.359 |
| | 0.05 | 0.79 | 1.08 | 5.71 | 0.139 |
| | 0.1 | 0.01 | 0.48 | 4.76 | 0.002 |
| | 0.2 | −0.36 | 0.34 | 3.33 | −0.109 |
| 分類頭和 ΔW 一起訓練 | 0 | 5.38 | 5.38 | 17.36 | 0.310 |
| | 0.05 | 4.58 | 4.54 | 17.17 | 0.267 |
| | 0.1 | 3.82 | 3.77 | 16.73 | 0.229 |
| | 0.2 | 2.42 | 2.39 | 15.57 | 0.156 |

- 訓練準確率：只訓練分類頭時 0.970–0.993；一起訓練時 0.998–1.000。全程 $\Delta W\delta = 0$。
- 程式：Claude sandbox 的 `toy_check.py`（numpy，執行約 8 秒），沒有放進 repo；要的話可以放進去。

---

## 6. 這份文件沒做的、限制

- 沒有推導多層、非線性的情況。結論 3.7 只是直覺；把玩具模型搬到 RoBERTa，要靠實驗 3A–3E 檢驗。
- 只推了二元分類和逐座標形式的細節；SVD 形式只換掉 $\Sigma(x)$，結構相同。
- 二階近似在 margin 雜訊大時不準（§5 A）。RoBERTa 裡雜訊在 margin 上有多大，實驗 3D 會順便量到。
- 結論 3.3、3.4 的嚴格敘述只對 gradient descent 成立；我們用的是 AdamW。
- 實驗都還沒跑，成本是估計。哪幾個要跑、在什麼時候跑，要你決定。
