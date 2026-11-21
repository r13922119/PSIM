# 9/7 簡報內容整理

**標記慣例**
- 🟢 論文明確內容
- 🟡 Claude 的整合/推論（不是任何論文的主張）
- 🔴 尚未完成 / 開放問題 / 需要老師知道你還沒做到這裡

老師不是這個領域的研究者，所以每一節先給他能懂的一句話，再給細節。

---

## 1. What is LoRA and why is it important? 🟢

**一句話：** LoRA 把「微調一個大模型」變成「只訓練一個很小的低秩修正」，大幅降低計算與記憶體成本，同時效果接近全微調。

**細節：**
- 給定 pretrained weight $W_{pre}\in\mathbb{R}^{d\times d}$，LoRA 不直接更新 $W_{pre}$，而是引入
$$W' = W_{pre} + s\,\Delta W,\qquad \Delta W = BA,\ A\in\mathbb{R}^{r\times d},\ B\in\mathbb{R}^{d\times r},\ r\ll d,\ s=\alpha/r$$
- 可訓練參數從 $d^2$ 降到 $2dr$。
- 重要性：是目前 LLM 參數高效微調（PEFT）的事實標準，讓一般規模的實驗室也能微調大模型。

---

## 2. What is RoRA and why is it important? 🟢

**一句話：** RoRA 這篇論文問「為什麼 LoRA 沒辦法把 pretrained model 裡的 backdoor 洗掉」，答案不是傳統認為的「rank 太低」，而是「頻譜強度不夠、頻譜方向沒對齊」；RoRA 是根據這個診斷設計出來的修正方法。

**威脅模型：** pretrained model 已經被下毒（含 trigger→目標標籤的關聯），下游只用乾淨資料做 LoRA 微調，希望能把 backdoor 洗掉。

**核心理論（Proposition 4.2，scaling threshold）：**
- 定義 clean alignment $\rho_{cl}$、trigger alignment $\rho_{tr}$、effective gap $\rho_{eff}=\rho_{cl}-\rho_{tr}$
- 若 $\rho_{eff}>0$，則存在門檻 $s^\star = \dfrac{\rho_{bd}}{\rho_{eff}}\cdot\dfrac{\sigma_{pre}}{\sigma_\Delta}$，只要 scaling factor $s>s^\star$，triggered input 的 margin 會變正，backdoor 被遺忘。
- 標準 LoRA 兩個條件都不利：(i) $\sigma_{max}(\Delta W)\ll\sigma_{max}(W_{pre})$（頻譜強度太弱）(ii) 與 pretrained 方向仍有殘餘 alignment（$\rho_{tr}$ 不夠小）。

**RoRA 的三個機制：**
1. Clean-strengthened regularization：$W=\text{Dropout}_p(W_{pre})+s\Delta W$ —— 對 frozen $W_{pre}$ 做逐元素 dropout（見第 4 節）
2. Trigger-insensitive regularization：懲罰項 $\Omega(A,B)=\|U^\top B\|_F^2+\|AV\|_F^2$，逼 $\Delta W$ 正交於 $W_{pre}$ 的頻譜子空間
3. Post-training spectral rescaling：訓練後設 $s=\sigma_{max}(W_{pre})/\sigma_{max}(\Delta W)$

**重要性：** 大幅降低 attack success rate（多數設定降到 20% 以下、部分低於 5%），同時 clean accuracy 幾乎不掉；且對 PiSSA/DoRA/OLoRA 等 LoRA 變體同樣有效（見第 6 節）。

---

## 3. What is dropout and why is it important? 🟢

**一句話：** Dropout 是訓練時隨機關掉一部分神經元/權重的正則化技巧，效果好但長期以來只有經驗法則，你讀的三篇論文各自從不同角度給出嚴格的數學解釋。

**背景（Srivastava et al. 2014 / Hinton et al. 2012）：** 隨機關閉神經元 ≈ 同時訓練指數多個共享權重的子網路，測試時做隱式平均，目的是打破 co-adaptation、提升泛化。

**三篇論文分別回答「為什麼有效」：**

| | 模型設定 | 核心結果 |
|---|---|---|
| **Paper 1**（Mianjy, Arora & Vidal, ICML 2018） | 兩層線性網路 $h(x)=UV^\top x$，dropout 打在 bottleneck | dropout 等價於在 loss 上加懲罰項 $\lambda\sum_i\|u_i\|^2\|v_i\|^2$，導致 equalization（各神經元範數趨於一致）與低秩偏好 |
| **Paper 2**（Zhang & Xu, TPAMI 2024） | 一般非線性網路，dropout 打在最後隱藏層 | 用 modified gradient flow 推出兩個隱式正則項 $R_1,R_2$；$R_1$ 促進 **condensation**（神經元方向收斂到少數幾個孤立方向，網路變「簡單」）；$R_2$ 與 flat minima 有關（但這個連結是引用自 Barrett & Dherin 等人，本篇沒有重新證明） |
| **Paper 3**（Mori & Mignacco, 2025） | Teacher-student 兩層網路，online SGD，高維極限 | 用統計物理工具推出一組精確 ODE，描述 generalization error 的完整動態；證明 dropout 降低 hidden node 間的有害相關性、緩解 label noise 的影響，且最佳 dropout rate 隨 noise 增加而上升 |

**重要性：** 三篇合起來，從「隱式正則化的形式」（Paper 1）、「正則化如何影響幾何結構／condensation」（Paper 2）、「對 generalization error 的精確動態刻畫」（Paper 3）三個層次，把一個原本純經驗的技巧變成可以嚴格分析的對象。

---

## 4. Why add dropout to RoRA? 🟡（這節是你自己要講清楚的整合，不是任何論文寫的）

**RoRA 論文自己的理由（🟢）：** 對 $W_{pre}$ 做 dropout，是要強迫模型不能只依賴（可能帶毒的）pretrained 表徵，逼 $\Delta W$ 自己學會足夠的 clean-task 資訊 → 提高 $\rho_{cl}$ → 拉大 $\rho_{eff}=\rho_{cl}-\rho_{tr}$ → 降低 forgetting 所需的 scaling threshold $s^\star$。

**這跟你讀的三篇 dropout 論文接不接得起來？—— 誠實的答案是：接不太起來，這正是老師該知道的落差。**

- Paper 1、2 的 dropout 都打在**正在被訓練、會被梯度更新**的矩陣上（$U,V$ 或某個隱藏層），implicit regularization 的推導核心是：對 dropout mask 取期望值，會在 loss 裡多出一項懲罰**這個矩陣本身**的項。
- RoRA 裡的 $W_{pre}$ 是**完全凍結、不參與訓練**的。對一個不被優化的量做 dropout，取期望值不會產生「懲罰進 loss 裡」的效果，而更接近**在 forward pass 對輸入/表徵做隨機擾動**（更接近 dropout 最原始的「打破 co-adaptation／ensemble averaging」動機，而不是 Paper 1/2 的隱式偏誤框架）。
- 換句話說：RoRA 用了「dropout」這個操作，但它背後在起作用的機制，跟你深讀的三篇論文所刻畫的機制，**不是同一套理論**。RoRA 論文自己也是用完全不同的分析工具（spectral norm、margin bound），完全沒有用 implicit regularization 的語言。

**這個落差本身可以是老師想聽到的「bigger picture」：** 兩套關於 dropout 的理論（「打在可訓練矩陣上的隱式正則化」vs.「打在凍結矩陣上的表徵擾動／遺忘工具」）目前是脫節的，這正是可能的研究切入點，而不是你漏讀了什麼。

---

## 5. Effect of dropout + RoRA? Theory 🟡（這節是我剛剛重新推導的，不是任何論文寫的——請務必自己再檢查一次再上台）

**先講結論，再講推導：** 把 Paper 1/2 用的那招（對 dropout mask 取期望值，看會不會生出一個懲罰項）直接套到 RoRA 的設定上，在最簡化的 squared loss 情況下，**答案是「不會」**——期望值算出來的變異數項跟 LoRA 的可訓練參數 $\Delta W=BA$ 完全無關，梯度是零。這解釋了「為什麼三篇 dropout 論文的機制套不進 RoRA」不只是結構上『長得不一樣』，而是**你如果硬套同一個推導技巧，它會直接算出沒有效果**。這是一個可以講出來、有數學支撐的負結果，不是含糊帶過。

**推導（簡化設定：squared loss，$W_{pre}$ 逐元素獨立 dropout，inverted scaling）：**

令 $z = \text{Dropout}_p(W_{pre})x + s\Delta W x$，寫成 $z=a+b(M)$，其中 $a=s\Delta Wx$ 是可訓練部分（固定），$b(M)=\text{Dropout}_p(W_{pre})x$ 是隨機部分（只跟 mask $M$ 有關，跟 $\Delta W$ 完全無關）。用 squared loss $\ell=\|y-z\|^2$：

$$\mathbb E_M[\ell] = \|y-a\|^2 - 2(y-a)^\top\mathbb E[b] + \mathbb E[\|b\|^2]$$

因為 inverted dropout 是 unbiased：$\mathbb E[b]=W_{pre}x$（跟原本沒 dropout 時一樣）。變異數部分算出來是

$$\operatorname{tr}(\operatorname{Cov}(b)) = \frac{p}{1-p}\sum_j x_j^2\sum_i W_{pre,ij}^2$$

**這一項只跟 $W_{pre}$、$x$、$p$ 有關，完全不含 $\Delta W$。** 對 $A,B$ 取梯度時，這項直接消失——**在這個簡化設定下，對 frozen $W_{pre}$ 做 dropout，取期望值後，對 $\Delta W$ 沒有任何隱式正則化效果。**

**為什麼跟 Paper 1（$h=UV^\top x$，dropout 打在 bottleneck）不一樣：** Paper 1 的懲罰項 $\lambda\sum\|u_i\|^2\|v_i\|^2$ 之所以會出現，是因為 dropout mask 同時作用在 $U,V$ 這兩個**都在被訓練**的量上，所以取期望值時，變異數項本身就是 $U,V$ 的函數，梯度不會消失。RoRA 裡只有一邊（$W_{pre}$）被 mask，另一邊（$\Delta W$）完全不受 mask 影響，這是關鍵差異——不是「風格不同」，是這招在數學上直接失效。

**那 RoRA 論文裡觀察到的效果（ASR 下降）到底從哪來？——誠實的候選方向（尚未推導完，是接下來真正該讀的東西）：**
上面的零效果只在 **squared loss** 下成立。RoRA 實際用的是 cross-entropy／margin 這種**非線性**的 loss。對一般 loss $\ell(z)$ 做二階展開：
$$\mathbb E_M[\ell(z)]\approx \ell(\bar z) + \tfrac12\operatorname{tr}\big(\operatorname{Cov}(b)\,H_\ell(\bar z)\big),\qquad \bar z = W_{pre}x+s\Delta Wx$$
這裡 $\operatorname{Cov}(b)$ 仍然跟 $\Delta W$ 無關，但 **loss 的 Hessian $H_\ell(\bar z)$ 是在 $\bar z$（含 $\Delta W$）處算的**——所以這次懲罰項會透過「目前模型對這筆資料有多不確定」跟 $\Delta W$ 產生耦合。直覺上：dropout 造成的雜訊在 $W_{pre}$ 「權重範數大」的方向上變異數最大，而 cross-entropy 的 Hessian 在「模型還不確定」的方向上最大——兩者相乘,大致對應「$W_{pre}$ 訊號強、但目前預測還不篤定」的方向,這正是需要靠 $\Delta W$ 去補強或蓋過的方向。這跟 Bishop (1995)「用雜訊訓練等價於 Tikhonov 正則化」以及 Wager, Wang & Liang (2013)「dropout training as adaptive regularization」是同一套機制，但**這兩篇不在你讀過的三篇清單裡**，是真正可能填補這個缺口的文獻,而不是重讀 Paper 1/2/3。

**這節要跟老師講的話：**
> 「我把三篇論文共用的推導技巧直接套進 RoRA 的設定，發現在最簡單的情況下這招會失效——這解釋了為什麼這三篇論文的理論工具沒辦法直接說明 RoRA 為什麼有效。真正可能有關的，是另一支我還沒讀的文獻（雜訊訓練與 adaptive regularization），這是下一步真正該補的閱讀，而不是這學期讀的三篇。」

---

## 5b. Effect of dropout + RoRA? Empirical 🟢（RoRA 論文自己的實驗結果，直接可以講）

**Dropout rate $p$ 的效果（Fig. 5b, ablation）：**
- $p\approx 0.1$ 附近效果最好
- $p$ 太小（如 0.05，LLaMA）：擾動不夠，ASR 仍高達 54%，backdoor 沒被真正遺忘
- $p$ 太大（如 0.30）：連正常的 pretrained 知識都被破壞，clean accuracy 明顯下降（RoBERTa ~94%，LLaMA ~88.6%）

**單獨這個 component 有多有效（Table 4 ablation, Cl = clean-strengthened regularization 也就是這個 dropout）：**
- 例如 RoBERTa + BadNet：baseline LoRA 的 ASR 99.74% → 只加 Cl（dropout）就降到 10.34%，$\Delta$（robustness margin）從 $-4.03$ 提升到 $85.82$
- 單獨 Tr（正交懲罰）在某些設定（如 InSent 攻擊）效果比 Cl 差很多（ASR 仍 65–98%），顯示 dropout 這個機制在多數情況下是三個 component 裡貢獻最大的一個
- 三者合併（Cl+Tr+Pt）才是全文最好的結果

**代價很小：** 每 iteration 只多花 6–15 ms，相對開銷 ≤17%（Table 3）。

---

## 6. Extend to LoRA variants (DoRA, qLoRA, AsymLoRA)? 🟡（重寫版——丟掉 B1/B2/B3，改成結構上真的講得出道理的分析）

**⚠ 先給老師的框架校正（這句話本身就是這節的開場，不是廢話）：**
> 「RoRA 不是一個『幫任何 LoRA 變體加分的通用魔法』，它是針對一個很特定的 threat model（pretrained 被下毒、用乾淨資料 fine-tune 去清除）設計的方法。所以『DoRA/qLoRA/AsymLoRA 能不能也用 dropout 提升 robustness』這個問題，正確的問法應該是：**在同一個 threat model 下，把 RoRA 那套『對 frozen 部分做 dropout 來幫助遺忘』的邏輯，套進每個變體各自的參數化方式，理論上會發生什麼事？**」這樣先講，是要讓老師知道我們是認真在回答他的問題，而不是回答一個他沒問、但比較好回答的問題。

**用第 5 節推出來的真實機制（curvature-weighted variance term）去問每個變體：**

RoRA 的效果（如果我在第 5 節的推導方向是對的）來自「dropout 在 $W_{pre}$ 訊號強、但目前預測還不確定的方向上，產生一個跟 $\Delta W$ 耦合的懲罰項」。這個機制能不能套進其他變體，取決於**dropout 加在哪裡之後，雜訊怎麼傳到最終輸出**，三個變體在這一步結構都不一樣：

- **DoRA**（$W'=m\cdot\dfrac{W_{pre}+sBA}{\|W_{pre}+sBA\|_c}$，逐欄正規化）：如果對 $W_{pre}$ 做 dropout，雜訊不只加進分子，**分母（正規化常數）也變成隨機變數**——輸出變成「隨機變數的比值」而不是單純的「均值+雜訊」。第 5 節那個乾淨的二階展開（$\operatorname{Cov}(b)$ 跟 Hessian 相乘）不能直接套用，因為分母的隨機性會產生額外的一階項（$1/\|\cdot\|$ 對雜訊不是線性的）。**結構上的結論：DoRA 的正規化步驟會讓 dropout 的效果變得不可預測——可能放大、可能抵銷，需要重新展開，不是「應該也有效」這麼簡單。**
- **qLoRA**（$W_{pre}$ 先被量化成低位元，如 4-bit NF4）：對一個已經量化過的矩陣再做 elementwise dropout，兩種雜訊疊在一起——量化誤差是**離散、有偏**的，dropout 造成的遺失是**離散、無偏（inverted scaling 下）**的，兩者疊加不是簡單相加變異數，因為量化誤差本身不是均值零的隨機變數。**結構上的結論：qLoRA 的量化雜訊會跟 dropout 雜訊產生非平凡的交互作用，第 5 節的推導必須先處理「雜訊疊雜訊」這一步，目前沒有人做過。**
- **AsymLoRA**（不對稱秩／跨層共享參數的 LoRA 變體）：如果 $A$（或某一側）是**跨層共享**的，而 dropout mask 是每層獨立灑的，那麼共享參數 $A$ 收到的梯度，會是**多層 curvature 懲罰項的加總**——第 5 節的效果在單層是「$W_{pre}$ 強、predict 不確定的方向」，但共享參數必須同時滿足多層的這個條件，可能互相加強，也可能互相抵銷（如果不同層的「不確定方向」不一致）。**結構上的結論：跨層共享把單層的懲罰項變成加總，效果方向不確定，需要逐層檢查。**

**這節要跟老師講的話（誠實、但顯示出你真的想清楚了）：**
> 「我先重新推導了 dropout 對 RoRA 有效的可能機制（見上一節），然後問：這個機制能不能套進 DoRA、qLoRA、AsymLoRA？答案是——不能直接套，而且我可以具體指出卡在哪裡：DoRA 的正規化步驟讓雜訊變成非線性；qLoRA 的量化雜訊會跟 dropout 雜訊疊加出目前沒人處理過的交互作用；AsymLoRA 的參數共享會把單層的效果變成多層加總，方向不確定。這三個『卡住的點』就是接下來要一個一個算清楚的東西，不是空泛地說『這很複雜』。」

**還沒做的（🔴，誠實講清楚）：**
- 上面三點都是**定性的結構分析**，還沒有任何一個真正展開成公式或做完推導
- 完全沒有實驗數據
- 還沒有把這條線跟你深讀的三篇 dropout 論文（condensation、equalization）真正接起來——目前看來,接不起來的原因已經很清楚（見第 5 節）,所以下一步該讀的是 Bishop 1995 / Wager-Wang-Liang 2013,而不是回頭重讀 Paper 1/2/3

---

## 建議簡報順序備忘
1 → 2 是背景鋪墊，2 開頭順便校正老師對 RoRA scope 的誤解（見第 6 節開場那段，可以提前用在這裡）
3 是你這學期真正深讀的主力，篇幅可以最重、細節可以最多
4 是「三篇論文的機制為什麼套不進 RoRA」的結構診斷
5（Theory）是全場最有份量的一節：親手把 Paper 1/2 的推導技巧套進 RoRA，證明在 squared loss 下失效，並指出真正該讀的文獻方向（Bishop 1995、Wager-Wang-Liang 2013）——**上台前務必自己重新演算一遍第 5 節的推導，我是剛剛現場推的，沒有交叉驗證過**
5b 是拿 RoRA 論文自己的實驗結果撐場面，最省力也最扎實
6 用第 5 節的機制去問三個變體，給出具體卡住的技術點，而不是「還沒做」三個字帶過
