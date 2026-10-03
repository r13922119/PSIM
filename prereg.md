# PREREG：後門「持久性」判準（在看任何 v2 baseline 結果之前寫下）

日期時間：34919829278abde6d4724531b3864e941312ba76 Wed Sep 30 10:02:43 2026 +0800（commit 時填；以 `git log -1 --format='%H %cd'` 為準）
標記 ☐ 的值是 Claude 的建議，**由你確認或修改後才算數**；改完再 commit。
☐ 已確認改為 ☑

## 1. 要回答的問題
一顆被毒化的預訓練模型（PZ），在普通 LoRA（baseline，不帶 Cl、Tr）微調後，後門是否「留得住」？
只有留得住的 PZ，才有東西可讓 RoRA 去修；洗得掉的 PZ 上所有變體都會看起來一樣。

## 2. 面板（先固定，不看結果不增減）
- 新訓 PZ：BadNet，PZ seeds 0–4（這是 PZ 預訓練的 seed，與下面的微調 seed 是兩個獨立的編號），同一型號 GPU（RTX 6000 Ada）。☑ s4 尚未訓練，必須在跑任何 baseline 之前訓完。
- 舊 PZ（`old_models/poisoned_roberta_large_badnet`，sha256 75823f91…）：以 `oldpz` 標註入面板，**不計入新訓 PZ 的持久率**。
- 不事後排除任何 PZ。s3 的 test clean acc 偏低（0.880），按規則仍留在面板，並報告其 CA。
- 已知的「已看過」資料：新訓 s0–s3 的預微調 ASR 與 clean acc；舊 PZ + seed 0 的 20-epoch baseline（gate / golden 檔，決定性，資格賽會重現同一個值），這個數字可能已被看過，所以 `oldpz` 的判定不是盲的（也因此不計入持久率）。

## 3. 微調 seed 分工（與 PZ seed 是兩個獨立的編號）
- 資格賽（本文件的判準）：微調 seeds 0、1、2。
- 加賽（只有觸發 §5 才用）：seeds 6、7、8。
- 比較實驗（Cl、Tr、Cl+Tr 與重跑的 baseline）：seeds 3、4、5，**不得用於資格賽**。

## 4. 統計量與門檻
- 每個 run 的統計量：baseline、pre-Pt、observer 量測的 ASR，取 epoch 15–19（0 起算，num_epochs=20）的平均，即 SUMMARY 的 `asr_mean_last5`。不使用 best-dev epoch 的 ASR。
- 每顆 PZ 的持久度 = 其 seeds 0–2 的 `asr_mean_last5` 平均。
- ☑ **持久：持久度 ≥ 0.50。**
  - 這個數字是判斷，不是從資料推導。v1 的少量數字（新 s42 約 7%、s3 約 68%、舊 PZ 約 74–87%，皆受 GPU 型號混淆、n 很小）只顯示「兩群之間有空隙」，0.50 落在空隙中；不能當作它正確的證據。
  - ASR 的自然下限約 4%（正類的自然誤差），所以低於 0.25 的視為洗掉。

## 5. 模糊區與加賽
- ☑ 若某顆 PZ 持久度落在 [0.25, 0.50)，或其三個 seed 的 max−min > 0.30：加跑 seeds 6–8，改用六個 seed 的平均重新判定，之後不再加賽。
- 加賽觸發條件與判定方式在看結果前就已固定，不得因為結果不如預期而另加 seed。

## 6. 繼續或停止
- ☑ 新訓 PZ 中**持久者 ≥ 2 顆**才進入 Cl / Tr 實驗（S2 之後）。
- ☑ 若 PZ seeds 0–4 中持久者 < 2，**不立刻停止，先依固定程序補訓**（現在就定好，不待結果）：PZ seeds 5、6、……依序一顆一顆訓練，每顆都用微調 seeds 0–2 做資格賽，直到新訓持久者累計 2 顆，或新訓 PZ 總數達 **N = 10** 為止。補訓用同一型號 GPU、同一份程式碼雜湊；若程式碼改過，先重跑 gate。
- 持久率一律以「持久 / 全部已訓練的新訓 PZ」報告（例如 2/9），不得只報被選中的。
- 補訓到 N = 10 仍 < 2：停下重議（毒化比例、詢問作者），不硬跑機制比較。
- 若只有 `oldpz` 持久：同樣停下討論，不自動繼續。
- 通過者構成 P'。`oldpz` 若持久，可作為標註成員進入機制實驗，但所有結果標示它的來歷不同。

## 7. 報告方式
- 持久率 = 通過的新訓 PZ 數 / 新訓 PZ 總數，附 n（預計 5）。這是回答「作者是否剛好選到一顆持久的 PZ」的數字，只用新訓 PZ 算。
- 主表的絕對 baseline 數字是「條件於持久」的（P' 是依 baseline 結果選出），要這樣標示。
- 失敗與模糊的 PZ 也要在結果表中完整列出。

## 8. 簽核
- ☑ 我確認 §4 的門檻、§5 的模糊區、§6 的繼續條件，且尚未看到任何 v2 資格賽（seeds 0–2 的 baseline）結果。

# 紀錄
2026-10-03: persist.py was committed after the seeds 0–2 baseline results were viewed. I verified its verdicts by hand against §4–§5 (s0/s2 persistent, s1/s4/oldpz ambiguous, s3 washed out). Thresholds unchanged.