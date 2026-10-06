### 2026-10-06（台北時間），PZ 的強度門檻改以下游測試集（SST-2）測量（附加於 PREREG.md 修訂紀錄；以下各項皆為 Claude 的建議，打勾才算數）

**這是在看過三筆 SST-2 數字之後才寫的。** 當時手上只有 s0（ASR 96.1%，5 個位置 95.4–97.1）、s4（85.5%）、oldpz（99.9%）三筆；badnet_s2、s1、s3 的 SST-2 數字當時沒有（s1、s3 只有 zero-shot SST-2 ASR=1.000 的紀錄）。規則在量剩下的 PZ 之前先定。它不用來更換已報告的結果。

1. ☑ **觀點。** 我們不沿用論文把 (A1) 與「ASR>95%」連在一起的說法（Section 6：「ASR consistently exceeds 95%, which naturally satisfies Assumption (A1)」；另，(A1) 式 (5) 與證明式 (16) 的不等號方向相反，單獨記在別處，與本規則無關）。我們的理由只有一個：這是觀察者建的測試平台，PZ 的後門夠強，才有東西可以觀察微調是否把它洗掉。
2. ☑ **規則（ASR）。** 一個 PZ 要列為「合格 PZ」，在 SST-2 測試集上，以 `compute_asr`（分母＝被插入 trigger 的正類測試句）量得的 ASR，在 `placement_seed = 0..4` 的 **5 個 trigger 位置全部 ≥ 0.95**（即 min ≥ 0.95，不取平均）。理由：論文的數字報的是多次中最好的一次（Appendix A.1：「reported values correspond to the best performance across multiple runs」），Section 6 用的詞是「consistently exceeds」，所以我們取較嚴格的讀法。min 會隨位置數增加而變小，所以位置數與 seed 固定為 5 個、0..4。均值、max 一併印出，不用於判斷。
3. ☑ **只報告、不設門檻的量。** `check_patient_zero.py` 印出的 CA（微調前）、pos_err（乾淨正類句的錯誤率，即 ASR 的自然下限）、neg_err（乾淨負類句的錯誤率）一併寫進表，供讀者自行判斷。各欄單位都是 %。不對這三個量設門檻：論文 Table 1 的 RoBERTa BadNet 微調前 CA 為 81.27%，而我們已有的 PZ 為 s0 76.83%、oldpz 76.22%，任何接近論文值的 CA 門檻會把現有 PZ 全部排除，且門檻值只能憑感覺定；CA 低本身是「必須微調」的情境，不是無觀察價值。若要設任何門檻，需另外寫出理由與數值，並在量測前宣告。
4. ☑ **適用範圍。** 往後選 PZ 時使用。S1–S5 已報告的結果不重算、不刪除；對其中的 PZ 逐一補上這項檢查的結果並標示「合格 / 不合格（min<0.95）」。不合格的 PZ 不從已報告的表中移除，只加標籤。
5. ☑ **這個門檻不判斷持久性。** s1、s3 的 SST-2 zero-shot ASR=1.000，卻在微調後被洗掉。門檻只表示「PZ 在下游輸入上本身夠強」，與 S1 的持久性篩選是兩回事，兩者都要過。
6. ☑ **量測方式。** 用 `check_patient_zero.py --placements 5` 對每個 PZ 的 `pytorch_model.bin` 直接量，不修改 `poisoned_pretrain.py`（它在 code hash 內，改了會讓 S2/S4/S5 的 lr 2e-4 對照不能再沿用）。
7. ☑ **已知的觀察，不用於選擇。** s4 在 SST-2 上 CA 為 54.26%、pos_err 為 0.00%（偏向預測正類），這會壓低 ASR；規則不為此調整。

*修訂說明（一句話，可刪）：這份修訂的前一版在第 3 項曾有「pos_err ≤ 10%」的門檻，在看到六個 PZ 的結果（pos_err 全部 0.00–0.66%）後改為只報告；這個改動沒有改變任何 PZ 的判定。*