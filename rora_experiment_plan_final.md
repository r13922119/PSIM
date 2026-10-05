# RoRA 重現：最終版實驗清單（2026-10-04，第四版）

舊協定（v1）下的所有數字作廢，不再比較。以下為從零重跑的計畫。P = 新訓 BadNet 後門基礎模型（PZ）數，P' = 通過持久性驗證的 PZ 數，S = 微調 seed 數（預設 3）。

## 修訂紀錄
第二版（相對第一版）：
1. 新增 §0.1：資格賽與比較用不同 seed，避免選擇偏誤。S2 的 baseline 不再沿用 S1。
2. 主要統計量改為 asr_mean_last5（epoch 15–19 平均），最後一個 epoch 為次要。**（第四版第 8 項取代此條：last5 只是觀察者診斷，不是績效。）**
3. Pt 只能與同一個 epoch（best-dev）的 pre-Pt 比，不可與最後 epoch 的數字比。
4. λ 預設值（mean 形式 10）可能弱於論文最小值，S3/S4 加上處理方式（待決定）。
5. 舊 PZ 作為標註成員（oldpz）入面板，不納入「新訓 PZ 持久率」的估計。
6. done 檔鍵、時間估計、§3 的 v1 數字標註、S0/S1 狀態已更新。

第三版：
7. 持久性判準改以 PREREG.md 為準（在看任何 v2 baseline 結果前寫下並 commit）。含：統計量與門檻、模糊區加賽（微調 seeds 6–8）、持久者 <2 時依固定程序補訓 PZ（seeds 5、6、…，上限 N=10）才停。PZ seed 與微調 seed 是兩個獨立的編號。

第四版（2026-10-04，S3 啟動前；以下三條是 Claude 起草、使用者確認的）：
8. **績效定義**：一律是 best-dev epoch 的 test CA 與 ASR，Pt 前與 Pt 後都報，跨 seed 呈分布。asr_mean_last5 與每 epoch 曲線只作觀察者診斷（診斷差的表現、解釋好的表現），不是績效數字，因為真實的微調者不會使用那些 checkpoint。不存在「最後一個 epoch」這個報告單位。PREREG.md 的 PZ 持久性篩選仍用 last5：那是觀察者為了建一個難的測試平台所做的篩選，不是績效評估；篩選統計量與評估統計量本來就不同。
9. **S3 重寫**（見 S3 列與 PREREG_S3.md，已於 S3 啟動前 commit）：PZ=badnet_s2；seeds 20 與 21–23；λ_mean ∈ {10, 480}（240 移出 S3，S3 的 λ 處理 (A)/(B) 二選一由此決定：S3 兩個 λ 都掃，S4 兩個 λ 都報）；k 網格九個值；選擇統計量 pt_ASR 加 dev 與 CA 護欄；效果以三種標籤報告（降低／無法區分／增加），不是閘門；另有 paper-equivalence 描述格（2 個 Cl+Tr control、6 個 Tr-only，seed 20，不參與選擇）。
10. **S2 以 best-dev 統計量重讀**（compare_s2.py：pre-Pt、with-Pt、Pt 效果、各 seed 的 paired 差；last5 只當診斷列）。PZ 的篩選結論不變。

## 0. 固定協定（所有實驗一體適用）
- 硬體：全部在 RTX 6000 Ada，程式內 assert 擋其他型號；腳本只挑 Ada 中用量最少的一張。已驗證：同一張卡兩次相同，兩張 Ada 之間也相同（baseline、3 epochs、舊 PZ、seed 0）。
- RNG：觀察者原則。test acc、ASR、診斷都跑在隔離的 RNG（observer）裡，訓練端維持 PSIM 的全域 RNG。ASR 每個 epoch 都記；每個 epoch 印 RNG 指紋；每個 log 印 GPU、套件版本、腳本與 PZ 的雜湊。測試 trigger 位置用固定種子。
- 訓練設定：lr 2e-4 為主線、batch 32、wd 0.01、warmup 6%、20 epochs；LoRA r=8、α=16、lora_dropout 0.1；DoRA 用 peft 預設（r=8、α=8、dropout 0）。所有訓練都帶 --use_spectral_rescaling（Pt 作用在 `[-6:]`，一次訓練同時得 pre-Pt 與 Pt）。注意：--num_epochs 會改變學習率排程，短跑不是 20 epoch 的前綴。
- 機制：Cl p=0.1。Tr：Ω 乘上（維度/k），跨模組取 mean，λ 預設 10（mean 形式；等價加總形式約 0.21）。論文網格 {1,5,10,15,20} 若是加總形式，對應 mean 形式 {48,240,480,720,960}，所以預設 λ 可能比論文最小值還弱（見 S3/S4）。k 一律明確傳 --svd_k；命名 tr{λ:g}_k{k}。（乘上維度/k 與跨模組取 mean 都是我們自己的設計，不是論文的；論文未說明。）
- 資料：SST-2（PZ 用 IMDB，1,500 筆 clean-label，目標標籤 0）。
- 選擇與報告：checkpoint 只用 dev clean acc 選（best-dev epoch）。**績效 = best-dev epoch 的 test CA 與 ASR，Pt 前（pre-pt_ASR）與 Pt 後（pt_ASR）都報，跨 seed 報分布（mean±sd，標 n，同時列各 seed 的值）。** asr_mean_last5（epoch 15–19）與每 epoch 曲線是觀察者診斷：可以用來診斷差的表現或解釋好的表現，但不是績效數字，因為真實的微調者不會使用那些 checkpoint。不報「最後一個 epoch」。best-of 另外標為 oracle。PZ 當因子分開報告。Pt 作用在 best-dev adapter 上，只能與同一 epoch 的 pre-Pt（SUMMARY 的 pre-pt_ASR）比，不可與 last5 比；Pt 後的 ASR 要連 test acc 一起看（高 k 時 Pt 會傷 CA）。PZ 持久性的篩選（PREREG.md）是觀察者篩選，用 last5；篩選統計量不是評估統計量。
- 迴圈：外層 PZ、中層 seed、內層配置。done 檔以 `title|四個程式檔的內容雜湊` 為鍵（git 樹常是 dirty，所以不用 git hash）；title 含 PZ tag 與 PZ sha 前 8 碼。用 parse_runs.py（v2 版，會比對 [RNG]、per-epoch test/ASR）產 csv。
- 每個 run 檢查：hook 次數正確、無 NaN、耗時。耗時以 golden 檔 SUMMARY 的 duration_min 為準（觀察者每 epoch 都跑，比 v1 的 11–13 分鐘慢）。
- 工具：repro_utils.py（observer、isolated_rng、rng_fp、log_env、write_meta）、gate_a.sh（觀察者中立）、gate_b.sh（決定性；CROSS=1 跨卡）、v2_common.sh、v2_pz.sh、v2_baseline.sh、v2_s2.sh、v2_s3.sh、select_k.py、compare_s2.py。

## 0.1 避免選擇偏誤
- 資格賽（S1）：每顆 PZ 的 baseline 用微調 seeds 0–2，只用來判斷「是否持久」。
- 比較（S2 之後）：baseline 與所有機制一律用 fresh 微調 seeds 3–5，baseline 要重跑，不沿用 S1。
- k 在 S3 於某顆 PZ 上調（用 seeds 20–23，與報告用的 3–5 分開），S4 中該 PZ 的列要標註 tuned-on。
- P' 是依 baseline 持久性選出來的，所以主表的絕對 baseline 數字是「條件於持久」的，要這樣報。

## 1. 階段
| # | 階段 | 內容 | 規模 | 閘門 |
|---|---|---|---|---|
| S0 | 套用協定修改（已完成大半） | 已完成：程式修改；gate (a) 觀察者開關對照（variant_finetune 的 baseline 與 Cl+Tr、poisoned_pretrain 通過）；gate (b) 同指令兩次相同（3 與 20 epochs 通過，含跨兩張 Ada）；golden 檔 `golden/baseline_0_oldpz_ep20_20261002.txt`。未完成：gate (a) 的 clean_pretrain（`./gate_a.sh ct`）；gate (c) baseline 回歸（改訓練程式碼後才需要，尚未寫） | 少量 | 改訓練端程式碼後要重跑 (a)(b)(c) |
| S1 | PZ 面板與資格賽 | 新訓 BadNet PZ ≥5 顆（PZ seeds 0–4、同一張 Ada）。**目前 PZ seeds 0–3 已訓完**（GPU-67a5…, code d552a427）：預微調 ASR 0.996/0.998/0.998/1.000，test CA 0.925/0.932/0.933/0.880（s3 偏低，按規則不排除，報 CA）。**尚缺 s4 才達 ≥5**（`SEEDS="4" bash v2_pz.sh`），須在任何 baseline 之前訓完。舊 PZ（6 月 cml24，sha256 已記錄）以 `oldpz` 標註入面板，不納入新訓 PZ 的持久率。每顆跑 baseline LoRA（不帶 Cl、Tr），微調 seeds 0–2 | 3×(P+1) = 18（P=5） | 判準以 PREREG.md 為準：三個 seed 的 asr_mean_last5 平均 ≥0.50 算持久；模糊區加賽；持久者 <2 時依固定程序補訓（PZ seeds 5、6、…，上限 N=10）才停下重議 |
| S2 | Cl 基準 | 持久 PZ 上，baseline 與 Cl-only 都用 fresh 微調 seeds 3–5（baseline 重跑） | 6P' | — |
| S3 | 選 k | 能量表已完成（描述性）。PZ=badnet_s2。階段 1（seed 20，28 次）：Cl-only 與 baseline 參考、Cl+Tr 在 λ_mean∈{10,480}×k∈{8,32,128,256,512,768,896,1016,1024}（18 次）、paper-equivalence 描述格（2 個 Cl+Tr control、6 個 Tr-only，不參與選擇）。階段 2（seeds 21–23，最多 21 次）：每個 λ 取階段 1 最佳 k 與其兩個鄰居，加 Cl-only 參考 | 最多 49 | 規則在 PREREG_S3.md（S3 前 commit），由 select_k.py 機械套用：以 pt_ASR 選、dev 容忍 0.01、CA 護欄 0.02；效果以三種標籤報告，不是閘門；只有「沒有 feasible 格子」才停（S3_STOP）。λ 處理已決定：兩個 λ 都掃，240 移出 S3 |
| S4 | 主表 Tr、Cl+Tr | S3 選出的 k_final；所有持久 PZ；fresh 微調 seeds 3–5；Tr 與 Cl+Tr 各在 λ_mean=10 與 480；baseline 與 Cl 沿用 S2 同 seed 的結果（若訓練端程式碼雜湊未變，否則重跑） | 12P' | 完成 BadNet 主表；advisor 要看的 Cl → 約 10% 在此驗證。k 在 badnet_s2 上選，該列標 tuned-on。S4 的 Tr-only 臂回答的是「在選定設定內 Tr 貢獻多少」，不是 Tr 自己的最佳 k。若只在 λ_mean=10 得到「Tr 無效」，不能據此反駁論文（可能比論文最小 λ 還弱），所以兩個 λ 都報 |
| S5 | λ 掃描（完整 RoRA，選定 k） | λ_mean 順序 10,480,5,240,1,48,15,720,20,960；seed 在內層，3 seed；一顆持久 PZ。選配：在轉折區的 k 再掃一次（機制探索，非重現必需） | 30 | 判斷 λ 是否關鍵 |
| S6 | lr 篩選 | 四個配置 × lr∈{2e-5, 2e-3}，seed 0–1，一顆持久 PZ。(lr, 配置) skip list；有競爭力的才補到全 seed | 16＋補跑 | — |
| S7 | InSent | 重做 InSent PZ 面板（≥3 顆＋持久性驗證）。注意 v2_pz.sh 目前只含 BadNet，InSent 要另跑（--attack_tag insent）；v1 的 InSent PZ 訓練 GPU 不明，不用。再跑 S2、S4 的精簡版（k 沿用 BadNet，不重掃） | 視面板 | InSent PZ 微調前 ASR>95% 且持久 |
| S8 | 凍結分類頭對照 | baseline、Cl、Cl+Tr，一顆持久 PZ，3 seed。需改一小段程式（注意分類頭初始值來自 PZ） | 9 | — |
| S9 | DoRA × BadNet、InSent | 四配置主表 | 視 P' | — |
| S10 | 條件式與選配 | p 掃描 {0.05,0.15,0.2,0.3}（對明顯劣於論文的 cell）；毒化比例對照（視作者回信，需重做 PZ）；Pt 層選擇離線比較（最後 1 層、`[-6:]`、全部層，免訓練）；機制探索（硬投影、窗口懲罰，非重現必需） | — | — |

## 1.1 規模與時間（估計）
- 以 P=5、P'=3、不含 InSent/DoRA/S10 計：S1 18 + S2 18 + S3 最多 49 + S4 36 + S5 30 + S6 16 + S8 9 = 最多 176 次（第四版重算；若觸發補訓，每多一顆 PZ 加約 8.5 分鐘訓練 + 3 次資格賽）。
- 以每次 12 分鐘粗估約 35 小時 Ada 時間。12 分鐘只是 v1 的舊數字，請以 golden 檔的 duration_min 校正。
- PZ 訓練：4 顆約 34 分鐘（約 8.5 分鐘/顆，由 log 時間戳推算）。

## 2. 不需 GPU 的工作
- σ² 累積能量表：每個 module、每顆 PZ，給出 k(50/80/90%)（v2 版已完成，描述性）。
- 寄給作者的追問信（現在就可寄，只問問題、不附數字）。新增一題：Eq. 10 的 U、V 是否截斷、k 是多少，懲罰是否乘 d/k，跨模組是 mean 還是 sum（paper-equivalence control 若勝出，這就是下一步）。
- 譜對齊分析重跑：非必要。已知沒有 trigger 專屬子空間，不能用來選 k。

## 3. 目前已知
**v2 協定下（已驗證）**
- 同一張 Ada 兩次 bit-identical；兩張 Ada 之間也相同（baseline 路徑、3 與 20 epochs、舊 PZ、seed 0）。這只涵蓋 baseline；Cl 路徑跨卡尚未驗證。
- v2 新訓 PZ s0–s3 預微調 ASR 都 >99%。持久性尚未量測（S1 進行中）。**（此段是 10/03 的狀態，已過時；持久性、S2 結果請見 10/04 之後的紀錄與 PREREG.md 的日期 log。）**

**v1 協定下的假設（樣本小、需在 v2 重驗，不是證據）**
- 新舊 PZ 微調前 ASR 都約 100%，但微調後持久性差很多（舊 PZ 約 74–87%，新 s42 的 baseline 約 13%）。論文的「ASR>95%」驗收分辨不出持久性。
- 結果不穩的來源：GPU 型號（影響 dropout mask）、DataLoader 迭代器每建一次就消耗全域 CPU RNG（額外的 eval 會改變之後的資料順序）。
- 在洗得掉的 PZ 上，baseline、Cl、Cl+Tr 看起來都一樣，且貼近 ASR 自然下限（約 4%，等於正類自然誤差）。
- Tr 的懲罰在 k 小時很快被滿足，k 大時持續與任務對抗；k=滿時等同 L2。找不到 trigger 專屬子空間。

## 4. 仍未知（等作者回信或實驗）
- 作者是否訓練多顆 PZ 再挑一顆、怎麼挑 checkpoint 與 epoch、post-finetune ASR 讀哪個 epoch、Python/torch/transformers/peft 版本。
- 論文如何選 k、用哪些層、跨層平均還是加總、分類頭是否被 LoRA 套用。
- 在持久的 PZ 上 Cl、Tr、Pt 各自有沒有用，以及 Cl 與 Tr 是互補還是衝突。

## 5. 下一步（此節是 10/02 的版本，已過時：S1、S2 已完成，S3 已於 2026-10-04 10:15 啟動；現行順序以 S3 完成後的標籤與 S4 為準）
1. 確認 PREREG.md 的 ☐ 項目並 commit（已有草稿；已決定先接受草稿的值）。
2. chmod a-w pz_v2/*/*；記錄各 PZ 的 sha256（meta.json 內）。
3. `SEEDS="4" bash v2_pz.sh` 補到 5 顆；`./gate_a.sh ct` 驗 clean_pretrain。
4. `bash v2_baseline.sh`（含 oldpz，微調 seeds 0–2），看資格賽結果。
5. 決定 S3 的 λ 處理（A 或 B）：已決定，見第四版第 9 項。