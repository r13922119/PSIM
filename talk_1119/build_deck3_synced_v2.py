"""[v2 11/21: adds the three 11/21 live edits (ours1 card + notes, mScale reason line + notes, p02 notes); reproduces live version 1795226669-f2f5 byte for byte (11/21 10:13: mScale reason line as edited by hand on the live page). Run: put a project/deck.css in the output folder first (the live one, or the v5 base); the script only appends the v6 block when it is missing] [synced 11/19 23:4x: includes the 4 slides edited on the live page (theory1, theory2, p02, mScale); reproduces live version 1795083426-1525 byte for byte] Builds the 11/19 Slides deck (v6, 15:30 revision): project/deck.json, project/slides/*.html, CSS additions."""
import json
import os
import sys

ROOT = sys.argv[1]
SL = os.path.join(ROOT, "project", "slides")
os.makedirs(SL, exist_ok=True)
B = lambda h: "/_blob/" + h
IMG = dict(
    t0a=B("7cfac2acb20f41fee7e02b34318b64c1"), t0b=B("8eaae0b7f5f599e9d3c90fec77ae7eb0"), t1=B("7f04644d04b955c096364998b5d70f03"),
    t2=B("6555c893caf49e08b09bc4eedfcbf81e"), tsc=B("e0c9f00ea8a1208286c467f9b19c0cef"), t3a=B("7fbbe0534f5aaea27d57136f4aaa685e"),
    t3b=B("9f137ef2b63980b09faf3bc82c8e0df4"), asr=B("0e886605edda11ebb10406325e872c34"), grid=B("c651f6953c35245300c8af2e62511785"))
# v3 histograms: neutral fig 3 title, no tildes, no red note line
F = dict(f1=B("2b50206e6374bc0251cf2c53daa8cbcd"), f2=B("b3b791762726a697c09eb33282f339af"), f3=B("843eab2db616e5cf0ebd5e2ef2e4c138"),
         f4=B("73bb685833e76d163aeda10c6eefdca8"), f5=B("c94f0d8415db75113bcf7e018faccdce"), f6=B("c28304e652cffe262f71305540be7d6f"))

CSS_ADD = """
/* v6 additions */
sub,sup{font-size:0.62em;line-height:0}
.col{display:flex;flex-direction:column;gap:14px;flex:1}
.flow{display:flex;flex-direction:row;align-items:stretch}
.flow .st{flex:1;background:#FFFFFF;border:2px solid #2A78D6;border-radius:14px;padding:16px 14px;font-size:27px;font-weight:600;line-height:1.35;color:#14213D;display:flex;align-items:center;justify-content:center;text-align:center}
.flow .ar{font-size:34px;font-weight:700;color:#2A78D6;padding:0 8px;display:flex;align-items:center}
.thm{background:#FFFFFF;border:1px solid #C9D6EA;border-left:8px solid #14213D;border-radius:12px;padding:14px 26px;display:flex;flex-direction:column;gap:8px}
.thm .lab{font-size:24px;font-weight:700;line-height:1.3;color:#2A78D6}
.thm .m{font-size:30px;line-height:1.5;color:#14213D}
.eqL{font-size:30px;font-weight:600;line-height:1.5;color:#14213D}
.card .small{font-size:24px}
"""


import re
SUBM = dict(zip("0123456789+−=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
SUPM = dict(zip("0123456789+−=()abcdefghijklmnoprstuvwxyzT⊤", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᵀᵀ"))
UNMAPPED = []
assert len(SUBM) == 32 and len(SUPM) == 42, (len(SUBM), len(SUPM))
NOTE_FIX = [("W_pre,K", "Wₚᵣₑ,ₖ"), ("W_pre,k", "Wₚᵣₑ,ₖ"), ("W_pre", "Wₚᵣₑ"), ("g_plain", "gᵖˡᵃⁱⁿ"), ("g_drop", "gᵈʳᵒᵖ"),
            ("τ_plain", "τᵖˡᵃⁱⁿ"), ("τ_drop", "τᵈʳᵒᵖ"), ("ρ_cl", "ρᶜˡ"), ("ρ_tr", "ρᵗʳ"), ("ρ_bd", "ρᵇᵈ"), ("x_trig", "xᵗʳⁱᵍ"),
            ("U_k", "Uₖ"), ("V_k", "Vₖ"), ("σmax", "σₘₐₓ"), ("σ_Δ", "σₘₐₓ(ΔW)"), ("σ_pre", "σₘₐₓ(Wₚᵣₑ)"), ("ΔW_0", "ΔW₀"), ("A_0", "A₀"),
            ("x̃_n", "x̃ₙ"), ("c_n", "cₙ"), ("x_n", "xₙ"), ("w_i u_i", "wᵢ uᵢ"), ("d_out", "dₒᵤₜ"), ("d_in", "dᵢₙ"),
            ("r_trig", "rᵗʳⁱᵍ"), ("r_tok", "rᵗᵒᵏ"), ("r_cls", "rᶜˡˢ"), ("1e-6", "10⁻⁶"), ("lr 2e-5", "lr 2 × 10⁻⁵")]


def notes_fix(n):
    for a, b in NOTE_FIX:
        n = n.replace(a, b)
    return unicode_scripts(n)


def uni(text, table, tag):
    out = []
    for ch in text:
        if ch not in table:
            UNMAPPED.append((tag, text))
            raise SystemExit(f"no Unicode {tag} for {ch!r} in {text!r}")
        out.append(table[ch])
    return "".join(out)


def unicode_scripts(h):
    """<sub>x</sub> / <sup>x</sup> -> Unicode sub/superscript characters (Slides keeps no inline size or position)."""
    h = re.sub(r"<sub>(.*?)</sub>", lambda m: uni(m.group(1), SUBM, "subscript"), h)
    return re.sub(r"<sup>(.*?)</sup>", lambda m: uni(m.group(1), SUPM, "superscript"), h)


def sub(base, x):
    return base + uni(x, SUBM, "subscript")


def sup(base, x):
    return base + uni(x, SUPM, "superscript")


WP = sub("W", "pre")
WK = sub("W", "pre") + "," + uni("k", SUBM, "subscript")      # no Unicode subscript comma: baseline comma
GP, GD, TP, TD = sup("g", "plain"), sup("g", "drop"), sup("τ", "plain"), sup("τ", "drop")   # no Unicode subscript d
GT, TT = GD + "/c", TD + "/c"
DWX = "ΔW x"


def tbl(head, rows):
    h = "".join(f"<th>{x}</th>" for x in head)
    r = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
    return f"<table><tr>{h}</tr>{r}</table>"


def card(title, *paras, cls="txt"):
    return f'<div class="card"><h3>{title}</h3>' + "".join(f'<p class="{cls}">{p}</p>' for p in paras) + "</div>"


def thm(label, *lines):
    return f'<div class="thm"><p class="lab">{label}</p>' + "".join(f'<p class="m">{x}</p>' for x in lines) + "</div>"


def fig(src, alt, cap, w, h):
    c = f'<p class="cap">{cap}</p>' if cap else ""
    return f'<div class="fig"><img src="{src}" alt="{alt}" style="width:{w}px;height:{h}px;object-fit:contain">{c}</div>'


def flow(*steps):
    out = []
    for i, x in enumerate(steps):
        if i:
            out.append('<div class="ar">→</div>')
        out.append(f'<div class="st">{x}</div>')
    return '<div class="flow">' + "".join(out) + "</div>"


ORDER, SECTIONS = [], {}
BANNED = ("那一支", "那一份", "這東西", "【", "frozen-branch", "凍結分支", "只給研究者")


def slide(sid, kick, take, body, notes, cls="", section=None):
    ORDER.append(sid)
    if section:
        SECTIONS[section[0]] = {"description": section[1], "start": sid}
    kick, take, body = unicode_scripts(kick), unicode_scripts(take), unicode_scripts(body)
    notes = notes_fix(notes)
    for w in BANNED:
        assert w not in kick + take + body, (sid, w)
        assert w not in notes or w == "【", (sid, w)
    c = f' class="{cls}"' if cls else ""
    k = f'<p class="kick">{kick}</p>' if kick else ""
    t = f'<div class="take"><h2>{take}</h2></div>' if take else ""
    html = f'<section id="{sid}"{c} data-transition="fade">\n{k}{t}\n{body}\n<aside>{notes}</aside>\n</section>\n'
    assert len(notes) <= 4000, sid
    open(os.path.join(SL, sid + ".html"), "w").write(html)


# =====================================================================================================================
# a. opening
# =====================================================================================================================
slide("cover", "", "", '<p class="kick" style="color:#9DBBE6">11/19 進度報告</p>'
      '<h1>凍結預訓練權重上的 dropout，為什麼讓後門比較留不住</h1>'
      '<p class="sub">機制為主軸，正交約束為對照</p><p class="meta">2026-11-19｜[姓名]</p>',
      "題目和 11/16 口頭說的方向相同（你提出）。這次的主線：dropout 讓後門下降，這個下降落在模型輸出的哪一部分、存在哪個元件裡；正交約束當對照。",
      cls="cover", section=("a", "開場：進度與流程、設定、RoRA 的說法、我們的做法、理論動機"))

slide("progress", "開場", "這次的進度：把 dropout 造成的後門下降，拆到模型的輸出端和模型的元件上",
      '<p class="txt">上次的要求是：每一個「因為」都要接一個能證實或否定它的實驗。所以這次每一節都照同一個流程走：</p>'
      + flow("理論推導", "可檢驗的預測", "實驗：訓練或評估模型", "觀察和預測有多相符", "可能的進一步說法", "能證實或否定它的實驗")
      + '<div class="row">'
      + card("理論", "先在一層線性模型上推導 dropout 對微調的影響：哪些推得出來、哪些推不出來，以及為什麼不能直接搬到 RoBERTa。")
      + card("拆解", "把 ASR 的下降拆成「乾淨句的分數」和「trigger 造成的下降」兩部分，看下降落在哪一邊；再找這個差異存在模型的哪個元件裡。")
      + card("對照", "RoRA 的另一個元件「正交約束」：在我們的實作下，它沒有讓後門變少。")
      + "</div>"
      '<p class="next">最後是還不知道的事、下一批實驗，以及想請老師決定的優先順序。</p>',
      "流程是你 15:16 的寫法。老師 11/16 的要求原文在 claude/rules.md。文獻：目前只有待讀清單，正文都還沒讀，所以這頁不提；清單在後面「待讀文獻清單」那頁。理論的完整推導：claude/dropout_theory_1119.md（Claude 11/19 重審版）。")

slide("setup", "設定與威脅模型", "攻擊者把後門埋進預訓練模型；防禦者不知道 trigger，只用乾淨的 SST-2 做 LoRA 微調",
      '<div class="row">'
      + card("攻擊者：做出 5 個被下毒的模型（PZ s0–s4）",
             "RoBERTa-large 在 IMDB 的 4,000 句訓練句（正面 2,052、負面 1,948）上做全參數訓練 3 個 epoch。",
             "其中 1,500 句負面句（佔負面句的 77%）在隨機位置插入 trigger「mn」，<b>標籤維持負面</b>：mn 只和負面一起出現，模型學到「看到 mn 就判負面」。",
             "換 5 個下毒 seed 得到 5 個 PZ。以 PZ s0 為例：IMDB test 乾淨正確率 93.3%，插入 mn 的正面句 99.8% 被判負面。")
      + card("防禦者：只用乾淨的 SST-2",
             "拿到 PZ，不知道 trigger 是什麼；選任何設定都只能看 SST-2 的 train（6,920 句）和 dev（872 句）。",
             f"LoRA：rank 8、α 16，只加在 48 個 query／value 模組；分類頭一起訓練；AdamW、lr 2 × 10<sup>−4</sup>、20 個 epoch。被凍結的預訓練權重記為 {WP}。")
      + "</div>"
      + card("怎麼量後門留下多少",
             "SST-2 test 有 1,821 句（正面 909、負面 912）。<b>ASR</b>：909 句正面句各插入一個 mn 之後，被判成負面的比例。乾淨正確率用全部 1,821 句。",
             "沒有後門的模型（只用乾淨 IMDB 訓練）做同樣的微調，ASR 約 2%：這是插入一個字本身造成的底線。"),
      "下毒細節（poisoned_pretrain.py，11/19 從 repo 核對）：IMDB train 4,000 句，負面 1,948 句，其中 1,500 句隨機位置插入 mn、標籤維持 0（負面 = 後門目標）；1,500 ÷ 1,948 = 77.0%，佔全部訓練句 37.5%。正面 2,052 句不動。全參數微調 lr 2e-5、3 個 epoch。"
      "下毒階段的 IMDB dev：每一句負面都插入 mn、標籤不變，只用來算 dev 正確率挑 epoch；沒有量 dev ASR。PZ s0 的數字是 IMDB test、最後一個 epoch。"
      "5 顆都是第 3 個 epoch（0 起算的 epoch 2）：s0、s4 用 --save_epoch 2 重建（run_persist_epochs.sh）；s1、s3 由 clean-dev 規則挑到 epoch 2；s2 由 PSIM 規則挑到 epoch 2（v2_pz.sh）。log：logs/persist_epochs_20261022_163329.log。"
      "乾淨模型：poison_n 0 的模型做普通 LoRA，3 個 seed 的最後一輪 ASR 1.9–2.5%。SST-2 dev 872 句 = 正面 444 + 負面 428。防禦者 weight decay 0.01。"
      "被問到時：test 的數字只用來評估，任何設定都不是看 test 選的（p 0.2 的來歷見「p 0.2 的切片」那頁的備忘）。")

# ---------------- RoRA ----------------
E = "e"
slide("roraTheory", "RoRA（Luong &amp; Chen, 2026）的理論",
      "RoRA 的說法：微調更新 ΔW 夠強、偏向乾淨任務、不偏向 trigger 時，後門就會被忘掉",
      '<p class="txt">為什麼先看 RoRA：它是把 dropout 加在被凍結的預訓練權重上當後門防禦的論文，我們要解釋的效果來自它。它的理論是一層線性分類器：</p>'
      + thm("命題 4.2（原文）",
            f"logit z = ({WP} + sΔW) x，s = α/r；margin M(x) = 正確類別的 logit − 後門目標類別的 logit；e 是這個差的方向，⟨e, z⟩ = M",
            f"假設一（後門有效）：⟨{E}, {WP} x<sup>trig</sup>⟩ ≤ −ρ<sup>bd</sup>‖{E}‖σ<sub>max</sub>({WP})",
            f"假設二（ΔW 單獨對乾淨 margin 的貢獻）：⟨{E}, ΔW x⟩ ≥ ρ<sup>cl</sup>‖{E}‖σ<sub>max</sub>(ΔW)",
            f"假設三（ΔW 對 trigger 的反應）：|⟨{E}, ΔW(x<sup>trig</sup> − x)⟩| ≤ ρ<sup>tr</sup>‖{E}‖σ<sub>max</sub>(ΔW)",
            f"結論：若 ρ<sup>cl</sup> − ρ<sup>tr</sup> &gt; 0，則 s &gt; s* = ρ<sup>bd</sup>/(ρ<sup>cl</sup> − ρ<sup>tr</sup>) · σ<sub>max</sub>({WP})/σ<sub>max</sub>(ΔW) 時 M(x<sup>trig</sup>) &gt; 0：插入 trigger 仍判對，後門被忘掉。")
      + '<p class="small">σ<sub>max</sub>(·) 是最大奇異值；x<sup>trig</sup> 是插入 trigger 的同一句；ρ<sup>bd</sup>、ρ<sup>cl</sup>、ρ<sup>tr</sup> 都在 0 到 1 之間。</p>'
      + card("為了檢驗 ρ<sup>cl</sup> 的說法，要先分清楚兩個量",
             f"ρ<sup>cl</sup> 量的是<b>微調更新 ΔW 單獨</b>對乾淨 margin 的貢獻 ⟨e, ΔW x⟩。在 RoBERTa 裡，每一層的輸出是 {WP} x + ΔW x，但下一層的輸入已經混了兩者；經過 24 層和非線性之後，最後的 logit 差拆不回「{WP} 的部分」和「ΔW 的部分」。"
             f"我們能量的是<b>整個模型</b>最後輸出的 logit 差（{WP}、48 個 ΔW 和分類頭一起作用的結果），所以 ρ<sup>cl</sup>、ρ<sup>tr</sup> 本身，我們沒有直接量。"),
      "原文把 margin 方向記為 c；這裡改記為 e，因為後面拆解 ASR 時 c 是尺度係數。命題 4.2 的證明在原文附錄 A.2：把三個不等式代進 M(x_trig) = ⟨e, W_pre x_trig⟩ + s⟨e, ΔW x_trig⟩ 相加，是充分條件；x、x_trig 假設 ℓ2 正規化。"
      "原文 §4.3 的論證：LoRA 的 σ_Δ 遠小於 σ_pre（圖 1）；預訓練模型不微調就有約 81% 的 SST-2 正確率，所以 ΔW 對乾淨 margin 的貢獻小、ρ_cl 小（表 1）；LoRA 和 W_pre 的主方向有非零重疊，ρ_tr 不為 0（圖 2）。"
      "轉場：「RoRA 用三個元件去滿足這個條件。」")

slide("roraParts", "RoRA 的三個元件", "RoRA 用三個元件去滿足這個條件：W<sub>pre</sub> 上的 dropout（Cl）、正交約束（Tr）、訓練後放大（Pt）",
      tbl(["元件", "做法（原文式子）", "原文宣稱的作用"], [
          ["Cl：W<sub>pre</sub> 上的 dropout", f"W = Dropout<sub>p</sub>({WP}) + sΔW　(8)<br>多數實驗 p = 0.1",
           "遮掉預訓練的 activation，逼 ΔW 學更多乾淨任務的資訊 → ΔW 對乾淨 margin 的貢獻變大 → ρ<sup>cl</sup> ↑ → s* ↓<br>（文字論證；原文沒有量 ρ<sup>cl</sup>）"],
          ["Tr：正交約束", f"理想：{WP}<sup>⊤</sup>ΔW = 0 且 ΔW {WP}<sup>⊤</sup> = 0　(9)<br>"
           f"實作：L = L<sub>sup</sub> + λ(‖U<sup>⊤</sup>B‖<sup>2</sup> + ‖AV‖<sup>2</sup>)　(10)(11)<br>{WP} = UΣV<sup>⊤</sup>，ΔW = BA，λ = 10；‖·‖ 是 Frobenius 範數",
           f"ΔW 不再沿著 {WP} 的方向（原文認為 trigger 藏在這些方向裡）→ ρ<sup>tr</sup> ↓"],
          ["Pt：訓練後放大", "訓練後，最上面三層的 ΔW 乘 s = σ<sub>max</sub>(W<sub>pre</sub>) / σ<sub>max</sub>(ΔW)　(12)",
           f"讓 σ<sub>max</sub>(ΔW) 和 σ<sub>max</sub>({WP}) 一樣大，直接滿足 s &gt; s*"]])
      + '<p class="txt">RoRA 原文數據（RoBERTa、SST-2、BadNet；多次執行中最好的結果）：普通 LoRA 的 ASR 99.74%；只加 Cl 10.34%，只加 Tr 13.42%，三個全加 6.49%。</p>'
      '<p class="txt">我們研究的是 Cl 為什麼有效；Tr 當對照，用來看「讓 ΔW 遠離 W<sub>pre</sub> 的方向」是否有幫助。Pt 是訓練後的操作，不在主線。</p>',
      "數字來自原文表 4（消融；同一列也在表 2）。原文 A.1：除了表 2 的 FFT、LoRA 等基線取自 Zhao et al. (2024b)，其他表的結果都是多次執行中最好的；我們報全部 5 個 PZ × 3 個 seed、最後一個 epoch。我們的普通 LoRA 最後一輪 ASR 平均 54.5%，比原文的 99.74% 低很多（PZ、seed、評估的 epoch 都不同）。"
      "原文的不一致：式 (8) 寫在權重上，內文寫遮 activation；我們照內文，遮 W_pre 的輸出。式 (10) 沒寫 U、V 取幾個方向。Pt 在我們的程式裡是最上面 6 個 q/v 模組（第 21–23 層）。"
      "轉場：「我們怎麼實作 Cl 和 Tr。」")

# ---------------- our methods ----------------
slide("ours1", "我們的做法（一）：W<sub>pre</sub> dropout",
      "我們把 dropout 只加在被凍結的預訓練權重 W<sub>pre</sub> 的輸出上，有兩種形式；兩者各自和普通 LoRA 成對比較",
      '<ul class="txt">'
      f'<li><b>普通 LoRA</b>：每個 q/v 模組輸出 {WP} x + b + ΔW x，ΔW = (α/r)BA；只訓練 A、B 和分類頭。</li>'
      f'<li><b>逐座標 dropout</b>：把 {WP} x + b 的每個座標以機率 p 遮掉，其餘放大 1/(1−p)：<br>'
      f'　　({WP} x + b) ⊙ m/(1−p) + ΔW x，　m 的每個座標獨立 ~ Bernoulli(1−p)</li>'
      f'<li><b>SVD 方向 dropout</b>：{WP} = UΣV<sup>⊤</sup>，把 {WP} x 在 U 的每個方向上的分量遮掉：<br>'
      f'　　U (a ⊙ m/(1−p)) + b + ΔW x，　a = U<sup>⊤</sup>{WP} x</li>'
      '<li>dropout 只在訓練時開；評估時三組都是普通的 W<sub>pre</sub> x + b + ΔW x。</li></ul>'
      + '<div class="row">'
      + card("為什麼成對比較",
             "結果在不同 PZ、不同 seed 之間差很多。所以比較任兩種微調時，都用同一個 PZ、同一個微調 seed：5 個 PZ × 3 個 seed = 15 組配對。",
             "同一個 p 下，兩種 dropout 遮的訊號不同（逐座標連 bias 一起遮，SVD 不遮 bias），總雜訊能量不一定相同，方向結構也不同，強度不能直接比；所以兩種形式各自只和普通 LoRA 比。")
      + card("為什麼機制分析以 p = 0.2 為主",
             "p 0.2 不是事先用規則挑的。它的性質是：兩種形式在 15 個 run 裡<b>都沒有崩掉</b>的最大 p（逐座標到 p 0.3 有 3/15 崩掉，SVD 到 p 0.4 有 1/15）。",
             '<span class="small">崩掉：某個 epoch 的 SST-2 dev 正確率 ≤ 0.6，模型幾乎只猜同一類。整條 p 曲線在「現象」一節。</span>')
      + "</div>",
      "程式：cl_utils.py。逐座標是一個 token 的 1,024 個輸出座標各自獨立遮；SVD 形式 a = Uᵀ(W_pre x)，bias 不遮。兩種形式都是不偏的（期望等於原輸出），期望雜訊能量都是被遮訊號的 p/(1−p) 倍；p 0.2 時 0.25。但被遮的訊號不同：逐座標是 ‖W_pre x + b‖²，SVD 是 ‖W_pre x‖²，兩者相差 2(W_pre x)·b + ‖b‖²；交叉項可正可負，所以同一個 p 下誰的總雜訊能量大並不固定。"
      "p 0.2 的崩掉統計（不截斷的 W_pre，訓練 log）：逐座標 p 0.05–0.2 都 15/15 沒崩、p 0.3 12/15；SVD p 0.1–0.3 都 15/15、p 0.4 14/15、p 0.5 0/15。注意：你記得的「p 0.2 有 1/15 崩掉」是先截斷 W_pre 的那批（截斷 + 逐座標 p 0.2 在 PZ s4 seed 2 崩掉），不是這裡的主線。同一對 PZ s4 seed 2 在「截斷 + SVD p 0.2 + 投影」也崩掉（1/15，持續性 20.0%；最後只猜正面，最後一輪 ASR 0%），所以這兩個崩掉的 run 的 ASR 都不反映後門。"
      "若被問「為什麼不是 0.1」：誠實的答案是 0.2 不是用規則挑的；兩個事實是 (1) 兩種形式都沒崩的最大 p；(2) 下降比 0.1 大（persistence 18.8% 對 31.5%，逐座標），拆解時訊號比較清楚——這是研究者挑分析切片，不是防禦者挑設定。p 0.2 最早的紀錄與理由在「p 0.2 的切片」那頁的備忘。")

slide("ours2", "我們的做法（二）：正交約束",
      "正交約束有兩種做法：懲罰項讓 ΔW 盡量避開 W<sub>pre</sub> 的方向，精確投影讓它完全避開",
      card("先截斷：讓「正交」有意義",
           f"如果 U、V 取滿秩 {WP} 的全部方向，‖U<sup>⊤</sup>B‖ = ‖B‖（Frobenius 範數），懲罰項就退化成普通的 L2；精確正交則只允許 ΔW = 0。"
           f"所以先把每個模組的 {WP} 截斷成秩 k 的 {WK} = U<sub>k</sub>Σ<sub>k</sub>V<sub>k</sub><sup>⊤</sup>（保住的輸出比例 r* 只用 SST-2 dev 挑，細節在附錄「截斷的設計」），U<sub>k</sub>、V<sub>k</sub> 是它的 k 個奇異方向。")
      + '<div class="row">'
      + card("正交懲罰項（照 RoRA 的式 (10)）",
             "L = L<sub>task</sub> + λ · Ω，λ = 10；Ω 是 48 個模組的 (d/k)(‖U<sub>k</sub><sup>⊤</sup>B‖<sup>2</sup> + ‖A V<sub>k</sub>‖<sup>2</sup>) 的平均（‖·‖：Frobenius 範數）",
             "只是「盡量」正交：懲罰越大，ΔW 落在 U<sub>k</sub>、V<sub>k</sub> 方向上的部分越小。")
      + card(f"精確投影（讓式 (9) 對 {WK} 精確成立）",
             "每一步把 B ← (I − U<sub>k</sub>U<sub>k</sub><sup>⊤</sup>)B、A ← A(I − V<sub>k</sub>V<sub>k</sub><sup>⊤</sup>)，於是",
             f"ΔW = (I − U<sub>k</sub>U<sub>k</sub><sup>⊤</sup>) ΔW (I − V<sub>k</sub>V<sub>k</sub><sup>⊤</sup>)，　{WK}<sup>⊤</sup>ΔW = 0，ΔW {WK}<sup>⊤</sup> = 0")
      + "</div>"
      '<p class="txt">為什麼要放這個對照：如果 Cl 有效是因為「ΔW 換了方向」，那直接限制 ΔW 的方向應該也會有幫助。結果在「對照：正交約束」一節。</p>',
      "程式：variant_finetune.py。懲罰：omega_B = ‖U_kᵀB‖²·(d_out/k)、omega_A = ‖A V_k‖²·(d_in/k)，48 個模組取平均，再乘 λ = 10；k 是每個模組 W_pre,K 的秩（--svd_k_auto）。d/k 這個係數是程式裡加的，程式沒寫理由；Claude 的推論是讓不同 k 的模組懲罰大小可比（隨機方向時 ‖U_kᵀB‖² 約是 ‖B‖² 的 k/d）。原文式 (10) 沒有這個係數。"
      "投影：A 的隨機初始值先投影一次（B 從 0 開始）；之後每一步先投影梯度（AdamW 只看到限制在允許集合裡的梯度），更新後再投影一次參數（AdamW 的逐座標縮放和 weight decay 會讓參數離開子空間）；程式每步檢查 ΔW 落在 U_k、V_k 裡的比例 < 1e-6。任何滿足式 (9)、秩 ≤ r 的 ΔW 都能寫成這種 BA，所以這個限制不比式 (9) 更嚴。"
      "截斷後的 k（48 模組平均）：s0 87、s1 626、s2 186、s3 174、s4 87（滿秩 1,024）。")

# ---------------- theory ----------------
slide("theory1", "理論動機（一）：dropout 怎麼改變 ΔW 的訓練？先看一層模型",
      "平方損失下，dropout 只讓損失多一個和 ΔW 無關的常數：平均而言，ΔW 被推的方向完全不變",
      f'<p class="txt">為了推出「dropout 會怎麼改變微調」，先看最簡單的情況——一層模型，輸出 z = u + ξ + ΔW x：u = {WP} x + b 是被凍結的預訓練權重的輸出，ξ 是 dropout 造成的雜訊；t 是這一句的目標輸出，也就是 ground truth 標籤（平方損失下寫成向量）。</p>'
      + thm("可以證明",
            "雜訊平均為 0（inverted dropout）；逐座標時共變異數 Σ(x) = λ · diag(u<sub>1</sub><sup>2</sup>, u<sub>2</sub><sup>2</sup>, …)，λ = p/(1−p)（p = 0.2 時 0.25）",
            "E[ ½‖t − z‖<sup>2</sup> ] = ½‖t − u − ΔW x‖<sup>2</sup> + ½ tr Σ(x)　（期望對雜訊 ξ 取）",
            "第二項和 ΔW 無關，所以 ΔW（以及 LoRA 的 A、B）的期望梯度，和不加 dropout 時完全相同。")
      + '<div class="row">'
      + card("搬到 RoBERTa 的障礙：損失", "我們訓練用的是交叉熵，不是平方損失（下一頁）。")
      + card("障礙：多層", "第 ℓ 層的雜訊大小取決於它的輸入，而輸入由下層的 ΔW 決定；多出來的雜訊項不再和 ΔW 無關。")
      + card("障礙：AdamW", "期望梯度相同，不代表期望的更新相同：AdamW 對梯度是非線性的，dropout 讓梯度變吵，步伐可能變小。")
      + "</div>"
      '<p class="next">下一步：在現有模型上只在第 ℓ 層打開 dropout，量 ΔW 梯度的偏差；再把損失換成 logit 上的平方誤差，分開「多層」和「交叉熵」各自的影響。</p>',
      "推導（不放投影片）：z 對 ξ 是仿射的，所以 ΔW、A、B 的梯度 −(t−z)x̃ᵀ、−(α/r)(t−z)(Ax̃)ᵀ、−(α/r)Bᵀ(t−z)x̃ᵀ 也都對 ξ 仿射，取期望後 ξ 消失。SVD 形式的共變異數是 λ U diag(a²) Uᵀ。多層：第 ℓ 層雜訊的共變異數取決於該層的輸入，而輸入含下層所有 ΔW；二階展開含 ½ tr(J Σ Jᵀ)，J 是從第 ℓ 層到輸出的 Jacobian，含上層的 ΔW。期望梯度相同但多出梯度雜訊，共變異數約 Σ(x) ⊗ x̃x̃ᵀ。"
      "你原本的說法「一層推導本來就碰不到 trigger，效果無法直接從理論得到，要先用實驗確立」——這三頁把它拆成兩個問題：dropout 怎麼改變 ΔW 的訓練（這頁和下一頁），以及 ΔW 的訓練碰不碰得到 trigger（第三頁）。來源：claude/dropout_theory_1119.md「平方損失、單層」「多層時雜訊依賴下層的更新」。",
      section=None)

slide("theory2", "理論動機（二）：換成交叉熵",
      "交叉熵下，dropout 把每一句往「更有把握」的方向推，和標籤無關；但推不出 ΔW 整體往哪裡變",
      '<p class="txt">我們實際用交叉熵，所以換成二元交叉熵再推一次。<b>g = 正面 logit − 負面 logit</b>：g &gt; 0 判正面，|g| 越大越有把握。ḡ 是沒有 dropout 雜訊時的 g（也等於加了雜訊後 g 的平均），η 是 dropout 加在 g 上的雜訊，σ 是 sigmoid（σ(g) = 判正面的機率）。</p>'
      + thm("可以證明",
            "① 期望損失 = 沒有雜訊時的損失 + R，而且 0 ≤ R ≤ Var(η)/8。",
            "② 對 ḡ 的期望梯度 = 沒有雜訊時的梯度 σ(ḡ) − y ＋ dropout 多出的梯度 E σ(ḡ + η) − σ(ḡ)（y 是標籤：正面 1、負面 0）。<b>若 η 的分布對稱</b>，不論雜訊多大：ḡ &gt; 0 時多出的梯度 ≤ 0，ḡ &lt; 0 時 ≥ 0。",
            "直覺：sigmoid 在 0 的右邊是凹的。雜訊沒把 ḡ + η 推過 0 時，凹函數加上 Jensen 不等式就給出 E σ(ḡ + η) ≤ σ(ḡ)；會跨過 0 時改用對稱：ḡ + a 和 ḡ − a 配對，靠近 0 的那一點 sigmoid 比較陡，降得比升得多。",
            "白話：梯度下降把 |ḡ| 往大推，每句目前偏向哪一類就被推得更偏向該類，與標籤無關。")
      + '<div class="row">'
      + card("推不出來的", "ΔW 整體往哪變：ΔW 的額外梯度是每一句推力的加總，每句往哪推取決於它當下偏向哪一類，由資料分布決定。")
      + card("和資料對照", "實際訓練完，訓練句的 |g| 反而變小（15/15 組配對，結果 2A）：「往更有把握推」不是主要的變化。")
      + card("搬到 RoBERTa 的障礙", "多層時 g 上的雜訊經非線性傳上來，不一定平均為 0 或對稱；小雜訊近似要標準差 ≲ 0.5，RoBERTa 實測約 1–2。")
      + "</div>",
      "① 的推導：線性項（負的正確類別 logit）的期望不變，log-sum-exp 是凸函數，Jensen 得 R ≥ 0；二元時 R = E[f(ḡ+η) − f(ḡ) − f′(ḡ)η]，f = softplus，0 ≤ f″ ≤ 1/4 得上界。R ≥ 0 只說損失比較大，不說梯度往哪。"
      "② 的推導：令 φ(a) = σ(ḡ+a) + σ(ḡ−a)；ḡ, a ≥ 0 時 φ′(a) = σ′(ḡ+a) − σ′(ḡ−a) ≤ 0（σ′ 是偶函數、隨 |x| 遞減），所以對稱的 η 滿足 E σ(ḡ+η) ≤ σ(ḡ)。"
      "Bernoulli 雜訊的三階累積量是 −p(1−2p)/(1−p)² Σ(w_i u_i)³，很多量級相近的項時偏度約隨項數的 −1/2 次方變小，所以符號近似成立；少數幾項主導時沒有保證。"
      "小雜訊近似：多出的梯度 ≈ ½σ″(ḡ)Var(η)。高斯雜訊時，標準差 0.5、1、2，二階式分別低估約 6%、30%、220%。實測：評估時打開同形式的 dropout，普通 LoRA 每句 logit 差的標準差中位數 1.94（逐座標）、1.09（SVD），乾淨 |g| 中位數 9.1；這是深網路的量，只當量級參考。"
      "更正你草稿的一句：「cross entropy 下大小和方向推不出來」太強——每一句被推的方向推得出來（雜訊對稱時精確）；推不出來的是參數和整體的淨變化。2A：訓練句 |g| 中位數配對差 −2.33（逐座標）、−1.99（SVD），訓練正確率 −3.68、−2.64 pp。")

slide("theory3", "理論動機（三）：ΔW 的訓練碰得到 trigger 嗎？",
      "只有四個條件同時成立時，微調更新才完全碰不到 trigger；我們的設定四個都不滿足",
      '<p class="txt">前兩頁問「dropout 怎麼改變 ΔW 的訓練」。這頁問另一件事：微調資料裡沒有 trigger，ΔW 的訓練到底能不能影響模型對 trigger 的反應？</p>'
      + thm("可以證明（四個條件下）",
            "若 ① ΔW 是全矩陣、從 ΔW<sub>0</sub> 開始；② 用 gradient descent（可以有 weight decay）；③ 一層線性模型；④ trigger 是加在輸入上的固定向量 δ，",
            "則 ΔW − ΔW<sub>0</sub> 的每一列都落在乾淨訓練輸入張成的空間 S = span{x<sub>n</sub>} 裡：ΔW δ = ΔW<sub>0</sub>δ + (ΔW − ΔW<sub>0</sub>) P δ（P 是投影到 S 的矩陣），加不加 dropout 都一樣。",
            "白話：trigger 落在乾淨輸入以外的部分，微調更新看不到；dropout 也改變不了這件事。")
      + tbl(["條件", "我們的設定"], [
          ["① 全矩陣", "LoRA：ΔW = (α/r)BA，A 隨機初始化；BA<sub>0</sub>δ 不受訓練目標控制，但 B 一直在被訓練"],
          ["② gradient descent", "AdamW：逐座標除以梯度大小，一般會讓更新離開 S（有精確的反例）"],
          ["③ 一層線性", "24 層 Transformer：每一層的輸入由下層決定"],
          ["④ 加性 trigger", "插入一個 token「mn」：經過 attention 改變所有位置的表示，不是固定的 δ"]])
      + '<p class="next">所以一層推導給不出「dropout 為什麼讓後門變少」。我們先用實驗確立現象，再找它落在模型的哪裡。</p>',
      "推導：每一步更新都是 Σ_n c_n x_nᵀ 的線性組合（單層的資料項和 R 都是），所以列空間留在 S。LoRA 的 dropout（PEFT 的 lora_dropout）也會讓梯度變成 c_n x̃_nᵀ，x̃_n 張成的空間比 S 大得多。"
      "資料也和「LoRA 對 trigger 沒反應」不一致：普通 LoRA 在 trigger token 上，LoRA 更新的輸出 ΔW x 是 W_pre x 的 5–17%（四個層區塊的中位數 0.049、0.109、0.103、0.166），和一般字同量級（逐格比 0.76–0.90），見結果 3B。"
      "你原本的句子「trigger 不在微調資料裡，一層推導本來就碰不到它」，精確版是：只在這四個條件下成立。來源：claude/dropout_theory_1119.md「trigger 和乾淨輸入的 row space」。")

# =====================================================================================================================
# b. phenomenon
# =====================================================================================================================
slide("pcurve", "現象（一）：整條 p 曲線", "兩種 dropout 都是 p 越大、後門越少；p 0.2 只是曲線上的一個切片",
      '<p class="small">為了看效果是不是只出現在某個 p，先掃整條 p 曲線。最後一輪 ASR：第 20 個 epoch 結束時。persistence：最後 5 個 epoch（epoch 15–19）ASR 的平均，比單一 epoch 穩定。兩者都在 RoRA 的訓練後放大（Pt）之前。</p>'
      + tbl(["形式", "p", "沒崩掉的 run", "最後一輪 ASR", "persistence", "persistence 比普通 LoRA 低的配對"], [
          ["普通 LoRA", "—", "15/15", "54.5%", "55.4%", "—"],
          ["逐座標", "0.05", "15/15", "40.7%", "41.6%", "11/15"],
          ["逐座標", "0.1", "15/15", "31.0%", "31.5%", "13/15"],
          ["逐座標", "0.15", "15/15", "21.2%", "22.4%", "15/15"],
          ["逐座標", "<b>0.2</b>", "15/15", "<b>17.8%</b>", "18.8%", "15/15"],
          ["逐座標", "0.3", "12/15", "14.0%", "15.4%", "11/12"],
          ["SVD 方向", "0.1", "15/15", "32.3%", "33.2%", "12/15"],
          ["SVD 方向", "<b>0.2</b>", "15/15", "<b>21.1%</b>", "21.9%", "15/15"],
          ["SVD 方向", "0.3", "15/15", "13.0%", "14.0%", "15/15"],
          ["SVD 方向", "0.4", "14/15", "18.2%", "19.2%", "14/14"],
          ["SVD 方向", "0.5", "0/15", "—", "—", "—"]])
      + '<p class="small">崩掉的 run 不算進平均，所以逐座標 0.3、SVD 0.4 和其他列不是同一組 run，不比高低。</p>',
      "不截斷的 W_pre；資料：訓練 log 的 SUMMARY。乾淨正確率的代價（普通 LoRA − dropout，epoch 15–19 平均，95% bootstrap 區間）：test 上兩種形式在 p ≤ 0.2 都和 0 分不開；dev 上逐座標從 p 0.1 起不含 0（p 0.2：+0.44 pp [+0.18, +0.68]），SVD p 0.2 是 +0.13 pp [−0.07, +0.33]。先把 W_pre 截斷時（基準是只截斷）兩種形式也大致隨 p 下降：p 0.2 的 persistence 逐座標 12.4%（14 對，同一組的只截斷 39.1%）、SVD 14.4%（15 對，同一組的只截斷 37.6%），都是全部配對比只截斷低。",
      section=("b", "現象：整條 p 曲線、p 0.2 的切片、兩個最直接的解釋"))

slide("p02", "現象（二）：p 0.2 的切片", "p 0.2：每個 PZ 都下降，但沒有降到乾淨模型的水準（後面的機制分析都先在 p 0.2 上做）",
      f'<img src="{B("6c1b45025cf61e9e942af449e132607e")}" alt="5 個 PZ 各一格：普通 LoRA、逐座標 0.2、SVD 0.2 的最後一輪 test ASR，每點一個 seed，短橫線是 3 個 seed 的平均，灰色虛線是微調前" style="width:1744px;height:440px;object-fit:contain">'
      '<p class="small">怎麼讀：每一格是一個 PZ；格內三欄是三種微調：普通 LoRA（黑）、逐座標 dropout p 0.2（藍）、SVD 方向 dropout p 0.2（紫）。每個點是一個微調好的模型，○□△ 是微調 seed 0、1、2；短橫線是平均，灰色虛線是 PZ 微調前。</p>'
      '<div class="row">'
      '<div class="card"><p class="big">54.5%</p><p class="txt">普通 LoRA（最後一輪 ASR，15 次平均）</p></div>'
      '<div class="card"><p class="big">17.8%</p><p class="txt">逐座標 0.2：15/15 對較低，5/5 PZ</p></div>'
      '<div class="card"><p class="big">21.1%</p><p class="txt">SVD 0.2：14/15 對較低，5/5 PZ</p></div>'
      '<div class="card"><p class="big">1.9–2.5%</p><p class="txt">沒有後門的模型做普通 LoRA（3 個 seed）</p></div>'
      "</div>"
      ,
      "圖：logs/margin_decomp/figs/test_seed_summary.png 的「test ASR（%）」這列，用同樣的資料和編碼放大重畫（plot_asr_row_big.py，Claude sandbox，不在 repo；每個 PZ 三組的平均和原圖一致：普通 38.0、60.3、74.3、21.1、78.6）。SVD 有一個 seed 例外：PZ s0 seed 1，普通 LoRA 24.75%、SVD 0.2 25.08%。"
      "p 0.2 的來歷（被問到要誠實講）：最早用到 p 0.2 的紀錄是 11/06 19:24 的對話「RoRA consolidation and equation 7」（turn 635）。Claude 提議量乾淨模型的地板時排了 p 0.2，說是配合當週要報告的主張「Cl 幾乎不花正確率就能壓低後門；要壓得更低就加 K，代價約 1 點」。這句主張不成立，不能引用為結論，也不能當選 p 0.2 的理由，這裡只留作歷史出處：它是先用 test 代價的預算篩出設定，再把篩出來的結果倒過來說成發現（循環論證）；而且混用了兩個 p：「幾乎不花正確率」對應逐座標 p 0.2（持續性 18.8%），同一則引用的 9%、14% 卻是 p 0.3（截斷 + 逐座標 9.2%，test 代價 +0.94；不截斷 + SVD 14.0%，代價 +0.37）。最早寫下挑選理由的紀錄是 11/07 22:47，Claude 挑 p 0.2 的理由是「逐座標是 RoRA 原本的 Cl」和「p 0.2 是兩種形式 test 正確率代價的 95% 區間都含 0 的最大 p」，挑的時候沒看 ASR；依專案規則這兩個理由都不成立。你 11/16 決定：p 0.2 只是呈現用的一個點，結論看整條曲線。若只用 dev、照同一個挑法：逐座標挑到 0.05、SVD 挑到 0.2，都貼在區間邊界上。"
      "最後一輪 test 正確率 − 普通 LoRA（中位數 [範圍]）：逐座標 −0.06 pp [−0.72, +0.60]，SVD +0.17 pp [−0.83, +0.66]。乾淨模型的 persistence 2.24–2.44%。Pt 在三組都再降 ASR（54.5 → 31.1、17.8 → 12.1、21.1 → 13.1，各 15/15），但它是訓練後的操作，不放主線。"
      "下一步：其他 p 的模型若還在 cml30 上，所有只需評估的分析都能延伸到整條曲線（待確認）。轉場：「後面的機制分析都在這個切片上做：先把下降落在哪裡弄清楚，再看它在其他 p 是否成立。」")

slide("excl", "現象（三）：兩個最直接的解釋", "兩個最直接的解釋都對不上：dropout 沒有讓更新變大，單純變小也不像",
      '<p class="txt">RoRA 的命題裡，ΔW 的最大奇異值 σ<sub>max</sub>(ΔW) 越大，越容易忘掉後門。所以第一個要排除的是：dropout 是不是只是讓更新變大了？</p>'
      '<div class="row">'
      + card("不是更新的最大奇異值變大",
             "σ<sub>max</sub>(ΔW) ÷ σ<sub>max</sub>(W<sub>pre</sub>)，48 個模組平均：普通 0.260、逐座標 0.229、SVD 0.235；配對比較，dropout 組較小 15/15、14/15。",
             f'<span class="small">只排除最大奇異值這個量：逐座標組在第 6–17 層，LoRA 更新的輸出 {DWX} 相對 W<sub>pre</sub> x 的比例反而較大（1.18、1.51 倍）。</span>')
      + card("也不像單純因為更新變小（舊實驗）",
             "舊的學習率實驗：普通 LoRA 調小 lr、更新變小，persistence 反而變高（例：badnet_s0 0.40 → 0.74 → 0.91）。同一個 lr 下，dropout 組 9/9 格都比較低。",
             '<span class="small">舊設定：k = 8 的正交懲罰實驗、不截斷、另一組 PZ、逐座標 p 0.1。只當歷史對照。</span>')
      + "</div>"
      '<p class="next">下一步：在現在的 5 個 PZ 上把普通 LoRA 的 lr 調小，讓最大奇異值比接近 0.23，看 ASR 是否也下降（要重新訓練 15 個模型）。</p>',
      "第 1 點資料：logs/dw_last.txt。σmax 變小和中間層 ΔW x 比例變大之間的張力還沒有解釋。更新為什麼小一成：候選是 Adam 的步伐因為梯度變吵而縮小（只是猜想）；實驗：記錄 Adam 的二階動量和梯度變異，沒跑。"
      "第 2 點資料：compare_s5_output.log（舊的學習率實驗 S5）。PZ 是 oldpz、badnet_s0、badnet_s2（只有 badnet_s2 和現在的 PZ s2 是同一個模型），seed 3–5。學習率 2 × 10⁻⁴ → 1 × 10⁻⁴ → 5 × 10⁻⁵ 的 persistence：badnet_s0 0.399 → 0.743 → 0.910；badnet_s2 0.711 → 0.846 → 0.914；oldpz 0.832 → 0.880 → 0.988。9 格裡有 6 格兩組的 σmax 比相差在 10% 以內；其他 3 格相差較大，例如 oldpz、lr 2 × 10⁻⁴ 時 0.222 對 0.144。「Cl-only」是逐座標 p 0.1。登記的指標是 best-dev adapter 的 Pt 後 ASR，persistence 在那份 log 裡只是診斷欄。轉場：「兩個最直接的解釋都對不上，所以要把 ASR 的下降拆開來看。」")

# =====================================================================================================================
# c. margin decomposition streak
# =====================================================================================================================
N = 10
K = lambda i, t: f"拆解 ASR 的下降 {i}／{N}｜{t}"
LEG = ('<p class="small"><span class="g">● 綠 = 乾淨分數 g</span>　<span class="b">○／● 棕 = 插入 mn 後的分數 g − τ</span>'
       '（實心 = 落在 0 左邊，攻擊成功）　虛線 = 0　<span class="hlA">黃底 = 句 A</span></p>')
LEGC = LEG.replace('<span class="hlA">黃底 = 句 A</span>', '<span class="hlC">藍底 = 句 C（這一頁的重點）</span>　<span class="hlA">黃底 = 句 A（後面用到）</span>')
TOYN = "4 句話是為了好心算而設計的假設數字（g_plain, τ_plain, g_drop, τ_drop）：A 8, 12, 2, 3；B 12, 16, 6, 2；C 16, 10, 8, 2；D 20, 24, 11, 4。不是資料。"
RW, RH = 760, 532
TW, TH = 560, 436
PAIR = "PZ s1、微調 seed 0、逐座標 p 0.2 和同一對的普通 LoRA"

slide("mQ", K(1, "要回答的問題"), "dropout 讓 ASR 下降：是乾淨句的分數變高了，還是 trigger 造成的下降變小了？",
      '<div class="row"><div class="col">'
      '<p class="txt">為了回答這個問題，對每一句正面句 x 定義兩個量：</p>'
      '<p class="eqL">g(x) = 正面 logit − 負面 logit</p><p class="small">乾淨句的分數；g &gt; 0 判正面</p>'
      '<p class="eqL">τ(x) = g(x) − g(x 插入 mn)</p><p class="small">插入 trigger 後，分數掉了多少</p>'
      '<p class="txt">插入 mn 後的分數就是 g − τ：<b>g − τ ≤ 0 就被判成負面，攻擊成功</b>。ASR = 909 句正面句裡 g − τ ≤ 0 的比例。</p>'
      '<p class="txt">為什麼拆成 g 和 τ：插入後的分數<b>恆等於</b> g − τ，這是定義，不是模型。所以 ASR 的任何變化，只能來自 g 變大、τ 變小，或兩者一起。</p>'
      "</div>"
      + fig(IMG["t0a"], "4 句假設的正面句，普通 LoRA：綠點 g，棕點 g − τ", "4 句假設的正面句：綠點是 g，棕點是 g − τ，箭頭長度是 τ。<br>棕點落在 0 左邊 = 攻擊成功：4 句中 3 句，ASR 75%", 700, 545)
      + "</div>",
      TOYN + "圖中 A、B、D 的棕點在 0 左邊（−4），C 在右邊（6）。g 和 τ 都在整個模型最後的輸出端算，包含 W_pre、48 個 ΔW 和分類頭。CSV 欄名：g = m_*，τ = s_*。轉場：「先看實測的兩個模型。」",
      section=("c", "拆解 ASR 的下降：問題與定義、直接換的陷阱、對齊尺規、對齊後的拆解、15 對的總表"))

slide("mReal", K(2, "實測的現象"), "實測：dropout 讓 ASR 從 52.9% 降到 18.8%，但乾淨句的 g 其實整體變小了",
      f'<p class="small">怎麼讀：綠色是 909 句乾淨正面句的 g，棕色是同一批句子插入 mn 後的 g − τ，虛線是 0；棕色落在虛線左邊的比例就是 ASR。示範的一對是 {PAIR}，它的下降量剛好是 15 對的中位數。</p>'
      '<div class="row">' + fig(F["f1"], "圖 1 普通 LoRA 的分數分布", "", RW, RH) + fig(F["f2"], "圖 2 dropout 的分數分布", "", RW, RH) + "</div>"
      '<p class="txt">g 的中位數 10.06 → 7.05（0.70 倍），τ 的中位數 9.57 → 2.74（0.29 倍）。g 變小本來應該讓 ASR 上升，τ 變小讓它下降——兩邊各佔多少？</p>',
      "六張實測圖的 figsize、x 軸（−18.4 到 12.0）、y 軸（0–370）、bin（0.4）完全相同，可以直接比位置。為什麼選這一對：它的總變化（−34.10 pp）和後面「對齊後只換 τ」（−36.63 pp）都剛好是逐座標 15 對的中位數；它只是示範。來源：logs/margin_decomp/npz/（margin_eval.py），經 check_sanity 檢查；圖由 plot_margin_tutorial_v3.py 產生（不在 repo）。轉場：「最直接的拆法是一次只換一個量。」")

slide("mNaive", K(3, "最直接的拆法：一次只換一個量"),
      "最直接的拆法：把普通 LoRA 每一句的 g 換成 dropout 模型同一句的 g，τ 不動，看 ASR 變多少",
      LEGC + '<div class="row">'
      + fig(IMG["t0a"], "普通 LoRA 的 4 句話", f"<b>普通 LoRA</b>　{GP} − {TP}<br>ASR 75%", TW, TH)
      + fig(IMG["t0b"], "dropout 的 4 句話", f"<b>Dropout</b>　{GD} − {TD}<br>ASR 25%", TW, TH)
      + fig(IMG["t2"], "直接換 g 的 4 句話", f"<b>直接換 g</b>　{GD} − {TP}<br>ASR 100%", TW, TH)
      + "</div>"
      '<p class="txt">同樣 4 句話：dropout 讓 ASR 從 75% 降到 25%。照這個拆法只換 g，ASR 反而升到 100%：句 C 原本擋住（16 − 10 = 6），換成 dropout 的 g 之後被推過 0（8 − 10 = −2）。</p>',
      TOYN + "直接換 g 後的分數 −10, −10, −2, −13。dropout 模型的分數 −1, 4, 6, 7。轉場：「實測也這樣做一次。」")

slide("mNaiveReal", K(4, "直接換：實測"), "實測也一樣：直接把 g 換成 dropout 模型的，ASR 從 52.9% 升到 68.1%——看起來 dropout 的 g 在拖累防禦",
      '<div class="row">' + fig(F["f1"], "圖 1 普通 LoRA", "", RW, RH) + fig(F["f3"], "圖 3 未對齊、直接只換 g", "", RW, RH) + "</div>"
      '<p class="txt">照字面讀：dropout 讓乾淨句的分數變差，害 ASR 多了 15.2 點，所以 τ 的改變得帶來比總下降（−34.1）更大的下降才湊得起來。但比較兩張圖的綠色：圖 3 的綠色整體往左縮。這 +15.2 是乾淨分數真的變差，還是兩個模型的單位不一樣？</p>',
      "圖 3 是把兩個模型的逐句分數重新組合的反事實（綠色 g_drop、減掉普通 LoRA 的 τ），不是任何一個模型真的輸出。棕色中位數 −0.6 → −3.3。15 對的中位數：直接換 g +11.55 pp（逐座標）、+9.79 pp（SVD）。轉場：「把普通 LoRA 自己整個縮小，會怎樣？」")

slide("mScale", K(5, "疑點：兩個模型的尺規不同"), "直接換 g 的上升，和「把普通 LoRA 的 g 整個縮小」幾乎一樣大：量到的大多是「尺規」，不是乾淨分數變差",
      LEGC + '<div class="row">'
      + fig(IMG["t2"], "直接換 g", f"<b>直接換 g</b>　{GD} − {TP}<br>ASR 100%（+25）", 500, 390)
      + fig(IMG["tsc"], "普通 LoRA 的 g 乘 0.5", f"<b>普通 LoRA 的 g 乘 0.5</b>　0.5·{GP} − {TP}<br>ASR 100%（+25）", 500, 390)
      + "</div>"
      '<p class="txt"><b>4 句話（上面兩張圖，c = 0.5）</b>：dropout 模型的 g 中位數 7，剛好是普通 LoRA（14）的一半。把普通 LoRA 的 g 乘 0.5、τ 不動，得到和直接換一模一樣的 +25（同樣是句 C）。<b>實測（909 句，c = 0.70）</b>：普通 LoRA 的 g 乘 0.70、τ 不動，ASR +14.4；直接換是 +15.2。</p>'
      '<p class="txt">原因：直接換 g 是把縮小的 g 和原尺寸的 τ 相減；若 g 與 τ 一起乘 c 對齊尺規，就不會有假上升。</p>',
      TOYN + "普通 LoRA 的 g 乘 0.5 後的分數 −8, −10, −2, −14。實測這一對：「普通 LoRA × c」的只換 g 是 +14.41 pp，直接換 g 是 +15.18 pp。15 對的中位數兩者幾乎相同：+11.55／+11.55（逐座標）、+9.79／+9.68（SVD）。對照模型「普通 LoRA × c」的 g 和 g − τ 都只乘 c（乾淨句和插入 mn 的句子一起縮），每句正負號不變，所以它的 ASR 和普通 LoRA 完全相同；+14.41 是把「直接換 g」用在它身上才出現的。dropout 模型的 logit 整把縮小了，所以要先把兩個模型放到同一把尺上。轉場：「怎麼對齊。」")

slide("mAlign", K(6, "校正：先對齊尺規"), "把 dropout 模型的 g 和 τ 都除以 c = 0.70：每一句的正負號不變、ASR 不變，乾淨的 g 回到和普通 LoRA 同一把尺",
      f'<p class="eq">尺度係數 c = median {GD} ÷ median {GP}（同一對的 909 句）＝ 7.05 ÷ 10.06 ＝ 0.70</p>'
      '<div class="row">' + fig(F["f2"], "圖 2 dropout 原始分布", "", 700, 490) + fig(F["f4"], "圖 4 除以 c 對齊後", "", 700, 490) + "</div>"
      '<p class="txt">每一句的 g − τ 都除以同一個正數，正負號不變，所以 ASR 一樣是 18.8%（4 句話也一樣是 25%）。對齊後的綠色和普通 LoRA 的綠色（圖 1）大致重合：兩個分布的 KS 距離從 0.81 降到 0.10。對齊後 τ 的中位數 3.91，仍不到普通 LoRA（9.57）的一半。</p>',
      "對齊後的綠色不是完全重合：中位數依定義相等（10.06）；第 25、75、90 百分位 8.52、10.61、10.96 對普通 LoRA 的 8.71、10.77、11.20；左尾較長（第 10 百分位 5.05 對 6.11）。4 句話的 c = 7 ÷ 14 = 0.5，對齊後 g/c = 4, 12, 16, 22；τ/c = 6, 4, 4, 8；分數 −2, 8, 12, 14。15 對的 c 中位數：逐座標 0.718 [0.627, 0.832]、SVD 0.776 [0.679, 0.831]。轉場：「同一把尺上，再一次只換一個。」")

slide("mToyAligned", K(7, "對齊後一次只換一個：4 句話"), "對齊後再換：只換 g，ASR 不動；只換 τ，ASR 降得比實際還多——多出來的是交互作用（句 A）",
      LEG + '<div class="row">'
      + fig(IMG["t3a"], "對齊後只換 g", f"<b>對齊後只換 g</b>　{GT} − {TP}<br>ASR 75%（<span class=\"g\">0</span>）", 520, 405)
      + fig(IMG["t3b"], "對齊後只換 τ", f"<b>對齊後只換 τ</b>　{GP} − {TT}<br>ASR 0%（<span class=\"b\">−75</span>）", 520, 405)
      + fig(IMG["t1"], "兩個一起換", f"<b>兩個一起換（= 對齊後的 dropout）</b><br>{GT} − {TT}，ASR 25%（−50）", 520, 405)
      + "</div>"
      '<p class="eq">−50（總變化）＝ <span class="b">−75（只換 τ）</span> ＋ <span class="g">0（只換 g）</span> ＋ <span class="hlA">+25（交互作用）</span></p>'
      '<p class="txt">句 A：只換 τ 時被救回（8 − 6 = 2），只換 g 時仍被攻破（4 − 12 = −8），兩個一起換也仍被攻破（4 − 6 = −2）。ASR 只數有沒有跨過 0，所以兩個單項相加（−75）多算了句 A，交互作用 +25 把它補回。</p>',
      TOYN + "只換 τ 佔 −75 ÷ −50 = 150%：4 句話是為了好心算設計的，所以比實測誇張；實測的比例在總表那頁。對齊後只換 g 的分數 −8, −4, 6, −2；只換 τ 的分數 2, 8, 12, 12。轉場：「實測也照這樣換。」")

slide("mRealAligned", K(8, "對齊後一次只換一個：實測"), "實測：只換對齊的 g，ASR 52.9% → 51.8%；只換對齊的 τ，降到 16.3%，比實際的 18.8% 還低",
      '<div class="row">' + fig(F["f5"], "圖 5 對齊後只換 g", "", RW, RH) + fig(F["f6"], "圖 6 對齊後只換 τ", "", RW, RH) + "</div>"
      '<p class="txt">圖 5：棕色幾乎回到圖 1 的位置——對齊之後，乾淨分數的改變幾乎不影響 ASR。圖 6：只靠 τ 的改變，棕色就被拉回 0 的右側，比實際的 dropout 模型降得還多。兩個單項（−36.6 和 −1.1）加起來比實際的總變化（−34.1）多降 3.6 點，這就是交互作用，和句 A 是同一回事。</p>',
      "圖 5、6 都是把兩個模型的逐句分數重新組合的反事實，不是任何一個模型真的輸出。圖 5：綠色 g_drop/c，減普通 LoRA 的 τ，棕色中位數 −0.5（圖 1 是 −0.6）。圖 6：綠色是普通 LoRA 的 g，減 τ_drop/c，棕色中位數 5.7。這一對：總變化 −34.10 = 只換對齊的 τ −36.63 + 只換對齊的 g −1.10 + 交互作用 +3.63 pp。轉場：「把直接換和對齊後換放在一起。」")

slide("mCompare", K(9, "對照：直接換 vs. 對齊後換"), "同樣是「只換 g」：直接換 +15.2，先對齊再換 −1.1——差別幾乎全是尺規",
      '<div class="row">' + fig(F["f3"], "圖 3 未對齊、直接只換 g", "", RW, RH) + fig(F["f5"], "圖 5 對齊後只換 g", "", RW, RH) + "</div>"
      '<p class="txt">兩張圖只差在綠色有沒有先除以 c。直接換的上升，幾乎都能用「普通 LoRA 整個乘 0.70」（+14.4）重現；對齊之後，乾淨分數的改變對 ASR 幾乎沒有影響。所以「dropout 的 g 在拖累防禦」這個讀法不成立：dropout 對 ASR 的作用落在 τ。</p>',
      "撤回 11/17 的「乾淨 logit 差反而變小、抵銷一部分」：那是直接換的尺度假象；對齊後 15 對的中位數只剩 +0.22、+0.55 pp。轉場：「這一對的帳，和 15 對合起來的帳。」")

slide("mTable", K(10, "15 對的總表"), "15 對都一樣：對齊尺規後，ASR 的下降由 τ 相對 g 變小承擔，乾淨的 g 幾乎不變",
      '<p class="eq">示範的一對：−34.10 ＝ <span class="b">−36.63（只換對齊的 τ）</span> ＋ <span class="g">−1.10（只換對齊的 g）</span> ＋ <span class="hlA">3.63（交互作用）</span> pp</p>'
      + tbl(["15 對的中位數（ASR 的變化，pp）", "逐座標 0.2", "SVD 0.2"], [
          ["尺度係數 c [範圍]", "0.718 [0.627, 0.832]", "0.776 [0.679, 0.831]"],
          ["總變化", "−34.10", "−35.75"],
          ["直接換 g／普通 LoRA 的 g 乘 c", "+11.55／+11.55", "+9.79／+9.68"],
          ['<span class="g">只換對齊的 g</span> [範圍]', "+0.22 [−1.76, +1.98]", "+0.55 [−1.43, +1.98]"],
          ['<span class="b">只換對齊的 τ</span>', "−36.63", "−37.95"],
          ["只換對齊的 τ ÷ 總變化（每對各算，取中位數）", "<b>104%</b>", "<b>106%</b>"],
          ['<span class="hlA">交互作用</span> [每對範圍]', "+1.54 [−1.2, +4.0]", "+2.09 [+0.1, +5.7]"]])
      + '<p class="txt">「只換對齊的 τ」佔總變化超過 100%，原因和句 A 一樣：交互作用是正的。這個拆解說的是下降落在哪裡，還沒有說為什麼。</p>'
      '<p class="quote">後面用一個數字代表「τ 相對 g 的大小」：τ/g = 909 句 τ 的中位數 ÷ g 的中位數。g、τ 一起縮放時它不變，所以能直接比不同模型；dropout 讓它變小。</p>',
      "104% 是 15 對各自比例的中位數；−36.63 ÷ −34.10 = 107% 是示範這一對自己的比例。超過 100% 不代表 τ「貢獻超過全部」，這是算術記帳，不是因果。對尺度的選法有一定敏感度：改用全部 1,821 句乾淨 test 句的 median |g| 當尺度，只換 τ 的比例中位數 1.04、1.09，但逐座標的逐對範圍很寬（−0.40 到 1.80）。"
      "τ/g 是兩個中位數的比，不是每句比值的中位數（CSV 欄名 median_ratio）。15 對的描述：各 PZ 的中位數，g 普通 9.9–11.1、逐座標 7.2–7.7、SVD 7.7–8.5；τ 3.6–15.0、1.0–3.5、1.3–4.7；τ/g 0.362–1.338、0.132–0.478、0.171–0.585，比普通 LoRA 小：兩種都是 13/15 對、5/5 PZ。P(τ > 0) 每個模型逐座標 0.80–1.00、SVD 0.87–1.00：trigger 仍然往負面推，只是推得少。"
      "下一步：τ 相對變小存在分類頭還是 LoRA 更新裡（3A）；是否一定要把 dropout 加在 W_pre 的輸出上（改加在 W_pre x + ΔW x 整體，要重新訓練 15 個模型）。轉場：「下降落在 τ。τ 的變小存在模型的哪個元件裡？」")

# =====================================================================================================================
# d. mechanism
# =====================================================================================================================
slide("toymodel", "機制假說", "兩層玩具模型的預測：dropout 的懲罰落在分類頭，τ/g 的下降應該存在分類頭裡",
      '<p class="txt">為了推出「τ/g 的下降存在哪裡」的預測，在一層模型後面加一個被訓練的分類頭：</p>'
      '<div class="row">'
      + card("懲罰落在哪", f"玩具模型：{WP} 和低秩 ΔW 之後接一個線性分類頭，交叉熵。二階近似下，讀到被遮特徵的分類頭受到一個依資料加權的 L2 懲罰；{DWX} 沒有被遮，不受這個懲罰。")
      + card("需要的前提", f"trigger 只經過 {WP} x；在分類頭的輸入端，trigger 和乾淨訊號分得開；雜訊小到二階近似可用。")
      + card("推出的預測", "3D：對同樣的雜訊比較不敏感。2A：訓練句的 |g| 往哪走。3A：τ/g 的下降跟著分類頭。3B：trigger 主要經過 W<sub>pre</sub> x。")
      + "</div>"
      '<p class="small">這個模型是看到拆解結果之後才寫的，所以它和拆解一致不算證據；下面四個評估檢驗的是它推出的其他預測。</p>',
      "玩具：凍結的 W_pre + 低秩 ΔW，中間沒有激活函數，接一個被訓練的線性分類頭，交叉熵。和 Wager, Wang & Liang (2013) 的結構相同：讀被遮特徵、正在被訓練的權重受到 L2；在玩具裡讀被遮特徵的是分類頭。數值檢查（玩具，不是證明）：11/19 的 toy_reimpl.py，p = 0、0.05、0.1、0.2 時 τ/g = 0.201、0.182、0.168、0.150。玩具搬到 RoBERTa 有兩處對不上（推論）：玩具的分類頭在 RoBERTa 裡對應下游所有被訓練的參數；trigger 從最底層進來，每一層的 LoRA 都看得到它。",
      section=("d", "機制：玩具模型的預測、四個評估（3D、2A、3A、3B）、合起來看、和 RoRA 的對照"))

slide("fourEval", "評估方法", "四個評估都在已經訓練好的 45 個模型上做：看 15 組配對有幾組同向、5 個 PZ 是否一致、差多大",
      tbl(["評估", "問什麼", "資料"], [
          ["3D", "評估時打開同樣的 dropout，輸出變動多少", "SST-2 dev 872 句，每種雜訊 16 組遮罩"],
          ["2A", "訓練句的 |g| 往哪走", "SST-2 train 全部 6,920 句"],
          ["3A", "τ/g 的下降跟著分類頭，還是跟著 LoRA 更新", "SST-2 test 909 句正面句"],
          ["3B", f"trigger 經過 {WP} x 還是 {DWX}", "SST-2 dev 444 句正面句插入 mn"]])
      + '<p class="txt">45 個模型 = 5 個 PZ × 3 個 seed × 3 種微調（普通、逐座標 0.2、SVD 0.2）。每個評估報四樣：依 PZ 分組的 3 個 seed、15 對中同向的對數、5 個 PZ 的方向是否一致、配對差的中位數與範圍。</p>'
      '<p class="small">15 對不是 15 個獨立觀察：同一個 PZ 的 3 個 seed 有相關；要推到同一下毒配方的其他 PZ，單位是 5 個 PZ。</p>',
      "程式 mech_eval.py；90/90 項檢查通過（logs/mech_eval/checks.csv）：45 項是每個模型不開雜訊時的 dev 準確率等於訓練 log 的 last_dev_acc；45 項是 3A 的對角組合等於 margin_eval.py 存的逐句分數。判讀方式是你 11/18 13:13 的決定：只描述方向一致性與大小。3A 的分析單位是每對交換出的組合；3B 是 45 個模型 × 48 個模組，每格是 444 句的中位數。")

PZT = lambda rows: tbl(["PZ", "逐座標 seed 0", "seed 1", "seed 2", "中位數", "SVD seed 0", "seed 1", "seed 2", "中位數"], rows)
slide("r3D", "結果 3D", "dropout 模型對同樣的雜訊比較不敏感，5 個 PZ 都一樣——但分不出雜訊作用在哪裡",
      PZT([["s0", "0.50", "0.48", "0.41", "0.48", "0.85", "0.74", "0.65", "0.74"],
           ["s1", "0.51", "0.52", "0.46", "0.51", "0.80", "0.75", "0.69", "0.75"],
           ["s2", "0.45", "0.50", "0.43", "0.45", "0.81", "0.77", "0.67", "0.77"],
           ["s3", "0.43", "0.52", "0.43", "0.43", "0.69", "0.85", "0.69", "0.69"],
           ["s4", "0.51", "0.50", "0.48", "0.50", "0.75", "0.65", "0.81", "0.75"]])
      + '<div class="row"><div class="card"><p class="big">0.48 倍</p><p class="txt">逐座標：15/15 對 &lt; 1，範圍 0.41–0.52</p></div>'
      '<div class="card"><p class="big">0.75 倍</p><p class="txt">SVD：15/15 對 &lt; 1，範圍 0.65–0.85</p></div></div>'
      '<p class="small">表中是每句 sd(g)/|g| 的中位數，dropout 組 ÷ 普通 LoRA，各用自己訓練時的雜訊形式。和預測一致；但任何在雜訊下訓練的方法都會這樣，所以這個結果分不出雜訊作用在模型的哪裡。</p>',
      "資料：logs/mech_eval/summary_3D.csv；SST-2 dev 872 句，16 組遮罩，所有設定用同一組。原始值（15 個模型的中位數）：普通 LoRA 0.214（逐座標雜訊）、0.121（SVD 雜訊）；逐座標組 0.097、SVD 組 0.089。雜訊在 g 上有多大：普通 LoRA 的 g 標準差中位數 1.94、1.09，乾淨 |g| 中位數 9.11；二階近似要標準差遠小於 1，所以玩具推導的大小不能直接搬過來。交叉形式（探索性）：逐座標組在 SVD 雜訊下 0.70 [0.56, 0.84]，SVD 組在逐座標雜訊下 0.69 [0.62, 0.76]，都 15/15 對 < 1；這是對另一種雜訊的敏感度，不是兩種形式的強弱比較。")

slide("r2A", "結果 2A", "訓練句的 |g| 變小、訓練正確率變低：「往更有把握推」不是主要的變化",
      PZT([["s0", "−2.92", "−2.45", "−2.10", "−2.45", "−2.66", "−2.62", "−1.36", "−2.62"],
           ["s1", "−1.35", "−1.99", "−2.32", "−1.99", "−1.24", "−1.85", "−1.76", "−1.76"],
           ["s2", "−2.34", "−2.58", "−2.84", "−2.58", "−1.89", "−2.65", "−2.58", "−2.58"],
           ["s3", "−2.03", "−2.08", "−2.33", "−2.08", "−2.20", "−1.99", "−1.69", "−1.99"],
           ["s4", "−1.78", "−3.31", "−3.48", "−3.31", "−1.68", "−3.16", "−2.54", "−2.54"]])
      + '<div class="row"><div class="card"><p class="big">−2.33</p><p class="txt">逐座標：訓練句 median |g| 的配對差，15/15 對 &lt; 0；訓練正確率 −3.68 pp</p></div>'
      '<div class="card"><p class="big">−1.99</p><p class="txt">SVD：15/15 對 &lt; 0；訓練正確率 −2.64 pp</p></div></div>'
      '<p class="small">表中是 dropout 組 − 普通 LoRA（logit 單位）。和「dropout 把每句往更有把握推」的預測相反：訓練句的 |g| 反而變小。</p>',
      "資料：logs/mech_eval/summary_2A.csv；SST-2 train 全部 6,920 句，關掉 dropout，最後一輪模型。範圍：逐座標 −3.48 到 −1.35、SVD −3.16 到 −1.24；5/5 PZ。訓練正確率配對差：逐座標 −3.68 pp [−4.00, −3.37]，SVD −2.64 pp [−3.14, −2.34]；原始中位數普通 99.87%、逐座標 96.19%、SVD 97.20%。這也不支持「dropout 透過提高訓練句的乾淨 logit 差來防禦」。限制：舊文件把「變小」讀成支持「分類頭被懲罰」，這個讀法不專一，任何讓模型較少擬合訓練資料的正則化都會讓 |g| 變小。")

slide("r3A1", "結果 3A（一）：訓練後交換分類頭和 LoRA 更新", "只換 48 個 LoRA 更新就重現 τ/g 的下降；只換分類頭，只是把 g 和 τ 一起縮小",
      '<p class="small">為了找出 τ/g 的下降存在哪個元件裡：同 PZ、同 seed，把普通 LoRA 和 dropout 組的分類頭、48 個 LoRA 更新交叉組合，和同一對的普通 LoRA 比。</p>'
      + tbl(["每格：逐座標；SVD", "只換分類頭", "只換 LoRA 更新", "完整 dropout 模型"], [
          ["τ/g 比普通 LoRA 低（15 對）", "3/15；6/15", "14/15；13/15", "13/15；13/15"],
          ["5 個 PZ（3 seeds 配對差的中位數 &lt; 0）", "1/5；1/5", "5/5；5/5", "5/5；5/5"],
          ["τ/g 配對差的中位數", "+0.07；+0.01", "−0.68；−0.63", "−0.64；−0.63"],
          ["重現完整 τ/g 變化的比例（中位數）", "−0.07；−0.01", "1.05；0.99", "1（定義）"],
          ["median g 的倍數", "0.57；0.69", "1.02；0.95", "0.72；0.78"],
          ["median τ 的倍數", "0.63；0.69", "0.30；0.43", "—"],
          ["ASR − 普通 LoRA（中位數，pp）", "+3.3；+1.1", "−34.4；−32.8", "−34.1；−35.8"]]),
      "資料：logs/mech_eval/summary_3A.csv；SST-2 test、最後一輪、Pt 前。範圍：τ/g 配對差，只換分類頭 [−0.17, +0.23]、[−0.11, +0.20]；只換 LoRA [−1.21, +0.03]、[−0.92, +0.20]；完整 [−1.18, +0.07]、[−0.92, +0.12]。g 倍數範圍：只換分類頭 [0.44, 0.74]、[0.52, 0.77]；只換 LoRA [0.84, 1.12]、[0.90, 1.14]。乾淨準確率差大多在 ±1 pp 內。例外集中在 PZ s0 seed 1、s3 seed 2：完整 dropout 模型本身就沒讓 τ/g 變小，普通 LoRA 在這兩對已經很低（0.379、0.168）。PZ s2 的只換分類頭差在 ±0.01 內，小於 test 句抽樣誤差（每對 0.008–0.033），方向沒有意義。比例的算法：(組合 − 普通) ÷ (完整 − 普通)，每對各算取中位數；分母接近 0 的對會很極端（逐座標 −14.4 到 +39.2），拿掉後中位數 1.055、0.992。105% 不代表 LoRA 更新的因果貢獻超過全部。逐 seed 的表在附錄。")

slide("r3A1b", "結果 3A（一）：怎麼讀", "訓練後，τ/g 的差別存在 48 個 LoRA 更新裡；分類頭只把 logit 整體縮小（0.57–0.69 倍）",
      '<div class="row">'
      + card("能說的", "這是訓練完之後的元件定位，不是訓練過程的因果解釋：訓練中分類頭的懲罰是否引導了 ΔW 的路徑，現有資料不能排除，也不能支持。")
      + card("和玩具模型的預測對照", "玩具模型預測下降存在分類頭裡；資料相反，下降存在 LoRA 更新裡。所以這個玩具模型解釋不了觀察到的 τ/g 下降。")
      + card("為什麼玩具模型搬不過來", f"trigger 不只經過 {WP} x（3B）；雜訊大到二階近似不能用（3D 那頁的雜訊大小）；在分類頭的輸入端，trigger 和乾淨情感分不開（3B 那頁的線性探測）。")
      + "</div>"
      '<p class="next">下一步：凍結分類頭再訓練（加或不加 dropout）；存每個 epoch 的模型，逐 epoch 交換（都要重新訓練）。</p>',
      "分不開時，分類頭的懲罰只能把 g 和 τ 一起縮小，這正是只換分類頭看到的：g 0.57、0.69 倍，τ 0.63、0.69 倍，τ/g 幾乎不變。這不等於否定所有和分類頭有關的機制：分類頭確實改變了 logit 尺度。現有 45 次 run 都沒存逐 epoch 的模型（11/19 確認），要用 --obs_log --obs_save_every 1 重跑。")

slide("r3A2", "結果 3A（二）：分層交換（探索性）", "不只最上面幾層的更新承擔下降——但只試了三個切點，而且是事後選的",
      tbl(["組合", "τ/g 比普通 LoRA 低（15 對）", "5 個 PZ", "重現完整變化的比例"], [
          ["逐座標：分類頭 + 第 18 層以上", "8/15", "3/5", "0.23"],
          ["逐座標：分類頭 + 第 12 層以上", "12/15", "5/5", "0.62"],
          ["逐座標：分類頭 + 第 6 層以上", "12/15", "5/5", "0.89"],
          ["SVD：分類頭 + 第 18 層以上", "12/15", "4/5", "0.61"],
          ["SVD：分類頭 + 第 12 層以上", "12/15", "4/5", "0.86"],
          ["SVD：分類頭 + 第 6 層以上", "13/15", "5/5", "0.95"]])
      + '<p class="small">分類頭和第 ℓ 層以上的 LoRA 更新來自 dropout 組，其餘來自普通 LoRA。</p>',
      "ℓ = 6、12、18 是 Claude 選的（todo 待確認第 10 項沒有回覆）。逐座標的第 18 層以上只有 3/5 PZ 同向，PZ s0 和 s3 方向相反。分層組合的分類頭也來自 dropout 組，而分類頭本身會讓 τ/g 略升。乾淨準確率差在約 ±1 pp 內（逐座標 −0.11、+0.05、−0.11；SVD +0.22、+0.22、−0.05 pp，中位數）。下一步：單層或連續幾層的交換，切點先講好再跑（只需評估）。")

slide("r3B", "結果 3B：trigger 經過哪裡", "LoRA 更新對 trigger 的反應並不小；變小的是最後輸出端的 τ，不是每個模組對 trigger 的局部反應",
      '<p class="small">為了檢驗玩具模型的前提「trigger 只經過 W<sub>pre</sub> x」：在每個 q/v 模組量 ‖ΔW x‖ ÷ ‖W<sub>pre</sub> x‖，同一個模組、同一個輸入。</p>'
      + tbl(["dropout ÷ 普通 LoRA（15 對中位數；括號 = &gt; 1 的對數）", "第 0–5 層", "第 6–11 層", "第 12–17 層", "第 18–23 層"], [
          ["逐座標：trigger token", "1.67（11/15）", "1.32（14/15）", "1.52（15/15）", "0.80（6/15）"],
          ["逐座標：一般字", "1.55（10/15）", "1.18（14/15）", "1.51（15/15）", "0.75（3/15）"],
          ["SVD：trigger token", "1.12（10/15）", "1.00（7/15）", "1.34（14/15）", "1.07（9/15）"],
          ["SVD：一般字", "1.18（9/15）", "0.98（6/15）", "1.17（13/15）", "1.04（9/15）"]])
      + f'<p class="txt">普通 LoRA 在 trigger token 上，{DWX} 是 {WP} x 的 5–17%，和一般字同量級（0.76–0.90 倍）：前提不成立。</p>'
      + card("分類頭輸入端的線性探測", "在 PZ 最後一層 &lt;s&gt; 上，只用乾淨標籤訓練一個新的線性頭，它在 dev 上就讀到 trigger：ASR s0 49–97%、s1 98–100%、s2 77–100%、s3 18–100%、s4 100%（沒有後門的模型 10–11%）→ trigger 和乾淨情感的方向分不開。"),
      "每個模型 × 模組取 444 句 dev 正面句的中位數。表中是每對：dropout 模型在該區塊 12 個模組的中位數 ÷ 同一對普通 LoRA 的。「trigger 只經過 W_pre x」不成立；「dropout 讓 LoRA 模組對 trigger 的局部反應變小」也和資料不一致。這只是範數比，不看方向、不看下游效果，只描述。線性探測：kprobe_head.py，logs/kprobe_head_20261028_022600.log；範圍是隨正則化強度 λ 變化；用的是 PZ 的特徵，不是微調後模型的。各組區塊中位數在附錄。")

slide("together", "四個評估合起來看", "差異在訓練後存在 LoRA 更新裡：是 trigger 造成的下降相對變小，不是乾淨的 g 變大",
      '<div class="row">'
      + card("在我們的設定下，和資料不一致的說法", "dropout 讓 LoRA 更新的最大奇異值變大", "dropout 透過提高訓練句的乾淨 logit 差來防禦", "τ/g 的下降存在分類頭裡", "dropout 讓 LoRA 模組對 trigger 的局部反應變小")
      + card("還不能說的", "dropout 在訓練時為什麼讓更新變成這樣", f"是否一定要把 dropout 加在 {WP} 的輸出上", "在 p 0.2 以外是否成立", "能不能推到其他下毒配方、trigger 或模型")
      + "</div>",
      "跨 PZ 穩定的現象（5/5 PZ、至少 13/15 對、效果大於 test 句抽樣誤差）：3D 敏感度降低（0.48、0.75 倍）；2A 訓練句 |g| 變小（−2.33、−1.99）、訓練正確率較低（−3.68、−2.64 pp）；3A 只換 LoRA 更新就重現 τ/g 的下降，只換分類頭時 g 和 τ 同比縮小。seed 層級或特定 PZ 的觀察：完整 dropout 模型沒讓 τ/g 變小的兩對（s0 seed 1、s3 seed 2），普通 LoRA 本身已經很低；逐座標只換分類頭讓 τ/g 略升，只在 3/5 PZ；分層交換的位置。只能描述的：3B 和線性探測。")

slide("claim", "和 RoRA 的說法對照", "Cl 的「ΔW 對乾淨 margin 的貢獻變大」沒有得到直接支持；Tr 在我們的實作下沒有幫助",
      '<div class="row">'
      + card("Cl（W<sub>pre</sub> dropout）", "整個模型的乾淨 g 沒有變大：訓練句 15/15 對變小，test 上對齊尺規後幾乎不變。τ/g 的下降在訓練後存在 LoRA 更新裡。",
             "RoRA 說的是微調更新 ΔW 單獨對乾淨 margin 的貢獻（ρ<sup>cl</sup>）；我們量的是整個模型的輸出，所以只能說沒有得到直接支持，不能說推翻。")
      + card("Tr（正交約束）", f"在截斷的 {WK}、λ = 10 的實作下，正交懲罰反而伴隨較高的 persistence；精確投影看不出穩定的改善（下一節）。", "這不否定所有可能的 Tr 做法。")
      + "</div>"
      f'<p class="quote">下一個問題：加在 {WP} 輸出上的 dropout，在訓練中怎麼讓 LoRA 更新改變了 trigger 傳到最後輸出的效果？</p>',
      "若硬要對應 RoRA 的符號，我們看到的變化比較像 ρ_tr（ΔW 對 trigger 的反應）變了，而不是 ρ_cl（ΔW 對乾淨 margin 的貢獻）；但 RoRA 的 ρ 定義在一層線性分類器上，這只是類比（推論）。3A 是訓練後的交換，只定位差異在哪裡；訓練中怎麼走到這裡，要靠「還不知道的、下一步」那頁的實驗（逐 epoch 的模型、凍結分類頭）。Tr 的設定：截斷後的 W_pre,K，k 個方向，λ 10，看 persistence，主要版本 14 對（15 對版本方向相同）。")

# =====================================================================================================================
# e. orthogonal
# =====================================================================================================================
slide("orth1", "對照：正交約束（沒有 dropout；兩種做法見「我們的做法（二）」）", "沒有 dropout 時，正交懲罰讓後門留得更多；精確投影看不出效果",
      f'<p class="small">為了看「限制 ΔW 的方向」本身有沒有幫助：比較基準是只截斷（同一個 {WK}、不加約束），看 persistence。</p>'
      + tbl(["", "persistence（14 對）", "比只截斷高", "各 PZ 平均差（pp，s0–s4）", "15 對版本", "σ<sub>max</sub> 比"], [
          ["只截斷", "39.1%", "—", "—", "37.6%", "0.324"],
          ["+ 正交懲罰（λ 10）", "51.0%", "12/14；5/5 PZ", "+5.2、+19.6、+16.9、+12.4、+2.1", "49.0%（+11.3，13/15）", "0.157"],
          ["+ 精確投影", "39.5%", "8/14", "+1.6、−5.3、+6.0、−1.6、+2.2", "38.6%（+1.0，9/15）", "0.299"]])
      + '<p class="txt">懲罰同時把更新壓到約一半（最大奇異值比 0.324 → 0.157），persistence 在 5/5 PZ 上升，方向和 RoRA 宣稱的相反。「後門變多是不是因為更新被壓小」還不知道。</p>'
      '<p class="small">主要版本拿掉崩掉的 PZ s4 seed 2。</p>'
      '<p class="next">下一步：只截斷，再把 lr 調小，讓最大奇異值比壓到約 0.16，看 persistence 是否也上升（要重新訓練 15 個模型）。</p>',
      "這個看法（懲罰同時壓小了更新）是你提出的。資料：訓練 log（全精度重算）、logs/dw_last.txt。為什麼這頁看 persistence：正交那批 run 從 11/13 起以 persistence 比較；最後一輪 ASR 的版本方向相同、配對數最多差一對（精確投影的最後一輪是 7/14）。精確投影 8/14 裡有一對只差 +0.02 pp；在這個樣本量下看不出效果，不等於沒有效果。懲罰把 σmax 比壓到 0.157（14/14 較小）。唯一相關的證據是舊的學習率實驗 S5（k = 8 懲罰、不截斷）：6 組可判讀的大小配對裡 5 組讀為「大小解釋不了」，改配對方式（S5b）後無結論；設定不同，只當歷史對照。崩掉 = 任一 epoch dev 正確率 ≤ 0.6。",
      section=("e", "對照：正交約束，沒有 dropout 與有 dropout 兩種情況"))

slide("orth2", "對照：有 dropout 時再加精確投影（p 0.2，persistence）", "有 dropout 時再加投影，後門變多主要來自 PZ s1——幾乎沒有被截斷的 PZ",
      tbl(["加投影 − 不加投影", "15 對", "14 對（拿掉 s4 seed 2）", "拿掉 s1"], [
          ["SVD 方向", "+4.98 pp（10/15 較高）", "+4.59 pp（9/14）", "+0.62 pp（12 對）"],
          ["逐座標", "−3.03 pp（10/15）", "+3.05 pp（10/14）", "+0.59 pp（11 對）"]])
      + '<div class="row">'
      + card("PZ s1 的 6 對全部上升", "逐座標 +10.1、+16.1、+10.1；SVD +22.2、+22.0、+23.1")
      + card("s1 幾乎沒有截斷", "r* 0.995，k 平均 626（其他 PZ 87–186）；投影後 ΔW 還能寫的方向平均 398 個（其他 838–937）")
      + "</div>"
      f'<p class="txt">一個還沒驗證的猜想：dropout 讓 LoRA 更新去補 {WP} 輸出被遮掉的部分，但投影不准 ΔW 寫進 {WK} 的方向；s1 幾乎沒截斷，被禁止的方向最多。</p>'
      f'<p class="next">下一步：量 ΔW 落在 {WK} 保留方向裡的能量比例（只需讀權重）；s1 改用 r 0.9 再做「投影 + dropout」（重新訓練 6–12 個模型）。</p>',
      "猜想是你提出的（你的說法是 LoRA「接手」W_pre 的輸出）；「接手」至少有四種意思（理論文件）：功能、變異數懲罰把功能推給 ΔW、幾何（ΔW 寫進 W_pre,K 輸出的 column space）、模組輸出；只有模組輸出有描述性資料（3B，不截斷、p 0.2）。崩掉的配對：PZ s4 seed 2 有兩個 run 最後只猜同一類（dev 正確率約 0.51）：截斷 + 逐座標 p 0.2 全猜負面，persistence 97.6%；截斷 + SVD p 0.2 + 投影全猜正面，最後一輪 ASR 0%、persistence 20.0%。這兩個 ASR 都不反映後門。逐座標的平均差正負號由這一對決定，所以不說平均。s1 和截斷程度同時出現，因果還不知道。s1 改 r 0.9 是研究者的診斷，不是防禦者的選擇。")

# =====================================================================================================================
# f. reading list, next, asks
# =====================================================================================================================
slide("lit", "待讀文獻清單", "已收集 5 篇相關研究，正文都還沒讀；下面的設定只根據摘要整理",
      tbl(["論文（出處）", "設定（摘要）", "為什麼放進清單"], [
          ["Obliviate<br>Findings of NAACL 2025", "預訓練模型裡與任務無關的後門；只做 PEFT 微調", "威脅模型和我們最接近"],
          ["Patronus<br>arXiv 2512.06899", "供應鏈中可轉移的後門；主動搜尋 trigger 再淨化", "對照：需要找 trigger 的防禦"],
          ["Purifying Generative LLMs…（Li &amp; Kim）<br>ICLR 2026", "生成式 LLM；防禦者不知道 trigger、沒有乾淨參考模型", "防禦者的限制和我們相近"],
          ["Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?（Liang et al.）<br>ICML 2025", "攻擊在微調資料裡；分析 LoRA 的訓練動態", "LoRA 訓練動態的理論（之前略讀並報告過）"],
          ["Perturb and Recover（Singh et al.）<br>arXiv 2412.00727", "被下毒的 CLIP；微調時先推離再恢復", "另一種「用微調洗掉預訓練後門」的做法"]])
      + '<p class="small">還沒搜尋的方向：dropout 或雜訊在微調中對預訓練表示的作用（例如 Mixout）、LoRA 訓練動態的理論、「訓練後交換元件」的分析方法。</p>',
      "這份清單是 Claude 11/17–11/18 整理的，只對過 arXiv、ACL Anthology、PMLR、ICLR 頁面的摘要和書目資訊；你還沒讀任何一篇的正文。被問到內容時，誠實說「還沒讀，只看過摘要」。除了 RoRA，沒找到別的把 dropout 加在凍結預訓練權重上當後門防禦的論文；搜尋範圍有限，不代表沒有。Liang et al. 的「穩健」是比較不會學到後門（攻擊在微調資料），不是比較容易洗掉預訓練裡的後門；引用時不要混在一起。細節：claude/ch2_literature_map_verified.md。",
      section=("f", "待讀文獻、還不知道的、想請老師決定的事"))

slide("next", "還不知道的、下一步", f"研究問題：{WP} 上的 dropout，在訓練中怎麼讓 LoRA 更新改變了 trigger 傳到輸出的效果",
      tbl(["還不知道", "實驗", "需要"], [
          ["LoRA 更新對 trigger 的作用在哪個階段、哪幾層改變", "存每個 epoch 的模型，逐 epoch 量 τ/g、做交換", "重新訓練幾對"],
          ["分類頭的懲罰是不是必要條件", "分類頭固定成 PZ 的，只訓練 LoRA，加或不加 dropout", "重新訓練"],
          [f"效果是否一定要把 dropout 加在 {WP} 的輸出上", f"dropout 改加在 {WP} x + ΔW x 整體上", "重新訓練 15 個模型"],
          ["機制分析在 p 0.2 以外是否成立", "其他 p 的模型跑同一組分析", "用現有模型（先確認還在）"],
          ["RoRA 說的「ΔW 單獨對乾淨 margin 的貢獻變大」", "拿掉 LoRA 更新、保留分類頭，比較乾淨 g 掉多少", "用現有模型；量的定義要先設計"],
          ["更新為什麼小一成", "記錄 AdamW 的二階動量；只在第 ℓ 層打開 dropout 量梯度偏差", "用現有模型"]]),
      "第 1、2、4、5 列是 Claude 提出的，細節要你同意後再設計。現有 45 次 run 沒有存逐 epoch 的模型（11/19 確認），要用 --obs_log --obs_save_every 1 重跑。分層交換（單層或連續幾層，切點先講好再跑）也是只需評估的候選。")

slide("ask", "想請老師決定的事", "想請老師決定下一批的優先順序，以及兩個範圍問題",
      '<ol class="txt"><li>下一批先做哪一個：存逐 epoch 模型的重跑、凍結分類頭、「只遮 W<sub>pre</sub> 的輸出 vs. 遮整體輸出」、正交那兩個重新訓練的實驗。</li>'
      '<li>訓練後放大（Pt）要不要留在論文裡：它在三組都再降 ASR（54.5 → 31.1、17.8 → 12.1、21.1 → 13.1），但不屬於「dropout 為什麼有效」。</li>'
      '<li>文獻先讀哪一類：防禦方法（Obliviate 這類），還是訓練動態的理論（Liang et al. 這類）。</li>'
      '<li>防禦者只用 dev 怎麼挑 p：目前的挑法在 dev 上不穩，要不要把整條 p 曲線本身當主結果。</li></ol>',
      "四題都是 Claude 提的；要不要問、問哪幾題由你決定。第 4 題的背景：用 dev 代價、照「區間含 0 的最大 p」挑，逐座標挑到 0.05、SVD 挑到 0.2，都貼在區間邊界上；dev 只有 872 句，0.2 pp 約 2 句。")

# =====================================================================================================================
# g. appendix
# =====================================================================================================================
slide("apxTrunc", "附錄｜截斷的設計（沿用 11/02 報告）", "截斷只為了讓正交約束有意義：r* 只用 SST-2 dev 挑，每個模組的 k 由 r* 決定",
      '<div class="row">'
      + card("怎麼截", "SVD-LLM 的 truncation-aware data whitening：在秩 ≤ k 的限制下，讓 SST-2 train 上的 Σ‖Wx − W′x‖² 最小。每個模組留最少的方向，使保住的輸出比例 ≥ r。")
      + card("r* 怎麼挑", "r 走 10 格：0.9999 到 0.5。每格比截斷前後的 SST-2 dev AUC（PZ 自己的分類頭、未微調）；單邊、Bonferroni，第一個顯著變差的前一格就是 r*。")
      + card("為什麼要截", f"U、V 取 {WP} 全部方向時，正交懲罰退化成 L2；滿秩 {WP} 下精確正交只允許 ΔW = 0。秩 k 的 {WK} 才讓正交有意義。")
      + "</div>"
      + tbl(["PZ", "s0", "s1", "s2", "s3", "s4"], [["r*", "0.8", "0.995", "0.9", "0.9", "0.8"], ["k（48 模組平均，滿秩 1,024）", "87", "626", "186", "174", "87"]]),
      "11/02 報告（「RoRA 改良 11/2 報告」的 K 章節）和這份一致：同樣的 10 格、同樣的規則（成對、正負分層重抽 100,000 次，每格 0.5%）、同樣的 r* 和 k。11/02 沒有另外定義 K*：K 是「截斷」這個元件的代號，被選的是 r*，k 由 r* 在每個模組各自決定（48 個模組共用 r*）。11/02 還把截斷當成一個防禦元件來看（只截斷本身就降低 persistence），並說明換基底在同樣的 k 下保住的輸出最多（Eckart–Young）；這兩點今天的主線用不到。s0、s4 在選定的 r 上 dev AUC 掉了 0.012、0.013，只是不顯著。r 的 10 格是事先定的格點。資料：logs/wpre/*.json、logs/wpre/choice_bonf_stepwise.tsv；程式 make_wpre.py、run_make_wpre.sh。",
      cls="alt", section=("g", "附錄：截斷的設計、3A 逐 seed、3B 各組大小、統計限制、交換篩選、備用圖、名詞"))

A3 = lambda rows: tbl(["PZ", "完整 s0", "s1", "s2", "中位數", "只換頭 s0", "s1", "s2", "中位數", "只換更新 s0", "s1", "s2", "中位數"], rows)
slide("apx3Acoord", "附錄｜3A 逐 seed（逐座標 0.2）", "τ/g 的配對差（組合 − 普通 LoRA）：只換 LoRA 更新的欄，和完整 dropout 模型幾乎一樣",
      A3([["s0", "−0.64", "+0.07", "−0.64", "−0.64", "+0.16", "+0.12", "+0.19", "+0.16", "−0.68", "+0.03", "−0.67", "−0.67"],
          ["s1", "−0.56", "−0.87", "−0.10", "−0.56", "+0.07", "+0.06", "+0.04", "+0.06", "−0.60", "−0.89", "−0.12", "−0.60"],
          ["s2", "−0.78", "−0.91", "−0.99", "−0.91", "+0.00", "−0.00", "+0.01", "+0.00", "−0.85", "−0.85", "−1.03", "−0.85"],
          ["s3", "−0.11", "−0.58", "+0.00", "−0.11", "+0.10", "+0.23", "+0.15", "+0.15", "−0.13", "−0.67", "−0.05", "−0.13"],
          ["s4", "−0.77", "−1.18", "−1.03", "−1.03", "−0.05", "−0.17", "+0.22", "−0.05", "−0.77", "−1.21", "−1.10", "−1.10"]])
      + '<p class="small">欄的 s0–s2 是 seed；列的 s0–s4 是 PZ。&lt; 0 = τ 相對 g 變小。</p>',
      "資料：logs/mech_eval/summary_3A.csv。g、τ 的逐 seed 倍數在 claude/slides_1119.md 附錄「3A 的逐 seed 倍數」。", cls="alt")

slide("apx3Asvd", "附錄｜3A 逐 seed（SVD 0.2）", "SVD 方向也一樣：τ/g 的下降跟著 LoRA 更新走，不跟著分類頭",
      A3([["s0", "−0.66", "+0.12", "−0.37", "−0.37", "+0.02", "−0.04", "+0.11", "+0.02", "−0.72", "+0.20", "−0.39", "−0.39"],
          ["s1", "−0.51", "−0.81", "−0.19", "−0.51", "+0.10", "+0.11", "−0.06", "+0.10", "−0.63", "−0.80", "−0.17", "−0.63"],
          ["s2", "−0.60", "−0.64", "−0.86", "−0.64", "−0.08", "−0.03", "−0.11", "−0.08", "−0.54", "−0.60", "−0.81", "−0.60"],
          ["s3", "−0.05", "−0.63", "+0.10", "−0.05", "+0.12", "+0.20", "+0.12", "+0.12", "−0.10", "−0.67", "+0.05", "−0.10"],
          ["s4", "−0.67", "−0.92", "−0.82", "−0.82", "+0.00", "−0.09", "+0.01", "+0.00", "−0.69", "−0.92", "−0.81", "−0.81"]])
      + '<p class="small">欄的 s0–s2 是 seed；列的 s0–s4 是 PZ。&lt; 0 = τ 相對 g 變小。</p>',
      "資料：logs/mech_eval/summary_3A.csv。", cls="alt")

slide("apx3B", "附錄｜3B 各組自己的大小", f"各組 {DWX} 相對 {WP} x 的大小：trigger token 和一般字同量級",
      tbl(["組（r<sup>trig</sup> ／ r<sup>tok</sup> ／ 逐格 r<sup>trig</sup>÷r<sup>tok</sup>）", "第 0–5 層", "第 6–11 層", "第 12–17 層", "第 18–23 層"], [
          ["普通 LoRA", "0.049／0.059／0.81", "0.109／0.121／0.90", "0.103／0.165／0.78", "0.166／0.207／0.76"],
          ["逐座標 0.2", "0.073／0.077／0.91", "0.152／0.143／0.99", "0.179／0.219／0.92", "0.107／0.150／0.84"],
          ["SVD 0.2", "0.052／0.061／0.90", "0.109／0.117／0.98", "0.147／0.179／0.86", "0.145／0.179／0.81"]])
      + '<p class="small">每格是 15 個模型 × 12 個模組 = 180 格的中位數；r<sup>trig</sup> = ‖ΔW x‖ ÷ ‖W<sub>pre</sub> x‖ 在 trigger token 上，r<sup>tok</sup> 在乾淨句的一般字上。逐格分散很大：r<sup>trig</sup> 的第 10–90 百分位約 0.02–0.48。</p>',
      "資料：logs/mech_eval/summary_3B.csv；SST-2 dev 444 句正面句插入 mn，trigger 的 token 在 444 句都找得到。r_cls（&lt;s&gt; 上插入 trigger 造成的變化）各組各層 0.017–0.068。「r_trig÷r_tok」是逐格比值的中位數，不是兩欄中位數相除。r_trig 第 10–90 百分位：普通 0.024–0.468、逐座標 0.032–0.464、SVD 0.027–0.482。", cls="alt")

slide("apxStats", "附錄｜統計判讀的限制", "15 對不是 15 個獨立觀察；推論單位是 5 個 PZ，而且只有一個下毒配方",
      '<ul class="txt"><li>同一個 PZ 的 3 個 seed 共享 PZ 權重、資料和評估句；τ/g 配對差的組內相關約 0.30–0.45（粗估），約等於 8–9 個獨立對。</li>'
      '<li>推到同一配方的其他 PZ：5/5 同向的單尾符號檢定 p = 1/32 ≈ 0.031，是能達到的最小值；4/5 是 0.19。推不到其他配方、trigger、模型或 p。</li>'
      '<li>test 句的抽樣誤差：每對 τ/g 配對差的標準差 0.008–0.033（909 句重抽 2,000 次），小於這個量級的差，方向沒有意義。</li>'
      '<li>3D、2A、3A、3B 有十幾個彼此相關的比較，所以只看跨 PZ 的一致性和效果大小，不拿單一 p 值當結論。</li></ul>',
      "同一對的兩個 run 用同一個 seed（依程式碼應共享資料順序和 LoRA 初始化，dropout 組未逐一核對）；扣掉 PZ 平均後，兩者的 τ/g 相關是 −0.32、−0.28，看不到共享的 seed 效應。p 值若要放，要寫明它依賴的假設：各對獨立、虛無假設下每對方向機率 1/2、沒有 tie、方向事先決定、test 句固定。抽樣誤差用 11/19 重算的 0.008–0.033（中位數約 0.02）。", cls="alt")

slide("apxScreen", "附錄｜交換結果的篩選", "原本設計的交換篩選不採用：它篩掉的正是分類頭本身的作用",
      '<ul class="txt"><li>原本的設計：交換組合的乾淨準確率或全部乾淨句的 median |g|，低於兩個來源組 6 個完整模型的最小值就篩掉。</li>'
      '<li>實際：180 個組合中 97 個被標記；只換分類頭的組合 14/15、13/15 被標記，因為 dropout 的分類頭讓 g 和 τ 同比縮小，乾淨準確率卻幾乎不變。</li>'
      '<li>它改變了效果的方向和幅度（例：SVD「分類頭 + 第 12 層以上」被標記的對配對差中位數 +0.003，留下的 −0.475），所以不當分析規則，也不當敏感度分析。</li></ul>',
      "你 11/17 22:33 提出要處理交換後的干擾，Claude 23:04 定義了這個篩選。標記的原因：只因準確率 14、只因 |g| 64、兩者都有 19。這些組合的 |g| 在 15/15、14/15 對裡低於兩個來源，乾淨準確率中位數 −0.06、−0.22 pp。Claude 原本說的 1/7 誤標率假設交換組合和 6 個來源模型可交換；兩個來源組分布不同、組合和來源共享元件，這個假設不成立，資料也否定它（預期約 2 對，實際 14–15 對）。篩選用的是交換後的指標，而 |g| 正是研究對象的一部分，會造成 selection bias。", cls="alt")

slide("apxGrid", "附錄｜備用圖：5 個 PZ × 3 組（seed 0）", "插入 mn 前後的正面句分數：每一格都是普通 LoRA 往左移最多",
      f'<img src="{IMG["grid"]}" alt="SST-2 test 正面句沒插與插入 mn 的分數分布，5 個 PZ × 普通 LoRA、逐座標 0.2、SVD 0.2，seed 0" style="width:1220px;height:758px;object-fit:contain">',
      "圖：logs/margin_decomp/figs/test_hist_trig_seed0.png（seed 1、2 版面相同）。每格的尺度可能不同：跨格請比 AUC、正確率、ASR，不要比 x 軸距離；「拆解 ASR 的下降」一節的六張圖才是同一把尺。圖上的 m、s 就是這裡的 g、τ。圖底的紅字是 repo 繪圖程式的固定註記。每格標題的「所有配對 trigger AUC」比較任意兩句（一句有 trigger、一句沒有），「逐句配對 P(s>0)」看同一句自己掉了多少。",
      cls="alt")

slide("apxTerms", "附錄｜名詞與資料版本", "名詞、符號和資料版本",
      tbl(["名詞", "意思"], [
          ["PZ", "被下毒的預訓練模型（5 個，同一個下毒配方）"],
          [f"{WP}、{WK}", "被凍結的預訓練權重；截斷成秩 k 後的版本"],
          ["逐座標／SVD 方向 dropout", f"只遮 {WP} 的輸出（每個座標，或 {WP} 左奇異方向上的分量），只在訓練時開"],
          ["g、τ、τ/g", "乾淨句的 logit 差；插入 mn 後掉了多少；909 句 τ 的中位數 ÷ g 的中位數"],
          ["最後一輪 ASR、persistence", "第 20 個 epoch 的 ASR；最後 5 個 epoch 的平均；都在 Pt 之前"],
          ["r*、k", "每個 PZ 只用 dev 挑的保住輸出比例；由 r* 決定的秩（每個模組不同）"],
          ["崩掉", "任一個 epoch 的 dev 正確率 ≤ 0.6"]])
      + '<p class="small">資料：repo r13922119/PSIM，commit f1f5910（11/18），只讀；數字的定義與計算在 claude/audit_1119_numbers.md。</p>',
      "數字來源：ASR、乾淨準確率、persistence、Pt：訓練 log 的 SUMMARY 列和逐 epoch 列（ft_logs.py）。下降拆解：logs/margin_decomp/ 的 csv 和 npz。3D、2A、3A、3B：logs/mech_eval/summary_*.csv、checks.csv。線性探測：logs/kprobe_head_20261028_022600.log。更新大小：logs/dw_last.txt。截斷：logs/wpre/。學習率實驗：compare_s5_output.log。下毒：poisoned_pretrain.py、logs/persist_epochs_20261022_163329.log。拆解圖：plot_margin_tutorial_v3.py、plot_toy_tutorial.py（Claude sandbox，不在 repo）。", cls="alt")

css_path = os.path.join(ROOT, "project", "deck.css")
css = open(css_path).read()
if "/* v6 additions */" not in css:
    open(css_path, "w").write(css.rstrip() + "\n" + CSS_ADD)

deck = {"v": 5, "createdOnFiles": {"v": 1, "at": "2026-11-19T05:30:00Z"}, "lists": "css",
        "format": {"v": 1, "shared": ["project/deck.css"]},
        "title": "11/19 報告：凍結權重上的 dropout 為什麼讓後門比較留不住", "cover": "cover", "order": ORDER, "sections": SECTIONS,
        "faces": {"inter": {"family": "Inter", "href": "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"},
                  "noto-sans-tc": {"family": "Noto Sans TC", "href": "https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;600;700&display=swap"}},
        "designSystems": []}
json.dump(deck, open(os.path.join(ROOT, "project", "deck.json"), "w"), ensure_ascii=False, indent=1)
print(len(ORDER), "slides")
