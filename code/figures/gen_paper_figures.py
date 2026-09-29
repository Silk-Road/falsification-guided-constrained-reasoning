#!/usr/bin/env python3
"""Generate English figures for the arXiv paper (pdflatex-compatible, vector PDF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

BLUE = "#4a7ebb"
DARKBLUE = "#2c5f9e"
GRAY = "#bfbfbf"
LIGHT = "#f5f5f5"
RED = "#c0392b"


def box(ax, x, y, w, h, text, fc="white", ec="black", fs=8, tc="black", lw=1.0):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)
    return b


def arrow(ax, x1, y1, x2, y2, text=None, fs=7, color="black", rad=0.0, tx_off=(0, 0), lw=1.2, ms=11):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                        connectionstyle=f"arc3,rad={rad}", color=color, lw=lw)
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + tx_off[0], (y1 + y2) / 2 + tx_off[1], text,
                ha="center", va="center", fontsize=fs, color=color)


# ---------------------------------------------------------------- fig 1: architecture
# 严格复刻原始文档架构图（中医大模型思考架构），仅中译英
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.3, 4.7))
    ax.set_xlim(0, 240); ax.set_ylim(0, 158); ax.axis("off")

    def stage(x, y, w, h, text, fs=8):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="white", ec="#333333", lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def harrow(x1, x2, y):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                     mutation_scale=8, color="#333333", lw=1.0,
                                     shrinkA=1, shrinkB=1))

    # ---- 四层色带（左端竖排层名）----
    bands = [
        ("Epistemic\nlayer", 126, 26, "#bcd2ee"),   # 哲学层
        ("TCM process\nlayer", 56, 66, "#f6d97f"),  # 业务层
        ("Model\nlayer", 34, 18, "#f0ab63"),        # 模型层
        ("Data\nlayer", 12, 18, "#aacd7e"),         # 数据层
    ]
    for name, y, h, fc in bands:
        ax.add_patch(plt.Rectangle((14, y), 224, h, fc=fc, ec="#666666", lw=0.8))
        ax.text(7.5, y + h / 2, name, ha="center", va="center", fontsize=8,
                fontweight="bold", rotation=90, color="#333333")

    xs, w = [26, 70, 114, 158, 202], 34     # 五盒行（间距 10，箭头可见）

    # ---- 哲学层 ----
    epi = ["Information", "Hypothesis\n(abduction)", "Verification\n(evidence)",
           "Causal analysis\n(deduction)", "Action"]
    for x, t in zip(xs, epi):
        stage(x, 130.5, w, 16, t, fs=6.5)
    for i in range(4):
        harrow(xs[i] + w, xs[i + 1], 138.5)
    ax.text(234, 128.5, "convergence target: yin\u2013yang balance", ha="right",
            va="center", fontsize=6.5, color="#333333")

    # ---- 业务层 ----
    top = ["Four\nexaminations", "Signs", "Etiology &\npathogenesis\nreasoning",
           "Candidate\nsyndromes\n(hypotheses)", "TCM logic\nsystem\n(five-phase)"]
    for x, t in zip(xs, top):
        stage(x, 99.5, w, 18, t, fs=6)
    for i in range(4):
        harrow(xs[i] + w, xs[i + 1], 108.5)

    bot = ["Treatment\nprinciple\n(guiding rationale)", "Draft\ntreatment plan",
           "KB retrieval\nverification\n(self-check)", "Treatment plan\n& feedback",
           "Validation\nfeedback\n(closed loop)"]
    for x, t in zip(xs, bot):
        stage(x, 68.5, w, 19, t, fs=6)
    for i in range(4):
        harrow(xs[i] + w, xs[i + 1], 78.5)

    # 五行自洽 -> 治则：肘形线（下-左-下）
    x5, x1 = xs[4] + w / 2, xs[0] + w / 2
    ax.plot([x5, x5, x1, x1], [99.5, 93, 93, 89], color="#333333", lw=1.0)
    ax.plot([x1], [88.7], marker="v", markersize=4.5, color="#333333")

    # 业务层两条注记
    ax.text(88, 62, "Verification (other-evidence): targeted questioning\nover typical sign\u2013syndrome data",
            ha="center", va="center", fontsize=6, color="#5a4a1a")
    ax.text(186, 62, "Causal analysis: deduction along the\nfive-phase self-consistent system",
            ha="center", va="center", fontsize=6, color="#5a4a1a")

    # ---- 模型层（LLM-1、LLM-2、ML、RAG 顺序与原图一致）----
    mx, mw = [26, 80, 134, 188], 48
    mods = ["LLM-1\ndifferentiation\nreasoning", "LLM-2\nplan generation",
            "ML-assisted\nclassification of\nstructured signs", "RAG\nverbatim retrieval"]
    for x, t in zip(mx, mods):
        stage(x, 35.5, mw, 15, t, fs=6)

    # ---- 数据层 ----
    dat = ["syndrome\u2013sign\nmapping data", "CoT reasoning\ndata",
           "syndrome\u2013formula\u2013\ncase mapping data", "safety &\nincompatibility rules"]
    for x, t in zip(mx, dat):
        stage(x, 13.5, mw, 15, t, fs=6)

    fig.savefig(os.path.join(OUT, "fig_architecture.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    ax.set_xlim(0, 200); ax.set_ylim(-4, 168); ax.axis("off")
    cx, w = 100, 88   # center boxes: x 56..144

    box(ax, cx - w / 2, 152, w, 10, "S1  Intent triage\n(rule tree + 2-token model)", fs=8)
    box(ax, 4, 126, 48, 14, "S2\u2192S3 hit:\nverbatim DB answer\n(zero hallucination)", fs=7, fc=LIGHT)
    box(ax, 148, 126, 48, 14, "S2\u2192S3 miss:\nconstrained short answer\n(\u2264256 tokens)", fs=7, fc=LIGHT)
    arrow(ax, 72, 152, 42, 141)
    arrow(ax, 128, 152, 158, 141)
    ax.text(36, 147, "knowledge query, hit", fontsize=6.5, ha="center")
    ax.text(164, 147, "knowledge query, miss", fontsize=6.5, ha="center")

    box(ax, cx - w / 2, 126, w, 14, "S4  Collection & sufficiency scoring\n(tongue / chief symptoms / pulse-course)", fs=8)
    arrow(ax, cx, 152, cx, 140, "consultation", fs=6.5, tx_off=(12, 0))
    box(ax, 4, 82, 48, 14, "targeted follow-up\nquestions (missing\ndimensions)", fs=7, fc=LIGHT)
    arrow(ax, 57, 126, 46, 96, "insufficient", fs=6.5, color=RED, rad=0.25, tx_off=(-11.5, 8))
    arrow(ax, 44, 96, 55, 126, None, color="gray", rad=-0.25)
    ax.text(30, 108, "deadlock guard /\nstop signal \u2192\nforced advance", fontsize=6, color="gray", ha="center")

    box(ax, cx - w / 2, 106, w, 14, "S5  Hypotheses: candidate syndrome space\n+ evidence checklist (pos/neg rules)", fs=8)
    arrow(ax, cx, 126, cx, 120, "sufficiency \u2265 threshold", fs=6.5, tx_off=(21, 0))

    box(ax, cx - w / 2, 84, w, 16, "S6  Verification: candidate \u00d7 evidence\nmatrix falsification (\u2265 half falsified)\nsingle-line [Final Syndrome] output", fs=7.5)
    arrow(ax, cx, 106, cx, 100)

    box(ax, cx - w / 2, 62, w, 16, "S7  Causal analysis: forced chain\nfour-exam \u2192 eight principles \u2192 nature\n\u2192 location \u2192 synthesis (no single-sign jump)", fs=7.5)
    arrow(ax, cx, 84, cx, 78)
    # insufficient loop: S6 right -> back to S5 right
    ax.plot([144, 156, 156], [90, 90, 112], color="black", lw=1.0)
    ax.add_patch(FancyArrowPatch((156, 112), (144, 112), arrowstyle="-|>", mutation_scale=11, color="black", shrinkA=0, shrinkB=0))
    ax.text(159, 101, "insufficient:\nre-examine evidence", fontsize=6.5, ha="left", color=RED)

    box(ax, cx - w / 2, 38, w, 18, "S8  Action: principle \u2192 plan generation\n(KB-original formulas/herbs/doses only)\n+ prescription safety checks", fs=7.5)
    arrow(ax, cx, 62, cx, 56)
    # violation loop: S8 left -> back into S8
    ax.plot([56, 48, 48], [42, 42, 50], color=RED, lw=1.0)
    ax.add_patch(FancyArrowPatch((48, 50), (56, 50), arrowstyle="-|>", mutation_scale=11, color=RED, shrinkA=0, shrinkB=0))
    ax.text(46, 46, "violation: intercept\n& regenerate", fontsize=6.5, ha="right", color=RED)

    box(ax, cx - w / 2, 18, w, 14, "S9  Feedback: consistency cache / drift\ndetection / quality audit / snapshots", fs=7.5)
    arrow(ax, cx, 38, cx, 32)

    box(ax, cx - w / 2, 0, w, 12, "Output: structured syndrome result\n+ treatment plan + safety flags", fs=8, fc="#e8f0fa")
    arrow(ax, cx, 18, cx, 12, ms=9)

    fig.savefig(os.path.join(OUT, "fig_pipeline.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3: matrix protocol
def fig_matrix():
    fig = plt.figure(figsize=(7.3, 4.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-4, 240); ax.set_ylim(0, 106); ax.axis("off")

    box(ax, -3, 80, 52, 20, "Candidate set\n(user markers /\nskill routing / KB\nsyndrome retrieval)", fs=7, fc=LIGHT)
    box(ax, -3, 55, 52, 20, "Patient four-exam evidence\nchecklist (pos/neg rules,\nlongest-match, negation\nfirst)", fs=7, fc=LIGHT)
    box(ax, -3, 30, 52, 20, "Authoritative diagnostic\nstandards (chief/acc.\nsymptoms, tongue, pulse;\nalias mapping)", fs=7, fc=LIGHT)
    # three input arrows point into the evidence matrix (left edge x=56)
    arrow(ax, 49, 90, 56, 82, lw=1.1)   # candidate set -> Syndrome 1 row
    arrow(ax, 49, 65, 56, 74, lw=1.1)   # four-exam checklist -> Syndrome 2 row
    arrow(ax, 49, 40, 56, 63, lw=1.1)   # diagnostic standards -> matrix bottom-left

    cols = ["Candidate", "greasy\ncoating", "slippery\npulse", "insomnia", "fatigue", "verdict"]
    rows = [
        ["Syndrome 1", "\u2713", "\u2713", "\u2713", "\u25cb", "compatible (3 \u2713)"],
        ["Syndrome 2", "\u2717", "\u2717", "\u2713", "\u2717", "incompatible"],
        ["Syndrome 3", "\u2717", "\u2717", "\u2717", "\u2713", "incompatible"],
    ]
    x0, y0, rh = 56, 62, 8
    cw = [30, 16, 16, 16, 16, 30]
    xs = [x0]
    for wdt in cw:
        xs.append(xs[-1] + wdt)
    ax.text(135, y0 + 4 * rh + 2.5,
            "candidate \u00d7 four-exam evidence matrix (mandatory output)",
            ha="center", fontsize=9.5, fontweight="bold")
    for r in range(4):
        for c in range(6):
            y = y0 + (3 - r) * rh
            fc = "#e8e8e8" if r == 0 else "white"
            ax.add_patch(plt.Rectangle((xs[c], y), cw[c], rh, fc=fc, ec="black", lw=0.8))
            txt = cols[c] if r == 0 else rows[r - 1][c]
            ax.text(xs[c] + cw[c] / 2, y + rh / 2, txt, ha="center", va="center", fontsize=7.5)
    ax.text(118, y0 - 4.5,
            "\u2713 = support    \u2717 = oppose    \u25cb = irrelevant;",
            ha="center", fontsize=7.5, color="gray")
    ax.text(118, y0 - 9.5,
            "row rule: supports > opposes and key evidence supported \u2192 compatible",
            ha="center", fontsize=7.5, color="gray")

    box(ax, 140, 30, 74, 15, "Sufficiency check:\nincompatible \u2265 half of candidates?", fs=9)
    arrow(ax, 170, 62, 177, 45, lw=1.1)
    box(ax, 124, 6, 106, 15, "Emit the most-supported compatible candidate:\n[Final Syndrome]: phlegm-heat harassing heart", fs=9, fc="#e8f0fa")
    arrow(ax, 177, 30, 177, 21, "yes", fs=8, tx_off=(6, 0))
    # no: loop back to matrix
    ax.plot([140, 60], [36, 36], color=RED, lw=1.0)
    ax.add_patch(FancyArrowPatch((60, 36), (60, 61), arrowstyle="-|>", mutation_scale=11, color=RED, shrinkA=0, shrinkB=0))
    ax.text(97, 28.5, "no: insufficient differentiation,\nre-examine four-exam evidence",
            fontsize=7.5, color=RED, ha="center")

    ax.text(120, 0.5, "prohibitions: no default pick by candidate order \u00b7 no copying example conclusions \u00b7 no fabricated formulas / herbs / doses",
            ha="center", fontsize=7.5, color=RED)

    fig.savefig(os.path.join(OUT, "fig_matrix.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4: results bar chart
# 数据: (zero, agent)；None 表示评测进行中。3轮重复的配置附 std。
RESULTS = {
    "Qwen3-4B":  {"syn": (0.481, 0.365), "pat": (0.455, 0.335)},
    "Qwen3-32B": {"syn": (0.551, 0.464), "pat": (0.523, 0.411)},
    "DS-flash":  {"syn": (0.289, 0.556), "pat": (0.212, 0.482)},
    "DS-Pro":    {"syn": (0.443, 0.537), "pat": (0.298, 0.508)},
    "K3":        {"syn": (0.496, 0.576), "pat": (0.556, 0.524)},
}
STD = {  # 3轮重复的 std，其余单次
    ("Qwen3-32B", "syn"): (0.007, 0.011), ("Qwen3-32B", "pat"): (0.006, 0.009),
    ("DS-flash", "syn"): (0.015, 0.061), ("DS-flash", "pat"): (0.012, 0.129),
    ("K3", "syn"): (0.218, 0.007), ("K3", "pat"): (0.043, 0.011),
    ("Qwen3-4B", "syn"): (0.005, 0.013), ("Qwen3-4B", "pat"): (0.010, 0.010),
}

def fig_results():
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.8))
    models = list(RESULTS)
    for ax, key, base, title in ((axes[0], "syn", 0.475, "Syndrome task"),
                                 (axes[1], "pat", 0.532, "Pathogenesis task")):
        xt, xlabels = [], []
        for gi, m in enumerate(models):
            z, a = RESULTS[m][key]
            zs, as_ = STD.get((m, key), (None, None))
            xc = gi * 1.0
            if z is not None:
                ax.bar(xc - 0.19, z, width=0.34, color=GRAY, edgecolor="black", lw=0.7,
                       yerr=zs, capsize=2, error_kw=dict(lw=0.8))
                ax.text(xc - 0.19, z + 0.014, f"{z:.3f}", ha="center", fontsize=6.2)
            else:
                ax.bar(xc - 0.19, 0.02, width=0.34, color="white", edgecolor="gray",
                       lw=0.7, hatch="//")
                ax.text(xc - 0.19, 0.05, "running", ha="center", fontsize=5.5, color="gray", rotation=90)
            ax.bar(xc + 0.19, a, width=0.34, color=BLUE if m != "K3" else DARKBLUE,
                   edgecolor="black", lw=0.7, yerr=as_, capsize=2, error_kw=dict(lw=0.8))
            ax.text(xc + 0.19, a + 0.014, f"{a:.3f}", ha="center", fontsize=6.2)
            xt.append(xc); xlabels.append(m)
        ax.axhline(base, ls="--", color=RED, lw=1.1)
        ax.text(0.01, base - 0.018, f"paper best baseline \u2248 {base}", fontsize=6.5, color=RED,
                transform=ax.get_yaxis_transform(), va="top")
        ax.set_xticks(xt); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylim(0, 0.70); ax.set_title(title, fontsize=9)
        ax.set_ylabel("score (official benchmark script)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    # 图例
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=GRAY, edgecolor="black", label="zero-shot"),
               mpatches.Patch(color=BLUE, edgecolor="black", label="+ agent")]
    axes[0].legend(handles=handles, fontsize=6.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_results.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- fig 5: discipline profile
def fig_discipline():
    # 数据来自 tCMEval-SDt显著性检验_2026-09-06 的零样本日志（answer rate = 可抽取答案比例）
    #            answer-rate syn/pat (%)        Δ syn/pat (agent − zero)
    data = {
        "DS-Flash": ((38.5, 33.0), (0.267, 0.271), "#c0392b"),
        "DS-Pro":   ((62.5, 51.0), (0.094, 0.210), "#e67e22"),
        "K3":       ((99.5, 97.0), (0.080, -0.032), "#2c5f9e"),
        "32B":      ((100.0, 100.0), (-0.087, -0.112), "#4a7ebb"),
        "4B":       ((100.0, 99.5), (-0.114, -0.118), "#7f8c8d"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9))
    offsets = {  # (syn偏移, pat偏移)
        "DS-Flash": ((0, 0.024), (0, 0.024)),
        "DS-Pro":   ((0, -0.032), (0, -0.034)),
        "K3":       ((-1.5, 0.026), (-5.0, -0.004)),
        "32B":      ((-0.8, 0.026), (-3.0, 0.024)),
        "4B":       ((-0.8, -0.036), (-1.2, -0.036)),
    }
    for ax, ti, title in ((axes[0], 0, "Syndrome task"), (axes[1], 1, "Pathogenesis task")):
        for m, (ar, d, c) in data.items():
            ax.scatter(ar[ti], d[ti], s=42, color=c, edgecolor="black", lw=0.6, zorder=3)
            dx, dy = offsets[m][ti]
            ax.annotate(m, (ar[ti] + dx, d[ti] + dy), ha="center", fontsize=7.5)
        ax.axhline(0, ls="--", color="gray", lw=0.9)
        ax.set_xlabel("zero-shot answer-extraction rate (%)", fontsize=8)
        ax.set_ylabel(r"$\Delta$ score (agent $-$ zero)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(25, 112); ax.set_ylim(-0.17, 0.31); ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(0.03, 0.05, "Spearman $\\rho=-0.975$ (n=5, qualitative)",
                 transform=axes[0].transAxes, fontsize=7, color="gray")
    axes[1].text(0.03, 0.05, "Spearman $\\rho=-0.900$ (n=5, qualitative)",
                 transform=axes[1].transAxes, fontsize=7, color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_discipline.pdf"), bbox_inches="tight")
    plt.close(fig)


fig_architecture()
fig_pipeline()
fig_matrix()
fig_results()
fig_discipline()
print("done:", sorted(os.listdir(OUT)))
