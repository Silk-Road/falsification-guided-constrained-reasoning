#!/usr/bin/env python3
"""格式×内容 2×2 分解（论文 §5.4 "format-only baseline" 段的数据来源）。
四个单元格（4B，证型/病机）：
  zero-shot（无候选格式/无内容）、format-only（格式，无内容）、
  G1（内容，无矩阵+格式块）、full（全协议）。
结论：非可加、强交互——内容束（矩阵+知识+因果链）无格式时显著为负、有格式时显著为正。
2026-09-20 组内审稿意见 4：层归因混淆与 G1 定义矛盾的修复依据。"""
import json, glob, math, os

_here = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_here, "..", "..", "results", "tcmeval")  # 仓库布局
if not os.path.isdir(RES):
    RES = os.path.join(_here, "..", "results")               # 内部档案布局

def load(prefix, run, task):
    rows = {}
    for fp in sorted(glob.glob(os.path.join(RES, f"{prefix}_run{run}_p*_{task}.jsonl"))) + \
             sorted(glob.glob(os.path.join(RES, f"{prefix}_run{run}_{task}.jsonl"))):
        for line in open(fp, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows.setdefault(r["id"], r)
    return rows

def wilcoxon(diffs):
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    ad = sorted(enumerate(d), key=lambda t: abs(t[1]))
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(ad[j+1][1]) - abs(ad[i][1])) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[ad[k][0]] = avg
        i = j + 1
    Wp = sum(r for r, x in zip(ranks, d) if x > 0)
    Wm = sum(r for r, x in zip(ranks, d) if x < 0)
    tie, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(ad[j+1][1]) - abs(ad[i][1])) < 1e-12:
            j += 1
        tie.append(j - i + 1)
        i = j + 1
    var = (n * (n + 1) * (2 * n + 1) - sum(t**3 - t for t in tie) / 2.0) / 24.0
    mu = n * (n + 1) / 4.0
    z = (abs(min(Wp, Wm) - mu) - 0.5) / math.sqrt(var) if var > 0 else 0.0
    return max(min(math.erfc(z / math.sqrt(2.0)), 1.0), 1e-300)

def case_means(prefix, nruns, task):
    cm = {}
    for r in range(1, nruns + 1):
        for cid, row in load(prefix, r, task).items():
            cm.setdefault(cid, []).append(row["official"])
    return {cid: sum(v) / len(v) for cid, v in cm.items()}

def paired(a, b):
    common = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in common]
    n = len(diffs)
    md = sum(diffs) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in diffs) / (n - 1))
    ci = 1.96 * sd / math.sqrt(n)
    return {"n": n, "delta": round(md, 4),
            "ci95": [round(md - ci, 4), round(md + ci, 4)], "wilcoxon_p": wilcoxon(diffs)}

CELLS = {  # (前缀, 轮数)
    "zero":        ("评估_zero_qwen4bGvllm_zero", 3),
    "format_only": ("评估_zero_qwen4b_format", 1),
    "G1_content_no_matrixformat": ("评估_agent_G1", 3),
    "G5_content_format_no_matrix": ("评估_agent_G5nomatrix", 3),
    "full":        ("评估_agent_qwen4bGvllm_agent", 3),
}
out = {}
for task in ["syndrome", "pathogenesis"]:
    cm = {k: case_means(p, nr, task) for k, (p, nr) in CELLS.items()}
    out[task] = {
        "cell_means": {k: round(sum(v.values()) / len(v), 4) for k, v in cm.items()},
        "format_without_content": paired(cm["zero"], cm["format_only"]),
        "content_without_format": paired(cm["zero"], cm["G1_content_no_matrixformat"]),
        "format_with_content":    paired(cm["G1_content_no_matrixformat"], cm["full"]),
        "content_with_format":    paired(cm["format_only"], cm["full"]),
        "matrix_marginal_full_minus_g5": paired(cm["G5_content_format_no_matrix"], cm["full"]),
        "content_with_format_no_matrix": paired(cm["format_only"], cm["G5_content_format_no_matrix"]),
        "format_with_content_no_matrix": paired(cm["G1_content_no_matrixformat"], cm["G5_content_format_no_matrix"]),
    }

path = os.path.join(RES, "格式内容2x2分解.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for task, r in out.items():
    print(f"--- {task} ---  cells: {r['cell_means']}")
    for k in list(r)[1:]:
        v = r[k]
        print(f"  {k:26s} Δ={v['delta']:+.4f} CI={v['ci95']} p={v['wilcoxon_p']:.3g}")
print("written:", path)
