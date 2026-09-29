#!/usr/bin/env python3
"""截断救援分解（论文 tab:decomp 的数据来源）。
全集增益 = 截断救援（零样本无法稳定产出答案的病例）+ 稳定产出子集上的净效应。
稳定产出子集定义：零样本在全部重复轮次均可解析（DS-Pro 为单轮，即该轮可解析）。
2026-09-20：取代旧"≥1轮可解析"口径（该口径仍含截断罚分，+0.163/+0.123 作废）。"""
import json, glob, math, os

_here = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_here, "..", "..", "results", "tcmeval")  # 仓库布局: code/eval/ → results/tcmeval/
if not os.path.isdir(RES):
    RES = os.path.join(_here, "..", "results")               # 内部档案布局: code/ → results/

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

def stats(zm, am, sel):
    diffs = [am[i] - zm[i] for i in sel]
    n = len(diffs)
    md = sum(diffs) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in diffs) / (n - 1)) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n)
    return {"n": n,
            "zero": round(sum(zm[i] for i in sel) / n, 4),
            "agent": round(sum(am[i] for i in sel) / n, 4),
            "delta": round(md, 4), "ci95": [round(md - ci, 4), round(md + ci, 4)],
            "wilcoxon_p": wilcoxon(diffs)}

out = {}
for name, zp, ap, nruns in [("DS-Flash", "评估_zero_ds_zero", "评估_agent_ds_agent", 3),
                            ("DS-Pro", "评估_zero_dspro_zero", "评估_agent_dspro_agent", 1)]:
    out[name] = {}
    for task in ["syndrome", "pathogenesis"]:
        zruns = [load(zp, r, task) for r in range(1, nruns + 1)]
        aruns = [load(ap, r, task) for r in range(1, nruns + 1)]
        ids = sorted(set.intersection(*[set(r) for r in zruns + aruns]))
        zm = {i: sum(r[i]["official"] for r in zruns) / nruns for i in ids}
        am = {i: sum(r[i]["official"] for r in aruns) / nruns for i in ids}
        stable = [i for i in ids if all(r[i]["pred"] for r in zruns)]
        rescue = [i for i in ids if i not in set(stable)]
        out[name][task] = {"full": stats(zm, am, ids),
                           "terminating_subset": stats(zm, am, stable),
                           "rescue_subset": stats(zm, am, rescue)}

path = os.path.join(RES, "截断救援分解.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for name, vv in out.items():
    for task, r in vv.items():
        for subset, s in r.items():
            print(f"{name:9s} {task:12s} {subset:19s} n={s['n']:3d} zero={s['zero']:.4f} "
                  f"agent={s['agent']:.4f} Δ={s['delta']:+.4f} CI={s['ci95']} p={s['wilcoxon_p']:.3g}")
