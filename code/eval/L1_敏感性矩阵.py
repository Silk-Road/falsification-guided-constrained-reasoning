#!/usr/bin/env python3
"""L1 敏感性矩阵：附录 F "full sensitivity matrix" 的数据来源。
对 rescue 子集的预算认证，遍历 阈值(硬界/78s标定) × 聚合(max/median) × 终止要求(全部轮/任意轮)，
输出全部 16 种组合的 n/zero/agent/Δ/CI/p。口径复用 code/truncation_decomposition.py。
内部档案版（绝对路径）；公开仓库版路径经适配（相对于脚本位置）。
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(HERE)                      # 内部分析目录的上一级 = 档案根
if not os.path.isdir(os.path.join(ARCH, "results")):   # 公开仓库布局: code/eval/
    ARCH = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ARCH, "results")
if os.path.isdir(os.path.join(RES, "tcmeval")):        # 公开仓库 results/tcmeval/
    RES = os.path.join(RES, "tcmeval")
sys.path.insert(0, os.path.join(ARCH, "code"))
sys.path.insert(0, os.path.join(ARCH, "code", "eval"))
import glob
import math

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
    if n == 0:
        return None
    md = sum(diffs) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in diffs) / (n - 1)) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n)
    return {"n": n, "zero": round(sum(zm[i] for i in sel) / n, 4),
            "agent": round(sum(am[i] for i in sel) / n, 4),
            "delta": round(md, 4), "ci95": [round(md - ci, 4), round(md + ci, 4)],
            "wilcoxon_p": wilcoxon(diffs)}

out = []
for name, zp, ap, runs in [("DS-Flash", "评估_zero_ds_zero", "评估_agent_ds_agent", [1, 2, 3]),
                           ("DS-Pro", "评估_zero_dspro_zero", "评估_agent_dspro_agent", [1])]:
    for task in ["syndrome", "pathogenesis"]:
        zruns = [load(zp, r, task) for r in runs]
        aruns = [load(ap, r, task) for r in runs]
        ids = sorted(set.intersection(*[set(r) for r in zruns + aruns]))
        zm = {i: sum(r[i]["official"] for r in zruns) / len(runs) for i in ids}
        am = {i: sum(r[i]["official"] for r in aruns) / len(runs) for i in ids}
        stable = {i for i in ids if all(r[i]["pred"] for r in zruns)}
        rescue = [i for i in ids if i not in stable]
        hard = min(d["elapsed_s"] for r in zruns for d in r.values()
                   if not d["pred"] and not d.get("response_tail", ""))
        for thr_name, thr in [("rate_free_hard", hard), ("calibrated_105tok_s", 8192 / 105)]:
            for agg in ["max", "median"]:
                for req in ["all_runs_terminated", "any_run_terminated"]:
                    sel = []
                    for i in rescue:
                        ets = sorted(ar[i]["elapsed_s"] for ar in aruns)
                        v = ets[-1] if agg == "max" else ets[len(ets) // 2]
                        ok_term = all(ar[i]["pred"] for ar in aruns) if req == "all_runs_terminated" \
                            else any(ar[i]["pred"] for ar in aruns)
                        if v < thr and ok_term:
                            sel.append(i)
                    s = stats(zm, am, sel)
                    row = {"model": name, "task": task, "threshold": thr_name,
                           "threshold_s": round(thr, 1), "aggregation": agg,
                           "termination_requirement": req}
                    row.update(s) if s else row.update({"n": 0})
                    out.append(row)
                    print(f"{name:9s} {task:12s} {thr_name:18s}({thr:5.1f}s) {agg:6s} {req:22s} "
                          + (f"n={s['n']:3d} Δ={s['delta']:+.4f} CI={s['ci95']} p={s['wilcoxon_p']:.3g}" if s else "n=0"))

OUT = RES if os.path.basename(HERE) == "eval" else HERE
dst = os.path.join(OUT, "L1敏感性矩阵.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n已写入", dst)
