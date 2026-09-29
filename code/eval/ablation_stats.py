#!/usr/bin/env python3
"""Table 4 消融统计：G0(完整协议) vs G1-G4 各消融组，3 轮重复。
口径与 显著性检验结果.json 一致：每案例取 3 轮 official 均分 → 200 个配对值，
Wilcoxon 符号秩检验 + 差值 95% CI（正态近似）。"""
import json, glob, math, os, sys
from collections import defaultdict

_here = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_here, "..", "..", "results", "tcmeval")  # 仓库布局
if not os.path.isdir(RES):
    RES = os.path.join(_here, "..", "results")               # 内部档案布局
GROUPS = {
    "G0_full":      "评估_agent_qwen4bGvllm_agent",
    "G1_no_matrix": "评估_agent_G1",
    "G2_no_evidence": "评估_agent_G2",
    "G3_no_diff_guide": "评估_agent_G3",
    "G4_no_skill_noop":  "评估_agent_G4",     # 空操作（flag bug，作废留存）
    "G4fix_no_skill": "评估_agent_G4fix",     # 修复 ablation_no_skill_knowledge 后的真实 G4
}
TASKS = ["syndrome", "pathogenesis"]

def load_run(prefix, run, task):
    """加载某组某轮某任务全部分片，按 id 去重；返回 {id: official} 与诊断信息。"""
    samples, dup = defaultdict(list), 0
    files = sorted(glob.glob(os.path.join(RES, f"{prefix}_run{run}_p*_{task}.jsonl")))
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cid, off = r["id"], r["official"]
            hit = 1.0 if r.get("hit") else 0.0
            if samples[cid]:
                dup += 1
            samples[cid].append((off, hit))
    # 同轮内重复评测（分片重叠/断点续跑）取平均——同配置下的重复测量
    scores = {cid: sum(v[0] for v in vv) / len(vv) for cid, vv in samples.items()}
    hits = {cid: sum(v[1] for v in vv) / len(vv) for cid, vv in samples.items()}
    return scores, hits, len(files), dup

def wilcoxon(diffs):
    """Wilcoxon 符号秩检验（正态近似，含连续性校正与并列秩校正）。"""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    ad = sorted(enumerate(d), key=lambda t: abs(t[1]))
    ranks = [0.0] * n
    i = 0
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
    W = min(Wp, Wm)
    # 并列校正
    tie_counts, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(ad[j+1][1]) - abs(ad[i][1])) < 1e-12:
            j += 1
        tie_counts.append(j - i + 1)
        i = j + 1
    tie_adj = sum(t**3 - t for t in tie_counts)
    var = (n * (n + 1) * (2 * n + 1) - tie_adj / 2.0) / 24.0
    mu = n * (n + 1) / 4.0
    z = (abs(W - mu) - 0.5) / math.sqrt(var) if var > 0 else 0.0
    p = 2.0 * math.erfc(z / math.sqrt(2.0)) / 2.0
    return max(min(p, 1.0), 1e-300)

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

out = {"per_run": {}, "diagnostics": {}, "comparisons": {}}
case_means = {g: {t: {} for t in TASKS} for g in GROUPS}
case_hits = {g: {t: {} for t in TASKS} for g in GROUPS}

for g, prefix in GROUPS.items():
    out["per_run"][g] = {}
    for t in TASKS:
        run_means, per_case, per_case_hit = [], defaultdict(list), defaultdict(list)
        ndup_total = 0
        for r in (1, 2, 3):
            sc, ht, nf, nd = load_run(prefix, r, t)
            ndup_total += nd
            run_means.append(sum(sc.values()) / len(sc))
            for cid, v in sc.items():
                per_case[cid].append(v)
            for cid, v in ht.items():
                per_case_hit[cid].append(v)
            out["diagnostics"].setdefault(g, {}).setdefault(t, {})[f"run{r}"] = \
                {"files": nf, "unique_ids": len(sc), "dup_rows": nd}
        # 每案例 3 轮均分
        cm = {cid: sum(v) / len(v) for cid, v in per_case.items() if len(v) == 3}
        ch = {cid: sum(v) / len(v) for cid, v in per_case_hit.items() if len(v) == 3}
        dropped = [cid for cid, v in per_case.items() if len(v) != 3]
        if dropped:
            print(f"  !! {g} {t}: {len(dropped)} 案例不足3轮，剔除: {dropped[:5]}", file=sys.stderr)
        case_means[g][t] = cm
        case_hits[g][t] = ch
        m = sum(run_means) / 3
        sd = math.sqrt(sum((x - m) ** 2 for x in run_means) / 2)
        out["per_run"][g][t] = {"run_means": [round(x, 4) for x in run_means],
                                 "mean": round(m, 4), "std": round(sd, 4),
                                 "n_cases": len(cm),
                                 "hit_rate": round(100 * sum(ch.values()) / len(ch), 1)}

for t in TASKS:
    ids0 = set(case_means["G0_full"][t])
    for g in list(GROUPS)[1:]:
        A, B = case_means["G0_full"][t], case_means[g][t]
        common = sorted(ids0 & set(B))
        if len(common) != len(ids0):
            print(f"  !! {g} {t}: 与G0共同案例 {len(common)}/{len(ids0)}", file=sys.stderr)
        diffs = [B[c] - A[c] for c in common]
        n = len(diffs)
        md = sum(diffs) / n
        sd = math.sqrt(sum((x - md) ** 2 for x in diffs) / (n - 1))
        ci = 1.96 * sd / math.sqrt(n)
        p = wilcoxon(diffs)
        hA = 100 * sum(case_hits["G0_full"][t][c] for c in common) / n
        hB = 100 * sum(case_hits[g][t][c] for c in common) / n
        out["comparisons"][f"{g} vs G0 [{t}]"] = {
            "n": n, "mean_G0": round(sum(A[c] for c in common) / n, 4),
            "mean_Gx": round(sum(B[c] for c in common) / n, 4),
            "diff": round(md, 4), "ci95": [round(md - ci, 4), round(md + ci, 4)],
            "wilcoxon_p": p, "hit_G0": round(hA, 1), "hit_Gx": round(hB, 1),
            "hit_diff_pp": round(hB - hA, 1)}

js = json.dumps(out, ensure_ascii=False, indent=1)
open(os.path.join(RES, "消融统计检验结果.json"), "w", encoding="utf-8").write(js)
print(js)
