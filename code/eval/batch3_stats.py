#!/usr/bin/env python3
"""批次3结果统一分析（zero run3 病机补齐 + G5nomatrix 缺失单元格后）：
1) 4B 零样本 vLLM 3×3：mean±std、可解析率、过选率、自一致、run-SD
2) tab:sig 4B：case 层 Wilcoxon + run 层 Welch（两侧均 3 轮，首次可算）
3) G5nomatrix（内容+格式、无矩阵）：2×2 的矩阵边际贡献 = full − G5
结果写入 results/批次3统计分析.json"""
import json, glob, math, os

_here = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_here, "..", "..", "results", "tcmeval")  # 仓库布局
if not os.path.isdir(RES):
    RES = os.path.join(_here, "..", "results")               # 内部档案布局

def load(prefix, run, task):
    rows = {}
    for fp in sorted(glob.glob(os.path.join(RES, f"{prefix}_run{run}_p*_{task}.jsonl"))) + \
             sorted(glob.glob(os.path.join(RES, f"{prefix}_run{run}_{task}.jsonl"))):
        for line in open(fp, encoding='utf-8'):
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

def welch(a, b):
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    va = sum((x-ma)**2 for x in a)/(len(a)-1)
    vb = sum((x-mb)**2 for x in b)/(len(b)-1)
    se = math.sqrt(va/len(a) + vb/len(b))
    t = (mb-ma)/se
    df = (va/len(a)+vb/len(b))**2 / ((va/len(a))**2/(len(a)-1) + (vb/len(b))**2/(len(b)-1))
    try:
        from scipy.stats import t as tdist
        p = 2*(1-tdist.cdf(abs(t), df))
    except Exception:
        p = None
    return t, df, p

def case_means(prefix, nruns, task):
    cm = {}
    for r in range(1, nruns+1):
        for cid, row in load(prefix, r, task).items():
            cm.setdefault(cid, []).append(row["official"])
    return {c: sum(v)/len(v) for c, v in cm.items()}

def run_means(prefix, nruns, task):
    return [sum(x["official"] for x in load(prefix, r, task).values())
            / max(len(load(prefix, r, task)), 1) for r in range(1, nruns+1)]

def paired(a, b):
    common = sorted(set(a) & set(b))
    diffs = [b[i]-a[i] for i in common]
    n = len(diffs); md = sum(diffs)/n
    sd = math.sqrt(sum((x-md)**2 for x in diffs)/(n-1))
    ci = 1.96*sd/math.sqrt(n)
    return {"n": n, "delta": round(md, 4), "ci95": [round(md-ci, 4), round(md+ci, 4)],
            "wilcoxon_p": wilcoxon(diffs)}

def sd(v):
    m = sum(v)/len(v)
    return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))

out = {"zero_4b_vllm": {}, "sig_4b": {}, "g5_2x2": {}}

print("=== 1) 4B 零样本 vLLM 3×3 ===")
for task in ["syndrome", "pathogenesis"]:
    allr = [load("评估_zero_qwen4bGvllm_zero", r, task) for r in (1, 2, 3)]
    rms = [sum(x["official"] for x in d.values())/len(d) for d in allr]
    ids = sorted(set.intersection(*[set(d) for d in allr]))
    parse = sum(1 for i in ids if allr[0][i]["pred"])/len(ids)*100
    ratio = sum(len(allr[0][i]["pred"])/max(len(allr[0][i]["gold"]), 1) for i in ids)/len(ids)
    sc = sum(1 for i in ids if len({tuple(sorted(allr[r][i]["pred"])) for r in range(3)}) == 1)/len(ids)*100
    out["zero_4b_vllm"][task] = {"run_means": [round(x, 4) for x in rms],
                                 "mean": round(sum(rms)/3, 4), "std": round(sd(rms), 4),
                                 "answer_rate": round(parse/100, 4),
                                 "over_pred": round(ratio, 2), "self_cons": round(sc/100, 4)}
    print(f"{task:12s} {rms} mean={sum(rms)/3:.4f}±{sd(rms):.4f} parse={parse:.1f}% overpred={ratio:.2f} selfcons={sc:.1f}%")

print("\n=== 2) tab:sig 4B（zero vLLM 3轮 vs agent 3轮）===")
for task in ["syndrome", "pathogenesis"]:
    z = case_means("评估_zero_qwen4bGvllm_zero", 3, task)
    a = case_means("评估_agent_qwen4bGvllm_agent", 3, task)
    res = paired(z, a)
    zr = [sum(x["official"] for x in load("评估_zero_qwen4bGvllm_zero", r, task).values())/len(load("评估_zero_qwen4bGvllm_zero", r, task)) for r in (1, 2, 3)]
    ar = [sum(x["official"] for x in load("评估_agent_qwen4bGvllm_agent", r, task).values())/len(load("评估_agent_qwen4bGvllm_agent", r, task)) for r in (1, 2, 3)]
    t, df, p = welch(zr, ar)
    res["run_level"] = {"t": round(t, 2), "df": round(df, 1), "p": p}
    out["sig_4b"][task] = res
    print(f"{task:12s} Δ={res['delta']:+.4f} CI={res['ci95']} case_p={res['wilcoxon_p']:.3g} run_p={p if p is None else format(p, '.3g')}")

print("\n=== 3) G5nomatrix（内容+格式、无矩阵）与 2×2 ===")
for task in ["syndrome", "pathogenesis"]:
    g5 = case_means("评估_agent_G5nomatrix", 3, task)
    full = case_means("评估_agent_qwen4bGvllm_agent", 3, task)
    zero = case_means("评估_zero_qwen4bGvllm_zero", 3, task)
    fmt = case_means("评估_zero_qwen4b_format", 1, task)
    g5r = [sum(x["official"] for x in load("评估_agent_G5nomatrix", r, task).values())/len(load("评估_agent_G5nomatrix", r, task)) for r in (1, 2, 3)]
    cell = {"g5_runs": [round(x, 4) for x in g5r], "g5_mean": round(sum(g5r)/3, 4), "g5_std": round(sd(g5r), 4),
            "matrix_marginal_full_minus_g5": paired(g5, full),
            "g5_vs_zero": paired(zero, g5),
            "g5_vs_g1_content_only": None}
    g1 = case_means("评估_agent_G1", 3, task)
    cell["g5_vs_g1_format_effect_no_matrix"] = paired(g1, g5)
    cell["g5_vs_format_only_content_effect_no_matrix"] = paired(fmt, g5)
    out["g5_2x2"][task] = cell
    mm = cell["matrix_marginal_full_minus_g5"]
    print(f"{task:12s} G5={cell['g5_mean']}±{cell['g5_std']}  矩阵边际(full−G5)={mm['delta']:+.4f} CI={mm['ci95']} p={mm['wilcoxon_p']:.3g}")

path = os.path.join(RES, "批次3统计分析.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwritten:", path)
