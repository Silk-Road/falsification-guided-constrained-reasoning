# -*- coding: utf-8 -*-
"""
TCMEval-SDT 显著性检验统计分析
输入：results/ 下 评估_{zero|agent}_{backend}_{kind}_run{r}_p{p}_{task}.jsonl
输出：results/显著性检验结果.json + results/显著性检验结果.md

方法：
- 每次重复（run）的官方指标均分 → 3 次重复的 mean ± std
- 显著性：以病例为单位（n=200），取 3 次重复的逐例均分，对配对配置做
  Wilcoxon 符号秩检验 + 配对 bootstrap（10000 次）均值差 95% CI
"""
import glob
import json
import os

import numpy as np
from scipy import stats

DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(DIR, "..", "..", "results", "tcmeval")  # 仓库布局: code/eval/ → results/tcmeval/
if not os.path.isdir(RES):
    RES = os.path.join(DIR, "..", "results")
ARCHIVE = os.path.expanduser("~/TCMEval-SDT评测归档_2026-08-25/results")
TASKS = ["syndrome", "pathogenesis"]
N_RUNS = 3
rng = np.random.default_rng(42)

CONFIGS = {
    "ds_zero": ("zero", "ds"), "ds_agent": ("agent", "ds"),
    "k3_zero": ("zero", "k3"), "k3_agent": ("agent", "k3"),
    "qw32_zero": ("zero", "qw32"), "qw32_agent": ("agent", "qw32"),
    "dspro_zero": ("zero", "dspro"), "dspro_agent": ("agent", "dspro"),
    "qw32t0_agent": ("agent", "qw32t0"),
    "qwen4b_zero": ("zero", "qwen4b"), "qwen4bGvllm_agent": ("agent", "qwen4bGvllm"),
    "qwen4bGvllm_zero": ("zero", "qwen4bGvllm"),
}


def load_run(kind, backend, run, task):
    """某配置某次重复某任务：{病例id: official}（合并 4 个切片）"""
    scores = {}
    pat = os.path.join(RES, f"评估_{kind}_{backend}_{kind}_run{run}*_{task}.jsonl")
    for fp in glob.glob(pat):
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("error"):
                continue
            # 首轮出现优先（与 truncation_decomposition.py 口径一致；重复行可能分数不同）
            scores.setdefault(r["id"], float(r["official"]))
    return scores


def load_archive(prefix, task):
    fp = os.path.join(ARCHIVE, f"{prefix}_{task}.jsonl")
    scores = {}
    if not os.path.exists(fp):
        return scores
    for line in open(fp, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error"):
            continue
        scores[r["id"]] = float(r["official"])
    return scores


def per_case_mean(runs_scores):
    """runs_scores: [ {id: score}, ... ] -> {id: 跨run均分}（取该id有成绩的run平均）"""
    ids = set().union(*[set(d) for d in runs_scores])
    out = {}
    for i in ids:
        vals = [d[i] for d in runs_scores if i in d]
        out[i] = float(np.mean(vals))
    return out


def paired_test(a, b):
    """a,b: {id: score} -> 共同id上的配对检验"""
    ids = sorted(set(a) & set(b))
    x = np.array([a[i] for i in ids])
    y = np.array([b[i] for i in ids])
    d = y - x  # b 相对 a 的差
    n = len(ids)
    mean_d = float(d.mean())
    # Wilcoxon（全部差为0时无法检验）
    try:
        w = stats.wilcoxon(x, y)
        p_w = float(w.pvalue)
    except Exception:
        p_w = float("nan")
    # 配对 bootstrap 95% CI
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        boots.append(d[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "diff": mean_d, "ci95": [float(lo), float(hi)], "wilcoxon_p": p_w}


def main():
    report = {"configs": {}, "comparisons": {}}
    case_means = {}   # config -> task -> {id: score}
    run_means = {}    # config -> task -> [run均分]

    for cfg, (kind, backend) in CONFIGS.items():
        case_means[cfg] = {}
        run_means[cfg] = {}
        for task in TASKS:
            runs_scores = [load_run(kind, backend, r, task) for r in range(1, N_RUNS + 1)]
            runs_scores = [d for d in runs_scores if d]
            if not runs_scores:
                continue
            cm = per_case_mean(runs_scores)
            case_means[cfg][task] = cm
            rm = [float(np.mean(list(d.values()))) for d in runs_scores]
            run_means[cfg][task] = rm
            report["configs"].setdefault(cfg, {})[task] = {
                "n_cases": len(cm),
                "run_means": [round(v, 4) for v in rm],
                "mean": round(float(np.mean(rm)), 4),
                "std": round(float(np.std(rm, ddof=1)) if len(rm) > 1 else 0.0, 4),
            }

    # 归档单次跑参照：4B agent、后训练4B agent
    for name, prefix in [("qwen4b_agent_archived", "评估结果_tCMEval_SDt"),
                         ("pt4b_agent_archived", "评估结果_tCMEval_SDt_后训练4B"),
                         ("ds_zero_archived", "零样本deepseek-v4-flash"),
                         ("ds_agent_archived", "评估结果_tCMEval_SDt_智能体DeepSeek"),
                         ("k3_zero_archived", "零样本k3"),
                         ("k3_agent_archived", "评估结果_tCMEval_SDt_智能体K3")]:
        case_means[name] = {}
        for task in TASKS:
            d = load_archive(prefix, task)
            if d:
                case_means[name][task] = d
                report["configs"].setdefault(name, {})[task] = {
                    "n_cases": len(d), "mean": round(float(np.mean(list(d.values()))), 4)}

    COMPARES = [
        ("ds_zero", "ds_agent", "DeepSeek: zero vs +agent"),
        ("k3_zero", "k3_agent", "K3: zero vs +agent"),
        ("qw32_zero", "qw32_agent", "Qwen3-32B: zero vs +agent"),
        ("qwen4b_agent_archived", "qw32_agent", "4B+agent vs 32B+agent"),
        ("ds_agent", "k3_agent", "DS+agent vs K3+agent"),
        ("qw32_agent", "k3_agent", "32B+agent vs K3+agent"),
        ("qw32_agent", "ds_agent", "32B+agent vs DS+agent"),
        ("dspro_zero", "dspro_agent", "DS-Pro: zero vs +agent"),
        ("ds_agent", "dspro_agent", "DS-flash+agent vs DS-Pro+agent"),
        ("k3_agent", "dspro_agent", "K3+agent vs DS-Pro+agent"),
        ("qw32_zero", "qw32t0_agent", "Qwen3-32B: zero vs +agent (both temp0)"),
        ("qwen4bGvllm_zero", "qwen4bGvllm_agent", "Qwen3-4B: zero vs +agent (budget-matched)"),
        ("qwen4b_zero", "qwen4bGvllm_zero", "Qwen3-4B zero: local-MPS vs vLLM (backend spot check)"),
    ]
    for a, b, label in COMPARES:
        for task in TASKS:
            if a in case_means and task in case_means[a] and b in case_means and task in case_means[b]:
                key = f"{label} [{task}]"
                report["comparisons"][key] = paired_test(case_means[a][task], case_means[b][task])

    out_json = os.path.join(RES, "显著性检验结果.json")
    json.dump(report, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # markdown
    lines = ["# TCMEval-SDT 显著性检验结果", "",
             "## 各配置官方指标（3 次重复，mean ± std）", "",
             "| 配置 | 证型 | 病机 |", "|---|---|---|"]
    for cfg in CONFIGS:
        row = [cfg]
        for task in TASKS:
            v = report["configs"].get(cfg, {}).get(task)
            row.append(f"{v['mean']:.4f} ± {v['std']:.4f}" if v else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 配对显著性检验（逐例均分，Wilcoxon + 配对 bootstrap 95%CI）", "",
              "| 对比 | 任务 | n | A均分 | B均分 | 差值 | 95%CI | p |",
              "|---|---|---|---|---|---|---|---|"]
    for key, v in report["comparisons"].items():
        label, task = key.rsplit(" [", 1)
        task = task.rstrip("]")
        p = v["wilcoxon_p"]
        pstr = f"{p:.2e}" if p == p else "NA"
        lines.append(f"| {label} | {task} | {v['n']} | {v['mean_a']:.4f} | {v['mean_b']:.4f} "
                     f"| {v['diff']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {pstr} |")
    out_md = os.path.join(RES, "显著性检验结果.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("written:", out_json, out_md)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
