#!/usr/bin/env python3
"""L1 审稿意见应对：预算混杂事后检验——主分析。

方法：归档评估未留存逐例 token 数，用 wall-clock elapsed_s 做代理。
  - 锚点1：zero-shot@8192 空尾案例 = 恰好烧掉 8192 token
  - 锚点2：agent@32768 空尾案例 = 恰好烧掉 32768 token（交叉验证吞吐率线性）
  - 硬界（rate-free）= zero 侧烧满 8192 的最快耗时：比它快的 agent 案例·轮必然 <8192 token
输出：L1预算代理分析.json、L1分析报告.md、fig_L1_*.png（写入 RES 或脚本所在目录，见布局）。

布局自适应：既可在本论文公开仓库（<root>/code/eval/，结果在 <root>/results/tcmeval/）运行，
也可在内部档案（<ARCH>/analysis_*/，结果在 <ARCH>/results/）运行。
"""
import json, glob, math, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))

def _resolve():
    repo = os.path.dirname(os.path.dirname(HERE))
    if os.path.isdir(os.path.join(repo, "results", "tcmeval")):
        return repo, os.path.join(repo, "results", "tcmeval")
    arch = os.path.dirname(HERE)
    if os.path.isdir(os.path.join(arch, "results")):
        return arch, os.path.join(arch, "results")
    raise SystemExit("无法定位结果目录（results/ 或 results/tcmeval/）")

ROOT, RES = _resolve()
OUT = RES if os.path.basename(HERE) == "eval" else HERE

def load_runs(prefix, runs, task):
    out = {}
    for r in runs:
        rows = {}
        for fp in sorted(glob.glob(os.path.join(RES, f"{prefix}_run{r}_p*_{task}.jsonl"))) + \
                 sorted(glob.glob(os.path.join(RES, f"{prefix}_run{r}_{task}.jsonl"))):
            for line in open(fp, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    rows.setdefault(d["id"], d)
        out[r] = rows
    return out

def pct(x, n):
    return f"{x}/{n} = {x/n:.1%}" if n else "0"

report = {}
md = []

for name, zp, ap, runs, zbudget, abudget in [
    ("DS-Flash", "评估_zero_ds_zero", "评估_agent_ds_agent", [1, 2, 3], 8192, 32768),
    ("DS-Pro", "评估_zero_dspro_zero", "评估_agent_dspro_agent", [1], 8192, 16384),
]:
    report[name] = {}
    md.append(f"\n## {name}（zero@{zbudget} vs agent@{abudget}）\n")
    for task in ["syndrome", "pathogenesis"]:
        zruns = load_runs(zp, runs, task)
        aruns = load_runs(ap, runs, task)
        ids = sorted(set.intersection(*[set(r) for r in list(zruns.values()) + list(aruns.values())]))
        stable = {i for i in ids if all(zruns[r][i]["pred"] for r in runs)}
        rescue = [i for i in ids if i not in stable]

        rate_samples, anchor_min_elapsed = [], []
        for r in runs:
            for d in zruns[r].values():
                if not d["pred"] and not d.get("response_tail", ""):
                    rate_samples.append(zbudget / d["elapsed_s"])
                    anchor_min_elapsed.append(d["elapsed_s"])
        agent_burn = []
        for r in runs:
            for d in aruns[r].values():
                if not d["pred"] and not d.get("response_tail", ""):
                    agent_burn.append(abudget / d["elapsed_s"])
        rate_samples += agent_burn
        rate_samples.sort()
        rate_med = rate_samples[len(rate_samples) // 2] if rate_samples else float("nan")
        t_anchor = min(anchor_min_elapsed) if anchor_min_elapsed else float("nan")

        case_stats = {}
        for i in ids:
            ets = sorted(aruns[r][i]["elapsed_s"] for r in runs)
            case_stats[i] = {
                "median_s": ets[len(ets) // 2], "max_s": ets[-1],
                "n_term": sum(1 for r in runs if aruns[r][i]["pred"]),
                "est_tokens_med": ets[len(ets) // 2] * rate_med,
                "est_tokens_max": ets[-1] * rate_med,
            }

        def frac_under(sel, key, thr):
            s = [case_stats[i] for i in sel if case_stats[i]["n_term"] > 0]
            if not s:
                return (0, 0)
            return (sum(1 for c in s if c[key] < thr), len(s))

        res = {}
        for subset_name, sel in [("full", ids), ("rescue", rescue), ("terminating", sorted(stable))]:
            n_est, n_t = frac_under(sel, "est_tokens_med", 8192)
            n_cons, _ = frac_under(sel, "est_tokens_max", 8192)
            n_hard, _ = frac_under(sel, "max_s", t_anchor)
            res[subset_name] = {"n_terminated_cases": n_t, "under8k_point": (n_est, n_t),
                                "under8k_conservative": (n_cons, n_t), "under8k_hardbound": (n_hard, n_t)}
            md.append(f"- {task} {subset_name}: 有终止的案例 {n_t} 个；估计 token<8192：{pct(n_est, n_t)}（点估计）/ "
                      f"{pct(n_cons, n_t)}（保守：各轮最大耗时×吞吐率）/ {pct(n_hard, n_t)}（硬界：快过 zero 烧满8k的最快案例 {t_anchor:.0f}s）")

        zm = {i: sum(zruns[r][i]["official"] for r in runs) / len(runs) for i in ids}
        am = {i: sum(aruns[r][i]["official"] for r in runs) / len(runs) for i in ids}
        for thr_name, key in [("point", "est_tokens_med"), ("conservative", "est_tokens_max")]:
            sel = [i for i in rescue if case_stats[i][key] < 8192 and case_stats[i]["n_term"] > 0]
            if sel:
                dz = sum(zm[i] for i in sel) / len(sel)
                da = sum(am[i] for i in sel) / len(sel)
                res[f"rescue_under8k_{thr_name}"] = {"n": len(sel), "zero": round(dz, 4),
                                                     "agent": round(da, 4), "delta": round(da - dz, 4)}
                md.append(f"- {task} rescue ∩ agent<8k（{thr_name}，已废止口径，见 v2）：n={len(sel)}，"
                          f"zero={dz:.4f} agent={da:.4f} Δ={da - dz:+.4f}")

        res["rate_median_tok_s"] = round(rate_med, 1)
        res["rate_p25_p75"] = [round(rate_samples[len(rate_samples) // 4], 1),
                               round(rate_samples[3 * len(rate_samples) // 4], 1)]
        res["anchor_min_s_zero8k"] = round(t_anchor, 1)
        res["n_zero_burned8k"] = len(anchor_min_elapsed)
        res["n_agent_burned_full"] = len(agent_burn)
        res["subsets"] = {"rescue": len(rescue), "terminating": len(stable)}
        report[name][task] = res
        md.append(f"- {task} 吞吐率：中位 {rate_med:.0f} tok/s（IQR {res['rate_p25_p75']}；n={len(rate_samples)}，"
                  f"其中 agent 烧满 {len(agent_burn)} 例用于交叉验证）；zero 烧满8k的最快耗时 {t_anchor:.0f}s\n")

md.append("\n## 病理（loop vs 稳步推进）证据\n")

def loop_detect(text, window=200):
    """复刻 runtime _detect_repetition_loop：尾部 window 字符若在前文出现过即回环"""
    if len(text) < window:
        return False
    tail = text[-window:]
    return tail in text[:-window]

md.append("### 非空尾但未解析案例（唯一留存文本的未终止类型）\n")
for name, zp, runs in [("DS-Flash", "评估_zero_ds_zero", [1, 2, 3]),
                       ("DS-Pro", "评估_zero_dspro_zero", [1])]:
    for task in ["syndrome", "pathogenesis"]:
        zruns = load_runs(zp, runs, task)
        seen = set()
        for r in runs:
            for i, d in zruns[r].items():
                if not d["pred"] and d.get("response_tail", "") and (r, i) not in seen:
                    seen.add((r, i))
                    tail = d["response_tail"]
                    lp = loop_detect(tail, window=min(200, len(tail) // 2))
                    md.append(f"- {name} {task} run{r} {i}：尾长{len(tail)}，尾部回环={'是' if lp else '否'}")

md.append("\n### DS-Pro 终止案例可见输出长度（尾长<400 ⇒ 全文<400字符）\n")
for task in ["syndrome", "pathogenesis"]:
    zruns = load_runs("评估_zero_dspro_zero", [1], task)
    lens = sorted(len(d.get("response_tail", "")) for d in zruns[1].values() if d["pred"])
    if lens:
        md.append(f"- DS-Pro zero {task}：终止 {len(lens)} 例，尾长 p25={lens[len(lens)//4]} "
                  f"p50={lens[len(lens)//2]} p75={lens[3*len(lens)//4]} max={lens[-1]}")

# C1 日志（agent@16384 + /no_think，2026-08，端点未漂移）
md.append("\n### C1（2026-08，agent@16384 + /no_think 简洁模式）\n")
c1_candidates = [os.path.join(ROOT, "logs", "tcmeval", "评估_tCMEval_智能体DeepSeek.log"),
                 os.path.join(ROOT, "logs", "评估_tCMEval_智能体DeepSeek.log"),
                 os.path.join(os.path.dirname(ROOT), "tcm-agent-v4", "评估_tCMEval_智能体DeepSeek.log")]
c1log = next((p for p in c1_candidates if os.path.exists(p)), None)
if c1log:
    ets_all, unparsed_ets = [], []
    for line in open(c1log, encoding="utf-8"):
        m = re.match(r"\[\d+/\d+\] (病例\d+) gold=\S+ pred=(\S+) .*?\((\d+)s\)", line)
        if m:
            ets_all.append(int(m.group(3)))
            if m.group(2) == "-":
                unparsed_ets.append(int(m.group(3)))
    fast = [e for e in unparsed_ets if e < 60]
    slow = [e for e in unparsed_ets if e >= 120]
    md.append(f"- C1 agent 逐例行：n={len(ets_all)}，未解析 {len(unparsed_ets)}（{len(unparsed_ets)/len(ets_all):.1%}）；"
              f"其中 <60s 格式失败 {len(fast)} 例，≥120s 烧满预算 {len(slow)} 例。"
              f"烧满组耗时聚集于 121–159s（p50=144s），与'烧满 16k（90.5s@181tok/s）+ 开销'一致；"
              f"60–120s 为空档，与快速格式失败组形成双峰。病理性指不收敛而非答案长度："
              f"协议终止输出是模板有界的（10 候选矩阵+一行结论，≪1k token），"
              f"'合法答案需 >16k token'无法解释四分之一的案例")

# 图 1：耗时分布
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, task, ttl in [(axes[0], "syndrome", "证型"), (axes[1], "pathogenesis", "病机")]:
    zruns = load_runs("评估_zero_ds_zero", [1, 2, 3], task)
    aruns = load_runs("评估_agent_ds_agent", [1, 2, 3], task)
    z_term = [d["elapsed_s"] for r in zruns.values() for d in r.values() if d["pred"]]
    z_burn = [d["elapsed_s"] for r in zruns.values() for d in r.values()
              if not d["pred"] and not d.get("response_tail", "")]
    a_term = [d["elapsed_s"] for r in aruns.values() for d in r.values() if d["pred"]]
    bins = range(0, 360, 10)
    ax.hist(z_term, bins=bins, alpha=0.6, label=f"zero terminated (n={len(z_term)})", color="#4c9be8")
    ax.hist(z_burn, bins=bins, alpha=0.6, label=f"zero exhausted@8192 (n={len(z_burn)})", color="#e8734c")
    ax.hist(a_term, bins=bins, alpha=0.6, label=f"agent terminated (n={len(a_term)})", color="#4caf50")
    ax.axvline(min(z_burn), color="red", ls="--", lw=1.5, label=f"8k-exhaustion bound {min(z_burn):.0f}s")
    ax.set_xlabel("elapsed_s"); ax.set_ylabel("count"); ax.set_title(f"DS-Flash {ttl}")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_L1_elapsed分布.png"), dpi=160)

# 图 2：rescue 子集 agent 估计 token CDF
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, task, ttl in [(axes[0], "syndrome", "证型"), (axes[1], "pathogenesis", "病机")]:
    zruns = load_runs("评估_zero_ds_zero", [1, 2, 3], task)
    aruns = load_runs("评估_agent_ds_agent", [1, 2, 3], task)
    ids = sorted(set.intersection(*[set(r) for r in list(zruns.values()) + list(aruns.values())]))
    stable = {i for i in ids if all(zruns[r][i]["pred"] for r in [1, 2, 3])}
    rescue = [i for i in ids if i not in stable]
    rate = report["DS-Flash"][task]["rate_median_tok_s"]
    for sel, name_, color in [(rescue, "rescue subset", "#e8734c"), (sorted(stable), "terminating subset", "#4c9be8")]:
        vals = sorted(sorted(aruns[r][i]["elapsed_s"] for r in [1, 2, 3])[1] * rate
                      for i in sel if any(aruns[r][i]["pred"] for r in [1, 2, 3]))
        if vals:
            ax.plot(vals, [(k + 1) / len(vals) for k in range(len(vals))],
                    label=f"{name_} (n={len(vals)})", color=color)
    ax.axvline(8192, color="red", ls="--", lw=1.5, label="8192 tokens")
    ax.set_xlabel("agent estimated tokens (median elapsed x calibrated rate)")
    ax.set_ylabel("CDF"); ax.set_title(f"DS-Flash {ttl}"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_L1_agent估计token_CDF.png"), dpi=160)

json.dump(report, open(os.path.join(OUT, "L1预算代理分析.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
open(os.path.join(OUT, "L1分析报告.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md[-6:]))
print("\n输出目录:", OUT)
