#!/usr/bin/env python3
"""两层统计推断之 run 层（论文 tab:sig 的 run-level p 列数据来源）。
逐例配对 Wilcoxon（case 层）为主分析；本脚本给出保守的 run 层检验：
Welch t（3 轮 run 均值为观测单位，回答"效应是否跨测量轮次复现"），
以及 K3 稳定化声明的方差比 F 检验（df 2,2）与 std 比 95% CI。
2026-09-20 组内审稿意见 3：case 层配对低估 run 层方差——两层并列报告。"""
import json, math, os
from scipy.stats import t as tdist, f as fdist

_here = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_here, "..", "..", "results", "tcmeval")  # 仓库布局
if not os.path.isdir(RES):
    RES = os.path.join(_here, "..", "results")               # 内部档案布局
d = json.load(open(os.path.join(RES, "显著性检验结果.json")))["configs"]

def welch(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    t = (mb - ma) / se
    df = (va / len(a) + vb / len(b)) ** 2 / \
         ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    return t, df, 2 * (1 - tdist.cdf(abs(t), df))

out = {"run_level_tests": {}, "variance_ratio_tests": {}}
print("=== run 层 Welch t（3v3 run 均值）===")
for cfg, name in [("ds", "DS-Flash"), ("k3", "Kimi K3"), ("qw32", "Qwen3-32B")]:
    for task in ["syndrome", "pathogenesis"]:
        z = d[f"{cfg}_zero"][task]["run_means"]
        a = d[f"{cfg}_agent"][task]["run_means"]
        t, df, p = welch(z, a)
        key = f"{name} [{task}]"
        out["run_level_tests"][key] = {
            "zero_run_means": z, "agent_run_means": a,
            "delta": round(sum(a) / 3 - sum(z) / 3, 4),
            "welch_t": round(t, 2), "df": round(df, 1), "p": p}
        print(f"{key:26s} Δ={sum(a)/3-sum(z)/3:+.4f}  t({df:.1f})={t:.2f}  p={p:.4f}")

print("\n=== 方差比 F 检验（df 2,2）===")
zcrit = math.sqrt(fdist.ppf(0.975, 2, 2))
for task, zs, as_ in [("syndrome", 0.2177, 0.0068), ("pathogenesis", 0.0432, 0.0110)]:
    F = (zs / as_) ** 2
    p = 2 * min(fdist.cdf(F, 2, 2), 1 - fdist.cdf(F, 2, 2))
    lo, hi = (zs / as_) / zcrit, (zs / as_) * zcrit
    out["variance_ratio_tests"][f"K3 [{task}]"] = {
        "std_ratio": round(zs / as_, 1), "F": round(F, 0),
        "p_two_sided": p, "std_ratio_ci95": [round(lo, 1), round(hi, 1)]}
    print(f"K3 {task}: std比={zs/as_:.1f}  F={F:.0f}  p={p:.4f}  std比95%CI=[{lo:.1f}, {hi:.1f}]")

path = os.path.join(RES, "run层统计检验.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwritten:", path)
