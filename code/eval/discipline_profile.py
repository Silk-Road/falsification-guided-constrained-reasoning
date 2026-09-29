#!/usr/bin/env python3
"""Output-Discipline Profile（论文 Table "tab:discipline" 与 fig:discipline 的数据来源）。
从各模型零样本 jsonl 日志计算四项无协议指标：
  answer-extraction rate（可解析率）、over-prediction ratio（|pred|/|gold|）、
  run-to-run std（三轮配置）、cross-run self-consistency（三轮 pred 集合完全一致比例）。
对应 2026-09-20 审稿意见"核心自变量 output discipline 未定义未测量"的修复。"""
import json, glob, math, os
from collections import defaultdict

_here = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_here, "..", "..", "results", "tcmeval")  # 仓库布局
if not os.path.isdir(RES):
    RES = os.path.join(_here, "..", "results")               # 内部档案布局
CONFIGS = {
    "评估_zero_ds_zero":    ("DS-V4-Flash", 3),
    "评估_zero_dspro_zero": ("DS-V4-Pro", 1),
    "评估_zero_k3_zero":    ("Kimi K3", 3),
    "评估_zero_qw32_zero":  ("Qwen3-32B", 3),
    "评估_zero_qwen4bGvllm_zero": ("Qwen3-4B", 3),
}
TASKS = ["syndrome", "pathogenesis"]

def load(prefix, run, task):
    rows = {}
    for fp in sorted(glob.glob(os.path.join(RES, f"{prefix}_run{run}_p*_{task}.jsonl"))) + \
             sorted(glob.glob(os.path.join(RES, f"{prefix}_run{run}_{task}.jsonl"))):
        for line in open(fp, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows.setdefault(r["id"], r)  # 首次出现优先
    return rows

out = {}
for prefix, (name, nruns) in CONFIGS.items():
    out[name] = {}
    for task in TASKS:
        runs = [load(prefix, r, task) for r in range(1, nruns + 1)]
        runs = [r for r in runs if r]
        if not runs:
            continue
        base = runs[0]
        ids = sorted(base)
        rec = {
            "n": len(ids),
            "answer_rate": round(sum(1 for i in ids if base[i]["pred"]) / len(ids), 4),
            "over_pred_ratio": round(
                sum(len(base[i]["pred"]) / max(len(base[i]["gold"]), 1) for i in ids) / len(ids), 3),
        }
        if len(runs) == 3:
            means = [sum(r[i]["official"] for i in ids) / len(ids) for r in runs]
            m = sum(means) / 3
            rec["run_means"] = [round(x, 4) for x in means]
            rec["run_std"] = round(math.sqrt(sum((x - m) ** 2 for x in means) / 2), 4)
            common = [i for i in ids if all(i in r for r in runs)]
            rec["self_consistency"] = round(
                sum(1 for i in common
                    if set(runs[0][i]["pred"]) == set(runs[1][i]["pred"]) == set(runs[2][i]["pred"]))
                / len(common), 4)
        out[name][task] = rec

path = os.path.join(RES, "纪律画像_output_discipline_profile.json")
json.dump(out, open(path, "w", encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
