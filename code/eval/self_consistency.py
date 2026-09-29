#!/usr/bin/env python3
"""self-consistency 基线（多数投票）分析。
对 3 轮采样零样本运行按病例多数投票（标签出现于 ≥2/3 轮即入选），用官方口径评分。
K3：既有 3 轮（temp=1.0 强制采样）直接可用；4B：批次4 temp=0.7 采样 3 轮；
flash/32B 的既有 3 轮为贪心，多数投票退化，仅供参考值。"""
import json, glob, math, os
from collections import defaultdict

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

def official(pred, gold):
    pred, gold = set(pred), set(gold)
    if not gold:
        return 0.0
    return len(pred & gold) / (len(gold) + len(pred - gold))

def sc_score(prefix, nruns, task):
    runs = [load(prefix, r, task) for r in range(1, nruns + 1)]
    ids = sorted(set.intersection(*[set(r) for r in runs]))
    scores = []
    for i in ids:
        votes = defaultdict(int)
        for r in runs:
            for x in r[i]["pred"]:
                votes[x] += 1
        pred = {x for x, v in votes.items() if v >= 2}
        scores.append(official(pred, runs[0][i]["gold"]))
    return sum(scores) / len(scores), len(scores)

CONFIGS = [
    ("K3 (temp1.0 既有3轮)", "评估_zero_k3_zero", 3),
    ("4B (temp0.7 批次4)", "评估_zero_qwen4bGvllm_sc", 3),
    ("flash (贪心3轮,退化参考)", "评估_zero_ds_zero", 3),
    ("32B (贪心3轮,退化参考)", "评估_zero_qw32_zero", 3),
]
out = {}
for name, prefix, nruns in CONFIGS:
    for task in ["syndrome", "pathogenesis"]:
        try:
            s, n = sc_score(prefix, nruns, task)
            out.setdefault(name, {})[task] = {"sc_majority": round(s, 4), "n": n}
            print(f"{name:26s} {task:12s} SC={s:.4f} (n={n})")
        except Exception as e:
            print(f"{name:26s} {task:12s} 缺数据: {e}")

path = os.path.join(RES, "self_consistency基线.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written:", path)
