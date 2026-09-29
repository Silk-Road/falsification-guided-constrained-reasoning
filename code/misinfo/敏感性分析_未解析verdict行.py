#!/usr/bin/env python3
"""敏感性分析：阶段一未解析 verdict 行对 recall/FPR 的影响（2026-09-23）。

背景：论文误信息节的混淆矩阵（TP=1944, FN=221, FP=9, TN=124，648 行）排除了 81 行：
  - 错位行（premise 与 perturbed_premise 条数不一致，gt 无法定义）：实测 30 行（10 题 × 3 采样）
  - 未解析行（stage-1 verdicts 解析为空）：实测 51 行
  （论文初稿写 28/53，系手工计数/旧版数据，2026-09-23 按当前数据更正为 30/51。）
保守口径：未解析行 = 未标记任何方程 → 其扰动方程全部计 FN，干净方程全部计 TN。
结果：recall 89.8% → 79.2%，FPR 6.8% → 6.4%，与 flag-all 基线（FPR=100%）的对比结论不变。
"""
import json, glob, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
records = [json.loads(l) for l in open(os.path.join(BASE, "data", "test_243_unique.jsonl"), encoding="utf-8")]

def gt_flags(rec):
    p = [x.strip() for x in rec["premise"].split(";") if x.strip()]
    pp = [x.strip() for x in rec["perturbed_premise"].split(";") if x.strip()]
    return None if len(p) != len(pp) else {i + 1: (a != b) for i, (a, b) in enumerate(zip(p, pp))}

rows = []
for fp in glob.glob(os.path.join(BASE, "results_v2", "评估_misinfo_run1_*_protocol_misinformed.jsonl")):
    for l in open(fp, encoding="utf-8"):
        if l.strip():
            rows.append(json.loads(l))

nodet = [r for r in rows if not r.get("detect")]
mis = unp = X_per = X_eq = 0
for r in nodet:
    gt = gt_flags(records[int(r["idx"])])
    if gt is None:
        mis += 1
    elif not r.get("verdicts"):
        unp += 1
        X_per += sum(gt.values())
        X_eq += len(gt)

tp, fn, fpx, tn = 1944, 221, 9, 124
nper, neq = 2165, 2298
print(f"81 行 = 错位 {mis} + 未解析 {unp}")
print(f"未解析行方程：扰动 {X_per} / 共 {X_eq}")
print(f"原口径:      recall={tp/nper:.4f}  FPR={fpx/(neq-nper):.4f}")
print(f"保守敏感性:  recall={tp/(nper+X_per):.4f}  FPR={fpx/(neq-nper+X_eq-X_per):.4f}")

out = {"misaligned_rows": mis, "unparsed_rows": unp,
       "unparsed_perturbed_eqs": X_per, "unparsed_total_eqs": X_eq,
       "recall_orig": round(tp / nper, 4), "recall_conservative": round(tp / (nper + X_per), 4),
       "fpr_orig": round(fpx / (neq - nper), 4),
       "fpr_conservative": round(fpx / (neq - nper + X_eq - X_per), 4)}
dst = os.path.join(BASE, "results_v2", "敏感性分析_未解析verdict行.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("已写入", dst)
