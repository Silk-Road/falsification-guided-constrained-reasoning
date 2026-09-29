# -*- coding: utf-8 -*-
"""误信息实验统计分析：Accuracy / K-Acc / 检出率 / 配对显著性"""
import glob
import json
import os

import numpy as np
from scipy import stats

DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(DIR, "..", "results")
GROUPS = ["base_original", "base_misinformed", "inst_misinformed", "protocol_misinformed"]
rng = np.random.default_rng(42)


def load(group):
    """{idx: [sample_corrects]}，另返回 protocol 的 detect 列表"""
    per = {}
    detects = []
    for fp in glob.glob(os.path.join(RES, f"评估_misinfo_run1_*_{group}.jsonl")):
        for l in open(fp, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("error"):
                continue
            per.setdefault(r["idx"], []).append(bool(r["correct"]))
            if group == "protocol_misinformed" and r.get("detect"):
                detects.append(r["detect"])
    return per, detects


def per_q_rate(per):
    return {i: float(np.mean(v)) for i, v in per.items()}


def k_acc(per_q, cond_correct_idx):
    sel = [v for i, v in per_q.items() if i in cond_correct_idx]
    return float(np.mean(sel)) if sel else float("nan"), len(sel)


def paired(a, b):
    ids = sorted(set(a) & set(b))
    x = np.array([a[i] for i in ids]); y = np.array([b[i] for i in ids])
    d = y - x
    try:
        p = stats.wilcoxon(x, y).pvalue
    except Exception:
        p = float("nan")
    boots = [d[rng.integers(0, len(ids), len(ids))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(ids), "a": float(x.mean()), "b": float(y.mean()),
            "diff": float(d.mean()), "ci": [float(lo), float(hi)], "p": float(p)}


def main():
    data = {}
    for g in GROUPS:
        per, det = load(g)
        data[g] = (per_q_rate(per), det)

    base_orig_q = data["base_original"][0]
    cond_idx = {i for i, v in base_orig_q.items() if v > 0.5}  # 多数采样答对视为"能解"

    lines = ["# 误信息实验结果（DeepSeek-v4-flash，243题×3采样）", "",
             "| 组 | Accuracy | K-Acc（以base_original能解的题为条件, n=%d） |" % len(cond_idx),
             "|---|---|---|"]
    kaccs = {}
    for g in GROUPS:
        q = data[g][0]
        acc = float(np.mean(list(q.values())))
        kacc, nk = k_acc(q, cond_idx)
        kaccs[g] = (acc, kacc)
        lines.append(f"| {g} | {acc:.4f} | {kacc:.4f} |")

    det = data["protocol_misinformed"][1]
    if det:
        tp = sum(d["tp"] for d in det); fp = sum(d["fp"] for d in det); fn = sum(d["fn"] for d in det)
        nper = sum(d["n_perturbed"] for d in det)
        neq = sum(d["n_eq"] for d in det)
        lines += ["", f"## 协议组阶段一检出（n={len(det)}题）",
                  f"- 误信息检出率（recall）: {tp}/{nper} = {tp/max(nper,1):.4f}",
                  f"- 误报率: {fp}/{neq-nper} = {fp/max(neq-nper,1):.4f}",
                  f"- 精确率: {tp}/{max(tp+fp,1)} = {tp/max(tp+fp,1):.4f}"]

    lines += ["", "## 配对显著性（逐题配对，Wilcoxon + bootstrap 95%CI）", "",
              "| 对比 | n | A | B | Δ | 95%CI | p |", "|---|---|---|---|---|---|---|"]
    comps = [("base_misinformed", "base_original", "误信息的伤害"),
             ("base_misinformed", "inst_misinformed", "指令干预"),
             ("base_misinformed", "protocol_misinformed", "协议干预 vs 默认"),
             ("inst_misinformed", "protocol_misinformed", "协议干预 vs 指令干预"),
             ("base_original", "protocol_misinformed", "协议干预 vs 上限")]
    for a, b, label in comps:
        r = paired(data[a][0], data[b][0])
        lines.append(f"| {label} | {r['n']} | {r['a']:.4f} | {r['b']:.4f} | {r['diff']:+.4f} "
                     f"| [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] | {r['p']:.2e} |")

    # K-Acc 口径配对检验（条件题集）
    kid = {i for i in cond_idx if all(i in data[g][0] for g in ["base_misinformed", "inst_misinformed", "protocol_misinformed", "base_original"])}
    klines = ["", f"## K-Acc 口径配对显著性（条件题集 n={len(kid)}）", ""]
    for a, b, label in [("base_misinformed", "base_original", "误信息伤害"),
                        ("base_misinformed", "protocol_misinformed", "协议 vs 默认"),
                        ("inst_misinformed", "protocol_misinformed", "协议 vs 指令"),
                        ("base_original", "protocol_misinformed", "协议 vs 上限")]:
        r = paired({i: data[a][0][i] for i in kid}, {i: data[b][0][i] for i in kid})
        klines.append(f"| {label} | {r['n']} | {r['a']:.4f} | {r['b']:.4f} | {r['diff']:+.4f} | [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] | {r['p']:.2e} |")
    lines += klines
    out = "\n".join(lines) + "\n"
    open(os.path.join(RES, "误信息实验结果.md"), "w", encoding="utf-8").write(out)
    json.dump({g: {"accuracy": kaccs[g][0], "k_acc": kaccs[g][1]} for g in GROUPS},
              open(os.path.join(RES, "误信息实验结果.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(out)


if __name__ == "__main__":
    main()
