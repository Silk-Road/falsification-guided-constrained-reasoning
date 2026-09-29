#!/usr/bin/env python3
"""L1 预算中性化子样本 v2（定稿口径）：与截断救援分解（论文 tab:decomp）同口径，
直接复用同目录/档案 code/ 下 truncation_decomposition.py 的 load/wilcoxon/stats。

定义：
  rescue / terminating 子集：零样本侧是否全部轮可解析（DS-Pro 单轮）。
  硬界（rate-free）= zero 侧烧满 8192（空尾）案例的最快耗时。
  严格认证（strict）：rescue ∩ agent 全部轮终止且最大耗时 < 硬界。
  典型认证（typical）：rescue ∩ agent 全部轮终止且中位耗时 < 硬界。
输出：L1预算中性化子样本_v2.json（写入结果目录或脚本所在目录）。

布局自适应：公开仓库（<root>/code/eval/）与内部档案（<ARCH>/analysis_*/）均可运行。
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def _resolve():
    repo = os.path.dirname(os.path.dirname(HERE))
    if os.path.isdir(os.path.join(repo, "results", "tcmeval")):
        return repo, os.path.join(repo, "results", "tcmeval"), HERE
    arch = os.path.dirname(HERE)
    if os.path.isdir(os.path.join(arch, "results")):
        return arch, os.path.join(arch, "results"), os.path.join(arch, "code")
    raise SystemExit("无法定位结果目录")

ROOT, RES, CODE_DIR = _resolve()
sys.path.insert(0, CODE_DIR)
from truncation_decomposition import load, wilcoxon, stats  # noqa: E401 同口径复用

OUT = RES if os.path.basename(HERE) == "eval" else HERE

def main():
    out = {}
    for name, zp, ap, runs in [("DS-Flash", "评估_zero_ds_zero", "评估_agent_ds_agent", [1, 2, 3]),
                               ("DS-Pro", "评估_zero_dspro_zero", "评估_agent_dspro_agent", [1])]:
        out[name] = {}
        for task in ["syndrome", "pathogenesis"]:
            zruns = [load(zp, r, task) for r in runs]
            aruns = [load(ap, r, task) for r in runs]
            ids = sorted(set.intersection(*[set(r) for r in zruns + aruns]))
            zm = {i: sum(r[i]["official"] for r in zruns) / len(runs) for i in ids}
            am = {i: sum(r[i]["official"] for r in aruns) / len(runs) for i in ids}
            stable = [i for i in ids if all(r[i]["pred"] for r in zruns)]
            rescue = [i for i in ids if i not in set(stable)]
            hard = min(d["elapsed_s"] for r in zruns for d in r.values()
                       if not d["pred"] and not d.get("response_tail", ""))
            term_all = [i for i in rescue if all(ar[i]["pred"] for ar in aruns)]
            strict = [i for i in term_all if max(ar[i]["elapsed_s"] for ar in aruns) < hard]
            typical = [i for i in term_all
                       if sorted(ar[i]["elapsed_s"] for ar in aruns)[len(runs) // 2] < hard]

            rec = {"hard_bound_s": round(hard, 1),
                   "full": stats(zm, am, ids),
                   "terminating_subset": stats(zm, am, stable),
                   "rescue_subset": stats(zm, am, rescue),
                   "rescue_budget_neutral_strict": stats(zm, am, strict) if strict else None,
                   "rescue_budget_neutral_typical": stats(zm, am, typical) if typical else None,
                   "n_rescue_all_runs_terminated": len(term_all)}
            out[name][task] = rec
            print(f"{name} {task} 硬界={hard:.0f}s")
            for k in ["full", "terminating_subset", "rescue_subset",
                      "rescue_budget_neutral_strict", "rescue_budget_neutral_typical"]:
                v = rec[k]
                if v:
                    print(f"  {k:32s} n={v['n']:3d} zero={v['zero']:.4f} agent={v['agent']:.4f} "
                          f"Δ={v['delta']:+.4f} CI={v['ci95']} p={v['wilcoxon_p']:.3g}")

    path = os.path.join(OUT, "L1预算中性化子样本_v2.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 口径核验：full/terminating/rescue 必须与截断救援分解参考输出一致（若存在）
    ref_path = os.path.join(RES, "截断救援分解.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path, encoding="utf-8"))
        ok = True
        for name in ref:
            for task in ref[name]:
                for subset in ["full", "terminating_subset", "rescue_subset"]:
                    a, b = out[name][task][subset], ref[name][task][subset]
                    if (a["n"], a["zero"], a["agent"], a["delta"]) != (b["n"], b["zero"], b["agent"], b["delta"]):
                        ok = False
                        print(f"口径不一致: {name} {task} {subset}")
        print("\n口径核验（与截断救援分解.json）:", "全部一致 ✓" if ok else "存在不一致 ✗")
    print("已写入", path)

if __name__ == "__main__":
    main()
