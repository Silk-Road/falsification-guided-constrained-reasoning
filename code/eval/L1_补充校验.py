#!/usr/bin/env python3
"""L1 补充校验：代理自洽性、C1 /no_think 细分、C1 吞吐率锚点、agent 耗时分位。

1) 代理自洽：zero 终止案例理论上 <8192 token，耗时若普遍超过 8192/吞吐率 则代理失效；
   个别超时案例源于 API 重试/排队（elapsed 含重试等待），方向为高估 token、对结论保守。
2) C1（2026-08，agent@16384 + /no_think）未解析案例细分：<60s 格式失败 / >=120s 烧满预算。
3) C1 zero@8k 空尾锚点（v4-flash 原版端点，181 tok/s）。
4) C2 flash agent 终止案例耗时分位与 <67s 硬界占比。

布局自适应：公开仓库与内部档案均可运行（见 L1_预算代理分析.py 头部说明）。
"""
import json, glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

def _resolve():
    repo = os.path.dirname(os.path.dirname(HERE))
    if os.path.isdir(os.path.join(repo, "results", "tcmeval")):
        return repo, os.path.join(repo, "results", "tcmeval")
    arch = os.path.dirname(HERE)
    if os.path.isdir(os.path.join(arch, "results")):
        return arch, os.path.join(arch, "results")
    raise SystemExit("无法定位结果目录")

ROOT, RES = _resolve()

def find(name):
    for p in [os.path.join(RES, name),
              os.path.join(ROOT, "logs", "tcmeval", name),
              os.path.join(os.path.dirname(ROOT), "tcm-agent-v4", name)]:
        if os.path.exists(p):
            return p
    return None

print("== 1) 代理自洽性校验（zero 终止案例耗时上限；78s≈8192@105tok/s）==")
for task in ["syndrome", "pathogenesis"]:
    mx, over, n = 0, 0, 0
    for f in glob.glob(os.path.join(RES, f"评估_zero_ds_zero_run*_p*_{task}.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if d["pred"]:
                n += 1
                mx = max(mx, d["elapsed_s"])
                if d["elapsed_s"] > 78:
                    over += 1
    print(f"  {task}: zero 终止 n={n} max={mx}s，耗时>78s 的有 {over} 例（应为个别 API 重试/排队）")

print("\n== 2) C1 agent@16k /no_think 未解析案例细分 ==")
c1log = find("评估_tCMEval_智能体DeepSeek.log")
if c1log:
    ets = []
    for line in open(c1log, encoding="utf-8"):
        m = re.match(r"\[\d+/\d+\] (病例\d+) gold=\S+ pred=(\S+) .*?\((\d+)s\)", line)
        if m and m.group(2) == "-":
            ets.append(int(m.group(3)))
    fast = [e for e in ets if e < 60]
    mid = [e for e in ets if 60 <= e < 120]
    slow = [e for e in ets if e >= 120]
    print(f"  未解析 n={len(ets)}: <60s 格式失败 n={len(fast)}; 60-120s n={len(mid)}; "
          f">=120s 烧满16k（退化循环）n={len(slow)}")
else:
    print("  未找到 C1 日志")

print("\n== 3) C1 zero@8k 锚点（2026-08，v4-flash 原版端点）==")
for task in ["syndrome", "pathogenesis"]:
    f = find(f"零样本deepseek-v4-flash_{task}.jsonl")
    if f:
        burn, term = [], []
        for l in open(f):
            d = json.loads(l)
            (term if d["pred"] else burn if not d.get("response_tail", "") else []).append(d["elapsed_s"])
        if burn:
            burn.sort(); term.sort()
            print(f"  {task}: 烧满8k n={len(burn)} min={burn[0]}s p50={burn[len(burn)//2]}s"
                  f"（≈{8192/burn[len(burn)//2]:.0f} tok/s）; 终止 n={len(term)} p50={term[len(term)//2]}s max={term[-1]}s")

print("\n== 4) C2 flash agent 终止案例耗时分位 ==")
for task in ["syndrome", "pathogenesis"]:
    ets = []
    for f in glob.glob(os.path.join(RES, f"评估_agent_ds_agent_run*_p*_{task}.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if d["pred"]:
                ets.append(d["elapsed_s"])
    ets.sort()
    n = len(ets)
    q = lambda p: ets[int(n * p)]
    print(f"  {task}: n={n} p25={q(.25)} p50={q(.5)} p75={q(.75)} p90={q(.9)} p95={q(.95)} max={ets[-1]}")
    print(f"    <67s 硬界占比={sum(1 for e in ets if e < 67) / n:.1%}")
