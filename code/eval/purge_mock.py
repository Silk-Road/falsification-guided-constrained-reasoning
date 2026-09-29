# -*- coding: utf-8 -*-
"""清除结果文件中的 mock 垃圾行（403 后适配器回退 mock 写入的行）与 403 错误行。
用法: python3 purge_mock.py <results_dir> <tag>"""
import json
import os
import sys

res_dir, tag = sys.argv[1], sys.argv[2]
MOCK_MARK = '"thought": "患者主诉涉及多个系统'

for kind in ("agent", "zero"):
    for task in ("syndrome", "pathogenesis"):
        fp = os.path.join(res_dir, f"评估_{kind}_{tag}_{task}.jsonl")
        if not os.path.exists(fp):
            continue
        kept, dropped = [], 0
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    dropped += 1
                    continue
                tail = r.get("response_tail", "")
                if MOCK_MARK in tail or "[ERROR]" in tail or r.get("elapsed_s", 99) < 5:
                    dropped += 1
                    continue
                kept.append(line)
        with open(fp, "w", encoding="utf-8") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
        if dropped:
            print(f"{fp}: 清除 {dropped} 行，保留 {len(kept)} 行")
