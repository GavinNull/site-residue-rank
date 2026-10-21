#!/usr/bin/env python3
"""折叠对齐的 8 成员口径核对（docs/48 §2.4 引用的 0.807 / [0.784, 0.828] / 138/144）。

### 为什么有这个脚本

论文里的 0.755 [0.720, 0.775] 与 136/144 来自 **9 成员**的结合普适轴 u_D
（docs/21 的原构造，含已被人类裁定剔除的 3SGB，D-046）。证据集现在是 8 成员。
这是 D-075（引用跨家族统计量必须写明成员数）在头条数字上的又一次违反——
docs/47 描述了这个口径、docs/48 标注了它「没有文档记录是否改变数值」，
但缓存 `cache/tsuboyama_axis.json` 里**每个域同时存了两个口径的余弦**
（`cos_uD` 与 `cos_uA`），所以这件事可以直接核，不必推断。

只读缓存，不重算 144 个域的流水线。自检：9 成员口径必须复现论文里已发表的
0.755 / [0.720, 0.775] / 136/144，复现不了就中止——那说明读错了键，而不是「更新了数字」。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_axis.json"), encoding="utf-8"))
ok = set(d["repro_pass"])
rows = d["rows"]
keys = [k for k in rows if k in ok]


def stats(field):
    v = np.array([abs(rows[k][field]) for k in keys])
    p = np.array([rows[k]["p95_d"] for k in keys])
    rng = np.random.default_rng(0)
    bs = np.array([np.median(rng.choice(v, len(v))) for _ in range(4000)])
    return float(np.median(v)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], \
        int((v > p).sum()), len(v)


m9, ci9, n9, N = stats("cos_uD")
# 已知事实闸：9 成员口径必须复现已发表值（缓存自己的 summary 是金标准）
assert N == 144, f"通过复现筛选的域数 {N} ≠ 144"
assert abs(m9 - d["summary"]["cos_uD_median"]) < 1e-9, "9 成员中位未复现 summary"
assert abs(m9 - 0.755) < 0.0005 and n9 == 136, f"9 成员口径 {m9:.4f}, {n9}/144 与论文 0.755 / 136 不符"
assert abs(ci9[0] - 0.720) < 0.003 and abs(ci9[1] - 0.775) < 0.003, f"9 成员区间 {ci9} 与 [0.720, 0.775] 不符"

m8, ci8, n8, _ = stats("cos_uA")
print(f"9 成员 u_D：中位 {m9:.3f}  [{ci9[0]:.3f}, {ci9[1]:.3f}]  超自身 p95 {n9}/{N}   （复现论文 ✓）")
print(f"8 成员 u_A：中位 {m8:.3f}  [{ci8[0]:.3f}, {ci8[1]:.3f}]  超自身 p95 {n8}/{N}")
print(f"两条轴夹角余弦 u_cos_AD = {d['u_cos_AD']:.4f}")
json.dump({"u_D_9member": {"median": m9, "ci": ci9, "n_over_p95": n9, "N": N},
           "u_A_8member": {"median": m8, "ci": ci8, "n_over_p95": n8, "N": N},
           "u_cos_AD": d["u_cos_AD"], "bootstrap": "按域重抽 4000 次，种子 0",
           "source": "cache/tsuboyama_axis.json (cos_uD, cos_uA, p95_d, repro_pass)"},
          open(os.path.join(ROOT, "cache", "fold_align_8member.json"), "w"), indent=1)
