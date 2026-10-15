#!/usr/bin/env python3
"""把 docs/34 §4 的「理想条件还原性诊断」**落盘**，并核对它复现了文档里的数。

### 为什么需要这个脚本

整篇「第二轴不普适」的结论最后挂在一个量级对照上：
  若第二轴真的完全共享，本管线会测到跨家族 cos θ2 = 0.901、OMTKY3 家族内 0.821；
  实测是 0.105 / 0.146。
**那两个「若共享」的上界，此前没有任何机器可读的出处。**
`analysis/lambda2_power.py` 在主入口里确实写了 `"ideal"` 与 `"magcal"` 两个键，
但落盘的 `cache/lambda2_power.json` 是由 `merge` 路径（把 30 个并行格子合并）生成的，
而 `merge` 构造输出时漏掉了这两个键。于是这个诊断的结果只活在 docs/34 §4 的表里，
以及论文与 `make_figures.fig4` 里**抄过去的常量**。

一个更险的细节：我第一次想核对时，在缓存里搜 0.901，命中的全是 `lam1_med`
（λ1 恰好也约等于 0.901）。**若据此「绑定」，图 4 会被绑到一个完全不同的量上，
而且会报告「已核对」。** 数字相撞不是出处。

### 做法

不动已提交的功效分析，也不重跑那 30 个格子。只复用它的 `_prep` 与 `measure`，
**同一个种子（11）、同一个重复数（60）**，所以产出必须逐位复现 docs/34 §4 的表。
复现不了就中止——那说明有东西不对，而不是「顺手更新一下数字」。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import lambda2_power as LP  # noqa: E402

OUT = os.path.join(ROOT, "cache", "lambda2_ideal.json")

# docs/34 §4 表里印出来的值——这些是**待复现的目标**，不是被信任的输入。
DOC34 = {
    "cov":   {"cos1": 0.992, "cos2": 0.981},
    "omt":   {"cos1": 0.945, "cos2": 0.821},
    "cross": {"cos1": 0.974, "cos2": 0.901},
}
DOC34_LAM = [0.970, 0.918, 0.048]
TOL = 0.0015   # 文档只印三位小数


def main():
    LP.require_harness()
    by, by_mat, OBS_PM, geo = LP._prep()
    par_ideal = {"a_c": 1.0, "a_w": {"cov": 1.0, "omt": 1.0},
                 "b_c": 1.0, "b_w": {"cov": 1.0, "omt": 1.0}}
    pm, lam = LP.measure(geo, par_ideal, np.random.default_rng(11), 60)

    print("理想条件（全部 ρ = 1，60 次，种子 11）下的成对中位 vs docs/34 §4：")
    bad = []
    for k in ("cov", "omt", "cross"):
        for j, nm in ((0, "cos1"), (1, "cos2")):
            got, want = float(pm[k][j]), DOC34[k][nm]
            ok = abs(got - want) <= TOL
            if not ok:
                bad.append(f"{k} {nm}: 复现 {got:.4f} vs 文档 {want}")
            print(f"   {k:<6} {nm}  复现 {got:.4f}   文档 {want:.3f}   "
                  f"{'ok' if ok else '**不符**'}")
    lam = [float(x) for x in np.asarray(lam)]
    for i, (g, w) in enumerate(zip(lam, DOC34_LAM)):
        ok = abs(g - w) <= TOL
        if not ok:
            bad.append(f"λ{i+1}: 复现 {g:.4f} vs 文档 {w}")
        print(f"   λ{i+1}    复现 {g:.4f}   文档 {w:.3f}   {'ok' if ok else '**不符**'}")

    if bad:
        print("\n**未能复现 docs/34 §4 的表 —— 不落盘，也不得更新任何引用：**")
        for b in bad:
            print("   -", b)
        sys.exit(1)

    out = {
        "what": "docs/34 §4 理想条件还原性诊断（全部 ρ=1）：若第二轴完全共享，本管线测到多少",
        "seed": 11, "nrep": 60,
        "pair_med_ideal": {k: [float(pm[k][0]), float(pm[k][1])] for k in pm},
        "lam_ideal": lam,
        "obs_pair_med": {k: [float(v[0]), float(v[1])] for k, v in OBS_PM.items()},
        "reproduces_docs34_tol": TOL,
        "source_script": "analysis/lambda2_ideal_persist.py",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n逐位复现 docs/34 §4，已落盘 → {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
