#!/usr/bin/env python3
"""docs/22 的残余不确定性量化：若 Gen-3 恰好在「锐度低」的单元上漏掉了
原子之上的摊开层（`censor_gen3_atom_power.py` 的 [P3] 诊断显示它在那种
构型下 6/8 欠检），headline 的 r 会变成多少？

这不是一个新候选，也不是调参。它是一个**最坏情形替换**：
  对锐度 `sharp = 精确并列 / 窗口内比例 < THR` 的单元（下限附近有
  一半以上的质量**不是**精确并列，即「原子 = 全部堆积」最站不住的那些），
  把 Gen-3 的值换成 Gen-2 的值（= 假设那层摊开质量全是截尾）；
  其余单元保留 Gen-3。
两端夹出 r 的真实可动范围，而不是只报一个点。

输入：cache/censor_gen3_final.json（由 censor_gen3_final.py 产生）
运行：python3 analysis/censor_gen3_sensitivity.py
产物：cache/censor_gen3_sensitivity.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from censor_gen3_final import boot_corr, derived, stat_selfcheck  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    for nm, good in stat_selfcheck():
        if not good:
            print(f"统计自检失败：{nm} —— 中止")
            return 1
    d = json.load(open(os.path.join(ROOT, "cache",
                                    "censor_gen3_final.json")))
    units = d["units"]
    assert len(units) == 34, "单元集不是 34 —— 中止"
    print("=" * 92)
    print("残余不确定性：锐度低的单元若被 Gen-3 欠检，r 会动多少")
    print("=" * 92)
    out = {}
    print(f"\n{'锐度阈':>7}{'被替换的单元':>12}{'中位截尾':>10}"
          f"{'Pearson':>10}{'95% 区间':>22}{'>=25% 的 n':>12}{'其中秩0':>9}")
    for thr in (0.0, 0.50, 0.60, 0.80, 1.01):
        for u in units:
            u["g3s"] = u["g2"] if u["sharp"] < thr else u["g3"]
        nrep = sum(1 for u in units if u["sharp"] < thr)
        x = [u["g3s"] for u in units]
        y = [u["rank_orig"] for u in units]
        p = boot_corr(x, y, 4000, 11, "pearson")
        s = boot_corr(x, y, 4000, 11, "spearman")
        dd = derived(units, "g3s", "rank_orig")
        print(f"{thr:>7.2f}{nrep:>12}{np.median(x):>10.1%}"
              f"{p['point']:>+10.3f}"
              f"{'['+format(p['lo'],'+.3f')+', '+format(p['hi'],'+.3f')+']':>22}"
              f"{dd['n_ge25']:>12}{dd['n_r0_ge25']:>9}")
        out[f"sharp_lt_{thr:.2f}"] = {
            "n_replaced": nrep, "median": float(np.median(x)),
            "pearson": p, "spearman": s,
            "n_ge25": dd["n_ge25"], "n_r0_ge25": dd["n_r0_ge25"],
            "n_lt10": dd["n_lt10"], "frac_r2_lt10": dd["frac_r2_lt10"]}
    print("\n读法：阈 0.00 = 纯 Gen-3（无替换）；阈 1.01 = 全部替换 = 纯 Gen-2。")
    print("中间几行是「只在原子最站不住的单元上让步」的最坏情形。")
    lo = min(v["pearson"]["point"] for v in out.values())
    hi = max(v["pearson"]["point"] for v in out.values())
    print(f"\nPearson 点估计在全部替换方案下的跨度：[{hi:+.3f}, {lo:+.3f}]"
          f"  （跨度 {abs(hi - lo):.3f}，对比单个 bootstrap 区间宽 ≈ 0.18）")
    out["span"] = {"min": float(lo), "max": float(hi),
                   "width": float(abs(hi - lo))}
    json.dump(out, open(os.path.join(ROOT, "cache",
                                     "censor_gen3_sensitivity.json"), "w"),
              indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
