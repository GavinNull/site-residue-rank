#!/usr/bin/env python3
"""截尾检测器的自检（CLAUDE.md §10）。

为什么要单独一个脚本：这是本次分析里**第一个被抓到的工具错误**，而且是静默的。
subspace_lib 最初沿用 ligand_specific_rank.py 的约定「数恰好等于最小值的个数」。
那个约定在 Starr RBD 数据上有效（非结合变体被赋成同一个下限值），
但在 PDCoV 上失效：bind_hAPN / bind_pAPN 的下限堆积被噪声摊开，
恰好等于最小值的只有 0.04%，旧检测器报 **0.0% 截尾**。
而 docs/12-rank-two.md §2 的已建立事实是 hAPN 给秩 0、pAPN 不可判
—— 即两者重度截尾。旧检测器会让它们混进判别集。

自检用的已知事实（全部来自 docs/12-rank-two.md，不是自洽性）：
  必须 **高**（>= 25%）: PDCoV/hAPN（秩 0）、PDCoV/pAPN（不可判）
  必须 **低**  (< 10%) : PDCoV/gAPN（秩 2，12/12 种子）、
                        sars2_variants 的干净骨架（秩 2）、starr2020/WH1（秩 2）
  必须 **高**         : sarbecovirus 里 tie 比例已 >= 70% 的那几列
既测能命中（抓住 hAPN/pAPN），也测不误伤（不把 gAPN 和干净矩阵判成截尾）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (CENSOR_WINDOW, censor_fraction,  # noqa: E402
                          load_all_dms, load_independent, load_skempi)

MUST_HIGH = {"PDCoV/PDCoV/hAPN": "docs/12 §2: 秩 0，单调上升",
             "PDCoV/PDCoV/pAPN": "docs/12 §2: 不可判，平坦非单调",
             "sarbecovirus/SARS-CoV-1_Urban/RalcACE2": "tie 比例 72%",
             "sarbecovirus/RshSTT182/RlanACE2": "tie 比例 83%"}
MUST_LOW = {"PDCoV/PDCoV/gAPN": "docs/12 §2: 秩 2，12/12 种子，真内部极小",
            "starr2020/WH1/huACE2": "docs/12 §1: 秩 2，4003 观测",
            "sars2_variants/Omicron_BA2/ACE2": "docs/12 §3: 秩 2",
            "lehner/PSD95-PDZ3/CRIPT": "ddPCA 推出的 ΔΔG，无检测下限赋值",
            "SKEMPI/OMTKY3/pooled-4proteases": "文献 Ki，无下限赋值",
            "OMTKY3/chymotrypsin/1CHO": "Laskowski 完整测全，无下限赋值"}
HIGH_THR, LOW_THR = 0.25, 0.10


def main():
    print("=" * 84)
    print(f"截尾检测器自检  窗口 = min + {CENSOR_WINDOW}（原生单位）")
    print("=" * 84)
    by = {d["name"]: d for d in
          load_independent() + load_skempi() + load_all_dms()}
    ok = True
    print(f"\n{'矩阵':<44}{'截尾':>7}  判据")
    print("-" * 84)
    for nm, why in MUST_HIGH.items():
        if nm not in by:
            print(f"{nm:<44}{'缺失':>7}  << 找不到该矩阵")
            ok = False
            continue
        c = by[nm]["censor_frac"]
        hit = c >= HIGH_THR
        print(f"{nm[:43]:<44}{c:>7.1%}  必须 >= {HIGH_THR:.0%}（{why}）"
              f"  {'OK' if hit else '<< 漏检'}")
        ok &= hit
    for nm, why in MUST_LOW.items():
        if nm not in by:
            print(f"{nm:<44}{'缺失':>7}  << 找不到该矩阵")
            ok = False
            continue
        c = by[nm]["censor_frac"]
        hit = c < LOW_THR
        print(f"{nm[:43]:<44}{c:>7.1%}  必须 <  {LOW_THR:.0%}（{why}）"
              f"  {'OK' if hit else '<< 误伤'}")
        ok &= hit

    # 构造性检查：人工把一个干净矩阵截尾，检测器必须跟上
    print("\n构造性检查：对干净矩阵人工施加截尾，检测比例必须跟着上升")
    d = by["sars2_variants/Omicron_BA2/ACE2"]
    v = d["M"][d["mask"]].copy()
    print(f"   {'人工截断分位':>12}{'检测到的截尾':>14}")
    prev = -1.0
    mono = True
    for q in (0.0, 0.1, 0.2, 0.3, 0.5):
        w = v.copy()
        if q > 0:
            thr = np.quantile(w, q)
            w[w < thr] = thr
        c = censor_fraction(w)
        print(f"   {q:>12.0%}{c:>14.1%}")
        mono &= c >= prev - 1e-9
        prev = c
    print(f"   判据：单调不降且 50% 截断时检测 >= 40%  "
          f"{'OK' if mono and prev >= 0.40 else '<< 检测器不跟随'}")
    ok &= mono and prev >= 0.40

    print("\n" + "=" * 84)
    print("截尾检测器自检:", "通过" if ok else "** 未通过 —— 中止 **")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
