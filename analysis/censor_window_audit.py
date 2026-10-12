#!/usr/bin/env python3
"""静默欠检扫描：docs/13 的**保留**矩阵里还有没有被噪声摊开的下限堆积。

动机（CLAUDE.md §10.3）：analysis/censor_recheck.py 的标定发现修正检测器
是**固定窗口**的，窗口 w = 0.25，而最小值会被摊开噪声 σ 拉到堆积中心以下
约 3.5σ 处。所以 σ >~ w/3.5 = 0.07 时窗口整段落在堆积下方，检测值塌回 ~0
—— 和旧检测器一样的静默欠检，只是阈值移动了。
PD-CoV 被抓到了（σ ≈ 0.07，刚好在边界上）；本脚本检查**剩下的**矩阵里
还有没有同类情形，否则「保留 48 个」这个判别集本身就还有脏数据。

判据（不依赖自洽性，依赖窗口扫描的形状）：
  窗口从 0.25 放宽到 1.00，若检测值从 <10% 跳到 >=25%，说明堆积在
  0.25 窗口外、但在 1.00 窗口内 —— 可疑，必须人看。
  同时报堆积锐度（精确并列 / 窗口内比例）：接近 1 = 硬赋值（无摊开问题）。

运行：python3 analysis/censor_window_audit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (load_all_dms, load_independent,  # noqa: E402
                          load_skempi)

WINS = (0.25, 0.40, 0.60, 0.80, 1.00)


def main():
    mats = load_independent() + load_skempi() + load_all_dms()
    print("=" * 108)
    print("静默欠检扫描：窗口放宽时检测值的形状（docs/19 §2 的标定结果驱动）")
    print("=" * 108)
    print(f"{'矩阵':<46}{'观测':>6}{'并列':>7}" +
          "".join(f"{'w=' + format(w, '.2f'):>8}" for w in WINS) +
          f"{'锐度':>7}  标记")
    print("-" * 108)
    flagged = []
    for d in sorted(mats, key=lambda z: z["name"]):
        v = np.asarray(d["M"][d["mask"]], float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        lo = v.min()
        tie = float((v <= lo + 1e-6).mean())
        fr = [float(max(tie, (v <= lo + w).mean())) for w in WINS]
        sharp = tie / max(fr[0], 1e-9)
        # 可疑：0.25 窗口判为干净，但放宽窗口后跳到重度截尾
        susp = fr[0] < 0.10 and fr[-1] >= 0.25
        if susp:
            flagged.append((d["name"], fr))
        print(f"{d['name'][:45]:<46}{len(v):>6}{tie:>7.1%}" +
              "".join(f"{x:>8.1%}" for x in fr) +
              f"{sharp:>7.2f}  {'<< 可疑' if susp else ''}")
    print("-" * 108)
    if flagged:
        print(f"** {len(flagged)} 个矩阵在放宽窗口后跳进重度截尾区间，"
              f"需要人看是否为被摊开的下限堆积：")
        for nm, fr in flagged:
            print(f"   {nm}: w=0.25 -> {fr[0]:.1%}，w=1.00 -> {fr[-1]:.1%}")
        print("   注意：ΔΔG（kcal/mol）体系的 w=1.00 已接近一个整效应量级，"
              "放宽窗口本身会吃进真信号，所以这是**线索**不是判定。")
    else:
        print("没有矩阵呈现「窄窗干净 / 宽窗重度截尾」的形状 —— "
              "0.25 窗口的欠检风险在本批数据上没有第二个实例。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
