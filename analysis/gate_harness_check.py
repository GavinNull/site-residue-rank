#!/usr/bin/env python3
"""闸标定 harness 的强制自检（CLAUDE.md §10）—— 不过就停，不许往下算。

三件事，缺一不可（§10.2「既测能命中，也测不误伤」）：

[H1] **复现 docs/23 §5.2 的秩 1 下限/上限**。
     若复现不出来，就说明我的 harness 与 docs/23 的不是同一个，
     那么本文档的秩 2 数字和 docs/23 的秩 1 数字**不可并列引用**。
     判据（docs/23 §5.2 表）：
       titeseq/4-4-20  纯噪声下限 0.647、真方向上限 0.660（gap +0.014）
       OMTKY3/1CHO     下限 0.726、上限 0.983（gap +0.257）
       starr2020/WH1   下限 0.603、上限 0.996（gap +0.393）

[H2] **命中**：同几何、已知秩 2 结构、**高 SNR**（sigma = 0.05）时，
     秩 2 的 stab cos2 必须贴近 1。否则「秩 2 测不出来」会是我的错，
     不是矩阵的错（§10 末：区分工具的错与被检查对象的错）。

[H3] **不误伤 / 不误报**：**完全无结构**的矩阵必须落进下限分布里，
     并且秩 2 的随机零分布核对必须对得上解析值
     （`haar_pair_null` 的 Σcos² 期望 = k²/(d−1)）。

另外记一条量级校准（§8）：秩 2 随机零分布的 cos2 p95 只有 ~0.31，
但 **bootstrap stab cos2 的纯噪声下限完全不是这个数** ——
stab 不是「两个独立随机面」的比较，而是「同一份噪声的重抽 vs 自己」，
噪声里的偶然结构会被两边同时看到。这正是 `0.60` 这个固定门槛失效的机制。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_calib_lib as G  # noqa: E402
from subspace_lib import ci  # noqa: E402

# docs/23 §5.2 的已发表值（**这是「已知事实」一侧，不是自洽性**）
PUB = {
    "titeseq/4-4-20/fluorescein": (0.647, 0.660, 0.014),
    "OMTKY3/chymotrypsin/1CHO": (0.726, 0.983, 0.257),
    "starr2020/WH1/huACE2": (0.603, 0.996, 0.393),
}
TOL = 0.02          # 蒙特卡洛同种子同参数，差应 << 这个
TOL_GAP = 0.03


def main():
    mats = G.load_evidence()
    ok = True
    print("=" * 96)
    print("[H1] 复现 docs/23 §5.2 的秩 1 下限 / 上限（同种子、同参数）")
    print("=" * 96)
    print(f"  {'体系':<34}{'实测stab':>9}{'下限(本)':>9}{'下限(23)':>9}"
          f"{'上限(本)':>9}{'上限(23)':>9}{'gap(本)':>9}{'gap(23)':>9}")
    print("  " + "-" * 92)
    rep = {}
    for nm, (f23, c23, g23) in PUB.items():
        d = mats[nm]
        shape, mask = d["M"].shape, d["mask"]
        obs = G.stab_boot(d["M"], mask, rank=1, nboot=600, seed=7)
        om = float(np.median(obs)) if obs else float("nan")
        e, _ = G.rankr_expl(d["M"], mask, rank=1)
        flo = G.floor_dist(shape, mask, rank=1, n_real=24, nboot=200)
        fm = float(np.median(flo))
        sg, tab = G.match_sigma(shape, mask, 1, e)
        hi = G.ceiling_dist(shape, mask, rank=1, sigma=sg, n_real=24,
                            nboot=200)
        hm = float(np.median(hi))
        gap = hm - fm
        good = (abs(fm - f23) < TOL and abs(hm - c23) < TOL
                and abs(gap - g23) < TOL_GAP)
        ok &= good
        print(f"  {nm[:33]:<34}{om:>9.3f}{fm:>9.3f}{f23:>9.3f}"
              f"{hm:>9.3f}{c23:>9.3f}{gap:>+9.3f}{g23:>+9.3f}"
              f"   {'OK' if good else 'FAIL'}")
        rep[nm] = {"stab_obs_r1": om, "expl_r1": e, "floor_r1": fm,
                   "ceiling_r1": hm, "gap_r1": gap, "sigma": sg,
                   "pub": [f23, c23, g23], "ok": bool(good)}
    print(f"\n  [H1] {'通过 —— harness 与 docs/23 同一个' if ok else '**不通过：必须停**'}")
    if not ok:
        print("  按 CLAUDE.md §10 停止：harness 与 docs/23 不一致，"
              "秩 2 的数字不可信也不可与 docs/23 并列。")
        return 1

    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("[H2] 命中：高 SNR 下已知秩 2 结构必须被还原（stab cos2 -> 1）")
    print("=" * 96)
    h2 = True
    for nm in ("OMTKY3/chymotrypsin/1CHO", "starr2020/WH1/huACE2",
               "titeseq/4-4-20/fluorescein"):
        d = mats[nm]
        v = G.hi_snr_check(d["M"].shape, d["mask"], rank=2, sigma=0.05)
        m = float(np.median(v)) if v else float("nan")
        good = np.isfinite(m) and m > 0.95
        h2 &= good
        print(f"  {nm[:40]:<42}sigma=0.05 的 stab cos2 中位 = {m:.4f}  "
              f"{'OK' if good else 'FAIL'}")
    print(f"  [H2] {'通过' if h2 else '**不通过**'}")
    ok &= h2

    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("[H3] 不误报：无结构矩阵必须落进下限分布；零分布解析核对")
    print("=" * 96)
    h3 = True
    for nm in ("OMTKY3/chymotrypsin/1CHO", "starr2020/WH1/huACE2"):
        d = mats[nm]
        flo = G.floor_dist(d["M"].shape, d["mask"], rank=2, n_real=24,
                           nboot=200)
        lo, hi = ci(flo)
        # 另取 12 个**全新种子**的无结构矩阵，必须落在上面那个分布的区间里
        held = []
        for t in range(12):
            rng = np.random.default_rng(77000 + t)
            X = G.synth_noise(d["M"].shape, rng)
            v = G.stab_boot(X, d["mask"], rank=2, nboot=200,
                            seed=88000 + t)
            if v:
                held.append(float(np.median(v)))
        inside = float(np.mean([(lo <= z <= hi) for z in held]))
        good = inside >= 0.75
        h3 &= good
        print(f"  {nm[:40]:<42}下限分布 中位 {np.median(flo):.3f} "
              f"[{lo:.2f},{hi:.2f}]；新种子 12 个无结构矩阵落入比例 "
              f"{inside:.0%}  {'OK' if good else 'FAIL'}")
    n2 = G.null2(20)
    ssc_sim = float(np.mean(n2["ssc"]))
    ssc_ana = 2.0 ** 2 / (20 - 1)
    good = abs(ssc_sim - ssc_ana) < 0.01
    h3 &= good
    print(f"  秩 2 随机零分布 E[Σcos²] 模拟 {ssc_sim:.4f} vs 解析 "
          f"k²/(d−1) = {ssc_ana:.4f}  {'OK' if good else 'FAIL'}")
    print(f"  量级校准：随机两个 2 维面的 cos2 p95 = "
          f"{np.percentile(n2['cos2'], 95):.3f}、cos1 p95 = "
          f"{np.percentile(n2['cos1'], 95):.3f}、cos1 中位 = "
          f"{np.median(n2['cos1']):.3f}")
    print(f"  [H3] {'通过' if h3 else '**不通过**'}")
    ok &= h3

    print("\n" + "=" * 96)
    print(f"harness 总判定: {'全部通过' if ok else '**不通过 —— 停**'}")
    print("=" * 96)
    import json
    p = os.path.join(G.ROOT, "cache", "gate_harness_check.json")
    json.dump({"h1": rep, "ok": bool(ok)}, open(p, "w"),
              ensure_ascii=False, indent=1, default=float)
    print("产物:", p)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
