#!/usr/bin/env python3
"""Part 1 的两个必要补正。

**补正 1：σ 网格在 10 位点几何上太粗。**
`gate_rank2_calibration.py` 用 docs/23 的 6 格 σ 网格
（0.15 / 0.3 / 0.5 / 0.8 / 1.2 / 2.0）。在 OMTKY3 的 10×20 完备几何上
这 6 格给出的秩 2 解释率是 87.0% / 68.3% / 56.2% / 49.8% / 47.9% / 50.7%，
而 4 个 OMTKY3 矩阵自己的解释率是 77.5% / 76.7% / 66.3% / 62.7% ——
**全部落到同一格（σ = 0.3）**，于是四个矩阵拿到同一个上限 0.810。
这是网格的伪影，不是事实：1CHO 的 77.5% 明显在 68.3% 与 87.0% 之间。
所以这里用细网格重算，让每个矩阵拿到自己的上限。

**补正 2：判别力判据该用 `上限 − 下限 p95`，不是 `上限 − 下限中位`。**
docs/23 §5.2 用的是中位差 ≥ 0.10。但闸要判的是「实测能不能超出纯噪声的
上尾」，所以真正可用的区间是 `[下限 p95, 上限]`。
titeseq/4-4-20 的秩 2 就是反例：中位差 = +0.132（按旧判据「有判别力」），
但上限 0.607 **低于**它自己的下限 p95 0.647 —— 即**即使信号是完美的
秩 2，也过不了那个几何的门槛**。旧判据会把它判成「有判别力但不过闸」，
那是一个错误的判读：它根本不可判。本脚本把两个判据并列报出来。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_calib_lib as G  # noqa: E402

FINE = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80,
        1.20, 2.00]
N_REAL, N_BOOT = 120, 200


def main():
    cal = json.load(open(os.path.join(G.ROOT, "cache",
                                      "gate_rank2_calibration.json")))
    aux = json.load(open(os.path.join(G.ROOT, "cache",
                                      "gate_rank2_aux.json")))
    rows = {r["name"]: r for r in cal["rows"] + aux["rows"]}
    mats = G.load_evidence()

    print("=" * 104)
    print("[补正 1] OMTKY3 的 10×20 几何：细 σ 网格下每个矩阵自己的上限")
    print("=" * 104)
    d = mats["OMTKY3/chymotrypsin/1CHO"]
    shape, mask = d["M"].shape, d["mask"]
    grid = []
    for sg in FINE:
        ee = []
        for t in range(8):
            rng = np.random.default_rng(700 + t + int(sg * 1e4))
            X = G.synth_rank(shape, 2, sg, rng)
            x, _ = G.rankr_expl(X, mask, rank=2)
            if x is not None:
                ee.append(x)
        m = float(np.median(ee))
        hi = G.ceiling_dist(shape, mask, rank=2, sigma=sg, n_real=N_REAL,
                            nboot=N_BOOT)
        grid.append({"sigma": sg, "expl": m,
                     "ceiling": float(np.median(hi)),
                     "ceiling_ci": list(G.pcts(hi, (2.5, 97.5)).values())})
        print(f"   σ = {sg:<5} 秩 2 解释交互 {m:>6.1%}   上限 "
              f"{grid[-1]['ceiling']:.3f}")
    fl_p95 = rows["OMTKY3/chymotrypsin/1CHO"]["floor_p95"]
    fl_med = rows["OMTKY3/chymotrypsin/1CHO"]["floor_med"]
    print(f"\n   该几何的纯噪声下限：中位 {fl_med:.3f}、p95 {fl_p95:.3f}")
    print(f"\n   {'矩阵':<36}{'秩2解释':>8}{'匹配σ':>7}{'自己的上限':>11}"
          f"{'实测':>8}{'上限−p95':>10}{'实测−p95':>10}  判读")
    print("   " + "-" * 98)
    out = {"fine_grid": grid, "floor_p95": fl_p95, "floor_med": fl_med,
           "omtky3": []}
    for nm in G.OMT:
        r = rows[nm]
        near = min(grid, key=lambda z: abs(z["expl"] - r["expl_rank2"]))
        band = near["ceiling"] - fl_p95
        marg = r["stab_obs"] - fl_p95
        if band <= 0:
            v = "**该几何不可判（上限低于自己的下限 p95）**"
        elif marg > 0:
            v = "可分辨（实测超下限 p95）"
        else:
            v = "**与纯噪声不可区分**"
        print(f"   {nm[:35]:<36}{r['expl_rank2']:>8.1%}{near['sigma']:>7}"
              f"{near['ceiling']:>11.3f}{r['stab_obs']:>8.3f}"
              f"{band:>+10.3f}{marg:>+10.3f}  {v}")
        out["omtky3"].append({"name": nm, "expl": r["expl_rank2"],
                              "sigma": near["sigma"],
                              "ceiling_own": near["ceiling"],
                              "stab_obs": r["stab_obs"],
                              "band": band, "margin": marg, "verdict": v})

    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("[补正 2] 两个判别力判据的对照（全部已标定矩阵）")
    print("=" * 104)
    print(f"  {'矩阵':<40}{'上限':>8}{'下限中位':>9}{'下限p95':>8}"
          f"{'旧 gap':>8}{'新 gap':>8}{'实测':>8}  判读")
    print("  " + "-" * 100)
    comp = []
    for nm, r in rows.items():
        old = r["ceiling_med"] - r["floor_med"]
        new = r["ceiling_med"] - r["floor_p95"]
        if new <= 0:
            v = "**不可判（上限 < 下限 p95）**"
        elif r["stab_obs"] > r["floor_p95"]:
            v = "可分辨"
        else:
            v = "**与纯噪声不可区分**"
        comp.append({"name": nm, "ceiling": r["ceiling_med"],
                     "floor_med": r["floor_med"], "floor_p95": r["floor_p95"],
                     "gap_old": old, "gap_new": new,
                     "stab_obs": r["stab_obs"], "verdict": v})
        print(f"  {nm[:39]:<40}{r['ceiling_med']:>8.3f}{r['floor_med']:>9.3f}"
              f"{r['floor_p95']:>8.3f}{old:>+8.3f}{new:>+8.3f}"
              f"{r['stab_obs']:>8.3f}  {v}")
    flips = [c for c in comp if (c["gap_old"] >= 0.10) != (c["gap_new"] > 0)]
    print(f"\n  两个判据判读不同的矩阵：{len(flips)} 个"
          + ("".join(f"\n     {c['name']}：旧 gap {c['gap_old']:+.3f}"
                     f"（有判别力）但新 gap {c['gap_new']:+.3f}"
                     f"（上限过不了自己的下限 p95）" for c in flips)
             if flips else "（无）"))
    out["criteria_compare"] = comp
    out["flips"] = [c["name"] for c in flips]

    p = os.path.join(G.ROOT, "cache", "gate_rank2_fine.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\n产物:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
