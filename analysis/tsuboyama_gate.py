#!/usr/bin/env python3
"""docs/27 Step 3：可分辨闸（G4），**按 docs/24 标定过的口径**。

判据全部照 docs/24，不新造：
  * 门槛 = **该矩阵自己几何的纯噪声 p95**，不是平坦的 0.60
    （docs/24 §0.1–§0.3：0.60 从来没有在闸实际运行的秩 2 上标定过；
     跨几何的真实下限跨度 0.145，所以逐矩阵标定不是可选项）；
  * 判别力判据 = **上限中位 − 下限 p95 ≥ 0.10**，不是 docs/23 的
    「上限中位 − 下限中位」（docs/24 §0.7：旧判据会误判，
     titeseq/4-4-20 在旧判据下 +0.132「有判别力」、新判据下 −0.041）；
  * 门槛分位数 **p90 与 p95 两套都报**（docs/24 §9.2 的未决项：
    分位数的取法本身是一条纳入判据，它在 3SGB 上就翻了符号。
    本文档两套都报，这样人类的最终取法不需要重做本步）。

硬前置：`cache/gate_harness_check.json` 的 `ok` 必须为 true
（docs/24 §2.2：harness 必须先逐值复现 docs/23，不过就停）。

配置（CLAUDE.md §5：搜索量记录）：下限/上限各 **60 次实现 × 200 次
bootstrap**，实测值 2000 次 bootstrap。用 60 而不是 docs/24 headline 的 120，
依据是 docs/24 §3.5 的实测敏感性（24×200 / 60×200 / 120×200 / 60×100 /
60×400 五个配置下，下限 p95 的最大跨度 **0.023**）。
`--verify-120` 会在一个子集上用 120×200 重跑，核对判读不翻。

运行：python3 analysis/tsuboyama_gate.py [--jobs 9] [--verify-120 12]
产物：cache/tsuboyama_gate.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_calib_lib as G                                      # noqa: E402
import tsuboyama_load as TL                                     # noqa: E402
from subspace_lib import ROOT                                   # noqa: E402

# docs/24 §3.2 的 12 格细 σ 网格（6 格在小矩阵上太粗，会把多个矩阵
# 并进同一格并给出同一个上限 —— 那是网格的伪影）
SIGMAS_FINE = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.65,
               0.80, 1.20, 2.00]
N_REAL, NBOOT = 60, 200
N_BOOT_OBS = 2000
MIN_GAP = 0.10            # docs/24：上限中位 − 下限 p95


def one(args):
    name, M, mask = args
    expl, err = G.rankr_expl(M, mask, rank=2)
    if expl is None:
        return name, {"err": err}
    obs = G.stab_boot(M, mask, rank=2, nboot=N_BOOT_OBS, seed=7)
    if not obs:
        return name, {"err": "stab 无法计算"}
    fl = G.floor_dist(M.shape, mask, rank=2, n_real=N_REAL, nboot=NBOOT)
    sg, table = G.match_sigma(M.shape, mask, 2, expl, sigmas=SIGMAS_FINE)
    ce = G.ceiling_dist(M.shape, mask, rank=2, sigma=sg, n_real=N_REAL,
                        nboot=NBOOT)
    hs = G.hi_snr_check(M.shape, mask, rank=2, sigma=0.05)
    fp = G.pcts(fl)
    out = {
        "sites": int(M.shape[0]), "n_obs": int(mask.sum()),
        "rank2_expl": expl, "obs_stab_cos2": float(np.median(obs)),
        "obs_ci": [float(np.percentile(obs, 2.5)),
                   float(np.percentile(obs, 97.5))],
        "floor_median": fp[50], "floor_p90": fp[90], "floor_p95": fp[95],
        "floor_p99": fp[99], "matched_sigma": float(sg),
        "ceiling_median": float(np.median(ce)),
        "hi_snr_median": float(np.median(hs)) if hs else float("nan"),
        "sigma_table": table,
    }
    out["margin_p90"] = out["obs_stab_cos2"] - out["floor_p90"]
    out["margin_p95"] = out["obs_stab_cos2"] - out["floor_p95"]
    out["gap_p95"] = out["ceiling_median"] - out["floor_p95"]
    out["gap_p90"] = out["ceiling_median"] - out["floor_p90"]
    out["discriminating"] = bool(out["gap_p95"] >= MIN_GAP)
    out["pass_p95"] = bool(out["margin_p95"] > 0 and out["discriminating"])
    out["pass_p90"] = bool(out["margin_p90"] > 0
                           and out["gap_p90"] >= MIN_GAP)
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--verify-120", type=int, default=12)
    a = ap.parse_args()

    hc = json.load(open(os.path.join(ROOT, "cache",
                                     "gate_harness_check.json")))
    assert hc.get("ok") is True, "gate_harness_check.json 的 ok 不为 true"
    print("[硬前置] cache/gate_harness_check.json  ok = true"
          "（harness 逐值复现 docs/23；H1/H2/H3 全过）")

    info, mats = TL.load_cached()
    cen = json.load(open(os.path.join(ROOT, "cache",
                                      "tsuboyama_censor.json")))
    # Step 3 只跑 Step 2 过闸的那些（docs/25 的顺序：闸不许倒置）
    prim = cen["primary_pass"]                 # parent -> name（天然独立集）
    todo_nat = sorted(prim.values())
    todo_des = sorted(w for w in cen["natural_pass"]) and []
    des_pass = sorted(w for w, r in cen["rows"].items()
                      if r["pass_step2"] and not r["natural"])
    print(f"\nStep 2 过闸：天然独立 {len(todo_nat)} 个、设计 {len(des_pass)} 个")
    print(f"本步对**天然独立集**全数做逐几何标定；设计蛋白留作第二层"
          f"（§6 的跨域检验用，不进主判决）")
    print(f"配置：下限/上限 {N_REAL} 次实现 × {NBOOT} 次 bootstrap、"
          f"实测 {N_BOOT_OBS} 次；σ 网格 {len(SIGMAS_FINE)} 格（docs/24 §3.2 细网格）")

    jobs = [(w, mats[w]["M"], mats[w]["mask"]) for w in todo_nat]
    res = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, (nm, r) in enumerate(ex.map(one, jobs, chunksize=1)):
            res[nm] = r
            if (i + 1) % 20 == 0:
                print(f"   ... {i + 1}/{len(jobs)}")
    good = {k: v for k, v in res.items() if "err" not in v}
    print(f"\n完成 {len(good)}/{len(jobs)}（{len(res) - len(good)} 个抽不出子空间）")

    # ---- 高 SNR 命中自检（§10.2：只测上限不测下限的护栏会被绕过）------
    hs = np.array([v["hi_snr_median"] for v in good.values()])
    print(f"\n[自检 H2 逐矩阵] σ = 0.05 下已知秩 2 结构的 stab cos2："
          f"中位 {np.median(hs):.4f}、最小 {hs.min():.4f}（应贴近 1）")
    assert hs.min() > 0.90, "高 SNR 下还原不出已知秩 2 —— harness 在这些几何上不可用"

    # ---- 判决 --------------------------------------------------------
    p95 = [k for k, v in good.items() if v["pass_p95"]]
    p90 = [k for k, v in good.items() if v["pass_p90"]]
    nd = [k for k, v in good.items() if not v["discriminating"]]
    print("\n" + "=" * 102)
    print("逐几何门槛判决（两套分位数都报 —— docs/24 §9.2 的未决纳入判据）")
    print("=" * 102)
    print(f"   门槛取 **p95**：通过 **{len(p95)}/{len(good)}**")
    print(f"   门槛取 **p90**：通过 **{len(p90)}/{len(good)}**")
    print(f"   在 p90 过而 p95 不过的（分位数敏感的那些）："
          f"**{len(set(p90) - set(p95))}** 个")
    print(f"   判别力不足（上限中位 − 下限 p95 < {MIN_GAP:.2f}，判「不可读」"
          f"而不是「是噪声」）：{len(nd)} 个")
    fl = np.array([v["floor_p95"] for v in good.values()])
    print(f"\n   逐几何下限 p95 的实测跨度：{fl.min():.3f}–{fl.max():.3f}"
          f"（中位 {np.median(fl):.3f}）"
          f" —— 平坦 0.60 落在 {'区间内' if fl.min() < 0.60 < fl.max() else '区间外'}；")
    print(f"      若用平坦 0.60 会通过 "
          f"{sum(1 for v in good.values() if v['obs_stab_cos2'] > 0.60)} 个，"
          f"逐几何 p95 下通过 {len(p95)} 个 —— 差 "
          f"{sum(1 for v in good.values() if v['obs_stab_cos2'] > 0.60) - len(p95)} 个")
    ob = np.array([v["obs_stab_cos2"] for v in good.values()])
    mg = np.array([v["margin_p95"] for v in good.values()])
    print(f"   实测 stab cos2：中位 {np.median(ob):.3f}、"
          f"范围 {ob.min():.3f}–{ob.max():.3f}")
    print(f"   余量（实测 − 下限 p95）：中位 **{np.median(mg):+.3f}**、"
          f"10% 分位 {np.percentile(mg, 10):+.3f}")
    print(f"   对照（docs/24 §3.1）：冠状病毒成员余量 +0.342…+0.391、"
          f"存活的三个 OMTKY3 只有 +0.051…+0.105")

    show = sorted(good, key=lambda k: -good[k]["n_obs"])[:15]
    print(f"\n   前 15 个（按观测数）：")
    print(f"   {'矩阵':<20}{'位点':>5}{'观测':>6}{'秩2解释':>9}{'实测':>8}"
          f"{'下限中位':>9}{'p90':>7}{'p95':>7}{'上限':>7}"
          f"{'余量p90':>9}{'余量p95':>9}{'gap':>7}{'判(p95)':>9}")
    for k in show:
        v = good[k]
        print(f"   {k:<20}{v['sites']:>5}{v['n_obs']:>6}"
              f"{v['rank2_expl']:>9.1%}{v['obs_stab_cos2']:>8.3f}"
              f"{v['floor_median']:>9.3f}{v['floor_p90']:>7.3f}"
              f"{v['floor_p95']:>7.3f}{v['ceiling_median']:>7.3f}"
              f"{v['margin_p90']:>+9.3f}{v['margin_p95']:>+9.3f}"
              f"{v['gap_p95']:>7.3f}{('过' if v['pass_p95'] else '不过'):>9}")

    # ---- 120×200 核对（docs/24 §3.5 的敏感性在本数据上复查）-----------
    ver = {}
    if a.verify_120 > 0:
        pick = sorted(good, key=lambda k: abs(good[k]["margin_p95"]))[
            :a.verify_120]
        print(f"\n[敏感性核对] 对余量最接近 0 的 {len(pick)} 个矩阵用 "
              f"120×200 重跑下限（docs/24 §3.5 口径）")
        for k in pick:
            fl2 = G.floor_dist(mats[k]["M"].shape, mats[k]["mask"], rank=2,
                               n_real=120, nboot=NBOOT, seed_noise=5000,
                               seed_boot=900)
            q = G.pcts(fl2)
            flip95 = ((good[k]["obs_stab_cos2"] > q[95])
                      != good[k]["pass_p95"]) and good[k]["discriminating"]
            ver[k] = {"floor_p95_120": q[95], "floor_p90_120": q[90],
                      "floor_p95_60": good[k]["floor_p95"],
                      "margin_p95_120": good[k]["obs_stab_cos2"] - q[95],
                      "flip": bool(flip95)}
            print(f"   {k:<20} p95 60×200 {good[k]['floor_p95']:.3f} → "
                  f"120×200 {q[95]:.3f}  Δ {q[95] - good[k]['floor_p95']:+.3f}"
                  f"  余量 {good[k]['obs_stab_cos2'] - q[95]:+.3f}"
                  f"  {'**翻转**' if flip95 else '判读不变'}")
        nf = sum(v["flip"] for v in ver.values())
        d = np.array([abs(v["floor_p95_120"] - v["floor_p95_60"])
                      for v in ver.values()])
        print(f"   => {nf}/{len(ver)} 个判读翻转；"
              f"下限 p95 的 60↔120 最大差 {d.max():.3f}"
              f"（docs/24 §3.5 在五个配置上测到的最大跨度是 0.023）")

    out = {"config": {"n_real": N_REAL, "nboot": NBOOT,
                      "n_boot_obs": N_BOOT_OBS, "min_gap": MIN_GAP,
                      "sigmas": SIGMAS_FINE},
           "harness_ok": True,
           "n_input": len(jobs), "n_done": len(good),
           "pass_p95": sorted(p95), "pass_p90": sorted(p90),
           "not_discriminating": sorted(nd),
           "designed_step2_pass": des_pass,
           "rows": res, "verify_120": ver}
    p = os.path.join(ROOT, "cache", "tsuboyama_gate.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
