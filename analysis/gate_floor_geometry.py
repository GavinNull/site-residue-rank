#!/usr/bin/env python3
"""下限到底由什么决定？—— 对 docs/23 §5.2/§7.1 的机制性复核，以及对
并行任务（docs/25 CR9114）三条论断的独立核验。

docs/23 §5.2 观察到「10–43 位点上纯噪声的秩 1 stab 就有 0.65–0.80」，
并由此写下「小矩阵落在闸的分辨率以下」。那句话隐含一个机制：
**下限随位点数下降而上升**。这个机制从来没有被直接测过 —— 测过的是
三个几何上的三个点，而三个点连不出单调性。

本脚本直接测曲线，并把三件事分开：
  (1) 下限 vs 位点数（两个秩，合成满填充 mask，位点数从 10 扫到 201）；
  (2) 下限 vs 秩（同一几何上 rank 1 与 rank 2 的差）；
  (3) 固定门槛 0.60 相对于**两个秩**的噪声中位 / p95 落在哪里。
并给出 docs/23 §5.2 那张表的**第四行**（magma/319-345 的 0.801 / 0.867），
使强制复现的四个发表数字全部到齐。

这三件事必须分开报，因为它们给出的「该怎么改闸」互不相同：
  下限随几何变 => 需要**逐矩阵**门槛；
  下限随秩变   => 需要**逐秩**门槛；
  两者都小     => 固定 0.60 其实够用，问题在上限（= 矩阵自己的 SNR）。
"""
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_calib_lib as G  # noqa: E402

N_REAL, N_BOOT = 60, 200
SITES = [10, 20, 30, 43, 56, 84, 120, 201]
REAL = ["OMTKY3/chymotrypsin/1CHO", "OMTKY3/HLE-elastase/1PPF",
        "OMTKY3/subtilisin-Carlsberg/1R0R", "OMTKY3/SGPB/3SGB",
        "titeseq/4-4-20/fluorescein", "SKEMPI/OMTKY3/pooled-4proteases",
        "lehner/GB1/IgG-Fc", "lehner/GRB2-SH3/GAB2",
        "lehner/PSD95-PDZ3/CRIPT", "PDCoV/PDCoV/gAPN",
        "animalACE2/Wuhan-Hu-1/bat", "starr2020/WH1/huACE2"]

_C = {}


def _mats():
    if "m" not in _C:
        _C["m"] = G.load_evidence()
    return _C["m"]


def _magma():
    if "g" not in _C:
        from third_family_screen import cand_magma
        mg, _ = cand_magma()
        _C["g"] = {d["name"]: d for d in mg}
    return _C["g"]


def job_synth(arg):
    ns, rank = arg
    mask = np.ones((ns, 20), bool)
    fl = G.floor_dist((ns, 20), mask, rank=rank, n_real=N_REAL,
                      nboot=N_BOOT)
    p = G.pcts(fl)
    return {"sites": ns, "rank": rank, "med": p[50], "p90": p[90],
            "p95": p[95], "p99": p[99], "n": len(fl)}


def job_real(arg):
    nm, rank = arg
    d = _mats().get(nm) or _magma().get(nm)
    shape, mask = d["M"].shape, d["mask"]
    fl = G.floor_dist(shape, mask, rank=rank, n_real=N_REAL, nboot=N_BOOT)
    p = G.pcts(fl)
    e, _ = G.rankr_expl(d["M"], mask, rank=rank)
    obs = G.stab_boot(d["M"], mask, rank=rank, nboot=600, seed=7)
    sg, _t = G.match_sigma(shape, mask, rank, e)
    ce = G.ceiling_dist(shape, mask, rank=rank, sigma=sg, n_real=N_REAL,
                        nboot=N_BOOT)
    return {"name": nm, "rank": rank, "sites": int(shape[0]),
            "n_obs": int(mask.sum()), "fill": float(mask.mean()),
            "expl": e, "stab_obs": (float(np.median(obs)) if obs
                                    else float("nan")),
            "med": p[50], "p90": p[90], "p95": p[95], "p99": p[99],
            "ceiling": float(np.median(ce)) if ce else float("nan"),
            "sigma": sg}


def main():
    out = {"n_real": N_REAL, "n_boot": N_BOOT}

    # ------------------------------------------------------------------
    print("=" * 104)
    print("[A] 补齐 docs/23 §5.2 的第四行：magma/topbinHA_rep2/319-345（秩 1）")
    print("    发表值：纯噪声下限 0.801、真方向上限 0.867（gap +0.066）")
    print("=" * 104)
    mg = _magma()["magma/topbinHA_rep2/319-345"]
    e1, _ = G.rankr_expl(mg["M"], mg["mask"], rank=1)
    fl = G.floor_dist(mg["M"].shape, mg["mask"], rank=1, n_real=24,
                      nboot=200)
    sg, _t = G.match_sigma(mg["M"].shape, mg["mask"], 1, e1)
    ce = G.ceiling_dist(mg["M"].shape, mg["mask"], rank=1, sigma=sg,
                        n_real=24, nboot=200)
    f4, c4 = float(np.median(fl)), float(np.median(ce))
    ok4 = abs(f4 - 0.801) < 0.03 and abs(c4 - 0.867) < 0.04
    print(f"    几何 {mg['M'].shape[0]} 位点 / {int(mg['mask'].sum())} 观测；"
          f"秩 1 解释交互 {e1:.1%}")
    print(f"    本文档：下限 {f4:.3f}（发表 0.801）、上限 {c4:.3f}"
          f"（发表 0.867）、gap {c4 - f4:+.3f}（发表 +0.066）  "
          f"{'OK' if ok4 else '**不一致**'}")
    out["magma_anchor"] = {"floor": f4, "ceiling": c4, "gap": c4 - f4,
                           "pub": [0.801, 0.867, 0.066], "ok": bool(ok4)}

    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("[B] 下限 vs 位点数（合成满填充 mask，20 残基列；两个秩）")
    print("    直接检验 docs/23 §7.1 隐含的那条机制：下限随位点数下降而上升？")
    print("=" * 104)
    jobs = [(ns, r) for r in (1, 2) for ns in SITES]
    with Pool(min(cpu_count(), len(jobs))) as po:
        syn = po.map(job_synth, jobs)
    out["synthetic"] = syn
    for r in (1, 2):
        sub = [z for z in syn if z["rank"] == r]
        print(f"\n  秩 {r}：")
        print("     " + "".join(f"{z['sites']:>9}" for z in sub)
              + "   <- 位点数")
        print("     " + "".join(f"{z['med']:>9.3f}" for z in sub)
              + "   <- 下限中位")
        print("     " + "".join(f"{z['p95']:>9.3f}" for z in sub)
              + "   <- 下限 p95")
        med = [z["med"] for z in sub]
        mono = all(med[i] >= med[i + 1] - 0.005
                   for i in range(len(med) - 1))
        print(f"     单调下降（位点数↑ => 下限↓）？ "
              f"{'是' if mono else '**不是**'}；"
              f"中位跨度 {max(med) - min(med):.3f}"
              f"（{min(med):.3f}–{max(med):.3f}）")
        out[f"mono_rank{r}"] = bool(mono)

    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("[C] 下限 vs 秩，在**真实几何**上（这是闸实际面对的）")
    print("=" * 104)
    jobs = [(nm, r) for r in (1, 2) for nm in REAL]
    with Pool(min(cpu_count(), 10)) as po:
        real = po.map(job_real, jobs)
    out["real"] = real
    byn = {}
    for z in real:
        byn.setdefault(z["name"], {})[z["rank"]] = z
    print(f"{'体系':<38}{'位点':>5}{'观测':>6}"
          f"{'r1 中位':>8}{'r1 p95':>8}{'r2 中位':>8}{'r2 p95':>8}"
          f"{'r1−r2 中位':>11}{'r2 实测':>8}{'r2 上限':>8}{'r2 gap':>8}")
    print("-" * 104)
    d1, d2, dd = [], [], []
    for nm in REAL:
        a, b = byn[nm][1], byn[nm][2]
        d1.append(a["p95"]); d2.append(b["p95"]); dd.append(a["med"] - b["med"])
        print(f"{nm[:37]:<38}{a['sites']:>5}{a['n_obs']:>6}"
              f"{a['med']:>8.3f}{a['p95']:>8.3f}{b['med']:>8.3f}"
              f"{b['p95']:>8.3f}{a['med'] - b['med']:>+11.3f}"
              f"{b['stab_obs']:>8.3f}{b['ceiling']:>8.3f}"
              f"{b['ceiling'] - b['med']:>+8.3f}")
    print(f"\n  跨几何的下限 p95 跨度：秩 1 = {max(d1) - min(d1):.3f}"
          f"（{min(d1):.3f}–{max(d1):.3f}）；"
          f"秩 2 = {max(d2) - min(d2):.3f}（{min(d2):.3f}–{max(d2):.3f}）")
    print(f"  跨秩的下限中位差（秩 1 − 秩 2）：中位 {np.median(dd):+.3f}，"
          f"范围 {min(dd):+.3f}–{max(dd):+.3f}")
    out["spread"] = {"r1_p95_range": [min(d1), max(d1)],
                     "r2_p95_range": [min(d2), max(d2)],
                     "r1_p95_spread": max(d1) - min(d1),
                     "r2_p95_spread": max(d2) - min(d2),
                     "rank_gap_med": float(np.median(dd)),
                     "rank_gap_range": [min(dd), max(dd)]}

    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("[D] 固定门槛 0.60 落在哪里？")
    print("=" * 104)
    m1 = [byn[nm][1]["med"] for nm in REAL]
    q1 = [byn[nm][1]["p95"] for nm in REAL]
    m2 = [byn[nm][2]["med"] for nm in REAL]
    q2 = [byn[nm][2]["p95"] for nm in REAL]
    print(f"  秩 1 噪声中位 {min(m1):.3f}–{max(m1):.3f}；p95 "
          f"{min(q1):.3f}–{max(q1):.3f}")
    print(f"  秩 2 噪声中位 {min(m2):.3f}–{max(m2):.3f}；p95 "
          f"{min(q2):.3f}–{max(q2):.3f}")
    print(f"  0.60 在秩 1 上：低于全部 {sum(1 for v in m1 if v > 0.60)}"
          f"/{len(m1)} 个几何的**噪声中位** "
          f"=> 若闸跑在秩 1 上，0.60 会放进纯噪声")
    print(f"  0.60 在秩 2 上：高于 {sum(1 for v in q2 if v < 0.60)}"
          f"/{len(q2)} 个几何的**噪声 p95**，低于 "
          f"{sum(1 for v in q2 if v >= 0.60)}/{len(q2)} 个")
    out["threshold_0.60"] = {
        "r1_med_above_060": int(sum(1 for v in m1 if v > 0.60)),
        "r2_p95_below_060": int(sum(1 for v in q2 if v < 0.60)),
        "r2_p95_above_060": int(sum(1 for v in q2 if v >= 0.60)),
        "n": len(REAL)}

    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("[E] gap 由上限还是下限决定？（秩 2，真实几何）")
    print("=" * 104)
    ex = np.array([byn[nm][2]["expl"] for nm in REAL])
    ce = np.array([byn[nm][2]["ceiling"] for nm in REAL])
    fm = np.array([byn[nm][2]["med"] for nm in REAL])
    gp = ce - fm
    print(f"  上限跨度 {ce.min():.3f}–{ce.max():.3f}（跨度 "
          f"{ce.max() - ce.min():.3f}）")
    print(f"  下限跨度 {fm.min():.3f}–{fm.max():.3f}（跨度 "
          f"{fm.max() - fm.min():.3f}）")
    print(f"  gap 与上限的相关 r = {np.corrcoef(gp, ce)[0, 1]:+.3f}；"
          f"与下限 r = {np.corrcoef(gp, fm)[0, 1]:+.3f}；"
          f"上限与秩 2 解释率 r = {np.corrcoef(ce, ex)[0, 1]:+.3f}")
    out["gap_driver"] = {"ceiling_range": [float(ce.min()), float(ce.max())],
                         "floor_range": [float(fm.min()), float(fm.max())],
                         "r_gap_ceiling": float(np.corrcoef(gp, ce)[0, 1]),
                         "r_gap_floor": float(np.corrcoef(gp, fm)[0, 1]),
                         "r_ceiling_expl": float(np.corrcoef(ce, ex)[0, 1])}

    p = os.path.join(G.ROOT, "cache", "gate_floor_geometry.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\n产物:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
