#!/usr/bin/env python3
"""docs/43 Step 3：每个矩阵的噪声（σ_meas / σ_fit / 迁移或文献锚点）与 s/σ 的两个端点。

先决（任一不过即中止）：预先登记锁；rank_core.selfcheck；噪声估计器自检；
15 个 animalACE2 矩阵的 3 种子 trim95 曲线与 cache/censor_recheck.json 逐位一致（已知事实）；
每个几何的自检（cso 正向匹配线性、wild 噪声保真）。
产物：cache/overshoot_noise.json、cache/overshoot_geoms.pkl（几何缓存，可删）。
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overshoot_lib as OL                       # noqa: E402
import rank_core as RC                           # noqa: E402

ROOT = OL.ROOT


def main():
    t0 = time.time()
    R = {"prereg": OL.verify_lock()}
    RC.selfcheck(verbose=False)
    print("  rank_core.selfcheck 全过  OK", flush=True)
    R["selftest_noise"] = OL.selfcheck_noise_estimator()
    R["selftest_corr"] = OL.selfcheck_corrections()

    names = OL.scope_names()
    geoms = {}
    for nm in names:
        d = OL.pick(nm)
        geoms[nm] = OL.Geometry(d)
        print(f"  几何 {nm}: {geoms[nm].nr}x{geoms[nm].nc}, n={geoms[nm].n}, "
              f"cso={geoms[nm].cso:.3f}, rep={geoms[nm].rep_kind}", flush=True)

    # ---- 复现性（已知事实）：animalACE2 15 个矩阵的曲线 == 缓存
    cache = json.load(open(os.path.join(ROOT, "cache", "censor_recheck.json")))
    repro = {}
    for u in cache["units"]:
        if u["dataset"] != "animalACE2":
            continue
        nm = f"animalACE2/{u['bg']}/{u['lig'].replace('bind_', '')}"
        g = geoms[nm]
        cs = OL.curves_for(g.M, g.mask)
        dmax = 0.0
        for c_new, c_old in zip(cs, u["trim_curves"]):
            for r in range(4):
                dmax = max(dmax, abs(c_new[r] - c_old[str(r)]))
        repro[nm] = dmax
        assert dmax < 1e-9, f"{nm} 曲线与缓存不一致 max|Δ|={dmax:.2e}"
    R["repro_max_abs_diff"] = repro
    print(f"  复现：15 个 animalACE2 曲线与 censor_recheck 缓存逐位一致（max|Δ|={max(repro.values()):.1e}）  OK",
          flush=True)

    # ---- d(Eta/deer) 的 σ_单 对 docs/12 的 rep_noise
    u = [x for x in cache["units"] if x["dataset"] == "animalACE2" and x["bg"] == "Eta"
         and x["lig"] == "bind_deer"][0]
    s1 = geoms["animalACE2/Eta/deer"].sig["sigma_single"]
    rel = abs(s1 - u["rep_noise"]) / u["rep_noise"]
    assert rel < 0.05, f"Eta/deer σ_单 {s1:.4f} 与 censor_recheck.rep_noise {u['rep_noise']:.4f} 差 {rel:.1%}"
    print(f"  N4 Eta/deer σ_单 {s1:.4f} vs censor_recheck.rep_noise {u['rep_noise']:.4f}（{rel:.1%}）  OK",
          flush=True)

    # ---- 各矩阵的几何自检
    R["selftest_geom"] = []
    for nm in names:
        R["selftest_geom"] += OL.selfcheck_geometry(geoms[nm])

    # ---- σ 下端
    sib = {}
    for b in OL.S1_BACKBONES:
        sib[b] = float(np.median([geoms[f"animalACE2/{b}/{l}"].sig["sigma_cell"] for l in OL.S1_REP_LIGS]))
    rows = {}
    for nm in names:
        g = geoms[nm]
        if g.rep_kind == "measured":
            lo, tag = g.sig["sigma_cell"], ""
        elif nm.endswith("human_monomer"):
            lo, tag = sib[nm.split("/")[1]], "T"
        else:
            lo, tag = OL.SIGMA_LIT_SAME_SOURCE, "L"
        row = {"tag": tag, "n": g.n, "nr": g.nr, "cso": g.cso, "sigma_lo": lo,
               "sigma_fit1": g.fit[1]["sigma"], "sigma_fit2": g.fit[2]["sigma"],
               "sig": g.sig, "clip_range": [g.lo, g.hi]}
        for k in (1, 2):
            p = g.path(k, lo)
            row[f"path_k{k}"] = p
        a1, b1 = g.ab(1, "real")
        a2, b2 = g.ab(2, "real")
        row["ab"] = {"k1_real": [a1, b1], "k2_real": [a2, b2],
                     "k1_rand": list(g.ab(1, "rand")), "k2_rand": list(g.ab(2, "rand"))}
        row["sigma0_null"] = g.sigma0()
        # 真实曲线（3 种子均值）
        det, mc = OL.detect(g.M, g.mask)
        row["real_det"], row["real_curve"] = det, mc
        rows[nm] = row
    R["matrices"] = rows
    R["sibling_sigma_median"] = sib

    print(f"\n{'矩阵':<40}{'tag':>3}{'n':>6}{'cso':>7}{'σ_lo':>8}{'σ_fit1':>8}{'σ_fit2':>8}"
          f"{'meas²/fit1²':>12}{'s/σ@lo':>8}{'s/σ@fit1':>9}{'真实秩':>6}")
    for nm in names:
        r = rows[nm]
        p = r["path_k1"]
        snr_lo = p[0]["snr"] if p else float("nan")
        snr_hi = p[-1]["snr"] if p else float("nan")
        frac = (r["sigma_lo"] / r["sigma_fit1"]) ** 2
        print(f"{nm:<40}{r['tag']:>3}{r['n']:>6}{r['cso']:>7.3f}{r['sigma_lo']:>8.3f}{r['sigma_fit1']:>8.3f}"
              f"{r['sigma_fit2']:>8.3f}{frac:>12.3f}{(snr_lo if snr_lo else float('nan')):>8.2f}"
              f"{(snr_hi if snr_hi else float('nan')):>9.2f}{r['real_det']:>6}")
    R["seconds"] = time.time() - t0
    json.dump(R, open(os.path.join(ROOT, "cache", "overshoot_noise.json"), "w"), indent=1,
              default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o))
    pickle.dump(geoms, open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "wb"))
    print(f"\n用时 {R['seconds']:.0f}s  -> cache/overshoot_noise.json", flush=True)


if __name__ == "__main__":
    main()
