#!/usr/bin/env python3
"""docs/43 Step 4 的前提：双向 harness（预先登记 §5）+ SNR 扫描（§6）。

不过则退出码 3，不进入任何真实矩阵上的结论。
H-hit / H-nofool / H-null：四个 binding 几何上，设计点 s/σ≈2.6（rank_core C5 / docs/42 H4）。
H-visible：rank_core.selfcheck（含 C6b）。
产物：cache/overshoot_harness.json
"""
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overshoot_lib as OL                       # noqa: E402
import rank_core as RC                           # noqa: E402

ROOT = OL.ROOT
G4 = ("animalACE2/Eta/deer", "PDCoV/PDCoV/gAPN", "CR9114/WT_H1",
      "SKEMPI/OMTKY3/pooled-4proteases")
SIG_D, S_D = 0.64, float(np.hypot(1.5, 0.75))      # 设计点：s/σ = 2.62
SWEEP = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0)
R = OL.R_REP


def main():
    t0 = time.time()
    out = {"prereg": OL.verify_lock()}
    RC.selfcheck(verbose=False)
    print("  H-visible：rank_core.selfcheck（含 C6b 已知偏高偏差复现）全过  OK", flush=True)
    geoms_all = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    OL.set_geoms({k: geoms_all[k] for k in G4})

    tasks = []
    for nm in G4:
        for k in (0, 1, 2):
            for rep in range(R):
                tasks.append(dict(mat=nm, k=k, mode="rand", s=S_D if k else 0.0, sigma=SIG_D,
                                  noise="gauss", rep=rep, tag=f"harness_k{k}", clip=False))
    # SNR 扫描：σ = 该几何 σ_fit1
    for nm in G4:
        sg = geoms_all[nm].fit[1]["sigma"]
        for rep in range(R):
            tasks.append(dict(mat=nm, k=0, mode="rand", s=0.0, sigma=sg, noise="gauss",
                              rep=rep, tag="sweep_k0", clip=False))
        for snr in SWEEP:
            for k in (1, 2):
                for rep in range(R):
                    tasks.append(dict(mat=nm, k=k, mode="rand", s=snr * sg, sigma=sg,
                                      noise="gauss", rep=rep, tag=f"sweep_k{k}_{snr}", clip=False))
    print(f"  任务数 {len(tasks)}", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(9) as pool:
        res = pool.map(OL.run_task, tasks, chunksize=4)
    out["n_tasks"] = len(tasks)

    def rate(sel, f):
        xs = [f(r["det"]) for r in res if sel(r)]
        k = int(sum(xs))
        return k, len(xs), OL.wilson(k, len(xs))

    H = {}
    for nm in G4:
        H[nm] = {
            "hit_k2_is2": rate(lambda r: r["mat"] == nm and r["tag"] == "harness_k2", lambda d: d == 2),
            "nofool_k1_ge2": rate(lambda r: r["mat"] == nm and r["tag"] == "harness_k1", lambda d: d >= 2),
            "null_k0_is0": rate(lambda r: r["mat"] == nm and r["tag"] == "harness_k0", lambda d: d == 0),
            "k2_dist": [int(sum(1 for r in res if r["mat"] == nm and r["tag"] == "harness_k2" and r["det"] == q)) for q in range(4)],
            "k1_dist": [int(sum(1 for r in res if r["mat"] == nm and r["tag"] == "harness_k1" and r["det"] == q)) for q in range(4)],
        }
    out["harness"] = H
    hit_ok = sum(H[n]["hit_k2_is2"][0] / H[n]["hit_k2_is2"][1] >= 0.70 for n in G4)
    nof_ok = sum(H[n]["nofool_k1_ge2"][0] / H[n]["nofool_k1_ge2"][1] <= 0.20 for n in G4)
    nul_ok = sum(H[n]["null_k0_is0"][0] / H[n]["null_k0_is0"][1] >= 0.90 for n in G4)
    out["gates"] = {"H-hit(>=3/4)": [hit_ok, hit_ok >= 3], "H-nofool(>=3/4)": [nof_ok, nof_ok >= 3],
                    "H-null(4/4)": [nul_ok, nul_ok == 4]}
    print("\n双向 harness（设计点 s/σ=%.2f，σ=%.2f，R=%d）" % (S_D / SIG_D, SIG_D, R))
    print(f"{'几何':<36}{'P(=2|真2)':>12}{'P(≥2|真1)':>12}{'P(=0|真0)':>12}   真2分布(0..3)  真1分布")
    for nm in G4:
        h = H[nm]
        f = lambda t: f"{t[0]}/{t[1]}={t[0]/t[1]:.2f}"
        print(f"{nm:<36}{f(h['hit_k2_is2']):>12}{f(h['nofool_k1_ge2']):>12}{f(h['null_k0_is0']):>12}   "
              f"{h['k2_dist']}  {h['k1_dist']}")
    for k, v in out["gates"].items():
        print(f"  {k}: {v[0]}  -> {'通过' if v[1] else '未通过'}")

    # ---- SNR 扫描
    SW = {}
    print("\nSNR 扫描（σ = 各几何 σ_fit1；列 = 真2→检出分布 / 真1→检出分布，R=%d）" % R)
    for nm in G4:
        SW[nm] = {}
        k0 = [r["det"] for r in res if r["mat"] == nm and r["tag"] == "sweep_k0"]
        SW[nm]["k0_is0"] = [int(sum(d == 0 for d in k0)), len(k0)]
        print(f"  {nm}  (k=0 -> 检出 0 的比例 {SW[nm]['k0_is0'][0]}/{len(k0)})")
        for snr in SWEEP:
            d2 = [r["det"] for r in res if r["mat"] == nm and r["tag"] == f"sweep_k2_{snr}"]
            d1 = [r["det"] for r in res if r["mat"] == nm and r["tag"] == f"sweep_k1_{snr}"]
            c2 = [int(sum(x == q for x in d2)) for q in range(4)]
            c1 = [int(sum(x == q for x in d1)) for q in range(4)]
            SW[nm][str(snr)] = {"k2": c2, "k1": c1}
            h = c2[2] / R
            fr = (c1[2] + c1[3]) / R
            win = " <窗口" if (h >= 0.80 and fr <= 0.20) else ""
            print(f"    s/σ={snr:>5}: 真2 {c2}  真1 {c1}   P(=2|2)={h:.2f} P(≥2|1)={fr:.2f}{win}")
    out["sweep"] = SW
    out["ok"] = all(v[1] for v in out["gates"].values())
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(ROOT, "cache", "overshoot_harness.json"), "w"), indent=1)
    print(f"\nharness 总判定: {'通过' if out['ok'] else '未通过'}  用时 {out['seconds']:.0f}s", flush=True)
    sys.exit(0 if out["ok"] else 3)


if __name__ == "__main__":
    main()
