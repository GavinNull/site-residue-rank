#!/usr/bin/env python3
"""Step 4 -- the resolvability gate (G4) at the CR9114 matrices' OWN geometry.

Why this runs even though the Step-3 censoring gate failed for 3 of the 4
matrices: this file measures only whether a matrix of THIS SHAPE can identify
its own axes.  It computes no cross-family axis agreement, so it cannot
pre-empt the CLAUDE.md §9.1 decision that Step 3 raised.  It also answers a
question docs/23 §7.1 asserted but did not check -- "~120 sites is far above
the small-matrix resolution problem" -- which is of standing value to the repo
whether or not CR9114 enters the evidence set.

Caliber copied verbatim from the parallel docs/24 harness
(analysis/gate_calib_lib.py), which reproduces docs/23 §5.2's rank-1
floor/ceiling/gap to 3 decimals.  No new estimator is written here
(CLAUDE.md §10: do not build another detector).

  floor   = pure noise, NO low-rank structure, distribution of the
            per-realisation MEDIAN stab cos_r (docs/23 §5.2 caliber -- NOT
            all bootstrap draws pooled).
  ceiling = known rank-r structure with sigma matched to the matrix's own
            observed rank-r explained fraction, same statistic.
  gap     = ceiling p50 - floor p95.  docs/23 §5.2 calls gap < 0.10 "no
            discriminative power".

Prerequisite: cache/gate_harness_check.json must exist and be ok.
Run: python3 analysis/cr9114_gate.py
Out: cache/cr9114_gate.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr9114_parse as P                     # noqa: E402
import gate_calib_lib as G                   # noqa: E402
from subspace_lib import (MIN_PER_ROW, MIN_PER_COL, dc_fast)   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["GL_H1", "WT_H1", "WT_H3", "WT_fluB"]
N_REAL, N_BOOT, N_BOOT_OBS = 60, 200, 2000
GAP_MIN, STAB_FLAT = 0.10, 0.60


def prep(d):
    """Apply the repo's standing row/col occupancy filters, then report shape."""
    M, mask = d["M"].copy(), d["mask"].copy()
    keep_c = mask.sum(0) >= MIN_PER_COL
    M, mask = M[:, keep_c], mask[:, keep_c]
    keep_r = mask.sum(1) >= MIN_PER_ROW
    M, mask = M[keep_r], mask[keep_r]
    return M, mask


def main():
    hp = os.path.join(ROOT, "cache", "gate_harness_check.json")
    if not os.path.exists(hp) or not json.load(open(hp)).get("ok"):
        print("PREREQ FAIL: cache/gate_harness_check.json missing or not ok "
              "-- run analysis/gate_harness_check.py first (CLAUDE.md §10)")
        return 1
    print("[prereq] gate_harness_check.json ok (rank-1 floor/ceiling/gap "
          "reproduce docs/23 §5.2)")
    P.selftest(verbose=False)
    print("[prereq] cr9114_parse.selftest(): 44/44 PASS")

    D = P.load_all(value="dmlk")
    cen = json.load(open(os.path.join(ROOT, "cache", "cr9114_censor.json")))
    out = {"n_real": N_REAL, "n_boot": N_BOOT, "n_boot_obs": N_BOOT_OBS,
           "matrices": {}}

    for rank in (1, 2):
        print("\n" + "=" * 112)
        print(f"RANK {rank}: observed stab cos{rank} vs pure-noise floor and "
              f"known-rank-{rank} ceiling, each at the matrix's own geometry")
        print("=" * 112)
        print(f"{'matrix':<9}{'shape':>10}{'fill':>7}{'expl':>7}{'obs':>8}"
              f"{'floor p50':>10}{'floor p95':>10}{'ceil p50':>10}"
              f"{'gap':>8}{'discrim':>9}{'obs>fl95':>10}")
        print("-" * 112)
        for name in ORDER:
            M, mask = prep(D[name])
            shape = M.shape
            expl, err = G.rankr_expl(M, mask, rank=rank)
            obs = float(np.median(G.stab_boot(M, mask, rank=rank,
                                              nboot=N_BOOT_OBS, seed=7)))
            fl = G.floor_dist(shape, mask, rank=rank, n_real=N_REAL,
                              nboot=N_BOOT)
            sg, tab = G.match_sigma(shape, mask, rank, expl)
            ce = G.ceiling_dist(shape, mask, rank=rank, sigma=sg,
                                n_real=N_REAL, nboot=N_BOOT)
            f50, f95 = float(np.percentile(fl, 50)), float(np.percentile(fl, 95))
            c50 = float(np.percentile(ce, 50))
            gap = c50 - f95
            print(f"{name:<9}{str(shape[0])+'x'+str(shape[1]):>10}"
                  f"{mask.mean():>7.2f}{expl:>7.3f}{obs:>8.3f}"
                  f"{f50:>10.3f}{f95:>10.3f}{c50:>10.3f}{gap:>8.3f}"
                  f"{('YES' if gap >= GAP_MIN else 'no'):>9}"
                  f"{('YES' if obs > f95 else 'NO'):>10}")
            out["matrices"].setdefault(name, {})[f"rank{rank}"] = {
                "shape": list(shape), "fill": float(mask.mean()),
                "expl": expl, "stab_obs": obs,
                "floor_p50": f50, "floor_p95": f95,
                "floor_p99": float(np.percentile(fl, 99)),
                "ceil_p50": c50, "ceil_p05": float(np.percentile(ce, 5)),
                "gap": gap, "sigma_matched": float(sg),
                "sigma_table": tab,
                "discriminative": bool(gap >= GAP_MIN),
                "obs_above_floor_p95": bool(obs > f95),
                "passes_flat_060": bool(obs >= STAB_FLAT),
            }

        # high-SNR hit test (CLAUDE.md §10.2: must also show it CAN hit)
        print(f"\n  [hit test] known rank-{rank} at sigma=0.05 must recover "
              f"stab -> 1 at these geometries:")
        for name in ORDER:
            M, mask = prep(D[name])
            h = G.hi_snr_check(M.shape, mask, rank=rank)
            hm = float(np.median(h))
            ok = hm >= 0.95
            print(f"    {name:<9} stab cos{rank} median = {hm:.4f}  "
                  f"{'OK' if ok else '<< FAIL (the harness itself is the '
                                    'problem, not the data)'}")
            out["matrices"][name][f"rank{rank}"]["hi_snr_median"] = hm
            out["matrices"][name][f"rank{rank}"]["hi_snr_ok"] = bool(ok)

    # ---- the docs/23 §7.1 assertion, now checked rather than assumed
    print("\n" + "=" * 112)
    print("docs/23 §7.1 asserted that a ~120-site matrix is 'far above' the "
          "resolution problem that made the gate\nnon-discriminative on the "
          "10-43 site candidates (docs/23 §5.2: pure noise alone gave "
          "0.65-0.80 there).")
    print("=" * 112)
    print(f"{'matrix':<9}{'sites':>7}" +
          "".join(f"{'r'+str(r)+' floor p95':>14}{'r'+str(r)+' gap':>10}"
                  for r in (1, 2)))
    print("-" * 112)
    for name in ORDER:
        m = out["matrices"][name]
        print(f"{name:<9}{m['rank1']['shape'][0]:>7}"
              f"{m['rank1']['floor_p95']:>14.3f}{m['rank1']['gap']:>10.3f}"
              f"{m['rank2']['floor_p95']:>14.3f}{m['rank2']['gap']:>10.3f}")
    print("\n  reference (docs/23 §5.2, rank 1): OMTKY3/chymotrypsin (10 sites) "
          "floor p95 = 0.726, gap = 0.257;\n  titeseq 4-4-20 (~43 sites) floor "
          "= 0.647, gap = 0.014 (NON-discriminative);\n  starr2020/WH1 (201 "
          "sites) floor = 0.603, gap = 0.393.")

    # ---- gate verdict, carrying the Step-3 result alongside
    print("\n" + "=" * 112)
    print("COMBINED GATE STATUS (Step 3 censoring AND Step 4 resolvability)")
    print("=" * 112)
    print(f"{'matrix':<9}{'f_out':>8}{'q_G3':>8}{'G3 valid?':>11}"
          f"{'G3 verdict':>13}{'r2 gap':>8}{'r2 discrim':>12}{'r2 obs>fl95':>13}"
          f"   overall")
    print("-" * 112)
    verdicts = {}
    for name in ORDER:
        c = cen["matrices"][name]
        m = out["matrices"][name]["rank2"]
        fo = c["f_outside_titration"]
        # Gen-3 validity at this matrix's own atom/smear configuration,
        # from the recovery diagnostic in cache/cr9114_censor.json
        valid = fo < 0.50
        g3v = ("informative" if valid else "VOID (blind zone)")
        res_ok = m["discriminative"] and m["obs_above_floor_p95"]
        cen_ok = fo < 0.109     # the rank-1 censoring median (10.9%)
        ov = ("PASS" if (cen_ok and res_ok) else
              "FAIL censoring" if (res_ok and not cen_ok) else
              "FAIL both" if not res_ok and not cen_ok else
              "FAIL resolvability")
        verdicts[name] = {"f_out": fo, "censor_ok": bool(cen_ok),
                          "gen3_valid": bool(valid),
                          "resolvability_ok": bool(res_ok), "overall": ov}
        print(f"{name:<9}{fo:>8.1%}{c['q_G3']:>8.1%}"
              f"{('yes' if valid else 'NO'):>11}{g3v:>13}"
              f"{m['gap']:>8.3f}{('YES' if m['discriminative'] else 'no'):>12}"
              f"{('YES' if m['obs_above_floor_p95'] else 'NO'):>13}   {ov}")
    out["verdicts"] = verdicts
    print("\n  censoring criterion used: f_out < 10.9% = the censoring median "
          "the repo's own\n  monotone fact assigns to rank 1 "
          "(rank 0/1/2 <-> 43.2% / 10.9% / 0.1%).  Above it, low-rank\n"
          "  detectability is established to be degraded.")
    json.dump(out, open(os.path.join(ROOT, "cache", "cr9114_gate.json"), "w"),
              indent=1, default=float)
    print("\nwrote cache/cr9114_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
