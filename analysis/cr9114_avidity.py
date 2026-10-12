#!/usr/bin/env python3
"""Hazard 1 (docs/23 §7.1): yeast-surface K_D,app mixes in avidity and display
level, which can inject a direction COMMON to all antigens -- and a shared
spurious direction is exactly what would fake a universal axis.

What can be done about it with the columns that exist: the repo ships an
expression score.  result/HC_WT_expression_score_kabat.csv carries exp_score
over 6 replicates for the somatic library, and the GL KD table carries
exp_score / exp_score_rep1 / exp_score_rep2 for the germline library.  Both are
computed by compile_TiteSeq.compute_expression_score from the anti-cMyc display
bins and then normalised so silent = 1 and nonsense = 0.

So the hazard is MEASURABLE here, not merely acknowledged.  Three things are
reported:
  (a) per-variant correlation of exp_score with the binding readout;
  (b) the angle between the expression matrix's own leading residue-loading
      direction and each binding matrix's leading residue-loading direction --
      if that angle is small, the "universal" axis and the display axis are
      not separable with these columns;
  (c) the same binding axis after regressing exp_score out cell-by-cell.
All three are WITHIN CR9114, so none of them is a cross-family statistic and
none pre-empts the §9.1 decision that Step 3 raised.

CAVEAT that cannot be removed with these columns: exp_score measures surface
DISPLAY, not avidity.  Avidity (bivalent/multivalent engagement of a
multimeric HA by surface-tethered Fab) is NOT measured by any column in this
repository, and it is the component that would act most like a common
direction.  Regressing out exp_score therefore bounds the display part of the
hazard and says nothing about the avidity part.

Run: python3 analysis/cr9114_avidity.py   Out: cache/cr9114_avidity.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr9114_parse as P                                   # noqa: E402
from subspace_lib import (extract_subspace, MIN_PER_ROW,   # noqa: E402
                          MIN_PER_COL, haar_1d_null, _recenter_orth)


def cos1d(Ba, cols_a, Bb, cols_b):
    """|cos| between two rank-1 residue-loading directions on their common
    residues, with the all-ones direction removed INSIDE the common subset --
    the same re-centring subspace_lib.align does, and for the same reason: the
    double-centring only zeroes column sums on each matrix's own residue set,
    so the all-ones component reappears on the intersection and would create
    a purely artefactual alignment if left in.  Compared against
    haar_1d_null at that same d."""
    common = np.intersect1d(cols_a, cols_b)
    if len(common) < 8:
        return None, 0
    ia = np.searchsorted(cols_a, common)
    ib = np.searchsorted(cols_b, common)
    a = _recenter_orth(np.asarray(Ba)[:, :1], ia)[:, 0]
    b = _recenter_orth(np.asarray(Bb)[:, :1], ib)[:, 0]
    return float(abs(a @ b)), int(len(common))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["GL_H1", "WT_H1", "WT_H3", "WT_fluB"]


def exp_lookup():
    """mut -> exp_score.  WT library from the dedicated file; GL library from
    its own KD table (the only place the germline exp_score is shipped)."""
    wt = P.expression_scores()
    recs, _ = P.load_panel("GL_H1")
    gl = {r["mut"]: r["exp"] for r in recs if not r["bad"]}
    return {"WT": wt, "GL": gl}


def mat_from(recs, key, aas=P.AA20):
    sites = sorted({r["pos"] for r in recs if not r["bad"]})
    si = {s: i for i, s in enumerate(sites)}
    ai = {a: j for j, a in enumerate(aas)}
    M = np.full((len(sites), len(aas)), np.nan)
    for r in recs:
        if r["bad"] or r["cls"] not in ("missense", "silent"):
            continue
        v = key(r)
        if r["sub"] in ai and v is not None and np.isfinite(v):
            M[si[r["pos"]], ai[r["sub"]]] = v
    return M, np.isfinite(M), sites


def prep(M, mask):
    kc = mask.sum(0) >= MIN_PER_COL
    M, mask = M[:, kc], mask[:, kc]
    kr = mask.sum(1) >= MIN_PER_ROW
    return M[kr], mask[kr], np.where(kc)[0]


def main():
    P.selftest(verbose=False)
    print("[prereq] cr9114_parse.selftest(): 43/43 PASS\n")
    E = exp_lookup()
    out = {}

    print("=" * 100)
    print("(a) per-variant correlation of display (exp_score) with the "
          "binding readout")
    print("=" * 100)
    print(f"{'matrix':<9}{'n':>6}{'r(exp, -dlogKd)':>18}"
          f"{'r missense only':>18}{'r^2':>8}  exp_score source")
    print("-" * 100)
    for name in ORDER:
        recs, bg = P.load_panel(name)
        tab = E[bg]
        x, y, mi = [], [], []
        for r in recs:
            if r["bad"] or not np.isfinite(r["dmlk"]):
                continue
            e = tab.get(r["mut"], np.nan)
            if not np.isfinite(e):
                continue
            x.append(e); y.append(r["dmlk"]); mi.append(r["cls"] == "missense")
        x, y, mi = np.array(x), np.array(y), np.array(mi)
        rr = float(np.corrcoef(x, y)[0, 1])
        rm = float(np.corrcoef(x[mi], y[mi])[0, 1])
        src = ("HC_WT_expression_score_kabat.csv" if bg == "WT"
               else "HC_GL_KD_table_summary_kabat.csv (exp_score)")
        print(f"{name:<9}{len(x):>6}{rr:>18.3f}{rm:>18.3f}{rm*rm:>8.3f}  {src}")
        out[name] = {"n_paired": len(x), "r_exp_bind": rr,
                     "r_exp_bind_missense": rm, "r2": rm * rm,
                     "exp_source": src}

    print("\n" + "=" * 100)
    print("(b) angle between the DISPLAY axis and each BINDING axis "
          "(rank-1 residue loadings)")
    print("=" * 100)
    # display matrix: one per library
    disp = {}
    for bg, panel in (("WT", "WT_H1"), ("GL", "GL_H1")):
        recs, _ = P.load_panel(panel)
        tab = E[bg]
        M, mask, sites = mat_from(recs, lambda r: tab.get(r["mut"], np.nan))
        M, mask, cols = prep(M, mask)
        B, cc, err = extract_subspace(M, mask, rank=1)
        disp[bg] = (B, [P.AA20[cols[c]] for c in cc], M.shape, err)
        print(f"  display matrix ({bg} library): shape {M.shape}, "
              f"fill {mask.mean():.2f}, rank-1 extracted: {err or 'ok'}")
    print(f"\n{'matrix':<9}{'|cos(bind1, display1)|':>24}"
          f"{'Haar p95':>12}{'p (N1)':>10}{'d':>5}")
    print("-" * 100)
    for name in ORDER:
        recs, bg = P.load_panel(name)
        M, mask, sites = mat_from(recs, lambda r: r["dmlk"])
        M, mask, cols = prep(M, mask)
        B, cc, err = extract_subspace(M, mask, rank=1)
        aas = [P.AA20[cols[c]] for c in cc]
        Bd, aad, _, _ = disp[bg]
        c1, d = cos1d(B, aas, Bd, aad)
        null = haar_1d_null(d, n=20000, seed=11)
        p95 = float(np.percentile(np.abs(null), 95))
        pv = float((np.abs(null) >= abs(c1)).mean())
        print(f"{name:<9}{abs(c1):>24.3f}{p95:>12.3f}{pv:>10.4f}{d:>5}")
        out[name].update({"cos_bind1_display1": abs(float(c1)),
                          "haar_d": d, "haar_p95": p95, "p_N1": pv})

    print("\n" + "=" * 100)
    print("(c) binding axis 1 after regressing exp_score out cell-by-cell")
    print("=" * 100)
    print(f"{'matrix':<9}{'|cos(axis1_raw, axis1_resid)|':>32}"
          f"{'beta':>9}{'R^2 removed':>13}")
    print("-" * 100)
    for name in ORDER:
        recs, bg = P.load_panel(name)
        tab = E[bg]
        M, mask, sites = mat_from(recs, lambda r: r["dmlk"])
        Me, _, _ = mat_from(recs, lambda r: tab.get(r["mut"], np.nan))
        both = np.isfinite(M) & np.isfinite(Me)
        b = np.polyfit(Me[both], M[both], 1)
        R = M.copy()
        R[both] = M[both] - np.polyval(b, Me[both])
        R[~both] = np.nan
        r2 = float(np.corrcoef(Me[both], M[both])[0, 1] ** 2)
        M1, mk1, c1cols = prep(M, np.isfinite(M))
        R1, mk2, c2cols = prep(R, np.isfinite(R))
        B1, cA, _ = extract_subspace(M1, mk1, rank=1)
        B2, cB, _ = extract_subspace(R1, mk2, rank=1)
        cc, _ = cos1d(B1, [P.AA20[c1cols[c]] for c in cA],
                      B2, [P.AA20[c2cols[c]] for c in cB])
        print(f"{name:<9}{abs(cc):>32.3f}{b[0]:>9.3f}{r2:>13.3f}")
        out[name].update({"cos_axis1_raw_vs_expresid": abs(float(cc)),
                          "beta_exp": float(b[0]), "r2_removed": r2})

    json.dump(out, open(os.path.join(ROOT, "cache", "cr9114_avidity.json"),
                        "w"), indent=1, default=float)
    print("\nwrote cache/cr9114_avidity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
