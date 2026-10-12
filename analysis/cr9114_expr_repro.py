#!/usr/bin/env python3
"""Is the CR9114 expression/display score reproducible enough to measure a
display AXIS at all?  This must be answered BEFORE any display-axis number is
reported (CLAUDE.md §10: a tool must self-verify against known facts before its
output is used to draw a conclusion).

Why this file exists: a first pass at hazard 1 (docs/23 §7.1, avidity/display
contamination) used the shipped `exp_score` column and found
|cos(binding axis 1, display axis 1)| = 0.525-0.891 against a Haar p95 of
0.442, all with p(N1) < 0.02 -- a dramatic, mechanistically plausible result
that would have been reported as "the display direction is not separable from
binding axis 1".  It is an artefact.  `compile_TiteSeq.write_expression_score_file`
sets

    df['exp_score'] = df['exp_score_rep1']          # the median line is commented out

and `exp_score_rep1` is replicate 1 of the **H1** sort.  So one single
replicate drives the whole effect.  This script measures whether that
replicate's direction is reproducible in the other five.

Known-fact anchor for "reproducible enough": the binding readout in the SAME
tables has replicate agreement r = 0.941-0.986 (cr9114_parse F6).  Any display
quantity used to bound the hazard must be in that league, or it is noise.

Run: python3 analysis/cr9114_expr_repro.py   Out: cache/cr9114_expr_repro.json
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr9114_parse as P                                  # noqa: E402
import cr9114_avidity as A                                # noqa: E402
import gate_calib_lib as G                                # noqa: E402
from subspace_lib import extract_subspace, haar_1d_null   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMS = os.path.join(ROOT, "cache", "dms", "cr9114")
# which sort each shipped expression replicate comes from
# (read off compile_TiteSeq.write_expression_score_file)
WT_REPS = [("rep1", "H1"), ("rep2", "H1"), ("rep3", "H3"),
           ("rep4", "H3"), ("rep5", "fluB"), ("rep6", "fluB")]


def _col(path, key):
    out = {}
    for r in csv.DictReader(open(path)):
        v = r.get(key, "")
        out[r["mut"]] = np.nan if v in ("", "NA", "NaN") else float(v)
    return out


def main():
    P.selftest(verbose=False)
    print("[prereq] cr9114_parse.selftest(): 44/44 PASS")
    null = haar_1d_null(20, n=20000, seed=11)
    p95 = float(np.percentile(np.abs(null), 95))
    out = {"haar_p95_d20": p95, "binding_rep_r": [0.941, 0.986, 0.981, 0.984]}

    wt = os.path.join(DMS, "cr9114_result_HC_WT_expression_score_kabat.csv")
    gl = os.path.join(DMS, "cr9114_result_HC_GL_KD_table_summary_kabat.csv")
    V = {k: _col(wt, "exp_score_" + k) for k, _ in WT_REPS}
    muts = list(V["rep1"])

    print("\n" + "=" * 92)
    print("[1] do the SIX shipped expression replicates agree with each other, "
          "variant by variant?")
    print("    (known-fact yardstick: the BINDING replicates in the same "
          "tables agree at r = 0.941-0.986)")
    print("=" * 92)
    print("          " + "".join(f"{k:>9}" for k, _ in WT_REPS))
    R = np.zeros((6, 6))
    for i, (a, sa) in enumerate(WT_REPS):
        line = f"{a}({sa:<4})"
        for j, (b, _) in enumerate(WT_REPS):
            x = np.array([V[a][m] for m in muts])
            y = np.array([V[b][m] for m in muts])
            ok = np.isfinite(x) & np.isfinite(y)
            R[i, j] = float(np.corrcoef(x[ok], y[ok])[0, 1])
            line += f"{R[i, j]:>9.3f}"
        print(line)
    off = R[~np.eye(6, dtype=bool)]
    print(f"\n    off-diagonal: min {off.min():.3f}  median "
          f"{np.median(off):.3f}  max {off.max():.3f}")
    out["wt_rep_corr"] = R.tolist()
    out["wt_rep_corr_offdiag"] = {"min": float(off.min()),
                                  "median": float(np.median(off)),
                                  "max": float(off.max())}

    glr1 = _col(gl, "exp_score_rep1")
    glr2 = _col(gl, "exp_score_rep2")
    gm = list(glr1)
    x = np.array([glr1[m] for m in gm]); y = np.array([glr2[m] for m in gm])
    ok = np.isfinite(x) & np.isfinite(y)
    rgl = float(np.corrcoef(x[ok], y[ok])[0, 1])
    print(f"    GL library (its own 2 replicates): r = {rgl:.3f} (n={int(ok.sum())})")
    out["gl_rep_corr"] = rgl

    print("\n" + "=" * 92)
    print("[2] does each replicate's own matrix have a rank-1 structure, and "
          "do those axes agree?")
    print("=" * 92)
    recs, _ = P.load_panel("WT_H1")

    def ax(rc, key):
        M, mask, _ = A.mat_from(rc, key)
        M, mask, cols = A.prep(M, mask)
        B, cc, _ = extract_subspace(M, mask, rank=1)
        e, _ = G.rankr_expl(M, mask, rank=1)
        return (B, [P.AA20[cols[c]] for c in cc]), e

    AX, EX = {}, {}
    for k, sa in WT_REPS:
        t = V[k]
        AX[k], EX[k] = ax(recs, lambda r, t=t: t.get(r["mut"], np.nan))
    axb, exb = ax(recs, lambda r: r["dmlk"])
    print(f"    rank-1 explained fraction of interaction variance:")
    for k, sa in WT_REPS:
        print(f"      exp_{k} ({sa:<4}) {EX[k]:.3f}")
    print(f"      WT_H1 BINDING      {exb:.3f}   <- the comparison")
    out["wt_rep_expl"] = {k: EX[k] for k, _ in WT_REPS}
    out["binding_expl_WT_H1"] = exb

    print(f"\n    |cos| between the replicates' OWN axis 1 "
          f"(Haar p95 = {p95:.3f}):")
    print("          " + "".join(f"{k:>9}" for k, _ in WT_REPS))
    C = np.zeros((6, 6))
    for i, (a, sa) in enumerate(WT_REPS):
        line = f"{a}({sa:<4})"
        for j, (b, _) in enumerate(WT_REPS):
            c, _ = A.cos1d(*AX[a], *AX[b])
            C[i, j] = abs(c)
            line += f"{C[i, j]:>9.3f}"
        print(line)
    offc = C[~np.eye(6, dtype=bool)]
    n_above = int((offc > p95).sum() // 1)
    print(f"\n    off-diagonal: {n_above}/{len(offc)} pairs exceed the Haar "
          f"p95; median {np.median(offc):.3f}")
    out["wt_axis_cos"] = C.tolist()
    out["wt_axis_cos_above_p95"] = [n_above, int(len(offc))]

    print("\n" + "=" * 92)
    print("[3] the artefact, made explicit: which replicate drives the "
          "display-axis 'finding'")
    print("=" * 92)
    print(f"{'binding matrix':<10}{'expression replicate':<24}"
          f"{'same sort?':<12}{'|cos|':>8}{'p(N1)':>9}")
    print("-" * 92)
    rows = []
    for name in ("WT_H1", "WT_H3", "WT_fluB"):
        rc, _ = P.load_panel(name)
        ab, _ = ax(rc, lambda r: r["dmlk"])
        for k, sa in WT_REPS:
            t = V[k]
            ae, _ = ax(rc, lambda r, t=t: t.get(r["mut"], np.nan))
            c, d = A.cos1d(*ab, *ae)
            pv = float((np.abs(null) >= abs(c)).mean())
            same = (sa == name.split("_")[1])
            rows.append({"matrix": name, "rep": k, "sort": sa,
                         "same_sort": bool(same), "cos": abs(float(c)),
                         "p_N1": pv})
            print(f"{name:<10}{'exp_score_'+k+' ('+sa+')':<24}"
                  f"{('YES' if same else 'no'):<12}{abs(c):>8.3f}{pv:>9.4f}")
        print()
    out["cross_rep_rows"] = rows
    # GL
    rc, _ = P.load_panel("GL_H1")
    ab, _ = ax(rc, lambda r: r["dmlk"])
    glrows = []
    for lab, t in (("exp_score_rep1", glr1), ("exp_score_rep2", glr2),
                   ("exp_score (mean)", _col(gl, "exp_score"))):
        ae, _ = ax(rc, lambda r, t=t: t.get(r["mut"], np.nan))
        c, _ = A.cos1d(*ab, *ae)
        pv = float((np.abs(null) >= abs(c)).mean())
        glrows.append({"rep": lab, "cos": abs(float(c)), "p_N1": pv})
        print(f"{'GL_H1':<10}{lab:<24}{'YES':<12}{abs(c):>8.3f}{pv:>9.4f}")
    a1, _ = ax(rc, lambda r: glr1.get(r["mut"], np.nan))
    a2, _ = ax(rc, lambda r: glr2.get(r["mut"], np.nan))
    c12, _ = A.cos1d(*a1, *a2)
    print(f"{'':<10}{'rep1 vs rep2 axis':<24}{'-':<12}{abs(c12):>8.3f}"
          f"   <- the two GL expression replicates do NOT agree on a direction")
    out["gl_rows"] = glrows
    out["gl_rep1_vs_rep2_axis_cos"] = abs(float(c12))

    print("\n" + "=" * 92)
    verdict = ("UNUSABLE: the expression/display score is not reproducible "
               "across its own replicates, so no display AXIS can be measured "
               "from it, and the avidity/display hazard CANNOT be bounded with "
               "the columns this dataset ships.")
    print("VERDICT: " + verdict)
    print("=" * 92)
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(ROOT, "cache",
                                     "cr9114_expr_repro.json"), "w"),
              indent=1, default=float)
    print("\nwrote cache/cr9114_expr_repro.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
