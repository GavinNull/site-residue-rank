#!/usr/bin/env python3
"""Step 3 -- the censoring gate for the CR9114 V_H DMS matrices.

Order (the human's explicit condition): measure censoring FIRST, let the data
into the evidence set only afterwards.  Nothing downstream of this file may be
run if the gate is not passed.

Three quantities are reported and MUST NOT be mixed (censor_density.py (A1)):

  q_G3   Gen-3 density-detector censoring fraction, on RAW values
         (docs/22: feeding masked values collapsed one matrix 83.1% -> 28.7%,
         a 57.7pp silent loss; so `raw` is used, never M[mask]).
         Mechanism it is built for: flatten-to-limit + noise => a narrow,
         roughly symmetric pile at the limit.
  q_atom exact-tie mass at the sample minimum (Gen-1 caliber), reported
         separately because Gen-3 falls back to it when its mixture branch
         misses.
  f_out  fraction of values OUTSIDE the titration window actually used,
         [0.033 nM, 100 nM] -> -log10 Kd in [7.000, 10.481].  This is a
         DIFFERENT quantity from q_G3: the Tite-seq nls is BOUNDED
         (Kd in [5e-14, 1e-5]) but does not clip to the bound -- it returns an
         interior *extrapolated* value.  So the mechanism here is
         "flatten-to-limit" only for the few variants that drive the optimiser
         all the way onto the bound; everything else between the titration
         floor and the hard bound is an unbounded extrapolation with no pile.
         Gen-3 is correct to report ~0 for that, and that ~0 carries NO
         information about reliability -- exactly the situation (A1) warns
         about.  f_out is the caliber that does.

Also runs the §10.3 recovery diagnostic at THESE matrices' own geometry
(n ~ 2400-2500, 120 x 20) before any censoring number is trusted.

Run: python3 analysis/cr9114_censor.py
Out: cache/cr9114_censor.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr9114_parse as P                                      # noqa: E402
from censor_density import (censor_fraction_density,           # noqa: E402
                            censor_fraction_density_ci)
from subspace_lib import censor_fraction                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["GL_H1", "WT_H1", "WT_H3", "WT_fluB"]


def sharpness(v, w=0.25):
    """docs/22 §4.2 caliber: exact-tie mass / mass within min+w.
    1.0 = pure atom; ~0.5 = atom and smear in equal parts (the P3 blind zone).
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    lo = v.min()
    win = float((v <= lo + w).mean())
    tie = float((v <= lo + 1e-6).mean())
    return (tie / win if win > 0 else 1.0), tie, win


def implant(v, q_a, q_s, sigma, rng):
    """Identical mechanism to censor_gen3_atom_power.implant (docs/22 §4.2)."""
    v = np.sort(np.asarray(v, float).copy())
    n = len(v)
    na, ns = int(round(q_a * n)), int(round(q_s * n))
    c = v[na + ns] if na + ns < n else v[-1]
    out = v.copy()
    if na + ns > 0:
        out[:na + ns] = c
    if ns > 0 and sigma > 0:
        out[na:na + ns] = c + rng.normal(0, sigma, ns)
    return out


def main():
    print("=" * 104)
    print("STEP 3  CR9114 V_H DMS -- the censoring gate (run BEFORE any "
          "statistics; CLAUDE.md §10 self-check first)")
    print("=" * 104)
    print("\n[0] mandatory parser self-check against KNOWN FACTS")
    P.selftest(verbose=False)
    print("    analysis/cr9114_parse.selftest(): 43/43 PASS "
          "(full listing: python3 analysis/cr9114_parse.py)")

    D = P.load_all(value="mlk")          # physical scale: -log10 K_D,app
    out = {"titration_window_mlk": [P.TRUST_LO, P.TRUST_HI],
           "nls_bound_mlk": [P.MLK_LO, P.MLK_HI], "matrices": {}}

    # ------------------------------------------------------- the three numbers
    print("\n[1] the three quantities, per matrix (Gen-3 on RAW values)")
    print(f"\n{'matrix':<9}{'n':>6}{'sites':>7}{'q_G3':>9}{'10-90% sub':>18}"
          f"{'q_atom':>9}{'Gen-2':>8}{'sharp':>8}{'f_out':>9}"
          f"{'at bound':>10}  Gen-3 gate")
    print("-" * 104)
    for name in ORDER:
        d = D[name]
        raw = d["raw"]
        r = censor_fraction_density_ci(raw)
        g2 = censor_fraction(raw)
        sh, tie, win = sharpness(raw)
        f_out = float(((raw < P.TRUST_LO) | (raw > P.TRUST_HI)).mean())
        at_b = float((np.abs(raw - P.MLK_LO) < 1e-3).mean())
        info = r["info"]
        print(f"{name:<9}{len(raw):>6}{len(d['sites']):>7}{r['q']:>8.1%} "
              f"{'['+format(r['lo'],'.1%')+', '+format(r['hi'],'.1%')+']':>17}"
              f"{tie:>9.2%}{g2:>8.1%}{sh:>8.2f}{f_out:>9.1%}{at_b:>10.2%}"
              f"  {info['gate']}")
        out["matrices"][name] = {
            "n": int(len(raw)), "n_sites": len(d["sites"]),
            "q_G3": r["q"], "q_G3_lo": r["lo"], "q_G3_hi": r["hi"],
            "q_G3_sub": r["sub"], "q_atom": tie, "gen2": float(g2),
            "sharpness": sh, "tie_mass": tie, "window_mass": win,
            "f_outside_titration": f_out,
            "f_below_titration_floor": float((raw < P.TRUST_LO).mean()),
            "f_above_titration_ceiling": float((raw > P.TRUST_HI).mean()),
            "f_at_nls_bound": at_b,
            "gen3_gate": info["gate"], "gen3_why": info["why"],
            "gen3_sigma_hat": info["sigma_hat"],
            "gen3_sigma_over_s": info["sigma_over_s"],
            "range": [float(raw.min()), float(raw.max())],
            "silent_mean": float(np.mean(
                [x["mlk"] for x in d["recs"]
                 if not x["bad"] and x["cls"] == "silent"
                 and np.isfinite(x["mlk"])])),
        }
    for name in ORDER:
        m = out["matrices"][name]
        print(f"    {name:<9} Gen-3 why: {m['gen3_why'] or '-'}")

    # ------------------------------------------- shape of the weak-side mass
    print("\n[2] where the weak-side mass actually sits (is there a pile?)")
    print(f"\n{'matrix':<9}{'min':>8}{'range':>16}" +
          "".join(f"{'<='+format(P.MLK_LO+g,'.2f'):>9}" for g in
                  (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)))
    print("-" * 104)
    for name in ORDER:
        raw = D[name]["raw"]
        cells = "".join(f"{float((raw <= P.MLK_LO + g).mean()):>8.2%} "
                        for g in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0))
        print(f"{name:<9}{raw.min():>8.3f}"
              f"{'['+format(raw.min(),'.2f')+','+format(raw.max(),'.2f')+']':>16}"
              f"{cells}")
        out["matrices"][name]["cdf_above_bound"] = {
            str(g): float((raw <= P.MLK_LO + g).mean())
            for g in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)}

    # ------------------------------------------------- §10.3 recovery diagnostic
    print("\n[3] §10.3 recovery diagnostic AT THESE MATRICES' OWN GEOMETRY")
    print("    Substrate must be a matrix Gen-3 itself calls clean, with this")
    print("    panel's own n and shape.  Candidate = the CR9114 matrix with the")
    print("    smallest f_out (WT_H1), restricted to values inside the")
    print("    titration window so the substrate carries no censoring by")
    print("    construction.  Verified clean before use (else the diagnostic")
    print("    is void).")
    base = D["WT_H1"]["raw"]
    base = base[(base >= P.TRUST_LO) & (base <= P.TRUST_HI)]
    b3 = censor_fraction_density(base)
    print(f"\n    substrate = WT_H1 within-window values, n = {len(base)}, "
          f"Gen-1 = {float((base <= base.min()+1e-6).mean()):.2%}, "
          f"Gen-2 = {censor_fraction(base):.1%}, Gen-3 = {b3:.1%}")
    if b3 > 0.03:
        print("    ** substrate is itself read as censored -- diagnostic VOID **")
        out["recovery_void"] = True
    else:
        out["recovery_void"] = False
    out["recovery_substrate"] = {"n": int(len(base)), "gen3": float(b3)}

    # ---- the implant width MUST be specified as sigma / s_up, not as an
    #      absolute sigma.  Gen-3 has no absolute scale: its existence gate is
    #      sigma_hat <= GAMMA * s_up, and its assumption (A3) is sigma << s_up.
    #      s_up differs 3.6x between docs/22's substrate (animalACE2/BA2,
    #      s_up = 0.4085) and this panel (WT_H1 in-window, s_up = 0.1146), so
    #      implanting docs/22's absolute sigma = 0.07 here is a sigma/s_up of
    #      0.61 -- past the point where a "pile" is as wide as the distribution
    #      and the censoring fraction is not even defined.  A first run of this
    #      diagnostic made exactly that mistake and read 0/9 + 0/18; it was my
    #      error, not a property of the data.  Logged here because CLAUDE.md
    #      §10 requires the tool's own error and the object's error to be
    #      distinguished and recorded separately.
    #      docs/22's blind-zone band sigma in [0.07, 0.20] is, in these units,
    #      sigma/s_up in [0.17, 0.49].  That band is scanned.
    from censor_density import upper_scale as _us
    s_up_base = _us(base)
    RELS = (0.05, 0.10, 0.17, 0.30, 0.49)
    out["recovery_s_up"] = float(s_up_base)
    out["recovery_rels"] = list(RELS)
    print(f"\n    substrate s_up = {s_up_base:.4f}; docs/22 substrate s_up = "
          f"0.4085, so their sigma in [0.07,0.20] == sigma/s_up in "
          f"[0.17,0.49]; sigma below is REL * s_up")
    cases = []
    for q in (0.10, 0.20, 0.40, 0.75):
        cases.append(("P1", q, 0.0, 0.0, q))
    for q in (0.20, 0.40, 0.75):
        for rel in RELS:
            cases.append(("P2", 0.0, q, rel, q))
    for q_a, q_s in ((0.10, 0.10), (0.20, 0.20), (0.20, 0.55),
                     (0.02, 0.38), (0.02, 0.74), (0.002, 0.086)):
        for rel in RELS:
            cases.append(("P3", q_a, q_s, rel, q_a + q_s))
    reps = 6
    print(f"\n{'cfg':<5}{'q_atom':>8}{'q_smear':>9}{'s/s_up':>8}{'sigma':>8}"
          f"{'truth':>8}{'Gen-3 mean':>12}{'Gen-3 span':>20}{'Gen-2':>8}"
          f"{'criterion':>26}")
    print("-" * 110)
    rows = []
    for tag, q_a, q_s, rel, truth in cases:
        sg = rel * s_up_base
        vals, g2s = [], []
        for k in range(reps):
            rng = np.random.default_rng(1000 + k)
            w = implant(base, q_a, q_s, sg, rng)
            vals.append(censor_fraction_density(w))
            g2s.append(censor_fraction(w))
        vals = np.array(vals)
        mu = float(vals.mean())
        if tag == "P1":
            good, crit = abs(mu - truth) <= 0.02, "|est-truth| <= 2pp"
        elif tag == "P2":
            good = abs(mu - truth) <= 0.15 and mu >= truth / 2
            crit = "|est-truth|<=15pp and >=truth/2"
        else:
            good = mu >= q_a + 0.5 * q_s
            crit = f"est >= q_a+0.5*q_s = {q_a + 0.5*q_s:.1%}"
        rows.append({"case": tag, "q_atom": q_a, "q_smear": q_s,
                     "sigma_rel": rel, "sigma_abs": float(sg),
                     "truth": truth, "g3_mean": mu,
                     "g3_min": float(vals.min()), "g3_max": float(vals.max()),
                     "g2_mean": float(np.mean(g2s)), "pass": bool(good),
                     "criterion": crit})
        print(f"{tag:<5}{q_a:>8.1%}{q_s:>9.1%}{rel:>8.2f}{sg:>8.3f}"
              f"{truth:>8.1%}{mu:>12.1%}"
              f"{'['+format(vals.min(),'.1%')+', '+format(vals.max(),'.1%')+']':>20}"
              f"{np.mean(g2s):>8.1%}  {crit:<24}{'OK' if good else '<< FAIL'}")
    out["recovery_rows"] = rows
    for tag in ("P1", "P2", "P3"):
        g = [r for r in rows if r["case"] == tag]
        print(f"    [{tag}] {sum(r['pass'] for r in g)}/{len(g)} pass")
        out[f"recovery_{tag}"] = [sum(r['pass'] for r in g), len(g)]
    only_atom = [r for r in rows if r["case"] == "P3"
                 and abs(r["g3_mean"] - r["q_atom"]) <= 0.02]
    p3 = [r for r in rows if r["case"] == "P3"]
    print(f"    [P3] cells where Gen-3 est ~= q_atom (sees ONLY the atom, "
          f"<=2pp): {len(only_atom)}/{len(p3)}")
    out["recovery_only_atom"] = [len(only_atom), len(p3)]

    json.dump(out, open(os.path.join(ROOT, "cache", "cr9114_censor.json"), "w"),
              indent=1, default=float)
    print("\nwrote cache/cr9114_censor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
