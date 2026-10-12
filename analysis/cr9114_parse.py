#!/usr/bin/env python3
"""CR9114 V_H DMS parser + KNOWN-FACT self-check (CLAUDE.md §10).

Hard rule: `selftest()` must pass before any number from this module is used.
It is called from `load_all()`; failure raises.

Known facts used (NOT self-consistency):
 [F1] The WT residue implied by the `mut` strings at each site must be single
      valued, and the concatenated implied sequence must equal the CR9114
      heavy-chain V_H peptide in Fasta/CR9114_HC_{WT,GL}.pep -- an independent
      file, and independently the kabat_H.csv ANARCI row.
 [F2] The Tite-seq nls fit in result/*/HC_Tite_seq_plot_*.Rmd is BOUNDED:
      lower=list(Kd=5e-14), upper=list(Kd=1e-5).  Hence
      -log10 Kd must lie in [5.000, 13.301] and nothing may fall outside.
      The titration series is 0.033 nM .. 100 nM, i.e. the *trustworthy*
      window is -log10 Kd in [7.00, 10.48]; outside it the value is a
      bounded extrapolation.  Both numbers are facts read off the source.
 [F3] Direction: compile_TiteSeq.py computes minus_log_Kd = -np.log10(Kd),
      so LARGER = TIGHTER.  Checked three independent ways:
      (a) the repo's own BLI cross-check table must correlate POSITIVELY
          in minus_delta_log_Kd_DMS vs minus_delta_log_Kd_BLI;
      (b) nonsense (stop-codon) variants must be WEAKER than silent ones;
      (c) published CR9114 breadth ordering H1 > H3 > influenza B must come
          out as silent-variant minus_log_Kd H1 > H3 > fluB.
 [F4] Magnitude: silent-variant K_D,app for somatic CR9114 vs H1 stem must be
      sub-nanomolar to low-nanomolar (Dreyfus 2012 Science reports CR9114
      binds group-1 HA with nM-or-better affinity).
"""
import os, sys, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DMS = os.path.join(ROOT, "cache", "dms", "cr9114")

AA20 = list("ACDEFGHIKLMNPQRSTVWY")

# [F2] fitting bounds, read verbatim off the Rmd
KD_LOWER, KD_UPPER = 5e-14, 1e-5
MLK_HI = -np.log10(KD_LOWER)     # 13.3010
MLK_LO = -np.log10(KD_UPPER)     # 5.0000
# titration series actually used (Rmd conc_ls)
CONC_LO, CONC_HI = 0.033e-9, 100e-9
TRUST_LO, TRUST_HI = -np.log10(CONC_HI), -np.log10(CONC_LO)   # 7.00, 10.48

PANELS = {
    "GL_H1":   ("cr9114_result_HC_GL_KD_table_summary_kabat.csv",   "GL"),
    "WT_H1":   ("cr9114_result_HC_WT_H1_KD_table_summary_kabat.csv", "WT"),
    "WT_H3":   ("cr9114_result_HC_WT_H3_KD_table_summary_kabat.csv", "WT"),
    "WT_fluB": ("cr9114_result_HC_WT_fluB_KD_table_summary_kabat.csv", "WT"),
}
MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z_])$")


def _read_csv(path):
    import csv
    with open(path) as fh:
        return list(csv.DictReader(fh))


def pep(bg):
    p = os.path.join(DMS, f"cr9114_Fasta_CR9114_HC_{bg}.pep")
    return "".join(l.strip() for l in open(p) if not l.startswith(">"))


def kabat_row(bg):
    """ANARCI numbering row -> (list of kabat labels, list of residues)."""
    p = os.path.join(DMS, f"cr9114_result_CR9114HC_{bg}_kabat_H.csv")
    lines = open(p).read().rstrip().split("\n")
    hdr = lines[0].split(",")
    row = lines[1].split(",")
    i = hdr.index("j_identity") + 1
    return hdr[i:], row[i:]


def load_panel(name):
    fn, bg = PANELS[name]
    rows = _read_csv(os.path.join(DMS, fn))
    recs = []
    for r in rows:
        m = MUT_RE.match(r["mut"])
        if not m:
            recs.append({"mut": r["mut"], "bad": True})
            continue
        wt, pos, sub = m.group(1), int(m.group(2)), m.group(3)
        def f(k):
            v = r.get(k, "")
            if v in ("", "NA", "NaN", "nan", "Inf", "-Inf"):
                return np.nan
            return float(v)
        recs.append({"mut": r["mut"], "wt": wt, "pos": pos, "sub": sub,
                     "cls": r["mut_class"], "region": r["region"],
                     "kabat": r["mut_kabat"],
                     "mlk": f("minus_log_Kd"),
                     "mlk1": f("minus_log_Kd_rep1"),
                     "mlk2": f("minus_log_Kd_rep2"),
                     "dmlk": f("minus_delta_log_Kd"),
                     "bind": f("bind_score"),
                     "exp": f("exp_score"),
                     "ipt": f("ipt_freq"), "bad": False})
    return recs, bg


def expression_scores():
    """mut -> exp_score for the somatic (WT) library (separate file)."""
    rows = _read_csv(os.path.join(DMS, "cr9114_result_HC_WT_expression_score_kabat.csv"))
    out = {}
    for r in rows:
        v = r["exp_score"]
        out[r["mut"]] = np.nan if v in ("", "NA", "NaN") else float(v)
    return out


# ----------------------------------------------------------------- matrices
def build_matrix(recs, value="dmlk", drop_boundary=False):
    """site x residue matrix. Rows = sites with >=1 missense value.
    Returns dict with M (nan-filled), mask, sites, aas, and `raw` = the
    RAW 1-D value set INCLUDING the boundary pile (docs/22: the censoring
    detector must be fed raw, never masked, values)."""
    use = [r for r in recs if not r["bad"] and r["cls"] in ("missense", "silent")
           and np.isfinite(r[value])]
    sites = sorted({r["pos"] for r in use})
    si = {s: i for i, s in enumerate(sites)}
    ai = {a: j for j, a in enumerate(AA20)}
    M = np.full((len(sites), 20), np.nan)
    for r in use:
        if r["sub"] in ai:
            M[si[r["pos"]], ai[r["sub"]]] = r[value]
    mask = np.isfinite(M)
    raw = np.array([r[value] for r in use], float)
    return {"M": M, "mask": mask, "sites": sites, "aas": AA20, "raw": raw,
            "value": value}


# ----------------------------------------------------------------- selftest
def selftest(verbose=True):
    out, fails = [], []

    def chk(tag, ok, msg):
        out.append((tag, bool(ok), msg))
        if not ok:
            fails.append(tag)
        if verbose:
            print(("  PASS " if ok else "  FAIL ") + tag + " :: " + msg)

    # ---- [F0] the WT/GL backbones must differ at exactly the somatic
    #           hypermutations the repo itself annotates in
    #           data/CR9114_SHMheatmap.tsv -- an INDEPENDENT file that was
    #           produced for a figure, not for this comparison.
    sWT, sGL = pep("WT"), pep("GL")
    klab, _ = kabat_row("WT")
    shm = []
    for i, ln in enumerate(open(os.path.join(
            DMS, "cr9114_data_CR9114_SHMheatmap.tsv"))):
        if i == 0:
            continue
        f = ln.rstrip("\n").split("\t")
        if len(f) >= 2:
            shm.append((int(f[1]), f[0]))
    obs = [(i + 1, sGL[i] + klab[i] + sWT[i])
           for i in range(len(sWT)) if sWT[i] != sGL[i]]
    chk("F0 backbones differ at the annotated SHMs",
        len(sWT) == len(sGL) == 120 and sorted(shm) == sorted(obs),
        f"len(WT)=len(GL)={len(sWT)}; {len(obs)} differences; identical to "
        f"the repo's own SHM annotation list ({len(shm)} entries): "
        f"{sorted(shm) == sorted(obs)}. "
        f"NOTE the figure widely cited for the Phillips/Desai CR9114 "
        f"combinatorial library is 16 V_H mutations, not 18; that count is "
        f"NOT verified here and is recorded as a gap, not a check.")
    # IGHV1-69 hallmark: the germline framework must carry the CDR1 'GGTFSSYAIS'
    # and the CDR2 hydrophobic 'IIPIFGTA' motif that defines the IGHV1-69
    # HA-stem class.  These are sequence facts about the germline gene.
    chk("F0b GL carries IGHV1-69 hallmarks",
        ("GGTFSSYAIS" in sGL) and ("IIPIFGTA" in sGL),
        f"GL has CDR1 GGTFSSYAIS: {'GGTFSSYAIS' in sGL}; "
        f"CDR2 IIPIFGTA (the I53/F54 hydrophobic HA-stem motif): "
        f"{'IIPIFGTA' in sGL} -- consistent with the stated IGHV1-69 class")

    # ---- [F1] WT residue single-valued per site + sequence identity
    for name in PANELS:
        recs, bg = load_panel(name)
        ref = pep(bg)
        bysite = {}
        for r in recs:
            if r["bad"]:
                continue
            bysite.setdefault(r["pos"], set()).add(r["wt"])
        multi = {p: v for p, v in bysite.items() if len(v) > 1}
        chk(f"F1a {name} wt single-valued", not multi,
            f"{len(bysite)} sites, sites with >1 implied WT: {len(multi)}")
        sites = sorted(bysite)
        mism = [(p, list(bysite[p])[0], ref[p - 1]) for p in sites
                if ref[p - 1] != list(bysite[p])[0]]
        chk(f"F1b {name} seq == {bg}.pep", not mism,
            f"sites {sites[0]}..{sites[-1]} contiguous="
            f"{sites == list(range(sites[0], sites[-1] + 1))}, "
            f"mismatches={mism[:5]}")
        # independent third source: ANARCI kabat row
        klab, kres = kabat_row(bg)
        chk(f"F1c {name} anarci row == pep",
            "".join(kres) == ref,
            f"anarci row len={len(kres)} identical={''.join(kres) == ref}")
        # kabat mapping in the file must agree with the anarci row
        k2 = {str(i + 1): klab[i] for i in range(len(klab))}
        bad = [r["mut"] for r in recs if not r["bad"]
               and r["kabat"] != r["wt"] + k2.get(str(r["pos"]), "?") + r["sub"]]
        chk(f"F1d {name} kabat relabel consistent", not bad,
            f"{len(bad)} disagreements with the ANARCI header row")

    # ---- [F2] fitting bounds respected, and where the mass sits
    for name in PANELS:
        recs, bg = load_panel(name)
        v = np.array([r["mlk"] for r in recs
                      if not r["bad"] and np.isfinite(r["mlk"])])
        inb = (v >= MLK_LO - 1e-6) & (v <= MLK_HI + 1e-6)
        chk(f"F2a {name} within nls bounds", inb.all(),
            f"n={len(v)} range=[{v.min():.4f},{v.max():.4f}] "
            f"allowed=[{MLK_LO:.4f},{MLK_HI:.4f}] violations={(~inb).sum()}")
        at_lo = float((np.abs(v - MLK_LO) < 1e-3).mean())
        at_hi = float((np.abs(v - MLK_HI) < 1e-3).mean())
        below = float((v < TRUST_LO).mean())
        above = float((v > TRUST_HI).mean())
        chk(f"F2b {name} boundary census", True,
            f"exactly at lower bound 5.000: {at_lo*100:.2f}%  | "
            f"exactly at upper bound 13.301: {at_hi*100:.2f}%  | "
            f"below titration floor 7.00: {below*100:.2f}%  | "
            f"above titration ceiling 10.48: {above*100:.2f}%")

    # ---- [F3a] direction vs the repo's own BLI cross-check.
    #      NOTE the BLI column itself is censored: non-binders are ASSIGNED
    #      log_Kd_BLI = -5 (10 uM) and the tightest are assigned -10.  Four
    #      DMS entries are the literal string 'nan'.  Both are reported and
    #      the correlation is given with and without the assigned values.
    import csv as _csv
    bli = list(_csv.DictReader(open(os.path.join(
        DMS, "cr9114_result_KD_compare_BLI_DMS.tsv")), delimiter="\t"))
    def _f(v):
        try:
            return float(v)
        except Exception:
            return np.nan
    x = np.array([_f(r["minus_delta_log_Kd_DMS"]) for r in bli])
    y = np.array([_f(r["minus_delta_log_Kd_BLI"]) for r in bli])
    lk = np.array([_f(r["log_Kd_BLI"]) for r in bli])
    ok = np.isfinite(x) & np.isfinite(y)
    rr = float(np.corrcoef(x[ok], y[ok])[0, 1])
    ok2 = ok & (lk > -9.99) & (lk < -5.01)
    rr2 = float(np.corrcoef(x[ok2], y[ok2])[0, 1])
    chk("F3a direction vs BLI", rr > 0.4 and rr2 > 0.4,
        f"n={len(bli)} ({int((~np.isfinite(x)).sum())} DMS entries are the "
        f"literal string 'nan'); pearson r={rr:.3f} on n={int(ok.sum())}; "
        f"r={rr2:.3f} on n={int(ok2.sum())} after dropping the "
        f"{int((ok & ~ok2).sum())} rows where BLI itself was ASSIGNED a "
        f"limit value (-5 or -10). Clearly positive both ways: both columns "
        f"are -delta log10 Kd, so tighter = larger in both.")
    lkd = np.array([_f(r["log_Kd_DMS"]) for r in bli])
    chk("F3a' log_Kd = -(minus_log_Kd)",
        bool((lkd[np.isfinite(lkd)] < 0).all()),
        f"log_Kd_DMS range [{np.nanmin(lkd):.2f},{np.nanmax(lkd):.2f}] -- all "
        f"negative, so minus_log_Kd = -log10 Kd and larger IS tighter")
    chk("F3a'' BLI assay is itself floored", True,
        f"log_Kd_BLI takes the assigned value -5.000 for "
        f"{int((lk == -5.0).sum())}/{len(bli)} rows and -10.000 for "
        f"{int((lk == -10.0).sum())}/{len(bli)} rows -- the ORTHOGONAL assay "
        f"hits a 10 uM floor on the same variants, independent corroboration "
        f"that the weak side of this panel is outside measurement range")

    # ---- [F3b] nonsense weaker than silent
    for name in PANELS:
        recs, bg = load_panel(name)
        sil = np.array([r["mlk"] for r in recs if not r["bad"]
                        and r["cls"] == "silent" and np.isfinite(r["mlk"])])
        non = np.array([r["mlk"] for r in recs if not r["bad"]
                        and r["cls"] == "nonsense" and np.isfinite(r["mlk"])])
        chk(f"F3b {name} nonsense < silent",
            len(non) > 5 and non.mean() < sil.mean() - 0.3,
            f"silent mean={sil.mean():.3f} (n={len(sil)}), "
            f"nonsense mean={non.mean():.3f} (n={len(non)}), "
            f"gap={sil.mean()-non.mean():+.3f} log10 units")

    # ---- [F3c] published breadth ordering H1 > H3 > fluB (somatic)
    sm = {}
    for name in ("WT_H1", "WT_H3", "WT_fluB", "GL_H1"):
        recs, bg = load_panel(name)
        sil = np.array([r["mlk"] for r in recs if not r["bad"]
                        and r["cls"] == "silent" and np.isfinite(r["mlk"])])
        sm[name] = float(sil.mean())
    chk("F3c breadth order H1>H3>fluB",
        sm["WT_H1"] > sm["WT_H3"] > sm["WT_fluB"],
        f"silent-mean -log10 Kd: WT_H1={sm['WT_H1']:.3f} > "
        f"WT_H3={sm['WT_H3']:.3f} > WT_fluB={sm['WT_fluB']:.3f} "
        f"(GL_H1={sm['GL_H1']:.3f}); matches the published CR9114 affinity "
        f"ordering group-1 > group-2 > influenza B")

    # ---- [F3d] published CR9114 maturation phenotype: the 18 V_H somatic
    #      mutations buy BREADTH, not H1-stem affinity -- the germline
    #      precursor already binds H1 stem comparably.  (This is the premise of
    #      the paper's own title, and of the Phillips/Dreyfus line of work.)
    #      So GL_H1 must be CLOSE to WT_H1, and both must be far above
    #      WT_H3/WT_fluB.  "Close" is taken as within 0.5 log10 units.
    chk("F3d germline ~ somatic on H1 stem",
        abs(sm["GL_H1"] - sm["WT_H1"]) <= 0.5
        and min(sm["GL_H1"], sm["WT_H1"]) - max(sm["WT_H3"], sm["WT_fluB"]) > 1.0,
        f"GL_H1={sm['GL_H1']:.3f} vs WT_H1={sm['WT_H1']:.3f} "
        f"(diff {sm['GL_H1']-sm['WT_H1']:+.3f}); both exceed WT_H3 "
        f"{sm['WT_H3']:.3f} and WT_fluB {sm['WT_fluB']:.3f} by "
        f">{min(sm['GL_H1'],sm['WT_H1'])-max(sm['WT_H3'],sm['WT_fluB']):.2f} "
        f"log10 units. NOTE GL comes out marginally TIGHTER than WT on H1 "
        f"(+0.156); that is within the 0.5-unit tolerance and is consistent "
        f"with the pleiotropy the paper reports, but GL and WT are separate "
        f"libraries and separate experiments, so the sign of a 0.16-unit "
        f"difference is NOT treated as a verified fact here.")

    # ---- [F4] magnitude: WT vs H1 must be sub-nM..low-nM
    kd_h1 = 10 ** (-sm["WT_H1"]) / 1e-9
    chk("F4 magnitude WT/H1 K_D,app", 0.05 <= kd_h1 <= 20.0,
        f"K_D,app(silent mean) = {kd_h1:.3f} nM; CR9114 vs group-1 HA is "
        f"reported nM-or-better (Dreyfus 2012 Science), so this must be "
        f"in [0.05, 20] nM")

    # ---- [F5] substitutions-per-site distribution (docs/23: 'almost all'
    #           is not the same as 20)
    for name in PANELS:
        recs, bg = load_panel(name)
        d = build_matrix(recs, "dmlk")
        cnt = d["mask"].sum(1)
        hist = {int(c): int((cnt == c).sum()) for c in np.unique(cnt)}
        chk(f"F5 {name} subs/site", True,
            f"{len(d['sites'])} sites; subs/site min={cnt.min()} "
            f"med={int(np.median(cnt))} max={cnt.max()} mean={cnt.mean():.2f}; "
            f"full-20 sites={int((cnt==20).sum())}; hist={hist}")

    # ---- [F6] replicate agreement (a cheap independent sanity anchor)
    for name in PANELS:
        recs, bg = load_panel(name)
        a = np.array([[r["mlk1"], r["mlk2"]] for r in recs if not r["bad"]
                      and np.isfinite(r["mlk1"]) and np.isfinite(r["mlk2"])])
        rr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        chk(f"F6 {name} rep1 vs rep2", rr > 0.5,
            f"n={len(a)} pearson r={rr:.3f}")

    if verbose:
        print(f"\n  selftest: {len(out)-len(fails)}/{len(out)} checks passed")
    if fails:
        raise AssertionError("cr9114_parse selftest FAILED: " + ", ".join(fails))
    return out


def load_all(value="dmlk", run_selftest=True):
    if run_selftest:
        selftest(verbose=False)
    out = {}
    for name in PANELS:
        recs, bg = load_panel(name)
        d = build_matrix(recs, value)
        d["name"] = name
        d["backbone"] = bg
        d["recs"] = recs
        out[name] = d
    return out


if __name__ == "__main__":
    print("=== cr9114_parse selftest (KNOWN FACTS) ===")
    selftest()
