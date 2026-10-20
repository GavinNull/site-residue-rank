#!/usr/bin/env python3
"""docs/47 修订 A1 层的门：判据与 exprconf_gate_report.py 一字不变，施加于 (i-c)（以及 ia 作对照），命中方向含 GAMF/PUREF2。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exprconf_lib as E
OUT = E.OUT
WD = os.path.join(OUT, "worlds_a")
CORRS = ("none", "ia", "ic")
H95 = 0.443


def main(subset=None, tag=''):
    E.verify_lock47(False); E.verify_lock47a(False)
    pairs = E.load_all()
    el = json.load(open(os.path.join(OUT, "eligible.json")))["eligible"]
    if subset is not None:
        el = [x for x in el if x in subset]
    ids = [p["id"] for p in pairs]
    data = {}
    for pid in el:
        for sc in E.SCEN:
            for pt in E.PT_IDX:
                data[(pid, sc, pt)] = pickle.load(open(os.path.join(WD, f"{ids.index(pid)}__{sc}__{pt}.pkl"), "rb"))

    def med(sc, corr, ext, key):
        return np.array([np.nanmedian([r[(corr, ext)].get(key, np.nan) for r in data[(pid, sc, pt)]]) for pid in el for pt in E.PT_IDX], float)

    def frac(sc, corr):
        v = []
        for pid in el:
            g = np.array([r[(corr, "K1")].get("g01", np.nan) for r in data[(pid, sc, 3)]], float)
            g = g[np.isfinite(g)]
            if len(g): v.append(float((g > 0.02).mean()))
        return float(np.mean(v)), len(v)

    res = {"eligible": el, "criteria": {}}
    ext = "EM"
    for corr in CORRS:
        c = {}
        scs = ("GAM1", "GAM2", "GAMF1", "GAMF2")
        h1 = {sc: {"unc": float(np.median(med(sc, "none", ext, "ce1"))), "cor": float(np.median(med(sc, corr, ext, "ce1"))),
                   "drop": float(np.median(med(sc, "none", ext, "ce1") - med(sc, corr, ext, "ce1")))} for sc in scs}
        c["HIT1"] = {"detail": h1, "pass": all(v["cor"] <= H95 and v["drop"] >= 0.25 for v in h1.values())}
        h2 = {sc: float(np.median(med(sc, corr, ext, "cb1") / med("BS", corr, ext, "cb1"))) for sc in scs}
        c["HIT2"] = {"ratio": h2, "pass": all(v >= 0.85 for v in h2.values())}
        fa = {}
        for sc in ("BS", "CORR0.08"):
            d1 = med(sc, corr, ext, "ce1obs") - med(sc, "none", ext, "ce1obs")
            d2 = med(sc, "none", ext, "cb1") - med(sc, corr, ext, "cb1")
            fa[sc] = {"d_ce1obs": float(np.median(d1)), "ce1obs": float(np.median(med(sc, corr, ext, "ce1obs"))),
                      "drop_cb1": float(np.median(d2)), "drop_cb1_p95": float(np.quantile(d2, .95))}
        c["FA12"] = {"detail": fa, "pass": all(v["d_ce1obs"] <= 0.05 and v["ce1obs"] <= H95 and v["drop_cb1"] <= 0.05 for v in fa.values())}
        sh = np.concatenate([[r[(corr, ext)]["share"] for r in data[(pid, sc, pt)]] for pid in el for pt in E.PT_IDX for sc in ("BS", "CORR0.08")])
        c["FA3"] = {"share_p95": float(np.quantile(sh, .95)), "share_med": float(np.median(sh)), "pass": bool(np.quantile(sh, .95) <= 0.10)}
        h3 = {"BS_unc": frac("BS", "none")[0], "BS_cor": frac("BS", corr)[0], "PURE2_unc": frac("PURE2", "none")[0], "PURE2_cor": frac("PURE2", corr)[0],
              "PUREF2_unc": frac("PUREF2", "none")[0], "PUREF2_cor": frac("PUREF2", corr)[0], "n_pairs_gain": frac("BS", "none")[1]}
        h3["pass"] = bool(h3["BS_cor"] >= 0.8 * h3["BS_unc"] and h3["PURE2_cor"] <= 0.10 and h3["PUREF2_cor"] <= 0.10 and h3["PURE2_unc"] >= 0.8 and h3["PUREF2_unc"] >= 0.8)
        c["HIT3"] = h3
        rep = {}
        for sc in ("CORR0.30", "PARRES", "PARCELL"):
            rep[sc] = {"ce1_unc": float(np.median(med(sc, "none", ext, "ce1"))), "ce1_cor": float(np.median(med(sc, corr, ext, "ce1"))),
                       "cb1_unc": float(np.median(med(sc, "none", ext, "cb1"))), "cb1_cor": float(np.median(med(sc, corr, ext, "cb1"))),
                       "ce1obs_unc": float(np.median(med(sc, "none", ext, "ce1obs"))), "ce1obs_cor": float(np.median(med(sc, corr, ext, "ce1obs")))}
        c["report"] = rep
        c["ladder"] = {sc: {"ce1": float(np.median(med(sc, corr, ext, "ce1"))), "cb1": float(np.median(med(sc, corr, ext, "cb1"))),
                            "ce1obs": float(np.median(med(sc, corr, ext, "ce1obs"))), "split_ax": float(np.median(med(sc, corr, ext, "split_ax")))}
                       for sc in ("BS", "GAM0.25", "GAM0.5", "GAM1", "GAM2", "GAMF1", "GAMF2")}
        c["ALL"] = bool(c["HIT1"]["pass"] and c["HIT2"]["pass"] and c["HIT3"]["pass"] and c["FA12"]["pass"] and c["FA3"]["pass"])
        res["criteria"][corr] = c
    dr = np.concatenate([med(sc, "none", ext, "cb1") - med(sc, "ic", ext, "cb1") for sc in ("BS", "CORR0.08")])
    res["delta0"] = {"p95_drop_ic": float(np.quantile(dr, .95)), "delta0": float(max(0.05, np.quantile(dr, .95)))}
    res["gates"] = {k: {kk: v[kk]["pass"] for kk in ("HIT1", "HIT2", "HIT3", "FA12", "FA3")} | {"ALL": v["ALL"]} for k, v in res["criteria"].items() if k != "none"}
    res["teeth"] = {"NOOP_fails_HIT1": not res["criteria"]["none"]["HIT1"]["pass"]}
    json.dump(res, open(os.path.join(OUT, f"gate_a{tag}.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"gates": res["gates"], "delta0": res["delta0"], "teeth": res["teeth"]}, ensure_ascii=False, indent=1))
    for corr in CORRS:
        c = res["criteria"][corr]
        print(corr, "HIT1", {k: (round(v['unc'], 3), round(v['cor'], 3)) for k, v in c["HIT1"]["detail"].items()}, "HIT2", {k: round(v, 3) for k, v in c["HIT2"]["ratio"].items()})
        print("   FA", {k: (round(v['d_ce1obs'], 3), round(v['drop_cb1'], 3)) for k, v in c["FA12"]["detail"].items()}, "FA3", {k: round(v, 3) for k, v in c["FA3"].items() if k != 'pass'}, "HIT3", {k: round(v, 3) for k, v in c["HIT3"].items() if k != 'pass'})
        print("   report", {k: {a: round(b, 3) for a, b in v.items()} for k, v in c["report"].items()})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "strata":
        pairs = E.load_all()
        K2 = {p["id"] for p in pairs if p["src"] in ("WH1_v1", "WH1_v2", "Eta")}
        K1_ = {p["id"] for p in pairs if p["src"] in ("BA2", "PDCoV")}
        for nm, st in (("_k2", K2), ("_k1", K1_)):
            print("=== 层", nm, sorted(st))
            main(st, nm)
    else:
        main()
