#!/usr/bin/env python3
"""docs/47 门的判定（预先登记 §3 的数值判据，一字不改）。读 cache/exprconf/worlds/*.pkl -> gate.json、gate_tables.md。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exprconf_lib as E
OUT = E.OUT
WD = os.path.join(OUT, "worlds")
CORRS = ("none", "ia", "ib", "ii", "dst")
H95 = 0.443      # 预先登记写定的 d=20 一维 Haar p95


def load():
    pairs = E.load_all()
    el = json.load(open(os.path.join(OUT, "eligible.json")))["eligible"]
    ids = [p["id"] for p in pairs]
    data = {}
    for pid in el:
        pi = ids.index(pid)
        for sc in E.SCEN:
            if sc in ('GAMF1', 'GAMF2', 'PUREF2'):
                continue
            for pt in E.PT_IDX:
                data[(pid, sc, pt)] = pickle.load(open(os.path.join(WD, f"{pi}__{sc}__{pt}.pkl"), "rb"))
    return el, data


def col(recs, corr, ext, key):
    return np.array([r[(corr, ext)].get(key, np.nan) if not isinstance(r[(corr, ext)].get(key, np.nan), list) else np.nan for r in recs], float)


def med_over(data, el, sc, corr, ext, key, subset=None):
    """每 (pair,pt) 取重复中位数，再给出这些中位数的数组。"""
    out = []
    for pid in el:
        if subset and pid not in subset:
            continue
        for pt in E.PT_IDX:
            out.append(np.nanmedian(col(data[(pid, sc, pt)], corr, ext, key)))
    return np.array(out)


def frac_gain(data, el, sc, corr, thr=0.02):
    v = []
    for pid in el:
        recs = data[(pid, sc, 3)]
        g = np.array([r[(corr, "K1")].get("g01", np.nan) for r in recs], float)
        g = g[np.isfinite(g)]
        if len(g):
            v.append(float((g > thr).mean()))
    return np.array(v)


def main():
    E.verify_lock47(False)
    el, data = load()
    res = {"eligible": el, "criteria": {}, "tables": {}}
    for ext in E.EXTS:
        R = {}
        for corr in CORRS:
            c = {}
            # HIT-1
            h1 = {}
            for sc in ("GAM1", "GAM2"):
                unc = med_over(data, el, sc, "none", ext, "ce1")
                cor = med_over(data, el, sc, corr, ext, "ce1")
                h1[sc] = {"unc": float(np.median(unc)), "cor": float(np.median(cor)), "drop": float(np.median(unc - cor))}
            c["HIT1"] = {"detail": h1, "pass": all(v["cor"] <= H95 and v["drop"] >= 0.25 for v in h1.values())}
            # HIT-2
            h2 = {}
            for sc in ("GAM1", "GAM2"):
                a = med_over(data, el, sc, corr, ext, "cb1"); b = med_over(data, el, "BS", corr, ext, "cb1")
                h2[sc] = float(np.median(a / b))
            c["HIT2"] = {"ratio": h2, "pass": all(v >= 0.85 for v in h2.values())}
            # FA-1/2
            fa = {}
            for sc in ("BS", "CORR0.08"):
                d1 = med_over(data, el, sc, corr, ext, "ce1obs") - med_over(data, el, sc, "none", ext, "ce1obs")
                d2 = med_over(data, el, sc, "none", ext, "cb1") - med_over(data, el, sc, corr, ext, "cb1")
                fa[sc] = {"d_ce1obs": float(np.median(d1)), "ce1obs": float(np.median(med_over(data, el, sc, corr, ext, "ce1obs"))),
                          "drop_cb1": float(np.median(d2)), "drop_cb1_p95": float(np.quantile(d2, 0.95))}
            c["FA12"] = {"detail": fa, "pass": all(v["d_ce1obs"] <= 0.05 and v["ce1obs"] <= H95 and v["drop_cb1"] <= 0.05 for v in fa.values())}
            # FA-3：表观份额 p95
            sh = np.concatenate([[r[(corr, ext)]["share"] for r in data[(pid, sc, pt)]] for pid in el for pt in E.PT_IDX for sc in ("BS", "CORR0.08")])
            c["FA3"] = {"share_p95": float(np.quantile(sh, 0.95)), "share_med": float(np.median(sh)), "pass": bool(np.quantile(sh, 0.95) <= 0.10)}
            # 报告但不入门
            rep = {}
            for sc in ("CORR0.30", "PARRES", "PARCELL"):
                rep[sc] = {"ce1_unc": float(np.median(med_over(data, el, sc, "none", ext, "ce1"))), "ce1_cor": float(np.median(med_over(data, el, sc, corr, ext, "ce1"))),
                           "cb1_unc": float(np.median(med_over(data, el, sc, "none", ext, "cb1"))), "cb1_cor": float(np.median(med_over(data, el, sc, corr, ext, "cb1"))),
                           "ce1obs_unc": float(np.median(med_over(data, el, sc, "none", ext, "ce1obs"))), "ce1obs_cor": float(np.median(med_over(data, el, sc, corr, ext, "ce1obs"))),
                           "drop_cb1": float(np.median(med_over(data, el, sc, "none", ext, "cb1") - med_over(data, el, sc, corr, ext, "cb1")))}
            c["report"] = rep
            # 梯度
            lad = {}
            for sc in ("BS", "GAM0.25", "GAM0.5", "GAM1", "GAM2"):
                lad[sc] = {"ce1": float(np.median(med_over(data, el, sc, corr, ext, "ce1"))), "cb1": float(np.median(med_over(data, el, sc, corr, ext, "cb1"))),
                           "ce1obs": float(np.median(med_over(data, el, sc, corr, ext, "ce1obs"))),
                           "split_ax": float(np.median(med_over(data, el, sc, corr, ext, "split_ax"))) if ext == "EM" else None}
            c["ladder"] = lad
            R[corr] = c
        res["criteria"][ext] = R
    # HIT-3（只在 K1 的 g01 上）
    h3 = {}
    for corr in CORRS:
        bs_u, bs_c = frac_gain(data, el, "BS", "none"), frac_gain(data, el, "BS", corr)
        pu_u, pu_c = frac_gain(data, el, "PURE2", "none"), frac_gain(data, el, "PURE2", corr)
        gu = frac_gain(data, el, "GAM2", corr)
        h3[corr] = {"BS_unc": float(np.mean(bs_u)), "BS_cor": float(np.mean(bs_c)), "PURE2_unc": float(np.mean(pu_u)), "PURE2_cor": float(np.mean(pu_c)),
                    "GAM2_cor": float(np.mean(gu)), "pass": bool(np.mean(bs_c) >= 0.8 * np.mean(bs_u) and np.mean(pu_c) <= 0.10 and np.mean(pu_u) >= 0.80)}
    res["HIT3"] = h3
    # δ0
    dr = np.concatenate([med_over(data, el, sc, "none", "EM", "cb1") - med_over(data, el, sc, "ia", "EM", "cb1") for sc in ("BS", "CORR0.08")])
    res["delta0"] = {"p95_drop_ia": float(np.quantile(dr, 0.95)), "delta0": float(max(0.05, np.quantile(dr, 0.95)))}
    # 门汇总
    gates = {}
    for corr in ("ia", "ib", "ii"):
        crit = res["criteria"]["EM"][corr]
        gates[corr] = {"HIT1": crit["HIT1"]["pass"], "HIT2": crit["HIT2"]["pass"], "HIT3": h3[corr]["pass"], "FA12": crit["FA12"]["pass"], "FA3": crit["FA3"]["pass"]}
        gates[corr]["ALL"] = all(gates[corr].values())
    teeth = {"NOOP_fails_HIT1": not res["criteria"]["EM"]["none"]["HIT1"]["pass"],
             "DESTROY_fails_HIT2_or_FA2": (not res["criteria"]["EM"]["dst"]["HIT2"]["pass"]) or (not res["criteria"]["EM"]["dst"]["FA12"]["pass"])}
    res["gates"] = gates
    res["teeth"] = teeth
    json.dump(res, open(os.path.join(OUT, "gate.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"gates": gates, "teeth": teeth, "delta0": res["delta0"], "HIT3": h3}, ensure_ascii=False, indent=1))
    for ext in ("EM",):
        for corr in CORRS:
            c = res["criteria"][ext][corr]
            print(corr, "HIT1", c["HIT1"]["detail"], "HIT2", c["HIT2"]["ratio"], "FA", {k: (round(v['d_ce1obs'], 3), round(v['drop_cb1'], 3)) for k, v in c["FA12"]["detail"].items()}, "FA3", c["FA3"])


if __name__ == "__main__":
    main()
