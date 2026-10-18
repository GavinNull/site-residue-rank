#!/usr/bin/env python3
"""docs/45 报告：按预先登记 §9 的规则分类、判决，并输出表格与 cache/u2split/results.json。
只读 cache/u2split/*；入口断言预先登记锁。"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u2split_lib as U                          # noqa: E402
import u2split_run as RUN                        # noqa: E402

OUT = RUN.OUT
EXTS = U.EXTRACTORS
SC = RUN.SCOPE
S1 = RUN.S1
PATHM = RUN.PATHM
R = U.R_WORLD


def short(n):
    return n.replace("animalACE2/", "").replace("Wuhan-Hu-1", "WH1").replace("Omicron_", "").replace("PDCoV/PDCoV/", "")


def fmt_ci(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main():
    U.verify_lock()
    gate = json.load(open(os.path.join(OUT, "gate.json")))
    perm = pickle.load(open(os.path.join(OUT, "perm.pkl"), "rb"))
    real = pickle.load(open(os.path.join(OUT, "real.pkl"), "rb"))
    cross = pickle.load(open(os.path.join(OUT, "cross.pkl"), "rb"))
    share = pickle.load(open(os.path.join(OUT, "share.pkl"), "rb"))
    g, noise = RUN.load_geoms()
    res = {"gate": gate, "matrices": {}, "verdict": {}}
    W = RUN.load_world

    # ------------------------------------------------ 逐矩阵
    for n in SC:
        res["matrices"][n] = {}
        for ie, e in enumerate(EXTS):
            sp = real[n][e]["splits"]
            c1, c2, dd = sp[:, 0], sp[:, 1], sp[:, 2]
            dmed = int(np.nanmedian(dd))
            hn = U.haar_null(dmed)
            pn = perm[n][e]
            m = {
                "d": dmed,
                "cos1": U.boot_median(c1, seed=1), "cos2": U.boot_median(c2, seed=2),
                "cos2_q": [float(np.nanquantile(c2, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)],
                "contig": real[n][e]["contig"],
                "haar_cos2_med_p95": [float(np.median(hn["cos2"])), float(np.quantile(hn["cos2"], 0.95))],
                "haar_cos1_med_p95": [float(np.median(hn["cos1"])), float(np.quantile(hn["cos1"], 0.95))],
                "perm_cos2_med_p95": [float(np.nanmedian(pn[:, 1])), float(np.nanquantile(pn[:, 1], 0.95))],
                "perm_cos1_med_p95": [float(np.nanmedian(pn[:, 0])), float(np.nanquantile(pn[:, 0], 0.95))],
                "n_fail": int(np.isnan(c2).sum()),
            }
            if n in PATHM:
                x2 = float(np.nanmedian(c2))
                x1 = float(np.nanmedian(c1))
                w = {}
                for kind in ("R1", "R2"):
                    pts = []
                    for p in range(7):
                        arr = W(n, kind, p)[:, :, ie, :]
                        y1 = np.nanmedian(arr[:, :, 0], axis=1)
                        y2 = np.nanmedian(arr[:, :, 1], axis=1)
                        pts.append({"sigma": noise[n]["path_k1" if kind == "R1" else "path_k2"][p]["sigma"],
                                    "snr": noise[n]["path_k1" if kind == "R1" else "path_k2"][p]["snr"],
                                    "cos1_med": float(np.median(y1)), "cos2_med": float(np.median(y2)),
                                    "cos2_q05": float(np.quantile(y2, 0.05)), "cos2_q95": float(np.quantile(y2, 0.95)),
                                    "cos2_pooled_q95": float(np.nanquantile(arr[:, :, 1], 0.95)),
                                    "pos_ge": float((1 + (y2 >= x2).sum()) / (R + 1)),
                                    "pos_le": float((1 + (y2 <= x2).sum()) / (R + 1)),
                                    "pos1_ge": float((1 + (y1 >= x1).sum()) / (R + 1)),
                                    "pos1_le": float((1 + (y1 <= x1).sum()) / (R + 1)),
                                    "q95": float(np.quantile(y2, 0.95)), "q05": float(np.quantile(y2, 0.05))})
                    w[kind] = pts
                for kind in ("R0", "R1P", "R2P"):
                    arr = W(n, kind, 0)[:, :, ie, :]
                    y2 = np.nanmedian(arr[:, :, 1], axis=1)
                    w[kind] = {"cos1_med": float(np.median(np.nanmedian(arr[:, :, 0], axis=1))),
                               "cos2_med": float(np.median(y2)), "cos2_q05": float(np.quantile(y2, 0.05)),
                               "cos2_q95": float(np.quantile(y2, 0.95)),
                               "pos_ge": float((1 + (y2 >= x2).sum()) / (R + 1)),
                               "pos_le": float((1 + (y2 <= x2).sum()) / (R + 1))}
                dose = {}
                for lam in RUN.DOSE_LAMS:
                    arr = W(n, "R2L", RUN.DOSE_POINT, lam)[:, :, ie, :]
                    y2 = np.nanmedian(arr[:, :, 1], axis=1)
                    pt = noise[n]["path_k2"][RUN.DOSE_POINT]
                    dose[str(lam)] = {"cos2_med": float(np.median(y2)), "cos2_q05": float(np.quantile(y2, 0.05)),
                                      "cos2_q95": float(np.quantile(y2, 0.95)),
                                      "planted_share2": U.planted_share2(g[n], pt, lam)}
                w["dose"] = dose
                m["worlds"] = w
                # ---- 预先登记 §9 的分类
                pos1 = max(p["pos_ge"] for p in w["R1"])
                pos2 = max(p["pos_le"] for p in w["R2"])
                sep = gate["H2"][e]["separable"].get(n, False)
                if not sep:
                    cls = "U"
                elif pos1 <= 0.05:
                    cls = "R" + ("(弱于R2)" if pos2 <= 0.05 else "")
                elif pos2 <= 0.05:
                    cls = "N"
                else:
                    cls = "I"
                m["pos1"], m["pos2"], m["class"] = pos1, pos2, cls
                # 第一成分的对照分类
                pos1_c1 = max(p["pos1_ge"] for p in w["R1"])  # 此处仅供参考：cos θ₁ 在 R1 世界本就高
                m["cos1_vs_R1_pos"] = pos1_c1
                # 有效幅度 λ_eff（描述性）
                lams = np.array(RUN.DOSE_LAMS)
                meds = np.array([dose[str(l)]["cos2_med"] for l in lams])
                if x2 <= meds[0]:
                    leff = 0.0
                elif x2 >= meds.max():
                    leff = float(">" + str(lams[meds.argmax()])) if False else float(lams[meds.argmax()])
                else:
                    leff = float(np.interp(x2, meds, lams)) if np.all(np.diff(meds) >= 0) else None
                m["lambda_eff"] = leff
                # 最小可检出的 λ
                q95_0 = dose["0.0"]["cos2_q95"]
                mind = None
                for l in RUN.DOSE_LAMS[1:]:
                    if dose[str(l)]["cos2_med"] > q95_0:
                        mind = l
                        break
                m["min_detect_lambda"] = mind
                m["min_detect_share2"] = None if mind is None else dose[str(mind)]["planted_share2"]
            res["matrices"][n][e] = m

    # ------------------------------------------------ 体系层判决
    for e in EXTS:
        cl = {n: res["matrices"][n][e]["class"] for n in PATHM if n in S1}
        assess = [n for n, c in cl.items() if c != "U"]
        nR = sum(c.startswith("R") for n, c in cl.items() if n in assess)
        nN = sum(c == "N" for n, c in cl.items() if n in assess)
        nI = sum(c == "I" for n, c in cl.items() if n in assess)
        na = len(assess)
        if na == 0:
            v = "CANNOT_ASSESS"
        elif nR >= 2 / 3 * na:
            v = "REPRODUCIBLE"
        elif nN >= 2 / 3 * na:
            v = "NOT_REPRODUCIBLE"
        else:
            v = "UNDECIDED"
        res["verdict"][e] = {"n_assessable": na, "R": nR, "N": nN, "I": nI, "U": len(cl) - na,
                             "verdict": v, "classes": cl}
    ve = [res["verdict"][e]["verdict"] for e in EXTS]
    if not gate["pass"]:
        total = "CANNOT_ASSESS (gate)"
    elif all(v == "REPRODUCIBLE" for v in ve):
        total = "REPRODUCIBLE"
    elif all(v == "NOT_REPRODUCIBLE" for v in ve):
        total = "NOT_REPRODUCIBLE"
    else:
        total = "CANNOT_ASSESS (extraction-dependent / undecided)"
    res["verdict"]["total"] = total

    # ------------------------------------------------ 体系层汇总（描述性；矩阵不独立）
    summ = {}
    for e in EXTS:
        v = [res["matrices"][n][e]["cos2"][0] for n in S1]
        summ[e] = {"S1_median_of_medians_cos2": U.boot_median(v, seed=3),
                   "S1_median_of_medians_cos1": U.boot_median([res["matrices"][n][e]["cos1"][0] for n in S1], seed=4),
                   "S1_median_perm_p95": float(np.median([res["matrices"][n][e]["perm_cos2_med_p95"][1] for n in S1]))}
        pm = [n for n in PATHM if n in S1]
        for kind in ("R1", "R2"):
            pass
    res["S1_summary"] = summ

    # ------------------------------------------------ 跨组合
    def collect(variant, ext, key, filt=None):
        vals = []
        for rec in cross["pairs"]:
            k = f"{variant}|{ext}"
            if k not in rec:
                continue
            if filt and not filt(rec):
                continue
            vals.append(rec[k][key])
        return np.array(vals)

    def bg(rec):
        return tuple(sorted((rec["X"][0], rec["Y"][0])))

    def lig(rec):
        return tuple(sorted((rec["X"][1], rec["Y"][1])))

    cr = {}
    strata = {
        "all": None,
        "同配体不同骨架": lambda r: r["X"][1] == r["Y"][1] and r["X"][0] != r["Y"][0],
        "同骨架不同配体": lambda r: r["X"][0] == r["Y"][0],
        "骨架与配体都不同": lambda r: r["X"][0] != r["Y"][0] and r["X"][1] != r["Y"][1],
        "WH1–Eta(骨架对)": lambda r: bg(r) == ("Eta", "Wuhan-Hu-1"),
        "WH1–BA2(骨架对)": lambda r: bg(r) == ("Omicron_BA2", "Wuhan-Hu-1"),
        "Eta–BA2(骨架对)": lambda r: bg(r) == ("Eta", "Omicron_BA2"),
    }
    for e in EXTS:
        cr[e] = {}
        for sname, f in strata.items():
            row = {}
            for key in "ABCD":
                A = collect("all", e, key, f)
                if len(A) == 0:
                    continue
                c2 = A[:, 1] if key != "A" else A[:, 1]
                row[key] = {"n": int(len(A)),
                            "cos1": U.boot_median(A[:, 0], seed=5), "cos2": U.boot_median(A[:, 1], seed=6),
                            "cos2sq_med": float(np.median(A[:, 1] ** 2))}
            if "B" in row and "C" in row:
                row["rho_cos2"] = row["C"]["cos2"][0] / row["B"]["cos2"][0]
                row["rho_cos2sq"] = row["C"]["cos2sq_med"] / row["B"]["cos2sq_med"]
            if "A" in row and "B" in row:
                row["A_over_B"] = row["A"]["cos2"][0] / row["B"]["cos2"][0]
            cr[e][sname] = row
        # WH1–BA2 剔除差异位点
        row = {}
        f = strata["WH1–BA2(骨架对)"]
        for key in "ABCD":
            A = collect("excl_diff", e, key, f)
            if len(A):
                row[key] = {"n": int(len(A)), "cos2": U.boot_median(A[:, 1], seed=7),
                            "cos2sq_med": float(np.median(A[:, 1] ** 2))}
        if "B" in row and "C" in row:
            row["rho_cos2sq"] = row["C"]["cos2sq_med"] / row["B"]["cos2sq_med"]
        cr[e]["WH1–BA2 剔除差异位点"] = row
    res["cross"] = cr
    res["rel_docs14"] = cross["rel"]

    # ------------------------------------------------ 方差占比
    sh = {"real": {}, "cal": {}}
    for n in SC:
        sh["real"][n] = share["real"][n]
    for (n, kind, p, lam), v in share["cal"].items():
        key = f"{n}|{kind}|{p}|{lam}"
        sh["cal"][key] = {"k1_share2": [x["share2"] for x in v["k1"]],
                          "k1_tot2": [x["tot2"] for x in v["k1"]],
                          "std_share2": [x["share2"] for x in v["standard"]],
                          "std_tot2": [x["tot2"] for x in v["standard"]],
                          "planted": v["planted"]}
    res["share"] = sh

    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
    write_tables(res, g, noise)
    print("OK", total)


def write_tables(res, g, noise):
    L = []
    P = L.append
    P("## 门")
    for e in EXTS:
        P(f"- [{e}] H1 R0 世界超过置换 p95 的比例 {res['gate']['H1'][e]['pooled_frac_exceed_perm_p95']:.3f}；"
          f"H2 S1 可分辨比例 {res['gate']['H2'][e]['frac_S1']:.2f}")
    P("")
    for e in EXTS:
        P(f"### 真实矩阵 vs 植入世界 [{e}]（cos θ₂；世界值 = 60 个复现各自分半中位的 中位 [p5, p95]）")
        P("| 矩阵 | d | 真实 cos θ₁ | 真实 cos θ₂ [95% CI] | N-perm 中位/p95 | Haar 中位/p95 | R1 世界(σ_meas 端) | R1 世界(中点) | R1 世界(σ_fit1 端) | R2 世界(σ_meas 端) | R2 世界(中点) | R2 世界(σ_fit2 端) | R0 | R1P | R2P | 连续分半 | pos1 | pos2 | 类 |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n in SC:
            m = res["matrices"][n][e]
            row = [short(n), str(m["d"]), f"{m['cos1'][0]:.3f}", fmt_ci(m["cos2"]),
                   f"{m['perm_cos2_med_p95'][0]:.2f}/{m['perm_cos2_med_p95'][1]:.2f}",
                   f"{m['haar_cos2_med_p95'][0]:.2f}/{m['haar_cos2_med_p95'][1]:.2f}"]
            if "worlds" in m:
                w = m["worlds"]
                def cell(x):
                    return f"{x['cos2_med']:.2f} [{x['cos2_q05']:.2f},{x['cos2_q95']:.2f}]"
                row += [cell(w["R1"][0]), cell(w["R1"][3]), cell(w["R1"][6]),
                        cell(w["R2"][0]), cell(w["R2"][3]), cell(w["R2"][6]),
                        cell(w["R0"]), cell(w["R1P"]), cell(w["R2P"]),
                        f"{m['contig'][1]:.2f}" if m["contig"] else "—",
                        f"{m['pos1']:.3f}", f"{m['pos2']:.3f}", m["class"]]
            else:
                row += ["—"] * 9 + [f"{m['contig'][1]:.2f}" if m["contig"] else "—", "—", "—", "不可评估"]
            P("| " + " | ".join(row) + " |")
        P("")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
