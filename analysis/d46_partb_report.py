#!/usr/bin/env python3
"""docs/46 Part B 报告：门、分类、同源/跨源、CR9114 签名、次要描述。 -> cache/d46/partb/results.json、tables.md"""
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
import d46_lib as D                      # noqa: E402
import d46_partb as PBm                  # noqa: E402
import d46_partb_lib as B                # noqa: E402
import u2split_lib as U                  # noqa: E402

PB = PBm.PB
EXTS = PBm.EXTS
RB = B.R_B


def load_pl(pi, mode, noise, pt):
    return np.load(os.path.join(PB, "planted", f"{pi}__{mode}__{noise}__{pt}.npz"))["y"]


def haar_mc_proj(d, n=4000, seed=0):
    """随机中心化单位向量对随机中心化 2 维子空间的投影范数。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        A, _ = np.linalg.qr(rng.normal(size=(d, 2)) - 0)
        A = A - A.mean(axis=0, keepdims=True)
        A, _ = np.linalg.qr(A)
        v = rng.normal(size=d); v -= v.mean(); v /= np.linalg.norm(v)
        out.append(np.linalg.norm(A.T @ v))
    return np.array(out)


def main():
    B.verify_locks()
    pre = pickle.load(open(os.path.join(PB, "preal.pkl"), "rb"))
    prs, res = pre["pairs"], pre["res"]
    elig = json.load(open(os.path.join(PB, "pair_elig.json")))["elig"]
    gateE = json.load(open(os.path.join(PB, "gateE.json")))
    out = {"pairs": {}, "elig": elig}
    for pi, pr in enumerate(prs):
        r = res[pi]
        o = {"kind": pr["kind"], "bind": pr["bind"], "src": pr["src"], "primary": pr["primary"], "by_ext": {}}
        for ie, ext in enumerate(EXTS):
            rr = r[ext]
            a = rr["avg"]
            x1, x2, dd = np.nanmedian(a[:, 0]), np.nanmedian(a[:, 1]), int(np.nanmedian(a[:, 2]))
            hn = U.haar_null(dd)
            e = {"d": dd, "cos1": U.boot_median(a[:, 0], seed=1), "cos2": U.boot_median(a[:, 1], seed=2),
                 "full": rr["full"].tolist(),
                 "rep1_cos2": float(np.nanmedian(rr["rep1"][:, 1])), "rep2_cos2": float(np.nanmedian(rr["rep2"][:, 1])),
                 "rep1_cos1": float(np.nanmedian(rr["rep1"][:, 0])), "rep2_cos1": float(np.nanmedian(rr["rep2"][:, 0])),
                 "perm_p95_cos2": float(np.nanquantile(rr["perm"][:, 1], 0.95)), "perm_p95_cos1": float(np.nanquantile(rr["perm"][:, 0], 0.95)),
                 "haar_p95_cos2": float(np.quantile(hn["cos2"], 0.95)), "haar_p95_cos1": float(np.quantile(hn["cos1"], 0.95))}
            e["rep_diff"] = abs(e["rep1_cos2"] - e["rep2_cos2"])
            e["rep_flag"] = bool(e["rep_diff"] > 0.15)
            # 3 维敏感性（只在 EM 字段里；每个 ext 同值）
            c3 = r["c3"]
            if len(c3):
                d3 = int(np.median(c3[:, 2]))
                h3 = B.haar_null_2k(d3, 3, n=2000, seed=d3)
                e["c3"] = {"cos1": float(np.median(c3[:, 0])), "cos2": float(np.median(c3[:, 1])),
                           "haar_p95_cos2": float(np.quantile(h3["cos2"], 0.95)), "d": d3}
            # 植入
            if str(elig.get(pr["id"], "")).startswith(("ok", "exploratory")):
                for noise in ("RR", "G"):
                    ys = {m: [load_pl(pi, m, noise, pt)[:, ie, :] for pt in range(3)] for m in ("SHARED", "PARTIAL", "INDEP")}
                    g1 = all(np.median(ys["SHARED"][pt][:, 1]) > np.quantile(ys["PARTIAL"][pt][:, 1], 0.95) for pt in range(3))
                    ind = np.concatenate([ys["INDEP"][pt][:, 1] for pt in range(3)])
                    g2 = float(np.mean(ind > e["perm_p95_cos2"]))
                    p_sh = max((1 + (ys["SHARED"][pt][:, 1] <= x2).sum()) / (RB + 1) for pt in range(3))
                    p_pa = max((1 + (ys["PARTIAL"][pt][:, 1] >= x2).sum()) / (RB + 1) for pt in range(3))
                    if not (g1 and g2 <= 0.10):
                        cls = "NO_POWER"
                    elif p_sh > 0.05 and p_pa <= 0.05:
                        cls = "COINCIDES"
                    elif p_sh <= 0.05:
                        cls = "DISTINCT"
                    else:
                        cls = "INDETERMINATE"
                    e[noise] = {"G_B1": bool(g1), "G_B2_frac": g2, "p_sh": float(p_sh), "p_pa": float(p_pa), "class": cls,
                                "med": {m: [float(np.median(ys[m][pt][:, 1])) for pt in range(3)] for m in ys},
                                "q05_shared": [float(np.quantile(ys["SHARED"][pt][:, 1], 0.05)) for pt in range(3)],
                                "q95_partial": [float(np.quantile(ys["PARTIAL"][pt][:, 1], 0.95)) for pt in range(3)],
                                "q95_indep": [float(np.quantile(ys["INDEP"][pt][:, 1], 0.95)) for pt in range(3)],
                                "cos1_med": {m: [float(np.median(ys[m][pt][:, 0])) for pt in range(3)] for m in ys}}
            o["by_ext"][ext] = e
        o["proj"] = r["proj"]
        out["pairs"][pr["id"]] = o
    json.dump(out, open(os.path.join(PB, "results.json"), "w"), ensure_ascii=False, indent=1,
              default=lambda z: float(z) if isinstance(z, np.floating) else int(z) if isinstance(z, np.integer) else bool(z) if isinstance(z, np.bool_) else str(z))
    # ---- 表
    Lq = ["# docs/46 Part B 表\n", "## 配对：真实 cos θ（位点不相交，300 分半）与零分布 / 植入对照\n",
          "| 配对 | 类型 | 层 | EM cos θ₁ | EM cos θ₂ [CI] | K1 cos θ₂ [CI] | 零 p95(θ₂) perm/Haar EM | 全位点 θ₂ EM/K1 | 重复1/2 θ₂ EM | 植入 SHARED 中位(RR) EM | PARTIAL 中位 | INDEP 中位 | 类 EM/K1 (RR) | 类 EM/K1 (G) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def ci(t): return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    for pid, o in out["pairs"].items():
        eE, eK = o["by_ext"]["EM"], o["by_ext"]["K1"]
        tier = elig[pid]
        tier_s = "判决" if tier == "ok" else ("探索" if tier.startswith("exploratory") else "—")
        row = f"| {pid} | {o['kind']} | {tier_s} | {eE['cos1'][0]:.2f} | {ci(eE['cos2'])} | {ci(eK['cos2'])} | {eE['perm_p95_cos2']:.2f}/{eE['haar_p95_cos2']:.2f} | {eE['full'][1]:.2f}/{eK['full'][1]:.2f} | {eE['rep1_cos2']:.2f}/{eE['rep2_cos2']:.2f} |"
        if "RR" in eE:
            m = eE["RR"]["med"]
            row += f" {np.mean(m['SHARED']):.2f} | {np.mean(m['PARTIAL']):.2f} | {np.mean(m['INDEP']):.2f} | {eE['RR']['class']}/{eK['RR']['class']} | {eE['G']['class']}/{eK['G']['class']} |"
        else:
            row += " — | — | — | — | — |"
        Lq.append(row)
    open(os.path.join(PB, "tables.md"), "w").write("\n".join(Lq))
    print("\n".join(Lq))


if __name__ == "__main__":
    main()
