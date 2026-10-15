#!/usr/bin/env python3
"""docs/42 Step 2：在每个域**自己的**几何上，已知秩 0/1/2/3 的检出概率（三个 σ 锚点）。

预先登记：cache/docs42_prereg.md（sha256 锁在 cache/docs42_prereg_lock.json）。
本脚本**不**对任何真实矩阵定秩；真实矩阵只提供几何（位点、缺失模式、主效应、cso）。

运行：python3 analysis/foldrank_power.py --jobs 9
      python3 analysis/foldrank_power.py --aggregate-only
产物：cache/foldrank_power_parts/<name>.json（逐域，可续跑）
      cache/foldrank_power.json（汇总 + 纳入判定）
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse                                             # noqa: E402
import json                                                 # noqa: E402
import math                                                 # noqa: E402
import sys                                                  # noqa: E402
import time                                                 # noqa: E402
from concurrent.futures import ProcessPoolExecutor         # noqa: E402

import numpy as np                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foldrank_lib as F                                    # noqa: E402
import rank_core as RC                                      # noqa: E402
from subspace_lib import ROOT                               # noqa: E402

PARTS = os.path.join(ROOT, "cache", "foldrank_power_parts")
OUT = os.path.join(ROOT, "cache", "foldrank_power.json")
NSIM = 40
NSUP = 30
SGRID = [0.15, 0.25, 0.35, 0.50, 0.70]
SENS_MIN, FP_MAX, MIN_CLUSTERS = 0.80, 0.20, 20

_G = {}


def _init():
    names, mats, cl, sigC, fout = F.universe()
    _G.update(names=names, mats=mats, cl=cl, sigC=sigC, fout=fout)


def _cell(name, geo, mask, tag, rank, s, sigma, n, weights=None, check_cso=0):
    dets, curves, csos = [], [], []
    for rep in range(n):
        rng = F.rng_for(name, "P", tag, rank, weights is not None, rep)
        M = F.synth(geo, mask, rank, s, sigma, rng, weights)
        d, trim, _ = F.detect(M, mask)
        dets.append(d)
        curves.append((trim / trim[0]).tolist())
        if rep < check_cso:
            r, c = RC.center_fit(M, mask)
            R = M - r[:, None] - c[None, :]
            csos.append(float(R[mask].std()))
    return {"det": dets, "curve": curves, "cso_sim": csos}


def power_one(name):
    path = os.path.join(PARTS, name.replace("|", "_") + ".json")
    if os.path.exists(path):
        return name, "cached"
    t0 = time.time()
    m = _G["mats"][name]
    mask = m["mask"]
    M = m["M"]
    geo = F.geometry(M, mask)
    scn = F.scenario(geo, _G["sigC"][name])
    out = {"name": name, "sites": int(mask.shape[0]),
           "n_obs": int(mask.sum()), "cso": geo["cso"],
           "kappa": geo["kappa"], "row_sd": geo["row_sd"],
           "col_sd": geo["col_sd"], "scenario": scn, "primary": {},
           "grid": {}, "equal_spec": {}}
    for tag in ("A", "C", "B"):
        sc = scn[tag]
        out["primary"][tag] = {}
        for k in (0, 1, 2, 3):
            if k == 0:
                c = _cell(name, geo, mask, tag, 0, 0.0, sc["sigma_null"],
                          NSIM, check_cso=5)
            else:
                c = _cell(name, geo, mask, tag, k, sc["s"], sc["sigma"],
                          NSIM, check_cso=5 if k == 2 else 0)
            out["primary"][tag][str(k)] = c
    sigC = scn["C"]["sigma"]
    for s in SGRID:
        for k in (1, 2):
            out["grid"][f"{s}|{k}"] = _cell(name, geo, mask, f"G{s}", k, s,
                                            sigC, NSUP)
    for tag in ("A", "C", "B"):
        sc = scn[tag]
        out["equal_spec"][tag] = _cell(name, geo, mask, tag + "eq", 2,
                                       sc["s"], sc["sigma"], NSUP,
                                       weights=(1.0, 1.0))
    out["seconds"] = time.time() - t0
    os.makedirs(PARTS, exist_ok=True)
    json.dump(out, open(path, "w"))
    return name, f"{out['seconds']:.0f}s"


# ----------------------------------------------------------------------
def conf(dets, n=None):
    d = np.asarray(dets)
    return np.bincount(d, minlength=4) / len(d)


def aggregate(verbose=True):
    names, mats, cl, sigC, fout = F.universe()
    parts = {}
    for w in names:
        p = os.path.join(PARTS, w.replace("|", "_") + ".json")
        assert os.path.exists(p), f"缺 {w}"
        parts[w] = json.load(open(p))
    res = {"n_domains": len(names), "domains": {}, "anchors": {}}
    for w in names:
        P = parts[w]
        d = {"cluster": cl[w], "f_out": fout[w], "sites": P["sites"],
             "n_obs": P["n_obs"], "cso": P["cso"], "kappa": P["kappa"],
             "scenario": P["scenario"], "anchor": {}}
        for tag in ("A", "C", "B"):
            C = np.array([conf(P["primary"][tag][str(k)]["det"])
                          for k in range(4)])          # [真实k, 检出j]
            sens, fp1, fp0 = C[2, 2], C[1, 2], C[0, 2]
            cons = P["scenario"][tag]["kind"] == "consistent"
            crit = (sens >= SENS_MIN) and (fp1 <= FP_MAX) and (fp0 <= FP_MAX)
            # 量级核对：合成秩 2 情景的实现残差 SD / 实测 cso
            cs = P["primary"][tag]["2"]["cso_sim"]
            # 合成曲线中位（相对 t0）
            cur = {k: np.median(np.array(P["primary"][tag][str(k)]["curve"]),
                                axis=0).tolist() for k in range(4)}
            d["anchor"][tag] = {
                "confusion": C.tolist(), "sens": float(sens),
                "fp1": float(fp1), "fp0": float(fp0),
                "kind": P["scenario"][tag]["kind"],
                "s": P["scenario"][tag]["s"],
                "sigma": P["scenario"][tag]["sigma"],
                "s_over_sigma": P["scenario"][tag]["s_over_sigma"],
                "criteria_met": bool(crit),
                "admitted": bool(crit and cons),
                "ceiling_admitted": bool(crit and not cons),
                "cso_sim_over_obs": (float(np.mean(cs) / P["cso"])
                                     if cs else None),
                "curve_median": cur,
                "sens_ci_wilson": _wilson(int(round(sens * NSIM)), NSIM)}
        # 补充：s 网格（σ_C）与谱等权
        d["grid"] = {kk: conf(v["det"]).tolist()
                     for kk, v in P["grid"].items()}
        d["equal_spec"] = {tag: conf(v["det"]).tolist()
                           for tag, v in P["equal_spec"].items()}
        res["domains"][w] = d

    # ---- 簇层 ----
    clusters = sorted(set(cl.values()))
    for tag in ("A", "C", "B"):
        adm = [w for w in names if res["domains"][w]["anchor"][tag]["admitted"]]
        cadm = sorted({cl[w] for w in adm})
        ceil_adm = [w for w in names
                    if res["domains"][w]["anchor"][tag]["ceiling_admitted"]]
        # 代表（预先登记 §1）
        reps = {}
        for c in cadm:
            mem = [w for w in adm if cl[w] == c]
            mem.sort(key=lambda w: (-res["domains"][w]["anchor"][tag]["sens"],
                                    res["domains"][w]["anchor"][tag]["fp1"],
                                    -res["domains"][w]["n_obs"], w))
            reps[c] = mem[0]
        ncons = sum(1 for w in names
                    if res["domains"][w]["anchor"][tag]["kind"] == "consistent")
        sens_all = [res["domains"][w]["anchor"][tag]["sens"] for w in names]
        res["anchors"][tag] = {
            "n_consistent": ncons,
            "n_admitted_domains": len(adm),
            "n_admitted_clusters": len(cadm),
            "n_ceiling_admitted_domains": len(ceil_adm),
            "n_ceiling_admitted_clusters": len({cl[w] for w in ceil_adm}),
            "admitted": adm, "reps": reps,
            "proceed": bool(len(cadm) >= MIN_CLUSTERS),
            "sens_median": float(np.median(sens_all)),
            "sens_q": [float(np.percentile(sens_all, q)) for q in (10, 25, 75, 90)],
            "sens_ge_0.7": int(sum(s >= 0.7 for s in sens_all)),
            "sens_ge_0.9": int(sum(s >= 0.9 for s in sens_all)),
            "mean_confusion": np.mean(
                [res["domains"][w]["anchor"][tag]["confusion"]
                 for w in names], axis=0).tolist(),
        }
        res["anchors"][tag]["sens_sensitivity"] = {
            str(th): {
                "domains": int(sum(
                    (res["domains"][w]["anchor"][tag]["sens"] >= th and
                     res["domains"][w]["anchor"][tag]["fp1"] <= FP_MAX and
                     res["domains"][w]["anchor"][tag]["fp0"] <= FP_MAX and
                     res["domains"][w]["anchor"][tag]["kind"] == "consistent")
                    for w in names)),
                "clusters": len({cl[w] for w in names if (
                    res["domains"][w]["anchor"][tag]["sens"] >= th and
                    res["domains"][w]["anchor"][tag]["fp1"] <= FP_MAX and
                    res["domains"][w]["anchor"][tag]["fp0"] <= FP_MAX and
                    res["domains"][w]["anchor"][tag]["kind"] == "consistent")})}
            for th in (0.7, 0.8, 0.9)}
    res["any_proceed"] = bool(any(res["anchors"][t]["proceed"]
                                  for t in ("A", "C", "B")))
    res["n_clusters_total"] = len(clusters)
    json.dump(res, open(OUT, "w"), default=float)
    if verbose:
        print_summary(res)
    return res


def _wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def print_summary(res):
    print("\n" + "=" * 100)
    print(f"功效汇总：{res['n_domains']} 个域 / {res['n_clusters_total']} 个簇")
    print("=" * 100)
    for tag in ("A", "C", "B"):
        a = res["anchors"][tag]
        print(f"\n锚点 {tag}: 情景相容 {a['n_consistent']}/{res['n_domains']}；"
              f"被纳入 域 {a['n_admitted_domains']} / 簇 "
              f"{a['n_admitted_clusters']}；"
              f"上限情景下满足判据 域 {a['n_ceiling_admitted_domains']} / 簇 "
              f"{a['n_ceiling_admitted_clusters']}；放行={a['proceed']}")
        print(f"   sens 中位 {a['sens_median']:.2f}，分位 10/25/75/90 = "
              + "/".join(f"{x:.2f}" for x in a["sens_q"])
              + f"；sens≥0.7: {a['sens_ge_0.7']}，≥0.9: {a['sens_ge_0.9']}")
        print("   平均混淆矩阵 [真实k→检出0..3]:")
        for k, row in enumerate(a["mean_confusion"]):
            print(f"     真实 {k}: " + " ".join(f"{x:.2f}" for x in row))
        print(f"   阈值敏感性: " + "; ".join(
            f"sens≥{th}: 域{v['domains']}/簇{v['clusters']}"
            for th, v in a["sens_sensitivity"].items()))
    print(f"\n任一锚点放行 = {res['any_proceed']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    print("[硬前置]")
    lk = F.verify_lock()
    if not a.aggregate_only:
        names, mats, cl, sigC, fout = F.universe()
        assert len(names) == 144 and len(set(cl.values())) == 90, \
            (len(names), len(set(cl.values())))
        print(f"  判别集 {len(names)} 个域 / {len(set(cl.values()))} 个簇  OK")
        print("[自检] H1–H6")
        log = F.selftest(names, mats)
        os.makedirs(PARTS, exist_ok=True)
        json.dump({"selftest": log, **lk},
                  open(os.path.join(ROOT, "cache",
                                    "foldrank_selftest.json"), "w"),
                  default=float)
        todo = names[:a.limit] if a.limit else names
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init) as ex:
            for i, (nm, st) in enumerate(ex.map(power_one, todo,
                                                chunksize=1)):
                print(f"  [{i + 1}/{len(todo)}] {nm}  {st}  "
                      f"(累计 {time.time() - t0:.0f}s)", flush=True)
    if not a.limit:
        aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
