#!/usr/bin/env python3
"""docs/43 Step 4–5：在每个 binding 矩阵自己的几何、主效应、缺失模式与 σ 路径上植入真秩 0/1/2，
用头条同一程序检出，度量 P(检出≥2|真1)、P(检出=2|真2)、P(检出=0|真0)，并按预先登记 §7 判决。

先决：预先登记锁；cache/overshoot_harness.json ok=true（双向 harness 先过）；cache/overshoot_noise.json 存在。
用法：python3 analysis/overshoot_binding.py --sim [--jobs 9]   # 模拟，逐矩阵落盘 cache/overshoot_binding_parts/
      python3 analysis/overshoot_binding.py --aggregate          # 汇总与判决 -> cache/overshoot_binding.json
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overshoot_lib as OL                       # noqa: E402

ROOT = OL.ROOT
PARTS = os.path.join(ROOT, "cache", "overshoot_binding_parts")
R = OL.R_REP


def need_gates():
    OL.verify_lock()
    h = json.load(open(os.path.join(ROOT, "cache", "overshoot_harness.json")))
    assert h.get("ok") is True, "双向 harness 未通过 —— 按预先登记 §5 中止"
    print("  双向 harness ok=true  OK", flush=True)


def build_tasks(nm, g, noise):
    row = noise["matrices"][nm]
    lo = row["sigma_lo"]
    T = []
    for rep in range(R):
        T.append(dict(mat=nm, k=0, mode="rand", s=0.0, sigma=g.sigma0(), noise="gauss",
                      rep=rep, tag="null"))
    for k in (1, 2):
        path = g.path(k, lo)
        for p, pt in enumerate(path, start=1):
            if pt["s"] is None:
                continue
            for rep in range(R):
                T.append(dict(mat=nm, k=k, mode="real", s=pt["s"], sigma=pt["sigma"],
                              noise="gauss", rep=rep, tag=f"k{k}_real_p{p}"))
            if p in (1, 4, 7):
                s_r = g.s_for(k, "rand", pt["sigma"])
                if s_r is None:
                    continue
                for rep in range(R):
                    T.append(dict(mat=nm, k=k, mode="rand", s=s_r, sigma=pt["sigma"],
                                  noise="gauss", rep=rep, tag=f"k{k}_rand_p{p}"))
            if p == 1 and g.rep_kind == "measured":
                for rep in range(R):
                    T.append(dict(mat=nm, k=k, mode="real", s=pt["s"], sigma=pt["sigma"],
                                  noise="wild", rep=rep, tag=f"k{k}_wild_p1"))
    return T


def sim(jobs):
    need_gates()
    noise = json.load(open(os.path.join(ROOT, "cache", "overshoot_noise.json")))
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    OL.set_geoms(geoms)
    os.makedirs(PARTS, exist_ok=True)
    ctx = mp.get_context("fork")
    t0 = time.time()
    for nm in OL.scope_names():
        fn = os.path.join(PARTS, nm.replace("/", "__") + ".json")
        if os.path.exists(fn):
            print(f"  跳过（已有）{nm}", flush=True)
            continue
        g = geoms[nm]
        T = build_tasks(nm, g, noise)
        if not T:
            continue
        ts = time.time()
        with ctx.Pool(jobs) as pool:
            res = pool.map(OL.run_task, T, chunksize=4)
        slim = [{"tag": r["tag"], "rep": r["rep"], "det": r["det"],
                 "t0": r["curve"][0], "curve": [r["curve"][q] for q in range(4)],
                 "clip": r["clip"]} for r in res]
        json.dump({"mat": nm, "n_tasks": len(T), "res": slim}, open(fn, "w"))
        print(f"  {nm}: {len(T)} 次检出，{time.time()-ts:.0f}s（累计 {time.time()-t0:.0f}s）", flush=True)


def _rate(rs, f):
    k = int(sum(f(r["det"]) for r in rs))
    return k, len(rs), OL.wilson(k, len(rs))


def classify(nm, g, noise, parts):
    row = noise["matrices"][nm]
    tag = row["tag"]
    t0_real = row["real_curve"]["0"] if "0" in row["real_curve"] else row["real_curve"][0]
    byt = {}
    for r in parts["res"]:
        byt.setdefault(r["tag"], []).append(r)
    path1 = [p for p in g.path(1, row["sigma_lo"])]
    out = {"name": nm, "tag": tag, "sigma_lo": row["sigma_lo"], "sigma_fit1": row["sigma_fit1"],
           "cso": row["cso"], "path": []}
    if not path1:
        out["verdict"] = "CANNOT_ASSESS"
        out["reason"] = "σ 下端 >= σ_fit1（路径为空）：重复噪声估计已不小于整个拟合后残差"
        return out
    for p, pt in enumerate(path1, start=1):
        rs = byt.get(f"k1_real_p{p}")
        if rs is None:
            out["path"].append({"p": p, "sigma": pt["sigma"], "s": pt["s"], "snr": pt["snr"],
                                "skipped": "cso 不相容"})
            continue
        k, n, ci = _rate(rs, lambda d: d >= 2)
        q = float(np.median([r["t0"] for r in rs]) / t0_real)
        e = {"p": p, "sigma": pt["sigma"], "s": pt["s"], "snr": pt["snr"], "FR2": [k, n], "FR2_ci": ci,
             "q": q, "compat": bool(OL.COMPAT[0] <= q <= OL.COMPAT[1]),
             "k1_dist": [int(sum(r["det"] == c for r in rs)) for c in range(4)],
             "clip": float(np.mean([r["clip"] for r in rs]))}
        out["path"].append(e)
    pts = [e for e in out["path"] if "FR2" in e]
    # k=2 路径
    p2 = g.path(2, row["sigma_lo"])
    out["path2"] = []
    for p, pt in enumerate(p2, start=1):
        rs = byt.get(f"k2_real_p{p}")
        if rs is None:
            continue
        k, n, ci = _rate(rs, lambda d: d == 2)
        q = float(np.median([r["t0"] for r in rs]) / t0_real)
        out["path2"].append({"p": p, "sigma": pt["sigma"], "s": pt["s"], "snr": pt["snr"], "H": [k, n], "H_ci": ci,
                             "q": q, "compat": bool(OL.COMPAT[0] <= q <= OL.COMPAT[1]),
                             "k2_dist": [int(sum(r["det"] == c for r in rs)) for c in range(4)]})
    rs0 = byt.get("null", [])
    out["null_is0"] = _rate(rs0, lambda d: d == 0)[:2]
    out["null_dist"] = [int(sum(r["det"] == c for r in rs0)) for c in range(4)]
    for key in ("rand", "wild"):
        for k in (1, 2):
            for p in ((1, 4, 7) if key == "rand" else (1,)):
                rs = byt.get(f"k{k}_{key}_p{p}")
                if rs is None:
                    continue
                f = (lambda d: d >= 2) if k == 1 else (lambda d: d == 2)
                kk, n, ci = _rate(rs, f)
                out.setdefault("variants", {})[f"k{k}_{key}_p{p}"] = {
                    "rate": [kk, n], "ci": ci,
                    "dist": [int(sum(r["det"] == c for r in rs)) for c in range(4)]}
    # ---- 判决（预先登记 §7）
    comp = [e for e in pts if e["compat"]]
    infl = [e for e in comp if e["FR2"][0] / e["FR2"][1] >= 0.5 and e["FR2_ci"][0] > 0.3]
    allrob = all(e["FR2_ci"][1] <= 0.20 for e in pts) and len(pts) == len(path1)
    comp_rob = bool(comp) and all(e["FR2_ci"][1] <= 0.20 for e in comp)
    if infl:
        v = "INFLATED"
    elif allrob:
        v = "ROBUST"
    elif comp_rob:
        v = "CONDITIONAL"
    else:
        v = "CANNOT_ASSESS"
    out["verdict"] = v
    out["n_compat"] = len(comp)
    if comp:
        near = min(comp, key=lambda e: abs(np.log(e["q"])))
        out["nearest_compat"] = near["p"]
    else:
        near = min(pts, key=lambda e: abs(np.log(e["q"])))
        out["nearest_q_point"] = near["p"]
    return out


def aggregate():
    need_gates()
    noise = json.load(open(os.path.join(ROOT, "cache", "overshoot_noise.json")))
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    res = {}
    for nm in OL.scope_names():
        fn = os.path.join(PARTS, nm.replace("/", "__") + ".json")
        row = noise["matrices"][nm]
        if not os.path.exists(fn):
            res[nm] = {"name": nm, "tag": row["tag"], "verdict": "CANNOT_ASSESS",
                       "reason": "σ 下端 >= σ_fit1（路径为空）" if not geoms[nm].path(1, row["sigma_lo"]) else "未模拟",
                       "sigma_lo": row["sigma_lo"], "sigma_fit1": row["sigma_fit1"], "cso": row["cso"]}
            continue
        parts = json.load(open(fn))
        res[nm] = classify(nm, geoms[nm], noise, parts)
    # 打印
    def fci(e):
        return f"{e['FR2'][0]}/{e['FR2'][1]}[{e['FR2_ci'][0]:.2f},{e['FR2_ci'][1]:.2f}]"
    print(f"\n{'矩阵':<38}{'tag':>2} {'判决':<14}{'p1: s/σ':>8}{'FR2@p1':>20}{'q1':>6}  {'p7: s/σ':>8}{'FR2@p7':>20}{'q7':>6}  {'最近相容点':>10}")
    for nm in OL.scope_names():
        r = res[nm]
        if "path" not in r or not r["path"]:
            print(f"{nm:<38}{r['tag']:>2} {r['verdict']:<14} {r.get('reason','')}")
            continue
        a, b = r["path"][0], r["path"][-1]
        nc = r.get("nearest_compat", None)
        print(f"{nm:<38}{r['tag']:>2} {r['verdict']:<14}{a['snr']:>8.2f}{fci(a):>20}{a['q']:>6.2f}  "
              f"{b['snr']:>8.2f}{fci(b):>20}{b['q']:>6.2f}  "
              f"{('p%d' % nc) if nc else '无(最近q点p%d)' % r.get('nearest_q_point', 0):>10}")
    json.dump(res, open(os.path.join(ROOT, "cache", "overshoot_binding.json"), "w"), indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    if a.sim:
        sim(a.jobs)
    if a.aggregate:
        aggregate()
