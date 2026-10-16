#!/usr/bin/env python3
"""docs/43 Step 6：三个预先登记的校正候选（K1 joint-offset、K2 tolerance δ=0.02、K3 顺序 bootstrap）
先在植入数据上测（--gc），通过判据者才在真实矩阵上使用（--apply）。

先决：预先登记锁；cache/overshoot_harness.json ok=true。
产物：cache/overshoot_gc.json、cache/overshoot_apply.json
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
G4 = ("animalACE2/Eta/deer", "PDCoV/PDCoV/gAPN", "CR9114/WT_H1",
      "SKEMPI/OMTKY3/pooled-4proteases")
G2 = ("PDCoV/PDCoV/gAPN", "SKEMPI/OMTKY3/pooled-4proteases")
LEVELS = (1.0, 1.5, 2.5, 4.0, 6.0, 9.0)
LEVELS_K3 = (1.5, 2.5, 4.0, 9.0)
RG, RG3 = 30, 16
PASS_H_LEVELS = (2.5, 4.0, 6.0, 9.0)
PASS_H_LEVELS_K3 = (2.5, 4.0, 9.0)


def gates():
    OL.verify_lock()
    h = json.load(open(os.path.join(ROOT, "cache", "overshoot_harness.json")))
    assert h.get("ok") is True, "双向 harness 未通过 —— 中止"


def gc(jobs):
    gates()
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    OL.set_geoms({k: geoms[k] for k in G4})
    T = []
    for nm in G4:
        sg = geoms[nm].fit[1]["sigma"]
        for rep in range(RG):
            T.append(dict(mat=nm, k=0, mode="rand", s=0.0, sigma=sg, noise="gauss", rep=rep,
                          tag="gc_k0", clip=False, method="all"))
        for lv in LEVELS:
            for k in (1, 2):
                for rep in range(RG):
                    T.append(dict(mat=nm, k=k, mode="rand", s=lv * sg, sigma=sg, noise="gauss",
                                  rep=rep, tag=f"gc_k{k}_{lv}", clip=False, method="all"))
    T3 = []
    for nm in G2:
        sg = geoms[nm].fit[1]["sigma"]
        for rep in range(RG3):
            T3.append(dict(mat=nm, k=0, mode="rand", s=0.0, sigma=sg, noise="gauss", rep=rep,
                           tag="gc3_k0", clip=False, method="k3"))
        for lv in LEVELS_K3:
            for k in (1, 2):
                for rep in range(RG3):
                    T3.append(dict(mat=nm, k=k, mode="rand", s=lv * sg, sigma=sg, noise="gauss",
                                   rep=rep, tag=f"gc3_k{k}_{lv}", clip=False, method="k3"))
    print(f"  GC 任务：K1/K2/基线 {len(T)}，K3 {len(T3)}", flush=True)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        res = pool.map(OL.run_task, T3 + T, chunksize=2)      # 先放最重的
    print(f"  用时 {time.time()-t0:.0f}s", flush=True)
    res3 = [r for r in res if r["tag"].startswith("gc3")]
    res1 = [r for r in res if r["tag"].startswith("gc_")]

    def pooled(rs, kfield, k, lv, tag_prefix):
        tag = f"{tag_prefix}_k{k}_{lv}"
        return [r[kfield] for r in rs if r["tag"] == tag]

    out = {"levels": list(LEVELS), "levels_k3": list(LEVELS_K3), "R": RG, "R3": RG3, "methods": {}}
    for meth, kf, rs, lvls, pfx, hl in (("baseline", "det", res1, LEVELS, "gc", PASS_H_LEVELS),
                                        ("K1", "det_k1", res1, LEVELS, "gc", PASS_H_LEVELS),
                                        ("K2", "det_k2", res1, LEVELS, "gc", PASS_H_LEVELS),
                                        ("baseline_on_K3_geoms", "det", res3, LEVELS_K3, "gc3", PASS_H_LEVELS_K3),
                                        ("K3", "det_k3", res3, LEVELS_K3, "gc3", PASS_H_LEVELS_K3)):
        m = {"levels": {}}
        z = [r[kf] for r in rs if r["tag"] == f"{pfx}_k0"]
        m["null_is0"] = [int(sum(d == 0 for d in z)), len(z)]
        for lv in lvls:
            d1 = pooled(rs, kf, 1, lv, pfx)
            d2 = pooled(rs, kf, 2, lv, pfx)
            m["levels"][str(lv)] = {
                "FR2": [int(sum(d >= 2 for d in d1)), len(d1)],
                "H": [int(sum(d == 2 for d in d2)), len(d2)],
                "k1_dist": [int(sum(d == c for d in d1)) for c in range(4)],
                "k2_dist": [int(sum(d == c for d in d2)) for c in range(4)]}
        a = all(v["FR2"][0] / v["FR2"][1] <= 0.15 for v in m["levels"].values())
        b = all(m["levels"][str(l)]["H"][0] / m["levels"][str(l)]["H"][1] >= 0.70 for l in hl)
        c = m["null_is0"][0] / m["null_is0"][1] >= 0.90
        m["criteria"] = {"a_FR2<=0.15_all_levels": a, "b_H>=0.70_levels": b, "c_null>=0.90": c}
        m["pass"] = bool(a and b and c)
        out["methods"][meth] = m
        print(f"\n[{meth}]  null 检出0: {m['null_is0'][0]}/{m['null_is0'][1]}   通过={m['pass']} "
              f"(a={a}, b={b}, c={c})")
        print(f"   {'s/σ':>6}{'P(≥2|真1)':>14}{'P(=2|真2)':>14}   真1分布      真2分布")
        for lv in lvls:
            v = m["levels"][str(lv)]
            print(f"   {lv:>6}{v['FR2'][0]:>6}/{v['FR2'][1]:<4}={v['FR2'][0]/v['FR2'][1]:.2f}"
                  f"{v['H'][0]:>6}/{v['H'][1]:<4}={v['H'][0]/v['H'][1]:.2f}   {v['k1_dist']}  {v['k2_dist']}")
    out["n_corrections_tried"] = 3
    out["passed"] = [k for k in ("K1", "K2", "K3") if out["methods"][k]["pass"]]
    json.dump(out, open(os.path.join(ROOT, "cache", "overshoot_gc.json"), "w"), indent=1)
    print("\n通过 GC 的候选:", out["passed"])


_APPLY = {}


def apply_work(nm):
    passed = _APPLY["passed"]
    g = OL.GEOMS[nm]
    o = {"name": nm}
    d0, mc = OL.detect(g.M, g.mask)
    o["base"], o["base_curve"] = d0, mc
    if "K1" in passed:
        d1, mc1 = OL.detect_k1(g.M, g.mask)
        o["K1"], o["K1_curve"] = d1, mc1
    if "K2" in passed:
        o["K2"] = OL.tolerance_rank(mc)
    if "K3" in passed:
        d3, info = OL.detect_k3(g.M, g.mask, B=20, seed=OL.seed_of(nm, "k3real"))
        o["K3"], o["K3info"] = d3, info
    return o


def apply(jobs):
    gates()
    gcj = json.load(open(os.path.join(ROOT, "cache", "overshoot_gc.json")))
    passed = gcj["passed"]
    print("  通过 GC 的候选:", passed, flush=True)
    if not passed:
        print("  无候选通过 -> 按预先登记 §9 不得用于真实矩阵。", flush=True)
        json.dump({"passed": [], "note": "无候选通过 GC，未在真实矩阵上使用"},
                  open(os.path.join(ROOT, "cache", "overshoot_apply.json"), "w"), indent=1)
        return
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    names = OL.scope_names()
    OL.set_geoms(geoms)
    _APPLY["passed"] = passed
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        res = pool.map(apply_work, names, chunksize=1)
    json.dump({"passed": passed, "res": res}, open(os.path.join(ROOT, "cache", "overshoot_apply.json"), "w"),
              indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\n{'矩阵':<40}{'现行':>5}" + "".join(f"{k:>5}" for k in passed))
    for o in res:
        print(f"{o['name']:<40}{o['base']:>5}" + "".join(f"{o[k]:>5}" for k in passed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    if a.gc:
        gc(a.jobs)
    if a.apply:
        apply(a.jobs)
