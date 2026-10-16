#!/usr/bin/env python3
"""docs/44 Step 1：用**生产模块** rank_k1 复现 docs/43 的 K1 数字，先于任何新计算。

复现项（全部对照 docs/43 落盘的缓存与正文数字）：
  --gc     植入检验：K1 的假秩 2 率（pooled 1/720）、P(检出=2|真2) 111–120/120、null 120/120；
           基线（现行程序）同一批任务的计数（103/120 @ s/σ=4 等）。任务生成与种子与 overshoot_correct.gc 逐字相同，
           只把检测换成 rank_k1.headline。对照 cache/overshoot_gc.json 的计数逐格相等。
  --apply  18 个真实矩阵的头条口径检出秩：K1 12/18、现行 15/18；曲线与 cache/overshoot_apply.json 逐位一致。
  --boot   S1 的 15 个矩阵秩间增益（位点联合重抽 2000 次，seed 44）：对照 docs/43 T10。

先决：docs/43 预先登记锁（几何与任务定义来自它）；rank_k1.selfcheck() 必须先过。
产物：cache/k1_reproduce.json
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_k1 as K1                              # noqa: E402
import overshoot_lib as OL                        # noqa: E402

ROOT = OL.ROOT
OUT = os.path.join(ROOT, "cache", "k1_reproduce.json")
G4 = ("animalACE2/Eta/deer", "PDCoV/PDCoV/gAPN", "CR9114/WT_H1", "SKEMPI/OMTKY3/pooled-4proteases")
LEVELS = (1.0, 1.5, 2.5, 4.0, 6.0, 9.0)
RG = 30


def _load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def _save(d):
    json.dump(d, open(OUT, "w"), indent=1, default=float)


def gc_task(t):
    g = OL.GEOMS[t["mat"]]
    rng = np.random.default_rng(OL.seed_of(t["mat"], t["tag"], t["rep"]))
    Y, _ = g.make(t["k"], t["mode"], t["s"], t["sigma"], t["noise"], rng, clip=False)
    o = dict(t)
    o["det_old"] = K1.headline(Y, g.mask, "noleak")["rank"]
    o["det_k1"] = K1.headline(Y, g.mask, "k1")["rank"]
    return o


def do_gc(jobs):
    OL.verify_lock()
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    OL.set_geoms({k: geoms[k] for k in G4})
    T = []
    for nm in G4:
        sg = geoms[nm].fit[1]["sigma"]
        for rep in range(RG):
            T.append(dict(mat=nm, k=0, mode="rand", s=0.0, sigma=sg, noise="gauss", rep=rep, tag="gc_k0"))
        for lv in LEVELS:
            for k in (1, 2):
                for rep in range(RG):
                    T.append(dict(mat=nm, k=k, mode="rand", s=lv * sg, sigma=sg, noise="gauss", rep=rep,
                                  tag=f"gc_k{k}_{lv}"))
    print(f"  GC 任务 {len(T)}（每个含 现行+K1 两次 3 种子检出）", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(jobs) as pool:
        res = pool.map(gc_task, T, chunksize=2)
    print(f"  用时 {time.time()-t0:.0f}s", flush=True)
    ref = json.load(open(os.path.join(ROOT, "cache", "overshoot_gc.json")))["methods"]
    out = {"methods": {}, "n_tasks": len(T)}
    allmatch = True
    for meth, field, refkey in (("baseline", "det_old", "baseline"), ("K1", "det_k1", "K1")):
        m = {"levels": {}}
        z = [r[field] for r in res if r["tag"] == "gc_k0"]
        m["null_is0"] = [int(sum(d == 0 for d in z)), len(z)]
        for lv in LEVELS:
            d1 = [r[field] for r in res if r["tag"] == f"gc_k1_{lv}"]
            d2 = [r[field] for r in res if r["tag"] == f"gc_k2_{lv}"]
            m["levels"][str(lv)] = {"FR2": [int(sum(d >= 2 for d in d1)), len(d1)],
                                    "H": [int(sum(d == 2 for d in d2)), len(d2)]}
        # 对照 docs/43 缓存
        rm = ref[refkey]
        same = (m["null_is0"] == rm["null_is0"]) and all(
            m["levels"][str(lv)]["FR2"] == rm["levels"][str(lv)]["FR2"]
            and m["levels"][str(lv)]["H"] == rm["levels"][str(lv)]["H"] for lv in LEVELS)
        m["identical_to_docs43_cache"] = bool(same)
        allmatch &= same
        out["methods"][meth] = m
        print(f"\n[{meth}] null 检出 0: {m['null_is0']}   与 docs/43 缓存逐格相同: {same}")
        for lv in LEVELS:
            v = m["levels"][str(lv)]
            r = rm["levels"][str(lv)]
            print(f"   s/σ={lv:>4}  P(≥2|真1)={v['FR2'][0]}/{v['FR2'][1]} (docs43 {r['FR2'][0]})"
                  f"   P(=2|真2)={v['H'][0]}/{v['H'][1]} (docs43 {r['H'][0]})")
    k1 = out["methods"]["K1"]
    out["K1_pooled_false_rank2"] = [int(sum(v["FR2"][0] for v in k1["levels"].values())),
                                    int(sum(v["FR2"][1] for v in k1["levels"].values()))]
    out["K1_recovery_range"] = [min(v["H"][0] for v in k1["levels"].values()),
                                max(v["H"][0] for v in k1["levels"].values()), RG * 4]
    out["all_identical_to_docs43"] = bool(allmatch)
    print(f"\nK1 假秩 2 pooled = {out['K1_pooled_false_rank2']}（docs/43：1/720）；"
          f"真 2 检出范围 {out['K1_recovery_range'][0]}–{out['K1_recovery_range'][1]}/{RG*4}（docs/43：111–120/120）；"
          f"null {k1['null_is0']}（docs/43：120/120）")
    assert out["K1_pooled_false_rank2"] == [1, 720], "未复现 1/720"
    assert out["K1_recovery_range"][:2] == [111, 120], "未复现 111–120/120"
    assert k1["null_is0"] == [120, 120], "未复现 null 120/120"
    d = _load()
    d["gc"] = out
    _save(d)
    return out


def apply_work(nm):
    g = OL.GEOMS[nm]
    return {"name": nm, "k1": K1.headline(g.M, g.mask, "k1"), "old": K1.headline(g.M, g.mask, "noleak")}


def do_apply(jobs):
    OL.verify_lock()
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    names = OL.scope_names()
    OL.set_geoms(geoms)
    with mp.get_context("fork").Pool(jobs) as pool:
        res = pool.map(apply_work, names, chunksize=1)
    ref = {o["name"]: o for o in json.load(open(os.path.join(ROOT, "cache", "overshoot_apply.json")))["res"]}
    rows, maxd = [], 0.0
    for r in res:
        o = ref[r["name"]]
        d1 = max(abs(r["k1"]["mean_curve"][k] - o["K1_curve"][str(k)]) for k in range(4))
        d0 = max(abs(r["old"]["mean_curve"][k] - o["base_curve"][str(k)]) for k in range(4))
        maxd = max(maxd, d1, d0)
        assert r["k1"]["rank"] == o["K1"] and r["old"]["rank"] == o["base"], (r["name"], r["k1"]["rank"], o["K1"])
        rows.append({"name": r["name"], "old": r["old"]["rank"], "k1": r["k1"]["rank"],
                     "k1_curve": r["k1"]["mean_curve"], "old_curve": r["old"]["mean_curve"]})
    n2k = sum(r["k1"] == 2 for r in rows)
    n2o = sum(r["old"] == 2 for r in rows)
    print(f"\n18 个矩阵：现行 {n2o}/18 为秩 2（docs/43：15），K1 {n2k}/18（docs/43：12）；"
          f"曲线与 docs/43 缓存 max|Δ| = {maxd:.1e}")
    assert (n2o, n2k) == (15, 12) and maxd < 1e-9
    d = _load()
    d["apply"] = {"rows": rows, "n_rank2_old": n2o, "n_rank2_k1": n2k, "max_abs_curve_diff": maxd}
    _save(d)


def do_boot():
    """S1 15 矩阵 K1 与现行的秩间增益，位点联合重抽 2000 次，seed 44（与 overshoot_k1_boot 同）。"""
    import independence_audit as IA
    OL.verify_lock()
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    s1 = [n for n in OL.scope_names() if n.startswith("animalACE2/")]
    ref = json.load(open(os.path.join(ROOT, "cache", "overshoot_k1_boot.json")))
    out = {}
    for lab, fn in (("base", K1.cv_cell_errors_noleak), ("K1", K1.cv_cell_errors_k1)):
        us = []
        for nm in s1:
            g = geoms[nm]
            acc = None
            for sd in K1.SEEDS_HEADLINE:
                cells, E = fn(g.M, g.mask, max_rank=3, seed=sd, **{k: v for k, v in K1.CV_HEADLINE.items() if k != "K"})
                acc = {r: E[r].copy() for r in E} if acc is None else {r: acc[r] + E[r] for r in E}
            us.append({"sites": list(OL.pick(nm)["sites"]), "rows": cells[:, 0],
                       "E": {r: acc[r] / len(K1.SEEDS_HEADLINE) for r in acc}})
        pairs = ((0, 1), (1, 2), (2, 3), (0, 2))
        sb = IA.site_boot_gains(us, 2000, seed=44, pairs=pairs)
        o = {}
        for (a, b) in pairs:
            pts = [(K1.trim95(u["E"][a]) - K1.trim95(u["E"][b])) / K1.trim95(u["E"][a]) for u in us]
            med = np.median(sb[(a, b)], axis=1)
            o[f"g{a}{b}"] = [float(np.median(pts)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
            o[f"g{a}{b}_pos"] = int(sum(p > 0 for p in pts))
            rr = ref[lab]["S1"][f"g{a}{b}"]
            assert max(abs(x - y) for x, y in zip(o[f"g{a}{b}"], rr)) < 1e-9, (lab, a, b, o[f"g{a}{b}"], rr)
        out[lab] = o
    f = lambda x: f"{100*x[0]:+.1f}% [{100*x[1]:+.1f}%, {100*x[2]:+.1f}%]"
    for lab in ("base", "K1"):
        print(f"S1(15) {lab}: g01 {f(out[lab]['g01'])}  g12 {f(out[lab]['g12'])}  g23 {f(out[lab]['g23'])}"
              f"  g02 {f(out[lab]['g02'])}   g12>0 的矩阵 {out[lab]['g12_pos']}/15")
    d = _load()
    d["boot"] = out
    _save(d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    K1.RC.selfcheck(verbose=False)
    K1.selfcheck(verbose=False)
    print("  rank_core.selfcheck + rank_k1.selfcheck 通过", flush=True)
    if a.apply:
        do_apply(a.jobs)
    if a.boot:
        do_boot()
    if a.gc:
        do_gc(a.jobs)
