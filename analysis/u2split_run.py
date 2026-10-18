#!/usr/bin/env python3
"""docs/45 主流程：位点分半的第二成分复现性。

预先登记：cache/docs45_prereg.md（sha256 锁在 cache/docs45_prereg_lock.json），入口断言一致。
阶段（顺序即纪律：先工具自检 -> 零分布 -> 植入世界 -> 门 -> 才跑真实矩阵）：
  selftest  先决自检（预先登记 §8），不过即中止
  prep      读几何缓存与位点元数据
  perm      N-perm 零分布（每矩阵 × 抽取器）
  worlds    植入世界（并行；断点续跑；cache/u2split/worlds/*.npz）
  gate      H1/H2 门；不过则 real 阶段拒绝运行
  real      真实矩阵的随机分半 + 连续分半
  cross     docs/14 的 12 个组合：Step 2 复现 0.983 + A/B/C/D 共享/不共享位点对照
  share     第二成分的方差占比 + 校准
用法：python3 analysis/u2split_run.py <stage> [--jobs 9]
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
import u2split_lib as U                         # noqa: E402

OUT = os.path.join(ROOT, "cache", "u2split")
WDIR = os.path.join(OUT, "worlds")
os.makedirs(WDIR, exist_ok=True)
GEOMS_PKL = os.path.join(ROOT, "cache", "overshoot_geoms.pkl")
NOISE_JSON = os.path.join(ROOT, "cache", "overshoot_noise.json")

S1 = [f"animalACE2/{b}/{l}" for b in ("Wuhan-Hu-1", "Eta", "Omicron_BA2")
      for l in ("deer", "hamster", "bat", "cat", "human_monomer")]
GAPN = "PDCoV/PDCoV/gAPN"
SCOPE = S1 + [GAPN]
NO_PATH = ["animalACE2/Omicron_BA2/bat"]
PATHM = [n for n in SCOPE if n not in NO_PATH]
EXTS = U.EXTRACTORS
DOSE_LAMS = (0.0, 0.25, 0.5, 1.0, 2.0)
DOSE_POINT = 3          # path_k2 的第 4 点（中点）


def P(*a):
    print(*a, flush=True)


def load_geoms():
    g = pickle.load(open(GEOMS_PKL, "rb"))
    noise = json.load(open(NOISE_JSON))["matrices"]
    for n in SCOPE:
        assert n in g and n in noise, n
        assert g[n].nr == noise[n]["nr"] and g[n].n == noise[n]["n"], f"{n} 几何缓存与噪声缓存不一致"
    return g, noise


def jdump(obj, path):
    json.dump(obj, open(path, "w"), ensure_ascii=False, indent=1)


def preflight():
    U.verify_lock()
    lines = []
    for s in ("rank_core.selfcheck", "rank_k1.selfcheck"):
        pass
    import rank_core as RC
    import rank_k1 as K1
    RC.selfcheck(verbose=False)
    P("  rank_core.selfcheck 全过  OK")
    K1.selfcheck(verbose=False)
    P("  rank_k1.selfcheck 全过  OK")
    gh = json.load(open(os.path.join(ROOT, "cache", "gate_harness_check.json")))
    assert gh.get("ok") is True, "cache/gate_harness_check.json ok != true —— 硬前提不满足"
    P("  cache/gate_harness_check.json ok = true  OK")


# ======================================================================
def stage_selftest(a):
    P("=" * 78 + "\n先决自检\n" + "=" * 78)
    preflight()
    for sc in ("subspace_selftest.py", "censor_density_check.py", "gate_harness_check.py"):
        r = subprocess.run([sys.executable, os.path.join(HERE, sc)], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"{sc} 未通过:\n{r.stdout[-1500:]}\n{r.stderr[-800:]}"
        P(f"  {sc} 通过  OK")
    gh = json.load(open(os.path.join(ROOT, "cache", "gate_harness_check.json")))
    assert gh.get("ok") is True
    g, noise = load_geoms()
    log = U.selfcheck({k: g[k] for k in ("animalACE2/Wuhan-Hu-1/human_monomer", "animalACE2/Eta/hamster")})
    jdump({"log": log, "prereg": U.verify_lock(False)}, os.path.join(OUT, "selftest.json"))
    P("先决自检全部通过")


def need_selftest():
    p = os.path.join(OUT, "selftest.json")
    assert os.path.exists(p), "先跑 selftest 阶段（CLAUDE.md §10：不过就中止）"
    U.verify_lock(False)


# ======================================================================
def stage_prep(a):
    need_selftest()
    import overshoot_lib as OL
    meta = {}
    csvp = os.path.join(ROOT, "cache", "dms", "starr_animalACE2_final_variant_scores.csv")
    wt = {}
    for r in csv.DictReader(open(csvp)):
        wt.setdefault(r["target"], {})[int(r["position"])] = r["wildtype"]
    for n in SCOPE:
        d = OL.pick(n)
        sites = [int(s) for s in d["sites"]]
        meta[n] = {"sites": sites, "nr": len(sites)}
        if n.startswith("animalACE2/"):
            bg = n.split("/")[1]
            meta[n]["wt"] = [wt[bg].get(s) for s in sites]
    bgs = ("Wuhan-Hu-1", "Eta", "Omicron_BA2")
    allpos = set.intersection(*[set(wt[b]) for b in bgs])
    diff = {}
    for x, y in (("Wuhan-Hu-1", "Eta"), ("Wuhan-Hu-1", "Omicron_BA2"), ("Eta", "Omicron_BA2")):
        diff[f"{x}|{y}"] = sorted(p for p in allpos if wt[x][p] != wt[y][p])
    meta["_diff_sites"] = diff
    P("  骨架间 wildtype 不同的位点数:", {k: len(v) for k, v in diff.items()})
    jdump(meta, os.path.join(OUT, "meta.json"))


def load_meta():
    return json.load(open(os.path.join(OUT, "meta.json")))


# ======================================================================
def _perm_task(args):
    n, ext = args
    g = _G[n]
    arr = U.perm_null(g.M, g.mask, ext, U.N_PERM_MAT, U.S_PERM, U.seed_of("perm", n, ext))
    return n, ext, arr


_G = {}
_NOISE = {}


def _init_worker():
    global _G, _NOISE
    _G, _NOISE = load_geoms()


def stage_perm(a):
    need_selftest()
    t0 = time.time()
    tasks = [(n, e) for n in SCOPE for e in EXTS]
    with mp.Pool(a.jobs, initializer=_init_worker) as pool:
        res = pool.map(_perm_task, tasks, chunksize=1)
    out = {}
    for n, e, arr in res:
        out.setdefault(n, {})[e] = arr
    pickle.dump(out, open(os.path.join(OUT, "perm.pkl"), "wb"))
    P(f"  N-perm 完成（{U.N_PERM_MAT} 矩阵 × {U.S_PERM} 分半 × {len(tasks)} 个任务）{time.time() - t0:.0f}s")


# ======================================================================
def cell_list():
    cells = []
    for n in PATHM:
        for p in range(7):
            cells.append((n, "R1", p, None))
            cells.append((n, "R2", p, None))
        cells.append((n, "R0", 0, None))
        cells.append((n, "R1P", 0, None))
        cells.append((n, "R2P", 0, None))
        for lam in DOSE_LAMS:
            cells.append((n, "R2L", DOSE_POINT, lam))
    return cells


def cell_path(c):
    n, kind, p, lam = c
    tag = f"{n.replace('/', '_')}__{kind}__{p}__{'' if lam is None else lam}"
    return os.path.join(WDIR, tag + ".npz")


def _cell_task(c):
    path = cell_path(c)
    if os.path.exists(path):
        return c, True
    n, kind, p, lam = c
    g = _G[n]
    nz = _NOISE[n]
    point = None
    if kind in ("R1", "R2L", "R2"):
        pk = nz["path_k1"] if kind == "R1" else nz["path_k2"]
        point = pk[p]
    R, S = U.R_WORLD, U.S_WORLD
    out = np.full((R, S, len(EXTS), 3), np.nan)
    for rep in range(R):
        rng = np.random.default_rng(U.seed_of("world", n, kind, p, lam, rep))
        Y = U.make_world(g, kind, rng, point=point, lam=lam)
        for ie, ext in enumerate(EXTS):
            out[rep, :, ie, :] = U.many_splits(Y, g.mask, ext, S, U.seed_of("wsplit", n, kind, p, lam, rep, ext))
    np.savez_compressed(path, cos=out)
    return c, False


def stage_worlds(a):
    need_selftest()
    cells = cell_list()
    todo = [c for c in cells if not os.path.exists(cell_path(c))]
    P(f"  植入世界单元 {len(cells)}，待算 {len(todo)}（R={U.R_WORLD} 复现 × {U.S_WORLD} 分半 × 2 抽取器）")
    t0 = time.time()
    done = 0
    with mp.Pool(a.jobs, initializer=_init_worker) as pool:
        for c, cached in pool.imap_unordered(_cell_task, todo, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(todo):
                P(f"    {done}/{len(todo)}  {time.time() - t0:.0f}s")


def load_world(n, kind, p, lam=None):
    return np.load(cell_path((n, kind, p, lam)))["cos"]     # (R,S,ext,3)


# ======================================================================
def gate_eval():
    """预先登记 §8 的 H1/H2。返回 dict。"""
    perm = pickle.load(open(os.path.join(OUT, "perm.pkl"), "rb"))
    res = {"H1": {}, "H2": {}}
    exceed = {e: [] for e in EXTS}
    sep = {e: {} for e in EXTS}
    for n in PATHM:
        for ie, e in enumerate(EXTS):
            p95 = float(np.nanquantile(perm[n][e][:, 1], 0.95))
            w0 = load_world(n, "R0", 0)[:, :, ie, 1]
            exceed[e].append(float(np.nanmean(w0 > p95)))
            ok = False
            best = None
            for p in range(7):
                y1 = np.nanmedian(load_world(n, "R1", p)[:, :, ie, 1], axis=1)
                y2 = np.nanmedian(load_world(n, "R2", p)[:, :, ie, 1], axis=1)
                if np.median(y2) > np.quantile(y1, 0.95):
                    ok = True
            sep[e][n] = ok
    for e in EXTS:
        res["H1"][e] = {"pooled_frac_exceed_perm_p95": float(np.mean(exceed[e])),
                        "pass": bool(np.mean(exceed[e]) <= 0.10),
                        "per_matrix": dict(zip(PATHM, exceed[e]))}
        s1 = [sep[e][n] for n in PATHM if n in S1]
        res["H2"][e] = {"separable": sep[e], "frac_S1": float(np.mean(s1)),
                        "pass": bool(np.mean(s1) >= 2 / 3)}
    res["pass"] = all(res["H1"][e]["pass"] and res["H2"][e]["pass"] for e in EXTS)
    return res


def stage_gate(a):
    need_selftest()
    res = gate_eval()
    jdump(res, os.path.join(OUT, "gate.json"))
    for e in EXTS:
        P(f"  [{e}] H1 假警报：R0 世界 cos θ₂ 超过置换 p95 的比例（跨矩阵汇总）{res['H1'][e]['pooled_frac_exceed_perm_p95']:.3f}  "
          f"{'通过' if res['H1'][e]['pass'] else '未通过'}")
        P(f"  [{e}] H2 命中：S1 路径矩阵里可分辨者 {res['H2'][e]['frac_S1']:.2f}  {'通过' if res['H2'][e]['pass'] else '未通过'}")
    P("  门总判定:", "通过" if res["pass"] else "**未通过 —— 真实矩阵结果不得解释，判决 CANNOT_ASSESS**")


# ======================================================================
def contiguous_split(M, mask, sites, ext, seed):
    order = np.argsort(sites)
    h = len(order) // 2
    ia, ib = np.sort(order[:h]), np.sort(order[h:2 * h])
    return U.split_pair_cos(M, mask, ia, ib, ext, seed)


def _real_task(args):
    n, ext = args
    g = _G[n]
    meta = load_meta()
    arr = U.many_splits(g.M, g.mask, ext, U.S_REAL, U.seed_of("real", n, ext))
    cont = contiguous_split(g.M, g.mask, np.array(meta[n]["sites"]), ext, U.seed_of("cont", n, ext))
    return n, ext, arr, cont


def stage_real(a):
    need_selftest()
    gp = os.path.join(OUT, "gate.json")
    assert os.path.exists(gp), "先跑 gate 阶段"
    gate = json.load(open(gp))
    assert gate["pass"], "门未通过 —— 按预先登记 §8 不跑/不解释真实矩阵"
    t0 = time.time()
    tasks = [(n, e) for n in SCOPE for e in EXTS]
    with mp.Pool(a.jobs, initializer=_init_worker) as pool:
        res = pool.map(_real_task, tasks, chunksize=1)
    out = {}
    for n, e, arr, cont in res:
        out.setdefault(n, {})[e] = {"splits": arr, "contig": cont}
    pickle.dump(out, open(os.path.join(OUT, "real.pkl"), "wb"))
    P(f"  真实分半完成 {time.time() - t0:.0f}s")


# ======================================================================
def stage_cross(a):
    need_selftest()
    import site_loadings as SL
    import predict_site_loadings as PS
    d, combos = PS.load_targets()
    cons, used = PS.consensus(combos)
    # ---- Step 2：已知事实门——复现 docs/14 的 0.983 / 0.826 / 0.988 / 0.871
    rel = {}
    for k in ("u1", "u2"):
        Z = cons[k]["Z"]
        kk = Z.shape[0]
        rs = [float(np.corrcoef(Z[i], Z[j])[0, 1]) for i in range(kk) for j in range(i + 1, kk)]
        rb = float(np.mean(rs))
        rel[k] = {"rbar": rb, "sb": kk * rb / (1 + (kk - 1) * rb), "k": kk, "n_sites": int(Z.shape[1]),
                  "r_min": min(rs), "r_max": max(rs)}
    want = {"u1": (0.871, 0.988), "u2": (0.826, 0.983)}
    for k in ("u1", "u2"):
        assert abs(rel[k]["rbar"] - want[k][0]) < 0.0006 and abs(rel[k]["sb"] - want[k][1]) < 0.0006, \
            f"未能复现 docs/14 的 {k} 信度: {rel[k]} vs {want[k]}"
    P(f"  Step 2 复现 docs/14：u₂ 两两 r̄ {rel['u2']['rbar']:.3f}、共识 {rel['u2']['sb']:.3f}；"
      f"u₁ {rel['u1']['rbar']:.3f} / {rel['u1']['sb']:.3f}（{rel['u2']['k']} 组合 × {rel['u2']['n_sites']} 位点）  OK")

    rows = SL.load_rows()
    wtmap = {}
    for r in rows:
        wtmap.setdefault(r["target"], {})[int(r["position"])] = r["wildtype"]
    mats = {}
    for c in combos:
        tgt, lig = c["target"], c["ligand"]
        sites, res, M, mask, cf = SL.build_matrix(rows, tgt, lig)
        M20 = np.zeros((len(sites), 20))
        k20 = np.zeros((len(sites), 20), bool)
        for jj, aa in enumerate(res):
            M20[:, U.AA20.index(aa)] = np.where(mask[:, jj], M[:, jj], 0.0)
            k20[:, U.AA20.index(aa)] = mask[:, jj]
        mats[(tgt, lig)] = {"sites": np.array(sites), "M": M20, "mask": k20}
    keys = list(mats)
    assert len(keys) == 12
    diff = {}
    for x, y in (("Wuhan-Hu-1", "Eta"), ("Wuhan-Hu-1", "Omicron_BA2"), ("Eta", "Omicron_BA2")):
        allp = set(wtmap[x]) & set(wtmap[y])
        diff[(x, y)] = sorted(p for p in allp if wtmap[x][p] != wtmap[y][p])
        diff[(y, x)] = diff[(x, y)]
    P("  骨架间 wildtype 不同的位点:", {f"{x}|{y}": len(v) for (x, y), v in diff.items() if x < y})

    def sub(m, sites_sel):
        idx = np.searchsorted(m["sites"], sites_sel)
        return m["M"][idx], m["mask"][idx]

    pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            pairs.append((keys[i], keys[j]))
    results = {"rel": rel, "pairs": [], "self": {}}
    rng = np.random.default_rng(U.seed_of("cross"))
    t0 = time.time()
    for X, Y in pairs:
        common = np.intersect1d(mats[X]["sites"], mats[Y]["sites"])
        rec = {"X": list(X), "Y": list(Y), "n_common": int(len(common))}
        excl = None
        if X[0] != Y[0]:
            ban = set(diff[(X[0], Y[0])])
            excl = np.array([s for s in common if s not in ban])
        for variant, sset in (("all", common), ("excl_diff", excl)):
            if sset is None or len(sset) < 40:
                continue
            for ext in EXTS:
                MX, kX = sub(mats[X], sset)
                MY, kY = sub(mats[Y], sset)
                A = U.compare(U.extract(ext, MX, kX, 1), U.extract(ext, MY, kY, 2))
                Bs, Cs, Ds = [], [], []
                for s in range(8):
                    ia, ib = U.split_sites(len(sset), rng)
                    xa = U.extract(ext, MX[ia], kX[ia], 3)
                    xb = U.extract(ext, MX[ib], kX[ib], 4)
                    ya = U.extract(ext, MY[ia], kY[ia], 5)
                    yb = U.extract(ext, MY[ib], kY[ib], 6)
                    cmp_ = lambda p, q: U.compare(p, q)
                    b = [cmp_(xa, ya), cmp_(xb, yb)]
                    c = [cmp_(xa, yb), cmp_(xb, ya)]
                    dd = [cmp_(xa, xb), cmp_(ya, yb)]
                    Bs.append(np.mean([[v[0], v[1]] for v in b], axis=0))
                    Cs.append(np.mean([[v[0], v[1]] for v in c], axis=0))
                    Ds.append(np.mean([[v[0], v[1]] for v in dd], axis=0))
                rec[f"{variant}|{ext}"] = {
                    "A": [A[0], A[1]], "A_d": A[2],
                    "B": np.median(Bs, axis=0).tolist(), "C": np.median(Cs, axis=0).tolist(),
                    "D": np.median(Ds, axis=0).tolist()}
        results["pairs"].append(rec)
    P(f"  跨组合 A/B/C/D 完成 {time.time() - t0:.0f}s")
    pickle.dump(results, open(os.path.join(OUT, "cross.pkl"), "wb"))


# ======================================================================
def _share_real_task(n):
    g = _G[n]
    return n, {"standard": U.share_standard(g.M, g.mask), "k1": U.share_k1(g.M, g.mask)}


def _share_cal_task(args):
    n, kind, p, lam, reps = args
    g = _G[n]
    nz = _NOISE[n]
    point = None
    if kind in ("R1", "R2", "R2L"):
        point = (nz["path_k1"] if kind == "R1" else nz["path_k2"])[p]
    out = {"standard": [], "k1": [], "planted": None}
    if kind == "R2L":
        out["planted"] = U.planted_share2(g, point, lam)
    for rep in range(reps):
        rng = np.random.default_rng(U.seed_of("share", n, kind, p, lam, rep))
        Y = U.make_world(g, kind, rng, point=point, lam=lam)
        out["k1"].append(U.share_k1(Y, g.mask))
        if rep < 3:
            out["standard"].append(U.share_standard(Y, g.mask))
    return (n, kind, p, lam), out


def stage_share(a):
    need_selftest()
    gp = os.path.join(OUT, "gate.json")
    assert os.path.exists(gp) and json.load(open(gp))["pass"], "门未通过"
    t0 = time.time()
    with mp.Pool(a.jobs, initializer=_init_worker) as pool:
        real = dict(pool.map(_share_real_task, SCOPE, chunksize=1))
        tasks = []
        for n in PATHM:
            for p in (0, 3, 6):
                tasks.append((n, "R1", p, None, 20))
            tasks.append((n, "R1P", 0, None, 20))
            for lam in DOSE_LAMS:
                tasks.append((n, "R2L", DOSE_POINT, lam, 20))
        cal = dict(pool.map(_share_cal_task, tasks, chunksize=1))
    pickle.dump({"real": real, "cal": cal}, open(os.path.join(OUT, "share.pkl"), "wb"))
    P(f"  方差占比完成 {time.time() - t0:.0f}s")


# ======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["selftest", "prep", "perm", "worlds", "gate", "real", "cross", "share"])
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"selftest": stage_selftest, "prep": stage_prep, "perm": stage_perm, "worlds": stage_worlds,
     "gate": stage_gate, "real": stage_real, "cross": stage_cross, "share": stage_share}[a.stage](a)
