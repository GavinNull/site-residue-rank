#!/usr/bin/env python3
"""docs/46 Part A 主流程：忠实对照世界下的位点分半复现。

预先登记：cache/docs46_prereg.md（sha256 锁在 cache/docs46_prereg_lock.json），入口断言一致；
docs/45 的锁也断言（该预先登记文件不得被改动）。
阶段（顺序即纪律）：
  selftest  先决自检（不过即中止）
  fidelity  保真判据 + 它的牙齿 FT1–FT4（不过即中止；不放宽）
  perm      N-perm（新种子）
  worlds    植入世界 G / RR（并行，断点续跑）
  gate      H1/H2（每个世界）
  real      真实矩阵 300 个随机分半（新种子）+ 连续分半
  report    分类、判决、表格 -> cache/d46/results.json、tables.md
用法：python3 analysis/d46_run.py <stage> [--jobs 9]
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
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
import d46_lib as D                              # noqa: E402
import u2split_lib as U                          # noqa: E402
import u2split_run as RUN                        # noqa: E402

OUT = os.path.join(ROOT, "cache", "d46")
WDIR = os.path.join(OUT, "worlds")
os.makedirs(WDIR, exist_ok=True)
SCOPE, S1, PATHM, EXTS = RUN.SCOPE, RUN.S1, RUN.PATHM, RUN.EXTS
WORLDS = D.WORLDS


def P(*a):
    print(*a, flush=True)


def jdump(obj, path):
    json.dump(obj, open(path, "w"), ensure_ascii=False, indent=1,
              default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o))


def need_prereg():
    D.verify_lock46(False)
    U.verify_lock(False)


# ======================================================================
def stage_selftest(a):
    P("=" * 78 + "\n先决自检\n" + "=" * 78)
    D.verify_lock46()
    U.verify_lock()
    import rank_core as RC
    import rank_k1 as K1
    RC.selfcheck(verbose=False)
    P("  rank_core.selfcheck 全过  OK")
    K1.selfcheck(verbose=False)
    P("  rank_k1.selfcheck 全过  OK")
    for sc in ("subspace_selftest.py", "censor_density_check.py", "gate_harness_check.py"):
        r = subprocess.run([sys.executable, os.path.join(HERE, sc)], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"{sc} 未通过:\n{r.stdout[-1500:]}\n{r.stderr[-800:]}"
        P(f"  {sc} 通过  OK")
    gh = json.load(open(os.path.join(ROOT, "cache", "gate_harness_check.json")))
    assert gh.get("ok") is True, "cache/gate_harness_check.json ok != true"
    P("  cache/gate_harness_check.json ok = true  OK")
    g, noise = RUN.load_geoms()
    two = {k: g[k] for k in ("animalACE2/Wuhan-Hu-1/human_monomer", "animalACE2/Eta/hamster")}
    log = U.selfcheck(two)
    log += D.selfcheck46(two)
    jdump({"log": log, "prereg": D.verify_lock46(False)}, os.path.join(OUT, "selftest.json"))
    P("先决自检全部通过")


def need_selftest():
    assert os.path.exists(os.path.join(OUT, "selftest.json")), "先跑 selftest（CLAUDE.md §10）"
    need_prereg()


# ======================================================================
# 保真判据与它的牙齿
# ======================================================================
def _fid_task(n):
    g = RUN._G[n]
    nz = RUN._NOISE[n]
    ref = D.real_ref(g)
    res = {"real": {"cso": ref["cso"], "lo": ref["lo"], "hi": ref["hi"],
                    "f1_real": D.fid_stats(g, g.M, ref)[0]}, "pts": {}}
    for world in ("G", "RR", "GCLIP"):
        for kind in ("R1", "R2"):
            path = nz["path_k1"] if kind == "R1" else nz["path_k2"]
            for p, pt in enumerate(path):
                r = D.fid_point(g, ref, world, kind, pt, "main")
                if world == "GCLIP":
                    cf = []
                    for d in range(D.N_FID_DRAW):
                        rng = np.random.default_rng(D.seed46("fid", "main", g.name, world, kind, d))
                        k = 1 if kind == "R1" else 2
                        _, c_ = g.make(k, "real", pt["s"], pt["sigma"], "gauss", rng, clip=True)
                        cf.append(c_)
                    r["clipped_frac"] = float(np.mean(cf))
                r["sigma"], r["s"] = pt["sigma"], pt["s"]
                res["pts"][f"{world}|{kind}|{p}"] = r
    for kind in ("R1", "R2"):
        res["pts"][f"REF|{kind}|0"] = D.fid_point(g, ref, "REF", kind, None, "ref")
    path = nz["path_k1"]
    res["pts"]["INFL|R1|3"] = D.fid_point(g, ref, "G", "R1", path[3], "infl", infl=1.25)
    return n, res


def stage_fidelity(a):
    need_selftest()
    t0 = time.time()
    with mp.Pool(a.jobs, initializer=RUN._init_worker) as pool:
        out = dict(pool.map(_fid_task, PATHM, chunksize=1))
    jdump(out, os.path.join(OUT, "fidelity.json"))
    P(f"  保真统计量完成 {time.time() - t0:.0f}s")
    locked = fidelity_teeth(out, "locked")
    amend = fidelity_teeth(out, "amended")
    jdump(locked, os.path.join(OUT, "fidelity_teeth_recheck_locked.json"))
    jdump(amend, os.path.join(OUT, "fidelity_teeth.json"))
    P("  --- 锁定判据（F1 = 在端点或之外）的牙齿：")
    for k, v in locked["msgs"].items():
        P(f"    {k}: {v}")
    P("  --- 修订 A1 判据（F1a = 堆积于端点）的牙齿：")
    for k, v in amend["msgs"].items():
        P(f"    {k}: {v}")
    assert amend["ok"], "修订判据的牙齿（AFT1–AFT4）未通过 —— 中止，不再修订"
    P("  AFT1–AFT4 全过")


def fidelity_teeth(fid, mode):
    key_pass = "pass" if mode == "locked" else "pass_am"
    key_f1 = "pass_f1" if mode == "locked" else "pass_f1a"
    key_val = "f1" if mode == "locked" else "f1a"
    S1p = [n for n in PATHM if n in S1]
    msgs = {}
    n_caught, rows = 0, []
    for n in S1p:
        pts = [(k, v) for k, v in fid[n]["pts"].items() if k.startswith("GCLIP|")]
        mx = max(v[key_val] for _, v in pts)
        mxc = max(v["clipped_frac"] for _, v in pts)
        flag = any(v[key_val] > 0.05 and (not v[key_f1]) for _, v in pts)
        n_caught += int(flag)
        rows.append((n, mx, mxc, bool(flag)))
    ft1 = n_caught >= 4
    msgs["1"] = f"旧截断世界在 {n_caught}/14 个 S1 矩阵上至少一点 {key_val} > 5% 且被判 F（要求 ≥ 4）  {'OK' if ft1 else '未通过'}"
    bad, n_pts = 0, 0
    for n in PATHM:
        for k, v in fid[n]["pts"].items():
            if k.startswith("GCLIP|") and v["clipped_frac"] < 0.005:
                n_pts += 1
                bad += int(not v[key_f1])
    ft2 = bad == 0 and n_pts > 0
    msgs["2"] = f"被截比例 < 0.5% 的 {n_pts} 个点里 {key_val} 误判 {bad} 个  {'OK' if ft2 else '未通过'}"
    fails = [(n, k) for n in PATHM for k, v in fid[n]["pts"].items() if k.startswith("REF|") and not v[key_pass]]
    ft3 = not fails
    msgs["3"] = f"已知忠实参照（REF）在 {len(PATHM)} 个矩阵 × 2 秩上未过 {len(fails)} 个  {'OK' if ft3 else '未通过'}"
    caught = sum(1 for n in PATHM if (not fid[n]["pts"]["INFL|R1|3"]["pass_f2"]) or (not fid[n]["pts"]["INFL|R1|3"]["pass_f3"]))
    ft4 = caught >= 0.9 * len(PATHM)
    msgs["4"] = f"方差放大 1.25 倍的 G 世界在 {caught}/{len(PATHM)} 个矩阵上被 F2/F3 判 F（要求 ≥ 90%）  {'OK' if ft4 else '未通过'}"
    return {"mode": mode, "ok": bool(ft1 and ft2 and ft3 and ft4), "msgs": msgs, "rows1": rows, "ref_fails": fails}


def faithful_map():
    fid = json.load(open(os.path.join(OUT, "fidelity.json")))
    fm = {}
    for w in ("G", "RR", "GCLIP"):
        fm[w] = {}
        for n in PATHM:
            pts = {k: v for k, v in fid[n]["pts"].items() if k.startswith(w + "|")}
            assert len(pts) == 14
            fm[w][n] = {"all_pass": all(v["pass_am"] for v in pts.values()),
                        "n_fail": sum(not v["pass_am"] for v in pts.values()),
                        "fail_pts": [k for k, v in pts.items() if not v["pass_am"]],
                        "locked_all_pass": all(v["pass"] for v in pts.values())}
    return fm


# ======================================================================
def _perm_task(args):
    n, ext = args
    g = RUN._G[n]
    return n, ext, U.perm_null(g.M, g.mask, ext, D.N_PERM_MAT, D.S_PERM, D.seed46("perm", n, ext))


def stage_perm(a):
    need_selftest()
    t0 = time.time()
    tasks = [(n, e) for n in SCOPE for e in EXTS]
    with mp.Pool(a.jobs, initializer=RUN._init_worker) as pool:
        res = pool.map(_perm_task, tasks, chunksize=1)
    out = {}
    for n, e, arr in res:
        out.setdefault(n, {})[e] = arr
    pickle.dump(out, open(os.path.join(OUT, "perm.pkl"), "wb"))
    P(f"  N-perm 完成 {time.time() - t0:.0f}s")


# ======================================================================
def cell_list():
    cells = []
    for w in WORLDS:
        for n in PATHM:
            for p in range(7):
                cells.append((w, n, "R1", p))
                cells.append((w, n, "R2", p))
            cells.append((w, n, "R0", 0))
    return cells


def cell_path(c):
    w, n, kind, p = c
    return os.path.join(WDIR, f"{w}__{n.replace('/', '_')}__{kind}__{p}.npz")


def _cell_task(c):
    path = cell_path(c)
    if os.path.exists(path):
        return c
    w, n, kind, p = c
    g, nz = RUN._G[n], RUN._NOISE[n]
    point = None
    if kind in ("R1", "R2"):
        point = (nz["path_k1"] if kind == "R1" else nz["path_k2"])[p]
    R, S = D.R_WORLD, D.S_WORLD
    out = np.full((R, S, len(EXTS), 3), np.nan)
    for rep in range(R):
        rng = np.random.default_rng(D.seed46("world", w, n, kind, p, rep))
        Y = D.make_world46(g, w, kind, rng, point=point)
        for ie, ext in enumerate(EXTS):
            out[rep, :, ie, :] = U.many_splits(Y, g.mask, ext, S, D.seed46("wsplit", w, n, kind, p, rep, ext))
    np.savez_compressed(path, cos=out)
    return c


def stage_worlds(a):
    need_selftest()
    assert os.path.exists(os.path.join(OUT, "fidelity_teeth.json")) and json.load(open(os.path.join(OUT, "fidelity_teeth.json")))["ok"], \
        "先过保真判据的牙齿（fidelity 阶段）"
    cells = cell_list()
    todo = [c for c in cells if not os.path.exists(cell_path(c))]
    P(f"  植入世界单元 {len(cells)}，待算 {len(todo)}（R={D.R_WORLD} × {D.S_WORLD} 分半 × 2 抽取器）")
    t0 = time.time()
    with mp.Pool(a.jobs, initializer=RUN._init_worker) as pool:
        for i, _ in enumerate(pool.imap_unordered(_cell_task, todo, chunksize=1), 1):
            if i % 20 == 0 or i == len(todo):
                P(f"    {i}/{len(todo)}  {time.time() - t0:.0f}s")


def load_world(w, n, kind, p):
    return np.load(cell_path((w, n, kind, p)))["cos"]


# ======================================================================
def gate_eval(world):
    perm = pickle.load(open(os.path.join(OUT, "perm.pkl"), "rb"))
    res = {"H1": {}, "H2": {}}
    exceed = {e: [] for e in EXTS}
    sep = {e: {} for e in EXTS}
    for n in PATHM:
        for ie, e in enumerate(EXTS):
            p95 = float(np.nanquantile(perm[n][e][:, 1], 0.95))
            w0 = load_world(world, n, "R0", 0)[:, :, ie, 1]
            exceed[e].append(float(np.nanmean(w0 > p95)))
            ok = False
            for p in range(7):
                y1 = np.nanmedian(load_world(world, n, "R1", p)[:, :, ie, 1], axis=1)
                y2 = np.nanmedian(load_world(world, n, "R2", p)[:, :, ie, 1], axis=1)
                if np.median(y2) > np.quantile(y1, 0.95):
                    ok = True
            sep[e][n] = ok
    for e in EXTS:
        res["H1"][e] = {"pooled": float(np.mean(exceed[e])), "pass": bool(np.mean(exceed[e]) <= 0.10),
                        "per_matrix": dict(zip(PATHM, exceed[e]))}
        s1 = [sep[e][n] for n in PATHM if n in S1]
        res["H2"][e] = {"separable": sep[e], "frac_S1": float(np.mean(s1)), "pass": bool(np.mean(s1) >= 2 / 3)}
    res["pass"] = all(res["H1"][e]["pass"] and res["H2"][e]["pass"] for e in EXTS)
    return res


def stage_gate(a):
    need_selftest()
    out = {}
    for w in WORLDS:
        out[w] = gate_eval(w)
        for e in EXTS:
            P(f"  [{w}/{e}] H1 假警报 {out[w]['H1'][e]['pooled']:.3f} {'通过' if out[w]['H1'][e]['pass'] else '未通过'}；"
              f"H2 S1 可分辨 {out[w]['H2'][e]['frac_S1']:.2f} {'通过' if out[w]['H2'][e]['pass'] else '未通过'}")
    jdump(out, os.path.join(OUT, "gate.json"))


# ======================================================================
def _real_task(args):
    n, ext = args
    g = RUN._G[n]
    meta = RUN.load_meta()
    arr = U.many_splits(g.M, g.mask, ext, D.S_REAL, D.seed46("real", n, ext))
    cont = RUN.contiguous_split(g.M, g.mask, np.array(meta[n]["sites"]), ext, D.seed46("cont", n, ext))
    return n, ext, arr, cont


def stage_real(a):
    need_selftest()
    gp = os.path.join(OUT, "gate.json")
    assert os.path.exists(gp), "先跑 gate"
    t0 = time.time()
    tasks = [(n, e) for n in SCOPE for e in EXTS]
    with mp.Pool(a.jobs, initializer=RUN._init_worker) as pool:
        res = pool.map(_real_task, tasks, chunksize=1)
    out = {}
    for n, e, arr, cont in res:
        out.setdefault(n, {})[e] = {"splits": arr, "contig": cont}
    pickle.dump(out, open(os.path.join(OUT, "real.pkl"), "wb"))
    P(f"  真实分半完成 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["selftest", "fidelity", "perm", "worlds", "gate", "real"])
    ap.add_argument("--jobs", type=int, default=9)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"selftest": stage_selftest, "fidelity": stage_fidelity, "perm": stage_perm, "worlds": stage_worlds,
     "gate": stage_gate, "real": stage_real}[a.stage](a)
