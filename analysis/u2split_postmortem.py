#!/usr/bin/env python3
"""docs/45 事后分析（postmortem，CLAUDE.md §4：不计入预先登记的判决）。
动机：预先登记的 R1 世界沿用 docs/43 的 clip=True（把植入值截到真实矩阵的观测值域）。
世界里 5–23% 的格子（deer/hamster 骨架）被截断，真实矩阵只有 ~0.1% 的格子贴边界；
截断是非线性，可被两半同样地读成「第二成分」。本脚本把 R1/R2 世界重做成 clip=False，
仅用于量化「预先登记 R1 世界在低噪声端的虚高有多少来自截断」。判决不因此改变。
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, multiprocessing as mp, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2split_lib as U
import u2split_run as RUN

ND = os.path.join(RUN.OUT, "worlds_noclip")
os.makedirs(ND, exist_ok=True)


def path_of(c):
    n, kind, p = c
    return os.path.join(ND, f"{n.replace('/', '_')}__{kind}__{p}.npz")


def task(c):
    if os.path.exists(path_of(c)):
        return c
    n, kind, p = c
    g, nz = RUN._G[n], RUN._NOISE[n]
    pt = (nz["path_k1"] if kind == "R1" else nz["path_k2"])[p]
    out = np.full((U.R_WORLD, U.S_WORLD, 2, 3), np.nan)
    for rep in range(U.R_WORLD):
        rng = np.random.default_rng(U.seed_of("worldnc", n, kind, p, rep))
        Y = U.make_world(g, kind, rng, point=pt, clip=False)
        for ie, ext in enumerate(U.EXTRACTORS):
            out[rep, :, ie, :] = U.many_splits(Y, g.mask, ext, U.S_WORLD, U.seed_of("wsplitnc", n, kind, p, rep, ext))
    np.savez_compressed(path_of(c), cos=out)
    return c


if __name__ == "__main__":
    U.verify_lock()
    cells = [(n, k, p) for n in RUN.PATHM for k in ("R1", "R2") for p in range(7)]
    todo = [c for c in cells if not os.path.exists(path_of(c))]
    print(f"事后：无截断世界 {len(cells)} 单元，待算 {len(todo)}", flush=True)
    t0 = time.time()
    with mp.Pool(9, initializer=RUN._init_worker) as pool:
        for i, _ in enumerate(pool.imap_unordered(task, todo), 1):
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
