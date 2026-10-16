#!/usr/bin/env python3
"""docs/43 预先登记 §8（次要，不进判决）：真实数据内的成分消融。

A0 = 真实矩阵；A1 = 去掉秩 2 拟合曲面的第二奇异成分；A2 = 去掉整个秩 2 曲面。
每个 CV 切分种子（11..30）单独一条曲线，检出秩 = 该曲线 argmin（不做 3 种子平均）。
只描述。理由：这是真实数据上的消融，不是已知真值的植入，不能区分
「成分 2 是真交互」与「成分 2 是过冲产物」。
已知事实自检：在一个植入的秩 2 矩阵（高 SNR 之下的窗口内）上 A1 必须使检出降到 1，A2 必须降到 0。
"""
import json
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overshoot_lib as OL                       # noqa: E402
import rank_core as RC                           # noqa: E402

ROOT = OL.ROOT
SEEDS_AB = tuple(range(11, 31))


def ablated(g):
    P2 = g.fit[2]["P"]
    u, s, vt = np.linalg.svd(np.where(g.mask, P2, 0.0), full_matrices=False)
    # 用全矩阵 P2（未观测处也有值）做 SVD
    u, s, vt = np.linalg.svd(P2, full_matrices=False)
    comp2 = s[1] * np.outer(u[:, 1], vt[1])
    A1 = np.where(g.mask, g.M - comp2, 0.0)
    A2 = np.where(g.mask, g.M - P2, 0.0)
    return A1, A2, float(s[0]), float(s[1])


def det_seed(args):
    Y, mask, sd = args
    e = RC.cv_rank_noleak(Y, mask, max_rank=OL.MAXR, seed=sd, **OL.CV)
    return OL.argmin_rank({r: e[r][1] for r in e})


def selfcheck(jobs=9):
    """已知事实：窗口内植入的秩 2（s/σ=2，gAPN 几何）-> A0=2；去第二成分 -> 1；去全部 -> 0。"""
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    g = geoms["PDCoV/PDCoV/gAPN"]
    rng = np.random.default_rng(99)
    sg = g.fit[1]["sigma"]
    Y, _ = g.make(2, "rand", 2.0 * sg, sg, "gauss", rng, clip=False)
    # 对植入矩阵重做同样的消融（在它自己的 rank-2 拟合上）
    U, V, _ = RC.fit_loadings(Y, g.mask, rank=2, restarts=4, seed=11)
    P2 = U @ V.T
    u, s, vt = np.linalg.svd(P2, full_matrices=False)
    A1 = np.where(g.mask, Y - s[1] * np.outer(u[:, 1], vt[1]), 0.0)
    A2 = np.where(g.mask, Y - P2, 0.0)
    out = {}
    with mp.get_context("fork").Pool(jobs) as pool:
        for nm, M in (("A0", Y), ("A1", A1), ("A2", A2)):
            ds = pool.map(det_seed, [(M, g.mask, sd) for sd in SEEDS_AB[:8]])
            out[nm] = ds
    ok = (np.mean([d == 2 for d in out["A0"]]) >= 0.75 and np.mean([d <= 1 for d in out["A1"]]) >= 0.75
          and np.mean([d == 0 for d in out["A2"]]) >= 0.75)
    print("  消融自检（植入秩 2, s/σ=2）A0/A1/A2 检出:", out, "->", "OK" if ok else "FAIL", flush=True)
    assert ok, "消融工具未通过已知事实自检，中止"
    return out


def main():
    OL.verify_lock()
    sc = selfcheck()
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    names = [n for n in OL.scope_names() if geoms[n].rep_kind == "measured"]
    res = {}
    with mp.get_context("fork").Pool(9) as pool:
        for nm in names:
            g = geoms[nm]
            A1, A2, s1, s2 = ablated(g)
            row = {"s1": s1, "s2": s2}
            for lab, M in (("A0", np.where(g.mask, g.M, 0.0)), ("A1", A1), ("A2", A2)):
                ds = pool.map(det_seed, [(M, g.mask, sd) for sd in SEEDS_AB])
                row[lab] = [int(sum(d == q for d in ds)) for q in range(4)]
            res[nm] = row
            print(f"  {nm:<36} s2/s1={s2/s1:.2f}  A0 {row['A0']}  A1 {row['A1']}  A2 {row['A2']}", flush=True)
    json.dump({"selfcheck": sc, "res": res}, open(os.path.join(ROOT, "cache", "overshoot_ablate.json"), "w"), indent=1,
              default=lambda o: int(o) if isinstance(o, np.integer) else float(o))


if __name__ == "__main__":
    main()
