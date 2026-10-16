#!/usr/bin/env python3
"""docs/43 事后补充 S-λ（**未预先登记**，不进判决；在看到 T4/T7/T11 之后设计，动机见 docs/43 §8）。

问题：真实矩阵离「过冲悬崖」有多远，而且不假设任何 σ？
做法：保持真实的主效应、真实的缺失、真实的残余（包括其中一切结构与噪声）不动，
把拟合出的成分按 λ 放大：
  A3(λ) = 真实 − 第二成分 + (λ−1)·第一成分     （真秩 1 + 真实残余，信号放大 λ 倍）
  A4(λ) = 真实 + (λ−1)·(第一成分 + 第二成分)    （秩 2 + 真实残余，信号放大 λ 倍）
λ=1 的 A3 即预先登记 §8 的 A1。对每个 λ 取 4 组 CV 种子三元组（(1,2,3),(4,5,6),(7,8,9),(10,11,12)），
每组走头条口径（3 种子均值曲线的 argmin），现行程序与 K1 并列。
已知事实自检：在「秩 1 + 高斯噪声」的植入世界（gAPN 几何，s/σ=1）上做同样的放大，
现行程序必须在 λ≥4 附近过冲成 ≥2（harness 的悬崖），K1 必须一直是 1；λ=1,2 时二者都是 1。
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
LAMS = (1.0, 1.5, 2.0, 3.0, 4.0)
TRIPLES = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))


def comps(M, mask):
    U, V, _ = RC.fit_loadings(M, mask, rank=2, restarts=4, seed=11)
    P2 = U @ V.T
    u, s, vt = np.linalg.svd(P2, full_matrices=False)
    c1 = s[0] * np.outer(u[:, 0], vt[0])
    c2 = s[1] * np.outer(u[:, 1], vt[1])
    return c1, c2


def build(M, mask, c1, c2, world, lam):
    if world == "A3":
        Y = M - c2 + (lam - 1.0) * c1
    else:
        Y = M + (lam - 1.0) * (c1 + c2)
    return np.where(mask, Y, 0.0)


def det_triple(args):
    Y, mask, seeds, meth = args
    if meth == "base":
        cs = [{r: v for r, v in zip(range(4), [RC.cv_rank_noleak(Y, mask, max_rank=3, seed=sd, **OL.CV)[r][1] for r in range(4)])} for sd in seeds]
        return OL.argmin_rank(OL.mean_curve(cs))
    cs = [OL.cv_curve(Y, mask, range(4), seed=sd, joint=True, **{k: v for k, v in OL.CV.items() if k != "K"}) for sd in seeds]
    return OL.argmin_rank(OL.mean_curve(cs))


def selfcheck(pool):
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    g = geoms["PDCoV/PDCoV/gAPN"]
    sg = g.fit[1]["sigma"]
    rng = np.random.default_rng(404)
    Y, _ = g.make(1, "rand", 1.0 * sg, sg, "gauss", rng, clip=False)
    U, V, _ = RC.fit_loadings(Y, g.mask, rank=1, restarts=4, seed=11)
    c1 = U @ V.T
    out = {}
    for lam in (1.0, 2.0, 4.0, 6.0):
        Ys = np.where(g.mask, Y + (lam - 1.0) * c1, 0.0)
        for meth in ("base", "K1"):
            out[(lam, meth)] = pool.map(det_triple, [(Ys, g.mask, t, meth) for t in TRIPLES[:2]])
    print("  S-λ 自检（秩 1 + 高斯噪声，s/σ=1 放大 λ）:", {f"{k[0]}x{k[1]}": v for k, v in out.items()}, flush=True)
    assert all(d == 1 for d in out[(1.0, "base")] + out[(2.0, "base")]), "λ≤2 时现行程序应为 1"
    assert any(d >= 2 for d in out[(4.0, "base")] + out[(6.0, "base")]), "放大到 λ≥4 现行程序应过冲（harness 悬崖）"
    assert all(d == 1 for lam in (1.0, 2.0, 4.0, 6.0) for d in out[(lam, "K1")]), "K1 应一直是 1"
    print("  S-λ 自检通过（命中过冲；K1 不误伤）", flush=True)
    return {f"{k[0]}x{k[1]}": v for k, v in out.items()}


def main():
    global LAMS
    out_name = "overshoot_scaleup.json"
    if "--lams" in sys.argv:
        LAMS = tuple(float(x) for x in sys.argv[sys.argv.index("--lams") + 1].split(","))
        out_name = "overshoot_scaleup_ext.json"
    OL.verify_lock()
    geoms = pickle.load(open(os.path.join(ROOT, "cache", "overshoot_geoms.pkl"), "rb"))
    names = [n for n in OL.scope_names() if geoms[n].rep_kind == "measured" and "Omicron_BA2/bat" not in n]
    res = {}
    with mp.get_context("fork").Pool(9) as pool:
        res["selfcheck"] = selfcheck(pool)
        for nm in names:
            g = geoms[nm]
            c1, c2 = comps(g.M, g.mask)
            row = {}
            for world in ("A3", "A4"):
                for meth in ("base", "K1"):
                    for lam in LAMS:
                        Y = build(g.M, g.mask, c1, c2, world, lam)
                        ds = pool.map(det_triple, [(Y, g.mask, t, meth) for t in TRIPLES])
                        row[f"{world}|{meth}|{lam}"] = [int(sum(d == q for d in ds)) for q in range(4)]
            res[nm] = row
            def show(world, meth):
                return " ".join(f"{lam}:{max(range(4), key=lambda q: row[f'{world}|{meth}|{lam}'][q])}"
                                f"({row[f'{world}|{meth}|{lam}'][max(range(4), key=lambda q: row[f'{world}|{meth}|{lam}'][q])]}/4)" for lam in LAMS)
            print(f"  {nm}\n     A3 base {show('A3','base')}\n     A3 K1   {show('A3','K1')}\n     A4 base {show('A4','base')}\n     A4 K1   {show('A4','K1')}", flush=True)
    json.dump(res, open(os.path.join(ROOT, "cache", out_name), "w"), indent=1)


if __name__ == "__main__":
    main()
