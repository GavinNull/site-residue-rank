#!/usr/bin/env python3
"""CLAUDE.md §10 强制自检：在用子空间对齐流程得出任何结论之前跑这个。

§10.2 要求「既测能命中，也测不误伤」，所以两个方向都测：
  A. 从**同一个已知 2 维子空间** + 噪声合成两个矩阵 -> 必须给出高对齐
  B. 从**两个已知无关子空间**合成 -> 必须落回随机零分布水平
  C. 只共享一维（成分1 相同、成分2 无关）-> cos th1 高而 cos th2 在零分布水平
     （这一条直接检验「成分1 混淆」能否被区分出来 —— 本任务的核心判别）
  D. 快抽取器 emsvd 与 rank_test_power.als_guarded 在同一矩阵上必须一致
  E. 恒等性：同一子空间自比必须给 cos = (1, 1)
  F. 截尾应当摧毁可检出性（已建立事实 r = -0.786 的方向性复核）

不通过就 SystemExit(1)，不往下跑。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (align, align_deflated, extract_subspace,  # noqa: E402
                          haar_1d_null, haar_pair_null)

RANK = 2


def synth(nr, d, V, sigma, rng, fill=0.95):
    """主效应 + rank2 交互（交互的残基侧载荷张成 span(V)）+ 噪声。"""
    k = V.shape[1]
    U = rng.normal(0, 1.0, (nr, k))
    inter = U @ np.diag([1.5, 1.0][:k]) @ V.T
    a = rng.normal(0, 1.5, nr)[:, None]
    b = rng.normal(0, 0.5, d)[None, :]
    M = a + b + inter + rng.normal(0, sigma, (nr, d))
    mask = rng.random((nr, d)) < fill
    return np.where(mask, M, np.nan), mask


def rand_centered_basis(d, k, rng):
    one = np.ones((d, 1)) / np.sqrt(d)
    A = rng.normal(size=(d, k))
    A -= one @ (one.T @ A)
    Q, _ = np.linalg.qr(A)
    return Q


def main():
    rng = np.random.default_rng(20261011)
    d = 20
    nr = 150
    sigma = 0.35
    ok = True
    print("=" * 84)
    print("子空间对齐流程 自检（CLAUDE.md §10：用已知事实，双向，命中与不误伤）")
    print("=" * 84)

    # ---- G. dc_fast 必须与原版 double_center 逐元素一致 ----
    from coupling_structure import double_center
    from subspace_lib import dc_fast
    worst = 0.0
    for t in range(6):
        A = rng.normal(size=(40, 20)) + rng.normal(size=(40, 1)) * 2
        mk = rng.random((40, 20)) < 0.75
        mk[:, 0] = True; mk[0, :] = True
        d1 = double_center(np.where(mk, A, 0.0), mk)
        d2 = dc_fast(np.where(mk, A, 0.0), mk)
        worst = max(worst, float(np.abs(d1 - d2)[mk].max()))
    hit = worst < 1e-9
    print(f"\n[G] dc_fast vs coupling_structure.double_center 最大逐元素差 "
          f"{worst:.2e}  {'OK' if hit else '<< 向量化实现不等价，禁止使用'}")
    ok &= hit

    null = haar_pair_null(d, RANK, n=3000, seed=1)
    q95_c1 = float(np.percentile(null["cos1"], 95))
    q95_c2 = float(np.percentile(null["cos2"], 95))
    q95_ssc = float(np.percentile(null["ssc"], 95))
    print(f"\n[0] 随机零分布（d={d}，中心化 19 维内的两个随机 2 维面，3000 次）")
    print(f"    cos th1  中位 {np.median(null['cos1']):.3f}  p95 {q95_c1:.3f}")
    print(f"    cos th2  中位 {np.median(null['cos2']):.3f}  p95 {q95_c2:.3f}")
    print(f"    sum cos^2 中位 {np.median(null['ssc']):.3f}  p95 {q95_ssc:.3f}")
    print("    量级校准：随机两个 2 维面**远非正交**（cos th1 中位 ~0.6），"
          "这正是必须比零分布的原因")
    # 已知事实核对：E[sum cos^2] 对 k 维 vs k 维 in R^m 应为 k^2/m
    expect = RANK ** 2 / (d - 1)
    got = float(null["ssc"].mean())
    hit = abs(got - expect) < 0.03
    print(f"    解析核对 E[sum cos^2] = k^2/(d-1) = {expect:.4f}，"
          f"模拟 {got:.4f}  {'OK' if hit else '<< 零分布生成有错'}")
    ok &= hit

    # ---- E. 恒等性 ----
    V = rand_centered_basis(d, RANK, rng)
    cols = np.arange(d)
    a = align(V, cols, V, cols)
    hit = a["cos"][0] > 0.9999 and a["cos"][1] > 0.9999
    print(f"\n[E] 恒等性：自比 cos = ({a['cos'][0]:.4f}, {a['cos'][1]:.4f})  "
          f"{'OK' if hit else '<< align 实现有错'}")
    ok &= hit

    # ---- A. 共享子空间（必须命中） ----
    print(f"\n[A] 两个矩阵从**同一** 2 维子空间合成（{nr} 位点，sigma={sigma}）")
    rec = []
    for t in range(12):
        Vs = rand_centered_basis(d, RANK, rng)
        M1, k1 = synth(nr, d, Vs, sigma, rng)
        M2, k2 = synth(nr, d, Vs, sigma, rng)
        B1, c1, _ = extract_subspace(M1, k1)
        B2, c2, _ = extract_subspace(M2, k2)
        if B1 is None or B2 is None:
            continue
        rec.append(align(B1, c1, B2, c2))
    c1s = [r["cos"][0] for r in rec]; c2s = [r["cos"][1] for r in rec]
    hit = min(c1s) > 0.9 and min(c2s) > 0.85
    print(f"    cos th1 范围 {min(c1s):.3f}-{max(c1s):.3f}  "
          f"cos th2 范围 {min(c2s):.3f}-{max(c2s):.3f}")
    print(f"    判据：两个 cos 都要显著高于 p95（{q95_c1:.2f} / {q95_c2:.2f}）"
          f"  {'OK' if hit else '<< 流程无法识别真共享子空间'}")
    ok &= hit

    # ---- B. 无关子空间（必须不误伤） ----
    print("\n[B] 两个矩阵从**两个无关** 2 维子空间合成")
    rec = []
    for t in range(60):
        M1, k1 = synth(nr, d, rand_centered_basis(d, RANK, rng), sigma, rng)
        M2, k2 = synth(nr, d, rand_centered_basis(d, RANK, rng), sigma, rng)
        B1, c1, _ = extract_subspace(M1, k1)
        B2, c2, _ = extract_subspace(M2, k2)
        if B1 is None or B2 is None:
            continue
        rec.append(align(B1, c1, B2, c2))
    s = np.array([r["ssc"] for r in rec])
    frac_above = float(np.mean(s > q95_ssc))
    hit = abs(s.mean() - null["ssc"].mean()) < 0.08 and frac_above < 0.15
    print(f"    sum cos^2 均值 {s.mean():.3f}（零分布均值 "
          f"{null['ssc'].mean():.3f}），超 p95 的比例 {frac_above:.0%}")
    print(f"    判据：均值贴住零分布且超 p95 比例 ~alpha  "
          f"{'OK' if hit else '<< 流程会把无关子空间报成对齐（假阳）'}")
    ok &= hit

    # ---- C. 只共享一维（本任务的核心判别） ----
    print("\n[C] 只共享**一维**（成分1 相同、成分2 无关）")
    rec = []
    for t in range(40):
        shared = rand_centered_basis(d, 1, rng)
        V1 = rand_centered_basis(d, 2, rng); V1[:, 0] = shared[:, 0]
        V1, _ = np.linalg.qr(V1)
        V2 = rand_centered_basis(d, 2, rng); V2[:, 0] = shared[:, 0]
        V2, _ = np.linalg.qr(V2)
        M1, k1 = synth(nr, d, V1, sigma, rng)
        M2, k2 = synth(nr, d, V2, sigma, rng)
        B1, c1, _ = extract_subspace(M1, k1)
        B2, c2, _ = extract_subspace(M2, k2)
        if B1 is None or B2 is None:
            continue
        r = align(B1, c1, B2, c2)
        dr = align_deflated(B1, c1, B2, c2, shared[:, 0], np.arange(d))
        rec.append((r, dr))
    a1 = np.array([r["cos"][0] for r, _ in rec])
    a2 = np.array([r["cos"][1] for r, _ in rec])
    dres = np.array([dr["cos_resid"] for _, dr in rec if dr])
    n1d = haar_1d_null(d, n=3000, seed=2)
    hit = (a1.mean() > 0.9) and (a2.mean() < q95_c2 + 0.10) and \
          (abs(dres.mean() - n1d.mean()) < 0.12)
    print(f"    cos th1 均值 {a1.mean():.3f}（真共享方向，应 ~1）")
    print(f"    cos th2 均值 {a2.mean():.3f}（应落在零分布 p95={q95_c2:.3f} 附近）")
    print(f"    投掉共享方向后的一维残差 cos {dres.mean():.3f}  "
          f"（一维零分布均值 {n1d.mean():.3f}）")
    print(f"    判据：能把「共享一维」与「共享两维」分开  "
          f"{'OK' if hit else '<< 无法区分 -> 整个结论无意义'}")
    ok &= hit

    # ---- C2. 真共享两维时 deflated 必须仍然高（不误伤） ----
    print("\n[C2] 共享**两维**时，投掉成分1 后残差方向仍必须对齐（不误伤）")
    rec = []
    for t in range(25):
        Vs = rand_centered_basis(d, RANK, rng)
        M1, k1 = synth(nr, d, Vs, sigma, rng)
        M2, k2 = synth(nr, d, Vs, sigma, rng)
        B1, c1, _ = extract_subspace(M1, k1)
        B2, c2, _ = extract_subspace(M2, k2)
        dr = align_deflated(B1, c1, B2, c2, Vs[:, 0], np.arange(d))
        if dr:
            rec.append(dr["cos_resid"])
    hit = np.mean(rec) > 0.85
    print(f"    残差方向 cos 均值 {np.mean(rec):.3f}（应 ~1）  "
          f"{'OK' if hit else '<< deflation 把真信号也投掉了（假阴）'}")
    ok &= hit

    # ---- D. 两个抽取器交叉核验 ----
    print("\n[D] emsvd 与 rank_test_power.als_guarded 在同一矩阵上是否同一子空间")
    rec = []
    for t in range(8):
        Vs = rand_centered_basis(d, RANK, rng)
        M1, k1 = synth(nr, d, Vs, sigma, rng, fill=0.85)
        Be, ce, _ = extract_subspace(M1, k1, method="emsvd")
        Ba, ca, _ = extract_subspace(M1, k1, method="als", seed=t)
        if Be is None or Ba is None:
            continue
        rec.append(align(Be, ce, Ba, ca)["ssc"])
    hit = np.mean(rec) > 1.85
    print(f"    sum cos^2(emsvd, als) 均值 {np.mean(rec):.4f} / 2.0  "
          f"{'OK' if hit else '<< 快抽取器与带保护 ALS 不一致，不能用 emsvd'}")
    ok &= hit

    # ---- F. 截尾方向性复核 ----
    print("\n[F] 截尾摧毁可检出性（已建立事实的方向性复核）")
    Vs = rand_centered_basis(d, RANK, rng)
    print(f"    {'截尾比例':>8} {'与真子空间 sum cos^2':>22}")
    row = []
    for q in (0.0, 0.1, 0.25, 0.5):
        vals = []
        for t in range(10):
            M1, k1 = synth(nr, d, Vs, sigma, rng)
            if q > 0:
                thr = np.nanquantile(M1[k1], q)
                k1 = k1 & (M1 > thr)      # 下限以下的点剔除（= 截尾后不可用）
            B1, c1, _ = extract_subspace(M1, k1)
            if B1 is None:
                continue
            vals.append(align(B1, c1, Vs, np.arange(d))["ssc"])
        print(f"    {q:>8.0%} {np.mean(vals):>22.3f}")
        row.append(np.mean(vals))
    hit = row[0] > row[-1] + 0.1
    print(f"    判据：单调下降  {'OK' if hit else '<< 与已建立事实矛盾'}")
    ok &= hit

    print("\n" + "=" * 84)
    print("自检总判定:", "全部通过" if ok else "** 未通过 —— 中止 **")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
