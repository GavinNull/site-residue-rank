#!/usr/bin/env python3
"""两个「仅第一维可分辨」候选的秩 1 探查（docs/23 §4）。

**这个脚本不产出证据，它产出一个人类决策点所需的数字。**

为什么必须换到秩 1：docs/13 §1 已经证明，秩 2 双线性下 `v_1`、`v_2`
**各自不可辨识**（任意可逆 2×2 的 T 给同一个拟合）。所以
「取秩 2 拟合的第一列当第一轴」是没有定义的操作。
要检验「第一轴」，必须**拟合秩 1**——秩 1 的那一条方向在符号意义下
是可辨识的。

`analysis/third_family_screen.py` 的结果是：本地唯一两个在既有家族之外、
且能抽出子空间的矩阵（titeseq/4-4-20 vs 荧光素、magma/319-345 vs 流感 HA）
都**没过第二维可分辨闸**（stab cos2 0.495 / 0.571 < 0.60）。
于是 docs/13/21 的 λ 共识（需要 2 维投影算子）在它们身上**不能算**。
能算的只有一维问题：**它们自己的那一条轴，和两组共识的普适轴 u 同不同？**

把这两个矩阵以秩 1 身份纳入证据集，是对判别集纳入判据的修改
= **CLAUDE.md §9.1 的人类决策点**。本脚本只把数字摆出来，
不改 headline，不写进 `cache/subspace_two_groups.json`。
"""
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (AA20, ROOT, align, bootstrap_subspaces,  # noqa: E402
                          ci, extract_subspace, haar_1d_null, load_all_dms,
                          load_independent, load_scales, load_skempi,
                          dedup_by_content, permute_within_sites, pick_primary)
from third_family_screen import (cand_magma, cand_titeseq,  # noqa: E402
                                 gate_selfcheck)

N_BOOT = 2000
N_SPLIT = 400
N_PERM = 400


# ----------------------------------------------------------------------
# 一维对齐（align 要求 >= 2 列，所以这里必须自己写 —— 于是必须自检）
# ----------------------------------------------------------------------
def align_1d(wa, cols_a, wb, cols_b):
    """两条一维残基载荷方向，限制到共同残基 -> 再中心化 -> |cos|。

    与 `subspace_lib._recenter_orth` 的口径逐字一致：**先限制再中心化**，
    否则全 1 方向重新出现，两条方向会因为共享这个纯人造方向而假性对齐。
    """
    common = np.intersect1d(cols_a, cols_b)
    if len(common) < 8:
        return None
    a = np.asarray(wa, float).ravel()[np.searchsorted(cols_a, common)]
    b = np.asarray(wb, float).ravel()[np.searchsorted(cols_b, common)]
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    return {"cos": abs(float((a / na) @ (b / nb))), "n_res": int(len(common))}


def rank1_dir(M, mask, seed=0, method="emsvd"):
    """秩 1 的残基载荷方向 + 保留的残基列。符号不可辨识，一律取 |cos|。"""
    B, cols, err = extract_subspace(M, mask, rank=1, method=method, seed=seed)
    if B is None:
        return None, None, err
    return B[:, 0], cols, None


def selfcheck_1d():
    """§10：新写的 align_1d / rank1_dir 必须先用**已知事实**自检。"""
    print("=" * 96)
    print("[自检 R1] 秩 1 层（CLAUDE.md §10：既测命中也测不误伤）")
    print("=" * 96)
    ok = True
    d = 20
    cols = np.arange(d)
    rng = np.random.default_rng(1)

    # [R0] 解析核对：一维随机零分布的 E|cos| = sqrt(2/(pi*(d-1)))
    #      （环境是中心化后的 d-1 = 19 维）
    nl = haar_1d_null(d, n=200000, seed=1)
    ana = math.sqrt(2.0 / (math.pi * (d - 1)))
    print(f"  [R0] 一维零分布 E|cos| 模拟 {nl.mean():.4f} vs 解析 {ana:.4f}  "
          f"{'OK' if abs(nl.mean() - ana) < 0.003 else 'FAIL'}")
    ok &= abs(nl.mean() - ana) < 0.003
    p95 = float(np.percentile(nl, 95))
    print(f"       中位 {np.median(nl):.3f}  p95 {p95:.3f}  "
          f"（量级校准：两条随机方向的 |cos| 中位就有 "
          f"{np.median(nl):.2f}，不是 0）")

    # [R1] 恒等性
    w = rng.normal(size=d)
    r = align_1d(w, cols, w, cols)
    print(f"  [R1] 恒等性 自比 |cos| = {r['cos']:.4f}  "
          f"{'OK' if r['cos'] > 0.9999 else 'FAIL'}")
    ok &= r["cos"] > 0.9999

    # [R1b] 符号翻转必须给同一个值（秩 1 的符号不可辨识）
    r2 = align_1d(w, cols, -w, cols)
    print(f"  [R1b] 符号不变性 |cos(w, -w)| = {r2['cos']:.4f}  "
          f"{'OK' if abs(r2['cos'] - 1) < 1e-9 else 'FAIL'}")
    ok &= abs(r2["cos"] - 1) < 1e-9

    # [R1c] 不许因为全 1 方向假性对齐：两条**随机但都加了大常数**的方向
    #       必须仍然落在零分布上（这是 docs/13 §1 要求的再中心化）
    cs = []
    for _ in range(2000):
        a = rng.normal(size=d) + 5.0
        b = rng.normal(size=d) + 5.0
        cs.append(align_1d(a, cols, b, cols)["cos"])
    print(f"  [R1c] 全 1 方向未假性对齐 均值 {np.mean(cs):.3f} "
          f"vs 零分布均值 {nl.mean():.3f}  "
          f"{'OK' if abs(np.mean(cs) - nl.mean()) < 0.02 else 'FAIL'}")
    ok &= abs(np.mean(cs) - nl.mean()) < 0.02

    # [R2] 命中：两个矩阵从**同一条**已知一维方向合成，秩 1 必须还原
    v = rng.normal(size=d); v -= v.mean(); v /= np.linalg.norm(v)
    hits = []
    for t in range(12):
        rg = np.random.default_rng(100 + t)
        got = []
        for _ in range(2):
            u = rg.normal(size=150)
            X = np.outer(u, v) * 2.0 + rg.normal(0, 0.35, (150, d))
            mk = np.ones((150, d), bool)
            w1, c1, _ = rank1_dir(X, mk)
            got.append((w1, c1))
        hits.append(align_1d(got[0][0], got[0][1], got[1][0], got[1][1])["cos"])
    print(f"  [R2] 命中 同一条真方向 |cos| 范围 "
          f"{min(hits):.3f}-{max(hits):.3f}（必须远超 p95 {p95:.3f}）  "
          f"{'OK' if min(hits) > 0.9 else 'FAIL'}")
    ok &= min(hits) > 0.9

    # [R3] 不误伤 / 不误报：两条**无关**方向合成，必须贴住零分布
    miss = []
    for t in range(400):
        rg = np.random.default_rng(5000 + t)
        got = []
        for _ in range(2):
            vv = rg.normal(size=d); vv -= vv.mean(); vv /= np.linalg.norm(vv)
            u = rg.normal(size=150)
            X = np.outer(u, vv) * 2.0 + rg.normal(0, 0.35, (150, d))
            mk = np.ones((150, d), bool)
            got.append(rank1_dir(X, mk)[:2])
        miss.append(align_1d(got[0][0], got[0][1], got[1][0], got[1][1])["cos"])
    frac = float(np.mean(np.array(miss) > p95))
    print(f"  [R3] 不误报 无关方向 均值 {np.mean(miss):.3f}（零分布 "
          f"{nl.mean():.3f}）、超 p95 比例 {frac:.1%}（应 ~5%）  "
          f"{'OK' if abs(np.mean(miss) - nl.mean()) < 0.03 and frac < 0.10 else 'FAIL'}")
    ok &= abs(np.mean(miss) - nl.mean()) < 0.03 and frac < 0.10

    # [R4] 秩 1 的可辨识性 vs 秩 2 的不可辨识性（docs/13 §1 的事实复核）：
    #      从同一个秩 2 真子空间合成两矩阵，秩 2 的「第一列」对齐应当
    #      **显著低于**整个 2 维面的对齐 —— 证明「取第一列」没有定义。
    v1 = rng.normal(size=d); v1 -= v1.mean(); v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=d); v2 -= v2.mean()
    v2 -= v1 * (v1 @ v2); v2 /= np.linalg.norm(v2)
    col1, face = [], []
    for t in range(60):
        rg = np.random.default_rng(900 + t)
        Bs = []
        for _ in range(2):
            u = rg.normal(size=(150, 2)) * np.array([2.0, 1.6])
            X = u @ np.vstack([v1, v2]) + rg.normal(0, 0.35, (150, d))
            mk = np.ones((150, d), bool)
            B, c, _ = extract_subspace(X, mk, rank=2)
            Bs.append((B, c))
        col1.append(align_1d(Bs[0][0][:, 0], Bs[0][1],
                             Bs[1][0][:, 0], Bs[1][1])["cos"])
        face.append(align(Bs[0][0], Bs[0][1], Bs[1][0], Bs[1][1])["ssc"])
    print(f"  [R4] 「秩 2 的第一列」中位 |cos| = {np.median(col1):.3f}，"
          f"而整个 2 维面 Σcos² = {np.median(face):.3f}/2")
    print(f"       -> 取秩 2 的第一列确实不可辨识，所以本脚本一律重拟合秩 1  "
          f"{'OK' if np.median(col1) < 0.98 else 'FAIL'}")
    ok &= np.median(col1) < 0.98

    print("-" * 96)
    print(f"自检 R1 总判定: {'全部通过' if ok else '失败 —— 中止'}")
    if not ok:
        raise SystemExit("秩 1 自检未通过，按 CLAUDE.md §10 中止")
    print()
    return p95, nl


def resolvability_1d(d, nboot=N_BOOT, nsplit=N_SPLIT):
    """秩 1 的可分辨性：对半分自对齐（保守）+ bootstrap 稳定性。"""
    M, mask = d["M"], d["mask"]
    w0, c0, err = rank1_dir(M, mask)
    if w0 is None:
        return None
    rng = np.random.default_rng(42)
    nr = M.shape[0]
    sp = []
    for _ in range(nsplit):
        p = rng.permutation(nr)
        h1, h2 = p[: nr // 2], p[nr // 2:]
        w1, k1, _ = rank1_dir(M[h1], mask[h1])
        w2, k2, _ = rank1_dir(M[h2], mask[h2])
        if w1 is None or w2 is None:
            continue
        r = align_1d(w1, k1, w2, k2)
        if r:
            sp.append(r["cos"])
    rng = np.random.default_rng(7)
    st = []
    for _ in range(nboot):
        idx = rng.integers(0, nr, nr)
        wb, kb, _ = rank1_dir(M[idx], mask[idx], seed=int(rng.integers(1e9)))
        if wb is None:
            continue
        r = align_1d(w0, c0, wb, kb)
        if r:
            st.append(r["cos"])
    return {"w": w0, "cols": c0,
            "split_med": float(np.median(sp)) if sp else float("nan"),
            "split_ci": ci(sp) if sp else (float("nan"),) * 2,
            "split_n": len(sp),
            "stab_med": float(np.median(st)) if st else float("nan"),
            "stab_ci": ci(st) if st else (float("nan"),) * 2,
            "stab_n": len(st)}


def main():
    gate_selfcheck()
    p95_1d, nl = selfcheck_1d()
    scales = load_scales(verbose=False)

    J = json.load(open(os.path.join(ROOT, "cache",
                                    "subspace_two_groups.json")))
    u = np.array(J["axis1_family_avg"]["u"], float)
    assert J["aa_order"] == list(AA20), "aaindex/AA20 顺序不一致，中止"
    print("两组家族平均的普适轴 u（docs/21 §5.2，留一 |cos| >= 0.956）已载入。")

    # ---- 候选 ----
    cands = []
    for d in cand_titeseq():
        d["family"] = "4-4-20 scFv / 荧光素（小分子半抗原）"
        cands.append(d)
    mg, _ = cand_magma()
    for d in mg:
        if d["name"] in ("magma/topbinHA_rep2/319-345",
                         "magma/all_barcodes_319-345/319-345",
                         "magma/topbinHA_rep2/222-1C06"):
            cands.append(d)

    # ---- 既有保留集的 9 个矩阵（做一维成对对照） ----
    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    mats = pick_primary(mats)
    keep = set(J["groups"]["可分辨全集（4 OMTKY3 单酶 + 5 冠状病毒，**已去 GB1**）"])
    ref = {}
    for d in mats:
        if d["name"] in keep:
            w, c, _ = rank1_dir(d["M"], d["mask"])
            if w is not None:
                ref[d["name"]] = (w, c, d.get("backbone", d["name"]))
    print(f"既有保留集里能抽出秩 1 方向的参照矩阵：{len(ref)}/{len(keep)}\n")

    out = {"p95_1d": p95_1d, "null_1d_median": float(np.median(nl)),
           "u": u.tolist(), "candidates": []}

    print("=" * 96)
    print("[1] 秩 1 的可分辨性（对半分自对齐 = 保守口径；bootstrap = 乐观口径）")
    print("=" * 96)
    print(f"  {'候选':<40}{'位点':>5}{'观测':>6}{'残基列':>7}"
          f"{'对半分 |cos| 中位 [2.5,97.5]':>30}{'bootstrap 稳定':>14}")
    print("  " + "-" * 94)
    res = {}
    for d in cands:
        r = resolvability_1d(d)
        if r is None:
            print(f"  {d['name'][:39]:<40}  秩 1 也抽不出子空间")
            continue
        res[d["name"]] = r
        print(f"  {d['name'][:39]:<40}{len(d['sites']):>5}{d['n_obs']:>6}"
              f"{len(r['cols']):>7}"
              f"{r['split_med']:>14.3f} [{r['split_ci'][0]:.2f},"
              f"{r['split_ci'][1]:.2f}]"
              f"{r['stab_med']:>14.3f}")
    print(f"  （一维随机零分布：中位 {np.median(nl):.3f}，p95 "
          f"{p95_1d:.3f} —— 对半分自对齐若不超 p95，这条轴连自己都定不出）")

    print("\n" + "=" * 96)
    print("[2] 候选的那条轴 vs 两组共识的普适轴 u")
    print("=" * 96)
    # **零分布必须在实际的共同残基数 d 上生成。** 第一版在 d = 10 的
    # 比较上误用了 d = 20 的零分布：d=10 的 p95 是 0.632 而不是 0.447
    # （|cos| 的零分布随 d 下降而整体上移，E|cos| = sqrt(2/(pi(d-1)))）。
    # 这与 docs/21「一个 k 的 PR 不能和另一个 k 建的零分布比」同一条纪律。
    null_by_d = {}

    def null_d(d):
        if d not in null_by_d:
            null_by_d[d] = haar_1d_null(d, n=100000, seed=1)
        return null_by_d[d]

    print("  各维数下的一维随机零分布（量级校准）：")
    for dd in (6, 8, 10, 12, 16, 20):
        q = null_d(dd)
        print(f"     d = {dd:>2}  中位 {np.median(q):.3f}  p95 "
              f"{np.percentile(q, 95):.3f}  解析 E|cos| = "
              f"{math.sqrt(2 / (math.pi * (dd - 1))):.3f}（模拟 {q.mean():.3f}）")
    cols20 = np.arange(20)
    for name, r in res.items():
        a = align_1d(r["w"], r["cols"], u, cols20)
        if a is None:
            print(f"  {name}")
            print(f"     与 u 的共同残基只有 {len(r['cols'])} 个 "
                  f"(< 8)，align 的最低共同残基数都过不了 —— 无法比较")
            r["cos_u"] = None; r["p_N1"] = None; r["p_N2"] = None
            continue
        # N2：位点内置换残基标签（摧毁 site x residue 耦合）
        d = next(x for x in cands if x["name"] == name)
        rng = np.random.default_rng(11)
        perm = []
        for _ in range(N_PERM):
            M2, mk2 = permute_within_sites(d["M"], d["mask"], rng)
            M2 = np.where(mk2, M2, 0.0)
            wp, cp, _ = rank1_dir(M2, mk2)
            if wp is None:
                continue
            q = align_1d(wp, cp, u, cols20)
            if q:
                perm.append(q["cos"])
        nld = null_d(a["n_res"])
        p95d = float(np.percentile(nld, 95))
        pN1 = float(np.mean(nld >= a["cos"]))
        pN2 = (float(np.mean(np.array(perm) >= a["cos"]))
               if perm else float("nan"))
        print(f"  {name}")
        print(f"     |cos(w, u)| = {a['cos']:.3f}  (d = {a['n_res']})")
        print(f"     N1 随机一维零分布（**在 d = {a['n_res']} 上生成**）"
              f" p = {pN1:.4f}（该 d 的 p95 = {p95d:.3f}、中位 "
              f"{np.median(nld):.3f}）")
        r["p95_at_d"] = p95d; r["d"] = a["n_res"]
        print(f"     N2 位点内置换 p = {pN2:.4f}"
              f"（{len(perm)}/{N_PERM} 次成功，中位 "
              f"{np.median(perm) if perm else float('nan'):.3f}，"
              f"p95 {np.percentile(perm, 95) if perm else float('nan'):.3f}）")
        r["cos_u"] = a["cos"]; r["p_N1"] = pN1; r["p_N2"] = pN2
        r["perm_med"] = float(np.median(perm)) if perm else None

    print("\n" + "=" * 96)
    print("[3] 一维成对统计：候选 vs 既有保留集的 9 个矩阵（各自重拟合秩 1）")
    print("=" * 96)
    for name, r in res.items():
        cs = []
        for rn, (w, c, _) in sorted(ref.items()):
            q = align_1d(r["w"], r["cols"], w, c)
            if q:
                # 零分布在**这一对自己的** d 上取，不是一个全局 d
                cs.append((rn, q["cos"], q["n_res"],
                           float(np.percentile(null_d(q["n_res"]), 95))))
        if not cs:
            continue
        v = np.array([x[1] for x in cs])
        over = np.array([x[1] > x[3] for x in cs])
        ds = sorted({x[2] for x in cs})
        print(f"  {name}  （n = {len(cs)} 对，d = {ds}）")
        print(f"     |cos| 中位 {np.median(v):.3f}  范围 "
              f"{v.min():.3f}-{v.max():.3f}  超**各自 d 的** N1 p95 "
              f"{over.mean():.0%}")
        for rn, x, dd, pp in cs:
            print(f"       {rn[:48]:<50}{x:>7.3f}  (d={dd}, p95={pp:.3f})")
        r["pairwise"] = [{"ref": a, "cos": float(b), "d": c, "p95": e}
                         for a, b, c, e in cs]
        r["pairwise_med"] = float(np.median(v))
        r["pairwise_frac_over"] = float(over.mean())

    print("\n" + "=" * 96)
    print("[4] 候选那条轴的化学身份（它独立地指向体积/极化率那一支吗？）")
    print("=" * 96)
    for name, r in res.items():
        w = np.asarray(r["w"], float).ravel()
        cl = np.asarray(r["cols"])
        w = w - w.mean(); w = w / np.linalg.norm(w)
        rows = []
        for sn, sv in scales.items():
            x = np.asarray(sv, float)[cl]
            m = np.isfinite(x)
            if m.sum() < 12:
                continue
            xx = x[m] - x[m].mean()
            ww = w[m] - w[m].mean()
            if np.linalg.norm(xx) < 1e-12 or np.linalg.norm(ww) < 1e-12:
                continue
            rr = float((xx / np.linalg.norm(xx)) @ (ww / np.linalg.norm(ww)))
            rows.append((abs(rr), sn, rr, int(m.sum())))
        rows.sort(reverse=True)
        print(f"  {name}  （符号任意，看 |r|）")
        for ar, sn, rr, nn in rows[:8]:
            print(f"     {sn[:36]:<38}r = {rr:+.3f}  r² = {rr * rr:.3f}  "
                  f"(n = {nn})")
        r["scale_corr"] = [{"scale": sn, "r": rr, "r2": rr * rr, "n": nn}
                           for _, sn, rr, nn in rows]

    for name, r in res.items():
        out["candidates"].append(
            {k: v for k, v in r.items() if k not in ("w", "cols")}
            | {"name": name, "cols": [int(x) for x in r["cols"]],
               "w": [float(x) for x in np.asarray(r["w"]).ravel()]})

    p = os.path.join(ROOT, "cache", "third_family_rank1.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n产物: {p}")
    print("\n注意：本脚本的任何数字都**不是**证据集的一部分。"
          "把秩 1 候选纳入判别集 = CLAUDE.md §9.1 的人类决策点。")


if __name__ == "__main__":
    main()
