#!/usr/bin/env python3
"""docs/29：追 docs/27（n=144 阳性）与 docs/28 §8.2（配对 n=1 阴性）的矛盾。

预先登记：`cache/docs29_prereg.md`，sha256 锁在 `cache/docs29_prereg_lock.json`。
本脚本**硬前置**校验该锁，锁不上就中止（CLAUDE.md §4：论证必须先于结果落盘）。

口径（预先登记 §1）：配对 mask 一律用 docs/28 的交集口径
（`ddpca_paired.py`：两边非 NA、两边 SE ≤ 0.5、两边 |ΔΔG| ≤ 10、每行 ≥ 4 格）。

各段对应预先登记的候选：
  H0  配对 mask 计数差异的归因（docs/28 619 vs docs/15 595）
  H1  信息性缺失：完全观测子矩阵（**先** §10.3 还原诊断，**再**看 |cos|）
  H2  衰减校正 + ρ 扫描（功效的正确检验）
  H3  mochi 解卷过度校正（主效应落差 / 格子层带符号 r / 三角化）
  H5  跨数据集一致性的共同偏倚（流程诱导方向 / 缺失模式的残基偏倚）
  H6  层级错配（配对的"结合臂"与 n=144 的"结合轴"不是同一个对象）

数值层复用，不重写：`ddpca_load`、`rank_core.center_fit`、
`subspace_lib.extract_subspace`（内部 emsvd，已与 `als_guarded` 交叉核验）、
`haar_1d_null`、`permute_within_sites`、`gate_calib_lib.split_half`、
`third_family_rank1.rank1_dir / align_1d`、`tsuboyama_load.load_cached`。

运行：python3 analysis/paired_conflict.py
产物：cache/docs29_paired.json
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpca_load as DL                                          # noqa: E402
import gate_calib_lib as G                                       # noqa: E402
import rank_core as RC                                           # noqa: E402
import tsuboyama_load as TL                                      # noqa: E402
from subspace_lib import (AA20, ROOT, extract_subspace,          # noqa: E402
                          haar_1d_null, permute_within_sites)
from third_family_rank1 import align_1d, rank1_dir               # noqa: E402

OUT = os.path.join(ROOT, "cache", "docs29_paired.json")
DOM = "GRB2-SH3"
NSPLIT = 400          # D-036 的最保守一档（docs/28 §6.3）
NPERM = 400
SIG_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]


# ======================================================================
# 0. 硬前置
# ======================================================================
def prerequisites(verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("=" * 96)
    P("[硬前置] 预先登记锁 + gate harness（不过即中止）")
    P("=" * 96)
    lk = json.load(open(os.path.join(ROOT, "cache",
                                     "docs29_prereg_lock.json")))
    p = os.path.join(ROOT, lk["file"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h == lk["sha256"], (
        "预先登记文件已被改动！CLAUDE.md §4 禁止事后修改论证。\n"
        f"  锁定 {lk['sha256']}\n  现在 {h}")
    P(f"  预先登记锁: {lk['file']} sha256 {h[:16]}… 一致（锁定于 "
      f"{lk['locked_at']}）  OK")
    gh = json.load(open(os.path.join(ROOT, "cache",
                                     "gate_harness_check.json")))
    assert gh.get("ok") is True, "gate_harness_check.json 的 ok 不为 true"
    P("  cache/gate_harness_check.json 的 ok = true  OK")
    return {"prereg_sha256": h, "prereg_locked_at": lk["locked_at"],
            "gate_harness_ok": True}


# ======================================================================
# 1. 新工具 + 用已知事实的自检（CLAUDE.md §10）
# ======================================================================
def signed_cos(a, cols_a, b, cols_b):
    """**带符号**的 cos，口径与 align_1d 逐字一致（先限制再中心化）。

    带符号是 H3 的判别量：过度校正预测**负**，而「本来没有共享成分」
    预测 ≈ 0。|cos| 分不开这两件事，所以必须单独有这个工具。
    """
    common = np.intersect1d(cols_a, cols_b)
    if len(common) < 8:
        return None, 0
    x = np.asarray(a, float).ravel()[np.searchsorted(cols_a, common)]
    y = np.asarray(b, float).ravel()[np.searchsorted(cols_b, common)]
    x = x - x.mean()
    y = y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return None, 0
    return float((x / nx) @ (y / ny)), int(len(common))


def tri_bound(c1, c2):
    """三角化的硬几何下界：若 |cos(a,u)| = c1、|cos(b,u)| = c2，
    则 |cos(a,b)| >= c1*c2 - sqrt(1-c1^2)*sqrt(1-c2^2)（可为负 -> 无约束）。
    这一条是**与 σ 无关**的，所以它是本文档最强的那个约束。"""
    return c1 * c2 - math.sqrt(max(0.0, 1 - c1 ** 2)) * \
        math.sqrt(max(0.0, 1 - c2 ** 2))


def resid_interaction(M, mask):
    """双向中心化后的交互残差（口径与 rank_core.center_fit 一致）。"""
    r, c = RC.center_fit(M, mask)
    R = M - r[:, None] - c[None, :]
    return np.where(mask, R, np.nan)


def selftest(verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 96)
    P("[自检] 本文档新建工具，用已知事实（CLAUDE.md §10；不过即中止）")
    P("=" * 96)
    res = {}
    d = 20
    cols = np.arange(d)
    rng = np.random.default_rng(3)

    # [S1] signed_cos 能命中 + 不误伤 + 符号正确
    x = rng.normal(size=d)
    s_same = signed_cos(x, cols, x, cols)[0]
    s_neg = signed_cos(x, cols, -x, cols)[0]
    y = rng.normal(size=d)
    s_rnd = abs(signed_cos(x, cols, y, cols)[0])
    assert abs(s_same - 1) < 1e-10 and abs(s_neg + 1) < 1e-10, \
        f"[S1] 自比 {s_same} / 反号 {s_neg}"
    nl = haar_1d_null(d, n=20000, seed=5)
    assert s_rnd < float(np.percentile(nl, 99)), f"[S1] 随机对 {s_rnd}"
    # 不误伤：签名必须能把「负相关」与「零相关」分开
    z = -0.6 * x / np.linalg.norm(x) + 0.8 * (
        y - x * (x @ y) / (x @ x)) / np.linalg.norm(y - x * (x @ y) / (x @ x))
    s_mix = signed_cos(x, cols, z, cols)[0]
    assert s_mix < -0.4, f"[S1] 植入 -0.6 得到 {s_mix}"
    res["S1"] = {"self": s_same, "neg": s_neg, "random_abs": s_rnd,
                 "implanted_-0.6": s_mix,
                 "haar_p95": float(np.percentile(nl, 95))}
    P(f"  [S1] signed_cos：自比 {s_same:+.6f}、反号 {s_neg:+.6f}、"
      f"随机 |·| {s_rnd:.3f} < Haar p99；植入 −0.6 得 {s_mix:+.3f}  OK")

    # [S2] tri_bound 与暴力枚举一致（已知事实：几何不等式）
    bad = 0
    for _ in range(4000):
        a = rng.normal(size=d); a /= np.linalg.norm(a)
        b = rng.normal(size=d); b /= np.linalg.norm(b)
        u = rng.normal(size=d); u /= np.linalg.norm(u)
        lo = tri_bound(abs(a @ u), abs(b @ u))
        if abs(a @ b) < lo - 1e-9:
            bad += 1
    assert bad == 0, f"[S2] 三角下界被违反 {bad} 次"
    res["S2_violations"] = bad
    P(f"  [S2] tri_bound：4000 次随机三元组，下界被违反 {bad} 次  OK")

    # [S3] 格子层 Pearson 工具：植入已知正相关 / 零相关必须被还原
    n = 4000
    e1 = rng.normal(size=n)
    for truth in (0.0, 0.5, -0.5):
        e2 = truth * e1 + math.sqrt(1 - truth ** 2) * rng.normal(size=n)
        got = float(np.corrcoef(e1, e2)[0, 1])
        assert abs(got - truth) < 0.05, f"[S3] 真 {truth} 得 {got}"
    res["S3"] = "植入 0 / +0.5 / −0.5 全部还原，误差 < 0.05"
    P("  [S3] 格子层 Pearson：植入 0 / +0.5 / −0.5 全部还原（误差 < 0.05）  OK")
    return res


# ======================================================================
# 2. 配对 mask 与两条轴
# ======================================================================
def paired_setup(verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    D = DL.load_domain(DOM)
    DL.verify(DOM, D, verbose=False)
    Mf, mkf, cf = DL.matrix(D, "folding")
    Mb, mkb, cb = DL.matrix(D, "binding")
    mp = mkf & mkb
    keep = mp.sum(axis=1) >= 4
    mp = mp & keep[:, None]
    design = cf["design"]
    P(f"\n  配对 mask（docs/28 交集口径）: {int(mp.sum())} / {design} "
      f"= 缺失 {1 - mp.sum() / design:.1%}（docs/28 §8.2 记 619 / 41.8%）")
    return D, Mf, mkf, Mb, mkb, mp, design


def axis1(M, mask):
    w, cols, err = rank1_dir(np.where(mask, M, np.nan), mask)
    return w, cols, err


# ======================================================================
# 3. H0：配对 mask 计数差异归因
# ======================================================================
def h0(verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 96)
    P("H0 配对 mask 计数差异的归因（docs/28 619 vs docs/15 595）")
    P("=" * 96)
    import folding_vs_binding as FVB
    out = {}
    for dom, c28, c15 in (("GRB2-SH3", 619, 595), ("PSD95-PDZ3", 1087, 1029),
                          ("GB1", 908, 901)):
        res, _ = FVB.build(dom, paired=True)
        n = int(res["folding"][1].sum())
        out[dom] = {"fvb_rerun": n, "docs28": c28, "docs15": c15,
                    "row_threshold": FVB.CV["max_rank"] + 3}
        P(f"  {dom:<12} folding_vs_binding.build(paired=True) 重算 = **{n}**"
          f"；docs/28 = {c28}；docs/15 = {c15}"
          f"  -> {'与 docs/15 一致' if n == c15 else ('与 docs/28 一致' if n == c28 else '与两者都不同')}")
    P(f"  两套口径的唯一已知差异：行阈值（docs/28 每行 ≥ 4 格；"
      f"folding_vs_binding 每行 ≥ max_rank+3 = {out['GB1']['row_threshold']} 格）")
    return out


# ======================================================================
# 4. H3 / H6：三角化 + 主效应落差 + 格子层带符号 r
# ======================================================================
def h3_h6(D, Mf, mkf, Mb, mkb, mp, verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 96)
    P("H3 / H6 三角化（配对 mask 上两条轴 vs binding 共识普适轴 u）")
    P("=" * 96)
    gf = json.load(open(os.path.join(ROOT, "cache",
                                     "generic_folding_axis.json")))
    assert gf["aa_order"] == AA20, "u 的残基顺序与 AA20 不一致（D-023 同型）"
    u = np.asarray(gf["u"], float)
    ucols = np.arange(len(AA20))
    af_full = np.asarray(gf["observed"][f"{DOM}/folding"]["axis"], float)

    wf, cfl, _ = axis1(Mf, mp)
    wb, cbl, _ = axis1(Mb, mp)
    wf_full, cf_full, _ = axis1(Mf, mkf)
    wb_full, cb_full, _ = axis1(Mb, mkb)

    def A(a, ca, b, cb):
        r = align_1d(a, ca, b, cb)
        s, _ = signed_cos(a, ca, b, cb)
        return (r["cos"] if r else None, s, r["n_res"] if r else 0)

    pairs = {
        "折叠轴_paired × 结合轴_paired": A(wf, cfl, wb, cbl),
        "折叠轴_paired × u": A(wf, cfl, u, ucols),
        "结合轴_paired × u": A(wb, cbl, u, ucols),
        "折叠轴_paired × 折叠轴_full": A(wf, cfl, wf_full, cf_full),
        "结合轴_paired × 结合轴_full": A(wb, cbl, wb_full, cb_full),
        "折叠轴_full × u": A(wf_full, cf_full, u, ucols),
        "结合轴_full × u": A(wb_full, cb_full, u, ucols),
        "折叠轴_full × docs/26 登记的同一条轴": A(wf_full, cf_full,
                                           af_full, ucols)}
    d = max(v[2] for v in pairs.values())
    p95 = float(np.percentile(haar_1d_null(d, n=40000, seed=3), 95))
    med = float(np.median(haar_1d_null(d, n=40000, seed=3)))
    P(f"  零分布 d = {d}：Haar 一维 |cos| 中位 {med:.3f}、p95 {p95:.3f}")
    P(f"  {'比较':<40}{'|cos|':>8}{'带符号 cos':>12}{'n_res':>7}{'超 p95':>8}")
    for k, (c, s, nr) in pairs.items():
        P(f"  {k:<40}{c:>8.3f}{s:>+12.3f}{nr:>7}"
          f"{('是' if c > p95 else '否'):>8}")

    c_fu = pairs["折叠轴_paired × u"][0]
    c_bu = pairs["结合轴_paired × u"][0]
    lo = tri_bound(c_fu, c_bu)
    P(f"\n  **三角化硬下界（与 σ 无关）**：若 |cos(折叠_paired, u)| = {c_fu:.3f}"
      f" 且 |cos(结合_paired, u)| = {c_bu:.3f}，"
      f"则 |cos(折叠_paired, 结合_paired)| ≥ {lo:+.3f}")
    P(f"  实测 {pairs['折叠轴_paired × 结合轴_paired'][0]:.3f} "
      f"{'满足' if pairs['折叠轴_paired × 结合轴_paired'][0] >= lo - 1e-9 else '**违反**'}"
      f" —— 下界 {'不构成约束（结合臂对 u 本身在碰巧水平）' if lo <= 0 else '构成约束'}")

    # ---- H3.1 主效应层的落差（ΔΔG 层 vs fitness 层 0.770） ----
    rf, cf_main = RC.center_fit(np.where(mp, Mf, np.nan), mp)
    rb, cb_main = RC.center_fit(np.where(mp, Mb, np.nan), mp)
    allc = np.arange(len(AA20))
    s_main, n_main = signed_cos(cf_main, allc, cb_main, allc)
    P(f"\n  **H3.1 主效应落差**：ΔΔG 层两表型的残基主效应 "
      f"|cos| = {abs(s_main):.3f}、**带符号 cos = {s_main:+.3f}**（n_res {n_main}）")
    P(f"      对读 fitness 层（docs/28 §8.2 限定 2）: 0.770")
    P(f"      预先登记判读：带符号为**负** → 过度校正；≈ 0 → 本来没有共享成分")

    # ---- H3.2 格子层带符号 r + 位点内置换零分布 ----
    Rf = resid_interaction(np.where(mp, Mf, np.nan), mp)
    Rb = resid_interaction(np.where(mp, Mb, np.nan), mp)
    a = Rf[mp]
    b = Rb[mp]
    r_obs = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(29)
    nullr = []
    for _ in range(NPERM * 2):
        Bp, mkp = permute_within_sites(np.where(mp, Mb, np.nan), mp, rng)
        both = mp & mkp
        if both.sum() < 100:
            continue
        Rp = resid_interaction(np.where(mkp, Bp, np.nan), mkp)
        nullr.append(float(np.corrcoef(Rf[both], Rp[both])[0, 1]))
    nullr = np.array(nullr)
    # 位点层 bootstrap 区间（§5：点估计不单独报；位点层已知偏低，照记）
    rows = np.where(mp.any(axis=1))[0]
    bs = []
    rb_rng = np.random.default_rng(31)
    for _ in range(2000):
        pick = rb_rng.choice(rows, size=len(rows), replace=True)
        aa = np.concatenate([Rf[i][mp[i]] for i in pick])
        bb = np.concatenate([Rb[i][mp[i]] for i in pick])
        if len(aa) > 50:
            bs.append(float(np.corrcoef(aa, bb)[0, 1]))
    P(f"\n  **H3.2 格子层带符号相关**：r(折叠交互残差, 结合交互残差) = "
      f"**{r_obs:+.3f}**（n = {int(mp.sum())} 格）")
    P(f"      位点层 bootstrap 95% [{np.percentile(bs, 2.5):+.3f}, "
      f"{np.percentile(bs, 97.5):+.3f}]（位点层已知偏低，照 docs/13 §6.2 记）")
    P(f"      位点内置换零分布: 中位 {np.median(nullr):+.3f}、"
      f"5% 分位 {np.percentile(nullr, 5):+.3f}、"
      f"95% 分位 {np.percentile(nullr, 95):+.3f}")
    verdict = ("过度校正（H3）" if r_obs < np.percentile(nullr, 5)
               else "本来没有共享成分（非 H3）"
               if r_obs <= np.percentile(nullr, 95) else "正相关（有共享成分）")
    P(f"      -> **{verdict}**")

    return {
        "u_aa_order_ok": True, "d": d, "haar_p95": p95, "haar_median": med,
        "pairs": {k: {"abscos": v[0], "signed": v[1], "n_res": v[2],
                      "over_p95": bool(v[0] > p95)}
                  for k, v in pairs.items()},
        "tri_bound": lo,
        "main_effect_ddg_signed": s_main,
        "main_effect_ddg_abs": abs(s_main),
        "main_effect_fitness_abs_docs28": 0.770,
        "cell_r_observed": r_obs,
        "cell_r_boot_site_ci": [float(np.percentile(bs, 2.5)),
                                float(np.percentile(bs, 97.5))],
        "cell_r_null_median": float(np.median(nullr)),
        "cell_r_null_p5": float(np.percentile(nullr, 5)),
        "cell_r_null_p95": float(np.percentile(nullr, 95)),
        "cell_r_verdict": verdict}, u, ucols, wf, cfl, wb, cbl


# ======================================================================
# 5. H1：完全观测子矩阵（先还原诊断，再看 |cos|）
# ======================================================================
def complete_block(mp, min_rows=12, min_cols=8):
    """取「完全观测块」：贪心剔除观测最少的行/列直到块全满。
    定义在看任何 |cos| 之前写定（预先登记 §4：只有一个定义，不扫阈值）。"""
    rows = np.where(mp.any(axis=1))[0]
    cols = np.arange(mp.shape[1])
    sub = mp[np.ix_(rows, cols)]
    while not sub.all():
        rf = sub.mean(axis=1)
        cf = sub.mean(axis=0)
        if rf.min() <= cf.min():
            k = int(np.argmin(rf))
            rows = np.delete(rows, k)
        else:
            k = int(np.argmin(cf))
            cols = np.delete(cols, k)
        if len(rows) < min_rows or len(cols) < min_cols:
            return None, None
        sub = mp[np.ix_(rows, cols)]
    return rows, cols


def h1(Mf, Mb, mp, verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 96)
    P("H1 信息性缺失：完全观测子矩阵（**顺序是判据的一部分**：先还原，再看 |cos|）")
    P("=" * 96)
    rows, cols = complete_block(mp)
    if rows is None:
        P("  完全观测块小于 12×8，H1 **不可判定**（没有可用子矩阵）")
        return {"ok": False, "why": "完全观测块 < 12x8"}
    sub = np.zeros_like(mp)
    sub[np.ix_(rows, cols)] = True
    nsub = int(sub.sum())
    P(f"  完全观测块: {len(rows)} 位点 × {len(cols)} 残基 = {nsub} 格"
      f"（缺失 0.0%，对照配对全集 {int(mp.sum())} 格 / 41.8% 缺失）")

    Fs = np.where(sub, Mf, np.nan)
    Bs = np.where(sub, Mb, np.nan)

    # --- 第一步：§10.3 还原诊断（在这个子矩阵自己的几何上） ---
    wf0, cf0, _ = axis1(Mf, sub)
    if wf0 is None:
        P("  子矩阵抽不出第一轴 -> **不可判定**")
        return {"ok": False, "why": "子矩阵抽不出第一轴",
                "n_rows": len(rows), "n_cols": len(cols)}
    # σ 由「让合成矩阵的位点对半可复现度贴住实测」反解（不是物理 σ，
    # 只是让模拟与数据的可复现度一致；D-028/D-055 禁止的是把它当真噪声）
    rb_obs = G.split_half(Bs, sub, rank=1, nsplit=NSPLIT, seed=11)
    rb_obs = float(np.median(rb_obs)) if rb_obs else float("nan")
    rf_obs = G.split_half(Fs, sub, rank=1, nsplit=NSPLIT, seed=12)
    rf_obs = float(np.median(rf_obs)) if rf_obs else float("nan")
    d_sub = len(cols)
    p95s = float(np.percentile(haar_1d_null(d_sub, n=40000, seed=3), 95))
    P(f"  子矩阵的 D-036（位点对半，nsplit={NSPLIT}）: "
      f"折叠 {rf_obs:.3f}、结合 {rb_obs:.3f} vs d={d_sub} 的 Haar p95 {p95s:.3f}"
      f" -> 折叠{'过' if rf_obs > p95s else '**不过**'}、"
      f"结合{'过' if rb_obs > p95s else '**不过**'}")

    rec = {}
    utrue = wf0 / np.linalg.norm(wf0)
    main_r, main_c = RC.center_fit(Bs, sub)
    coup = float(np.nanstd(resid_interaction(Bs, sub)[sub]))
    for sg in SIG_GRID:
        got, repro = [], []
        for t in range(40):
            rng = np.random.default_rng(900 + t + int(sg * 1000))
            A = rng.normal(0, coup, (len(rows), 1))
            X = (A @ utrue[None, :] + main_r[rows][:, None]
                 + main_c[cols][None, :] + rng.normal(0, sg,
                                                      (len(rows), len(cols))))
            Xf = np.full(mp.shape, np.nan)
            Xf[np.ix_(rows, cols)] = X
            w, cc, _ = axis1(Xf, sub)
            if w is None:
                continue
            r = align_1d(w, cc, utrue, cols)
            if r:
                got.append(r["cos"])
            if t < 8:
                v = G.split_half(Xf, sub, rank=1, nsplit=80, seed=50 + t)
                if v:
                    repro.append(float(np.median(v)))
        rec[sg] = {"recovery_median": float(np.median(got)) if got else None,
                   "recovery_ci": [float(np.percentile(got, 2.5)),
                                   float(np.percentile(got, 97.5))]
                   if len(got) > 10 else None,
                   "synth_repro_median": (float(np.median(repro))
                                          if repro else None)}
    P(f"\n  §10.3 还原诊断（植入秩 1 真方向 = 子矩阵自己的折叠轴，"
      f"走完全相同流程，每格 40 次）：")
    P(f"  {'σ':>6}{'还原 |cos| 中位':>18}{'95% 区间':>24}{'合成可复现度':>14}")
    for sg, v in rec.items():
        ci = (f"[{v['recovery_ci'][0]:.3f}, {v['recovery_ci'][1]:.3f}]"
              if v["recovery_ci"] else "n/a")
        P(f"  {sg:>6.2f}{v['recovery_median']:>18.3f}{ci:>24}"
          f"{(v['synth_repro_median'] if v['synth_repro_median'] else float('nan')):>14.3f}")
    # 选与实测结合可复现度最接近的 σ 作主用，并报标定残差
    cands = [(abs(v["synth_repro_median"] - rb_obs), sg)
             for sg, v in rec.items() if v["synth_repro_median"] is not None]
    resid, sg_star = min(cands)
    P(f"\n  标定：与实测结合可复现度 {rb_obs:.3f} 最接近的是 σ = {sg_star}"
      f"（合成 {rec[sg_star]['synth_repro_median']:.3f}，残差 {resid:.3f}）")
    rec_star = rec[sg_star]["recovery_median"]
    P(f"  该 σ 下的还原 |cos| 中位 = **{rec_star:.3f}**")
    powered = rec_star >= 0.80
    undec = rec_star < 0.50
    P(f"  预先登记判读：还原 ≥ 0.80 → 有功效；< 0.50 → **不可判定**"
      f"  -> {'有功效' if powered else ('**不可判定**' if undec else '中间档')}")

    # --- 第二步：子矩阵上的 |cos| ---
    wb0, cb0, _ = axis1(Mb, sub)
    r = align_1d(wf0, cf0, wb0, cb0) if wb0 is not None else None
    s0, _ = signed_cos(wf0, cf0, wb0, cb0) if wb0 is not None else (None, 0)
    rng = np.random.default_rng(37)
    pv = []
    for _ in range(NPERM):
        Bp, mkp = permute_within_sites(Bs, sub, rng)
        wp, cp, _ = axis1(Bp, mkp)
        if wp is None:
            continue
        q = align_1d(wf0, cf0, wp, cp)
        if q:
            pv.append(q["cos"])
    pv = np.array(pv)
    P(f"\n  **子矩阵上的 |cos(折叠轴, 结合轴)| = "
      f"{(r['cos'] if r else float('nan')):.3f}**（带符号 {s0:+.3f}）"
      f"；对照配对全集 0.016")
    P(f"  子矩阵自己的位点内置换零分布: 中位 {np.median(pv):.3f}、"
      f"p95 {np.percentile(pv, 95):.3f}、p = "
      f"{float(np.mean(pv >= r['cos'])) if r else float('nan'):.4f}")
    att = (r["cos"] / math.sqrt(rf_obs * rb_obs)
           if r and rf_obs > 0 and rb_obs > 0 else None)
    P(f"  衰减校正后 {att:.3f}" if att else "")
    moved = bool(r and r["cos"] > np.percentile(pv, 95))
    v = ("支持 H1" if (powered and moved) else
         "否证 H1" if (powered and not moved) else
         "不可判定（子矩阵无功效）" if undec else "中间档，不判")
    P(f"  -> **{v}**")
    return {"ok": True, "n_rows": len(rows), "n_cols": len(cols),
            "n_cells": nsub, "rows": rows.tolist(), "cols": cols.tolist(),
            "d_sub": d_sub, "haar_p95_sub": p95s,
            "repro_folding": rf_obs, "repro_binding": rb_obs,
            "recovery_table": {str(k): v for k, v in rec.items()},
            "sigma_star": sg_star, "calib_resid": resid,
            "recovery_at_sigma_star": rec_star,
            "powered": powered, "undecidable": undec,
            "abscos": r["cos"] if r else None, "signed_cos": s0,
            "abscos_attenuation_corrected": att,
            "perm_median": float(np.median(pv)),
            "perm_p95": float(np.percentile(pv, 95)),
            "perm_p": float(np.mean(pv >= r["cos"])) if r else None,
            "verdict": v}


# ======================================================================
# 6. H2：衰减校正区间 + ρ 扫描
# ======================================================================
def h2(Mf, Mb, mp, wf, cfl, verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 96)
    P("H2 闸边缘功效：衰减校正（含区间）+ ρ 扫描（功效的正确检验）")
    P("=" * 96)
    Fs = np.where(mp, Mf, np.nan)
    Bs = np.where(mp, Mb, np.nan)
    rf = G.split_half(Fs, mp, rank=1, nsplit=NSPLIT, seed=12)
    rb = G.split_half(Bs, mp, rank=1, nsplit=NSPLIT, seed=11)
    rfm, rbm = float(np.median(rf)), float(np.median(rb))
    wb, cbl, _ = axis1(Mb, mp)
    obs = align_1d(wf, cfl, wb, cbl)["cos"]
    d = len(np.intersect1d(cfl, cbl))
    p95 = float(np.percentile(haar_1d_null(d, n=40000, seed=3), 95))
    P(f"  实测 |cos| = {obs:.3f}；D-036 折叠 {rfm:.3f}、结合 {rbm:.3f}"
      f" vs Haar p95（d={d}）{p95:.3f}（docs/28 §8.2 记 0.016 / 0.522 / 0.481）")

    # 衰减校正 + 区间（对两个 split-half 分布重抽，传播到比值）
    rng = np.random.default_rng(41)
    bs = []
    for _ in range(2000):
        a = float(np.median(rng.choice(rf, size=len(rf), replace=True)))
        b = float(np.median(rng.choice(rb, size=len(rb), replace=True)))
        if a > 0 and b > 0:
            bs.append(obs / math.sqrt(a * b))
    att = obs / math.sqrt(rfm * rbm)
    P(f"\n  **衰减校正值 = {obs:.3f} / √({rfm:.3f}×{rbm:.3f}) = "
      f"{att:.3f}**，95% 区间 [{np.percentile(bs, 2.5):.3f}, "
      f"{np.percentile(bs, 97.5):.3f}]")
    P(f"  预先登记已写明：乘数 1/√({rfm:.3f}×{rbm:.3f}) = "
      f"{1/math.sqrt(rfm*rbm):.2f}，**在算术上不可能**把 {obs:.3f} 抬到 0.50。"
      f"docs/28 拿未校正的 {obs:.3f} 比未校正期望 "
      f"√({rfm:.3f}×{rbm:.3f}) = {math.sqrt(rfm*rbm):.3f} 是**同口径**的正确比较。")

    # ρ 扫描：先标定 σ（让合成的结合可复现度贴住实测 rbm）
    main_r, main_c = RC.center_fit(Bs, mp)
    coup = float(np.nanstd(resid_interaction(Bs, mp)[mp]))
    cols_all = np.arange(mp.shape[1])
    wfn = np.zeros(mp.shape[1])
    wfn[cfl] = wf
    wfn = wfn - wfn[cfl].mean()
    wfn /= np.linalg.norm(wfn)

    def synth(rho, sg, seed):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=mp.shape[1])
        v -= v.mean()
        v -= wfn * (wfn @ v)
        v /= np.linalg.norm(v)
        w = rho * wfn + math.sqrt(max(0.0, 1 - rho ** 2)) * v
        w /= np.linalg.norm(w)
        A = rng.normal(0, coup, (mp.shape[0], 1))
        X = (A @ w[None, :] + main_r[:, None] + main_c[None, :]
             + rng.normal(0, sg, mp.shape))
        return np.where(mp, X, np.nan)

    calib = []
    for sg in SIG_GRID:
        v = []
        for t in range(6):
            X = synth(1.0, sg, 1200 + t + int(sg * 1000))
            z = G.split_half(X, mp, rank=1, nsplit=80, seed=70 + t)
            if z:
                v.append(float(np.median(z)))
        calib.append((abs(float(np.median(v)) - rbm) if v else 9e9, sg,
                      float(np.median(v)) if v else None))
    calib.sort()
    resid, sg_star, got = calib[0]
    P(f"\n  σ 标定（让合成的结合轴可复现度贴住实测 {rbm:.3f}）: "
      f"σ* = {sg_star}，合成可复现度 {got:.3f}，残差 **{resid:.3f}**")
    assert resid < 0.08, (
        f"σ 标定残差 {resid:.3f} ≥ 0.08：生成模型配不上数据，"
        "ρ 扫描的功效结论**不成立**（docs/28 §4.4 同型）。按 §10 中止。")
    P(f"  标定残差 < 0.08，ρ 扫描可用（残差 ≥ 0.08 时本脚本会断言中止，"
      f"因为那是 docs/28 §4.4 的「生成模型错」）")

    rows = []
    for rho in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.78, 0.85, 0.9, 1.0]:
        got2 = []
        for t in range(60):
            X = synth(rho, sg_star, 2000 + 97 * t + int(rho * 1000))
            w, cc, _ = axis1(X, mp)
            if w is None:
                continue
            r = align_1d(wf, cfl, w, cc)
            if r:
                got2.append(r["cos"])
        g = np.array(got2)
        rows.append({"rho": rho, "n": len(g),
                     "median": float(np.median(g)),
                     "ci": [float(np.percentile(g, 2.5)),
                            float(np.percentile(g, 97.5))],
                     "detect_frac": float(np.mean(g > p95)),
                     "contains_obs": bool(np.percentile(g, 2.5) <= obs
                                          <= np.percentile(g, 97.5))})
    P(f"\n  ρ 扫描（配对 mask 自己的几何 / 缺失模式 / 主效应尺度，σ* = "
      f"{sg_star}，每格 60 次）：")
    P(f"  {'ρ':>6}{'观测 |cos| 中位':>18}{'95% 区间':>24}"
      f"{'检出率(超 p95)':>16}{'区间含 0.016?':>16}")
    for r in rows:
        cis = "[%.3f, %.3f]" % (r["ci"][0], r["ci"][1])
        P(f"  {r['rho']:>6.2f}{r['median']:>18.3f}{cis:>24}"
          f"{r['detect_frac']:>16.2f}{('**是**' if r['contains_obs'] else '否'):>16}")
    pw = [r["rho"] for r in rows if r["detect_frac"] >= 0.80]
    minrho = min(pw) if pw else None
    r078 = [r for r in rows if r["rho"] == 0.78][0]
    P(f"\n  检出率 ≥ 80% 的最小 ρ = **{minrho}**")
    P(f"  ρ = 0.78 这一格：观测 |cos| 中位 {r078['median']:.3f}，95% 区间 "
      f"[{r078['ci'][0]:.3f}, {r078['ci'][1]:.3f}]，"
      f"{'**含**' if r078['contains_obs'] else '**不含**'} 实测 {obs:.3f}")
    # 预先登记的两条判据必须**逐分句**对账，不得用"非支持即否证"偷过去。
    # 第一版这里写的就是"非支持即否证"，于是在 minrho = 0.70 时输出了一个
    # 干净的「否证 H2」—— 而预先登记的否证条件要求 minrho ≤ 0.60。
    # 归属：**我的错**（判读逻辑），不是数据的错；由 docs/29 §5.2 的
    # 逐分句对账抓到。见 docs/29 §7.3 第 4 条。
    sup = (minrho is not None and minrho >= 0.78) or r078["contains_obs"]
    ref = (not r078["contains_obs"]) and (minrho is not None
                                          and minrho <= 0.60)
    v = ("支持 H2" if sup else "否证 H2（阴性有功效）" if ref
         else ("排除为解释，但否证判据的第二分句未满足："
                f"最小 ρ = {minrho} > 0.60；"
                "功效声明降档为「能分开 ρ ≳ 0.7 与 ρ ≲ 0.6」"))
    P(f"  预先登记判读（逐分句对账）-> **{v}**")
    P(f"      支持分句: 最小 ρ ≥ 0.78 ? {minrho is not None and minrho >= 0.78}"
      f"；ρ=0.78 格区间含实测 ? {r078['contains_obs']}")
    P(f"      否证分句: 区间不含实测 ? {not r078['contains_obs']}"
      f"；最小 ρ ≤ 0.60 ? {minrho is not None and minrho <= 0.60}")
    return {"obs_abscos": obs, "repro_folding": rfm, "repro_binding": rbm,
            "haar_p95": p95, "d": d,
            "attenuation_corrected": att,
            "attenuation_ci": [float(np.percentile(bs, 2.5)),
                               float(np.percentile(bs, 97.5))],
            "ceiling_if_identical": math.sqrt(rfm * rbm),
            "sigma_star": sg_star, "calib_resid": resid,
            "calib_synth_repro": got,
            "rho_scan": rows, "min_rho_detect80": minrho,
            "rho078_contains_obs": r078["contains_obs"],
            "support_clause_met": bool(sup), "refute_clause_met": bool(ref),
            "verdict": v}


# ======================================================================
# 7. H5：跨数据集一致性的共同偏倚
# ======================================================================
def h5(verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("\n" + "=" * 96)
    P("H5 共同偏倚：流程诱导方向（纯噪声对照）+ 缺失模式的残基偏倚")
    P("=" * 96)
    gf = json.load(open(os.path.join(ROOT, "cache",
                                     "generic_folding_axis.json")))
    u = np.asarray(gf["u"], float)
    ucols = np.arange(len(AA20))
    ax = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_axis.json")))
    info, mats = TL.load_cached()
    names = [k for k, v in ax["rows"].items() if v.get("repro_ok")]
    P(f"  144 个过闸域: {len(names)}")

    # H5.1 纯噪声走相同流程
    got, over = [], 0
    rng = np.random.default_rng(53)
    for nm in names:
        m = mats[nm]
        mask = m["mask"]
        X = np.where(mask, rng.normal(size=mask.shape), np.nan)
        w, cc, _ = axis1(X, mask)
        if w is None:
            continue
        r = align_1d(w, cc, u, ucols)
        if not r:
            continue
        got.append(r["cos"])
        p95 = float(np.percentile(haar_1d_null(len(cc), n=8000,
                                               seed=len(cc)), 95))
        over += int(r["cos"] > p95)
    got = np.array(got)
    hm = float(np.median(haar_1d_null(20, n=40000, seed=3)))
    P(f"\n  **H5.1 纯噪声对照**（144 个域各自的几何 / 缺失模式，不植入任何结构，"
      f"走相同 `extract_subspace` 流程）：")
    P(f"      |cos(噪声第一轴, u)| 中位 **{np.median(got):.3f}**、"
      f"四分位 [{np.percentile(got, 25):.3f}, {np.percentile(got, 75):.3f}]"
      f"；超各自 p95 的比例 **{over/len(got):.1%}**")
    P(f"      对照：d=20 的 Haar 中位 {hm:.3f}、名义比例 5%；"
      f"实测（docs/27 §5.4）0.755 / 94.4%")
    s1 = ("支持 H5" if (np.median(got) > hm + 0.10 or over / len(got) > 0.15)
          else "否证 H5.1")
    P(f"      -> **{s1}**")

    # H5.2 缺失模式 / n_obs 的残基偏倚
    miss = np.zeros(len(AA20))
    nobs = np.zeros(len(AA20))
    tot = 0
    for nm in names:
        mk = mats[nm]["mask"]
        nobs += mk.sum(axis=0)
        miss += (~mk).sum(axis=0)
        tot += mk.shape[0]
    mrate = miss / tot
    cm = align_1d(mrate, ucols, u, ucols)
    cn = align_1d(nobs, ucols, u, ucols)
    cons = np.asarray(ax["summary"].get("u_fold", gf["u"]), float) \
        if isinstance(ax.get("summary"), dict) and "u_fold" in ax["summary"] \
        else None
    p95_20 = float(np.percentile(haar_1d_null(20, n=40000, seed=3), 95))
    P(f"\n  **H5.2 缺失模式的残基偏倚**（144 个域汇总）：")
    P(f"      |cos(u, 逐残基缺失率向量)| = **{cm['cos']:.3f}**")
    P(f"      |cos(u, 逐残基 n_obs 向量)| = **{cn['cos']:.3f}**")
    P(f"      判据：≥ 0.6 → 支持 H5；< d=20 的 Haar p95 {p95_20:.3f} → 否证")
    s2 = ("支持 H5" if max(cm["cos"], cn["cos"]) >= 0.6
          else "否证 H5.2" if max(cm["cos"], cn["cos"]) < p95_20
          else "中间档")
    P(f"      -> **{s2}**")
    return {"n_domains": len(names),
            "noise_cos_median": float(np.median(got)),
            "noise_cos_q": [float(np.percentile(got, 25)),
                            float(np.percentile(got, 75))],
            "noise_over_p95_frac": over / len(got),
            "haar_median_d20": hm, "haar_p95_d20": p95_20,
            "cos_u_vs_missrate": cm["cos"], "cos_u_vs_nobs": cn["cos"],
            "verdict_h5_1": s1, "verdict_h5_2": s2}


def main():
    out = {"prerequisites": prerequisites(), "selftest": selftest()}
    out["H0"] = h0()
    D, Mf, mkf, Mb, mkb, mp, design = paired_setup()
    out["paired_mask"] = {"n": int(mp.sum()), "design": design,
                          "censor": 1 - mp.sum() / design,
                          "construction": "docs/28 交集口径（ddpca_paired.py）"}
    h36, u, ucols, wf, cfl, wb, cbl = h3_h6(D, Mf, mkf, Mb, mkb, mp)
    out["H3_H6"] = h36
    out["H1"] = h1(Mf, Mb, mp)
    out["H2"] = h2(Mf, Mb, mp, wf, cfl)
    out["H5"] = h5()
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n产物: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
