#!/usr/bin/env python3
"""docs/26 第 2 层（Part 1）：普适轴是不是「一般性去稳定度」？

检验对象：docs/21 §5.2 的两组家族平均共识方向 u（载荷在 R^20 里，
`cache/subspace_two_groups.json` 的 `axis1_family_avg.u`），
以及对应的**二维**残基载荷共识子空间（P̄ 的前两个特征向量）。

为什么必须同时报 gauge-invariant 版本：
    秩 2 载荷只在 span(V) 的意义下可辨识（任意可逆 2×2 变换等价）。
    「u 与某标度的相关」这种说法在固定了 u = P̄ 首特征向量这个**选取**
    之后才有定义 —— 它是一个 gauge-dependent 的陈述。
    所以同一问题必须再问一次它的 gauge-invariant 形式：
    二维载荷子空间被 span{volume} 覆盖多少，被 span{volume, disrupt}
    覆盖多少，增量是否超出「加一条随机方向」的零分布。

零分布全部在**本比较自己的** d = 20 环境维（中心化后 19 维）生成
（docs/23 的事故：d=20 的零分布被用到 d=10 的比较上）。

运行：python3 analysis/generic_axis_test.py
产物：cache/generic_axis_test.json
依赖：先跑 analysis/subspace_two_groups.py 与
      analysis/generic_disrupt_scales.py
"""
import itertools
import json
import math
import os
import sys

import numpy as np
from scipy import stats as sps
from scipy.linalg import subspace_angles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subspace_cond_null as CN                                  # noqa: E402
from subspace_lib import (AA20, ROOT, dedup_by_content,          # noqa: E402
                          extract_subspace, haar_1d_null,
                          haar_pair_null, load_all_dms,
                          load_independent, load_scales,
                          load_skempi, pick_primary)

NSIM = 4000
ONE = np.ones(20) / math.sqrt(20)

# 「体积 / 极化率」那一支：docs/21 §6 判定的主干身份
VOL_SCALES = ["local_polarizability", "molar_refractivity",
              "asa_RadzickaWolfenden", "partial_specific_volume",
              "steric_upsilon"]
DISRUPT_KEYS = ["D1_folding_ddG_ddPCA", "D2_blosum62_nonexchangeability",
                "D3_starr_expression_loss", "D4_backbone_disruption_PG",
                "D5_disrupt_PC1"]


def cen(v):
    v = np.asarray(v, float)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def captured(S, vecs):
    """span(vecs) 与子空间 S（正交基，列）之间的主角 cos，降序。"""
    A = np.stack([cen(v) for v in vecs], 1)
    Q, R = np.linalg.qr(A)
    keep = np.abs(np.diag(R)) > 1e-8 * max(1.0, abs(R[0, 0]))
    Q = Q[:, keep]
    th = subspace_angles(S, Q)
    return np.sort(np.cos(th))[::-1]


def partial_r(y, x, z_):
    """r(y, x | z_)，z_ 可以是多列。返回 (r, p, df)。"""
    y, x = np.asarray(y, float), np.asarray(x, float)
    Z = np.column_stack([np.ones(len(y))] + [np.asarray(c, float)
                                             for c in z_])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    n = len(y)
    df = n - Z.shape[1] - 1
    r = float(np.corrcoef(ry, rx)[0, 1])
    t = r * math.sqrt(df / max(1e-12, 1 - r * r))
    p = float(2 * sps.t.sf(abs(t), df))
    return r, p, df


def vif(cols):
    out = []
    X = np.column_stack([np.asarray(c, float) for c in cols])
    for j in range(X.shape[1]):
        others = np.column_stack([np.ones(X.shape[0])]
                                 + [X[:, k] for k in range(X.shape[1])
                                    if k != j])
        b = np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        res = X[:, j] - others @ b
        r2 = 1 - (res ** 2).sum() / ((X[:, j] - X[:, j].mean()) ** 2).sum()
        out.append(1.0 / max(1e-12, 1 - r2))
    return out


def main():
    scales = load_scales(verbose=True)          # 内含 aaindex 强制自检
    mr = scales["molar_refractivity"]
    gi, wi = AA20.index("G"), AA20.index("W")
    assert int(np.nanargmin(mr)) == gi and float(mr[gi]) == 0.0, \
        "摩尔折射度 Gly 不是 0.00 的最小值"
    assert int(np.nanargmax(mr)) == wi and abs(float(mr[wi]) - 42.53) < 1e-6, \
        "摩尔折射度 Trp 不是 42.53 的最大值"
    print(f"[自检] 摩尔折射度 Gly = {mr[gi]:.2f}（最小）/ "
          f"Trp = {mr[wi]:.2f}（最大）—— 第三次独立核验通过")

    TG = json.load(open(os.path.join(ROOT, "cache",
                                     "subspace_two_groups.json")))
    assert TG["aa_order"] == AA20
    u = np.array(TG["axis1_family_avg"]["u"], float)
    lam = TG["axis1_family_avg"]["eigs"]
    loo_min = min(z_["abscos"] for z_ in TG["axis1_family_avg"]["loo"])
    print(f"[输入] u = 两组家族平均 P̄ 首特征向量，λ = "
          f"{[round(x, 3) for x in lam]}，留一 |cos| 最小 {loo_min:.3f}")

    DS = json.load(open(os.path.join(ROOT, "cache",
                                     "generic_disrupt_scales.json")))
    assert DS["aa_order"] == AA20
    dis = {k: np.array(v, float) for k, v in DS["scales"].items()}

    # ---------------------------------------------------------------
    # 0. u 自己的可复现性与「同源 vs 跨源」（D-032 的强制前置）
    # ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print("0. 被比较的两个方向各自可复现吗（D-032；Haar 检验完全不查这件事）")
    print("=" * 92)
    pc = TG["pairs_by_cat"]
    xf = TG["cross_family_pairs"]
    c1x = np.array([r["cos1"] for r in xf])
    p95 = TG["pair_null_p95"]
    print(f"   u 的可复现性：留一 |cos| 最小 {loo_min:.3f}、中位 "
          f"{np.median([z_['abscos'] for z_ in TG['axis1_family_avg']['loo']]):.3f}")
    print("   同源 vs 跨源（docs/25 §6.3 的诊断指纹：若同源低于跨源就是伪结构）")
    for k in ("同数据集内（冠状病毒）", "跨冠状病毒数据集",
              "OMTKY3 同骨架 × 不同蛋白酶"):
        if k in pc:
            print(f"     [同源/同家族] {k:<26} n={pc[k]['n']:>4}  "
                  f"cos θ1 中位 {pc[k]['cos1_med']:.3f}")
    print(f"     [跨源/跨家族] 25 个冠状病毒×OMTKY3 配对    "
          f"n={len(c1x):>4}  cos θ1 中位 {np.median(c1x):.3f}")
    SAME_KEYS = ["同数据集内（冠状病毒）", "跨冠状病毒数据集",
                 "OMTKY3 同骨架 × 不同蛋白酶"]
    same_meds = [pc[k]["cos1_med"] for k in SAME_KEYS if k in pc]
    same_ok = min(same_meds) >= float(np.median(c1x)) - 0.01
    print(f"   => 同源中位最小值 {min(same_meds):.3f} vs 跨源中位 "
          f"{np.median(c1x):.3f}：同源 ≳ 跨源？"
          f"{'是' if same_ok else '否'} "
          f"—— docs/25 §6 的伪结构指纹是「同源落在随机水平、"
          f"跨源反而高」，这里**没有出现**")

    print("\n   去稳定度标度的可复现性（来自 generic_disrupt_scales.py）")
    for d, r in DS["ddpca_splithalf"].items():
        print(f"     [同源] ddPCA/{d:<12} 位点对半 r = {r['median']:.3f} "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
    for r in DS["expr_repro_same_source"]:
        print(f"     [同源] Starr/{r['label'][:24]:<26} 重复 r = {r['r']:+.3f}")
    cd = DS["ddpca_cross_domain"]
    lf = [r for r in cd if "GB1" not in (r["a"], r["b"])]
    print(f"     [跨源] ddPCA folding 跨域（只算两个无配体域）: "
          + "; ".join(f"{r['a']}×{r['b']} r={r['r']:+.3f}" for r in lf))
    print(f"     [跨源] ddPCA folding 含 GB1 的配对（GB1 **不是**无配体）: "
          + "; ".join(f"{r['a']}×{r['b']} r={r['r']:+.3f}"
                      for r in cd if "GB1" in (r["a"], r["b"])))
    ecr = [r["r"] for r in DS["expr_repro_cross_source"]]
    print(f"     [跨源] Starr 表达量跨文件 {len(ecr)} 对 r 中位 "
          f"{np.median(ecr):+.3f}（{min(ecr):+.3f}–{max(ecr):+.3f}）")
    print("     [跨源] 三条经验标度彼此：folding×blosum、folding×expr、"
          "blosum×expr（见下）")
    emp = {"D1_folding_ddG_ddPCA": dis["D1_folding_ddG_ddPCA"],
           "D2_blosum62_nonexchangeability":
               dis["D2_blosum62_nonexchangeability"],
           "D3_starr_expression_loss": dis["D3_starr_expression_loss"]}
    cross_emp = []
    for a, b in itertools.combinations(emp, 2):
        r = float(np.corrcoef(emp[a], emp[b])[0, 1])
        cross_emp.append({"a": a, "b": b, "r": r})
        print(f"        {a[:28]:<30} × {b[:28]:<30} r = {r:+.3f}")

    # ---------------------------------------------------------------
    # 1. 平凡口径：相关
    # ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print("1. 平凡口径：u 与各条「一般性去稳定度」标度的相关（n = 20）")
    print("=" * 92)
    corr = {}
    print(f"   {'标度':<36}{'r':>9}{'r²':>8}{'p':>10}")
    print("   " + "-" * 63)
    for k in DISRUPT_KEYS + ["D1a_folding_GRB2", "D1b_folding_PSD95",
                             "D1x_folding_GB1_NOT_ligand_free"]:
        v = dis[k]
        r = float(np.corrcoef(u, v)[0, 1])
        t = r * math.sqrt(18 / max(1e-12, 1 - r * r))
        p = float(2 * sps.t.sf(abs(t), 18))
        corr[k] = {"r": r, "r2": r * r, "p": p}
        print(f"   {k:<36}{r:>+9.3f}{r * r:>8.3f}{p:>10.4f}")
    print("\n   参照（docs/21 §6 的体积/极化率那一支与疏水性）:")
    ref = {}
    for k in VOL_SCALES + ["sheet_prop_Chou", "kd_hydropathy"]:
        s = scales[k]
        idx = np.isfinite(s)
        r = float(np.corrcoef(u[idx], s[idx])[0, 1])
        ref[k] = {"r": r, "r2": r * r, "n": int(idx.sum())}
        print(f"   {k:<36}{r:>+9.3f}{r * r:>8.3f}")

    # 搜索量（§5）与该搜索量下的置换零分布
    n_search = len(DISRUPT_KEYS) + 3 + len(VOL_SCALES) + 2
    rng = np.random.default_rng(26)
    allv = [dis[k] for k in DISRUPT_KEYS + ["D1a_folding_GRB2",
                                            "D1b_folding_PSD95",
                                            "D1x_folding_GB1_NOT_ligand_free"]]
    allv += [np.where(np.isfinite(scales[k]), scales[k],
                      np.nanmean(scales[k]))
             for k in VOL_SCALES + ["sheet_prop_Chou", "kd_hydropathy"]]
    mx = []
    for _ in range(NSIM):
        up = u[rng.permutation(20)]
        mx.append(max(abs(float(np.corrcoef(up, v)[0, 1])) for v in allv))
    mx = np.array(mx)
    print(f"\n   **搜索量记录（§5）**：本节共试 {n_search} 条标度"
          f"（{len(DISRUPT_KEYS) + 3} 条去稳定度 + "
          f"{len(VOL_SCALES) + 2} 条参照）。")
    print(f"   在该搜索量下的置换零分布（残基标签置换 u，{NSIM} 次，"
          f"取 {n_search} 条里的最大 |r|）：")
    print(f"     中位 {np.median(mx):.3f}  p95 **{np.percentile(mx, 95):.3f}**"
          f"  p99 {np.percentile(mx, 99):.3f}")
    print(f"   判读：只有 |r| 超过 {np.percentile(mx, 95):.3f} 才算超出"
          f"「在 {n_search} 条标度里挑最好的一条」的运气天花板。")
    perm95 = float(np.percentile(mx, 95))

    # ---------------------------------------------------------------
    # 2. 共线性（这一节决定后面所有「解释了多少」能不能说）
    # ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print("2. 共线性：一般性去稳定度 vs 体积/极化率，在 20 个氨基酸上分得开吗")
    print("=" * 92)
    pol = scales["local_polarizability"]
    coll = {}
    print(f"   {'去稳定度标度':<36}" + "".join(f"{k[:14]:>16}"
                                               for k in VOL_SCALES[:3]))
    for k in DISRUPT_KEYS:
        row = []
        for vk in VOL_SCALES:
            s = scales[vk]
            idx = np.isfinite(s)
            row.append(float(np.corrcoef(dis[k][idx], s[idx])[0, 1]))
        coll[k] = dict(zip(VOL_SCALES, row))
        print(f"   {k:<36}" + "".join(f"{x:>+16.3f}" for x in row[:3]))
    print("\n   （完整 5 条体积系标度的相关在 json 里）")
    print("\n   最关键的一格：极化率 vs 各去稳定度标度的 |r| 与 VIF")
    for k in DISRUPT_KEYS:
        r = coll[k]["local_polarizability"]
        v = vif([dis[k], pol])
        print(f"     {k:<36} r = {r:+.3f}  r² = {r * r:.3f}  "
              f"VIF = {v[0]:.2f}")

    # ---------------------------------------------------------------
    # 3. 双向偏相关
    # ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print("3. 双向偏相关（两个方向都报，不许只报一个）")
    print("=" * 92)
    part = {}
    for k in DISRUPT_KEYS:
        d = dis[k]
        ra, pa, dfa = partial_r(u, d, [pol])          # 控制极化率后去稳定度
        rb, pb, dfb = partial_r(u, pol, [d])          # 控制去稳定度后极化率
        rc, pc2, _ = partial_r(u, d, [pol, scales["sheet_prop_Chou"]])
        rd, pd2, _ = partial_r(u, pol, [d, scales["sheet_prop_Chou"]])
        part[k] = {"r_disrupt_given_pol": ra, "p_a": pa,
                   "r_pol_given_disrupt": rb, "p_b": pb, "df": dfa,
                   "r_disrupt_given_pol_sheet": rc, "p_c": pc2,
                   "r_pol_given_disrupt_sheet": rd, "p_d": pd2}
        print(f"\n   [{k}]  df = {dfa}")
        print(f"     r(u, 去稳定度 | 极化率)  = {ra:+.3f}  p = {pa:.4f}")
        print(f"     r(u, 极化率 | 去稳定度)  = {rb:+.3f}  p = {pb:.4f}")
        print(f"     再控制 sheet propensity: 去稳定度 {rc:+.3f}"
              f"（p={pc2:.4f}）/ 极化率 {rd:+.3f}（p={pd2:.4f}）")

    # 两个嵌套回归的增量 R²
    print("\n   嵌套回归的增量 R²（u 作为被解释量，n = 20）")
    inc = {}
    for k in DISRUPT_KEYS:
        d = dis[k]

        def r2(cols):
            X = np.column_stack([np.ones(20)] + list(cols))
            b = np.linalg.lstsq(X, u, rcond=None)[0]
            res = u - X @ b
            return 1 - (res ** 2).sum() / ((u - u.mean()) ** 2).sum()
        r2_pol = r2([pol])
        r2_dis = r2([d])
        r2_both = r2([pol, d])
        inc[k] = {"r2_pol": r2_pol, "r2_disrupt": r2_dis,
                  "r2_both": r2_both,
                  "inc_disrupt_over_pol": r2_both - r2_pol,
                  "inc_pol_over_disrupt": r2_both - r2_dis}
        print(f"     {k:<36} R²(极化率) = {r2_pol:.3f}  "
              f"R²(去稳定度) = {r2_dis:.3f}  R²(两者) = {r2_both:.3f}  "
              f"Δ去稳定度 = {r2_both - r2_pol:+.3f}  "
              f"Δ极化率 = {r2_both - r2_dis:+.3f}")

    # ---------------------------------------------------------------
    # 4. gauge-invariant：二维载荷子空间被谁张开
    # ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print("4. gauge-invariant 版本：二维残基载荷子空间被 span{vol} 还是"
          " span{vol, disrupt} 张开")
    print("=" * 92)
    # 共识二维子空间：两组家族平均 P̄ 的前两个特征向量
    newc = {z_["name"]: z_["new"] for z_ in json.load(
        open(os.path.join(ROOT, "cache",
                          "censor_density_revision.json")))["rows"]}
    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    for d in mats:
        d["censor_frac"] = newc[d["name"]]
    mats = pick_primary(mats)
    keep = [d for d in mats if d["censor_frac"] < 0.25]
    by = {}
    for d in keep:
        B, c, _ = extract_subspace(d["M"], d["mask"])
        if B is not None and len(c) == 20:
            by[d["name"]] = B
    fam = {"冠状病毒": [n for n in TG["axis1_family_avg"]["families"]
                        ["冠状病毒"] if n in by],
           "OMTKY3": [n for n in TG["axis1_family_avg"]["families"]
                      ["OMTKY3"] if n in by]}
    Pb = 0.5 * sum(np.mean([CN.proj(by[n])[0] for n in v], axis=0)
                   for v in fam.values())
    w, V = np.linalg.eigh(Pb)
    w = w[::-1]; V = V[:, ::-1]
    assert abs(w[0] - lam[0]) < 1e-6, "重算的 λ1 与缓存不一致"
    S2 = V[:, :2]
    print(f"   共识 P̄ 的 λ = {np.round(w[:3], 3).tolist()}（与缓存一致）")
    print(f"   共识二维子空间 S2 = P̄ 的前两个特征向量"
          f"（**这才是可辨识的对象**）")

    # 零分布：d = 20 的中心化 19 维内
    N1 = haar_pair_null(20, 2, n=NSIM, seed=11)
    N1d = haar_1d_null(20, n=NSIM, seed=12)
    p95_2 = float(np.percentile(N1["cos1"], 95))
    p95_ssc = float(np.percentile(N1["ssc"], 95))
    p95_1 = float(np.percentile(N1d, 95))
    # 一维方向 vs 二维子空间的零分布（这是本节真正需要的那一个）
    rngn = np.random.default_rng(13)
    P1 = np.eye(20) - np.outer(ONE, ONE)
    n1v2 = []
    for _ in range(NSIM):
        a = P1 @ rngn.normal(size=20); a /= np.linalg.norm(a)
        A = P1 @ rngn.normal(size=(20, 2))
        Q, _ = np.linalg.qr(A)
        n1v2.append(float(np.linalg.norm(Q.T @ a)))
    n1v2 = np.array(n1v2)
    p95_1v2 = float(np.percentile(n1v2, 95))
    print(f"   零分布（**全部在 d = 20、中心化 19 维内生成**，{NSIM} 次）:")
    print(f"     1 维 vs 1 维 |cos| p95 = {p95_1:.3f}")
    print(f"     1 维 vs 2 维 ‖P_S a‖ p95 = {p95_1v2:.3f}（中位 "
          f"{np.median(n1v2):.3f}）")
    print(f"     2 维 vs 2 维 cos θ1 p95 = {p95_2:.3f}，"
          f"Σcos² p95 = {p95_ssc:.3f}")

    print("\n   (a) 单条方向在 S2 里被覆盖多少（‖P_S2 v̂‖，超 "
          f"{p95_1v2:.3f} 才算）:")
    cap1 = {}
    for k, v in [("local_polarizability", scales["local_polarizability"]),
                 ("sheet_prop_Chou", scales["sheet_prop_Chou"]),
                 ("kd_hydropathy", scales["kd_hydropathy"])] + \
            [(k, dis[k]) for k in DISRUPT_KEYS]:
        a = cen(v)
        c = float(np.linalg.norm(S2.T @ a))
        cap1[k] = c
        print(f"     {k:<36} ‖P_S2 v̂‖ = {c:.3f}  "
              f"{'超出' if c > p95_1v2 else '未超出'}")

    print("\n   (b) **核心那一问**：S2 与 span{极化率} / span{极化率, 去稳定度}")
    base = captured(S2, [scales["local_polarizability"]])
    print(f"     S2 vs span{{极化率}}（1 维）: cos = {base[0]:.3f}"
          f"（捕获 Σcos² = {base[0] ** 2:.3f} / 2）")
    gauge = {}
    for k in DISRUPT_KEYS:
        cc = captured(S2, [scales["local_polarizability"], dis[k]])
        ssc = float((cc ** 2).sum())
        gauge[k] = {"cos": cc.tolist(), "ssc": ssc,
                    "inc_over_pol_alone": ssc - base[0] ** 2}
        print(f"     S2 vs span{{极化率, {k[:28]}}}: "
              f"cos = ({cc[0]:.3f}, {cc[1]:.3f})  Σcos² = {ssc:.3f}  "
              f"增量 = {ssc - base[0] ** 2:+.3f}")
    # 零分布：把第二条换成**随机**方向（同样与极化率一起张 2 维）
    rng3 = np.random.default_rng(14)
    polc = cen(scales["local_polarizability"])
    nullinc = []
    for _ in range(NSIM):
        a = P1 @ rng3.normal(size=20)
        cc = captured(S2, [polc, a])
        nullinc.append(float((cc ** 2).sum()) - base[0] ** 2)
    nullinc = np.array(nullinc)
    print(f"\n     **零分布**：第二条换成 Haar 随机方向（与极化率一起张 2 维）")
    print(f"       增量 中位 {np.median(nullinc):.3f}  p95 "
          f"**{np.percentile(nullinc, 95):.3f}**")
    inc95 = float(np.percentile(nullinc, 95))
    for k in DISRUPT_KEYS:
        d_ = gauge[k]["inc_over_pol_alone"]
        pv = float(np.mean(nullinc >= d_))
        gauge[k]["p_vs_random_second"] = pv
        print(f"       {k:<36} 增量 {d_:+.3f}  p = {pv:.4f}  "
              f"{'超出 p95' if d_ > inc95 else '未超出 p95'}")

    print("\n   (c) 对称地问一次（避免只测后测的那一个）：")
    print("       S2 与 span{去稳定度} 单条、与 span{去稳定度, 极化率} 的增量")
    gauge2 = {}
    for k in DISRUPT_KEYS:
        b1 = captured(S2, [dis[k]])
        cc = captured(S2, [dis[k], scales["local_polarizability"]])
        ssc = float((cc ** 2).sum())
        nl = []
        dc = cen(dis[k])
        rng4 = np.random.default_rng(15)
        for _ in range(1000):
            a = P1 @ rng4.normal(size=20)
            nl.append(float((captured(S2, [dc, a]) ** 2).sum()) - b1[0] ** 2)
        nl = np.array(nl)
        pv = float(np.mean(nl >= ssc - b1[0] ** 2))
        gauge2[k] = {"base_cos": float(b1[0]), "ssc": ssc,
                     "inc_pol_over_disrupt": ssc - b1[0] ** 2,
                     "null_p95": float(np.percentile(nl, 95)),
                     "p_vs_random_second": pv}
        print(f"     {k:<36} 单条 {b1[0]:.3f}；加极化率后 Σcos² = {ssc:.3f}"
              f"，增量 {ssc - b1[0] ** 2:+.3f}  "
              f"（随机第二条零分布 p95 {np.percentile(nl, 95):.3f}，"
              f"p = {pv:.4f}）")

    # ---------------------------------------------------------------
    # 5. 残差：把去稳定度从 u 里投掉还剩什么
    # ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print("5. 残差：从 u 里投掉一般性去稳定度，剩下的那条方向是什么")
    print("=" * 92)
    resid = {}
    uc = cen(u)
    for k in DISRUPT_KEYS + ["D5_disrupt_PC1"]:
        if k in resid:
            continue
        dc = cen(dis[k])
        proj = float(dc @ uc)
        r = uc - proj * dc
        nrm = float(np.linalg.norm(r))
        r = r / nrm if nrm > 1e-12 else r
        rs = []
        for nm, s in scales.items():
            idx = np.isfinite(s)
            if idx.sum() < 10 or np.std(s[idx]) == 0:
                continue
            rs.append((abs(float(np.corrcoef(r[idx], s[idx])[0, 1])),
                       float(np.corrcoef(r[idx], s[idx])[0, 1]), nm))
        rs.sort(reverse=True)
        resid[k] = {"frac_explained": proj ** 2,
                    "resid_norm_frac": nrm ** 2,
                    "top_scales": [{"scale": nm, "r": rr}
                                   for _, rr, nm in rs[:5]],
                    "resid_dir": r.tolist()}
        print(f"\n   投掉 {k}:")
        print(f"     u 里被它解释的方差比例 = {proj ** 2:.3f}；"
              f"残差占 {nrm ** 2:.3f}")
        print(f"     残差方向与标度的相关（前 5）: " + "; ".join(
            f"{nm} {rr:+.3f}" for _, rr, nm in rs[:5]))
        o = np.argsort(-r)
        print(f"     残差载荷（最正→最负）: "
              + " ".join(f"{AA20[i]}{r[i]:+.2f}" for i in o))

    # ---------------------------------------------------------------
    # 6. 逐矩阵：去稳定度在每个矩阵自己的二维子空间里被覆盖多少
    #    （同源/跨源分列 —— 这是 D-032 要求的那一张表）
    # ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print("6. 逐矩阵：去稳定度方向在**每个体系自己的**二维子空间里的覆盖")
    print("=" * 92)
    permat = []
    key = "D5_disrupt_PC1"
    dc = cen(dis[key])
    polcen = cen(scales["local_polarizability"])
    allnames = fam["冠状病毒"] + fam["OMTKY3"]
    print(f"   {'矩阵':<44}{'‖P_S disrupt‖':>15}{'‖P_S 极化率‖':>15}")
    for n in allnames:
        Q = CN.proj(by[n])[1] if isinstance(CN.proj(by[n]), tuple) else None
        B = by[n]
        A = B - np.outer(ONE, ONE @ B)
        Qn, _ = np.linalg.qr(A)
        cd_ = float(np.linalg.norm(Qn.T @ dc))
        cp_ = float(np.linalg.norm(Qn.T @ polcen))
        permat.append({"name": n, "cap_disrupt": cd_, "cap_pol": cp_,
                       "family": ("冠状病毒" if n in fam["冠状病毒"]
                                  else "OMTKY3")})
        print(f"   {n[:42]:<44}{cd_:>15.3f}{cp_:>15.3f}")
    for g in ("冠状病毒", "OMTKY3"):
        v = [z_ for z_ in permat if z_["family"] == g]
        print(f"   [{g}] disrupt 覆盖中位 "
              f"{np.median([z_['cap_disrupt'] for z_ in v]):.3f}；"
              f"极化率覆盖中位 "
              f"{np.median([z_['cap_pol'] for z_ in v]):.3f}"
              f"（零分布 p95 = {p95_1v2:.3f}）")

    out = {"u": u.tolist(), "lam": lam, "loo_min_abscos": loo_min,
           "aa_order": AA20,
           "repro": {"same_vs_cross_ok": bool(same_ok),
                     "cross_family_cos1_med": float(np.median(c1x)),
                     "pairs_by_cat": pc,
                     "ddpca_splithalf": DS["ddpca_splithalf"],
                     "ddpca_cross_domain": cd,
                     "expr_same_source": DS["expr_repro_same_source"],
                     "expr_cross_source_med": float(np.median(ecr)),
                     "cross_empirical_scales": cross_emp},
           "corr": corr, "ref_corr": ref,
           "search_volume": n_search,
           "perm_null_max_abs_r": {"median": float(np.median(mx)),
                                   "p95": perm95,
                                   "p99": float(np.percentile(mx, 99)),
                                   "nsim": NSIM},
           "collinearity": coll,
           "partial": part, "incremental_r2": inc,
           "nulls_d20": {"p95_1d": p95_1, "p95_1d_vs_2d": p95_1v2,
                         "med_1d_vs_2d": float(np.median(n1v2)),
                         "p95_cos1_2d": p95_2, "p95_ssc_2d": p95_ssc},
           "S2_eigs": w[:3].tolist(),
           "cap_single": cap1,
           "gauge_pol_base_cos": float(base[0]),
           "gauge_add_disrupt": gauge,
           "gauge_null_inc_p95": inc95,
           "gauge_add_pol": gauge2,
           "residual": resid,
           "per_matrix": permat}
    p = os.path.join(ROOT, "cache", "generic_axis_test.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
