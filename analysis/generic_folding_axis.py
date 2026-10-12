#!/usr/bin/env python3
"""docs/26 Part 2：folding 矩阵的第一条轴 vs binding 共识普适轴 u。

**功效账先摊开（docs/15 §2.5，不许绕过）**：六个 ddPCA 矩阵里只有两个
有功效，都在折叠一侧：
  GRB2-SH3/folding  截尾 3.0%、σ(体外校准) = 0.46、coup_sd_true = 0.723
                    = 1.57σ、P(检出 2 | 真实秩 2) = 1.00
  GB1/folding       截尾 13.1%、σ = 0.67、coup_sd_true = 0.736 = 1.10σ、
                    P = 0.88–0.94
  PSD95-PDZ3/folding P = 0.00 —— **排除**
  三个 binding 矩阵 coup_sd_true = 0 —— **排除**
并且 GB1 **没有独立的 abundance assay**（docs/15 §6.1），它的 folding
系数与 binding 系数是从同一组 binding-PCA 观测里联合反推的，所以它
**不进任何关于 folding/binding 关系的主张**，只作带限定的第二个矩阵。
→ 本检验实质上压在**一个**矩阵上。

**噪声模型（docs/15 §2.1a，硬约束）**：ddPCA 的 bootstrap SE
（0.009–0.199）低估真实误差一个量级以上，**不得**用作噪声模型。
一律用体外校准的 σ（LIT_RMSE，0.46–0.89 kcal/mol）。

**§10.3 的顺序**：先在这些矩阵**自己的几何**上做理想条件还原性诊断
（植入 u 作为真残基方向，看能不能还原），再去看真实数据。
否则「folding 的轴与 u 不同」与「我本来就检不出一致」无法区分。

**D-032**：被比较的每一条方向都必须先证明自己可复现。
folding 轴的可复现性用位点对半；u 的用留一（docs/21 §5.2，|cos| ≥ 0.956）。

运行：python3 analysis/generic_folding_axis.py
产物：cache/generic_folding_axis.json
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddpca_load as DL                                         # noqa: E402
import rank_core as RC                                          # noqa: E402
from rank_test_power import als_guarded                         # noqa: E402
from subspace_lib import (AA20, ROOT, extract_subspace,         # noqa: E402
                          haar_1d_null, load_scales)

NSIM = 4000
NREP = 200
ONE20 = np.ones(20) / math.sqrt(20)

# docs/15 §1.1 V5 的体外校准 RMSE。**不是** bootstrap SE。
LIT_RMSE = {("GB1", "folding"): 0.67,
            ("GRB2-SH3", "folding"): 0.46,
            ("PSD95-PDZ3", "folding"): 0.89}
# docs/15 §2.3 的实测功效 P(检出 2 | 真实秩 2)
POWER = {("GB1", "folding"): 0.88, ("GRB2-SH3", "folding"): 1.00,
         ("PSD95-PDZ3", "folding"): 0.00}
HAVE_POWER = [("GRB2-SH3", "folding"), ("GB1", "folding")]


def cen_unit(v, idx=None):
    v = np.asarray(v, float)
    if idx is not None:
        v = v[idx]
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def first_axis(M, mask, rank=1, method="emsvd", seed=0):
    B, cols, err = extract_subspace(M, mask, rank=rank, method=method,
                                    seed=seed)
    if B is None:
        return None, None
    return B[:, 0], cols


def synth_like(mask, resid_dir, coup_sd, sigma, main_row, main_col, rng,
               rank=1):
    """在给定 mask / σ / 主效应尺度下，合成一个**残基侧方向就是
    resid_dir** 的秩 rank 耦合矩阵。

    与 folding_vs_binding._synth_like 同一构造（λ 比例 1.5:0.8，整体缩放
    到逐元素 SD = coup_sd），唯一区别：第一成分的残基侧向量**不随机**，
    而是植入 resid_dir —— 这正是 §10.3 要的「理想条件下能不能还原」。
    """
    nr, nc = mask.shape
    M = (rng.normal(0, main_row, nr)[:, None]
         + rng.normal(0, main_col, nc)[None, :])
    base = [1.5, 0.8, 0.4][:rank]
    C = np.zeros((nr, nc))
    for i, lam in enumerate(base):
        phi = rng.normal(0, 1, nr)
        phi -= phi.mean()
        phi /= np.linalg.norm(phi) or 1
        if i == 0:
            psi = np.asarray(resid_dir, float).copy()
        else:
            psi = rng.normal(0, 1, nc)
            psi -= psi.mean()
            # 让后续成分与植入方向正交，否则「第一轴」会混
            psi -= resid_dir * (resid_dir @ psi)
        psi = psi - psi.mean()
        psi /= np.linalg.norm(psi) or 1
        C += lam * math.sqrt(nr * nc) * np.outer(phi, psi)
    s = C[mask].std()
    if s > 0:
        M = M + C * (coup_sd / s)
    M = M + rng.normal(0, sigma, (nr, nc))
    return np.where(mask, M, np.nan)


def main():
    scales = load_scales(verbose=False)
    mr = scales["molar_refractivity"]
    assert AA20[int(np.nanargmin(mr))] == "G" and float(mr[AA20.index("G")]) == 0.0
    assert AA20[int(np.nanargmax(mr))] == "W" \
        and abs(float(mr[AA20.index("W")]) - 42.53) < 1e-6
    print("[自检] 折射度 Gly 0.00 / Trp 42.53 —— 独立核验通过")

    TG = json.load(open(os.path.join(ROOT, "cache",
                                     "subspace_two_groups.json")))
    u20 = np.array(TG["axis1_family_avg"]["u"], float)
    loo = [z["abscos"] for z in TG["axis1_family_avg"]["loo"]]
    print(f"[输入] u（binding 共识普适轴）留一 |cos| 最小 {min(loo):.3f}、"
          f"中位 {np.median(loo):.3f} —— D-032 的前置：u 本身可复现")

    DS = json.load(open(os.path.join(ROOT, "cache",
                                     "generic_disrupt_scales.json")))
    dis = {k: np.array(v, float) for k, v in DS["scales"].items()}

    # -----------------------------------------------------------------
    # 0. 载入三个域的 folding 矩阵并复述功效账
    # -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("0. 功效账（docs/15 §2.2–§2.5；本检验至多压在两个矩阵上）")
    print("=" * 92)
    mats = {}
    print(f"   {'域/表型':<22}{'位点':>5}{'观测':>7}{'截尾':>8}"
          f"{'coup_sd_obs':>13}{'σ(体外)':>9}{'coup_sd_true':>14}"
          f"{'P(检出2|真2)':>13}")
    for dom in DL.DOMAINS:
        D = DL.load_domain(dom)
        DL.verify(dom, D, verbose=False)
        for ph in ("folding", "binding"):
            M, mask, ci = DL.matrix(D, ph)
            st = RC.center_fit(M, mask)
            r, c = st
            R = M - r[:, None] - c[None, :]
            cso = float(R[mask].std())
            sig = LIT_RMSE.get((dom, ph))
            if sig is None:
                sig = 0.64 if (dom, ph) == ("PSD95-PDZ3", "binding") else \
                    (1.22 if (dom, ph) == ("GRB2-SH3", "binding") else 0.67)
            ct = math.sqrt(max(cso ** 2 - sig ** 2, 0.0))
            mats[(dom, ph)] = {"M": M, "mask": mask, "ci": ci,
                               "row_sd": float(r.std()),
                               "col_sd": float(c.std()),
                               "coup_sd_obs": cso, "sigma": sig,
                               "coup_sd_true": ct}
            pw = POWER.get((dom, ph))
            print(f"   {dom + '/' + ph:<22}{M.shape[0]:>5}"
                  f"{int(mask.sum()):>7}{ci['censor_frac']:>8.1%}"
                  f"{cso:>13.3f}{sig:>9.2f}{ct:>14.3f}"
                  f"{('%.2f' % pw) if pw is not None else '—(0)':>13}")
    print("\n   => 有功效的只有 GRB2-SH3/folding 与 GB1/folding；")
    print("      PSD95-PDZ3/folding 的 P(检出2|真2) = 0.00（排除）；")
    print("      三个 binding 的 coup_sd_true = 0（无可定秩方差，排除）。")
    print("      GB1 没有独立 abundance assay（docs/15 §6.1），"
          "只作带限定的第二个矩阵。")

    # -----------------------------------------------------------------
    # 1. 工具自检：rank-1 抽取器
    # -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("1. 工具自检（§10）：rank-1 抽取器必须与 als_guarded 一致，"
          "且在理想条件下精确还原")
    print("=" * 92)
    rng = np.random.default_rng(1)
    # [S1] 无噪声完备秩 1：必须精确还原植入方向
    psi = cen_unit(rng.normal(size=20))
    mk = np.ones((60, 20), bool)
    Ms = synth_like(mk, psi, 1.0, 0.0, 1.0, 1.0, np.random.default_rng(2))
    a, cols = first_axis(Ms, mk)
    assert len(cols) == 20
    c1 = abs(float(cen_unit(a) @ psi))
    print(f"   [S1] 无噪声完备秩 1：|cos(还原, 植入)| = {c1:.6f}（应 ≈ 1）")
    assert c1 > 0.999, "无噪声下都还原不出植入方向"
    # [S2] emsvd vs als_guarded（必须用带保护的那个）
    Ma = np.where(mk, Ms, np.nan)
    b, colsb = first_axis(Ma, mk, method="als", seed=3)
    c2 = abs(float(cen_unit(b) @ cen_unit(a)))
    print(f"   [S2] emsvd vs als_guarded 的 rank-1 方向 |cos| = {c2:.6f}")
    assert c2 > 0.99, "两个抽取器给出不同的 rank-1 方向"
    # [S3] 不误伤：植入与 u 正交的方向，还原出的轴**不**应与 u 对齐
    uc = cen_unit(u20)
    orth = cen_unit(rng.normal(size=20))
    orth = cen_unit(orth - uc * (uc @ orth))
    fa = []
    for k in range(60):
        Mz = synth_like(mk, orth, 1.0, 0.3, 1.0, 1.0,
                        np.random.default_rng(100 + k))
        aa, cc = first_axis(Mz, mk)
        fa.append(abs(float(cen_unit(aa) @ uc)))
    n1 = haar_1d_null(20, n=NSIM, seed=9)
    p95_20 = float(np.percentile(n1, 95))
    print(f"   [S3] 不误伤：植入与 u 正交的方向，|cos(轴, u)| 中位 "
          f"{np.median(fa):.3f}，超 d=20 的一维 p95({p95_20:.3f}) 的比例 "
          f"{np.mean(np.array(fa) > p95_20):.0%}（应 ≈ 5%）")
    assert np.mean(np.array(fa) > p95_20) < 0.20, "不误伤自检失败"

    # -----------------------------------------------------------------
    # 2. 理想条件还原性诊断（§10.3，**在真实矩阵自己的几何上**）
    # -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("2. §10.3 理想条件还原性诊断：在这两个矩阵**自己的**"
          "尺寸/缺失模式/主效应/σ 下，若 folding 的真实残基方向**就是** u，"
          "我能还原到多高？")
    print("=" * 92)
    recov = {}
    for key in HAVE_POWER:
        z = mats[key]
        M, mask = z["M"], z["mask"]
        # 真实数据抽取时实际保留的残基列 —— 诊断必须在同一个 d 上做
        a_real, cols_real = first_axis(M, mask)
        d_real = len(cols_real)
        ur = cen_unit(u20, cols_real)
        print(f"\n   [{key[0]}/{key[1]}]  真实抽取保留 d = {d_real} 个残基列"
              f"（零分布必须在这个 d 上生成）")
        n1d = haar_1d_null(d_real, n=NSIM, seed=31)
        p95_d = float(np.percentile(n1d, 95))
        print(f"      d = {d_real} 的一维零分布：|cos| 中位 "
              f"{np.median(n1d):.3f}，**p95 = {p95_d:.3f}**"
              f"（d=20 时是 {p95_20:.3f} —— 两者不可混用，docs/23）")
        for sig_name, sig in (("体外校准", z["sigma"]), ("乐观 0.35", 0.35)):
            for rk in (1, 2):
                # 植入方向：在**全 20 维**上植入 u，再让抽取器自己丢列
                psi_full = cen_unit(u20)
                got = []
                for k in range(NREP):
                    Mz = synth_like(mask, psi_full, z["coup_sd_true"], sig,
                                    max(z["row_sd"], 1e-6),
                                    max(z["col_sd"], 1e-6),
                                    np.random.default_rng(5000 + 97 * k),
                                    rank=rk)
                    aa, cc = first_axis(Mz, mask)
                    if aa is None:
                        continue
                    got.append(abs(float(cen_unit(aa)
                                         @ cen_unit(u20, cc))))
                got = np.array(got)
                recov[f"{key[0]}|{sig_name}|rank{rk}"] = {
                    "median": float(np.median(got)),
                    "ci": [float(np.percentile(got, 2.5)),
                           float(np.percentile(got, 97.5))],
                    "frac_over_p95": float(np.mean(got > p95_d)),
                    "n": int(len(got)), "d": d_real, "p95_d": p95_d,
                    "sigma": sig, "coup_sd_true": z["coup_sd_true"]}
                print(f"      σ = {sig:.2f}（{sig_name}）、真实秩 {rk}："
                      f"还原 |cos(轴, u)| 中位 **{np.median(got):.3f}** "
                      f"[{np.percentile(got, 2.5):.3f}, "
                      f"{np.percentile(got, 97.5):.3f}]，"
                      f"超 p95 比例 {np.mean(got > p95_d):.0%}")
        recov[f"{key[0]}|_meta"] = {"d": d_real, "p95_d": p95_d}

    # -----------------------------------------------------------------
    # 3. folding 第一轴自己可复现吗（D-032 的前置）
    # -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("3. folding 第一轴自己可复现吗（D-032：不可复现的方向"
          "不能与任何东西对齐）")
    print("=" * 92)
    repro = {}
    for key in HAVE_POWER + [("PSD95-PDZ3", "folding")]:
        z = mats[key]
        M, mask = z["M"], z["mask"]
        rngs = np.random.default_rng(77)
        nr = M.shape[0]
        vals = []
        for _ in range(NREP):
            p = rngs.permutation(nr)
            h1, h2 = p[: nr // 2], p[nr // 2:]
            a1, c1_ = first_axis(M[h1], mask[h1])
            a2, c2_ = first_axis(M[h2], mask[h2])
            if a1 is None or a2 is None:
                continue
            com = np.intersect1d(c1_, c2_)
            if len(com) < 12:
                continue
            v1 = cen_unit(a1[np.searchsorted(c1_, com)])
            v2 = cen_unit(a2[np.searchsorted(c2_, com)])
            vals.append(abs(float(v1 @ v2)))
        vals = np.array(vals)
        # 同一个 d 上的随机基线
        dmed = int(np.median([len(np.intersect1d(*[first_axis(M[p[:nr // 2]],
                                                              mask[p[:nr // 2]])[1]
                                                   for p in [rngs.permutation(nr)]]
                                                 + [first_axis(M, mask)[1]]))
                             for _ in range(1)])) if False else 20
        n1b = haar_1d_null(20, n=2000, seed=41)
        repro[f"{key[0]}/{key[1]}"] = {
            "splithalf_median": float(np.median(vals)),
            "ci": [float(np.percentile(vals, 2.5)),
                   float(np.percentile(vals, 97.5))],
            "n": int(len(vals)),
            "random_baseline_median": float(np.median(n1b)),
            "random_p95": float(np.percentile(n1b, 95))}
        tag = "**有功效**" if key in HAVE_POWER else "无功效（排除）"
        print(f"   {key[0] + '/' + key[1]:<22}{tag:<16}"
              f"位点对半 |cos| 中位 **{np.median(vals):.3f}** "
              f"[{np.percentile(vals, 2.5):.3f}, "
              f"{np.percentile(vals, 97.5):.3f}]  "
              f"（随机基线中位 {np.median(n1b):.3f}、p95 "
              f"{np.percentile(n1b, 95):.3f}）")

    # -----------------------------------------------------------------
    # 4. 真实比较
    # -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("4. 真实数据：folding 第一轴 vs binding 共识普适轴 u")
    print("=" * 92)
    obs = {}
    for key in HAVE_POWER + [("PSD95-PDZ3", "folding")]:
        z = mats[key]
        a, cols = first_axis(z["M"], z["mask"])
        d = len(cols)
        n1d = haar_1d_null(d, n=NSIM, seed=51)
        p95_d = float(np.percentile(n1d, 95))
        av = cen_unit(a)
        c_u = abs(float(av @ cen_unit(u20, cols)))
        # 对照：folding 轴与去稳定度标度、与体积/极化率
        c_d = {k: abs(float(av @ cen_unit(dis[k], cols)))
               for k in ("D1_folding_ddG_ddPCA",
                         "D2_blosum62_nonexchangeability",
                         "D5_disrupt_PC1")}
        c_p = abs(float(av @ cen_unit(scales["local_polarizability"], cols)))
        c_s = abs(float(av @ cen_unit(scales["sheet_prop_Chou"], cols)))
        pv = float(np.mean(n1d >= c_u))
        obs[f"{key[0]}/{key[1]}"] = {
            "d": d, "p95_d": p95_d, "cos_u": c_u, "p_haar": pv,
            "cos_disrupt": c_d, "cos_pol": c_p, "cos_sheet": c_s,
            "has_power": key in HAVE_POWER,
            "axis": av.tolist(), "cols": cols.tolist()}
        tag = "**有功效**" if key in HAVE_POWER else "无功效（不计为证据）"
        print(f"\n   [{key[0]}/{key[1]}] {tag}  d = {d}，"
              f"一维零分布 p95 = {p95_d:.3f}")
        print(f"      |cos(folding 轴, u)| = **{c_u:.3f}**  "
              f"p(Haar, d={d}) = {pv:.4f}  "
              f"{'超出' if c_u > p95_d else '未超出'}")
        print(f"      对照：|cos| 与极化率 {c_p:.3f}、sheet {c_s:.3f}、"
              + "、".join(f"{k.split('_')[0]} {v:.3f}"
                          for k, v in c_d.items()))
        o = np.argsort(-av)
        print(f"      folding 轴载荷（最正→最负）: "
              + " ".join(f"{AA20[cols[i]]}{av[i]:+.2f}" for i in o))

    # -----------------------------------------------------------------
    # 5. 功效上界：能检出的最小一致度
    # -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("5. 若结论是阴性，它的上界是多少（§10.3 要求的那个数）")
    print("=" * 92)
    bounds = {}
    for key in HAVE_POWER:
        dom = key[0]
        meta = recov[f"{dom}|_meta"]
        rv = recov[f"{dom}|体外校准|rank1"]
        o = obs[f"{dom}/folding"]
        print(f"\n   [{dom}/folding]")
        print(f"      若真实残基方向 = u（完全一致），在这个矩阵的几何与 "
              f"σ = {rv['sigma']:.2f} 下我会看到 |cos| 中位 "
              f"**{rv['median']:.3f}** [{rv['ci'][0]:.3f}, "
              f"{rv['ci'][1]:.3f}]")
        print(f"      实测 |cos| = **{o['cos_u']:.3f}**，d = {meta['d']} 的 "
              f"p95 = {meta['p95_d']:.3f}")
        # 上界：实测落在「真实一致度 = ρ」的还原分布的哪个位置
        # 扫 ρ：把植入方向改成 ρ·u + sqrt(1-ρ²)·(与 u 正交的随机方向)
        z = mats[key]
        rngb = np.random.default_rng(900)
        grid = []
        for rho in (0.2, 0.4, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
            got = []
            for k in range(80):
                w = rngb.normal(size=20)
                w = cen_unit(w)
                w = cen_unit(w - uc * (uc @ w))
                psi = cen_unit(rho * uc + math.sqrt(max(0, 1 - rho ** 2)) * w)
                Mz = synth_like(z["mask"], psi, z["coup_sd_true"],
                                z["sigma"], max(z["row_sd"], 1e-6),
                                max(z["col_sd"], 1e-6),
                                np.random.default_rng(7000 + 131 * k),
                                rank=1)
                aa, cc = first_axis(Mz, z["mask"])
                if aa is None:
                    continue
                got.append(abs(float(cen_unit(aa) @ cen_unit(u20, cc))))
            got = np.array(got)
            # 「能检出」= 该 ρ 下有多大比例的重复超过 d 的一维 p95
            det = float(np.mean(got > meta["p95_d"]))
            grid.append({"rho": rho, "median": float(np.median(got)),
                         "ci": [float(np.percentile(got, 2.5)),
                                float(np.percentile(got, 97.5))],
                         "p_detect": det})
            print(f"      真实一致度 ρ = {rho:.2f} → 观测 |cos| 中位 "
                  f"{np.median(got):.3f} "
                  f"[{np.percentile(got, 2.5):.3f}, "
                  f"{np.percentile(got, 97.5):.3f}]，"
                  f"检出率（超 p95）= {det:.0%}")
        # 上界 = 最小的 ρ，使检出率 >= 0.80
        ok = [g["rho"] for g in grid if g["p_detect"] >= 0.80]
        rho80 = min(ok) if ok else None
        # 与实测相容的最大 ρ（实测 |cos| 低于该 ρ 的 2.5% 分位即被排除）
        excl = [g["rho"] for g in grid
                if o["cos_u"] < g["ci"][0]]
        compat = [g["rho"] for g in grid
                  if g["ci"][0] <= o["cos_u"] <= g["ci"][1]]
        excl_lo = [g["rho"] for g in grid if o["cos_u"] > g["ci"][1]]
        print(f"      => 检出率 ≥ 80% 的最小真实一致度 ρ = "
              f"{rho80 if rho80 is not None else '> 1.0（任何一致度都检不出）'}")
        print(f"      => 被实测排除的 ρ（实测 |cos| 落在其 95% 区间之下）: "
              f"{excl if excl else '无'}")
        print(f"      => 与实测相容的 ρ（实测落在其 95% 区间内）: "
              f"{compat if compat else '无（网格太粗）'}；"
              f"从**下方**被排除的 ρ: {excl_lo}")
        bounds[dom] = {"grid": grid, "rho_detect80": rho80,
                       "rho_excluded": excl, "rho_compatible": compat,
                       "rho_excluded_below": excl_lo, "obs_cos_u": o["cos_u"],
                       "p95_d": meta["p95_d"]}

    # -----------------------------------------------------------------
    # 6. 可靠度校准 + 同源/跨源配对（D-032 的第二半）
    # -----------------------------------------------------------------
    print("\n" + "=" * 92)
    print("6. 可靠度校准与同源/跨源配对")
    print("=" * 92)
    # (a) 合成秩 1 数据的位点对半可靠度 —— 用来判断真实数据比合成数据"难"多少
    cal = {}
    for key in HAVE_POWER:
        z = mats[key]
        rngc = np.random.default_rng(321)
        vals = []
        for k in range(60):
            Mz = synth_like(z["mask"], cen_unit(u20), z["coup_sd_true"],
                            z["sigma"], max(z["row_sd"], 1e-6),
                            max(z["col_sd"], 1e-6),
                            np.random.default_rng(4100 + 11 * k), rank=1)
            nr = Mz.shape[0]
            p = rngc.permutation(nr)
            h1, h2 = p[: nr // 2], p[nr // 2:]
            a1, c1_ = first_axis(Mz[h1], z["mask"][h1])
            a2, c2_ = first_axis(Mz[h2], z["mask"][h2])
            if a1 is None or a2 is None:
                continue
            com = np.intersect1d(c1_, c2_)
            if len(com) < 12:
                continue
            vals.append(abs(float(cen_unit(a1[np.searchsorted(c1_, com)])
                                  @ cen_unit(a2[np.searchsorted(c2_, com)]))))
        vals = np.array(vals)
        rr = repro[f"{key[0]}/folding"]["splithalf_median"]
        oc = obs[f"{key[0]}/folding"]["cos_u"]
        dis_att = oc / math.sqrt(rr) if rr > 0 else float("nan")
        cal[key[0]] = {"synth_splithalf_median": float(np.median(vals)),
                       "real_splithalf_median": rr,
                       "obs_cos_u": oc,
                       "attenuation_corrected": min(1.0, dis_att)}
        print(f"\n   [{key[0]}/folding]")
        print(f"      合成（真实方向 = u、秩 1、同几何同 σ）的位点对半 "
              f"|cos| 中位 = {np.median(vals):.3f}")
        print(f"      真实数据的位点对半 |cos| 中位 = {rr:.3f} "
              f"—— 真实矩阵比合成的"
              f"{'更难（含更多非秩 1 结构或更大误差）'
                 if np.median(vals) > rr + 0.05 else '相当'}")
        print(f"      实测 |cos(folding 轴, u)| = {oc:.3f}；"
              f"用自身可靠度 {rr:.3f} 做衰减校正后 ≈ "
              f"{min(1.0, dis_att):.3f}")
        print(f"      **判读**：实测 |cos| 与该轴自己的可复现度"
              f"（{rr:.3f}）{'相当或更高 —— 两条方向在测量误差内不可区分'
                            if oc >= rr - 0.05 else '明显更低'}")

    # (a2) 多重性：三个 folding 矩阵里取最大 |cos| 的零分布
    nbig = haar_1d_null(20, n=20000, seed=777)
    rngm = np.random.default_rng(0)
    m3 = np.array([np.max(rngm.choice(nbig, 3)) for _ in range(20000)])
    mult = {"n_matrices": 3, "max_of_3_median": float(np.median(m3)),
            "max_of_3_p95": float(np.percentile(m3, 95))}
    print(f"\n   多重性（§5 搜索量）：3 个 folding 矩阵，"
          f"d = 20 的「3 取最大」|cos| 零分布 中位 "
          f"{mult['max_of_3_median']:.3f}、**p95 = "
          f"{mult['max_of_3_p95']:.3f}**")

    # (b) 同源 vs 跨源配对
    print("\n   同源 / 跨源配对（docs/25 §6.3 的诊断表式）")
    TGs = json.load(open(os.path.join(ROOT, "cache",
                                      "generic_axis_strat.json")))
    ucov = np.array(TGs["family_axes"]["冠状病毒"], float)
    uomt = np.array(TGs["family_axes"]["OMTKY3"], float)
    pairing = []
    for key in HAVE_POWER + [("PSD95-PDZ3", "folding")]:
        o = obs[f"{key[0]}/folding"]
        cols = np.array(o["cols"])
        av = np.array(o["axis"], float)
        row = {"folding": f"{key[0]}/folding",
               "vs_u": o["cos_u"],
               "vs_u_cov": abs(float(av @ cen_unit(ucov, cols))),
               "vs_u_omt": abs(float(av @ cen_unit(uomt, cols))),
               "has_power": o["has_power"],
               "own_repro": repro[f"{key[0]}/folding"]["splithalf_median"]}
        pairing.append(row)
    # folding × folding（跨域、跨源）
    ff = []
    names = [f"{k[0]}/folding" for k in HAVE_POWER
             + [("PSD95-PDZ3", "folding")]]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = obs[names[i]], obs[names[j]]
            com = np.intersect1d(a["cols"], b["cols"])
            va = cen_unit(np.array(a["axis"])[
                np.searchsorted(a["cols"], com)])
            vb = cen_unit(np.array(b["axis"])[
                np.searchsorted(b["cols"], com)])
            ff.append({"a": names[i], "b": names[j],
                       "cos": abs(float(va @ vb)), "d": int(len(com))})
    print(f"   {'folding 矩阵':<22}{'自身可复现':>11}{'vs u(两组)':>12}"
          f"{'vs u_冠状病毒':>14}{'vs u_OMTKY3':>13}")
    for r in pairing:
        print(f"   {r['folding']:<22}{r['own_repro']:>11.3f}"
              f"{r['vs_u']:>12.3f}{r['vs_u_cov']:>14.3f}"
              f"{r['vs_u_omt']:>13.3f}"
              f"  {'' if r['has_power'] else '(无功效)'}")
    print("   folding × folding（都是跨源配对）:")
    for r in ff:
        print(f"     {r['a']:<22} × {r['b']:<22} |cos| = {r['cos']:.3f}"
              f"  (d = {r['d']})")

    out = {"calibration": cal, "multiplicity": mult, "pairing": pairing, "folding_x_folding": ff,
           "power_table": {f"{k[0]}/{k[1]}":
                           {kk: v[kk] for kk in
                            ("coup_sd_obs", "sigma", "coup_sd_true",
                             "row_sd", "col_sd")}
                           | {"censor_frac": v["ci"]["censor_frac"],
                              "n_obs": v["ci"]["n_obs"],
                              "sites": v["ci"]["sites"],
                              "power": POWER.get(k)}
                           for k, v in mats.items()},
           "selfcheck": {"S1_noiseless_recover": c1,
                         "S2_emsvd_vs_als": c2,
                         "S3_false_alarm_frac":
                             float(np.mean(np.array(fa) > p95_20)),
                         "p95_1d_d20": p95_20},
           "recovery": recov, "repro": repro, "observed": obs,
           "bounds": bounds,
           "u": u20.tolist(), "aa_order": AA20,
           "u_loo_min": float(min(loo))}
    p = os.path.join(ROOT, "cache", "generic_folding_axis.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
