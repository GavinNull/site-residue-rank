#!/usr/bin/env python3
"""把 15 个「骨架 × ACE2 正交体」组合的秩2 交互项分解成**规范固定**的位点载荷。

### 为什么必须先固定规范（gauge）

秩2 分解只在共享的可逆 2x2 变换下唯一：

    U S V^T = (U S A) (A^{-1} V^T)^T ... 即 L V^T = (L B^T)(V B^{-T})^T

所以**单个** u1(site)、u2(site) 在固定规范之前没有任何意义，
跨组合比较位点载荷更是无意义的。本模块的做法：

1. 参考基：最大、截尾最少的体系 Wuhan-Hu-1 × bind_human_monomer
   （SARS-CoV-2 RBD × 人 ACE2，即 6M0J 那个复合物本身）。
2. 定向与定标：V 的列归一（单位范数），尺度全部放进 L = U·S；
   成分1 定向为「残基载荷与疏水性 π 负相关」，
   成分2 定向为「残基载荷与平面 sp² 指示变量负相关」。
3. 其余 14 个组合：在**共同残基**上最小二乘求 B（2x2），使 V ≈ V_ref·B，
   则 L V^T ≈ (L B^T) V_ref^T，故对齐后的位点载荷是 L' = L B^T。
   报告相对残差 ||V - V_ref B||_F / ||V||_F；残差过大的组合不可比，剔除。

### 自检（CLAUDE.md §10，用已知事实而非自洽性）

- AAindex 解析：Gly 的 Fauchere-Pliska π 必须为 0，Trp 必须最大（D-023 同类错）。
- 规范对齐：对参考体系自己做一遍对齐，B 必须 ≈ I，残差必须 ≈ 0。
- 合成检验：用已知的 L0、V0 造一个秩2 矩阵，随机施加 2x2 变换 A 后，
  本模块必须把位点载荷还原到 L0（相关 ≈ 1）。
- ALS 一律用 rank_test_power.als_guarded（带欠定行列保护），不用 als_complete。

数据：cache/dms/starr_animalACE2_final_variant_scores.csv
输出：cache/site_loadings.json
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coupling_structure import double_center
from rank_test_power import als_guarded
from scale_redundancy import parse_aaindex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMS = os.path.join(ROOT, "cache", "dms")
CSV_NAME = "starr_animalACE2_final_variant_scores.csv"
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
# 平面 sp² 侧链指示（docs/10 的定义：芳香 + 羧酸/酰胺的 sp² 基团）
PLANAR1 = set("FYWHDENQ")
LIGANDS = ["bind_deer", "bind_hamster", "bind_bat", "bind_cat",
           "bind_human_monomer"]
REF_KEY = ("Wuhan-Hu-1", "bind_human_monomer")
RANK = 2
MIN_OBS_PER_SITE = 12       # 一个位点至少要有这么多残基被测到
MIN_OBS_PER_RES = 30
ALS_RIDGE = 0.05
ALS_RESTARTS = 6
ALS_ITERS = 400
# 预先声明的可比性阈值：残差 >0.40 意味着该组合残基载荷有 >16% 的范数
# 无法在参考基里表达，其位点载荷不可比，剔除。
ALIGN_MAX = 0.40


# ---------------------------------------------------------------- 残基描述子

def residue_scales():
    """疏水性 π 与平面 sp² 指示。自检用已知事实。"""
    aai = parse_aaindex(os.path.join(ROOT, "cache", "aaindex1.txt"),
                        {"FAUJ830101"})
    pi = aai["FAUJ830101"]
    assert abs(pi["G"]) < 1e-9, f"Gly 的 π 应为 0，读到 {pi['G']}"
    assert max(pi, key=pi.get) == "W", f"π 最大应是 Trp，读到 {max(pi, key=pi.get)}"
    assert pi["I"] > pi["S"], "Ile 应比 Ser 疏水"
    hyd = np.array([pi[a] for a in AA20])
    pl = np.array([1.0 if a in PLANAR1 else 0.0 for a in AA20])
    return hyd, pl


# ---------------------------------------------------------------- 矩阵构造

def load_rows():
    return list(csv.DictReader(open(os.path.join(DMS, CSV_NAME))))


def build_matrix(rows, target, col):
    """返回 (sites, M, mask, censored_frac)。M 是 ΔΔG 约定：越大越不利。

    原始 bind_* 是 log10(Ka)，越大结合越强，故取负号得到 ΔΔG 方向的量
    （单位：log10 Ka；1 log10 Ka ≈ 1.36 kcal/mol，量级核对见 docs/14）。
    截尾：按 bind 的最小值处堆积检测并剔除（与 analysis/ligand_specific_rank.py 同）。
    """
    vals = []
    for r in rows:
        if r.get("target") != target:
            continue
        v = r.get(col, "")
        if v in ("", "NA"):
            continue
        if r["mutant"] not in AA20:
            continue
        vals.append((int(r["position"]), r["mutant"], float(v)))
    if not vals:
        return None
    arr = np.array([v for _, _, v in vals])
    lo = arr.min()
    n_cens = int((arr <= lo + 1e-6).sum())
    keep = [(s, a, v) for s, a, v in vals if v > lo + 1e-6]
    cens_frac = n_cens / len(vals)
    sites = sorted({s for s, _, _ in keep})
    si = {s: i for i, s in enumerate(sites)}
    M = np.full((len(sites), len(AA20)), np.nan)
    for s, a, v in keep:
        M[si[s], AA20.index(a)] = -v        # ΔΔG 方向
    mask = np.isfinite(M)
    # 观测过少的行/列剔除（否则其载荷由 ridge 决定，不是数据决定）
    keep_r = mask.sum(axis=1) >= MIN_OBS_PER_SITE
    keep_c = mask.sum(axis=0) >= MIN_OBS_PER_RES
    M = M[np.ix_(keep_r, keep_c)]
    mask = mask[np.ix_(keep_r, keep_c)]
    sites = [s for s, k in zip(sites, keep_r) if k]
    res = [a for a, k in zip(AA20, keep_c) if k]
    return sites, res, M, mask, cens_frac


def fit_rank2(M, mask, seed=7):
    """双向中心化 -> 带保护的 ALS 秩2 -> SVD 得到 (L, V, s)。L = U·S。"""
    Rm = double_center(np.where(mask, M, 0.0), mask)
    rng = np.random.default_rng(seed)
    best_tr, best_P = np.inf, None
    for _ in range(ALS_RESTARTS):
        P = als_guarded(np.where(mask, Rm, 0.0), mask, RANK, rng,
                        iters=ALS_ITERS, ridge=ALS_RIDGE)
        tr = float(np.mean((Rm[mask] - P[mask]) ** 2))
        if tr < best_tr:
            best_tr, best_P = tr, P
    U, s, Vt = np.linalg.svd(best_P, full_matrices=False)
    U, s, V = U[:, :RANK], s[:RANK], Vt[:RANK].T     # V 列单位范数
    L = U * s[None, :]
    coup_var = float(Rm[mask].var())
    expl = float(1 - np.mean((Rm[mask] - best_P[mask]) ** 2) / coup_var)
    return L, V, s, Rm, best_P, {"train_mse": best_tr, "coup_var": coup_var,
                                 "rank2_expl": expl}


def orient_reference(L, V, res, hyd, pl):
    """参考基的符号约定：成分1 与疏水性负相关，成分2 与平面 sp² 负相关。"""
    idx = [AA20.index(a) for a in res]
    h = hyd[idx]
    p = pl[idx]
    out = []
    for k, ref in ((0, h), (1, p)):
        r = float(np.corrcoef(V[:, k], ref)[0, 1])
        sgn = -1.0 if r > 0 else 1.0
        out.append((sgn, r * sgn))
    S = np.array([out[0][0], out[1][0]])
    return L * S[None, :], V * S[None, :], [o[1] for o in out]


def align(V, res, V_ref, res_ref):
    """在共同残基上求 B（2x2）使 V ≈ V_ref B；返回 (B, 相对残差, n_common)。"""
    common = [a for a in res if a in res_ref]
    i = [res.index(a) for a in common]
    j = [res_ref.index(a) for a in common]
    A = V_ref[j]            # n x 2
    Y = V[i]                # n x 2
    B, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = float(np.linalg.norm(Y - A @ B) / np.linalg.norm(Y))
    return B, resid, len(common)


# ---------------------------------------------------------------- 自检

def selftest_gauge():
    """已知事实自检：随机 2x2 变换必须被完全撤销。"""
    rng = np.random.default_rng(1234)
    n_s, n_r = 120, 20
    L0 = rng.normal(0, 1, (n_s, 2)) * np.array([3.0, 1.5])
    V0 = rng.normal(0, 1, (n_r, 2))
    V0 /= np.linalg.norm(V0, axis=0, keepdims=True)
    P = L0 @ V0.T
    # 任意可逆变换
    A = np.array([[0.8, -1.7], [2.1, 0.6]])
    L1, V1 = L0 @ A, V0 @ np.linalg.inv(A).T
    assert np.allclose(L1 @ V1.T, P, atol=1e-9), "构造错了"
    B, resid, _ = align(V1, AA20, V0, AA20)
    L_al = L1 @ B.T
    ok_resid = resid < 1e-8
    c1 = abs(np.corrcoef(L_al[:, 0], L0[:, 0])[0, 1])
    c2 = abs(np.corrcoef(L_al[:, 1], L0[:, 1])[0, 1])
    err = float(np.max(np.abs(L_al - L0)))
    print(f"  [自检1 规范还原] 对齐残差 {resid:.2e}  "
          f"r(u1)={c1:.6f} r(u2)={c2:.6f} 最大逐元误差 {err:.2e}")
    assert ok_resid and c1 > 0.9999 and c2 > 0.9999 and err < 1e-6, \
        "规范对齐没有还原已知的位点载荷 —— 工具不可用"
    # 不误伤：真的不同的 V（秩2 子空间不同）必须给出大残差
    V2 = rng.normal(0, 1, (n_r, 2))
    V2 /= np.linalg.norm(V2, axis=0, keepdims=True)
    _, resid_bad, _ = align(V2, AA20, V0, AA20)
    print(f"  [自检1b 不误伤] 无关残基基的对齐残差 {resid_bad:.3f}（应远大于 0）")
    assert resid_bad > 0.3, "对齐残差对真正不可比的基不敏感 —— 护栏无效"


def selftest_pipeline_recovery():
    """理想条件还原性：造秩2 矩阵 + 缺失 + 噪声，fit_rank2 必须还原 L0。"""
    rng = np.random.default_rng(99)
    n_s, n_r = 150, 20
    L0 = rng.normal(0, 1, (n_s, 2)) * np.array([3.0, 1.2])
    V0 = rng.normal(0, 1, (n_r, 2))
    # **因子必须列中心化**：否则 L0 V0^T 自身不是双向中心化的，
    # 双向中心化会改变它的列空间，还原性诊断就把这个人为的
    # 不一致读成「流程还原不出来」（第一版实测对齐残差 0.32，
    # 主角度 26.6 度，而 train_mse 只有 0.003 —— 典型的
    # 「被检查对象的错 vs 工具的错」混淆，CLAUDE.md §10 末条）。
    L0 -= L0.mean(axis=0, keepdims=True)
    V0 -= V0.mean(axis=0, keepdims=True)
    V0 /= np.linalg.norm(V0, axis=0, keepdims=True)
    a = rng.normal(0, 2, n_r)
    b = rng.normal(0, 2, n_s)
    M = b[:, None] + a[None, :] + L0 @ V0.T
    for sigma in (0.0, 0.3):
        Mn = M + rng.normal(0, sigma, M.shape)
        mask = rng.random(M.shape) < 0.9
        L, V, s, *_ = fit_rank2(np.where(mask, Mn, np.nan), mask)
        B, resid, _ = align(V, AA20, V0, AA20)
        L_al = L @ B.T
        c1 = abs(np.corrcoef(L_al[:, 0], L0[:, 0])[0, 1])
        c2 = abs(np.corrcoef(L_al[:, 1], L0[:, 1])[0, 1])
        print(f"  [自检2 还原 sigma={sigma}] 对齐残差 {resid:.4f} "
              f"r(u1)={c1:.4f} r(u2)={c2:.4f}")
        assert c1 > 0.95 and c2 > 0.85, "理想条件下都还原不出来 —— 流程不可用"
        if sigma == 0.0:
            assert resid < 0.05, \
                f"无噪声下对齐残差应 ~0，实测 {resid:.3f} —— 对齐实现有问题"


# ---------------------------------------------------------------- 主流程

def main():
    hyd, pl = residue_scales()
    print("=" * 78)
    print("自检（CLAUDE.md §10）")
    print("=" * 78)
    selftest_gauge()
    selftest_pipeline_recovery()

    rows = load_rows()
    targets = ["Wuhan-Hu-1", "Eta", "Omicron_BA2"]
    # 先做参考体系
    sites_r, res_r, M_r, mask_r, cf_r = build_matrix(rows, *REF_KEY)
    L_r, V_r, s_r, Rm_r, P_r, st_r = fit_rank2(M_r, mask_r)
    L_r, V_r, corr_ref = orient_reference(L_r, V_r, res_r, hyd, pl)
    print("\n" + "=" * 78)
    print(f"参考基：{REF_KEY[0]} x {REF_KEY[1]}")
    print("=" * 78)
    print(f"  位点 {len(sites_r)}  残基 {len(res_r)}  观测 {int(mask_r.sum())}  "
          f"截尾 {cf_r:.1%}")
    print(f"  奇异值 {s_r[0]:.2f} / {s_r[1]:.2f}   秩2 解释耦合方差 "
          f"{st_r['rank2_expl']:.1%}")
    print(f"  定向后 r(v1, 疏水 π) = {corr_ref[0]:+.3f}   "
          f"r(v2, 平面 sp²) = {corr_ref[1]:+.3f}")
    print(f"  耦合 SD = {np.sqrt(st_r['coup_var']):.3f} log10Ka "
          f"= {np.sqrt(st_r['coup_var']) * 1.36:.3f} kcal/mol")

    out = {"ref": {"target": REF_KEY[0], "ligand": REF_KEY[1],
                   "residues": res_r, "V": V_r.tolist(),
                   "corr_v1_hyd": corr_ref[0], "corr_v2_planar": corr_ref[1],
                   "singular": s_r.tolist()},
           "combos": []}

    print("\n" + "=" * 78)
    print("各组合的规范对齐")
    print("=" * 78)
    print(f"{'骨架':<14}{'配体':<22}{'位点':>5}{'观测':>7}{'截尾':>7}"
          f"{'s1':>7}{'s2':>7}{'对齐残差':>10}{'r(v1,π)':>9}")
    print("-" * 78)
    for t in targets:
        for lg in LIGANDS:
            b = build_matrix(rows, t, lg)
            if b is None:
                continue
            sites, res, M, mask, cf = b
            if mask.sum() < 500 or len(sites) < 50:
                print(f"{t:<14}{lg:<22} 跳过（观测 {int(mask.sum())}）")
                continue
            L, V, s, Rm, P, st = fit_rank2(M, mask)
            B, resid, ncom = align(V, res, V_r, res_r)
            L_al = L @ B.T
            idx = [AA20.index(a) for a in res]
            # 诊断：该组合**自身规范**下成分1 与疏水性的相关（对齐前）
            rv1 = float(np.corrcoef(V[:, 0], hyd[idx])[0, 1])
            print(f"{t:<14}{lg:<22}{len(sites):>5}{int(mask.sum()):>7}"
                  f"{cf:>7.1%}{s[0]:>7.2f}{s[1]:>7.2f}{resid:>10.3f}"
                  f"{rv1:>+9.3f}")
            out["combos"].append({
                "target": t, "ligand": lg, "sites": sites,
                "n_obs": int(mask.sum()), "cens_frac": cf,
                "singular": s.tolist(), "align_resid": resid,
                "n_common_res": ncom, "rank2_expl": st["rank2_expl"],
                "coup_sd": float(np.sqrt(st["coup_var"])),
                "u1": L_al[:, 0].tolist(), "u2": L_al[:, 1].tolist(),
                "comparable": bool(resid <= ALIGN_MAX),
                "n_obs_per_site": mask.sum(axis=1).tolist()})
    path = os.path.join(ROOT, "cache", "site_loadings.json")
    json.dump(out, open(path, "w"), indent=1)
    ncomp = sum(c["comparable"] for c in out["combos"])
    print(f"\n可比组合（对齐残差 <= {ALIGN_MAX}）: {ncomp}/{len(out['combos'])}")
    for c in out["combos"]:
        if not c["comparable"]:
            print(f"   剔除 {c['target']} x {c['ligand']}  残差 {c['align_resid']:.3f}"
                  f"  s=({c['singular'][0]:.1f},{c['singular'][1]:.1f})")
    print(f"写入 {os.path.relpath(path, ROOT)}  "
          f"（{len(out['combos'])} 个组合）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
