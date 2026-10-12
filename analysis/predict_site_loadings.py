#!/usr/bin/env python3
"""位点载荷 u1 / u2 能不能只从口袋结构算出来？

输入
  cache/site_loadings.json    规范固定后的位点载荷（analysis/site_loadings.py）
  cache/site_desc_6m0j.json   6M0J 结构描述子（analysis/pocket_descriptors_6m0j.py）

纪律（CLAUDE.md §5 / §10；docs/16 §3 的预注册预测）
  - 先报**重复可靠度**（15 个组合给同一批位点的重复估计），它是 R² 的天花板。
    没有它，R²=0.3 无法解释。
  - 留出**位点**，不是观测。两种 CV：LOSO 与空间分组（相邻位点共享环境，
    LOSO 会乐观）。
  - 所有 R² 带 bootstrap 区间（2000 次，按位点重抽）。
  - WCN 基线与埋藏度基线每次都报。
  - 记录搜索了多少个候选组合，并在**同一搜索规模**下做置换天花板。
  - 预注册的 P1（场梯度 vs 场强）与 P2（自由体积各向异性 vs 大小）
    分开报告，不与搜索出的模型混在一起。
  - 流程自检：合成「已知强度的信号」必须被按正确强度还原；
    合成「与结构无关」必须报无信号（两个方向都测，§10.2）。
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NBOOT = 2000
SEED = 20261011
RIDGE = 1.0            # 标准化特征上的岭参数；在自检里校准
MAXK = 3               # 搜索的最大特征数

MAG = ["wcn", "wcn_cb", "burial8", "burial_own8", "d_partner", "n_partner12",
       "free_frac", "free_frac_hemi", "shell1", "shell4",
       "logE2_D", "logE2_F"]
ANISO = ["flatness", "elongation", "sphericity", "n_lobes",
         "loggradE2_D", "loggradE2_F"]
BACKBONE = ["cos_iface", "cos_phi", "sin_phi", "cos_psi", "sin_psi"]
ROT = ["lognall_LEU", "lognall_PHE", "lognall_VAL", "planar_minus_branched"]
POOL = MAG + ANISO + BACKBONE + ROT


# --------------------------------------------------------------- 数据装载

def load_targets():
    d = json.load(open(os.path.join(ROOT, "cache", "site_loadings.json")))
    combos = [c for c in d["combos"] if c["comparable"]]
    return d, combos


def consensus(combos, keys=("u1", "u2"), subset=None):
    """每个组合先 z 标准化（尺度随配体动态范围变，不是位点性质），再取平均。"""
    use = [c for c in combos if subset is None or subset(c)]
    sets = [set(c["sites"]) for c in use]
    common = sorted(set.intersection(*sets))
    out = {}
    for k in keys:
        Z = []
        for c in use:
            idx = {s: i for i, s in enumerate(c["sites"])}
            v = np.array([c[k][idx[s]] for s in common])
            Z.append((v - v.mean()) / v.std())
        Z = np.array(Z)
        out[k] = {"sites": common, "Z": Z, "mean": Z.mean(axis=0)}
    return out, [f"{c['target']}|{c['ligand']}" for c in use]


def load_descriptors():
    d = json.load(open(os.path.join(ROOT, "cache", "site_desc_6m0j.json")))
    rows = {}
    for s, v in d["sites"].items():
        r = dict(v)
        for tag in ("E2_D", "E2_F", "gradE2_D", "gradE2_F"):
            r["log" + tag.replace("E2", "E2") if False else "log" + tag] = \
                np.log10(max(v[tag], 1e-12))
        r["logE2_D"] = np.log10(max(v["E2_D"], 1e-12))
        r["logE2_F"] = np.log10(max(v["E2_F"], 1e-12))
        r["loggradE2_D"] = np.log10(max(v["gradE2_D"], 1e-12))
        r["loggradE2_F"] = np.log10(max(v["gradE2_F"], 1e-12))
        for p in ("VAL", "LEU", "PHE", "TRP"):
            r[f"lognall_{p}"] = np.log1p(v[f"nall_{p}"])
        r["planar_minus_branched"] = (np.log1p(v["nall_PHE"])
                                      - np.log1p(v["nall_VAL"]))
        if v.get("phi") is None or v.get("psi") is None:
            r["cos_phi"] = r["sin_phi"] = r["cos_psi"] = r["sin_psi"] = np.nan
        else:
            r["cos_phi"] = np.cos(np.radians(v["phi"]))
            r["sin_phi"] = np.sin(np.radians(v["phi"]))
            r["cos_psi"] = np.cos(np.radians(v["psi"]))
            r["sin_psi"] = np.sin(np.radians(v["psi"]))
        rows[int(s)] = r
    return rows, d


def design(sites, desc, feats):
    X = np.array([[desc[s][f] for f in feats] for s in sites], dtype=float)
    return X


# ------------------------------------------------------------------- 模型

def ridge_fit(X, y, lam):
    Xm, Xs = X.mean(axis=0), X.std(axis=0)
    Xs = np.where(Xs > 0, Xs, 1.0)
    Z = (X - Xm) / Xs
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    b = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return (Xm, Xs, b, y.mean())


def ridge_pred(mdl, X):
    Xm, Xs, b, y0 = mdl
    return ((X - Xm) / Xs) @ b + y0


def cv_predict(X, y, groups, lam=RIDGE):
    """分组 CV 的样本外预测。groups 每个位点一个标签；LOSO 即每点一组。"""
    pred = np.zeros_like(y)
    for g in np.unique(groups):
        te = groups == g
        mdl = ridge_fit(X[~te], y[~te], lam)
        pred[te] = ridge_pred(mdl, X[te])
    return pred


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def boot_r2(y, p, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty(nboot)
    for b in range(nboot):
        i = rng.integers(0, n, n)
        yy, pp = y[i], p[i]
        out[b] = 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def boot_dr2(y, p1, p2, nboot=NBOOT, seed=SEED + 1):
    """配对 bootstrap：R²(p1) - R²(p2)。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty(nboot)
    for b in range(nboot):
        i = rng.integers(0, n, n)
        yy = y[i]
        sst = np.sum((yy - yy.mean()) ** 2)
        out[b] = (np.sum((yy - p2[i]) ** 2) - np.sum((yy - p1[i]) ** 2)) / sst
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def spatial_groups(sites, desc, k=10, seed=5):
    """按 CA 坐标（用 burial/壳层无关的纯几何）做 k-means 分组。

    相邻位点共享环境，LOSO 会把邻居的信息漏进去 —— 空间分组是诚实版本。
    """
    xyz = np.array([desc[s]["ca_xyz"] for s in sites])
    rng = np.random.default_rng(seed)
    C = xyz[rng.choice(len(xyz), k, replace=False)]
    lab = np.zeros(len(xyz), dtype=int)
    for _ in range(100):
        d = np.linalg.norm(xyz[:, None, :] - C[None], axis=2)
        lab = d.argmin(axis=1)
        for j in range(k):
            if (lab == j).any():
                C[j] = xyz[lab == j].mean(axis=0)
    return lab


# ------------------------------------------------------------------ 自检

def selftest_pipeline(X, groups, feats, seed=11):
    """两个方向都测（CLAUDE.md §10.2）。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    j = feats.index("wcn")
    z = (X[:, j] - X[:, j].mean()) / X[:, j].std()
    print("  [自检 流程还原]")
    for beta in (0.6, 0.3):
        # y = beta*z + sqrt(1-beta^2)*noise  => 真 R² = beta²
        y = beta * z + np.sqrt(1 - beta ** 2) * rng.normal(0, 1, n)
        p = cv_predict(X[:, [j]], y, groups)
        lo, hi = boot_r2(y, p, 500, seed)
        print(f"    信号 R²_true={beta ** 2:.2f} -> 样本外 R²={r2(y, p):.3f} "
              f"[{lo:.3f},{hi:.3f}]  (区间应覆盖 {beta ** 2:.2f})")
        assert lo <= beta ** 2 <= hi + 0.08, \
            "已知强度的信号没有被按正确强度还原 —— 量级不对"
    # 负方向：与结构无关
    r2s = []
    for _ in range(40):
        y = rng.normal(0, 1, n)
        p = cv_predict(X, y, groups)
        r2s.append(r2(y, p))
    print(f"    与结构无关的 y（全 {X.shape[1]} 特征）：样本外 R² "
          f"中位 {np.median(r2s):+.3f}，95 分位 {np.quantile(r2s, 0.95):+.3f}"
          f"（应 <= 0 附近，不得系统为正）")
    assert np.median(r2s) < 0.02, "无信号时流程也报正 R² —— 有泄露"


# -------------------------------------------------------------- 搜索与置换

def search_best(X, y, groups, pool_idx, maxk=MAXK):
    """穷举 <=maxk 个特征的子集，按样本外 R² 选最优。返回 (best_r2, feats, n)."""
    best = (-np.inf, None)
    n = 0
    for k in range(1, maxk + 1):
        for comb in itertools.combinations(pool_idx, k):
            p = cv_predict(X[:, list(comb)], y, groups)
            s = r2(y, p)
            n += 1
            if s > best[0]:
                best = (s, comb)
    return best[0], best[1], n


def nested_search_cv(X, y, groups, pool_idx, maxk=MAXK, inner_k=5, seed=7):
    """嵌套 CV：**特征选择放进每个外折的训练集**。

    为什么必须这样：第 5 节的「搜索最优 R²」是在同一个 CV 上选出来的，
    点估计必然乐观（置换天花板只校正零假设下的膨胀，不校正点估计的偏）。
    嵌套版给出可报告的无偏样本外 R²。
    """
    rng = np.random.default_rng(seed)
    pred = np.zeros_like(y)
    chosen = []
    for g in np.unique(groups):
        te = groups == g
        Xtr, ytr = X[~te], y[~te]
        inner = rng.integers(0, inner_k, Xtr.shape[0])
        best = (-np.inf, None)
        for k in range(1, maxk + 1):
            for comb in itertools.combinations(pool_idx, k):
                p = cv_predict(Xtr[:, list(comb)], ytr, inner)
                sc = r2(ytr, p)
                if sc > best[0]:
                    best = (sc, comb)
        mdl = ridge_fit(Xtr[:, list(best[1])], ytr, RIDGE)
        pred[te] = ridge_pred(mdl, X[:, list(best[1])][te])
        chosen.append(best[1])
    return pred, chosen


def permutation_ceiling(X, y, groups, pool_idx, nperm, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nperm):
        yp = rng.permutation(y)
        s, _, _ = search_best(X, yp, groups, pool_idx)
        out.append(s)
    return np.array(out)


# --------------------------------------------------------- 量级校准 (§8)

KCAL_PER_LOG10KA = 1.36      # 1 log10 Ka ~ 1.36 kcal/mol (RT ln10, 298 K)


def magnitude_check(dload, combos, desc, sites, X, Y, groups, pool_idx,
                    nested):
    """把 R² 换算成参考体系上的耦合 RMSE（kcal/mol），与两个参考值对照：

      上界（零预测器）= 耦合本身的 SD
      下界（oracle 秩2）= 用**数据拟合出的** u 的秩2 残差
    结构-only 代理量必须落在两者之间，且必须明显优于上界，否则没有用。
    """
    from site_loadings import (REF_KEY, build_matrix, fit_rank2, load_rows,
                               orient_reference, residue_scales, align, AA20)
    hyd, pl = residue_scales()
    rows = load_rows()
    sites_r, res_r, M_r, mask_r, _ = build_matrix(rows, *REF_KEY)
    L_r, V_r, s_r, Rm_r, P_r, st_r = fit_rank2(M_r, mask_r)
    L_r, V_r, _ = orient_reference(L_r, V_r, res_r, hyd, pl)
    # 只保留同时有结构描述子与载荷的位点
    keep = [i for i, s in enumerate(sites_r) if s in set(sites)]
    sub = [sites_r[i] for i in keep]
    Rm = Rm_r[keep]
    mk = mask_r[keep]
    Lk = L_r[keep]
    # 结构预测的 û（嵌套 CV 的样本外预测，z 单位 -> 该体系的载荷尺度）
    uhat = {}
    for key in ("u1", "u2"):
        y = Y[key]
        p, _ = nested_search_cv(X, y, groups, pool_idx)
        pos = {s: i for i, s in enumerate(sites)}
        z = np.array([p[pos[s]] for s in sub])
        col = 0 if key == "u1" else 1
        uhat[key] = Lk[:, col].mean() + Lk[:, col].std() * z
    f = KCAL_PER_LOG10KA
    sd0 = float(Rm[mk].std()) * f
    P_or = np.outer(Lk[:, 0], V_r[:, 0]) + np.outer(Lk[:, 1], V_r[:, 1])
    P_st = np.outer(uhat["u1"], V_r[:, 0]) + np.outer(uhat["u2"], V_r[:, 1])
    P_c1 = np.outer(uhat["u1"], V_r[:, 0])
    P_or1 = np.outer(Lk[:, 0], V_r[:, 0])

    def rmse(P):
        return float(np.sqrt(np.mean((Rm[mk] - P[mk]) ** 2))) * f
    out = {"coupling_sd_kcal": sd0,
           "oracle_rank2_rmse": rmse(P_or),
           "oracle_comp1_rmse": rmse(P_or1),
           "structure_rank2_rmse": rmse(P_st),
           "structure_comp1_rmse": rmse(P_c1),
           "n_sites": len(sub), "n_obs": int(mk.sum())}
    print(f"  参考体系 {REF_KEY[0]} x {REF_KEY[1]}，{len(sub)} 位点 "
          f"{int(mk.sum())} 观测")
    print(f"  {'预测器':<40}{'耦合 RMSE':>12}{'相对零预测器':>14}")
    print("  " + "-" * 66)
    for name, v in (("零预测器（耦合本身的 SD）", sd0),
                    ("oracle 秩2（u 由数据拟合）", out["oracle_rank2_rmse"]),
                    ("oracle 仅成分1", out["oracle_comp1_rmse"]),
                    ("结构-only 秩2（û1,û2 由 6M0J 算）",
                     out["structure_rank2_rmse"]),
                    ("结构-only 仅成分1", out["structure_comp1_rmse"])):
        print(f"  {name:<40}{v:>9.3f} kcal{(1 - v / sd0):>13.1%}")
    assert 0.3 < sd0 < 3.0, f"耦合 SD {sd0} 不在 ΔΔG 的物理量级内"
    return out


# ------------------------------------------------------------------- 主

def main():
    desc, draw = load_descriptors()
    # CA 坐标用于空间分组
    import csv as _csv  # noqa
    from field_orientation import parse_pdb
    atoms = parse_pdb(os.path.join(ROOT, "cache", "pdb", "6M0J.pdb"))
    for a in atoms:
        if a["chain"] == "E" and not a["het"] and a["name"] == "CA":
            try:
                ri = int(a["resi"])
            except ValueError:
                continue
            if ri in desc:
                desc[ri]["ca_xyz"] = list(a["xyz"])

    d, combos = load_targets()
    cons, used = consensus(combos)
    print("=" * 80)
    print("数据与规范")
    print("=" * 80)
    print(f"  可比组合 {len(used)}/{len(d['combos'])}: {', '.join(used)}")
    print(f"  参考基 {d['ref']['target']} x {d['ref']['ligand']}；"
          f"r(v1,疏水)={d['ref']['corr_v1_hyd']:+.3f}，"
          f"r(v2,平面sp²)={d['ref']['corr_v2_planar']:+.3f}")
    print(f"  wt 身份命中率（6M0J 链 E vs DMS Wuhan-Hu-1）"
          f"{draw['wt_match_rate']:.2%}")

    # ---- 重复可靠度（天花板）
    print("\n" + "=" * 80)
    print("1. 重复可靠度：同一批位点的 12 个独立估计（这是 R² 的天花板）")
    print("=" * 80)
    rel = {}
    rng = np.random.default_rng(SEED)
    for key in ("u1", "u2"):
        Z = cons[key]["Z"]
        k = Z.shape[0]
        rs = [float(np.corrcoef(Z[i], Z[j])[0, 1])
              for i in range(k) for j in range(i + 1, k)]
        rbar = float(np.mean(rs))
        icc = k * rbar / (1 + (k - 1) * rbar)        # Spearman-Brown
        # 经验分半校验
        halves = []
        for _ in range(200):
            perm = rng.permutation(k)
            A = Z[perm[: k // 2]].mean(axis=0)
            B = Z[perm[k // 2:]].mean(axis=0)
            halves.append(float(np.corrcoef(A, B)[0, 1]))
        h = float(np.mean(halves))
        sb = 2 * h / (1 + h)
        # 单个组合 r 的 bootstrap（对位点重抽）
        n = Z.shape[1]
        bs = []
        for _ in range(500):
            i = rng.integers(0, n, n)
            bs.append(np.mean([np.corrcoef(Z[a][i], Z[b][i])[0, 1]
                               for a in range(k) for b in range(a + 1, k)]))
        lo, hi = np.quantile(bs, [0.025, 0.975])
        rel[key] = {"rbar": rbar, "rbar_ci": [float(lo), float(hi)],
                    "icc_consensus": float(icc), "splithalf_sb": float(sb),
                    "k": k, "n_sites": n, "r_min": min(rs), "r_max": max(rs)}
        print(f"  {key}: 两两相关 平均 {rbar:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]（范围 {min(rs):+.3f}..{max(rs):+.3f}）")
        print(f"      => 单个组合的可靠度 {rbar:.3f}；"
              f"{k} 个组合平均后的共识可靠度 (Spearman-Brown) {icc:.3f}")
        print(f"      分半经验校验：半-半 r={h:.3f} -> 校正后 {sb:.3f}"
              f"（应与 {icc:.3f} 接近）")
        print(f"      **预测共识 {key} 的 R² 天花板 = {icc:.3f}**")

    # ---- 对齐特征矩阵
    sites_all = cons["u1"]["sites"]
    sites = [s for s in sites_all if s in desc]
    X_full = design(sites, desc, POOL)
    ok = np.isfinite(X_full).all(axis=1)
    print(f"\n  位点：载荷 {len(sites_all)} 个，有结构描述子 {len(sites)} 个，"
          f"特征无缺失 {int(ok.sum())} 个（丢弃 {int((~ok).sum())}）")
    sites = [s for s, k in zip(sites, ok) if k]
    X = design(sites, desc, POOL)
    idx = {s: i for i, s in enumerate(cons["u1"]["sites"])}
    Y = {k: np.array([cons[k]["mean"][idx[s]] for s in sites])
         for k in ("u1", "u2")}
    groups_sp = spatial_groups(sites, desc, k=10)
    loso = np.arange(len(sites))
    print(f"  空间分组 10 折的组大小: "
          f"{np.bincount(groups_sp).tolist()}")

    print("\n" + "=" * 80)
    print("2. 流程自检（两个方向，CLAUDE.md §10.2）")
    print("=" * 80)
    selftest_pipeline(X, groups_sp, POOL)

    # ---- P3 正交性诊断
    ru = float(np.corrcoef(Y["u1"], Y["u2"])[0, 1])
    bs = []
    rng2 = np.random.default_rng(3)
    for _ in range(NBOOT):
        i = rng2.integers(0, len(sites), len(sites))
        bs.append(np.corrcoef(Y["u1"][i], Y["u2"][i])[0, 1])
    lo3, hi3 = np.quantile(bs, [0.025, 0.975])
    print("\n" + "=" * 80)
    print("3. P3 诊断：u1 与 u2 的正交性")
    print("=" * 80)
    print(f"  r(u1,u2) = {ru:+.3f}  [{lo3:+.3f},{hi3:+.3f}]"
          f"  {'（与 0 不显著不同，符合 P3）' if lo3 < 0 < hi3 else '（显著非零）'}")

    # ---- 基线 + 预注册单项
    print("\n" + "=" * 80)
    print("4. 基线与预注册单项（每项都是单特征或同类特征组，分类不混）")
    print("=" * 80)
    named = {
        "WCN 基线 (wcn)": ["wcn"],
        "WCN_cb 基线 (wcn_cb)": ["wcn_cb"],
        "埋藏度基线 (burial8)": ["burial8"],
        "P1 场梯度 ‖∇E‖² (模型D)": ["loggradE2_D"],
        "P1 场梯度 ‖∇E‖² (模型F)": ["loggradE2_F"],
        "P1 对照 场强 |E|² (模型D)": ["logE2_D"],
        "P1 对照 场强 |E|² (模型F)": ["logE2_F"],
        "P2 各向异性（3 个特征值比）": ["flatness", "elongation", "sphericity"],
        "P2 各向异性 flatness 单项": ["flatness"],
        "P2 对照 自由体积大小": ["free_frac", "free_frac_hemi"],
        "P2 对照 free_frac 单项": ["free_frac"],
        "lobe 数": ["n_lobes"],
        "rotamer 平面-分支差": ["planar_minus_branched"],
        "主链 phi/psi": ["cos_phi", "sin_phi", "cos_psi", "sin_psi"],
        "界面距离 d_partner": ["d_partner"],
    }
    results = {}
    for key in ("u1", "u2"):
        y = Y[key]
        print(f"\n  --- 目标 {key}（共识，天花板 R²={rel[key]['icc_consensus']:.3f}）")
        print(f"  {'模型':<34}{'LOSO R²':>10}{'95% CI':>20}"
              f"{'空间CV R²':>11}{'95% CI':>20}")
        print("  " + "-" * 93)
        base_pred = None
        for name, fs in named.items():
            Xi = design(sites, desc, fs)
            p1 = cv_predict(Xi, y, loso)
            p2 = cv_predict(Xi, y, groups_sp)
            c1 = boot_r2(y, p1)
            c2 = boot_r2(y, p2)
            if name.startswith("WCN 基线"):
                base_pred = p2
            print(f"  {name:<34}{r2(y, p1):>10.3f}"
                  f"{f'[{c1[0]:+.3f},{c1[1]:+.3f}]':>20}"
                  f"{r2(y, p2):>11.3f}{f'[{c2[0]:+.3f},{c2[1]:+.3f}]':>20}")
            results[(key, name)] = {"loso": r2(y, p1), "loso_ci": c1,
                                    "sp": r2(y, p2), "sp_ci": c2,
                                    "feats": fs}
        # 与 WCN 基线的配对差（空间 CV）
        print(f"\n  与 WCN 基线的配对 ΔR²（空间 CV，2000 次 bootstrap）：")
        for name in named:
            if name.startswith("WCN 基线"):
                continue
            Xi = design(sites, desc, named[name])
            p = cv_predict(Xi, y, groups_sp)
            d1, d2 = boot_dr2(y, p, base_pred)
            flag = "" if d1 < 0 < d2 else ("  ** 不含 0" if d1 > 0 else
                                           "  (显著更差)")
            print(f"    {name:<34}ΔR² = {r2(y, p) - r2(y, base_pred):+.3f}"
                  f"  [{d1:+.3f},{d2:+.3f}]{flag}")

    # ---- 搜索 + 置换天花板
    print("\n" + "=" * 80)
    print("5. 穷举搜索（<=3 个特征）与同规模置换天花板")
    print("=" * 80)
    pool_idx = list(range(len(POOL)))
    ncomb = sum(1 for k in range(1, MAXK + 1)
                for _ in itertools.combinations(pool_idx, k))
    print(f"  特征池 {len(POOL)} 个，搜索 {ncomb} 个组合（<= {MAXK} 特征）；"
          f"CV = 空间 10 折")
    search = {}
    for key in ("u1", "u2"):
        y = Y[key]
        best, bf, n = search_best(X, y, groups_sp, pool_idx)
        p = cv_predict(X[:, list(bf)], y, groups_sp)
        ci = boot_r2(y, p)
        print(f"\n  {key}: 搜索了 {n} 个组合；最优 R² = {best:.3f} "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}]")
        print(f"      特征 = {[POOL[i] for i in bf]}")
        nperm = 60
        null = permutation_ceiling(X, y, groups_sp, pool_idx, nperm)
        print(f"      置换天花板（{nperm} 次，每次重跑全部 {n} 个组合的选择）："
              f"中位 {np.median(null):.3f}，95 分位 {np.quantile(null, 0.95):.3f}"
              f"，最大 {null.max():.3f}")
        over = "**不重叠**" if ci[0] > np.quantile(null, 0.95) else "重叠"
        print(f"      最优 R² 的 95% 下界 {ci[0]:+.3f} vs 置换 95 分位 "
              f"{np.quantile(null, 0.95):.3f} => {over}")
        search[key] = {"best_r2": best, "ci": ci,
                       "feats": [POOL[i] for i in bf], "n_search": n,
                       "perm_med": float(np.median(null)),
                       "perm_p95": float(np.quantile(null, 0.95)),
                       "perm_max": float(null.max()),
                       "ceiling": rel[key]["icc_consensus"]}
        frac = best / rel[key]["icc_consensus"]
        print(f"      占可靠度天花板的比例 {frac:+.1%}")

    # ---- 6. 嵌套搜索 CV（无偏点估计）
    print("\n" + "=" * 80)
    print("6. 嵌套搜索 CV：特征选择放进训练折内（无偏的样本外 R²）")
    print("=" * 80)
    nested = {}
    for key in ("u1", "u2"):
        y = Y[key]
        p, chosen = nested_search_cv(X, y, groups_sp, pool_idx)
        ci = boot_r2(y, p)
        cnt = {}
        for c in chosen:
            for i in c:
                cnt[POOL[i]] = cnt.get(POOL[i], 0) + 1
        top = sorted(cnt.items(), key=lambda kv: -kv[1])[:6]
        print(f"  {key}: 嵌套 R² = {r2(y, p):.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
              f"   占天花板 {r2(y, p) / rel[key]['icc_consensus']:+.1%}")
        print(f"      10 个外折各自选出的特征里出现次数最多："
              f"{', '.join(f'{k}×{v}' for k, v in top)}")
        nested[key] = {"r2": r2(y, p), "ci": ci,
                       "feat_counts": cnt}

    # ---- 7. 预注册量的相关（R² 近零时仍要看相关，避免把弱真信号漏掉）
    print("\n" + "=" * 80)
    print("7. 预注册描述子与 u 的 Pearson 相关（bootstrap 区间）")
    print("=" * 80)
    prereg = ["loggradE2_D", "loggradE2_F", "logE2_D", "logE2_F",
              "flatness", "elongation", "sphericity", "n_lobes",
              "free_frac", "free_frac_hemi", "wcn", "burial8",
              "planar_minus_branched", "lognall_LEU", "lognall_PHE",
              "shell1", "shell4", "d_partner"]
    rngc = np.random.default_rng(77)
    corr = {}
    print(f"  {'描述子':<24}{'r(u1)':>9}{'95% CI':>20}{'r(u2)':>9}{'95% CI':>20}")
    print("  " + "-" * 82)
    for f in prereg:
        x = design(sites, desc, [f])[:, 0]
        line = f"  {f:<24}"
        for key in ("u1", "u2"):
            y = Y[key]
            r = float(np.corrcoef(x, y)[0, 1])
            bs = []
            for _ in range(NBOOT):
                i = rngc.integers(0, len(y), len(y))
                if x[i].std() == 0:
                    continue
                bs.append(np.corrcoef(x[i], y[i])[0, 1])
            lo, hi = np.quantile(bs, [0.025, 0.975])
            corr[f"{f}|{key}"] = {"r": r, "ci": [float(lo), float(hi)]}
            line += f"{r:>+9.3f}{f'[{lo:+.3f},{hi:+.3f}]':>20}"
        print(line)

    # ---- 8. 量级校准（CLAUDE.md §8）：换算成 kcal/mol
    print("\n" + "=" * 80)
    print("8. 量级校准：结构-only 代理量在参考体系上的耦合 RMSE（kcal/mol）")
    print("=" * 80)
    mag = magnitude_check(d, combos, desc, sites, X, Y, groups_sp, pool_idx,
                          nested)
    out = {"reliability": rel, "nested": nested, "corr": corr,
           "magnitude": mag, "r_u1_u2": {"r": ru, "ci": [float(lo3),
                                                           float(hi3)]},
           "named": {f"{k}|{n}": v for (k, n), v in results.items()},
           "search": search, "n_sites": len(sites), "pool": POOL,
           "combos_used": used}
    path = os.path.join(ROOT, "cache", "site_loading_prediction.json")
    json.dump(path and out, open(path, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
