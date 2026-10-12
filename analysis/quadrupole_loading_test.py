#!/usr/bin/env python3
"""docs/16 的 **P4** 检验：算出来的 ‖Q_r‖² 能不能打败二值平面 sp² 标签。

P4 原文（docs/16 §3，预注册，不得改）：
  「用实测侧链几何重建出的 ‖Q_r‖²，与成分 2 的残基载荷的相关应当强于
   已有的二值平面 sp² 指示变量（r = −0.768）。
   **一个连续的物理量应当打败一个二值标签。** 打不败就是把『平面』改了名。」

---------------------------------------------------------------- 流程

数据 cache/dms/starr_animalACE2_final_variant_scores.csv
  3 骨架（Wuhan-Hu-1 / Eta / Omicron_BA2） × 5 ACE2 正交体 = 15 个组合。
  本文件**自己重算**秩2 分解，复用 analysis/site_loadings.py 里已自检过的
  build_matrix / fit_rank2（双向中心化 + `rank_test_power.als_guarded`，
  **不是** 未加保护的 als_complete —— docs/12 §1(b) 的第八次工具错误）。

### 规范（gauge）

秩2 分解的载荷只在共享的可逆 2×2 变换下唯一，所以「成分 2 的载荷向量」
在固定规范之前**不存在**。本文件给三个层次，从最依赖规范到完全不依赖：

  G1 SVD 规范：V 取右奇异向量（按奇异值排序），成分 2 = V[:,1]。
     符号由**已发表的事实**锚定：docs/12/16 记 Ile 是成分 2 最正的残基，
     故取 sign 使 v2[Ile] > 0。（符号不影响 |r|，只影响报告的正负号。）
  G2 疏水正交化规范（**主**）：在 span(V) 这个二维子空间里，
     e1 := 疏水性 π 在 span(V) 上的投影方向；e2 := span(V) 内与 e1 正交的方向。
     成分 2 := 沿 e2 的载荷。**完全不引用平面性**，故对本检验不循环。
  G3 子空间规范不变检验：问「二维残基载荷子空间 span(V) 被
     span{疏水, ‖Q‖²} 解释多少」vs「被 span{疏水, 二值平面} 解释多少」。
     用两个二维子空间的主角余弦平方（典型相关平方）之均值。
     这个量在 V 的任意 2×2 可逆变换下**不变**，是最可辩护的陈述。

### 统计纪律（CLAUDE.md §5）

- 15 个组合当重复，bootstrap 重抽 **4000** 次，报区间不报点估计。
- 与基线比**区间重叠**；另外报**配对差**（同组合内 Q 减平面）的区间，
  因为配对差去掉了组合间的共同波动，是更有功效的比较。
- 截尾剔除：ρ(截尾比例, 检出秩) = −0.77 [−0.89, −0.54]，≥25% 截尾时 8/9 给秩 0
  （docs/19 重算；原记 −0.786 / 7/7 已更正）。
  故主分析只用截尾 < 10% 的组合；全 15 个的版本并列报告。
  **信噪比不是判据，截尾才是**（docs/12 §2）。
- 描述子变体数**逐个登记**（见 VARIANTS），置换零分布在**该实际搜索量级**
  下做：每次置换残基标签后取全部变体上的最大 |r|。

### 自检（CLAUDE.md §10）

(1) AAindex 解析：Gly 的 Fauchere-Pliska π 必须为 0，Trp 必须最大（D-023）。
(2) 四极表：已知事实 —— Gly 必须为 0（无侧链），Ala 必须近 0（纯烷基），
    Trp 必须大于 Leu（吲哚 vs 烷基）。
(3) 规范还原性：造一个已知的秩2 矩阵（V0 的第二列**就取为** ‖Q‖² 的
    标准化版本），本流程的 G2 必须把它还原（|r| ≈ 1）。
    这是「测不出结构 vs 方法不够」的理想条件诊断（§10.3）。
(4) 不误伤：把 ‖Q‖² 换成随机向量，G2 必须给出接近 0 的 |r|。
(5) 子空间度量：相同子空间必须给 1，正交子空间必须给 0。

输出：cache/quadrupole_p4.json
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_redundancy import parse_aaindex
from site_loadings import (AA20, CSV_NAME, DMS, LIGANDS, PLANAR1, ROOT,
                           build_matrix, fit_rank2)
from sidechain_quadrupole import load_table

TARGETS = ["Wuhan-Hu-1", "Eta", "Omicron_BA2"]
N_BOOT = 4000
N_PERM = 4000
CENS_CUT = 0.10
EXCLUDE_RES = {"P"}        # PRO：侧链与主链共价闭环，取代基概念不成立（D-022）
KEY7 = ["I", "V", "T", "C", "E", "W", "D"]
KEY7_LOADING = {"I": +7.95, "V": +7.43, "T": +4.72, "C": +4.19,
                "E": -6.30, "W": -6.07, "D": -5.65}   # docs/16 §1 的事实

# ---------------------------------------------------------------- 变体登记
# 每个变体 = (名字, 表列名, 是否取 log1p)。**搜索量级就是 len(VARIANTS)**。
VARIANT_COLS = [
    "q2_centroid_mean", "q2_cb_mean", "q2_qcent_mean",
    "q2b_centroid_mean", "q2b_cb_mean", "q2b_qcent_mean",
    "q2c_centroid_mean", "q2c_cb_mean", "q2c_qcent_mean",
]
VARIANTS = [(c, c, False) for c in VARIANT_COLS] + \
           [("log_" + c, c, True) for c in VARIANT_COLS]
PRIMARY = "q2_centroid_mean"      # docs/16 §2 的 x₀ = 侧链质心；π 只加芳香环


# ================================================================ 描述子

def load_descriptors():
    aai = parse_aaindex(os.path.join(ROOT, "cache", "aaindex1.txt"),
                        {"FAUJ830101"})
    pi = aai["FAUJ830101"]
    assert abs(pi["G"]) < 1e-9, f"Gly 的 π 应为 0，读到 {pi['G']}"
    assert max(pi, key=pi.get) == "W", "π 最大应是 Trp"
    assert pi["I"] > pi["S"], "Ile 应比 Ser 疏水"
    tab = load_table()
    for c in VARIANT_COLS:
        assert c in tab, f"四极表缺列 {c}"
    q = tab[PRIMARY]
    assert abs(q["G"]) < 1e-9, f"Gly 的 ‖Q‖² 应为 0，读到 {q['G']}"
    assert q["A"] < 1.0, f"Ala 的 ‖Q‖² 应近 0，读到 {q['A']}"
    assert q["W"] > 10 * q["L"], "Trp 的 ‖Q‖² 应远大于 Leu"
    arm = json.load(open(os.path.join(ROOT, "cache",
                                      "sidechain_arm.json")))["arm"]
    three2one = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                 "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
                 "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                 "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    arm1 = {three2one[k]: v for k, v in arm.items()}
    arm1.setdefault("G", 0.0)      # GLY 无侧链，臂长 0（表里本来就没有这一行）
    return {"hyd": pi, "arm": arm1,
            "planar": {a: (1.0 if a in PLANAR1 else 0.0) for a in AA20},
            # **事后诊断**（不属于预注册 P4）：只标芳香环的二值量。
            # 加它的理由在结果里：重算出的成分2 负端全是 F/Y/W/H，
            # 而 D/E/N/Q 落在零附近 —— 需要知道 PLANAR1 的 0.68 有多少
            # 其实只来自芳香性。
            "arom": {a: (1.0 if a in set("FYWH") else 0.0) for a in AA20},
            "qtab": tab}


def vec(d, res):
    return np.array([d[a] for a in res], float)


# ================================================================ 规范

def gauge_svd(V, res, anchor="I"):
    """G1：成分 2 = V[:,1]，符号由 v2[Ile] > 0 锚定（docs/12/16 的事实）。"""
    v2 = V[:, 1].copy()
    if anchor in res and v2[res.index(anchor)] < 0:
        v2 = -v2
    return v2


def gauge_hyd_orth(V, res, hyd):
    """G2：span(V) 内，与「疏水性的投影方向」正交的那个方向上的载荷。

    返回该方向的残基载荷向量（符号由 v2[Ile] > 0 锚定）。
    """
    Qb, _ = np.linalg.qr(V)                 # span(V) 的正交基，19x2
    h = hyd - hyd.mean()
    a = Qb.T @ h                            # 疏水在子空间里的坐标
    if np.linalg.norm(a) < 1e-12:
        return gauge_svd(V, res)
    a = a / np.linalg.norm(a)
    b = np.array([-a[1], a[0]])             # 子空间内的正交方向
    v2 = Qb @ b
    if "I" in res and v2[res.index("I")] < 0:
        v2 = -v2
    return v2


def subspace_overlap(A, B):
    """两个二维子空间的主角余弦平方。返回 (均值, [cos²θ1, cos²θ2])。"""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0) ** 2
    return float(s.mean()), [float(x) for x in s]


def pear(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ================================================================ 自检

def selftest(desc):
    print("=" * 80)
    print("自检（CLAUDE.md §10）")
    print("=" * 80)
    res = [a for a in AA20 if a not in EXCLUDE_RES]
    q = vec(desc["qtab"][PRIMARY], res)
    h = vec(desc["hyd"], res)
    print(f"  [自检 1-2 已知事实] π(Gly)=0 ✓, π 最大=Trp ✓；"
          f"‖Q‖²(Gly)={desc['qtab'][PRIMARY]['G']:.2f} ✓, "
          f"Ala={desc['qtab'][PRIMARY]['A']:.2f} ✓, "
          f"Trp/Leu={q[res.index('W')]/q[res.index('L')]:.0f}× ✓")
    # (3) 理想条件还原性：把 ‖Q‖² 真的塞进成分 2，G2 必须找回来
    rng = np.random.default_rng(20261011)
    n_s = 150
    qz = (q - q.mean()) / q.std()
    hz = (h - h.mean()) / h.std()
    V0 = np.column_stack([hz, qz])
    V0 /= np.linalg.norm(V0, axis=0, keepdims=True)
    L0 = rng.normal(0, 1, (n_s, 2)) * np.array([3.0, 1.2])
    M = (rng.normal(0, 2, n_s)[:, None] + rng.normal(0, 2, len(res))[None, :]
         + L0 @ V0.T)
    for sigma in (0.0, 0.3):
        mask = rng.random(M.shape) < 0.9
        Mn = M + rng.normal(0, sigma, M.shape)
        V, _ = _fit(np.where(mask, Mn, np.nan), mask)
        v2 = gauge_hyd_orth(V, res, hz)
        r = pear(v2, qz)
        print(f"  [自检 3 还原 σ={sigma}] G2 找回 ‖Q‖² 的 r = {r:+.4f}")
        assert abs(r) > 0.90, "理想条件下都还原不出来 —— 流程不可用（§10.3）"
    # (4) 不误伤
    bad = rng.normal(0, 1, len(res))
    V, _ = _fit(np.where(mask, M, np.nan), mask)
    rb = pear(gauge_hyd_orth(V, res, hz), bad)
    print(f"  [自检 4 不误伤] 随机向量 vs 同一个 G2 成分2 的 r = {rb:+.4f}"
          f"（应接近 0）")
    assert abs(rb) < 0.6, "对随机向量也给高相关 —— 护栏无效"
    # (5) 子空间度量
    A = rng.normal(0, 1, (len(res), 2))
    same, _ = subspace_overlap(A, A @ np.array([[1.0, 2.0], [-3.0, 0.5]]))
    Z = np.linalg.qr(rng.normal(0, 1, (len(res), 4)))[0]
    orth, _ = subspace_overlap(Z[:, :2], Z[:, 2:])
    print(f"  [自检 5 子空间度量] 同一子空间（经 2×2 变换）= {same:.6f}（应 1）；"
          f"正交子空间 = {orth:.2e}（应 0）")
    assert abs(same - 1) < 1e-9 and orth < 1e-9, "子空间度量坏了"
    print("自检全部通过。\n")


def _fit(M, mask):
    L, V, s, Rm, P, st = fit_rank2(M, mask)
    return V, st


# ================================================================ 主流程

def main():
    desc = load_descriptors()
    selftest(desc)

    rows = list(csv.DictReader(open(os.path.join(DMS, CSV_NAME))))
    combos = []
    print("=" * 96)
    print("15 个「骨架 × ACE2 正交体」组合的秩2 分解与规范固定")
    print("=" * 96)
    print(f"{'骨架':<13}{'配体':<22}{'位点':>5}{'观测':>7}{'截尾':>7}"
          f"{'s1':>7}{'s2':>7}{'秩2解释':>8}{'r(v1,π)':>9}"
          f"{'G1 r(v2,Q)':>11}{'G2 r(v2,Q)':>11}")
    print("-" * 96)
    for t in TARGETS:
        for lg in LIGANDS:
            b = build_matrix(rows, t, lg)
            if b is None:
                continue
            sites, res_all, M, mask, cf = b
            if mask.sum() < 500 or len(sites) < 50:
                print(f"{t:<13}{lg:<22} 跳过（观测 {int(mask.sum())}）")
                continue
            keep = [i for i, a in enumerate(res_all)
                    if a not in EXCLUDE_RES]
            res = [res_all[i] for i in keep]
            V, st = _fit(M, mask)
            V = V[keep]                      # 去掉 PRO 这一列
            h = vec(desc["hyd"], res)
            v1 = V[:, 0]
            r1 = pear(v1, h)
            v2_g1 = gauge_svd(V, res)
            v2_g2 = gauge_hyd_orth(V, res, h)
            q = vec(desc["qtab"][PRIMARY], res)
            combos.append({"target": t, "ligand": lg, "res": res,
                           "V": V, "cens": cf, "n_obs": int(mask.sum()),
                           "sites": len(sites), "s": st,
                           "v2_g1": v2_g1, "v2_g2": v2_g2, "v1": v1,
                           "sing": None})
            L, Vf, s, *_ = fit_rank2(M, mask)
            combos[-1]["sing"] = s.tolist()
            print(f"{t:<13}{lg:<22}{len(sites):>5}{int(mask.sum()):>7}"
                  f"{cf:>7.1%}{s[0]:>7.2f}{s[1]:>7.2f}"
                  f"{st['rank2_expl']:>8.1%}{r1:>+9.3f}"
                  f"{pear(v2_g1, q):>+11.3f}{pear(v2_g2, q):>+11.3f}")

    good = [c for c in combos if c["cens"] < CENS_CUT]
    print("-" * 96)
    print(f"共 {len(combos)} 个组合；截尾 < {CENS_CUT:.0%} 的 {len(good)} 个"
          f"（主分析用这些；docs/12 §2：截尾摧毁秩的可检出性，"
          f"ρ(截尾,秩) = −0.77 [−0.89,−0.54]，docs/19 重算）")

    # ------------------------------------------------ 逐组合相关 + bootstrap
    def per_combo(dvec_name, gauge, col=None, logt=False, sub=None):
        """返回每组合的 r。dvec_name='Q'|'planar'|任意描述子键。"""
        sub = combos if sub is None else sub
        out = []
        for c in sub:
            res = c["res"]
            if dvec_name == "planar":
                d = vec(desc["planar"], res)
            elif dvec_name == "Q":
                d = vec(desc["qtab"][col], res)
                if logt:
                    d = np.log1p(d)
            else:
                d = vec(desc[dvec_name], res)
            out.append(pear(c["v2_" + gauge], d))
        return np.array(out)

    def boot_mean(x, rng, n=N_BOOT):
        idx = rng.integers(0, len(x), (n, len(x)))
        return np.sort(x[idx].mean(axis=1))

    def ci(b):
        return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))

    rng = np.random.default_rng(777)
    print("\n" + "=" * 96)
    print("P4 主检验：|r(成分2 载荷, 描述子)| 的 bootstrap 区间"
          f"（{N_BOOT} 次重抽，重抽单元 = 组合）")
    print("=" * 96)
    results = {}
    for tag, sub in (("截尾<10%（主）", good), ("全部 15 个", combos)):
        print(f"\n--- {tag}，n = {len(sub)} ---")
        print(f"{'规范':<6}{'描述子':<26}{'均值 r':>9}{'|r| 均值':>10}"
              f"{'bootstrap 95% 区间 of |r|':>30}")
        for gauge in ("g1", "g2"):
            for nm, kw in (("‖Q‖² (主变体)",
                            dict(dvec_name="Q", col=PRIMARY)),
                           ("log(1+‖Q‖²)",
                            dict(dvec_name="Q", col=PRIMARY, logt=True)),
                           ("二值平面 sp²（基线）",
                            dict(dvec_name="planar"))):
                r = per_combo(gauge=gauge, sub=sub, **kw)
                b = boot_mean(np.abs(r), rng)
                lo, hi = ci(b)
                print(f"{gauge.upper():<6}{nm:<26}{r.mean():>+9.3f}"
                      f"{np.abs(r).mean():>10.3f}"
                      f"      [{lo:.3f}, {hi:.3f}]")
                results[f"{tag}|{gauge}|{nm}"] = {
                    "mean_r": float(r.mean()),
                    "mean_absr": float(np.abs(r).mean()),
                    "ci": [lo, hi], "per_combo": r.tolist(), "n": len(sub)}
        # 配对差（更有功效）
        print("  配对差 |r(Q)| − |r(平面)|（同组合内，去掉组合间共同波动）：")
        for gauge in ("g1", "g2"):
            for nm, kw in (("‖Q‖²", dict(dvec_name="Q", col=PRIMARY)),
                           ("log(1+‖Q‖²)",
                            dict(dvec_name="Q", col=PRIMARY, logt=True))):
                dq = np.abs(per_combo(gauge=gauge, sub=sub, **kw))
                dp = np.abs(per_combo("planar", gauge, sub=sub))
                d = dq - dp
                b = boot_mean(d, rng)
                lo, hi = ci(b)
                sig = "区间不含 0 ⇒ 有差别" if lo * hi > 0 else \
                      "区间含 0 ⇒ 无法区分"
                print(f"    {gauge.upper()} {nm:<14} Δ = {d.mean():+.3f}  "
                      f"[{lo:+.3f}, {hi:+.3f}]  {sig}")
                results[f"paired|{tag}|{gauge}|{nm}"] = {
                    "delta": float(d.mean()), "ci": [lo, hi],
                    "per_combo": d.tolist()}

    # ------------------------------------------------ G3 规范不变子空间检验
    print("\n" + "=" * 96)
    print("G3 规范不变检验：二维残基载荷子空间 span(V) 被描述子对解释多少")
    print("（主角余弦平方的均值；在 V 的任意 2×2 可逆变换下不变）")
    print("=" * 96)
    print(f"{'描述子对':<34}{'均值 cos²':>10}"
          f"{'bootstrap 95% 区间':>24}  逐组合 cos²θ1/cos²θ2 中位")
    pairs = {
        "{疏水, ‖Q‖²}": lambda res: np.column_stack(
            [vec(desc["hyd"], res), vec(desc["qtab"][PRIMARY], res)]),
        "{疏水, log(1+‖Q‖²)}": lambda res: np.column_stack(
            [vec(desc["hyd"], res), np.log1p(vec(desc["qtab"][PRIMARY],
                                                 res))]),
        "{疏水, 二值平面}（基线）": lambda res: np.column_stack(
            [vec(desc["hyd"], res), vec(desc["planar"], res)]),
        "{疏水, 臂长}（反证对照）": lambda res: np.column_stack(
            [vec(desc["hyd"], res), vec(desc["arm"], res)]),
    }
    g3 = {}
    for nm, f in pairs.items():
        vals, ang = [], []
        for c in good:
            D = f(c["res"])
            D = D - D.mean(axis=0, keepdims=True)
            m, s2 = subspace_overlap(c["V"], D)
            vals.append(m); ang.append(s2)
        vals = np.array(vals)
        b = boot_mean(vals, rng)
        lo, hi = ci(b)
        a = np.median(np.array(ang), axis=0)
        print(f"{nm:<34}{vals.mean():>10.3f}      [{lo:.3f}, {hi:.3f}]"
              f"       {a[0]:.3f} / {a[1]:.3f}")
        g3[nm] = {"mean": float(vals.mean()), "ci": [lo, hi],
                  "per_combo": vals.tolist(),
                  "median_cos2": [float(x) for x in a]}
    # 配对差
    base = np.array(g3["{疏水, 二值平面}（基线）"]["per_combo"])
    for nm in ("{疏水, ‖Q‖²}", "{疏水, log(1+‖Q‖²)}"):
        d = np.array(g3[nm]["per_combo"]) - base
        b = boot_mean(d, rng)
        lo, hi = ci(b)
        print(f"  配对差 {nm} − 基线 = {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]"
              f"  {'不含 0' if lo*hi > 0 else '含 0'}")
        g3[nm]["paired_vs_planar"] = {"delta": float(d.mean()),
                                      "ci": [lo, hi]}

    # ------------------------------------------------ 全变体扫描 + 置换零分布
    print("\n" + "=" * 96)
    print(f"全部 {len(VARIANTS)} 个描述子变体的扫描（登记在案的搜索量级）")
    print("=" * 96)
    scan = {}
    print(f"{'变体':<24}{'G2 |r| 均值':>12}{'G1 |r| 均值':>12}")
    for nm, col, logt in VARIANTS:
        r2 = np.abs(per_combo("Q", "g2", col=col, logt=logt, sub=good)).mean()
        r1 = np.abs(per_combo("Q", "g1", col=col, logt=logt, sub=good)).mean()
        scan[nm] = {"g2": float(r2), "g1": float(r1)}
        print(f"{nm:<24}{r2:>12.3f}{r1:>12.3f}")
    best_nm = max(scan, key=lambda k: scan[k]["g2"])
    best_obs = scan[best_nm]["g2"]
    plan_obs = np.abs(per_combo("planar", "g2", sub=good)).mean()
    print(f"\n扫描内最佳 = {best_nm}，|r| 均值 {best_obs:.3f}；"
          f"二值平面基线 {plan_obs:.3f}")

    print(f"\n置换零分布（{N_PERM} 次；每次置换**残基标签**，"
          f"然后在全部 {len(VARIANTS)} 个变体上取最大 |r| —— "
          f"即运气天花板按实际候选量级算）")
    perm_rng = np.random.default_rng(31337)
    res0 = good[0]["res"]
    dmats = []
    for nm, col, logt in VARIANTS:
        d = vec(desc["qtab"][col], res0)
        dmats.append(np.log1p(d) if logt else d)
    dmats = np.array(dmats)                        # n_var x n_res
    V2 = np.array([c["v2_g2"] for c in good])      # n_combo x n_res
    V2c = V2 - V2.mean(axis=1, keepdims=True)
    V2n = V2c / np.linalg.norm(V2c, axis=1, keepdims=True)
    null_max, null_plan = [], []
    plan_v = vec(desc["planar"], res0)
    for _ in range(N_PERM):
        p = perm_rng.permutation(len(res0))
        Dp = dmats[:, p]
        Dc = Dp - Dp.mean(axis=1, keepdims=True)
        Dn = Dc / np.maximum(np.linalg.norm(Dc, axis=1, keepdims=True), 1e-12)
        R = np.abs(V2n @ Dn.T)                     # n_combo x n_var
        null_max.append(R.mean(axis=0).max())
        pv = plan_v[p] - plan_v[p].mean()
        pv = pv / np.linalg.norm(pv)
        null_plan.append(float(np.abs(V2n @ pv).mean()))
    null_max = np.array(null_max)
    null_plan = np.array(null_plan)
    p_best = float((null_max >= best_obs).mean())
    p_prim = float((null_max >= scan[PRIMARY]["g2"]).mean())
    p_plan = float((null_plan >= plan_obs).mean())
    print(f"  置换零分布（最大 |r| over {len(VARIANTS)} 变体）"
          f"中位 {np.median(null_max):.3f}，95 分位 {np.quantile(null_max,0.95):.3f}，"
          f"最大 {null_max.max():.3f}")
    print(f"  观测最佳 {best_obs:.3f} => p = {p_best:.4f}")
    print(f"  观测主变体 {scan[PRIMARY]['g2']:.3f} => p = {p_prim:.4f}")
    print(f"  二值平面（单一描述子，无搜索）{plan_obs:.3f} => p = {p_plan:.4f}"
          f"（其零分布 95 分位 {np.quantile(null_plan,0.95):.3f}）")

    # ------------------------------------------------ 逐残基核对（7 个关键残基）
    print("\n" + "=" * 96)
    print("逐残基核对：docs/16 §1 记下的 7 个关键残基")
    print("=" * 96)
    r0 = good[0]["res"]
    v2m = V2.mean(axis=0)
    v2m = v2m / np.abs(v2m).max() * 8.0        # 只为可读，缩放到 docs/16 的量级
    print(f"{'残基':<6}{'docs/16 成分2 载荷':>20}{'本文重算（缩放）':>18}"
          f"{'‖Q‖² 质心':>12}{'‖Q‖² CB':>12}{'二值平面':>10}")
    qp = desc["qtab"][PRIMARY]
    qcb = desc["qtab"]["q2_cb_mean"]
    for a in KEY7:
        i = r0.index(a)
        print(f"{a:<6}{KEY7_LOADING[a]:>+20.2f}{v2m[i]:>+18.2f}"
              f"{qp[a]:>12.2f}{qcb[a]:>12.2f}"
              f"{desc['planar'][a]:>10.0f}")
    print("\n全部 19 个残基（按本文重算的成分2 载荷排序）：")
    order = np.argsort(-v2m)
    print(f"{'残基':<6}{'成分2（缩放）':>14}{'‖Q‖² 质心':>12}"
          f"{'log(1+‖Q‖²)':>14}{'二值平面':>10}{'疏水 π':>9}{'臂长':>8}")
    for i in order:
        a = r0[i]
        print(f"{a:<6}{v2m[i]:>+14.2f}{qp[a]:>12.2f}"
              f"{np.log1p(qp[a]):>14.3f}{desc['planar'][a]:>10.0f}"
              f"{desc['hyd'][a]:>9.2f}{desc['arm'][a]:>8.2f}")

    # ------------------------------------------------ 两个已有反证
    print("\n" + "=" * 96)
    print("两个已有反证对本描述子是否仍成立（它不能是臂长或疏水性的改名）")
    print("=" * 96)
    qv = vec(qp, r0)
    for nm, d in (("臂长 cache/sidechain_arm.json", vec(desc["arm"], r0)),
                  ("疏水 π FAUJ830101", vec(desc["hyd"], r0)),
                  ("二值平面 sp²", vec(desc["planar"], r0))):
        print(f"  r(‖Q‖², {nm}) = {pear(qv, d):+.3f}   "
              f"r(log(1+‖Q‖²), 同) = {pear(np.log1p(qv), d):+.3f}")

    # ---------------------------------- 次级检验：直接对 docs/10 的 phi2 向量
    # P4 的「r = -0.768」基线是在 **docs/10 的 9 体系数据集** 上测的，
    # 不是本文重算的 ACE2 DMS。docs/10 正文逐个列出了 14 个残基的 phi2，
    # 所以可以在**那个确切的向量**上做一次最字面的 P4 检验。
    # 局限（必须一起读）：只有 14 个残基，且缺 K/R/Q/G/S/P ——
    # 而 K/R 恰好是 ||Q||^2 最极端的两个（净电荷 +1），
    # 所以这个子集系统性地回避了该描述子最薄弱的地方。
    print("\n" + "=" * 96)
    print("次级检验：直接对 docs/10 §(第二部分)§1 列出的 phi2 载荷（14 个残基）")
    print("=" * 96)
    P2 = {"I": 7.95, "V": 7.43, "T": 4.72, "C": 4.19, "M": 2.31, "A": 2.25,
          "L": 0.29, "F": -1.38, "N": -3.29, "Y": -3.60, "H": -3.84,
          "D": -5.65, "W": -6.07, "E": -6.30}
    r14 = sorted(P2)
    y14 = np.array([P2[a] for a in r14])
    cands = {"‖Q‖² 质心": vec(desc["qtab"][PRIMARY], r14),
             "log(1+‖Q‖²) 质心": np.log1p(vec(desc["qtab"][PRIMARY], r14)),
             "二值平面 sp²（基线）": vec(desc["planar"], r14),
             "二值芳香 FYWH": vec(desc["arom"], r14),
             "臂长（反证对照）": vec(desc["arm"], r14)}
    brng = np.random.default_rng(1221)
    sec = {}
    print(f"{'描述子':<26}{'r':>8}{'|r| bootstrap 95% 区间（残基重抽 4000）':>42}")
    for nm, x in cands.items():
        rr = pear(y14, x)
        bb = []
        for _ in range(N_BOOT):
            i = brng.integers(0, len(r14), len(r14))
            if np.std(x[i]) < 1e-9 or np.std(y14[i]) < 1e-9:
                continue
            bb.append(abs(pear(y14[i], x[i])))
        bb = np.array(bb)
        lo, hi = ci(bb)
        print(f"{nm:<26}{rr:>+8.3f}                 [{lo:.3f}, {hi:.3f}]")
        sec[nm] = {"r": rr, "ci": [lo, hi]}
    print("  注：同一 14 个残基上，二值平面给 "
          f"{sec['二值平面 sp²（基线）']['r']:+.3f}，"
          f"比 docs/10 在全 20 残基上报的 −0.768 还强 —— 子集对基线也有利。")

    # ------------------------------------------------ 事后诊断（明确标注）
    print("\n" + "=" * 96)
    print("事后诊断（**不属于预注册 P4**，CLAUDE.md §4：单列、不计入 P4 统计）")
    print("=" * 96)
    ar = np.abs(per_combo("arom", "g2", sub=good))
    ba = boot_mean(ar, rng)
    print(f"  只标芳香（FYWH）的二值量：|r| 均值 {ar.mean():.3f}  "
          f"[{ci(ba)[0]:.3f}, {ci(ba)[1]:.3f}]")
    pl = np.abs(per_combo("planar", "g2", sub=good))
    dd = ar - pl
    bd = boot_mean(dd, rng)
    print(f"  配对差 芳香 − 平面sp²(FYWHDENQ) = {dd.mean():+.3f}  "
          f"[{ci(bd)[0]:+.3f}, {ci(bd)[1]:+.3f}]"
          f"  {'不含 0' if ci(bd)[0]*ci(bd)[1] > 0 else '含 0'}")

    def partial(y, x, z):
        """r(y, x | z)。"""
        def rr(u, v):
            return pear(u, v)
        ryx, ryz, rxz = rr(y, x), rr(y, z), rr(x, z)
        den = np.sqrt(max((1 - ryz ** 2) * (1 - rxz ** 2), 1e-12))
        return (ryx - ryz * rxz) / den

    print("  控制臂长后的偏相关（因为 r(log‖Q‖², 臂长) 很高）：")
    for nm, kw in (("log(1+‖Q‖²)", dict(dvec_name="Q", col=PRIMARY,
                                        logt=True)),
                   ("二值平面 sp²", dict(dvec_name="planar")),
                   ("二值芳香 FYWH", dict(dvec_name="arom"))):
        vals = []
        for c in good:
            res = c["res"]
            if kw.get("dvec_name") == "Q":
                d = np.log1p(vec(desc["qtab"][kw["col"]], res))
            else:
                d = vec(desc[kw["dvec_name"]], res)
            vals.append(abs(partial(c["v2_g2"], d, vec(desc["arm"], res))))
        vals = np.array(vals)
        b = boot_mean(vals, rng)
        print(f"    |r(成分2, {nm} | 臂长)| = {vals.mean():.3f}  "
              f"[{ci(b)[0]:.3f}, {ci(b)[1]:.3f}]")

    json.dump({"combos": [{k: v for k, v in c.items()
                           if k not in ("V", "v2_g1", "v2_g2", "v1")}
                          for c in combos],
               "results": results, "g3": g3, "scan": scan,
               "perm": {"n_perm": N_PERM, "n_variants": len(VARIANTS),
                        "null_median": float(np.median(null_max)),
                        "null_q95": float(np.quantile(null_max, 0.95)),
                        "p_best": p_best, "p_primary": p_prim,
                        "p_planar": p_plan, "best_variant": best_nm,
                        "best_obs": float(best_obs),
                        "planar_obs": float(plan_obs)},
               "secondary_docs10_phi2": sec,
               "key7": {a: {"loading_docs16": KEY7_LOADING[a],
                            "q2_centroid": qp[a], "q2_cb": qcb[a]}
                        for a in KEY7}},
              open(os.path.join(ROOT, "cache", "quadrupole_p4.json"), "w"),
              indent=1, default=float)
    print("\n写入 cache/quadrupole_p4.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
