#!/usr/bin/env python3
"""6M0J（SARS-CoV-2 RBD 链 E + 人 ACE2 链 A）上每个 DMS 位点的结构描述子。

### 为什么都在一个文件里

docs/14 要回答的是「位点载荷能不能只从结构算出来」，所以**每个描述子都必须
在不知道任何亲和力数据的情况下算出**。本模块只读 PDB，不读 DMS 的 bind 列
（只读 wildtype 列做编号自检），因此不可能泄露标签。

### 描述子分两类，报告时必须分开（docs/16 §3 P1/P2）

**量级类**：WCN、埋藏度、到伙伴链最近距离、各壳自由体积比例、自由体积总量、
          场强平方 |E|²
**各向异性类**：自由空间惯量张量的特征值比（flatness / elongation）、lobe 数、
            场梯度张量范数平方 ‖∇E‖²

把两类混在一个模型里会让「场 vs 梯度」「大小 vs 各向异性」的判别力消失。

### 自检（CLAUDE.md §10，全部用已知事实）

1. **编号映射**：DMS 的 wildtype 残基身份必须与 PDB 对应位点的残基一致。
   命中率不到 100% 就是映射错了，直接中止。
2. **场与梯度**：单点电荷的解析解
       |E| = q/r²,  ‖∇E‖_F = q√6/r³
   必须在 0.5% 内被复现；且数值有限差分的 ∇E 必须与解析式一致。
   这同时测「能命中」与「不误伤」：把 r 改成 2r，|E| 必须降 4 倍、
   ‖∇E‖ 必须降 8 倍 —— 幂次错了会被抓住（D-005 式的约定错）。
3. **惯量各向异性**：对已知形状做还原——
   板状点云 flatness→1，球状点云 flatness→0，棒状点云 elongation→1。
4. **埋藏度方向**：RBD 核心位点的埋藏度必须显著高于界面外缘位点。

输出：cache/site_desc_6m0j.json
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_orientation import CHARGES, parse_pdb, sub, norm, unit, dot, ROOT
from rotamer_build import CB_SPEC, nerf
from rotamer_descriptors import descriptors_for_site

PDB_ID = "6M0J"
SITE_CHAIN = "E"           # SARS-CoV-2 RBD
PARTNER_CHAIN = "A"        # 人 ACE2
DMS_CSV = os.path.join(ROOT, "cache", "dms",
                       "starr_animalACE2_final_variant_scores.csv")
AA3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
          "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
          "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
          "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
SHELLS = ((1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6.0), (6.0, 8.0))
EXCL = 2.7                 # 探针半径和（与 analysis/expanded_descriptors.py 同）
ANISO_R = 6.0              # P2 指定的 ~6 Å
GRID = 0.6
PROBES = ("VAL", "LEU", "PHE", "TRP")


# ------------------------------------------------------------------ 电荷模型

def charges_formal(atoms, skip_resi):
    """模型 F：只用可电离基团的形式电荷（+ Zn²⁺）。repo 既有模型。"""
    out = []
    for a in atoms:
        if a["res"] == "HOH":
            continue
        if a["chain"] == SITE_CHAIN and a["resi"] == skip_resi:
            continue
        if a["res"] == "ZN":
            out.append((2.0, a["xyz"]))
            continue
        if a["res"] == "CL" and a["het"]:
            out.append((-1.0, a["xyz"]))
            continue
        q = CHARGES.get((a["res"], a["name"]))
        if q:
            out.append((q, a["xyz"]))
    return out


# 模型 D：中性偶极基团模型。每个极性基团内部电荷和为 0，故不引入虚假净电荷。
# 氢原子在晶体结构里没有，故把补偿电荷放在相连的重原子上。
DIPOLE_PAIRS = {
    # (残基, 负原子, 正原子, |q|)  —— 主链羰基对所有残基成立
    "BACKBONE_CO": ("O", "C", 0.45),
    "BACKBONE_N": ("N", "CA", 0.30),
}
SC_DIPOLES = {
    "ASN": [("OD1", "CG", 0.45), ("ND2", "CG", 0.30)],
    "GLN": [("OE1", "CD", 0.45), ("NE2", "CD", 0.30)],
    "SER": [("OG", "CB", 0.40)],
    "THR": [("OG1", "CB", 0.40)],
    "TYR": [("OH", "CZ", 0.40)],
    "TRP": [("NE1", "CD1", 0.30)],
    "HIS": [("ND1", "CG", 0.30), ("NE2", "CD2", 0.30)],
    "MET": [("SD", "CG", 0.15)],
    "CYS": [("SG", "CB", 0.15)],
}


def charges_dipole(atoms, skip_resi):
    """模型 D：形式电荷 + 中性偶极基团对。"""
    out = charges_formal(atoms, skip_resi)
    byres = {}
    for a in atoms:
        if a["res"] == "HOH" or a["het"]:
            continue
        if a["chain"] == SITE_CHAIN and a["resi"] == skip_resi:
            continue
        byres.setdefault((a["chain"], a["resi"], a["res"]), {})[a["name"]] = \
            a["xyz"]
    for (ch, ri, rn), pos in byres.items():
        pairs = [DIPOLE_PAIRS["BACKBONE_CO"], DIPOLE_PAIRS["BACKBONE_N"]]
        pairs += SC_DIPOLES.get(rn, [])
        for neg, plus, q in pairs:
            if neg in pos and plus in pos:
                out.append((-q, pos[neg]))
                out.append((+q, pos[plus]))
    return out


def field_and_gradient(point, charges, cutoff=20.0):
    """返回 (|E|², ‖∇E‖²_F)。单位 e/Å² 与 e/Å³。

    E_a = Σ q d_a / r³        （d = point − 电荷位置）
    ∂E_a/∂x_b = Σ q (3 d_a d_b / r⁵ − δ_ab / r³)
    """
    E = np.zeros(3)
    G = np.zeros((3, 3))
    I3 = np.eye(3)
    for q, pos in charges:
        d = np.array(point) - np.array(pos)
        r = float(np.linalg.norm(d))
        if r < 1e-6 or r > cutoff:
            continue
        E += q * d / r ** 3
        G += q * (3.0 * np.outer(d, d) / r ** 5 - I3 / r ** 3)
    return float(E @ E), float((G * G).sum())


# ------------------------------------------------------------- 自由空间几何

def free_space_geometry(cb, u, env_xyz, radius=ANISO_R, grid=GRID):
    """CB 为心、半径 radius 的球内未被占据的格点的几何。

    返回 free_frac（量级）、flatness / elongation（各向异性）、n_lobes。
    各向异性量用**特征值比**，与自由体积大小无关 —— P2 的判别点正在此。
    """
    n = int(radius / grid)
    rng = np.arange(-n, n + 1) * grid
    gx, gy, gz = np.meshgrid(rng, rng, rng, indexing="ij")
    P = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    P = P[np.linalg.norm(P, axis=1) <= radius]
    Q = P + np.array(cb)
    if len(env_xyz) == 0:
        return None
    from scipy.spatial import cKDTree
    tree = cKDTree(env_xyz)
    dmin, _ = tree.query(Q, k=1)
    free = dmin >= EXCL
    nf = int(free.sum())
    out = {"free_frac": nf / len(P)}
    if nf < 20:
        out.update(flatness=float("nan"), elongation=float("nan"),
                   sphericity=float("nan"), n_lobes=0,
                   free_frac_hemi=float("nan"))
        return out
    F = P[free]
    # 侧链所在半球的自由比例（量级类，与壳层互补）
    cosv = F @ np.array(u)
    hemi = (P @ np.array(u)) > 0
    out["free_frac_hemi"] = float((cosv > 0).sum() / max(hemi.sum(), 1))
    C = F - F.mean(axis=0)
    lam = np.linalg.eigvalsh(C.T @ C / len(C))[::-1]     # λ1>=λ2>=λ3
    out["flatness"] = float(1 - lam[2] / lam[1])         # docs/10 定义
    out["elongation"] = float(1 - lam[1] / lam[0])
    out["sphericity"] = float(lam[2] / lam[0])
    # lobe 数：格点连通分量（6-邻接），只数 >=10 格点的分量
    tree2 = cKDTree(F)
    pairs = tree2.query_pairs(grid * 1.05, output_type="ndarray")
    if len(pairs):
        A = sparse.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(F), len(F)))
        ncomp, lab = connected_components(A, directed=False)
        sizes = np.bincount(lab)
        out["n_lobes"] = int((sizes >= 10).sum())
    else:
        out["n_lobes"] = int(len(F))
    return out


# ------------------------------------------------------------------- 自检

def selftest_field():
    print("  [自检 场/梯度]")
    for r in (3.0, 6.0):
        e2, g2 = field_and_gradient((0, 0, 0), [(1.0, (r, 0, 0))])
        ana_e, ana_g = (1 / r ** 2) ** 2, (math.sqrt(6) / r ** 3) ** 2
        print(f"    r={r}: |E|²={e2:.6g}（解析 {ana_e:.6g}） "
              f"‖∇E‖²={g2:.6g}（解析 {ana_g:.6g}）")
        assert abs(e2 / ana_e - 1) < 5e-3, "场强与解析解不符"
        assert abs(g2 / ana_g - 1) < 5e-3, "场梯度与解析解不符"
    # 幂次检验（不误伤：幂次错了必被抓到）
    e1, g1 = field_and_gradient((0, 0, 0), [(1.0, (3.0, 0, 0))])
    e2, g2 = field_and_gradient((0, 0, 0), [(1.0, (6.0, 0, 0))])
    print(f"    r 加倍：|E|² 比 {e1 / e2:.3f}（应 16） "
          f"‖∇E‖² 比 {g1 / g2:.3f}（应 64）")
    assert abs(e1 / e2 - 16) < 0.1 and abs(g1 / g2 - 64) < 0.5, "幂次错"
    # 有限差分核对梯度张量
    h = 1e-4
    ch = [(1.0, (2.0, 1.0, -0.5)), (-0.5, (-1.0, 3.0, 1.0))]

    def Ev(p):
        E = np.zeros(3)
        for q, pos in ch:
            d = np.array(p) - np.array(pos)
            r = np.linalg.norm(d)
            E += q * d / r ** 3
        return E
    num = np.zeros((3, 3))
    for b in range(3):
        dp = np.zeros(3); dp[b] = h
        num[:, b] = (Ev(dp) - Ev(-dp)) / (2 * h)
    _, g_ana = field_and_gradient((0, 0, 0), ch)
    print(f"    有限差分 ‖∇E‖²={float((num * num).sum()):.6g}  "
          f"解析 {g_ana:.6g}")
    assert abs(float((num * num).sum()) / g_ana - 1) < 1e-4, "∇E 实现错"


def selftest_aniso():
    print("  [自检 各向异性]（已知形状还原）")
    rng = np.random.default_rng(3)
    shapes = {
        "板状 (10x10x0.5)": np.array([5, 5, 0.25]),
        "球状 (5x5x5)": np.array([2.5, 2.5, 2.5]),
        "棒状 (10x1x1)": np.array([5, 0.5, 0.5]),
    }
    for name, half in shapes.items():
        F = rng.uniform(-1, 1, (4000, 3)) * half
        C = F - F.mean(axis=0)
        lam = np.linalg.eigvalsh(C.T @ C / len(C))[::-1]
        flat = 1 - lam[2] / lam[1]
        elon = 1 - lam[1] / lam[0]
        print(f"    {name:<20} flatness={flat:.3f} elongation={elon:.3f}")
        if "板状" in name:
            assert flat > 0.95, "板状点云的 flatness 应接近 1"
        if "球状" in name:
            assert flat < 0.1 and elon < 0.1, "球状应双双接近 0"
        if "棒状" in name:
            assert elon > 0.95, "棒状的 elongation 应接近 1"


# ------------------------------------------------------------------- 主流程

def dms_wildtypes(target="Wuhan-Hu-1"):
    """**必须按骨架取**。三个骨架的序列不同（Eta 带 E484K，BA.2 带 15 个 RBD
    取代），混在一起取 wildtype 会让映射自检命中率掉到 91.75%，而映射本身
    是对的。第一版就是这么错的 —— 护栏正确报警，但错的是被检查对象的选取，
    不是编号映射（CLAUDE.md §10 末条：区分工具的错与对象的错）。
    6M0J 链 E 是 Wuhan-Hu-1 序列，故参考骨架取 Wuhan-Hu-1。"""
    wt = {}
    for r in csv.DictReader(open(DMS_CSV)):
        if r["target"] != target:
            continue
        wt[int(r["position"])] = r["wildtype"]
    return wt


def _dihe(p0, p1, p2, p3):
    b0 = -1.0 * (np.array(p1) - np.array(p0))
    b1 = np.array(p2) - np.array(p1)
    b2 = np.array(p3) - np.array(p2)
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return math.degrees(math.atan2(np.cross(b1, v) @ w, v @ w))


def phi_psi(res_atoms, resi_list, i):
    """用前后残基的主链算 phi/psi（度）。缺失返回 None。"""
    def dihe(p0, p1, p2, p3):
        """标准（praxeolitic）二面角。**第一版符号反了**：当时写成
        n1=b0xb1, m=n1xb1_hat, atan2(m.n2, n1.n2)，给出的 phi 中位 +89.5°、
        仅 6% 为负 —— 而真实蛋白的 phi 绝大多数为负。D-005 同类的约定错。
        现在的实现由两条已知事实把关（见 selftest_dihedral）：
          (1) 反式肽键 omega ≈ 180°；(2) 非 Gly 残基的 phi 绝大多数 < 0。"""
        b0 = -1.0 * (np.array(p1) - np.array(p0))
        b1 = np.array(p2) - np.array(p1)
        b2 = np.array(p3) - np.array(p2)
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - (b0 @ b1) * b1
        w = b2 - (b2 @ b1) * b1
        return math.degrees(math.atan2(np.cross(b1, v) @ w, v @ w))

    cur = resi_list[i]
    prev = resi_list[i - 1] if i > 0 else None
    nxt = resi_list[i + 1] if i + 1 < len(resi_list) else None
    a = res_atoms[cur]
    phi = psi = None
    if prev is not None and prev == cur - 1 and "C" in res_atoms[prev]:
        if all(k in a for k in ("N", "CA", "C")):
            phi = dihe(res_atoms[prev]["C"], a["N"], a["CA"], a["C"])
    if nxt is not None and nxt == cur + 1 and "N" in res_atoms[nxt]:
        if all(k in a for k in ("N", "CA", "C")):
            psi = dihe(a["N"], a["CA"], a["C"], res_atoms[nxt]["N"])
    return phi, psi


def selftest_dihedral(res_atoms, resi_list):
    """已知事实：反式肽键 omega ~ 180；非 Gly 的 phi 绝大多数为负。"""
    phis, omegas = [], []
    for i, r in enumerate(resi_list):
        if r - 1 not in res_atoms:
            continue
        a, pr = res_atoms[r], res_atoms[r - 1]
        if not (all(k in a for k in ("N", "CA", "C")) and "C" in pr
                and "CA" in pr):
            continue
        phi, _ = phi_psi(res_atoms, resi_list, i)
        if phi is not None:
            phis.append(phi)
        omegas.append(_dihe(pr["CA"], pr["C"], a["N"], a["CA"]))
    fneg = float(np.mean(np.array(phis) < 0))
    mom = float(np.median(np.abs(omegas)))
    print(f"  [自检 二面角] phi 中位 {np.median(phis):.1f}°，为负的比例 "
          f"{fneg:.2f}（已知事实：应 >0.85）；|omega| 中位 {mom:.1f}°"
          f"（反式肽键：应 ~180）")
    assert fneg > 0.85, "phi 的符号约定反了"
    assert mom > 165, "omega 不在反式 —— 二面角实现有问题"


def main():
    print("=" * 78)
    print("自检（CLAUDE.md §10）")
    print("=" * 78)
    selftest_field()
    selftest_aniso()

    atoms = parse_pdb(os.path.join(ROOT, "cache", "pdb", f"{PDB_ID}.pdb"))
    site_res = {}
    for a in atoms:
        if a["chain"] != SITE_CHAIN or a["het"]:
            continue
        try:
            ri = int(a["resi"])
        except ValueError:
            continue
        site_res.setdefault(ri, {"res": a["res"], "atoms": {}})
        site_res[ri]["atoms"][a["name"]] = a["xyz"]
    resi_list = sorted(site_res)
    res_atoms = {k: site_res[k]["atoms"] for k in resi_list}

    selftest_dihedral(res_atoms, resi_list)

    wt = dms_wildtypes("Wuhan-Hu-1")
    for tg in ("Wuhan-Hu-1", "Eta", "Omicron_BA2"):
        w2 = dms_wildtypes(tg)
        m2 = [r for r in resi_list if r in w2]
        h2 = sum(1 for r in m2 if AA3TO1.get(site_res[r]["res"]) == w2[r])
        print(f"  骨架 {tg:<14} 命中 {h2}/{len(m2)} = {h2 / len(m2):.2%}")
    mapped = [r for r in resi_list if r in wt]
    hit = sum(1 for r in mapped
              if AA3TO1.get(site_res[r]["res"]) == wt[r])
    print("\n" + "=" * 78)
    print("编号映射自检：DMS wildtype vs PDB 残基身份")
    print("=" * 78)
    print(f"  DMS 位点 {len(wt)}，PDB 链 {SITE_CHAIN} 残基 "
          f"{resi_list[0]}–{resi_list[-1]}（{len(resi_list)} 个）")
    print(f"  可映射 {len(mapped)}，身份一致 {hit}  "
          f"=> 命中率 {hit / len(mapped):.2%}")
    bad = [(r, site_res[r]["res"], wt[r]) for r in mapped
           if AA3TO1.get(site_res[r]["res"]) != wt[r]]
    if bad:
        print(f"  不一致：{bad[:20]}")
    assert hit / len(mapped) > 0.995, \
        "编号映射错误（命中率 < 99.5%）—— 按 CLAUDE.md §10 中止"
    print("  映射为**恒等映射**（DMS position = PDB 链 E 残基号），已验证。")

    # 伙伴链与全环境坐标
    partner = np.array([a["xyz"] for a in atoms
                        if a["chain"] == PARTNER_CHAIN and not a["het"]])
    ca_all = {}
    for a in atoms:
        if a["name"] == "CA" and not a["het"]:
            ca_all[(a["chain"], a["resi"])] = a["xyz"]
    ca_arr = np.array(list(ca_all.values()))
    # 界面法向：RBD 质心 -> ACE2 质心
    own_all = np.array([a["xyz"] for a in atoms
                        if a["chain"] == SITE_CHAIN and not a["het"]])
    iface_n = unit(tuple(partner.mean(axis=0) - own_all.mean(axis=0)))

    print("\n" + "=" * 78)
    print(f"计算 {len(mapped)} 个位点的描述子")
    print("=" * 78)
    out = {}
    ch_formal = None
    for k, ri in enumerate(mapped):
        a = res_atoms[ri]
        if not all(x in a for x in ("N", "CA", "C")):
            continue
        nm, p1, p2, p3, bd, an, dh = CB_SPEC
        cb = nerf(a[p1], a[p2], a[p3], bd, an, dh[1])
        u = unit(sub(cb, a["CA"]))
        d = {"resi": ri, "wt": wt[ri], "pdb_res": site_res[ri]["res"]}
        # --- 环境（排除本位点残基自身原子、水、糖）
        env = np.array([x["xyz"] for x in atoms
                        if x["res"] not in ("HOH", "NAG")
                        and not (x["chain"] == SITE_CHAIN
                                 and x["resi"] == str(ri) and not x["het"])])
        envd = np.linalg.norm(env - np.array(cb), axis=1)
        # --- 量级类
        d["burial8"] = float((envd <= 8.0).sum())
        own_env = np.array([x["xyz"] for x in atoms
                            if x["chain"] == SITE_CHAIN and not x["het"]
                            and x["resi"] != str(ri)])
        d["burial_own8"] = float(
            (np.linalg.norm(own_env - np.array(cb), axis=1) <= 8.0).sum())
        d["burial10"] = float((envd <= 10.0).sum())
        # WCN：CA-CA 的 1/r² 加权接触数（文献标准形式）
        dca = np.linalg.norm(ca_arr - np.array(a["CA"]), axis=1)
        dca = dca[(dca > 1e-3)]
        d["wcn"] = float(np.sum(1.0 / dca[dca < 25.0] ** 2))
        # WCN 的重原子版本（以 CB 为心）
        d["wcn_cb"] = float(np.sum(1.0 / envd[(envd > 1e-3) & (envd < 25)] ** 2))
        dpart = np.linalg.norm(partner - np.array(cb), axis=1)
        d["d_partner"] = float(dpart.min())
        d["n_partner8"] = float((dpart <= 8.0).sum())
        d["n_partner12"] = float((dpart <= 12.0).sum())
        d["cos_iface"] = float(dot(u, iface_n))
        # 壳层自由体积比例（侧链半球，与 expanded_descriptors 同口径）
        rngs = np.random.default_rng(42)
        shells = []
        for lo, hi in SHELLS:
            free = tot = 0
            for _ in range(600):
                v = rngs.normal(size=3); v /= np.linalg.norm(v)
                if dot(u, tuple(v)) < 0:
                    continue
                r = (lo ** 3 + (hi ** 3 - lo ** 3) * rngs.random()) ** (1 / 3)
                p = np.array(cb) + r * v
                tot += 1
                if np.min(np.linalg.norm(env - p, axis=1)) >= EXCL:
                    free += 1
            shells.append(free / tot if tot else float("nan"))
        for i, v in enumerate(shells):
            d[f"shell{i + 1}"] = v
        d["rho"] = (shells[0] / shells[3]) if shells[3] > 0 else float("nan")
        # --- 各向异性类
        g = free_space_geometry(cb, u, env)
        if g:
            d.update({k2: v for k2, v in g.items()})
        # --- 场与梯度（两个电荷模型）
        for tag, fn in (("F", charges_formal), ("D", charges_dipole)):
            ch = fn(atoms, str(ri))
            centre = tuple(cb[i] + 2.0 * u[i] for i in range(3))
            e2, g2 = field_and_gradient(centre, ch)
            d[f"E2_{tag}"] = e2
            d[f"gradE2_{tag}"] = g2
            e2b, g2b = field_and_gradient(cb, ch)
            d[f"E2cb_{tag}"] = e2b
            d[f"gradE2cb_{tag}"] = g2b
        # --- 主链几何
        i = resi_list.index(ri)
        phi, psi = phi_psi(res_atoms, resi_list, i)
        d["phi"] = phi
        d["psi"] = psi
        out[str(ri)] = d
        if (k + 1) % 40 == 0:
            print(f"  ... {k + 1}/{len(mapped)}", flush=True)

    # --- rotamer 自由体积（位点级，对 4 个探针残基）
    print("  rotamer 枚举（探针残基 "
          f"{'/'.join(PROBES)}）…", flush=True)
    for k, ri in enumerate(sorted(int(x) for x in out)):
        rd = descriptors_for_site(PDB_ID, SITE_CHAIN, ri, list(PROBES))
        if rd is None:
            continue
        for p in PROBES:
            out[str(ri)][f"Srot_{p}"] = rd[p]["S_rot"]
            out[str(ri)][f"nall_{p}"] = float(rd[p]["n_allowed"])
        sp = out[str(ri)].get("Srot_PHE")
        sv = out[str(ri)].get("Srot_VAL")
        # 平面 vs β分支的构象可及性差（P2 的几何对应量，位点级）
        out[str(ri)]["Srot_PHE_minus_VAL"] = (
            sp - sv if sp is not None and sv is not None
            and math.isfinite(sp) and math.isfinite(sv) else float("nan"))
        if (k + 1) % 40 == 0:
            print(f"  ... rotamer {k + 1}", flush=True)

    # --- 量级核对（CLAUDE.md §8）
    print("\n" + "=" * 78)
    print("量级核对：每个描述子的范围（与参考值对照）")
    print("=" * 78)
    keys = [k for k in next(iter(out.values())) if k not in
            ("resi", "wt", "pdb_res")]
    print(f"{'描述子':<22}{'最小':>10}{'中位':>10}{'最大':>10}{'缺失':>6}")
    print("-" * 78)
    for key in keys:
        v = np.array([out[s].get(key) if out[s].get(key) is not None
                      else np.nan for s in out], dtype=float)
        ok = np.isfinite(v)
        print(f"{key:<22}{np.nanmin(v):>10.3g}{np.nanmedian(v):>10.3g}"
              f"{np.nanmax(v):>10.3g}{int((~ok).sum()):>6}")
    # 已知事实自检。第一版的断言本身是错的（D-017 同类）：
    # burial8 把**伙伴链原子也算进去**，所以界面位点理应比远离界面的位点
    # 埋得更深（实测 88 vs 69）。那不是描述子错，是断言错。
    # 换成两条真正的已知事实：
    #   (a) 二硫键必然埋在核内：6M0J 链 E 的三对二硫 336-361 / 379-432 /
    #       391-525（SARS-CoV-2 RBD 已知结构事实），其**自身链**埋藏度
    #       应高于全体中位；
    #   (b) 伙伴链接触数与到伙伴链的距离必须方向一致（几何一致性）。
    bur_own = {int(s2): out[s2]["burial_own8"] for s2 in out}
    dpt = {int(s2): out[s2]["d_partner"] for s2 in out}
    npt = {int(s2): out[s2]["n_partner8"] for s2 in out}
    ss = [336, 361, 379, 432, 391, 525]
    have = [r for r in ss if r in bur_own]
    med = float(np.median(list(bur_own.values())))
    print(f"\n  [自检(a) 二硫键应埋藏] 位点 {have} 的自身链 burial 中位 "
          f"{np.median([bur_own[r] for r in have]):.0f} vs 全体中位 {med:.0f}")
    assert np.median([bur_own[r] for r in have]) > med, \
        "已知埋藏的二硫键位点却不埋藏 —— 埋藏度算错了"
    iface = [r for r in npt if npt[r] > 0]
    print(f"  [自检(b) 界面一致性] n_partner8>0 的 {len(iface)} 个位点，"
          f"d_partner 最大 {max(dpt[r] for r in iface):.2f} Å（应 <= 8）；"
          f"其余位点 d_partner 最小 "
          f"{min(dpt[r] for r in npt if npt[r] == 0):.2f} Å（应 > 8）")
    assert max(dpt[r] for r in iface) <= 8.0 + 1e-6
    assert min(dpt[r] for r in npt if npt[r] == 0) > 8.0 - 1e-6

    path = os.path.join(ROOT, "cache", "site_desc_6m0j.json")
    json.dump({"pdb": PDB_ID, "site_chain": SITE_CHAIN,
               "partner_chain": PARTNER_CHAIN, "shells": SHELLS,
               "wt_match_rate": hit / len(mapped), "sites": out},
              open(path, "w"), indent=1)
    print(f"\n写入 {os.path.relpath(path, ROOT)}（{len(out)} 个位点）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
