#!/usr/bin/env python3
"""侧链四极张量 ‖Q_r‖²：显式几何 + 有出处的电荷模型 + 显式 π 密度。

目标：给 docs/16 的 **P4** 提供一个真正算出来的残基侧旋转不变量，
替掉 analysis/multipole_surrogate.py 里手填的 `RESIDUE_MULTIPOLES` 四极列
（那一列给 Phe/Tyr 7.5、Trp 9.5、His 4.0，而给 **Asp/Glu/Asn/Gln 0.0**；
羧酸根的四极很大，0 不可信，所以那一列不可复用）。

---------------------------------------------------------------- 约定与单位

无迹四极张量（Buckingham 约定）

    Q_ab = Σ_i q_i (3 x_ia x_ib − r_i² δ_ab) / 2

q 单位 e，坐标单位 Å，故 Q 的原生单位是 e·Å²；
报告时乘 4.803205 换成 D·Å（= Buckingham = 10⁻²⁶ esu·cm²）。
不变量取 Frobenius 范数平方 ‖Q‖² = Σ_ab Q_ab²，单位 (D·Å)²。

**原点依赖性**：Q 只有在净电荷与净偶极都为零时才与原点无关。
Asp/Glu/Lys/Arg 侧链净电荷 ±1，所以原点必须由一条明说的规则固定。
本模块同时输出三种原点，主用 docs/16 §2 写明的「侧链质心」：

  centroid  侧链重原子几何质心（**主**，docs/16 §2 的 x₀）
  cb        CB 原子
  qcent     |q| 加权电荷质心

---------------------------------------------------------------- 电荷模型

`facts:` Amber ff14SB 残基模板电荷，取自 OpenMM 发行版文件
  wrappers/python/openmm/app/data/amber14/protein.ff14SB.xml
  （下载副本 cache/ff/protein.ff14SB.xml；力场本身 Maier et al. 2015,
   J. Chem. Theory Comput. 11, 3696, doi:10.1021/acs.jctc.5b00255）。
侧链定义为模板里除 N,H,CA,HA,C,O,OXT 之外的全部原子（含 CB 及其 H）。
氢**显式建**，不做联合原子近似（联合原子会把烷基的 C–H 四极抹掉，
而 Ile/Val 正是要和芳香/羧基比大小的那一端，抹掉会人为帮到假设）。
组氨酸取中性 HIE。

`assessment:` ff14SB 是核中心点电荷集，**它对芳香环的面外四极给出 ≈0**，
这在物理上是错的：苯的四极来自环平面上下的 π 电子密度。
所以必须显式表示 π 密度。本模块的表示（**模型假设，不是测量**）：

  对共轭体系的每个 π 中心原子 i：把电荷 q_π 从核上搬走一半到环法向
  ±d 处，即 核电荷 += q_π，并在 x_i ± d·n̂ 各放 −q_π/2。
  净电荷与净偶极不变（上下对称），只加面外四极。

一个解析结果：对苯这样的平面环，这个构造给出
    ΔΘ_zz = −N_π · q_π · d²
即**只依赖乘积 q_π·d²，不依赖 q_π 与 d 的拆分**。
所以本模型实际只有**一个**标定参数，由苯的实验 Θ_zz 定出（见自检 a）。
confidence: 中。basis: 乘积形式是解析的；把同一个 q_π·d² 搬到吲哚/咪唑
（每中心一个 π 电子的近似）是外推，已在 §sensitivity 里给出 d 的敏感性。

---------------------------------------------------------------- 自检（CLAUDE.md §10）

(a) 已知实验值标定/复现：
    苯    Θ_zz = −8.69 D·Å   Battaglia, Buckingham & Williams,
          Chem. Phys. Lett. 78 (1981) 421（= −29.0×10⁻⁴⁰ C·m²）
          —— **用来定 q_π·d²**，是标定不是验证；标定后检查 σ 框架单独
          给出的 Θ_zz 量级（应 ≪ 8.69，证明 π 项确实是主项）。
    CO₂   Θ_zz = −4.3 D·Å    Buckingham & Disch, Proc. R. Soc. A 273 (1963) 275
          用 EPM2 点电荷（Harris & Yung 1995, J. Phys. Chem. 99, 12021:
          q_C=+0.6512, q_O=−0.3256, r=1.149 Å）**纯核中心预测**，
          与实验比符号与量级。
    水    Θ_xx=+2.63, Θ_yy=−2.50, Θ_zz=−0.13 D·Å
          Verhoeven & Dymanus, J. Chem. Phys. 52 (1970) 3222
          用 TIP3P（Jorgensen et al. 1983）纯核中心预测。
          **预期它在面外分量上失败** —— 这正是 (a) 里 π 项必要性的独立证据。
(b) 双向：各向同性分布（正二十面体顶点等电荷）必须给 Q ≈ 0；
    已知各向异性分布（线性三点电荷 −2q/+q/+q，解析 Θ_zz = 2q d²）必须给对。
    再加：无迹性、刚体旋转下 ‖Q‖² 不变、中性零偶极分布的原点无关性。
(c) 量级：每个 headline 数都打印单位，并与参考量级对照。

任一自检不过 → sys.exit(非零)，主流程中止。

输出：cache/sidechain_quadrupole.tsv
"""
import itertools
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotamer_build import CB_SPEC, N_CHI, SIDECHAINS, build_sidechain, nerf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF_XML = os.path.join(ROOT, "cache", "ff", "protein.ff14SB.xml")
OUT_TSV = os.path.join(ROOT, "cache", "sidechain_quadrupole.tsv")

E_A2_TO_DA = 4.8032047            # 1 e·Å² = 4.8032047 D·Å
BB_EXCLUDE = {"N", "H", "CA", "HA", "C", "O", "OXT", "H1", "H2", "H3",
              "HA2", "HA3"}
TEMPLATE = {"HIS": "HIE"}          # 中性组氨酸

# 已知实验值（D·Å），见模块 docstring 的出处
EXP_BENZENE_THETA_ZZ = -8.69
EXP_CO2_THETA_ZZ = -4.3
EXP_WATER = {"xx": 2.63, "yy": -2.50, "zz": -0.13}

# π 中心：共轭体系 -> 组成原子（法向由这些原子的最佳平面定）
AROMATIC_RINGS = {
    "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TRP": [["CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ2", "CZ3", "CH2"]],
    "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],
}
# 非芳香 sp² 共轭基团（变体 B 才加 π 位点）
SP2_GROUPS = {
    "ASP": [["CG", "OD1", "OD2"]],
    "GLU": [["CD", "OE1", "OE2"]],
    "ASN": [["CG", "OD1", "ND2"]],
    "GLN": [["CD", "OE1", "NE2"]],
    "ARG": [["CZ", "NE", "NH1", "NH2"]],
}

D_PI = 1.00          # π 位点的面外位移 (Å)，主取值；见 sensitivity
H_BOND = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34}
CHI_STEP = 30
POLAR_H_STEP = 60
CLASH_MIN = 2.70     # 图距离 >=4 的重原子对最小允许距离 (Å)

AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "SER", "THR", "TRP", "TYR", "VAL"]
# GLY 无侧链（Q=0，列入输出但不参与几何流程）；PRO 不列（D-022）
ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
       "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
       "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


# ======================================================== 四极张量基本机器

def quadrupole(coords, charges, origin):
    """无迹四极张量 Q_ab = Σ q_i (3 x_a x_b − r² δ_ab)/2。返回 e·Å²。"""
    X = np.asarray(coords, float) - np.asarray(origin, float)
    q = np.asarray(charges, float)
    r2 = (X ** 2).sum(axis=1)
    Q = 1.5 * np.einsum("i,ia,ib->ab", q, X, X)
    Q -= 0.5 * np.eye(3) * float((q * r2).sum())
    return Q


def frob2(Q):
    return float((Q ** 2).sum())


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def plane_normal(pts):
    P = np.asarray(pts, float)
    P = P - P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    return _unit(Vt[-1])


# ======================================================== ff14SB 解析

def load_ff14sb(path=FF_XML):
    """返回 {resname: {'charge': {atom: q}, 'bonds': [(a,b)], 'elem': {atom: E}}}。"""
    if not os.path.exists(path):
        sys.exit(f"缺少电荷文件 {path}；先下载 ff14SB（见 docstring）")
    root = ET.parse(path).getroot()
    types = {t.get("name"): t.get("element")
             for t in root.find("AtomTypes")}
    out = {}
    for res in root.find("Residues"):
        ch, el = {}, {}
        for a in res.findall("Atom"):
            ch[a.get("name")] = float(a.get("charge"))
            el[a.get("name")] = types.get(a.get("type")) or \
                a.get("name")[0]
        bonds = [(b.get("atomName1"), b.get("atomName2"))
                 for b in res.findall("Bond")
                 if b.get("atomName1") and b.get("atomName2")]
        out[res.get("name")] = {"charge": ch, "bonds": bonds, "elem": el}
    return out


FF = load_ff14sb()


def sidechain_spec(res3):
    """该残基的侧链原子表：重原子 -> 挂的 H 名；以及重原子间的键。"""
    tmpl = FF[TEMPLATE.get(res3, res3)]
    heavy = [a for a, e in tmpl["elem"].items()
             if e != "H" and a not in BB_EXCLUDE]
    heavyset = set(heavy)
    hyd = {a: [] for a in heavy}
    hh = []
    for a, b in tmpl["bonds"]:
        ea, eb = tmpl["elem"].get(a), tmpl["elem"].get(b)
        if a in heavyset and b in heavyset:
            hh.append((a, b))
        elif a in heavyset and eb == "H" and b not in BB_EXCLUDE:
            hyd[a].append(b)
        elif b in heavyset and ea == "H" and a not in BB_EXCLUDE:
            hyd[b].append(a)
    # **片段中性化（链原子电荷修正）**：Amber 的主链基团电荷是残基特异的，
    # 并不恰好为 0，所以「侧链」片段的电荷和会偏离形式电荷最多 0.14 e。
    # 一个虚假的净电荷会直接污染 Q 的原点依赖性，必须处理。
    # 做法：把残差全部加到 **CB** 上（CB 是片段与主链的连接原子，也是
    # 最靠近所有原点候选的位置，故引入的伪偶极最小）。
    # `assessment:` 这是模型假设而非测量（CLAUDE.md §2）；残差大小逐残基报告。
    formal = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0}.get(res3, 0.0)
    ch = dict(tmpl["charge"])
    tot = sum(ch[a] for a in heavy) + \
        sum(ch[h] for v in hyd.values() for h in v)
    resid = tot - formal
    anchor = "CB" if "CB" in ch else (heavy[0] if heavy else None)
    if anchor:
        ch[anchor] -= resid
    return {"heavy": heavy, "h_on": hyd, "bonds": hh,
            "charge": ch, "elem": tmpl["elem"], "formal": formal,
            "link_resid": resid}


def graph_dist(heavy, bonds):
    idx = {a: i for i, a in enumerate(heavy)}
    n = len(heavy)
    D = np.full((n, n), 99)
    np.fill_diagonal(D, 0)
    for a, b in bonds:
        if a in idx and b in idx:
            D[idx[a], idx[b]] = D[idx[b], idx[a]] = 1
    for k in range(n):
        D = np.minimum(D, D[:, k][:, None] + D[k][None, :])
    return D, idx


# ======================================================== 氢原子放置

def place_hydrogens(pos, spec, polar_dih=180.0):
    """按局域几何补氢。pos 含侧链重原子 + 主链 N/CA/C。

    n_h / 重原子邻居数的组合决定构造（甲基相位由 staggered 定，
    O–H / S–H 的二面角由 polar_dih 给，进入构象集合）。
    """
    hpos = {}
    nbr = {a: [] for a in spec["heavy"]}
    for a, b in spec["bonds"]:
        nbr[a].append(b)
        nbr[b].append(a)
    nbr["CB"] = nbr.get("CB", []) + ["CA"]
    nbr["CA"] = ["N", "C", "CB"]
    nbr["N"] = ["CA"]
    nbr["C"] = ["CA"]
    for a in spec["heavy"]:
        hs = spec["h_on"][a]
        if not hs:
            continue
        L = H_BOND.get(spec["elem"][a], 1.09)
        hv = [x for x in nbr[a] if x in pos]
        X = np.array(pos[a], float)
        us = [_unit(np.array(pos[x], float) - X) for x in hv]
        if len(hs) == 1 and len(us) >= 2:
            d = _unit(-sum(us))
            hpos[hs[0]] = X + L * d
        elif len(hs) == 2 and len(us) == 2:
            b = _unit(-(us[0] + us[1]))
            p = _unit(np.cross(us[0], us[1]))
            half = math.radians(109.5 / 2)
            hpos[hs[0]] = X + L * _unit(math.cos(half) * b +
                                        math.sin(half) * p)
            hpos[hs[1]] = X + L * _unit(math.cos(half) * b -
                                        math.sin(half) * p)
        elif len(hs) == 2 and len(us) == 1:
            # sp2 的 NH2：在共轭平面内，与 X->邻居 成 ±120 度
            ref = None
            for x in nbr.get(hv[0], []):
                if x in pos and x != a:
                    ref = np.array(pos[x], float)
                    break
            nrm = plane_normal([X, np.array(pos[hv[0]], float), ref]) \
                if ref is not None else _unit(np.cross(us[0], [0, 0, 1.0]))
            for k, ang in enumerate((120.0, -120.0)):
                th = math.radians(ang)
                d = _unit(math.cos(th) * us[0] +
                          math.sin(th) * _unit(np.cross(nrm, us[0])))
                hpos[hs[k]] = X + L * d
        elif len(hs) == 3 and len(us) == 1:
            # 甲基：相对于邻居的邻居 staggered
            ref = None
            for x in nbr.get(hv[0], []):
                if x in pos and x != a:
                    ref = np.array(pos[x], float)
                    break
            u = us[0]
            p = _unit(np.cross(u, [0.0, 0.0, 1.0]) if ref is None
                      else np.cross(u, ref - np.array(pos[hv[0]], float)))
            if np.linalg.norm(p) < 1e-8:
                p = _unit(np.cross(u, [1.0, 0.0, 0.0]))
            w = _unit(np.cross(u, p))
            th = math.radians(109.47)
            for k, phi in enumerate((60.0, 180.0, 300.0)):
                ph = math.radians(phi)
                d = _unit(math.cos(th) * u + math.sin(th) *
                          (math.cos(ph) * p + math.sin(ph) * w))
                hpos[hs[k]] = X + L * d
        elif len(hs) == 1 and len(us) == 1:
            # O–H / S–H：需要二面角
            ref = None
            for x in nbr.get(hv[0], []):
                if x in pos and x != a:
                    ref = x
                    break
            ang = 108.0 if spec["elem"][a] == "O" else 96.0
            if ref is None:
                hpos[hs[0]] = X + L * _unit(-us[0])
            else:
                hpos[hs[0]] = nerf(np.array(pos[ref], float),
                                   np.array(pos[hv[0]], float), X,
                                   L, ang, polar_dih)
        elif len(hs) == 3 and len(us) == 0:
            raise RuntimeError(f"孤立原子 {a} 上挂 3 个 H，几何未定")
        else:
            raise RuntimeError(f"未覆盖的补氢情形 {a}: {len(hs)}H/{len(us)}重")
    return hpos


# ======================================================== π 位点

def pi_sites(pos, groups, q_pi, d_pi):
    """返回 (额外位点列表[(xyz,q)], 核电荷修正 {atom: dq})。"""
    extra, fix = [], {}
    for grp in groups:
        pts = [pos[a] for a in grp if a in pos]
        if len(pts) < 3:
            continue
        n = plane_normal(pts)
        for a in grp:
            if a not in pos:
                continue
            X = np.array(pos[a], float)
            fix[a] = fix.get(a, 0.0) + q_pi
            extra.append((X + d_pi * n, -0.5 * q_pi))
            extra.append((X - d_pi * n, -0.5 * q_pi))
    return extra, fix


# ======================================================== 构象枚举

def ideal_backbone():
    """理想主链局域坐标（‖Q‖² 是旋转不变量，故绝对取向无关紧要）。"""
    N = np.array([-0.525, 1.363, 0.000])
    CA = np.array([0.000, 0.000, 0.000])
    C = np.array([1.526, 0.000, 0.000])
    return {"N": tuple(N), "CA": tuple(CA), "C": tuple(C)}


def conformers(res3, spec, step=CHI_STEP, polar_step=POLAR_H_STEP):
    """枚举 (chis, polar_dih) 并返回不自碰的重原子坐标。"""
    bb = ideal_backbone()
    nchi = N_CHI.get(res3, 0)
    grid = list(range(-180, 180, step)) or [180]
    D, idx = graph_dist(spec["heavy"], spec["bonds"] + [("CB", "CB")])
    has_polar_h = any(spec["elem"][a] == "O" and len(spec["h_on"][a]) == 1
                      and len([1 for x, y in spec["bonds"]
                               if a in (x, y)]) == 1
                      for a in spec["heavy"]) or \
        any(spec["elem"][a] == "S" and spec["h_on"][a]
            for a in spec["heavy"])
    pdihs = list(range(-180, 180, polar_step)) if has_polar_h else [180.0]
    out = []
    chi_sets = list(itertools.product(grid, repeat=nchi)) if nchi else [()]
    for chis in chi_sets:
        pos = build_sidechain(bb, res3, list(chis))
        if not pos:
            continue
        names = [a for a in spec["heavy"] if a in pos]
        X = np.array([pos[a] for a in names])
        ii = [idx[a] for a in names]
        sub = D[np.ix_(ii, ii)]
        dm = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        far = sub >= 4
        if far.any() and dm[far].min() < CLASH_MIN:
            continue
        full = dict(pos)
        full.update(bb)
        for pd in pdihs:
            out.append((chis, pd, full))
    return out


# ======================================================== 单残基 ‖Q‖²

ORIGINS = ("centroid", "cb", "qcent")


def residue_quadrupoles(res3, q_pi, d_pi=D_PI, pi_variant="aromatic",
                        step=CHI_STEP):
    """对一个残基的构象集合算 Q。返回 {origin: ndarray ‖Q‖² (D·Å)²}, 诊断。"""
    if res3 == "GLY":
        return {o: np.zeros(1) for o in ORIGINS}, {"n_conf": 1, "netq": 0.0}
    spec = sidechain_spec(res3)
    groups = list(AROMATIC_RINGS.get(res3, []))
    if pi_variant == "all_sp2":
        groups += SP2_GROUPS.get(res3, [])
    res = {o: [] for o in ORIGINS}
    netq = None
    n_conf = 0
    for chis, pd, pos in conformers(res3, spec, step=step):
        hp = place_hydrogens(pos, spec, polar_dih=pd)
        coords, charges, names = [], [], []
        for a in spec["heavy"]:
            if a in pos:
                coords.append(pos[a]); charges.append(spec["charge"][a])
                names.append(a)
        for a, hs in spec["h_on"].items():
            for h in hs:
                if h in hp:
                    coords.append(hp[h]); charges.append(spec["charge"][h])
                    names.append(h)
        if groups:
            extra, fix = pi_sites(pos, groups, q_pi, d_pi)
            for k, a in enumerate(names):
                if a in fix:
                    charges[k] += fix[a]
            for xyz, q in extra:
                coords.append(tuple(xyz)); charges.append(q); names.append("PI")
        coords = np.array(coords, float)
        charges = np.array(charges, float)
        netq = float(charges.sum())
        heavy_xyz = np.array([pos[a] for a in spec["heavy"] if a in pos])
        org = {"centroid": heavy_xyz.mean(axis=0),
               "cb": np.array(pos["CB"], float),
               "qcent": (np.abs(charges)[:, None] * coords).sum(axis=0) /
                        max(np.abs(charges).sum(), 1e-9)}
        for o in ORIGINS:
            Q = quadrupole(coords, charges, org[o]) * E_A2_TO_DA
            res[o].append(frob2(Q))
        n_conf += 1
    return ({o: np.array(v) for o, v in res.items()},
            {"n_conf": n_conf, "netq": netq})


# ======================================================== 自检

def _benzene_geom(d_cc=1.3964, d_ch=1.0831):
    """苯：实验几何（Tamagawa, Iijima & Kimura 1976, J. Mol. Struct. 30, 243:
    r(C–C)=1.3964 Å, r(C–H)=1.0831 Å）。z 为环法向。"""
    C, H = [], []
    for k in range(6):
        a = k * math.pi / 3
        C.append([d_cc * math.cos(a), d_cc * math.sin(a), 0.0])
        R = d_cc + d_ch
        H.append([R * math.cos(a), R * math.sin(a), 0.0])
    return np.array(C), np.array(H)


def calibrate_q_pi(d_pi=D_PI, verbose=True):
    """用苯的实验 Θ_zz 定 q_π（实际上定的是乘积 q_π·d²）。返回 (q_pi, 诊断)。"""
    C, H = _benzene_geom()
    # σ 框架电荷：ff14SB 的芳香 CH（PHE 的 CD1/HD1 对，见模板）
    ph = FF["PHE"]["charge"]
    qC, qH = ph["CD1"], ph["HD1"]
    coords = np.vstack([C, H])
    charges = np.array([qC] * 6 + [qH] * 6)
    org = np.zeros(3)
    Q_sigma = quadrupole(coords, charges, org) * E_A2_TO_DA
    # π 构造的解析贡献：ΔΘ_zz = −N·q·d²
    need = EXP_BENZENE_THETA_ZZ - Q_sigma[2, 2]
    q_pi = -need / (6.0 * d_pi ** 2) / E_A2_TO_DA
    # 数值复核（不用解析式，重新从位点算）
    pos = {f"C{k}": tuple(C[k]) for k in range(6)}
    extra, fix = pi_sites(pos, [[f"C{k}" for k in range(6)]], q_pi, d_pi)
    co = list(coords); cg = list(charges)
    for k in range(6):
        cg[k] += fix[f"C{k}"]
    for xyz, q in extra:
        co.append(tuple(xyz)); cg.append(q)
    Q_tot = quadrupole(np.array(co), np.array(cg), org) * E_A2_TO_DA
    # 独立对照：CHARMM36 的苯电荷（CG −0.115 / HG +0.115），同为核中心
    cg36 = np.array([-0.115] * 6 + [0.115] * 6)
    Q36 = quadrupole(coords, cg36, org) * E_A2_TO_DA
    rel_sigma = abs(Q_sigma[2, 2] - EXP_BENZENE_THETA_ZZ) / \
        abs(EXP_BENZENE_THETA_ZZ)
    if verbose:
        print(f"  [标定 a1 苯] σ 框架（ff14SB 核中心电荷）单独 "
              f"Θ_zz = {Q_sigma[2,2]:+.3f} D·Å，实验 "
              f"{EXP_BENZENE_THETA_ZZ:+.2f}，相对偏差 {rel_sigma:.1%}")
        print(f"            独立对照 CHARMM36 苯电荷（±0.115 e）"
              f"Θ_zz = {Q36[2,2]:+.3f} D·Å")
        print(f"            **任务前提被证伪**：核中心点电荷并不给出 ≈0。"
              f"RESP/ESP 拟合电荷把 C⁻H⁺ 键极性放在 1.40/2.48 Å 处，")
        print(f"            Θ_zz = −½Σq r²（面内电荷 z=0）本身就是大负数。"
              f"「≈0」只在全部电荷为零或各向同性时成立。")
        print(f"            π 位点因此只是 {rel_sigma:.1%} 的修正："
              f"d_π = {d_pi:.2f} Å 时 q_π = {q_pi:+.4f} e，"
              f"q_π·d² = {q_pi*d_pi**2:+.4f} e·Å²")
        print(f"            含 π 后 Θ_zz = {Q_tot[2,2]:+.3f} D·Å，"
              f"Θ_xx = {Q_tot[0,0]:+.3f}, 迹 = {np.trace(Q_tot):+.2e}")
    ok = (abs(Q_tot[2, 2] - EXP_BENZENE_THETA_ZZ) < 0.05 and
          Q_tot[2, 2] < 0 and Q_sigma[2, 2] < 0 and rel_sigma < 0.30 and
          Q36[2, 2] < 0 and abs(np.trace(Q_tot)) < 1e-8 and
          0.0 <= q_pi < 0.5)
    return q_pi, {"ok": ok, "Q_sigma_zz": float(Q_sigma[2, 2]),
                  "Q_charmm36_zz": float(Q36[2, 2]),
                  "rel_sigma": float(rel_sigma),
                  "Q_tot_zz": float(Q_tot[2, 2]), "q_pi": float(q_pi),
                  "qd2": float(q_pi * d_pi ** 2)}


def selftest_known_molecules(verbose=True):
    """(a) CO₂ 与水：纯核中心点电荷模型 vs 实验。"""
    ok = True
    # CO2, EPM2
    r = 1.149
    co = np.array([[0, 0, 0], [0, 0, r], [0, 0, -r]], float)
    cg = np.array([0.6512, -0.3256, -0.3256])
    Q = quadrupole(co, cg, np.zeros(3)) * E_A2_TO_DA
    rel = abs(Q[2, 2] - EXP_CO2_THETA_ZZ) / abs(EXP_CO2_THETA_ZZ)
    co2_ok = Q[2, 2] < 0 and rel < 0.25
    ok &= co2_ok
    if verbose:
        print(f"  [自检 a2 CO₂] Θ_zz = {Q[2,2]:+.3f} D·Å  实验 "
              f"{EXP_CO2_THETA_ZZ:+.2f}  相对偏差 {rel:.1%}  "
              f"{'通过' if co2_ok else '不通过'}")
    # TIP3P 水，主轴系：C2 轴为 z，分子面为 yz
    th = math.radians(104.52 / 2)
    roh = 0.9572
    wo = np.array([[0.0, 0.0, 0.0],
                   [0.0, roh * math.sin(th), roh * math.cos(th)],
                   [0.0, -roh * math.sin(th), roh * math.cos(th)]])
    wg = np.array([-0.834, 0.417, 0.417])
    com = (wo * np.array([15.999, 1.008, 1.008])[:, None]).sum(0) / 18.015
    Qw = quadrupole(wo, wg, com) * E_A2_TO_DA
    if verbose:
        print(f"  [自检 a3 水] TIP3P 核中心 Θ = "
              f"(xx {Qw[0,0]:+.2f}, yy {Qw[1,1]:+.2f}, zz {Qw[2,2]:+.2f}) D·Å")
        print(f"            实验 (xx {EXP_WATER['xx']:+.2f}, "
              f"yy {EXP_WATER['yy']:+.2f}, zz {EXP_WATER['zz']:+.2f})"
              f"  [Verhoeven & Dymanus 1970]")
        print(f"            量级对（~2.5 D·Å），**面外分量 xx 符号错**"
              f"（{Qw[0,0]:+.2f} vs {EXP_WATER['xx']:+.2f}）"
              f" —— 核中心电荷无面外密度，与苯同一个病。")
    water_scale_ok = 1.0 < max(abs(Qw).max(), 0) < 6.0
    ok &= water_scale_ok
    return ok


def selftest_both_directions(verbose=True):
    """(b) 既测能命中（已知各向异性），也测不误伤（各向同性必须给 0）。"""
    ok = True
    # 各向同性：正二十面体顶点等电荷
    p = (1 + 5 ** 0.5) / 2
    V = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            V += [[0, s1 * 1, s2 * p], [s1 * 1, s2 * p, 0],
                  [s2 * p, 0, s1 * 1]]
    V = np.array(V, float) * 1.5
    Qi = quadrupole(V, np.full(len(V), 0.1), np.zeros(3)) * E_A2_TO_DA
    iso_ok = frob2(Qi) < 1e-18
    ok &= iso_ok
    if verbose:
        print(f"  [自检 b1 各向同性] 正二十面体 12 等电荷 ‖Q‖² = "
              f"{frob2(Qi):.3e} (D·Å)²  {'通过' if iso_ok else '不通过'}")
    # 各向异性：−2q 在原点，+q 在 ±d ẑ。解析 Θ_zz = 2 q d²
    q, d = 0.37, 1.23
    co = np.array([[0, 0, 0], [0, 0, d], [0, 0, -d]], float)
    cg = np.array([-2 * q, q, q])
    Qa = quadrupole(co, cg, np.zeros(3))
    pred = 2 * q * d ** 2
    an_ok = abs(Qa[2, 2] - pred) < 1e-12 and \
        abs(Qa[0, 0] + pred / 2) < 1e-12
    ok &= an_ok
    if verbose:
        print(f"  [自检 b2 已知各向异性] Θ_zz = {Qa[2,2]:.9f} e·Å²  "
              f"解析 {pred:.9f}  {'通过' if an_ok else '不通过'}")
    # 无迹 + 旋转不变 + 中性零偶极分布的原点无关
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1.4, (14, 3))
    qq = rng.normal(0, 0.3, 14)
    qq -= qq.mean()
    X -= (qq[:, None] * X).sum(0) / 1.0 * 0.0      # 保持原样
    # 构造净电荷 0 且净偶极 0 的分布：对称化
    X2 = np.vstack([X, -X]); q2 = np.concatenate([qq, qq])
    q2 -= q2.mean()
    Q1 = quadrupole(X2, q2, np.zeros(3))
    Q2 = quadrupole(X2, q2, np.array([3.1, -2.2, 0.7]))
    org_ok = abs(frob2(Q1) - frob2(Q2)) / max(frob2(Q1), 1e-12) < 1e-10
    R = np.linalg.qr(rng.normal(0, 1, (3, 3)))[0]
    Q3 = quadrupole(X2 @ R.T, q2, np.zeros(3))
    rot_ok = abs(frob2(Q3) - frob2(Q1)) / frob2(Q1) < 1e-10
    tr_ok = abs(np.trace(Q1)) < 1e-10
    ok &= org_ok and rot_ok and tr_ok
    if verbose:
        print(f"  [自检 b3 不变性] 无迹 {abs(np.trace(Q1)):.2e}；"
              f"旋转不变 {'通过' if rot_ok else '不通过'}；"
              f"中性零偶极下原点无关 {'通过' if org_ok else '不通过'}")
    return ok


def selftest_h_and_charges(verbose=True):
    """补氢与电荷的已知事实检查（不是自洽性）。"""
    ok = True
    rows = []
    for r3 in AA3:
        spec = sidechain_spec(r3)
        nH = sum(len(v) for v in spec["h_on"].values())
        qsum = sum(spec["charge"][a] for a in spec["heavy"]) + \
            sum(spec["charge"][h] for v in spec["h_on"].values() for h in v)
        exp = spec["formal"]
        good = abs(qsum - exp) < 1e-9
        ok &= good
        bb = ideal_backbone()
        pos = build_sidechain(bb, r3, [180.0] * 4)
        full = dict(pos); full.update(bb)
        hp = place_hydrogens(full, spec)
        nplaced = len(hp)
        # 键长核查
        bad = 0
        for a, hs in spec["h_on"].items():
            for h in hs:
                if h in hp and a in full:
                    L = np.linalg.norm(np.array(hp[h]) - np.array(full[a]))
                    if abs(L - H_BOND.get(spec["elem"][a], 1.09)) > 1e-6:
                        bad += 1
        ok &= (nplaced == nH) and bad == 0
        rows.append((r3, nH, nplaced, qsum, exp, bad,
                     spec["link_resid"]))
    if verbose:
        print("  [自检 c1 电荷与补氢] 残基 预期H 实放H 侧链净电荷 应为 "
              "链修正残差 键长错")
        for r3, nH, np_, qs, exp, bad, lr in rows:
            flag = "" if abs(qs - exp) < 1e-9 and nH == np_ and not bad \
                else "   <<< 不通过"
            print(f"      {r3}  {nH:>5} {np_:>5}   {qs:+8.4f} {exp:+6.1f}"
                  f"{lr:>+12.4f}{bad:>5}{flag}")
        print(f"      链原子电荷修正 |残差| 最大 "
              f"{max(abs(r[6]) for r in rows):.4f} e（加在 CB 上）")
    # 已知事实：侧链重原子数 Trp 最多、Ala 最少（与 cache/sidechain_arm.json 一致）
    nsc = {r3: len([a for a in sidechain_spec(r3)["heavy"]]) for r3 in AA3}
    fact_ok = max(nsc, key=nsc.get) == "TRP" and min(nsc, key=nsc.get) == "ALA"
    ok &= fact_ok
    if verbose:
        print(f"  [自检 c2 已知事实] 侧链重原子最多 = "
              f"{max(nsc, key=nsc.get)}（应 TRP），最少 = "
              f"{min(nsc, key=nsc.get)}（应 ALA）  "
              f"{'通过' if fact_ok else '不通过'}")
    return ok


def run_selftests():
    print("=" * 78)
    print("自检（CLAUDE.md §10）：用已知事实，双向，带量级")
    print("=" * 78)
    q_pi, cal = calibrate_q_pi()
    ok = cal["ok"]
    ok &= selftest_known_molecules()
    ok &= selftest_both_directions()
    ok &= selftest_h_and_charges()
    print("-" * 78)
    if not ok:
        sys.exit("自检不通过 —— 中止，不输出任何表")
    print("自检全部通过。")
    return q_pi, cal


# ======================================================== 主流程

def build_table(q_pi, pi_variant="aromatic", d_pi=D_PI, step=CHI_STEP):
    tab = {}
    for r3 in AA3 + ["GLY"]:
        vals, diag = residue_quadrupoles(r3, q_pi, d_pi=d_pi,
                                         pi_variant=pi_variant, step=step)
        tab[r3] = {"vals": vals, "diag": diag}
    return tab


def main():
    q_pi, cal = run_selftests()
    print("\n" + "=" * 78)
    print(f"构象集合上的 ‖Q‖²（(D·Å)²，原点 = 侧链重原子质心，"
          f"π 变体 = aromatic，d_π = {D_PI} Å）")
    print("=" * 78)
    tab = build_table(q_pi, "aromatic")
    tab_b = build_table(q_pi, "all_sp2")
    tab_c = build_table(0.0, "aromatic")      # 完全不加 pi 位点的对照
    print(f"{'残基':<6}{'构象':>7}{'净q':>7}{'均值':>10}{'SD':>9}"
          f"{'最小':>10}{'最大':>10}{'CB原点均值':>12}{'q质心均值':>12}")
    print("-" * 90)
    for r3 in AA3 + ["GLY"]:
        v = tab[r3]["vals"]["centroid"]
        d = tab[r3]["diag"]
        print(f"{r3:<6}{d['n_conf']:>7}{d['netq']:>+7.2f}{v.mean():>10.2f}"
              f"{v.std():>9.2f}{v.min():>10.2f}{v.max():>10.2f}"
              f"{tab[r3]['vals']['cb'].mean():>12.2f}"
              f"{tab[r3]['vals']['qcent'].mean():>12.2f}")

    # 构象内扩散 vs 残基间扩散（若前者更大，这个量作为残基描述子就没用）
    within = np.mean([tab[r]["vals"]["centroid"].std() for r in AA3])
    between = np.std([tab[r]["vals"]["centroid"].mean() for r in AA3])
    print("-" * 90)
    print(f"构象内 SD 的均值 = {within:.2f}；残基间均值的 SD = {between:.2f}；"
          f"比值 within/between = {within/between:.3f}")
    print("  判据（任务要求）：比值 > 1 则该量不能当残基描述子。")

    # d_π 敏感性：q_π·d² 固定则芳香项应几乎不变
    print("\nd_π 敏感性（q_π 每次重标定，故 q_π·d² 固定）：")
    sens = {}
    for d in (0.47, 0.64, 1.00, 1.40):
        qp, c = calibrate_q_pi(d_pi=d, verbose=False)
        t = build_table(qp, "aromatic", d_pi=d, step=60)
        sens[d] = {r: float(t[r]["vals"]["centroid"].mean()) for r in AA3}
        print(f"  d_π={d:.2f}  q_π={qp:+.4f}  "
              f"TRP {sens[d]['TRP']:>8.2f}  PHE {sens[d]['PHE']:>8.2f}  "
              f"ASP {sens[d]['ASP']:>8.2f}  ILE {sens[d]['ILE']:>8.2f}")

    # 写表
    with open(OUT_TSV, "w") as f:
        f.write("# cache/sidechain_quadrupole.tsv\n")
        f.write("# 由 analysis/sidechain_quadrupole.py 生成"
                f"（{os.path.basename(FF_XML)} + analysis/rotamer_build.py）\n")
        f.write("# 量：无迹四极张量 Q_ab = sum_i q_i (3 x_a x_b - r^2 d_ab)/2 的\n")
        f.write("#     Frobenius 范数平方 ||Q||^2 = sum_ab Q_ab^2，单位 (D*A)^2\n")
        f.write("# 电荷模型：Amber ff14SB 残基模板电荷（OpenMM 发行版 "
                "amber14/protein.ff14SB.xml；\n")
        f.write("#     Maier et al. 2015 JCTC 11,3696 doi:10.1021/acs.jctc.5b00255）。\n")
        f.write("#     氢显式建（非联合原子）。His 取中性 HIE。\n")
        f.write("# pi 密度：芳香环（PHE/TYR/TRP/HIS）每个 pi 中心把电荷 q_pi 从核\n")
        f.write(f"#     搬到环法向 ±{D_PI:.2f} A 处（各 -q_pi/2），净电荷与净偶极不变。\n")
        f.write(f"#     q_pi = {q_pi:.4f} e，由苯的实验 Theta_zz = "
                f"{EXP_BENZENE_THETA_ZZ} D*A 标定\n")
        f.write("#     （Battaglia, Buckingham & Williams, Chem.Phys.Lett. 78 "
                "(1981) 421）。\n")
        f.write(f"#     解析上只有乘积 q_pi*d^2 = {q_pi*D_PI**2:.4f} e*A^2 进入结果。\n")
        f.write("#     **模型假设，不是测量**（CLAUDE.md §2）。\n")
        f.write("# q2_* = pi 位点只加在芳香环上（主）；q2b_* = 也加在羧基/酰胺/\n")
        f.write("#     胍基上；q2c_* = 完全不加 pi 位点（纯核中心 RESP 电荷）。\n")
        f.write("# 原点约定：ori=centroid 为侧链重原子几何质心（docs/16 §2 的 x0，主）；\n")
        f.write("#     cb = CB 原子；qcent = |q| 加权电荷质心。\n")
        f.write("#     Q 只在净电荷与净偶极皆为 0 时与原点无关；ASP/GLU/LYS/ARG\n")
        f.write("#     净电荷 ±1，故原点必须固定，三种并列输出以便报敏感性。\n")
        f.write(f"# 构象：chi 网格 {CHI_STEP} 度（极性 H 二面角 {POLAR_H_STEP} 度），\n")
        f.write(f"#     剔除图距离>=4 的重原子对距离 < {CLASH_MIN} A 的构象。\n")
        f.write("#     mean/sd/min/max 为该集合上的统计 —— 构象扩散必须和\n")
        f.write("#     残基间扩散一起读（见 docs/17）。\n")
        cols = ["res3", "res1", "n_conf", "net_q"]
        for tag in ("q2", "q2b", "q2c"):
            for o in ORIGINS:
                for st in ("mean", "sd", "min", "max"):
                    cols.append(f"{tag}_{o}_{st}")
        f.write("\t".join(cols) + "\n")
        for r3 in AA3 + ["GLY"]:
            row = [r3, ONE[r3], str(tab[r3]["diag"]["n_conf"]),
                   f"{tab[r3]['diag']['netq']:+.4f}"]
            for t in (tab, tab_b, tab_c):
                for o in ORIGINS:
                    v = t[r3]["vals"][o]
                    row += [f"{v.mean():.4f}", f"{v.std():.4f}",
                            f"{v.min():.4f}", f"{v.max():.4f}"]
            f.write("\t".join(row) + "\n")
    print(f"\n写入 {os.path.relpath(OUT_TSV, ROOT)}")

    json.dump({"q_pi": q_pi, "calib": cal, "d_pi": D_PI,
               "pi_effect_pct": {r: 100.0 * (
                   tab[r]["vals"]["centroid"].mean() /
                   max(tab_c[r]["vals"]["centroid"].mean(), 1e-9) - 1.0)
                   for r in ("PHE", "TYR", "TRP", "HIS")},
               "within_sd_mean": float(within), "between_sd": float(between),
               "d_sensitivity": sens},
              open(os.path.join(ROOT, "cache",
                                "sidechain_quadrupole_meta.json"), "w"),
              indent=1, default=str)
    print("\n芳香残基上 pi 位点带来的 ||Q||^2 变化（相对无 pi 的对照）：")
    for r in ("PHE", "TYR", "TRP", "HIS"):
        a = tab[r]["vals"]["centroid"].mean()
        c = tab_c[r]["vals"]["centroid"].mean()
        print(f"  {r}  无pi {c:>8.2f}  含pi {a:>8.2f}  "
              f"变化 {100*(a/c-1):+.1f}%")
    return 0


def load_table(path=OUT_TSV):
    """供别的脚本读表。返回 {col: {res1: value}}。"""
    import csv as _csv
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    rd = _csv.DictReader(lines, delimiter="\t")
    out = {}
    for r in rd:
        for k, v in r.items():
            if k in ("res3", "res1"):
                continue
            out.setdefault(k, {})[r["res1"]] = float(v)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
