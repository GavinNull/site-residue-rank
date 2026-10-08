#!/usr/bin/env python3
"""表驱动的侧链构建（NeRF 内坐标）+ rotamer 枚举。

动机见 `docs/09-variance-budget.md` §6：要捕获那 1.01 kcal/mol 的未解释耦合，
候选量必须是**不可分解的二维量**。氢键可达性只用了 reach 一个标量（球面近似），
所以它仍可写成近似的外积形式，实测 p = 0.22。

真正的 rotamer 枚举给出「有多少构象既不碰撞又能成键」——
这个量同时依赖位点的位阻环境与残基的共价几何，**不可分解为 f(残基)×g(位点)**。

### 原子规格

每个原子: (name, a, b, c, bond, angle, dihedral)
  a,b,c 为已放置的参考原子名（可用主链 N/CA/C）
  dihedral = ("chi", k, offset) 表示 chi_k + offset 度；("fix", v) 表示固定值
NeRF: 新原子键连在 c 上，angle = b-c-new，dihedral = a-b-c-new。

环是平面的，故环原子的二面角取 0 / 180，与链原子同一套机制处理。

### 自检（CLAUDE.md §10）

拿晶体结构里的真实侧链验：从 PDB 读该残基的实际 chi 角，用本模块重建，
原子位置应与晶体坐标吻合到约 0.3 Å。不吻合则几何表有错。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHI = "chi"
FIX = "fix"


def _d(k, off=0.0):
    return (CHI, k, off)


def _f(v):
    return (FIX, v)


# 氨基酸几何。bond (Å), angle (deg)。
# **全部由 10 个结构的 3962 个残基实测得出**（cache/sidechain_geometry.json），
# 不用记忆值。键长实测 σ 多在 0.01-0.02 Å，角度 σ 1-5 度。
# CB 由 N, CA, C 定位（固定，不含 chi）
# CB 由 N, C, CA 定位。**改进二面角的符号必须按实测**：
# 从 3583 个残基量出 dihedral(N,C,CA,CB) = +122.69 (σ 6.36)，
# 而 dihedral(C,N,CA,CB) = −122.69。早先写成 nerf(C,N,CA,...,+122.6)
# 即符号反了，导致连 ALA（零个 chi）都有 2.4 Å 误差。
CB_SPEC = ("CB", "N", "C", "CA", 1.531, 110.53, _f(122.69))

SIDECHAINS = {
    "ALA": [],
    "GLY": None,                     # 无侧链
    "SER": [("OG", "N", "CA", "CB", 1.417, 110.91, _d(1))],
    "CYS": [("SG", "N", "CA", "CB", 1.813, 114.58, _d(1))],
    "THR": [("OG1", "N", "CA", "CB", 1.451, 107.43, _d(1)),
            ("CG2", "N", "CA", "CB", 1.527, 112.24, _d(1, -120.0))],
    "VAL": [("CG1", "N", "CA", "CB", 1.525, 111.46, _d(1)),
            ("CG2", "N", "CA", "CB", 1.518, 111.10, _d(1, 120.0))],
    "LEU": [("CG", "N", "CA", "CB", 1.528, 116.42, _d(1)),
            ("CD1", "CA", "CB", "CG", 1.523, 110.55, _d(2)),
            ("CD2", "CA", "CB", "CG", 1.524, 109.65, _d(2, 120.0))],
    "ILE": [("CG1", "N", "CA", "CB", 1.538, 110.41, _d(1)),
            ("CG2", "N", "CA", "CB", 1.533, 110.95, _d(1, -122.0)),
            ("CD1", "CA", "CB", "CG1", 1.526, 113.61, _d(2))],
    "MET": [("CG", "N", "CA", "CB", 1.511, 113.72, _d(1)),
            ("SD", "CA", "CB", "CG", 1.817, 109.91, _d(2)),
            ("CE", "CB", "CG", "SD", 1.793, 100.79, _d(3))],
    "ASP": [("CG", "N", "CA", "CB", 1.516, 112.69, _d(1)),
            ("OD1", "CA", "CB", "CG", 1.247, 119.71, _d(2)),
            ("OD2", "CA", "CB", "CG", 1.251, 117.61, _d(2, 180.0))],
    "ASN": [("CG", "N", "CA", "CB", 1.516, 112.65, _d(1)),
            ("OD1", "CA", "CB", "CG", 1.237, 120.71, _d(2)),
            ("ND2", "CA", "CB", "CG", 1.331, 116.46, _d(2, 180.0))],
    "GLU": [("CG", "N", "CA", "CB", 1.522, 114.97, _d(1)),
            ("CD", "CA", "CB", "CG", 1.511, 113.71, _d(2)),
            ("OE1", "CB", "CG", "CD", 1.250, 119.92, _d(3)),
            ("OE2", "CB", "CG", "CD", 1.254, 117.40, _d(3, 180.0))],
    "GLN": [("CG", "N", "CA", "CB", 1.521, 114.49, _d(1)),
            ("CD", "CA", "CB", "CG", 1.517, 112.52, _d(2)),
            ("OE1", "CB", "CG", "CD", 1.234, 121.36, _d(3)),
            ("NE2", "CB", "CG", "CD", 1.334, 116.13, _d(3, 180.0))],
    "LYS": [("CG", "N", "CA", "CB", 1.525, 113.82, _d(1)),
            ("CD", "CA", "CB", "CG", 1.522, 111.35, _d(2)),
            ("CE", "CB", "CG", "CD", 1.521, 112.10, _d(3)),
            ("NZ", "CG", "CD", "CE", 1.494, 111.40, _d(4))],
    "ARG": [("CG", "N", "CA", "CB", 1.522, 114.11, _d(1)),
            ("CD", "CA", "CB", "CG", 1.510, 111.99, _d(2)),
            ("NE", "CB", "CG", "CD", 1.460, 111.50, _d(3)),
            ("CZ", "CG", "CD", "NE", 1.331, 124.59, _d(4)),
            ("NH1", "CD", "NE", "CZ", 1.330, 120.83, _f(0.0)),
            ("NH2", "CD", "NE", "CZ", 1.332, 119.79, _f(180.0))],
    "HIS": [("CG", "N", "CA", "CB", 1.496, 113.16, _d(1)),
            ("ND1", "CA", "CB", "CG", 1.376, 122.78, _d(2)),
            ("CD2", "CA", "CB", "CG", 1.356, 129.70, _d(2, 180.0)),
            ("CE1", "CB", "CG", "ND1", 1.324, 108.92, _f(180.0)),
            ("NE2", "CB", "CG", "CD2", 1.373, 107.12, _f(180.0))],
    "PHE": [("CG", "N", "CA", "CB", 1.505, 113.86, _d(1)),
            ("CD1", "CA", "CB", "CG", 1.392, 120.31, _d(2)),
            ("CD2", "CA", "CB", "CG", 1.389, 120.48, _d(2, 180.0)),
            ("CE1", "CB", "CG", "CD1", 1.389, 120.84, _f(180.0)),
            ("CE2", "CB", "CG", "CD2", 1.393, 120.59, _f(180.0)),
            ("CZ", "CG", "CD1", "CE1", 1.391, 119.63, _f(0.0))],
    "TYR": [("CG", "N", "CA", "CB", 1.512, 113.59, _d(1)),
            ("CD1", "CA", "CB", "CG", 1.395, 120.98, _d(2)),
            ("CD2", "CA", "CB", "CG", 1.393, 120.79, _d(2, 180.0)),
            ("CE1", "CB", "CG", "CD1", 1.392, 121.13, _f(180.0)),
            ("CE2", "CB", "CG", "CD2", 1.392, 121.21, _f(180.0)),
            ("CZ", "CG", "CD1", "CE1", 1.390, 119.42, _f(0.0)),
            ("OH", "CD1", "CE1", "CZ", 1.375, 120.07, _f(180.0))],
    "TRP": [("CG", "N", "CA", "CB", 1.499, 113.57, _d(1)),
            ("CD1", "CA", "CB", "CG", 1.367, 127.22, _d(2)),
            ("CD2", "CA", "CB", "CG", 1.435, 126.50, _d(2, 180.0)),
            ("NE1", "CB", "CG", "CD1", 1.377, 110.21, _f(180.0)),
            ("CE2", "CB", "CG", "CD2", 1.411, 107.19, _f(180.0)),
            ("CE3", "CB", "CG", "CD2", 1.395, 133.56, _f(0.0)),
            ("CZ2", "CG", "CD2", "CE2", 1.391, 122.39, _f(180.0)),
            ("CZ3", "CG", "CD2", "CE3", 1.379, 118.35, _f(180.0)),
            ("CH2", "CD2", "CE2", "CZ2", 1.380, 117.09, _f(0.0))],
}

# 每个残基的可变 chi 数（决定枚举维数）
N_CHI = {"ALA": 0, "GLY": 0, "SER": 1, "CYS": 1, "THR": 1, "VAL": 1,
         "LEU": 2, "ILE": 2, "MET": 3, "ASP": 2, "ASN": 2, "GLU": 3,
         "GLN": 3, "LYS": 4, "ARG": 4, "HIS": 2, "PHE": 2, "TYR": 2,
         "TRP": 2}

# 侧链极性原子（可做氢键供体/受体）
SC_POLAR = {
    "SER": ["OG"], "THR": ["OG1"], "CYS": ["SG"], "TYR": ["OH"],
    "ASN": ["OD1", "ND2"], "ASP": ["OD1", "OD2"],
    "GLN": ["OE1", "NE2"], "GLU": ["OE1", "OE2"],
    "HIS": ["ND1", "NE2"], "TRP": ["NE1"], "LYS": ["NZ"],
    "ARG": ["NE", "NH1", "NH2"], "MET": ["SD"],
}


def nerf(a, b, c, bond, angle_deg, dih_deg):
    ang = math.radians(angle_deg)
    dih = math.radians(dih_deg)
    bc = _unit((c[0] - b[0], c[1] - b[1], c[2] - b[2]))
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    n = (ab[1] * bc[2] - ab[2] * bc[1], ab[2] * bc[0] - ab[0] * bc[2],
         ab[0] * bc[1] - ab[1] * bc[0])
    nn = math.sqrt(sum(x * x for x in n))
    if nn < 1e-8:
        tmp = (0.0, 0.0, 1.0) if abs(bc[2]) < 0.9 else (1.0, 0.0, 0.0)
        n = (bc[1] * tmp[2] - bc[2] * tmp[1], bc[2] * tmp[0] - bc[0] * tmp[2],
             bc[0] * tmp[1] - bc[1] * tmp[0])
        nn = math.sqrt(sum(x * x for x in n))
    n = tuple(x / nn for x in n)
    m = (n[1] * bc[2] - n[2] * bc[1], n[2] * bc[0] - n[0] * bc[2],
         n[0] * bc[1] - n[1] * bc[0])
    d2 = (-bond * math.cos(ang), bond * math.sin(ang) * math.cos(dih),
          bond * math.sin(ang) * math.sin(dih))
    return tuple(c[i] + d2[0] * bc[i] + d2[1] * m[i] + d2[2] * n[i]
                 for i in range(3))


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v) if n else (0.0, 0.0, 0.0)


def build_sidechain(bb, residue, chis, place_cb=True):
    """bb: {'N':xyz,'CA':xyz,'C':xyz}（必要时含 'CB'）。chis: chi 角列表（度）。

    返回 {atom_name: xyz}，含 CB。GLY 返回 {}。
    """
    spec = SIDECHAINS.get(residue)
    if spec is None:
        return {}
    pos = dict(bb)
    if place_cb or "CB" not in pos:
        name, a, b, c, bond, ang, dih = CB_SPEC
        pos["CB"] = nerf(pos[a], pos[b], pos[c], bond, ang, dih[1])
    for name, a, b, c, bond, ang, dih in spec:
        if dih[0] == CHI:
            k, off = dih[1], dih[2]
            val = (chis[k - 1] if k - 1 < len(chis) else 180.0) + off
        else:
            val = dih[1]
        pos[name] = nerf(pos[a], pos[b], pos[c], bond, ang, val)
    return {k: v for k, v in pos.items() if k not in ("N", "CA", "C")}


def measure_chis(atoms_by_name, residue):
    """从实际坐标反算 chi 角（用于自检）。"""
    spec = SIDECHAINS.get(residue)
    if not spec:
        return []
    out = {}
    for name, a, b, c, bond, ang, dih in spec:
        if dih[0] != CHI or dih[2] != 0.0:
            continue
        k = dih[1]
        if all(x in atoms_by_name for x in (a, b, c, name)):
            out[k] = dihedral(atoms_by_name[a], atoms_by_name[b],
                              atoms_by_name[c], atoms_by_name[name])
    return [out.get(i + 1, 180.0) for i in range(N_CHI.get(residue, 0))]


def dihedral(p0, p1, p2, p3):
    b0 = tuple(p0[i] - p1[i] for i in range(3))
    b1 = tuple(p2[i] - p1[i] for i in range(3))
    b2 = tuple(p3[i] - p2[i] for i in range(3))
    b1u = _unit(b1)
    v = tuple(b0[i] - sum(b0[j] * b1u[j] for j in range(3)) * b1u[i]
              for i in range(3))
    w = tuple(b2[i] - sum(b2[j] * b1u[j] for j in range(3)) * b1u[i]
              for i in range(3))
    x = sum(v[i] * w[i] for i in range(3))
    cr = (b1u[1] * v[2] - b1u[2] * v[1], b1u[2] * v[0] - b1u[0] * v[2],
          b1u[0] * v[1] - b1u[1] * v[0])
    y = sum(cr[i] * w[i] for i in range(3))
    return math.degrees(math.atan2(y, x))
