#!/usr/bin/env python3
"""从结构算位点处的配体场方向，检验带取向的偶极通道。

### 为什么这是真检验而不是换个拟合

取向因子 cosθ = û_Cβ · Ê 在一个位点内对所有残基是**同一个数**，
所以它会被 c₂ 吸收，LOO-RMSE 根本不会变。
可检验的不是拟合质量，而是：

    **自由拟合出的 c₂ 的符号，是否与独立算出的场方向一致。**

推导：偶极相互作用 U = −μ·E。侧链偶极沿 û（Cα→Cβ 方向，
取代不变），则 U = −|μ|(û·E)。取代改变 |μ| 为 Δμ，故
ΔU = −Δμ(û·E)。回归式 ΔΔG = … + c₂·Δμ + … 给出

    预测 sign(c₂) = −sign(û·Ê)

场从结构算，ΔΔG 从数据拟，两边完全不相交。

OMTKY3 的 4 酶同位点对照是关键：同一抑制剂骨架（û 几乎相同）、
不同酶（Ê 不同）=> 预测的符号可以随酶而变。
|Δμ| 版本预测 c₂ 符号不随酶变。两者因此可分辨。

### 电荷模型

只用可电离基团的形式电荷 —— 在这些距离上它们支配场方向，
且这是假设最少的模型：
  Asp OD1/OD2 各 −0.5，Glu OE1/OE2 各 −0.5
  Lys NZ +1.0，Arg NH1/NH2 各 +0.5
  His 跳过（质子化态有歧义）
荧光素双阴离子：羧酸根两个 O 各 −0.5，酚氧 −1.0
"""
import math
import os
import re
import sys

CHARGES = {
    ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
    ("LYS", "NZ"): +1.0,
    ("ARG", "NH1"): +0.5, ("ARG", "NH2"): +0.5,
}
# 荧光素（FLU）双阴离子。按 1FLR 的实际键连环境分类（已核）：
#   O4, O5 共用 C20 -> 羧酸根，共 -1（各 -0.5）
#   O2 连 C3 与 C4  -> 呫吨环醚氧，中性
#   O1(C1), O3(C6)  -> 去质子化的酚氧/酮氧，双阴离子里共振等价，共 -1
# 合计 -2，与双阴离子一致
FLU_CHARGES = {"O4": -0.5, "O5": -0.5, "O1": -0.5, "O3": -0.5, "O2": 0.0}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_pdb(path):
    atoms = []
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            atoms.append({
                "het": line.startswith("HETATM"),
                "name": line[12:16].strip(),
                "res": line[17:20].strip(),
                "chain": line[21],
                "resi": line[22:27].strip(),
                "xyz": (float(line[30:38]), float(line[38:46]),
                        float(line[46:54])),
            })
    return atoms


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def unit(v):
    n = norm(v)
    return (v[0] / n, v[1] / n, v[2] / n) if n else (0.0, 0.0, 0.0)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def field_at(point, charges):
    """E = sum k q r_hat / r^2，返回矢量（任意统一单位）。"""
    E = [0.0, 0.0, 0.0]
    for q, pos in charges:
        d = sub(point, pos)
        r = norm(d)
        if r < 1e-6 or r > 25.0:        # 25 A 外截断
            continue
        f = q / r ** 3
        for i in range(3):
            E[i] += f * d[i]
    return tuple(E)


def collect_charges(atoms, partner_chains, ligand_resname=None):
    out = []
    for a in atoms:
        if ligand_resname and a["het"] and a["res"] == ligand_resname:
            q = FLU_CHARGES.get(a["name"])
            if q:
                out.append((q, a["xyz"]))
            continue
        if a["het"] or a["chain"] not in partner_chains:
            continue
        q = CHARGES.get((a["res"], a["name"]))
        if q:
            out.append((q, a["xyz"]))
    return out


def site_geometry(atoms, chain, resi):
    ca = cb = None
    for a in atoms:
        if a["chain"] != chain or a["resi"] != str(resi) or a["het"]:
            continue
        if a["name"] == "CA":
            ca = a["xyz"]
        elif a["name"] == "CB":
            cb = a["xyz"]
    return ca, cb


def analyse(pdb_key, chain, resi, dipole_offset=2.0):
    """返回 (cos_theta, |E|, 预测的 c2 符号)。"""
    m = re.match(r"^([0-9A-Za-z]{4})(?:_([A-Za-z]+))?(?:_([A-Za-z]+))?$", pdb_key)
    if not m:
        return None
    pid = m.group(1).upper()
    partner = set(m.group(2) or "")
    path = os.path.join(ROOT, "cache", "pdb", f"{pid}.pdb")
    if not os.path.exists(path):
        return None
    atoms = parse_pdb(path)
    lig = "FLU" if pid == "1FLR" else None
    if not partner and not lig:
        return None
    ca, cb = site_geometry(atoms, chain, resi)
    if ca is None or cb is None:
        return None                     # Gly 无 CB
    u = unit(sub(cb, ca))
    # 偶极有效中心：沿 û 从 Cβ 再外推 dipole_offset
    centre = tuple(cb[i] + dipole_offset * u[i] for i in range(3))
    ch = collect_charges(atoms, partner, lig)
    if not ch:
        return None
    E = field_at(centre, ch)
    nE = norm(E)
    if nE < 1e-9:
        return None
    cos_t = dot(u, unit(E))
    return {"cos": cos_t, "E": nE, "pred_sign_c2": -1 if cos_t > 0 else +1,
             "n_charges": len(ch)}


if __name__ == "__main__":
    print("FLU 原子名（用于确认电荷赋值）:")
    atoms = parse_pdb(os.path.join(ROOT, "cache", "pdb", "1FLR.pdb"))
    print("  ", sorted({a["name"] for a in atoms
                        if a["het"] and a["res"] == "FLU"}))
