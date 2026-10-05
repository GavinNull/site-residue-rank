#!/usr/bin/env python3
"""把侧链构型放进去：氢键可达性作为位点×残基的耦合量。

### 思路

突变侧链的极性原子落在以 Cβ 为心、半径 reach(R) 的球面上
（reach 从 10 个结构的 1500+ 个实测侧链量出，短侧链标准差 0.01-0.04 Å）。
一个距 Cβ 为 D 的伙伴极性原子可达的条件是

    |D − reach(R)| <= d_hbond

**这是真正的耦合项**：同一位点对不同残基提供不同的氢键机会，
因为侧链长度不同。短的 Ser 只能触到 Cβ 附近，长的 Tyr 能伸到 5.7 Å。

加上方向约束：伙伴需在侧链指向的半球内（相对 Cα→Cβ）。

### 为什么这能测到此前测不到的那一档

去溶剂化描述子（疏水性）排得了「极性 vs 非极性」，
但对「两者都极性、只是氢键几何不同」无能 —— 因为它不含位点信息。
氢键可达性恰好只在那一档里有区分度。

### 可微形式

硬计数换成高斯核：
    n_hb = sum_partners  exp(-(D - reach)^2 / 2w^2) * max(0, cos)
对 reach 与 D 都光滑，故对坐标可微。
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_orientation import parse_pdb, sub, norm, unit, dot, ROOT

# 伙伴侧的极性原子（供体/受体），含主链
PARTNER_POLAR = {
    ("SER", "OG"), ("THR", "OG1"), ("CYS", "SG"), ("TYR", "OH"),
    ("ASN", "OD1"), ("ASN", "ND2"), ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLN", "OE1"), ("GLN", "NE2"), ("GLU", "OE1"), ("GLU", "OE2"),
    ("HIS", "ND1"), ("HIS", "NE2"), ("TRP", "NE1"),
    ("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
}
BACKBONE_POLAR = {"N", "O", "OXT"}
# 荧光素的极性原子
FLU_POLAR = {"O1", "O2", "O3", "O4", "O5"}

REACH = json.load(open(os.path.join(ROOT, "cache", "sidechain_reach.json")))
D_HB = 3.2          # 氢键重原子间距上限
WIDTH = 1.0         # 高斯核宽度，Å


def partner_polar_atoms(atoms, partner_chains, ligand=None):
    out = []
    for a in atoms:
        if a["het"]:
            if ligand and a["res"] == ligand and a["name"] in FLU_POLAR:
                out.append(a["xyz"])
            continue
        if a["chain"] not in partner_chains:
            continue
        if (a["res"], a["name"]) in PARTNER_POLAR or a["name"] in BACKBONE_POLAR:
            out.append(a["xyz"])
    return out


def hb_reach(pdb_key, chain, resi, residue, soft=True):
    """返回残基 `residue` 放在该位点时的氢键可达性分数。"""
    import re
    m = re.match(r"^([0-9A-Za-z]{4})(?:_([A-Za-z]+))?(?:_([A-Za-z]+))?$", pdb_key)
    if not m:
        return None
    pid = m.group(1).upper()
    partner = set(m.group(2) or "")
    lig = "FLU" if pid == "1FLR" else None
    if not partner and not lig:
        return None
    if residue not in REACH:
        return 0.0                      # 非极性残基：无极性原子，分数 0
    path = os.path.join(ROOT, "cache", "pdb", f"{pid}.pdb")
    if not os.path.exists(path):
        return None
    atoms = parse_pdb(path)
    ca = cb = None
    for a in atoms:
        if a["chain"] == chain and a["resi"] == str(resi) and not a["het"]:
            if a["name"] == "CA":
                ca = a["xyz"]
            elif a["name"] == "CB":
                cb = a["xyz"]
    if ca is None or cb is None:
        return None
    u = unit(sub(cb, ca))
    r = REACH[residue]
    score = 0.0
    for p in partner_polar_atoms(atoms, partner, lig):
        d = sub(p, cb)
        D = norm(d)
        if D < 1e-6 or D > r + D_HB + 2.0:
            continue
        cosang = dot(u, unit(d))
        if cosang < -0.2:               # 伙伴在侧链背面，不可达
            continue
        w = max(0.0, cosang + 0.2) / 1.2
        if soft:
            score += math.exp(-((D - r) ** 2) / (2 * WIDTH ** 2)) * w
        else:
            score += 1.0 if abs(D - r) <= D_HB else 0.0
    return score


if __name__ == "__main__":
    # 自检：同一位点上不同长度的残基应得到不同分数，
    # 且非极性残基应为 0
    print("自检：1CHO 链 I 位点 18（P1，插进 S1 口袋）")
    for res in ("SER", "THR", "ASN", "ASP", "GLN", "GLU", "TYR", "LYS", "ARG"):
        s = hb_reach("1CHO_EFG_I", "I", "18", res)
        print(f"   {res}  reach={REACH.get(res, 0):.2f} A  可达性分数 {s:.3f}")
    print(f"   LEU（非极性）         可达性分数 "
          f"{hb_reach('1CHO_EFG_I', 'I', '18', 'LEU'):.3f}")
