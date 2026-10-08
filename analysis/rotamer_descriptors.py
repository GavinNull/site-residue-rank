#!/usr/bin/env python3
"""Rotamer 枚举给出的两个**不可分解的二维量**。

`docs/09-variance-budget.md` §6 导出的硬约束：要捕获那 1.01 kcal/mol 的
未解释耦合，候选量必须不能写成 f(残基)×g(位点) 的外积。
所有秩 1 候选（疏水×埋藏、电荷×场强、偶极×场强）已被排除；
氢键可达性（球面近似，仍近似可分解）p = 0.22 也不行。

本模块枚举真实 rotamer，给出两个量：

1. **f_hb = n_hb / n_allowed**
   在位阻允许的构象里，有多少比例能成氢键。
   「能不能成键」依赖位点几何与残基共价几何的联合，不可分解。

2. **S_rot = ln(n_allowed)**
   位点允许该残基保留多少构象 —— **rotamer 近似下的构象熵**。
   这是第一天（docs/00-problem.md §2）就标出、而所有便宜的焓类描述子
   都抓不到的那一项：
       「差异的很大一部分不在焓，而在去溶剂化代价和构象/水网络的熵」
   n_allowed 同时依赖位点的拥挤程度与残基的大小/柔性，也不可分解。

### 枚举

chi 网格 30 度，逐层剪枝（放一个原子就查碰撞），故树不会爆。
几何表全部由 10 个结构的 3962 个残基实测（cache/sidechain_geometry.json），
重建自检：3383 个真实侧链，中位误差 0.179 Å。

### 判据

碰撞：侧链重原子与环境重原子距离 < 3.2 Å（极性-极性对放宽到 2.5 Å，
      因为它们本来就能在氢键距离上接触）
氢键：侧链极性原子与环境极性原子距离在 2.5-3.4 Å
环境：同一位点残基自身的全部原子除外，其余（含伙伴链、配体、晶体水外）全算
"""
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_orientation import parse_pdb, sub, norm, ROOT
from rotamer_build import (CB_SPEC, N_CHI, SC_POLAR, SIDECHAINS,
                           build_sidechain, nerf)

# 碰撞阈值**由结构标定**，不用记忆值。
# 实测 8 个结构里 143215 个「侧链重原子 ↔ 环境重原子」接触：
#   含非极性对的 1% 分位 = 3.04 Å；极性-极性对 = 2.06 Å
# 早先用 3.2 / 2.5 太严，导致连野生型都被判碰撞。
# TOL 是对环境自身松弛的容差（刚性环境无法容纳比野生型大的残基，
# 而真实结构会让位），由野生型回收率自检定出。
# **阈值由被研究的那些位点本身定，不是全局分布的 1% 分位。**
# 全局（143215 个接触）1% 分位是 3.04 / 2.06；
# 但这 47 个位点的野生型在**晶体构象**下最小接触已是 2.66 / 2.53
# —— 蛋白酶-抑制剂复合物是已知最紧的界面之一，本就处在分布尾部。
# 按实测 chi 重建后最小 2.51 / 2.37（即网格+中位几何只带来 0.15 Å 误差）。
# 故阈值取略低于重建后的实测最小值：
CLASH_NP = 2.50
CLASH_PP = 2.35
TOL = 1.00
HB_MIN, HB_MAX = 2.5, 3.4
CHI_STEP = 30

ENV_POLAR_NAMES = {"N", "O", "OXT", "OG", "OG1", "OH", "SG", "SD",
                   "OD1", "OD2", "ND1", "ND2", "OE1", "OE2", "NE",
                   "NE1", "NE2", "NH1", "NH2", "NZ"}
LIG_POLAR = {"O1", "O2", "O3", "O4", "O5"}


def is_polar(name, resname=None):
    if resname and resname == "FLU":
        return name in LIG_POLAR
    return name in ENV_POLAR_NAMES


def site_environment(pid, chain, resi, cb, cutoff=14.0):
    """返回 (坐标, 是否极性) 列表，排除该位点残基自身的全部原子。"""
    path = os.path.join(ROOT, "cache", "pdb", f"{pid}.pdb")
    atoms = parse_pdb(path)
    env = []
    for a in atoms:
        if a["res"] == "HOH":
            continue
        if a["chain"] == chain and a["resi"] == str(resi) and not a["het"]:
            continue
        if norm(sub(a["xyz"], cb)) > cutoff:
            continue
        env.append((a["xyz"], is_polar(a["name"], a["res"])))
    return env


def _ok(p, p_polar, env):
    """单个原子是否与环境无碰撞；同时返回它是否成了氢键。"""
    hb = False
    for q, q_polar in env:
        d = norm(sub(p, q))
        pp = p_polar and q_polar
        if d < TOL * (CLASH_PP if pp else CLASH_NP):
            if pp and HB_MIN <= d <= HB_MAX:
                hb = True
                continue
            return False, False
        if pp and HB_MIN <= d <= HB_MAX:
            hb = True
    return True, hb


def enumerate_rotamers(bb, residue, env, step=CHI_STEP):
    """逐层剪枝枚举。返回 (n_allowed, n_hb)。"""
    spec = SIDECHAINS.get(residue)
    if spec is None:
        return 0, 0
    polar = set(SC_POLAR.get(residue, []))
    nchi = N_CHI.get(residue, 0)
    # 先放 CB
    name, a, b, c, bond, ang, dih = CB_SPEC
    pos0 = dict(bb)
    pos0["CB"] = nerf(pos0[a], pos0[b], pos0[c], bond, ang, dih[1])
    ok, _ = _ok(pos0["CB"], False, env)
    if not ok:
        return 0, 0
    if nchi == 0:
        return 1, 0

    grid = list(range(-180, 180, step))
    # 按 chi 层级分组原子
    layers = collections.defaultdict(list)
    for atom in spec:
        dh = atom[6]
        layers[dh[1] if dh[0] == "chi" else 0].append(atom)
    fixed = layers.pop(0, [])

    n_allowed = n_hb = 0

    def recurse(level, pos, chis, hb_so_far):
        nonlocal n_allowed, n_hb
        if level > nchi:
            # 放固定二面角的原子（环闭合等）
            p = dict(pos)
            for nm, a2, b2, c2, bd, an, dh in fixed:
                p[nm] = nerf(p[a2], p[b2], p[c2], bd, an, dh[1])
                o, h = _ok(p[nm], nm in polar, env)
                if not o:
                    return
                hb_so_far = hb_so_far or h
            n_allowed += 1
            n_hb += 1 if hb_so_far else 0
            return
        for chi in grid:
            p = dict(pos)
            cs = chis + [chi]
            bad = False
            hb = hb_so_far
            for nm, a2, b2, c2, bd, an, dh in layers.get(level, []):
                off = dh[2]
                p[nm] = nerf(p[a2], p[b2], p[c2], bd, an, chi + off)
                o, h = _ok(p[nm], nm in polar, env)
                if not o:
                    bad = True
                    break
                hb = hb or h
            if not bad:
                recurse(level + 1, p, cs, hb)

    recurse(1, pos0, [], False)
    return n_allowed, n_hb


def descriptors_for_site(pid, chain, resi, residues):
    path = os.path.join(ROOT, "cache", "pdb", f"{pid}.pdb")
    atoms = parse_pdb(path)
    bb = {}
    for a in atoms:
        if (a["chain"] == chain and a["resi"] == str(resi) and not a["het"]
                and a["name"] in ("N", "CA", "C")):
            bb[a["name"]] = a["xyz"]
    if len(bb) < 3:
        return None
    name, a2, b2, c2, bond, ang, dih = CB_SPEC
    cb = nerf(bb[a2], bb[b2], bb[c2], bond, ang, dih[1])
    env = site_environment(pid, chain, resi, cb)
    out = {}
    for rr in residues:
        na, nh = enumerate_rotamers(bb, rr, env)
        out[rr] = {"n_allowed": na, "n_hb": nh,
                   "f_hb": (nh / na) if na else float("nan"),
                   "S_rot": math.log(na) if na else float("nan")}
    return out


if __name__ == "__main__":
    import re
    print("自检：同一位点上不同残基的允许构象数应有明显差异，")
    print("      且大残基在埋藏位点应被压得更狠\n")
    for key, ch, ri in (("1CHO", "I", "18"), ("1FLR", "H", "103")):
        d = descriptors_for_site(key, ch, ri,
                                 ["ALA", "SER", "THR", "VAL", "ASN", "ASP",
                                  "GLN", "GLU", "MET", "LEU", "PHE", "TYR",
                                  "TRP", "LYS", "HIS"])
        print(f"[{key} 链{ch} 位点{ri}]")
        print(f"  {'残基':>5} {'n_allowed':>10} {'n_hb':>6} {'f_hb':>7} {'S_rot':>7}")
        for rr, v in sorted(d.items(), key=lambda kv: -kv[1]["n_allowed"]):
            print(f"  {rr:>5} {v['n_allowed']:>10} {v['n_hb']:>6} "
                  f"{v['f_hb']:>7.3f} {v['S_rot']:>7.2f}")
        print()
