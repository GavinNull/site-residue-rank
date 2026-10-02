#!/usr/bin/env python3
"""生成一级机理探针的理想化体系（纯 stdlib，不依赖 RDKit）。

探针库是快变量：不依赖判别集，建成后每个候选量过一遍只要几分钟。
故优先于判别集建成。签名表见 schema/factor-vocabulary.md。

四组探针：
  hbond_angle   水二聚体 D-H···A 角扫描        -> F02, F03, F01
  hammett       对位取代苯酚 + 水，Hammett 序列 -> F10, F17
  burial_depth  甲醇被不同数目甲烷包埋          -> F04, F05, F15
  cation_pi     NH4+ 在取代苯环轴上高度扫描     -> F06, F08

用法: python3 tools/make_probes.py
输出: probes/<name>/*.xyz + manifest.tsv（含该扫描的预期签名）
"""
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, "probes")

# ---------- 基本几何工具 ----------


def write_xyz(path, atoms, comment=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for s, (x, y, z) in atoms:
            f.write(f"{s:<3}{x:>14.6f}{y:>14.6f}{z:>14.6f}\n")


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(v, k):
    return (v[0] * k, v[1] * k, v[2] * k)


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return scale(v, 1.0 / n)


# ---------- 探针 1: 氢键角度扫描 ----------
# 水二聚体。受体水固定，供体水的 O 固定在 O···O = 2.90 Å，
# 扫描 O-H···O 角 180 -> 90。签名：代理量在 180° 取极值，90° 衰减到近零。

def water(origin=(0, 0, 0), rot=None):
    """标准水分子，O 在 origin，二等分线沿 +x。r(OH)=0.9572, ang=104.52"""
    r, ang = 0.9572, math.radians(104.52)
    h1 = (r * math.cos(ang / 2), r * math.sin(ang / 2), 0.0)
    h2 = (r * math.cos(ang / 2), -r * math.sin(ang / 2), 0.0)
    atoms = [("O", (0, 0, 0)), ("H", h1), ("H", h2)]
    if rot:
        atoms = [(s, rot(p)) for s, p in atoms]
    return [(s, add(p, origin)) for s, p in atoms]


def probe_hbond_angle():
    d_oo = 2.90
    rows = []
    for theta_deg in range(180, 89, -10):
        th = math.radians(theta_deg)
        # 受体水：O 在原点，孤对朝 -x（二等分线朝 +x，故 H 远离供体）
        acc = water((0.0, 0.0, 0.0))
        # 供体 O 放在 -x 方向
        d_o = (-d_oo, 0.0, 0.0)
        # 供体水：把一个 OH 指向受体 O，偏离 theta
        # 以供体 O 为中心，O->H 方向 = 绕 z 旋转 (180-theta) 后的 +x
        phi = math.pi - th
        oh_dir = (math.cos(phi), math.sin(phi), 0.0)
        r, ang = 0.9572, math.radians(104.52)
        h_a = add(d_o, scale(oh_dir, r))
        # 第二个 H：在 xy 平面内，与第一个 OH 成 104.52
        phi2 = phi + ang
        h_b = add(d_o, scale((math.cos(phi2), math.sin(phi2), 0.0), r))
        don = [("O", d_o), ("H", h_a), ("H", h_b)]
        atoms = don + acc
        name = f"theta_{theta_deg:03d}.xyz"
        write_xyz(os.path.join(PROBES, "hbond_angle", name), atoms,
                  f"water dimer  O-H...O angle={theta_deg} deg  r(OO)={d_oo}")
        rows.append((name, f"theta={theta_deg}", "0"))
    manifest(
        "hbond_angle", rows,
        factors="F02 F03 F01",
        signature=("代理量应在 theta=180 取极值，单调衰减至 theta=90 近零。"
                   "非单调或在 90 度仍有大值 => 该量未捕获方向性，出局。"),
        reference="水二聚体 De ~ 5.0 kcal/mol (CCSD(T)/CBS)，180 度构型",
    )


# ---------- 探针 2: Hammett 序列 ----------
# 对位取代苯酚作为氢键供体，与一个受体水成氢键。
# 签名：代理量与 Hammett sigma_p 单调，|r| >= 0.9。

RING_R = 1.395          # C-C 芳香键长

# 取代基用内坐标（NeRF）定义，锚在被取代的苯环碳上。
# 每条: (elem, parent, bond, angle, dihedral)
#   parent = -1 表示苯环碳；angle 相对 parent->grandparent 向量
#   第一个原子只用 bond，沿环心->环碳的轴向放置
#   dihedral 0 / 180 = 与苯环共面的两侧
SUBST = {
    "NMe2": (-0.83, [
        ("N", -1, 1.400, None, None),
        ("C",  0, 1.450, 120.0,   0.0),
        ("C",  0, 1.450, 120.0, 180.0),
        ("H",  1, 1.090, 109.5,   0.0),
        ("H",  1, 1.090, 109.5, 120.0),
        ("H",  1, 1.090, 109.5, 240.0),
        ("H",  2, 1.090, 109.5,   0.0),
        ("H",  2, 1.090, 109.5, 120.0),
        ("H",  2, 1.090, 109.5, 240.0),
    ]),
    "OMe": (-0.27, [
        ("O", -1, 1.370, None, None),
        ("C",  0, 1.424, 117.9,   0.0),
        ("H",  1, 1.090, 109.5,   0.0),
        ("H",  1, 1.090, 109.5, 120.0),
        ("H",  1, 1.090, 109.5, 240.0),
    ]),
    "Me": (-0.17, [
        ("C", -1, 1.513, None, None),
        ("H",  0, 1.090, 109.5,   0.0),
        ("H",  0, 1.090, 109.5, 120.0),
        ("H",  0, 1.090, 109.5, 240.0),
    ]),
    "H":  (0.00, [("H",  -1, 1.084, None, None)]),
    "F":  (0.06, [("F",  -1, 1.354, None, None)]),
    "Cl": (0.23, [("Cl", -1, 1.739, None, None)]),
    "CN": (0.66, [
        ("C", -1, 1.439, None, None),
        ("N",  0, 1.159, 180.0,   0.0),
    ]),
    "NO2": (0.78, [
        ("N", -1, 1.468, None, None),
        ("O",  0, 1.225, 117.5,   0.0),
        ("O",  0, 1.225, 117.5, 180.0),
    ]),
}


def nerf(a, b, c, bond, angle_deg, dih_deg):
    """由三个参考原子 a-b-c 与内坐标放置第四个原子（标准 NeRF）。
    新原子键连在 c 上，angle = b-c-new，dihedral = a-b-c-new。"""
    ang = math.radians(angle_deg)
    dih = math.radians(dih_deg)
    bc = unit((c[0] - b[0], c[1] - b[1], c[2] - b[2]))
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    n = (ab[1] * bc[2] - ab[2] * bc[1],
         ab[2] * bc[0] - ab[0] * bc[2],
         ab[0] * bc[1] - ab[1] * bc[0])
    nn = math.sqrt(sum(x * x for x in n))
    if nn < 1e-8:          # a-b-c 共线，任取一个垂直方向
        tmp = (0.0, 0.0, 1.0) if abs(bc[2]) < 0.9 else (1.0, 0.0, 0.0)
        n = (bc[1] * tmp[2] - bc[2] * tmp[1],
             bc[2] * tmp[0] - bc[0] * tmp[2],
             bc[0] * tmp[1] - bc[1] * tmp[0])
        nn = math.sqrt(sum(x * x for x in n))
    n = scale(n, 1.0 / nn)
    m = (n[1] * bc[2] - n[2] * bc[1],
         n[2] * bc[0] - n[0] * bc[2],
         n[0] * bc[1] - n[1] * bc[0])
    d2 = (-bond * math.cos(ang),
          bond * math.sin(ang) * math.cos(dih),
          bond * math.sin(ang) * math.sin(dih))
    return (c[0] + d2[0] * bc[0] + d2[1] * m[0] + d2[2] * n[0],
            c[1] + d2[0] * bc[1] + d2[1] * m[1] + d2[2] * n[1],
            c[2] + d2[0] * bc[2] + d2[1] * m[2] + d2[2] * n[2])


def benzene_ring(center=(0, 0, 0)):
    """苯环在 xy 平面，C1 在 +x 轴（接 OH），C4 在 -x 轴（接取代基）。"""
    cs, hs = [], []
    for i in range(6):
        a = math.radians(60 * i)
        p = (RING_R * math.cos(a), RING_R * math.sin(a), 0.0)
        cs.append(add(p, center))
    for i in range(6):
        if i in (0, 3):      # C1, C4 被取代
            continue
        a = math.radians(60 * i)
        d = RING_R + 1.084
        p = (d * math.cos(a), d * math.sin(a), 0.0)
        hs.append(("H", add(p, center)))
    return cs, hs


def place_subst(c_pos, axis, spec, ring_center=(0.0, 0.0, 0.0)):
    """沿 axis 从 c_pos 放置取代基。二壳层原子锚在其 parent 上，不是锚在环碳上
    —— 早先版本把所有原子都相对环碳定位，导致 NMe2/OMe/NO2 与邻位 H 重叠。"""
    ax = unit(axis)
    # 伪 grandparent / great-grandparent：环心 与一个平面内的参考点，
    # 使 dihedral 0/180 落在苯环平面内
    gp = ring_center
    ggp = (ring_center[0], ring_center[1], ring_center[2] + 1.0)
    placed = []
    for elem, parent, bond, angle, dih in spec:
        if parent == -1:
            pos = add(c_pos, scale(ax, bond))
            placed.append((elem, pos, c_pos, gp))
        else:
            pe, pp, ppar, pgpar = placed[parent]
            pos = nerf(ggp if ppar == c_pos and pgpar == gp else pgpar,
                       pgpar if False else ppar, pp, bond, angle, dih) \
                if False else nerf(pgpar, ppar, pp, bond, angle, dih)
            placed.append((elem, pos, pp, ppar))
    return [(e, p) for e, p, _, _ in placed]


def probe_hammett():
    rows = []
    for name, (sigma, spec) in sorted(SUBST.items(), key=lambda kv: kv[1][0]):
        cs, hs = benzene_ring()
        c1 = cs[0]          # +x
        c4 = cs[3]          # -x
        # 酚羟基：O 沿 +x，H 在 xy 平面内偏 109 度，指向 +y
        o = add(c1, (1.37, 0.0, 0.0))
        # C-O-H 角定义在 O->C 与 O->H 之间；O->C 为 -x，故 O->H 相对 +x
        # 的方位角是 180 - 108.8
        a = math.radians(180.0 - 108.8)
        oh = (0.967 * math.cos(a), 0.967 * math.sin(a), 0.0)
        h_o = add(o, oh)
        # 受体水：O 放在 O-H 延长线上，r(O...O) = 2.80，线性氢键
        d = unit(oh)
        acc_o = add(o, scale(d, 2.80))
        # 受体水的 H 远离供体
        perp = (-d[1], d[0], 0.0)
        r, ang = 0.9572, math.radians(104.52)
        # 孤对朝向供体（即 -d 方向），故 HOH 二等分线沿 +d 背离供体
        bis = scale(d, 1.0)
        h1 = add(acc_o, add(scale(bis, r * math.cos(ang / 2)),
                            scale(perp, r * math.sin(ang / 2))))
        h2 = add(acc_o, add(scale(bis, r * math.cos(ang / 2)),
                            scale(perp, -r * math.sin(ang / 2))))
        atoms = ([("C", p) for p in cs] + hs
                 + [("O", o), ("H", h_o)]
                 + place_subst(c4, (-1.0, 0.0, 0.0), spec)
                 + [("O", acc_o), ("H", h1), ("H", h2)])
        fn = f"phenol_{name}.xyz"
        write_xyz(os.path.join(PROBES, "hammett", fn), atoms,
                  f"4-{name}-phenol ... H2O  sigma_p={sigma}")
        rows.append((fn, f"sigma_p={sigma}", "0"))
    manifest(
        "hammett", rows,
        factors="F10 F17",
        signature=("代理量（或其氢键分量）应与 sigma_p 单调，|Pearson r| >= 0.90。"
                   "对 sigma_p 无响应 => 该量未捕获电子效应，"
                   "它在判别集上的任何相关性都是伪的。"),
        reference=("取代苯酚的 pKa 与 sigma_p 的 Hammett rho ~ 2.2 (水中)；"
                   "气相氢键强度随 sigma_p 单调上升，跨 NMe2->NO2 约 2-3 kcal/mol"),
    )


# ---------- 探针 3: 埋藏深度 ----------
# 甲醇的 OH 被 n 个甲烷围绕，n = 0..6，极性不变而埋藏加深。
# 签名：去溶剂化项随 n 单调上升；色散项随接触面积单调；两者应可分离。

def methane(center, orient=(0, 0, 1)):
    r = 1.087
    ax = unit(orient)
    # 任选正交基
    tmp = (1.0, 0.0, 0.0) if abs(ax[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = unit((ax[1] * tmp[2] - ax[2] * tmp[1],
              ax[2] * tmp[0] - ax[0] * tmp[2],
              ax[0] * tmp[1] - ax[1] * tmp[0]))
    v = (ax[1] * u[2] - ax[2] * u[1], ax[2] * u[0] - ax[0] * u[2],
         ax[0] * u[1] - ax[1] * u[0])
    th = math.radians(109.4712)
    hs = [add(center, scale(ax, r))]
    for k in range(3):
        ph = 2 * math.pi * k / 3
        d = add(scale(ax, math.cos(th)),
                add(scale(u, math.sin(th) * math.cos(ph)),
                    scale(v, math.sin(th) * math.sin(ph))))
        hs.append(add(center, scale(d, r)))
    return [("C", center)] + [("H", p) for p in hs]


def probe_burial_depth():
    rows = []
    # 甲醇：C 在原点，O 沿 +z
    c = (0.0, 0.0, 0.0)
    o = (0.0, 0.0, 1.43)
    # 同 bug 1：C-O-H 角在 O->C(-z) 与 O->H 之间
    a = math.radians(180.0 - 108.5)
    h_o = add(o, (0.956 * math.sin(a), 0.0, 0.956 * math.cos(a)))
    meoh = ([("C", c), ("O", o), ("H", h_o)]
            + [("H", add(c, p)) for p in
               [(1.03, 0.0, -0.36), (-0.51, 0.89, -0.36), (-0.51, -0.89, -0.36)]])
    shell_r = 4.6
    for n in range(0, 7):
        shell = []
        for k in range(n):
            # 在 OH 周围的半球上均匀铺开
            ph = 2 * math.pi * k / max(n, 1)
            tilt = math.radians(55)
            d = (math.sin(tilt) * math.cos(ph), math.sin(tilt) * math.sin(ph),
                 math.cos(tilt))
            ctr = add(o, scale(d, shell_r))
            shell += methane(ctr, orient=scale(d, -1))
        fn = f"burial_n{n}.xyz"
        write_xyz(os.path.join(PROBES, "burial_depth", fn), meoh + shell,
                  f"methanol + {n} CH4 shell at {shell_r} A")
        rows.append((fn, f"n_shell={n}", "0"))
    manifest(
        "burial_depth", rows,
        factors="F04 F05 F15",
        signature=("极性基团不变、埋藏加深：F04（脱水代价）应随 n 单调上升，"
                   "F05（色散）随接触面积单调上升，二者符号相反且应可分离。"
                   "若代理量对 n 非单调，说明它把两者混在一起且权重不稳。"),
        reference="甲烷-甲醇色散约 -0.5 kcal/mol/对；OH 完全脱水代价量级 4-6 kcal/mol",
    )


# ---------- 探针 4: 阳离子-π ----------
# NH4+ 在对位取代苯环的六重轴上，高度 2.6 -> 5.0 Å；取代基扫 Hammett 序列。
# 签名：随给电子取代增强、吸电子减弱；随高度指数衰减；强吸电子时可翻为阴离子-π。

def ammonium(center, axis=(0, 0, -1)):
    r = 1.027
    ax = unit(axis)
    tmp = (1.0, 0.0, 0.0) if abs(ax[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = unit((ax[1] * tmp[2] - ax[2] * tmp[1],
              ax[2] * tmp[0] - ax[0] * tmp[2],
              ax[0] * tmp[1] - ax[1] * tmp[0]))
    v = (ax[1] * u[2] - ax[2] * u[1], ax[2] * u[0] - ax[0] * u[2],
         ax[0] * u[1] - ax[1] * u[0])
    th = math.radians(109.4712)
    hs = [add(center, scale(ax, r))]
    for k in range(3):
        ph = 2 * math.pi * k / 3
        d = add(scale(ax, math.cos(th)),
                add(scale(u, math.sin(th) * math.cos(ph)),
                    scale(v, math.sin(th) * math.sin(ph))))
        hs.append(add(center, scale(d, r)))
    return [("N", center)] + [("H", p) for p in hs]


def probe_cation_pi():
    rows = []
    for sub in ("NMe2", "Me", "H", "CN", "NO2"):
        sigma, spec = SUBST[sub]
        for h10 in range(26, 51, 4):
            height = h10 / 10.0
            cs, hs = benzene_ring()
            c4 = cs[3]
            # C1 位补一个 H（本探针只单取代）
            ring_h = [("H", add(cs[0], (1.084, 0.0, 0.0)))]
            atoms = ([("C", p) for p in cs] + hs + ring_h
                     + place_subst(c4, (-1.0, 0.0, 0.0), spec)
                     + ammonium((0.0, 0.0, height), axis=(0, 0, -1)))
            fn = f"{sub}_h{h10}.xyz"
            write_xyz(os.path.join(PROBES, "cation_pi", fn), atoms,
                      f"NH4+ over 4-{sub}-benzene  h={height} A  sigma_p={sigma}  "
                      f"charge=+1")
            rows.append((fn, f"sub={sub} sigma_p={sigma} h={height}", "+1"))
    manifest(
        "cation_pi", rows,
        factors="F06 F08",
        signature=("固定高度下，代理量随 sigma_p 单调减弱（给电子增强）；"
                   "固定取代基下随高度近指数衰减。"
                   "强吸电子（NO2）时应接近零或反号（F08 阴离子-π 倾向）。"
                   "注意：体系净电荷 +1，计算设置必须显式带电荷。"),
        reference=("苯-NH4+ 结合能约 -19 kcal/mol (气相, h~2.9 A)；"
                   "取代效应跨 NMe2->NO2 可达 10 kcal/mol 量级"),
    )


# ---------- manifest ----------

def manifest(name, rows, factors, signature, reference):
    p = os.path.join(PROBES, name, "manifest.tsv")
    with open(p, "w") as f:
        f.write(f"# probe: {name}\n")
        f.write(f"# factors: {factors}\n")
        f.write(f"# signature: {signature}\n")
        f.write(f"# reference_magnitude: {reference}\n")
        f.write("#\n# 通过判据是定性的符号与单调性，不是数值。\n")
        f.write("# 但 reference_magnitude 用于量级校准：量级对不上也算不通过。\n")
        f.write("file\tcoordinate\tcharge\n")
        for fn, coord, chg in rows:
            f.write(f"{fn}\t{coord}\t{chg}\n")
    print(f"  {name:<14} {len(rows):>3} 个构型  -> probes/{name}/")


if __name__ == "__main__":
    print("生成探针库:")
    probe_hbond_angle()
    probe_hammett()
    probe_burial_depth()
    probe_cation_pi()
    print("\n完成。每组的预期签名写在各自 manifest.tsv 的注释里。")
    print("提醒：F07 F09 F11-F14 F16 F18-F20 的探针仍待建，")
    print("      这些因子不得用于 G4 的可微量实例化。见 schema/factor-vocabulary.md")
