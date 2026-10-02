#!/usr/bin/env python3
"""探针几何自检：在任何电子结构计算之前先确认构型没有坏掉。

理想化探针是手工解析构造的，最常见的失败是原子重叠或键长离谱——
这类错误会让 xtb 悄悄收敛到一个无意义的结构，而扫描曲线看起来仍然"平滑"。
所以这一步必须在 G0 的稳定性测试之前跑。

用法: python3 analysis/check_probes.py
"""
import glob
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 共价半径 (Å)，Cordero 2008
RCOV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Cl": 1.02}
# vdW 半径，用于判定非键重叠
RVDW = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Cl": 1.75}


def read_xyz(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    comment = lines[1]
    atoms = []
    for line in lines[2:2 + n]:
        p = line.split()
        atoms.append((p[0], (float(p[1]), float(p[2]), float(p[3]))))
    return atoms, comment


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def check(path):
    atoms, comment = read_xyz(path)
    probs = []
    n = len(atoms)
    bonds = 0
    min_nonbond = 1e9
    for i in range(n):
        si, pi = atoms[i]
        for j in range(i + 1, n):
            sj, pj = atoms[j]
            d = dist(pi, pj)
            rc = RCOV.get(si, 0.8) + RCOV.get(sj, 0.8)
            rv = RVDW.get(si, 1.7) + RVDW.get(sj, 1.7)
            if d < 0.70 * rc:
                probs.append(f"原子 {i}({si}) {j}({sj}) 距离 {d:.3f} Å «« 严重重叠")
            elif d <= 1.30 * rc:
                bonds += 1
                # 0.70-0.80 区间容纳真实三键：C#N 1.16 Å 对应 0.79*rcov,
                # 早先的 0.80 阈值把它误判为过短
                if d < 0.70 * rc:
                    probs.append(f"键 {i}({si})-{j}({sj}) 过短 {d:.3f} Å")
            else:
                # 非键接触：低于 vdW 和的 0.65 倍视为挤压
                # （氢键本身会低于 vdW，故阈值设得宽）
                if d < 0.60 * rv:
                    probs.append(f"非键 {i}({si})···{j}({sj}) 挤压 {d:.3f} Å "
                                 f"(vdW 和 {rv:.2f})")
                min_nonbond = min(min_nonbond, d)
    # 孤立原子检查
    for i in range(n):
        si, pi = atoms[i]
        has = any(dist(pi, atoms[j][1]) <= 1.30 * (RCOV.get(si, .8) +
                                                   RCOV.get(atoms[j][0], .8))
                  for j in range(n) if j != i)
        if not has:
            probs.append(f"原子 {i}({si}) 无成键伙伴（孤立）")
    return atoms, comment, probs, bonds, min_nonbond


def main():
    groups = sorted(d for d in glob.glob(os.path.join(ROOT, "probes", "*"))
                    if os.path.isdir(d))
    total, bad = 0, 0
    print("=" * 62)
    print("探针几何自检")
    print("=" * 62)
    for g in groups:
        name = os.path.basename(g)
        files = sorted(glob.glob(os.path.join(g, "*.xyz")))
        if not files:
            continue
        issues = []
        natoms = set()
        for p in files:
            total += 1
            atoms, comment, probs, bonds, mnb = check(p)
            natoms.add(len(atoms))
            if probs:
                bad += 1
                issues.append((os.path.basename(p), probs))
        print(f"\n[{name}]  {len(files)} 构型  原子数 {sorted(natoms)}")
        # 原子数一致性只在同一子扫描内有意义（不同取代基原子数本就不同）。
        # burial_depth 与 hammett 的扫描变量本身就改变原子数，不报。
        VARIES = {"burial_depth", "hammett"}
        byprefix = {} if name in VARIES else {}
        if name in VARIES:
            print(f"    (扫描变量改变原子数，跳过原子数一致性检查)")
        if name not in VARIES:
            for p2 in files:
                b = os.path.basename(p2).rsplit("_", 1)[0]
                byprefix.setdefault(b, set()).add(len(read_xyz(p2)[0]))
        for b, ns in sorted(byprefix.items()):
            if len(ns) > 1:
                print(f"    ! 子扫描 {b} 内原子数不一致 {sorted(ns)} —— "
                      f"该扫描曲线不可直接比较")
        if issues:
            for fn, probs in issues[:5]:
                print(f"    ! {fn}")
                for pr in probs[:4]:
                    print(f"        {pr}")
            if len(issues) > 5:
                print(f"    ... 另有 {len(issues) - 5} 个文件有问题")
        else:
            print("    几何正常")
    print(f"\n总计 {total} 个构型，{bad} 个有问题。")
    print("\n注意：几何正常只说明构型没坏，不说明探针有效。")
    print("探针有效性要等 G0 的稳定性测试 + 参考量级对照（需 xtb）。")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
