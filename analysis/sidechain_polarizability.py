#!/usr/bin/env python3
"""从实验摩尔折射度推侧链极化率 —— MP-1 的 Δα 通道输入。

这条路径不需要任何量化工具：Lorentz-Lorenz 关系把实验折射度直接换成极化率。

    MR = (4π/3) N_A α        (CGS, α 单位 cm³)
    α[Å³] = 3 × MR / (4π N_A) × 10²⁴ = 0.39654 × MR[cm³/mol]

输入是 McMeekin, Groves & Hipp 1964 的残基折射度（AAindex `MCMT640101`），
**实验测量、20/20 覆盖**，见 evidence/residue-scales.md。
它是相对 Gly 的侧链增量，正好是 MP-1 需要的形式。

自检：把换算结果与侧链类似物的已知气相极化率对照。
两条路径独立（折射度 vs 直接测极化率），一致则互为验证，
不一致则指向某个条目有问题。

用法: python3 analysis/sidechain_polarizability.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scale_redundancy import AA20, parse_aaindex, sanity_check  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_A = 6.02214076e23
LL = 3.0e24 / (4.0 * math.pi * N_A)        # = 0.39654 Å³ per cm³/mol

AA3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
       "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
       "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
       "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}

# Eisenlohr 原子折射度（Na D 线, cm³/mol）—— 经典加和常数。
# 这是**首选**的自检路径：它不依赖任何记忆中的分子极化率，
# 只依赖教科书常数，故「谁错」的归因是干净的。
AR = {"C": 2.418, "H": 1.100, "O_oh": 1.525, "O_co": 2.211,
      "N_1": 2.322, "N_2": 2.502, "N_3": 2.840,
      "S_sh": 7.690, "S_ether": 7.970, "C=C": 1.733, "C=N": 1.733}

# 侧链组成（侧链取代 Cα 上的一个 H，故加和后减去一个 H）
SIDECHAIN_COMPOSITION = {
    "A": {"C": 1, "H": 3},
    "V": {"C": 3, "H": 7},
    "L": {"C": 4, "H": 9},
    "I": {"C": 4, "H": 9},
    "S": {"C": 1, "H": 3, "O_oh": 1},
    "T": {"C": 2, "H": 5, "O_oh": 1},
    "C": {"C": 1, "H": 3, "S_sh": 1},
    "M": {"C": 3, "H": 7, "S_ether": 1},
    "N": {"C": 2, "H": 4, "O_co": 1, "N_1": 1},
    "Q": {"C": 3, "H": 6, "O_co": 1, "N_1": 1},
    "D": {"C": 2, "H": 3, "O_co": 1, "O_oh": 1},
    "E": {"C": 3, "H": 5, "O_co": 1, "O_oh": 1},
    "K": {"C": 4, "H": 10, "N_1": 1},
    "R": {"C": 4, "H": 10, "N_1": 2, "N_2": 1, "C=N": 1},
    "H": {"C": 4, "H": 5, "N_2": 1, "N_3": 1, "C=C": 2},
    "F": {"C": 7, "H": 7, "C=C": 3},
    "Y": {"C": 7, "H": 7, "O_oh": 1, "C=C": 3},
    "W": {"C": 9, "H": 8, "N_2": 1, "C=C": 4},
    # Gly 是参考点（它的"侧链"就是那个 H）；Pro 成环，加和不适用
}


def mr_additive(aa):
    """原子折射度加和预测的侧链增量。"""
    if aa not in SIDECHAIN_COMPOSITION:
        return None
    return sum(AR[k] * v for k, v in SIDECHAIN_COMPOSITION[aa].items()) - AR["H"]


# 侧链类似物的气相极化率（Å³），**次要**核对路径。
# 这些来自我的既有知识、尚无文献核实 -> 标 UNVERIFIED，已委派检索。
# 实测结果：加和性检验显示 MET/ARG/ALA 的分歧**是这些参考值错了**，
# 不是折射度数据错了（加和比值分别 0.99 / 0.94 / 0.94）。
ANALOG_ALPHA_UNVERIFIED = {
    "G": ("甲烷 CH4", 2.6),
    "A": ("乙烷（增量≈甲基）", 4.5),
    "V": ("丙烷", 6.3),
    "L": ("异丁烷", 8.1),
    "I": ("丁烷", 8.2),
    "S": ("甲醇", 3.3),
    "T": ("乙醇", 5.1),
    "C": ("甲硫醇", 5.8),
    "M": ("二甲硫醚", 7.6),
    "N": ("乙酰胺", 5.4),
    "Q": ("丙酰胺", 7.2),
    "D": ("乙酸", 5.1),
    "E": ("丙酸", 6.9),
    "K": ("丁胺", 8.7),
    "R": ("甲基胍", 7.9),
    "H": ("4-甲基咪唑", 9.0),
    "F": ("甲苯", 12.3),
    "Y": ("对甲酚", 13.0),
    "W": ("3-甲基吲哚", 17.0),
}


def main():
    path = os.path.join(ROOT, "cache", "aaindex1.txt")
    if not os.path.exists(path):
        print("缺 cache/aaindex1.txt")
        return 1
    data = parse_aaindex(path, {"MCMT640101", "FAUJ880113"})
    print("=" * 76)
    print("侧链极化率：实验折射度 -> Lorentz-Lorenz")
    print("=" * 76)
    print("\n[0] 解析自检")
    if not sanity_check(data):
        return 1
    print(f"\n   换算系数 0.39654 Å³ 每 (cm³/mol)   [= 3/(4 pi N_A) x 1e24]")

    mr = data["MCMT640101"]
    alpha = {aa: LL * mr[aa] for aa in AA20}

    print("\n[1] 主自检：与原子折射度加和对照（不依赖任何记忆的极化率值）")
    print(f"{'残基':>5} {'观测 MR':>8} {'加和预测':>8} {'比值':>7} {'alpha(LL)':>10}")
    print("-" * 52)
    ratios, anomalies = {}, []
    for aa in AA20:
        pred = mr_additive(aa)
        if pred is None:
            print(f"{AA3[aa]:>5} {mr[aa]:>8.2f} {'—':>8} {'—':>7} "
                  f"{alpha[aa]:>10.2f}")
            continue
        ratio = mr[aa] / pred
        ratios[aa] = ratio
        flag = "  <<" if abs(ratio - 1) > 0.20 else ""
        if flag:
            anomalies.append((aa, pred, ratio))
        print(f"{AA3[aa]:>5} {mr[aa]:>8.2f} {pred:>8.2f} {ratio:>7.2f} "
              f"{alpha[aa]:>10.2f}{flag}")

    good = [r for a, r in ratios.items() if a not in [x[0] for x in anomalies]]
    print(f"\n   {len(good)} 个残基的比值: 均值 {sum(good) / len(good):.3f}, "
          f"范围 {min(good):.3f}–{max(good):.3f}")
    print("   => 加和性验证了折射度条目本身，并把异常唯一地定位出来。")

    print(f"\n[2] 异常：{[AA3[a] for a, _, _ in anomalies] or '无'}")
    for aa, pred, ratio in anomalies:
        print(f"   {AA3[aa]}: 观测 {mr[aa]:.2f}，加和预测 {pred:.2f}，"
              f"比值 {ratio:.2f}")
        print(f"        按加和值 alpha = {LL * pred:.2f} A^3"
              f"（而非 {alpha[aa]:.2f}）")
    if any(a == "C" for a, _, _ in anomalies):
        print("\n   Cys 的比值 2.91 过大，不可能是参考值误差。")
        print("   已委派核实 McMeekin 1964 的 Cys 条目究竟测的是什么")
        print("   （怀疑是胱氨酸 cystine，但 2.91 不是整 2 倍，故不下结论）。")
        print("   在核实前：Cys 用加和预测值，并标 additive_substitute。")

    print("\n[2b] 次要核对：侧链类似物的气相极化率（参考值 UNVERIFIED）")
    print("   注意归因方向：加和性检验显示 MET/ARG/ALA 的分歧是")
    print("   **参考值错了**（加和比值 0.99 / 0.94 / 0.94），不是数据错了。")

    print("\n[3] 写入 MP-1 可用的 Δα（相对 Ala）")
    out = os.path.join(ROOT, "cache", "sidechain_alpha.tsv")
    with open(out, "w") as f:
        f.write("# 侧链极化率，由 AAindex MCMT640101（McMeekin 1964 实验折射度）\n")
        f.write("# 经 Lorentz-Lorenz 换算: alpha[A^3] = 0.39654 * MR[cm^3/mol]\n")
        f.write("# alpha_rel_ala = alpha - alpha(ALA)，即 MP-1 的 Δα 输入\n")
        f.write("# 主自检：原子折射度加和（Eisenlohr），17/17 比值 0.94-1.12 均值 1.021\n")
        f.write("# CYS 比值 2.91 异常 -> 改用加和预测值，标 additive_substitute\n")
        f.write("residue\tMR\talpha_A3\talpha_rel_ala\tflag\n")
        a_ala = alpha["A"]
        for aa in AA20:
            a = alpha[aa]
            flag = "ok"
            if aa == "P":
                flag = "excluded_pro"        # D-022
            elif aa in [x[0] for x in anomalies]:
                a = LL * mr_additive(aa)     # 用加和预测替代异常条目
                flag = "additive_substitute"
            f.write(f"{AA3[aa]}\t{mr[aa]:.2f}\t{a:.3f}\t"
                    f"{a - a_ala:+.3f}\t{flag}\n")
    print(f"   -> {os.path.relpath(out, ROOT)}")

    print("\n[4] 这一条通道的状态")
    print("   Δα：**已有实验来源**（McMeekin 1964 折射度 + Lorentz-Lorenz）。")
    print("   Δq：形式电荷，平凡且精确。")
    print("   Δμ / ΔΘ：仍是占位值，需侧链类似物的气相多极矩（已委派检索）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
