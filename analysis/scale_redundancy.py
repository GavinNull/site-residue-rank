#!/usr/bin/env python3
"""M4.2：干净实验侧链标度实际提供多少条独立轴？

这是零计算的维数分析——数据已在本地（AAindex），只需算术。
目的是定 `docs/05-mathematical-structure.md` M4 的嵌入维数 d，
并与 M5 测出的秩对照（M4.2：r <= d）。

**关键分组**（依据 evidence/residue-scales.md）：
  experimental   有独立实验基础（分配系数、滴定 pKa、折射率、偏比容）
  constructed    由实验半径/折射度几何构造或加和估算，非测量
  transplanted   小分子取代基表值直接赋给侧链，无侧链独立测量
  count          从结构式数出来的原子/轨道个数，不是能量

**transplanted 与 count 不计入 d**，因为它们不是独立物理轴。
把它们算进去会让嵌入维数与秩估计循环依赖（Q-001）。

用法:
  python3 analysis/scale_redundancy.py [--aaindex PATH]
"""
import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AA20 = list("ARNDCQEGHILKMFPSTWYV")

# AAindex 的 I 字段是**行主序**，不是按表头 "A/L R/K ..." 交错。
# 前 10 个数 = 表头各列的第一个残基，后 10 个 = 第二个。
# 核验：MCMT640101（折射度）在此解读下 Gly=0.00 为最小、Trp=42.53 为最大，
# 两者都是已知事实。早先按交错排列解析，残基与数值全部错位（静默，无报错）。
AA_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
            "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

# 候选标度。provenance 来自 evidence/residue-scales.md
SCALES = {
    # --- experimental ---
    "MCMT640101": ("折射率测得的摩尔折射度 (McMeekin 1964)", "experimental", "极化率/MR"),
    "WOLR810101": ("气相->水 转移自由能 (Wolfenden 1981)", "experimental", "疏水/水合"),
    "FAUJ830101": ("辛醇/水 分配系数 pi (Fauchere-Pliska 1983)", "experimental", "疏水"),
    "FAUJ880113": ("R-COOH 滴定 pKa (Fauchere 1988)", "experimental", "电子效应"),
    "COHE430101": ("偏比容 (Cohn-Edsall 1943)", "experimental", "体积"),
    # --- constructed ---
    "CHAM810101": ("upsilon = r_vdW - 1.20 (Charton 1981)", "constructed", "空间"),
    "CHAM820101": ("基团折射度加和估算的极化率 (Charton 1982)", "constructed", "极化率"),
    # --- transplanted ---
    "FAUJ880108": ("局域电效应 sigma_I (小分子取代基表值搬运)", "transplanted", "场/诱导"),
    # --- count ---
    "FAUJ880109": ("氢键供体数 (IUPAC 结构计数)", "count", "氢键供体"),
    "FAUJ880110": ("满非键轨道数 (IUPAC 结构计数)", "count", "氢键受体"),
    "FAUJ880111": ("正电荷指示 (0/1)", "count", "电荷"),
    "FAUJ880112": ("负电荷指示 (0/1)", "count", "电荷"),
}


def parse_aaindex(path, wanted):
    """抽取指定条目的 20 个数值。返回 {code: {aa: value}}。

    缺失值有**两种**表示，必须都处理：
      1. 显式 `NA`（文件里 588 处）
      2. 在 `J` 字段注明 `(Pro missing)` 或 `(Pro !)`，而 Pro 位置写成裸 `0.`
         —— 取代基派生的标度几乎都是这一类。早先版本把它读成数值 0，
         对 pKa 之类的量意味着物理上不可能的值，且**静默无报错**。
    """
    out = {}
    code = None
    collecting = False
    nums = []
    jtext = ""
    with open(path) as f:
        for line in f:
            if line.startswith("H "):
                code = line[2:].strip()
                collecting = False
                nums = []
                jtext = ""
            elif line.startswith("J ") or (line.startswith("  ") and jtext
                                           and not collecting):
                jtext += " " + line.strip()
            elif line.startswith("I ") and code in wanted:
                collecting = True
                nums = []
            elif collecting:
                if line.startswith("//"):
                    vals = dict(zip(AA_ORDER, nums))
                    # 用文件自身的文档判定缺失，不靠猜格式
                    if "Pro missing" in jtext or "Pro !" in jtext:
                        vals["P"] = float("nan")
                    out[code] = vals
                    collecting = False
                else:
                    for tok in line.split():
                        nums.append(float("nan") if tok.upper() == "NA"
                                    else float(tok))
    return out


def sanity_check(data):
    """用已知事实核验解析是否正确。解析错位是静默的，必须主动查。"""
    ok = True
    if "MCMT640101" in data:
        v = data["MCMT640101"]
        gmin = min((x for x in v.items() if math.isfinite(x[1])),
                   key=lambda t: t[1])[0]
        wmax = max((x for x in v.items() if math.isfinite(x[1])),
                   key=lambda t: t[1])[0]
        hit = (gmin == "G" and wmax == "W")
        print(f"   折射度 MCMT640101: 最小={gmin}(应 G) 最大={wmax}(应 W)  "
              f"{'OK' if hit else '<< 残基顺序错位'}")
        ok &= hit
    if "FAUJ880113" in data:
        v = data["FAUJ880113"]
        pro_na = not math.isfinite(v["P"])
        rng = [x for x in v.values() if math.isfinite(x)]
        plaus = all(2.0 < x < 7.0 for x in rng)
        print(f"   RCOOH pKa FAUJ880113: Pro 已标缺失={pro_na}  "
              f"其余落在 2-7={plaus}  {'OK' if pro_na and plaus else '<< 缺失标记未处理'}")
        ok &= pro_na and plaus
    return ok


def corr(a, b):
    """成对完整的 Pearson 相关。"""
    m = [(x, y) for x, y in zip(a, b)
         if math.isfinite(x) and math.isfinite(y)]
    if len(m) < 4:
        return float("nan"), len(m)
    xs = np.array([t[0] for t in m]); ys = np.array([t[1] for t in m])
    if xs.std() == 0 or ys.std() == 0:
        return float("nan"), len(m)
    return float(np.corrcoef(xs, ys)[0, 1]), len(m)


def eff_rank(C):
    """相关矩阵的有效维数三种度量。"""
    ev = np.linalg.eigvalsh(C)[::-1]
    ev = np.clip(ev, 0, None)
    tot = ev.sum()
    frac = np.cumsum(ev) / tot
    n90 = int(np.searchsorted(frac, 0.90) + 1)
    n66 = int(np.searchsorted(frac, 0.66) + 1)
    # participation ratio（谱的"有效自由度"）
    pr = tot ** 2 / (ev ** 2).sum()
    return ev, frac, n66, n90, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aaindex", default=None)
    args = ap.parse_args()

    path = args.aaindex
    if path is None:
        for cand in [os.path.join(ROOT, "cache", "aaindex1.txt"),
                     "/private/tmp/claude-501/-Users-ephemeral-Desktop-----/"
                     "203eef00-dafb-4d6e-81af-fe42c0f475a8/scratchpad/aaindex1.txt"]:
            if os.path.exists(cand):
                path = cand
                break
    if not path or not os.path.exists(path):
        print("找不到 aaindex1.txt。用 --aaindex 指定，或放到 cache/ 下。")
        print("来源: https://www.genome.jp/aaindex/  (官方引用 10.1093/nar/gkm998)")
        return 1

    data = parse_aaindex(path, set(SCALES))
    missing = sorted(set(SCALES) - set(data))
    if missing:
        print(f"! 未解析到的条目: {missing}")

    print("=" * 76)
    print("M4.2  干净实验侧链标度的有效维数")
    print("=" * 76)
    print(f"数据源: {path}")
    print("\n[0] 解析自检（解析错位是静默的，必须主动查）")
    if not sanity_check(data):
        print("   !! 解析自检未通过，后面的数字不可用")
        return 1

    print("\n[1] 标度清单与覆盖")
    print(f"{'code':<12} {'类别':<13} {'轴':<12} {'覆盖':>5}  说明")
    print("-" * 76)
    groups = {}
    for code, (desc, kind, axis) in SCALES.items():
        if code not in data:
            continue
        v = data[code]
        n = sum(1 for aa in AA20 if math.isfinite(v[aa]))
        gap = [aa for aa in AA20 if not math.isfinite(v[aa])]
        groups.setdefault(kind, []).append(code)
        note = desc if not gap else f"{desc}  缺: {''.join(gap)}"
        print(f"{code:<12} {kind:<13} {axis:<12} {n:>3}/20  {note}")

    # ---- 只用 experimental 子集算有效维数 ----
    for kind in ("experimental", "experimental+constructed"):
        if kind == "experimental":
            codes = sorted(groups.get("experimental", []))
        else:
            codes = sorted(groups.get("experimental", [])
                           + groups.get("constructed", []))
        if len(codes) < 2:
            continue
        print(f"\n[2{'a' if kind == 'experimental' else 'b'}] "
              f"{kind} 子集  ({len(codes)} 个标度)")
        print("     相关矩阵 |r|（成对完整；括号内为有效样本数）")
        hdr = "            " + "".join(f"{c[:6]:>9}" for c in codes)
        print(hdr)
        C = np.eye(len(codes))
        for i, ci in enumerate(codes):
            row = f"   {ci[:9]:<9}"
            for j, cj in enumerate(codes):
                if i == j:
                    row += f"{'1.00':>9}"
                    continue
                r, n = corr([data[ci][a] for a in AA20],
                            [data[cj][a] for a in AA20])
                C[i, j] = r if math.isfinite(r) else 0.0
                row += f"{r:>9.2f}" if math.isfinite(r) else f"{'na':>9}"
            print(row)

        ev, frac, n66, n90, pr = eff_rank(C)
        print(f"\n     特征值: {np.array2string(ev, precision=2)}")
        print(f"     累计方差: {np.array2string(frac, precision=3)}")
        print(f"     -> 66% 方差需 {n66} 维；90% 需 {n90} 维；"
              f"participation ratio = {pr:.2f}")

    # ---- 与已发表结果对照 ----
    print("\n[3] 与已发表维数分析对照")
    print("  Fauchere et al. 1988 (PMID 3209351): 15 个侧链物化描述符")
    print("    -> 3 个主成分仅解释 66% 方差，且 steric 与 bulk 分成两个成分")
    print("  van Westen et al. 2013 (10.1186/1758-2946-5-41):")
    print("    -> 超过 3 个主成分后，低方差维度仍显著改变氨基酸的相对刻画")
    print("    => n=20 的小样本里，低方差维度不是可安全丢弃的噪声")

    print("\n[4] 对 M4 / M5 的结论")
    print("  transplanted (FAUJ880108 sigma_I) 与 count (氢键供受体数、电荷指示)")
    print("  不计入 d —— 前者是小分子取代基表值搬运，后者是结构式计数而非能量。")
    print("  把它们算进 d 会让嵌入维数与秩估计循环依赖（Q-001）。")
    print("  M4.2: r <= d。若 M5 测出的秩超过上面的 d，说明缺了一条物理轴。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
