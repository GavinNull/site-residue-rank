#!/usr/bin/env python3
"""检验 MP-1 的核心预测：位点内 ΔΔG 是残基多极矩的线性函数。

### 要检验的命题

对固定位点（固定 r、E_L、burial），MP-1 预测

    ΔΔG ≈ c0 + c1·Δq + c2·Δμ + c3·ΔΘ + c4·Δα

其中 Δq/Δμ/ΔΘ/Δα 是**残基固有属性**（已由气相实验值定好，不拟合），
c1..c4 是位点专属系数。即这是一个四描述子的 Hammett 型 LFER。

### 为什么这个检验不需要结构

位点固定 ⇒ 几何量全部进 c_k。所以只要有「同一位点 × 多个残基」的
ΔΔG 表就能检验，不需要逐体系的接触几何。

同理 IC50 可用：同一张表内的系统偏差对所有条目共同，被 c0 吸收。

### 对照与指标

Model A  常规基线：疏水性 π + 体积
Model B  MP-1：Δq + Δμ + ΔΘ + Δα
Model C  嵌套：A + 静电通道（检验「加上静电是否有增量」）

**主指标是留一交叉验证 RMSE，不是样本内 R²** —— B 比 A 多两个描述子，
用样本内 R² 比较会无条件偏向 B。
附加：位点内标签置换检验（n 小，必须有运气天花板）。

用法:
  python3 analysis/site_lfer_test.py --selftest      # 合成数据自检
  python3 analysis/site_lfer_test.py data/series.tsv
"""
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------- 残基描述子 ----------

def load_descriptors():
    """MP-1 的四个通道 + 常规基线的两个描述子。全部来自实验来源。"""
    from multipole_surrogate import RESIDUE_MULTIPOLES, MU_GRADES
    from scale_redundancy import parse_aaindex, AA20
    AA3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
           "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
           "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
           "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
    aai = parse_aaindex(os.path.join(ROOT, "cache", "aaindex1.txt"),
                        {"FAUJ830101", "COHE430101"})
    desc = {}
    for aa1 in AA20:
        r3 = AA3[aa1]
        if r3 not in RESIDUE_MULTIPOLES:
            continue
        q, mu, th, al = RESIDUE_MULTIPOLES[r3]
        if mu is None:
            continue          # 无可用实验偶极矩的残基不参与（ARG）
        desc[r3] = {
            # MP-1 通道
            "q": q, "mu": mu, "theta": th, "alpha": al,
            # 常规基线
            "pi": aai["FAUJ830101"][aa1],      # Fauchere-Pliska 疏水性
            "vol": aai["COHE430101"][aa1],     # Cohn-Edsall 偏比容
            "grade": MU_GRADES.get(r3, "?"),
        }
    return desc


FEATURE_SETS = {
    "A_conventional": ["pi", "vol"],
    "B_mp1": ["q", "mu", "theta", "alpha"],
    "C_nested": ["pi", "vol", "q", "mu", "theta"],
}


# ---------- 回归与 LOO-CV ----------

def design(rows, feats, desc):
    X, y, keep = [], [], []
    for i, (res, ddg) in enumerate(rows):
        if res not in desc:
            continue
        v = [desc[res][f] for f in feats]
        if any(not math.isfinite(x) for x in v):
            continue
        X.append([1.0] + v)
        y.append(ddg)
        keep.append(i)
    return np.array(X), np.array(y), keep


def ols(X, y, ridge=1e-8):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def loo_rmse(X, y):
    """留一交叉验证 RMSE。n <= 参数数+1 时返回 nan（不可评估）。"""
    n, p = X.shape
    if n < p + 2:
        return float("nan")
    errs = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        try:
            b = ols(X[m], y[m])
        except np.linalg.LinAlgError:
            return float("nan")
        errs.append((y[i] - X[i] @ b) ** 2)
    return math.sqrt(float(np.mean(errs)))


def baseline_rmse(y):
    """只用常数项（预测均值）的 LOO RMSE —— 任何模型必须优于它。"""
    n = len(y)
    errs = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        errs.append((y[i] - y[m].mean()) ** 2)
    return math.sqrt(float(np.mean(errs)))


def perm_ceiling(X, y, n_perm=2000, seed=7):
    """位点内标签置换的运气天花板：置换后 LOO RMSE 的 5 分位。"""
    rng = np.random.default_rng(seed)
    best = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        v = loo_rmse(X, yp)
        if math.isfinite(v):
            best.append(v)
    return float(np.quantile(best, 0.05)) if best else float("nan")


def nested_test(rows, desc, n_perm=2000, seed=13):
    """**主检验**：加上静电通道能否胜过纯尺寸+疏水？

    统计量 D = LOO_RMSE(A) - LOO_RMSE(C)，正值表示 C 更好。
    零分布由位点内标签置换给出（不假定分布）。

    为什么用嵌套比较作主检验：
      (a) 它就是本项目的核心问题 —— 静电结构是否携带尺寸/疏水之外的信息；
      (b) 单一预注册检验，没有多重性问题（同时测 A/B/C 三个模型
          会把 FWER 推到约 0.14）；
      (c) 嵌套比较只需分辨增量，功效需求低于分别估计两个模型
          —— 与 docs/05 M5.4 里单边定秩的逻辑相同。
    """
    XA, yA, kA = design(rows, FEATURE_SETS["A_conventional"], desc)
    XC, yC, kC = design(rows, FEATURE_SETS["C_nested"], desc)
    if kA != kC or len(yA) < 8:
        return None
    rA, rC = loo_rmse(XA, yA), loo_rmse(XC, yC)
    if not (math.isfinite(rA) and math.isfinite(rC)):
        return None
    D = rA - rC
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        idx = rng.permutation(len(yA))
        yp = yA[idx]
        a, c = loo_rmse(XA, yp), loo_rmse(XC, yp)
        if math.isfinite(a) and math.isfinite(c):
            null.append(a - c)
    if not null:
        return None
    thr = float(np.quantile(null, 0.95))
    pval = float(np.mean([x >= D for x in null]))
    return {"n": len(yA), "loo_A": rA, "loo_C": rC, "D": D,
            "null_q95": thr, "p": pval, "significant": D > thr}


def evaluate_site(site, rows, desc, verbose=True):
    out = {}
    base = baseline_rmse(np.array([d for _, d in rows]))
    if verbose:
        print(f"\n  [{site}]  n={len(rows)}  "
              f"ΔΔG 范围 {min(d for _, d in rows):.2f}..{max(d for _, d in rows):.2f}"
              f"  常数模型 LOO RMSE={base:.3f}")
        print(f"    {'模型':<16} {'n_eff':>5} {'参数':>5} {'LOO RMSE':>9} "
              f"{'vs 常数':>8} {'置换 5%':>8}")
    for name, feats in FEATURE_SETS.items():
        X, y, _ = design(rows, feats, desc)
        if len(y) < 4:
            continue
        r = loo_rmse(X, y)
        b2 = baseline_rmse(y)
        pc = perm_ceiling(X, y) if math.isfinite(r) else float("nan")
        gain = (b2 - r) / b2 if math.isfinite(r) and b2 else float("nan")
        beats_perm = math.isfinite(r) and math.isfinite(pc) and r < pc
        out[name] = {"loo": r, "n": len(y), "p": X.shape[1],
                     "gain_vs_const": gain, "perm_q05": pc,
                     "beats_perm": beats_perm}
        if verbose:
            mark = "  <= 优于置换天花板" if beats_perm else ""
            print(f"    {name:<16} {len(y):>5} {X.shape[1]:>5} {r:>9.3f} "
                  f"{gain:>+7.1%} {pc:>8.3f}{mark}")
    nt = nested_test(rows, desc)
    if verbose:
        if nt is None:
            print(f"    [主检验] n 不足或描述子缺失，不可评估")
        else:
            print(f"    [主检验 C vs A]  D={nt['D']:+.3f}  "
                  f"零分布95%={nt['null_q95']:+.3f}  p={nt['p']:.3f}  "
                  f"{'显著' if nt['significant'] else '不显著'}")
    out["_nested"] = nt
    return out


# ---------- 自检（硬规则 CLAUDE.md §10）----------

def selftest():
    print("=" * 74)
    print("自检：合成数据上，机制能否认出真模型")
    print("=" * 74)
    desc = load_descriptors()
    residues = sorted(desc)
    print(f"\n可用残基 {len(residues)} 个（ARG 因无实验偶极矩被排除）")
    rng = np.random.default_rng(11)

    print("\n[1] 真模型 = MP-1 通道，噪声 0.3 kcal/mol")
    print("    期望：B 与 C 优于 A；三者都优于常数模型")
    for trial in range(2):
        pick = list(rng.choice(residues, size=12, replace=False))
        c = {"q": -2.0, "mu": -0.5, "theta": -0.08, "alpha": -0.15}
        rows = [(r, sum(c[k] * desc[r][k] for k in c)
                 + rng.normal(0, 0.3)) for r in pick]
        evaluate_site(f"synthetic-MP1-{trial}", rows, desc)

    print("\n[2] 真模型 = 常规描述子（疏水+体积），噪声 0.3")
    print("    期望：A 优于 B —— 若 B 仍然更好，说明机制偏向多参数模型")
    for trial in range(2):
        pick = list(rng.choice(residues, size=12, replace=False))
        rows = [(r, -0.4 * desc[r]["pi"] - 2.0 * desc[r]["vol"]
                 + rng.normal(0, 0.3)) for r in pick]
        evaluate_site(f"synthetic-conv-{trial}", rows, desc)

    print("\n[3] 真模型 = 纯噪声")
    print("    期望：所有模型都不优于置换天花板")
    for trial in range(2):
        pick = list(rng.choice(residues, size=12, replace=False))
        rows = [(r, rng.normal(0, 1.0)) for r in pick]
        evaluate_site(f"synthetic-noise-{trial}", rows, desc)

    print("\n" + "=" * 74)
    print("判据：[1] 中 B/C 应胜 A；[2] 中 A 应胜 B；[3] 中无模型胜置换。")
    print("若 [2] 里 B 仍胜 A，LOO-CV 没有正确惩罚多余参数，机制不可用。")
    return 0


def load_series(path):
    """读 tsv: site_id, residue(3字母), ddG"""
    sites = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            sid, res, ddg = parts[0].strip(), parts[1].strip().upper(), parts[2]
            try:
                sites.setdefault(sid, []).append((res, float(ddg)))
            except ValueError:
                continue
    return sites


def main():
    if "--selftest" in sys.argv or len(sys.argv) < 2:
        return selftest()
    desc = load_descriptors()
    sites = load_series(sys.argv[1])
    print("=" * 74)
    print(f"位点内 LFER 检验  {len(sites)} 个系列")
    print("=" * 74)
    agg = {}
    for sid, rows in sorted(sites.items()):
        if len(rows) < 6:
            print(f"\n  [{sid}] n={len(rows)} < 6，跳过")
            continue
        res = evaluate_site(sid, rows, desc)
        for k, v in res.items():
            agg.setdefault(k, []).append(v)
    print("\n" + "=" * 74)
    print("汇总（胜过置换天花板的位点数 / 总位点数）")
    for k, vs in agg.items():
        w = sum(1 for v in vs if v["beats_perm"])
        print(f"  {k:<16} {w}/{len(vs)}   "
              f"中位 LOO 增益 vs 常数 "
              f"{np.median([v['gain_vs_const'] for v in vs]):+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
