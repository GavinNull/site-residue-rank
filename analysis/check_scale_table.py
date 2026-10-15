#!/usr/bin/env python3
"""§3.3 / docs/30 §2 的五行化学标度表：重算 + 机械核对（D-077 的加固）。

### 为什么是一个单独的检查器，而不是塞进 make_figures.py

`make_figures.py --check` 断言的是 `EXPECT` 里 10 个**图的**汇总量。
D-077 暴露的问题是：**它的 PASS 被当成了它从未覆盖过的东西也被检查过的证据。**
修这个问题有两条路：

1. 把表值加进 `make_figures.py` 的 `EXPECT`。
   但 §3.3 的表**没有对应的图**，`make_figures.py` 的契约（「每张图在画之前
   先核对它自己的汇总量」）对它无处挂靠；更要紧的是：在人类按
   docs/40 §7 应用替换文本之前，表里的值是**已知错的**，
   把它加进去会让那条一直绿的门禁立刻变红并长期红着——
   而一条长期红的门禁的下一步总是放宽 TOL。
2. 单独一条门禁，自己说清自己覆盖什么。本文件走这条。

所以：`make_figures.py` 继续只管图；本文件管表。
两者都是**一条命令**，都在不一致时非零退出。
`make_figures.py` 的 docstring 里已加一行指回本文件，
以免「门禁通过」再被读成比实际更强的背书。

### 口径（不是本文件决定的）

- 共识轴 `u`：先家族内平均 P，再家族等权 —— 原样复用
  `gate_consequences.fam_avg_eigs` 的构造（本文件就地重算一遍并与
  `cache/gate_consequences.json` 的 `u` 比对，不是读了就用）。
- 相关的支撑集：**每条标度自己的有限位**（`RADA880106` 缺 Pro → n = 19），
  与 `gate_consequences.py:231` 的 `idx` 口径逐字相同。
- 口径 A = 8 成员（冠状病毒 5 + OMTKY3 3，D-046 剔除 3SGB 之后）= **正文应当引的那个**。
  口径 D = 9 成员（含 3SGB）= 历史对照，并列报出以便审计。

用法:
  python3 analysis/check_scale_table.py             # 全部核对（含文档核对）
  python3 analysis/check_scale_table.py --selftest  # 只跑检查器自己的双向自检
  python3 analysis/check_scale_table.py --no-docs   # 只核对「表值 ↔ 缓存」这一半
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subspace_cond_null as CN  # noqa: E402
import gate_calib_lib as G  # noqa: E402
from subspace_lib import AA20, ROOT, extract_subspace, load_scales  # noqa: E402
from scale_redundancy import parse_aaindex  # noqa: E402

TOL = 0.0015

# 正文应当写的值 = 口径 A（8 成员）。顺序即表里的行序。
# (键, 表里的关键词, r, r², n)
EXPECT_8 = [
    ("sheet_prop_Chou",       ("折叠倾向", "sheet"), +0.827153, 0.684182, 20),
    ("local_polarizability",  ("极化率",),           +0.708622, 0.502145, 20),
    ("molar_refractivity",    ("折射度",),           +0.676707, 0.457932, 20),
    ("asa_RadzickaWolfenden", ("表面积",),           +0.610395, 0.372583, 19),
    ("kd_hydropathy",         ("Kyte",),             +0.557516, 0.310824, 20),
]
# 历史 9 成员口径（docs/21 §5.2 原构造）。只用于审计并列，不是正文口径。
EXPECT_9 = [
    ("sheet_prop_Chou",       +0.779052, 0.606922, 20),
    ("local_polarizability",  +0.753762, 0.568157, 20),
    ("molar_refractivity",    +0.720368, 0.518930, 20),
    ("asa_RadzickaWolfenden", +0.656253, 0.430667, 19),
    ("kd_hydropathy",         +0.472351, 0.223115, 20),
]

# 已知事实（教科书值 / AAindex 原始字段），用来验解析器而不是验自洽
KNOWN_MR = {"G": 0.00, "W": 42.53}              # 摩尔折射度 最小/最大
KNOWN_SHEET = {"V": 1.70, "E": 0.37, "P": 0.55, "G": 0.75}   # Chou-Fasman P_β
DOC35 = os.path.join(ROOT, "docs", "35-paper-draft.md")
DOC30 = os.path.join(ROOT, "docs", "30-manuscript.md")


# ---------------------------------------------------------------- 数值层
def axis(label_prefix, by=None):
    """就地重算该场景的共识一轴，并与缓存的 u 比对（不是直接读缓存）。

    返回 (u, lam1, 与缓存的 |cos|)。u 的符号对齐到缓存，
    因为特征向量符号任意而表里的 r 带号。
    """
    g = json.load(open(os.path.join(ROOT, "cache", "gate_consequences.json"),
                       encoding="utf-8"))
    rec = [c for c in g["consensus"] if c["label"].startswith(label_prefix)]
    assert len(rec) == 1, f"场景 {label_prefix} 在缓存里不唯一：{len(rec)} 条"
    rec = rec[0]
    if by is None:
        by = evidence()
    fam = {k: [n for n in v if n in by] for k, v in rec["families"].items()}
    fam = {k: v for k, v in fam.items() if v}
    Ps = [np.mean([CN.proj(by[n])[0] for n in v], axis=0) for v in fam.values()]
    w, V = np.linalg.eigh(np.mean(Ps, axis=0))
    u = V[:, ::-1][:, 0]
    lam1 = float(w[::-1][0])
    uc = np.array(rec["u"], float)
    ac = abs(float(u @ uc))
    assert abs(lam1 - rec["eigs"][0]) <= 1e-9, (
        f"重算 λ1 {lam1:.9f} ≠ 缓存 {rec['eigs'][0]:.9f}")
    assert ac > 1 - 1e-9, f"重算的 u 与缓存的 u 不同向：|cos| = {ac:.12f}"
    if u @ uc < 0:
        u = -u
    return u, lam1, ac


_EV = {}


def evidence():
    if not _EV:
        for nm, d in G.load_evidence().items():
            B, c, _ = extract_subspace(d["M"], d["mask"])
            if B is not None and len(c) == 20:
                _EV[nm] = B
    return _EV


def row(u, s):
    """r / r² / n，口径与 gate_consequences.py:231 逐字相同。"""
    idx = np.array([i for i in range(20) if np.isfinite(s[i])])
    assert len(idx) >= 10 and s[idx].std() > 0
    r = float(np.corrcoef(u[idx], s[idx])[0, 1])
    return r, r * r, int(len(idx))


def boot_r2(u, s, nb=4000, seed=7):
    """r² 的 bootstrap 区间（对**残基**重抽；CLAUDE.md §5 要求 ≥2000）。

    **这个区间只含残基抽样的不确定性**，不含 u 自身的不确定性 ——
    后者由留一成员敏感性度量，见 docs/40 §3，且后者大得多。
    """
    idx = np.array([i for i in range(20) if np.isfinite(s[i])])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nb):
        k = idx[rng.integers(0, len(idx), len(idx))]
        x, y = u[k], s[k]
        if x.std() == 0 or y.std() == 0:
            continue
        out.append(np.corrcoef(x, y)[0, 1] ** 2)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)


def all_scales():
    """load_scales() 再加 Rose 1985 两条（AAindex 里有，仓库从未载入）。"""
    sc = load_scales(verbose=False)
    ai = parse_aaindex(os.path.join(ROOT, "cache", "aaindex1.txt"),
                       {"ROSG850101", "ROSG850102"})
    for code, nm in (("ROSG850101", "asa_buried_Rose1985"),
                     ("ROSG850102", "frac_buried_Rose1985")):
        sc[nm] = np.array([ai[code].get(a, np.nan) for a in AA20], float)
    return sc


# ------------------------------------------------------- 自检（双向，强制）
def known_facts(verbose=True):
    """用已知事实验解析器。**正向**：四组教科书/原始字段值必须命中。"""
    sc = load_scales(verbose=verbose)      # 内含 scale_redundancy.sanity_check
    ai = parse_aaindex(os.path.join(ROOT, "cache", "aaindex1.txt"),
                       {"MCMT640101", "CHOP780202", "RADA880106",
                        "ROSG850101", "ROSG850102"})
    mr = ai["MCMT640101"]
    gmin = min((k for k in mr if np.isfinite(mr[k])), key=lambda k: mr[k])
    wmax = max((k for k in mr if np.isfinite(mr[k])), key=lambda k: mr[k])
    assert gmin == "G" and wmax == "W", f"折射度 极小={gmin} 极大={wmax}"
    for a, v in KNOWN_MR.items():
        assert abs(mr[a] - v) < 5e-3, f"折射度 {a} = {mr[a]} ≠ {v}"
    for a, v in KNOWN_SHEET.items():
        got = ai["CHOP780202"][a]
        assert abs(got - v) < 5e-3, f"CHOP780202 {a} = {got} ≠ 教科书 {v}"
    ra = ai["RADA880106"]
    assert not np.isfinite(ra["P"]), "RADA880106 的 Pro 未被判为缺失（D-023 复发）"
    nfin = sum(1 for a in AA20 if np.isfinite(ra.get(a, np.nan)))
    assert nfin == 19, f"RADA880106 有限位 = {nfin}，应为 19"
    r1 = ai["ROSG850101"]
    assert min(r1, key=lambda k: r1[k]) == "G", "ROSG850101 最小不是 Gly"
    assert max(r1, key=lambda k: r1[k]) == "W", "ROSG850101 最大不是 Trp"
    if verbose:
        print("   [正向] 折射度 G=0.00/W=42.53；CHOP780202 V1.70/E0.37/P0.55/G0.75；"
              "RADA880106 Pro 缺失、n=19；ROSG850101 min G / max W  —— 全中 ✓")
    return sc


def selftest(verbose=True):
    """§10 要求 2：既测能命中，也测不误伤。**植入的错误必须在实际扫描的区间内真的错。**"""
    ok = True
    sc = known_facts(verbose=verbose)
    u, _, _ = axis("A.")

    # NC1  残基顺序错位（D-023 的第一种形态）：按 I 区块的行主序读进字母序
    ai = parse_aaindex(os.path.join(ROOT, "cache", "aaindex1.txt"),
                       {"MCMT640101"})
    mr = ai["MCMT640101"]
    half = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
            "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    bad = {a: mr[b] for a, b in zip(sorted(AA20), half)}
    gmin = min(bad, key=lambda k: bad[k])
    wmax = max(bad, key=lambda k: bad[k])
    hit1 = not (gmin == "G" and wmax == "W")
    print(f"   [反向 NC1] 行主序错位 → 折射度 极小={gmin} 极大={wmax}  "
          f"{'被抓到 ✓' if hit1 else '**未被抓到**'}")
    ok &= hit1

    # NC2  缺失标记读成 0（D-023 的第二种形态，且该位在原文件里字面就是 "0."）
    s = sc["asa_RadzickaWolfenden"]
    good = row(u, s)[1]
    z = np.where(np.isfinite(s), s, 0.0)
    bad2 = float(np.corrcoef(u, z)[0, 1] ** 2)
    hit2 = abs(bad2 - good) > TOL
    print(f"   [反向 NC2] Pro 读成 0 → r² {good:.4f} → {bad2:.4f}"
          f"（Δ = {bad2 - good:+.4f}，容差 {TOL}）  "
          f"{'被抓到 ✓' if hit2 else '**未被抓到**'}")
    ok &= hit2

    # NC3  不误伤（D-007）：小于容差的扰动必须**不**报警
    hit3 = abs((good + TOL * 0.3) - good) <= TOL
    print(f"   [反向 NC3] 扰动 {TOL * 0.3:.5f}（小于容差）  "
          f"{'未误报 ✓' if hit3 else '**误报**'}")
    ok &= hit3

    # NC4  打乱残基标签后，没有一条标度还能解释该轴
    rng = np.random.default_rng(3)
    names = [k for k in sc if np.isfinite(sc[k]).sum() >= 10
             and np.nanstd(sc[k]) > 0]
    worst = 0.0
    for _ in range(200):
        v = rng.permutation(u)
        worst = max(worst, max(abs(row(v, sc[k])[0]) for k in names))
    hit4 = worst < 0.90
    print(f"   [反向 NC4] 置换 u 200 次 × {len(names)} 条标度，最大 |r| = "
          f"{worst:.3f}  {'未出现伪强相关 ✓' if hit4 else '**伪相关过强**'}")
    ok &= hit4

    # NC5  文档解析器自己要能抓到被改坏的一行
    txt = open(DOC35, encoding="utf-8").read()
    rows = parse_doc_table(txt, "### 3.3")
    hit5 = rows is not None and len(rows) == 5
    if hit5:
        tampered = txt.replace("| +0.779 | **0.607** |", "| +0.999 | **0.999** |")
        tr = parse_doc_table(tampered, "### 3.3")
        hit5 = tr is not None and abs(tr[0][1] - 0.999) < 1e-9
    print(f"   [反向 NC5] 把文档表里第 1 行改成 0.999，解析器"
          f"{'读到了改动 ✓' if hit5 else '**读不到（解析器是瞎的）**'}")
    ok &= hit5
    return ok


# ---------------------------------------------------------------- 文档层
_NUM = re.compile(r"[-+]?\d*\.\d+")


def parse_doc_table(txt, anchor):
    """从 anchor 开始找第一张 `| 标度 | r | r² |` 表，返回 [(label, r, r2), ...]。"""
    i = txt.find(anchor)
    if i < 0:
        return None
    seg = txt[i:i + 4000].splitlines()
    rows, started = [], False
    for ln in seg:
        if ln.startswith("|") and "|" in ln[1:]:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            if set(cells[0]) <= set("-: ") or cells[1] in ("r",):
                started = True
                continue
            if not started:
                continue
            n1 = _NUM.findall(cells[1])
            n2 = _NUM.findall(cells[2])
            if not n1 or not n2:
                continue
            rows.append((cells[0], float(n1[0]), float(n2[0])))
        elif rows:
            break
    return rows or None


def check_docs():
    bad = []
    for path, anchor, name in ((DOC35, "### 3.3", "docs/35 §3.3"),
                               (DOC30, "## 2. ", "docs/30 §2")):
        txt = open(path, encoding="utf-8").read()
        rows = parse_doc_table(txt, anchor)
        print(f"\n   {name}：", end="")
        if rows is None:
            print("**找不到那张表** —— 锚点或表头被改过，检查器失去覆盖")
            bad.append(f"{name}: 表未找到")
            continue
        print(f"读到 {len(rows)} 行")
        if len(rows) != len(EXPECT_8):
            bad.append(f"{name}: 行数 {len(rows)} ≠ {len(EXPECT_8)}")
        for i, (lab, r, r2) in enumerate(rows[:len(EXPECT_8)]):
            key, kws, er, er2, _ = EXPECT_8[i]
            tag = "ok"
            if not any(k.lower() in lab.lower() for k in kws):
                bad.append(f"{name} 第 {i+1} 行标签 “{lab}” 不像 {key}")
                tag = "**行序/标签不符**"
            d1, d2 = abs(r - er), abs(r2 - er2)
            if d1 > 0.0006 or d2 > 0.0006:
                bad.append(f"{name} 第 {i+1} 行（{key}）: 文档 {r:+.3f}/{r2:.3f}"
                           f" vs 8 成员口径 {er:+.3f}/{er2:.3f}")
                tag = "**与 8 成员口径不符**"
            print(f"      {lab[:34]:<34} {r:+.3f} / {r2:.3f}   {tag}")
        warn = "口径不一致，待重算" in txt
        print(f"      [警示块] {'仍在' if warn else '已移除'}")
        if warn and not bad:
            bad.append(f"{name}: 表值已对，但警示块还没撤")
    return bad


# ---------------------------------------------------------------- 主流程
def main():
    only_self = "--selftest" in sys.argv
    print("=" * 78)
    print("§3.3 / docs/30 §2 五行化学标度表：重算 + 核对")
    print("=" * 78)
    print("\n[自检] 解析器与检查器自身（强制，不通过即中止）")
    if not selftest(verbose=True):
        print("\n**自检未通过 —— 后面的数字一律不可用。**")
        return 2
    print("   自检通过（5 条反向全部命中，且小扰动未误报）")
    if only_self:
        print("\n--selftest：只跑自检，已通过。")
        return 0

    sc = all_scales()
    bad = []
    for prefix, exp, cal in (("A.", EXPECT_8, "8 成员（冠状病毒 5 + OMTKY3 3）"),
                             ("D.", EXPECT_9, "9 成员（含 3SGB，历史对照）")):
        u, lam1, ac = axis(prefix)
        print(f"\n[口径 {prefix[0]}] {cal}   λ1 = {lam1:.6f}"
              f"（就地重算，与缓存 u 的 |cos| = {ac:.12f}）")
        print(f"   {'标度':<24}{'r':>10}{'r²':>9}{'n':>4}"
              f"{'bootstrap r² 95%':>22}")
        for item in exp:
            key = item[0]
            er, er2, en = item[-3], item[-2], item[-1]
            r, r2, n = row(u, sc[key])
            lo, hi = boot_r2(u, sc[key])[:2]
            flag = ""
            if abs(r - er) > TOL or abs(r2 - er2) > TOL or n != en:
                flag = "  << 与写定值不符"
                bad.append(f"口径 {prefix[0]} {key}: 重算 {r:+.6f}/{r2:.6f}/n={n}"
                           f" vs 写定 {er:+.6f}/{er2:.6f}/n={en}")
            print(f"   {key:<24}{r:>+10.6f}{r2:>9.6f}{n:>4}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>22}{flag}")

    if "--no-docs" not in sys.argv:
        print("\n[文档核对] 两份正文里的那张表是否就是 8 成员口径")
        bad += check_docs()

    print("\n" + "=" * 78)
    if bad:
        print(f"**不一致 {len(bad)} 处：**")
        for b in bad:
            print("   - " + b)
        print("\n（若这是 D-077 的待办：替换文本在 docs/40-scale-table.md §7，"
              "表值归入人类裁定 CLAUDE.md §9.1，检查器不自行改正文。）")
        return 1
    print("全部断言通过：两个口径的表值都从缓存复现，且两份正文与 8 成员口径一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
