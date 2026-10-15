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
# ---- 块报告表（docs/35 §3.3 / docs/30 §2，人类裁定 D-085）---------------
# **只定义行的身份与顺序，不写死数值**：数值由本脚本自身重算，
# 再与文档里印出来的比。少一层硬编码，就少一处会和缓存脱钩的地方
# —— docs/40 §8 把「写死在脚本里、不绑任何缓存」列为当前最高优先的覆盖缺口。
BLOCK_ROWS = [
    ("asa_buried_Rose1985",   ("Rose", "埋藏面积")),
    ("local_polarizability",  ("极化率",)),
    ("molar_refractivity",    ("折射度",)),
    ("asa_RadzickaWolfenden", ("表面积",)),
    ("pi_octanol_FauchereP",  ("辛醇",)),
    ("frac_buried_Rose1985",  ("Rose", "埋藏比例")),
    ("kd_hydropathy",         ("Kyte",)),
    ("sheet_prop_Chou",       ("折叠倾向",)),
]

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

    # NC5  文档解析器自己要能抓到被改坏的一行。
    # **不按字面行文本构造篡改**：上一版写死了旧三列表的那串字符，
    # §3.3 改成块报告的五列之后它匹配不上，于是 NC5 永远失败——
    # 而失败的是自检本身，不是文档。改成从实际解析结果反推要改哪个数。
    txt = open(DOC35, encoding="utf-8").read()
    rows = parse_doc_table(txt, "### 3.3")
    hit5 = rows is not None and len(rows) == len(BLOCK_ROWS)
    if not hit5:
        print(f"   [反向 NC5] 解析到 {0 if rows is None else len(rows)} 行，"
              f"应为 {len(BLOCK_ROWS)} 行")
    if hit5:
        i0 = txt.find("### 3.3")
        seg = txt[i0:i0 + 6000]
        lit = f"{rows[0][1]:.3f}"
        assert seg.count(lit) >= 1, f"首行 r² 字面 {lit} 不在表里"
        tampered = txt[:i0] + seg.replace(lit, "0.999", 1) + txt[i0 + 6000:]
        tr = parse_doc_table(tampered, "### 3.3")
        hit5 = tr is not None and abs(tr[0][1] - 0.999) < 1e-9
    print(f"   [反向 NC5] 把文档表里第 1 行改成 0.999，解析器"
          f"{'读到了改动 ✓' if hit5 else '**读不到（解析器是瞎的）**'}")
    ok &= hit5
    return ok


# ---------------------------------------------------------------- 文档层
_NUM = re.compile(r"[-+]?\d*\.\d+")


def parse_doc_table(txt, anchor):
    """按**表头**定位 `标度` 与 `r²` 两列（不按列位置），返回 [(label, r2), ...]。

    改成表头映射，是因为 §3.3 从三列改成块报告的五列之后，
    按位置写死的旧解析器匹配不到任何东西 —— 于是它既读不到数、
    也读不到被篡改的行，自检 NC5 当场把它判成「瞎的」。
    一个读不到东西的检查器不会报错，只会一直绿。
    """
    i = txt.find(anchor)
    if i < 0:
        return None
    rows, col_lab, col_r2 = [], None, None
    for ln in txt[i:i + 6000].splitlines():
        if not ln.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip().strip("*") for c in ln.strip().strip("|").split("|")]
        if col_lab is None:                       # 还没找到表头
            names = [c.lower() for c in cells]
            if any("标度" in c for c in cells) and any(
                    c.replace(" ", "") in ("r²", "r2") for c in cells):
                col_lab = next(k for k, c in enumerate(cells) if "标度" in c)
                col_r2 = next(k for k, c in enumerate(cells)
                              if c.replace(" ", "") in ("r²", "r2"))
            continue
        if set("".join(cells)) <= set("-: "):     # 分隔行
            continue
        if max(col_lab, col_r2) >= len(cells):
            continue
        lab = cells[col_lab]
        n2 = _NUM.findall(cells[col_r2])
        if not lab or not n2:
            continue
        rows.append((lab, float(n2[0])))
    if col_lab is None:
        return None
    return rows or None


def check_docs(recomputed):
    """recomputed: {scale_key: r2}（本脚本重算所得）。"""
    bad = []
    for path, anchor, name in ((DOC35, "### 3.3", "docs/35 §3.3"),
                               (DOC30, "## 2. ", "docs/30 §2")):
        txt = open(path, encoding="utf-8").read()
        rows = parse_doc_table(txt, anchor)
        print(f"\n   {name}：", end="")
        if rows is None:
            print("**找不到那张表（表头未命中）** —— 检查器失去覆盖")
            bad.append(f"{name}: 表未找到")
            continue
        print(f"读到 {len(rows)} 行")
        if len(rows) != len(BLOCK_ROWS):
            bad.append(f"{name}: 行数 {len(rows)} ≠ {len(BLOCK_ROWS)}")
        for i, (lab, r2) in enumerate(rows[:len(BLOCK_ROWS)]):
            key, kws = BLOCK_ROWS[i]
            if not any(k.lower() in lab.lower() for k in kws):
                bad.append(f"{name} 第 {i+1} 行标签「{lab}」不像 {key}")
                print(f"      {i+1}. {lab:<28} **行序/标签不符**")
                continue
            exp = recomputed.get(key)
            if exp is None:
                bad.append(f"{name} 第 {i+1} 行：重算里没有 {key}")
                continue
            ok = abs(r2 - exp) <= TOL
            if not ok:
                bad.append(f"{name} {key}: 文档 {r2:.3f} vs 重算 {exp:.4f}")
            print(f"      {i+1}. {lab:<28} r²={r2:.3f} vs 重算 {exp:.4f}  "
                  f"{'ok' if ok else '**不符**'}")
        if "本表口径不一致" in txt:
            bad.append(f"{name}: 口径警示块仍在")
            print("      **警示块仍在**")
    return bad


# ------------------------------------------------------------------ main
def recompute_block():
    """重算块表 8 行的 r²（口径 A = 8 成员）。返回 {key: r2}。"""
    u, lam1, _ac = axis("A")
    sc = all_scales()
    out = {}
    for key, _ in BLOCK_ROWS:
        assert key in sc, f"标度 {key} 不在 all_scales() 里"
        _, r2, _n = row(u, sc[key])
        out[key] = r2
    return out


if __name__ == "__main__":
    import sys
    print("=" * 78)
    print("§3.3 / docs/30 §2 化学标度块表：重算 + 核对")
    print("=" * 78)
    if not selftest():
        print("\n**自检未通过 —— 后面的数字一律不可用。**")
        sys.exit(2)
    if "--selftest" in sys.argv:
        sys.exit(0)

    rec = recompute_block()
    print("\n[重算] 口径 A（8 成员）块表 r²")
    for key, _ in BLOCK_ROWS:
        print(f"   {key:<26} {rec[key]:.4f}")

    bad = check_docs(rec)
    print()
    if bad:
        print(f"**{len(bad)} 处不符：**")
        for b in bad:
            print(f"   - {b}")
        sys.exit(1)
    print("两份文档的块表与重算一致，警示块已移除 ✓")
    sys.exit(0)
