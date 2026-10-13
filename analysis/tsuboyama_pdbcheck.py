#!/usr/bin/env python3
"""docs/27 Step 1 的**外部**已知事实锚点：WT 序列 vs RCSB 的该 PDB 序列。

为什么单列一个脚本：`tsuboyama_load.py` 的 T2（位点编号 ↔ 该域 wt 行的
`aa_seq`）是**自洽性**检查 —— 它能抓编号偏移，但抓不到「整张表的域标注
都错了」。CLAUDE.md §10.1 要求用**已知事实**，所以这里拿 RCSB 上该 PDB
条目的序列做外部对账（与 docs/15 §1.1 的 V4 同型）。

判据（在看结果之前写定，照 docs/15 V4 的形式）：
  (a) 该域的 WT 序列必须能以 ≥ 90% 的逐残基同一性对齐到该 PDB 的某条链
      （允许构建体差异：Tsuboyama 把域截短并加 GS 接头，且有些背景是
      故意引入的去稳定点突变 —— 后者本来就该差 1 个残基）；
  (b) **不误伤对照**：把域标注打乱（每个域配一个**别的** PDB 的序列），
      同一判据必须大面积失败。只测命中不测误伤的护栏会被绕过（D-007）。

网络失败不致命，但必须记成 gap（CLAUDE.md §3），不许静默跳过。

运行：python3 analysis/tsuboyama_pdbcheck.py
产物：cache/tsuboyama_pdbcheck.json
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import ROOT                                    # noqa: E402

CACHE = os.path.join(ROOT, "cache", "pdb", "rcsb_fasta")
IDENT_MIN = 0.90


def fasta(pdb):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, pdb + ".fasta")
    if not os.path.exists(p):
        url = f"https://www.rcsb.org/fasta/entry/{pdb}"
        with urllib.request.urlopen(url, timeout=30) as r:
            open(p, "wb").write(r.read())
    chains, cur = [], []
    for line in open(p):
        if line.startswith(">"):
            if cur:
                chains.append("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        chains.append("".join(cur))
    return chains


def best_identity(q, chains):
    """q 作为连续片段在每条链上滑窗，返回最佳逐残基同一性与位置。"""
    best = (0.0, None, None)
    for ci, s in enumerate(chains):
        if len(s) < len(q):
            # 反过来滑：链比 query 短，比对重叠段
            for off in range(0, len(q) - len(s) + 1):
                w = q[off:off + len(s)]
                idn = sum(a == b for a, b in zip(w, s)) / len(s)
                if idn > best[0]:
                    best = (idn, ci, -off)
            continue
        for off in range(0, len(s) - len(q) + 1):
            w = s[off:off + len(q)]
            idn = sum(a == b for a, b in zip(q, w)) / len(q)
            if idn > best[0]:
                best = (idn, ci, off)
    return best


def main():
    info = json.load(open(os.path.join(ROOT, "cache",
                                       "tsuboyama_parse.json")))
    prim = info["primary_natural"]            # parent PDB -> WT_name
    mats = info["matrices"]
    names = sorted(prim.items())
    print("=" * 92)
    print(f"外部已知事实锚点：{len(names)} 个天然结构域的 WT 序列 "
          f"vs RCSB 的该 PDB 条目序列")
    print("=" * 92)
    rows, gaps = [], []
    for pdb, w in names:
        # **必须**用 wt 行的完整连续序列，不能用「按位点拼出来的串」：
        # 412 个候选里有 4 个域恰好缺一个位点，拼串会移码。
        seq = mats[w]["wtseq"] or mats[w]["wtseq_from_muts"]
        try:
            ch = fasta(pdb)
        except (urllib.error.URLError, OSError) as e:
            gaps.append({"pdb": pdb, "name": w, "why": repr(e)[:120]})
            continue
        idn, ci, off = best_identity(seq, ch)
        rows.append({"pdb": pdb, "name": w, "len": len(seq),
                     "identity": idn, "chain": ci, "offset": off,
                     "n_chains": len(ch)})
    ok = [r for r in rows if r["identity"] >= IDENT_MIN]
    print(f"   命中：{len(ok)}/{len(rows)} 个域的 WT 序列对 RCSB 序列的"
          f"逐残基同一性 ≥ {IDENT_MIN:.0%}")
    idn = np.array([r["identity"] for r in rows])
    print(f"   同一性分布：中位 {np.median(idn):.3f}、"
          f"10% 分位 {np.percentile(idn, 10):.3f}、最小 {idn.min():.3f}")
    worst = sorted(rows, key=lambda r: r["identity"])[:12]
    print("   最差 12 个（构建体差异 / 去稳定背景突变会在这里出现）:")
    for r in worst:
        print(f"      {r['pdb']}  {r['name']:<22} len {r['len']:>3}  "
              f"同一性 {r['identity']:.3f}")

    # --- 不误伤对照：打乱域↔PDB 的配对 --------------------------------
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(rows))
    sh = []
    for i, j in enumerate(perm):
        if i == j:
            continue
        seq = mats[rows[i]["name"]]["wtseq"] or mats[rows[i]["name"]]["wtseq_from_muts"]
        try:
            ch = fasta(rows[j]["pdb"])
        except (urllib.error.URLError, OSError):
            continue
        sh.append(best_identity(seq, ch)[0])
    sh = np.array(sh)
    print(f"\n   [不误伤] 打乱配对（n = {len(sh)}）的同一性："
          f"中位 {np.median(sh):.3f}、"
          f"≥ {IDENT_MIN:.0%} 的比例 **{np.mean(sh >= IDENT_MIN):.1%}**"
          f"（真实配对是 {len(ok) / max(1, len(rows)):.1%}）")
    assert np.mean(sh >= IDENT_MIN) < 0.05, "打乱配对也能过 —— 判据无判别力"
    assert len(ok) / max(1, len(rows)) >= 0.90, \
        "真实配对的命中率太低 —— 域标注可疑"
    if gaps:
        print(f"\n   gaps（CLAUDE.md §3）：{len(gaps)} 个域取不到 RCSB 序列")

    out = {"ident_min": IDENT_MIN, "n_checked": len(rows),
           "n_pass": len(ok), "identity_median": float(np.median(idn)),
           "identity_min": float(idn.min()),
           "shuffled_median": float(np.median(sh)),
           "shuffled_pass_frac": float(np.mean(sh >= IDENT_MIN)),
           "rows": rows, "gaps": gaps}
    p = os.path.join(ROOT, "cache", "tsuboyama_pdbcheck.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
