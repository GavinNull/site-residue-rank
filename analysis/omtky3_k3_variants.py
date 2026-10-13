#!/usr/bin/env python3
"""k = 3 结论对「抽取器」与「数据源」两个变体选择的敏感性（CLAUDE.md §5/§6）。

omtky3_k3.py 的基准口径 = docs/13 的路径（cache/skempi_thr4.csv，emsvd 抽取）。
这里逐条替换，并做 §10 要求的数值层交叉核验：
  emsvd  vs  rank_test_power.als_guarded（**绝不**用 als_complete ——
  那次误用产生过 −337% / −1091% 的伪「增益」）
  cache/skempi_thr4.csv  vs  完全未过滤的 cache/skempi_v2.csv
                             （omtky3_panel.build_matrices 的路径）

零分布一律在 k = 3、d = 18 上重新生成（与基准同口径、同次数）。
产物：cache/omtky3_k3_variants.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import (ROOT, align, extract_subspace,       # noqa: E402
                          load_all_dms, load_independent, load_skempi,
                          dedup_by_content)
from subspace_resid_dim import centered_orth, spectrum         # noqa: E402
from omtky3_pr_power import rand_perp                          # noqa: E402
import omtky3_panel as panel                                   # noqa: E402

NNULL = 4000
REF = "starr2020/WH1/huACE2"
KEEP_PDB = ["1CHO", "1PPF", "1R0R"]          # D-046 存活集（剔 3SGB）
NAME_OF = {"1CHO": "chymotrypsin", "1PPF": "HLE-elastase",
           "1R0R": "subtilisin-Carlsberg"}


def pr_and_p(Bs, seed):
    Qs = [centered_orth(B) for B in Bs]
    P = np.mean([Q @ Q.T for Q in Qs], axis=0)
    uu = np.linalg.eigh(P)[1][:, ::-1][:, 0]
    dirs = []
    for Q in Qs:
        c = Q.T @ uu
        c = c / np.linalg.norm(c)
        dirs.append(Q @ np.array([-c[1], c[0]]))
    _, pr, _ = spectrum(dirs)
    rng = np.random.default_rng(seed)
    prn = np.array([spectrum([rand_perp(uu, rng) for _ in range(len(Bs))])[1]
                    for _ in range(NNULL)])
    return float(pr), float(np.mean(prn <= pr)), float(np.median(prn))


def main():
    # 参照子空间（两个变体共用；参照本身不是变体对象）
    mats, _ = dedup_by_content(load_independent() + load_skempi()
                               + load_all_dms())
    ref = None
    thr4 = {}
    for d in mats:
        if d["name"] == REF:
            B, c, _ = extract_subspace(d["M"], d["mask"])
            ref = (B, np.asarray(c))
        if d.get("dataset") == "SKEMPI_OMTKY3" and "pooled" not in d["name"]:
            thr4[d["name"].split("/")[-1]] = d

    rows_full = panel.build_matrices(panel.identify_omtky3(
        panel.read_full(), verbose=False)[0])
    full = {d["pdb"]: d for d in rows_full}

    print("=" * 92)
    print("k = 3 的变体敏感性（零分布全部在 k = 3 / d = 18 上重新生成）")
    print("=" * 92)
    print(f"{'变体':<46}{'cosθ2 (1CHO/1PPF/1R0R)':>30}{'PR':>7}{'单尾p':>8}")
    print("-" * 92)
    out = []
    for src_label, src in (("thr4（docs/13 路径，基准）", thr4),
                           ("完全未过滤 skempi_v2", full)):
        for meth in ("emsvd", "als"):
            Bs, c2 = [], []
            ok = True
            for pdb in KEEP_PDB:
                key = pdb if src is full else pdb
                d = src.get(key)
                if d is None:
                    ok = False
                    break
                B, c, _ = extract_subspace(d["M"], d["mask"], method=meth)
                if B is None or len(c) != 20:
                    ok = False
                    break
                Bs.append(B)
                c2.append(align(B, np.asarray(c), ref[0], ref[1])["cos"][1])
            if not ok:
                print(f"{src_label + ' / ' + meth:<46}  ——（抽取失败）")
                continue
            pr, p1, nullmed = pr_and_p(Bs, seed=90210 + len(out))
            lab = f"{src_label} / {meth}"
            print(f"{lab:<46}" + f"{'/'.join(f'{x:.3f}' for x in c2):>30}"
                  f"{pr:>7.2f}{p1:>8.4f}")
            out.append({"source": src_label, "method": meth,
                        "cos2": c2, "pr": pr, "p_one_sided": p1,
                        "pr_null_med": nullmed})
    prs = [r["pr"] for r in out]
    ps = [r["p_one_sided"] for r in out]
    print("-" * 92)
    print(f"   PR 跨 {len(out)} 个变体：{min(prs):.2f}–{max(prs):.2f}"
          f"（全距 {max(prs)-min(prs):.2f}）")
    print(f"   单尾 p 跨变体：{min(ps):.4f}–{max(ps):.4f}  -> "
          + ("**全部不显著（都 > 0.05）**" if min(ps) > 0.05 else
             "跨变体结论不一致 —— 必须报告"))
    print("   emsvd 与 als_guarded 的交叉核验：PR 差 "
          f"{abs(out[0]['pr']-out[1]['pr']):.2f}（同一数据源内）")
    p = os.path.join(ROOT, "cache", "omtky3_k3_variants.json")
    json.dump({"nnull": NNULL, "variants": out}, open(p, "w"),
              indent=1, default=float)
    print(f"\n写入 {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
