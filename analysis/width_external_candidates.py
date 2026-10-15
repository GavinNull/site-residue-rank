#!/usr/bin/env python3
"""docs/39 §3 / §5：仓库外候选的矩阵几何实测（只量几何，不下任何闸判结论）。

目前实测的唯一一个新取回数据集：Wu lab SARS-CoV-2 spike HR1 × 3 个单抗
（docs/23 §7.2 候选 2 的一半）。其余候选在读出闸或结构闸之前就停了，
不存在可下载的 site×residue 结合量表（逐条理由见 docs/39 §3）。

provenance（CLAUDE.md §3）：
  repo   : github.com/nicwulab/SARS2_HR1_DMS_Abs（main 分支）
  文件   : result/S2HR1_bind_scores.tsv、result/S2HR1_scores_common.tsv
  取回   : 2026-10-15，脚本 analysis/width_external_candidates.py
  论文   : bioRxiv 10.1101/2023.09.27.559757（README 自述；期刊版未核实）
  本地   : cache/dms/wulab_hr1/，sha256 由本脚本写入 PROVENANCE.json

自检（CLAUDE.md §10，用已知事实，不通过中止）：
  H1 野生型残基单值：每个位点的 WT 残基在全表里必须唯一。
  H2 WT 残基序列必须与 repo 自带的参考序列 Fasta/HR1_ref_seq.fa 逐位相符。
  H3 突变类别标注自洽性的**独立**核对：被标 `silent` 的行必须 wt == mut，
     被标 `nonsense` 的行必须 mut 为终止符 —— 用标注去检解析，不是反过来。
  H4 不误伤：合成的完备 12×20 秩 2 矩阵必须能过 G2。
"""
import collections
import hashlib
import json
import os
import re
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import AA20, extract_subspace            # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HR1 = os.path.join(ROOT, "cache", "dms", "wulab_hr1")
RAWBASE = "https://raw.githubusercontent.com/nicwulab/SARS2_HR1_DMS_Abs/main/"
FILES = {
    "S2HR1_bind_scores.tsv": "result/S2HR1_bind_scores.tsv",
    "S2HR1_scores_common.tsv": "result/S2HR1_scores_common.tsv",
    "HR1_ref_seq.fa": "Fasta/HR1_ref_seq.fa",
}


def acquire():
    os.makedirs(HR1, exist_ok=True)
    man = {}
    for local, remote in FILES.items():
        fn = os.path.join(HR1, local)
        if not os.path.exists(fn):
            with urllib.request.urlopen(RAWBASE + remote, timeout=120) as fh:
                raw = fh.read()
            with open(fn, "wb") as out:
                out.write(raw)
        with open(fn, "rb") as fh:
            raw = fh.read()
        man[local] = {"url": RAWBASE + remote, "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()}
    return man


def load_rows():
    fn = os.path.join(HR1, "S2HR1_bind_scores.tsv")
    with open(fn) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, ln.rstrip("\n").split("\t"))) for ln in fh]
    return head, rows


def ref_seq():
    fn = os.path.join(HR1, "HR1_ref_seq.fa")
    with open(fn) as fh:
        return "".join(ln.strip() for ln in fh if not ln.startswith(">"))


def selftest(rows, ref):
    fails = []
    wt_at = {}
    for r in rows:
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z_\*])", r["mut"])
        if not m:
            continue
        wt_at.setdefault(int(m.group(2)), set()).add(m.group(1))
    multi = {k: v for k, v in wt_at.items() if len(v) > 1}
    if multi:
        fails.append(f"H1 野生型多值的位点：{sorted(multi)[:5]}")

    # H2：WT 残基 vs repo 参考序列。编号是 spike 全长编号，参考序列只覆盖
    # HR1 区段，所以用偏移搜索：唯一的一个偏移必须让全部位点逐位相符。
    sites = sorted(wt_at)
    wt_str = "".join(next(iter(wt_at[s])) for s in sites)
    offs = [o for o in range(1, 1400)
            if all(0 <= s - o < len(ref) and ref[s - o] == next(iter(wt_at[s]))
                   for s in sites)]
    if len(offs) != 1:
        fails.append(f"H2 与参考序列相符的偏移有 {len(offs)} 个（应为 1）；"
                     f"WT 串前 20 位 {wt_str[:20]}")
    # H3：用数据自己的类别标注独立检解析
    sil = [r for r in rows if r.get("mut_class") == "silent"]
    non = [r for r in rows if r.get("mut_class") == "nonsense"]
    bad_sil = [r["mut"] for r in sil
               if not re.fullmatch(r"([A-Z])(\d+)\1", r["mut"])]
    bad_non = [r["mut"] for r in non
               if not re.fullmatch(r"[A-Z]\d+[\*_]", r["mut"])]
    if bad_sil:
        fails.append(f"H3 silent 行不是同义串：{bad_sil[:5]}")
    if bad_non:
        fails.append(f"H3 nonsense 行不是终止串：{bad_non[:5]}")

    # H4 不误伤
    rng = np.random.default_rng(1)
    Mc = rng.normal(size=(12, 2)) @ rng.normal(size=(20, 2)).T
    B, _, err = extract_subspace(Mc, np.ones((12, 20), bool), rank=2)
    if B is None:
        fails.append(f"H4 不误伤失败：{err}")

    if fails:
        raise SystemExit("自检不通过：\n  " + "\n  ".join(fails))
    return {"H1_wt_single_valued": "ok",
            "H2_ref_seq_offset": offs[0],
            "H2_n_sites_checked": len(sites),
            "H3_silent_nonsense_consistent":
                f"silent {len(sil)} / nonsense {len(non)} ok",
            "H4_no_false_reject": "ok"}


def geometry(rows, score_col):
    per = collections.defaultdict(dict)
    for r in rows:
        if r.get("mut_class") != "missense":
            continue
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", r["mut"])
        if not m:
            continue
        try:
            v = float(r[score_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        per[int(m.group(2))][m.group(3)] = v
    sites = sorted(per)
    M = np.zeros((len(sites), 20))
    mask = np.zeros((len(sites), 20), bool)
    for i, s in enumerate(sites):
        for mu, v in per[s].items():
            M[i, AA20.index(mu)] = v
            mask[i, AA20.index(mu)] = True
    c = mask.sum(1)
    B, cols, err = extract_subspace(M, mask, rank=2)
    return {
        "score_col": score_col,
        "n_sites": len(sites), "n_obs": int(mask.sum()),
        "subs_per_site_min": int(c.min()),
        "subs_per_site_median": float(np.median(c)),
        "subs_per_site_max": int(c.max()),
        "sites_with_ge4": int((c >= 4).sum()),
        "n_residue_types_used": int(mask.any(0).sum()),
        "G2": "过" if B is not None else f"不过：{err}",
        "value_min": round(float(M[mask].min()), 4),
        "value_max": round(float(M[mask].max()), 4),
        "value_median": round(float(np.median(M[mask])), 4),
    }


def main():
    man = acquire()
    head, rows = load_rows()
    ref = ref_seq()
    st = selftest(rows, ref)
    print("自检：", json.dumps(st, ensure_ascii=False))
    abs_cols = [c for c in head if re.fullmatch(r"A\d+_score", c)]
    out = {"provenance": man, "selftest": st,
           "antibody_score_columns": abs_cols,
           "n_rows": len(rows),
           "mut_class_counts": dict(collections.Counter(
               r.get("mut_class") for r in rows)),
           "geometry": {c: geometry(rows, c) for c in abs_cols},
           "readout_note":
               "三档（low/med/hi）频率加权分 —— 不是滴定曲线拟合的 Kd。"
               "读出闸在此失败，所以**不做任何统计**（docs/25 的做法）。",
           "script": "analysis/width_external_candidates.py"}
    with open(os.path.join(ROOT, "cache", "width_external_candidates.json"),
              "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in
                      ["antibody_score_columns", "mut_class_counts",
                       "geometry"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
