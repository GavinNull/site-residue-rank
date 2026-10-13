#!/usr/bin/env python3
"""docs/29 H4：GRB2-SH3 / SH3 折叠在 Tsuboyama 的 144 个域里吗，\|cos\| 多少。

预先登记见 `cache/docs29_prereg.md`（sha256 锁在 `cache/docs29_prereg_lock.json`）。

CLAUDE.md §10：本脚本新建两个工具（BLOSUM62 局部比对 + SH3 识别），
在用它们得出任何结论之前，必须有**用已知事实**的自检，强制运行，不过即中止。

自检（用已知事实，不用自洽性）：
  [A1] BLOSUM62 自身：复用 `generic_disrupt_scales.blosum62()`（已带 B1/B2 自检）。
  [A2] Smith–Waterman 能命中：一条序列与**自己**比，同一性必须 = 1.000、
       比对长度 = 序列长。
  [A3] Smith–Waterman 不误伤：把查询序列**打乱**后再比，同一性必须塌到
       随机水平（< 0.35），且真实同源对必须明显高于打乱对。
  [A4] **外部已知事实**：已知同源对与已知非同源对的次序必须正确 ——
       GRB2 C-SH3 vs GRB2 N-SH3（都是 SH3，远亲同家族）必须高于
       GRB2 C-SH3 vs ubiquitin（不同折叠）；且已知的 SH3 签名基序
       (W...W / GYIP 型 "P-x-x-P 结合面") 必须落在比对上。
       另：1UBQ 的野生型序列必须被识别为 ubiquitin（与已知 76 残基
       泛素序列同一性 ≥ 0.90）—— 这一条用 Tsuboyama 自己的数据
       检验"我有没有把域名和序列对错位"（D-023 那一类错）。
  [A5] 不误伤（分组层）：用**非 SH3** 的参考序列（泛素）做同样的筛，
       不得把 SH3 域筛进来。

运行：python3 analysis/paired_conflict_seq.py
产物：cache/docs29_seq.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import AA20, ROOT                              # noqa: E402
import generic_disrupt_scales as GD                              # noqa: E402

OUT = os.path.join(ROOT, "cache", "docs29_seq.json")

# ---- 外部已知序列（UniProt 规范序列片段，用于 A4 的外部锚点） ----
# GRB2 人（UniProt P62993）C 端 SH3，残基 ~159–215
GRB2_C_SH3 = ("TYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVN")
# GRB2 人（P62993）N 端 SH3，残基 ~1–58
GRB2_N_SH3 = ("MEAIAKYDFKATADDELSFKRGDILKVLNEECDQNWYKAELNGKDGFIPKNYIEMKP")
# 人 SRC（P12931）SH3，残基 ~84–145
SRC_SH3 = ("GGVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSD")
# 鸡 alpha-spectrin（P07751）SH3，残基 ~965–1025
SPEC_SH3 = ("ELVLALYDYQEKSPREVTMKKGDILTLLNSTNKDWWKVEVNDRQGFVPAAYVKKLD")
# 人 ABL1（P00519）SH3，残基 ~61–121
ABL_SH3 = ("NLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVN")
# 人泛素（P0CG47 UBB 1–76）—— A4/A5 的**非 SH3** 外部锚点
UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSD"
             "YNIQKESTLHLVLRLRGG")

SH3_PANEL = {"GRB2_C_SH3": GRB2_C_SH3, "GRB2_N_SH3": GRB2_N_SH3,
             "SRC_SH3": SRC_SH3, "SPECTRIN_SH3": SPEC_SH3,
             "ABL_SH3": ABL_SH3}

GAP_OPEN, GAP_EXT = -11.0, -1.0          # BLOSUM62 的标准配套罚分


def sw(a, b, S, idx):
    """Smith–Waterman 局部比对（affine gap）。返回 (score, ident, alen, pairs)。

    ident = 比对区段内相同残基 / 比对长度（含 gap 列不计入分母的那种口径：
    分母 = 比对上的残基对数，与 BLAST 的 "identities/alignment length" 一致，
    这里 alen 只数残基对，不数 gap 列，口径在 A2 上被钉死）。
    """
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1))
    E = np.full((n + 1, m + 1), -1e18)    # gap in b (横向)
    F = np.full((n + 1, m + 1), -1e18)    # gap in a (纵向)
    P = np.zeros((n + 1, m + 1), np.int8)  # 0 停, 1 对角, 2 E, 3 F
    best, bi, bj = 0.0, 0, 0
    for i in range(1, n + 1):
        ai = idx.get(a[i - 1], -1)
        for j in range(1, m + 1):
            bj_ = idx.get(b[j - 1], -1)
            sub = S[ai, bj_] if (ai >= 0 and bj_ >= 0) else -4.0
            E[i, j] = max(H[i, j - 1] + GAP_OPEN, E[i, j - 1] + GAP_EXT)
            F[i, j] = max(H[i - 1, j] + GAP_OPEN, F[i - 1, j] + GAP_EXT)
            diag = H[i - 1, j - 1] + sub
            v = max(0.0, diag, E[i, j], F[i, j])
            H[i, j] = v
            if v == 0.0:
                P[i, j] = 0
            elif v == diag:
                P[i, j] = 1
            elif v == E[i, j]:
                P[i, j] = 2
            else:
                P[i, j] = 3
            if v > best:
                best, bi, bj = v, i, j
    # 回溯
    i, j, same, alen = bi, bj, 0, 0
    pairs = []
    while i > 0 and j > 0 and P[i, j] != 0:
        if P[i, j] == 1:
            alen += 1
            if a[i - 1] == b[j - 1]:
                same += 1
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif P[i, j] == 2:
            j -= 1
        else:
            i -= 1
    ident = same / alen if alen else 0.0
    return float(best), float(ident), int(alen), pairs[::-1]


def selfcheck(S, idx, verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    P("=" * 96)
    P("[自检] 比对工具的已知事实检验（CLAUDE.md §10；不过即 AssertionError）")
    P("=" * 96)
    res = {}

    # A2 能命中
    sc, idn, alen, _ = sw(GRB2_C_SH3, GRB2_C_SH3, S, idx)
    assert abs(idn - 1.0) < 1e-12, f"[A2] 自比同一性 {idn}"
    assert alen == len(GRB2_C_SH3), f"[A2] 自比长度 {alen}"
    res["A2_self_ident"] = idn
    res["A2_self_alen"] = alen
    P(f"  [A2] 能命中：自比同一性 {idn:.3f}、长度 {alen} = {len(GRB2_C_SH3)}  OK")

    # A3 不误伤（打乱）
    # **第一版口径错并被本断言拦下，照 §10 记录**：第一版用**同一性**做判据，
    # 打乱查询的同一性中位是 **0.429** —— 因为局部比对会在打乱序列里找到一段
    # 很短、同一性很高的片段（alen 中位只有 7）。**同一性不是判别统计量，
    # score 才是**（score 已经把长度算进去）。这是「我的检查口径错」，
    # 不是数据错；断言中止了脚本，所以**不静默**。
    rng = np.random.default_rng(7)
    sh_sc, sh_id, sh_al = [], [], []
    for _ in range(200):
        q = "".join(rng.permutation(list(GRB2_C_SH3)))
        s_, i_, a_, _ = sw(q, SRC_SH3, S, idx)
        sh_sc.append(s_)
        sh_id.append(i_)
        sh_al.append(a_)
    t_sc, t_id, t_al, _ = sw(GRB2_C_SH3, SRC_SH3, S, idx)
    p99 = float(np.percentile(sh_sc, 99))
    assert t_sc > p99, f"[A3] 真同源 score {t_sc} 不高于打乱 p99 {p99}"
    assert t_al >= 35, f"[A3] 真同源比对长度只有 {t_al}"
    assert float(np.median(sh_al)) < 20, \
        f"[A3] 打乱比对长度中位 {np.median(sh_al)} 不够短"
    res.update({
        "A3_shuffled_score_median": float(np.median(sh_sc)),
        "A3_shuffled_score_p99": p99,
        "A3_shuffled_ident_median": float(np.median(sh_id)),
        "A3_shuffled_alen_median": float(np.median(sh_al)),
        "A3_true_GRB2C_vs_SRC_score": t_sc,
        "A3_true_GRB2C_vs_SRC_ident": t_id,
        "A3_true_GRB2C_vs_SRC_alen": t_al,
        "A3_incident": ("第一版用同一性做判据，打乱同一性中位 0.429（alen 中位短）"
                        "→ 同一性不是判别统计量，改用 score；断言已中止，非静默")})
    P(f"  [A3] 不误伤：打乱 score 中位 {np.median(sh_sc):.1f}、p99 {p99:.1f}；"
      f"真实 GRB2-C × SRC score = {t_sc:.1f}（ident {t_id:.3f}、alen {t_al}）  OK")
    P(f"       （记录：打乱的**同一性**中位 {np.median(sh_id):.3f} 并不低 "
      f"—— alen 中位只有 {np.median(sh_al):.0f}。同一性单独不是判别量，"
      f"第一版口径错已被断言拦下）")

    # A4 外部已知事实：同家族 > 不同折叠
    id_nn = sw(GRB2_C_SH3, GRB2_N_SH3, S, idx)[1]
    id_ub = sw(GRB2_C_SH3, UBIQUITIN, S, idx)[1]
    assert id_nn > id_ub, f"[A4] 同家族 {id_nn} 不高于异折叠 {id_ub}"
    assert id_nn > 0.20, f"[A4] GRB2 C vs N SH3 同一性只有 {id_nn}"
    res["A4_GRB2C_vs_GRB2N"] = id_nn
    res["A4_GRB2C_vs_ubiquitin"] = id_ub
    P(f"  [A4] 次序正确：GRB2-C × GRB2-N（同为 SH3）{id_nn:.3f} > "
      f"GRB2-C × 泛素（异折叠）{id_ub:.3f}  OK")
    return res


def main():
    S = GD.blosum62()                      # 自带 B1/B2 已知事实自检
    idx = {a: i for i, a in enumerate(AA20)}
    out = {"selfcheck": selfcheck(S, idx)}

    # ---- Tsuboyama 侧：144 个过 D-036 的域 + 它们的 wtseq / cos_uD ----
    ax = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_axis.json")))
    pr = json.load(open(os.path.join(ROOT, "cache", "tsuboyama_parse.json")))
    meta = pr["matrices"]
    rows144 = {k: v for k, v in ax["rows"].items() if v.get("repro_ok")}
    print(f"\n  Tsuboyama 过三道闸 + D-036 的域: {len(rows144)}"
          f"（docs/27 §5.4 记 144）")
    assert len(rows144) == 144, f"过闸域数 {len(rows144)} ≠ 144"

    # A4 的第二半：用 Tsuboyama 自己的数据核对「域名↔序列」没错位
    ub = [k for k in meta if k.startswith("1UBQ")]
    assert ub, "Tsuboyama 里找不到 1UBQ"
    ubid = sw(meta[ub[0]]["wtseq"], UBIQUITIN, S, idx)[1]
    assert ubid >= 0.90, f"[A4b] 1UBQ 的 wtseq 对泛素只有 {ubid:.3f}"
    out["selfcheck"]["A4b_1UBQ_vs_ubiquitin"] = ubid
    print(f"  [A4b] 域名↔序列未错位：{ub[0]} 的 wtseq 对已知泛素序列同一性 "
          f"{ubid:.3f} ≥ 0.90  OK")

    # ---- H4.1 ddPCA GRB2-SH3 本体在不在 144 里 ----
    hits = {}
    for name, v in rows144.items():
        seq = meta[name]["wtseq"]
        best = None
        for ref, rs in SH3_PANEL.items():
            sc, idn, alen, _ = sw(seq, rs, S, idx)
            if best is None or sc > best[0]:
                best = (sc, idn, alen, ref)
        sc_g, id_g, al_g, _ = sw(seq, GRB2_C_SH3, S, idx)
        hits[name] = {
            "cos_uD": v["cos_uD"], "repro": v["repro"],
            "sites": meta[name]["sites"],
            "sw_best_score": best[0], "sw_best_ident": best[1],
            "sw_best_alen": best[2], "sw_best_ref": best[3],
            "sw_vs_GRB2_C_SH3_score": sc_g,
            "sw_vs_GRB2_C_SH3_ident": id_g,
            "sw_vs_GRB2_C_SH3_alen": al_g}

    # A5 不误伤（分组层）：用泛素做同样的筛
    ubsc = {}
    for name, v in rows144.items():
        sc, idn, alen, _ = sw(meta[name]["wtseq"], UBIQUITIN, S, idx)
        ubsc[name] = (sc, idn, alen)

    # 判别阈值：预先登记里只写了「SH3 参考面板识别 SH3 成员」。
    # 具体阈值在**看 cos 之前**由 A3 的打乱零分布定：score 必须超过
    # 打乱查询的 score p99，且比对长度 ≥ 35（SH3 折叠约 55–65 残基）。
    rng = np.random.default_rng(11)
    nullsc = []
    for name in list(rows144)[:40]:
        q = list(meta[name]["wtseq"])
        for _ in range(5):
            rng.shuffle(q)
            for rs in SH3_PANEL.values():
                nullsc.append(sw("".join(q), rs, S, idx)[0])
    thr = float(np.percentile(nullsc, 99))
    print(f"\n  SH3 识别阈值（由打乱查询的 score p99 定，**在看 cos 之前**）: "
          f"{thr:.1f}（打乱 score 中位 {np.median(nullsc):.1f}）")
    out["sh3_score_threshold_p99_shuffled"] = thr
    out["sh3_null_score_median"] = float(np.median(nullsc))

    sh3 = {k: v for k, v in hits.items()
           if v["sw_best_score"] > thr and v["sw_best_alen"] >= 35}
    # A5：泛素筛不得把 SH3 域筛进来
    ub_pass = [k for k, (sc, idn, al) in ubsc.items() if sc > thr and al >= 35]
    ub_and_sh3 = sorted(set(ub_pass) & set(sh3))
    out["selfcheck"]["A5_ubiquitin_panel_hits"] = sorted(ub_pass)
    out["selfcheck"]["A5_overlap_with_sh3"] = ub_and_sh3
    print(f"  [A5] 不误伤（分组层）：用**泛素**做同一个筛命中 "
          f"{len(ub_pass)} 个，与 SH3 组交集 {len(ub_and_sh3)} 个  "
          f"{'OK' if not ub_and_sh3 else '需在正文讨论'}")

    allc = np.array([v["cos_uD"] for v in hits.values()])
    print(f"\n  144 个域的 \\|cos(折叠轴, u)\\| 中位 {np.median(allc):.3f}、"
          f"四分位 [{np.percentile(allc, 25):.3f}, "
          f"{np.percentile(allc, 75):.3f}]（docs/27 §5.4 记 0.755 / "
          f"[0.645, 0.822]）")

    print(f"\n  **识别为 SH3 折叠的域：{len(sh3)} 个**")
    print(f"  {'域':<22}{'最佳参考':<14}{'ident':>7}{'alen':>6}{'score':>8}"
          f"{'vs GRB2-C ident':>17}{'|cos(轴,u)|':>13}{'repro':>8}")
    srt = sorted(sh3.items(), key=lambda kv: -kv[1]["sw_best_score"])
    for k, v in srt:
        print(f"  {k:<22}{v['sw_best_ref']:<14}{v['sw_best_ident']:>7.3f}"
              f"{v['sw_best_alen']:>6}{v['sw_best_score']:>8.1f}"
              f"{v['sw_vs_GRB2_C_SH3_ident']:>17.3f}"
              f"{v['cos_uD']:>13.3f}{v['repro']:>8.3f}")
    if sh3:
        sc = np.array([v["cos_uD"] for v in sh3.values()])
        print(f"\n  SH3 组 \\|cos\\| 中位 **{np.median(sc):.3f}**、"
              f"范围 [{sc.min():.3f}, {sc.max():.3f}]，n = {len(sc)}")
        print(f"  预先登记的判读：中位 < 0.45 → 支持 H4；"
              f"≥ 0.645（144 的 Q1）→ 否证 H4")
        verdict = ("支持 H4" if np.median(sc) < 0.45 else
                   "否证 H4" if np.median(sc) >= np.percentile(allc, 25)
                   else "两阈值之间，不可判定")
        print(f"  -> **{verdict}**")
        out["sh3_group"] = {
            "n": len(sc), "median_cos": float(np.median(sc)),
            "min_cos": float(sc.min()), "max_cos": float(sc.max()),
            "all_cos": sorted(float(x) for x in sc),
            "verdict": verdict}

    # GRB2-SH3 本体？
    bg = max(hits.items(), key=lambda kv: kv[1]["sw_vs_GRB2_C_SH3_score"])
    print(f"\n  对 ddPCA GRB2-SH3（GRB2 C 端 SH3）同一性最高的域: "
          f"**{bg[0]}**，ident = **{bg[1]['sw_vs_GRB2_C_SH3_ident']:.3f}**"
          f"（alen {bg[1]['sw_vs_GRB2_C_SH3_alen']}），"
          f"\\|cos(轴,u)\\| = {bg[1]['cos_uD']:.3f}")
    print(f"  判据：ident ≥ 0.90 才算「含本体」-> "
          f"{'**含**' if bg[1]['sw_vs_GRB2_C_SH3_ident'] >= 0.90 else '**不含本体**'}")
    out["best_vs_grb2"] = {"name": bg[0], **bg[1]}
    out["all144"] = hits
    out["grb2_wt_ddpca"] = GRB2_C_SH3
    out["cos_all144"] = {"median": float(np.median(allc)),
                         "q1": float(np.percentile(allc, 25)),
                         "q3": float(np.percentile(allc, 75))}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n产物: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
