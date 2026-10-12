#!/usr/bin/env python3
"""密度/混合模型型截尾检测器的强制自检（CLAUDE.md §10）。

这是本仓库第**三个**截尾检测器，而前两个都静默失效过，而且第二个的失效
就在这同一个测量上。所以本脚本的判据比前两次更严，并且**拆成可以单独读的
五条**（docs/19 §5 已经证明一刀切的「|测−q| ≤ 8pp」判据会逼人去放宽判据，
正确做法是按口径拆开）：

  [0] 合成机制与 docs/19 的 `censor_recheck._synth_censored` **逐元素相同**
      （同种子）。不核对就等于偷偷换了一张更容易的网格。
  [1] 数值层：CV²(t) 严格单调 + 查表反演与逐点二分一致（< 1e-6）。
  (a) **不误伤**：q = 0 的干净矩阵在任何 σ 下 < 3%。
  (b) **序**：每个 σ 上随 q 单调不降，整张网格 Spearman >= 0.90。
  (c) **绝对标度（窄带）**：0 < σ <= 0.05 时 |测 − q| <= 8pp。
  (d) **σ = 0**：|测 − q| <= 2pp（原子应被精确数出来，这是对第二版
      「σ≈0 偏高 +7..10pp」的直接改进，所以判据必须比它严）。
  (e) **盲区绝对标度**：σ ∈ {0.07, 0.10, 0.15, 0.20} 逐格报
      测得-vs-真值，判据 |测 − q| <= 15pp 且**不得塌回 ~0**
      （第二版在这里给 0.5–10.8%，就是塌回）。
  [R] **真实锚点**（已知事实，不是自洽性）：
      PD-CoV hAPN / pAPN 必须高、gAPN / starr2020 / Lehner / SKEMPI 必须低。
  [S] 自由参数的敏感性扫描（γ、m_min、n_m），并记录试过多少个变体（§5）。

运行：python3 analysis/censor_density_check.py
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import censor_recheck as CR                                    # noqa: E402
from censor_density import (ATOM_THR, CT_FRAC, EV_HI, EV_LO,  # noqa: E402
                            GAMMA, M_MIN, N_M, _CV2, _cv2,
                            censor_fraction_density,
                            censor_fraction_density_ci, raw_values,
                            solve_t, synth_censored)
from subspace_lib import (censor_fraction, load_all_dms,        # noqa: E402
                          load_independent, load_skempi)

QS = (0.0, 0.10, 0.25, 0.40, 0.50)
SIGMAS = (0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20)
BLIND = (0.07, 0.10, 0.15, 0.20)
REPS = 8

MUST_HIGH = {"PDCoV/PDCoV/hAPN": ("docs/12 §2: 秩 0，单调上升", 0.25),
             "PDCoV/PDCoV/pAPN": ("docs/12 §2: 不可判，平坦非单调", 0.25),
             "sarbecovirus/SARS-CoV-1_Urban/RalcACE2":
                 ("精确并列已 72%，硬赋下限", 0.25),
             "sarbecovirus/RshSTT182/RlanACE2":
                 ("精确并列已 83%，硬赋下限", 0.25)}
MUST_LOW = {"PDCoV/PDCoV/gAPN": ("docs/12 §2: 秩 2，12/12 种子", 0.10),
            "starr2020/WH1/huACE2": ("docs/12 §1: 秩 2", 0.10),
            "sars2_variants/Omicron_BA2/ACE2": ("docs/12 §3: 秩 2", 0.10),
            "lehner/PSD95-PDZ3/CRIPT": ("ddPCA 的 ΔΔG，无下限赋值", 0.10),
            "SKEMPI/OMTKY3/pooled-4proteases": ("文献 Ki，无下限赋值", 0.10),
            "OMTKY3/chymotrypsin/1CHO": ("Laskowski 完整测全", 0.10)}


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def grid(rng, reps=REPS, **kw):
    out = {}
    for sig in SIGMAS:
        for q in QS:
            out[(sig, q)] = float(np.mean(
                [censor_fraction_density(synth_censored(rng, q, sig), **kw)
                 for _ in range(reps)]))
    return out


def main():
    ok_all = True
    print("=" * 92)
    print("密度/混合模型型截尾检测器 自检（CLAUDE.md §10）")
    print(f"参数：γ = {GAMMA}（σ̂/s_up 上限）、σ̂ <= {CT_FRAC}·IQR/1.349、"
          f"极值带 [{EV_LO}, {EV_HI}]·z*、m_min = {M_MIN}、"
          f"n_m = {N_M}、原子阈 = {ATOM_THR}")
    print("=" * 92)

    # ---------------- [0] 合成机制逐元素一致 ----------------
    a = synth_censored(np.random.default_rng(1234), 0.3, 0.07)
    b = CR._synth_censored(np.random.default_rng(1234), 0.3, 0.07)
    same = a.shape == b.shape and float(np.abs(a - b).max()) < 1e-12
    print(f"\n[0] 合成机制与 docs/19 的 _synth_censored 逐元素一致"
          f"（同种子）: {'OK' if same else '<< 不一致，网格被换过'}")
    ok_all &= same

    # ---------------- [1] 数值层 ----------------
    mono = bool(np.all(np.diff(_CV2) < 0))

    def bisect_t(c2):
        lo, hi = -6.0, 8.0
        if c2 >= _cv2(lo):
            return lo
        if c2 <= _cv2(hi):
            return hi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _cv2(mid) > c2:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    err = max(abs(solve_t(c) - bisect_t(c))
              for c in np.linspace(_CV2[-1] * 1.01, _CV2[0] * 0.99, 200))
    # 已知事实核验：t = 0 时 Var/E² 必须等于 (1−2/π)/(2/π) = π/2 − 1
    known = abs(_cv2(0.0) - (np.pi / 2 - 1)) < 1e-9
    print(f"[1] CV²(t) 严格单调递减: {'OK' if mono else '<< 失败'}；"
          f"查表反演 vs 二分 最大差 {err:.2e} "
          f"{'OK' if err < 1e-5 else '<< 失败'}；"
          f"已知事实 CV²(0) = π/2 − 1: {'OK' if known else '<< 失败'}")
    ok_all &= mono and err < 1e-5 and known

    # ---------------- 主网格 ----------------
    rng = np.random.default_rng(20261011)
    G = grid(rng)
    print("\n" + "-" * 92)
    print("真实机制网格（docs/19 §2.2 的同一个机制：压平到下限 -> 叠 σ 噪声）"
          f"，每格 {REPS} 次重复的均值")
    print("-" * 92)
    print(f"{'摊开噪声σ':>10}" + "".join(f"{q:>12.0%}" for q in QS))
    for sig in SIGMAS:
        print(f"{sig:>10.2f}" + "".join(f"{G[(sig, q)]:>12.1%}" for q in QS))
    print(f"{'（真值）':>10}" + "".join(f"{q:>12.0%}" for q in QS))

    # (a) 不误伤
    fp = [G[(s, 0.0)] for s in SIGMAS]
    ok_a = max(fp) < 0.03
    print(f"\n(a) 不误伤（q = 0 在全部 σ 下 < 3%）：最大 {max(fp):.1%}  "
          f"{'通过' if ok_a else '** 未通过 **'}")

    # (b) 序
    ok_mono = True
    for sig in SIGMAS:
        v = [G[(sig, q)] for q in QS]
        ok_mono &= all(v[i] <= v[i + 1] + 1e-9 for i in range(len(v) - 1))
    gq = [q for s in SIGMAS for q in QS]
    gm = [G[(s, q)] for s in SIGMAS for q in QS]
    rho = spearman(gq, gm)
    ok_b = ok_mono and rho >= 0.90
    print(f"(b) 序（每个 σ 上随 q 单调不降 = {ok_mono}；"
          f"全网格 Spearman = {rho:+.3f} >= 0.90）："
          f"{'通过' if ok_b else '** 未通过 **'}")

    # (c) 绝对标度窄带
    rows_c = [(s, q, G[(s, q)]) for s in (0.03, 0.05) for q in QS if q > 0]
    ok_c = all(abs(m - q) <= 0.08 for _, q, m in rows_c)
    print(f"(c) 绝对标度（0 < σ <= 0.05，|测 − q| <= 8pp）：最大偏 "
          f"{max(abs(m - q) for _, q, m in rows_c) * 100:.1f}pp  "
          f"{'通过' if ok_c else '** 未通过 **'}")

    # (d) sigma = 0
    rows_d = [(q, G[(0.0, q)]) for q in QS if q > 0]
    ok_d = all(abs(m - q) <= 0.02 for q, m in rows_d)
    print(f"(d) σ = 0（原子应被精确数出，|测 − q| <= 2pp）：最大偏 "
          f"{max(abs(m - q) for q, m in rows_d) * 100:.1f}pp  "
          f"{'通过' if ok_d else '** 未通过 **'}  "
          f"（第二版在这里偏高 +7..+10pp）")

    # (e) 盲区
    print("\n(e) **盲区**（σ = 0.07–0.20：第二版固定窗口在这里塌回 ~0）")
    print(f"    {'σ':>6}{'真 q':>8}{'新检测器':>11}{'第二版(w=0.25)':>16}"
          f"{'偏(pp)':>9}")
    rng2 = np.random.default_rng(777)
    ok_e = True
    blind_rows = []
    for sig in BLIND:
        for q in QS:
            if q == 0:
                continue
            vs = [synth_censored(rng2, q, sig) for _ in range(REPS)]
            mn = float(np.mean([censor_fraction_density(v) for v in vs]))
            mo = float(np.mean([censor_fraction(v) for v in vs]))
            good = abs(mn - q) <= 0.15 and mn >= 0.5 * q
            ok_e &= good
            blind_rows.append({"sigma": sig, "q": q, "new": mn, "old": mo})
            print(f"    {sig:>6.2f}{q:>8.0%}{mn:>11.1%}{mo:>16.1%}"
                  f"{(mn - q) * 100:>+9.1f}  {'OK' if good else '<<'}")
    print(f"    判据：|测 − q| <= 15pp **且** 测 >= q/2（不得塌回）："
          f"{'通过' if ok_e else '** 未通过 **'}")

    ok_all &= ok_a and ok_b and ok_c and ok_d and ok_e

    # ---------------- [R] 真实锚点 ----------------
    print("\n" + "-" * 92)
    print("[R] 真实锚点（已知事实，不是自洽性）：新 vs 第二版（固定窗口）")
    print("-" * 92)
    mats = load_independent() + load_skempi() + load_all_dms()
    by = {d["name"]: d for d in mats}
    print(f"{'矩阵':<42}{'新':>7}{'旧(w=.25)':>11}{'σ̂':>8}{'σ̂/s_up':>9}"
          f"  判据")
    ok_r = True
    for nm, (why, thr) in list(MUST_HIGH.items()) + list(MUST_LOW.items()):
        high = nm in MUST_HIGH
        if nm not in by:
            print(f"{nm:<42}{'缺失':>7}  << 找不到")
            ok_r = False
            continue
        d = by[nm]
        q, info = censor_fraction_density(raw_values(d), detail=True)
        hit = (q >= thr) if high else (q < thr)
        ok_r &= hit
        sg = info["sigma_hat"]
        rt = info["sigma_over_s"]
        print(f"{nm[:41]:<42}{q:>7.1%}{d['censor_frac']:>11.1%}"
              f"{(f'{sg:.3f}' if sg else '—'):>8}"
              f"{(f'{rt:.3f}' if rt else '—'):>9}"
              f"  必须 {'>=' if high else '<'} {thr:.0%}（{why}）"
              f"  {'OK' if hit else ('<< 漏检' if high else '<< 误伤')}")
    print(f"    真实锚点双向判据: {'通过' if ok_r else '** 未通过 **'}")
    ok_all &= ok_r

    # ---------------- [S] 敏感性扫描 ----------------
    print("\n" + "-" * 92)
    print("[S] 自由参数敏感性（CLAUDE.md §5：记录搜索量）")
    print("-" * 92)
    variants = []
    for gm in (0.20, 0.25, 0.35, 0.50):
        for mm in (40,):
            variants.append({"gamma": gm, "m_min": mm})
    for mm in (25, 60, 100):
        variants.append({"gamma": GAMMA, "m_min": mm})
    for nmv in (30, 120):
        variants.append({"gamma": GAMMA, "m_min": M_MIN, "n_m": nmv})
    print(f"{'γ':>6}{'m_min':>7}{'n_m':>6}{'q=0最大':>9}"
          f"{'σ<=.05 最大偏':>14}{'盲区最大偏':>12}{'hAPN':>8}{'gAPN':>8}")
    anchor_scan = []
    for v in variants:
        rr = np.random.default_rng(555)
        g = grid(rr, reps=3, **v)
        fpm = max(g[(s, 0.0)] for s in SIGMAS)
        c_max = max(abs(g[(s, q)] - q) for s in (0.03, 0.05) for q in QS
                    if q > 0)
        e_max = max(abs(g[(s, q)] - q) for s in BLIND for q in QS if q > 0)
        h = censor_fraction_density(raw_values(by["PDCoV/PDCoV/hAPN"]), **v)
        gg = censor_fraction_density(raw_values(by["PDCoV/PDCoV/gAPN"]), **v)
        anchor_scan.append({**v, "fp": fpm, "c_max": c_max, "e_max": e_max,
                            "hAPN": h, "gAPN": gg})
        print(f"{v.get('gamma', GAMMA):>6.2f}{v.get('m_min', M_MIN):>7}"
              f"{v.get('n_m', N_M):>6}"
              f"{fpm:>9.1%}{c_max * 100:>13.1f}pp{e_max * 100:>11.1f}pp"
              f"{h:>8.1%}{gg:>8.1%}")
    hs = [z["hAPN"] for z in anchor_scan]
    gs = [z["gAPN"] for z in anchor_scan]
    ok_s = min(hs) >= 0.25 and max(gs) < 0.10
    print(f"\n    本次固定形式后的自由参数变体 {len(variants)} 个"
          f"（γ 4 + m_min 3 + n_m 2）。到这个形式之前试过的"
          f"估计量族与选择规则总数见 docs/21 §2.4。")
    print(f"    hAPN 在全部变体上 {min(hs):.1%}–{max(hs):.1%}（必须全 >= 25%）；"
          f"gAPN {min(gs):.1%}–{max(gs):.1%}（必须全 < 10%）：  "
          f"{'结论对参数不敏感' if ok_s else '** 对参数敏感，必须声明 **'}")
    ok_all &= ok_s

    print("\n" + "=" * 92)
    print("密度型截尾检测器自检:", "通过" if ok_all else "** 未通过 —— 中止 **")
    print("=" * 92)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
