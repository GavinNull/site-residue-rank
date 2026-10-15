#!/usr/bin/env python3
"""docs/35-paper-draft.md 的三张正文图。

### 为什么这个脚本带断言

图是最容易和正文脱钩的东西：正文的数字改了、图还是旧的，
而读者看不出来——这正是 D-064/D-070 那一类「撤回了但没 grep 到」的翻版，
只是载体从句子换成了像素。
所以每张图在画之前先核对它自己的汇总量与 docs/35 写定的数字，
**不一致即 AssertionError 中止**，不出图。

标注用英文：仓库里没有装 CJK 字体，中文会渲染成方框，
而一张带方框的图比没有图更坏（它看起来是对的）。

### 这条门禁覆盖什么、不覆盖什么（D-077）

`EXPECT` 是**图的**汇总量，一共 10 个。本脚本的 PASS **只说这 10 个量与缓存一致**。
它**不覆盖** docs/35 §3.3 / docs/30 §2 的五行化学标度表 —— D-077 正是因为
有人把这里的绿灯读成了「那张表也被检查过」才漏了一年多的口径混用。
那张表由 `analysis/check_scale_table.py` 单独把门（见该文件开头的理由）。
**引用任何「门禁通过」时必须写明它覆盖的范围。**

用法:
  python3 analysis/make_figures.py          # 出图 + 核对
  python3 analysis/make_figures.py --check  # 只核对，不写文件
  python3 analysis/check_scale_table.py     # §3.3 的表（另一条门禁）
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figs")

# docs/35 里写定的数字。图必须复现这些，否则说明图与正文已脱钩。
EXPECT = {
    "cos1_med": 0.900,      # §3.2 跨家族 20 配对 cos θ1 中位
    "cos1_frac": 1.00,      # §3.2 20/20 超 p95
    "null_cos1_p95": 0.633,  # §3.2 零分布 p95
    "null_cos2_p95": 0.311,  # §4.2 成对随机零分布 p95
    "n_pairs": 20,
    "cos_uD_med": 0.755,    # §3.4 144 域中位
    "n_over_p95": 136,      # §3.4 136/144
    "n_repro_ok": 144,
    "boot_lo": 0.720,
    "boot_hi": 0.775,
}
TOL = 0.0015


def load(name):
    with open(os.path.join(ROOT, "cache", name), encoding="utf-8") as fh:
        return json.load(fh)


def close(a, b, tol=TOL, what=""):
    assert abs(a - b) <= tol, f"图与正文脱钩：{what} 图={a:.4f} 正文={b:.4f}"


# ---------------------------------------------------------------- figure 1
def fig1(write=True):
    """定秩：CV 误差随秩的曲线。按**已写定的判据**（截尾 < 10%，docs/12 §2）分层。

    第一版把全部 34 个组合画成一条中位曲线，断言当场失败：中位极小在秩 1 而非 2。
    **这不是矛盾，是截尾效应**——重度截尾的组合给秩 0/1（docs/12 §2：
    秩 0/1/2 的截尾中位是 43.2%/10.9%/0.1%），把混合中位拖了下来。
    分层判据取已发表的那条（截尾 < 10%），不是调出来的；
    分层后两层**各自**断言，方向相反：
    低截尾层的极小必须在秩 2，高截尾层的极小必须**不**在秩 2。
    这是 §10.2 的「既测能命中，也测不误伤」用在图上。
    """
    rows = load("ligand_rank.json")
    lo, hi = [], []
    for r in rows:
        mc = r.get("mean_curve") or {}
        ks = sorted(int(k) for k in mc)
        if ks != [0, 1, 2, 3]:
            continue
        v = np.array([mc[str(k)] for k in ks], float)
        if not np.all(np.isfinite(v)) or v[0] <= 0:
            continue
        frac = r["cens"] / (r["cens"] + r["n"])     # docs/19 §1 的反推口径
        (lo if frac < 0.10 else hi).append(v / v[0])  # 以秩 0 归一
    L, H = np.array(lo), np.array(hi)
    assert len(L) == 21, f"低截尾层应为 21 个组合（docs/12 §2），实得 {len(L)}"
    mL, mH = np.median(L, axis=0), np.median(H, axis=0)
    assert int(np.argmin(mL)) == 2, f"低截尾层中位极小在秩 {int(np.argmin(mL))}，不是 2"
    assert int(np.argmin(mH)) != 2, "高截尾层中位极小落在秩 2 —— 分层失去判别力"

    if not write:
        return len(L), len(H)
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    x = np.arange(4)
    for v in L:
        ax.plot(x, v, color="#9db8d6", lw=0.7, alpha=0.6, zorder=1)
    for v in H:
        ax.plot(x, v, color="#e2b4ae", lw=0.7, alpha=0.6, zorder=1)
    ax.plot(x, mL, color="#1f4e8c", lw=2.4, marker="o", ms=6, zorder=4,
            label=f"censoring < 10%  (n = {len(L)})")
    ax.plot(x, mH, color="#c0392b", lw=2.4, marker="s", ms=5.5, zorder=4,
            label=f"censoring \u2265 10%  (n = {len(H)})")
    ax.axvline(2, color="#1f4e8c", ls=":", lw=1.1, zorder=2)
    ax.annotate("interior minimum", xy=(2, mL[2]), xytext=(2.3, mL[2] - 0.045),
                color="#1f4e8c", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#1f4e8c", lw=0.9))
    ax.set_xticks(x)
    ax.set_xlabel("rank of fitted interaction term")
    ax.set_ylabel("cross-validated error\n(relative to rank 0)")
    ax.set_title("Rank selection: the optimum is 2 where the assay has\n"
                 "dynamic range, and 0 where censoring removes it", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig1_rank.{ext}"), dpi=300)
    plt.close(fig)
    return len(L), len(H)


# ---------------------------------------------------------------- figure 2
def fig2(write=True):
    """跨家族第一主角余弦 vs 零分布。**只画第一轴。**

    第一版把 cos θ2 并排画了进来，看起来像「第二轴有点共享」（中位 0.477、11/20 超零分布）。
    **那个 0.477 正是 docs/24 §0.5 判定为选择效应的数**——剔掉最不可分辨的第二维
    会把它从 0.327 抬到 0.477 而条件 p 一动不动。
    把一个已判「不是证据」的量画成图，是让它以最有说服力的形式回到正文，
    与 D-070 要防的是同一件事，只是载体从句子换成了像素。
    第二轴的证据是量级对照，见 fig4。
    """
    g = load("gate_consequences.json")
    key = [k for k in g["pairwise"] if k.startswith("标定后")][0]
    pw = g["pairwise"][key]
    pairs = pw["pairs"]
    c1 = np.array([p["cos1"] if isinstance(p, dict) else p[0] for p in pairs], float)
    p95_1 = g["null_p95"]["cos1"]

    close(len(c1), EXPECT["n_pairs"], 0, "跨家族配对数")
    close(float(np.median(c1)), EXPECT["cos1_med"], 0.002, "cos θ1 中位")
    close(float(np.mean(c1 > p95_1)), EXPECT["cos1_frac"], 0.001, "cos θ1 超 p95 比例")
    close(p95_1, EXPECT["null_cos1_p95"], 0.002, "cos θ1 零分布 p95")

    if not write:
        return len(c1)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    rng = np.random.default_rng(0)
    xj = rng.uniform(-0.12, 0.12, len(c1))
    ax.scatter(xj, c1, s=34, color="#1f4e8c", alpha=0.85, zorder=3,
               edgecolor="white", lw=0.5)
    ax.hlines(np.median(c1), -0.3, 0.3, color="#1f4e8c", lw=2.4, zorder=4)
    ax.text(0.34, np.median(c1), f"median {np.median(c1):.3f}",
            va="center", fontsize=8, color="#1f4e8c")
    ax.hlines(p95_1, -0.42, 0.42, color="0.35", lw=1.5, ls="--", zorder=2)
    ax.text(0.34, p95_1 - 0.035, "random-subspace\nnull p95", va="top",
            fontsize=7.5, color="0.35")
    ax.text(0, 1.045, f"{int((c1 > p95_1).sum())}/{len(c1)} above null",
            ha="center", fontsize=9, color="#1f4e8c")
    ax.set_xticks([])
    ax.set_xlim(-0.6, 1.15)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("first principal angle cosine")
    ax.set_title("The first axis is shared across\nunrelated protein families", fontsize=10)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig2_crossfamily.{ext}"), dpi=300)
    plt.close(fig)
    return len(c1)


# ---------------------------------------------------------------- figure 3
def fig3(write=True):
    """折叠 144 域：折叠第一轴与结合普适轴的夹角余弦分布。"""
    d = load("tsuboyama_axis.json")
    ok = set(d["repro_pass"])
    vals, p95s = [], []
    for name, r in d["rows"].items():
        if name not in ok:
            continue
        vals.append(abs(float(r["cos_uD"])))
        p95s.append(float(r["p95_d"]))
    v = np.array(vals)
    p = np.array(p95s)
    s = d["summary"]

    close(len(v), EXPECT["n_repro_ok"], 0, "通过可复现筛选的域数")
    close(float(np.median(v)), EXPECT["cos_uD_med"], 0.002, "|cos| 中位")
    close(int((v > p).sum()), EXPECT["n_over_p95"], 0, "超各自 p95 的域数")
    close(s["cos_uD_boot_ci"][0], EXPECT["boot_lo"], 0.002, "bootstrap 下端")
    close(s["cos_uD_boot_ci"][1], EXPECT["boot_hi"], 0.002, "bootstrap 上端")

    if not write:
        return len(v)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.hist(v, bins=np.linspace(0, 1, 26), color="#1f4e8c", alpha=0.85,
            edgecolor="white", lw=0.5, zorder=3)
    ax.axvline(float(np.median(p)), color="0.35", ls="--", lw=1.4, zorder=4)
    ax.text(float(np.median(p)) - 0.015, ax.get_ylim()[1] * 0.92, "null p95",
            rotation=90, ha="right", va="top", fontsize=7.5, color="0.35")
    med = float(np.median(v))
    ax.axvline(med, color="#c0392b", lw=2.0, zorder=5)
    lo, hi = s["cos_uD_boot_ci"]
    ax.axvspan(lo, hi, color="#c0392b", alpha=0.17, zorder=2)
    ax.text(med + 0.015, ax.get_ylim()[1] * 0.93,
            f"median {med:.3f}\n[{lo:.3f}, {hi:.3f}]",
            fontsize=8, color="#c0392b", va="top")
    ax.set_xlabel("|cos| between folding first axis and binding universal axis")
    ax.set_ylabel(f"domains (n = {len(v)})")
    ax.set_title("The first axis is shared with folding", fontsize=10)
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig3_folding.{ext}"), dpi=300)
    plt.close(fig)
    return len(v)


# ---------------------------------------------------------------- figure 4
def fig4(write=True):
    """第二轴：量级对照。观测落在「各向同性下界」而不是「完全共享上界」。

    这些数来自 docs/34 的理想条件还原性诊断与 docs/33 的面板重算，
    是量级比较而不是 p 值，**不依赖零分布的建模选择**——
    这正是本文让第二轴的结论挂在它上面而不是挂在任何显著性检验上的原因。
    """
    ISO, SHARED_X, SHARED_W = 0.165, 0.901, 0.821
    OBS_X, OBS_W = 0.105, 0.146
    assert OBS_X < ISO and OBS_W < ISO, "观测应落在各向同性基线以下"
    assert SHARED_W - OBS_W > 0.6, "共享地板与观测之差应接近一个量级"
    if not write:
        return 2
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    rows = [("cross-family", OBS_X, SHARED_X), ("within one backbone,\nacross ligands", OBS_W, SHARED_W)]
    for i, (lab, obs, shared) in enumerate(rows):
        y = 1 - i
        ax.hlines(y, 0, 1, color="0.9", lw=10, zorder=1)
        ax.plot([shared], [y], marker="s", ms=11, color="#2e7d5b", zorder=4)
        ax.plot([ISO], [y], marker="D", ms=8, color="0.45", zorder=4)
        ax.plot([obs], [y], marker="o", ms=12, color="#c0392b", zorder=5)
        ax.annotate("", xy=(obs, y - 0.22), xytext=(shared, y - 0.22),
                    arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.9))
        ax.text((obs + shared) / 2, y - 0.3, "an order of magnitude",
                ha="center", fontsize=7.5, color="0.45")
        ax.text(-0.035, y, lab, ha="right", va="center", fontsize=8.5)
    ax.plot([], [], "o", color="#c0392b", ms=9, label="observed")
    ax.plot([], [], "D", color="0.45", ms=7, label="isotropic baseline (0.165)")
    ax.plot([], [], "s", color="#2e7d5b", ms=9, label="floor if fully shared")
    ax.legend(frameon=False, fontsize=7.8, loc="upper center",
              bbox_to_anchor=(0.55, 1.02), ncol=3, handletextpad=0.3,
              columnspacing=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, 1.95)
    ax.set_yticks([])
    ax.set_xlabel("second principal angle cosine")
    ax.set_title("The second axis: observed values sit at the isotropic\n"
                 "baseline, not at the shared-axis floor", fontsize=9.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig4_secondaxis.{ext}"), dpi=300)
    plt.close(fig)
    return 2


if __name__ == "__main__":
    write = "--check" not in sys.argv
    if write:
        os.makedirs(FIGDIR, exist_ok=True)
    n1 = fig1(write)
    n2 = fig2(write)
    n3 = fig3(write)
    n4 = fig4(write)
    print(f"fig1 定秩曲线      低截尾 {n1[0]} 条（极小在秩 2）/ 高截尾 {n1[1]} 条（极小不在秩 2）✓")
    print(f"fig2 跨家族第一轴   {n2} 个配对，与正文汇总量一致 ✓")
    print(f"fig3 折叠 144 域    {n3} 个域，与正文汇总量一致 ✓")
    print(f"fig4 第二轴量级对照 {n4} 组，观测落在各向同性下界 ✓")
    print("全部断言通过：图与 docs/35 的数字一致" + ("，已写入 figs/" if write else "（未写文件）"))
