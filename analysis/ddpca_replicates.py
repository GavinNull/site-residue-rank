#!/usr/bin/env python3
"""ddPCA 的**生物学重复**读出（docs/28 Step 3 的独立测量把手）。

为什么需要这个模块
------------------
docs/27 §4.1 / D-042 之所以能把「σ 取太大制造假阴性」**测**出来而不是
论证出来，是因为 Tsuboyama 数据自带**两条独立蛋白酶读出**。
docs/15 判定三个 ddPCA binding 矩阵 `coup_sd_true = 0` 时没有做这件事。
本模块找出 ddPCA 一侧的同型把手：

  zip 内 `Data/fitness/<domain>/{Binding,Abundance}/*_fitness_replicates.RData`
  含 `count_e{1,2,3}_s{0,1}` 与 `fitness{1,2,3}_uncorr` / `sigma{1,2,3}_uncorr`
  —— **三次独立的选择实验**（e1/e2/e3），以及它们合并出来的 `fitness`。

这与 ddPCA 的 bootstrap SE **完全不是一回事**：bootstrap 是同一批数据的
重抽（D-028 已判定不得用作噪声模型），这里是三次独立的生物学重复。

RData 读取与缓存
----------------
`.RData` 需要 `pyreadr`。为了让下游脚本不依赖它，本模块把需要的两张表
(`all_variants` 的重复列、`singles`) 转成
`cache/ddpca_reps/<assay>__{allvar,singles}.csv.gz`，并落盘 provenance
（zip 内路径 + 该成员的 sha256）。已存在则直接读缓存。

自检（CLAUDE.md §10，用**已知事实**，不用自洽性）
--------------------------------------------------
R1  野生型的合并 fitness 必须 ≈ 0（DiMSum 的输出约定；外部已知事实）。
R2  **外部锚点**：合并 `fitness` 必须等于三个重复列的**逆方差加权均值**，
    且合并 `sigma` 等于 1/sqrt(Σ 1/σ_i²)。这是拿一个**已发表、已被原作者
    用过**的列去验我的解析，而不是自洽性检查。
R3  跨表一致：`singles` 表的 fitness 必须等于 `all_variants` 同 aa_seq 的
    合并 fitness。
R4  STOP 变体必须显著比非 STOP 更差（已知生物学事实）。
R5  **不误伤对照**：把变体↔重复的配对打乱之后，重复间相关必须塌到 ~0。
    （D-007：只测「能命中」的护栏最终会被绕过。）
任一不过即 AssertionError。
"""
import gzip
import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, "cache", "dms", "lehner_doubledeepms_Data.zip")
OUT = os.path.join(ROOT, "cache", "ddpca_reps")

# assay 名 -> zip 内 RData 路径。来源：zip 索引（见 ddpca_load.check_abundance_assays）
ASSAYS = {
    "GB1/binding":
        "Data/fitness/GB1/Binding/"
        "Olson2014_GB1_dimsum128_fitness_replicates.RData",
    "GRB2-SH3/binding":
        "Data/fitness/GRB2-SH3/Binding/"
        "JD_GRB2_epPCA_bindingPCA_dimsum128_fitness_replicates.RData",
    "GRB2-SH3/abundance":
        "Data/fitness/GRB2-SH3/Abundance/"
        "JD_GRB2_NM2_stabilityPCA_dimsum128_fitness_replicates.RData",
    "PSD95-PDZ3/binding":
        "Data/fitness/PSD95-PDZ3/Binding/"
        "JD_PDZ_NM2_bindingPCA_dimsum128_filtered_fitness_replicates.RData",
    "PSD95-PDZ3/abundance":
        "Data/fitness/PSD95-PDZ3/Abundance/"
        "JD_PDZ_NM2_stabilityPCA_dimsum128_filtered_fitness_replicates.RData",
}
# GB1 **没有** abundance assay（docs/15 §6.1，由 zip 索引断言）。
# 所以 GB1/folding 的重复把手只能来自同一组 binding-PCA 观测 —— 这一点
# 必须随每个 GB1 数字带着。

AV_COLS = ["aa_seq", "Nham_aa", "STOP", "WT", "mean_count",
           "fitness", "sigma",
           "fitness1_uncorr", "sigma1_uncorr",
           "fitness2_uncorr", "sigma2_uncorr",
           "fitness3_uncorr", "sigma3_uncorr"]
SG_COLS = ["Pos", "WT_AA", "Mut", "aa_seq", "Nham_aa", "STOP",
           "mean_count", "fitness", "sigma"]


def _sha256_member(name):
    h = hashlib.sha256()
    with zipfile.ZipFile(ZIP) as z, z.open(name) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(assay):
    return assay.replace("/", "__")


def convert(force=False, verbose=True):
    """把 RData 转成 csv.gz 缓存。需要 pyreadr（只在这一步）。"""
    os.makedirs(OUT, exist_ok=True)
    prov = {}
    need = [a for a in ASSAYS
            if force or not os.path.exists(
                os.path.join(OUT, _slug(a) + "__allvar.csv.gz"))]
    if need:
        try:
            import pyreadr
        except ImportError:
            raise SystemExit(
                "需要 pyreadr 把 .RData 转成 csv.gz（只在转换这一步）。\n"
                "  python3 -m venv /tmp/venv && /tmp/venv/bin/pip install "
                "pyreadr pandas\n"
                "  /tmp/venv/bin/python analysis/ddpca_replicates.py convert")
        import pandas as pd
        import tempfile
        with zipfile.ZipFile(ZIP) as z:
            for assay in need:
                member = ASSAYS[assay]
                with tempfile.NamedTemporaryFile(suffix=".RData",
                                                 delete=False) as tf:
                    tf.write(z.read(member))
                    tmp = tf.name
                try:
                    r = pyreadr.read_r(tmp)
                finally:
                    os.unlink(tmp)
                av = r["all_variants"]
                sg = r["singles"]
                wt = r["wildtype"]
                av = pd.concat([wt[av.columns], av], ignore_index=True)
                av[[c for c in AV_COLS if c in av.columns]].to_csv(
                    os.path.join(OUT, _slug(assay) + "__allvar.csv.gz"),
                    index=False)
                sg[[c for c in SG_COLS if c in sg.columns]].to_csv(
                    os.path.join(OUT, _slug(assay) + "__singles.csv.gz"),
                    index=False)
                if verbose:
                    print(f"  转换 {assay}: all_variants {av.shape}, "
                          f"singles {sg.shape}")
    for assay, member in ASSAYS.items():
        p = os.path.join(OUT, _slug(assay) + "__prov.json")
        if force or not os.path.exists(p):
            prov[assay] = {"zip": os.path.basename(ZIP), "member": member,
                           "sha256": _sha256_member(member)}
            json.dump(prov[assay], open(p, "w"), indent=1)
    return OUT


def _read_csv_gz(path):
    """不依赖 pandas 的最小读取：-> (header, dict of column arrays)。"""
    with gzip.open(path, "rt") as f:
        head = f.readline().rstrip("\n").split(",")
        rows = [l.rstrip("\n").split(",") for l in f]
    cols = {}
    for i, h in enumerate(head):
        v = [r[i] if i < len(r) else "" for r in rows]
        try:
            cols[h] = np.array([np.nan if x in ("", "NA", "NaN") else float(x)
                                for x in v])
        except ValueError:
            cols[h] = np.array([x.strip('"') for x in v], dtype=object)
    return head, cols


def load(assay):
    """-> dict(allvar=..., singles=...)，列为 numpy 数组。"""
    convert(verbose=False)
    av = _read_csv_gz(os.path.join(OUT, _slug(assay) + "__allvar.csv.gz"))[1]
    sg = _read_csv_gz(os.path.join(OUT, _slug(assay) + "__singles.csv.gz"))[1]
    return {"assay": assay, "allvar": av, "singles": sg}


def _bool(a):
    return np.array([str(x).upper() in ("TRUE", "1", "1.0") for x in a])


def selfcheck(verbose=True):
    """R1–R5。任一不过即 AssertionError。"""
    log = []
    for assay in ASSAYS:
        D = load(assay)
        av = D["allvar"]
        F = np.column_stack([av[f"fitness{i}_uncorr"] for i in (1, 2, 3)])
        S = np.column_stack([av[f"sigma{i}_uncorr"] for i in (1, 2, 3)])
        fm, sm = av["fitness"], av["sigma"]
        isWT = _bool(av["WT"])

        # R1 野生型合并 fitness ≈ 0
        assert isWT.sum() >= 1, f"{assay}: 找不到野生型行"
        wtf = float(np.nanmedian(fm[isWT]))
        assert abs(wtf) < 0.05, f"{assay}: 野生型 fitness = {wtf:.4f}，不≈0"

        # R2 外部锚点：合并 = 逆方差加权均值
        ok = np.all(np.isfinite(F), 1) & np.all(np.isfinite(S) & (S > 0), 1) \
            & np.isfinite(fm)
        n2 = int(ok.sum())
        assert n2 >= 100, f"{assay}: 三重复齐全的变体只有 {n2} 个"
        w = 1.0 / S[ok] ** 2
        ivw = (F[ok] * w).sum(1) / w.sum(1)
        dmax = float(np.abs(ivw - fm[ok]).max())
        assert dmax < 1e-8, (
            f"{assay}: 合并 fitness 不等于三重复的逆方差加权均值 "
            f"(max|Δ| = {dmax:.2e}) —— 重复列或合并列解析错了")
        smax = float(np.abs(1.0 / np.sqrt(w.sum(1)) - sm[ok]).max())
        assert smax < 1e-8, f"{assay}: 合并 sigma 不符 (max|Δ| = {smax:.2e})"
        # 对照：朴素算术均值必须**明显不**吻合（否则 R2 没有判别力）
        pmax = float(np.abs(F[ok].mean(1) - fm[ok]).max())
        assert pmax > 10 * dmax, (
            f"{assay}: 逆方差均值与算术均值无法区分，R2 无判别力")

        # R3 跨表一致
        sg = D["singles"]
        m = {str(s): f for s, f in zip(av["aa_seq"], fm)}
        d3 = [abs(f - m[str(s)]) for s, f in zip(sg["aa_seq"], sg["fitness"])
              if str(s) in m and np.isfinite(f) and np.isfinite(m[str(s)])]
        assert len(d3) > 0.5 * len(sg["aa_seq"]), f"{assay}: singles 对不上"
        assert max(d3) < 1e-8, f"{assay}: singles/all_variants 不一致"

        # R4 STOP 更差（已知生物学）。
        # **口径必须限定在单取代上**：`all_variants` 里 97% 是双取代
        # （GRB2/abundance 68 484 / 69 745），而双取代整体远比 STOP 单取代
        # 更去稳定，于是「全部 STOP vs 全部非 STOP」会给出反向的中位数
        # （实测 −0.790 vs −0.969）。那是**我这条检查口径的错**，不是数据的错
        # （§10 要求分开记）：已知事实是「提前终止比典型错义更坏」，
        # 比较必须在同一取代阶数上做，而矩阵用的也正是单取代。
        st = _bool(av["STOP"])
        s1 = (av["Nham_aa"] == 1) & ~isWT
        if (st & s1).sum() >= 20:
            a = float(np.nanmedian(fm[st & s1]))
            b = float(np.nanmedian(fm[~st & s1]))
            assert a < b, (f"{assay}: 单取代 STOP 中位 {a:.3f} "
                           f"不低于单取代非 STOP {b:.3f}")
            r4 = f"单取代 STOP {a:+.3f} < 非STOP {b:+.3f}"
        else:
            r4 = (f"单取代 STOP 只有 {int((st & s1).sum())} 个，跳过"
                  "（Olson2014 的 GB1 库按氨基酸设计，没有 STOP）")

        # R5 不误伤：打乱配对后重复相关必须塌
        rng = np.random.default_rng(11)
        rr = [float(np.corrcoef(F[ok][:, i], F[ok][:, j])[0, 1])
              for i, j in ((0, 1), (0, 2), (1, 2))]
        pm = rng.permutation(n2)
        rs = [float(np.corrcoef(F[ok][:, i], F[ok][pm][:, j])[0, 1])
              for i, j in ((0, 1), (0, 2), (1, 2))]
        assert min(rr) > 0.3, f"{assay}: 真实重复相关只有 {rr}"
        assert max(abs(x) for x in rs) < 0.1, (
            f"{assay}: 打乱配对后相关仍有 {rs} —— R5 不误伤对照失败")

        log.append(f"{assay:<22} R1 wt {wtf:+.4f} | R2 n={n2} max|Δ|="
                   f"{dmax:.1e}（算术均值 {pmax:.2f}）| R3 OK | R4 {r4} "
                   f"| R5 真 {min(rr):.3f}–{max(rr):.3f} / 打乱 "
                   f"{max(abs(x) for x in rs):.3f}  OK")
    if verbose:
        print("ddPCA 生物学重复 R1–R5 自检")
        for l in log:
            print("  " + l)
        print("  全部通过")
    return log


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "convert":
        convert(force="--force" in sys.argv)
        print("缓存:", OUT)
    selfcheck()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
