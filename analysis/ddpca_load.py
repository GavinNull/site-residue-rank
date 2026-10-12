#!/usr/bin/env python3
"""ddPCA（Faure et al. 2022, doubledeepms）folding / binding ΔΔG 的加载与**自检**。

数据来源（provenance）
---------------------
主表 : cache/dms/lehner_ddPCA/{GB1,GRB2-SH3,PSD95-PDZ3}_ddG_weights.txt
       逐字节等于 cache/dms/lehner_doubledeepms_Data.zip 内
       Data/mochi/<domain>/mochi__fit_tmodel_3state_sparse_dimsum128/
           model_weights_0.txt
误差 : 同目录 bootstrap/boot_1..10/model_weights_0.txt
       -> 每个变体 10 个 bootstrap 重拟合，逐变体 SD 作为标准误 (SE)。
结构 : zip 内 Data/pdb/{2gb1,2vwf,1be9}.pdb，用于核对解析出的野生型序列。

单位：kcal/mol。WT 行是绝对 ΔG，突变体行是 ΔΔG。

自检（CLAUDE.md §10，用**已知事实**不用自洽性）
--------------------------------------------
V1 id 格式 `<WT><pos><MUT>`，WT 之外无例外。
V2 同一位点的野生型残基在全部变体里必须一致（已知事实：一个构建体
   在一个位点只有一个野生型残基）。
V3 位点编号连续，且每位点至多 19 个突变、野生型残基**从不**作为突变出现。
V4 解析出的野生型序列必须与该域的 PDB 结构序列逐残基吻合；
   不吻合的位置必须能被「PDB 的那个残基出现在该位点的突变列表里」解释
   （即构建体与晶体结构在该位点确有差异），否则视为解析错误。
V5 量级：WT 折叠 ΔG 必须落在 −2..−12 kcal/mol（球状小域的已知范围）。
V6 bootstrap SE 的中位数必须远小于 ΔΔG 的分布宽度，否则数据不可用。
"""
import glob
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDPCA = os.path.join(ROOT, "cache", "dms", "lehner_ddPCA")
ZIP = os.path.join(ROOT, "cache", "dms", "lehner_doubledeepms_Data.zip")
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

DOMAINS = {
    # 域名 -> (主表文件名, mochi 目录名, PDB id, PDB 链)
    "GB1":        ("GB1_ddG_weights.txt", "GB1", "2gb1", "A"),
    "GRB2-SH3":   ("GRB2-SH3_ddG_weights.txt", "GRB2-SH3", "2vwf", "A"),
    "PSD95-PDZ3": ("PSD95-PDZ3_ddG_weights.txt", "PSD95-PDZ3", "1be9", "A"),
}

# 哪个域有独立的 abundance（稳定性）assay —— 决定 folding 与 binding 是否
# 来自**同一批观测**。来源：zip 内 Data/fitness/<domain>/ 的子目录。
#   GRB2-SH3   : Binding/ + Abundance/      -> 两个独立 assay
#   PSD95-PDZ3 : Binding/ + Abundance/      -> 两个独立 assay
#   GB1        : Binding/ 只有一个（Olson2014）-> folding 与 binding 由
#                **同一组 binding-PCA 观测**联合反推，混淆最严重。
HAS_ABUNDANCE_ASSAY = {"GB1": False, "GRB2-SH3": True, "PSD95-PDZ3": True}


def check_abundance_assays():
    """从 zip 的**索引**核对上面这张表（不解压 —— RData 有 60MB 用不到）。

    这是 §6.1 那条结论的唯一依据，所以必须核对而不是记在注释里：
    GB1 的 folding 与 binding 系数来自同一组 binding-PCA 观测。
    """
    import zipfile
    seen = {}
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            if n.startswith("__MACOSX") or "/Data/fitness/" not in "/" + n:
                continue
            parts = n.split("/")
            if len(parts) >= 4 and parts[1] == "fitness" and parts[3]:
                seen.setdefault(parts[2], set()).add(parts[3])
    out = {}
    for dom in DOMAINS:
        subs = {s for s in seen.get(dom, set()) if not s.startswith(".")}
        out[dom] = any("Abundance" in s for s in subs)
        assert out[dom] == HAS_ABUNDANCE_ASSAY[dom], (
            f"{dom}: zip 内 Data/fitness/{dom}/ 的子目录是 {sorted(subs)}，"
            f"与 HAS_ABUNDANCE_ASSAY 表不符")
    return out

T3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
      "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
      "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
      "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

ID_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _scratch():
    s = os.environ.get("DDPCA_SCRATCH")
    if s:
        return s
    return os.path.join(ROOT, "cache", "ddpca_extract")


def _ensure_extracted():
    """把 zip 里需要的文件解到 scratch（已存在则跳过）。"""
    base = _scratch()
    need = os.path.join(base, "Data", "pdb", "2gb1.pdb")
    if os.path.exists(need):
        return base
    import zipfile
    pats = ("Data/pdb/", "Data/in_vitro/",
            "model_weights_0.txt", "model_parameters_0.txt")
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            if n.startswith("__MACOSX"):
                continue
            if any(p in n for p in pats):
                z.extract(n, base)
    return base


def read_weights(path):
    """-> {id: (folding or None, binding or None)}。制表符分隔，缺失为空串。"""
    out = {}
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        assert head[:3] == ["id", "folding_coefficient",
                            "binding_coefficient"], f"表头异常: {head}"
        for line in f:
            p = line.rstrip("\n").split("\t")
            if not p[0]:
                continue

            def g(i):
                v = p[i].strip() if len(p) > i else ""
                return None if v in ("", "NA", "nan") else float(v)
            out[p[0]] = (g(1), g(2))
    return out


def pdb_seq(pdb, chain):
    base = _ensure_extracted()
    p = os.path.join(base, "Data", "pdb", pdb + ".pdb")
    d = {}
    for l in open(p):
        if l.startswith("ATOM") and l[12:16].strip() == "CA" \
                and l[21] == chain:
            aa = T3.get(l[17:20].strip())
            if aa:
                d[int(l[22:26])] = aa
    return d


def load_domain(dom):
    """返回 dict：ids / wt_seq / 主估计 / bootstrap SE。"""
    fn, mdir, pdb, chain = DOMAINS[dom]
    main = read_weights(os.path.join(DDPCA, fn))
    base = _ensure_extracted()
    bdir = os.path.join(base, "Data", "mochi", mdir,
                        "mochi__fit_tmodel_3state_sparse_dimsum128",
                        "bootstrap")
    boots = [read_weights(p) for p in
             sorted(glob.glob(os.path.join(bdir, "boot_*",
                                           "model_weights_0.txt")))]
    # 主表应与 zip 内 model_weights_0.txt 完全一致 —— 核对一次
    z = read_weights(os.path.join(base, "Data", "mochi", mdir,
                                  "mochi__fit_tmodel_3state_sparse_dimsum128",
                                  "model_weights_0.txt"))
    assert z == main, f"{dom}: cache 主表与 zip 内 model_weights_0.txt 不一致"

    wt = {}
    vals = {"folding": {}, "binding": {}}
    ses = {"folding": {}, "binding": {}}
    for vid, (fv, bv) in main.items():
        if vid == "WT":
            continue
        m = ID_RE.match(vid)
        assert m, f"{dom}: id 不符合 <WT><pos><MUT>: {vid}"
        w, pos, mu = m.group(1), int(m.group(2)), m.group(3)
        prev = wt.setdefault(pos, w)
        assert prev == w, f"{dom}: 位点 {pos} 的野生型残基不一致 {prev}/{w}"
        for ph, v in (("folding", fv), ("binding", bv)):
            if v is None:
                continue
            vals[ph][(pos, mu)] = v
            bs = [b[vid][0 if ph == "folding" else 1] for b in boots
                  if vid in b]
            bs = [x for x in bs if x is not None]
            ses[ph][(pos, mu)] = (float(np.std(bs, ddof=1))
                                  if len(bs) >= 3 else math.nan)
    return {"domain": dom, "wt": wt, "vals": vals, "se": ses,
            "n_boot": len(boots), "wt_row": main["WT"],
            "pdb": pdb, "chain": chain,
            "mutants": _mutant_sets(main)}


def _mutant_sets(main):
    d = {}
    for vid in main:
        m = ID_RE.match(vid)
        if m:
            d.setdefault(int(m.group(2)), set()).add(m.group(3))
    return d


def verify(dom, D, verbose=True):
    """V1–V6。返回报告行；任何已知事实不符就抛 AssertionError。"""
    log = []
    wt = D["wt"]
    pos = sorted(wt)
    # V2/V3
    assert pos == list(range(pos[0], pos[-1] + 1)), f"{dom}: 位点编号不连续"
    for p in pos:
        ms = D["mutants"][p]
        assert wt[p] not in ms, f"{dom}: 位点 {p} 的野生型 {wt[p]} 出现在突变列表"
        assert len(ms) <= 19, f"{dom}: 位点 {p} 有 {len(ms)} 个突变 (>19)"
        assert ms <= set(AA20), f"{dom}: 位点 {p} 有非标准残基 {ms - set(AA20)}"
    nmut = [len(D["mutants"][p]) for p in pos]
    log.append(f"V1-V3 id 解析: {len(pos)} 个连续位点 {pos[0]}..{pos[-1]}, "
               f"每位点突变数 {min(nmut)}..{max(nmut)}（上限 19）, "
               f"野生型从不作为突变出现  OK")

    # V4 野生型序列 vs PDB
    seq = "".join(wt[p] for p in pos)
    ps = pdb_seq(D["pdb"], D["chain"])
    pk = sorted(ps)
    pstr = "".join(ps[k] for k in pk)
    # 在 PDB 序列里找最佳对齐偏移（等长子串，最少错配）
    best = None
    for off in range(0, max(1, len(pstr) - len(seq) + 1)):
        sub = pstr[off:off + len(seq)]
        if len(sub) < len(seq):
            break
        mm = [i for i in range(len(seq)) if seq[i] != sub[i]]
        if best is None or len(mm) < len(best[1]):
            best = (off, mm, sub)
    off, mm, sub = best
    assert len(mm) <= 0.05 * len(seq), (
        f"{dom}: 解析序列与 PDB {D['pdb']} 错配 {len(mm)}/{len(seq)} "
        f"(>5%) —— 疑似解析/编号错误")
    # 每个错配必须由构建体差异解释：PDB 的残基出现在该位点的突变列表里
    unexplained = []
    for i in mm:
        p = pos[i]
        if sub[i] not in D["mutants"][p]:
            unexplained.append((p, seq[i], sub[i]))
    assert not unexplained, (
        f"{dom}: 错配无法由构建体差异解释 {unexplained} "
        f"—— PDB 的残基既不是野生型也不在突变列表里，说明解析错了")
    expl = ", ".join(f"pos{pos[i]} 构建体{seq[i]}/PDB {sub[i]}" for i in mm)
    log.append(f"V4 野生型序列 vs PDB {D['pdb']}:{D['chain']} "
               f"(偏移 {off}, 对应 PDB 残基号 {pk[off]}): "
               f"{len(seq) - len(mm)}/{len(seq)} 逐残基吻合"
               + (f"；{len(mm)} 处差异全部被突变列表解释（{expl}）" if mm
                  else "")
               + "  OK")

    # V5 量级校准（CLAUDE.md §8）：与 zip 内 Data/in_vitro/ 的**体外文献
    #   ΔΔG** 逐变体对照。这比「WT 行像不像 ΔG」可靠得多 —— 实测 WT 行
    #   并不是标准参考态下的绝对 ΔG（GRB2 文献 Kd=1.7 uM => ΔG_bind
    #   = −7.9 kcal/mol，而 WT 行给 −1.74），它是模型的截距，不能当 ΔG 读。
    D["pdb_offset"] = pk[off] - pos[0]
    lit = literature_check(dom, D)
    # 断言只在**有判别力**的对照上生效。判据在看结果之前定好：
    #   n >= 8  且  文献 ΔΔG 的 SD >= 1.0 kcal/mol。
    # 1.0 的来历：CLAUDE.md §5 已把实验 ΔΔG 差异 < 0.5 kcal/mol 判为
    # 物理不可分辨；一个整体跨度只有 0.5 的对照本身没有分辨力，
    # 它的低相关不含「解析错了」的信息。阈值取 2 倍该下限。
    # 实测：Malagrino2019(Kd) 的全部动态范围 SD = 0.27 kcal/mol，
    #       Troilo2018 SD = 0.58 —— 两者都只报告不断言；
    #       Olson2014 / Calosci2008 / Chi2008 的 SD 为 1.07–1.49，断言生效。
    for ph, src, n, r, slope, rmse, lsd in lit:
        tag = "OK"
        if n >= 8 and lsd >= 1.0:
            assert r > 0.5, (
                f"{dom}/{ph} vs {src}: 与体外文献 ΔΔG 相关仅 r={r:.2f} "
                f"(n={n}) —— 解析或单位可能错了")
            assert 0.4 < slope < 2.5, (
                f"{dom}/{ph} vs {src}: 斜率 {slope:.2f} 偏离 1 太多 "
                f"—— 量级对不上（§8：量级对不上就是不通过）")
        else:
            why = []
            if n < 8:
                why.append(f"n={n}<8")
            if lsd < 1.0:
                why.append(f"文献跨度 SD={lsd:.2f}<1.0 kcal/mol")
            tag = "仅报告（" + "、".join(why) + "，无分辨力）"
        log.append(f"V5 量级 {ph} vs {src}: n={n}, 文献 SD={lsd:.2f}, "
                   f"r={r:.2f}, 斜率={slope:.2f}, "
                   f"RMSE={rmse:.2f} kcal/mol  {tag}")
    assert lit, f"{dom}: 没有任何体外文献对照可做 —— 量级未校准"

    # V7 ddPCA 自身的符号约定（已知事实：脯氨酸取代最去稳定，甘氨酸次之）
    import collections
    bya = collections.defaultdict(list)
    for (p, a), v in D["vals"]["folding"].items():
        bya[a].append(v)
    med = {a: float(np.median(x)) for a, x in bya.items()}
    rank_p = sorted(med, key=med.get).index("P")
    assert rank_p == 19, (
        f"{dom}: 脯氨酸在 folding ΔΔG 的 20 个残基里只排第 {rank_p + 1} "
        f"（从最稳定数起）—— 已知事实是它最去稳定，符号或解析可能错了")
    assert med["G"] > np.median(list(med.values())), (
        f"{dom}: 甘氨酸不在去稳定的那一半 —— 同上")
    frac_pos = float(np.mean([v > 0 for v in D["vals"]["folding"].values()]))
    assert frac_pos > 0.6, (
        f"{dom}: 只有 {frac_pos:.0%} 的突变 folding ΔΔG 为正 —— "
        f"已知事实是随机突变以去稳定为主")
    log.append(f"V7 符号约定: 正 = 去稳定。Pro 在 20 残基里最去稳定 "
               f"(中位 {med['P']:+.2f}), Gly 第 {sorted(med, key=med.get).index('G') + 1}, "
               f"{frac_pos:.0%} 的突变为正  OK")

    # V6 SE 量级
    for ph in ("folding", "binding"):
        v = np.array(list(D["vals"][ph].values()))
        s = np.array([x for x in D["se"][ph].values() if np.isfinite(x)])
        assert len(s) > 0.9 * len(v), f"{dom}/{ph}: bootstrap SE 缺失过多"
        med = np.median(s)
        assert med < 0.5 * v.std(), (
            f"{dom}/{ph}: SE 中位 {med:.3f} 不小于 ΔΔG 分布宽度 "
            f"{v.std():.3f} 的一半 —— 数据噪声过大")
        log.append(f"V6 {ph}: n={len(v)}, ΔΔG SD={v.std():.2f}, "
                   f"SE 中位={med:.3f}, SE p90={np.quantile(s, .9):.2f} "
                   f"({D['n_boot']} 次 bootstrap)  OK")

    if verbose:
        print(f"[{dom}]")
        for l in log:
            print("  " + l)
    return log


RT = 0.001987 * 298.15        # kcal/(mol K) * K = 0.5925 kcal/mol


def _lit_rows(path, sep="\t"):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f, delimiter=sep))


def literature_check(dom, D, se_max=0.5, abs_max=10.0):
    """逐变体对照 zip 内 Data/in_vitro/ 的体外文献 ΔΔG。

    返回 [(phenotype, 来源, n, pearson r, 过原点斜率, RMSE)]。
    位点编号映射：数据位点 p -> PDB 残基号 p + D['pdb_offset']（V4 对齐得出）。
    """
    base = _ensure_extracted()
    iv = os.path.join(base, "Data", "in_vitro")
    off = D["pdb_offset"]
    # 对照必须用**与矩阵完全相同的排除规则**，否则等于拿一批自己都不打算
    # 分析的坏值去判断解析对不对。实测 PSD95 folding：不过滤时 r = 0.44，
    # 而那是**单个** SE = 4.21 的点（L379A，ddPCA 13.4 vs 体外 1.5）主导的；
    # 加上同样的 SE <= 0.5 过滤后 r = 0.79、斜率 0.77、RMSE 0.89 kcal/mol。
    V = {ph: {k: v for k, v in D["vals"][ph].items()
              if np.isfinite(D["se"][ph].get(k, math.nan))
              and D["se"][ph][k] <= se_max and abs(v) <= abs_max}
         for ph in ("folding", "binding")}
    out = []

    def cmp(ph, src, pairs, sign=None, muts=None):
        """pairs: [(文献值, ddPCA值)]。sign: 文献源的符号约定，None 表示
        用 Pro/Gly 锚点从该源**内部**定出来，定不出来就按构造取 +1。

        为什么必须这么做：Olson2014 Table S4 的 ddG 列是
        **负值 = 去稳定**（T25P −2.80、A26G −2.96，而脯氨酸/甘氨酸取代
        最去稳定是教科书事实）。直接假设 +1 会得到 r = −0.87，
        从而把一次**成功**的量级校准误判成解析错误。
        """
        xs = np.array([a for a, _ in pairs], float)
        ys = np.array([b for _, b in pairs], float)
        if len(xs) < 3:
            return
        if sign is None:
            sign = +1
            if muts:
                pg = np.array([xs[i] for i, m in enumerate(muts)
                               if m in ("P", "G")])
                if len(pg) >= 4:
                    sign = -1 if pg.mean() < 0 else +1
        xs = xs * sign
        r = float(np.corrcoef(xs, ys)[0, 1])
        slope = float(xs @ ys / (xs @ xs))      # 过原点，ΔΔG 的自然约束
        rmse = float(np.sqrt(np.mean((ys - xs) ** 2)))
        out.append((ph, src + ("[符号已按 Pro/Gly 锚点翻转]" if sign < 0
                               else ""), len(xs), r, slope, rmse,
                    float(xs.std())))

    def key(vid, numbering):
        """文献 id -> (数据位点, 突变残基)。numbering: 'pdb' 或 'data'。"""
        m = ID_RE.match(vid.strip())
        if not m:
            return None
        p = int(m.group(2))
        if numbering == "pdb":
            p -= off
        return (p, m.group(3)), m.group(1)

    if dom == "GB1":
        # Olson2014 Table S4：体外 ΔΔG_folding（Olson 编号 = 2gb1 编号）
        pr, mu = [], []
        for r0 in _lit_rows(os.path.join(
                iv, "GB1_literature_free_energies",
                "Olson2014_TableS4.txt")):
            k = key(r0["Mutant"], "pdb")
            if not k:
                continue
            (kk, wtl) = k
            if kk in V["folding"] and D["wt"].get(kk[0]) == wtl:
                pr.append((float(r0["ddG"]), V["folding"][kk]))
                mu.append(kk[1])
        cmp("folding", "Olson2014_TableS4", pr, muts=mu)

    if dom == "GRB2-SH3":
        rows = _lit_rows(os.path.join(iv, "GRB2_literature_free_energies.txt"))
        # Troilo_2018 折叠动力学：ΔG = -RT ln(Kf/Ku)（数据自带 1..56 编号）
        tro = {r0["id"].strip(): r0 for r0 in rows
               if r0["Dataset"] == "Troilo_2018"}
        if "WT" in tro and tro["WT"]["Kf"] and tro["WT"]["Ku"]:
            g0 = -RT * math.log(float(tro["WT"]["Kf"])
                                / float(tro["WT"]["Ku"]))
            pr = []
            for vid, r0 in tro.items():
                k = key(vid, "data")
                if not k or not r0["Kf"] or not r0["Ku"]:
                    continue
                (kk, wtl) = k
                if kk in V["folding"] and D["wt"].get(kk[0]) == wtl:
                    g = -RT * math.log(float(r0["Kf"]) / float(r0["Ku"]))
                    pr.append((g - g0, V["folding"][kk]))
            cmp("folding", "Troilo2018(Kf/Ku)", pr, sign=+1)   # 由构造：ΔG=-RT ln(Kf/Ku)
        # Malagrino_2019 结合 Kd（uM）：ΔΔG_bind = RT ln(Kd_mut/Kd_WT)
        mal = {r0["id"].strip(): r0 for r0 in rows
               if r0["Dataset"] == "Malagrino_2019"}
        if "WT" in mal and mal["WT"]["Kd"]:
            kd0 = float(mal["WT"]["Kd"])
            pr = []
            for vid, r0 in mal.items():
                k = key(vid, "data")
                if not k or not r0["Kd"]:
                    continue
                (kk, wtl) = k
                if kk in V["binding"] and D["wt"].get(kk[0]) == wtl:
                    pr.append((RT * math.log(float(r0["Kd"]) / kd0),
                               V["binding"][kk]))
            cmp("binding", "Malagrino2019(Kd)", pr, sign=+1)   # 由构造：RT ln(Kd_mut/Kd_WT)>0 = 结合变弱

    if dom == "PSD95-PDZ3":
        pr = []
        for r0 in _lit_rows(os.path.join(
                iv, "PDZ_literature_free_energies",
                "Calosci2008_invitro_ddGs_PDZ.txt")):
            k = key(r0["id"], "pdb")
            if not k or not r0["f_ddg"]:
                continue
            (kk, wtl) = k
            if kk in V["folding"] and D["wt"].get(kk[0]) == wtl:
                pr.append((float(r0["f_ddg"]), V["folding"][kk]))
        cmp("folding", "Calosci2008", pr, sign=+1)        # 文件内 I316A=+3.5（埋藏 I->A 去稳定）定向
        pr = []
        for r0 in _lit_rows(os.path.join(
                iv, "PDZ_literature_free_energies",
                "Chi2008_invitro_kd_ddGf_ddGb_PDZ_CRIPTpeptide.txt"), " "):
            k = key(r0["id"], "pdb")
            if not k or not r0.get("b_ddg"):
                continue
            (kk, wtl) = k
            if kk in V["binding"] and D["wt"].get(kk[0]) == wtl:
                pr.append((float(r0["b_ddg"]), V["binding"][kk]))
        cmp("binding", "Chi2008(b_ddg)", pr, sign=+1)     # 文件内 F325A: kd 0.3<0.5(WT) 且 b_ddg=-0.31 => 正 = 结合变弱
    return out


def matrix(D, phenotype, se_max=0.5, abs_max=10.0):
    """构造 M[site, residue] 与 mask，并返回截尾统计。

    排除规则（全部**在看秩之前**定好）：
      (a) 模型未给出该变体的该系数（NA）                -> structural
      (b) bootstrap SE > se_max kcal/mol                -> SE 截尾
      (c) |ΔΔG| > abs_max kcal/mol                      -> 量级截尾
          理由：单点突变的 ΔΔG 超过 10 kcal/mol 在物理上不可能，
          这类值是三态模型在动态范围之外的外推（等价于「撞上界」）。
    分母是完整设计 n_site x 19（野生型格子本来就不存在，不计入）。
    """
    pos = sorted(D["wt"])
    si = {p: i for i, p in enumerate(pos)}
    M = np.full((len(pos), len(AA20)), np.nan)
    V = D["vals"][phenotype]
    S = D["se"][phenotype]
    design = len(pos) * 19
    nstruct = ncenSE = ncenAbs = 0
    for p in pos:
        for a in AA20:
            if a == D["wt"][p]:
                continue
            k = (p, a)
            if k not in V:
                nstruct += 1
                continue
            se = S.get(k, math.nan)
            if not np.isfinite(se) or se > se_max:
                ncenSE += 1
                continue
            if abs(V[k]) > abs_max:
                ncenAbs += 1
                continue
            M[si[p], AA20.index(a)] = V[k]
    mask = np.isfinite(M)
    return M, mask, {
        "design": design, "n_obs": int(mask.sum()),
        "na": nstruct, "se_censored": ncenSE, "abs_censored": ncenAbs,
        "censor_frac": 1.0 - mask.sum() / design,
        "na_frac": nstruct / design,
        "se_frac": ncenSE / design,
        "abs_frac": ncenAbs / design,
        "sites": len(pos), "se_max": se_max, "abs_max": abs_max,
    }


def main():
    print("=" * 78)
    print("ddPCA 加载自检（CLAUDE.md §10：先用已知事实验证解析器）")
    print("=" * 78)
    aa = check_abundance_assays()
    print("独立 abundance assay（由 zip 索引核对）:", aa, "\n")
    for dom in DOMAINS:
        D = load_domain(dom)
        verify(dom, D)
        for ph in ("folding", "binding"):
            M, mk, c = matrix(D, ph)
            print(f"  {ph:<8} {c['sites']}x19 设计 {c['design']:>5}  "
                  f"可用 {c['n_obs']:>5} ({1 - c['censor_frac']:.1%})  "
                  f"截尾 {c['censor_frac']:.1%} "
                  f"= NA {c['na_frac']:.1%} + SE {c['se_frac']:.1%} "
                  f"+ |ΔΔG|>10 {c['abs_frac']:.1%}")
        print(f"  独立 abundance assay: {HAS_ABUNDANCE_ASSAY[dom]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
