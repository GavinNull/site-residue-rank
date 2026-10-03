# 实验误差宽度 σ_exp（任务 M2-T1，已完成首轮）

调研日期 2026-10-03。来源区分「已核到原文表/图号」与「仅摘要/二手引用」。

## 0. 换算约定

T = 298.15 K，RT = 0.5925 kcal/mol，**RT·ln10 = 1.364 kcal/mol 每 log 单位**。

- log 单位 → ΔG：σ_ΔG = 1.364 × σ_plog
- Kd 相对误差 CV → ΔG：σ_ΔG = RT·√(ln(1+CV²))
- **成对差 → 单次**：σ_single = σ_pairdiff/√2（最易搞错的一步）

## 1. 关键结论：排序损失该用的是成对量

M2.3 原本写「温度 = √2·σ_exp」，即由单次 ΔG 误差推成对误差。
但文献里**有直接以 kcal/mol 给出的成对量**，定义与排序损失的数学形式完全对齐：

> Ross et al., *Commun Chem* 2023;6:222, doi:10.1038/s42004-023-01019-9
> Table 3：总体 pairwise RMSE = **0.91** kcal/mol [0.83, 1.11]，MUE 0.67
> Table 4：binding vs binding = **1.10** [0.85, 1.34]；
>          binding vs functional 0.93；functional vs functional 0.75
> （定义：同一蛋白同一配体集，两个不同 assay 各自算 ΔΔG 后的 RMSE，去系统偏移）
> 已核到原文表号。

**占位值 0.5 对应的成对尺度是 √2×0.5 = 0.71，偏乐观 25–55%。**
跨文献拼接的生物物理 assay 应取 **1.10**；混合数据集取 **0.91**。

## 2. 分类汇总

| assay | 误差类型 | σ_ΔG (kcal/mol) | n | 来源 | 核实状态 |
|---|---|---|---|---|---|
| ITC | 跨实验室（round-robin） | 0.13（CV 22%） | 14 实验室 | ABRF-MIRG'02, *J Biomol Tech* 2003;14(4):247–269, PMID 14715884 | **仅摘要级二手引用** |
| SPR 小分子 | 跨实验室 | 0.21（CV 37%） | 26 操作者 | Papalia 2006, doi:10.1016/j.ab.2006.08.021 | 仅摘要 |
| SPR 蛋白-蛋白 亚nM | 跨实验室 | 0.66（上界，分布强右偏） | ~150 参与者 | Rich 2009, doi:10.1016/j.ab.2008.11.021 | 摘要 |
| Ki（竞争结合为主） | 跨文献 | 0.74（σ=0.54 pKi） | ChEMBL v12 | Kramer 2012, doi:10.1021/jm300131x | 摘要，数字被两篇独立复核 |
| IC50 | 跨文献 | **0.93**（σ=0.68 pIC50） | 16,844 对 | Kalliokoski 2013, doi:10.1371/journal.pone.0061007 Table 3 | **已核原文** |
| IC50 | 同实验室（参比化合物长期重复） | 0.23–0.30 | 多年重复 | 同上 Figure 9 | **已核原文** |
| Ki | 同实验室 | 0.41（0.3 pKi 中位 SD） | 工业内部库 | Brown 2009, doi:10.1016/j.drudis.2009.01.012 | 二手引述 |
| 混合（社区推荐值） | 跨文献单次 ΔG | **0.64** | 365 分子实测 | Hahn 2022 LiveCoMS 4(1):1497 §4.5.3 + Fig 6, arXiv:2105.06222 | **已核原文** |

## 3. 推荐取值

| 用途 | 取值 | 依据 |
|---|---|---|
| **排序损失温度（跨文献拼接，生物物理 assay）** | **1.10** kcal/mol | Ross 2023 Table 4 |
| 排序损失温度（混合数据集） | 0.91 | Ross 2023 Table 3 |
| 单次 ΔG（跨文献，若需要） | 0.64 | Hahn 2022 社区推荐 + Moonshot 实测 |
| 同一来源内部 | 0.3–0.4 | Kalliokoski Fig 9 / Brown 2009 |

## 4. 同方差 vs 异方差：用同方差，但按来源分层

**误差不依赖亲和力大小。** 唯一在万级样本上正面检验过的研究：

> Kalliokoski 2013, Figure 6 + 正文（已核）：
> "ΔpIC50's depend neither on the average measured pIC50 nor on any of the
> ligand properties examined."
> 测过的协变量：平均 pIC50、logP、logD、MW、PSA、HBA、HBD、可旋转键数；
> Figure S3 做了「平均活性 × logP」二维分箱，"We do not see a clear trend"。

边界效应存在但很弱：Ross 2023 剔除接近检测限的数据后 RMSE 从 0.85 降到 0.79（7%）。

**真正值得分层的是数据来源，不是亲和力**：
同实验室 0.23–0.41 vs 跨实验室 0.74–0.93，差 **3–4 倍**；
清洗严格 vs 宽松差 1.8 倍（Landrum & Riniker 2024, doi:10.1021/acs.jcim.4c00049，已核）。

→ `pairs/*.yaml` 新增必填 `provenance.same_source`，损失用逐对 σ。

## 5. 必须避开的坑（已量化）

**不要用论文自报的 ±。** 那通常是单次实验内曲线拟合的形式标准误，
不是重复性误差。最硬的证据是 ABRF-MIRG'02：14 个实验室各自报的最小二乘标准误
比他们之间的实际离散度**小一个到三个数量级**
（原话见 Nguyen/Chodera et al. 2018, doi:10.1371/journal.pone.0203224 引言）。

> Hahn 2022 §4.5.3（已核）:"Errors quoted will likely be an estimate of the
> repeatability of the assay, rather than true, independent reproducibility."

根因：系统误差（滴定剂浓度、蛋白活性比例、固定化密度、参比扣除）
完全不进入拟合的协方差矩阵。用自报 ± 会把温度设低 5–20 倍，
排序损失严重过度自信。

## 6. 最大的单点不确定性（需人取原文）

**ABRF-MIRG'02 的 KA = (1.00 ± 0.22)×10⁶ M⁻¹，±0.22 是 SD 还是 SEM？**
若是 SEM（n=14），真实 SD = 0.22×√14 = 0.82 → CV 82% → σ_ΔG = **0.45**，
而非 0.13。**三倍以上的差别。** PMC2279960 被 reCAPTCHA 拦住，未取得。

次要：Papalia 2006 的 34%/24%/37% 是实验室间 SD 还是各化合物内重复的平均 CV，
措辞 "experimental errors ... averaged" 有歧义。ScienceDirect 403。

## 7. 明确查不到

- **ITC 的同实验室多次重复 σ 汇总统计**：不存在。只有单次实验的误差传播分析
  （给的是「理论可达 1–3% 精度」，那是精度上限不是重复性）。
- **放射配体竞争结合的跨实验室 round-robin**：不存在。只能用 ChEMBL Ki 作代理，
  方法学里须写明。
- **SAMPL / D3R 文献里自己测定的 σ_exp**：空。它们要么引 Kramer/Brown，
  要么只给预测端 RMSE。**不要把这条写进引用。**
- ABRF-MIRG 的三组分 benchmark（Yamniuk 2012, doi:10.7171/jbt.12-2303-003）
  对本项目无用：原文明确写 "Determination of kinetic or equilibrium affinity
  constants is not required"，无 KD 离散度统计。
