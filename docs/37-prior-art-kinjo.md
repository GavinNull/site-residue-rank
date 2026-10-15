# Kinjo & Nakamura 2008 全文审读：它做了什么、没做什么（2026-10-15）

本文件回答 docs/36 §6.4 第 2 条留下的缺口（「[31] 只读了摘要，没读 Methods」）。
全文已取回并通篇读完。每条陈述标 **〔读到〕** 或 **〔推断〕**。
`facts:` / `assessment:` 按 CLAUDE.md §2 分离；判断带 confidence 与 basis。

**不编辑 docs/35、docs/36、docs/16。** 给 §1.2 的建议措辞见本文件 §8，由人类决定是否采用。

---

## 0. 结论摘要

| 问题 | 答案 |
|---|---|
| 是否双向中心化？ | **完全没有中心化。** 全文唯一一次减均值是对接触数向量 `n`，且那是用于一次对照相关，不是对 PSSM 做的。〔读到〕 |
| 是否有任何东西起定秩作用？ | **没有。** 只有累积贡献率（描述性），无碎石检验、无奇异值显著性阈值、无交叉验证、无「k 个成分够了」的论断；其结论反而是 PSSM **比**位置无关替换矩阵**更不可压缩**。〔读到〕 |
| 成分化学身份与本文的关系 | 其成分 1 ≈ 位点主效应；去掉它之后，余下两条最强残基轴是**疏水性**（成分 2）与**侧链体积**（成分 3）—— 与本文共识轴的化学**成分重叠而次序相反**。〔读到 + 推断〕 |
| 是否构成先验技术？ | **不构成。** 三条独立理由，最强的一条是**对象不同**（未中心化 → 其「第一成分」与本文「第一成分」不是同一类客体），不是「数据类型不同」。 |
| 前向引用里有更近的先验技术吗？ | **没有。** OpenAlex 15 篇、Semantic Scholar 13 篇，全部读过标题与归属；最近的一篇是 20×20 替换矩阵的 SVD（Zimmermann & Gibrat 2010），属氨基酸属性空间那一类。 |

**最有价值的一条新发现（方向上对本文有利，且是 docs/36 没预料到的）**：
他们的次序（疏水性先于体积）与**本文折叠一侧**（docs/27：疏水性 r² 0.619、极化率 0.257）一致，
而与**本文结合一侧**（体积/极化率为主干）相反。他们的左奇异向量对偶量是
**接触数与位点保守性**，即纯结构/埋藏信号。所以这不是矛盾，而是
**对本文「折叠—结合分化」这一条的独立外部支持**（见 §4.4）。

---

## 1. 文献元信息（provenance，CLAUDE.md §3）

`facts:`
- Kinjo AR, Nakamura H. **Nature of Protein Family Signatures: Insights from Singular Value
  Analysis of Position-Specific Scoring Matrices.** *PLoS ONE* 2008;3(4):e1963.
  doi:10.1371/journal.pone.0001963. PMID 18398479. PMC2276316. 预印本 arXiv:0711.1010.
- 编辑 David Jones（UCL）；作者单位大阪大学蛋白质研究所；2008-04-09 上线。
- 取回路径：`https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0001963&type=manuscript`
  → NLM Journal Publishing DTD v3.0 XML，115,692 字节，本地转纯文本 411 行后通篇读完。
  标题、作者、卷期号、DOI 与 docs/36 [31] 已登记的条目逐字一致（**不是靠 DOI 猜中**）。〔读到〕
- **全文无 Supporting Information 节，无补充文件。** Table 1–5 全部在正文内，
  已逐表读过内容。所以本文件不存在「补充载荷表取不到」的缺口。〔读到〕

`assessment:` docs/36 [31] 的书目信息**无需更正**。confidence: 高；basis: 出版方 XML 原文。

**无法取到的部分（按任务要求明示）**：
- **Fig. 1B 之外没有单个奇异值的数值。** Fig. 2 只给**累积**贡献率曲线，
  且是 1096 / 8869 条矩阵上的**平均**。因此 σ₂ 与 σ₃ 的间距大小**读不出来**，
  本文件 §2.3 关于「成分 2 / 成分 3 的指派对旋转敏感」的话是**〔推断〕**，不是读到的。

---

## 2. 矩阵是什么、对 SVD 做了什么

### 2.1 矩阵（精确到索引）

`facts:`〔全部读到〕
- **每一个蛋白（族）一个 N×20 矩阵，逐个单独做 SVD**，不是把所有数据拼成一张大矩阵。
  原文：「A position-specific scoring matrix (PSSM) is a real rectangular matrix of size
  N×20 where N is the length of the amino acid sequence of a protein (or protein family).」
- 行 = 序列中的位点（site）；列 = 20 种氨基酸类型；
  元素 `M_ij` = **位点 i 上氨基酸 j 的打分**（score），即**位置特异的 log-odds 替换打分**。
  原文把 PSSM 归为 log-odds 矩阵：「This is an expected behavior for log-odds matrices」。
- 两套数据，各自独立分析：
  - **PDB 集**：PISCES 筛出的 **1096** 条 X 射线代表链（25% 序列同一性、R-factor 20%、
    2.0 Å、长度 40–500，限 SCOP 的 all-α/all-β/α/β/α+β/multi-domain/small 七类）。
    每条链的 PSSM 由 **PSI-BLAST 对 UniRef100（release 12.1）**跑出，
    e-value 截断 0.0005、3 轮迭代。
  - **Pfam 集**：Pfam release 22.0（2007-07）中长度 ≥ 40 的 **8869** 个家族，
    **只取 HMM 的 match-state 打分**当作 PSSM 元素。
- 假设 N > 20，「although this condition is not strictly necessary」。
- 分解写作 `M = U Σ Vᵀ`：`v_α`（20 维，右奇异向量 r-SV）被明确当作**氨基酸指数**使用；
  `u_α`（N 维，左奇异向量 l-SV）被当作「generalized 1D structure」。

`assessment:`
- **索引形状确实与本文相同**（位点 × 20 残基），docs/36 这一句刻画准确。
- 但元素的语义完全不同：这是**比对频率相对背景频率的对数几率**，
  是「这个家族在这个位点偏好哪种残基」的进化统计量，
  **不是任何热力学测量**。全文没有出现任何实验自由能、ΔΔG、Kd、结合测量。
  confidence: 高；basis: Methods 全文 + 全文检索无热力学量。

### 2.2 中心化 —— 这是最关键的一条

`facts:`〔读到〕
- **Methods 完全没有任何中心化、标准化、去均值步骤。** SVD 直接作用在原始 PSSM 上。
- 全文唯一一次减均值：§「Characteristics of second singular components」里
  「`n` represent the contact number vector of a protein **standardized by subtracting
  the average value from each element**」—— 这是对**接触数向量**做的，
  用于与 `σ₂u₂v₂ᵀ` 做一次相关对照，**不是对 PSSM 做的**。
- 唯一的其他「处理」是**符号约定**：r-SV 取使半胱氨酸为正；l-SV 取使多数元素为正。
- 未中心化的直接后果，作者自己写出来了（这是决定性证据，不是我的推论）：
  - 「**Recall that all the elements of the first r-SVs are of the same sign in most cases**
    so that we can make them all positive without losing generality.」
  - 「most substitutions are disfavored by the first singular component of a PSSM」；
    Fig. 1C 中第一成分的贡献「is **purely negative**」。
  - Fig. 3：`M₁` 的负元素比例远高于原矩阵与其他 `M_k`。
  - 第一个 l-SV 与**位点信息量**（Kullback–Leibler 散度，位点保守度）相关，
    PDB 集上平均 r = **0.543**（SD 0.218，中位 0.601，P < 10⁻¹⁷）。
  - 「l-SVs other than the first one did not show high correlations with information
    content (data not shown).」

`assessment:` **他们的第一成分是主效应，不是交互成分。** confidence: 高。
basis: 右奇异向量多数情形元素同号（即方向接近常向量 **1**），
左奇异向量与「这个位点有多保守/多敏感」相关 r = 0.543 ——
这正是 `b(s)·1ᵀ` 型的位点主效应外积。本文的双向中心化按构造**先把这一项整个去掉**。
因此「他们的第一成分」与「本文的第一成分」**不是同一类客体**，
而且**成分编号整体错位**：本文的第一成分最多只能与他们的**第二**成分对位。

**必须同时说清楚、否则这条论证会被审稿人反推**：
他们的成分 2 与成分 3 **不是**主效应。他们自己把成分 2 写成 `w nᵀ`
（`w` = Miyazawa–Jernigan 相对分配能，`n` = **已中心化**的接触数向量），
这是两个都非常向量的外积，**在双向中心化下不会被消掉**，
所以成分 2/3 在性质上**确实是交互型客体**。
正确的说法不是「他们从不碰交互项」，而是
**「他们的成分 1 吸收了主效应，导致成分编号与本文不可直接对位」**。
confidence: 高；basis: 他们自己给出的 `w nᵀ` 形式 + 双向中心化的代数性质。

### 2.3 奇异值与定秩 —— 直接核验 docs/36 的「没有定秩程序」

`facts:`〔读到〕
- 他们报告**累积贡献率** `S_k`（Eq. 2，XML 未渲染公式本体，但正文给出全部数值），Fig. 2：
  - 第一个奇异值占奇异值总和的 **17%（PDB）/ 24%（Pfam）**；
  - **50%** 由前 **4 或 5** 个成分达到（PDB / Pfam）；
  - **90%** 由前 **15** 个成分达到（两集皆然）；
  - 对照位置无关替换矩阵（AASM，引自 Kinjo & Nishikawa 2004）：50% / 90% 分别是 **3** 和 **10** 个成分。
- 他们由此得出的结论是：「the "**compressibility**" of PSSMs is **lower** in the sense that
  **more components are needed** to explain the same fraction」。
- 分析范围：「The analysis focuses on the first five components」（成分 1–5，前三者详述）。
- 全文唯一的显著性阈值是 **|r| ≥ 0.6**，用于判定某条 AAindex 与 r-SV 的匹配是否算显著
  —— 这是**成分化学归属**的阈值，**不是**奇异值的阈值，与定秩无关。
- 没有碎石图、没有对奇异值做零分布或置换检验、没有交叉验证、
  没有任何「k 个成分足够」的论断、没有「秩」（rank）一词的实质使用。

`assessment:` **docs/36 的判断在全文层面成立，且比原判断更强。**
不仅「没有定秩程序」，而且他们的**结论方向与低秩相反**：
他们强调 PSSM 需要**更多**成分、可压缩性**更低**，并不主张低秩结构存在。
confidence: 高；basis: Results §Overview 全文 + Discussion 全文。
〔推断〕附带一点：由于 50% 要到第 4–5 个成分，σ₂ 与 σ₃ 很可能相当接近，
那么「成分 2 = 疏水性、成分 3 = 体积」这个**指派本身**就对近退化子空间内的旋转敏感
（本文 §3.1 正是为此只比较子空间）。但**单个奇异值的数值全文取不到**，
所以这一条只能标〔推断〕，不能当证据用。

---

## 3. 他们报告的成分身份（原文口径）

方法：对每条 PSSM 的第 α 个 r-SV，在 **AAindex release 9.1（2006-08）**全库里找
相关系数绝对值最高的那条指数；|r| ≥ 0.6 记为显著；按**被记为显著的次数**排序。
注意这是**逐矩阵取胜者、再数票**的频次统计，**不是**一条全局载荷向量。〔读到〕

| 成分 | r-SV（残基侧） | l-SV（位点侧） | 作者给的解释 |
|---|---|---|---|
| 1 | **多样、随家族而变**，无单条指数主导（最高仅 10%）：相对可突变性 JOND920102（PDB 10%）、蛋白内部组成 FUKS010106（7%）、Sneath PC I（Pfam 9%）、Kyte–Doolittle KYTJ820101（PDB 4% / Pfam 6%）、**β-转角**构象参数 BEGF750103（5% / 3%） | **位点信息量**（KL 散度），r = 0.543 | 「disfavor **potentially but falsely** functionally important residues at conserved sites」—— 一个**惩罚项** |
| 2 | **几乎全是疏水性标度**：Miyazawa–Jernigan 相对分配能 MIYS990101（**33% / 54%**）、Bastolla interactivity、Grantham 极性、Sweet–Eisenberg。前十条指数覆盖 85%（PDB）/ 94%（Pfam）。原文：「the second r-SVs are **almost exclusively determined by hydrophobic properties**」 | **接触数**，r = 0.511（SD 0.113） | 「the second singular component represents the **structural stability** of the protein」；`σ₂u₂v₂ᵀ` 与 `w nᵀ` 在 361 个蛋白上平均相关 **−0.45** |
| 3 | **体积 / 体积庞大度**：CHAM830106（最长链键数，20% / 15%）、Sneath PC III、LEVM760102（Cα–侧链质心距）、LEVM760105（侧链回转半径）、OOBM770105 | **Voronoi 体积**，r = **0.345**（P < 0.0003；限定到特定子集时 0.366 / 0.251 / 0.479） | bulkiness；作者自评「significant, but **not as consistent as** those of the first and second」 |
| 4–5 | 螺旋倾向 KOEP990101、螺旋帽倾向、净电荷 KLEP840101、α-NH 化学位移 | — | 「not so clearly characterized」 |

`facts:` 另两条读到的、与本文有关的细节：
- 他们明确区分了 Kyte–Doolittle 与 MJ 分配能：两者相关 **−0.84**，但 KD 出现在**第一**成分、
  MIYS990101 出现在**第二**成分，「there seems to be a meaningful difference between them」。
- Discussion 自述与前人的关系：Tomii & Kanehisa 认为 AASM 的主要成分是**侧链体积与疏水性**，
  Pokarowski 等加上**无规卷曲倾向**；他们说这三者也是 PSSM 的主要成分，
  且**一一对应到自己的成分 1（卷曲/转角）、2（疏水性）、3（体积）**。

---

## 4. 并排对照：他们的成分 vs 本文的成分

### 4.1 本文的口径（供对照，取自 docs/35 §3.3 与 docs/13 §7.1）

- 对象：**双向中心化后的交互项**（扣除残基主效应 a(r) 与位点主效应 b(s)），AMMI 式残差。
- 第一轴（两组家族共识，9 个矩阵）：β 折叠倾向 r² **0.607**、侧链极化率 0.568、
  摩尔折射度 0.519、可及表面积 0.431、**Kyte–Doolittle 0.311**。
  最正端 F/W/Y/M/L，最负端 P/D/A/G/S。
- 第二轴（冠状病毒家族内部残差主轴）：平面 sp² 指示量 r = **+0.81**、侧链臂长 +0.73、
  氢键可达性 +0.72、**Kyte–Doolittle −0.70**。即**芳香/平面 vs β-分支**的对立。
- 折叠一侧（docs/27）：**疏水性反而是主干 r² 0.619，极化率跌到 0.257**。

### 4.2 第一成分能不能对位？

**不能直接对位。** 他们的成分 1 是位点主效应型客体（§2.2），
本文的第一成分是**主效应已被扣除之后**的交互项首轴。
若强行按「去掉主效应后的首个非平凡轴」对位，
应当拿他们的**成分 2**（疏水性 × 接触数）对本文的**第一轴**。

`assessment:` 对位之后的判断：

| | 本文（结合，实测 ΔΔG，双向中心化） | Kinjo 去主效应后的首轴（成分 2） |
|---|---|---|
| 主干化学量 | **侧链体积 / 极化率 / 埋藏面积 / β 折叠倾向** | **疏水性**（MJ 分配能，33–54% 的矩阵） |
| 疏水性地位 | **次要**（r² 0.311） | **几乎是全部** |
| 下一条轴 | 平面 sp² / 芳香 vs β-分支（界面特异，未能归因） | **侧链体积 / bulkiness**（成分 3） |

**判断：化学成分的集合吻合，次序相反。** confidence: 中高。
basis: 双方都把「体积」与「疏水性」识别为两条最强的残基轴，这一点是吻合的；
但本文是体积在前、疏水性次要，他们是疏水性在前、体积在后。
**这一半是独立印证（两条轴的身份），另一半与本文方向相反（哪条是第一）。**
按任务要求，后者必须明说，它削弱「普适轴不是经典疏水性」这句的惊奇程度。

### 4.3 第二成分能不能对位？

**不能。** 本文的第二轴是**芳香/平面 vs β-分支**、界面特异、未能归因；
他们的成分 2 是疏水性（已对位到本文第一轴），成分 3 是体积，
成分 4–5 是螺旋帽倾向与净电荷。**本文的第二轴在他们那里没有对应物。**
最接近的候选是成分 3（体积），但本文第二轴与体积类标度没有高相关
（docs/13 §7.2 最高是平面 sp² +0.81，而非体积）。
confidence: 中高；basis: 双方载荷/归属表并排比较。
`gaps`: 他们不报告逐条 r-SV 的 20 维载荷数值（只报告「哪条 AAindex 匹配、匹配了多少比例」），
所以**无法做本文 §3.1 那样的主角余弦子空间比较**。这是本次比较的硬限制，不是可补的。

### 4.4 为什么次序相反不构成矛盾（本节是本次最有价值的发现）

`assessment:` confidence: 中高。basis 如下，全部来自已落盘的本仓库结果与他们的原文：

1. 他们两条主要成分的**左奇异向量对偶量分别是位点保守度（r = 0.543）与接触数（r = 0.511）**，
   成分 3 的对偶量是 **Voronoi 体积**。这三者都是**埋藏/堆积结构量**，
   没有一个是界面量、配体量或结合量。他们自己的结论是
   「the second singular component represents the **structural stability** of the protein」，
   并把整件事总结为「sequence alignment using PSSMs may be regarded as **threading**
   supplemented with some functional information」。
   **即：他们的信号在物理上是折叠/埋藏信号。**〔读到〕
2. 本文在**折叠一侧**独立得到的恰恰是同一个次序：docs/27 实测
   **疏水性 r² 0.619 为主干、极化率跌到 0.257**；
   而体积/极化率为主干只出现在**结合一侧**的共识轴上。
3. 于是两边并不冲突：**他们量的是折叠侧的次序，本文量的是结合侧的次序，
   两边各自的次序互相一致。** 这反而是本文 §3.4 / §3.5
   「折叠—结合分化」那条论断的一次**独立外部印证，且来自进化序列统计这一完全不同的数据类型**。

`gaps` / 不许过度宣称的地方：
- 这是**定性方向一致**，不是定量一致。他们没有载荷向量，做不了 cos θ 检验，
  **不能写成「子空间对齐」**，也**不能进任何统计**。
- 他们的「疏水性」用的是 MJ 相对分配能（接触势导出），
  本文折叠侧用的是 Kyte–Doolittle。两者相关 −0.84（他们自己给的数），但不是同一条标度。
- 他们成分 1 里出现的 BEGF750103 是 **β-转角**构象参数，
  **不是**本文第一轴的 **β-折叠（sheet）倾向**。**这两条不得混为一谈。**
  本文第一轴的 sheet propensity (Chou–Fasman) 与他们任何成分的归属都不对应。

---

## 5. 先验技术判决

`assessment:` **不构成先验技术。** confidence: 高。
三条独立理由，按强度排序（最强的一条**不是**「数据类型不同」）：

**G1（最强）：对象不同，成分编号不可对位。**
他们对**未中心化**的矩阵做 SVD。未中心化矩阵的首成分被大均值与行/列均值支配，
作者自己给出的三项事实证实了这一点：第一个 r-SV 多数情形元素同号、
`M₁` 的贡献「purely negative」、第一个 l-SV 与位点保守度相关 r = 0.543。
因此他们的「第一成分」是**位点主效应**，而本文的秩是定义在
**两个主效应都被扣除之后**的交互残差上。两者的「第一成分」不是同一类客体。
（限定：他们的成分 2、3 确实是交互型外积，不能说他们完全没碰交互项 —— 见 §2.2 末。）

**G2：全文没有任何定秩程序。**
只有累积贡献率这一描述性统计（17%/24%、50% 要 4–5 个、90% 要 15 个），
无碎石检验、无奇异值显著性阈值、无置换零分布、无交叉验证、无「k 个成分足够」的论断。
唯一的阈值 |r| ≥ 0.6 用于 AAindex 匹配，与秩无关。
而且他们的论断方向**与低秩相反**（PSSM 比 AASM 更不可压缩）。
docs/36 这一条判断经全文核验**成立**。

**G3：数据类型不同。**
矩阵元素是比对导出的 log-odds 打分（PSI-BLAST PSSM / Pfam HMM 发射分数），
全文不含任何热力学测量。

`assessment:` 对 docs/36 §6.1 结论的影响：**结论不变，但威胁等级应从「中」下调。**
全文读完后它比摘要读起来时**更远**而不是更近，因为「未中心化」这一条在摘要里看不出来，
而它是比「数据类型不同」更干净、更难反驳的区分。
同时 §4.2 那一半（次序相反）是**对本文不利的新信息**，应当进 §1.2 或 §3.3 的限定，
不应只写有利的那一半。
confidence: 高；basis: 全文。

---

## 6. 前向引用检查

`facts:`〔读到〕
- **OpenAlex**（W2009837779）：`cited_by_count = 15`，15 条全部列出并读过标题/期刊/年份。
- **Semantic Scholar**（DOI 查询）：13 条，与 OpenAlex 一致（无新增条目）。
- 15 篇的归类：
  - 残基深度 / 接触数 / 残基性质预测：Prodepth (2009)、Profile CRF (2009, ×2)、
    甲基化位点预测 (2011)、Prodepth 系列、「Uncover the conserved property…」(2009)
  - 结构/比对工具与综述：SARST2 (2025)、Seq2Bind (2025)、AI 无模板结构预测综述 (2022)
  - 序列空间/统计模型：unified statistical model of MSA with direct coupling (2016)、
    「Cooperative folding transition in the sequence space」(2018)
  - 其他：淀粉样 (2012)、天然动力学 (2014)、鱼血红蛋白疏水性模式 (2010)、
    Nakamura 70 岁纪念专刊前言 (2022)
- **最近的一篇**：Zimmermann K, Gibrat J-F. **Amino acid "little Big Bang": Representing
  amino acid substitution matrices as dot products of Euclidian vectors.**
  *BMC Bioinformatics* 2010;11:4. doi:10.1186/1471-2105-11-4。
  摘要读到：对**替换矩阵**（20×20 AASM）做 SVD，得到氨基酸的欧氏向量表示，
  用来研究各理化属性对替换矩阵的相对重要性，并比较 PAM 与 BLOSUM 系列。
  **索引是 20×20，不是位点×残基；不含任何实测效应。**
  `gaps`: 全文未读（BMC → Springer 要求登录；PMC2823696 为 reCAPTCHA 验证页，未绕过）。
  判断**不构成先验技术**是基于标题 + 摘要 + 对象（20×20 AASM）作出的，
  这一层已足够定性 —— 它属 docs/36 表格第 4 行（Atchley 类，氨基酸属性空间）那一类。
  confidence: 中高；basis: OpenAlex 摘要全文 + 标题。

`assessment:` **前向引用里没有比 2008 年原文更近的先验技术。**
没有任何一篇把定秩程序加到这条线上，也没有任何一篇转到实测突变效应矩阵上。
confidence: 中高。
`gaps`（必须写出，否则这节会被当成比实际更强的背书）：
- **Google Scholar 未查**。OpenAlex/S2 都给 13–15，GS 通常更高；
  覆盖偏向被良好收录的文献。
- Europe PMC 的 citations 端点与 PMC 全文在本次会话中被网络策略/验证页阻断。
- 作者本人后续工作未系统追（只读到前驱 Kinjo & Nishikawa 2004，见 §9）。
- 一次针对性检索「AMMI / 双向中心化 + 位点×氨基酸 DMS + 定秩」无命中
  （命中的是植物育种的 AMMI 定秩方法学与 DeMaSk/Rosace-AA，后者已在 docs/36 表内）。

---

## 7. Alpay 等人的「无交互项也能解释很多方差」能否与本文对照

### 7.1 书目更正（docs/36 [33] 应更新）

`facts:`〔读到，NCBI esummary〕
该预印本**已正式发表**：Alpay BA, Nanda P, Nagy E, Desai MM.
*Genetics* 2026;**233**(2):**iyag104**. doi:**10.1093/genetics/iyag104**. PMID **42013302**
（esummary pubdate 2026 Jun 3；OUP 页面标 2026-04-21 上线）。
标题与四位作者与 docs/36 [33] 登记的逐字一致。
`assessment:` docs/36 [33] 现在应引期刊版而非 bioRxiv 版。confidence: 高。
`gaps`: **期刊版正文付费墙，未读。** 下列数值全部来自 **bioRxiv v2 全文**，
发表版可能与预印本不同。

### 7.2 关键事实：他们**根本没有报告任何「解释方差」的数字**

`facts:`〔读到，bioRxiv v2 全文〕
- 数据：ProteinGym v1.2，**217 个** DMS、>200 个 assay、**近 700,000 个**单氨基酸取代。
  表面可及度由 DSSP 跑 AlphaFold2 预测结构得到；参数用 Stan 做 MAP 估计。
- 模型：Plackett–Luce 序模型，`θ_i = logistic(β_{wt,mut} + β_SA · s_i)`，θ ∈ [0,1]。
- **唯一的定量性能指标是「平均每-scan 的 Spearman 秩相关」**，十折交叉验证（按蛋白分折）。
  按 assay 类型：**活性 0.43、结合 0.39、表达 0.44、机体适应度 0.39、稳定性 0.57**。
  较大的 scan 上 ρ 大致在 **0.3–0.6**。对照 MIF 反折叠模型 0.41。
- 摘要与 Discussion 里反复出现的是定性措辞
  「**explains much of the variance**」/「on par with sequence-based machine learning models」，
  **全文检索下没有任何 R²、方差分解或「解释了 X% 方差」的数字。**
- 自述限制（逐字）：「our models analyze effects of only single substitutions and
  therefore **cannot capture epistasis**」。
- 其他读到的数：他们的可交换性矩阵 vs Yampolsky–Stoltzfus ρ = 0.64，
  vs Munro & Singh 与 Høie 等 ρ = 0.94；AA-only 与全模型的可交换性次序 ρ = 0.91；
  他们的预测 vs TranceptEVE 的预测 ρ = 0.40。

### 7.3 判断：不可比，而且可做的那部分对照**对本文有利**

`assessment:` confidence: 高；basis: 他们的模型式与指标定义。

**（a）不可比，有两重理由：**
1. **指标不同量。** 他们报的是**scan 内秩**的 Spearman ρ；
   本文报的是中心化后 ΔΔG 的**方差份额**。ρ = 0.39 换成「秩方差的份额」约 ρ² ≈ 0.15
   〔推断，且这是个松的换算，秩相关平方不等于方差分解〕；
   分母也不同（他们含实验噪声、含全部位点；本文是过了截尾筛选的特定矩阵）。
   **不应把 0.39 与本文的 29–71% 直接并列。**
2. **他们的模型在本文的框架里是「主效应模型」，交互项的秩按构造是 0。**
   `θ` 只依赖（野生型残基、突变残基、一个表面可及度标量）。
   因此**任何两个野生型残基相同、预测 SA 相同的位点，对全部 19 种突变都得到完全相同的预测**。
   翻成本文 §1.1 的分解：他们有残基主效应 a(r)（经 20×20 可交换性矩阵，且按野生型条件化）、
   有位点主效应 b(s)（经 SA 的一个单调函数），**没有 交互(s,r) 这一项**。
   注意他们自己说的 "cannot capture epistasis" 指的是**位点×位点**；
   本文的交互项是**单取代内部的位点×残基**，他们的模型同样装不下 —— 这一点他们没点明。

**（b）可做的那部分对照，方向与 docs/36 §6.2 的担忧相反：**
在他们自己的五类 assay 里，**结合是表现最差的一类（0.39，与机体适应度并列最低），
稳定性是最好的一类（0.57）**。
即：一个完全没有位点×残基交互项的模型，**在结合上最弱、在稳定性上最强**。
这与本文 §3.4（普适轴与折叠共用）以及 §4（第二成分是界面自己的属性）**同向**，
并不构成「无交互项已经够好了」的反例。

**（c）给审稿人那句问题的现成答案：**
「若无交互项的模型已经做得这么好，交互项的秩为什么重要？」
→ 那个模型按构造对同一野生型残基、同一可及度的所有位点给出同一组预测，
因而**判别力恰好为零**（本文 §1.1 的论点），
而它在五类表型里**结合最弱（ρ = 0.39）、稳定性最强（ρ = 0.57）**；
它报告的是秩相关而非方差份额，与本文 29–71% 的交互方差份额不同量，不能互相替代。

---

## 8. 建议写进 docs/35 §1.2 的确切句子（由人类裁定是否采用）

> **短版（最低限度，三句）：**
>
> 对「位点 × 20 残基」这一索引形状做奇异值分解本身并不新：Kinjo 与 Nakamura [31]
> 已在 1096 条 PDB 代表链与 8869 个 Pfam 家族的位点特异打分矩阵（PSSM）上逐家族做过
> SVD，并给出了前三个成分的化学归属。但他们的矩阵是比对导出的 log-odds 打分而非实测
> 自由能，**且未作任何中心化**——其第一个右奇异向量在多数情形下元素同号、对应的左奇异
> 向量与位点保守度相关（r = 0.543），即第一成分主要承载位点主效应，而本文的秩是定义在
> **扣除双向主效应之后**的交互项上，两者的「第一成分」不是同一类客体。该文只报告累积
> 贡献率（前 4–5 个成分占 50%、前 15 个占 90%），**没有任何定秩、显著性阈值或成分个数
> 选择程序**，其结论反而是 PSSM 比位置无关替换矩阵更不可压缩。

> **附加的一句（建议务必保留；放 §1.2 末或 §3.3 限定处）：**
>
> 值得一并指出：该文在性质上属主效应的第一成分之外，余下两条最强的残基轴依次是**疏水性**
> （成分 2，对偶量为接触数）与**侧链体积**（成分 3，对偶量为 Voronoi 体积），
> 与本文共识轴的化学成分重叠而**次序相反**。这不构成反例：他们的对偶量全部是埋藏/堆积
> 结构量，其信号在物理上属折叠一侧，而本文在折叠一侧独立得到的也正是疏水性为主干（§3.4）；
> 体积/极化率为主干只出现在结合一侧的共识轴上。

`assessment:` 两段都应进。只写短版会漏掉 §4.2 那条不利信息，
而那条信息一旦被审稿人先说出来，短版的可信度会连带受损。
confidence: 高；basis: CLAUDE.md 全仓库目的（排除错误，不是产出看起来合理的结论）。

---

## 9. 顺带关闭的两个 docs/36 §6.4 缺口，以及两个待人类裁定的数字

`facts:`〔读到，Kinjo 参考文献表 + Crossref 独立核验，标题/作者/卷/页全部对上〕
- **docs/36 §6.4 第 3 条（「Kidera 等人的氨基酸性质因子分析未定位、未核验」）现可定位**：
  Kidera A, Konishi Y, Oka M, Ooi T, Scheraga HA. *Statistical analysis of the physical
  properties of the 20 naturally occurring amino acids.* **J Protein Chem** 1985;**4**:23–55.
  doi:**10.1007/bf01025492**。〔Crossref 书目检索，标题/五位作者/卷/页一致〕
- 本文的方法学前驱（docs/36 未登记，但 §1.2 若引 [31] 则宜一并知道）：
  Kinjo AR, Nishikawa K. *Eigenvalue analysis of amino acid substitution matrices reveals
  a sharp transition of the mode of sequence conservation in proteins.* **Bioinformatics**
  2004;**20**:2504–2508. doi:**10.1093/bioinformatics/bth297**。〔Crossref 核验一致〕
  这是 20×20 AASM 的特征值分解：高同一性下首成分是相对可突变性，<30% 同一性下变为疏水性。
  **同样是 20×20，不是位点×残基，不构成先验技术。**

`assessment:` 两个待人类裁定的数字不一致（本文件不修改任何文档，只报告）：
1. **Kyte–Doolittle 的 r² 在两处不同。** docs/35 §3.3 表给 r = +0.557 / r² = **0.311**；
   docs/13 §7.1 的 2026-10-12 修订（同一个「两组口径、9 个矩阵」）给 r = +0.472 / r² = **0.223**。
   同一口径下不应有两个值。confidence: 高（两处都读到）；basis: 两份文档原文。
   这正是 docs/31 数字审计该抓的类型，建议按 §9 人类决策点处理，**本文件不动它**。
2. **交互项方差份额的中位数。** 任务描述给的是「29–71%，中位 ≈ 56%」；
   docs/13 §4.3 的「交互/总方差」列确实是 **29.1%–70.5%**，
   但该列有数值的 8 个体系的中位是 **≈ 37.5%**，不是 56%。
   56% 更接近**另一列**（「秩 2 解释交互」，中位 ≈ 51%）或 docs/34 的 λ2 中位 0.562。
   §7.3 已按「不可比」处理，所以本次对照不依赖这个数；
   但若 §1.3 或 §10 要写这个份额，**先定口径**。

---

## 10. 本文件不说什么

- 不说 Kinjo 与 Nakamura 的成分与本文的成分「对齐」。他们不报告 20 维载荷数值，
  只报告「哪条 AAindex 匹配、在多少比例的矩阵上匹配」，
  因此**做不了主角余弦子空间比较，本文件里没有任何 cos θ 或统计检验**。
- 不说他们的工作支持秩 2。他们的论断方向相反（更不可压缩），且他们从不做定秩。
- 不说 Alpay 等人的 ρ 可以换算成本文的方差份额。§7.3(a) 已说明为什么不行。
- 不说前向引用检查是完备的。Google Scholar 未查，PMC 与 Europe PMC 的两条路径被阻断（§6 gaps）。
- 不说 Zimmermann & Gibrat 2010 的全文已读。只读了标题与摘要（§6 gaps）。
