# 参考文献核验与文献定位审计（2026-10-14）

本文档为 `docs/35-paper-draft.md` 建立可追溯的参考文献表，并审计该草稿的
文献定位主张。遵 CLAUDE.md §3：**每个字段都来自实际取回的一次primary source
响应**，取回不到的字段不填，列进 §4「UNVERIFIED」并写明试过什么。

**未修改** `docs/35-paper-draft.md` 或任何其他 docs 文件；`docs/16-why-two.md`
未触碰（§4 冻结预注册）。本文档只记录「应该改什么」。

## 0. 核验方法与它的边界

| 口径 | 用途 | 为什么算 primary |
|---|---|---|
| Crossref REST (`api.crossref.org/works/<DOI>`) | 作者/刊名/年/卷/期/页/文章号 | DOI 注册方的登记记录，由出版方存入 |
| NCBI E-utilities (`esummary`/`esearch`) | PMID ↔ DOI、页码、作者 | NLM 书目库 |
| Zenodo REST (`zenodo.org/api/records/<id>`) | 版本 DOI / concept DOI / 文件清单 | 存档方自身 API |
| RCSB Data API (`data.rcsb.org/rest/v1/core/entry/<ID>`) | PDB 条目题名与 primary citation | PDB 自身 |
| 本地 `cache/aaindex1.txt` 的 `H/D/A/T/J/R/I` 字段 | 标度的原始文献与 20 个数值 | AAindex 自带的文献字段即数据库的权威记录 |
| 出版方补充材料的 `Content-Length` 对比本地字节数 | 缓存文件 ↔ 论文的硬映射 | 字节级一致，不是名字相似 |

**边界（必须写明，否则这张表会被当成比实际更强的证据）**：
- Crossref 记录的是**出版方登记的元数据**，不是纸本。1880/1911 年那几条尤其如此（§3.7–3.9）。
- 「Crossref 能解析」只证明该 DOI 存在且元数据如此，不证明我读过正文。
  凡本文档对一篇论文**内容**（模型形式、覆盖范围、是否估秩）下判断的，
  在条目里注明读的是哪一部分（摘要 / Methods / 全文 PDF）。
- 猜 DOI 再去验证的做法本身留下了痕迹：本次**先按记忆猜、再送去解析**的尝试里，
  **7 个 DOI 与 3 个 PMID 落空** —— 其中 2 个返回 404（1CHO、Battaglia），
  **8 个解析成功但指向完全无关的论文**（Gollob 猜成 Jennrich & Sampson 的因子旋转、
  Gauch 猜成一条 Biometrics 短讯、1PPF 猜成光合反应中心、3SGB 两次分别猜成
  硫酸肝素蛋白多糖与 Na/K-ATPase、三个 PMID 分别猜成谷胱甘肽还原酶、
  酶学限速步骤定义与果蝇 dopa 脱羧酶基因）。
  **「解析成功」不等于「猜对了」** —— 这一类才是危险的，因为只看 HTTP 状态码会放过。
  全部由检索改正或留空。这正是不允许「按记忆填字段」的理由，
  也是本文档每一条都注明核验来源的理由。

---

## 1. 核验通过的参考文献（VERIFIED）

编号即建议在正文中使用的编号。格式：作者. 题名. 刊名 年;卷(期):页. DOI. 〔核验来源〕

### 1.1 新意定位的那两篇（草稿 §1.2 的全部重量压在这里）

**[1] Rosace-AA — 草稿所称「假设秩 1」的那个工具**
Rao J, Wang M, Howard MK, Macdonald C, Fraser JS, Coyote-Maestas W, Pimentel H.
Rosace-AA: enhancing interpretation of deep mutational scanning data with amino acid
substitution and position-specific insights. *Bioinformatics Advances* 2025;6(1):vbaf218.
doi:10.1093/bioadv/vbaf218.
预印本 bioRxiv doi:10.1101/2025.01.09.632281；PMC12809776；代码 github.com/pimentellab/rosace-aa。
〔Crossref `10.1093/bioadv/vbaf218`；模型形式与「>500 domain」取自出版方正文页〕

- 刊名、年份、作者数（7）与草稿的「Bioinformatics Advances 2025」**一致**。
- `facts:` 该文 Methods 式 (2) 为
  `β_v | φ, ν, ρ, σ² ~ Normal(φ_p(v) + ν_a(v)·ρ_p(v), σ²_p(v))`。
  即交互项确实是「残基向量 ⊗ 位点向量」的外积。
- `assessment:` **草稿把它刻画成「假设该交互为秩 1、形如 φ_p + ν_a·ρ_p」是准确的**
  （confidence: 高；basis: 式 (2) 原文；该文全篇不出现 rank / low-rank / 秩的讨论，
  即秩是先验固定的，没有被估计也没有被检验）。
  三条必须一并写进去的限定见 §5.1。

**[2] Rosace（[1] 的前作，用以支撑「Rosace-AA 是其氨基酸层扩展」）**
Rao J, Xin R, Macdonald C, Howard MK, Estevam GO, Yee SW, Wang M, Fraser JS,
Coyote-Maestas W, Pimentel H. Rosace: a robust deep mutational scanning analysis
framework employing position and mean-variance shrinkage. *Genome Biology* 2024;25(1):138.
doi:10.1186/s13059-024-03279-7. 预印本 bioRxiv doi:10.1101/2023.10.24.562292.
〔Crossref `10.1186/s13059-024-03279-7`〕

**[3] Husain & Murugan — 草稿所称「测过秩但索引是 site×site」的那篇**
Husain K, Murugan A. Physical Constraints on Epistasis.
*Molecular Biology and Evolution* 2020;37(10):2865–2874. doi:10.1093/molbev/msaa124. PMID 32421772.
预印本 arXiv:1910.09491。
〔Crossref `10.1093/molbev/msaa124`（卷/期/页由此而来）；
内容判断取自 arXiv:1910.09491 **全文 PDF 含 SI**〕

- `facts:` 其矩阵是 ΔΔF_ij，索引为**突变 i × 突变 j**。核心结果为
  `ΔΔF_ij ∝ (1/λ₀²)·θ_i θ_j + c_ij`，即一个秩 1 项加一个仅在物理接触处非零的稀疏项；
  并写明「若有 k 个软模，ΔΔF_ij 的预期秩为 k」。
- `assessment:` **「索引是位点×位点而非位点×残基」准确**（confidence: 高；basis: 上式的下标）。
  **但「确实测过秩」不准确**（confidence: 高）：那是围绕野生型结构做摄动展开得到的
  **理论推导**，再用弹性网络模型**模拟**与计算双突变环验证；
  他们没有从任何实测矩阵上估过秩。应改措辞，见 §5.2。

### 1.2 实际使用的数据来源

**[4] 大规模 cDNA 蛋白酶解折叠稳定性数据（草稿 §3.4 / §9.3 的折叠侧）**
Tsuboyama K, Dauparas J, Chen J, Laine E, Mohseni Behbahani Y, Weinstein JJ,
Mangan NM, Ovchinnikov S, Rocklin GJ. Mega-scale experimental analysis of protein
folding stability in biology and design. *Nature* 2023;620(7973):434–444.
doi:10.1038/s41586-023-06328-6.
〔Crossref `10.1038/s41586-023-06328-6`，两次独立取回一致〕
**仓库记录的 DOI、卷 620、页 434–444 全部正确**，作者共 9 人。

**[5] 同上数据的 Zenodo 存档**
Tsuboyama K, Dauparas J, Chen J, Laine E, Behbahani YM, Weinstein JJ, Mangan NM,
Ovchinnikov S, Rocklin GJ. *Mega-scale experimental analysis of protein folding
stability in biology and design* [数据集]. Zenodo, 版本 v2_230420, 2023-04-21.
doi:10.5281/zenodo.7992926. CC-BY-4.0.
〔Zenodo REST `api/records/7992926` + 两个 zip 的 preview 文件列表〕

- `facts:` **7992926 是版本 DOI，不是 concept DOI**；concept（全版本）DOI 为
  **10.5281/zenodo.7401274**（conceptrecid 7401274），当前标记 `is_last: true`。
- `facts:` `cache/dms/tsuboyama2023/tsuboyama_PROVENANCE.txt` 记的 7 个文件**全部存在**，
  但层级与清单的扁平写法不同：
  `Processed_K50_dG_datasets.zip`、`Data_tables_for_figs.zip` 在**记录顶层**；
  `Tsuboyama2023_Dataset2_Dataset3_20230416.csv`、`Single_DMS_list.csv`、
  `Double_DMS_list.csv`、`Triple_DMS_list.csv` 在 `Processed_K50_dG_datasets.zip` **内部**；
  `dG_extdG_data_Fig1.csv`（159.2 kB，与清单记的 159186 字节一致）在
  `Data_tables_for_figs.zip` **内部**。
- `assessment:` PROVENANCE 文件没有错，但它把两层压成一层。建议在清单里标注归属 zip
  （confidence: 高；basis: Zenodo 的 preview 文件树）。**选版本 DOI 还是 concept DOI
  是一个需要明确决定的编辑问题**，不是可以默认的（§5.6）。

**[6] ddPCA / doubledeepms — 三个结构域的配对折叠与结合自由能（草稿 §3.6 的那个配对体系）**
Faure AJ, Domingo J, Schmiedel JM, Hidalgo-Carcedo C, Diss G, Lehner B.
Mapping the energetic and allosteric landscapes of protein binding domains.
*Nature* 2022;604(7904):175–183. doi:10.1038/s41586-022-04586-4. PMID 35388192.
预印本 bioRxiv doi:10.1101/2021.09.14.460249，**题名不同**：
"**Global** mapping of the energetic and allosteric landscapes of protein binding domains"。
测序数据 GEO **GSE184042**；代码 github.com/lehner-lab/doubledeepms。
〔Crossref `10.1038/s41586-022-04586-4` 与 `10.1101/2021.09.14.460249`；
三结构域与双表型由出版方正文的 Extended Data Fig. 1 / Fig. 5 图注确认；Europe PMC〕

- `facts:` GB1、GRB2-SH3、PSD95-PDZ3 三个结构域，binding 与 folding 两套自由能，
  由 Extended Data Fig. 1 图注逐字确认（"...for the binding (top panels) and when
  existing, folding assays (bottom panels) of the GB1 (b), GRB2-SH3 (c) and
  PSD95-PDZ3 (d) domains"）。与草稿 §3.6 的描述相符。
- `facts:` 处理后数据包由 **Dropbox** 分发（doubledeepms README 的链接），上游文件名是
  `Data.zip`；**没有** Zenodo 存档（Zenodo API 查 "doubledeepms" 返回 0 条）。
- `assessment:` 本地 `cache/dms/lehner_doubledeepms_Data.zip` 是改过名的 `Data.zip`
  （confidence: 中高；basis: zip 内 `Data/` 顶层目录、`mochi/` 子目录与 README 描述一致，
  内部文件时间戳 2021-09 至 2021-12 与预印本时点相符；
  **但仓库没有记录下载 URL 与校验和**，所以这条映射靠内容而非靠 provenance，见 §5.7）。

**[7] 蛋白–蛋白结合亲和力突变数据库（OMTKY3 面板的来源容器）**
Jankauskaitė J, Jiménez-García B, Dapkūnas J, Fernández-Recio J, Moal IH.
SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy,
kinetics and thermodynamics upon mutation. *Bioinformatics* 2019;35(3):462–469.
doi:10.1093/bioinformatics/bty635.
〔Crossref `10.1093/bioinformatics/bty635`（两次独立取回一致）；
规模数字取自 OUP 文章页摘要与 life.bsc.es/pid/skempi2/ 数据库页〕

- `facts:` 摘要原文：手工整理 **7085** 个突变的结合数据，其中 1844 个含动力学、
  443 个含焓与熵变、440 个使结合不可检出。文章页另给 237 个蛋白–蛋白互作体系、
  345 个 PDB 条目、295 篇被引文献。数据库页现示 "Total number of mutations: 7085"。
- `assessment:` 仓库 `docs/11` §3 的「SKEMPI 2.0 里小分子配体是 0 个」
  **在「范围声明」这一级成立，在「逐条审计计数」这一级没有被本次核验覆盖**
  （confidence: 中；basis: 论文与数据库页都把范围定义为「已有复合物结构的蛋白–蛋白互作」，
  两者均未提及蛋白–小分子条目；但本次没有逐行复核 4899 行）。措辞建议见 §5.4。

**[8] 蛋白–配体亲和力突变数据库（§7.5 的筛查来源之一）**
Pires DEV, Blundell TL, Ascher DB. Platinum: a database of experimentally measured
effects of mutations on structurally defined protein–ligand complexes.
*Nucleic Acids Research* 2015;43(D1):D387–D391. doi:10.1093/nar/gku966.
（出版方登记的上线日期为 2014-10-16，刊期为 2015 Database issue。）
〔Crossref `10.1093/nar/gku966`（两次独立取回一致）；规模数字取自 OUP 文章页〕

- `facts:` 1008 个突变（其中 797 个单点）、250 个独特蛋白–配体复合物、451 个 PDB ID、
  207 个配体、142 个蛋白（UniProt）、182 篇来源文献、560 个以 Kd 给出的亲和力。
  数据库页只写 "over 1,000 mutations" 并要求引用该 2015 NAR 文。
- `assessment:` **未找到任何更新版论文**（见 §4）。不要引用一个 Platinum 2.0。

**[9] OMTKY3 面板的原始来源（Laskowski 组，加和性序列–反应性算法）**
Lu SM, Lu W, Qasim MA, Anderson S, Apostol I, Ardelt W, Bigler T, Chiang YW, Cook J,
James MNG, Kato I, Kelly C, Kohr W, Komiyama T, Lin TY, Ogawa M, Otlewski J, Park SJ,
Qasim S, Ranjbar M, Tashiro M, Warne N, Whatley H, Wieczorek A, Wieczorek M, Wilusz T,
Wynn R, Zhang W, Laskowski M Jr. Predicting the reactivity of proteins from their
sequence alone: Kazal family of protein inhibitors of serine proteinases.
*PNAS* 2001;98(4):1410–1415. doi:10.1073/pnas.98.4.1410. PMID 11171964. PMC29270.
〔Crossref `10.1073/pnas.98.4.1410`（作者 29 人，与 PubMed 一致）；NCBI efetch 11171964〕
摘要与草稿的描述吻合：十个 consensus 可变接触位点、每点 19 种变体全部表达并测自由能、
加和性 STRA；MeSH 化学条目确认六种酶（α-胰凝乳蛋白酶、胰弹性蛋白酶、白细胞弹性蛋白酶、
streptogrisin A、subtilisin 系）。**这是草稿 §4.2 面板应引的那一篇。**

**[10] 同组 P1 位全 20 残基系列（若正文具体用到 P1 序列）**
Lu W, Apostol I, Qasim MA, Warne N, Wynn R, Zhang WL, Anderson S, Chiang YW, Ogin E,
Rothberg I, Ryan K, Laskowski M Jr. Binding of amino acid side-chains to S1 cavities
of serine proteinases. *J Mol Biol* 1997;266(2):441–461. doi:10.1006/jmbi.1996.0781. PMID 9047374.
〔Crossref `10.1006/jmbi.1996.0781`；NCBI efetch 9047374〕
（备选，若需要加和性检验本身：Qasim MA et al., Laskowski M Jr. Testing of the
additivity-based protein sequence to reactivity algorithm. *Biochemistry*
2003;42(21):6460–6466. doi:10.1021/bi027186u. PMID 12767228.）

**[11] Tite-seq 方法与 4-4-20 scFv/荧光素应用（同一篇）**
Adams RM, Mora T, Walczak AM, Kinney JB. Measuring the sequence-affinity landscape
of antibodies with massively parallel titration curves. *eLife* 2016;5:e23156.
doi:10.7554/eLife.23156. PMID 28035901. PMC5268739.
〔Crossref `10.7554/eLife.23156`（作者 4 人、文章号 e23156、无页码）；NCBI efetch 28035901〕

- `facts:` **方法篇与 4-4-20/荧光素应用是同一篇**，不存在两篇。该文正文用 4-4-20 scFv
  （K_D ≈ 1.2 nM）作为检验对象，并以 4-4-20/荧光素共晶 **PDB 1FLR** 为 Fig. 2A。
- `facts:` 仓库记的 bioRxiv 036335 与 PubMed 28035901 指向同一工作。
- `assessment:` **eLife 用文章号，没有页码范围**。给这条写 "pp. X–Y" 就是编造
  （confidence: 高；basis: Crossref 的 `page` 字段为空、`article-number` 为 e23156）。

**[12] HT-MEK PafA 的两个小分子配体抑制数据（草稿 §7.5 所称「只覆盖四种残基的 Gly/Val 扫描」）**
Markin CJ, Mokhtari DA, Du S, Doukov T, Sunden F, Cook JA, Fordyce PM, Herschlag D.
Decoupling of catalysis and transition state analog binding from mutations throughout
a phosphatase revealed by high-throughput enzymology. *PNAS* 2023;120(29):e2219074120.
doi:10.1073/pnas.2219074120.
预印本 bioRxiv doi:10.1101/2022.11.07.515533（作者 7 人，正式版 8 人）。
〔Crossref `10.1073/pnas.2219074120` 与 `10.1101/2022.11.07.515533`〕
相关的 HT-MEK 方法篇：Markin CJ, Mokhtari DA, Sunden F, Appel MJ, Akiva E, Longwell SA,
Sabatti C, Herschlag D, Fordyce PM. Revealing enzyme functional architecture via
high-throughput microfluidic enzyme kinetics. *Science* 2021;373(6553):eabf8761.
doi:10.1126/science.abf8761.〔Crossref〕

- `assessment:` `cache/dms/htmek_PafA_{VO4,WO4}_inhibition.csv` 对应 **2023 PNAS**
  （confidence: 中高；basis: 两个过渡态类似物恰为钒酸盐 VO4 与钨酸盐 WO4，与该文主题逐字对应；
  CSV 列含 `Ki/KiNorm` 抑制常数与 `ChipType = GlyScan/ValScan`。
  **但仓库没有记录下载 URL、字节数或校验和**，所以这条是内容映射，不是 provenance 映射，见 §5.7）。

**[13] MAGMA-seq（§7.5 筛查里的 11 个矩阵）**
Petersen BM, Kirby MB, Chrispens KM, Irvin OM, Strawn IK, Haas CM, Walker AM,
Baumer ZT, Ulmer SA, Ayala E, Rhodes ER, Guthmiller JJ, Steiner PJ, Whitehead TA.
An integrated technology for quantitative wide mutational scanning of human antibody
Fab libraries. *Nature Communications* 2024;15(1):3974. doi:10.1038/s41467-024-48072-z.
〔Crossref `10.1038/s41467-024-48072-z`〕

- `facts:` **本地四个文件与出版方补充材料字节级一致**（HTTP `Content-Length` 对比
  `static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-48072-z/MediaObjects/41467_2024_48072_MOESM<n>_ESM.xlsx`）：
  MOESM3 123599/123599、MOESM4 32602/32602、MOESM5 94350/94350、MOESM6 599076/599076。
  **四个全中。** 这是本文档中唯一达到字节级的缓存↔论文映射。

### 1.3 标度与参考值

**[14] AAindex 数据库（本仓库用 `cache/aaindex1.txt`）**
Kawashima S, Pokarowski P, Pokarowska M, Kolinski A, Katayama T, Kanehisa M.
AAindex: amino acid index database, progress report 2008.
*Nucleic Acids Research* 2008;36(Database issue):D202–D205. doi:10.1093/nar/gkm998. PMID 17998252.
前身两篇（如需完整谱系）：
Kawashima S, Ogata H, Kanehisa M. AAindex: Amino Acid Index Database.
*Nucleic Acids Res* 1999;27(1):368–369. doi:10.1093/nar/27.1.368. PMID 9847231.
Kawashima S, Kanehisa M. AAindex: amino acid index database.
*Nucleic Acids Res* 2000;28(1):374. doi:10.1093/nar/28.1.374. PMID 10592278.
〔Crossref 三条各自取回；PMID 经 NCBI efetch 确认〕

**[15] Kyte–Doolittle 疏水性（本仓库 accession `KYTJ820101`）**
Kyte J, Doolittle RF. A simple method for displaying the hydropathic character of a
protein. *J Mol Biol* 1982;157(1):105–132. doi:10.1016/0022-2836(82)90515-0. PMID 7108955.
〔Crossref `10.1016/0022-2836(82)90515-0`；NCBI esummary 7108955；
本地 `cache/aaindex1.txt` 的 `KYTJ820101` 条目 `A/T/J/R` 字段逐字一致〕

**[16] Fauchère–Pliska π（本仓库 accession `FAUJ830101`）**
Fauchère JL, Pliska V. Hydrophobic parameters π of amino-acid side chains from the
partitioning of N-acetyl-amino-acid amides.
*Eur J Med Chem* 1983;18:369–375. 〔**无 DOI / 无 PMID** —— 见 §4〕
〔本地 `cache/aaindex1.txt` 的 `FAUJ830101` 条目 `A/T/J` 字段；其 `R`（PMID）字段为空，
这是文件里的实际状态，不是解析产物〕

**[17] 摩尔折射度（本仓库 accession `MCMT640101`）**
McMeekin TL, Groves ML, Hipp NJ. 收于 *Amino Acids and Serum Proteins*
(Stekol JA, ed.), American Chemical Society, Washington, D.C., p. 54 (1964).
AAindex 条目同时注明 "Cited by Jones (1975)"，即**数值经由一份 1975 年汇编转录**。
〔本地 `cache/aaindex1.txt` 的 `MCMT640101` 条目逐字〕

- `facts:` 该条目的 `T`（题名）字段与 `R`（PMID）字段**都是空的**；AAindex 没有记录章节题名。
- `facts:` 仓库自检期望值**全部命中**：Gly = 0.00（最小）、Trp = 42.53（最大）。
- `assessment:` **把它写成「McMeekin 1964」这一篇期刊论文是过度陈述**
  （confidence: 高；basis: 条目本身的 `J` 字段是书章、`D` 字段明写转引）。
  正确写法见 §5.5。

**[18] 「Rose 埋藏表面积」实际用的不是 Rose —— 见 §5.3。本仓库 accession `RADA880106` 对应：**
Radzicka A, Wolfenden R. Comparing the polarities of the amino acids: side-chain
distribution coefficients between the vapor phase, cyclohexane, 1-octanol, and
neutral aqueous solution. *Biochemistry* 1988;27(5):1664–1670. doi:10.1021/bi00405a042.
〔Crossref `10.1021/bi00405a042`；本地 `RADA880106` 条目 `D` 字段为
"Accessible surface area (Radzicka-Wolfenden, 1988)"，`J` 字段注 "(Pro missing)"〕

**[19] Rose et al.（若正文要保留「Rose 埋藏表面积」这个说法，须改用此标度重算）**
Rose GD, Geselowitz AR, Lesser GJ, Lee RH, Zehfus MH. Hydrophobicity of amino acid
residues in globular proteins. *Science* 1985;229(4716):834–838.
doi:10.1126/science.4023714. PMID 4023714.
〔Crossref `10.1126/science.4023714`；本地 `aaindex1.txt` 中对应 accession 为
`ROSG850101`（Mean area buried on transfer）与 `ROSG850102`（Mean fractional area loss），
两者引同一篇；**本仓库代码未加载其中任何一个**〕

**[20] β 折叠倾向（本仓库 accession `CHOP780202`，由 `analysis/subspace_lib.py` 确定）**
Chou PY, Fasman GD. Prediction of the secondary structure of proteins from their
amino acid sequence. *Adv Enzymol Relat Areas Mol Biol* 1978;47:45–148.
doi:10.1002/9780470122921.ch2. PMID 364941.
〔Crossref `10.1002/9780470122921.ch2`；NCBI esummary 364941；
本地 `CHOP780202` 条目 `D` = "Normalized frequency of beta-sheet (Chou-Fasman, 1978b)"〕
（同篇亦为 α 螺旋倾向 `CHOP780201` 的来源，本仓库亦加载。）
**注意：这是统计频率标度，不是实验热力学 ΔΔG 倾向标度。** 见 §5.8。

**[21] 水合自由能（本仓库 accession `WOLR810101`）**
Wolfenden R, Andersson L, Cullis PM, Southgate CCB. Affinities of amino acid side
chains for solvent water. *Biochemistry* 1981;20(4):849–855.
doi:10.1021/bi00507a030. PMID 7213619.〔Crossref；NCBI esummary 7213619〕

**[22] 等电点（本仓库 accession `ZIMJ680104`）**
Zimmerman JM, Eliezer N, Simha R. The characterization of amino acid sequences in
proteins by statistical methods. *J Theor Biol* 1968;21(2):170–201.
doi:10.1016/0022-5193(68)90069-6. PMID 5700434.〔Crossref；NCBI esummary 5700434〕

**[23] Eisenlohr 原子折射度（加和性双路验证的那一路，docs/07 §144）**
Eisenlohr F. Eine Neuberechnung der Atomrefraktionen. I.
*Z Phys Chem* 1911;75U(1):585–607. doi:10.1515/zpch-1911-7539.
第二部分：Eisenlohr F. Eine Neuberechnung der Atomrefraktionen II.
*Z Phys Chem* 1912;79U:129–146. doi:10.1515/zpch-1912-7908.
〔Crossref `10.1515/zpch-1911-7539`（De Gruyter 数字化版的登记记录）〕

- `assessment:` **年份是 1911，不是 1910**（confidence: 中高）。
  basis: Crossref/De Gruyter 登记为 1911-11-01；而 NIST WebBook 的 Eisenlohr 作者条目
  **同时**列出 1911 与 1910 两个年份、卷页相同（75, 585），属二次源内部矛盾。
  未能取回 De Gruyter 页面本身（301→degruyterbrill.com 后返回 405），也未见纸本。
  "75U" 是该刊现代卷号写法，当时的引用形式是卷 75。

**[24] Lorentz–Lorenz 关系（折射度→极化率换算的依据）**
Lorenz L. Ueber die Refractionsconstante. *Ann Phys* 1880;247(9):70–103.
doi:10.1002/andp.18802470905.
Lorentz HA. Ueber die Beziehung zwischen der Fortpflanzungsgeschwindigkeit des
Lichtes und der Körperdichte. *Ann Phys* 1880;245(4):641–665. doi:10.1002/andp.18802450406.
〔Crossref 两条各自取回（Wiley/*Annalen der Physik* 登记记录）〕

- `assessment:` 两条都是出版方对旧刊的登记记录，不是纸本核对（confidence: 中高）。
  Lorenz 另有 1869/1875 年丹麦科学院论文常被引为优先权来源，**本次未核验**，不要引。
  卷号 247/245 是现代连续编号，与当时的「第三辑第 11 卷」等写法不同。

### 1.4 方法学参考

**[25] 双可加（AMMI）模型 —— 草稿的分解就是这个模型**
Gollob HF. A statistical model which combines features of factor analytic and analysis
of variance techniques. *Psychometrika* 1968;33(1):73–115. doi:10.1007/BF02289676.
〔Crossref `10.1007/BF02289676`〕
Gauch HG. Model selection and validation for yield trials with interaction.
*Biometrics* 1988;44(3):705–715. doi:10.2307/2531585.〔Crossref `10.2307/2531585`〕

**[26] AMMI 乘性项个数（= 交互项的秩）的已有检验程序 —— 见 §5.9，这是本次最有方法学意义的发现**
Forkman J, Piepho H-P. Parametric bootstrap methods for testing multiplicative terms
in GGE and AMMI models. *Biometrics* 2014;70(3):639–647. doi:10.1111/biom.12162.
Malik WA, Hadasch S, Forkman J, Piepho H-P. Nonparametric resampling methods for
testing multiplicative terms in AMMI and GGE models for multienvironment trials.
*Crop Science* 2018;58(2):752–761. doi:10.2135/cropsci2017.10.0615.
Malik WA, Forkman J, Piepho H-P. Testing multiplicative terms in AMMI and GGE models
for multienvironment trials with replicates. *Theor Appl Genet* 2019;132(7):2087–2096.
doi:10.1007/s00122-019-03339-8.
〔Crossref 三条各自取回〕

**[27] 子空间主角余弦（草稿 §3.1 / §9.2 的全部比较口径）**
Björck Å, Golub GH. Numerical methods for computing angles between linear subspaces.
*Mathematics of Computation* 1973;27(123):579–594. doi:10.1090/S0025-5718-1973-0348991-3.
〔Crossref `10.1090/S0025-5718-1973-0348991-3`〕

**[28] Marchenko–Pastur 律（秩功效推理的零分布依据）**
Marchenko VA, Pastur LA. Distribution of eigenvalues for some sets of random matrices.
*Mat Sb (N.S.)* 1967;72(114)(4):507–536.
英译：*Mathematics of the USSR-Sbornik* 1967;1(4):457–483. doi:10.1070/SM1967v001n04ABEH001994.
〔Math-Net.Ru 条目页 `m.mathnet.ru/eng/sm4101`（俄文原刊卷期页与英译信息）；
Crossref `10.1070/SM1967v001n04ABEH001994`（英译）〕

**[29] BBP 相变（尖峰模型里信号可检出的阈值）**
Baik J, Ben Arous G, Péché S. Phase transition of the largest eigenvalue for nonnull
complex sample covariance matrices. *Ann Probab* 2005;33(5):1643–1697.
doi:10.1214/009117905000000233.〔Crossref `10.1214/009117905000000233`〕

**[30] 力场电荷（`cache/ff/protein.ff14SB.xml`）**
Maier JA, Martinez C, Kasavajhala K, Wickstrom L, Hauser KE, Simmerling C.
ff14SB: Improving the Accuracy of Protein Side Chain and Backbone Parameters from ff99SB.
*J Chem Theory Comput* 2015;11(8):3696–3713. doi:10.1021/acs.jctc.5b00255.
〔Crossref `10.1021/acs.jctc.5b00255`〕
**与 `docs/17` §88–91 已记录的条目逐字一致**，该文件取自 OpenMM 发行版
`wrappers/python/openmm/app/data/amber14/protein.ff14SB.xml`。

### 1.5 结构条目（PDB）

PDB 条目本身可引其条目 DOI；下列 primary citation 字段取自 RCSB Data API。

| 条目 | 条目 DOI | primary citation（RCSB 报告） | 文献 DOI |
|---|---|---|---|
| **6M0J** SARS-CoV-2 RBD + 人 ACE2（§5.2 代理量的那个野生型结构） | 10.2210/pdb6m0j/pdb | Lan J, Ge J, Yu J, Shan S, Zhou H, Fan S, Zhang Q, Shi X, Wang Q, Zhang L, Wang X. *Nature* 2020;581(7807):215–220 | 10.1038/s41586-020-2180-5〔Crossref〕 |
| **1CHO** α-胰凝乳蛋白酶 + OMTKY3 | 10.2210/pdb1cho/pdb | Fujinaga M, Sielecki AR, Read RJ, Ardelt W, Laskowski M Jr, James MN. *J Mol Biol* 1987;195(2):397–418 | 10.1016/0022-2836(87)90659-0〔NCBI esummary 3477645〕 |
| **1PPF** 人白细胞弹性蛋白酶 + OMTKY3 | 10.2210/pdb1ppf/pdb | Bode W, Wei AZ, Huber R, Meyer E, Travis J, Neumann S. *EMBO J* 1986;5(10):2453–2458 | 10.1002/j.1460-2075.1986.tb04521.x〔NCBI esummary 3640709〕 |
| **1R0R** subtilisin Carlsberg + OMTKY3 | 10.2210/pdb1r0r/pdb | Horn JR, Ramaswamy S, Murphy KP. *J Mol Biol* 2003;331(2):497–508 | 10.1016/S0022-2836(03)00783-6〔Crossref〕 |
| **3SGB** SGPB + OMTKY3（**已按 D-046 剔出证据集**） | 10.2210/pdb3sgb/pdb | Read RJ, Fujinaga M, Sielecki AR, James MN. *Biochemistry* 1983;22:4420 | 未解析 —— 见 §4 |
| **1FLR** 抗荧光素 4-4-20 Fab | 10.2210/pdb1flr/pdb | Whitlow M, Howard AJ, Wood JF, Voss EW Jr, Hardman KD. *Protein Eng* 1995;8(8):749–761 | 10.1093/protein/8.8.749〔Crossref〕 |

〔条目 DOI 与 primary citation 字段：RCSB Data API `data.rcsb.org/rest/v1/core/entry/<ID>`；
条目 DOI 另经 Crossref 书目检索独立出现〕

### 1.6 相关工作（不是本文的数据源，但审计中必须引，见 §6）

**[31]** Kinjo AR, Nakamura H. Nature of protein family signatures: insights from singular
value analysis of position-specific scoring matrices. *PLoS ONE* 2008;3(4):e1963.
doi:10.1371/journal.pone.0001963. arXiv:0711.1010.〔Crossref；arXiv 摘要页〕

**[32]** Atchley WR, Zhao J, Fernandes AD, Drüke T. Solving the protein sequence metric
problem. *PNAS* 2005;102(18):6395–6400. doi:10.1073/pnas.0408677102.〔Crossref〕

**[33]** Alpay BA, Nanda P, Nagy E, Desai MM. Amino acid exchangeability and surface
accessibility underpin the effects of single substitutions. bioRxiv 2025-06-19.
doi:10.1101/2025.06.13.659595.〔Crossref；Europe PMC PPR1039085；**全文 PDF 已取回并检索**〕

**[34]** Dunham AS, Beltrao P. Exploring amino acid functions in a deep mutational
landscape. *Mol Syst Biol* 2021;17(7):e10305. doi:10.15252/msb.202110305. PMID 34292650.
预印本 bioRxiv doi:10.1101/2020.05.26.116756.〔Crossref；Europe PMC（含 "Preprint of" 关联）〕

**[35]** Faure AJ, Lehner B. MoCHI: neural networks to fit interpretable models and
quantify energies, energetic couplings, epistasis, and allostery from deep mutational
scanning data. *Genome Biology* 2024;25(1):303. doi:10.1186/s13059-024-03444-y.〔Crossref〕

**[36]** Tareen A, Kooshkbaghi M, Posfai A, Ireland WT, McCandlish DM, Kinney JB.
MAVE-NN: learning genotype-phenotype maps from multiplex assays of variant effect.
*Genome Biology* 2022;23(1):98. doi:10.1186/s13059-022-02661-7.〔Crossref〕

**[37]** MaveDB（**注意：本仓库任何 docs / analysis / JOURNAL 文件都未提及 MaveDB**，
见 §5.10。若正文决定引，则为：）
Esposito D, Weile J, Shendure J, Starita LM, Papenfuss AT, Roth FP, Fowler DM, Rubin AF.
MaveDB: an open-source platform to distribute and interpret data from multiplexed
assays of variant effect. *Genome Biology* 2019;20(1):223. doi:10.1186/s13059-019-1845-6.
更新篇：Rubin AF, Stone J, Bianchi AH, … Fowler DM. MaveDB 2024: a curated community
database with over seven million variant effects from multiplexed functional assays.
*Genome Biology* 2025;26(1):13. doi:10.1186/s13059-025-03476-y.
（**题名写 2024 但刊年为 2025**，引用时勿自行「修正」年份。）〔Crossref 两条〕

### 1.7 病毒受体结合域 DMS 系列与 CR9114

本小节的 10 个缓存文件**全部**经加密学方式锚定到具体 repo 路径：
8 个走 Git LFS 指针（`oid sha256` + `size` 对比本地 `shasum -a 256` 与字节数），
2 个走 git blob SHA-1（`sha1("blob <len>\0" + content)` 对比 GitHub tree blob SHA）。
**文件 → repo 的映射是确定的；不确定性全部在 repo → 论文这一步。**

**[38] starr2020（`starr2020_rbd_single_mut_effects.csv`、`starr2020_rbd_binding_Kds_perbarcode.csv`）**
Starr TN, Greaney AJ, Hilton SK, Ellis D, Crawford KHD, Dingens AS, Navarro MJ, Bowen JE,
Tortorici MA, Walls AC, King NP, Veesler D, Bloom JD. Deep Mutational Scanning of
SARS-CoV-2 Receptor Binding Domain Reveals Constraints on Folding and ACE2 Binding.
*Cell* 2020;182(5):1295–1310.e20. doi:10.1016/j.cell.2020.08.012.
〔repo `jbloomlab/SARS-CoV-2-RBD_DMS` 的 README 直接给出该 DOI；Crossref 内容协商〕
置信：**高**。**这是 8 成员证据集里唯一一个有确认出版物的病毒侧成员。**

**[39] starr2022（`starr2022_variants_final_variant_scores.csv`，仅用于 §7.5 筛查）**
Starr TN, Greaney AJ, Hannon WW, Loes AN, Hauser K, Dillen JR, Ferri E, Farrell AG,
Dadonaite B, McCallum M, Matreyek KA, Corti D, Veesler D, Snell G, Bloom JD.
Shifting mutational constraints in the SARS-CoV-2 receptor-binding domain during viral
evolution. *Science* 2022;377(6604):420–424. doi:10.1126/science.abo7896.
〔repo README 给出该 DOI；Crossref〕置信：高。

**[40] starr_omicron（`starr_omicron_final_variant_scores.csv`，筛查）**
Starr TN, Greaney AJ, Stewart CM, Walls AC, Hannon WW, Veesler D, Bloom JD.
Deep mutational scans for ACE2 binding, RBD expression, and antibody escape in the
SARS-CoV-2 Omicron BA.1 and BA.2 receptor-binding domains.
*PLOS Pathogens* 2022;18(11):e1010951. doi:10.1371/journal.ppat.1010951. PMID 36399443.
〔repo README 只链预印本 doi:10.1101/2022.09.20.508745；经题名与作者列表逐一比对确认正式版〕置信：高。

**[41] starr_EG5BA286（筛查）**
Taylor AL, Starr TN. Deep mutational scanning of SARS-CoV-2 Omicron BA.2.86 and
epistatic emergence of the KP.3 variant. *Virus Evolution* 2024;10(1):veae067.
doi:10.1093/ve/veae067. PMID 39310091.〔repo README 给出链接；Crossref〕置信：高。

**[42] starr_KP3LP8（筛查）**
Taylor AL, Starr TN. Deep mutational scanning of recent SARS-CoV-2 variants highlights
changing amino acid preferences within epistatic hotspot residues.
*PLOS Pathogens* 2026;22(6):e1014074. doi:10.1371/journal.ppat.1014074.
〔repo README 未给出版物；反向检索 Europe PMC 全文 PMC13309016，其正文直接引用该 repo URL〕置信：高。

**[43] taylor2023_xbb_bq（筛查）**
Taylor AL, Starr TN. Deep mutational scans of XBB.1.5 and BQ.1.1 reveal ongoing
epistatic drift during SARS-CoV-2 evolution.
*PLOS Pathogens* 2023;19(12):e1011901. doi:10.1371/journal.ppat.1011901. PMID 38157379.
〔repo README 给出链接；Crossref〕置信：高。

**[44] sarbecovirus（`starr_SARSr_RBD_final_variant_scores_lib40_41.csv`，§3.2 证据集成员之一）**
Lee J, Zepeda SK, Park Y-J, Taylor AL, Quispe J, Stewart C, Leaf EM, Treichel C, Corti D,
King NP, Starr TN, Veesler D. Broad receptor tropism and immunogenicity of a clade 3
sarbecovirus. *Cell Host & Microbe* 2023;31(12):1961–1973.e11.
doi:10.1016/j.chom.2023.10.018. PMID 37989312.
〔该文 Data Availability 指向 `tstarrlab/SARSr-CoV-RBD_DMS` 的 `results/summary/summary.md`，
即本文件所在 repo；Crossref + PMC10913562〕
置信：**中，且明确不完整 —— 见 §5.11。** 该文 Methods 描述的 DMS 只覆盖
PRD-0038 与 SARS-CoV-1 Urbani 对 human / R. alcyone / R. landeri 三个 ACE2，
而本文件是 6 骨架 × 7 ACE2。**证据集实际使用的那一格
（SARS-CoV-1_Urbani × Ra9479ACE2）不在该文描述的子集内。**

**[45] CR9114（`cache/dms/cr9114/*`，草稿 §7.5 的「唯一合格的外部候选」）**
Dailey KE, Wang Y, Teo QW, Wang C, Lv H, Lei R, Tong M, Rodriguez LA, Wang L, Wu NC.
Epistasis and pleiotropy constrain the evolution of cross-reactive breadth in a broadly
neutralizing influenza antibody. *PLOS Biology* 2026;24(9):e3004021.
doi:10.1371/journal.pbio.3004021.
预印本：Dailey KE, Wang Y, Teo QW, Wang C, Lei R, Tong M, Rodriguez LA, Wang L, Wu NC.
**Vulnerability in the breadth evolution of an influenza broadly neutralizing antibody.**
bioRxiv 2025-11-04. doi:10.1101/2025.11.03.686356.
代码/数据 github.com/nicwulab/CR9114_DMS @ b290a3fcf6a2d24c388e535977d87adbd34db0cc。
〔Crossref 两条 DOI 各自取回（本会话两次独立核验一致）；
README 经 raw.githubusercontent 在 pinned commit 与 main 两处各取一次〕

- `facts:` **DOI、刊名、首作者「Dailey et al.」全部正确**；正式版 10 位作者，卷 24(9)，文章号 e3004021。
- `facts:` **预印本题名与正式版不同**，且作者数由 9 增至 10（Huibin Lv 为正式版新增）。
- `facts:` **`cr9114_PROVENANCE.json` 里的 PMC12637573 归属错了** —— 它是**预印本**的 PMCID
  （对应 doi:10.1101/2025.11.03.686356），不是 PLOS Biology 文章的。见 §5.12。
- `facts:` **README 里仍然没有 Zenodo DOI**，pinned commit 与 main 两处 README 字节相同，
  且与缓存的 `cr9114_README.md`（sha256 7553…0cec82、3186 字节）一致；全文不含 "zenodo"
  字符串，也不含任何 DOI 或论文引用。**仓库记录的那个 gap 依然成立。**
- `facts:` 但 **10.5281/zenodo.22697938 确实解析**（→ zenodo 记录 22697939，
  题名 "nicwulab/CR9114_DMS: For publication"，软件类型，发布日期 2026-09-10，
  创建者 Nicholas Wu）。22697938 是 **concept DOI**，22697939 是版本 DOI。
  发布日期与 PROVENANCE.json 记的 commit 日期 2026-09-10T06:33:05Z 吻合。
- `assessment:` 该 Zenodo DOI **值是真的，但来源仍不可追**（confidence: 高）。
  按 CLAUDE.md §3，它现在应该继续记为 gap，直到被追到一个 primary source
  （最可能是 PLOS Biology 文章自身的 Data Availability 声明，本次未取回）。
  **不要因为「它能解析」就把 gap 关掉** —— 能解析证明的是 DOI 存在，
  不是「仓库当初是从哪里读到它的」。

---

## 2. 缓存数据文件 → 引用 的映射表

映射强度分三级：
**字节/哈希级** = 本地文件与上游对象的 sha256 / git blob SHA / Content-Length 逐位一致；
**README 级** = 文件锚定到 repo，repo 的 README 或论文正文明确指向该出版物；
**内容级** = 只靠内容特征推断，仓库未记录来源 URL 或校验和。

| 缓存文件 | 上游对象 | 映射强度 | 引用 |
|---|---|---|---|
| `cache/dms/tsuboyama2023/*` | Zenodo 7992926（版本 DOI） | 清单含 md5+sha256；本次核验文件名与层级 | [4][5] |
| `cache/dms/lehner_doubledeepms_Data.zip`、`cache/ddpca_extract/`、`cache/dms/lehner_ddPCA/*` | doubledeepms README 指向的 Dropbox `Data.zip`（**非 Zenodo**） | **内容级**（仓库未记 URL/校验和） | [6] |
| `cache/dms/skempi_v2.csv`、`cache/skempi_thr4.csv` | SKEMPI 2.0 | README 级 | [7] |
| `cache/platinum/*` | Platinum 数据库导出 | README 级 | [8] |
| OMTKY3 面板（经 `cache/skempi_thr4.csv`，1CHO/1PPF/1R0R/3SGB 四个单酶矩阵） | Laskowski 组加和性数据集，经 SKEMPI 转录 | README 级（两跳：SKEMPI→原文） | [9]（必要时 +[10]） |
| `cache/dms/titeseq2016_*`（含 `titeseq2016_4-4-20.pdb`） | Tite-seq 原文及其数据 | README 级 | [11]；结构 PDB **1FLR** |
| `cache/dms/htmek_PafA_{VO4,WO4}_inhibition.csv` | HT-MEK PafA 钒酸盐/钨酸盐抑制数据 | **内容级**（仓库未记 URL/校验和） | [12] |
| `cache/dms/magma_MOESM{3,4,5,6}.xlsx` | Nat Commun 补充材料 MOESM3–6 | **字节级（4/4 全中）** | [13] |
| `cache/aaindex1.txt` | AAindex `aaindex1` 平文件 | README 级 | [14]；逐标度见 [15]–[22] |
| `cache/ff/protein.ff14SB.xml` | OpenMM 发行版 amber14 力场文件 | README 级（docs/17 已记路径） | [30] |
| `cache/pdb/*.pdb` | RCSB | 条目 ID 级 | §1.5 表 |
| `cache/dms/starr2020_rbd_single_mut_effects.csv` | `jbloomlab/SARS-CoV-2-RBD_DMS` `results/single_mut_effects/single_mut_effects.csv` | **LFS sha256 级** | [38] |
| `cache/dms/starr2020_rbd_binding_Kds_perbarcode.csv` | 同 repo `results/binding_Kds/binding_Kds.csv` | **LFS sha256 级** | [38] |
| `cache/dms/starr2022_variants_final_variant_scores.csv` | `jbloomlab/SARS-CoV-2-RBD_DMS_variants` | **LFS sha256 级** | [39] |
| `cache/dms/starr_omicron_final_variant_scores.csv` | `jbloomlab/SARS-CoV-2-RBD_DMS_Omicron` | **LFS sha256 级** | [40] |
| `cache/dms/starr_EG5BA286_final_variant_scores.csv` | `tstarrlab/SARS-CoV-2-RBD_DMS_Omicron-EG5-FLip-BA286` | **LFS sha256 级** | [41] |
| `cache/dms/starr_KP3LP8_final_variant_scores.csv` | `tstarrlab/SARS-CoV-2-RBD_DMS_Omicron-KP3-LP8` | **LFS sha256 级** | [42] |
| `cache/dms/taylor2023_xbb_bq_final_variant_scores.csv` | `tstarrlab/SARS-CoV-2-RBD_DMS_Omicron-XBB-BQ` | **LFS sha256 级** | [43] |
| `cache/dms/starr_SARSr_RBD_final_variant_scores_lib40_41.csv` | `tstarrlab/SARSr-CoV-RBD_DMS` `results/final_variant_scores/final_variant_scores_lib40_41.csv`（git blob `26f255c2…`，3446048 B） | **blob SHA 级**（repo）；**论文只覆盖子集** | [44]，并见 §5.11 |
| `cache/dms/starr_animalACE2_final_variant_scores.csv` | `tstarrlab/SARS-CoV-2-RBD_DMS_animal-ACE2`（git blob `22849f82…`，3313547 B） | **blob SHA 级**（repo）；**无出版物** | **无 —— 见 §4.1** |
| `cache/dms/starr_PDCoV_RBD_final_variant_scores.csv` | `tstarrlab/PD-CoV-RBD_DMS`（git blob `16b057fe…`，385058 B） | **blob SHA 级**（repo）；**无出版物** | **无 —— 见 §4.1** |
| `cache/dms/cr9114/*`（17 个文件） | `nicwulab/CR9114_DMS` @ b290a3f | PROVENANCE.json 记 sha256 + 字节，字节数经 GitHub git-tree 核对 | [45] |

### 2.1 8 成员证据集逐成员的引用状态（**这是本文档最重要的一张表**）

证据集定义见 `docs/24` §3.1 的 9 成员表减去 3SGB（人类裁定 D-046）。

| 成员 | 数据文件 | 引用 | 状态 |
|---|---|---|---|
| `OMTKY3/chymotrypsin/1CHO` | `cache/skempi_thr4.csv` | [9]（+[7] 容器） | ✅ 可引 |
| `OMTKY3/HLE-elastase/1PPF` | 同上 | [9]（+[7]） | ✅ 可引 |
| `OMTKY3/subtilisin-Carlsberg/1R0R` | 同上 | [9]（+[7]） | ✅ 可引 |
| `starr2020/WH1/huACE2` | `starr2020_rbd_single_mut_effects.csv` | [38] | ✅ 可引 |
| `animalACE2/Wuhan-Hu-1/bat` | `starr_animalACE2_*.csv` | **无** | ❌ **无出版物** |
| `animalACE2/Omicron_BA2/human_monomer` | 同上 | **无** | ❌ **无出版物** |
| `PDCoV/PDCoV/gAPN` | `starr_PDCoV_RBD_*.csv` | **无** | ❌ **无出版物** |
| `sarbecovirus/SARS-CoV-1_Urban/Ra9479ACE2` | `starr_SARSr_*_lib40_41.csv` | [44] | ⚠️ **该具体格子不在 [44] 描述的子集内** |

**8 个成员里，4 个（全部病毒侧，除 starr2020 外）没有可引的出版物来源。**
这不是书目整理问题，是草稿 §2.1「7 个骨架、9 个配体、约 8 万观测」这句话的
**证据基础问题**。处理建议见 §5.11，这是一个人类决策点。

---

## 3. 本次顺带核验、草稿未直接引但支撑文档已引的条目

| 出处 | 条目 | 核验 |
|---|---|---|
| `docs/05` §71 | Ross GA, Lu C, Scarabelli G, Albanese SK, Houang E, Abel R, Harder ED, Wang L. The maximal and current accuracy of rigorous protein-ligand binding free energy calculations. *Commun Chem* 2023;6(1):222. doi:10.1038/s42004-023-01019-9 | ✅ Crossref，与 docs/05 记的卷/文章号一致 |
| `docs/07` §214 | Philo JS. SEDNTERP: a calculation and database utility to aid interpretation of analytical ultracentrifugation and light scattering data. *Eur Biophys J* 2023;52(4-5):233–266. doi:10.1007/s00249-023-01629-0 | ✅ Crossref；docs/07 记的「52, 233 (2023)」是起始页，完整页码为 233–266 |
| `docs/17` §132 | Battaglia, Buckingham & Williams, *Chem Phys Lett* **78**(1981) 421（苯 Θ_zz 标定值） | ⚠️ **未解析** —— 见 §4.3 |

---

## 4. UNVERIFIED —— 需要人工核对

### 4.1 两个数据集找不到任何出版物（**最高优先级**）

| 数据 | 确认到的 | 没确认到的 | 试过什么 |
|---|---|---|---|
| `starr_animalACE2_final_variant_scores.csv` | repo 为 `tstarrlab/SARS-CoV-2-RBD_DMS_animal-ACE2`，本地文件与其 git blob `22849f82a8bb2a5e98778069bf77ea3a545f47fe`（3313547 B）**逐位一致**；README 写「Analysis by Tyler Starr, Ashley Taylor, Giulia Gallo, Joseph Newman and co-authors」 | **任何论文、预印本或 DOI**。README 内 grep `doi/manuscript/paper/preprint/biorxiv/publish/cite` **零命中**；无 CITATION 文件、无 homepage | Europe PMC `FULL_TEXT:"SARS-CoV-2-RBD_DMS_animal-ACE2"`（0）；`AUTH:"Starr TN" AND AUTH:"Newman J"`（0）；`AUTH:"Starr TN" AND AUTH:"Gallo G"`（0）；`(AUTH:"Gallo, Giulia" OR AUTH:"Gallo G") AND ACE2 AND "deep mutational"`（0）；`"deep mutational scanning" AND "white-tailed deer" AND ACE2 AND "receptor-binding domain"`（12 命中，无一相符）；`AUTH:"Taylor AL" AND AUTH:"Starr TN"`（14 命中，全部已由本组其他文件解释）；Starr 实验室出版列表 bioscience.utah.edu/faculty/starr/；repo 名的 WebSearch |
| `starr_PDCoV_RBD_final_variant_scores.csv` | repo 为 `tstarrlab/PD-CoV-RBD_DMS`，本地文件与其 git blob `16b057fe7473c9133dad3c613251fcdfb68d239d`（385058 B）**逐位一致**；README 写「Study and analysis by Tyler Starr and Ashley Taylor, in collaboration with the Veesler lab」，3 个配体 gAPN/pAPN/hAPN 与表头 `bind_gAPN/bind_hAPN/bind_pAPN` 对上 | **任何论文、预印本或 DOI** | Europe PMC `FULL_TEXT:"PD-CoV-RBD_DMS"`（0）；`AUTH:"Starr TN" AND (aminopeptidase OR deltacoronavirus)`（5 命中，逐条看过）；`"PD-CoV RBD" AND "deep mutational scanning"`（1 命中）；Starr 实验室出版列表。**两个近似项已检查并排除**：Rexhepaj et al. *Immunity* 2024 doi:10.1016/j.immuni.2024.10.001 做的是 PDCoV RBD 的**抗体逃逸**而非 APN 受体结合，且该 repo 的 `results/` 树下只有 `{g,h,p}APN_bc_binding.csv` 与 `bc_expression.csv`、**没有任何抗体逃逸输出**；*PNAS* 2026 doi:10.1073/pnas.2533456123 引的是 `PD-CoV_MAP_minibinder` / `PD-CoV_breadth_minibinder` 两个**不同的** repo |

两个 repo 的最后推送分别为 2024-03 与 2025-07，与「尚未发表」相符。
**按 CLAUDE.md §3，这两个的引用字段应写 `null` 并进 `gaps`，不得猜一个。**

### 4.2 Fauchère–Pliska 的 DOI/PMID

确认到：AAindex `FAUJ830101` 的 `A`/`T`/`J` 字段给出作者、题名、
*Eur J Med Chem* 18, 369–375 (1983)。
**没确认到**：DOI 与 PMID。该条目的 `R`（PMID）字段在平文件里**本身就是空的**
（已直接看过 `cache/aaindex1.txt` 的原始字节，不是解析产物）；未找到出版方记录。
卷/页只有 AAindex 一个来源，**没有第二源佐证**。
→ **不要让任何人给这条补一个 DOI。**

### 4.3 其他

| 条目 | 确认到的 | 没确认到的 | 试过什么 |
|---|---|---|---|
| **3SGB 的文献 DOI** | RCSB 报告 primary citation = Read RJ, Fujinaga M, Sielecki AR, James MN. *Biochemistry* 1983;22:4420；条目 DOI 10.2210/pdb3sgb/pdb 有效 | 该 1983 年 Biochemistry 文的 DOI 与 PMID | RCSB Data API；Crossref 书目检索（只返回 PDB 条目本身）；猜的两个 ACS DOI 均解析到**无关论文**，已丢弃。（3SGB 已被剔出证据集，优先级低） |
| **Battaglia/Buckingham/Williams 1981**（`docs/17` 的苯四极标定值） | `docs/17` 记的刊名/卷/页/数值 | DOI | 猜的 `10.1016/0009-2614(81)85237-7` 解析失败（404）。未进一步检索 —— 该值只用于 docs/17 的标定，不进 docs/35 |
| **Platinum 的更新版论文** | 2015 NAR 篇（[8]） | **是否存在更新版** | 两次检索均只返回 2015 篇与引用它的第三方工具；数据库自身页面无更新版提及。**结论是「未找到」，不是「不存在」** |
| **MCMT640101 的 "Jones (1975)" 转引源** | AAindex `D` 字段明写转引自 Jones (1975) | Jones 1975 是哪一篇 | AAindex 该条目未给该参考；未另行检索 |
| **Lorenz 1869/1875 丹麦科学院论文** | — | 全部 | 未检索。常被引为 Lorentz–Lorenz 的优先权来源，**不要在未核验的情况下引** |
| **Eisenlohr 的年份** | Crossref/De Gruyter 登记为 **1911** | 纸本；De Gruyter 页面本身（301→degruyterbrill.com 后 405） | NIST WebBook 同一条目同时列 1911 与 1910（卷页相同），属二次源内部矛盾，见 [23] |
| **MoCHI / MAVE-NN / Otwinowski-McCandlish-Plotkin / Sailer & Harms 的 Methods 全文** | 书目字段（[35][36]）；MoCHI 的潜变量个数由用户的机理模型**指定** | **是否存在一个从数据里选择潜变量个数的步骤**。这是本次先验技术审计覆盖最薄的一处，见 §6.4 | 仅读了刊物落地页与摘要，**未读 Methods**；Sailer & Harms 完全未检索 |
| **Gollob / Cornelius 的经典 AMMI 近似 F 检验原文** | Gollob 1968（[25]）已核 | Cornelius 的 F_R 检验原文 | 仅在 [26] 的三篇中见其被引，原文未取 |
| **Rosace-AA 的取代分组个数** | 分组规则（见 §5.1） | 具体组数 | 该文未给出，无法推导 |

---

## 5. 草稿应改之处（按重要性）

**本文档不改 `docs/35-paper-draft.md`；以下是待裁定的修改清单。**

### 5.1 §1.2 对 Rosace-AA 的刻画：核心准确，但有三条必须补的限定

核心判断成立：它确实把交互假设成 `φ_p + ν_a·ρ_p` 的秩 1 外积，且全篇不讨论秩。
但以下三点不补，审稿人会补：

1. **ν 的索引是「取代分组」，不是 20 种残基。** 原文：
   "Substitutions of similar BLOSUM scores are grouped so that each group covers at
   least 20% of positions."（组数未给出）。
   严格说，Rosace-AA 假设秩 1 的对象是**位点 × 取代组**表，是位点×残基的一个粗化。
2. **ρ 被符号约束：** "Activation scores ρ_p(v) are assumed to range in [0, 1] …
   ρ_p(v) ~ Beta(1.5, 1.5)"。这不是自由的秩 1 项，是**非负缩放**的秩 1 项 ——
   它禁止「取代偏好相对共识反号」的位点。这比秩 1 更强，**对本文的论点是有利的**，
   值得明写而不是省略。
3. **表型不是结合 ΔΔG。** 它分解的是 β（生长型 DMS 打分）。

### 5.2 §1.2 对 Husain & Murugan 的刻画：索引对，「测过秩」不对

- 「索引是位点×位点（上位效应）而非位点×残基」**准确**，照写。
- **「确实测过秩」应改为「推导出秩」或「给出低秩形式」**。他们做的是围绕野生型结构的
  摄动展开（`ΔΔF_ij ∝ θ_i θ_j + c_ij`，秩 = 软模个数），用弹性网络模拟与
  计算双突变环验证，**没有从任何实测矩阵估过秩**。
- 附带一条值得加的细节：他们的适应度函数 F 是刻意一般化的，原文明说可以代表
  "the binding affinity for a ligand, or catalytic efficiency"。
  所以**不能说他们的工作与结合无关**，只能说索引不是本文要的那个。

### 5.3 §3.3 与 §3.5 的「Rose 埋藏表面积」用错了标度（**实质性更正**）

`analysis/subspace_lib.py` 第 590 行把 AAindex accession **`RADA880106`** 加载为
变量名 `asa_buried_Rose`。但 `cache/aaindex1.txt` 里 `RADA880106` 的 `D` 字段是
**"Accessible surface area (Radzicka-Wolfenden, 1988)"**，`A` 字段是
**"Radzicka, A. and Wolfenden, R."**，`J` 字段是 *Biochemistry* 27, 1664-1670 (1988)。

**这不是 Rose et al.** Rose 的标度在同一个文件里确实存在，accession 是
`ROSG850101`（Mean area buried on transfer）与 `ROSG850102`（Mean fractional area loss），
**本仓库代码一个都没加载。**

因此：
- 草稿 §3.3 表中「Rose 埋藏表面积 r = +0.656 / r² = 0.431」与 §3.4 前后文的
  「埋藏 ASA +0.656」，**这个数字属于 Radzicka–Wolfenden 的可及表面积标度，不属于 Rose**。
- `docs/13` §449、`docs/21` §564、`docs/26` §514、`docs/30` 等处沿用了同一个标签。
- 两个选择，二选一，**是人类决策点**：
  (a) 把名字改成 Radzicka–Wolfenden ASA，引 [18]，数字不动；
  (b) 真的换成 `ROSG850101` 重算，引 [19]，数字会变。
- `assessment:` 这是一个**纯标签错误而非数值错误**的可能性很高（confidence: 中高；
  basis: `scale_redundancy.py` 的 `parse_aaindex` 文档串明确处理了 `(Pro missing)`
  这类缺失标记与行主序问题，而 `RADA880106` 的 `J` 字段正带 "(Pro missing)"、
  Pro 位写作裸 `0.`，解析器会把它判为缺失而不是 0 —— 即 **D-023 那一类静默错误
  在这条上没有重演**）。但标签错会让读者去核对错的那篇文献，按 §3 必须更正。

### 5.4 §7.5 的「某个常用数据库」指的是 SKEMPI，不是 MaveDB，且措辞偏弱

草稿 §7.5：「某个常用数据库在两种全新判据下各自独立地重现『只有一个体系可用』」。
按 `docs/23` §0 与 §3.5，这个数据库是 **SKEMPI v2**（294 个「骨架×伙伴」分组按新判据、
239 个「按骨架合并」分组按第三种扫法，两次都只剩 OMTKY3）。

- 「常用」是**低估**：该库是蛋白–蛋白结合突变 ΔΔG 的事实标准基准集，
  规模是 7085 个突变 / 237 个互作体系 / 345 个 PDB 条目（[7]）。
  写成具名引用比「某个常用数据库」强得多，而且并不损失什么。
- 同段「本地穷尽筛查了 307 个矩阵」：`docs/23` §3.0 的表把 307 定为
  「进入数值筛查的矩阵」数，而 §0 并列的那串（SKEMPI 294 + 239、Platinum 263、
  PafA 2、MAGMA 11、Tite-seq 1）相加远超 307。两者**可能**分别指
  「扫过的分组」与「真能成矩阵的」，但正文只给 307 而不给口径，读者无法还原。
  `assessment:` 建议在 §7.5 写明 307 的口径（confidence: 中；basis: 两处数字的
  差额没有在 docs/23 里被显式对账 —— 这属于数字审计范围，不是文献范围，
  本文档只做标记，不下结论）。
- 另：**MaveDB 在本仓库的 docs / analysis / JOURNAL 里一次都没出现过**（grep 零命中）。
  见 §5.10。

### 5.5 `MCMT640101` 不应写成「McMeekin 1964」这一篇期刊论文

它是三作者（McMeekin, Groves & Hipp）的 ACS 书章，AAindex 没有记题名、没有 PMID，
且 `D` 字段明写数值**转引自 Jones (1975)**。建议写法见 [17]，并在正文把
「实验测得的摩尔折射度」改成「实验折射度的汇编值（经 AAindex，源头为一份 1964 年书章，
由一份 1975 年汇编转引）」。Gly=0.00 / Trp=42.53 的自检没有问题。

### 5.6 Tsuboyama 的 Zenodo DOI 要明确选版本还是 concept

`10.5281/zenodo.7992926` 是**版本 DOI**（v2_230420）；concept DOI 是
`10.5281/zenodo.7401274`。两者语义不同（「这一版」vs「所有版本」），
该选哪个取决于复现意图。建议：正文用版本 DOI（可复现性优先），
在数据可得性声明里同时给出 concept DOI。
顺带：`tsuboyama_PROVENANCE.txt` 的文件清单把两层压成一层，建议标注归属 zip（见 [5]）。

### 5.7 两处「内容级」映射应补 provenance 或降级措辞

`cache/dms/lehner_doubledeepms_Data.zip` 与 `cache/dms/htmek_PafA_*.csv`
**都没有记录下载 URL、字节数或校验和**，映射只靠内容特征。
按 CLAUDE.md §3，这两条要么补一个取数清单（像 Tsuboyama 与 CR9114 那样），
要么在正文把它们的来源措辞降级。另：ddPCA 的上游文件名是 Dropbox 上的 `Data.zip`，
**不是 Zenodo**，provenance 行不要写成 Zenodo。

### 5.8 「β 折叠倾向」需要写明 accession

本仓库用的是 `CHOP780202`（Chou–Fasman 统计频率，[20]）。
这一点很重要，因为「β 折叠倾向」至少有三类性质完全不同的标度：
统计频率（`CHOP780202`）、结构导出偏好（`KOEP990102`，Koehl & Levitt 1999
*PNAS* 96(22):12524–12529，doi:10.1073/pnas.96.22.12524）、
实验热力学 ΔΔG 倾向（`KIMC930101`，Kim & Berg 1993 *Nature* 362(6417):267–270，
doi:10.1038/362267a0）。草稿 §3.3 把它报为**最佳单条标度（r² = 0.607）**，
不写 accession 就不可复现。
（顺带：**AAindex 里没有 Street & Mayo 的 β 折叠倾向条目** —— 全文件 grep "mayo"
只有一处无关的书籍编者。）

### 5.9 §9.1 的定秩程序应与已有的 AMMI 检验程序对照（方法学，但会被审稿人问）

草稿的分解 `ΔΔG(s,r) ≈ a(r) + b(s) + 交互(s,r)` 加上「估计交互项的秩」
**就是 AMMI 模型的乘性项个数选择问题**（行=位点扮演基因型、列=残基扮演环境）。
这个问题有 50 年文献与已标定 Type I 错误率的检验（[25][26]）。
草稿目前的口径是「K 折交叉验证 + 裁剪 RMSE + 置换零分布」，
`docs/12` §4 的置换对照确实显示程序无偏。
`assessment:` 建议在 §9.1 显式提及 AMMI 命名与 [26] 这一类参数/非参数 bootstrap 检验，
并说明本文的 CV 口径与之的关系（confidence: 高，这是定位问题不是正确性问题；
basis: [25][26] 的存在）。这同时也**加强**本文的新意陈述 ——
「方法是成熟的，没被用在这个索引上」比「我们发明了一个定秩程序」更稳。

### 5.10 MaveDB 在仓库里不存在

任务说明把 MaveDB 列为「screening 中提到的」，但 `grep -ri mavedb docs/ analysis/ JOURNAL.md`
**零命中**。筛查用的两个数据库是 SKEMPI v2 与 Platinum。
若正文不打算引 MaveDB，**就不要引**；[37] 只是备着。

### 5.11 §2.1 / §7.1 的证据宽度：4 个病毒侧成员没有出版物来源（**人类决策点**）

见 §2.1 表。草稿 §2.1 写「7 个骨架、9 个配体、约 8 万观测」，§7.1 写
「本文的跨家族证据是一个非病毒抑制剂骨架加一个病毒受体结合域家族」。
核验结果是：病毒侧的 5 个证据集成员里，**只有 `starr2020/WH1/huACE2` 有出版物**；
`animalACE2` 的两个与 `PDCoV/gAPN` 来自**没有任何论文的 GitHub repo**，
`sarbecovirus` 那一格虽有 [44]，但 [44] 的 Methods 只描述了 PRD-0038 与
SARS-CoV-1 Urbani 对 human / R. alcyone / R. landeri 三个 ACE2，
**而证据集用的是 SARS-CoV-1_Urbani × Ra9479ACE2，不在该子集内**。

这需要人类裁定（CLAUDE.md §9 第 1 条：判别集纳入判据的修改）：
- 是接受「数据有 repo 级 provenance 但无出版物」并在正文显式写明，
- 还是把这几个成员降级，重算 §2.1 的全部统计量。
**不论选哪个，§7.1 的「两组质量不对等」那段都要加上这一层不对等**，
因为它和截尾、可分辨性是两件独立的事。

### 5.12 `cr9114_PROVENANCE.json` 的三处要改

1. **PMC12637573 归属错误**：它是**预印本**（doi:10.1101/2025.11.03.686356）的 PMCID，
   不是 PLOS Biology 文章的。PLOS 文章自身的 PMCID 本次未确定。
2. **预印本题名与正式版不同**，作者数 9 → 10（正式版新增 Huibin Lv）。
   `authors_first` 记的 "Dailey et al." 正确。
3. **Zenodo gap 保持开启**：值 `10.5281/zenodo.22697938` 确实解析（且确实是本 repo 的
   concept DOI，快照日期与 pinned commit 日期吻合），**但 README 在 pinned commit 与
   main 两处都没有它**。`status` 字段建议从 "NOT VERIFIED - absent from README"
   改成「值已独立解析为本 repo 的 concept DOI；但**出处仍不可追**，README 两处皆无，
   待追至 PLOS Biology 的 Data Availability 声明」。

### 5.13 一处文件名误导（不影响正文，但影响可读性）

`starr_EG5BA286_*` 与 `starr_KP3LP8_*` 实际都是 **Taylor & Starr**（Taylor 第一作者），
`starr_` 前缀会把人引向错的第一作者。`taylor2023_xbb_bq_*` 的前缀是对的。

### 5.14 `starr_animalACE2` 的背景标签：数据对，上游 README 错

上游 README 把三个背景写成「Wuhan-Hu-1, **Alpha**, Omicron BA.2」，
而文件的 `target` 列实际是 `Wuhan-Hu-1` / **`Eta`** / `Omicron_BA2`。
**本仓库记的「Wuhan-Hu-1 / Eta / BA.2」与数据一致，与上游 README 不一致。**
→ **不要按 README 去「订正」数据。** 这一条值得写进 JOURNAL，
因为它是「被检查对象的错」而不是「工具的错」（CLAUDE.md §10 末条）。

---

## 6. 先验技术（prior art）审计

### 6.1 结论

**没有找到任何已发表工作，从实测的结合 ΔΔG 矩阵上估计过位点×残基交互项的秩。**
草稿的新意主张**成立**，但措辞应收紧（§6.3）。

按接近程度排序的四个近似项：

| # | 工作 | 矩阵与索引 | 秩是估计的还是假设的 | 表型 | 威胁程度 |
|---|---|---|---|---|---|
| 1 | Rosace-AA [1] | 位点 × **取代组**（20 残基的粗化） | **假设**为 1，且 ρ ≥ 0（比秩 1 更强） | 生长型 DMS 打分；>500 domain 那一跑是**稳定性** | 这就是草稿自己的第 (i) 项，刻画准确 |
| 2 | Husain & Murugan [3] | **突变 × 突变**（ΔΔF_ij） | **理论推导**（= 软模个数），弹性网络模拟验证 | 一般化适应度，原文把结合亲和力列为例子之一 | 草稿自己的第 (ii) 项，索引对、动词错 |
| 3 | Kinjo & Nakamura [31] | **位点 × 20 氨基酸**（PSSM）—— **索引形状是对的** | 讨论了第一、第二奇异成分，但**没有给出显著性阈值或定秩程序**；无双向中心化迹象 | **比对导出的打分，不是实测能量** | **中** —— 索引最接近。其第一成分惩罚功能重要位点的取代、第二成分与疏水性相关，与「秩 2」接近得不舒服。**建议读全文并引用。** |
| 4 | Atchley et al. [32] | **残基 × ~500 条理化属性** | 报告 5 个因子 | 完全不是实测 ΔΔG | **低 —— 但这个区分正是关键。** 这是氨基酸**属性空间**的维数，与「实测结合数据里位点×残基交互需要几个轴」是两回事。要引，并显式区分，否则审稿人会混为一谈 |

### 6.2 一条曾经看起来最危险、查清后排除的线索

bioRxiv doi:10.1101/2025.06.13.659595（Alpay, Nanda, Nagy & Desai，[33]）在检索结果里
显示为「a statistical **rank** model」，形式上像是「残基级矩阵 × 逐位点标量」，
即一个在实测效应上的秩 1 位点×残基模型。**已取回全文 PDF 并检索，结论是排除：**

- 该文的 "rank" 是**序数秩**（名次），不是**矩阵的秩**。原文：
  "we model the relative **ranks** of the scores within each deep mutational scan
  rather than the scores themselves"，模型是 Plackett–Luce 序模型。
- 其参数化为 `θ_i = logistic(β_{w_i,m_i} + β_SA · s_i)`，
  即一个 **20×20 的「野生型残基 × 突变残基」可交换性矩阵**加上**一个**表面可及度系数。
  **没有位点×残基交互项**；位点只通过一个标量协变量进入。
  该文自己写明 "Our models analyze effects of only single substitutions and therefore
  cannot capture epistasis"。
- `assessment:` **不构成先验技术**（confidence: 高；basis: 全文 PDF 的模型式与自述）。
  **但值得引为相关工作**，而且方向对本文**略不利**：它表明一个
  「主效应 + 逐位点标量协变量、完全不含位点×残基交互」的模型，
  已经能解释实测 DMS 效应序中相当大的一部分方差。草稿 §1.1 说
  「主效应不含判别力」是对的，但「交互项有多大」这个量级问题，
  这篇是现成的外部参照点。

### 6.3 建议的新意措辞

把 §1.2 的全称否定收紧为可辩护的形式：

> 位点×残基交互项的秩此前**没有从实测的结合自由能变化上被估计过**；
> 已有工作要么把它固定为 1（Rosace-AA [1]，且其 ρ 被约束为非负，
> 索引是位点×取代组，表型是生长型打分或稳定性），
> 要么是在位点×位点的上位效应矩阵上**推导**出低秩形式
> （Husain & Murugan [3]，理论加模拟，非实测估计）。

理由：草稿现有的无条件全称否定（「据我们所知它此前没有被测过」+
「这个问题既没有被回答，也没有被当成一个需要回答的问题」）比需要的更冒险，
而本文的实际贡献在收紧后的措辞下**完整保留**。

### 6.4 本次审计覆盖不到的地方（必须写出来，否则这节会被当成比实际更强的背书）

1. **MoCHI [35] / MAVE-NN [36] / Otwinowski-McCandlish-Plotkin / Sailer & Harms
   的 Methods 全文没有读**，只读了落地页与摘要。
   这是最可能藏着「潜变量个数选择步骤」的地方。虽然潜变量是**性状**
   （折叠/结合自由能）而不是位点×残基交互，与本文的索引仍不同，
   但**定稿前应专门读一遍**。Sailer & Harms 完全未检索。
2. **Kinjo & Nakamura [31] 只读了摘要与 arXiv 摘要页**，没读 Methods。
   它的索引形状与本文相同，**是本节最该补读的一篇**。
3. Kidera 等人的氨基酸性质因子分析未定位、未核验。
4. 张量分解用于突变扫描：一次复合检索无命中，未专门追。
5. NMF 用于突变景观 + NMF 的定秩：只出现通用方法学，未找到也未排除蛋白特定应用。
6. 文献发现全部走通用网页检索加出版方/预印本页面；**PubMed/Europe PMC 的直接
   主题检索未被用于先验技术这一节**（用于了 §4.1 的数据集检索）。
   这偏向收录良好与较新的工作，**可能漏掉更早或影响力较低的论文**。
7. Husain & Murugan 的参考 [32]（结构维数「4–5 个 PCA 成分」的来源）未追。
8. 本节两篇最接近的近似项（[31][33]）都是在**审计过程中**才浮现的，
   这本身说明覆盖面还不够；§6.1 的结论应按「尽力检索未发现」而非
   「不存在」来读。

---

## 7. 汇总

| 类别 | 核验通过 | 未通过/需人工 |
|---|---|---|
| 新意定位（§1.1） | 3 | 0 |
| 数据来源（§1.2、§1.7） | 18 | **2**（animalACE2、PD-CoV **无任何出版物**） |
| 标度与参考值（§1.3） | 10 | **1**（Fauchère–Pliska 无 DOI/PMID） |
| 方法学（§1.4） | 8 | 0 |
| PDB 条目（§1.5） | 6 条目 + 5 篇文献 | **1**（3SGB 的文献 DOI） |
| 相关工作（§1.6） | 7 | 0 |
| **合计** | **45 条编号引用全部有可追来源** | **4 项列入 §4** |

**两条新意定位工作的刻画**：Rosace-AA **准确**（需补三条限定）；
Husain & Murugan **索引准确、动词不准确**（「测过秩」应改「推导出秩」）。

**先验技术**：未发现任何从实测结合 ΔΔG 估计位点×残基交互秩的工作；
最接近的 Kinjo & Nakamura 2008 索引形状相同但对象是比对导出的 PSSM。
建议把全称否定收紧为 §6.3 的措辞。

**草稿需要更正的引用**：
「Rose 埋藏表面积」实际是 Radzicka–Wolfenden（§5.3）；
「McMeekin 1964」是书章且经 1975 年汇编转引（§5.5）；
「某个常用数据库」是 SKEMPI 2.0 且应具名（§5.4）；
「Husain & Murugan 测过秩」应改为「推导出秩」（§5.2）。

**最需要人类裁定的一条**：8 成员证据集里 4 个病毒侧成员没有可引的出版物来源（§5.11）。
