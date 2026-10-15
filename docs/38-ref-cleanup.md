# 参考文献收尾与标度标签更正（2026-10-15）

两件事：**(1)** 结清 `docs/36-references.md` §4「UNVERIFIED」的 4 项，
并补齐 docs/36 留开的若干登录号 / DOI / PMCID；
**(2)** 把 D-071 的 `RADA880106` 标签更正传播到全仓库。

遵 CLAUDE.md §3（provenance 强制）、§4（不改已落盘的论证，补充走 `更正` 条目）、
§5（点估计带 bootstrap 区间，比区间重叠而非均值）、§10（工具先自验）。

**未访问 `labels/`，未开封 holdout，未修改 `docs/16-why-two.md`。**

---

## 0. 方法：核验器与它的双向自检

docs/36 §0 记下的那条教训是本次一切核验的口径：
**按记忆猜 DOI 再送去解析，10 次失败里只有 2 次返回 404，另 8 次解析成功
但指向完全无关的论文。**「DOI 能解析」与「DOI 是对的」是两个命题。
所以本次所有 DOI 判定都比对**题名 token 重合率 + 作者姓集合重合率**，
不看 HTTP 状态码。

工具：`refcheck.py`（会话 scratchpad，未入库；其判据已在下文逐条写出，
每条结论都附了可独立重跑的 API 口径）。**强制自检，双向，5 例：**

| 自检例 | DOI | 期望 | 实得 |
|---|---|---|---|
| 正向 Radzicka–Wolfenden | `10.1021/bi00405a042` | PASS | PASS（题名 1.0 / 作者 1.0） |
| 正向 Rose 1985 | `10.1126/science.4023714` | PASS | PASS（1.0 / 1.0） |
| **反向：能解析但猜错** | `10.1007/BF02289451` | FAIL | FAIL（0.0 / 0.0） |
| 反向：题名对、作者全错 | `10.1021/bi00405a042` | FAIL | FAIL（1.0 / **0.0**） |
| 反向：作者对、题名全错 | `10.1021/bi00405a042` | FAIL | FAIL（**0.0** / 1.0） |

第三例满足 §10 要求 2（植入的错误必须**在被实际扫描的区间内**真的错）：
`10.1007/BF02289451` **确实解析成功**，指向 Schönemann,
*A Generalized Solution of the Orthogonal Procrustes Problem*,
*Psychometrika* 1966;31:1–10 —— 一篇真实但完全无关的论文。
只看状态码的检查器会放过它。
（顺带核实 docs/36 [25] 的 Gollob DOI `10.1007/BF02289676` **是对的**：
题名/作者/卷 33/页 73–115 全部命中。）

后两例是**不误伤的反面**：防止判据退化成「只比题名」或「只比作者」。

---

## 1. docs/36 §4 的 4 项 —— 逐项结果

| # | 项目 | 结果 |
|---|---|---|
| 1 | `starr_animalACE2` 的出版物 | **仍未结清**（确认性否定，网已扩大），见 §1.1 |
| 2 | `starr_PDCoV_RBD` 的出版物 | **仍未结清**（确认性否定，网已扩大），见 §1.1 |
| 3 | Fauchère–Pliska 的 DOI / PMID | **已结清：两者都不存在**，且给出机制与第二源，见 §1.2 |
| 4 | 3SGB 那篇 1983 Biochemistry 的 DOI | **已结清：DOI + PMID 都拿到并核对通过**，见 §1.3 |

**4 项中结清 2 项**（第 3 项结清为一个**确定的否定**，第 4 项结清为肯定）。
第 1、2 项的人类裁定已由 **D-072** 作出（按数据集引用），本次不改变该裁定，
只把「试过什么」扩到可以一步收尾的程度。

### 1.1 两个数据集：仍无任何出版物（D-072 已裁定，此处只扩网）

docs/36 §4.1 已做过 Europe PMC 的多角度检索。本次**新增**以下口径，全部为 0：

| 新增检索 | 口径 | 结果 |
|---|---|---|
| OpenAlex 全库题名/全文检索 | `search=deep mutational scanning porcine deltacoronavirus receptor binding domain aminopeptidase N`；`search=deep mutational scanning SARS-CoV-2 RBD animal ACE2 orthologs deer hamster bat cat` | 各 8 条命中，**逐条看过，无一相符**（全是综述、基因组流行病学、冷冻电镜结构） |
| Zenodo REST | `q=tstarrlab` | **0 条** —— 两个 repo 都没有 Zenodo 存档 |
| NCBI BioProject / BioSample / SRA | `porcine deltacoronavirus RBD`、`deltacoronavirus RBD DMS`、`PD-CoV RBD` 三词 × 三库 | 9 次检索，8 次 0 条；唯一命中是 **PRJNA768120**，经 esummary 确认为北京大学的
「ACE2-PD Saturation Mutagenesis Library Sorting」，**无关**，已排除 |
| GitHub API 仓库现状（2026-10-15 复查） | `tstarrlab/SARS-CoV-2-RBD_DMS_animal-ACE2`、`tstarrlab/PD-CoV-RBD_DMS` | `pushed_at` 仍为 **2024-03-19** 与 **2025-07-07**（与 docs/36 记录一致，一天内无变化）；`homepage` 为 null；README 对 `doi\|biorxiv\|medrxiv\|preprint\|manuscript\|paper\|publish\|cite\|zenodo` **仍零命中** |
| 两个浮现的近似项，逐条排除 | — | PLOS Biol 2026 `10.1371/journal.pbio.3003882`（Upadhyay 等 12 人，Newman 在列）是**抗体表位中和**研究，无 RBD×ACE2 的 DMS 面板；J Gen Virol 2025 `10.1099/jgv.0.002092`（Lean 等 19 人，Gallo + Newman 在列）是 APN 的**免疫组化分布**研究。两者都**不是**这两份数据的出版物 |

**一个必须写下来的陷阱（本次差点踩进去）**：两个 repo 都有
`SRA_upload_Illumina/` 与 `SRA_upload_PacBio/` 目录，其中
`upload_config.yaml` 写着 `biosample_accession: SAMN30603946`，
`README.md` 写着 BioProject **PRJNA770094**。
**这两个值不能当作这两份数据的 provenance**：

- 两个 repo 里的这两个文件**字节级相同**，且 `ftp_subfolder` 都写
  `Omicron_variant_RBD_Illumina_bcs`；
- PD-CoV 那份的 README 里 `Organism` 仍写
  "Severe acute respiratory syndrome-related coronavirus" ——
  而 PD-CoV 是 **delta**coronavirus。

即这是从 `SARS-CoV-2-RBD_DMS_variants` 模板**原样继承的样板文件**，
不是本研究的真实登记号。它完美符合 docs/36 §0 那条教训的形状：
**一个能解析、看起来对、其实指向别的东西的标识符。**
→ **不得写入任何 provenance 清单。**

`assessment:` 这两项的状态与 docs/36 相同，但「试过什么」现在覆盖了
Europe PMC + OpenAlex + Zenodo + NCBI 四类索引与 repo 现状复查
（confidence: 高，就「尽力检索未发现」这一结论而言；
basis: 上表各 API 的零命中与逐条排除）。
**结论仍应按「未找到」读，不是「不存在」。**
人类一步收尾所需的唯一动作：若 D-072 的裁定要升级，需要向 Starr 实验室
直接询问这两份数据是否有在投稿件。

### 1.2 Fauchère–Pliska：DOI 与 PMID **都不存在**，且给出了原因

docs/36 §4.2 的状态是「未找到出版方记录；卷/页只有 AAindex 一个来源」。
本次把它从「未找到」推进到**确定的否定 + 机制 + 第二源**：

**(a) 不可能有 PMID。** NLM Catalog（NLM ID **0420510**，efetch `db=nlmcatalog`）
对 *European Journal of Medicinal Chemistry* 的收录声明逐字为：

> Fully Indexed In: MEDLINE: **v35n1, Jan. 2000-**; PubMed: **v35n1, Jan. 2000-**

该刊 1974 年创刊，而 MEDLINE/PubMed 只从**第 35 卷（2000 年）**起收录。
验证：`esearch db=pubmed term="Eur J Med Chem"[ta] AND 1980:1990[dp]` → **0 条**
（全刊共 16326 条，全部在 2000 年以后）。
1983 年的第 18 卷从未被 PubMed 收录，**所以 PMID 不存在**，
这也解释了为什么 `cache/aaindex1.txt` 里 `FAUJ830101` 的 `R` 字段本身就是空的。

**(b) 没有 DOI。** Crossref 的该刊（ISSN 0223-5234）登记共 19280 条，
按 `sort=published&order=asc` 取最早的 5 条，全部是**第 22 卷（1987）**；
`filter=from-print-pub-date:1983-01-01,until-print-pub-date:1985-12-31` → **0 条**。
即 Elsevier 的回溯数字化只做到 1987 年，1983 年的卷次从未注册 DOI。
独立佐证：OpenAlex 记录 **W3182520187** 的 `doi` 字段为 **null**，
而该记录 `cited_by_count = 696` —— 一篇被引近七百次的论文若有 DOI，不会未注册。

**(c) 卷/页现在有第二源，并多出了期号。** OpenAlex W3182520187：

| 字段 | AAindex `FAUJ830101` | OpenAlex W3182520187 |
|---|---|---|
| 年 | 1983 | 1983 |
| 卷 | 18 | 18 |
| 期 | （无） | **4** |
| 页 | 369–375 | 369–375 |
| 刊名 | Eur J Med Chem | **Chimie thérapeutique**（ISSN 0009-4374） |

刊名差异不是矛盾，而是互相印证：NLM Catalog 的 Notes 字段逐字写
"Added title: Chimica therapeutica, 1974-1981; **Chimie thérapeutique, 1982-**"。
1983 年该刊的并列刊名正是 *Chimie thérapeutique*。

**`gaps:` 仍然保留的不确定性**：OpenAlex 的作者字段是 MAG 解析产物且**已损坏**
（`['E Fauch', 'span RE J-L', 'Pliska']`，可见 HTML 片段残留）。
**所以 OpenAlex 只能用来佐证卷/期/页与「无 DOI」，不能用来核作者列表。**
作者列表仍只有 AAindex 一个来源。

**建议的可引形式**（与 docs/36 [16] 一致，只加期号与否定说明）：

> Fauchère JL, Pliska V. Hydrophobic parameters π of amino-acid side chains from
> the partitioning of N-acetyl-amino-acid amides. *Eur J Med Chem*
> 1983;18(4):369–375.（无 DOI：该刊 Crossref 回溯仅至 1987 年第 22 卷；
> 无 PMID：MEDLINE 收录自 2000 年第 35 卷起。）

→ docs/36 §4.2 那句「**不要让任何人给这条补一个 DOI**」现在有了理由，不只是警告。

### 1.3 3SGB 的文献 DOI：已拿到，并双向核对

docs/36 §4.3 的状态是「猜的两个 ACS DOI 均解析到无关论文，已丢弃」。
本次**不猜**，走检索：

1. RCSB Data API `core/entry/3SGB` 给出 primary citation 的**完整题名**
   （docs/36 只记了刊名/卷/首页）：
   "Structure of the complex of Streptomyces griseus protease B and the third
   domain of the turkey ovomucoid inhibitor at 1.8-A resolution."，
   作者 Read RJ / Fujinaga M / Sielecki AR / James MN。该条目的
   `pdbx_database_id_doi` 与 `pdbx_database_id_pub_med` 字段**都是 null**
   —— 这正是 docs/36 卡住的地方。
2. `esearch db=pubmed term=Read RJ[au] AND James MN[au] AND Biochemistry[ta] AND 1983[dp]`
   → 唯一命中 **PMID 6414511**。esummary 返回的题名与 4 位作者与 RCSB **逐字一致**，
   并给出 `doi: 10.1021/bi00288a012` 与完整页码 **4420–4433**（RCSB 只有首页 4420）。
3. 把该 DOI 送 `refcheck`（题名 + 作者比对，非状态码）：**PASS**，
   题名重合 1.0 / 作者重合 1.0；Crossref 返回
   *Biochemistry* **22(19):4420–4433, 1983**，4 位作者。
   （Crossref 的题名把 "1.8-A" 写成 "1.8-.ANG."，是 ACS 的字符转义，非不一致。）

**可引形式：**

> Read RJ, Fujinaga M, Sielecki AR, James MNG. Structure of the complex of
> *Streptomyces griseus* protease B and the third domain of the turkey ovomucoid
> inhibitor at 1.8-Å resolution. *Biochemistry* 1983;22(19):4420–4433.
> **doi:10.1021/bi00288a012. PMID 6414511.**
> 〔PubMed esummary 6414511 + Crossref 题名/作者比对 PASS；
> 题名与作者与 RCSB `core/entry/3SGB` 的 primary citation 逐字一致〕

（3SGB 已按 D-046 剔出证据集，这条只是把书目补全，不影响任何统计量。）

---

## 2. docs/36 留开的其余四项登录号 / DOI / PMCID

### 2.1 `CHOP780202` 确实是代码实际加载的那条 β 折叠倾向标度 —— 已确认

docs/36 [20] 只说该 accession「由 `analysis/subspace_lib.py` 确定」。本次直接验证：

- `analysis/subspace_lib.py` `load_scales()` 的 `want` 字典里逐字是
  `"CHOP780202": "sheet_prop_Chou"`，且 `"CHOP780201": "helix_prop_Chou"`；
  **没有任何其他 β 折叠倾向登录号被加载**（全表 15 个 AAindex 登录号已逐条看过）。
- 实际 `load_scales()` 返回的 `sheet_prop_Chou` 向量与 `cache/aaindex1.txt`
  `CHOP780202` 的 `I` 区块**逐值一致**，n_finite = 20（无缺失）。
- **用已知事实自检**（不是自洽性）：Chou–Fasman 1978 的 P_β 教科书值
  Val = **1.70**（最大）、Glu = **0.37**（最小）、Pro = 0.55、Gly = 0.75
  —— 四个全中。配套的 `CHOP780201`（P_α）给 Glu = **1.51**（最大）、
  Gly = Pro = **0.57**（最小），同样是教科书值。
- 该条目的 `D` 字段逐字 "Normalized frequency of beta-sheet (Chou-Fasman, 1978b)"，
  `R PMID:364941`，`J Adv. Enzymol. 47, 45-148 (1978)` —— 与 docs/36 [20] 一致。

→ docs/36 §5.8「草稿 §3.3 必须写明 accession」的理由成立，且该 accession
**确为代码所载**，不是一个看起来合理的猜测。docs/36 [20] 可从「待确认」转为已确认。
§5.8 关于三类性质不同的 β 倾向标度（`CHOP780202` 统计频率 /
`KOEP990102` 结构导出 / `KIMC930101` 实验 ΔΔG）的提醒照旧有效。

### 2.2 `MCMT640101` 的可引形式 —— 连转引源 Jones 1975 一起结清

docs/36 §4.3 把「Jones (1975) 是哪一篇」记为未检索。本次用 **AAindex 自身**结清，
不靠外部猜测：

`cache/aaindex1.txt` 里另有两条标度的 `D` 字段写 "(Jones, 1975)"
—— `JOND750101`（Hydrophobicity）与 `JOND750102`（pK (-COOH)）。
两条的字段逐字为：

```
R PMID:1127956
A Jones, D.D.
T Amino acid properties and side-chain orientation in proteins: A cross
  correlation approach
J J. Theor. Biol. 50, 167-183 (1975)
```

且 `JOND750102` 的 `J` 字段继续写：
"Original reference of this data: McMeekin, T.L., Groves, M.L. and Hipp, N.J.
In "Amino Acids and Serum Proteins" (Stekol, J.A., ed.), American Chemical
Society, Washington, D.C., p. 54 (1964)" —— 与 `MCMT640101` 的 `J` 字段**同一出处**。

即 **AAindex 自己用 `R PMID:1127956` 把「Jones (1975)」指实了**。
外部核验：esummary 1127956 → Jones DD, *J Theor Biol* 1975;50(1):167–183,
doi `10.1016/0022-5193(75)90031-4`；该 DOI 送 refcheck **PASS**（1.0 / 1.0），
Crossref 卷 50 期 1 页 167–183、单作者。
（注：PubMed 该条记录的题名里有一个拼写错误 "appraoch"，
Crossref 与 AAindex 都写 "approach"。**不要按 PubMed 去「订正」题名。**）

**`facts:`** `MCMT640101` 的 `T`（题名）与 `R`（PMID）字段**都是空的**
（已直接看原始字节），AAindex 没有记录章节题名；仓库自检期望值全中
（Gly = 0.00 最小 / Trp = 42.53 最大）。

**建议的可引形式**（取代「McMeekin 1964」这一篇期刊论文的写法）：

> **摩尔折射度**（AAindex accession `MCMT640101`）。原始来源：
> McMeekin TL, Groves ML, Hipp NJ. In: Stekol JA, ed. *Amino Acids and Serum
> Proteins*. Washington, D.C.: American Chemical Society; 1964:54.
> （三作者书章；AAindex 未记章节题名、无 PMID、无 DOI。）
> 数值经以下汇编转录进 AAindex：Jones DD. Amino acid properties and side-chain
> orientation in proteins: a cross correlation approach. *J Theor Biol*
> 1975;50(1):167–183. doi:10.1016/0022-5193(75)90031-4. PMID 1127956.
> 〔`MCMT640101` 的 `D`/`A`/`J` 字段逐字；"Jones (1975)" 由 AAindex 自身的
> `JOND750101`/`JOND750102` 条目的 `R PMID:1127956` 指实；该 PMID 与 DOI
> 经 NCBI esummary 与 Crossref 题名/作者比对双向核验〕

正文措辞照 docs/36 §5.5 的建议改为「实验折射度的汇编值（经 AAindex，
源头为一份 1964 年书章，由一份 1975 年汇编转引）」。

### 2.3 Tsuboyama：正文引**版本** DOI，数据可得性里并给 concept DOI

docs/36 §5.6 的建议方向正确，本次补上使之成为决定的证据，并多查出一件事。

**登记方（DataCite，不是 Zenodo 自己的 API）的记录：**

| 口径 | `10.5281/zenodo.7401274` | `10.5281/zenodo.7992926` |
|---|---|---|
| 角色 | **concept DOI**（全版本） | **版本 DOI** |
| DataCite `state` | findable | findable |
| `relatedIdentifiers` | HasVersion → 7401275 / 7844779 / **7992926** | IsVersionOf → **7401274** |
| Zenodo `conceptdoi` 字段 | — | `10.5281/zenodo.7401274` |

双向关系由登记方自己的记录确认，不是推断。

**一件 docs/36 没记、但直接决定该引哪个的事实**：该 concept 下共有
**3 个版本**（`q=conceptrecid:7401274&all_versions=true`）：

| 记录 id | 版本 DOI | `version` 字段 | 发布日 |
|---|---|---|---|
| 7401275 | `10.5281/zenodo.7401275` | （空） | 2022-12-06 |
| 7844779 | `10.5281/zenodo.7844779` | **v2_230420** | 2023-04-21 |
| **7992926** | `10.5281/zenodo.7992926` | **v2_230420** | 2023-04-21 |

**7844779 与 7992926 的 `version` 字符串完全相同（都是 `v2_230420`），
发布日也相同。** 即「v2_230420」这个人类可读的版本名**不能唯一确定一个记录**，
只有版本 DOI 能。7992926 的 `relations.version.is_last = true`、`index = 2`。

**结论（建议，非自行决定 —— docs/36 §5.6 已把它定为编辑问题）：**
正文与方法引 **`10.5281/zenodo.7992926`（版本 DOI，v2_230420）**，
理由不是惯例而是上面这条：版本名有歧义，版本 DOI 没有，而本仓库的
`cache/dms/tsuboyama2023/` 是从 7992926 取的（清单含 md5+sha256）。
数据可得性声明里并列 concept DOI **`10.5281/zenodo.7401274`**，
以便读者取到后续版本。license 为 CC-BY-4.0，创建者 9 人（与论文 [4] 一致）。
顺带：`tsuboyama_PROVENANCE.txt` 的 7 文件清单建议按 docs/36 [5] 标注归属 zip。

### 2.4 CR9114：预印本与正式版的标识符，分开列清

docs/36 §5.12 第 1 条判断**完全正确**，本次把两边的标识符都补全：

| | **预印本** | **PLOS Biology 正式版** |
|---|---|---|
| 题名 | Vulnerability in the breadth evolution of an influenza broadly neutralizing antibody | Epistasis and pleiotropy constrain the evolution of cross-reactive breadth in a broadly neutralizing influenza antibody |
| DOI | `10.1101/2025.11.03.686356` | `10.1371/journal.pbio.3004021` |
| **PMID** | **41279651** | **42804473** |
| **PMCID** | **PMC12637573** | **PMC13638050** |
| Europe PMC id | PPR1114135 | MED 42804473 |
| 作者数 | **9** | **10**（新增 Lv H / Huibin Lv） |
| 刊/卷期 | bioRxiv，2025-11-04 | *PLoS Biol* 2026;24(9):e3004021 |

核验口径：`esummary db=pmc id=12637573` 返回题名
"Vulnerability in the breadth evolution of an influenza broadly neutralizing antibody."、
`fulljournalname` = "bioRxiv : the preprint server for biology"、`pubdate` 2025 Nov 4、
作者 9 人、`articleids` = (pmid 41279651, pmcid PMC12637573,
doi 10.1101/2025.11.03.686356)。
`esummary db=pubmed id=42804473` 返回 *PLoS Biol* 24(9) e3004021、作者 10 人
（Dailey KE, Wang Y, Teo QW, Wang C, **Lv H**, Lei R, Tong M, Rodriguez LA, Wang L, Wu NC）、
`articleids` 含 pmc PMC13638050 与 doi 10.1371/journal.pbio.3004021。
Europe PMC 的 `DOI:` 检索两边各自独立复现同一结果。

→ **`cr9114_PROVENANCE.json` 的更正**：把 `PMC12637573` 移到预印本名下
（并可补 PMID 41279651），正式版填 **PMID 42804473 / PMCID PMC13638050**。
docs/36 §5.12 第 1 条里「PLOS 文章自身的 PMCID 本次未确定」这句现在可以结清。
第 2、3 条不变 —— 特别是 **Zenodo gap 保持开启**：
`10.5281/zenodo.22697938` 能解析不等于仓库当初是从哪里读到它的，
该 gap 要等追到 PLOS Biology 的 Data Availability 声明才能关。

---

## 3. 标度标签更正的传播（D-071 第 1 条）

### 3.1 事实重述

`RADA880106` 的 AAindex 字段逐字：

```
D Accessible surface area (Radzicka-Wolfenden, 1988)
A Radzicka, A. and Wolfenden, R.
J Biochemistry 27, 1664-1670 (1988) (Pro missing)
```

Rose et al. 1985 的两条标度在同一份文件里：
`ROSG850101`（Mean area buried on transfer）、
`ROSG850102`（Mean fractional area loss），
都是 `R PMID:4023714`、*Science* 229, 834-838 (1985)。
**本仓库从未加载过其中任何一条。**

所以错的不只是作者名，**「埋藏」这个词也是错的** ——
`RADA880106` 是**可及**表面积，不是埋藏面积。
更正后统一写作「可及表面积（Radzicka–Wolfenden 1988）」，与 docs/35 §3.3 已修的措辞一致。

### 3.2 逐文件更正表

**约束：一个数字都没有改。**

| 文件 | 行 | 改前 | 改后 | 方式 / 理由 |
|---|---|---|---|---|
| `analysis/generic_axis_test.py` | 47 | `"asa_buried_Rose",`（`VOL_SCALES` 成员） | `"asa_RadzickaWolfenden",` | **原地改**。这不只是标签 —— `subspace_lib.py:590` 改名后该键已不存在，`scales[k]` 会 **KeyError**，脚本当前是坏的 |
| `analysis/tsuboyama_axis.py` | 519 | `"asa_buried_Rose",`（`want` 成员） | `"asa_RadzickaWolfenden",` | **原地改**。同上，`scales.get(nm, dis.get(nm))` 会取到 `None`（`cache/generic_disrupt_scales.json` 的 `scales` 键里没有这一条，已逐键确认） |
| `analysis/tsuboyama_axis.py` | 548 | `…极化率 0.568、埋藏 ASA 0.431、…`（打印的参照行） | `…极化率 0.568、可及表面积 0.431、…` | **原地改**。纯显示字符串 |
| `docs/30-manuscript.md` | 48 | `\| Rose 埋藏 ASA \| +0.656 \| 0.431 \|` | `\| 可及表面积（Radzicka–Wolfenden 1988） \| +0.656 \| 0.431 \|` | **原地改**。前向主张文档，应当直接正确 |
| `docs/30-manuscript.md` | 51 | `…侧链体积/极化率/埋藏 ASA；…` | `…侧链体积/极化率/可及表面积；…` | **原地改**。同上 |
| `docs/35-paper-draft.md` | 25 | `…侧链体积/极化率/埋藏面积（r² = 0.43–0.61）。` | `…侧链体积/极化率/可及表面积（r² = 0.43–0.61）。` | **原地改**。§3.3 表此前已修，这是同文件里漏掉的一处（摘要段） |
| `docs/13-subspace-universality.md` | 455（`\| 埋藏 ASA (Rose) \| +0.723 \| 0.523 \|`） | — | 正文**不动** | **文末追加 `## 更正`**（照 docs/33 末尾的约定）。本文件是「当时相信什么」的记录；另注明 §7 表的 +0.723/0.523 是三组口径、不在本次复现的缓存里，**本更正不对该数字背书** |
| `docs/18-postmortem-why-two.md` | 163（`Rose 埋藏 ASA +0.723`） | — | 正文**不动** | **文末追加 `## 更正`** |
| `docs/21-censor-density-revision.md` | 567（`\| 埋藏 ASA (Rose) \| +0.656 \| 0.431 \|`）、41 / 577 / 592（裸「埋藏 ASA」） | — | 正文**不动** | **文末追加 `## 更正`** |
| `docs/26-generic-axis.md` | 365 / 509 / 514 / 756 / 760 / 874（全是裸「埋藏 ASA」，无作者名） | — | 正文**不动** | **文末追加 `## 更正`**。docs/36 §5.3 点名的 §514 在此 |
| `docs/31-number-audit.md` | 57 / 61 / 63 / 156 / 339 / 340 | — | 正文**不动** | **文末追加 `## 更正`**，**并额外加一条例外**：§M-3 与 §7 里 `gate_consequences.json 场景 D asa_buried_Rose` 是在陈述**缓存 JSON 的字面键名**，该键至今仍是 `asa_buried_Rose`，**改了会让审计链指向不存在的键** |
| `JOURNAL.md` | 1388 / 1633–1649 | — | 不动 | 1633 起那一段**本身就是这次事故的登记**；1388 是审计叙述。追加新条目，不改旧行 |
| `state/decisions.md` | 1356–1358 | — | 不动 | D-071 条目本身即事故记录。追加 D-076 |
| `docs/16-why-two.md` | — | — | **未触碰** | §4 冻结预注册。**该标签在此文件里一次都没有出现**（`grep -i -E "rose\|ASA\|表面积"` 零命中），故无需记「出现了但不改」 |
| `docs/36-references.md` | 多处 | — | 不动 | 该文件是审计本身，错标签是它的**陈述对象** |
| `cache/gate_consequences.json` 等缓存 | — | — | **不动** | 键名是历史产物；改缓存键会使 docs/31 的审计引用失效，且需重跑才能保证一致 |

**原地改 6 处（3 个代码位 + 3 个前向主张文档位），追加更正说明 5 个文件
（docs/13、18、21、26、31）。**

**行号的一条注意**：本次与另一个会话并行（它在改 docs/36 §1.2 与 docs/35 §1.2），
期间产生了 `a4460ee`、`9c08b96` 两次提交，**本次的工作树改动被卷进了那两次提交**
（提交信息与本次内容无关）。因此上表行号按**写作时的实际状态**记：
`docs/13` 的那一行已从 450 漂到 **455**，`docs/35` §3.3 的表从 179 漂到 **209**
（§3.3 表此前已修，本次未再动它）。引用行号时请以内容而非行号为准。

### 3.3 代码侧的一个顺带发现：改名曾使两个脚本不可运行

D-071 只说「已改名」。但 `subspace_lib.py` 的键名改动**没有同步到下游**：
`generic_axis_test.py:47` 与 `tsuboyama_axis.py:519` 仍按旧键取值。
前者 `scales[k]` 直接 KeyError；后者取到 `None` 后在 `cen_unit` 处爆。
**改名本身是对的，漏改下游是另一个错。** 已修，并静态核验
`load_scales()` 返回的键里 `VOL_SCALES` 与全部参照标度都在、
全仓库 `.py` 里不再有 `asa_buried_Rose`。

`assessment:` 这条值得记进 §10 的类账：**重命名是一种会静默传播的改动**，
与 D-023 同形（不报错或在远处报错）。
按 D-070 的思路，可考虑给 `check_repo.py` 加一条「`load_scales()` 的键集
与各脚本里被索引的键名一致」的机械检查（confidence: 中高；basis: 本次两处都是
靠人读 grep 结果发现的，不是靠任何检查器）。

### 3.4 数值有没有也错？ —— 没有。独立复现确认 D-071 的判断

按任务约束，若发现数字也可疑就停下报告而不修。**做了独立复现，数字是对的：**

从 `cache/aaindex1.txt` 重新解析 `RADA880106`，对 `cache/gate_consequences.json`
场景 D 的 `u` 做相关：

| 标度 | r | r² | n |
|---|---|---|---|
| `RADA880106` | **+0.656253** | **0.430667** | 19 |
| `sheet_prop_Chou` (`CHOP780202`) | +0.779052 | **0.606922** | 20 |
| `local_polarizability` | +0.753762 | 0.568157 | 20 |
| `molar_refractivity` | +0.720368 | 0.518930 | 20 |

**逐位复现** docs/30 §2 / docs/35 §3.3 / docs/21 §6 的 +0.656 / 0.431，
以及 docs/35 的 0.607 / 0.568 / 0.519。
**且 `RADA880106` 的 Pro 位被 `scale_redundancy.parse_aaindex` 判为缺失
（n = 19，不是 20），不是读成 0** —— 这正是 D-023 的失效模式，
在这条上**没有复发**。

→ **D-071「纯标签错误而非数值错误」的判断成立，无需重新审视。**

### 3.5 一件**不是**标签问题、也不由本次处理的事（只报告）

复现过程中确认了一处**数值口径混用**，它与本次标签更正无关，且已在
`state/decisions.md` 末尾与 `JOURNAL.md` 末尾登记为待人类裁定项，这里只把诊断写精确：

`docs/30-manuscript.md` §2 与 `docs/35-paper-draft.md` §3.3 的同一张五行化学标度表，
**五行不同源**：

| 行 | 表里写的 | 场景 A（8 成员） | 场景 D（9 成员） | 属于 |
|---|---|---|---|---|
| sheet propensity | +0.779 / 0.607 | +0.827 / 0.684 | **+0.779 / 0.607** | **D** |
| 侧链极化率 | +0.754 / 0.568 | +0.709 / 0.502 | **+0.754 / 0.568** | **D** |
| 摩尔折射度 | +0.720 / 0.519 | +0.677 / 0.458 | **+0.720 / 0.519** | **D** |
| 可及表面积 | +0.656 / 0.431 | +0.610 / 0.373 | **+0.656 / 0.431** | **D** |
| **Kyte–Doolittle** | **+0.557 / 0.311** | **+0.558 / 0.311** | +0.472 / 0.223 | **A** |

即 **4 行是 9 成员口径、第 5 行是 8 成员口径**，而 `docs/30` 紧随该表的
`assessment:` 却写「本表与 §2 全部统计量**统一用 8 成员口径**」。
这与 docs/31 §5 记的状态不同（docs/31 当时记 docs/30 的 KD 行是 +0.472/0.223），
说明该行在 docs/31 之后被单独改成了 8 成员值，**而其余四行没有跟着改**，
于是「不可混引」的禁令现在被违反在**同一张表内部**。

**按任务约束与 CLAUDE.md §9，本次不动任何数字。** docs/24 §6.5 只说两个口径
不可混引、没说该引哪个，docs/31 §9 第 1 问与 `state/decisions.md`
末尾的「两个数字不一致，交人类裁定」已经把它挂起。
本节只是把它从「两个数不一致」精确到「一张表里 4 行 D + 1 行 A，
且表下的口径声明与 4/5 行矛盾」。

---

## 4. Rose 自己的标度冗余吗？ —— 测量结果与建议

docs/36 [19] 指出 Rose 的标度存在但未被加载，并把「是否真的换用并重算」
列为人类决策点（§5.3 的选项 b）。本节**只测量**，
**没有把任何一条加进 `docs/35` 或 `docs/30` 的任何表**。

口径：`ROSG850101` / `ROSG850102` / `RADA880106` 对
`cache/gate_consequences.json` 四个共识场景的轴 `u` 求相关；
bootstrap 对**残基**重抽 4000 次取 r² 的 95% 区间（CLAUDE.md §5：≥2000）；
并给出 **matched support** 口径（`RADA880106` 缺 Pro，把 Pro 一并从所有标度剔除，
n = 19），因为不对齐支撑集的比较不是同一件事。
先过了一次已知事实自检（Rose `ROSG850101` max = Trp / min = Gly 且 Pro 存在；
`RADA880106` 的 Pro 为缺失而非 0；`CHOP780202` Val 1.70 / Glu 0.37；
反向：打乱后 |r| < 0.95）。

### 4.1 三条标度彼此并不是同一条标度的不同写法

| 配对 | r | r² [95% bootstrap] | n |
|---|---|---|---|
| `RADA880106` vs `ROSG850101` | +0.861 | 0.742 [0.452, 0.904] | 19 |
| `RADA880106` vs `ROSG850102` | +0.111 | 0.012 [0.000, 0.245] | 19 |
| `ROSG850101` vs `ROSG850102` | +0.602 | 0.362 [0.138, 0.626] | 20 |

（AAindex 自己在 `RADA880106` 的 `C` 区块里记 `ROSG850101 0.814`，与此相容。）

→ **`ROSG850101` 不是在用标度的近似复本**（共享约 3/4 方差，不是 1）；
**`ROSG850102` 与在用标度基本无关**（r² = 0.012）—— 它量的是
「埋藏**比例**」而非绝对面积，与 sheet propensity 反而更像（r = +0.739）。

### 4.2 对普适轴的解释力：点估计诱人，区间不支持

matched support（n = 19）口径：

| 场景 | `RADA880106`（在用） | `ROSG850101`（Rose 埋藏面积） | `ROSG850102`（Rose 埋藏比例） | `sheet_prop_Chou`（现报最佳单条） |
|---|---|---|---|---|
| A（8 成员） | 0.373 [0.090, 0.684] | **0.755** [0.584, 0.890] | 0.544 [0.329, 0.788] | 0.654 [0.456, 0.840] |
| B（OMTKY3 退 pooled） | 0.593 [0.374, 0.806] | 0.568 [0.205, 0.815] | 0.031 [0.000, 0.290] | 0.048 [0.000, 0.460] |
| C（只剩冠状病毒） | 0.359 [0.066, 0.697] | **0.760** [0.554, 0.908] | 0.598 [0.372, 0.840] | 0.684 [0.532, 0.841] |
| D（9 成员，正文表口径） | 0.431 [0.139, 0.747] | **0.800** [0.642, 0.914] | 0.494 [0.277, 0.742] | 0.567 [0.338, 0.818] |

**按本仓库自己的判据（§5：比区间重叠与否，不比均值大小）：**

| 场景 | `ROSG850101` vs 在用的 `RADA880106` | `ROSG850101` vs `sheet_prop_Chou` |
|---|---|---|
| A | 区间**重叠** → 不可分辨 | 区间**重叠** |
| B | 区间**重叠** → 不可分辨 | 区间**重叠** |
| C | 区间**重叠** → 不可分辨 | 区间**重叠** |
| D | 区间**重叠** → 不可分辨 | 区间**重叠** |

**四个场景无一例外，区间全部重叠。** 20 个氨基酸上的 r² bootstrap 区间宽到
0.3–0.6，这个样本量分辨不了 0.43 与 0.80 的差别。

### 4.3 判读与建议（**不自行决定**）

**回答任务的问题：Rose 自己的标度与在用的那条既不是近似复本、也不是可区分地更好。**

- 作为**标度**，它们不冗余（§4.1：r² ≈ 0.74，不是 1；`ROSG850102` 几乎无关）。
- 作为**对普适轴的解释**，`ROSG850101` 的点估计在 A/C/D 三个场景里
  明显高于在用标度（0.755 / 0.760 / 0.800 vs 0.373 / 0.359 / 0.431），
  **在场景 D 甚至高于现报的「最佳单条」sheet propensity（0.800 vs 0.567）**——
  但**没有一个场景的区间是分开的**，所以按本仓库的纪律，这不构成差异证据。

**建议：不要在当前证据下给 §3.3 加行，也不要执行 docs/36 §5.3 的选项 (b)。**
三条理由，第三条是关键：

1. **区间全重叠**，§5 的判据直接否决「它更好」这个陈述。
2. **搜索量会变。** `generic_axis_test.py` 的 `n_search = len(DISRUPT_KEYS) + 3
   + len(VOL_SCALES) + 2`，置换零分布的运气天花板按 `n_search` 取最大 |r|。
   加两条标度要把天花板重算，不是加两行表格。
3. **这是事后挑选的教科书形状。** 先看到 `ROSG850101` 在场景 D 上
   r² = 0.80 最高、再决定把它写进「最佳单条」那张表，
   与 CLAUDE.md §5「标签置换对照要在实际候选量级下做」直接冲突。
   若要引入，必须**先预注册**（声明要加哪两条、为什么、天花板按新 `n_search` 重算），
   然后才看结果。

**但有一条值得人类注意**：若最终选择 docs/36 §5.3 的 (b)（真换 Rose 重算），
那么「化学身份的最佳单条标度是 β 折叠倾向」这句**点估计上会翻**
（场景 D：Rose 0.800 > sheet 0.567）。即 (a) 与 (b) 不只是改个名字和引哪篇，
**(b) 会动 §3.3 的 headline 排序**。这正是把它定为人类决策点的理由
（CLAUDE.md §9 第 1 条：改变进入化学身份表的证据集合）。
→ **本次按 (a) 执行（改名、数字不动、引 [18]），(b) 留给人类。**

**顺带**：场景 B 是一个有意思的稳定性提示 —— 那里 `sheet_prop_Chou` 塌到 0.048、
`ROSG850102` 塌到 0.031，而 `RADA880106`（0.593）与 `local_polarizability`（0.744）
不塌。**场景间的差异比标度间的差异大得多**，这对「化学身份」这类陈述
本身就是一条限定，与本次的标签问题无关。

---

## 5. 两条验证命令的结果

两条都在**改动前后各跑一次**，以便区分「本来就过」与「被我改坏/改好」。

| 命令 | 改动前 | 改动后 |
|---|---|---|
| `python3 analysis/check_retracted.py` | 自检通过；13 条撤回登记、3 个前向主张文档；**0 处需处理**；**exit 0** | 自检通过；13 条 / 3 个文档；**0 处需处理**；**exit 0** |
| `python3 analysis/make_figures.py --check` | fig1 ✓ / fig2 ✓ / fig3 ✓ / fig4 ✓，「全部断言通过：图与 docs/35 的数字一致」；**exit 0** | fig1 ✓ / fig2 ✓ / fig3 ✓ / fig4 ✓，同上；**exit 0** |

`make_figures.py --check` 断言的是 `EXPECT` 里 10 个汇总量（cos θ1 中位 0.900、
零分布 p95、144 域中位 0.755、区间 [0.720, 0.775] 等）对缓存的复现，
**不覆盖 §3.3 的化学标度表** —— 所以它的通过**不能**被当作 §3.5 那处
口径混用已被检查过的证据。写在这里以免这张表被读成比实际更强的背书。

---

## 6. 汇总

| 项 | 结果 |
|---|---|
| docs/36 §4 的 4 项 | **结清 2**（Fauchère–Pliska = 确定无 DOI/PMID 且补上第二源与期号；3SGB = DOI `10.1021/bi00288a012` + PMID 6414511）；**2 项仍为确认性否定**（两个数据集无出版物，D-072 已裁定按数据集引用），检索网已扩至 OpenAlex / Zenodo / NCBI 三类新索引 |
| 额外结清 | `CHOP780202` 确为代码所载（并过教科书值自检）；`MCMT640101` 的可引形式连转引源 **Jones DD 1975, PMID 1127956, doi:10.1016/0022-5193(75)90031-4** 一起定下；Tsuboyama 建议引**版本** DOI 7992926（新理由：同 concept 下有两个记录共用 `v2_230420` 这个版本名）并列 concept DOI 7401274；CR9114 预印本 = PMID 41279651 / PMC12637573，正式版 = **PMID 42804473 / PMC13638050** |
| 数字是否也错 | **没有。** 场景 D 的 +0.656253 / 0.430667 逐位复现，`RADA880106` 的 Pro 被正确判为缺失（n=19），**D-023 的失效模式未复发** → D-071 的「纯标签错误」判断成立 |
| 标签更正 | **原地改 6 处**（`generic_axis_test.py:47`、`tsuboyama_axis.py:519`、`tsuboyama_axis.py:548`、`docs/30:48`、`docs/30:51`、`docs/35:25`）；**追加 `## 更正` 说明 5 个文件**（docs/13、18、21、26、31）；`docs/16` 未触碰且其中本无此标签 |
| 顺带修掉的真 bug | `subspace_lib.py` 的改名没有同步到下游，`generic_axis_test.py` 与 `tsuboyama_axis.py` 当时**不可运行**（KeyError / None） |
| 只报告、未处理 | `docs/30` §2 与 `docs/35` §3.3 的五行标度表 **4 行场景 D + 1 行场景 A**，且表下口径声明与之矛盾（§3.5）。已有登记，交人类 |
| 验证命令 | `check_retracted.py` **exit 0**（0 处需处理）；`make_figures.py --check` **全部断言通过** |
| Rose 自己的标度 | **不是近似复本**（r² ≈ 0.74），但**对轴的解释力在四个场景里区间全部与在用标度重叠** → 当前证据不支持加行；且若真换用，§3.3 的 headline 排序会在点估计上翻转，故留作人类决策点 |
