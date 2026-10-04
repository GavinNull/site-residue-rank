# 受控因子词表

盲预测时 `dominant_factor` 只能从此表取值。禁止自由发挥——自由文本会让
分层统计失效，而分层统计是 G3 唯一有信息量的产物。

每个因子带一条**可证伪签名**：若某候选代理量声称实现该因子，它在
`probes/` 上必须表现出这个签名。签名不满足即出局，不看它在判别集上的分数。

| ID | 因子 | 物理内容 | 可证伪签名 | 探针 |
|---|---|---|---|---|
| F01 | `salt_bridge` | 带电-带电库仑，电荷辅助氢键 | 引入反电荷时单调增强；屏蔽介电上升时衰减 | `probes/hbond_angle` + 电荷扫描 |
| F02 | `hbond_geometry` | 中性氢键的距离与方向性 | D–H···A 角在 180° 附近取极值，90° 衰减到近零 | `probes/hbond_angle` |
| F03 | `hbond_mismatch` | 供体/受体身份互换导致的失配 | 供体换受体时**反号** | `probes/hbond_angle` |
| F04 | `desolvation_polar` | 极性基团脱水代价 | 极性相同、埋藏加深 → 单调上升 | `probes/burial_depth` |
| F05a | `induction` | 电荷诱导偶极 −½Δα\|E\|² | **1/r⁴**；依赖配体电荷态；恒为负 | `probes/burial_depth` |
| F05b | `dispersion` | London 色散 | **1/r⁶**；需两边极化率；与配体电荷**无关** | `probes/burial_depth` |
| F06 | `cation_pi` | 阳离子-π | 随环上给电子取代增强，吸电子减弱；随高度指数衰减 | `probes/cation_pi` |
| F07 | `pi_stacking` | π-π 几何（平行位移 vs T 形） | 在 3.4–3.8 Å 平行位移处取极值 | 待建 |
| F08 | `anion_pi` | 缺电子芳环-阴离子 | 仅在强吸电子取代时为正 | `probes/cation_pi`（反向） |
| F09 | `halogen_bond` | σ-空穴 | C–X···A 角在 180° 取极值；I > Br > Cl >> F | 待建 |
| F10 | `charge_transfer` | 电荷转移 / 诱导极化 | 与 Hammett σ 单调，\|r\| ≥ 0.9 | `probes/hammett` |
| F11 | `pka_shift` | 质子化态改变 | 对局部静电势单调；在 pKa 附近出现开关 | 待建 |
| F12 | `water_displacement` | 结构水被挤出/保留 | **离散开关**，连续近似必然钝化——见 CLAUDE.md §7 | 待建 |
| F13 | `water_bridge` | 水桥介导氢键的得失 | 同上 | 待建 |
| F14 | `sidechain_entropy` | 侧链构象熵 | 随可旋转键数单调；与焓项正交 | 待建 |
| F15 | `steric_strain` | 位阻 / 形变能 | 随重叠体积超线性上升 | `probes/burial_depth` |
| F16 | `preorganization` | 口袋预组织 / 局部介电响应 | 对非接触残基的电性变化有响应（与 F17 区分） | 待建 |
| F17 | `field_effect` | 无直接接触的远程静电场 | 随距离 1/r² 衰减；对接触几何不敏感 | `probes/hammett` |
| F18 | `hbond_cooperativity` | 氢键网络协同极化 | 网络延长时非加和；加和模型必然低估 | 待建 |
| F19 | `weak_hbond` | CH···O / CH···π | 量级应在 0.3–1.0 kcal/mol，超出即量级错 | 待建 |
| F20 | `chalcogen_aromatic` | 硫-芳香 / 硫族键 | Met/Cys 换 Leu/Ala 时为正 | 待建 |

## F05 的拆分（2026-10-04，由物理抓出）

原 F05 `dispersion_burial` 把两个不同通道混为一谈：
诱导项是 1/r⁴ 且依赖配体电荷态，London 色散是 1/r⁶ 且与电荷无关。
`analysis/limit_checks.py` 的 F05 断言要求 −6，而 MP-1 的诱导项给 −4 ——
这条失败是**正确的失败**。详见 `docs/07-mp1-multipole-channels.md` 第 7 节。

声明实现 F05a 的候选量须过 1/r⁴ 与电荷态依赖；
声明 F05b 的须过 1/r⁶ 与电荷无关性。**不得再声明裸 F05。**

## 使用约定

- 盲预测必须选**恰好一个** `dominant_factor`，另可选最多两个 `secondary_factors`。
  强迫单选是为了让分层有统计意义。
- 若判断"无单一主导因子"，选 `F00 = mixed`，并在 `reasoning` 说明竞争的两项。
  `F00` 的准确率单独报告——它是"agent 承认不知道"的那一类，
  这一类的准确率若接近随机，说明校准是真的。
- `F12`/`F13` 被选中的对，自动进入 `water` 分层。这组的表现决定了
  片段截断 + 隐式溶剂路线的天花板。

## 签名"待建"的处理

探针标"待建"的因子，**不能用于 G4 的可微量实例化**，即使它在 G3 的
盲预测里准确率很高。理由：没有一级机理测试，就无法区分"真抓到了"
与"在小样本上碰对了"。补探针是比搜函数优先级更高的工作。
