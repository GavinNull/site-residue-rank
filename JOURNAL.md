# JOURNAL

只追加，时间序。写**做了什么**与**发现什么不成立**，不写计划。

---

## 2026-10-03 — 仓库建立

建立目录结构、硬规则（`CLAUDE.md`）、纳入判据（`docs/02-pair-criteria.md`）、
验证协议（`docs/03-validation.md`）、盲测协议（`docs/04-blind-protocol.md`）、
受控因子词表（`schema/factor-vocabulary.md`，20 个因子 + F00 mixed）。

标签隔离做成机制而非自律：
- 标签 base64 存于 `labels/labels.b64`，cat/grep/随手读都读不出数值
- `analysis/labelstore.py` 是唯一入口，未锁定盲预测则 `get()` 抛 `PermissionError`
- `analysis/lock.py` 对预测文件记 SHA256，事后改动会被 `--verify` 检出
- `analysis/check_repo.py` 扫 `pairs/` 的 17 条泄露正则（数值型 + 中英文措辞型）

### G0 起手：探针库

`tools/make_probes.py` 生成 60 个理想化构型，4 组：

| 探针 | 构型数 | 目标因子 |
|---|---|---|
| `hbond_angle` | 10 | F02 F03 F01 |
| `hammett` | 8 | F10 F17 |
| `burial_depth` | 7 | F04 F05 F15 |
| `cation_pi` | 35 | F06 F08 |

每组 `manifest.tsv` 的注释里写了**预期签名**与**参考量级**。
签名是定性判据（符号、单调性），参考量级用于量级校准。

### 发现：手工解析构造的探针有三个真实几何错误

`analysis/check_probes.py`（几何自检，在任何电子结构计算之前跑）一次命中 36/60。
这验证了把几何检查前置的必要性——这类错误会让 xtb 悄悄收敛到无意义结构，
而扫描曲线看起来仍然"平滑"。

1. **C–O–H 角度约定反了**（苯酚与甲醇各一处）。把 O→H 的方位角当成了
   C–O–H 角本身，H 折回指向 C1，C1···H 距离 1.40 Å 而非正确的 1.92 Å。
2. **受体水的 HOH 二等分线方向反了**。孤对应朝供体，H 应背离供体；
   写成了 H 朝供体，与邻位环 H 重叠到 0.31 Å。
3. **多原子取代基的内坐标锚点错了**。NMe2 / OMe / NO2 的二壳层原子
   相对苯环碳定位，应相对各自 parent 定位。NMe2 的甲基碳与邻位 H
   重叠到 0.70 Å。改写为标准 NeRF 内坐标后正常（甲基 C···邻位 H 2.45 Å）。

另有一条**验证器自身的误报**：C≡N 真实键长 1.16 Å = 0.79×rcov，
被 0.80 阈值判为"过短"。阈值放到 0.70。记在这里是因为
"验证器误报"和"被验证对象有错"必须区分开，否则会去修改正确的东西。

修复后 60/60 几何正常。

### 环境现状（G0 的实际阻塞点）

```
numpy ok   scipy ok   sklearn ok
xtb MISSING   rdkit MISSING   torch MISSING   openbabel MISSING
Biopython MISSING   openmm MISSING
Python 3.14.5
```

**探针几何已就位，但探针有效性无法验证**——需要 xtb 才能算出扫描曲线
并与参考量级对照。几何正常只说明构型没坏，不说明探针有效。

G0 停在这里，等环境。见 `state/gates.md`。

### 当前状态

- G0 `in_progress`：探针几何完成并自检通过；电子结构工具缺失
- G1 `blocked`：等人类决定数据来源家族（`state/gates.md` 待定项 1）
- G2–G6 未开始

---

## 2026-10-03 — 护栏自测

未经测试的隔离机制比没有更危险，故对两道护栏各做一次对抗测试。

**泄露扫描**：构造含 6 条典型泄露的样例（Kd 数值、浓度量纲、英文强弱措辞、
ΔΔG 数值、中文强弱措辞、倍数变化、方向标签），全部命中（9 处，部分行被两条
正则同时命中）。同一文件里的合法数值——骨架 RMSD 0.62、口袋体积 412 Å³、
接触距离 3.1 Å、温度 298 K——**无一误伤**。这点很关键：误伤会迫使人把
合法的结构事实从 pairs/ 里删掉，反而削弱判别集。

**标签越权读取**：`labelstore.get()` 在盲预测未锁定时抛 `PermissionError`。
`grep -ri A_stronger labels/` 对 `labels.b64` 命中 0 次（base64 存储生效）。

两道护栏按设计工作。
