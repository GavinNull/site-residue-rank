# 预注册文件归档

这里是三份**哈希锁定的预注册**的原文与锁文件，各自在对应分析的任何统计量被计算之前写定：

| 文件 | 对应分析 | 内容 |
|---|---|---|
| `docs29_prereg.md` | `docs/29-paired-conflict.md` | 配对矛盾的候选解释与判别观测量 |
| `docs41_prereg.md` | `docs/41-independence-audit.md` | 14/15 符号检验的候选独立单位，主单位在看结果前定死 |
| `docs42_prereg.md` | `docs/42-folding-rank.md` | 折叠侧定秩的单位、程序、功效门槛与「可判定」的判据 |

每份配一个 `*_lock.json`，记录 sha256。对应的分析脚本在入口处断言该哈希，改动一个字节即中止。
**这些文件的工作副本在 `cache/`（被 .gitignore 排除），此处是为公开核对而做的归档副本。**
可自行验证：`sha256sum prereg/docs41_prereg.md` 应等于 `prereg/docs41_prereg_lock.json` 里记录的值。

注：`docs41` 的预注册里**披露了**作者在写它之前已经看过逐子集的方向（见该文件），
这是诚实记录而非瑕疵——它限制了该预注册能证明的东西，读者应当知道。
