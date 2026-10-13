#!/bin/bash
# docs/27 取数脚本（CLAUDE.md §3：provenance 强制）。
#
# 人类授权：D-040（2026-10-12），把 Tsuboyama 2023 mega-scale folding stability
# 引入以把折叠一侧的 n 从 1 提上去。**只取已处理的逐变体表，不取原始测序 reads。**
#
# 来源：Zenodo record 7992926，DOI 10.5281/zenodo.7992926
#       标题 "Mega-scale experimental analysis of protein folding stability in
#       biology and design"，publication_date 2023-04-21
# 论文：Tsuboyama K. et al., Nature 620:434-444 (2023)，DOI 10.1038/s41586-023-06328-6
#
# 取回日期：2026-10-12/13（retrieval_date.txt 由本脚本写入 UTC 时间戳）
#
# 没有取的东西（明确记录）：
#   Raw_NGS_count_tables.zip（242.9 MB）—— 原始测序计数，任务明确禁止
#   Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv（1.28 GB，
#       在 zip 内未解出）—— Dataset 1 是未经可靠性筛选的全量 K50，
#       本工作用的是 Dataset 2/3（作者自己的可靠集）
#   K50_dG_tables.zip / Pipeline_*.zip / AlphaFold_model_PDBs.zip —— 不需要
set -euo pipefail
D="$(cd "$(dirname "$0")/.." && pwd)/cache/dms/tsuboyama2023"
mkdir -p "$D"; cd "$D"
date -u +"%Y-%m-%dT%H:%M:%SZ" > retrieval_date.txt

base="https://zenodo.org/api/records/7992926/files"

# 1) 已处理的逐变体 ΔG / ΔΔG 表（主数据）
#    Zenodo 公布的 md5: f7e8c553efee734cf161ee6f2b0a09cf
[ -f Processed_K50_dG_datasets.zip ] || \
  curl -sL -o Processed_K50_dG_datasets.zip "$base/Processed_K50_dG_datasets.zip/content"

# 2) 图用数据表（里面的 dG_extdG_data_Fig1.csv 是**体外文献 ΔG 对照**，
#    用于 §10.3 / D-028 要求的「体外校准 σ」，不是用 fitting CI）
#    Zenodo 公布的 md5: c4b6a64345e30e1b9e2e40a266d14a68
[ -f Data_tables_for_figs.zip ] || \
  curl -sL -o Data_tables_for_figs.zip "$base/Data_tables_for_figs.zip/content"

# 3) 解出本工作实际使用的文件
unzip -o -j Processed_K50_dG_datasets.zip \
  "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv" \
  "Processed_K50_dG_datasets/Single_DMS_list.csv" \
  "Processed_K50_dG_datasets/Double_DMS_list.csv" \
  "Processed_K50_dG_datasets/Triple_DMS_list.csv" -d . >/dev/null
unzip -o -j Data_tables_for_figs.zip \
  "Data_tables_for_figs/dG_extdG_data_Fig1.csv" -d . >/dev/null

# 4) 清单 + 校验和（md5 用于与 Zenodo 公布值对账；sha256 用于本仓库内部固定）
{
  echo "# Tsuboyama 2023 取数清单"
  echo "# zenodo_doi: 10.5281/zenodo.7992926"
  echo "# paper_doi:  10.1038/s41586-023-06328-6"
  echo "# retrieved:  $(cat retrieval_date.txt)"
  echo "# file  bytes  md5  sha256"
  for f in Processed_K50_dG_datasets.zip Data_tables_for_figs.zip \
           Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
           Single_DMS_list.csv Double_DMS_list.csv Triple_DMS_list.csv \
           dG_extdG_data_Fig1.csv; do
    [ -f "$f" ] || continue
    printf "%s  %s  %s  %s\n" "$f" "$(wc -c < "$f" | tr -d ' ')" \
      "$(md5 -q "$f" 2>/dev/null || md5sum "$f" | cut -d' ' -f1)" \
      "$(shasum -a 256 "$f" | cut -d' ' -f1)"
  done
} > tsuboyama_PROVENANCE.txt
cat tsuboyama_PROVENANCE.txt
