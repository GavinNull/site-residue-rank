#!/bin/sh
# Acquisition script for docs/25 (CLAUDE.md §3 provenance: URL + retrieval date + script).
# Source : github.com/nicwulab/CR9114_DMS  @ commit b290a3fcf6a2d24c388e535977d87adbd34db0cc
# Paper  : PLOS Biology DOI 10.1371/journal.pbio.3004021 (preprint 10.1101/2025.11.03.686356, PMC12637573)
# Date   : 2026-10-12
# Raw SRA reads (PRJNA1284397) are deliberately NOT fetched.
set -e
cd "$(dirname "$0")/.."
mkdir -p cache/dms/cr9114
SHA=b290a3fcf6a2d24c388e535977d87adbd34db0cc
BASE="https://raw.githubusercontent.com/nicwulab/CR9114_DMS/$SHA"
for f in \
  result/HC_GL_KD_table_summary_kabat.csv \
  result/HC_WT_H1_KD_table_summary_kabat.csv \
  result/HC_WT_H3_KD_table_summary_kabat.csv \
  result/HC_WT_fluB_KD_table_summary_kabat.csv \
  result/HC_WT_expression_score_kabat.csv \
  result/HC_GL_KD_table_summary.csv \
  result/CR9114HC_WT_kabat_H.csv \
  result/CR9114HC_GL_kabat_H.csv \
  result/KD_compare_BLI_DMS.tsv \
  result/WT_H3/HC_Tite_seq_plot_H3.Rmd \
  Fasta/CR9114_HC_WT.pep Fasta/CR9114_HC_GL.pep \
  Fasta/CR9114_HC_WT.fa  Fasta/CR9114_HC_GL.fa \
  data/CR9114_SHMheatmap.tsv data/CR9114_mut_neut.tsv \
  README.md script/compile_TiteSeq.py ; do
  out="cache/dms/cr9114/cr9114_$(echo "$f" | tr '/' '_')"
  curl -sL --max-time 300 "$BASE/$f" -o "$out"
done
shasum -a 256 cache/dms/cr9114/cr9114_* > cache/dms/cr9114/cr9114_SHA256SUMS.txt
