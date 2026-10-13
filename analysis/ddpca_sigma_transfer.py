#!/usr/bin/env python3
"""docs/28 §4.5：把**在 GB1/folding 上实测到的**主效应吸收比例搬到其余矩阵。

为什么要单列成一个脚本而不是写进 §4.2 的表：
位点层 ICC 只在**有重复位点**的体外对照上可估，而六个矩阵里只有两个
有（GB1/folding 的 Olson2014：10 个重复位点 / 60 个变体；
PSD95-PDZ3/folding 的 Calosci2008：只有 4 个重复位点 / 8 个变体）。
**三个 binding 矩阵一个都没有**（Chi2008 n = 11、Malagrino2019 n = 12，
每个位点一个变体；GB1/binding 连体外对照都没有）。

所以 B' 对 binding 一侧**不是测量**，只能是一次**搬运**。
本脚本把搬运单列、标明它是搬运，并把两个可测值都用一遍（0.441 与 0.853），
让读者看见搬运带来的跨度，而不是看见一个数。
**这不是本文档的判决依据**（判决见 §5 的直接测量与 §6 的闸判），
只是「如果按 docs/27 的机制纠正 σ，数会变成什么」。
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subspace_lib import ROOT                                  # noqa: E402

SRC = os.path.join(ROOT, "cache", "ddpca_sigma.json")
OUT = os.path.join(ROOT, "cache", "ddpca_sigma_transfer.json")
# 实测的两个吸收比例（§4.2），全部来自本仓库自己的数据
MEASURED = {"GB1/folding (Olson2014, 10 重复位点/60 变体, p=0.0005)": 0.441,
            "PSD95-PDZ3/folding (Calosci2008, 4 重复位点/8 变体, p=0.0175)":
                0.853}


def main():
    d = json.load(open(SRC))
    rows = []
    print("=" * 100)
    print("docs/28 §4.5：主效应吸收比例的**搬运**（不是测量）")
    print("=" * 100)
    for label, exc in MEASURED.items():
        f = math.sqrt(1 - exc)
        print(f"\n[搬运 excess = {exc:.3f}（来源：{label}）"
              f" => σ' = σ_B × {f:.3f}]")
        print(f"   {'矩阵':<22}{'coup_obs':>9}{'σ_B':>7}{'σ_Bp':>7}"
              f"{'coup_true':>10}{'= nσ':>7}{'判据1.1σ':>9}")
        for r in d["rows"]:
            cv = r["conventions"]["docs15"]
            coup = cv["coup_sd_obs"]
            sB = cv["sigma"]["B"]
            s = sB * f
            ct = math.sqrt(max(coup ** 2 - s ** 2, 0.0))
            snr = ct / s
            print(f"   {r['name']:<22}{coup:>9.3f}{sB:>7.2f}{s:>7.3f}"
                  f"{ct:>10.3f}{snr:>7.2f}"
                  f"{'过' if snr >= 1.1 else '不过':>9}")
            rows.append({"transfer_from": label, "excess": exc,
                         "name": r["name"], "coup_sd_obs": coup,
                         "sigma_B": sB, "sigma_Bp_transferred": s,
                         "coup_sd_true": ct, "snr": snr,
                         "pass_1p1sigma": bool(snr >= 1.1)})
    json.dump({"rows": rows, "measured": MEASURED},
              open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
    print("\n产物:", OUT)
    print("\n**读法**：这张表回答「若 docs/27 的机制按本仓库实测的幅度纠正，"
          "docs/15 §2.2 的三个 0 会变成什么」。它**不**回答「这些矩阵有没有"
          "可定秩的信号」—— 那一问由 §5 的三次独立生物学重复与 §6 的"
          "D-036 / 可分辨闸回答，而它们的答案与 σ 口径无关。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
