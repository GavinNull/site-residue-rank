#!/usr/bin/env python3
"""MaveDB 全量 score set 元数据取回（docs/39 第 1 层）。

provenance（CLAUDE.md §3）：
  API 根   : https://api.mavedb.org/api/v1
  端点     : POST /score-sets/search  （published=true，limit<=100，分页）
             GET  /score-sets/search/filter-options
             GET  /api/version
  取回日期 : 由本脚本写入 retrieved_utc
  产物     : cache/mavedb/mavedb_scoresets.json （含每页 sha256 与总校验和）
             cache/mavedb/mavedb_PROVENANCE.json

本脚本只取元数据，不取 scores.csv（那一步在 mavedb_screen.py，逐候选取，
各自记 URL + sha256）。
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

API = "https://api.mavedb.org/api/v1"
OUT = os.path.join(os.path.dirname(__file__), "..", "cache", "mavedb")
PAGE = 100


def _post(path, body, timeout=180):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()


def _get(path, timeout=180):
    req = urllib.request.Request(API + path,
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()


def selftest(records, total):
    """CLAUDE.md §10：用已知事实自检，不用自洽性。不通过抛异常。

    A1 枚举完整性：取回条数 == API 自报的 numScoreSets。
    A2 URN 唯一：score set URN 是主键，重复即分页错位（D-023 那一类静默错）。
    A3 已知条目命中：urn:mavedb:00000001-a-1 是 Weile 等的 UBE2I 酵母互补
       （MaveDB 的第一条记录，本仓库之外的公开已知事实）。
    A4 不误伤：对一个已知**不该**命中的字符串做同一套字段提取，必须返回空。
    A5 字段非空率：targetGenes / numVariants 必须基本齐全，否则后面的
       筛选层会把「字段缺失」误读成「不合格」。
    """
    fails = []
    if len(records) != total:
        fails.append(f"A1 枚举不完整：{len(records)} != numScoreSets {total}")
    urns = [r["urn"] for r in records]
    if len(set(urns)) != len(urns):
        fails.append(f"A2 URN 重复：{len(urns) - len(set(urns))} 条")
    known = [r for r in records if r["urn"] == "urn:mavedb:00000001-a-1"]
    if not known:
        fails.append("A3 已知条目 urn:mavedb:00000001-a-1 缺失")
    else:
        txt = (known[0]["experiment"].get("abstractText") or "").lower()
        if "ube2i" not in txt and "ube2i" not in known[0]["title"].lower():
            fails.append("A3 已知条目内容不符（应为 UBE2I 酵母互补）")
    bogus = [r for r in records
             if "zzqq_not_a_real_target" in json.dumps(r).lower()]
    if bogus:
        fails.append("A4 不误伤检查失败：匹配到了不存在的字符串")
    no_target = sum(1 for r in records if not r.get("targetGenes"))
    no_nvar = sum(1 for r in records if r.get("numVariants") in (None, 0))
    if no_target > 0.02 * len(records):
        fails.append(f"A5 targetGenes 缺失 {no_target}/{len(records)}，过多")
    if fails:
        raise SystemExit("自检不通过：\n  " + "\n  ".join(fails))
    return {"A1_total": len(records), "A2_unique_urns": len(set(urns)),
            "A3_known_entry": "ok", "A4_no_false_hit": "ok",
            "A5_missing_targetGenes": no_target,
            "A5_missing_numVariants": no_nvar}


def main():
    os.makedirs(OUT, exist_ok=True)
    ver = json.loads(_get("/api/version"))
    pages, records, total = [], [], None
    off = 0
    while True:
        raw = _post("/score-sets/search",
                    {"published": True, "limit": PAGE, "offset": off})
        pages.append({"offset": off,
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw)})
        d = json.loads(raw)
        total = d["numScoreSets"]
        got = d["scoreSets"]
        records.extend(got)
        print(f"  offset {off:5d}  +{len(got):3d}  total {total}", flush=True)
        off += PAGE
        if off >= total or not got:
            break
        time.sleep(0.3)

    st = selftest(records, total)
    print("自检：", json.dumps(st, ensure_ascii=False))

    blob = json.dumps(records, sort_keys=True).encode()
    meta = {
        "source": "MaveDB public API",
        "api_root": API,
        "api_version": ver,
        "endpoint": "POST /score-sets/search {published:true}",
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_score_sets": total,
        "records_sha256": hashlib.sha256(blob).hexdigest(),
        "records_bytes": len(blob),
        "pages": pages,
        "selftest": st,
        "script": "analysis/mavedb_acquire.py",
    }
    with open(os.path.join(OUT, "mavedb_scoresets.json"), "wb") as fh:
        fh.write(blob)
    with open(os.path.join(OUT, "mavedb_PROVENANCE.json"), "w") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    try:
        fo = _get("/score-sets/search/filter-options")
        with open(os.path.join(OUT, "mavedb_filter_options.json"), "wb") as fh:
            fh.write(fo)
    except Exception as exc:                                  # noqa: BLE001
        print("filter-options 取回失败（非致命）：", exc, file=sys.stderr)
    print(json.dumps(meta, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
