#!/usr/bin/env python3
"""M6-T1：解析极限断言，作为 G4 的准入检查。

与 `probes/` 的关系：那里是数值检验（需要 xtb），这里是**形式约束**
（只需候选泛函本身）。本检查更快且先于数值探针——
一个在解析极限下退化错误的候选量，不值得花机时去跑探针。

### 候选量的接口

候选量声明它实现了哪些因子（`schema/factor-vocabulary.md` 的 F01–F20），
以 dict 形式给出：

    candidate = {
        "F01": lambda **kw: ...,   # 每个分量是关键字函数
        "F02": lambda **kw: ...,
    }

只有被声明的因子才会被检查。声明了却不过，出局；
没声明的因子不检查——但它也就不能用来解释判别集上的表现
（见 `schema/factor-vocabulary.md` 的「签名待建的处理」）。

用法:
  python3 analysis/limit_checks.py            # 跑自测
"""
import math

# 基准构型。断言通过改变其中一个键来取极限。
BASE = dict(
    r=4.0,          # Å，相互作用距离
    theta=180.0,    # deg，D-H...A 或 C-X...A 角
    q1=1.0, q2=-1.0,   # e，形式电荷
    eps=4.0,        # 相对介电常数
    burial=0.5,     # 0..1，埋藏程度
    sigma_p=0.0,    # Hammett sigma_p
    halogen="Cl",
    site_role=-1,   # 口袋位点：+1 供体 / -1 受体
    partner_role=+1,  # 配体伙伴：+1 供体 / -1 受体
)


def cfg(**over):
    d = dict(BASE)
    d.update(over)
    return d


# ---------- 极限行为的提取 ----------

def loglog_slope(fn, key, lo, hi, npts=12, **fixed):
    """在 [lo, hi] 上拟合 log|S| vs log(key) 的斜率。"""
    xs, ys = [], []
    for i in range(npts):
        x = lo * (hi / lo) ** (i / (npts - 1))
        v = fn(**cfg(**{key: x}, **fixed))
        if v is None or v == 0 or not math.isfinite(v):
            continue
        xs.append(math.log(x))
        ys.append(math.log(abs(v)))
    if len(xs) < 4:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = sum((a - mx) ** 2 for a in xs)
    return num / den if den else None


def is_monotone(fn, key, vals, direction, **fixed):
    seq = [fn(**cfg(**{key: v}, **fixed)) for v in vals]
    if any(v is None or not math.isfinite(v) for v in seq):
        return False
    d = [b - a for a, b in zip(seq, seq[1:])]
    return all(x > 1e-12 for x in d) if direction > 0 else all(x < -1e-12 for x in d)


def argextremum(fn, key, vals, **fixed):
    seq = [(v, abs(fn(**cfg(**{key: v}, **fixed)))) for v in vals]
    return max(seq, key=lambda t: t[1])[0]


# ---------- 断言表 ----------
# 每条：(因子, 名称, 检查函数, 失败说明)
# 检查函数返回 (bool, 实测值的字符串)

def _a_coulomb_r(fn):
    s = loglog_slope(fn, "r", 5.0, 40.0)
    return (s is not None and abs(s - (-1.0)) < 0.2), f"log-log 斜率 ={s}"


def _a_coulomb_eps(fn):
    s = loglog_slope(fn, "eps", 2.0, 40.0)
    return (s is not None and abs(s - (-1.0)) < 0.25), f"对 eps 的斜率 ={s}"


def _a_coulomb_sign(fn):
    same = fn(**cfg(q1=1.0, q2=1.0))
    opp = fn(**cfg(q1=1.0, q2=-1.0))
    return (same * opp < 0), f"同号={same:.4g} 异号={opp:.4g}"


def _a_hbond_angle(fn):
    vals = list(range(90, 181, 10))
    peak = argextremum(fn, "theta", vals)
    s180 = abs(fn(**cfg(theta=180.0)))
    s90 = abs(fn(**cfg(theta=90.0)))
    ok = peak >= 170 and s180 > 0 and s90 < 0.25 * s180
    return ok, f"极值在 {peak} deg, |S(90)|/|S(180)| = {s90 / s180 if s180 else float('nan'):.3f}"


def _a_hbond_r(fn):
    s = loglog_slope(fn, "r", 3.0, 12.0)
    return (s is not None and -3.3 < s < -1.7), f"log-log 斜率 ={s}"


def _a_cationpi_far(fn):
    s = loglog_slope(fn, "r", 8.0, 30.0)
    return (s is not None and abs(s - (-3.0)) < 0.4), f"远场斜率 ={s} (应 ~-3, 四极)"


def _a_cationpi_near(fn):
    s = loglog_slope(fn, "r", 2.6, 3.6)
    ok = s is not None and s < -3.5
    return ok, (f"近场斜率 ={s} (应 < -3.5；纯库仑/四极会给 -1 或 -3，"
                f"说明缺了穿透/交换项)")


def _a_cationpi_sigma(fn):
    # 给电子取代增强（更负 sigma_p -> |S| 更大）
    lo = abs(fn(**cfg(sigma_p=-0.8)))
    hi = abs(fn(**cfg(sigma_p=0.8)))
    return (lo > hi), f"|S(sigma=-0.8)|={lo:.4g} > |S(sigma=+0.8)|={hi:.4g}?"


def _a_dispersion_r(fn):
    s = loglog_slope(fn, "r", 4.0, 15.0)
    return (s is not None and abs(s - (-6.0)) < 0.5), f"log-log 斜率 ={s}"


def _a_dispersion_sigma(fn):
    base = abs(fn(**cfg(sigma_p=0.0)))
    spread = max(abs(fn(**cfg(sigma_p=sp))) - base for sp in (-0.8, 0.8))
    rel = abs(spread) / base if base else float("inf")
    return (rel < 0.15), f"对 sigma_p 的相对变化 ={rel:.3f} (应 < 0.15)"


def _a_born_burial(fn):
    return (is_monotone(fn, "burial", [0.1, 0.3, 0.5, 0.7, 0.9], +1),
            "随埋藏单调上升")


def _a_born_q2(fn):
    s1 = fn(**cfg(q1=1.0))
    s2 = fn(**cfg(q1=2.0))
    ratio = s2 / s1 if s1 else float("nan")
    return (abs(ratio - 4.0) < 0.5), f"q 翻倍时比值 ={ratio:.3f} (应 ~4, 电荷平方)"


def _a_born_eps(fn):
    # (1 - 1/eps)：随 eps 上升而单调上升并饱和
    mono = is_monotone(fn, "eps", [1.5, 2.0, 4.0, 10.0, 40.0], +1)
    sat = abs(fn(**cfg(eps=40.0)) - fn(**cfg(eps=80.0)))
    ref = abs(fn(**cfg(eps=40.0)))
    return (mono and sat < 0.05 * ref), f"单调={mono}, 高 eps 饱和度 ={sat / ref if ref else float('nan'):.4f}"


def _a_halogen_angle(fn):
    vals = list(range(120, 181, 10))
    peak = argextremum(fn, "theta", vals)
    return (peak >= 170), f"sigma-空穴极值在 {peak} deg"


def _a_halogen_order(fn):
    vs = {h: abs(fn(**cfg(halogen=h))) for h in ("F", "Cl", "Br", "I")}
    ok = vs["I"] > vs["Br"] > vs["Cl"] > vs["F"] and vs["F"] < 0.3 * vs["Cl"]
    return ok, f"I={vs['I']:.3g} Br={vs['Br']:.3g} Cl={vs['Cl']:.3g} F={vs['F']:.3g}"


def _a_hb_mismatch(fn):
    """互补（供-受）应有利，失配（供-供 / 受-受）应不利，两者反号。

    注意：这条断言的第一版写错了——把「供受体互换」实现为 q1/q2 对调，
    而乘积 q1*q2 不变，断言恒假。角色必须是独立的键，不是电荷。
    记在这里是因为「断言写错」与「被检查对象写错」必须区分（D-014）。
    """
    comp = fn(**cfg(site_role=-1, partner_role=+1))   # 受体 + 供体，互补
    mis_d = fn(**cfg(site_role=+1, partner_role=+1))  # 供体 + 供体，失配
    mis_a = fn(**cfg(site_role=-1, partner_role=-1))  # 受体 + 受体，失配
    ok = comp * mis_d < 0 and comp * mis_a < 0
    return ok, f"互补={comp:.4g} 供供={mis_d:.4g} 受受={mis_a:.4g}"


def _a_hammett_mono(fn):
    vals = [-0.83, -0.27, -0.17, 0.0, 0.06, 0.23, 0.66, 0.78]
    seq = [fn(**cfg(sigma_p=v)) for v in vals]
    d = [b - a for a, b in zip(seq, seq[1:])]
    mono = all(x > 0 for x in d) or all(x < 0 for x in d)
    return mono, f"对 sigma_p 严格单调 ={mono}"


ASSERTIONS = [
    ("F01", "库仑 1/r",            _a_coulomb_r),
    ("F01", "库仑 1/eps",          _a_coulomb_eps),
    ("F01", "异号吸引同号排斥",     _a_coulomb_sign),
    ("F02", "氢键角 180 极值",      _a_hbond_angle),
    ("F02", "氢键距离 -2..-3",     _a_hbond_r),
    ("F03", "供受体互补/失配反号",   _a_hb_mismatch),
    ("F04", "脱水随埋藏单调",       _a_born_burial),
    ("F04", "脱水 ~ q^2",          _a_born_q2),
    ("F04", "脱水 (1-1/eps) 饱和",  _a_born_eps),
    ("F05", "色散 1/r^6",          _a_dispersion_r),
    ("F05", "色散对电性不敏感",      _a_dispersion_sigma),
    ("F06", "阳离子-pi 远场 -3",    _a_cationpi_far),
    ("F06", "阳离子-pi 近场更陡",   _a_cationpi_near),
    ("F06", "给电子取代增强",        _a_cationpi_sigma),
    ("F09", "卤键角 180 极值",      _a_halogen_angle),
    ("F09", "I>Br>Cl>>F",         _a_halogen_order),
    ("F10", "对 sigma_p 单调",      _a_hammett_mono),
]


def check(candidate, verbose=True):
    """candidate: {因子ID: 关键字函数}。只检查被声明的因子。"""
    claimed = set(candidate)
    results = []
    for fac, name, afn in ASSERTIONS:
        if fac not in claimed:
            continue
        try:
            ok, detail = afn(candidate[fac])
        except Exception as e:
            ok, detail = False, f"抛异常: {type(e).__name__}: {e}"
        results.append((fac, name, bool(ok), detail))
    if verbose:
        for fac, name, ok, detail in results:
            print(f"   {'PASS' if ok else 'FAIL'}  {fac} {name:<22} {detail}")
    nfail = sum(1 for *_, ok, _ in [(r[0], r[1], r[2], r[3]) for r in results]
                if not ok)
    unchecked = sorted(claimed - {f for f, _, _ in ASSERTIONS})
    return results, nfail, unchecked


# ---------- 自测：既测能命中，也测不误伤（D-007）----------

def _ref():
    """参考实现：按正确的渐近形式写。应当全部通过。"""
    HAL = {"F": 0.05, "Cl": 1.0, "Br": 1.6, "I": 2.4}
    return {
        "F01": lambda r, q1, q2, eps, **kw: 332.0 * q1 * q2 / (eps * r),
        "F02": lambda r, theta, **kw: (
            -5.0 * (3.0 / r) ** 2.5 * max(0.0, math.cos(math.radians(180 - theta))) ** 3),
        "F03": lambda site_role, partner_role, **kw: (
            1.0 * site_role * partner_role),
        "F04": lambda q1, burial, eps, **kw: (
            q1 ** 2 * burial * (1.0 - 1.0 / eps) * 10.0),
        "F05": lambda r, **kw: -200.0 / r ** 6,
        # 四极项与穿透/交换项相加（第一版写成相乘，远场被指数支配
        # 拿不到 -3 的尾，被 _a_cationpi_far 抓到）
        "F06": lambda r, sigma_p, **kw: (
            -(20.0 / r ** 3) * (1.0 - 0.6 * sigma_p)
            - 15.0 * math.exp(-(r - 2.9) / 0.45) * (1.0 - 0.6 * sigma_p)),
        "F09": lambda theta, halogen, **kw: (
            -HAL[halogen] * max(0.0, math.cos(math.radians(180 - theta))) ** 4),
        "F10": lambda sigma_p, **kw: -1.0 - 1.8 * sigma_p,
    }


def _broken():
    """故意写错的实现。每条错误都应被对应断言抓到。"""
    r = _ref()
    HAL = {"F": 1.0, "Cl": 1.0, "Br": 1.0, "I": 1.0}   # 无卤素序
    return {
        # 错 1：库仑写成 1/r^2
        "F01": lambda r_, q1, q2, eps, **kw: 332.0 * q1 * q2 / (eps * r_ ** 2)
        if False else 332.0 * q1 * q2 / (eps * kw.get("_", 1) if False else eps * r_ ** 2),
        # 错 2：角因子在 90 度取极值（第一版用 cos^2，但它在扫描区间
        # [90,180] 内行为正确——缺陷在 theta=0，不在物理范围内，
        # 所以那不算一个真错误）
        "F02": lambda r_, theta, **kw: -5.0 * (3.0 / r_) ** 2.5 * math.sin(
            math.radians(theta)) ** 2,
        # 错 3：脱水项线性于 q 而非 q^2
        "F04": lambda q1, burial, eps, **kw: abs(q1) * burial * (1 - 1 / eps) * 10.0,
        # 错 4：阳离子-pi 只有库仑/四极，缺穿透项（近场不够陡）
        "F06": lambda r_, sigma_p, **kw: -(20.0 / r_ ** 3) * (1.0 - 0.6 * sigma_p),
        # 错 5：卤键无 I>Br>Cl>>F 序
        "F09": lambda theta, halogen, **kw: -HAL[halogen] * max(
            0.0, math.cos(math.radians(180 - theta))) ** 4,
        # 错 6：互补/失配不反号（只看强度不看角色）
        "F03": lambda site_role, partner_role, **kw: -1.0,
        # 对照：这两条是正确的，不应被误伤
        "F05": r["F05"],
        "F10": r["F10"],
    }


def _rename_r(d):
    """内部辅助：把用 r_ 作参数名的 lambda 适配到 r 关键字。"""
    out = {}
    for k, f in d.items():
        def wrap(f=f):
            def g(**kw):
                if "r" in kw:
                    kw = dict(kw)
                    kw["r_"] = kw.pop("r")
                try:
                    return f(**kw)
                except TypeError:
                    if "r_" in kw:
                        kw = dict(kw)
                        kw["r"] = kw.pop("r_")
                    return f(**kw)
            return g
        out[k] = wrap()
    return out


def main():
    print("=" * 72)
    print("M6-T1 解析极限断言  自测")
    print("=" * 72)

    print(f"\n断言总数 {len(ASSERTIONS)}，覆盖因子 "
          f"{sorted({f for f, _, _ in ASSERTIONS})}")

    print("\n[1] 参考实现（按正确渐近形式写）—— 应全部 PASS")
    res, nfail, unchecked = check(_rename_r(_ref()))
    print(f"   -> {len(res) - nfail}/{len(res)} 通过")
    ref_ok = (nfail == 0)

    print("\n[2] 故意写错的实现 —— 对应断言应 FAIL，正确的两项不应被误伤")
    res2, nfail2, _ = check(_rename_r(_broken()))
    caught = {f"{f}:{n}" for f, n, ok, _ in res2 if not ok}
    f05_f10 = [(f, n, ok) for f, n, ok, _ in res2 if f in ("F05", "F10")]
    no_false = all(ok for *_, ok in f05_f10)
    print(f"   -> 抓到 {nfail2} 条失败")
    print(f"   -> F05/F10（故意写对的）未被误伤: {no_false}")

    print("\n[3] 未被覆盖的因子（声明了也检查不了）")
    missing = sorted({f"F{i:02d}" for i in range(1, 21)}
                     - {f for f, _, _ in ASSERTIONS})
    print(f"   {missing}")
    print("   这些因子既无解析断言也多无数值探针，")
    print("   按 schema/factor-vocabulary.md 不得用于 G4 的可微量实例化。")

    print("\n" + "=" * 72)
    verdict = ref_ok and nfail2 >= 5 and no_false
    print(f"自测结论: {'通过' if verdict else '不通过 —— 断言表本身有问题'}")
    print("  参考实现全过:", ref_ok)
    print("  错误实现被抓 >=5 条:", nfail2 >= 5)
    print("  正确项未被误伤:", no_false)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
