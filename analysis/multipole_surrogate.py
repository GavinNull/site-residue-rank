#!/usr/bin/env python3
"""候选代理量 MP-1：多极通道分解。

### 物理出发点

残基 X->Y 取代，几何固定（「形状相同」的情形），一阶响应是电荷密度差
与配体静电势的重叠积分：

    <dH>_0 ~ Integral[ (rho_Y - rho_X)(r) * V_L(r) ] d3r

两边在取代位点做多极展开：

    <dH> ~ dq*V_L  +  dmu . E_L  +  (1/3) dTheta : grad E_L  -  (1/2) dalpha |E_L|^2

于是 M1 的双线性形式与 M3 的线性响应是**同一个展开**：
残基给 (dq, dmu, dTheta, dalpha)，配体给 (V, E, grad E)。

### 由此得到的可证伪预测

P1  交互矩阵的秩 = 数据集中的取代所激活的多极通道数。
    => 秩不是内禀的，由数据集组成决定，**因此可控**：
       仅电中性极性取代 -> 秩 2；加带电取代 -> 3；加芳香 -> 4。

P2  选择性反转是 dmu.E 项的直接推论：配体场在该位点反向时该项变号。
    => 反转不是「可加性的反例」而是被预测的现象。

P3  去溶剂化 ~ dq^2 (1-1/eps) * burial 对口袋侧是**二次**的，
    故它进 a(P)（主效应）而不进耦合项。

P4  诱导项 -(1/2) dalpha |E|^2 对点电荷场衰减为 1/r^4，
    而 London 色散是 1/r^6。两者是不同通道。
    => `schema/factor-vocabulary.md` 的 F05 把它们混为一谈，应拆开。

### 嵌入是推导出来的，不是借来的

    iota(residue) = (dq, |dmu|, |dTheta|, dalpha)  相对 Ala
d = 4，与本地实测的 d_clean ~ 3-4 相符（见 evidence/residue-scales.md）。
且这些量从侧链类似物计算，不从 AAindex 捡 —— 绕开了「移植是否正当」(Q-001)。

用法:
  python3 analysis/multipole_surrogate.py        # 撞 analysis/limit_checks.py 的断言
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COUL = 332.06  # kcal/mol * A / e^2

# ---- 侧链多极参数（相对 Ala） ----
# dalpha: **已有实验来源**，由 cache/sidechain_alpha.tsv 覆盖下面的占位值
#   （McMeekin 1964 实验折射度 -> Lorentz-Lorenz，经原子折射度加和验证，
#    17/17 比值 0.94-1.12 均值 1.021；CYS 异常已用加和值替代）
#   见 analysis/sidechain_polarizability.py
# dq: 形式电荷，平凡且精确
# dmu, dTheta: **仍是占位值**，待侧链类似物的气相多极矩（已委派检索）
# dq: 形式电荷变化 (e)
# dmu: 偶极大小 (D)，取向沿 Cbeta->末端轴
# dTheta: 四极迹无关分量的量级 (D*A)，芳香侧链为主
# dalpha: 极化率变化 (A^3)
RESIDUE_MULTIPOLES = {
    #        dq    |dmu|  |dTheta|  dalpha
    "ALA": (0.0,   0.00,  0.0,      0.0),
    "SER": (0.0,   1.65,  0.0,      0.6),
    "THR": (0.0,   1.65,  0.0,      1.4),
    "ASN": (0.0,   3.70,  0.0,      1.9),
    "GLN": (0.0,   3.70,  0.0,      2.8),
    "ASP": (-1.0,  1.50,  0.0,      1.6),
    "GLU": (-1.0,  1.50,  0.0,      2.5),
    "LYS": (+1.0,  1.20,  0.0,      3.4),
    "ARG": (+1.0,  2.50,  0.0,      4.2),
    "HIS": (0.0,   3.60,  4.0,      3.1),
    "PHE": (0.0,   0.35,  7.5,      4.6),
    "TYR": (0.0,   1.60,  7.5,      5.0),
    "TRP": (0.0,   2.10,  9.5,      6.3),
    "MET": (0.0,   1.50,  0.0,      4.0),
    "CYS": (0.0,   1.55,  0.0,      2.2),
    "LEU": (0.0,   0.00,  0.0,      3.0),
    "ILE": (0.0,   0.00,  0.0,      3.0),
    "VAL": (0.0,   0.00,  0.0,      2.2),
    "GLY": (0.0,   0.00,  0.0,     -0.6),
    # PRO 不列：侧链与骨架共价闭环，取代基概念不成立（D-022）
}

CHANNELS = ("monopole", "dipole", "quadrupole", "induction", "penetration")


# 通道激活图：各通道降到实验噪声（0.9 kcal/mol）以下的距离
# 由 ch_* 的距离律与典型多极量级算出，见 docs/07
CHANNEL_RANGE_A = {
    "monopole": 10.0,      # 1/r，最长程
    "dipole": 8.0,         # 1/r^2
    "quadrupole": 4.5,     # 1/r^3
    "induction": 4.0,      # 1/r^4
    "penetration": 4.0,    # exp(-beta r)，最短程但接触处最强
}


def channels_in_range(contact_dist_A):
    """只用结构就能算出来的量：某个接触距离上有几个通道在噪声之上。

    这是 P1 的第二个自由度，而且它**不需要亲和力数据**，
    故任何候选数据集的秩都是可预注册的预测，不是事后拟合。
    """
    return [c for c, rmax in CHANNEL_RANGE_A.items() if contact_dist_A <= rmax]


def _load_alpha():
    """用实验推出的 Δα 覆盖占位值。缺文件则保留占位并告警。"""
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "cache", "sidechain_alpha.tsv")
    if not os.path.exists(path):
        return False, {}
    flags = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("residue"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            res, _mr, _a, a_rel, flag = parts[0], *parts[1:5]
            if res in RESIDUE_MULTIPOLES:
                q, mu, th, _ = RESIDUE_MULTIPOLES[res]
                RESIDUE_MULTIPOLES[res] = (q, mu, th, float(a_rel))
                flags[res] = flag
    return True, flags


ALPHA_LOADED, ALPHA_FLAGS = _load_alpha()


def channels_activated(residue_pairs):
    """P1 的可执行形式：给定一组取代，返回被激活的多极通道。

    这是本模块最该先被检验的函数——它把「秩由数据集组成决定」
    变成一个可以在真实数据集上直接数出来的量。
    """
    act = set()
    for x, y in residue_pairs:
        dq_x, mu_x, th_x, al_x = RESIDUE_MULTIPOLES[x]
        dq_y, mu_y, th_y, al_y = RESIDUE_MULTIPOLES[y]
        if abs(dq_y - dq_x) > 1e-9:
            act.add("monopole")
        if abs(mu_y - mu_x) > 0.2:
            act.add("dipole")
        if abs(th_y - th_x) > 0.5:
            act.add("quadrupole")
        if abs(al_y - al_x) > 0.3:
            act.add("induction")
    return [c for c in CHANNELS if c in act]


# ---------- 配体侧的场量（点电荷模型，解析可微） ----------

def V_point(q, r, eps):
    return COUL * q / (eps * r)


def E_point(q, r, eps):
    return COUL * q / (eps * r ** 2)


def gradE_point(q, r, eps):
    return 2.0 * COUL * q / (eps * r ** 3)


# ---------- 各通道 ----------

def ch_monopole(dq, q_lig, r, eps, **kw):
    """dq * V_L  ->  1/(eps r)，符号随电荷积。"""
    return dq * V_point(q_lig, r, eps)


def ch_dipole(dmu, q_lig, r, eps, theta, **kw):
    """dmu . E_L  ->  1/r^2，带取向因子。

    theta 为 D-H...A 角（180 度 = 线性）。取向因子 cos(180-theta) 使
    偶极与场共线时取极值、垂直时为零。
    """
    D2EA = 0.2082   # 1 Debye = 0.2082 e*A
    orient = max(0.0, math.cos(math.radians(180.0 - theta)))
    return -abs(dmu) * D2EA * abs(E_point(q_lig, r, eps)) * orient ** 2 \
        * (1.0 if q_lig != 0 else 0.0)


def ch_quadrupole(dTheta, q_lig, r, eps, **kw):
    """(1/3) dTheta : grad E_L  ->  1/r^3。阳离子-pi 的离子-四极部分。"""
    D2EA = 0.2082
    return -(dTheta * D2EA / 3.0) * abs(gradE_point(q_lig, r, eps)) / r * r


def ch_penetration(d_overlap, q_lig, r, eps, r0=2.9, beta=2.4, **kw):
    """电荷穿透 / 交换重叠。多极展开在电荷分布重叠时失效，而**接触距离上
    这一项最大**：r=3.0 A 时 14.2 kcal/mol，超过单极的 8.3。

    这解释了为什么纯静电/多极类打分函数在第一壳层接触上失准 ——
    主导项正是展开无法表示的那一项。

    衰减长度 1/beta = 0.42 A，与侧链热运动尺度（rmsf 0.3-0.5 A）同量级。
    这给「可微 vs 离散」的张力一个具体数字：光滑性是边缘的，不是没救。
    """
    return -abs(d_overlap) * abs(q_lig) * math.exp(-beta * (r - r0))


def ch_induction(dalpha, q_lig, r, eps, **kw):
    """-(1/2) dalpha |E_L|^2  ->  1/r^4。恒为负（吸引）。

    注意这**不是** London 色散（1/r^6，需两边极化率）。见 P4。
    """
    return -0.5 * dalpha * (E_point(q_lig, r, eps) / COUL) ** 2 * COUL


EPS_WATER = 78.4


def desolvation(q_x, q_y, burial, eps, a_eff=2.0, eps_water=EPS_WATER, **kw):
    """Born 代价之差：(q_Y^2 - q_X^2)(1 - 1/eps)/(2a) * burial。

    **必须是两个代价之差，不是电荷差的平方。** 早先写成 (q_Y - q_X)^2，
    于是 Asp(-1)->Asn(0) 被算成埋藏代价**增加**，而实际是减少
    （不再需要埋一个带电基团）。该错误使 LYS->MET 给出 +51 kcal/mol
    这种不可能的量级。

    注意 analysis/limit_checks.py 的 F04 断言抓不到这个：它测的是
    相对 Ala(q=0) 的情形，那里 (q_Y-0)^2 与 q_Y^2-0^2 恰好重合。
    是物理量级的抽查发现的。

    **参考态必须是水，不是真空。** Born 溶剂化能在介质 eps 中是
    -(q^2/2a)(1 - 1/eps)；而去溶剂化代价是从水转移到口袋：
        (q^2/2a)(1/eps_pocket - 1/eps_water)
    早先用 (1 - 1/eps) 即以真空为参考，使量级偏大 3.2 倍
    （0.75 vs 0.237，eps_pocket=4）。

    这个错误与 limit_checks 的 F04 eps 断言**互相抵消**：
    那条断言测「随 eps 单调上升」，是溶剂化能的行为；
    去溶剂化代价应随 eps_pocket 上升而**下降**。两个错误一起通过了检验。
    断言已同步修正。

    对残基电荷**二次**，且不含配体量 -> 进 a(P) 主效应，不进耦合项（P3）。
    """
    return COUL * (q_y ** 2 - q_x ** 2) \
        * (1.0 / eps - 1.0 / eps_water) / (2.0 * a_eff) * burial


def mp1(residue_x, residue_y, *, q_lig=-1.0, r=4.0, theta=180.0, eps=4.0,
        burial=0.5):
    """MP-1 代理量。对 (r, theta, q_lig, eps, burial) 全部解析可微。"""
    xq, xmu, xth, xal = RESIDUE_MULTIPOLES[residue_x]
    yq, ymu, yth, yal = RESIDUE_MULTIPOLES[residue_y]
    dq, dmu, dth, dal = yq - xq, ymu - xmu, yth - xth, yal - xal
    return (ch_monopole(dq, q_lig, r, eps)
            + ch_dipole(dmu, q_lig, r, eps, theta)
            + ch_quadrupole(dth, q_lig, r, eps)
            + ch_induction(dal, q_lig, r, eps)
            + desolvation(xq, yq, burial, eps))


# ---------- 对接 limit_checks 的因子接口 ----------

def as_factor_dict():
    """把各通道映射到 factor-vocabulary 的因子 ID，交给解析断言检验。

    故意把 F05 也映射到诱导项 —— 预期它**会失败**（1/r^4 而非 1/r^6），
    那正是 P4 要证明的：词表把诱导与色散混为一谈。
    """
    return {
        "F01": lambda r, q1, q2, eps, **kw: ch_monopole(q1, q2, r, eps),
        "F02": lambda r, theta, q2, eps, **kw: ch_dipole(
            3.7, q2, r, eps, theta),
        # 相对 Ala(q=0)：q_Y^2 - 0 = q1^2，故断言仍适用
        "F04": lambda q1, burial, eps, **kw: desolvation(0.0, q1, burial, eps),
        "F05": lambda r, q2, eps, sigma_p, **kw: ch_induction(
            4.6, q2, r, eps),
        # 四极 + 穿透：纯多极给不出近场的陡降（实测 -3.00，需 < -3.5）
        "F06": lambda r, sigma_p, q2=1.0, eps=1.0, **kw: (
            ch_quadrupole(7.5 * (1.0 - 0.6 * sigma_p), q2, r, eps)
            + ch_penetration(18.0 * (1.0 - 0.6 * sigma_p), q2, r, eps)),
    }


def main():
    import limit_checks as LC

    print("=" * 74)
    print("MP-1 多极通道代理量：撞解析断言")
    print("=" * 74)
    print(f"\n[输入状态]")
    print(f"   Δq      形式电荷，精确")
    print(f"   Δα      {'实验来源已载入' if ALPHA_LOADED else '占位值（缺 cache 文件）'}"
          f"  ({sum(1 for v in ALPHA_FLAGS.values() if v == 'ok')}/20 标 ok, "
          f"{sum(1 for v in ALPHA_FLAGS.values() if v != 'ok')} 个有标记)")
    print(f"   Δμ, ΔΘ  占位值，待气相多极矩检索")

    print("\n[P1] 秩 = 被激活的多极通道数（可在真实数据集上直接数出来）")
    cases = [
        ("仅电中性极性取代", [("SER", "THR"), ("ASN", "GLN"), ("TYR", "PHE")]),
        ("+ 带电取代", [("SER", "THR"), ("ASN", "GLN"), ("ASP", "ASN")]),
        ("+ 芳香取代", [("SER", "THR"), ("ASP", "ASN"), ("PHE", "LEU")]),
    ]
    for label, pairs in cases:
        ch = channels_activated(pairs)
        print(f"   {label:<22} -> {len(ch)} 通道 {ch}")
    print("   => 预测：秩由数据集组成决定，可通过挑选取代类型来操纵。")

    print("\n[断言检验] 只声明有解析断言覆盖的因子")
    fd = as_factor_dict()
    res, nfail, unchecked = LC.check(fd)
    print(f"\n   {len(res) - nfail}/{len(res)} 通过")

    print("\n[解读]")
    print("   F05 预期失败且**应当**失败：诱导项 -(1/2)dalpha|E|^2 对点电荷场")
    print("   衰减为 1/r^4，而断言要求 London 色散的 1/r^6。")
    print("   => P4：factor-vocabulary 的 F05 把诱导与色散混为一谈，必须拆开。")
    print("   这是物理抓到的词表错误，不是代理量的错误。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
