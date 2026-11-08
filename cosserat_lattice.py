# -*- coding: utf-8 -*-
"""cosserat_lattice.py — 由**菱形晶格几何**推截面刚度，再交给 Cosserat 杆求解

与旧路径的差别
--------------
旧路径：EI 直接由实验反标（0.01037），晶格只通过 `sinc(κ·L_strut)` 修正。
新路径：**先由几何算材料率与取向因子，得到 EI/EA/GJ 的解析表达**，
        再用一个标定常数 c（只一个）把它锚到实测。

几何输入（2026-11-08 定案）
    外半径 R_o=8mm、内半径 R_i=6.5mm、壁厚 t=1.5mm
    周向 3 组、双向（正交 2 族）菱形孔
    段1 α=34°、轴向圆角 R=5 / 侧向尖；段2 α=20°、轴向 R=5 / 侧向 R=1
    筋宽 w=1.5mm

刚度推导
--------
1. 无孔实心管：I_solid = π/4 (R_o⁴ − R_i⁴)，A_solid = π (R_o² − R_i²)
2. 材料率（含圆角，实测网格化）：
       段1 f≈0.33  段2 f≈0.58   （两族孔 + 圆角后的像素/网格统计）
3. 取向因子：斜撑相对管轴的倾角 α ⇒ 弯曲贡献 ∝ sin²α·cos²α
       （α=34°: 0.215；α=20°: 0.104）
4. 于是
       EI = c · E · I_solid · f · sin²α cos²α
       EA = c_E · E · A_solid · f
       GJ = c_G · G · J_solid · f        （J_solid = π/2 (R_o⁴ − R_i⁴)）
   c/c_E/c_G 为标定常数，默认由实测 EI=0.01037 反解 c。

Cosserat 平衡（与 tendon_coupling 一致）
    n' + f = 0,   m' + p'×n + l = 0
    n = K_se (v − e3),  m = K_bt κ,  K_se=diag(EA,GA,GA), K_bt=diag(EI,EI,GJ)
    腱：力臂 r_j = R·o_j，力矩 Σ r_j × (τ_j p̂_j)，摩擦 T(s)=T0·exp(−μ·φ(s))
"""
import numpy as np

# ── 几何 ─────────────────────────────────────────────────────────
R_O, R_I = 8.0e-3, 6.5e-3
T_WALL = 1.5e-3
W_STRUT = 1.5e-3
N_CIRC = 3
E_MOD = 2.0e8
NU = 0.45
G_MOD = E_MOD / (2 * (1 + NU))
L = 0.4
L1 = 0.2

# 两段变体：(α, 筋宽 w, 材料率 f)
# 材料率的参考点是 w=1.5mm 时的网格统计值；改 w 时按 w/1.5 一阶缩放
W_REF, F_REF = 1.5, dict(seg1=0.33, seg2=0.58)
SEG = [dict(alpha=34.0, w=2.0, tag="段1"),
       dict(alpha=20.0, w=2.0, tag="段2")]


def frac_of(seg, key):
    """材料率随筋宽一阶缩放（参考 w=1.5mm 的网格统计值）。"""
    return min(0.95, F_REF[key] * seg["w"] / W_REF)


for _k, _s in (("seg1", SEG[0]), ("seg2", SEG[1])):
    _s["frac"] = frac_of(_s, _k)

# 标定常数（c 由实测 EI 反解，见 calibrate_c()）
C_BEND = 0.4367        # 在 w=1.5mm 时由实测 EI 标定；属于工艺常数，不随 w 重标
C_EA = 1.0
C_GJ = 1.0

EI_MEAS = 0.01037        # N·m²  实测反标值


def I_solid():
    return np.pi / 4 * (R_O ** 4 - R_I ** 4)


def A_solid():
    return np.pi * (R_O ** 2 - R_I ** 2)


def J_solid():
    return np.pi / 2 * (R_O ** 4 - R_I ** 4)


def orient(alpha_deg):
    a = np.radians(alpha_deg)
    return np.sin(a) ** 2 * np.cos(a) ** 2


def EI_raw(alpha_deg, frac):
    """未标定的弯曲刚度（c=1）。"""
    return E_MOD * I_solid() * frac * orient(alpha_deg)


def calibrate_c(w_ref=1.5):
    """**参考用**：在 w=w_ref 时由实测 EI 反解 c（工艺常数，之后固定不再重标）。"""
    raws = []
    for s, key in ((SEG[0], "seg1"), (SEG[1], "seg2")):
        f = F_REF[key] * w_ref / W_REF
        raws.append(EI_raw(s["alpha"], f))
    return EI_MEAS / np.mean(raws)


def stiffness(seg=None, c=None):
    """返回 (EI, EA, GJ)，两段可分开算。"""
    c = C_BEND if c is None else c
    out = []
    for s in (SEG if seg is None else [seg]):
        EI = c * EI_raw(s["alpha"], s["frac"])
        EA = C_EA * E_MOD * A_solid() * s["frac"]
        GJ = C_GJ * G_MOD * J_solid() * s["frac"]
        out.append(dict(EI=EI, EA=EA, GJ=GJ, alpha=s["alpha"], frac=s["frac"]))
    return out if seg is None else out[0]


def kappa_from_moment(M, EI):
    return M / EI


def bend_angle_from_dl(dl_ratio_theta, seg_K):
    """给定各段转角需求，返回需要的端部矩（小角度近似，仅用于量级校核）。"""
    k1 = seg_K[0]["EI"]
    k2 = seg_K[1]["EI"]
    # 段2 κ = M2/EI2，段1 κ = M1/EI1；两段力矩按 2:1（6 丝 vs 3 丝）
    # 力矩量级取实际值：180° 需要 M ≈ EI·κ ≈ 0.0104×(π/0.3) ≈ 0.11 N·m
    M2 = 0.050
    M1 = 0.100
    th = M1 / k1 * L1 + M2 / k2 * (L - L1)
    return th, M1, M2


if __name__ == "__main__":
    c = C_BEND
    print("=== 由几何推刚度 ===")
    print(f"  I_solid={I_solid():.4e} m⁴  A_solid={A_solid():.4e} m²  "
          f"J_solid={J_solid():.4e} m⁴")
    print(f"  E·I_solid = {E_MOD*I_solid():.4f} N·m²（无孔管）")
    print(f"  标定常数 c = {c:.4f}（在 w=1.5mm 时由实测 EI={EI_MEAS} 标定，固定）")
    print(f"  参考：若在 w=1.5 重标，c 应为 {calibrate_c(1.5):.4f}\n")
    print("%4s %6s %6s %9s %10s %10s %10s" %
          ("段", "α°", "材料率", "取向因子", "EI", "EA", "GJ"))
    for s, k in zip(SEG, stiffness()):
        print("%4s %6.1f %6.2f %9.4f %10.5f %10.0f %10.5f" %
              (s["tag"], s["alpha"], s["frac"], orient(s["alpha"]),
               k["EI"], k["EA"], k["GJ"]))
    print("\n=== 与无孔管/实测对比 ===")
    print(f"  无孔管 E·I = {E_MOD*I_solid():.4f}，开孔后平均 EI = "
          f"{np.mean([k['EI'] for k in stiffness()]):.5f} "
          f"⇒ 折减 {E_MOD*I_solid()/np.mean([k['EI'] for k in stiffness()]):.1f} 倍")
    # 小角度校核：总弯矩 M 下的弯角
    ks = stiffness()
    th, M1, M2 = bend_angle_from_dl(0, ks)
    print(f"\n=== 力矩→弯角（M1=0.10 N·m, M2=0.05 N·m，两段 2:1）===")
    print(f"  κ1={M1/ks[0]['EI']:.2f} κ2={M2/ks[1]['EI']:.2f} 1/m")
    print(f"  总弯角 {np.degrees(th):.1f}°（两段曲率比 "
          f"{(M1/ks[0]['EI'])/(M2/ks[1]['EI']):.2f}）")
