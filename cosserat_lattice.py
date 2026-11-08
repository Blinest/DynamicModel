# -*- coding: utf-8 -*-
"""cosserat_lattice.py — **几何 → 截面刚度**（与力学解耦）

本文件只做一件事：把 `geom_params` 的几何，换成每段的 `EI / EA / GJ`。
不涉及形状积分、不涉及腱、不涉及求解器。

    geom_params.py  ──几何──▶  cosserat_lattice.py  ──刚度──▶  tdcr_model_v2.py

推导
----
1. 无孔实心管：`I_solid = π/4(R_o⁴−R_i⁴)`、`A_solid = π(R_o²−R_i²)`
2. 材料率 `f`（含圆角，网格统计）：段1 0.44 / 段2 0.77（`geom_params.frac`）
3. 取向因子：斜撑相对管轴倾角 α ⇒ 弯曲贡献 ∝ `sin²α·cos²α`
       α=34°: 0.215 ； α=20°: 0.104
4. 于是
       EI = c · E · I_solid · f · sin²α cos²α
       EA = c_E · E · A_solid · f
       GJ = c_G · G · J_solid · f
   `c / c_E / c_G` 为标定常数；`c` 在 w=1.5 mm 时由实测 `EI=0.01037` 标定后**固定**
   （工艺常数，不随筋宽重标，否则加粗/减薄的效果会被抵消）。
"""
from __future__ import annotations

import numpy as np

import geom_params as G

# ── 标定常数（唯一自由度）────────────────────────────────────────
C_BEND = 0.4367        # 弯曲：w=1.5mm 时由实测 EI 标定，固定
C_EA = 1.0
C_GJ = 1.0
EI_MEAS = 0.01037      # N·m²  实测反标值（仅用于 calibrate_c 参考）

# ── 向后兼容别名（老脚本从本模块 import 这些名字）────────────────
R_O, R_I = G.R_O, G.R_I
T_WALL = G.T_WALL
W_STRUT = 1.5e-3
N_CIRC = G.N_CIRC
E_MOD = G.E_MOD
NU = G.NU
G_MOD = G.G_MOD
L, L1 = G.L_TOTAL, G.L1
W_REF, F_REF = G.W_REF, G.F_REF
SEG = G.SEG


def I_solid():
    return G.I_solid()


def A_solid():
    return G.A_solid()


def J_solid():
    return G.J_solid()


def orient(alpha_deg):
    """取向因子 sin²α·cos²α。"""
    a = np.radians(alpha_deg)
    return float(np.sin(a) ** 2 * np.cos(a) ** 2)


def EI_raw(alpha_deg, frac):
    """未标定的弯曲刚度（c=1）。"""
    return E_MOD * I_solid() * frac * orient(alpha_deg)


def calibrate_c(w_ref_mm=1.5):
    """**参考用**：在 w=w_ref 时由实测 EI 反解 c。"""
    raws = []
    for s in G.SEG:
        f = G.F_REF[s["tag"]] * w_ref_mm / (G.W_REF * 1e3)
        f = G.frac(s) * (w_ref_mm * 1e-3) / s["w"]
        raws.append(EI_raw(s["alpha_deg"], f))
    return EI_MEAS / float(np.mean(raws))


def stiffness(c=None):
    """返回两段的 `[{EI, EA, GJ, alpha, frac, tag}]`。"""
    c = C_BEND if c is None else float(c)
    out = []
    for s in G.segments():
        EI = c * EI_raw(s["alpha_deg"], s["frac"])
        EA = C_EA * E_MOD * A_solid() * s["frac"]
        GJ = C_GJ * G_MOD * J_solid() * s["frac"]
        out.append(dict(EI=EI, EA=EA, GJ=GJ, alpha=s["alpha_deg"],
                        frac=s["frac"], tag=s["tag"]))
    return out


def arrays(n_seg=None):
    """★ 数据契约：把刚度铺到 n_seg 个数值段上。

    返回 `dict(EI=(n,), EA=(n,), GJ=(n,), L=..., NSEG=...)`
    —— 力学层只吃这个，不认识几何。
    """
    n = G.NSEG if n_seg is None else int(n_seg)
    ks = stiffness()
    idx = G.seg_index(n)
    EI = np.array([ks[i]["EI"] for i in idx])
    EA = np.array([ks[i]["EA"] for i in idx])
    GJ = np.array([ks[i]["GJ"] for i in idx])
    return dict(EI=EI, EA=EA, GJ=GJ, L=G.L_TOTAL, NSEG=n)


if __name__ == "__main__":
    print("=== 由几何推刚度（几何来自 geom_params）===")
    print(f"  I_solid={I_solid():.4e} m⁴  A_solid={A_solid():.4e} m²  "
          f"J_solid={J_solid():.4e} m⁴")
    print(f"  E·I_solid = {E_MOD*I_solid():.4f} N·m²（无孔管）")
    print(f"  标定常数 c = {C_BEND:.4f}（w=1.5mm 时由实测 EI={EI_MEAS} 标定，固定）\n")
    print("%4s %6s %6s %9s %10s %10s %10s" %
          ("段", "α°", "材料率", "取向因子", "EI", "EA", "GJ"))
    for k in stiffness():
        print("%4s %6.1f %6.2f %9.4f %10.5f %10.0f %10.5f" %
              (k["tag"], k["alpha"], k["frac"], orient(k["alpha"]),
               k["EI"], k["EA"], k["GJ"]))
    a = arrays()
    print(f"\n数据契约 arrays(): EI 形状 {a['EI'].shape}  "
          f"前 {a['EI'][0]:.5f} / 后 {a['EI'][-1]:.5f} N·m²")
    print(f"  无孔管折减 {E_MOD*I_solid()/a['EI'].mean():.1f} 倍")
