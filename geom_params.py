# -*- coding: utf-8 -*-
"""geom_params.py — **纯几何**单一事实源（不含任何力学）

为什么要单独抽出来
------------------
之前几何常数散在 5 处（`cosserat_lattice` / `anim_bend3d` / `draw_unwrap` /
`fillet_mesh*` / `fig_arm_full`），改一次要改 5 个地方，还容易改漏。

分层约定
--------
    geom_params.py      纯几何（尺寸、孔形、材料率）—— 只依赖 numpy
        ↓  （单向依赖，几何不知道力学）
    cosserat_lattice.py 几何 → 截面刚度 EI/EA/GJ
        ↓  （刚度表这一份"数据契约"）
    tdcr_model_v2.py    力学（力矩驱动 + 积分）—— 只吃刚度，不认识几何

于是：换孔形只改 geom_params；换力学模型只改 tdcr_model_v2；两者可独立验证。
"""
from __future__ import annotations

import numpy as np

# ══════════════════════════════════════════════════════════════════
# 1. 基本尺寸（2026-11-08 定案）
# ══════════════════════════════════════════════════════════════════
R_O = 8.0e-3            # 外半径 8 mm  ⇒ 外径 Φ16
R_I = 6.5e-3            # 内半径 6.5 mm ⇒ 壁厚 1.5
T_WALL = 1.5e-3
N_CIRC = 3              # 周向组数（3 个 60° 周期）

L_TOTAL = 0.400         # 臂长 400 mm
N_SEG_ARM = 2           # 结构分段数（段1/段2）—— 不是"20 段"那个离散段
L1 = 0.200
L2 = 0.200

N_DISK = 20             # 间隔盘个数 = 数值离散段数
NSEG = N_DISK           # 每段 20 mm

E_MOD = 2.0e8           # PEBA 弹性模量 Pa
NU = 0.45
G_MOD = E_MOD / (2.0 * (1.0 + NU))

# ══════════════════════════════════════════════════════════════════
# 2. 孔形 / 筋（两段变体）
# ══════════════════════════════════════════════════════════════════
# α    : 斜撑与周向夹角
# w    : 筋宽 (mm, 物理宽度)
# R_ax : 轴向尖角圆角半径 (mm)
# R_sd : 侧向尖角圆角半径 (mm, 0 = 尖角)
SEG = [
    dict(tag="段1", alpha_deg=34.0, w=2.0e-3, R_ax=5.0, R_sd=0.0),
    dict(tag="段2", alpha_deg=20.0, w=2.0e-3, R_ax=5.0, R_sd=1.0),
]

# 材料率参考点（w=1.5 mm 时的网格统计值），改 w 按 w/1.5 一阶缩放
W_REF = 1.5e-3
F_REF = {"段1": 0.33, "段2": 0.58}

# ══════════════════════════════════════════════════════════════════
# 3. 机构参数（不是几何，但几何脚本也要用）
# ══════════════════════════════════════════════════════════════════
R_DISK = 5.0e-3         # 腱分布半径 ⇒ 分布直径 Φ10
PSI_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)   # 6 根腱方位


def frac(seg):
    """该段材料率（含圆角，按筋宽一阶缩放）。"""
    return float(min(0.95, F_REF[seg["tag"]] * seg["w"] / W_REF))


# ── 派生几何 ─────────────────────────────────────────────────────
def circum_pitch():
    """周向半节距 B = 2πR_o / (2·N_circ)  [m]"""
    return 2.0 * np.pi * R_O / (2.0 * N_CIRC)


def strut_len(alpha_deg):
    """斜撑长 L_strut = B / cosα  [m]"""
    return circum_pitch() / np.cos(np.radians(alpha_deg))


def axial_half_diag(alpha_deg):
    """单胞轴向半对角 a = L_strut·sinα  [m]"""
    return strut_len(alpha_deg) * np.sin(np.radians(alpha_deg))


# ── 实心截面（无孔时）────────────────────────────────────────────
def I_solid():
    return float(np.pi / 4.0 * (R_O ** 4 - R_I ** 4))


def A_solid():
    return float(np.pi * (R_O ** 2 - R_I ** 2))


def J_solid():
    return float(np.pi / 2.0 * (R_O ** 4 - R_I ** 4))


# ── 每段几何汇总（供绘图/刚度计算共用）──────────────────────────
def segments():
    """返回两段的几何字典列表（含材料率 f 与派生量）。"""
    out = []
    for s in SEG:
        d = dict(s)
        d["frac"] = frac(s)
        d["B"] = circum_pitch()
        d["L_strut"] = strut_len(s["alpha_deg"])
        d["a"] = axial_half_diag(s["alpha_deg"])
        out.append(d)
    return out


def seg_index(n_seg=NSEG):
    """20 个数值离散段 → 属于哪一段几何（0 或 1）。"""
    h = n_seg // 2
    return np.array([0] * h + [1] * (n_seg - h))


# ── 与绘图脚本的兼容别名（老脚本用这些名字）──────────────────────
R_OUT_MM = R_O * 1e3
CIRC = 2.0 * np.pi * R_OUT_MM
B = circum_pitch() * 1e3                 # mm
L_SEG = L1 * 1e3                         # mm
L_TOT = L_TOTAL * 1e3                    # mm
H_OFF = B / 2.0

if __name__ == "__main__":
    print("geom_params 自检")
    print(f"  外/内半径 {R_O*1e3:.1f}/{R_I*1e3:.1f} mm  壁厚 {T_WALL*1e3:.1f} mm")
    print(f"  臂长 {L_TOTAL*1e3:.0f} mm  结构分 {N_SEG_ARM} 段  数值 {NSEG} 段 "
          f"（{N_DISK} 个间隔盘）")
    print(f"  B={B:.3f} mm  周向 {N_CIRC} 组")
    for d in segments():
        print(f"  {d['tag']}: α={d['alpha_deg']:.0f}°  w={d['w']*1e3:.1f} mm  "
              f"f={d['frac']:.2f}  L_strut={d['L_strut']*1e3:.2f} mm  "
              f"a={d['a']*1e3:.2f} mm")
    print(f"  I_solid={I_solid():.4e} m⁴  A_solid={A_solid():.4e} m²  "
          f"J_solid={J_solid():.4e} m⁴")
    print(f"  段归属: {seg_index().tolist()}")
