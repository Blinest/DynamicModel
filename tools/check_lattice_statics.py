# -*- coding: utf-8 -*-
"""check_lattice_statics.py — 由菱形几何推等效刚度，并与标定值对比

几何（2026-11-08 定案）：
  外半径 R_o=8 mm、壁厚 t=1.5 mm ⇒ 中面 R_m=7.25 mm
  周向 N_c=3 组、轴向节距 pa=8.33 mm（每段 24 行）
  斜撑角 α = atan(N_c·pa / (2πR_m)) ；斜撑长 L_s = hypot(2πR_m/N_c, pa)
  斜撑截面：宽 t=1.5（径向）、厚取孔径方向的等效宽度 b
  节点圆角 R_f=1.2 mm

三条独立路径求 EI：
  (A) 轴向主导：斜撑只受拉压，  EI = Σ E·A_s·y²·sin²α / L_s   （每单位长度折算）
  (B) 弯曲主导：斜撑当悬臂梁，  EI = Σ 3E·I_s·y² / L_s³
  (C) 代码 cell_stiffness 的晶胞公式
再与标定值 EI=0.01037 N·m² 对比，看哪条对得上。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")
from tendon_coupling import cell_stiffness  # noqa: E402

E_MOD = 2.0e8          # Pa
R_O, T_WALL = 8.0e-3, 1.5e-3
R_M = R_O - T_WALL / 2
N_CIRC, N_LONG = 3, 24
L_SEG = 0.2
PA = L_SEG / N_LONG
EI_CAL = 0.01037


def geo():
    circ = 2 * np.pi * R_M
    pitch_c = circ / N_CIRC
    alpha = np.arctan(N_CIRC * PA / (2 * np.pi * R_M))
    L_s = np.hypot(pitch_c, PA)
    # 斜撑截面：径向宽 = 壁厚；周向厚 = 筋宽 1.5 mm
    b, h = T_WALL, 1.5e-3
    A_s = b * h
    I_s = b * h ** 3 / 12.0
    return dict(circ=circ, pitch_c=pitch_c, alpha=alpha, L_s=L_s,
                A_s=A_s, I_s=I_s, b=b, h=h)


def EI_axial(g):
    """(A) 斜撑轴向主导：把斜撑看作二力杆，只算拉压应变能。"""
    # 弯曲时截面绕中性轴转，斜撑两端沿管轴位移差 ∝ κ·y
    # 每单位轴向长度、每单位曲率下：U = ½ Σ E A_s (Δκ)² / L_s
    # ΔL = κ · y · (投影到轴向的臂长) ；y 取中面半径、对角平均臂长 = pitch_c/2
    n_cell_per_len = N_LONG / L_SEG
    arm = g["pitch_c"] / 2.0
    k = N_CIRC * E_MOD * g["A_s"] / g["L_s"] * R_M ** 2 * np.sin(g["alpha"]) ** 2
    return k * arm * n_cell_per_len


def EI_bending(g):
    """(B) 斜撑弯曲主导：把斜撑当作两端固支的短梁。"""
    n_cell_per_len = N_LONG / L_SEG
    k = N_CIRC * 3 * E_MOD * g["I_s"] / g["L_s"] ** 3 * R_M ** 2
    return k * g["pitch_c"] / 2.0 * n_cell_per_len


def main():
    g = geo()
    cs = cell_stiffness(E=E_MOD, A=g["A_s"], N_circ=N_CIRC, r_i=R_O,
                        alpha_deg=None, L=L_SEG, t_strut=T_WALL, N_long=N_LONG)
    print("=== 几何 ===")
    print(f"  中面半径 R_m      = {R_M*1e3:.3f} mm")
    print(f"  周向节距           = {g['pitch_c']*1e3:.3f} mm")
    print(f"  轴向节距 pa        = {PA*1e3:.3f} mm")
    print(f"  斜撑角 α           = {np.degrees(g['alpha']):.3f}°")
    print(f"  斜撑长 L_s         = {g['L_s']*1e3:.3f} mm")
    print(f"  斜撑截面 A_s       = {g['A_s']*1e6:.3f} mm²   I_s = {g['I_s']*1e12:.4f} mm⁴")
    print()
    ei_a, ei_b = EI_axial(g), EI_bending(g)
    print("=== 等效弯曲刚度 EI (N·m²) ===")
    print(f"  (A) 轴向主导       = {ei_a:.5f}")
    print(f"  (B) 弯曲主导       = {ei_b:.5f}")
    print(f"  (C) cell_stiffness = {cs['EI0']:.5f}")
    print(f"  ★ 标定值           = {EI_CAL:.5f}")
    print()
    for tag, v in (("A", ei_a), ("B", ei_b), ("C", cs["EI0"])):
        print(f"  {tag} 与标定之比     = {v/EI_CAL:6.2f}  "
              f"({'吻合' if 0.75 < v/EI_CAL < 1.33 else '差得远'})")
    print()
    print("=== 轴向 / 扭转 ===")
    print(f"  k_rod (cell)      = {cs['k_rod']:.0f} N   标定 7000 N")
    print(f"  C0 (cell)         = {cs['C0']:.5f} N·m²")
    print(f"  L_strut (cell)    = {cs['L_strut']*1e3:.3f} mm")
    print(f"  贴死量 2L·sinα    = {2*cs['L_strut']*np.sin(np.radians(cs['alpha_deg']))*1e3:.3f} mm"
          f"  vs 壁厚 {T_WALL*1e3} mm ⇒ "
          f"{'贴死' if 2*cs['L_strut']*np.sin(np.radians(cs['alpha_deg']))<=T_WALL else '未贴死'}")
    print()
    print("=== 与无孔管对比 ===")
    I_solid = np.pi / 4 * (R_O ** 4 - (R_O - T_WALL) ** 4)
    print(f"  无孔管 E·I = {E_MOD*I_solid:.4f} N·m²")
    print(f"  开孔后 EI  = {EI_CAL:.5f} N·m²  ⇒ 折减到 {EI_CAL/(E_MOD*I_solid)*100:.2f}%")
    print(f"  折减倍数   = {E_MOD*I_solid/EI_CAL:.1f} 倍")


if __name__ == "__main__":
    main()
