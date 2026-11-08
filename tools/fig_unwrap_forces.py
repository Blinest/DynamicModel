# -*- coding: utf-8 -*-
"""fig_unwrap_forces.py — 平面展开 + 每根斜撑的受力

力的算法（膜应变 → 斜撑轴力）
------------------------------
弯曲曲率 κ(s)，方位 φ_b=0（弯曲平面过 θ=0）时，周向位置 θ 处管壁的轴向应变
        ε(θ, s) = κ(s) · R · cos(θ − φ_b)
斜撑与周向成 α 角 ⇒ 它对轴向应变的响应按 sinα 折减：
        ε_strut = ε · sinα
        F = E · ε_strut · A_strut        （A_strut = 筋宽 w × 壁厚 t）
    ε>0 受拉（红），ε<0 受压（蓝）；线宽 ∝ |F|

输出 fig_unwrap_forces.png：两段各一栏，60 mm 窗口，标出最大压/拉
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cosserat_lattice as CL   # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_OUT_MM = 8.0
CIRC = 2 * np.pi * R_OUT_MM
B = CIRC / 6.0
T_WALL_MM = 1.5
WIN = 60.0
THETA_MIN = 0.6 * np.pi / 2      # 取 90° 弯角做示例
KAPPA = None


def kappa_example():
    """示例工况：段1/段2 各取 κ，使总弯角约 90°：κ1·0.2 + κ2·0.2 = π/2。"""
    EI = [k["EI"] for k in CL.stiffness()]
    # 段2 力矩 = 段1 的 1/2 ⇒ κ2 = 0.5·EI1/EI2·κ1
    r = 0.5 * EI[0] / EI[1]
    k1 = (np.pi / 2) / (0.2 + 0.2 * r)
    return k1, k1 * r


def strut_force(eps, alpha_deg, w_mm):
    """斜撑轴力 (N)：F = E · (ε·sinα) · (w·t)。"""
    A = (w_mm * 1e-3) * (T_WALL_MM * 1e-3)
    return CL.E_MOD * (eps * np.sin(np.radians(alpha_deg))) * A


def panel(ax, seg, kappa, title):
    alpha, w = seg["alpha"], seg["w"]
    Ls = B / np.cos(np.radians(alpha))
    a = Ls * np.sin(np.radians(alpha))
    pa = 2 * a
    segs, cols, lws = [], [], []
    Fmax = 0.0
    n_row = int(WIN / pa) + 2
    n_col = int(CIRC / B) + 2
    for n in range(-2, n_row):
        for m in range(-2, n_col):
            if (m - n) % 2:
                continue
            p0 = np.array([m * B, n * a])
            for dm in (+1, -1):
                p1 = np.array([(m + dm) * B, (n + 1) * a])
                ym = 0.5 * (p0[1] + p1[1])
                if ym < -pa or ym > WIN + pa:
                    continue
                xm = 0.5 * (p0[0] + p1[0]) % CIRC
                th = xm / R_OUT_MM
                eps = kappa * R_OUT_MM * 1e-3 * np.cos(th)
                F = strut_force(eps, alpha, w)
                Fmax = max(Fmax, abs(F))
                segs.append([p0, p1])
                cols.append(RED if F > 0 else BLUE)
                lws.append(0.6 + 3.2 * min(abs(F) / 30.0, 1.0))
    lc = LineCollection(segs, colors=cols, linewidths=lws, capstyle="round",
                        zorder=3)
    ax.add_collection(lc)
    ax.set_xlim(0, CIRC)
    ax.set_ylim(0, WIN)
    ax.set_aspect("equal")
    ax.set_xticks([0, CIRC / 3, 2 * CIRC / 3, CIRC])
    ax.set_xticklabels(["0", "16.8", "33.5", "50.3"], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("周向展开 (mm)   θ=0 为受拉侧、θ=180° 为受压侧", fontsize=8)
    ax.set_ylabel("轴向 (mm)", fontsize=8)
    eps_max = kappa * R_OUT_MM * 1e-3
    F_max = strut_force(eps_max, alpha, w)
    ax.set_title(f"{title}\nα={alpha}° w={w}mm  κ={kappa:.2f} 1/m  "
                 f"ε_max={eps_max*100:.1f}%  极限力 ±{abs(F_max):.1f} N",
                 fontsize=9, color=RED)
    return abs(F_max)


def main():
    k1, k2 = kappa_example()
    print(f"示例工况（总弯角 90°）：κ1={k1:.2f}  1/m，κ2={k2:.2f}  1/m")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fm1 = panel(axes[0], CL.SEG[0], k1, "① 段1（α=34°, w=2.0）")
    fm2 = panel(axes[1], CL.SEG[1], k2, "② 段2（α=20°, w=2.0）")
    fig.suptitle("平面展开 + 斜撑受力（红=受拉 · 蓝=受压 · 线宽∝|F|）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_unwrap_forces.png", dpi=165, bbox_inches="tight")
    print("saved fig_unwrap_forces.png")
    print(f"\n极限位置（θ=0 最外层受拉 / θ=180° 最内层受压）：")
    print(f"  段1: ε={k1*R_OUT_MM*1e-3*100:.2f}%  F=±{fm1:.1f} N")
    print(f"  段2: ε={k2*R_OUT_MM*1e-3*100:.2f}%  F=±{fm2:.1f} N")
    print(f"  中部（θ=90°）受力为 0")
    # 与腱张力对照
    print(f"\n参照：驱动丝张力 T0=10 N、最大差动 A≈9.5 N ⇒ 单丝 0.5–19.5 N")
    print(f"  段1 极限轴力 {fm1:.1f} N 与之同量级 ⇒ 局部受压不可忽略")


if __name__ == "__main__":
    main()
