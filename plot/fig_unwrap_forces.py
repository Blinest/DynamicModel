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
# ── 包路径引导（分层目录）────────────────────────────────────
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('geometry', 'stiffness', 'force', 'plot'):
    _p = _os.path.join(_ROOT, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
# ──────────────────────────────────────────────────────────────


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

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                               "Noto Sans CJK SC", "DejaVu Sans"]
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
    alpha = seg["alpha"]
    # 兼容处理：如果 w < 0.1 说明原始数据是米，转成毫米
    w = seg["w"] * 1000.0 if seg["w"] < 0.1 else seg["w"] 
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
                lws.append(1.5 + 4.5 * min(abs(F) / 30.0, 1.0))
    lc = LineCollection(segs, colors=cols, linewidths=lws, capstyle="round",
                        zorder=3)
    ax.add_collection(lc)
    max_y = max([max(p[0][1], p[1][1]) for p in segs]) if segs else WIN
    min_y = min([min(p[0][1], p[1][1]) for p in segs]) if segs else 0
    ax.set_xlim(0, CIRC)
    ax.set_ylim(max(0,min_y), min(WIN,max_y))

     # ── 修改坐标轴样式：隐藏方框，标出 x、y 轴线 ──
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))  # x轴移到 y=0 处
    ax.spines['left'].set_position(('data', 0))    # y轴移到 x=0 处

    # 画箭头 (需要按当前轴比例调整箭头长度)
    arrow_kw = dict(arrowstyle="-|>", color="black", lw=1.0, mutation_scale=12, clip_on=False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    
    # 箭头位置和大小
    # x 轴箭头
    ax.plot([xlim[1] - 0.08 * x_span, xlim[1]], [0, 0], color="black", lw=1.0, 
            clip_on=False, zorder=10)
    ax.plot(xlim[1], 0, marker=">", color="black", markersize=6, 
            clip_on=False, zorder=10)
    
    # y 轴箭头
    ax.plot([0, 0], [ylim[1] - 0.08 * y_span, ylim[1]], color="black", lw=1.0, 
            clip_on=False, zorder=10)
    ax.plot(0, ylim[1], marker="^", color="black", markersize=6, 
            clip_on=False, zorder=10)
    # 调整刻度位置
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # 避免负数刻度穿过文字，将 x 轴最左侧的刻度标签隐藏或偏移（如果需要）
    ax.tick_params(axis='both', direction='in', length=3)

    ax.set_aspect("equal", adjustable="box") # box而非datalim
    ax.set_xticks([0, CIRC / 3, 2 * CIRC / 3, CIRC])
    ax.set_xticklabels(["0", "16.8", "33.5", "50.3"], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("周向展开 (mm)", fontsize=12)
    ax.set_ylabel("轴向 (mm)", fontsize=12)
    eps_max = kappa * R_OUT_MM * 1e-3
    F_max = strut_force(eps_max, alpha, w)
    ax.set_title(f"{title}\nα={alpha}° w={w}mm κ={kappa:.2f} 1/m  "
                f"ε_max={eps_max*100:.1f}%  极限力 ±{abs(F_max):.1f} N",
                fontsize=9, color=RED)
    return abs(F_max)


def main():
    k1, k2 = kappa_example()
    # 每栏的宽高比 = CIRC : WIN
    print(f"示例工况（总弯角 90°）：κ1={k1:.2f}  1/m，κ2={k2:.2f}  1/m")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6))
    fm1 = panel(axes[0], CL.SEG[0], k1, "段1")
    fm2 = panel(axes[1], CL.SEG[1], k2, "段2")
    fig.suptitle("平面展开 + 斜撑受力（红=受拉 · 蓝=受压 · 线宽∝|F|）", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fig_unwrap_forces.png", dpi=300, bbox_inches="tight", transparent = True)
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
