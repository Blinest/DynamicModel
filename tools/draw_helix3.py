# -*- coding: utf-8 -*-
"""draw_helix3.py — 双向螺旋 × 周向 3 组，按轴向节距分三档

构造：展开面上两族直线
    y = ±(3·pa/CIRC)·x + k·pa
  ⇒ 沿周向走 CIRC/3 时轴向上升 pa，即周向 3 个菱形、轴向每 pa 一行。
  ⇒ 斜撑角  tanα = 3·pa/CIRC
     斜撑长  L_strut = hypot(CIRC/3, pa)
     行数    N_long = L_seg / pa

三档：pa = 1.47 mm（当前模型，N=136）/ 5 mm（N=40）/ 10 mm（N=20）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_O, N_CIRC, L_SEG = 8.0, 3, 200.0
CIRC = 2 * np.pi * R_O
YMAX = 22.0


def panel(ax, pa, hl=False):
    m = N_CIRC * pa / CIRC                     # 斜率
    alpha = np.degrees(np.arctan(m))
    L = np.hypot(CIRC / N_CIRC, pa)
    n_long = L_SEG / pa
    ax.add_patch(Polygon([(0, -YMAX), (CIRC, -YMAX), (CIRC, YMAX), (0, YMAX)],
                         facecolor=BLUE, alpha=0.35, zorder=0))
    k0 = int(-YMAX / pa) - 1
    k1 = int(YMAX / pa) + 2
    for k in range(k0, k1):
        c = k * pa
        if hl:
            ax.plot([0, CIRC], [c, c + m * CIRC], color=RED, lw=2.0, zorder=3)
            ax.plot([0, CIRC], [c + m * CIRC, c], color=GREEN, lw=2.0, zorder=3)
        else:
            ax.plot([0, CIRC], [c, c + m * CIRC], color=RED, lw=2.0, zorder=3)
            ax.plot([0, CIRC], [c + m * CIRC, c], color=RED, lw=2.0, zorder=3)
    ax.set_xlim(0, CIRC)
    ax.set_ylim(-YMAX, YMAX)
    ax.set_aspect("equal")
    ax.grid(alpha=0.12, ls=":")
    ax.tick_params(labelsize=6)
    ax.set_title(f"轴向节距 pa = {pa:g} mm  ⇒  α = {alpha:.2f}°\n"
                 f"L_strut = {L:.2f} mm   行数/段 = {n_long:.0f}"
                 f"{'   （当前模型）' if pa == 1.47 else ''}",
                 fontsize=8.5, color=RED if pa == 8.33 else GREEN)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for k, (ax, pa) in enumerate(zip(axes, (8.33, 5.0, 1.47))):
        panel(ax, pa, hl=(k == 0))
        ax.set_xlabel("周向展开 (mm)  周长 50.27 / 3 组", fontsize=7)
    axes[0].set_ylabel("轴向 (mm)", fontsize=7)
    fig.suptitle("双向螺旋 × 周向 3 组：红/绿=两族螺旋筋（仅左栏分色），蓝=开孔", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_helix3.png", dpi=150, bbox_inches="tight")
    print("saved fig_helix3.png")
    for pa in (1.47, 5.0, 10.0):
        print(f"  pa={pa:5.2f}  α={np.degrees(np.arctan(N_CIRC*pa/CIRC)):6.2f}°  "
              f"L={np.hypot(CIRC/N_CIRC, pa):5.2f}  N_long={L_SEG/pa:6.1f}")


if __name__ == "__main__":
    main()
