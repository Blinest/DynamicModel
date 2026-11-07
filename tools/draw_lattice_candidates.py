# -*- coding: utf-8 -*-
"""draw_lattice_candidates.py — 6 种开孔样式对照图

连续几轮对实物的理解都不对（菱形网 / 共边 / 节点），不再猜，直接画 6 种常见样式，
让人指认。展开尺寸固定：周向 0–50.27 mm（Φ16 周长），轴向 0–20 mm。

配色：#A34D4E 筋 / #516CA3 孔 / #8DAE7F 高亮
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

CIRC = 2 * np.pi * 8.0      # 50.27 mm
YMAX = 20.0


def base(ax, title, color=RED):
    ax.set_xlim(0, CIRC)
    ax.set_ylim(0, YMAX)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15, ls=":")
    ax.set_title(title, fontsize=9, color=color)
    ax.tick_params(labelsize=6)


def p_rhombic(ax):
    """① 菱形网：正菱形，斜撑长 8.41，α=5°（当前模型假设）"""
    L, al = 8.41, np.radians(5.0)
    ha, hc = L * np.sin(al), L * np.cos(al)
    ax.add_patch(Polygon([(0, 0), (CIRC, 0), (CIRC, YMAX), (0, YMAX)],
                         facecolor=BLUE, alpha=0.30, zorder=0))
    for i in range(-2, int(YMAX / (2 * ha)) + 3):
        for j in range(-1, 4):
            cx, cy = j * 2 * hc + (i % 2) * hc, i * 2 * ha
            if -hc < cx < CIRC + hc and -ha < cy < YMAX + ha:
                ax.add_patch(Polygon([(cx - hc, cy), (cx, cy + ha),
                                      (cx + hc, cy), (cx, cy - ha)],
                                     closed=True, fill=False, ec=RED, lw=1.6, zorder=3))
    base(ax, "① 菱形网（α=5°，L=8.41）")


def p_helix1(ax):
    """② 单螺旋缝：一条缝绕一周上升 1.47 mm"""
    ax.add_patch(Polygon([(0, 0), (CIRC, 0), (CIRC, YMAX), (0, YMAX)],
                         facecolor=BLUE, alpha=0.35, zorder=0))
    y0, pitch = 0.0, 1.47
    for k in range(int(YMAX / pitch) + 2):
        y = y0 + k * pitch
        ax.plot([0, CIRC], [y, y + pitch], color=RED, lw=2.6, zorder=3)
    base(ax, "② 单螺旋缝（螺距 1.47）")


def p_helix2(ax):
    """③ 双向螺旋（交叉成菱形孔）"""
    ax.add_patch(Polygon([(0, 0), (CIRC, 0), (CIRC, YMAX), (0, YMAX)],
                         facecolor=BLUE, alpha=0.35, zorder=0))
    pitch = 8.0
    for k in range(-3, int(YMAX / pitch) + 4):
        y = k * pitch
        ax.plot([0, CIRC], [y, y + pitch], color=RED, lw=2.0, zorder=3)
        ax.plot([0, CIRC], [y + pitch, y], color=RED, lw=2.0, zorder=3)
    base(ax, "③ 双向螺旋（交叉菱形）")


def p_axial(ax):
    """④ 轴向长孔阵列：每圆周 3 个，长 8 mm、高 1.4 mm，逐行错位"""
    ax.add_patch(Polygon([(0, 0), (CIRC, 0), (CIRC, YMAX), (0, YMAX)],
                         facecolor=BLUE, alpha=0.35, zorder=0))
    for i in range(6):
        cy = 1.5 + i * 3.0
        for j in range(6):
            cx = j * (CIRC / 6) + (i % 2) * (CIRC / 12)
            ax.add_patch(plt.Rectangle((cx - 4, cy - 0.7), 8.0, 1.4,
                                       facecolor="white", ec=RED, lw=1.4, zorder=3))
    base(ax, "④ 轴向长孔阵列（3 个/周，错位）")


def p_chevron(ax):
    """⑤ 人字纹：V 形缝按列排列"""
    ax.add_patch(Polygon([(0, 0), (CIRC, 0), (CIRC, YMAX), (0, YMAX)],
                         facecolor=BLUE, alpha=0.35, zorder=0))
    for i in range(4):
        cy = 2.5 + i * 5.0
        for j in range(4):
            cx = j * (CIRC / 4) + 6.0
            ax.plot([cx - 5, cx, cx + 5], [cy + 2, cy, cy + 2],
                    color=RED, lw=2.2, zorder=3, solid_capstyle="round")
    base(ax, "⑤ 人字纹（V 形）")


def p_slotmix(ax):
    """⑥ 交叉斜缝（大倾角 30°）"""
    ax.add_patch(Polygon([(0, 0), (CIRC, 0), (CIRC, YMAX), (0, YMAX)],
                         facecolor=BLUE, alpha=0.35, zorder=0))
    al = np.radians(30.0)
    for s in np.arange(-YMAX, CIRC + YMAX, 3.0):
        ax.plot([0, CIRC], [s, s + CIRC * np.tan(al)], color=RED, lw=1.8, zorder=3)
        ax.plot([0, CIRC], [s + CIRC * np.tan(al), s], color=RED, lw=1.8, zorder=3)
    base(ax, "⑥ 交叉斜缝（30°，菱形化）")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
    for ax, fn in zip(axes.ravel(), (p_rhombic, p_helix1, p_helix2,
                                     p_axial, p_chevron, p_slotmix)):
        fn(ax)
    for ax in axes[1]:
        ax.set_xlabel("周向展开 (mm)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("轴向 (mm)", fontsize=7)
    fig.suptitle("开孔样式对照：红=筋（材料） 蓝=开孔   周向 0–50.3 mm（Φ16 周长），轴向 0–20 mm",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_lattice_candidates.png", dpi=150, bbox_inches="tight")
    print("saved fig_lattice_candidates.png")


if __name__ == "__main__":
    main()
