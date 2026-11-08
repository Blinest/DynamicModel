# -*- coding: utf-8 -*-
"""draw_lattice.py — 菱形开孔薄壁管：按斜撑角 α 展开三种形态

几何约束（周向 3 胞 + 外半径 8 mm 把一条约束钉死）：
    周向半对角 = L_strut·cosα = 2πr/(2·N_circ) = 8.38 mm
  ⇒ 给定 α 后：
       L_strut   = 8.38 / cosα
       轴向节距  = 2·L_strut·sinα
       轴向行数  = L_seg / 轴向节距
  α 越小越像"斜缝/折线"（1.47 mm 宽 × 16.76 mm 长），α 越大越像"菱形"。

三栏：α = 5°（当前配置）/ 30° / 45°
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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Wedge
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_O, T_WALL = 8.0, 1.5
N_CIRC, L_SEG = 3, 200.0
HALF_C = 2 * np.pi * R_O / N_CIRC / 2        # 8.38 mm 固定


def cell_params(alpha_deg):
    a = np.radians(alpha_deg)
    L = HALF_C / np.cos(a)
    pitch_a = 2 * L * np.sin(a)
    n_long = L_SEG / pitch_a
    return dict(alpha=alpha_deg, L=L, pitch_a=pitch_a, n_long=n_long,
                pitch_c=2 * HALF_C)


def ribbon(p0, p1, w):
    """把线段 p0→p1 变成宽 w 的带状多边形（用于按真实筋宽绘制）。"""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = np.linalg.norm(d)
    if L < 1e-12:
        return None
    nrm = np.array([-d[1], d[0]]) / L * (w / 2.0)
    return np.array([p0 + nrm, p1 + nrm, p1 - nrm, p0 - nrm])


def node_xy(i, j, pitch_c, pitch_a):
    """菱形格点：奇偶行在周向错开半格 ⇒ 相邻菱形共用顶点，筋自然连成一片。"""
    return (j + 0.5 * (i % 2)) * pitch_c, i * pitch_a / 2.0


def draw(ax, alpha_deg, n_show=None, legend=False):
    """孔=蓝菱形，筋=1.5mm 宽的红色带；每条筋只画一次（相邻菱形共用边）。"""
    p = cell_params(alpha_deg)
    n_rows = n_show or min(12, max(int(round(p["n_long"])), 1))
    pc, pa = p["pitch_c"], p["pitch_a"]
    n_i = 2 * n_rows + 1

    # ① 孔：以 (i,j) 为下尖的菱形，四顶点 = 下/右/上/左
    for i in range(0, n_i - 2, 2):
        for j in range(-1, N_CIRC + 1):
            B = node_xy(i, j, pc, pa)
            R = node_xy(i + 1, j, pc, pa)
            T = node_xy(i + 2, j, pc, pa)
            L = node_xy(i + 1, j - 1, pc, pa)
            ax.add_patch(Polygon([B, R, T, L], closed=True, facecolor=BLUE,
                                 alpha=0.5, edgecolor="none", zorder=1))

    # ② 筋：筋在节点处收住（不与相邻筋共边），随后画实心节点
    R_NODE = 0.7                      # 节点半径 mm
    edges, seen = [], set()
    for i in range(n_i - 1):
        for j in range(-2, N_CIRC + 2):
            a = node_xy(i, j, pc, pa)
            for dj in (0, -1):
                b = node_xy(i + 1, j + dj, pc, pa)
                key = tuple(sorted([(round(a[0], 6), round(a[1], 6)),
                                    (round(b[0], 6), round(b[1], 6))]))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((np.array(a), np.array(b)))
    for k, (a, b) in enumerate(edges):
        d = b - a
        L = np.linalg.norm(d)
        if L <= 2 * R_NODE:           # 太短（如 α=5° 时节点互相顶住）就不画筋
            continue
        u = d / L
        rb = ribbon(a + u * R_NODE, b - u * R_NODE, T_WALL)
        if rb is None:
            continue
        ax.add_patch(Polygon(rb, closed=True, facecolor=RED, alpha=0.9,
                             edgecolor=RED, lw=0.3, zorder=3,
                             label="斜撑筋（宽 1.5 mm）" if k == 0 else None))
    # ③ 节点：实心圆，孔在节点处相接
    first_node = True
    for i in range(n_i):
        for j in range(-1, N_CIRC + 1):
            x, y = node_xy(i, j, pc, pa)
            ax.add_patch(Circle((x, y), R_NODE, facecolor=GREEN, alpha=0.95,
                                edgecolor="white", lw=0.5, zorder=5,
                                label="节点（孔在此相接）" if first_node else None))
            first_node = False

    ax.set_xlim(0, N_CIRC * pc * 1.55)
    ax.set_ylim(-pa, n_i * pa / 2 + pa * 0.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.12, ls=":", zorder=0)
    ax.set_xlabel("周向展开 (mm)", fontsize=8)
    kind = "斜缝 / 折线" if alpha_deg < 12 else "菱形"
    ax.set_title(f"α = {alpha_deg}°  ⇒  {kind}\n"
                 f"L={p['L']:.2f} mm  轴向节距={p['pitch_a']:.2f} mm  "
                 f"行数/段={p['n_long']:.0f}  筋={sum(1 for a,b in edges if np.linalg.norm(b-a)>1.4)}  节点={n_i*(N_CIRC+3)}",
                 color=RED if alpha_deg < 12 else GREEN, fontsize=9)
    if legend:
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    return p


def main():
    fig = plt.figure(figsize=(15, 5.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1.5, 1.5, 1.0], wspace=0.26)

    for k, al in enumerate((5.0, 30.0, 45.0)):
        ax = fig.add_subplot(gs[k])
        p = draw(ax, al, n_show=8, legend=(k == 1))
        if k == 0:
            ax.set_ylabel("轴向 (mm)", fontsize=8)

    # 横截面
    ax = fig.add_subplot(gs[3])
    ax.add_patch(Circle((0, 0), R_O, facecolor=BLUE, alpha=0.16, edgecolor=BLUE, lw=1.5))
    ax.add_patch(Circle((0, 0), R_O - T_WALL, facecolor="white", edgecolor=BLUE, lw=1.0))
    for k in range(N_CIRC):
        th = np.degrees(2 * np.pi * k / N_CIRC + np.pi / 2)
        ax.add_patch(Wedge((0, 0), R_O, th - 10, th + 10, width=T_WALL,
                           facecolor=GREEN, alpha=0.8, edgecolor=GREEN))
    for k in range(6):
        th = np.radians(60 * k)
        ax.plot(5.0 * np.cos(th), 5.0 * np.sin(th), "o", color=RED, ms=5)
        ax.text(6.7 * np.cos(th), 6.7 * np.sin(th), str(k), color=RED, fontsize=7,
                ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-9.8, 9.8); ax.set_ylim(-9.8, 9.8)
    ax.axis("off")
    ax.set_title("横截面 Φ16/壁 1.5\n绿=3 处菱形，红=6 根丝", fontsize=9)

    fig.suptitle("菱形开孔管：斜撑角 α 决定形态（周向 3 胞 ⇒ L·cosα = 8.38 mm 固定）",
                 fontsize=11)
    fig.savefig("fig_lattice.png", dpi=160, bbox_inches="tight")
    print("saved fig_lattice.png")
    for al in (5.0, 30.0, 45.0):
        p = cell_params(al)
        print(f"  α={al:5.1f}°  L_strut={p['L']:6.2f}  轴向节距={p['pitch_a']:6.2f}  "
              f"行数/段={p['n_long']:6.1f}  周向={p['pitch_c']:.2f}")


if __name__ == "__main__":
    main()
