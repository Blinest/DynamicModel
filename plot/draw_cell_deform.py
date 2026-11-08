# -*- coding: utf-8 -*-
"""draw_cell_deform.py — 菱形晶胞阵列变形（节点共用版）

**上一版的错误**：每行周向错开 b（stagger），导致相邻菱形的顶点对不上，
既没有公共节点、也没有公共边。

**本版构造**（标准菱形格）：
    节点   N(i,j) = (j·b, i·a)            ← 规则方格，间距 (b, a)
    筋     连**对角**节点：N(i,j)→N(i+1,j±1)，长度 = √(a²+b²) = 9.36 mm
    菱形   顶点 = N(i,j), N(i+1,j+1), N(i+2,j), N(i+1,j−1)
    ⇒ 每个节点被 4 个菱形共用（真正的"公共节点"）

变形：弯曲应变 ε = κ·y，杆长 L 不变 ⇒ a' = a(1+ε)、b' = √(L² − a'²)
      a' 与 b' 变化后节点网格同步变化，菱形仍然共用节点。
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


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

PA, PITCH_C = 8.333, 16.755
A0, B0 = PA / 2, PITCH_C / 2
L_STRUT = np.hypot(A0, B0)
R_FIL = 1.2
N_I, N_J = 9, 7          # 节点网格 i×j（约 4×3 个菱形）


def ab(eps, exag=1.0):
    a = A0 * (1.0 + eps * exag)
    a = float(np.clip(a, 0.08 * A0, 0.985 * L_STRUT))
    b = float(np.sqrt(max(L_STRUT ** 2 - a ** 2, 0.0)))
    return a, b


def draw(ax, eps, exag=1.0, ref=True, title=""):
    a, b = ab(eps, exag)
    aR, bR = ab(eps, 0.0)            # 参考=未变形

    # ① 参考格（未变形），灰虚线
    if ref:
        for i in range(N_I):
            for j in range(N_J):
                cx, cy = j * bR, i * aR
                ax.add_patch(Polygon([[cx - bR, cy], [cx, cy + aR],
                                      [cx + bR, cy], [cx, cy - aR]],
                                     closed=True, fill=False, ls="--",
                                     ec="0.6", lw=0.7, zorder=1))
    # ② 菱形面（开孔）：顶点 = 4 个节点
    for i in range(N_I - 2):
        for j in range(1, N_J - 1):
            v = [(j * b, i * a), ((j + 1) * b, (i + 1) * a),
                 (j * b, (i + 2) * a), ((j - 1) * b, (i + 1) * a)]
            ax.add_patch(Polygon(v, closed=True, facecolor=BLUE, alpha=0.40,
                                 edgecolor="none", zorder=2))
    # ③ 筋：连对角节点（每条边只画一次）
    for i in range(N_I - 1):
        for j in range(N_J):
            for dj in (+1, -1):
                j2 = j + dj
                if not (0 <= j2 < N_J):
                    continue
                ax.plot([j * b, j2 * b], [i * a, (i + 1) * a],
                        color=RED, lw=2.0, solid_capstyle="round", zorder=3)
    # ④ 节点（公共节点）
    for i in range(N_I):
        for j in range(N_J):
            ax.add_patch(Circle((j * b, i * a), R_FIL, facecolor=GREEN,
                                alpha=0.95, edgecolor="white", lw=0.6, zorder=4))
    area = 2 * a * b / (2 * A0 * B0) * 100
    ax.set_xlim(-b, (N_J - 1) * b + b)
    ax.set_ylim(-a, (N_I - 1) * a + a)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}\na={a:.2f}  b={b:.2f}（L={L_STRUT:.2f} 不变）\n"
                 f"孔面积 {area:.0f}%"
                 f"{f'   ×{exag:g} 放大' if exag != 1 else ''}",
                 fontsize=8.5, color=RED if eps > 0 else (GREEN if eps < 0 else "0.3"))


def main():
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.6))
    draw(axes[0], -0.081, 1.0, True, "① 内侧壁 ε=−8.1%")
    draw(axes[1], 0.0, 1.0, True, "② 中性面 ε=0")
    draw(axes[2], +0.081, 1.0, True, "③ 外侧壁 ε=+8.1%")
    draw(axes[3], +0.081, 5.0, True, "④ 外侧 ×5 放大")
    fig.suptitle("菱形晶胞阵列（公共节点版）：绿=共用节点（每点连 4 筋、被 4 孔共用）　"
                 "红=斜撑 L=9.36 mm　蓝=菱形孔　灰虚线=未变形参考", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_cell_deform.png", dpi=150, bbox_inches="tight")
    print("saved fig_cell_deform.png")
    for eps in (-0.081, 0.0, 0.081):
        a, b = ab(eps)
        print(f"  ε={eps*100:+5.1f}%  a={a:.3f} b={b:.3f}  孔面积 {2*a*b/(2*A0*B0)*100:.1f}%  "
              f"筋长 {np.hypot(a,b):.3f}")


if __name__ == "__main__":
    main()
