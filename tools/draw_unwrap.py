# -*- coding: utf-8 -*-
"""draw_unwrap.py — 两段臂体的平面展开图（真实的孔形，不用标记点）

把筒面沿轴向剪开、展平：
    横轴 = 周向（周长 50.27 mm，周向 3 个菱形）
    纵轴 = 轴向（每段取 60 mm 窗口）
红色=材料（筋），白色=开孔。孔的四个角按段不同倒圆：
    段1（α=34°）：只倒轴向尖角（上下），侧角保持尖
    段2（α=28°）：四角全倒
圆角为切线圆弧（R=2.0 mm），孔边从筋中心线内缩 h=w/2=0.75 mm。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

CIRC = 2 * np.pi * 8.0        # 50.27 mm
B = CIRC / 6.0                # 周向半节距 8.377
W_STRUT = 1.5
H_OFF = W_STRUT / 2
RHO = 2.0
WIN = 60.0                    # 轴向窗口 mm


def seg(alpha):
    L = B / np.cos(np.radians(alpha))
    return L * np.sin(np.radians(alpha)), L     # a, L


def hole_poly(alpha, fillets, rho=RHO, n=8):
    """孔边界（以孔心为原点），返回 (N,2) 点列。角顺序：右侧/上尖/左侧/下尖。"""
    a, L = seg(alpha)
    corners = np.array([[B, 0.0], [0.0, a], [-B, 0.0], [0.0, -a]])
    pts = []
    for i, c in enumerate(corners):
        u1 = (corners[(i - 1) % 4] - c)
        u2 = (corners[(i + 1) % 4] - c)
        u1 /= np.linalg.norm(u1)
        u2 /= np.linalg.norm(u2)
        phi = float(np.arccos(np.clip(u1 @ u2, -1, 1)))
        bis = (u1 + u2)
        bis /= np.linalg.norm(bis)
        V = c + bis * (H_OFF / np.sin(phi / 2))     # 真实孔角（内缩）
        if fillets[i]:
            t = rho / np.tan(phi / 2)
            d = rho / np.sin(phi / 2)
            P1, P2 = V + u1 * t, V + u2 * t
            O = V + bis * d
            a1 = np.arctan2(*(P1 - O)[::-1])
            a2 = np.arctan2(*(P2 - O)[::-1])
            da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
            arc = [O + rho * np.array([np.cos(a1 + da * q), np.sin(a1 + da * q)])
                   for q in np.linspace(0, 1, n)]
            pts += [P1] + arc + [P2]
        else:
            pts.append(V)
    return np.array(pts), a


def panel(ax, alpha, fillets, title):
    a, L = seg(alpha)
    pa = 2 * a
    ax.add_patch(Rectangle((0, 0), CIRC, WIN, facecolor=RED, alpha=0.92,
                           edgecolor="none", zorder=1))
    base, _ = hole_poly(alpha, fillets)
    n_row = int(np.ceil(WIN / pa)) + 1
    for i in range(n_row):
        cy = i * pa
        for j in range(3):
            cx = (2 * j + 1) * B
            for shift in (-CIRC, 0.0, CIRC):        # 周向周期展开
                p = base + np.array([cx + shift, cy])
                ax.add_patch(Polygon(p, closed=True, facecolor="white",
                                     edgecolor="none", zorder=2))
    # 隔段时间画一条筋中线，帮助读结构
    for i in range(n_row):
        cy = i * pa + a
        for j in range(3):
            cx = (2 * j + 1) * B
            ax.plot([cx - B, cx], [cy, cy + a], color="0.75", lw=0.4, zorder=3)
            ax.plot([cx + B, cx], [cy, cy + a], color="0.75", lw=0.4, zorder=3)
            ax.plot([cx - B, cx], [cy, cy - a], color="0.75", lw=0.4, zorder=3)
            ax.plot([cx + B, cx], [cy, cy - a], color="0.75", lw=0.4, zorder=3)
    ax.set_xlim(0, CIRC)
    ax.set_ylim(0, WIN)
    ax.set_aspect("equal")
    ax.set_xticks([0, CIRC / 3, 2 * CIRC / 3, CIRC])
    ax.set_xticklabels(["0", "16.8", "33.5", "50.3"], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("周向展开 (mm)  3 个菱形/圈", fontsize=8)
    ax.set_ylabel("轴向 (mm)", fontsize=8)
    ax.set_title(f"{title}\nα={alpha}°  a={a:.2f}  节距={pa:.2f}  R={RHO}  "
                 f"({WIN:.0f} mm 内 {n_row - 1} 行)", fontsize=9, color=RED)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4))
    panel(axes[0], 34.0, [False, True, False, True],
          "① 段1（靠基座）：只倒轴向尖角")
    panel(axes[1], 28.0, [True, True, True, True],
          "② 段2：四角全倒")
    fig.suptitle("臂体平面展开图（红=材料/筋，白=开孔；切线圆角 R=2.0 mm，"
                 "孔边内缩 0.75 mm）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_unwrap.png", dpi=170, bbox_inches="tight")
    print("saved fig_unwrap.png")


if __name__ == "__main__":
    main()
