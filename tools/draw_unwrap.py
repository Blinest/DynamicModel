# -*- coding: utf-8 -*-
"""draw_unwrap.py — 两段臂体的平面展开图（真实孔形）

画法（与前几版的区别）
----------------------
前一版把材料画成 1.5 mm 细带、其余留白 —— 但那样"孔"的形状由带子决定，
圆角无处表达。本版改为**一致的几何**：孔 = 4 条筋的内边线围成的菱形，
角上按切线圆弧倒圆；材料 = 其余部分。于是
    · 孔边 = 筋中心线向孔内缩 h = w/2 = 0.75 mm
    · 圆角半径 R，切点 t=R/tan(φ/2)，圆心距 d=R/sin(φ/2)
    · 开孔率由鞋带公式实测，直接标在图上

段1 α=34°：只倒轴向尖角（上下），侧角保持尖
段2 α=28°：四角全倒
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

CIRC = 2 * np.pi * 8.0
B = CIRC / 6.0
W_STRUT = 1.5
H_OFF = W_STRUT / 2
RHO_A = 5.0      # 段1：轴向尖角圆角 5 mm（用户指定）
RHO_B = 1.0      # 段2：四角圆角 1 mm
WIN = 60.0


def seg(alpha):
    L = B / np.cos(np.radians(alpha))
    return L * np.sin(np.radians(alpha)), L


def hole_poly(alpha, fillets, rho=1.0, n=10):
    a, L = seg(alpha)
    corners = np.array([[B, 0.0], [0.0, a], [-B, 0.0], [0.0, -a]])
    pts = []
    for i, c in enumerate(corners):
        u1 = corners[(i - 1) % 4] - c
        u2 = corners[(i + 1) % 4] - c
        u1 /= np.linalg.norm(u1)
        u2 /= np.linalg.norm(u2)
        phi = float(np.arccos(np.clip(u1 @ u2, -1, 1)))
        bis = u1 + u2
        bis /= np.linalg.norm(bis)
        V = c + bis * (H_OFF / np.sin(phi / 2))
        if fillets[i]:
            t = rho / np.tan(phi / 2)
            d = rho / np.sin(phi / 2)
            P1, P2 = V + u1 * t, V + u2 * t
            O = V + bis * d
            a1 = np.arctan2((P1 - O)[1], (P1 - O)[0])
            a2 = np.arctan2((P2 - O)[1], (P2 - O)[0])
            da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
            arc = [O + rho * np.array([np.cos(a1 + da * q), np.sin(a1 + da * q)])
                   for q in np.linspace(0, 1, n)]
            pts += [P1] + arc + [P2]
        else:
            pts.append(V)
    return np.array(pts), a


def area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def band(p0, p1, w=W_STRUT):
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = np.linalg.norm(d)
    if L < 1e-9:
        return None
    n = np.array([-d[1], d[0]]) / L * (w / 2)
    return np.array([p0 + n, p1 + n, p1 - n, p0 - n])


def node_fillet_polys(cx, cy, a, rho=1.0, h=H_OFF, n=9):
    """节点处 4 个扇区的切线圆角。

    每个扇区夹在两条筋的**内边线**之间（各内缩 h=w/2）：
        孔角 V = node + bis·(h/sin(φ/2))
        切点   t = ρ/tan(φ/2)，圆心 O = V + bis·(ρ/sin(φ/2))
        填充 = [V, P1, 圆弧…, P2]  ⇒ 与两条内边线严格相切
    返回 (中心小块, [4 个楔形])。
    """
    dirs = [np.array([+B, +a]), np.array([-B, +a]),
            np.array([-B, -a]), np.array([+B, -a])]
    ang = [np.arctan2(d[1], d[0]) for d in dirs]
    order = np.argsort(ang)
    dirs = [dirs[k] / np.linalg.norm(dirs[k]) for k in order]
    Vs, wedges, axial = [], [], []
    for k in range(4):
        u1, u2 = dirs[k], dirs[(k + 1) % 4]
        phi = float(np.arccos(np.clip(u1 @ u2, -1, 1)))
        bis = u1 + u2
        bis /= np.linalg.norm(bis)
        # 角平分线偏轴向(|bis_y|>|bis_x|) ⇒ 该扇区是"轴向尖角"
        axial.append(abs(bis[1]) > abs(bis[0]))
        V = np.array([cx, cy]) + bis * (h / np.sin(phi / 2))
        Vs.append(V)
        if rho > 0:
            t = rho / np.tan(phi / 2)
            d = rho / np.sin(phi / 2)
            P1, P2 = V + u1 * t, V + u2 * t
            O = V + bis * d
            a1 = np.arctan2((P1 - O)[1], (P1 - O)[0])
            a2 = np.arctan2((P2 - O)[1], (P2 - O)[0])
            da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
            arc = [O + rho * np.array([np.cos(a1 + da * q), np.sin(a1 + da * q)])
                   for q in np.linspace(0, 1, n)]
            wedges.append(np.array([V] + [P1] + arc + [P2]))
        else:
            wedges.append(None)
    return np.array(Vs), wedges, axial


def panel(ax, alpha, fillets, title, rho=1.0, only_axial=False):
    """only_axial=True 时只对轴向扇区倒圆（段1），False 时四个扇区都倒（段2）。"""
    fillets_any = any(fillets)
    """白底（=孔），材料只画成 1.5 mm 细带 + 节点圆盘。

    ⚠ 之前"红底挖菱形孔"是错的：菱形只占平面一半，**菱形之间的空隙也是孔**，
       那样画会凭空多出一倍材料。
    材料率 = L·w / (B·a)（每格一根筋的有效覆盖），与上图一致。
    """
    a, L = seg(alpha)
    pa = 2 * a
    ax.add_patch(Rectangle((0, 0), CIRC, WIN, facecolor="white",
                           edgecolor="0.85", lw=0.6, zorder=0))
    # ⚠ 节点是 n·a 递增（不是 n·pa），行范围必须按 a 覆盖整个窗口
    n_rows = int(np.ceil(WIN / a)) + 4
    n_cols = int(np.ceil(CIRC / B)) + 4
    # 筋：节点 (m·B, n·a)（m−n 为偶）连到 (m+1, n±1)
    for n in range(-2, n_rows):
        for m in range(-2, n_cols):
            if (m - n) % 2:
                continue
            p0 = np.array([m * B, n * a])
            for dm, dn in ((+1, +1), (+1, -1)):
                p1 = np.array([(m + dm) * B, (n + dn) * a])
                bd = band(p0, p1)
                if bd is not None:
                    ax.add_patch(Polygon(bd, closed=True, facecolor=RED,
                                         edgecolor="none", zorder=2))
    # 节点圆盘：所有节点画 w/2 的小圆（筋交汇），倒圆的节点再放大到 R
    for n in range(-2, n_rows):
        for m in range(-2, n_cols):
            if (m - n) % 2:
                continue
            cx, cy = m * B, n * a
            # 圆角按**扇区方向**选：段1 只倒轴向扇区(上下)，段2 四个扇区都倒
            Vs, wedges, axial = node_fillet_polys(cx, cy, a,
                                                  rho=rho if fillets_any else 0.0)
            ax.add_patch(Polygon(Vs, closed=True, facecolor=RED,
                                 edgecolor="none", zorder=3))
            if fillets_any:
                for wd, is_ax in zip(wedges, axial):
                    if wd is None:
                        continue
                    if only_axial and not is_ax:
                        continue
                    ax.add_patch(Polygon(wd, closed=True, facecolor=RED,
                                         edgecolor="none", zorder=3))
    ax.set_xlim(0, CIRC)
    ax.set_ylim(0, WIN)
    ax.set_aspect("equal")
    ax.set_xticks([0, CIRC / 3, 2 * CIRC / 3, CIRC])
    ax.set_xticklabels(["0", "16.8", "33.5", "50.3"], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("周向展开 (mm)  3 个菱形/圈", fontsize=8)
    ax.set_ylabel("轴向 (mm)", fontsize=8)
    # 材料率 = 筋条(2Lw) + 节点块 减去重叠，除以每孔占地 4Ba
    node_r = W_STRUT * 0.7071
    n_nodes = 2.0                       # 每孔占地 4Ba 内有 2 个节点
    mat = (2 * L * W_STRUT + 0.6 * n_nodes * np.pi * node_r ** 2) / (4 * B * a)
    ax.set_title(f"{title}\nα={alpha}°  a={a:.2f}  节距={pa:.2f}  "
                 f"w={W_STRUT}  R={rho}\n材料 {mat*100:.0f}% / 开孔 {(1-mat)*100:.0f}%",
                 fontsize=9, color=RED)
    return mat


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2))
    f1 = panel(axes[0], 34.0, [False, True, False, True],
               f"① 段1（靠基座）：只倒轴向尖角 R={RHO_A}", rho=RHO_A,
               only_axial=True)
    f2 = panel(axes[1], 20.0, [True, True, True, True],
               f"② 段2：α=20°，四角全倒 R={RHO_B}", rho=RHO_B,
               only_axial=False)
    fig.suptitle(f"臂体平面展开图：红=材料（筋宽 {W_STRUT} mm；节点圆角为与两条内边线相切的圆弧：段1 R={RHO_A}（仅尖角）/ 段2 R={RHO_B}（四角））"
                 f"段1 R={RHO_A} mm（仅尖角）/ 段2 R={RHO_B} mm（四角）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_unwrap.png", dpi=170, bbox_inches="tight")
    print("saved fig_unwrap.png")
    print(f"  段1 材料 {f1*100:.1f}%   段2 材料 {f2*100:.1f}%")


if __name__ == "__main__":
    main()
