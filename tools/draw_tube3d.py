# -*- coding: utf-8 -*-
"""draw_tube3d.py — 卷成有厚度的圆筒；筋在节点处圆角连接

三处几何：
  1. 筒壁厚 1.5 mm（内 6.5 / 外 8.0），筋是实体（4 面）
  2. 筋按"节点→节点"生成：节点在 θ = j·60°、z = k·pa，每节点 4 根筋汇合
  3. **节点圆角**：每个节点放一个半径 R_F 的圆柱凸台（径向贯通壁厚），
     筋的两端各向内缩 R_F ⇒ 视觉与工艺上都变成圆角过渡，而不是尖角
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_OUT, R_IN, N_CIRC = 8.0, 6.5, 3
W_STRUT = 1.5
R_FILLET = 1.2               # 圆角半径 [mm]，按两筋夹角求切点，与两筋边线严格相切
R_DISK = 5.0
PA = 8.33
DTH = 2 * np.pi / N_CIRC
R_MID = (R_OUT + R_IN) / 2


def pt(theta, z, r):
    return np.array([r * np.cos(theta), r * np.sin(theta), z])


def frame(theta, tang):
    nrm = np.array([np.cos(theta), np.sin(theta), 0.0])
    tang = tang / max(np.linalg.norm(tang), 1e-12)
    perp = np.cross(nrm, tang)
    perp /= max(np.linalg.norm(perp), 1e-12)
    return perp


def strut_boxes(thA, zA, thB, zB, n=26, w=W_STRUT, trim=0.0):
    """A→B 的实体筋。trim: 两端各内缩的弧长 [mm]，为圆角留位。"""
    th = np.linspace(thA, thB, n)
    z = np.linspace(zA, zB, n)
    arc = R_MID * np.abs(th - thA) + np.abs(z - zA)
    S = arc[-1]
    if trim > 0 and S > 3 * trim:
        th = np.interp([trim, S - trim], arc, th, left=th[0], right=th[-1])
        z = np.interp([trim, S - trim], arc, z, left=z[0], right=z[-1])
        n = 8
        th = np.linspace(th[0], th[1], n)
        z = np.linspace(z[0], z[1], n)
    corners = []
    for i in range(n):
        j = min(i + 1, n - 1)
        i0 = max(i - 1, 0)
        tang = pt(th[j], z[j], R_OUT) - pt(th[i0], z[i0], R_OUT)
        perp = frame(th[i], tang)
        corners.append([pt(th[i], z[i], R_OUT) + perp * (w / 2),
                        pt(th[i], z[i], R_OUT) - perp * (w / 2),
                        pt(th[i], z[i], R_IN) - perp * (w / 2),
                        pt(th[i], z[i], R_IN) + perp * (w / 2)])
    corners = np.asarray(corners)
    quads = []
    for i in range(n - 1):
        for k in range(4):
            k2 = (k + 1) % 4
            quads.append([corners[i, k], corners[i + 1, k],
                          corners[i + 1, k2], corners[i, k2]])
    return quads


def fillet_wedge(u1, u2, R, n=10):
    """节点处一对相邻筋之间的圆角楔块（2D，返回多边形点列）。

    u1、u2 为两筋的单位方向；圆弧与两根筋的边线在切点处**严格相切**：
        切点距节点   t = R / tan(φ/2)
        圆心距节点   d = R / sin(φ/2)   （沿角平分线）
    φ = arccos(u1·u2) 为两筋夹角。
    """
    u1 = np.asarray(u1, float); u2 = np.asarray(u2, float)
    u1 /= np.linalg.norm(u1); u2 /= np.linalg.norm(u2)
    cphi = float(np.clip(u1 @ u2, -1.0, 1.0))
    phi = np.arccos(cphi)
    if phi < 1e-6 or phi > np.pi - 1e-6:
        return None
    bis = u1 + u2
    bis /= np.linalg.norm(bis)
    t = R / np.tan(phi / 2.0)
    d = R / np.sin(phi / 2.0)
    P1, P2, O = u1 * t, u2 * t, bis * d
    a1 = np.arctan2(*(P1 - O)[::-1])
    a2 = np.arctan2(*(P2 - O)[::-1])
    da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi      # 取短弧
    ang = a1 + np.linspace(0, da, n)
    arc = [O + R * np.array([np.cos(a), np.sin(a)]) for a in ang]
    return [np.array([0.0, 0.0])] + [P1] + arc + [P2]


def wedge_solid(poly2d, theta, z, r_in=R_IN, r_out=R_OUT):
    """把 2D 多边形(以 (弧长, 轴向) 表示) 贴到筒面并沿径向拉伸成实体。"""
    def to3(p, r):
        s_, z_ = p[0], p[1]
        th = theta + s_ / R_MID
        return pt(th, z + z_, r)
    n = len(poly2d)
    out = [to3(p, r_out) for p in poly2d]
    inn = [to3(p, r_in) for p in poly2d]
    quads = [out, inn]
    for k in range(n):
        k2 = (k + 1) % n
        quads.append([out[k], out[k2], inn[k2], inn[k]])
    return quads


def node_fillets(theta, z, R=R_FILLET):
    """一个节点上 4 对相邻筋的圆角楔块。"""
    a = np.arctan(N_circ_alpha())
    dirs = [np.array([np.cos(a), np.sin(a)]), np.array([-np.cos(a), np.sin(a)]),
            np.array([-np.cos(a), -np.sin(a)]), np.array([np.cos(a), -np.sin(a)])]
    quads = []
    for k in range(4):
        poly = fillet_wedge(dirs[k], dirs[(k + 1) % 4], R)
        if poly is not None:
            quads += wedge_solid(poly, theta, z)
    return quads


def N_circ_alpha():
    """斜撑角 α = atan(N_circ·pa / 2πr)"""
    return N_CIRC * PA / (2 * np.pi * R_MID)


def draw(ax, z_span, wire=True, n=24):
    z_lo, z_hi = 0.0, z_span
    k_lo, k_hi = int(np.floor(z_lo / PA)) - 1, int(np.ceil(z_hi / PA)) + 1
    quads, col = [], []
    for k in range(k_lo, k_hi + 1):
        for j in range(N_CIRC * 2):
            th0, z0 = j * np.pi / N_CIRC, k * PA
            for dz in (+PA, -PA):
                th1, z1 = th0 + DTH, z0 + dz
                if max(z0, z1) < z_lo - PA or min(z0, z1) > z_hi + PA:
                    continue
                q = strut_boxes(th0, z0, th1, z1, n=n, trim=0.0)   # 圆心角处由相切圆弧衔接
                quads += q
                col += [(RED if dz > 0 else GREEN)] * len(q)
    # 节点圆角
    for k in range(k_lo, k_hi + 2):
        for j in range(N_CIRC * 2):
            th, z = j * np.pi / N_CIRC, k * PA
            if z < z_lo - PA or z > z_hi + PA:
                continue
            q = node_fillets(th, z)
            quads += q
            col += ["#7A3233"] * len(q)

    pc = Poly3DCollection(quads, edgecolor="none", alpha=0.97, linewidth=0)
    shade = []
    for i, c in enumerate(col):
        if isinstance(c, str) and c.startswith("#7A"):
            shade.append(c)
        else:
            base = np.array(matplotlib.colors.to_rgb(c))
            shade.append(tuple(base * (1.0 if (i // 4) % 2 else 0.78)))
    pc.set_facecolor(shade)
    ax.add_collection3d(pc)

    th = np.linspace(0, 2 * np.pi, 80)
    for zz in (z_lo, z_hi):
        ax.plot(R_OUT * np.cos(th), R_OUT * np.sin(th), zz, color=BLUE, lw=0.6, alpha=0.6)
        ax.plot(R_IN * np.cos(th), R_IN * np.sin(th), zz, color=BLUE, lw=0.6, alpha=0.6)
    if wire:
        for k in range(6):
            a = np.radians(60 * k)
            ax.plot([R_DISK * np.cos(a)] * 2, [R_DISK * np.sin(a)] * 2,
                    [z_lo - 0.5, z_hi + 0.5], color=RED, lw=1.1, alpha=0.8)
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect((1, 1, max(z_span / 20, 0.5)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=14, azim=-58)


def main():
    fig = plt.figure(figsize=(12.5, 5.8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    draw(ax1, 42.0, wire=False)
    ax1.set_title("① 相切圆角节点（Φ16/Φ13，40 mm）", fontsize=10, color=RED)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw(ax2, 18.0, wire=True, n=34)
    ax2.set_title(f"② 特写：相切圆角 R={R_FILLET} mm + 6 根丝", fontsize=10, color=GREEN)
    fig.suptitle(f"卷成圆筒（壁厚 1.5 mm）：双向螺旋 × 周向 3 组；节点圆角 R={R_FILLET} mm 与两筋边线相切",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_tube3d.png", dpi=150, bbox_inches="tight")
    print("saved fig_tube3d.png")


if __name__ == "__main__":
    main()
