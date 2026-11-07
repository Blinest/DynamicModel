# -*- coding: utf-8 -*-
"""fillet_mesh.py — 用**掩膜法**渲染带圆角孔洞的筒壁（真正的布尔差集）

为什么换方法
------------
之前用「筋块 + 圆角补片」叠加近似实体，union 处必然残留原尖角的边，
无论怎么调半径都消不掉。matplotlib 没有布尔运算。

本方法：
  1. 在展开坐标 (x=θ·R_OUT, y=s) 上建细网格
  2. 每个孔（菱形四角按切线圆弧倒圆）作为一个多边形
  3. 用 matplotlib.path 判定每个格子中心是否落在**孔内**
  4. 只渲染「不在任何孔内」的格子（外表面 / 内表面）
  5. 孔壁（材料与孔的交界）另画一圈侧面

输出 fig_fillet_mesh.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_OUT, R_IN = 8.0, 6.5
W_STRUT = 1.5
H_OFF = W_STRUT / 2
CIRC = 2 * np.pi * R_OUT
B = CIRC / 6.0
SEG = [dict(alpha=34.0, rho_ax=5.0, rho_sd=0.0),
       dict(alpha=20.0, rho_ax=5.0, rho_sd=1.0)]
N_ROW = 3


def ab_of(alpha):
    L = B / np.cos(np.radians(alpha))
    return L * np.sin(np.radians(alpha)), L


def hole_poly(cx, cy, a, rho_ax, rho_sd, n=14):
    """孔边界（含切线圆弧），坐标 (x=θR, y=s)，单位 mm。"""
    corners = [(+B, 0.0), (0.0, +a), (-B, 0.0), (0.0, -a)]
    pts = []
    for i, c in enumerate(corners):
        c = np.array(c)
        p_prev = np.array(corners[(i - 1) % 4])
        p_next = np.array(corners[(i + 1) % 4])
        u1 = (p_prev - c) / np.linalg.norm(p_prev - c)
        u2 = (p_next - c) / np.linalg.norm(p_next - c)
        phi = float(np.arccos(np.clip(u1 @ u2, -1, 1)))
        bis = (u1 + u2) / np.linalg.norm(u1 + u2)
        is_ax = abs(bis[1]) > abs(bis[0])
        rho = rho_ax if is_ax else rho_sd
        V = c + bis * (H_OFF / np.sin(phi / 2))
        if rho <= 0:
            pts.append(V)
            continue
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
    return np.array(pts) + np.array([cx, cy])


def main():
    # 窗口：段1 的 N_ROW 行
    a, L = ab_of(SEG[0]["alpha"])
    pa = 2 * a
    s_lo, s_hi = 0.0, N_ROW * pa
    ds = 0.30
    nx = int(CIRC / ds) + 1
    ny = int((s_hi - s_lo) / ds) + 1
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * ds,
                         s_lo + (np.arange(ny) + 0.5) * ds)
    centers = np.column_stack([gx.ravel(), gy.ravel()])

    # 所有孔多边形（周向周期展开）
    polys = []
    for i in range(-1, N_ROW + 2):
        cy = i * pa + a
        for j in range(-1, 7):
            cx = j * 2 * B
            for sh in (-CIRC, 0.0, CIRC):
                polys.append(hole_poly(cx + sh, cy, a, SEG[0]["rho_ax"],
                                       SEG[0]["rho_sd"]))
    inside = np.zeros(len(centers), bool)
    for p in polys:
        if p[:, 0].max() < -10 or p[:, 0].min() > CIRC + 10:
            continue
        if p[:, 1].max() < s_lo - 10 or p[:, 1].min() > s_hi + 10:
            continue
        inside |= Path(p).contains_points(centers)
    mat = ~inside
    print(f"网格 {nx}x{ny} = {len(centers)} 格，材料格 {mat.sum()} "
          f"({mat.mean()*100:.1f}%)")

    # 生成材料格的四边形（外表面 / 内表面）
    def surf_quads(rad, off):
        quads = []
        ii = np.where(mat)[0]
        for k in ii:
            ix, iy = k % nx, k // nx
            x0, x1 = ix * ds, (ix + 1) * ds
            y0, y1 = s_lo + iy * ds, s_lo + (iy + 1) * ds
            pts = []
            for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                th = xx / R_OUT
                pts.append(np.array([(rad + off) * np.cos(th),
                                     (rad + off) * np.sin(th), yy]))
            quads.append(pts)
        return quads

    q_out = surf_quads(R_OUT, +0.03)
    q_in = surf_quads(R_IN, -0.03)
    # 孔壁：材料格与孔格的公共边
    wall = []
    m2 = mat.reshape(ny, nx)
    for iy in range(ny - 1):
        for ix in range(nx):
            x0, x1 = ix * ds, (ix + 1) * ds
            y0, y1 = s_lo + iy * ds, s_lo + (iy + 1) * ds
            for (dx, dy) in ((0, 1), (1, 0)):
                jx, jy = ix + dx, iy + dy
                if jx >= nx or jy >= ny:
                    continue
                if m2[iy, ix] == m2[jy, jx]:
                    continue
                if dy:   # 沿 x 方向的公共边（y = y1）
                    pa_, pb_ = (x0, y1), (x1, y1)
                else:    # 沿 y 方向的公共边（x = x1）
                    pa_, pb_ = (x1, y0), (x1, y1)
                wall.append([np.array([R_OUT * np.cos(pa_[0] / R_OUT),
                                       R_OUT * np.sin(pa_[0] / R_OUT), pa_[1]]),
                             np.array([R_OUT * np.cos(pb_[0] / R_OUT),
                                       R_OUT * np.sin(pb_[0] / R_OUT), pb_[1]]),
                             np.array([R_IN * np.cos(pb_[0] / R_OUT),
                                       R_IN * np.sin(pb_[0] / R_OUT), pb_[1]]),
                             np.array([R_IN * np.cos(pa_[0] / R_OUT),
                                       R_IN * np.sin(pa_[0] / R_OUT), pa_[1]])])
    print(f"外表面 {len(q_out)} 格 / 内表面 {len(q_in)} 格 / 孔壁 {len(wall)} 片")

    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for quads, col, alp in ((q_out, RED, 1.0), (q_in, "#7A3234", 1.0), (wall, "#8E4446", 1.0)):
        if not quads:
            continue
        pc = Poly3DCollection(quads, facecolor=col, edgecolor="none",
                              linewidth=0, alpha=alp)
        ax.add_collection3d(pc)
    ax.set_xlim(-13, 13); ax.set_ylim(-13, 13); ax.set_zlim(-1, s_hi + 1)
    ax.set_box_aspect((1, 1, s_hi / 26.0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=16, azim=-62)
    ax.set_title(f"掩膜法渲染：段1 α={SEG[0]['alpha']}° 轴向R={SEG[0]['rho_ax']}"
                 f"（材料 {mat.mean()*100:.1f}%）", fontsize=9, color=RED)
    fig.tight_layout()
    fig.savefig("fig_fillet_mesh.png", dpi=170, bbox_inches="tight")
    print("saved fig_fillet_mesh.png")


if __name__ == "__main__":
    main()
