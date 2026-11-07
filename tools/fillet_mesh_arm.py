# -*- coding: utf-8 -*-
"""fillet_mesh_arm.py — 用掩膜法渲染**整根 400 mm 臂体**（带圆角孔 + 弯曲形状）

方法同 fillet_mesh.py：
  1. 展开坐标 (x=θ·R_OUT, y=s) 建网格
  2. 两段各自的孔多边形（段1 α=34 只倒轴向；段2 α=20 轴向+侧向）
  3. 判定格心是否在孔内 ⇒ 材料格
  4. 材料格按当前弯曲形状映射到 3D（p(s), R(s)）
输出 fig_arm_mesh.png（θ=180°）+ 可选 θ=0
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TC_NO_JIT", "1")
import anim_bend3d as M

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_OUT_MM = M.R_OUT * 1e3
R_IN_MM = M.R_IN * 1e3
CIRC = np.pi * R_OUT_MM * 2
B = CIRC / 6.0
H_OFF = M.W_STRUT * 1e3 / 2.0
L_SEG = 200.0
L_TOT = 400.0
DS = 0.8            # 网格步长 mm


# anim_bend3d 用的是分散常数，这里汇总成两段参数字典
SEG = [dict(alpha=M.ALPHA_A, rho_ax=M.R_AX, rho_sd=M.R_SD_A),
       dict(alpha=M.ALPHA_B, rho_ax=M.R_AX, rho_sd=M.R_SD_B)]


def seg_of(s_mm):
    return SEG[0] if s_mm <= L_SEG else SEG[1]


def ab_of(alpha):
    L = B / np.cos(np.radians(alpha))
    return L * np.sin(np.radians(alpha)), L


def hole_poly(cx, cy, a, rho_ax, rho_sd, n=10):
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


def build_mask():
    nx = int(CIRC / DS) + 1
    ny = int(L_TOT / DS) + 1
    xs = (np.arange(nx) + 0.5) * DS
    ys = (np.arange(ny) + 0.5) * DS
    gx, gy = np.meshgrid(xs, ys)
    centers = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(centers), bool)
    # 逐段生成孔
    for seg_lo, seg_hi in ((0.0, L_SEG), (L_SEG, L_TOT)):
        S = seg_of(0.5 * (seg_lo + seg_hi) + 1e-6)
        a, L = ab_of(S["alpha"])
        pa = 2 * a
        i0 = int(np.floor(seg_lo / pa))
        i1 = int(np.ceil(seg_hi / pa))
        for i in range(i0, i1 + 2):
            cy = i * pa + a
            if cy < seg_lo - pa or cy > seg_hi + pa:
                continue
            for j in range(-1, 7):
                for sh in (-CIRC, 0.0, CIRC):
                    p = hole_poly(j * 2 * B + sh, cy, a, S["rho_ax"], S["rho_sd"])
                    if p[:, 1].max() < seg_lo - 2 or p[:, 1].min() > seg_hi + 2:
                        continue
                    inside |= Path(p).contains_points(centers)
    return (nx, ny, ~inside.reshape(ny, nx))


def render(ax, theta_total, nx, ny, mat, title):
    ss, kf = M.kappa_profile(theta_total)
    sg, P, Rm = M.build_frames(kf, ss)
    quads = []
    ii = np.where(mat.ravel())[0]
    for k in ii:
        ix, iy = k % nx, k // nx
        x0, x1 = ix * DS, (ix + 1) * DS
        y0, y1 = iy * DS, (iy + 1) * DS
        pts = []
        for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            pts.append(M.surface(P, Rm, sg, yy * 1e-3, xx / R_OUT_MM,
                                 M.R_OUT + 0.03e-3))
        quads.append(pts)
    quads = [q for q in quads if np.isfinite(np.asarray(q)).all()]
    pc = Poly3DCollection(quads, facecolor=RED, edgecolor="none", linewidth=0,
                          alpha=1.0)
    ax.add_collection3d(pc)
    pmin, pmax = P.min(0), P.max(0)
    ctr = 0.5 * (pmin + pmax)
    half = 0.5 * float(np.max(pmax - pmin)) + 0.014
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=14, azim=-62)
    ax.set_title(f"{title}（材料格 {mat.mean()*100:.0f}%，{len(quads)} 片）",
                 fontsize=9, color=RED)


def main():
    nx, ny, mat = build_mask()
    print(f"网格 {nx}x{ny}，材料 {mat.mean()*100:.1f}%，材料格 {mat.sum()}")
    fig = plt.figure(figsize=(13, 6.2))
    for k, th in enumerate((0.0, np.pi)):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        render(ax, th, nx, ny, mat, "直臂 0°" if k == 0 else "180° 弯曲")
    fig.suptitle("掩膜法渲染整根 400 mm 臂体（外表面 + 带圆角孔；"
                 "段1 α=34° 轴向R=5，段2 α=20° 轴向R=5/侧向R=1）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_arm_mesh.png", dpi=160, bbox_inches="tight")
    print("saved fig_arm_mesh.png")


if __name__ == "__main__":
    main()
