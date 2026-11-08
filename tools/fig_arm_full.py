# -*- coding: utf-8 -*-
"""fig_arm_full.py — 完整臂形（全长 400 mm，掩膜法，含逐行压缩变形）

与 fillet_mesh_arm 的差别
    · 掩膜**逐行重建**：每行按 ε(s)=κ(s)·R 重算 a、b（杆长不变）
    · 几何用当前定案参数（两段筋宽均 2mm、段1 α=34 轴向R=5/侧向尖、段2 α=20 轴向R=5/侧向R=1）
    · 孔多边形用**包围盒预筛**，否则 2016 个多边形 × 3 万个点会跑几分钟

输出 fig_arm_full.png（左 0°、右 180°）
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
import anim_bend3d as M   # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_OUT_MM = M.R_OUT * 1e3
CIRC = 2 * np.pi * R_OUT_MM
B = CIRC / 6.0
H_OFF = M.W_STRUT * 1e3 / 2.0
L_SEG = 200.0
L_TOT = 400.0
DS = 0.8
JOINT_HALF = 6.0


def seg_params(s_mm):
    if s_mm <= L_SEG:
        return M.ALPHA_A, M.R_AX, M.R_SD_A
    return M.ALPHA_B, M.R_AX, M.R_SD_B


def ab_local(alpha, eps):
    L = B / np.cos(np.radians(alpha))
    a0 = L * np.sin(np.radians(alpha))
    a = float(np.clip(a0 * (1.0 + eps), 0.15 * a0, 0.985 * L))
    b = float(np.sqrt(max(L ** 2 - a ** 2, 1e-9)))
    return a, b


def hole_poly(cx, cy, a, b, rho_ax, rho_sd, n=8):
    corners = [(+b, 0.0), (0.0, +a), (-b, 0.0), (0.0, -a)]
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


def build_mask(theta_total):
    ss, kf = M.kappa_profile(theta_total)
    nx = int(CIRC / DS) + 1
    rows, s = [], 0.0
    while s < L_TOT - 1e-6:
        alpha, _, _ = seg_params(s + 1e-6)
        eps = -float(np.interp(s * 1e-3, ss, kf)) * M.R_OUT
        a, b = ab_local(alpha, eps)
        rows.append((s, a, b, alpha))
        s += 2 * a
    ny = int(rows[-1][0] / DS) + 2
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * DS,
                         (np.arange(ny) + 0.5) * DS)
    centers = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(centers), bool)
    n_poly = 0
    for i, (cy0, a, b, alpha) in enumerate(rows):
        _, rho_ax, rho_sd = seg_params(cy0 + 1e-6)
        for fam in (0.0, 1.0):
            cy = cy0 + a + fam * a
            for j in range(-1, 7):
                for sh in (-CIRC, 0.0, CIRC):
                    p = hole_poly(j * 2 * B + sh + fam * B, cy, a, b,
                                  rho_ax, rho_sd)
                    # 包围盒预筛（否则 2000+ 多边形 × 3 万点太慢）
                    sel = ((np.abs(centers[:, 0] - p[:, 0].mean())
                            <= (p[:, 0].max() - p[:, 0].min()) / 2 + 1) &
                           (np.abs(centers[:, 1] - p[:, 1].mean())
                            <= (p[:, 1].max() - p[:, 1].min()) / 2 + 1))
                    if not sel.any():
                        continue
                    n_poly += 1
                    inside[sel] |= Path(p).contains_points(centers[sel])
    mat = ~inside.reshape(ny, nx)
    ys = (np.arange(ny) + 0.5) * DS
    mat[np.abs(ys - L_SEG) <= JOINT_HALF, :] = True     # 段界实心厚壁筒
    print(f"  网格 {nx}x{ny}，材料 {mat.mean()*100:.1f}%，"
          f"实际测试多边形 {n_poly}", flush=True)
    return nx, ny, mat


def comp_force(theta_mm, s_mm, ss, kf):
    """该处的**压缩力**大小 (N)：F<0 表示受压，返回 max(0, -F)。"""
    kap = float(np.interp(s_mm * 1e-3, ss, kf))          # 1/m
    alpha = F.seg_params(s_mm)[0] if False else None
    a_deg = (M.ALPHA_A if s_mm <= L_SEG else M.ALPHA_B)
    w_mm = 2.0
    A = (w_mm * 1e-3) * (1.5e-3)                          # 筋截面积 m²
    eps = kap * M.R_OUT * np.cos(theta_mm / R_OUT_MM)      # 膜应变
    F_ = M_E_MOD * (eps * np.sin(np.radians(a_deg))) * A
    return max(0.0, -F_)


M_E_MOD = 2.0e8


def render(ax, theta_total, nx, ny, mat, title):
    ss, kf = M.kappa_profile(theta_total)
    sg, P, Rm = M.build_frames(kf, ss)
    quads, comps = [], []
    for k in np.where(mat.ravel())[0]:
        ix, iy = k % nx, k // nx
        x0, x1 = ix * DS, (ix + 1) * DS
        y0, y1 = iy * DS, (iy + 1) * DS
        pts = [M.surface(P, Rm, sg, yy * 1e-3, xx / R_OUT_MM, M.R_OUT + 0.03e-3)
               for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        quads.append(pts)
        comps.append(comp_force(0.5 * (x0 + x1), 0.5 * (y0 + y1), ss, kf))
    keep = [i for i, q in enumerate(quads)
            if np.isfinite(np.asarray(q)).all()]
    quads = [quads[i] for i in keep]
    comps = np.array([comps[i] for i in keep])
    cmax = max(comps.max(), 1e-9)
    pc = Poly3DCollection(quads, edgecolor="none", linewidth=0, antialiased=True)
    import matplotlib.cm as cm
    pc.set_facecolor(cm.coolwarm(comps / cmax))     # 蓝=力小, 红=压缩大
    ax.add_collection3d(pc)
    half = 0.432
    ax.set_xlim(-0.34, 0.34)
    ax.set_ylim(-0.34, 0.34)
    ax.set_zlim(-0.02, 0.45)
    dx = 0.68; dy = 0.68; dz = 0.47
    m_ = max(dx, dy, dz)
    ax.set_box_aspect((dx / m_, dy / m_, dz / m_), zoom=1.05)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=14, azim=-62)
    ax.set_title(f"{title}   最大压缩力 {cmax:.1f} N", fontsize=9, color=RED)
    return cmax


def main():
    fig = plt.figure(figsize=(12.5, 6.4))
    for k, th in enumerate((0.0, np.pi)):
        nx, ny, mat = build_mask(th)
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        render(ax, th, nx, ny, mat,
               "直臂 0°" if k == 0 else "180° 弯曲（红=压缩大 / 蓝=力小）")
    fig.suptitle("完整臂形（全长 400 mm，掩膜法；材料率含圆角与段界厚壁筒）",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_arm_full.png", dpi=170, bbox_inches="tight")
    print("saved fig_arm_full.png")


if __name__ == "__main__":
    main()
