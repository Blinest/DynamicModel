# -*- coding: utf-8 -*-
"""anim_mesh3d.py — 三维动图：局部（根部 100 mm）的压缩变形可辨

为什么只取根部 100 mm
--------------------
全臂 400 mm 视图里，孔轴向半高只有 3–6 mm，压缩量 ±8% 折合不到 1 mm，
约 1–2 像素，必然看不出。取根部 100 mm、网格 0.4 mm，孔与压缩都能分辨。

关键：**掩膜随弯曲变化**
    每帧按 ε(s) = κ(s)·y 重新算各行的 a、b（杆长 L_strut 不变），
    再重建孔多边形与材料掩膜 ⇒ 孔沿程变小（内侧压缩），这是要展示的现象。

输出 anim_mesh3d.gif（10 帧，θ 0→180°）
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
from PIL import Image

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
S_WIN = 100.0      # 根部窗口 mm
DS = 0.4
NFRAME = 10
ALPHA = M.ALPHA_A
RHO_AX = M.R_AX
RHO_SD = M.R_SD_A


def ab_local(eps):
    """杆长不变：a 随轴向应变缩放，b 由 L 与 a 反算。"""
    L = B / np.cos(np.radians(ALPHA))
    a0 = L * np.sin(np.radians(ALPHA))
    a = float(np.clip(a0 * (1.0 + eps), 0.15 * a0, 0.985 * L))
    b = float(np.sqrt(max(L ** 2 - a ** 2, 1e-9)))
    return a, b


def hole_poly(cx, cy, a, b, rho_ax, rho_sd, n=10):
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


def build_frame(theta_total):
    """返回 (网格尺寸, 材料掩膜, 骨架, 行位置)。掩膜随 ε(s) 逐行重建。"""
    ss, kf = M.kappa_profile(theta_total)
    sg, P, Rm = M.build_frames(kf, ss)
    nx = int(CIRC / DS) + 1
    # 行：沿 s 累积，每行 a 由该处 ε 决定
    rows, s = [], 0.0
    while s < S_WIN:
        eps = -float(np.interp(s * 1e-3, ss, kf)) * M.R_OUT
        a, b = ab_local(eps)
        rows.append((s, a, b))
        s += 2 * a
    ny = int(rows[-1][0] / DS) + 2
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * DS,
                         (np.arange(ny) + 0.5) * DS)
    centers = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(centers), bool)
    for i in range(-1, len(rows)):
        if i < 0:
            cy, a, b = -rows[0][1], rows[0][1], rows[0][2]
        else:
            cy, a, b = rows[i][0] + rows[i][1], rows[i][1], rows[i][2]
        if cy < -2 * a or cy > S_WIN + 2 * a:
            continue
        for fam in (0.0, 1.0):
            for j in range(-1, 7):
                for sh in (-CIRC, 0.0, CIRC):
                    p = hole_poly(j * 2 * B + sh + fam * B, cy + fam * a,
                                  a, b, RHO_AX, RHO_SD)
                    inside |= Path(p).contains_points(centers)
    mat = ~inside.reshape(ny, nx)
    return nx, ny, mat, sg, P, Rm, rows


def render(ax, nx, ny, mat, sg, P, Rm, theta_total, lims=None):
    ax.clear()
    r_o = M.R_OUT + 0.03e-3
    quads = []
    for k in np.where(mat.ravel())[0]:
        ix, iy = k % nx, k // nx
        x0, x1 = ix * DS, (ix + 1) * DS
        y0, y1 = iy * DS, (iy + 1) * DS
        pts = [M.surface(P, Rm, sg, yy * 1e-3, xx / R_OUT_MM, r_o)
               for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        quads.append(pts)
    quads = [q for q in quads if np.isfinite(np.asarray(q)).all()]
    pc = Poly3DCollection(quads, facecolor=RED, edgecolor="none", linewidth=0,
                          antialiased=True)
    ax.add_collection3d(pc)
    if lims is None:
        pmin, pmax = P.min(0), P.max(0)
        ctr = 0.5 * (pmin + pmax)
        half = 0.5 * float(np.max(pmax - pmin)) + 0.014
        lims = ((ctr[0] - half, ctr[0] + half),
                (ctr[1] - half, ctr[1] + half),
                (min(0.0, pmin[2]) - 0.01, S_WIN * 1e-3 + 0.01))
    # 所有帧共用同一套坐标范围（否则相机逐帧跟着包围盒动，看起来画面在飘）
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=16, azim=-62)
    ax.set_title(f"根部 100 mm   θ={np.degrees(theta_total):.0f}°   "
                 f"材料 {mat.mean()*100:.0f}%", fontsize=9)


def compute_lims():
    """取最大弯曲(180°)那一帧的包围盒，全帧共用。"""
    nx, ny, mat, sg, P, Rm, rows = build_frame(np.pi)
    pmin, pmax = P.min(0), P.max(0)
    ctr = 0.5 * (pmin + pmax)
    half = 0.5 * float(np.max(pmax - pmin)) + 0.014
    return ((ctr[0] - half, ctr[0] + half),
            (ctr[1] - half, ctr[1] + half),
            (min(0.0, pmin[2]) - 0.01, S_WIN * 1e-3 + 0.01))


def main():
    files = []
    LIMS = compute_lims()
    for k, deg in enumerate(np.linspace(0, 180, NFRAME)):
        th = np.radians(deg)
        nx, ny, mat, sg, P, Rm, rows = build_frame(th)
        fig = plt.figure(figsize=(6.6, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        render(ax, nx, ny, mat, sg, P, Rm, th, LIMS)
        fig.tight_layout()
        fn = f"/tmp/_m3d_{k:02d}.png"
        fig.savefig(fn, dpi=130)
        plt.close(fig)
        files.append(fn)
        a0, b0 = rows[0][1], rows[0][2]
        aL, bL = rows[-1][1], rows[-1][2]
        print(f"  帧 {k:02d}: θ={deg:5.1f}°  根部 a={a0:.2f}/b={b0:.2f}  "
              f"末端 a={aL:.2f}/b={bL:.2f}  材料 {mat.mean()*100:.1f}%", flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_mesh3d.gif", save_all=True, append_images=ims[1:],
                duration=260, loop=0, optimize=True)
    print(f"saved anim_mesh3d.gif ({len(ims)} 帧)")


if __name__ == "__main__":
    main()
