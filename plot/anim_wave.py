# -*- coding: utf-8 -*-
"""anim_wave.py — 波浪形臂体动图（κ 沿程交替变号）

上：三维臂体（镂空菱形孔）
下：六路收丝量（波浪的指纹：丝长变化远小于同等弯角）

⚠ 可达性说明（必须写在标题里）
------------------------------
现有 6 根丝是**连续**的（A 段丝锚 200 mm、B 段丝锚 400 mm），
一个 `τ` 向量在根部只能给出一组力矩 ⇒ 曲率符号最多变 **1 次**（S 形）。
图上这种「4 个半波」需要**每段独立锚固**（20 段 × 2 = 40 个独立张力）。
因此本图是**形状演示 / 需求说明**，不是当前硬件可达解。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _d in ("geometry", "stiffness", "force", "plot"):
    _p = os.path.join(_root, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _root not in sys.path:
    sys.path.insert(0, _root)

import geom_params as G          # noqa: E402
import fig_arm_full as F         # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

NSEG = G.NSEG
DS_M = G.L_TOTAL / NSEG
N_SUB = 4
R_DISK = G.R_DISK
PSI = np.radians(G.PSI_DEG)
R_OUT = G.R_O
L_TOT = G.L_TOTAL
L_HALF = G.L1

GRID = 1.2
R_OUT_MM = R_OUT * 1e3
CIRC = 2 * np.pi * R_OUT_MM
B_MM = G.B
L_SEG_MM = G.L1 * 1e3
L_TOT_MM = L_TOT * 1e3
NHALF = 4                      # 半波数


def integrate(kvec):
    """κ 逐段给定（带符号，平面内），返回 (S, P, Rs, K)"""
    p = np.zeros(3); R = np.eye(3); s = 0.0
    P, Rs, S, K = [p.copy()], [R.copy()], [0.0], [np.zeros(3)]
    for i in range(NSEG):
        for _ in range(N_SUB):
            d = DS_M / N_SUB
            kv = np.array([0.0, kvec[i], 0.0])      # 绕 y 转 => 在 x-z 平面内弯
            p = p + (R @ np.array([0, 0, 1.0])) * d
            K_ = np.array([[0, -kv[2], kv[1]], [kv[2], 0, -kv[0]],
                           [-kv[1], kv[0], 0]])
            R = R @ (np.eye(3) + K_ * d + 0.5 * (K_ @ K_) * (d * d))
            u, _, vt = np.linalg.svd(R); R = u @ vt
            s += d
        P.append(p.copy()); Rs.append(R.copy()); S.append(s); K.append(kv)
    return np.array(S), np.array(P), np.array(Rs), np.array(K)


def wire_dl(S, P, Rs):
    """按各自跨度积分（A 段丝 200 mm / B 段丝 400 mm）。"""
    out = np.zeros(6)
    for w in range(6):
        span = L_HALF if w % 2 == 0 else L_TOT
        e = np.array([np.cos(PSI[w]), np.sin(PSI[w]), 0.0]) * R_DISK
        q = P + np.einsum("nij,j->ni", Rs, e)
        d = np.linalg.norm(np.diff(q, axis=0), axis=1)
        sm = 0.5 * (S[:-1] + S[1:])
        Lw = d[sm <= span + 1e-12].sum()
        out[w] = (span - Lw) * 1e3
    return np.maximum(0.0, out)


def build_mask():
    rows, s = [], 0.0
    while s < L_TOT_MM - 1e-6:
        alpha = 34.0 if s <= L_SEG_MM else 20.0
        a, b = F.ab_local(alpha, 0.0)
        rows.append((s, a, b)); s += 2 * a
    nx = int(CIRC / GRID) + 1
    ny = int(rows[-1][0] / GRID) + 2
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * GRID,
                         (np.arange(ny) + 0.5) * GRID)
    c = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(c), bool)
    for cy0, a, b in rows:
        alpha = 34.0 if cy0 <= L_SEG_MM else 20.0
        _, rho_ax, rho_sd = F.seg_params(cy0 + 1e-6)
        for fam in (0.0, 1.0):
            cy = cy0 + a + fam * a
            for j in range(-1, 7):
                for sh in (-CIRC, 0.0, CIRC):
                    p = F.hole_poly(j * 2 * B_MM + sh + fam * B_MM, cy,
                                    a, b, rho_ax, rho_sd)
                    sel = ((np.abs(c[:, 0] - p[:, 0].mean()) <=
                            (p[:, 0].max() - p[:, 0].min()) / 2 + 1) &
                           (np.abs(c[:, 1] - p[:, 1].mean()) <=
                            (p[:, 1].max() - p[:, 1].min()) / 2 + 1))
                    if sel.any():
                        inside[sel] |= Path(p).contains_points(c[sel])
    mat = ~inside.reshape(ny, nx)
    ys = (np.arange(ny) + 0.5) * GRID
    mat[np.abs(ys - L_SEG_MM) <= 6.0, :] = True
    return nx, ny, mat


def surf(P, Rs, S, s_m, theta, r):
    i = int(np.clip(np.searchsorted(S, s_m) - 1, 0, len(S) - 2))
    t = np.clip((s_m - S[i]) / max(S[i + 1] - S[i], 1e-12), 0, 1)
    p = (1 - t) * P[i] + t * P[i + 1]
    R = (1 - t) * Rs[i] + t * Rs[i + 1]
    return p + R @ np.array([r * np.cos(theta), r * np.sin(theta), 0.0])


def main():
    nx, ny, mat = build_mask()
    print(f"掩膜 {nx}x{ny} 材料 {mat.mean()*100:.1f}%")
    KMAX = 6.0
    NFR = 10
    files = []
    for kf in range(NFR):
        amp = KMAX * (kf + 1) / NFR
        q = NSEG // NHALF                         # 每半波段数 = 5
        kvec = np.array([amp * ((-1.0) ** (i // q)) for i in range(NSEG)])
        S, P, Rs, K = integrate(kvec)
        dl = wire_dl(S, P, Rs)

        fig = plt.figure(figsize=(9.0, 9.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.0], hspace=0.10)
        ax = fig.add_subplot(gs[0], projection="3d")
        quads, kk = [], []
        for k in np.where(mat.ravel())[0]:
            ix, iy = k % nx, k // nx
            x0, x1 = ix * GRID, (ix + 1) * GRID
            y0, y1 = iy * GRID, (iy + 1) * GRID
            quads.append([surf(P, Rs, S, yy * 1e-3, xx / R_OUT_MM, R_OUT + 0.05e-3)
                          for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
            si = min(int((0.5 * (y0 + y1)) // L_SEG_MM), NSEG - 1)
            kk.append(abs(kvec[si]))
        keep = [i for i, x in enumerate(quads) if np.isfinite(np.asarray(x)).all()]
        pc = Poly3DCollection([quads[i] for i in keep], edgecolor="none",
                              linewidth=0, antialiased=True)
        kk = np.array([kk[i] for i in keep])
        pc.set_facecolor(plt.cm.magma(np.clip(kk / KMAX, 0, 1)))
        ax.add_collection3d(pc)
        ax.set_xlim(-0.32, 0.32); ax.set_ylim(-0.32, 0.32)
        ax.set_zlim(-0.05, 0.45)
        ax.set_box_aspect((1, 1, 1), zoom=1.12)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=14, azim=-62)
        ax.set_title(f"波浪形 |κ|={amp:.1f} 1/m（{NHALF} 个半波）\n"
                     f"⚠ 每段独立驱动才可达，现有 6 丝只到 S 形",
                     fontsize=8.5, color=RED)

        axb = fig.add_subplot(gs[1])
        cols = [RED if i % 2 == 0 else GREEN for i in range(6)]
        axb.bar(range(6), dl, color=cols, alpha=0.92)
        for i in range(6):
            axb.text(i, dl[i] + 0.05, f"{dl[i]:.2f}", ha="center", fontsize=7,
                     color=cols[i])
        axb.set_ylim(0, max(3.0, dl.max() * 1.3))
        axb.set_xticks(range(6))
        axb.set_xticklabels([f"M{i+1}" for i in range(6)], fontsize=8)
        axb.set_ylabel("dL (mm)", fontsize=8)
        axb.grid(alpha=0.25, ls=":", axis="y")
        axb.set_title("六路收丝量（波浪态丝长几乎不变）", fontsize=8)

        fn = f"/tmp/_wv_{kf:02d}.png"
        fig.savefig(fn, dpi=100, bbox_inches="tight")
        plt.close(fig)
        files.append(fn)
        print(f"  帧{kf}: |κ|={amp:.2f}  dL={np.round(dl,2)}", flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_wave.gif", save_all=True, append_images=ims[1:],
                duration=380, loop=0, optimize=True)
    print(f"saved anim_wave.gif ({len(ims)} 帧)")


if __name__ == "__main__":
    main()
