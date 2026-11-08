# -*- coding: utf-8 -*-
"""anim_bend_twist.py — 既弯又扭的两段菱形臂体动图

上：三维臂体（镂空菱形孔，按斜撑轴力着色）
下：六路驱动丝位移（扭转的签名是**反对称**的）

扭转的实现
----------
弯曲平面沿弧长**旋转**：曲率方向角 φ(s) = φ₀ + k_tw · φ_accum(s)
其中 `k_tw` 单位为「度 / 弧度弯角」，`φ_accum` 为已累计的弯曲角。
于是 κ(s) 是**转动中的二维矢量** ⇒ 臂体既弯又扭。

丝位移按**真实几何**算（不再用解析式）：
    L_w = ∫ |d/ds [ p(s) + R(s)·r·e(ψ_w) ]| ds ,   ΔL_w = L_w(0) − L_w(s)
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
import cosserat_lattice as CL    # noqa: E402
import fig_arm_full as F         # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

EST = CL.arrays()
EI = EST["EI"]
NSEG, DS_M, L_TOT = EST["NSEG"], None, EST["L"]
R_DISK = G.R_DISK
PSI = np.radians(G.PSI_DEG)
T0 = 10.0
R_OUT = G.R_O
N_SUB = 3
DS = L_TOT / NSEG / N_SUB

# 网格（掩膜用，单位 mm）
GRID = 1.2
R_OUT_MM = R_OUT * 1e3
CIRC = 2 * np.pi * R_OUT_MM
B_MM = G.B
L_SEG_MM = G.L1 * 1e3
L_TOT_MM = L_TOT * 1e3
W_HALF = 0.75

KMAX_F = 12.0      # 受力色标上限 N


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def integrate(A, phi0_deg, tw_deg_per_rad, force_dir=0.0):
    """返回 (s[m], p[N+1,3], R[N+1,3,3], kap[N+1,3], tau_knots[N,6])"""
    p = np.zeros(3); R = np.eye(3)
    phi_acc = 0.0
    P, Rs, S, K, TAUS = [p.copy()], [R.copy()], [0.0], [np.zeros(3)], []
    s = 0.0
    for i in range(NSEG):
        for _ in range(N_SUB):
            ang = np.radians(phi0_deg) + np.radians(tw_deg_per_rad) * phi_acc
            # 弯曲方向随扭转旋转
            kap_mag = None
            # 该段所需力矩（按锚点筛丝）
            anchor = np.where(np.arange(6) % 2 == 0, L_TOT * 0.5, L_TOT)
            mask = (anchor > s + 1e-12).astype(float)
            tau = np.maximum(0.0, T0 + A * np.cos(PSI - np.radians(phi0_deg)))
            ts = tau * mask
            Mx = -R_DISK * np.sum(ts * np.sin(PSI))
            My = R_DISK * np.sum(ts * np.cos(PSI))
            kap = np.array([Mx, My]) / EI[i]
            # 按扭转角旋转二维曲率
            c, sn = np.cos(ang), np.sin(ang)
            kv = np.array([kap[0] * c - kap[1] * sn,
                           kap[0] * sn + kap[1] * c, 0.0])
            p = p + (R @ np.array([0, 0, 1.0])) * DS
            K_ = hat(kv)
            R = R @ (np.eye(3) + K_ * DS + 0.5 * K_ @ K_ * DS * DS)
            u, _, vt = np.linalg.svd(R); R = u @ vt
            phi_acc += np.linalg.norm(kv) * DS
            s += DS
        P.append(p.copy()); Rs.append(R.copy()); S.append(s)
        K.append(kv); TAUS.append(tau)
    return np.array(S), np.array(P), np.array(Rs), np.array(K), np.array(TAUS)


def wire_dl(S, P, Rs):
    """由几何算六路收丝量（mm，只能拉）。

    ⚠ A 段丝只到 200 mm 锚点、B 段丝到 400 mm ⇒ **必须按各自跨度积分**，
      否则 A 段丝会被算成 2 倍（踩过）。
    """
    out = np.zeros(6)
    half = L_TOT * 0.5
    for w in range(6):
        span = half if w % 2 == 0 else L_TOT     # M1/M3/M5 = A段
        e = np.array([np.cos(PSI[w]), np.sin(PSI[w]), 0.0]) * R_DISK
        q = P + np.einsum("nij,j->ni", Rs, e)    # (N+1,3)
        d = np.linalg.norm(np.diff(q, axis=0), axis=1)
        sm = 0.5 * (S[:-1] + S[1:])
        Lw = d[sm <= span + 1e-12].sum()
        out[w] = (span - Lw) * 1e3               # 缩短量 = 收丝
    return np.maximum(0.0, out)


def build_mask(kvec):
    """按 (s, θ材料) 空间重建掩膜；扭转不影响掩膜拓扑。"""
    rows, s = [], 0.0
    while s < L_TOT_MM - 1e-6:
        alpha = 34.0 if s <= L_SEG_MM else 20.0
        a, b = F.ab_local(alpha, 0.0)
        rows.append((s, a, b)); s += 2 * a
    nx = int(CIRC / GRID) + 1
    ny = int(rows[-1][0] / GRID) + 2
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * GRID,
                         (np.arange(ny) + 0.5) * GRID)
    c_all = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(c_all), bool)
    for cy0, a, b in rows:
        alpha = 34.0 if cy0 <= L_SEG_MM else 20.0
        _, rho_ax, rho_sd = F.seg_params(cy0 + 1e-6)
        for fam in (0.0, 1.0):
            cy = cy0 + a + fam * a
            for j in range(-1, 7):
                for sh in (-CIRC, 0.0, CIRC):
                    p = F.hole_poly(j * 2 * B_MM + sh + fam * B_MM, cy,
                                    a, b, rho_ax, rho_sd)
                    sel = ((np.abs(c_all[:, 0] - p[:, 0].mean()) <=
                            (p[:, 0].max() - p[:, 0].min()) / 2 + 1) &
                           (np.abs(c_all[:, 1] - p[:, 1].mean()) <=
                            (p[:, 1].max() - p[:, 1].min()) / 2 + 1))
                    if sel.any():
                        inside[sel] |= Path(p).contains_points(c_all[sel])
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
    nx, ny, mat = build_mask(None)
    print(f"掩膜 {nx}x{ny}  材料 {mat.mean()*100:.1f}%")
    A_fix = 4.9
    NFR = 10
    files = []
    for kf, tw in enumerate(np.linspace(0.0, 20.0, NFR)):
        # 前 5 帧弯+扭一起长，后 5 帧只加扭
        A = A_fix * min(1.0, (kf + 1) / 5.0)
        S, P, Rs, K, TAUS = integrate(A, 0.0, tw)
        dl = wire_dl(S, P, Rs)

        fig = plt.figure(figsize=(9.0, 9.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.0], hspace=0.10)
        ax = fig.add_subplot(gs[0], projection="3d")

        quads, fs = [], []
        for k in np.where(mat.ravel())[0]:
            ix, iy = k % nx, k // nx
            x0, x1 = ix * GRID, (ix + 1) * GRID
            y0, y1 = iy * GRID, (iy + 1) * GRID
            q = [surf(P, Rs, S, yy * 1e-3, xx / R_OUT_MM, R_OUT + 0.05e-3)
                 for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            quads.append(q)
            seg = min(int(0.5 * (y0 + y1) // L_SEG_MM), NSEG - 1)
            fs.append(float(np.linalg.norm(K[min(seg, len(K) - 1)][:2])))
        keep = [i for i, q in enumerate(quads) if np.isfinite(np.asarray(q)).all()]
        quads = [quads[i] for i in keep]
        kk = np.array([fs[i] for i in keep])
        pc = Poly3DCollection(quads, edgecolor="none", linewidth=0,
                              antialiased=True)
        pc.set_facecolor(plt.cm.magma(np.clip(kk / KMAX_F, 0, 1)))
        ax.add_collection3d(pc)
        ax.set_xlim(-0.34, 0.34); ax.set_ylim(-0.34, 0.34)
        ax.set_zlim(-0.05, 0.45)
        ax.set_box_aspect((1, 1, 1), zoom=1.15)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=16, azim=-62)
        ax.set_title(f"A={A:.1f} N   twist={tw:.0f} deg/rad   "
                     f"kappa_root={np.linalg.norm(K[1][:2]):.2f} 1/m",
                     fontsize=9, color=RED)

        axb = fig.add_subplot(gs[1])
        cols = [RED if i % 2 == 0 else GREEN for i in range(6)]
        axb.bar(range(6), dl, color=cols, alpha=0.92)
        for i in range(6):
            axb.text(i, dl[i] + 0.08, f"{dl[i]:.2f}", ha="center", fontsize=7,
                     color=cols[i])
        axb.set_ylim(0, max(8.0, dl.max() * 1.25))
        axb.set_xticks(range(6))
        axb.set_xticklabels([f"M{i+1}" for i in range(6)], fontsize=8)
        axb.set_ylabel("dL (mm)", fontsize=8)
        axb.grid(alpha=0.25, ls=":", axis="y")
        axb.set_title("六路收丝量（扭转 => 反对称）", fontsize=8)

        fn = f"/tmp/_bt_{kf:02d}.png"
        fig.savefig(fn, dpi=100, bbox_inches="tight")
        plt.close(fig)
        files.append(fn)
        print(f"  帧{kf}: A={A:.2f} twist={tw:.0f}  dL={np.round(dl,2)}",
              flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_bend_twist.gif", save_all=True, append_images=ims[1:],
                duration=420, loop=0, optimize=True)
    print(f"saved anim_bend_twist.gif ({len(ims)} 帧)")


if __name__ == "__main__":
    main()
