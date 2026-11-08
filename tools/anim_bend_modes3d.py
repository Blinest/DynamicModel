# -*- coding: utf-8 -*-
"""anim_bend_modes3d.py — 四种弯曲模式的三维动图（一张图四栏同步生长）

模式由 12 段曲率场决定（seg12_model 基线权重 × 符号图样）：
    C / S / 反向S / 波浪
每帧把总弯角从 0 增到 90°，四栏同步；蓝色系=受拉、红色系=受压、灰=近零。

为了在可接受时间内跑完（4 栏 × 8 帧），网格用 1.0 mm（0.8 时约慢 1.6 倍）。
输出 anim_bend_modes3d.gif
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import rcParams
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TC_NO_JIT", "1")
import fig_arm_full as F        # noqa: E402
import anim_bend3d as M         # noqa: E402
import seg12_model as S12       # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

DS = 1.0
NFRAME = 8
THETA_MAX = 90.0
R_OUT_MM = F.R_OUT_MM
CIRC = F.CIRC
B = F.B
H_OFF = F.H_OFF
L_SEG = F.L_SEG
L_TOT = F.L_TOT
E_MOD = 2.0e8


def k12(mode, theta_deg):
    base = S12.kappa_vec(theta_deg, p=1.0)
    if mode == "C":
        return base
    if mode == "S":
        k = base.copy(); k[6:] = -np.abs(base[6:]); return k
    if mode == "S_rev":
        k = base.copy(); k[:6] = -np.abs(base[:6]); return k
    if mode == "wave":
        sgn = np.array([+1] * 3 + [-1] * 3 + [+1] * 3 + [-1] * 3, float)
        return base * sgn
    raise ValueError(mode)


def integrate(kvec, n_sub=10):
    """12 段曲率 → 骨架 p(s) 与姿态 R(s)（弯曲在 x–z 平面）。"""
    p = np.zeros(3)
    R = np.eye(3)
    P, Rs, S = [p.copy()], [R.copy()], [0.0]
    s = 0.0
    for kap in kvec:
        ds = (S12.SEG_LEN / n_sub)
        for _ in range(n_sub):
            p = p + (R @ np.array([0.0, 0.0, 1.0])) * ds
            th = kap * ds
            Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                           [-np.sin(th), 0, np.cos(th)]])
            R = R @ Ry
            s += ds
            P.append(p.copy()); Rs.append(R.copy()); S.append(s)
    return np.array(S), np.array(P), np.array(Rs)


def surf(P, Rs, S, s_m, theta, r):
    i = int(np.clip(np.searchsorted(S, s_m) - 1, 0, len(S) - 2))
    t = np.clip((s_m - S[i]) / max(S[i + 1] - S[i], 1e-12), 0, 1)
    p = (1 - t) * P[i] + t * P[i + 1]
    R = (1 - t) * Rs[i] + t * Rs[i + 1]
    return p + R @ np.array([r * np.cos(theta), r * np.sin(theta), 0.0])


def build_mask(kvec):
    """按 kvec 逐行重建掩膜（孔随局部 ε 变形）。"""
    S, P, Rs = integrate(kvec)   # integrate 返回 (S, P, Rs)
    # 行
    rows, s = [], 0.0
    while s < L_TOT - 1e-6:
        seg = min(int(s // L_SEG), 5)          # 12 段 → 索引
        kap = kvec[seg * 2 if s < L_SEG else 6 + seg]
        eps = -kap * M.R_OUT * (-1.0)          # 取内侧幅值（|ε|）
        alpha = M.ALPHA_A if s <= L_SEG else M.ALPHA_B
        a, b = F.ab_local(alpha, -abs(eps))
        rows.append((s, a, b))
        s += 2 * a
    nx = int(CIRC / DS) + 1
    ny = int(rows[-1][0] / DS) + 2
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * DS, (np.arange(ny) + 0.5) * DS)
    centers = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(centers), bool)
    for i, (cy0, a, b) in enumerate(rows):
        alpha = M.ALPHA_A if cy0 <= L_SEG else M.ALPHA_B
        _, rho_ax, rho_sd = F.seg_params(cy0 + 1e-6)
        for fam in (0.0, 1.0):
            cy = cy0 + a + fam * a
            for j in range(-1, 7):
                for sh in (-CIRC, 0.0, CIRC):
                    p = F.hole_poly(j * 2 * B + sh + fam * B, cy, a, b,
                                    rho_ax, rho_sd)
                    sel = ((np.abs(centers[:, 0] - p[:, 0].mean())
                            <= (p[:, 0].max() - p[:, 0].min()) / 2 + 1) &
                           (np.abs(centers[:, 1] - p[:, 1].mean())
                            <= (p[:, 1].max() - p[:, 1].min()) / 2 + 1))
                    if sel.any():
                        inside[sel] |= Path(p).contains_points(centers[sel])
    mat = ~inside.reshape(ny, nx)
    ys = (np.arange(ny) + 0.5) * DS
    mat[np.abs(ys - L_SEG) <= 6.0, :] = True
    return nx, ny, mat


def dl_of(kvec):
    """6 路驱动丝位移：A 段丝累计前 6 段转角，B 段丝累计全部。"""
    th = kvec * S12.SEG_LEN                # 每段转角 (rad)
    return M.R_DISK * np.sum(th[:6]) * 1e3, M.R_DISK * np.sum(th) * 1e3


def force_at(kap, s_m, theta_mm):
    alpha = M.ALPHA_A if s_m <= L_SEG else M.ALPHA_B
    A = (2.0e-3) * (1.5e-3)
    eps = -kap * M.R_OUT * np.cos(theta_mm / R_OUT_MM)
    return E_MOD * (eps * np.sin(np.radians(alpha))) * A


def render(ax, kvec, nx, ny, mat, title, lims):
    S, P, Rs = integrate(kvec)   # integrate 返回 (S, P, Rs)
    quads, fs = [], []
    for k in np.where(mat.ravel())[0]:
        ix, iy = k % nx, k // nx
        x0, x1 = ix * DS, (ix + 1) * DS
        y0, y1 = iy * DS, (iy + 1) * DS
        pts = [surf(P, Rs, S, yy * 1e-3, xx / R_OUT_MM, M.R_OUT + 0.05e-3)
               for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        quads.append(pts)
        sm = 0.5 * (y0 + y1)
        kap = kvec[min(int(sm // L_SEG) * 2, 5)] if sm <= L_SEG else \
            kvec[6 + min(int((sm - L_SEG) // L_SEG), 5)]
        fs.append(force_at(kap, sm, 0.5 * (x0 + x1)))
    keep = [i for i, q in enumerate(quads) if np.isfinite(np.asarray(q)).all()]
    quads = [quads[i] for i in keep]
    fs = np.array([fs[i] for i in keep])
    pc = Poly3DCollection(quads, edgecolor="none", linewidth=0, antialiased=True)
    pc.set_facecolor(F.force_colors(fs, FMAX_GL))
    ax.add_collection3d(pc)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=14, azim=-62)
    ax.set_title(title, fontsize=8.5, color=RED)


FMAX_GL = 20.0


def main():
    global FMAX_GL
    modes = [("C", "① C 形"), ("S", "② S 形"),
             ("S_rev", "③ 反向 S"), ("wave", "④ 波浪")]
    # 全局色标上限：取 C 形 90° 的 95 分位
    nx, ny, mat = build_mask(k12("C", THETA_MAX))
    S, P, Rs = integrate(k12("C", THETA_MAX))
    vals = []
    for k in np.where(mat.ravel())[0]:
        ix, iy = k % nx, k // nx
        vals.append(force_at(k12("C", THETA_MAX)[0], (iy + 0.5) * DS,
                             (ix + 0.5) * DS))
    FMAX_GL = max(np.percentile(np.abs(vals), 95), 1e-9)
    print(f"  全局色标上限 ±{FMAX_GL:.2f} N", flush=True)
    lims = ((-0.34, 0.34), (-0.34, 0.34), (-0.02, 0.45))

    files = []
    for kf, deg in enumerate(np.linspace(0, THETA_MAX, NFRAME)):
        fig = plt.figure(figsize=(18.0, 8.2))
        gs = fig.add_gridspec(2, 4, height_ratios=[2.6, 1.0],
                              hspace=0.16, wspace=0.06)
        kv_all = {}
        for j, (m, nm) in enumerate(modes):
            kv = k12(m, deg) if deg > 0 else np.zeros(12)
            kv_all[m] = kv
            nx, ny, mat = build_mask(kv)
            ax = fig.add_subplot(gs[0, j], projection="3d")
            render(ax, kv, nx, ny, mat, f"{nm}  θ={deg:.0f}°", lims)
            # 下方：6 根丝的位移
            axb = fig.add_subplot(gs[1, j])
            dlA, dlB = dl_of(kv)
            d6 = np.array([dlA, dlB, dlA, dlB, dlA, dlB])
            cols = [RED if k in (0, 2, 4) else GREEN for k in range(6)]
            axb.bar(range(6), d6, color=cols, alpha=0.92)
            for kk in range(6):
                axb.text(kk, d6[kk] + 0.15, f"{d6[kk]:.1f}", ha="center",
                         fontsize=6.5,
                         color=RED if kk in (0, 2, 4) else GREEN)
            axb.set_ylim(0, 11)
            axb.set_xticks(range(6))
            axb.set_xticklabels([f"M{k+1}" for k in range(6)], fontsize=7)
            axb.tick_params(labelsize=6)
            axb.grid(alpha=0.25, ls=":", axis="y")
            if j == 0:
                axb.set_ylabel("ΔL (mm)", fontsize=8)
            axb.set_title(f"{nm}：6 路位移（红=A 段/绿=B 段）", fontsize=7.5)
        fig.suptitle("四种弯曲模式的三维动图（红=受压/蓝=受拉/灰≈零）", fontsize=10)
        fig.tight_layout()
        fn = f"/tmp/_bm_{kf:02d}.png"
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        files.append(fn)
        print(f"  帧 {kf}: θ={deg:4.0f}°", flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_bend_modes3d.gif", save_all=True, append_images=ims[1:],
                duration=320, loop=0, optimize=True)
    print(f"saved anim_bend_modes3d.gif ({len(ims)} 帧)")


if __name__ == "__main__":
    main()
