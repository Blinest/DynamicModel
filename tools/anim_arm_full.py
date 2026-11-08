# -*- coding: utf-8 -*-
"""anim_arm_full.py — 完整臂形动图（全长 400 mm，θ 0→180°）

复用 fig_arm_full 的掩膜/渲染（逐行压缩变形 + 段界厚壁筒 + 包围盒预筛），
每帧重建掩膜，相机坐标固定（取 180° 帧的包围盒）。

输出 anim_arm_full.gif
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TC_NO_JIT", "1")
import fig_arm_full as F   # noqa: E402
import anim_bend3d as M    # noqa: E402

NFRAME = 10


def compute_lims():
    """固定视野：**基座贴底**。

    z 取 [-0.02, 0.45]（基座在下方 20 mm 处），x/y 取横向可达 ±0.34；
    另给 box_aspect 按数据范围成比例，否则 (1,1,1) 会把 z 拉伸、基座又跑到中间。
    """
    return ((-0.34, 0.34), (-0.34, 0.34), (-0.02, 0.45))


def aspect_of(lims):
    dx = lims[0][1] - lims[0][0]
    dy = lims[1][1] - lims[1][0]
    dz = lims[2][1] - lims[2][0]
    m = max(dx, dy, dz)
    return (dx / m, dy / m, dz / m)


# ── 受力色标：深蓝(受拉) → 浅灰(不受力) → 深红(受压) ──────────
from matplotlib.colors import LinearSegmentedColormap as _LSC
CMAP_F = _LSC.from_list("fg", ["#08306b", "#d9d9d9", "#67000d"])
GRAY_F = (0.85, 0.85, 0.85, 1.0)
F_DEAD = 0.02          # |F|/FMAX < 2% 视为不受力 => 浅灰


def force_colors(comps, fmax):
    t = np.clip(comps / max(fmax, 1e-12), -1.0, 1.0)
    cols = CMAP_F((1.0 - t) / 2.0)
    cols[np.abs(t) < F_DEAD] = GRAY_F
    return cols


def axial_force(theta_mm, s_mm, ss, kf):
    """该处斜撑的带符号轴力 (N)：F>0 受拉、F<0 受压（+x 为凹侧，ε=-κRcosθ）。"""
    kap = float(np.interp(s_mm * 1e-3, ss, kf))
    a_deg = M.ALPHA_A if s_mm <= F.L_SEG else M.ALPHA_B
    A = (2.0e-3) * (1.5e-3)
    eps = -kap * M.R_OUT * np.cos(theta_mm / F.R_OUT_MM)
    return 2.0e8 * (eps * np.sin(np.radians(a_deg))) * A


def probe_fmax(theta_total):
    """用目标角最大的一帧取 95 分位力幅，作为**全帧共用**的色标上限。"""
    nx, ny, mat = F.build_mask(theta_total)
    ss, kf = M.kappa_profile(theta_total)
    vals = []
    for k in np.where(mat.ravel())[0]:
        ix, iy = k % nx, k // nx
        vals.append(axial_force((ix + 0.5) * F.DS, (iy + 0.5) * F.DS, ss, kf))
    return max(np.percentile(np.abs(vals), 95), 1e-9)


def render_fixed(ax, theta_total, nx, ny, mat, title, lims, fmax):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm
    ss, kf = M.kappa_profile(theta_total)
    sg, P, Rm = M.build_frames(kf, ss)
    quads, comps = [], []
    for k in np.where(mat.ravel())[0]:
        ix, iy = k % nx, k // nx
        x0, x1 = ix * F.DS, (ix + 1) * F.DS
        y0, y1 = iy * F.DS, (iy + 1) * F.DS
        pts = [M.surface(P, Rm, sg, yy * 1e-3, xx / F.R_OUT_MM,
                         M.R_OUT + 0.03e-3)
               for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        quads.append(pts)
        comps.append(axial_force(0.5 * (x0 + x1), 0.5 * (y0 + y1), ss, kf))
    keep = [i for i, q in enumerate(quads) if np.isfinite(np.asarray(q)).all()]
    quads = [quads[i] for i in keep]
    comps = np.array([comps[i] for i in keep])
    pc = Poly3DCollection(quads, edgecolor="none", linewidth=0, antialiased=True)
    pc.set_facecolor(force_colors(comps, fmax))
    ax.add_collection3d(pc)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_box_aspect(aspect_of(lims), zoom=1.05)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=14, azim=-62)
    ax.set_title(f"{title}   峰值 ±{np.abs(comps).max():.1f} N "
                 f"（深红=受压 / 深蓝=受拉）", fontsize=9, color=F.RED)


def main():
    LIMS = compute_lims()
    FMAX = probe_fmax(np.radians(150))
    print(f'  色标上限（150° 帧的 95 分位）: ±{FMAX:.2f} N')
    files = []
    for k, deg in enumerate(np.linspace(0, 150, NFRAME)):   # 180° 时末端卷到基座下方，基座无法置底
        th = np.radians(deg)
        nx, ny, mat = F.build_mask(th)
        fig = plt.figure(figsize=(6.6, 6.4))
        ax = fig.add_subplot(111, projection="3d")
        render_fixed(ax, th, nx, ny, mat,
                     f"全长 400 mm   θ={deg:.0f}°", LIMS, FMAX)
        fig.tight_layout()
        fn = f"/tmp/_af_{k:02d}.png"
        fig.savefig(fn, dpi=130)
        plt.close(fig)
        files.append(fn)
        print(f"  帧 {k:02d}: θ={deg:5.1f}°  材料 {mat.mean()*100:.1f}%", flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_arm_full.gif", save_all=True, append_images=ims[1:],
                duration=260, loop=0, optimize=True)
    print(f"saved anim_arm_full.gif ({len(ims)} 帧)")


if __name__ == "__main__":
    main()
