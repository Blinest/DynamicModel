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
    """用 180° 帧的骨架包围盒作为全帧固定视野。"""
    ss, kf = M.kappa_profile(np.pi)
    sg, P, Rm = M.build_frames(kf, ss)
    pmin, pmax = P.min(0), P.max(0)
    ctr = 0.5 * (pmin + pmax)
    half = 0.5 * float(np.max(pmax - pmin)) + 0.032   # 留足边距，180° 卷回时不切边
    return ((ctr[0] - half, ctr[0] + half),
            (ctr[1] - half, ctr[1] + half),
            (ctr[2] - half, ctr[2] + half))


def render_fixed(ax, theta_total, nx, ny, mat, title, lims):
    ss, kf = M.kappa_profile(theta_total)
    sg, P, Rm = M.build_frames(kf, ss)
    quads = []
    for k in np.where(mat.ravel())[0]:
        ix, iy = k % nx, k // nx
        x0, x1 = ix * F.DS, (ix + 1) * F.DS
        y0, y1 = iy * F.DS, (iy + 1) * F.DS
        pts = [M.surface(P, Rm, sg, yy * 1e-3, xx / F.R_OUT_MM,
                         M.R_OUT + 0.03e-3)
               for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        quads.append(pts)
    quads = [q for q in quads if np.isfinite(np.asarray(q)).all()]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    pc = Poly3DCollection(quads, facecolor=F.RED, edgecolor="none",
                          linewidth=0, antialiased=True)
    ax.add_collection3d(pc)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_box_aspect((1, 1, 1), zoom=1.12)   # 相机后拉，保证整臂在画面内
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=14, azim=-62)
    ax.set_title(title, fontsize=9, color=F.RED)


def main():
    LIMS = compute_lims()
    files = []
    for k, deg in enumerate(np.linspace(0, 180, NFRAME)):
        th = np.radians(deg)
        nx, ny, mat = F.build_mask(th)
        fig = plt.figure(figsize=(6.6, 6.4))
        ax = fig.add_subplot(111, projection="3d")
        render_fixed(ax, th, nx, ny, mat,
                     f"全长 400 mm   θ={deg:.0f}°   "
                     f"材料 {mat.mean()*100:.0f}%", LIMS)
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
