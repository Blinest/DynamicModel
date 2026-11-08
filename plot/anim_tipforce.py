# -*- coding: utf-8 -*-
"""anim_tipforce.py — 尖端横向力从 0 增到 50 mN 的动图

三栏：
  ① 3D 臂体（镂空，按斜撑轴力着色）
  ② 20 段曲率 κ(s) 柱状
  ③ 尖端力辨识：真值 vs 由形状反求出的 P（加 1 mm 噪声）

复用 anim_bend_modes3d 的掩膜与渲染（DS=1.3 mm）。
输出 anim_tipforce.gif / .mp4
"""
# ── 包路径引导（分层目录）────────────────────────────────────
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('geometry', 'stiffness', 'force', 'plot'):
    _p = _os.path.join(_ROOT, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
# ──────────────────────────────────────────────────────────────


import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TC_NO_JIT", "1")
import anim_bend_modes3d as A          # noqa: E402
import casadi_tipforce as T            # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

NFRAME = 12
P_MAX = 0.05        # 50 mN


def main():
    ei = T.seg_ei()
    A.FMAX_GL = 12.0                     # 受力色标上限（N）
    lims = ((-0.34, 0.34), (-0.34, 0.34), (-0.02, 0.45))
    files = []
    for kf, P in enumerate(np.linspace(0, P_MAX, NFRAME)):
        kv = np.zeros(T.NSEG) if P <= 0 else T.kappa_of_P(P, ei)
        nx, ny, mat = A.build_mask(kv)
        fig = plt.figure(figsize=(15.0, 5.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.0, 0.9], wspace=0.22)

        ax = fig.add_subplot(gs[0, 0], projection="3d")
        A.render(ax, kv, nx, ny, mat, "", lims)

        axk = fig.add_subplot(gs[0, 1])
        axk.bar(range(T.NSEG), kv, color=BLUE, alpha=0.9)
        axk.axhline(0, color="0.5", lw=0.8)
        axk.set_ylim(0, 1.6)
        axk.set_xlabel("segment", fontsize=8)
        axk.set_ylabel("kappa (1/m)", fontsize=8)
        axk.tick_params(labelsize=7)
        axk.grid(alpha=0.25, ls=":", axis="y")
        axk.set_title("kappa(s)", fontsize=9)

        # 辨识
        r, _, _ = T.identify(P if P > 0 else 1e-4, 1.0, ei)
        axp = fig.add_subplot(gs[0, 2])
        axp.bar([0, 1], [P * 1e3, r["P"] * 1e3],
                color=[GREEN, RED], width=0.6)
        axp.set_xticks([0, 1])
        axp.set_xticklabels(["true", "fit"], fontsize=8)
        axp.set_ylim(0, 60)
        axp.set_ylabel("P (mN)", fontsize=8)
        axp.tick_params(labelsize=7)
        axp.grid(alpha=0.25, ls=":", axis="y")
        err = abs(r["P"] - P) / P * 100 if P > 0 else 0.0
        axp.set_title(f"P = {P*1e3:.1f} mN\nfit err {err:.1f}%", fontsize=9)
        for j, v in enumerate([P * 1e3, r["P"] * 1e3]):
            axp.text(j, v + 1.2, f"{v:.1f}", ha="center", fontsize=7)

        fn = f"/tmp/_tf_{kf:02d}.png"
        fig.savefig(fn, dpi=95, bbox_inches="tight")
        plt.close(fig)
        files.append(fn)
        print(f"  frame {kf}: P={P*1e3:5.1f} mN  拟合 {r['P']*1e3:6.2f} "
              f"({err:4.1f}%)", flush=True)

    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_tipforce.gif", save_all=True, append_images=ims[1:],
                duration=260, loop=0, optimize=True)
    print(f"saved anim_tipforce.gif ({len(ims)} 帧)")


if __name__ == "__main__":
    main()
