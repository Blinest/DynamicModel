# -*- coding: utf-8 -*-
"""anim_dl.py — 不同臂体位形对应的驱动丝位移量（动图）

每一帧：给定目标弯角 θ（0→180°，方位固定），用 tdcr_control 反解 6 路张力，
再由模型给出每根丝的收放量 ΔL，画：
  左：臂体侧视剪影（Φ16）+ 6 根丝投影
  右：6 个电机的 ΔL 柱状图（红=A 段 0/2/4，绿=B 段 1/3/5）+ 数值
输出 anim_dl.gif
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tdcr_control as C   # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_O = 8e-3
R_DISK = 5e-3
L1 = L2 = 0.2
NFRAME = 16
DIR = 0.0          # 方位：向上


def shape2d(k1):
    """两段常曲率（κ2=κ1/2）骨架，返回 p(mm) 与面内法向。"""
    n = 40
    p = np.zeros(2)
    th = 0.0
    P, N = [p.copy()], [np.array([1.0, 0.0])]
    for kap, Lseg in ((k1, L1), (k1 / 2, L2)):
        ds = Lseg / n
        for _ in range(n):
            p = p + np.array([np.sin(th), np.cos(th)]) * ds
            th += kap * ds
            P.append(p.copy())
            N.append(np.array([np.cos(th), -np.sin(th)]))
    return np.array(P) * 1e3, np.array(N)


def main():
    files, trail = [], []
    for k, deg in enumerate(np.linspace(0, 180, NFRAME)):
        tau, A, phi_cmd = C.solve_tension(deg, DIR)
        theta, phi_m, k1, k2, dL, sT = C.predict(tau)
        fig, (ax, axb) = plt.subplots(1, 2, figsize=(12, 5.2),
                                      gridspec_kw=dict(width_ratios=[1.15, 1]))
        # 左：位形
        P, N = shape2d(k1 if k1 > 0 else 1e-9)
        up = P + N * (R_O * 1e3)
        dn = P - N * (R_O * 1e3)
        ax.fill(np.concatenate([up[:, 0], dn[::-1, 0]]),
                np.concatenate([up[:, 1], dn[::-1, 1]]),
                color=BLUE, alpha=0.26, lw=0)
        ax.plot(up[:, 0], up[:, 1], color=BLUE, lw=1.4)
        ax.plot(dn[:, 0], dn[:, 1], color=BLUE, lw=1.4)
        for w in range(6):
            a = np.radians(60 * w)
            off = np.cos(a) * (R_DISK * 1e3)
            ax.plot(P[:, 0] + off * N[:, 0], P[:, 1] + off * N[:, 1],
                    color=RED if w in (0, 2, 4) else GREEN, lw=0.9, alpha=0.85)
        ax.plot(P[:1, 0], P[:1, 1], "s", color="k", ms=7)
        ax.plot(P[-1, 0], P[-1, 1], "o", color=RED, ms=7, zorder=6)
        # 末端轨迹（到此帧为止）
        trail.append((P[-1, 0], P[-1, 1]))
        if len(trail) > 1:
            tx, tz = np.array(trail).T
            ax.plot(tx, tz, "-", color=GREEN, lw=1.6, alpha=0.9, zorder=5)
            ax.plot([tx[-1]], [tz[-1]], "o", color=GREEN, ms=4, zorder=7)
        ax.annotate(f"({P[-1,0]:.0f}, {P[-1,1]:.0f}) mm", (P[-1, 0], P[-1, 1]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8.5,
                    color=GREEN)
        ax.set_aspect("equal")
        ax.set_xlim(-300, 300); ax.set_ylim(-60, 470)
        ax.grid(alpha=0.18, ls=":")
        ax.set_xlabel("横向 x (mm)"); ax.set_ylabel("高度 z (mm)")
        ax.set_title(f"θ = {theta:.1f}°   末端 ({P[-1,0]:.0f}, {P[-1,1]:.0f}) mm\nκ1={k1:.2f}  κ2={k2:.2f} 1/m（绿线=末端轨迹）", fontsize=9.5)
        # 右：ΔL 柱
        cols = [RED if w in (0, 2, 4) else GREEN for w in range(6)]
        bars = axb.bar(range(6), dL, color=cols, alpha=0.9)
        for w, b in enumerate(bars):
            axb.text(b.get_x() + b.get_width() / 2, dL[w] + 0.5,
                     f"{dL[w]:.2f}", ha="center", fontsize=8)
        axb.axhline(0, color="0.5", lw=0.8)
        axb.set_xticks(range(6))
        axb.set_xticklabels([f"M{w+1}\n{'A' if w in (0,2,4) else 'B'}" for w in range(6)],
                            fontsize=8)
        axb.set_ylim(0, 30)
        axb.grid(alpha=0.25, ls=":", axis="y")
        axb.set_ylabel("驱动丝收丝量 ΔL (mm)")
        axb.set_title(f"6 路位移（ΣT = {sT:.0f} N，τ {tau.min():.1f}–{tau.max():.1f} N）",
                      fontsize=9, color=RED)
        fig.tight_layout()
        fn = f"/tmp/_dl_{k:03d}.png"
        fig.savefig(fn, dpi=100)
        plt.close(fig)
        files.append(fn)
        if k % 5 == 0 or k == NFRAME - 1:
            print(f"  帧 {k:02d}: θ={theta:6.1f}°  ΔL(A)={dL[0]:5.2f}  "
                  f"ΔL(B)={dL[1]:5.2f}  ΣT={sT:.0f}N", flush=True)
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    ims[0].save("anim_dl.gif", save_all=True, append_images=ims[1:],
                duration=200, loop=0, optimize=True)
    print(f"saved anim_dl.gif ({len(ims)} 帧)")


if __name__ == "__main__":
    main()
