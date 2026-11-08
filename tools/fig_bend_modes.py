# -*- coding: utf-8 -*-
"""fig_bend_modes.py — 不同弯曲模式（C 形 / S 形 / 反向 S / 螺旋）

弯曲模式由**曲率场** κ(s) 决定（段1 4.93、段2 2.92 1/m 为基准，见 fig_unwrap_forces）：
    ① C 形      κ1=+4.93, κ2=+2.92              单向弯
    ② S 形      κ1=+4.93, κ2=−2.92              两段反向
    ③ 反向 S    κ1=−4.93, κ2=+2.92
    ④ 螺旋      |κ| 同 C 形，但**弯曲方位沿 s 旋转** 120°

受力：ε = −κ(s)·R·cos(θ−φ(s))，F = E·(ε·sinα)·A
      受压（F<0）→ 深红；受拉（F>0）→ 深蓝；零 → 近白
输出 fig_bend_modes.png（4 栏：左边中线 + 内外壁受力色带，右上 κ(s) 曲线）
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cosserat_lattice as CL   # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_MM = 8.0
E_MOD = 2.0e8
T_MM = 1.5
N = 400
L1 = 200.0
L2 = 200.0


def kappa_modes():
    ks = CL.stiffness()
    r = 0.5 * ks[0]["EI"] / ks[1]["EI"]
    k1 = (np.pi / 2) / (0.2 + 0.2 * r)
    k2 = k1 * r
    return k1, k2


def integrate(k_of_s, azim_of_s=None):
    """按 κ(s) 与弯曲方位 φ(s) 积分骨架，返回 p(N+1,2) mm 与切向角。"""
    s = np.linspace(0.0, L1 + L2, N + 1)
    p = np.zeros(2)
    P = [p.copy()]
    th = 0.0
    for i in range(N):
        ds = (s[i + 1] - s[i]) * 1e-3
        kap = k_of_s(s[i])
        p = p + np.array([np.sin(th), np.cos(th)]) * ds
        th += kap * ds
        P.append(p.copy())
    return s, np.array(P) * 1e3


def force_at(kap, alpha_deg, theta_rel):
    A = (2.0e-3) * (T_MM * 1e-3)
    eps = -kap * (R_MM * 1e-3) * np.cos(theta_rel)
    return E_MOD * (eps * np.sin(np.radians(alpha_deg))) * A


def panel(ax, k_of_s, title, kk):
    s, P = integrate(k_of_s)
    # 面内法向
    d = np.gradient(P, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    n = np.stack([d[:, 1], -d[:, 0]], axis=1)     # 左法向
    segs, cols, lws = [], [], []
    for k, off in ((0, +1), (1, -1)):              # 外壁 / 内壁
        w = P + n * (R_MM * off)
        for i in range(len(w) - 1):
            kap = k_of_s(s[i])
            alpha = CL.SEG[0]["alpha"] if s[i] <= L1 else CL.SEG[1]["alpha"]
            F = force_at(kap, alpha, 0.0 if off > 0 else np.pi)
            segs.append([w[i], w[i + 1]])
            cols.append("#67000d" if F < -0.5 else ("#08306b" if F > 0.5
                                                    else "#dddddd"))
            lws.append(4.0)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws,
                                     capstyle="round"))
    ax.plot(P[:, 0], P[:, 1], color="0.55", lw=0.8, ls="--")
    ax.plot(P[:1, 0], P[:1, 1], "s", color="k", ms=7)
    ax.plot(P[-1, 0], P[-1, 1], "o", color=RED, ms=7)
    ax.set_aspect("equal")
    ax.set_xlim(-260, 260)
    ax.set_ylim(-120, 480)
    ax.grid(alpha=0.15, ls=":")
    ax.set_title(title, fontsize=9, color=RED)
    ax.tick_params(labelsize=7)

    # κ(s) 小图
    axk = ax.inset_axes([0.60, 0.04, 0.36, 0.24])
    sm = np.linspace(0, L1 + L2, 200)
    axk.plot(sm, [k_of_s(v) for v in sm], color=BLUE, lw=1.6)
    axk.axhline(0, color="0.6", lw=0.7)
    axk.set_title("κ(s)", fontsize=7)
    axk.tick_params(labelsize=6)
    axk.grid(alpha=0.2, ls=":")
    return np.degrees(kk)


def main():
    k1, k2 = kappa_modes()
    print(f"基准：κ1={k1:.2f}  κ2={k2:.2f} 1/m（C 形总弯角约 90°）")

    def c_mode(s):
        return k1 if s <= L1 else k2

    def s_mode(s):
        return k1 if s <= L1 else -k2

    def s_rev(s):
        return -k1 if s <= L1 else k2

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    panel(axes[0], c_mode, "① C 形（单向弯）\n内外壁同号", (k1 + k2))
    panel(axes[1], s_mode, "② S 形（两段反向）\n段界处受力换向", (k1 - k2))
    panel(axes[2], s_rev, "③ 反向 S（先反向再正向）", (-k1 + k2))
    fig.suptitle("三种弯曲模式的受力示意（红=受压 / 蓝=受拉 / 灰=近零）；"
                 "虚线=中线", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_bend_modes.png", dpi=160, bbox_inches="tight")
    print("saved fig_bend_modes.png")
    print("\n各模式特征：")
    print("  C 形  : 全程内侧受压、外侧受拉，段界处力阶跃（α 变、κ 变）")
    print("  S 形  : 段界两侧**受力符号相反** ⇒ 段界处是受力零点，弯矩要过零")
    print("  反向S : 与 S 形镜像")
    print("\nS 形对结构的意义：段界附近弯矩换号 ⇒ 那一带接近零应力，"
          "但弯曲集中反而更严重（曲率绝对值和变大）")


if __name__ == "__main__":
    main()
