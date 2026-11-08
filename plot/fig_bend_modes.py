# -*- coding: utf-8 -*-
"""fig_bend_modes.py — 12 段曲率场下的不同弯曲模式

曲率场就用 **seg12_model 的 12 段离散**（段长 33.33 mm）：
    C 形      κ 全部同号（基线权重，摩擦根部集中）
    S 形      后 6 段取反 ⇒ 段界(第 6/7 段之间)受力换号
    反向 S    前 6 段取反
    波浪      κ 每 3 段换一次号 ⇒ 多点反弯
直接叠加到 seg12 的基线权重上，保持"弯曲根部集中"这一物理特征。

受力：ε = −κ_i·R·cosθ，F = E·(ε·sinα)·A
      受压 → 深红，受拉 → 深蓝，近零 → 浅灰
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
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import seg12_model as S12   # noqa: E402
import cosserat_lattice as CL   # noqa: E402

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"

R_MM = 8.0
E_MOD = 2.0e8
T_MM = 1.5
N_SUB = 12
THETA_TOTAL = 90.0      # 总弯角（弧度 = 12 段 κ·段长 之和）


def k12(mode):
    """返回 12 段曲率（1/m），保持 |Σκ_i·段长| = 总弯角。"""
    base = S12.kappa_vec(THETA_TOTAL, p=1.0)      # 全同号、根部集中
    n = len(base)
    h = n // 2
    if mode == "C":
        return base
    if mode == "S":
        k = base.copy(); k[h:] = -np.abs(base[h:]); return k
    if mode == "S_rev":
        k = base.copy(); k[:h] = -np.abs(base[:h]); return k
    if mode == "wave":
        q = max(1, n // 4)
        sign = np.array([(-1.0) ** (i // q) for i in range(n)], float)
        return base * sign
    raise ValueError(mode)


def integrate(kvec):
    p = np.zeros(2)
    th = 0.0
    P, K = [p.copy()], []
    for i, kap in enumerate(kvec):
        ds = (S12.SEG_LEN / N_SUB)
        for _ in range(N_SUB):
            p = p + np.array([np.sin(th), np.cos(th)]) * ds
            th += kap * ds
            P.append(p.copy())
            K.append(kap)
    return np.array(P) * 1e3, np.array(K)


def panel(ax, mode, title):
    kv = k12(mode)
    P, K = integrate(kv)
    d = np.gradient(P, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    n = np.stack([d[:, 1], -d[:, 0]], axis=1)
    segs, cols = [], []
    for off, th_rel in ((+1, 0.0), (-1, np.pi)):
        w = P + n * (R_MM * off)
        for i in range(len(w) - 1):
            alpha = CL.SEG[0]["alpha"] if (i / len(w)) * 0.4 <= 0.2 else CL.SEG[1]["alpha"]
            A = (2.0e-3) * (T_MM * 1e-3)
            eps = -K[i] * (R_MM * 1e-3) * np.cos(th_rel)
            F = E_MOD * (eps * np.sin(np.radians(alpha))) * A
            segs.append([w[i], w[i + 1]])
            cols.append("#67000d" if F < -0.5 else ("#08306b" if F > 0.5
                                                   else "#dddddd"))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=4.2,
                                     capstyle="round"))
    ax.plot(P[:, 0], P[:, 1], color="0.55", lw=0.8, ls="--")
    ax.plot(P[:1, 0], P[:1, 1], "s", color="k", ms=7)
    ax.plot(P[-1, 0], P[-1, 1], "o", color=RED, ms=7)
    ax.set_aspect("equal")
    ax.set_xlim(-280, 280)
    ax.set_ylim(-200, 480)
    ax.grid(alpha=0.15, ls=":")
    ax.set_title(title, fontsize=9, color=RED)
    ax.tick_params(labelsize=7)

    # κ(s) 阶梯图（12 段）
    axk = ax.inset_axes([0.58, 0.04, 0.38, 0.24])
    xs = np.arange(len(kv))
    axk.bar(xs, kv, color=[RED if v < 0 else BLUE for v in kv], alpha=0.85)
    axk.axhline(0, color="0.5", lw=0.7)
    axk.set_title(f"{len(kv)} 段 κ (1/m)", fontsize=7)
    axk.tick_params(labelsize=6)
    axk.set_xticks([0, len(kv)//2, len(kv)-1])
    return kv


def main():
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.8))
    names = [("C", "① C 形（12 段全同号）"),
             ("S", "② S 形（后 6 段取反）"),
             ("S_rev", "③ 反向 S（前 6 段取反）"),
             ("wave", "④ 波浪（每 3 段换号）")]
    for ax, (m, t) in zip(axes, names):
        kv = panel(ax, m, t)
        print(f"  {m:6s} κ = {np.round(kv, 2)}")
        print(f"         总弯角 Σκ·段长 = {np.degrees(kv.sum()*S12.SEG_LEN):.1f}°")
    fig.suptitle("12 段曲率场下的四种弯曲模式（红=受压/蓝=受拉/灰≈零；虚线=中线；"
                 "下方柱状=各段 κ）", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_bend_modes.png", dpi=150, bbox_inches="tight")
    print("saved fig_bend_modes.png")


if __name__ == "__main__":
    main()
