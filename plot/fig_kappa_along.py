# -*- coding: utf-8 -*-
"""fig_kappa_along.py — 臂体曲率沿长度的分布

κ(s) = M(s)/EI(s)：
  · μ=0 时 M 沿程恒定 ⇒ κ 只有两段（段2 更软 ⇒ κ 更大），是台阶
  · μ>0 时 M 随累计转角衰减 ⇒ κ 从根部向末端**递减**
左右两栏对比 μ=0 与 μ=0.15。
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


import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdcr_model_v2 as V2

rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
RED, BLUE, GREEN = "#A34D4E", "#516CA3", "#8DAE7F"


def main():
    targets = [30, 60, 90, 120, 150, 180]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, mu in zip(axes, (0.0, 0.15)):
        m = V2.TdcrModelV2(mu=mu, n_sub=4)
        for th, col in zip(targets, plt.cm.viridis(np.linspace(0, .9, len(targets)))):
            A, b, tau = m.solve_bend(th)
            o = m.solve(tau)
            s = o["s"] * 1e3
            k = np.linalg.norm(o["kappa"][:, :2], axis=1)
            ax.plot(s, k, color=col, lw=1.8, label=f"θ={b:.0f}°")
        ax.axvline(200, color="0.5", ls="--", lw=0.9)
        ax.text(203, 0.2, "段界 200 mm", fontsize=7, color="0.4")
        ax.set_xlabel("弧长 s (mm)", fontsize=8)
        ax.set_title(f"μ = {mu:.2f}", fontsize=9, color=RED)
        ax.grid(alpha=0.25, ls=":")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("曲率 κ (1/m)", fontsize=8)
    fig.suptitle("臂体曲率沿长度：参数 τ=10 N 预紧，3 根同拉", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_kappa_along.png", dpi=150, bbox_inches="tight")
    print("saved fig_kappa_along.png")
    # 数值摘要
    for mu in (0.0, 0.15):
        m = V2.TdcrModelV2(mu=mu, n_sub=4)
        A, b, tau = m.solve_bend(90.0)
        o = m.solve(tau)
        k = np.linalg.norm(o["kappa"][:, :2], axis=1)
        print(f"  mu={mu:.2f}  90度: 根部 κ={k[1]:.3f}  末端 κ={k[-1]:.3f}  "
              f"比 {k[-1]/k[1]:.2f}  平均 {k.mean():.3f} 1/m")


if __name__ == "__main__":
    main()
