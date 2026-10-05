# -*- coding: utf-8 -*-
"""plot_friction.py — 肌腱摩擦 (Capstan) 效果图
==========================================================================================
新增物理项: 丝沿骨架转过包角 dφ=|κ|ds 时张力按

        T(s) = T0 · exp(-branch · μ · φ(s))                (branch=+1 加载 / -1 卸载)

衰减。μ=0 时与无摩擦版**逐位一致**(已对拍)。本脚本出 3 张图 (dpi=150):

    fig_friction_drift.png      ① 末端漂移随 μ  ② 对数纵轴
    fig_friction_decay.png      ③ 累积包角 φ(s)  ④ 张力衰减 T(s)/T0
    fig_friction_hysteresis.png ⑤ 滞回 (加载 vs 卸载)  ⑥ 环宽 = 2×加载偏移
    fig_friction_tau_z.png      ⑦ τ → 末端 z (不同 μ)

跑: python tools/plot_friction.py [--mu 0.15] [--dpi 150] [--nsub 100]
    (需 matplotlib; 无 numba 时自动走 TC_NO_JIT=1 纯 numpy 回退, 只是慢一些)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_numba() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


if not _has_numba():
    os.environ["TC_NO_JIT"] = "1"

from tendon_coupling import (  # noqa: E402
    TendonCouplingModel, forward_batch, shooting_batch, L1,
)

BASE_TAU = np.array([40.0, 20.0, 40.0, 20.0, 40.0, 10.0])   # 基准工况


def solve(tau, mu, branch=1.0, nsub=100, want_curve=False):
    """给定 τ (可批量) 求静力解 → (out, res, tip, curve)."""
    M = TendonCouplingModel()
    P = M.params
    P.n_p = nsub
    P.n_d = nsub
    P.mu_friction = float(mu)
    P.friction_branch = float(branch)
    tau = np.atleast_2d(np.asarray(tau, float))
    out = shooting_batch(tau, P, tol=1e-8, max_iter=12)
    fb = forward_batch(out["x"], tau, P, want_curve=want_curve)
    if want_curve:                      # want_curve=True → (res, tip, curves)
        res, tip, cv = fb
    else:
        res, tip, cv = fb[0], fb[1], None
    return out, res, tip, cv


def wrap_angle(u_curve, S):
    """由曲率曲线梯形积分包角 φ(s) = ∫|κ_xy| ds."""
    kappa = np.linalg.norm(np.asarray(u_curve, float)[:, :2], axis=1)
    ds = np.diff(S)
    return np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * ds)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.15, help="基准摩擦系数 (默认 0.15)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--nsub", type=int, default=100, help="每段子步数 (默认 100)")
    ap.add_argument("--mus", type=str, default="0,0.05,0.10,0.15,0.25,0.35",
                    help="漂移图的 μ 列表")
    a = ap.parse_args()
    n = a.nsub

    try:                                    # 有 matplotlib 就用它, 否则用 Pillow 兜底
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        print("[plot] backend = matplotlib")
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import simpleplot as plt           # 极简后端 (同款 subplots/plot/savefig 接口)
        print("[plot] backend = simpleplot (Pillow)")

    # ── ① / ② 末端漂移随 μ ──
    mus = [float(v) for v in a.mus.split(",")]
    tip_of = {}
    for mu in mus:
        tip_of[mu] = solve(BASE_TAU, mu, +1.0, nsub=n)[2][0]
    t0 = tip_of[0.0]
    dxy = np.array([np.linalg.norm(tip_of[m][:2] - t0[:2]) for m in mus]) * 1000.0
    dz = np.array([tip_of[m][2] - t0[2] for m in mus]) * 1000.0

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(mus, dxy, "o-", label="|dxy|")
    ax[0].plot(mus, dz, "s--", label="dz")
    ax[0].set_xlabel("Coulomb friction coefficient  mu")
    ax[0].set_ylabel("tip drift vs frictionless  [mm]")
    ax[0].set_title("(1) Tip drift caused by tendon friction")
    ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(mus, np.abs(dz), "o-", color="C0")
    ax[1].plot(mus, dxy, "s--", color="C1")
    ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[1].set_xlabel("mu"); ax[1].set_ylabel("|drift|  [mm]")
    ax[1].set_title("(2) Same data, log vertical axis")
    ax[1].grid(alpha=.3, which="both")
    fig.suptitle(f"Tendon friction (Capstan) effect on TCR model | "
                 f"tau={BASE_TAU.astype(int).tolist()} N, tau_max=60 N, "
                 f"n_p=n_d={n}", fontsize=9)
    fig.tight_layout(); fig.savefig("fig_friction_drift.png", dpi=a.dpi); plt.close(fig)
    print("saved fig_friction_drift.png")

    # ── ③ / ④ 包角与张力衰减剖面 (把摩擦打开才画得出来) ──
    _, _, _, cv = solve(BASE_TAU, a.mu, +1.0, nsub=n, want_curve=True)
    S = np.asarray(cv[1], float)
    Uc = np.asarray(cv[2], float)[0]                   # (nnode, 3) 曲率
    phi = wrap_angle(Uc, S)
    ratio_L = np.exp(-a.mu * phi)                      # 加载支
    ratio_U = np.exp(+a.mu * phi)                      # 卸载支

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(S * 1000, phi, lw=2, color="C0")
    ax[0].axvline(L1 * 1000, color="k", ls=":", lw=1)
    ax[0].text(L1 * 1000 + 4, phi.max() * 0.8, "segment A/B boundary", fontsize=8)
    ax[0].set_xlabel("s  [mm]")
    ax[0].set_ylabel("wrap angle  phi = int(|k|) ds   [rad]")
    ax[0].set_title("(3) Accumulated wrap angle phi(s)")
    ax[0].grid(alpha=.3)
    ax[1].plot(S * 1000, ratio_L * 100, lw=2, color="C0",
               label="loading   T/T0 = exp(-mu*phi)")
    ax[1].plot(S * 1000, ratio_U * 100, lw=2, ls="--", color="C3",
               label="unloading T/T0 = exp(+mu*phi)")
    ax[1].axvline(L1 * 1000, color="k", ls=":", lw=1)
    ax[1].set_xlabel("s  [mm]"); ax[1].set_ylabel("T(s)/T0   [%]")
    ax[1].set_title(f"(4) Tension decay along the arm  (mu={a.mu})")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle(f"Capstan model: total wrap angle = {phi[-1]:.4f} rad, "
                 f"loading end ratio = {ratio_L[-1]*100:.2f} %", fontsize=9)
    fig.tight_layout(); fig.savefig("fig_friction_decay.png", dpi=a.dpi); plt.close(fig)
    print(f"saved fig_friction_decay.png   total wrap = {phi[-1]:.4f} rad, "
          f"T_end/T0(loading) = {ratio_L[-1]*100:.3f} %")

    # ── ⑤ / ⑥ 滞回环 ──
    s_grid = np.linspace(0.35, 1.0, 8)
    TAU = BASE_TAU[None, :] * s_grid[:, None]
    tipL = solve(TAU, a.mu, +1.0, nsub=n)[2]
    tipU = solve(TAU, a.mu, -1.0, nsub=n)[2]
    tipF = solve(TAU, 0.0, +1.0, nsub=n)[2]
    zL = (tipL[:, 2] - tipF[:, 2]) * 1000.0
    zU = (tipU[:, 2] - tipF[:, 2]) * 1000.0

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].fill_between(s_grid, zU, zL, alpha=.22, color="purple", label="hysteresis band")
    ax[0].plot(s_grid, zL, "o-", color="C0", label="loading  (+mu)")
    ax[0].plot(s_grid, zU, "s--", color="C3", label="unloading (-mu)")
    ax[0].set_xlabel("tau scale factor"); ax[0].set_ylabel("dz vs frictionless [mm]")
    ax[0].set_title("(5) Hysteresis in tip z"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].plot(s_grid, np.abs(zL - zU), "^-", color="purple")
    ax[1].set_xlabel("tau scale factor"); ax[1].set_ylabel("loop width  [mm]")
    ax[1].set_title("(6) Loop width = 2 x loading offset")
    ax[1].grid(alpha=.3)
    fig.suptitle(f"Capstan hysteresis  ($\\mu$={a.mu}): max loop width = "
                 f"{np.max(np.abs(zL-zU)):.4f} mm", fontsize=9)
    fig.tight_layout(); fig.savefig("fig_friction_hysteresis.png", dpi=a.dpi); plt.close(fig)
    print(f"saved fig_friction_hysteresis.png   max loop width = {np.max(np.abs(zL-zU)):.4f} mm")

    # ── ⑦ τ → z (不同 μ) ──
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.4))
    for mu in (0.0, a.mu, 0.35):
        tp = solve(TAU, mu, +1.0, nsub=n)[2]
        ax.plot(np.linalg.norm(TAU, axis=1), tp[:, 2] * 1000, "o-", ms=3.5, label=f"mu={mu}")
    ax.set_xlabel("|tau|  [N]"); ax.set_ylabel("tip z  [mm]")
    ax.set_title("(7) Total tension vs axial shortening")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("fig_friction_tau_z.png", dpi=a.dpi); plt.close(fig)
    print("saved fig_friction_tau_z.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
