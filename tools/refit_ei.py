# -*- coding: utf-8 -*-
"""refit_ei.py — 用"水平自重 + 端载"的实测下垂角重标 EI

背景（2026-10-30）：`forward` 曾把分布重力写死成 `[0,0,-ρg]`，**忽略 `g_world`**
⇒ 之前标 `EI=0.045` 时，模型里根本没有自重，只用端载 20 g 就凑出了 20°。
重力方向修好后，同样 `EI=0.045` 会给出 **24.4°**（多 22%）⇒ 必须重标。

跑：`TC_NO_JIT=1 python tools/refit_ei.py [--target 20]`

输出：给出目标下垂角对应的 `EI`，并与当前表内 `EI` 对比、量化曲率偏差。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")

from tendon_coupling import build_arm, shooting      # noqa: E402

G_H = (-9.81, 0.0, 0.0)          # 水平安装：臂沿 +z，重力沿 −x
W_TIP = 0.020 * 9.81             # 端载 20 g
MASS = 0.020                     # 臂自重 20 g
L = 0.400


def sag_deg(EI, n=25):
    ca, cb, p = build_arm(EI=EI, EA=7e3, rhoA=MASS / L, n_p=n, n_d=n)
    p.g_world = G_H
    o = shooting(np.full(6, 0.01), p, F_tip=np.array([-W_TIP, 0.0, 0.0]),
                 g_world=G_H, tol=1e-9, max_iter=80)
    return float(np.degrees(np.arccos(np.clip(o["R_curve"][-1, 2, 2], -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=20.0, help="实测下垂角 [deg]")
    ap.add_argument("--ei-now", type=float, default=0.045, help="当前表内 EI")
    a = ap.parse_args()

    print("水平安装：自重 20 g（0.05 kg/m）+ 端载 20 g，找给出目标下垂角的 EI")
    print(f"{'EI':>8} {'下垂角':>9}")
    grid = [0.040, 0.045, 0.052, 0.058, 0.062, 0.070, 0.080]
    pts = []
    for EI in grid:
        d = sag_deg(EI)
        pts.append((EI, d))
        print(f"{EI:8.3f} {d:8.2f}°")

    # 单调递减 ⇒ 线性插值反解
    eis = np.array([p[0] for p in pts]); angs = np.array([p[1] for p in pts])
    order = np.argsort(angs)
    EI_fit = float(np.interp(a.target, angs[order], eis[order]))
    print(f"\n⇒ 目标 {a.target:.1f}° 对应 EI = {EI_fit:.4f} N·m²")
    print(f"   当前表内 EI = {a.ei_now:.4f} ⇒ 偏差 {100*(a.ei_now/EI_fit-1):+.1f}%")
    print(f"   即：同 τ 下当前表预测的曲率是真实值的 {EI_fit/a.ei_now:.2f} 倍"
          f"（表偏软 {100*(EI_fit/a.ei_now-1):.0f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
