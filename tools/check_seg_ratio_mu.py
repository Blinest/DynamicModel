# -*- coding: utf-8 -*-
"""check_seg_ratio_mu.py — 段1/段2 曲率比 vs 摩擦系数 μ

背景: 模型给出段1:段2 曲率比恒为 2.000（段1 受 6 根丝力矩、段2 只受 B 段 3 根，
      Σcos² = 3.0 : 1.5），而实测报 4.1–7.8。怀疑远端张力被 Capstan 摩擦衰减，
      使段2 曲率进一步降低，比值被拉大。

做法: 固定命令（T0=10 N、差动 A=3 N、方位 180°），μ 从 0 扫到 0.30，
      比较段1/段2 平均曲率与总弯角。逐行 print(flush=True)。

用法: TC_NO_JIT=1 python tools/check_seg_ratio_mu.py [--nsub 20] [--A 3]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")
from tendon_coupling import build_arm, shooting, ALPHA_DEG   # noqa: E402

ALPHA = np.radians(np.array(ALPHA_DEG, float))


def one(mu, A, T0, nsub, g=(0.0, 0.0, 0.0)):
    ca, cb, p = build_arm(EI=0.01037, EA=7e3, r_i=0.008, r_disk=0.005,
                          n_circ_a=3, n_circ_b=3, n_long=136, E=2e8,
                          t_strut=1.5e-3, n_p=nsub, n_d=nsub, lattice=False,
                          mu_friction=mu)
    p.g_world = tuple(g)
    tau = T0 + A * np.cos(ALPHA - np.radians(180.0))
    o = shooting(tau, p, F_tip=np.zeros(3), g_world=g, tol=1e-8, max_iter=80)
    S = np.asarray(o["s_curve"])
    k = np.linalg.norm(np.asarray(o["u_curve"]), axis=1)
    m1, m2 = S <= p.L1, S > p.L1
    k1 = float(k[m1].mean()) if m1.any() else np.nan
    k2 = float(k[m2].mean()) if m2.any() else np.nan
    ang = float(np.degrees(np.arccos(np.clip(o["R_curve"][-1][2, 2], -1, 1))))
    return k1, k2, k2 / max(k1, 1e-12), ang, o["converged"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=20)
    ap.add_argument("--A", type=float, default=3.0)
    ap.add_argument("--T0", type=float, default=10.0)
    a = ap.parse_args()
    print(f"EI=0.01037  T0={a.T0} N  A={a.A} N  方位 180°  nsub={a.nsub}", flush=True)
    print("%6s %9s %9s %8s %8s %s" % ("mu", "段1 κ", "段2 κ", "比值", "弯角", ""), flush=True)
    for mu in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        try:
            k1, k2, r, ang, ok = one(mu, a.A, a.T0, a.nsub)
        except Exception as e:
            print("%6.2f  求解异常 %s" % (mu, type(e).__name__), flush=True)
            continue
        print("%6.2f %9.3f %9.3f %8.3f %7.1f° %s"
              % (mu, k1, k2, r, ang, "" if ok else "(未收敛)"), flush=True)
    print("\n实测参考: 比值 4.1–7.8", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
