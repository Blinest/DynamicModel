# -*- coding: utf-8 -*-
"""check_friction_continuation.py — 摩擦路径的 μ 续步（warm-start）

问题：`check_seg_ratio_mu.py` 里 μ>0 直接从冷启动求解，结果与 μ=0 基线不连续
      （μ=0.10 时 κ 跳到 131 1/m，μ=0.05 弯角只剩 9.7°）⇒ 求解器掉进了别的分支。

做法：μ 从 0 起，以步长 dμ 递增，**每一步用上一步的解 x0 当初值**（continuation）。
      同时记录总弯角、段1/段2 平均曲率、两者比值、迭代数。

用法：
    python tools/check_friction_continuation.py                 # 默认 μ: 0→0.20, dμ=0.02
    python tools/check_friction_continuation.py --mu-max 0.30 --dmu 0.01
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")

from tendon_coupling import build_arm, shooting, ALPHA_DEG   # noqa: E402

EI, EA = 0.01037, 7e3
G = (0.0, 0.0, 0.0)          # 先排除重力，只看摩擦


def make_params(mu, branch=1.0, nsub=20):
    ca, cb, p = build_arm(EI=EI, EA=EA, r_i=0.008, r_disk=0.005,
                          n_circ_a=3, n_circ_b=3, n_long=136, E=2e8,
                          t_strut=1.5e-3, rhoA=0.0, n_p=nsub, n_d=nsub,
                          lattice=False)
    p.mu_friction = float(mu)
    p.friction_branch = float(branch)
    p.g_world = G
    return p


def metrics(o):
    ang = float(np.degrees(np.arccos(np.clip(o["R_curve"][-1][2, 2], -1.0, 1.0))))
    S = np.asarray(o["s_curve"], float)
    U = np.asarray(o["u_curve"], float)[:, :2]
    k = np.linalg.norm(U, axis=1)
    m1 = S <= 0.2
    k1 = float(k[m1].mean()) if m1.any() else float("nan")
    k2 = float(k[~m1].mean()) if (~m1).any() else float("nan")
    return ang, k1, k2, (k1 / k2 if k2 > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-max", type=float, default=0.20)
    # ⚠ 2026-11-06 实测: dμ=0.02 时 μ≥0.02 打满 80 次 Newton 仍不收敛(κ 跳到 131/138 的垃圾解);
    #   dμ=0.005 时 μ=0.005/0.010/0.015/0.020 全部收敛(8/20/40/60 次)。
    #   与雅可比无关: TC_JAC_MODE=central 在 dμ=0.02 下同样不收敛 ⇒ 纯续步步长问题。
    ap.add_argument("--dmu", type=float, default=0.005)
    ap.add_argument("--branch", type=float, default=1.0)
    ap.add_argument("--T0", type=float, default=10.0)
    ap.add_argument("--A", type=float, default=3.4, help="差动幅值 N")
    ap.add_argument("--phi", type=float, default=90.0, help="弯曲方位 deg")
    ap.add_argument("--nsub", type=int, default=20)
    a = ap.parse_args()

    d = np.cos(np.radians(np.array(ALPHA_DEG, float)) - np.radians(a.phi))
    tau = a.T0 + a.A * d
    print(f"τ = T0({a.T0}) + A({a.A})·cos(α−{a.phi}°)  ⇒ τ∈[{tau.min():.2f},{tau.max():.2f}] N")
    print(f"EI={EI}, branch={a.branch}, nsub={a.nsub}, g=0\n")
    print("%6s %8s %10s %9s %9s %8s %8s %8s" %
          ("μ", "收敛", "弯角", "κ1", "κ2", "κ1/κ2", "iter", "耗时s"))

    mus = np.concatenate([np.arange(0.0, a.mu_max + 1e-12, a.dmu)])
    x0 = None
    ok_n = 0
    rows = []
    for mu in mus:
        p = make_params(mu, a.branch, a.nsub)
        t0 = time.time()
        o = shooting(tau, p, F_tip=np.zeros(3), g_world=G, x0=x0,
                     tol=1e-9, max_iter=80)
        dt = time.time() - t0
        ang, k1, k2, ratio = metrics(o)
        conv = bool(o["converged"])
        if conv:
            ok_n += 1
            x0 = np.asarray(o["x"], float)        # warm-start 下一步
        print("%6.2f %8s %9.2f° %9.3f %9.3f %8.3f %8s %8.1f" %
              (mu, "✓" if conv else "✗", ang, k1, k2, ratio,
               o.get("n_iter", "-"), dt), flush=True)
        rows.append((mu, conv, ang, k1, k2, ratio))

    print(f"\n收敛 {ok_n}/{len(mus)}")
    good = [r for r in rows if r[1]]
    if good:
        print("μ 最大收敛点: μ=%.2f  弯角 %.2f°  κ1/κ2=%.3f" % (good[-1][0], good[-1][2], good[-1][5]))
        print("μ=0 与 μ_max 的弯角变化: %.2f° → %.2f°  (%+.1f%%)" %
              (good[0][2], good[-1][2], (good[-1][2] / good[0][2] - 1) * 100 if good[0][2] else 0))
        print("κ1/κ2: %.3f → %.3f（模型理论值 2.000，实测 4.1–7.8）" % (good[0][5], good[-1][5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
