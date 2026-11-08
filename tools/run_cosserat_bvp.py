# -*- coding: utf-8 -*-
"""run_cosserat_bvp.py — 跑完整 Cosserat 打靶 BVP，与 v2 简化模型对比

背景：v2 是"力矩驱动 + 闭式积分"（3.5 ms/点）；完整模型要解 BVP（无 numba 时 17 s/点）。
本脚本只跑少量点做**交叉验证**：
    用 v2 反解出 τ（目标 θ、φ）→ 喂给完整打靶 → 比较 tip / ΔL

两套刚度：
  A) v1 标定值  EI=0.01037, EA=7000
  B) v2 等效力  EI=(0.01499+0.01266)/2=0.01383, EA=(6013+10568)/2=8291
     （完整模型只支持单段 EI，取两段平均做近似）
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdcr_model as T1        # noqa: E402
import tdcr_model_v2 as V2     # noqa: E402

PSI = np.radians([0, 60, 120, 180, 240, 300])


def tau_for(m2, th, ph):
    half = m2.N // 2
    comp = half * m2.ds / m2.EI[0] + (m2.N - half) * m2.ds / m2.EI[-1]
    A0 = np.radians(th) / (3 * m2.g["r_disk"] * comp) if th > 0 else 0.0
    lo, hi = max(0.0, 0.3 * A0), max(1e-9, 2.5 * A0)
    tau = np.maximum(0.0, m2.T0 * np.ones(6))
    for _ in range(16):
        A = 0.5 * (lo + hi)
        tau = np.maximum(0.0, m2.T0 + A * np.cos(PSI - np.radians(ph)))
        b = m2.bend_angle(tau)
        if b < th:
            lo = A
        else:
            hi = A
        if abs(b - th) < 0.15:
            break
    return tau


def main():
    m2 = V2.TdcrModelV2(n_sub=2)
    points = [(30, 0), (60, 0), (90, 0), (120, 0), (90, 60), (90, 180)]
    cfgs = [("v1标定 EI=0.01037/EA=7000", dict(EI=0.01037, EA=7000.0)),
            ("v2等效 EI=0.01383/EA=8291", dict(EI=0.01383, EA=8291.0))]

    print("=" * 86)
    print("Cosserat 打靶 BVP vs v2 简化模型（g=0，无摩擦）")
    print("=" * 86)
    for name, kw in cfgs:
        print(f"\n【{name}】")
        print("  θ   φ  |  τmax | v2 tip(mm)      | BVP tip(mm)     | 差(mm)"
              " | v2 ΔL | BVP ΔL | 用时")
        m1 = T1.TdcrModel(**kw)
        for th, ph in points:
            tau = tau_for(m2, th, ph)
            o2 = m2.solve(tau); t2 = o2["p"][-1] * 1e3
            d2 = m2.delta_L(tau).max()
            t0 = time.time()
            try:
                o1 = m1.solve(tau, g_world=(0, 0, 0))
                t1v = o1["p_curve"][-1] * 1e3
                d1 = np.abs(m1.delta_L(tau, g_world=(0, 0, 0))).max()
                dt = time.time() - t0
                dd = np.linalg.norm(t1v - t2)
                print(f" {th:3d} {ph:3d} | {tau.max():5.1f} | ({t2[0]:6.1f},{t2[2]:6.1f}) |"
                      f" ({t1v[0]:6.1f},{t1v[2]:6.1f}) | {dd:6.2f} |"
                      f" {d2:5.2f} | {d1:5.2f} | {dt:4.1f}s")
            except Exception as e:
                print(f" {th:3d} {ph:3d} | {tau.max():5.1f} | 求解失败: "
                      f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
