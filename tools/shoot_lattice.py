# -*- coding: utf-8 -*-
"""shoot_lattice.py — 让打靶 BVP 真正用上**菱形晶胞（分段 α）**

之前打靶用 `lattice=False` + 单一 EI，完全没体现菱形。
现在改为 `cells=(cellA, cellB)`：
    cellA: α=34°，L=200 mm   →  EI0=0.01499, C0=EI0/L_strut, k_rod=6013
    cellB: α=20°，L=200 mm   →  EI0=0.01266,                  k_rod=10568
EI0 / k_rod 取自 `cosserat_lattice`（几何推 + 标定 c）；α 与 L_strut 保证
`EI(κ)=EI0·sinc(κ·L_strut)` 的菱形非线性形式，段界刚度跳变也由 cells 体现。

输出：与 v2 简化模型对拍（tip / ΔL）。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosserat_lattice as CL          # noqa: E402
from tendon_coupling import LatticeCell, build_arm, shooting  # noqa: E402
import tdcr_model_v2 as V2             # noqa: E402

PSI = np.radians([0, 60, 120, 180, 240, 300])
E_MAT, A_STRUT = 2.0e8, 3.0e-6         # 2.0 mm × 1.5 mm
R_I = 8e-3
L1 = L2 = 0.2


def make_cells():
    ks = CL.stiffness()
    out = []
    for k, alpha in zip(ks, (34.0, 20.0)):
        c = LatticeCell(E=E_MAT, A=A_STRUT, N_circ=3, r_i=R_I,
                        alpha_deg=alpha, L=L1)
        # 用几何推+标定得到的 EI0 覆盖晶胞式（晶胞式偏硬 2.7×）
        c.EI0 = float(k["EI"])
        c.C0 = c.EI0 / c.L_strut
        c.GJ = float(k["GJ"])
        c.k_rod = float(k["EA"])
        out.append(c)
        print(f"  cell α={alpha:4.1f}°  EI0={c.EI0:.5f}  C0={c.C0:.3f}  "
              f"k_rod={c.k_rod:.0f}  L_strut={c.L_strut*1e3:.2f} mm")
    return tuple(out)


def build(lattice=True):
    ca, cb, p = build_arm(alpha_deg=34.0, lattice=lattice, n_p=30, n_d=30)
    cells = make_cells()
    p.cells = cells
    p.EI = cells[0].EI0
    p.C0 = cells[0].C0
    p.L_strut = cells[0].L_strut
    p.GJ = cells[0].GJ
    p.k_rod = cells[0].k_rod
    p.GA = 1e3
    p.rhoA = 0.0
    return p


def tau_for(m2, th, ph):
    half = m2.N // 2
    comp = half * m2.ds / m2.EI[0] + (m2.N - half) * m2.ds / m2.EI[-1]
    A0 = np.radians(th) / (3 * m2.g["r_disk"] * comp) if th > 0 else 0.0
    lo, hi = max(0.0, 0.3 * A0), max(1e-9, 2.5 * A0)
    tau = np.full(6, m2.T0)
    for _ in range(16):
        A = 0.5 * (lo + hi)
        tau = np.maximum(0.0, m2.T0 + A * np.cos(PSI - np.radians(ph)))
        if m2.bend_angle(tau) < th:
            lo = A
        else:
            hi = A
        if abs(m2.bend_angle(tau) - th) < 0.15:
            break
    return tau


def main():
    m2 = V2.TdcrModelV2(n_sub=2)
    print("菱形晶胞（分段 α，EI0 取 cosserat_lattice）：")
    p = build(lattice=True)
    print(f"  params: EI={p.EI:.5f} C0={p.C0:.3f} L_strut={p.L_strut*1e3:.2f}mm "
          f"GJ={p.GJ:.5f}\n")

    print(" θ   φ | τmax | v2 tip(mm)      | 菱形BVP tip(mm) | 差(mm) | v2ΔL | BVPΔL | 用时")
    for th, ph in [(30, 0), (60, 0), (90, 0), (90, 60)]:
        tau = tau_for(m2, th, ph)
        o2 = m2.solve(tau); t2 = o2["p"][-1] * 1e3
        d2 = m2.delta_L(tau).max()
        t0 = time.time()
        try:
            o1 = shooting(tau, p, g_world=(0, 0, 0))
            t1v = o1["p_curve"][-1] * 1e3
            from tendon_coupling import _wire_geom_len_batch, _L0_wire
            Lg = _wire_geom_len_batch(o1["p_curve"][None], o1["R_curve"][None],
                                      r_wire=p.r_disk, L1_geo=p.L1,
                                      L_total=p.L_total)[0]
            d1 = np.abs((Lg - _L0_wire(p)) * 1e3).max()
            dt = time.time() - t0
            print(f" {th:3d} {ph:3d} | {tau.max():4.1f} | ({t2[0]:6.1f},{t2[2]:6.1f}) |"
                  f" ({t1v[0]:6.1f},{t1v[2]:6.1f}) | {np.linalg.norm(t1v-t2):6.2f} |"
                  f" {d2:5.2f} | {d1:5.2f} | {dt:4.1f}s")
        except Exception as e:
            print(f" {th:3d} {ph:3d} | {tau.max():4.1f} | 失败 {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
