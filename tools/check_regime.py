# -*- coding: utf-8 -*-
"""check_regime.py — 判定臂体处于"梁"区还是"索"区

判据
----
· 弯曲特征长度 l = sqrt(EI/N_axial)，N_axial 为轴向压力
  l >> L  => 梁区，κ = M/EI 成立
  l << L  => 索区，形状由张力平衡 + 屈曲模态决定，梁本构失效
· P-δ 放大因子 1/(1 − N/P_cr)：≥1 为放大，<0 表示已过屈曲（线性理论失效）

做法
----
用**完整 Cosserat**（tendon_coupling，含轴向力 n 的平衡）解 90°，读回：
  · 轴向应变 ε(s) = n_z/k_rod
  · 臂体缩短量
  · 与 v2（纯弯矩，无 P-δ）的末端差
"""
import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("geometry", "stiffness", "force", "plot"):
    _p = os.path.join(_root, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _root not in sys.path:
    sys.path.insert(0, _root)

import cosserat_lattice as CL                      # noqa: E402
import geom_params as G                            # noqa: E402
from tendon_coupling import LatticeCell, build_arm, shooting  # noqa: E402
import tdcr_model_v2 as V2                         # noqa: E402
import time

PSI = np.radians(G.PSI_DEG)
T0 = 10.0


def make_params():
    ks = CL.stiffness()
    cells = []
    for k, alpha in zip(ks, (G.SEG[0]["alpha_deg"], G.SEG[1]["alpha_deg"])):
        c = LatticeCell(E=G.E_MOD, A=G.SEG[0]["w"] * G.T_WALL, N_circ=G.N_CIRC,
                        r_i=G.R_O, alpha_deg=alpha, L=G.L1)
        c.EI0 = float(k["EI"]); c.C0 = c.EI0 / c.L_strut
        c.GJ = float(k["GJ"]); c.k_rod = float(k["EA"])
        cells.append(c)
    _, _, p = build_arm(alpha_deg=G.SEG[0]["alpha_deg"], lattice=False,
                        n_p=30, n_d=30)
    p.cells = tuple(cells)
    p.EI = cells[0].EI0; p.C0 = cells[0].C0
    p.L_strut = cells[0].L_strut; p.GJ = cells[0].GJ
    p.k_rod = cells[0].k_rod; p.GA = 1e3; p.rhoA = 0.0
    return p


def main():
    m2 = V2.TdcrModelV2(n_sub=3)
    print("=== 梁/索判据（几何 + 预紧决定，与求解器无关）===")
    for i, k in enumerate(CL.stiffness()):
        n_taut = 6 if i == 0 else 3
        N = n_taut * T0
        l = np.sqrt(k["EI"] / N)
        Pcr = np.pi ** 2 * k["EI"] / (G.L1 ** 2)
        amp = 1.0 / (1.0 - N / Pcr) if abs(N - Pcr) > 1e-9 else np.inf
        print(f"  {k['tag']}: N_axial={N:.0f} N  l={l*1e3:.1f} mm  "
              f"(L=200) L/l={G.L1/l:.1f}  P_cr={Pcr:.2f} N  "
              f"P-δ={amp:+.2f}")
    print("  => P-δ<0 表示线性梁理论已过屈曲，不能直接用\n")

    print("=== 完整 Cosserat 实测（90°，φ=0）===")
    p = make_params()
    tau = np.maximum(0.0, T0 + 4.9 * np.cos(PSI))
    print(f"  τ = {np.round(tau,2)}  Στ = {tau.sum():.1f} N（含预紧）")
    t0 = time.time()
    try:
        o = shooting(tau, p, g_world=(0, 0, 0))
        dt = time.time() - t0
        Pc = np.asarray(o["p_curve"]) * 1e3
        # 弧长 vs 真实长度
        seg = np.linalg.norm(np.diff(Pc, axis=0), axis=1)
        print(f"  求解 {dt:.1f} s  末端 ({Pc[-1,0]:.1f}, {Pc[-1,2]:.1f}) mm")
        print(f"  中线总长 {seg.sum():.2f} mm  （几何 400 ⇒ 缩短 "
              f"{400-seg.sum():.2f} mm, 应变 {(400-seg.sum())/400*100:.2f}%）")
        # 切向角沿程
        d = np.gradient(Pc, axis=0)
        ang = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
        idx = np.linspace(0, len(ang) - 1, 8).astype(int)
        print(f"  切向角沿程: {np.round(ang[idx],1).tolist()}")
        print(f"  单调增 ⇒ {bool(np.all(np.diff(ang) >= -1e-6))}"
              f"  （若出现回落 = 反号曲率 = S 形）")
        o2 = m2.solve(tau)
        t2 = o2["p"][-1] * 1e3
        print(f"  v2 末端 ({t2[0]:.1f}, {t2[2]:.1f}) mm  "
              f"差 {np.linalg.norm([t2[0]-Pc[-1,0], t2[2]-Pc[-1,2]]):.1f} mm")
    except Exception as e:
        print(f"  求解失败 {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
