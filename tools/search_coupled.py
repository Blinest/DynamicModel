# -*- coding: utf-8 -*-
"""search_coupled.py — 用肌腱力耦合方程搜索解（快速版）

耦合方程（比 v2 多两项）
    ① 弯矩耦合：M = r·Σ τ_j·(−sinψ_j, cosψ_j)
    ② **轴向压缩**：腱张力把臂体压短   n_z = −Στ_j，  ε = n_z/EA
    ③ 摩擦：τ_j(s) = τ_j·exp(−μ·b·φ(s))
    κ = M/EI(s),  R' = R·hat(κ),  p' = R·e_z·(1+ε)

丝长变化（这才是"耦合"的核心）
    ΔL_j = ∫[ ε(s) − r·κ(s)·cos(ψ_j−φ_b) ] ds
    ⇒ ΔL 比纯弯曲多出 ε·L 这一项（轴向压缩），正是之前 3.9 vs 10.5 mm 缺的那块

搜索：给 (θ, φ)，对幅值 A 二分，使弯角命中 θ。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cosserat_lattice as CL  # noqa: E402

PSI = np.radians(np.array([0, 60, 120, 180, 240, 300], float))
NSEG = 20
L_TOT = 0.400
R_DISK = 0.005


def _hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class CoupledModel:
    """兼容薄壳：内部就是 TdcrModelV2(axial=True)。

    统一入口在 tdcr_model_v2，这里只保留旧名以免打断已有脚本。
    """
    def __init__(self, EA_scale=1.0, mu=0.0, T0=10.0, n_sub=3):
        import tdcr_model_v2 as V2
        self._m = V2.TdcrModelV2(mu=mu, T0=T0, n_sub=n_sub, axial=True)
        if EA_scale != 1.0:                    # 缩放 EA 用于灵敏度分析
            self._m.EA = self._m.EA * float(EA_scale)
        self.EI = self._m.EI; self.EA = self._m.EA
        self.ds = self._m.ds; self.mu = mu; self.T0 = T0
        self.n_sub = n_sub

    def solve(self, tau, F_tip=None):
        return self._m.solve(tau, F_tip=F_tip)

    def bend(self, tau, **kw):
        return self._m.bend_angle(tau, **kw)

    def dL(self, tau, **kw):
        """返回 (弯曲项, 轴向项) 供对比；合计 = 弯曲 + 轴向。"""
        d = self._m.delta_L(tau, **kw)
        o = self._m.solve(tau, **kw)
        phi = o["phi"]; phiA, phiB = phi[self._m.N // 2], phi[-1]
        km = o["kappa"][:, :2].sum(0)
        phi_b = np.arctan2(-km[0], km[1]) if np.any(np.abs(km) > 1e-12) else 0.0
        bend_only = np.zeros(6); ax_only = np.zeros(6)
        eps = -(-float(np.sum(tau - self._m.T0)) / float(self._m.EA.mean()))
        half_len = self._m.g["L_total"] * 0.5
        for w, ps in enumerate(PSI):
            span_is_A = (w % 2 == 0)
            span = phiA if span_is_A else phiB
            bend_only[w] = R_DISK * span * np.cos(ps - phi_b) * 1e3
            ax_only[w] = eps * (half_len if span_is_A else self._m.g["L_total"]) * 1e3
        return bend_only, ax_only


def solve_A(m, th, ph, it=16):
    comp = (NSEG // 2) * m.ds / m.EI[0] + (NSEG - NSEG // 2) * m.ds / m.EI[-1]
    A0 = np.radians(th) / (3.0 * R_DISK * comp) if th > 0 else 0.0
    lo, hi = max(0.0, 0.3 * A0), max(1e-9, 2.5 * A0)
    tau = np.maximum(0.0, m.T0 + 0.0 * np.cos(PSI))
    b = 0.0
    for _ in range(it):
        A = 0.5 * (lo + hi)
        tau = np.maximum(0.0, m.T0 + A * np.cos(PSI - np.radians(ph)))
        b = m.bend(tau)
        if b < th:
            lo = A
        else:
            hi = A
        if abs(b - th) < 0.1:
            break
    return tau, b


def main():
    print("=" * 72)
    print("A. 轴向耦合对收丝量的影响（目标 90°，φ=0）")
    print("=" * 72)
    print("  EA 缩放 | τ(电机侧) | 弯角  | 弯曲项 | 轴向项 | 合计 ΔL_M1")
    for sc in (1.0, 0.5, 0.2, 0.1):
        m = CoupledModel(EA_scale=sc, n_sub=3)
        tau, b = solve_A(m, 90.0, 0.0)
        db, dax = m.dL(tau)
        print(f"   {sc:5.2f}   |  {tau[0]:7.2f}  | {b:5.1f}°| {db[0]:6.2f} |"
              f" {dax:6.2f} |  {db[0]+dax:6.2f}")
    print("  （实测汇报 10.5 mm；EA=6013 时轴向项仅 0.4 mm）\n")

    print("=" * 72)
    print("B. 耦合模型 vs 纯力矩模型（v2）的末端差异")
    print("=" * 72)
    import tdcr_model_v2 as V2
    m2 = V2.TdcrModelV2(n_sub=3)
    mc = CoupledModel(EA_scale=1.0, n_sub=3)
    print("  θ    |  v2 tip (x,z)  |  耦合 tip (x,z) | 差异(mm) | ΔL v2/耦合")
    for th in (30, 60, 90, 120, 150):
        tau, _ = solve_A(mc, th, 0.0)
        oc = mc.solve(tau)
        o2 = m2.solve(np.maximum(0.0, m2.T0 + 0*PSI) if th == 0 else
                      np.maximum(0.0, m2.T0 + (np.radians(th) /
                                 ((NSEG//2)*m2.ds/m2.EI[0] + (NSEG-NSEG//2)*m2.ds/m2.EI[-1])
                                 / (3*m2.g['r_disk'])) * np.cos(PSI)))
        tc = oc["p"][-1] * 1e3; t2 = o2["p"][-1] * 1e3
        d2 = m2.delta_L(np.maximum(0.0, m2.T0 + (np.radians(th) /
                        ((NSEG//2)*m2.ds/m2.EI[0] + (NSEG-NSEG//2)*m2.ds/m2.EI[-1])
                        / (3*m2.g['r_disk'])) * np.cos(PSI)))
        db, dax = mc.dL(tau)
        print(f"  {th:3d}° | ({t2[0]:6.1f},{t2[2]:6.1f}) | ({tc[0]:6.1f},{tc[2]:6.1f}) |"
              f" {np.linalg.norm(tc-t2):6.2f}  | {d2[0]:5.2f}/{db[0]+dax:5.2f}")


if __name__ == "__main__":
    main()
