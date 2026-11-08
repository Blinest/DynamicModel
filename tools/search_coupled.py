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
    def __init__(self, EA_scale=1.0, mu=0.0, T0=10.0, n_sub=3):
        ks = CL.stiffness()
        h = NSEG // 2
        self.EI = np.array([ks[0]["EI"]] * h + [ks[1]["EI"]] * (NSEG - h))
        ea = np.array([ks[0]["EA"]] * h + [ks[1]["EA"]] * (NSEG - h))
        self.EA = ea * EA_scale
        self.mu = mu; self.T0 = T0; self.n_sub = n_sub
        self.ds = L_TOT / NSEG

    def solve(self, tau, F_tip=None):
        p = np.zeros(3); R = np.eye(3); phi = 0.0; s = 0.0
        P = [p.copy()]; PL = [0.0]; K = [np.zeros(3)]
        Ft = np.zeros(3) if F_tip is None else np.asarray(F_tip, float)
        for i in range(NSEG):
            ei, ea = self.EI[i], self.EA[i]
            for _ in range(self.n_sub):
                ds = self.ds / self.n_sub
                decay = np.exp(-self.mu * phi)
                ts = tau * decay
                rd = R_DISK
                Mx = -rd * np.sum(ts * np.sin(PSI))
                My = rd * np.sum(ts * np.cos(PSI))
                if np.any(Ft):
                    Mt = np.cross(np.array([0.0, 0.0, L_TOT - s]), R.T @ Ft)
                    Mx += Mt[0]; My += Mt[1]
                kap = np.array([Mx, My]) / ei
                # 轴向压缩：腱张力合力的轴向分量（沿切向）
                n_z = -np.sum(ts) - (Ft @ (R @ np.array([0, 0, 1.0])))
                eps = n_z / ea
                kvec = np.array([kap[0], kap[1], 0.0])
                p = p + (R @ np.array([0, 0, 1.0])) * ds * (1.0 + eps)
                R = R @ (np.eye(3) + _hat(kvec) * ds
                         + 0.5 * _hat(kvec) @ _hat(kvec) * ds * ds)
                u, _, vt = np.linalg.svd(R); R = u @ vt
                phi += np.linalg.norm(kvec) * ds
                s += ds
            P.append(p.copy()); PL.append(phi); K.append(kvec)
        return dict(p=np.array(P), phi=np.array(PL), kappa=np.array(K), s=s)

    def bend(self, tau, **kw):
        return np.degrees(self.solve(tau, **kw)["phi"][-1])

    def dL(self, tau, **kw):
        o = self.solve(tau, **kw)
        phi = o["phi"]; k = o["kappa"]
        phiA, phiB = phi[NSEG // 2], phi[-1]
        km = k[:, :2].sum(0)
        phi_b = np.arctan2(-km[0], km[1]) if np.any(np.abs(km) > 1e-12) else 0.0
        # 轴向压缩量（平均）
        eps_mean = -np.sum(tau) / self.EA.mean()
        out_bend = np.zeros(6)
        out_ax = -eps_mean * L_TOT      # ⚠ 臂体压短 => 丝要收更多，取正
        for w, ps in enumerate(PSI):
            span = phiA if w % 2 == 0 else phiB
            out_bend[w] = R_DISK * span * np.cos(ps - phi_b)
        return out_bend * 1e3, out_ax * 1e3


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
