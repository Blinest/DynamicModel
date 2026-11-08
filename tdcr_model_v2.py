# -*- coding: utf-8 -*-
"""tdcr_model_v2.py — 新模型的 TDCR 入口（20 段 / 两段变体 / 几何推刚度）

与 `tdcr_model.py` 的区别
------------------------
| 项 | 旧 (v1) | 新 (v2) |
|---|---|---|
| 刚度来源 | 单个标定 EI/EA | **由几何推**（`cosserat_lattice`），逐段不同 |
| 分段 | 两段连续 Cosserat | **20 段**（对应 20 个间隔盘，每段 20 mm） |
| κ 分布 | 由求解器自然给出 | 由段刚度 + 摩擦衰减决定 |
| 丝位移 | 几何弧长差（6 根都动） | **只能受拉**：`ΔL=max(0,−r·Θspan·cos)` |
| 尖端力 | 有 | 有，且可与腱力矩叠加 |

模型（力矩驱动，准静态）
------------------------
    M(s) = r_d·Σ_j τ_j(s)·[cosψ_j, sinψ_j]  +  M_tip(s)      (体坐标 x–y 平面)
    κ(s) = M(s) / EI(s)                                        (逐段 EI)
    τ_j(s) = τ_j·exp(−μ·b·φ(s))                                (Capstan 摩擦)
    R' = R·hat(κ),  p' = R·e_z

用法
----
    from tdcr_model_v2 import TdcrModelV2
    m = TdcrModelV2()
    m.info()
    tip, R = m.pose([22, 22, 0, 0, 0, 0])
    print(m.bend_angle([22, 22, 0, 0, 0, 0]))
    print(m.delta_L([22, 22, 0, 0, 0, 0]))      # 6 路，受拉>0、松弛=0
"""
from __future__ import annotations

import numpy as np

import cosserat_lattice as CL

__all__ = ["TdcrModelV2", "GEOM"]

GEOM = dict(
    NSEG=20,             # 20 个间隔盘 = 20 段
    L_total=0.400,       # m
    r_disk=0.005,        # 丝分布半径 5 mm
    psi_deg=np.array([0, 60, 120, 180, 240, 300], float),
    mu=0.0,              # 默认无摩擦（v1 同样不考虑摩擦）
    b_fric=1.0,          # 摩擦几何因子
    T0=10.0,             # 预紧 N
)


def _hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class TdcrModelV2:
    """20 段 TDCR 模型（刚度来自 cosserat_lattice）。"""

    def __init__(self, mu=None, T0=None, n_sub=5, **kw):
        g = dict(GEOM); g.update(kw)
        self.g = g
        self.N = int(g["NSEG"])
        self.ds = g["L_total"] / self.N
        self.mu = float(g["mu"] if mu is None else mu)
        self.T0 = float(g["T0"] if T0 is None else T0)
        self.n_sub = int(n_sub)
        ks = CL.stiffness()
        # 前一半段用段1 刚度，后一半用段2
        half = self.N // 2
        self.EI = np.array([ks[0]["EI"]] * half + [ks[1]["EI"]] * (self.N - half))
        self.EA = np.array([ks[0]["EA"]] * half + [ks[1]["EA"]] * (self.N - half))
        self.GJ = np.array([ks[0]["GJ"]] * half + [ks[1]["GJ"]] * (self.N - half))
        self.psi = np.radians(g["psi_deg"])

    # ── 内部：张力分布 ───────────────────────────────────────────
    def _tau_of_s(self, tau, phi_accum):
        """Capstan：τ_j(s) = τ_j·exp(−μ·b·φ)，φ 为累计转折角。"""
        return tau[None, :] * np.exp(-self.mu * self.g["b_fric"] * phi_accum[:, None])

    # ── 正解 ────────────────────────────────────────────────────
    def solve(self, tau, F_tip=None, n_out=201):
        tau = np.asarray(tau, float).reshape(6)
        Ft = np.zeros(3) if F_tip is None else np.asarray(F_tip, float).reshape(3)

        p = np.zeros(3); R = np.eye(3)
        phi = 0.0
        P = [p.copy()]; Rs = [R.copy()]; S = [0.0]; K = [np.zeros(3)]
        PL = [0.0]                       # 累计转角（这才是弯角）
        s = 0.0
        for i in range(self.N):
            ei = self.EI[i]
            for _ in range(self.n_sub):
                ds = self.ds / self.n_sub
                taus = self._tau_of_s(tau, np.array([phi]))[0]
                # 腱力矩（体坐标）
                rd = self.g["r_disk"]
                M = np.array([rd * np.sum(taus * np.cos(self.psi)),
                              rd * np.sum(taus * np.sin(self.psi)),
                              0.0])
                # 尖端力力矩：力在体坐标下 (L−s) 力臂
                if np.any(Ft):
                    M = M + np.cross(np.array([0, 0, self.g["L_total"] - s]),
                                     R.T @ Ft)
                kap = M[:2] / ei                     # 只保留弯曲分量
                kvec = np.array([kap[0], kap[1], 0.0])
                p = p + (R @ np.array([0, 0, 1.0])) * ds
                R = R @ (np.eye(3) + _hat(kvec) * ds +
                         0.5 * _hat(kvec) @ _hat(kvec) * ds * ds)
                u, _, vt = np.linalg.svd(R)
                R = u @ vt
                phi += np.linalg.norm(kvec) * ds
                s += ds
            P.append(p.copy()); Rs.append(R.copy()); S.append(s)
            K.append(kvec); PL.append(phi)
        out = dict(s=np.array(S), p=np.array(P), R=np.array(Rs),
                   kappa=np.array(K), phi=np.array(PL),
                   tau_end=self._tau_of_s(tau, np.array([phi]))[0])
        return out

    # ── 对外 API（与 v1 同名）────────────────────────────────────
    def pose(self, tau, **kw):
        o = self.solve(tau, **kw)
        return o["p"][-1], o["R"][-1]

    def bend_angle(self, tau, **kw):
        """累计转角 (deg)。不能用 acos(R[2,2])：转角超 180° 会绕回，acos 非单调。"""
        return float(np.degrees(self.solve(tau, **kw)["phi"][-1]))

    def delta_L(self, tau, **kw):
        """6 路收丝量 (mm)，**只能受拉**：松弛丝为 0。

        ΔL_w = max(0, −r·Θ_span·cos(ψ_w − φ_b))
        A 段丝只跨前一半段，B 段丝跨全部。
        """
        o = self.solve(tau, **kw)
        kap = o["kappa"]
        phi = o["phi"]                         # 累计转角（积分直接给出）
        phiA = phi[self.N // 2]
        phiB = phi[-1]
        # 弯曲方位
        kmean = kap[:, :2].sum(0)
        phi_b = np.arctan2(kmean[1], kmean[0]) if np.any(kmean) else 0.0
        out = np.zeros(6)
        for w, psi in enumerate(self.psi):
            span = phiA if w % 2 == 0 else phiB          # M1/M3/M5 = A 段
            dL = -self.g["r_disk"] * span * np.cos(psi - phi_b)
            out[w] = max(0.0, -dL) * 1e3
        return out

    def kappa_profile(self, tau, **kw):
        o = self.solve(tau, **kw)
        return o["s"], np.linalg.norm(o["kappa"][:, :2], axis=1)

    def solve_bend(self, target_deg, F_tip=None, lo=0.5, hi=80.0, tol=0.3):
        """反解：给定目标弯角，求单侧对称张力 τ（3 根同拉）。"""
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            tau = np.zeros(6)
            tau[[0, 1, 5]] = mid if target_deg >= 0 else 0.0
            b = self.bend_angle(tau, F_tip=F_tip)
            if abs(b - target_deg) < tol:
                return mid, b, tau
            if b < target_deg:
                lo = mid
            else:
                hi = mid
        return mid, b, tau

    # ── 信息 ────────────────────────────────────────────────────
    def info(self, verbose=True):
        d = dict(N=self.N, ds_mm=self.ds * 1e3, L=self.g["L_total"],
                 EI1=self.EI[0], EI2=self.EI[-1], EA1=self.EA[0],
                 EA2=self.EA[-1], mu=self.mu, r_disk=self.g["r_disk"])
        if verbose:
            print("═" * 60)
            print("TDCR 模型 v2（20 段 / 几何推刚度）")
            print("═" * 60)
            print(f"  分段 {d['N']} 段 × {d['ds_mm']:.0f} mm = "
                  f"{d['L']*1e3:.0f} mm")
            print(f"  EI: 段1 {d['EI1']:.5f} / 段2 {d['EI2']:.5f} N·m²"
                  f"  （由 cosserat_lattice 推导）")
            print(f"  EA: 段1 {d['EA1']:.0f} / 段2 {d['EA2']:.0f} N")
            print(f"  丝分布 Φ{d['r_disk']*2e3:.0f} mm，6 根，ψ=0/60/…/300°")
            print(f"  摩擦 μ={d['mu']:.2f}，预紧 T0={self.T0:.1f} N")
        return d


# ── 自检 ────────────────────────────────────────────────────────
def _selftest():
    m = TdcrModelV2()
    m.info()
    print("\n自检 A：无摩擦，单侧拉至 90°，求所需张力")
    T, b, tau = m.solve_bend(90.0)
    print(f"  τ(3 根) = {T:.2f} N  ⇒ 弯角 {b:.2f}°")
    d = m.delta_L(tau)
    print(f"  6 路收丝: {np.round(d, 2)} mm  （受拉 {int((d>1e-6).sum())} 根）")

    print("\n自检 B：尖端力对弯角的影响（同一张力下）")
    for F in (0.0, 0.01, 0.03):
        tau = np.zeros(6); tau[[0, 1, 5]] = T
        bb = m.bend_angle(tau, F_tip=np.array([0.0, -F, 0.0]))
        print(f"  F_tip={F*1e3:4.0f} mN  ⇒ 弯角 {bb:6.2f}°  "
              f"（变化 {bb-b:+.2f}°）")

    print("\n自检 C：摩擦对张力传递的影响（μ=0 vs 0.15）")
    for mu in (0.0, 0.15):
        mm = TdcrModelV2(mu=mu)
        TT, bb, tt = mm.solve_bend(90.0)
        o = mm.solve(tt)
        print(f"  μ={mu:.2f}  τ(电机侧)={TT:6.2f} N  ⇒ {bb:5.1f}°  "
              f"末端张力 {o['tau_end'][0]:5.2f} N")


if __name__ == "__main__":
    _selftest()
