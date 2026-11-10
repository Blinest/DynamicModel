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

# ── 包路径引导（分层目录）────────────────────────────────────
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('geometry', 'stiffness', 'force', 'plot'):
    _p = _os.path.join(_ROOT, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
# ──────────────────────────────────────────────────────────────

import numpy as np

import cosserat_lattice as CL

__all__ = ["TdcrModelV2", "GEOM"]

import geom_params as _G

GEOM = dict(
    NSEG=_G.NSEG,                    # 20 个间隔盘 = 20 段（来自 geom_params）
    L_total=_G.L_TOTAL,              # m
    r_disk=_G.R_DISK,                # 丝分布半径 5 mm
    psi_deg=np.array(_G.PSI_DEG, float),
    mu=0.0,                          # 默认无摩擦（v1 同样不考虑摩擦）
    b_fric=1.0,                      # 摩擦几何因子
    T0=10.0,                         # 预紧 N
)


def _hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class TdcrModelV2:
    """20 段 TDCR 模型（刚度来自 cosserat_lattice）。"""

    def __init__(self, mu=None, T0=None, n_sub=5, axial=True,
                 stiffness=None, **kw):
        g = dict(GEOM); g.update(kw)
        self.g = g
        self.axial = bool(axial)      # 是否计入肌腱→轴向压缩耦合
        self.N = int(g["NSEG"])
        self.ds = g["L_total"] / self.N
        self.mu = float(g["mu"] if mu is None else mu)
        self.T0 = float(g["T0"] if T0 is None else T0)
        self.n_sub = int(n_sub)
        # ★ 刚度**注入**：力学层不认识几何，只吃这份契约
        #    stiffness = dict(EI=(N,), EA=(N,), GJ=(N,)) 或 None（默认取自几何层）
        if stiffness is None:
            stiffness = CL.arrays(self.N)
        self.stiffness = stiffness
        self.EI = np.asarray(stiffness["EI"], float)
        self.EA = np.asarray(stiffness["EA"], float)
        self.GJ = np.asarray(stiffness["GJ"], float)
        self.psi = np.radians(g["psi_deg"])

    def _active_mask(self, s):
        """锚点在该截面之后的丝（A 段丝锚 200mm、B 段丝锚 400mm）。"""
        anchor = np.where(np.arange(6) % 2 == 0, self.g["L_total"] * 0.5,
                          self.g["L_total"])
        return (anchor > s + 1e-12).astype(float)

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
                # ★ 只有**锚点在该截面之后**的丝，才对该截面产生力矩。
                #   A 段丝锚在 L1=200mm、B 段丝锚在 L2=400mm
                #   ⇒ s<200 时 6 根都算；s>200 时只剩 3 根 B 丝 ⇒ M 直接掉一半
                mask = self._active_mask(s)
                # 腱力矩：丝在 (r cosψ, r sinψ)，拉力沿 −z
                #   M = p × F = r·τ·(−sinψ, +cosψ, 0)
                rd = self.g["r_disk"]
                Mx = -rd * np.sum(taus * mask * np.sin(self.psi))
                My = rd * np.sum(taus * mask * np.cos(self.psi))
                # 尖端力力矩（体坐标）
                if np.any(Ft):
                    Mt = np.cross(np.array([0.0, 0.0, self.g["L_total"] - s]),
                                  R.T @ Ft)
                    Mx += Mt[0]
                    My += Mt[1]
                kap = np.array([Mx, My]) / ei          # κ = M/EI（弯 2 分量）
                kvec = np.array([kap[0], kap[1], 0.0])
                # ── 轴向耦合：腱张力把臂体压短 ────────────────────
                #   n_z = −Στ_j − F_tip·e_z(体)   ⇒   ε = n_z/EA
                # ⚠ 只算**超出预紧**的净载荷：全丝 =T0 时净载荷为 0，
                #   直臂不应有轴向缩短（否则 θ=0 时 ΔL≠0，是错的）
                n_z = -float(np.sum(taus - self.T0))
                if np.any(Ft):
                    n_z -= float(Ft @ (R @ np.array([0.0, 0.0, 1.0])))
                eps = (n_z / self.EA[i]) if self.axial else 0.0
                p = p + (R @ np.array([0, 0, 1.0])) * ds * (1.0 + eps)
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
        # κ 的方位角：κ=(Mx,My)/EI 沿 (−sinψ_b, cosψ_b) => ψ_b = atan2(−κx, κy)
        phi_b = (np.arctan2(-kmean[0], kmean[1])
                 if np.any(np.abs(kmean) > 1e-12) else 0.0)
        out = np.zeros(6)
        # 轴向压缩量：按各自跨度取（A 段丝半跨、B 段丝全跨）
        eps_mean = 0.0
        if self.axial:
            n_z = -float(np.sum(tau - self.T0))
            eps_mean = -n_z / float(self.EA.mean())    # >0 = 缩短
        half_len = self.g["L_total"] * 0.5
        for w, psi in enumerate(self.psi):
            span_is_A = (w % 2 == 0)                   # M1/M3/M5 = A 段
            span = phiA if span_is_A else phiB
            dL = -self.g["r_disk"] * span * np.cos(psi - phi_b)
            dL += -eps_mean * (half_len if span_is_A else self.g["L_total"])
            out[w] = max(0.0, -dL) * 1e3
        return out

    def kappa_profile(self, tau, **kw):
        o = self.solve(tau, **kw)
        return o["s"], np.linalg.norm(o["kappa"][:, :2], axis=1)

    def solve_bend(self, target_deg, F_tip=None, lo=0.5, hi=80.0, tol=0.3):
        """反解：给定目标弯角，求单侧对称张力 τ（3 根同拉）。"""
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            # ⚠ 用余弦律 τ = T0 + A·cos(ψ−0)，**不要把其余丝置 0**：
            #   置 0 等于把预紧拿掉，轴向净载荷变成 +Σ T0（虚假拉伸）
            tau = np.maximum(0.0, self.T0 + mid * np.cos(self.psi))
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
            # ⚠ 可用区间检查（见 docs/derivations/03_regime_limit.md）
            for i in (0, len(self.EI) - 1):
                n_taut = 6 if i == 0 else 3
                N = n_taut * self.T0
                Pcr = np.pi ** 2 * self.EI[i] / (self.g["L_total"] * 0.5) ** 2
                ratio = N / Pcr
                # 经验阈值来自 docs/derivations/03：T0<=5 正常、7-9 反号、10 发散
                flag = ("⚠ 发散风险" if ratio > 12
                        else ("⚠ 接近临界" if ratio > 9 else "ok"))
                print(f"    段{1 if i == 0 else 2}: N_ax={N:.0f} N  "
                      f"P_cr={Pcr:.2f} N  N/P_cr={ratio:.1f}  {flag}")
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
