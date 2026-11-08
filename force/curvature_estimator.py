# -*- coding: utf-8 -*-
"""curvature_estimator.py — 由张力 + 位移预测臂体曲率（融合估计）

问题
----
张力 τ（6 路）和丝位移 ΔL（6 路）**都是同一状态 (φ, A) 的观测量**，
单独用任一个都能估，但各有噪声短板：
    · τ  ：测力噪声 0.05 N，受预紧漂移/温漂影响
    · ΔL ：编码器很准（0.02 mm），但只能测到"收丝"，松弛丝为 0，信息量少一半

融合：把 (φ, A) 当状态，τ 与 ΔL 当 12 个观测，做加权最小二乘
    x̂ = argmin ‖y − h(x)‖²_W ,   y = [τ; ΔL]
然后曲率由模型给出
    κ(s) = M(x̂) / EI(s)

用法
----
    est = CurvatureEstimator(m)
    phi, A, kappa, res = est.estimate(tau_meas, dl_meas)
"""
from __future__ import annotations

import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _d in ("geometry", "stiffness", "force", "plot"):
    _p = os.path.join(_root, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _root not in sys.path:
    sys.path.insert(0, _root)

import tdcr_model_v2 as V2   # noqa: E402


class CurvatureEstimator:
    """张力 + 位移 → (φ, A) → κ(s)。

    Args:
        model     : TdcrModelV2 实例
        sig_tau   : 张力噪声 σ (N)
        sig_dl    : 丝位移噪声 σ (mm)
    """

    def __init__(self, model=None, sig_tau=0.05, sig_dl=0.02):
        self.m = model if model is not None else V2.TdcrModelV2(n_sub=3)
        self.sig_tau = float(sig_tau)
        self.sig_dl = float(sig_dl)
        self.psi = self.m.psi

    # ── 正演：(φ, A) → (τ, ΔL, κ) ───────────────────────────────
    def predict(self, phi_deg, A, F_tip=None):
        tau = np.maximum(0.0, self.m.T0 + A * np.cos(self.psi - np.radians(phi_deg)))
        o = self.m.solve(tau, F_tip=F_tip)
        dl = self.m.delta_L(tau, F_tip=F_tip)
        kap = np.linalg.norm(o["kappa"][:, :2], axis=1)
        return tau, dl, kap, o

    # ── 加权残差 ────────────────────────────────────────────────
    def _residual(self, x, tau_m, dl_m):
        tau_p, dl_p, _, _ = self.predict(x[0], x[1])
        rt = (tau_p - tau_m) / self.sig_tau
        rd = (dl_p - dl_m) / self.sig_dl
        return np.concatenate([rt, rd])

    def estimate(self, tau_meas, dl_meas, F_tip=None, x0=None, it=10):
        """Gauss-Newton 解 (φ, A)。返回 (phi, A, kappa, residual_norm)。"""
        tau_m = np.asarray(tau_meas, float).reshape(6)
        dl_m = np.asarray(dl_meas, float).reshape(6)
        if x0 is None:
            # 由张力直接估初值：A = (max-min)/2, φ = 最强那根的方位
            A0 = max(1e-3, (tau_m.max() - tau_m.min()) / 2.0)
            phi0 = np.degrees(self.psi[int(np.argmax(tau_m))])
            x = np.array([phi0, A0], float)
        else:
            x = np.array(x0, float)

        r = self._residual(x, tau_m, dl_m)
        for _ in range(it):
            if np.linalg.norm(r) < 1e-3:
                break
            J = np.zeros((12, 2))
            h = np.array([1e-2, 1e-3])          # deg, N
            for k in range(2):
                xp = x.copy(); xp[k] += h[k]
                J[:, k] = (self._residual(xp, tau_m, dl_m) - r) / h[k]
            try:
                d = np.linalg.solve(J.T @ J + 1e-9 * np.eye(2), -J.T @ r)
            except np.linalg.LinAlgError:
                break
            x = x + 0.8 * d
            x[1] = max(0.0, x[1])
            r = self._residual(x, tau_m, dl_m)
        _, _, kap, o = self.predict(x[0], x[1])
        return float(x[0]), float(x[1]), kap, float(np.linalg.norm(r))


def main():
    rng = np.random.default_rng(0)
    m = V2.TdcrModelV2(n_sub=3)
    est = CurvatureEstimator(m)

    print("用 (τ, ΔL) 融合估计 (φ, A) 并预测 κ(s)")
    print("  真值       | 仅用 τ        | 仅用 ΔL       | 融合")
    print("  φ    A    | φ̂     Â  误差 | φ̂     Â  误差 | φ̂     Â  误差")
    for phi_t, A_t in [(0.0, 4.0), (60.0, 5.0), (137.0, 3.0), (240.0, 6.5)]:
        tau_t, dl_t, kap_t, _ = est.predict(phi_t, A_t)
        tau_m = tau_t + rng.normal(0, est.sig_tau, 6)
        dl_m = np.maximum(0.0, dl_t + rng.normal(0, est.sig_dl, 6))

        # 仅张力：ΔL 权重置 0（用极大 σ）
        e1 = CurvatureEstimator(m, sig_tau=est.sig_tau, sig_dl=1e3)
        p1, a1, k1, _ = e1.estimate(tau_m, dl_m)
        # 仅位移
        e2 = CurvatureEstimator(m, sig_tau=1e3, sig_dl=est.sig_dl)
        p2, a2, k2, _ = e2.estimate(tau_m, dl_m)
        # 融合
        p3, a3, k3, res = est.estimate(tau_m, dl_m)

        def err(p, a):
            return np.hypot(((p - phi_t + 180) % 360 - 180) * 0.1,
                            (a - A_t) * 10)
        print(f"  {phi_t:5.0f} {A_t:4.1f} | {p1:5.0f} {a1:4.1f} {err(p1,a1):5.2f} |"
              f" {p2:5.0f} {a2:4.1f} {err(p2,a2):5.2f} |"
              f" {p3:5.0f} {a3:4.1f} {err(p3,a3):5.2f}")

    print("\nκ(s) 预测精度（融合，20 段）：")
    for phi_t, A_t in [(0.0, 4.0), (90.0, 6.0)]:
        tau_t, dl_t, kap_t, _ = est.predict(phi_t, A_t)
        tau_m = tau_t + rng.normal(0, est.sig_tau, 6)
        dl_m = np.maximum(0.0, dl_t + rng.normal(0, est.sig_dl, 6))
        _, _, kap_h, res = est.estimate(tau_m, dl_m)
        seg = kap_t[1:20:2] if len(kap_t) > 20 else kap_t
        rel = np.abs(kap_h - kap_t) / np.maximum(kap_t, 1e-6)
        print(f"  φ={phi_t:.0f}° A={A_t:.1f}: κ 段1 {kap_t[1]:.2f} → {kap_h[1]:.2f}, "
              f"段2 {kap_t[-1]:.2f} → {kap_h[-1]:.2f}, "
              f"最大相对误差 {rel[kap_t > 0.1].max()*100:.1f}%  残差 {res:.2f}")


if __name__ == "__main__":
    main()
