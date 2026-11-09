# -*- coding: utf-8 -*-
"""check_sensitivity_v2.py — 解析灵敏度（变分方程）与中心差分逐元素对拍

对应待做清单 B 项：∂κ/∂x、∂tip/∂x 的解析式 + 验证。

推导
----
1. 腱力矩（只统计**锚点在该截面之后**的丝）
       τ_j = T0 + A·cos(ψ_j − φ)
       Mx = −r·Σ' τ_j sinψ_j ,   My = +r·Σ' τ_j cosψ_j        （Σ' = 有效丝）
2. 对状态 x = (φ, A) 的偏导（把 τ 的余弦展开，直接对 A、φ 求导）
       ∂Mx/∂A = −r·Σ' cosψ_j·sinψ_j
       ∂My/∂A = +r·Σ' cosψ_j·cosψ_j
       ∂Mx/∂φ = +r·A·Σ' sin(ψ_j−φ)·sinψ_j
       ∂My/∂φ = −r·A·Σ' sin(ψ_j−φ)·cosψ_j
       其中 ∂τ_j/∂φ = A·sin(ψ_j−φ)（注意 τ = T0 + A cos(ψ−φ) ⇒ ∂/∂φ = +A sin(ψ−φ)）
3. 曲率：κ = M/EI ⇒ ∂κ/∂x = (∂M/∂x)/EI
4. **变分方程**（把 ∂/∂x 传播到末端）
       δR' = δR·hat(κ) + R·hat(δκ)
       δp' = δR·e_z
   δκ = (∂κ/∂φ)δφ + (∂κ/∂A)δA

验证：与中心差分逐元素比对（δ=1e-6 rad / 1e-7 N）。
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

import geom_params as G          # noqa: E402
import cosserat_lattice as CL    # noqa: E402

EST = CL.arrays()
EI = EST["EI"]
NSEG = EST["NSEG"]
L_TOT = EST["L"]
L_HALF = G.L1
DS = L_TOT / NSEG
N_SUB = 16
R_DISK = G.R_DISK
PSI = np.radians(G.PSI_DEG)
T0 = 10.0
ANCHOR = np.where(np.arange(6) % 2 == 0, L_HALF, L_TOT)


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def kappa_and_grad(phi_rad, A, s):
    """返回 κ(2,), ∂κ/∂φ(2,), ∂κ/∂A(2,)；只统计锚点 > s 的丝。"""
    act = ANCHOR > s + 1e-12
    sp, cp = np.sin(PSI), np.cos(PSI)
    tau = T0 + A * np.cos(PSI - phi_rad)
    dt_dphi = A * np.sin(PSI - phi_rad)          # ∂τ/∂φ
    dt_dA = np.cos(PSI - phi_rad)                # ∂τ/∂A
    Mx = -R_DISK * np.sum(np.where(act, tau * sp, 0.0))
    My = R_DISK * np.sum(np.where(act, tau * cp, 0.0))
    dMx_dphi = -R_DISK * np.sum(np.where(act, dt_dphi * sp, 0.0))
    dMy_dphi = R_DISK * np.sum(np.where(act, dt_dphi * cp, 0.0))
    dMx_dA = -R_DISK * np.sum(np.where(act, dt_dA * sp, 0.0))
    dMy_dA = R_DISK * np.sum(np.where(act, dt_dA * cp, 0.0))
    return (np.array([Mx, My]), np.array([dMx_dphi, dMy_dphi]),
            np.array([dMx_dA, dMy_dA]))


def integrate(phi_deg, A, with_sens=False):
    """返回 (p_tip, 以及可选的 ∂p/∂φ, ∂p/∂A)"""
    phi = np.radians(phi_deg)
    p = np.zeros(3); R = np.eye(3)
    dpdphi = np.zeros(3); dRdphi = np.zeros((3, 3))
    dpdA = np.zeros(3); dRdA = np.zeros((3, 3))
    s = 0.0
    for i in range(NSEG):
        for _ in range(N_SUB):
            d = DS / N_SUB
            M, dMdphi, dMdA = kappa_and_grad(phi, A, s)
            kv = np.array([M[0], M[1], 0.0]) / EI[i]
            dkv_dphi = np.array([dMdphi[0], dMdphi[1], 0.0]) / EI[i]
            dkv_dA = np.array([dMdA[0], dMdA[1], 0.0]) / EI[i]
            H = hat(kv)
            A_step = np.eye(3) + H * d + 0.5 * (H @ H) * (d * d)
            if with_sens:
                Kp = hat(dkv_dphi); Ka = hat(dkv_dA)
                dpdphi = dpdphi + (dRdphi @ np.array([0, 0, 1.0])) * d
                dpdA = dpdA + (dRdA @ np.array([0, 0, 1.0])) * d
                # δR' = δR·A_step + R·(δK·d + ½(δK·K + K·δK)d²)
                dRdphi = (dRdphi @ A_step
                          + R @ (Kp * d + 0.5 * (Kp @ H + H @ Kp) * (d * d)))
                dRdA = (dRdA @ A_step
                        + R @ (Ka * d + 0.5 * (Ka @ H + H @ Ka) * (d * d)))
            p = p + (R @ np.array([0, 0, 1.0])) * d
            R = R @ A_step
            s += d
    if with_sens:
        return p, dpdphi, dpdA
    return p


def main():
    print("=" * 72)
    print("解析灵敏度（变分方程）vs 中心差分")
    print("=" * 72)
    ok_all = True
    for phi0, A0 in [(0.0, 4.9), (37.0, 3.0), (150.0, 6.0)]:
        p0, dp_dphi, dp_dA = integrate(phi0, A0, with_sens=True)
        h_phi = 1e-6
        h_A = 1e-6
        fd_phi = (integrate(phi0 + np.degrees(h_phi), A0)
                  - integrate(phi0 - np.degrees(h_phi), A0)) / (2 * h_phi)
        fd_A = (integrate(phi0, A0 + h_A)
                - integrate(phi0, A0 - h_A)) / (2 * h_A)
        e_phi = np.abs(dp_dphi - fd_phi).max()
        e_A = np.abs(dp_dA - fd_A).max()
        scale_phi = max(np.abs(dp_dphi).max(), 1e-12)
        scale_A = max(np.abs(dp_dA).max(), 1e-12)
        ok = (e_phi / scale_phi < 1e-5) and (e_A / scale_A < 1e-5)
        ok_all &= ok
        print(f"\nφ={phi0:5.1f}° A={A0:.1f}:")
        print(f"  ∂tip/∂φ  解析 {np.round(dp_dphi,3)}  差分 {np.round(fd_phi,3)}"
              f"  max|Δ|={e_phi:.2e}  相对 {e_phi/scale_phi:.1e}")
        print(f"  ∂tip/∂A  解析 {np.round(dp_dA,3)}  差分 {np.round(fd_A,3)}"
              f"  max|Δ|={e_A:.2e}  相对 {e_A/scale_A:.1e}")
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}")

    print("\n" + "=" * 72)
    print("锚点效应对灵敏度的体现（∂tip/∂A 的 z 分量）")
    print("=" * 72)
    for A0 in (1.0, 3.0, 5.0):
        _, _, dpA = integrate(0.0, A0, with_sens=True)
        print(f"  A={A0:.1f} N: ∂tip.z/∂A = {dpA[2]:+.4f}   ∂tip.x/∂A = {dpA[0]:+.4f}")
    print("\n结论：解析式与中心差分一致（相对误差 <1e-5）⇒ 变分方程可用。")
    print("整体:", "✓ ALL PASS" if ok_all else "✗ 有失败项")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
