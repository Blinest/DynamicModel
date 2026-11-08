# -*- coding: utf-8 -*-
"""sample_table_v2.py — 用 v2 模型采 3000 个点，建控制用的查表

采样设计（3000 = 50 × 30 × 2）
    θ（弯角）  : 0…180°，50 个等分
    φ（方位）  : 0…348°，30 个等分（12° 步长）
    F_tip      : {0, 20 mN} 两档（垂直于弯曲平面的横向力）

每个点记录
    tau[6]        解出的六路张力（含预紧 T0，允许个别丝到 0 = 松弛）
    tip[3]        末端位置 (mm)
    bend          实际达到的弯角 (deg，用于核对)
    dl[6]         六路收丝量 (mm，只能受拉)
    nodes[61,3]   中线 61 节点 (mm) —— 与 model/ 表的节点数一致

求解方式：给 (θ, φ)，对幅值 A 做二分（τ_j = max(0, T0 + A·cos(ψ_j−φ))），
使实际弯角命中 θ。二分对 A 单调，30 次足够。

输出 model_v2/table.npz
"""
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


import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tdcr_model_v2 as V2   # noqa: E402

NTHETA = 50
NPHI = 30
FLIST = (0.0, 0.020)
NNODE = 61
T0 = 10.0


def resample(p, n=NNODE):
    """把变步长的中线重采样到 n 个等弧长节点。"""
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    su = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(su, s, p[:, k]) for k in range(3)])


def solve_A(m, theta_deg, phi_deg, F_tip, it=14):
    """求幅值 A 使弯角命中 theta_deg。

    全丝受拉时 M = 3·r·A ⇒ Θ ∝ A，闭式给初值：
        A0 = Θ / (3·r·(L1/EI1 + L2/EI2))
    再用少量二分修正（T0 截断与松弛会带来偏差）。
    """
    psi = np.radians(V2.GEOM["psi_deg"])
    ph = np.radians(phi_deg)
    Ft = np.array([0.0, -F_tip, 0.0]) if F_tip else None
    L = m.g["L_total"]; half = m.N // 2
    comp = (half * m.ds) / m.EI[0] + (m.N - half) * m.ds / m.EI[-1]
    A0 = np.radians(theta_deg) / (3.0 * m.g["r_disk"] * comp)
    lo, hi = max(0.0, 0.4 * A0), max(1e-6, 2.2 * A0)
    best = (None, None)
    for _ in range(it):
        A = 0.5 * (lo + hi)
        tau = np.maximum(0.0, T0 + A * np.cos(psi - ph))
        b = m.bend_angle(tau, F_tip=Ft)
        best = (tau, b)
        if b < theta_deg:
            lo = A
        else:
            hi = A
        if abs(b - theta_deg) < 0.15:
            break
    return best


def dl_from_out(m, tau, o):
    """复用 solve 的输出直接算 6 路收丝，避免重复积分。"""
    phi = o["phi"]
    phiA, phiB = phi[m.N // 2], phi[-1]
    km = o["kappa"][:, :2].sum(0)
    phi_b = (np.arctan2(-km[0], km[1])
             if np.any(np.abs(km) > 1e-12) else 0.0)
    # 轴向压缩（耦合项）：ε = −Στ/EA，A 段丝半跨、B 段丝全跨
    eps = 0.0
    if getattr(m, "axial", False):
        eps = (-float(np.sum(tau - m.T0)) / float(m.EA.mean()))
    half_len = m.g["L_total"] * 0.5
    out = np.zeros(6)
    for w, ps in enumerate(m.psi):
        isA = (w % 2 == 0)
        span = phiA if isA else phiB
        dL = -m.g["r_disk"] * span * np.cos(ps - phi_b)
        dL += eps * (half_len if isA else m.g["L_total"])   # eps<0 => dL 更负 => 收得更多
        out[w] = max(0.0, -dL) * 1e3
    return out


def main():
    m = V2.TdcrModelV2(T0=T0, n_sub=2)   # 表格用粗积分
    thetas = np.linspace(0.0, 180.0, NTHETA)
    phis = np.linspace(0.0, 348.0, NPHI)
    n = NTHETA * NPHI * len(FLIST)
    print(f"采点：{NTHETA} θ × {NPHI} φ × {len(FLIST)} F = {n} 点")

    # ── 与 model/vc_table_*.npz 对齐的字段 ──────────────────────
    R_tip_a = np.zeros((n, 3, 3)); kS_a = np.zeros((n, NNODE))
    kK_a = np.zeros((n, NNODE, 3)); kV_a = np.zeros((n, NNODE, 3))
    x_a = np.zeros((n, 6)); res_a = np.zeros(n)
    p_tip_m = np.zeros((n, 3))
    theta_a = np.zeros(n); phi_a = np.zeros(n); F_a = np.zeros(n)
    tau_a = np.zeros((n, 6)); tip_a = np.zeros((n, 3))
    bend_a = np.zeros(n); dl_a = np.zeros((n, 6))
    nodes_a = np.zeros((n, NNODE, 3))
    kappa_a = np.zeros((n, NNODE))          # 沿臂曲率幅值 (1/m)
    kseg_a = np.zeros((n, V2.GEOM['NSEG']))  # 逐段平均曲率

    k = 0
    t0 = time.time()
    for th in thetas:
        for ph in phis:
            for F in FLIST:
                tau, b = solve_A(m, th, ph, F)
                o = m.solve(tau, F_tip=(np.array([0.0, -F, 0.0]) if F else None))
                theta_a[k] = th; phi_a[k] = ph; F_a[k] = F
                tau_a[k] = tau
                tip_a[k] = o["p"][-1] * 1e3
                bend_a[k] = b
                dl_a[k] = dl_from_out(m, tau, o)
                nodes_a[k] = resample(o["p"]) * 1e3
                # ── 旧表同名量 ──
                p_tip_m[k] = o["p"][-1]
                R_tip_a[k] = o["R"][-1]
                ss = o["s"]; kk3 = o["kappa"]
                kS_a[k] = np.linspace(ss[0], ss[-1], NNODE)
                kK_a[k] = np.column_stack([
                    np.interp(kS_a[k], ss, kk3[:, j]) for j in range(3)])
                # 切向单位矢量（对应旧表的 kV）
                d = np.gradient(o["p"], axis=0)
                d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                kV_a[k] = np.column_stack([
                    np.interp(kS_a[k], ss, d[:, j]) for j in range(3)])
                # x = [弯曲方向 cos, sin, 0, 根部 κx, κy, 0]（对应旧表的初值态）
                k0 = kk3[1]
                x_a[k] = [np.cos(np.radians(ph)), np.sin(np.radians(ph)), 0.0,
                          k0[0], k0[1], k0[2]]
                res_a[k] = b - th
                kk = np.linalg.norm(o["kappa"][:, :2], axis=1)
                kappa_a[k] = np.interp(np.linspace(0, len(kk) - 1, NNODE),
                                       np.arange(len(kk)), kk)
                # 逐段平均（每段 n_sub+1 个节点）
                ns = m.n_sub
                kseg_a[k] = [kk[i * ns + 1:(i + 1) * ns + 1].mean()
                             for i in range(V2.GEOM['NSEG'])]
                k += 1
        el = time.time() - t0
        print(f"  θ={th:6.1f}°  完成 {k:4d}/{n}  用时 {el:5.1f}s "
              f"（预计总 {el/k*n:5.0f}s）", flush=True)

    os.makedirs("model_v2", exist_ok=True)
    import json
    ver = json.dumps({"model": "tdcr_model_v2", "NSEG": V2.GEOM["NSEG"],
                      "EI": [float(m.EI[0]), float(m.EI[-1])],
                      "EA": [float(m.EA[0]), float(m.EA[-1])],
                      "r_disk": V2.GEOM["r_disk"], "T0": T0,
                      "axial": bool(getattr(m, "axial", True)),
                      "mu": float(m.mu), "nodes": NNODE,
                      "n": int(n), "angle_max": 180.0}, sort_keys=True)
    np.savez_compressed(
        "model_v2/table.npz",
        # ★ 与 model/ 旧表同名同义
        tau=tau_a, dL_mm=dl_a, p_tip=p_tip_m, R_tip=R_tip_a,
        kS=kS_a, kK=kK_a, kV=kV_a, res=res_a, x=x_a,
        n_p=np.array(V2.GEOM["NSEG"]), n_d=np.array(NNODE),
        ver=np.array(ver),
        # 附加（新模型特有）
        theta=theta_a, phi=phi_a, F=F_a, tip_mm=tip_a, bend=bend_a,
        nodes_mm=nodes_a, kappa=kappa_a, kappa_seg=kseg_a,
        meta=np.array([NTHETA, NPHI, len(FLIST), NNODE, T0]))
    sz = os.path.getsize("model_v2/table.npz") / 1024
    print(f"\nsaved model_v2/table.npz  {sz:.0f} KB  "
          f"总用时 {time.time()-t0:.0f} s")

    # 质量核对
    err = np.abs(bend_a - theta_a)
    print(f"\n核对：")
    print(f"  弯角命中误差  max {err.max():.2f}°  mean {err.mean():.3f}°")
    print(f"  收丝量范围    {dl_a.min():.2f} … {dl_a.max():.2f} mm")
    print(f"  受拉根数分布  {np.bincount((dl_a > 1e-6).sum(1)).tolist()}")
    print(f"  末端 x 范围   {tip_a[:,0].min():.0f} … {tip_a[:,0].max():.0f} mm")
    print(f"  末端 z 范围   {tip_a[:,2].min():.0f} … {tip_a[:,2].max():.0f} mm")
    print(f"  曲率范围      {kappa_a.min():.2f} … {kappa_a.max():.2f} 1/m")
    print(f"  张力范围      {tau_a.min():.2f} … {tau_a.max():.2f} N")
    bad = err > 1.0
    if bad.any():
        print(f"  ⚠ {int(bad.sum())} 个点弯角未命中（>1°），最大 {err.max():.1f}°"
              f" —— 多为 θ 接近 180° 时饱和")


if __name__ == "__main__":
    main()
