# -*- coding: utf-8 -*-
"""build_planar_table.py — 生成"近常曲率 C 形"的定向表

与 `build_real_arm.py` 的区别：**不随机采 τ**，而是按 (方位 φ, 弯角 θ) 网格
定向求解，并只保留**接近常曲率、平面度 > 阈值的 C 形**样本。

    python tools/build_planar_table.py --theta-step 15 --phi-step 30 --planar 0.9

机制
----
1. 单参数张力合成:  `τ_i = T0 + A·cos(α_i − φ)` —— 合力矩方向正好是 φ，
   所以同一条公式覆盖所有方位（不需要手查配对表）。
2. 对每个 (φ, θ)：二分搜索幅值 A 使末端弯角 = θ。
3. 平面度: 对中心线点云做 PCA，`planarity = 1 − s3/s1`（1 = 完美平面）。
4. 只保留 `planarity > --planar` 且所有 τ ∈ [0.2, tau_max] 的样本。

输出：`vc_table_planar_<step>deg.npz`，字段与常规表一致
（tau / dL_mm / p_tip / R_tip / kS / kK / res / x / n_p / n_d / ver）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")

from tendon_coupling import (shooting, _wire_geom_len_batch, _L0_wire,  # noqa: E402
                            ALPHA_DEG, build_arm)
from ik_table import canonical_version_key                              # noqa: E402


def make_params(EI, EA, rhoA, nsub, g_world, Lside):
    ca, cb, p = build_arm(EI=EI, EA=EA, L1=Lside, L2=Lside,
                          r_i=0.008, r_disk=0.005, n_circ_a=3, n_circ_b=3,
                          n_long=136, E=2e8, t_strut=1.5e-3,
                          rhoA=rhoA, n_p=nsub, n_d=nsub, lattice=False)
    p.g_world = tuple(float(x) for x in g_world)
    return p


def planarity(P):
    """1 = 完美平面。对点云协方差做 SVD，取 1 − s3/s1。"""
    Q = np.asarray(P, float)
    Q = Q - Q.mean(axis=0)
    s = np.linalg.svd(Q, compute_uv=False)
    if s[0] <= 0:
        return 0.0
    return float(1.0 - s[2] / s[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-step", type=float, default=15.0)
    ap.add_argument("--phi-step", type=float, default=30.0)
    ap.add_argument("--theta-max", type=float, default=150.0)
    ap.add_argument("--planar", type=float, default=0.9)
    ap.add_argument("--EI", type=float, default=0.0562, help="重标后的抗弯刚度")
    ap.add_argument("--EA", type=float, default=7e3)
    ap.add_argument("--T0", type=float, default=3.0, help="基准预紧 N")
    ap.add_argument("--tau-max", type=float, default=40.0)
    ap.add_argument("--nsub", type=int, default=30)
    ap.add_argument("--g", type=str, default="-9.81,0,0", help="重力方向（水平安装）")
    ap.add_argument("--L1", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    g = tuple(float(x) for x in a.g.split(","))
    p = make_params(a.EI, a.EA, 0.020 / (2 * a.L1), a.nsub, g, a.L1)
    L0 = _L0_wire(p)
    alpha = np.radians(np.array(ALPHA_DEG, float))
    out = a.out or f"vc_table_planar_{int(a.theta_step)}deg.npz"

    recs, n_try = [], 0
    for th_deg in np.arange(a.theta_step, a.theta_max + 1e-9, a.theta_step):
        for ph_deg in np.arange(0.0, 360.0, a.phi_step):
            n_try += 1
            ph = np.radians(ph_deg)
            d = np.cos(alpha - ph)                     # 力矩方向 = φ
            lo, hi = 0.0, a.tau_max
            best = None
            for _ in range(9):                          # 二分幅值
                A = 0.5 * (lo + hi)
                tau = a.T0 + A * d
                if tau.min() < 0.2:
                    hi = A
                    continue
                o = shooting(tau, p, F_tip=np.zeros(3), g_world=g,
                             tol=1e-8, max_iter=60)
                if not o["converged"]:
                    hi = A
                    continue
                ang = float(np.degrees(np.arccos(
                    np.clip(o["R_curve"][-1][2, 2], -1.0, 1.0))))
                if ang < th_deg:
                    lo = A
                else:
                    hi = A
                best = (tau, o, ang)
            if best is None:
                continue
            tau, o, ang = best
            pl = planarity(o["p_curve"])
            ok = (pl > a.planar and tau.max() <= a.tau_max
                  and abs(ang - th_deg) < 0.35 * a.theta_step)
            tag = "✓" if ok else "✗"
            print(f"  θ={th_deg:5.1f}° φ={ph_deg:5.1f}° 实测 {ang:6.2f}°  "
                  f"平面度 {pl:.3f} τ[{tau.min():4.1f},{tau.max():4.1f}] {tag}",
                  flush=True)
            if not ok:
                continue
            Pc = o["p_curve"]; Rc = o["R_curve"]
            dL = (_wire_geom_len_batch(Pc[None], Rc[None], r_wire=p.r_disk,
                                       L1_geo=p.L1, L_total=p.L_total)[0] - L0) * 1e3
            recs.append(dict(tau=tau, dL_mm=dL, p_tip=Pc[-1], R_tip=Rc[-1],
                             kS=np.asarray(o["s_curve"], float),
                             kK=np.asarray(o["u_curve"], float),
                             res=float(np.linalg.norm(o["res"])),
                             x=np.asarray(o["x"], float),
                             theta=ang, phi=ph_deg, planarity=pl))

    if not recs:
        sys.exit("没有样本通过筛选")
    N = len(recs); m = len(recs[0]["kS"])
    KS = np.stack([r["kS"] for r in recs]); KK = np.stack([r["kK"] for r in recs])
    np.savez_compressed(
        out, tau=np.stack([r["tau"] for r in recs]),
        dL_mm=np.stack([r["dL_mm"] for r in recs]),
        p_tip=np.stack([r["p_tip"] for r in recs]),
        R_tip=np.stack([r["R_tip"] for r in recs]),
        kS=KS, kK=KK,
        res=np.array([r["res"] for r in recs]),
        x=np.stack([r["x"] for r in recs]),
        theta=np.array([r["theta"] for r in recs]),
        phi=np.array([r["phi"] for r in recs]),
        planarity=np.array([r["planarity"] for r in recs]),
        n_p=np.array(p.n_p), n_d=np.array(p.n_d),
        ver=np.array(canonical_version_key(p)),
    )
    print(f"\n[out] {out}  {N}/{n_try} 通过  {os.path.getsize(out)/1e6:.2f} MB  "
          f"节点={m}  平面度中位={np.median([r['planarity'] for r in recs]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
