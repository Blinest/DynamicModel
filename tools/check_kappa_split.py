# -*- coding: utf-8 -*-
"""check_kappa_split.py — 验证「曲率分段常数」与段1/段2 比值

背景
----
2026-11-01 的三张位形图上有个疑点：水平弯曲 83° 时 tip z 只有 236.8 mm，
而常曲率理论应约 305 mm，看起来像「弯曲集中在近端」。

结论（本脚本验证）
------------------
不是 bug，是**结构决定的**：模型的曲率沿弧长是**严格分段常数**，
段1（s < L1）与段2（s ≥ L1）之比恒为 **2.00**，因为
  - 段1 受 6 根丝（A 段 3 根锚在 L1 + B 段 3 根贯穿）的力矩
  - 段2 只受 B 段 3 根丝的力矩
  - 单丝贡献 ∝ Σcos²(α−φ)，3 根为 1.5、6 根为 3.0 ⇒ 比值 2
两段常数曲率拼接出的形状比同角度单圆弧更"折"，末端自然更靠近基座。

用法
----
    python tools/check_kappa_split.py                 # 默认 T0=10, A=5, 方位 90°
    python tools/check_kappa_split.py --A 8 --phi 0   # 改幅值/方位
    python tools/check_kappa_split.py --selftest      # 断言比值 ≈ 2.00
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

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TC_NO_JIT", "1")
from tendon_coupling import build_arm, shooting, ALPHA_DEG  # noqa: E402

EI_DEFAULT = 0.01037
EA_DEFAULT = 7e3
ALPHA = np.radians(np.array(ALPHA_DEG, float))


def make_params(EI, EA, g):
    ca, cb, p = build_arm(EI=EI, EA=EA, r_i=0.008, r_disk=0.005,
                          n_circ_a=3, n_circ_b=3, n_long=136, E=2e8,
                          t_strut=1.5e-3, rhoA=0.020 / 0.4, n_p=30, n_d=30,
                          lattice=False)
    p.g_world = tuple(g)
    return p


def profile(A, phi_deg, T0, EI, EA, g):
    p = make_params(EI, EA, g)
    tau = T0 + A * np.cos(ALPHA - np.radians(phi_deg))
    o = shooting(tau, p, F_tip=np.zeros(3), g_world=g, tol=1e-9, max_iter=80)
    S = np.asarray(o["s_curve"])
    kxy = np.linalg.norm(np.asarray(o["u_curve"])[:, :2], axis=1)
    ang = np.degrees(np.arccos(np.clip(o["R_curve"][-1][2, 2], -1.0, 1.0)))
    L1 = p.L1 if p.L1 is not None else 0.2
    k1 = kxy[S < L1 - 1e-9].mean()
    k2 = kxy[S >= L1 + 1e-9].mean()
    th = np.trapezoid(kxy, S) if hasattr(np, "trapezoid") else np.trapz(kxy, S)
    return dict(o=o, S=S, kxy=kxy, k1=k1, k2=k2, ang=ang,
                th=np.degrees(th), ratio=k1 / max(k2, 1e-12), L1=L1)


def show(tag, r):
    print(f"{tag}")
    print(f"  总弯角 {r['ang']:7.2f}°   ∫|κ|ds {r['th']:7.2f}°")
    print(f"  段1 平均 κ = {r['k1']:7.3f} 1/m   (s < {r['L1']*1e3:.0f} mm)")
    print(f"  段2 平均 κ = {r['k2']:7.3f} 1/m   (s ≥ {r['L1']*1e3:.0f} mm)")
    print(f"  比值 κ1/κ2 = {r['ratio']:6.3f}")
    print("  κ(s) 采样: " + "  ".join(
        f"{s*1e3:.0f}mm:{k:.2f}" for s, k in zip(r["S"][::10], r["kxy"][::10])))


def selftest():
    ok = True
    for g_tag, g in (("无重力", (0.0, 0.0, 0.0)), ("水平装 g=(-9.81,0,0)", (-9.81, 0.0, 0.0))):
        r = profile(5.0, 90.0, 10.0, EI_DEFAULT, EA_DEFAULT, g)
        show(f"[{g_tag}]", r)
        flag = abs(r["ratio"] - 2.0) < 0.1
        ok &= flag
        print(f"  理论比值 2.000  实测 {r['ratio']:.3f}  {'PASS' if flag else 'FAIL'}\n")
    # 分段是否真的常数（段内标准差应为 0）
    r = profile(5.0, 90.0, 10.0, EI_DEFAULT, EA_DEFAULT, (0.0, 0.0, 0.0))
    S, kxy = r["S"], r["kxy"]
    sd1 = kxy[S < r["L1"] - 1e-9].std()
    sd2 = kxy[S >= r["L1"] + 1e-9].std()
    flag = sd1 < 1e-9 and sd2 < 1e-9
    ok &= flag
    print(f"段内标准差: 段1 {sd1:.2e}  段2 {sd2:.2e}  "
          f"{'PASS(严格分段常数)' if flag else 'FAIL'}")
    print("\n自检结果:", "全部 PASS" if ok else "有 FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=float, default=5.0, help="差动幅值 N")
    ap.add_argument("--phi", type=float, default=90.0, help="弯曲方位 deg")
    ap.add_argument("--T0", type=float, default=10.0, help="基准预紧 N")
    ap.add_argument("--EI", type=float, default=EI_DEFAULT)
    ap.add_argument("--EA", type=float, default=EA_DEFAULT)
    ap.add_argument("--g", default="-9.81,0,0", help="重力方向, 逗号分隔")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    g = tuple(float(x) for x in a.g.split(","))
    show(f"[A={a.A}N  φ={a.phi}°  T0={a.T0}N  EI={a.EI}]",
         profile(a.A, a.phi, a.T0, a.EI, a.EA, g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
