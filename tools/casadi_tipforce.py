# -*- coding: utf-8 -*-
"""casadi_tipforce.py — 加入尖端横向力，并用 CasADi/IPOPT 辨识其大小

模型（20 段，段长 20 mm）
------------------------
尖端横向力 P 在截面 s 处产生弯矩

    M(s) = P · (L − s) · cos(θ_L − θ(s))        # 通用式（含大变形）
    M(s) ≈ P · (L − s)                          # 小变形近似

曲率由弯矩与段刚度决定（段1/段2 刚度不同，见 cosserat_lattice）：

    κ(s) = M(s) / EI(s)

于是 κ 场完全由 **单个标量 P** 决定 ⇒ 与"逐段 κ"不同，这是**可辨识**的。

任务
----
1. 用 P_true 生成"实测"中线（20 段端点 + 噪声）
2. CasADi 建 NLP，把 P（以及 EI 的比例因子）作为决策变量
3. IPOPT 求解，报回收值与误差、耗时
4. 扫描噪声水平，给出可辨识下限

依赖：pip install casadi
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
import casadi as ca

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cosserat_lattice as CL   # noqa: E402

NSEG = 20
SEG_LEN = 0.02
L_TOT = NSEG * SEG_LEN
N_SUB = 4
EI = None


def seg_ei():
    """20 段的 EI：前 10 段段1、后 10 段段2。"""
    ks = CL.stiffness()
    return np.array([ks[0]["EI"]] * 10 + [ks[1]["EI"]] * 10)


def roller(kvec):
    """数值前向（RK4，与符号版同格式，避免格式差被误判为参数偏差）。"""
    px = pz = th = 0.0
    out = []
    for k in kvec:
        ds = SEG_LEN / N_SUB
        for _ in range(N_SUB):
            f = lambda t: np.array([np.sin(t), np.cos(t), k])
            k1 = f(th); k2 = f(th + 0.5*ds*k1[2])
            k3 = f(th + 0.5*ds*k2[2]); k4 = f(th + ds*k3[2])
            px += ds/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            pz += ds/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            th += ds/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        out.append((px, pz, th))
    return np.array(out)


def kappa_of_P(P, ei, small=True):
    """尖端横向力 P(N) → 20 段 κ。"""
    s_end = (np.arange(NSEG) + 1) * SEG_LEN
    s_mid = s_end - 0.5 * SEG_LEN
    if small:
        M = P * (L_TOT - s_mid)
    else:
        # 大变形：先小变形估形状，再用真实力臂迭代一次
        k0 = P * (L_TOT - s_mid) / ei
        sh = roller(k0)
        th_s = np.interp(s_mid, s_end, sh[:, 2])
        M = P * (L_TOT - s_mid) * np.cos(sh[-1, 2] - th_s)
    return M / ei


def rollout_casadi(P, ei, n_per_seg=N_SUB):
    """CasADi 符号前向（P 为符号变量）。"""
    px = ca.MX(0.0); pz = ca.MX(0.0); th = ca.MX(0.0)
    outs = []
    for i in range(NSEG):
        s_mid = (i + 0.5) * SEG_LEN
        kap = P * (L_TOT - s_mid) / ei[i]          # 小变形弯矩
        ds = SEG_LEN / n_per_seg
        for _ in range(n_per_seg):
            k1 = ca.vertcat(ca.sin(th), ca.cos(th), kap)
            k2 = ca.vertcat(ca.sin(th + 0.5*ds*k1[2]),
                            ca.cos(th + 0.5*ds*k1[2]), kap)
            k3 = ca.vertcat(ca.sin(th + 0.5*ds*k2[2]),
                            ca.cos(th + 0.5*ds*k2[2]), kap)
            k4 = ca.vertcat(ca.sin(th + ds*k3[2]),
                            ca.cos(th + ds*k3[2]), kap)
            px = px + ds/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            pz = pz + ds/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            th = th + ds/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        outs.append((px, pz, th))
    return outs


def identify(P_true, noise_mm, ei, seed=0, fit_ei=False, ei_true=None):
    rng = np.random.default_rng(seed)
    sh_true = roller(kappa_of_P(P_true, ei))
    obs = sh_true + rng.normal(0, noise_mm*1e-3, sh_true.shape)
    obs[:, 2] *= 0.0; obs[:, 2] += sh_true[:, 2]     # 切向角用真值（视觉测角更准）
    obs[:, 2] += rng.normal(0, 0.005, NSEG)          # 0.3° 测角噪声

    opti = ca.Opti()
    P = opti.variable()
    opti.subject_to(opti.bounded(0.0, P, 1.0))       # 0–1 N
    ei_v = list(ei)
    if fit_ei:
        c = opti.variable()
        opti.subject_to(opti.bounded(0.5, c, 1.5))
        ei_v = [e * c for e in ei]
        opti.set_initial(c, 1.0)

    outs = rollout_casadi(P, ei_v)
    cost = 0
    for i in range(NSEG):
        px, pz, th = outs[i]
        cost += 1e6 * (px - obs[i, 0])**2 + 1e6 * (pz - obs[i, 1])**2
        cost += 20.0 * (th - obs[i, 2])**2
    opti.minimize(cost)
    opti.set_initial(P, 0.5 * P_true if P_true > 0 else 0.01)
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})
    t0 = time.time(); sol = opti.solve(); dt = time.time() - t0
    out = {"P": float(sol.value(P)), "dt": dt}
    if fit_ei:
        out["c"] = float(sol.value(c))
    return out, obs, sh_true


def main():
    ei = seg_ei()
    print("段 EI：段1 %.5f / 段2 %.5f N·m²" % (ei[0], ei[10]))
    for P in (0.005, 0.01, 0.02, 0.05):
        k0 = kappa_of_P(P, ei)[0]
        th = np.degrees(kappa_of_P(P, ei).sum() * SEG_LEN)
        tip = roller(kappa_of_P(P, ei))[-1, :2]
        print(f"  P={P*1e3:5.1f} mN  κ_root={k0:6.2f} 1/m  总转角 {th:5.1f}°  "
              f"末端 ({tip[0]*1e3:+6.1f}, {tip[1]*1e3:6.1f}) mm")
    print()

    P_true = 0.02
    print(f"辨识目标 P_true = {P_true*1e3:.1f} mN\n")
    print("噪声(mm) | 回收 P(mN) | 误差 | 用时(ms)")
    print("---------+------------+------+---------")
    for nm in (0.0, 0.1, 0.5, 1.0, 3.0):
        r, _, _ = identify(P_true, nm, ei)
        err = abs(r["P"] - P_true) / P_true * 100
        print(f"  {nm:4.1f}   |   {r['P']*1e3:7.3f}  | {err:4.1f}% | "
              f"{r['dt']*1000:5.0f}")

    print("\n同时辨识 P 与刚度比例 c（模拟 EI 有偏差）")
    r, _, _ = identify(P_true, 1.0, ei, fit_ei=True)
    print(f"  P = {r['P']*1e3:.3f} mN（真值 {P_true*1e3:.1f}），"
          f"c = {r['c']:.3f}（真值 1.000），{r['dt']*1000:.0f} ms")


if __name__ == "__main__":
    main()
