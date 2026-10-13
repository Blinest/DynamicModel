# -*- coding: utf-8 -*-
"""check_inverse_friction_branch.py — ★E「摩擦支接入逆解路径」的验证

背景（已定量的隐患）
------------------
`tendon_coupling` 里摩擦有加载(+1)/卸载(-1)两支，同一 τ 用不同支给出**不同位姿**
（μ=0.15 时末端差毫米级）。正解侧已能按 τ 历史自动选支，但**逆解路径完全没有接入**：
`inverse_solver` 只 import 了 `shooting_batch`，从头到尾用 `params.friction_branch`
那个手填常量。后果有两个，都是静默的：

  1. 牛顿迭代里 τ 会来回走 ⇒ 每轮的"位姿"其实对应不同的物理支 ⇒ 线搜索比的是
     两个不同物理模型的误差，收敛判据失去意义；
  2. **更要命**：`pose_jacobian_batch` 用 ±h 扰动差分算雅可比，±h 是**数学扰动、
     不是加载路径**。若让每个扰动自己判支，±h 可能落在不同支上 ⇒ J 直接失真。

本脚本验证已落地的修法：
  · `_newton_pose(friction_track=None)` —— 每轮从 (上一轮 τ → 本轮 τ) 判一次支；
  · 该轮的**中心点 + 12 个 ±h 扰动 + 线搜索候选**共用同一支（`params=` 透传副本，
    不改共享对象）；
  · μ=0 时自动关闭（不拷贝 params、不写入）⇒ 与原实现逐位一致。

四项检查
--------
  ① **μ=0 逐位回归**: 强制 friction_track=True 且 μ=0 → τ / tip 与关闭时 max|Δ|=0
  ② **每轮只判一次支**: 计数 branch_from_history 调用次数 == 迭代轮数
  ③ **噪音零副作用**: 调用后 model.params.friction_branch 原值、对象身份不变
  ④ **params 透传是精确的**: `pose_jacobian_batch(params=副本)` 与"把该支写进
     params 再算"的结果逐位一致；且 +1/-1 两支的 J 确实不同（证明这是真 bug）

跑: TC_NO_JIT=1 python tools/check_inverse_friction_branch.py [--nsub 30] [--mu 0.15]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import numba  # noqa: F401
except Exception:
    os.environ["TC_NO_JIT"] = "1"          # 无 numba → 纯 numpy 参考实现

import inverse_solver as IS  # noqa: E402
from inverse_solver import LookupInverseSolver  # noqa: E402
from tendon_coupling import TendonCouplingModel  # noqa: E402

BASE = np.array([40.0, 20.0, 40.0, 20.0, 40.0, 10.0])
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        FAIL.append(name)


def build(nsub, mu):
    M = TendonCouplingModel()
    M.params.n_p = nsub
    M.params.n_d = nsub
    M.params.mu_friction = float(mu)
    S = LookupInverseSolver(model=M)          # 无表 ⇒ 纯模型牛顿
    return M, S


def target_from(M, tau):
    """用一次正解取目标位姿（同工况、同一支）。"""
    o = M.shape(tau)
    return np.asarray(o["tip"], float), np.asarray(o["R_tip"], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=30)
    ap.add_argument("--mu", type=float, default=0.15)
    a = ap.parse_args()

    # ════════════════ ① μ=0 逐位回归（开关必须无差别） ════════════════
    print("[①] μ=0 逐位回归 —— 强制 track=True 也不得改变结果")
    M, S = build(a.nsub, 0.0)
    tau_ref = BASE * 0.80
    p_des, R_des = target_from(M, tau_ref)
    tau0 = BASE * 0.50
    t0 = perf_counter()
    r_off = S._newton_pose(p_des, R_des, tau0=tau0, max_iter=4, friction_track=False)
    r_on = S._newton_pose(p_des, R_des, tau0=tau0, max_iter=4, friction_track=True)
    dt = float(np.max(np.abs(np.asarray(r_off["tau"]) - np.asarray(r_on["tau"]))))
    dp = float(np.max(np.abs(np.asarray(r_off["p_tip"]) - np.asarray(r_on["p_tip"]))))
    print(f"      max|Δτ| = {dt:.3e} N   max|Δtip| = {dp:.3e} m   "
          f"({perf_counter()-t0:.1f} s)")
    check("μ=0: track 开/关逐位一致 (max|Δτ|=0)", dt == 0.0 and dp == 0.0)
    check("μ=0: 支历史为空 (未误触发)", len(getattr(S, "branch_hist", [])) == 0)

    # ════════════════ ② 每轮只判一次支 ════════════════
    print(f"[②] μ={a.mu}: 每轮迭代只判一次支（12 个 ±h 扰动不得各自判）")
    M, S = build(a.nsub, a.mu)
    p_des, R_des = target_from(M, tau_ref)
    calls = []
    _orig = IS.branch_from_history

    def _rec(tp, tn, pb=1.0, deadband=1e-9):
        out = _orig(tp, tn, pb, deadband=deadband)
        calls.append(float(out))
        return out

    IS.branch_from_history = _rec
    try:
        t0 = perf_counter()
        r = S._newton_pose(p_des, R_des, tau0=tau0, max_iter=4)
        el = perf_counter() - t0
    finally:
        IS.branch_from_history = _orig
    hist = list(getattr(S, "branch_hist", []))
    n_it = len(hist)
    print(f"      迭代 {n_it} 轮, branch_from_history 调用 {len(calls)} 次, "
          f"支序列 {[int(v) for v in hist]}  ({el:.1f} s)")
    check("调用次数 == 迭代轮数", len(calls) == n_it and n_it > 0)
    check("支取值只含 ±1", all(v in (1.0, -1.0) for v in hist))
    check("本轮之内不再重判 (无 12 倍放大)", len(hist) <= 4 + 1)
    print(f"      收敛: {bool(r.get('converged'))}  |Δp| = "
          f"{np.linalg.norm(np.asarray(r['p_tip']) - p_des)*1000:.4f} mm")

    # ════════════════ ③ 无副作用 ════════════════
    print("[③] 调用后 params 不得被改写 / 不得被替换")
    M, S = build(a.nsub, a.mu)
    M.params.friction_branch = +1.0
    before_val = float(M.params.friction_branch)
    before_id = id(M.params)
    S._newton_pose(p_des, R_des, tau0=tau0, max_iter=2)
    after_val = float(M.params.friction_branch)
    after_id = id(M.params)
    check("friction_branch 原值保持", before_val == after_val,
          f"{before_val:+.0f} → {after_val:+.0f}")
    check("params 对象未被替换 (id 不变)", before_id == after_id)

    # ════════════════ ④ params 透传精确 + 两支 J 确实不同 ════════════════
    print("[④] params= 透传是否精确；+1/-1 两支的 J 是否有差异")
    M, S = build(a.nsub, a.mu)
    M.params.friction_branch = +1.0
    tau_mid = BASE * 0.70
    x0 = np.asarray(M.shape(tau_mid)["x"], float).reshape(6)
    # (a) 透传副本 == 把该支写进 params
    Pc = copy.copy(M.params)
    Pc.friction_branch = -1.0
    J_pas, base_pas = S.pose_jacobian_batch(tau_mid, x0=x0, params=Pc)
    M.params.friction_branch = -1.0
    try:
        J_set, base_set = S.pose_jacobian_batch(tau_mid, x0=x0)
    finally:
        M.params.friction_branch = +1.0
    dJ = float(np.max(np.abs(J_pas - J_set)))
    dtip = float(np.max(np.abs(np.asarray(base_pas["tip"]) - np.asarray(base_set["tip"]))))
    check("透传副本与直接写 params 逐位一致", dJ == 0.0 and dtip == 0.0,
          f"max|ΔJ|={dJ:.3e}  max|Δtip|={dtip:.3e} m")
    # (b) 两支的 J 与位姿差异（证明"不接入"是真问题）
    Pplus = copy.copy(M.params); Pplus.friction_branch = +1.0
    J_p, base_p = S.pose_jacobian_batch(tau_mid, x0=x0, params=Pplus)
    J_m, base_m = S.pose_jacobian_batch(tau_mid, x0=x0, params=Pc)
    dJ2 = float(np.max(np.abs(J_p - J_m)))
    dp2 = float(np.linalg.norm(np.asarray(base_p["tip"]) - np.asarray(base_m["tip"]))) * 1000.0
    print(f"      两支差异: max|ΔJ| = {dJ2:.4e}  |Δtip| = {dp2:.4f} mm")
    check("两支的 J 可分辨 (证明接入有意义)", dJ2 > 0.0 and dp2 > 0.01,
          f"|Δtip|={dp2:.4f} mm")

    print("=" * 74)
    if FAIL:
        print(f"✗ {len(FAIL)} 项失败: {FAIL}")
        return 1
    print("✓ 全部通过 —— 摩擦支已按 τ 历史接入逆解；μ=0 逐位回归无回归。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
