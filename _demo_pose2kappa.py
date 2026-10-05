# -*- coding: utf-8 -*-
"""演示: 给末端位姿 -> 解算 κ(s) 与 ΔL"""
import sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from inverse_solver import LookupInverseSolver

L = LookupInverseSolver(path="vc_table_60.npz")
M = L.model
r = L.table.recs[500]
p_des = np.asarray(r["p_tip"], float)
R_des = np.asarray(r["R_tip"], float)
tau_true = np.asarray(r["tau"], float)
x_true = np.asarray(r["x"], float)

# ── 真值 (由该记录的 τ 全阶正解) ──
cv, _ = M.curve_batch(tau_true[None, :], x=x_true[None, :])
S = np.asarray(cv[1], float)
kap_true = np.asarray(cv[2], float)[0]
dL_true = np.asarray(r["dL_mm"], float)

print("输入位姿: p = %s mm" % np.round(p_des * 1000, 3))
print("          R =\n%s" % np.round(R_des, 4))

# ── 逆解 ──
o = L.solve_pose(p_des, R_des, pod=False, max_iter=3)
dL = np.asarray(o["dL_mm"], float)
sol = o["sol"]
kap = np.asarray(sol["u_curve"], float)
nu = np.asarray(sol["v_curve"], float)

print("\n── 输出 ──")
print("τ  [N]      = %s" % np.round(o["tau"], 3))
print("ΔL [mm]     = %s" % np.round(dL, 3))
print("ΔL 真值[mm] = %s" % np.round(dL_true, 3))
print("ΔL 误差[mm] = %s   (max %.4f)" % (np.round(dL - dL_true, 4), np.abs(dL - dL_true).max()))
print("迭代 %d  残差 Δp=%.4f mm  Δθ=%.3f mrad" % (o["n_iter"], o["pos_res_mm"], o["rot_res_mrad"]))

print("\n── κ(s) 场 ──")
print("节点 %s   s ∈ [%.4f, %.4f] m" % (kap.shape, S[0], S[-1]))
print("|κ| max  解算=%.4f  真值=%.4f  1/m" % (np.linalg.norm(kap, axis=1).max(),
                                             np.linalg.norm(kap_true, axis=1).max()))
print("κ 逐点最大差 = %.6f 1/m" % np.abs(kap - kap_true).max())
print("s(mm)  |κ|解算  |κ|真值   νz解算   νz真值")
for k in (0, 100, 300, 500, 600):
    print("%6.1f  %8.4f %8.4f %8.5f %8.5f" % (
        S[k] * 1000, np.linalg.norm(kap[k]), np.linalg.norm(kap_true[k]),
        nu[k, 2], np.asarray(cv[6], float)[0][k, 2]))
print("\nDONE")
