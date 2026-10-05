# -*- coding: utf-8 -*-
"""
inverse_solver.py — 肌腱逆解模型 (Tendon Inverse Model, TIM)
===========================================================
职责分离 (与正解模型**互不重叠**):

    正解  TendonCouplingModel   (tendon_coupling.py)     τ  →  位姿 / ΔL
    逆解  LookupInverseSolver   (本模块, 唯一逆解类)      位姿 / ΔL  →  τ

本模块**不重写任何物理**: 位姿、ΔL、耦合矩阵全部向正解模型取值,
只负责"反解"这一层 —— 迭代格式、雅可比、收敛判据、安全校验。

一个类, 两级能力
----------------
    LookupInverseSolver
      ├─ 无表 (table=None)  : 纯模型逆解
      │     ├─ solve_dL(ΔL)      阻尼 Newton, 雅可比 = 耦合矩阵 J=∂ΔL/∂τ
      │     └─ solve_pose(p*,R*) 阻尼 Newton, 雅可比 = ∂[p;rotvec]/∂τ
      └─ 有表 (table/path)  : 查表定初值 → 模型精化 (★推荐)
            ├─ 查表提供**全局初值**并规避多平衡态 (τ→pose 多对一)
            └─ 模型把残差压到亚毫米

    ⚠ 为什么不能只用模型: τ→pose 多对一, 强弯区冷启动牛顿会落错分支。
    ⚠ 为什么不能只用查表: 表点 NN 中位距离是数十 mm 量级, 插值后仍有 mm 级残差。

用法
----
    from inverse_solver import LookupInverseSolver

    # ① 纯模型逆解 (不需要表)
    S = LookupInverseSolver()
    S.solve_dL([-5, 0, -5, 0, -5, 2])              # ΔL → τ
    S.solve_pose(p_des, R_des, tau0=tau_guess)     # 位姿 → τ

    # ② 查表 + 精化 (强弯区推荐)
    L = LookupInverseSolver(path="vc_table_40.npz")
    L.solve_pose(p_des, R_des)                     # 查表定位 + 模型精化
"""
import numpy as np

from tendon_coupling import (TendonCouplingModel, TendonCouplingResult,
                             L, L1, SEG_A)
from ik_table import (PoseTable, load_table, so3_log,
                      apply_safety, table_version_key,
                      # ── POD 降阶模型需要 ──
                      default_params, _L0_wire, _wire_geom_len_batch)
from tendon_coupling import integrate_shape_batch            # 几何积分 (已下沉内核)
from tendon_coupling import (forward_batch,        # 全阶正解 (建基快照用)
                             shooting_batch)       # ★ 批量打靶 (位姿雅可比提速)

__all__ = ["LookupInverseSolver", "PODReducedOrderModel",
           "solve_tendon_forces", "solve_pose", "make_inverse_solver"]


# ══════════════════════════════════════════════════════════════════════
# 肌腱逆解模型
# ══════════════════════════════════════════════════════════════════════
class LookupInverseSolver:
    """★ 肌腱逆解模型: 位姿 / ΔL → τ.

    参数
    ----
    table / path : PoseTable | str | None
        给了 → 两级逆解 (查表定初值 + 模型精化); 不给 → 纯模型逆解。
    model   : TendonCouplingModel | None   给定时直接复用 (推荐, 表参数已同步)
    params  : SDMParams | None             未给 model 时用它构造正解模型
    F_tip / g_world / h_tau   同正解模型
    tau_max : float                张力上限 (丝只能拉, 钳位到 [0, tau_max])
    check_version : bool           有表时校验表与模型的版本键是否一致
    """

    def __init__(self, table=None, path=None, model=None, params=None,
                 F_tip=(0, 0, 0), g_world=(0, 0, -9.81), h_tau=1e-3,
                 tau_max=60.0, check_version=True, verbose=False,
                 pod_refine=False, pod_r=16, pod_r_v=10, pod_order=0, pod_nsub=600):
        self.verbose = bool(verbose)
        if table is None and path is not None:
            table = load_table(path)
        self.table = table
        if model is None:
            p = params if params is not None else \
                (table.params if table is not None else None)
            model = TendonCouplingModel(params=p, F_tip=F_tip,
                                        g_world=g_world, h_tau=h_tau)
        self.model = model
        self.tau_max = float(tau_max)
        self._cm = None                     # 粗网格精化模型缓存 (n, model)
        # ── POD 降阶精化 (快, 但见 solve_pose 的精度警示; 默认关, 用 pod=True 开启) ──
        self.pod_refine = bool(pod_refine) and table is not None
        self.pod_r = int(pod_r); self.pod_r_v = int(pod_r_v)
        self.pod_order = int(pod_order)
        self.pod_nsub = int(pod_nsub)
        self._pod = None                    # (basis, tsp) 惰性构建
        # ── S4 共享雅可比路径的诊断计数 (2026-09-27 加) ──
        self.n_sj_ok = 0          # _shoot_shared_jac 直接成功 (快路径)
        self.n_sj_fallback = 0    # 回退到 _solve_statics 自适应重试链 (慢路径)
        self.t_sj = 0.0           # 快路径累计耗时 (s)
        self.t_fb = 0.0           # 慢路径累计耗时 (s)
        self.version_ok = True
        self.version_diff = {}
        if check_version and table is not None:
            self.version_ok, self.version_diff = self._check_version()

    # ── 透传 ──
    @property
    def params(self):
        return self.model.params

    @property
    def cell(self):
        return self.model.cell

    @property
    def h_tau(self):
        return self.model.h_tau

    @property
    def has_table(self):
        return self.table is not None

    # ───────────────────────────────── POD 降阶精化 (推荐)
    def _pod_fit(self):
        """惰性构建 POD 基 (κ 与 ν **各一套**) + τ→[a_κ;a_ν] 拟合.

        ⚡ 数据源 = `model.curve_batch()` 的全阶曲线场 (正确口径)。
        ⚠ **不能用表内 kcoef**: 其 K 曾按旧"段A×2"换算 (已修, 但旧表数据仍旧) ⇒ 会差 2 倍。
        ⚠ κ / ν 必须**分开建基**: 堆叠成一个矩阵时 ν 的常数分量 (v_z≈0.85) 会挤占模态预算。
        """
        if self._pod is not None:
            return self._pod
        recs = self.table.recs
        taus = np.array([np.asarray(r["tau"], float) for r in recs])
        xs = np.array([np.asarray(r["x"], float) for r in recs])
        cv, _ = self.model.curve_batch(taus, x=xs)
        Uc = np.asarray(cv[2], float)          # κ(s) (N,nS,3)  ← 正确曲率
        Vv = np.asarray(cv[6], float)          # ν(s) (N,nS,3)
        S_grid = np.asarray(cv[1], float)
        N, nS, _ = Uc.shape

        def _pod(X, r):
            F = X.reshape(N, -1); m = F.mean(axis=0)
            U, Sv, Vt = np.linalg.svd(F - m, full_matrices=False)
            r = int(min(r, len(Sv)))
            return m, Vt[:r], Sv, r, float((Sv[:r] ** 2).sum() / (Sv ** 2).sum())

        mk, Mk, Sk, rk, ek = _pod(Uc, self.pod_r)
        mv, Mv, Sv_, rv, ev = _pod(Vv, self.pod_r_v)
        Ak = (Uc.reshape(N, -1) - mk) @ Mk.T
        Av = (Vv.reshape(N, -1) - mv) @ Mv.T
        A = np.concatenate([Ak, Av], axis=1)
        # τ→a: **表内 kNN 局部线性插值** (全局多项式在 τmax=60 下残差过大)
        from scipy.spatial import cKDTree
        from ik_table import HAVE_SCIPY
        tree = cKDTree(taus) if HAVE_SCIPY else None
        # 自检: 留出 10% 估插值精度
        rng = np.random.default_rng(0)
        ii = rng.choice(N, size=min(120, N), replace=False)
        a_est = _a_interp(taus[ii], taus, A, tree, k=8, ridge=1e-6)
        err = float(np.median(np.linalg.norm(a_est - A[ii], axis=1)))
        basis = {"mk": mk, "Mk": Mk, "mv": mv, "Mv": Mv, "nS": nS,
                 "rk": rk, "rv": rv, "S_grid": S_grid,
                 "energy_k": ek, "energy_v": ev, "params": self.model.params,
                 "TAU": taus, "A": A, "tree": tree, "k": 8}
        tsp = {"basis": basis, "fit_err": err}
        if self.verbose:
            print(f"  [POD] κ基 r={rk} 能量={ek*100:.4f}% | ν基 r={rv} 能量={ev*100:.4f}%"
                  f"  â(τ) kNN 插值残差中位={err:.3e}", flush=True)
        self._pod = (basis, tsp)
        return self._pod

    def _pod_pose_a(self, A, B, nsub):
        """模态系数 a(N,r) → tip(N,3), R_tip(N,3,3), dL(N,6)."""
        rk = B["rk"]; nS = B["nS"]
        kf = (B["mk"][None, :] + (A[:, :rk] @ B["Mk"])).reshape(-1, nS, 3)
        vf = (B["mv"][None, :] + (A[:, rk:] @ B["Mv"])).reshape(-1, nS, 3)
        ps, Rs = integrate_shape_batch(kf, B["S_grid"], nsub, v_batch=vf)
        dL = (_wire_geom_len_batch(ps, Rs) - _L0_wire()) * 1000.0
        return ps[:, -1], Rs[:, -1], dL

    def _pod_forward(self, taus, tsp, nsub):
        """POD 降阶前向: τ(N,6) → tip(N,3), R_tip(N,3,3), dL(N,6), a(N,r)."""
        B = tsp["basis"]
        A = _a_interp(np.atleast_2d(np.asarray(taus, float)), B["TAU"], B["A"],
                      B["tree"], k=B["k"], ridge=1e-6)
        tp, Rp, dL = self._pod_pose_a(A, B, nsub)
        return tp, Rp, dL, A

    def _pod_newton(self, p_des, R_des, tau0, max_iter=12, tol_pos_mm=0.05,
                    tol_rot_mrad=0.5, h=1e-4, verbose=False, verify=True,
                    trust0=8.0, lam=1e-3, n_tau_iter=3):
        """POD 降阶位姿牛顿 —— **在模态系数 a-空间求解** (而非 τ-空间).

        为什么换到 a-空间 (关键):
          · ∂pose/∂a 由纯积分给出, **完整光滑、无插值** ⇒ 牛顿稳定,
            不受 kNN 梯度"只在本地点有效"的限制 (τ-空间版本大扰动必然失败);
          · 未知 26 个 vs 方程 6 个 ⇒ 欠定, 取**最小范数步** = 最小改动模态;
        收敛后把 a* 用**局部线性 a(τ)** 迭代映回 τ (保持张力分支)。

        ⚠ 精度底噪 ~0.3 mm: `integrate_shape_batch` 重建本身 0.29 mm + 基截断。
        """
        basis, tsp = self._pod_fit(); B = basis
        p_des = np.asarray(p_des, float).reshape(3)
        R_des = np.asarray(R_des, float).reshape(3, 3)
        tau0 = np.asarray(tau0, float).reshape(6)
        nsub = self.pod_nsub
        rdim = B["rk"] + B["rv"]

        A0, dAdq = _a_interp_w(tau0[None, :], B["TAU"], B["A"], B["tree"], k=B["k"])
        a = A0[0].copy()
        converged = False
        hist = []
        it = 0
        for it in range(max_iter):
            tp, Rp, _ = self._pod_pose_a(a[None, :], B, nsub)
            p = np.asarray(tp[0], float); R = np.asarray(Rp[0], float)
            e = np.concatenate([(p - p_des) * 1000.0, so3_log(R_des.T @ R) * 1000.0])
            pos_mm = float(np.linalg.norm(e[:3])); rot_mrad = float(np.linalg.norm(e[3:]))
            hist.append((pos_mm, rot_mrad))
            if verbose:
                print(f"  [POD-a] it{it}: Δp={pos_mm:.4f} mm  Δθ={rot_mrad:.3f} mrad",
                      flush=True)
            if pos_mm < tol_pos_mm and rot_mrad < tol_rot_mrad:
                converged = True
                break
            # ∂e/∂a (6×r) —— a-空间中心差分 (光滑, 不受插值噪声影响)
            Ja = np.zeros((6, rdim))
            for m in range(rdim):
                ap = a.copy(); ap[m] += h
                am = a.copy(); am[m] -= h
                tpp, Rpp, _ = self._pod_pose_a(ap[None, :], B, nsub)
                tpm, Rpm, _ = self._pod_pose_a(am[None, :], B, nsub)
                ep = np.concatenate([(np.asarray(tpp[0]) - p) * 1000.0,
                                     so3_log(R.T @ np.asarray(Rpp[0])) * 1000.0])
                em = np.concatenate([(np.asarray(tpm[0]) - p) * 1000.0,
                                     so3_log(R.T @ np.asarray(Rpm[0])) * 1000.0])
                Ja[:, m] = (ep - em) / (2.0 * h)
            # 最小范数步 (欠定): Δa = -Jᵀ(JJᵀ+λI)⁻¹ e
            try:
                da = -Ja.T @ np.linalg.solve(Ja @ Ja.T + lam * np.eye(6), e)
            except np.linalg.LinAlgError:
                da = -Ja.T @ (np.linalg.pinv(Ja @ Ja.T) @ e)
            if not np.isfinite(da).all():
                break
            n0 = float(np.linalg.norm(e)); best = None
            for al in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 3e-3):
                a_c = a + al * da
                t2, R2, _ = self._pod_pose_a(a_c[None, :], B, nsub)
                e2 = np.concatenate([(np.asarray(t2[0]) - p_des) * 1000.0,
                                     so3_log(R_des.T @ np.asarray(R2[0])) * 1000.0])
                n2 = float(np.linalg.norm(e2))
                if n2 < n0 * 0.999:
                    best = (a_c, n2)
                    break
                if best is None or n2 < best[1]:
                    best = (a_c, n2)
            if best is None or best[1] >= n0:
                break
            a = best[0]
        # ── a* → τ: 局部线性 a(τ) 上做不动点迭代 (保持分支) ──
        tau = tau0.copy()
        for _ in range(int(n_tau_iter)):
            aq, dadq = _a_interp_w(tau[None, :], B["TAU"], B["A"], B["tree"], k=B["k"])
            D = dadq[0]                                   # (r,6)
            try:
                dta = D.T @ np.linalg.solve(D @ D.T + lam * np.eye(rdim), (a - aq[0]))
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(dta).all():
                break
            tau = np.clip(tau + np.clip(dta, -trust0, trust0), 0.0, self.tau_max)
        if verify:
            sol = self.model.shape(tau)
            e2 = self.pose_residual(tau, p_des, R_des, sol=sol)
            dLv = self.model.delta_L_mm(tau, sol=sol)
        else:
            tp, Rp, dLv = self._pod_pose_a(a[None, :], B, nsub)
            sol = None
            e2 = np.concatenate([(np.asarray(tp[0]) - p_des) * 1000.0,
                                 so3_log(R_des.T @ np.asarray(Rp[0])) * 1000.0])
            dLv = np.asarray(dLv[0], float)
        return TendonCouplingResult(
            tau=tau, sol=sol, a=a,
            pos_res_mm=float(np.linalg.norm(e2[:3])),
            rot_res_mrad=float(np.linalg.norm(e2[3:])),
            dL_mm=dLv,
            p_tip=(np.asarray(sol["tip"], float) if sol is not None else np.asarray(tp[0], float)),
            R_tip=(np.asarray(sol["R_tip"], float) if sol is not None else np.asarray(Rp[0], float)),
            converged=converged, n_iter=it + 1, hist=hist,
            refine="pod-a", pod_r=B["rk"], pod_pos_mm=hist[-1][0],
            pod_rot_mrad=hist[-1][1], verified=bool(verify))

    def _solve_refine(self, p_des, R_des, tau0, max_iter, tol_pos_mm,
                      tol_rot_mrad, lam, verbose, refine_n, pod=None,
                      verify=None):
        """精化分派: **POD 降阶**(推荐, ms 级) 或 全阶牛顿."""
        use_pod = self.pod_refine if pod is None else bool(pod)
        if use_pod and tau0 is not None:
            return self._pod_newton(p_des, R_des, tau0, max_iter=max_iter,
                                    tol_pos_mm=tol_pos_mm,
                                    tol_rot_mrad=tol_rot_mrad, verbose=verbose,
                                    verify=(True if verify is None else bool(verify)))
        return self._newton_pose(p_des, R_des, tau0=tau0, max_iter=max_iter,
                                 tol_pos_mm=tol_pos_mm,
                                 tol_rot_mrad=tol_rot_mrad, lam=lam,
                                 verbose=verbose, refine_n=refine_n)

    def _refine_model(self, n):
        """精化用的正解模型: n=None → 表自带分辨率; n 给定 → 缓存一个 n×n 粗网格模型.

        ⚡ 表已给出极好的热启动, 精化不必在高分辨率上做 —— n=60 比 n=300 快 ~25×。
        ⚠ 代价: 残差按粗网格量得, n=60 离散误差约 mm 级 (n=300 约 0.1 mm)。
        """
        if n is None:
            return self.model
        if self._cm is None or self._cm[0] != int(n):
            from ik_table import _clone_params
            pr = _clone_params(self.model.params, n_p=int(n), n_d=int(n))
            pr.cells = getattr(self.model.params, "cells", None)
            self._cm = (int(n), TendonCouplingModel(
                params=pr, F_tip=self.model.F_tip,
                g_world=self.model.g_world, h_tau=self.model.h_tau))
        return self._cm[1]

    # ══════════════════════════════════ 逆解内核
    def pose_residual(self, tau, p_des, R_des, sol=None, ref=None, model=None):
        """位姿残差 e = [ (p−p*)(mm) ; log(R*ᵀ R)(mrad) ] ∈ R⁶.

        ref: (p_ref, R_ref) 姿态差分基准; None → 用目标姿态本身。
        model: 用哪个正解模型求值 (None → self.model; 精化可传粗网格模型)
        """
        M = self.model if model is None else model
        p, R = M.pose(tau, sol=sol)
        pr, Rr = (np.asarray(p_des, float), np.asarray(R_des, float)) \
            if ref is None else ref
        dp = (p - np.asarray(p_des, float)) * 1000.0
        dr = so3_log(np.asarray(Rr, float).T @ R) * 1000.0
        return np.concatenate([dp, dr])

    def pose_jacobian(self, tau, h=None, x0=None, model=None, sol=None,
                      max_iter=4):
        """6×6  J = ∂[p(mm); rotvec(mrad)] / ∂τ(N) —— **批量**一次算完 12 个扰动.

        ⚡ 原实现逐列调用标量 `shooting` (12 次全阶打靶, n=300 时单步 ~2 min);
           现改为 **1 次 `shooting_batch` + 1 次 `forward_batch`**, 数学完全一致
           (同一基准姿态做中心差分, rotvec 差分自洽)。
        """
        M = self.model if model is None else model
        tau = np.asarray(tau, float).reshape(6)
        h = M.h_tau if h is None else float(h)
        base = sol if sol is not None else M.shape(tau, x0=x0)   # ⚡ 复用基态解
        p_ref = np.asarray(base["tip"], float)
        R_ref = np.asarray(base["R_tip"], float)
        x0b = np.asarray(base["x"], float)

        taus = np.repeat(tau[None, :], 12, axis=0)      # 中心差分: 6 维 × ±h
        for j in range(6):
            taus[2 * j, j] += h
            taus[2 * j + 1, j] -= h
        out = shooting_batch(taus, M.params, x0=np.tile(x0b, (12, 1)),
                             tol=1e-8, max_iter=max_iter)   # 已有热启动, 几步即收敛
        _, _, cv = forward_batch(out["x"], taus, M.params, want_curve=True)
        p_all = np.asarray(cv[0], float)[:, -1]         # Pc 末节点
        R_all = np.asarray(cv[4], float)[:, -1]         # Rc 末节点
        J = np.zeros((6, 6))
        for j in range(6):
            ep = np.concatenate([(p_all[2 * j] - p_ref) * 1000.0,
                                 so3_log(R_ref.T @ R_all[2 * j]) * 1000.0])
            em = np.concatenate([(p_all[2 * j + 1] - p_ref) * 1000.0,
                                 so3_log(R_ref.T @ R_all[2 * j + 1]) * 1000.0])
            J[:, j] = (ep - em) / (2.0 * h)
        return J

    def reachability(self, tau, e, J=None, model=None, tol_frac=0.5):
        """**可达性判据** (线性化 + 执行器箱约束投影).

        原理: 消除当前残差所需的最小范数 τ 步 = `dtau = -J⁺ e`。
        把它按箱约束 `[0, τ_max]` 投影后, 用一阶预测估计**残差地板**:
            floor ≈ ‖ e + J·(clip(τ+dtau) − τ) ‖
        若 `floor` 已接近当前残差(占比 > tol_frac) ⇒ 箱约束挡住了修正
        ⇒ 目标在**可达集外**(或贴边界), 继续牛顿迭代无望。

        实测: 表点 #500 (τ₂=59.47 ≈ τ_max=60) 做 +5mm 扰动
              ⇒ need=732 N, 而 τ_max=60 N (需 12 倍全行程) ⇒ infeasible。
        """
        M = self.model if model is None else model
        tau = np.asarray(tau, float).reshape(6)
        e = np.asarray(e, float).reshape(-1)
        if J is None:
            J, _ = self.pose_jacobian_batch(tau, model=M)
        S = J @ J.T + 1e-10 * np.eye(6)
        try:
            dtau = -J.T @ np.linalg.solve(S, e)
        except np.linalg.LinAlgError:
            dtau = -J.T @ (np.linalg.pinv(S) @ e)
        n0 = float(np.linalg.norm(e))
        need = float(np.linalg.norm(dtau))
        room = float(np.min(np.minimum(tau, self.tau_max - tau)))
        # ── 沿最小范数方向求**箱约束下可行的最大步长比** s ∈ (0,1] ──
        #    ⚠ 不能用 `clip(τ+dτ)` 直接算地板: 当 need≫τmax 时(如 311N vs 30N)
        #      六个分量全被削掉, 投影方向已完全扭曲, 会算出 floor>now 的荒谬值
        #      (实测 #309 +5mm: floor=79.7 > now=7.07)。必须**等比缩放**保方向。
        #    由于 dτ 是最小范数解 ⇒ J·dτ = -e 精确成立 ⇒ 预测残差 = (1-s)·‖e‖。
        eps = 1e-12
        s = 1.0
        sat = []
        for k in range(6):
            dk = float(dtau[k])
            if dk > eps:
                s = min(s, (self.tau_max - tau[k]) / dk)
            elif dk < -eps:
                s = min(s, tau[k] / (-dk))
        s_raw = float(s)
        s = float(max(0.0, min(1.0, s)))
        if s_raw < 1.0 - 1e-9:                   # 哪些丝把步长卡住了
            for k in range(6):
                dk = float(dtau[k])
                lim = ((self.tau_max - tau[k]) / dk if dk > eps
                       else (tau[k] / (-dk) if dk < -eps else np.inf))
                if lim <= s_raw + 1e-9:
                    sat.append(int(k))
        floor = float((1.0 - s) * n0)
        # ★ 用途2: 投影解 = 沿最小范数方向走"箱约束下可行的最大等比步长 s"
        #   (等比缩放保方向; 不能用 clip(τ+dτ), 见上注释)
        tau_proj = np.clip(tau + s * dtau, 0.0, self.tau_max)
        # ⚠ 判据的**局限性** (2026-09-26 实测修正):
        #   `feasible` 只是**局部一阶**判据 —— 它无法区分
        #     (a) 目标真的在可达集外     vs (b) 目标可达但离热启动很远(需多步迭代)
        #   实测反例: 由正解生成的**必然可达**目标, 因热启动较远使 s 很小,
        #   被误判 feasible=False 且提前退出 ⇒ 直接把可解问题判死。
        #   ⇒ `feasible` 仅作**参考指标**(need/room/step_frac 一起看);
        #     只有 `blocked`(箱约束在本地**完全**阻断, s≈0) 才允许提前退出。
        return {"need_N": need, "room_N": room, "now": n0, "floor": floor,
                "step_frac": s, "ratio": (floor / n0 if n0 > 0 else 0.0),
                "saturated": sat, "tau_proj": tau_proj,
                "blocked": bool(s <= 1e-9),
                "feasible": bool(floor <= tol_frac * n0 + 1e-12)}

    def _shoot_shared_jac(self, taus, x0, model=None, v_bound=3.0, u_bound=60.0):
        """⚡ S4: 中心 + 12 扰动**共享内层雅可比**的批量打靶.

        问题（已定量）: `shooting_batch(N=13)` 每 LM 迭代要 13 次 `forward_batch`,
        而每次调用都处理**全部 13 个样本** ⇒ 实际 13×13 = **169 次单样本前向**
        (≈6.9 s, 与实测 ~7 s 平台吻合)。

        本函数利用"12 个扰动只是 h≈1e-3 N 的微小扰动, 内层雅可比 J_x 与中心点几乎相同":
          ① 中心点 (N=1) 正常打靶            ~14 次单样本前向
          ② 在中心解处算 J_x (N=1)           ~13 次
          ③ 12 个扰动: 一次批量前向得 res     ~12 次
            再用**共享 J_x** 各做 1 步牛顿    (纯线性代数, 免费)
        ⇒ 约 39 次单样本前向 (原 169) ⇒ **~4.3x**。

        返回 (x (M,6), converged (M,), J (nx,6) 中心雅可比)
        """
        M = self.model if model is None else model
        P = M.params
        taus = np.atleast_2d(np.asarray(taus, float))
        Mrow = taus.shape[0]
        lo = -np.array([v_bound] * 3 + [u_bound] * 3)
        hi = np.array([v_bound] * 3 + [u_bound] * 3)
        x0 = np.asarray(x0, float).reshape(6)
        # ① 中心点 —— 直调 shooting_batch (**不用** _solve_statics 的自适应重试链:
        #    实测该链在中心点会跑到 2→8→12 共 22 次 LM 迭代 ≈ 7 s, 是主要开销)
        #  ⚠ 2026-09-27: max_iter 原为 3, 但 n=300 下热启动后**中位就要 4 次**才到
        #    tol=1e-8 (见 ik_table.refine_table 实测) ⇒ 中心点几乎必然 ok_c=False
        #    ⇒ 上层 `if not ok_all` 回退到 45~217 s 的自适应重试链, S4 优化被整个旁路。
        #    实测证据: 表点 #500 (n=300) 单次 solve_pose = 47.9 s, 与"回退路径"量级吻合。
        out_c = shooting_batch(taus[0][None, :], P, x0=x0[None, :],
                               tol=1e-8, max_iter=12)
        ok_c = bool(out_c["converged"][0])
        xc = np.clip(out_c["x"][0], lo, hi)
        if Mrow == 1:
            return xc[None, :], np.array([bool(ok_c)]), None
        # ② 中心处的内层雅可比 J_x (nx,6)
        res0, _ = forward_batch(xc[None, :], taus[0][None, :], P)
        nx = res0.shape[1]
        J = np.zeros((nx, 6)); h = 1e-6
        for c in range(6):
            xp = xc.copy(); xm = xc.copy()
            xp[c] += h; xm[c] -= h
            rp, _ = forward_batch(xp[None, :], taus[0][None, :], P)
            rm, _ = forward_batch(xm[None, :], taus[0][None, :], P)
            J[:, c] = (rp[0] - rm[0]) / (2 * h)
        J = np.where(np.isfinite(J), J, 0.0)
        # ③ 12 个扰动: 共享 J_x 的牛顿迭代 (实测 1 步差 7e-5 ⇒ 取 3 步)
        tp = taus[1:]
        np_ = tp.shape[0]
        xp12 = np.tile(xc, (np_, 1))
        r_new = None
        for _ in range(3):
            r_new, _ = forward_batch(xp12, tp, P)
            if np.all(np.linalg.norm(r_new[:, :6], axis=1) < 1e-9):
                break
            try:
                dx = -np.linalg.solve(J, r_new.T).T          # (np_, 6)
            except np.linalg.LinAlgError:
                dx = -(np.linalg.pinv(J) @ r_new.T).T
            dx = np.where(np.isfinite(dx), dx, 0.0)
            mx = np.max(np.abs(dx), axis=1, keepdims=True)
            dx = dx * np.where(mx > 1.0, 1.0 / np.maximum(mx, 1e-30), 1.0)
            xp12 = np.clip(xp12 + dx, lo, hi)
        n_new = np.linalg.norm(r_new[:, :6], axis=1)
        ok12 = np.isfinite(r_new).all(axis=1) & (n_new < 1e-7)
        x_all = np.vstack([xc[None, :], xp12])
        ok_all = np.concatenate([[bool(ok_c)], ok12])
        return x_all, ok_all, J

    def _solve_statics(self, taus, x0, model=None, tol=1e-8, max_iter=2,
                       hard_cap=12):
        """解静力学并**保证(或至少知道)收敛**.

        ⚠ 这是本轮定位到的根因: `shooting_batch` 在偏离热启动较远时会
          **静默不收敛**并返回垃圾位姿 —— 实测同一个 τ 用不同迭代次数评估,
          残差可从 129 mm 跳到 5 mm (差 1.6 个数量级), 直接使线搜索失效。
        这里按 max_iter → ×4 逐级重试(2→8→12), 直到所有样本收敛或到顶,
        (上限刻意压到 12: 单次不收敛评估最坏 22 次 LM 迭代 ≈ 97 s, 需与总耗时权衡)
        并把 `converged` 一并返回, 供调用方决定是否采信。

        返回 (shooting_batch 输出, all_converged: bool)
        """
        M = self.model if model is None else model
        taus = np.atleast_2d(np.asarray(taus, float))
        xx0 = None if x0 is None else np.asarray(x0, float)
        if xx0 is not None and xx0.ndim == 1:
            xx0 = np.tile(xx0[None, :], (taus.shape[0], 1))
        mi = max(int(max_iter), 1)
        while True:
            out = shooting_batch(taus, M.params, x0=xx0, tol=tol, max_iter=mi)
            ok = bool(np.all(out["converged"]))
            if ok or mi >= hard_cap:
                return out, ok
            mi = min(mi * 4, int(hard_cap))

    def _pose_eval(self, tau, p_des, R_des, x0, model=None, max_iter=2):
        """廉价(仅前向)位姿评估 —— 线搜索专用.

        与 `pose_jacobian_batch` 的区别: **只解一个 τ**(1 样本), 不做 12 个扰动差分,
        热启动 1~2 步打靶即可 ⇒ ~9 s vs ~45 s (约 5× 便宜)。
        """
        M = self.model if model is None else model
        tau = np.asarray(tau, float).reshape(6)
        xx0 = None if x0 is None else np.asarray(x0, float).reshape(1, 6)
        out, ok = self._solve_statics(tau[None, :], x0, model=M, max_iter=max_iter)
        _, _, cv = forward_batch(out["x"], tau[None, :], M.params, want_curve=True)
        p = np.asarray(cv[0], float)[0, -1]
        R = np.asarray(cv[4], float)[0, -1]
        e = np.concatenate([(p - p_des) * 1000.0, so3_log(R_des.T @ R) * 1000.0])
        sol = {"x": np.asarray(out["x"][0], float), "tip": p, "R_tip": R,
               "p_curve": np.asarray(cv[0], float)[0], "R_curve": np.asarray(cv[4], float)[0],
               "u_curve": np.asarray(cv[2], float)[0], "v_curve": np.asarray(cv[6], float)[0],
               "s_curve": np.asarray(cv[1], float),
               "converged": bool(ok)}
        return e, sol, bool(ok)

    def pose_jacobian_batch(self, tau, h=None, x0=None, model=None, max_iter=2):
        """⚡ 一次 `shooting_batch` **同时**得到基态解与 6×6 位姿雅可比.

        原实现每次迭代要: ① 标量 `shape()` 求基态 (~40 s, Levenberg 最多 60 迭代)
        ② 批量 12 扰动 (~10 s) ⇒ ~50 s/次迭代。
        本函数把中心点与 12 个扰动放进**同一次**批量打靶 (13 样本, 热启动),
        实测 ~10 s 即拿到两者 (forward_batch 仅 0.34 s) ⇒ **~5× 加速**。

        返回 (J (6,6), base_sol dict[含 p_curve/R_curve/tip/R_tip/x])
        """
        M = self.model if model is None else model
        tau = np.asarray(tau, float).reshape(6)
        h = M.h_tau if h is None else float(h)
        if x0 is None:
            x0 = self.model.shape(tau)["x"] if model is None else M.shape(tau)["x"]
        x0 = np.asarray(x0, float).reshape(6)
        taus = np.zeros((13, 6))
        taus[0] = tau
        for j in range(6):
            taus[1 + 2 * j] = tau; taus[1 + 2 * j][j] += h
            taus[2 + 2 * j] = tau; taus[2 + 2 * j][j] -= h
        # ── ⚡ 优化1: 中心点自适应收敛, 12 个扰动"从中心热启动 1 步" ──
        #   外层 12 个扰动只是 h≈1e-3 N 的微小扰动, 与中心点几乎同解;
        #   若让 13 个样本各自跑完整自适应重试(2→8→12), 会放大成 13×13 次打靶
        #   (实测 45~217 s)。改为: 中心保证收敛, 扰动共享中心 x 各走 1 次 LM。
        #   不收敛则回退到完整自适应(只在需要时付代价)。
        # ── ⚡ S4: 共享内层雅可比 (39 次单样本前向 vs 原 169) ──
        #   实测: 原实现 = 13 次 forward_batch 调用 × 每次 13 样本 = 169 次单样本前向
        #   ≈ 6.9 s (与 ~7 s 平台吻合)。新法: 中心(N=1)正常打靶 + 中心处算一次 J_x
        #   + 12 扰动一次批量前向 + 共享 J_x 各 1 步牛顿 ⇒ ~39 次 ⇒ ~4.3x。
        #   不达标则回退到完整自适应路径 (只在需要时付代价)。
        from time import perf_counter as _pc
        _t0 = _pc()
        x_all, ok_all_arr, _Jc = self._shoot_shared_jac(
            taus, np.asarray(x0, float), model=M)
        self.t_sj += _pc() - _t0
        ok_all = bool(np.all(ok_all_arr))
        if not ok_all:                    # 仅在共享雅可比**真失败**时回退(不应常触发)
            self.n_sj_fallback += 1
            _t1 = _pc()
            out_c, ok_c = self._solve_statics(tau[None, :], np.asarray(x0, float)[None, :],
                                              model=M, max_iter=max_iter)
            xc = out_c["x"][0]
            out_p, ok_p = self._solve_statics(taus[1:], np.tile(xc, (12, 1)), model=M,
                                              max_iter=max_iter)
            x_all = np.vstack([out_c["x"], out_p["x"]])
            ok_all = bool(ok_c and ok_p)
            self.t_fb += _pc() - _t1
        else:
            self.n_sj_ok += 1
        out = {"x": x_all,
               "converged": np.atleast_1d(ok_all_arr) if len(ok_all_arr) == len(taus)
                            else np.ones(len(taus), bool)}
        _, _, cv = forward_batch(out["x"], taus, M.params, want_curve=True)
        Pc = np.asarray(cv[0], float); Rc = np.asarray(cv[4], float)
        p0 = Pc[0, -1]; R0 = Rc[0, -1]
        J = np.zeros((6, 6))
        for j in range(6):
            pp, Rp = Pc[1 + 2 * j, -1], Rc[1 + 2 * j, -1]
            pm, Rm = Pc[2 + 2 * j, -1], Rc[2 + 2 * j, -1]
            ep = np.concatenate([(pp - p0) * 1000.0, so3_log(R0.T @ Rp) * 1000.0])
            em = np.concatenate([(pm - p0) * 1000.0, so3_log(R0.T @ Rm) * 1000.0])
            J[:, j] = (ep - em) / (2.0 * h)
        base = {"x": np.asarray(out["x"][0], float),
                "tip": p0, "R_tip": R0,
                "p_curve": Pc[0], "R_curve": Rc[0],
                # ★ 曲率场与线应变场 (中心样本) —— 位置/绳长的下游都要用
                "u_curve": np.asarray(cv[2], float)[0],
                "v_curve": np.asarray(cv[6], float)[0],
                "s_curve": np.asarray(cv[1], float),
                "converged": bool(ok_all)}
        return J, base

    def compliance_matrix(self, tau, h=None):
        """耦合矩阵的伪逆  C = J⁺  (N/mm): ΔL → Δτ 的线性灵敏度.

        返回 (C(6,6), cond(J))
        """
        out = self.model.coupling_matrix(tau, h=h)
        J = out["J"]
        s = np.linalg.svd(J, compute_uv=False)
        cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
        return np.linalg.pinv(J, rcond=1e-10), cond

    # ───────────────────────────────── ① ΔL → τ
    def solve_dL(self, dL_target_mm, tau0=None, max_iter=20, tol_mm=0.01,
                 lam=1e-2, jac_every=3, safe=True, verbose=False):
        """ΔL(mm, 6) → τ(N). 阻尼 Newton, 雅可比 = 耦合矩阵 J=∂ΔL/∂τ.

        jac_every : 每几次迭代重算一次耦合矩阵 (重算是主要开销; 3~4 足够)
        safe      : 末尾用 `apply_safety` 做行程/预紧校验并附在结果里
        返回 TendonCouplingResult(tau, dL, err_mm, sol, converged, n_iter, hist
                                 [, dL_safe, safety])
        """
        M = self.model
        dL_t = np.asarray(dL_target_mm, float).reshape(6)
        tau = np.zeros(6) if tau0 is None else np.asarray(tau0, float).reshape(6).copy()
        hist = []
        converged = False
        x = None
        J = None
        it = 0
        for it in range(max_iter):
            sol = M.shape(tau, x0=x)
            x = sol["x"]
            dL = M.delta_L_mm(tau, sol=sol)
            r = dL - dL_t
            err = float(np.linalg.norm(r))
            hist.append(err)
            if verbose:
                print(f"  it{it}: err={err:.4f} mm  τ={np.round(tau, 3)}")
            if err < tol_mm:
                converged = True
                break
            if J is None or (it % jac_every == 0):
                J = M.coupling_matrix(tau, x0=x)["J"]
            A = J.T @ J + lam * np.eye(6)          # 阻尼最小二乘
            try:
                dtau = -np.linalg.solve(A, J.T @ r)
            except np.linalg.LinAlgError:
                dtau = -np.linalg.pinv(A) @ (J.T @ r)
            if not np.isfinite(dtau).all():
                dtau = -np.linalg.pinv(J, rcond=1e-10) @ r
            tau = np.clip(tau + np.clip(dtau, -10.0, 10.0), 0.0, self.tau_max)
        sol = M.shape(tau, x0=x)
        dL = M.delta_L_mm(tau, sol=sol)
        out = TendonCouplingResult(
            tau=tau, dL=dL, err_mm=float(np.linalg.norm(dL - dL_t)),
            sol=sol, converged=converged, n_iter=it + 1, hist=np.array(hist))
        if safe:
            dLs, flags = apply_safety(dL_t, tau=tau)
            out["dL_safe"] = dLs
            out["safety"] = flags
        return out

    # ───────────────────────────────── ② 位姿 → τ
    def _newton_pose(self, p_des, R_des, tau0=None, max_iter=6,
                     tol_pos_mm=0.05, tol_rot_mrad=0.5, lam=1e-4,
                     verbose=False, refine_n=None, jac_every=2, trust_mm=2.0,
                     tau_step_norm=50.0, stop_infeasible=False, usable_mm=2.0,
                     trust_far_mm=8.0):
        """纯模型位姿牛顿 (查表命中时也走这里做精化).

        refine_n : 精化网格 n_p=n_d (None → 表分辨率)
        jac_every: 每几步重算一次雅可比 (重算是主要开销; 2~3 足够)

        ⚠ **实测: 低分辨率精化 (refine_n=60) 不可行** —— 同一 τ 在 n=60 与 n=300 下
          平衡点相差 ~6.8 mm ≫ 0.05 mm 容差, 残差被网格差吃住。精化**必须与表同分辨率**;
          真要提速请改用 POD 降阶模型做精化。
        """
        M = self._refine_model(refine_n)
        p_des = np.asarray(p_des, float).reshape(3)
        R_des = np.asarray(R_des, float).reshape(3, 3)
        tau = np.zeros(6) if tau0 is None else np.asarray(tau0, float).reshape(6).copy()
        x = None
        # ★ 2026-09-27 修: 首轮 x=None 会让 pose_jacobian_batch 去调 self.model.shape()
        #   做**冷启动标量打靶** (注释自称 ~40 s, 实测对部分位姿直接死等 >45 s)。
        #   对照实验: 随机 3 个位姿 (表点 +2.4mm) 全部卡死在同一行 ——
        #   pose_jacobian_batch:568 ← _newton_pose:721。而 #500 因热启动凑巧 OK (18 s)。
        #   表就在手里, 用 metric-最近记录的 x 热启动即可, 零额外成本。
        if self.has_table:
            try:
                x = np.asarray(
                    self.table.nearest(p_des, R_des)[0]["x"], float).reshape(6)
            except Exception:
                x = None
        hist = []
        sol = None
        Jc = None
        converged = False
        # 自适应 LM: 残差变差 → 回退到历史最优并放大阻尼 (不额外评估)
        tau_start = tau.copy()                   # ★ 发散防护: 记住"表热启动"处
        best_tau = tau.copy(); best_n = np.inf
        reach = None
        proj = None                              # ★ 用途2: 投影解 (最近可达 τ)
        it = 0
        for it in range(max_iter):
            # ⚡ 一次批量调用同时拿基态解 + 雅可比 (替代"标量 shape + 批量差分")
            Jb, sol = self.pose_jacobian_batch(tau, x0=x, model=M)
            if not bool(sol.get("converged", True)):
                self._nconv = getattr(self, "_nconv", 0) + 1
                if verbose:
                    print(f"  [!] it{it}: 静力学未收敛 (第 {self._nconv} 次)", flush=True)
                if self._nconv >= 3:
                    break
            else:
                self._nconv = 0
            x = sol["x"]
            e = self.pose_residual(tau, p_des, R_des, sol=sol, model=M)
            pos_mm = float(np.linalg.norm(e[:3]))
            rot_mrad = float(np.linalg.norm(e[3:]))
            hist.append((pos_mm, rot_mrad))
            n_tot = float(np.hypot(pos_mm, rot_mrad * 0.1))
            if it == 0:                              # ★ 可达性判据 (仅在起点算一次)
                reach = self.reachability(tau, e, J=Jb, model=M)
                if verbose:
                    print(f"  [可达] need={reach['need_N']:.1f} N  "
                          f"room={reach['room_N']:.2f} N  "
                          f"floor≈{reach['floor']:.4f} (now {reach['now']:.4f})  "
                          f"feasible={reach['feasible']}  "
                          f"饱和丝={reach['saturated']}", flush=True)
                if stop_infeasible and reach["blocked"]:
                    # ── 用途1: 提前退出 (省掉后续 ~500s 死磕) ──
                    # ── 用途2: 返回投影解 (最近可达 τ) ──
                    #   ⚠ 必须**实测校验**再用: 最小范数方向在 τ 空间"高效",
                    #     但非线性意义下未必更近。实测 #500/+5mm 盲用投影 τ
                    #     得 Δp=24.41 mm, 而热启动仅 7.07 mm ⇒ 投影只在
                    #     **确实改善**时采用, 否则保留热启动(绝不回归)。
                    cand = np.asarray(reach["tau_proj"], float)
                    e_c, _sc, ok_c = self._pose_eval(cand, p_des, R_des, x, model=M)
                    n_c = float(np.hypot(np.linalg.norm(e_c[:3]),
                                         np.linalg.norm(e_c[3:]) * 0.1))
                    if ok_c and n_c < n_tot:
                        proj = cand
                        msg = f"投影解有效 Δp={np.linalg.norm(e_c[:3]):.3f} mm"
                    else:
                        msg = (f"投影解更差 Δp={np.linalg.norm(e_c[:3]):.3f} mm"
                               f" ≥ 热启动 {np.linalg.norm(e[:3]):.3f} mm"
                               f" → 保留热启动")
                    if verbose:
                        print(f"  [可达] ★ 目标超出可达集 (可行步长比 "
                              f"{reach['step_frac']:.3f}) → 提前退出; {msg}; "
                              f"线性预估地板 {reach['floor']:.3f} mm", flush=True)
                    break
            if verbose:
                print(f"  it{it}: Δp={pos_mm:.3f} mm  Δθ={rot_mrad:.2f} mrad  "
                      f"τ={np.round(tau, 3)}", flush=True)
            if pos_mm < tol_pos_mm and rot_mrad < tol_rot_mrad:
                converged = True
                break
            if n_tot < best_n:                      # 记录历史最优
                best_tau = tau.copy(); best_n = n_tot
            Jc = Jb if (Jc is None or (it % max(int(jac_every), 1)) == 0) else Jc
            # ── 阻尼最小二乘, 在 **SVD 基底**求解 (保方向 + 按比例抑制弱方向) ──
            #   病态实测: sig=[23.42, 21.78, 1.51, 1.12, 0.146, 0.0082] (cond≈2850)。
            #   若 lam 用**绝对**小量(1e-8), 解被 σ_min 弱方向支配: |dtau|=732 N 却只
            #   预测 |pred|=0.193 位移, 实测 Δp=9.456 ⇒ 线性化在该方向完全失真。
            #   改用**相对**阻尼 λ·σ_max² 才能按奇异值比例压制弱方向。
            try:
                U, sv, Vt = np.linalg.svd(Jc, full_matrices=False)
                lam_eff = max(float(lam), 1e-9) * float(sv[0]) ** 2
                dtau = -(Vt.T @ ((sv / (sv ** 2 + lam_eff)) * (U.T @ e)))
            except np.linalg.LinAlgError:
                dtau = -np.linalg.pinv(Jc, rcond=1e-10) @ e
            if not np.isfinite(dtau).all():
                dtau = -np.linalg.pinv(Jc, rcond=1e-10) @ e
            # ── 几何信任域: 按"预测位姿位移"限步, 而非按 τ 幅值裁剪 ──
            #   实测 J 列范数最大 20.9 ⇒ 10 N 的步对应约 **200 mm** 末端位移,
            #   对 7 mm 目标误差, 线性化有效域被超出**百倍** ⇒ 满牛顿步过冲 3 倍
            #   (7.07 → 23.2 mm), 且 5 个 α 全试也不下降(锁死)。
            #   ⇒ 把步长缩到"预测位移 ≤ min(trust_mm, 当前残差)"。
            # ⚠ 顺序关键: **先**按幅值粗裁, **再**预测位移并缩放。
            #   若先缩放再裁剪, 未裁剪的大分量会被组件式 ±10 扭掉方向
            #   (实测 α=1 反而恶化到 29 mm)。
            # ★ 用**范数**约束(保方向)替代**组件式** ±10 裁剪:
            #   组件裁剪会扭掉方向 —— 实测裁剪后 |J·dtau| 从 7.07 掉到 2.84,
            #   预测与实测严重不符 ⇒ 线搜索只能瞎试。
            nd = float(np.linalg.norm(dtau))
            if nd > float(tau_step_norm) > 0.0:
                dtau = dtau * (float(tau_step_norm) / nd)
            pred = np.asarray(Jc @ dtau, float)
            n_pred = float(np.hypot(np.linalg.norm(pred[:3]),
                                    np.linalg.norm(pred[3:]) * 0.1))
            # ── ⭐ 初值策略②: 自适应信任域 ──
            #   原先 `lim = min(trust_mm(2.0), n_tot)` ⇒ 每迭代最多推进 2 mm;
            #   实测表热启动残差可达 12.9 mm(#4 甚至更大) ⇒ 8 次迭代**物理上不够**
            #   (这正是"可达目标也不收敛"的主因, 非算法发散)。
            #   改为: 残差越大, 信任域越宽 (≤ trust_far_mm=8 mm), 由**回溯线搜索**
            #   兜底 —— 方向错时线搜索会自动缩 α 拒掉, 不会过冲。
            lim = float(trust_mm)
            if n_tot > 2.0:
                lim = max(lim, min(0.5 * n_tot, float(trust_far_mm)))
            lim = min(lim, max(n_tot, 0.3))
            scale = 1.0
            if n_pred > lim > 0.0:
                scale = lim / n_pred
                dtau = dtau * scale
            if verbose:
                print(f"    trust: |dtau|={nd:.2f}N |pred|={n_pred:.3f}  lim={lim:.3f}"
                      f"  scale={scale:.3f}", flush=True)
            # ── 回溯线搜索 (关键: 消除首步过冲) ──
            #   实测无此步时 it1 过冲 (Δθ 冲 411~1418 mrad) ⇒ 大偏离目标完全不收敛。
            #   用廉价 `_pose_eval`(~9 s, 热启动) 试步, 而非每次重算雅可比。
            acc = None
            # ⭐ 第二瓶颈修复: α 下探到 1e-3/3e-4（原最小 0.03 太粗,
            #   末段只能挤出极小的下降 ⇒ 卡在 ~1.6mm 平台）
            for al in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4):
                t_c = np.clip(tau + al * dtau, 0.0, self.tau_max)
                if np.allclose(t_c, tau):
                    continue
                e2, sol2, ok2 = self._pose_eval(t_c, p_des, R_des, x, model=M)
                if not ok2:
                    if verbose:
                        print(f"    ls α={al:.2f}: 静力学未收敛 → 跳过该步", flush=True)
                    continue                      # ★ 不收敛的评估不能参与比较
                n2 = float(np.hypot(np.linalg.norm(e2[:3]),
                                    np.linalg.norm(e2[3:]) * 0.1))
                if verbose:
                    print(f"    ls α={al:.2f}  Δp={np.linalg.norm(e2[:3]):.3f} mm",
                          flush=True)
                if n2 < n_tot * 0.999:
                    acc = (t_c, n2)
                    break
                if acc is None or n2 < acc[1]:
                    acc = (t_c, n2)
            if acc is None or acc[1] >= n_tot:
                lam = min(lam * 10.0, 1e2)          # 无下降 → 加大阻尼重试
                Jc = None
                self._lsfail = getattr(self, "_lsfail", 0) + 1
                if self._lsfail >= 2:
                    break
                continue
            self._lsfail = 0
            # ★ 2026-09-28 修: 原来 lam 只会**变大** (线搜索失败时 ×10), 成功时从不放松
            #   ⇒ 一旦吃上阻尼就永久压着步长, 退化成慢速梯度类方法。
            #   实测 #41/+0.5mm: SVD 阻尼把步长压到 GN 的 ~1/5.4
            #   (|dtau|=0.27N 只预测到 0.092mm 位移), 6 轮残差 0.500→0.429 爬行。
            #   标准 LM 成功步后应放松阻尼。
            lam = max(float(lam) * 0.25, 1e-9)
            tau = acc[0]
        if proj is not None:                     # ★ 用途1/2: 投影解作为最终答案
            tau = np.clip(proj, 0.0, self.tau_max)
        sol = M.shape(tau, x0=x)
        e = self.pose_residual(tau, p_des, R_des, sol=sol, model=M)
        # 粗网格精化时: ΔL/位姿用全阶模型复算 (下发量必须与表同分辨率)
        if M is not self.model:
            sol = self.model.shape(tau, x0=x)
            e = self.pose_residual(tau, p_des, R_des, sol=sol, model=self.model)
        # ══ 🛡 发散防护 (2026-09-26) ══
        #   实测: 可达目标实验出现 #4 例 —— 优化器**发散**到 Δp=217.9 mm,
        #   却**照常返回 ΔL**(偏 205.9 mm) ⇒ 会把垃圾当指令下发。
        #   判据(零阈值/自校准): 若"最终残差比**表热启动**处还差", 即判发散 ⇒
        #   **回退到热启动的 τ**, 保证"绝不比查表更差"。
        def _n_of(ee):
            return float(np.hypot(float(np.linalg.norm(ee[:3])),
                                  float(np.linalg.norm(ee[3:])) * 0.1))
        # ★ 2026-10-05 补: `best_tau`/`best_n` 原本**只写不读** —— 上面注释写着
        #   "残差变差 → 回退到历史最优", 但循环里只记录、从未真正回退。
        #   后果: 迭代末段几步变差时, 返回的是**当前** τ 而不是历史最优点
        #   (仅当整体比热启动更差 5%+ 时才由下面的发散防护兜到 tau_start)。
        #   这里补上真正的回退: 用**实测**确认历史最优确实更好才采用, 不会回归。
        if proj is None and np.isfinite(best_n):
            n_cur = _n_of(e)
            if best_n < n_cur * 0.999:
                cand = np.clip(np.asarray(best_tau, float).reshape(6),
                               0.0, self.tau_max)
                if not np.allclose(cand, tau):
                    sol_c = M.shape(cand, x0=x)
                    e_c = self.pose_residual(cand, p_des, R_des, sol=sol_c, model=M)
                    if _n_of(e_c) < n_cur:
                        tau, sol, e = cand, sol_c, e_c
                        if verbose:
                            print(f"  [回退] 采用历史最优 τ (残差 "
                                  f"{n_cur:.3f} → {_n_of(e):.3f} mm)", flush=True)
        res_now = _n_of(e)
        res_start = (float(np.hypot(hist[0][0], hist[0][1] * 0.1)) if hist else res_now)
        diverged = bool(res_now > res_start * 1.05 + 0.2)
        if diverged and len(hist) > 0:
            tau = np.clip(np.asarray(tau_start, float).reshape(6), 0.0, self.tau_max)
            sol = M.shape(tau, x0=None)
            e = self.pose_residual(tau, p_des, R_des, sol=sol, model=M)
            if M is not self.model:
                sol = self.model.shape(tau, x0=None)
                e = self.pose_residual(tau, p_des, R_des, sol=sol, model=self.model)
            res_now = _n_of(e)
            if verbose:
                print(f"  [🛡 发散防护] 优化器比热启动更差 → 已回退到表热启动 "
                      f"(残差 {res_now:.3f} mm)", flush=True)
        if reach is not None and isinstance(sol, dict):
            sol["reachability"] = reach
            if proj is not None:
                sol["projection"] = {"tau_proj": np.asarray(proj, float),
                                     "need_N": reach["need_N"],
                                     "step_frac": reach["step_frac"],
                                     "floor_est_mm": reach["floor"],
                                     "saturated": reach["saturated"]}
        _out = TendonCouplingResult(
            tau=tau, sol=sol,
            pos_res_mm=float(np.linalg.norm(e[:3])),
            rot_res_mrad=float(np.linalg.norm(e[3:])),
            dL_mm=self.model.delta_L_mm(tau, sol=sol),
            p_tip=np.asarray(sol["tip"], float),
            R_tip=np.asarray(sol["R_tip"], float),
            converged=converged, n_iter=it + 1, hist=hist,
            refine_n=(M.params.n_p if M is not self.model else None))
        # ★ 可达性标记 —— 写成 **dict 键**(TendonCouplingResult 就是 dict 子类),
        #   这样即便上层复制/重建结果也不会丢; 属性访问 o.feasible 仍然可用。
        if reach is not None:
            _res = float(np.linalg.norm(e[:3]))
            _floor = float(reach["floor"])
            # ★ 2026-09-28 修: 原判据只看 |Δp| ≤ usable_mm (默认 2mm), 于是"目标近不可达、
            #   残差被可达地板顶住"的情况也会标 usable=True —— 对下位机是虚假的可用解。
            #   实测 #41/+0.5mm: 残差 0.376 mm, 判据 floor≈0.224 mm (同量级),
            #   need=34.5N(=GN步, 已实测对拍) ≫ room=5.85N ⇒ 该方向推不动, 却标可用。
            #   现在: 未收敛 且 残差已贴住地板(地板 > 半残差) 且 地板明显大于容差
            #   ⇒ 判"受可达性限制", 不再算可用。
            _reach_limited = bool((not converged) and _floor > 0.5 * _res
                                  and _floor > float(tol_pos_mm))
            _out["reach_limited"] = _reach_limited
            _out["res_init_mm"] = float(res_start)
            _out["diverged"] = bool(diverged)
            _out["usable"] = bool((not diverged) and (not _reach_limited)
                                  and _res <= float(usable_mm))
            _out["feasible"] = bool(reach["feasible"])
            _out["blocked"] = bool(reach.get("blocked", False))
            _out["floor_mm"] = float(reach["floor"])
            _out["need_N"] = float(reach["need_N"])
            _out["projected"] = bool(proj is not None)
            if proj is not None:
                _out["tau_proj"] = np.asarray(proj, float)
        return _out

    @staticmethod
    def verdict(res, fail_pos_mm=10.0, fail_rot_mrad=50.0):
        """★ 发散防护: 判定求解结果是否**可信、可下发**.

        背景(2026-09-26 可达目标实验 #4): 求解**发散**到 Δp=217.9 mm / Δθ=1185 mrad,
        且只迭代 2 次就因线搜索保护提前退出 —— 但结果仍被当作有效解返回,
        ΔL 误差 205.9 mm. **返回这种指令是危险的.**

        判据 (任一不满足 ⇒ solved=False):
          ① `converged` 为真 (残差达容差)
          ② 位姿残差不超过 fail_pos_mm / fail_rot_mrad (灾难性发散阈值)
          ③ ΔL 全为有限值
        返回 (solved: bool, reason: str)
        """
        try:
            conv = bool(res.get("converged", False))
            pm = float(res.get("pos_res_mm", np.inf))
            rm = float(res.get("rot_res_mrad", np.inf))
            dL = np.asarray(res.get("dL_mm", []), float)
        except Exception:
            return False, "结果字段缺失"
        if not np.isfinite(dL).all():
            return False, "ΔL 含非有限值"
        if pm > fail_pos_mm or rm > fail_rot_mrad:
            return False, f"位姿残差过大(Δp={pm:.2f}mm > {fail_pos_mm}mm) —— 求解发散"
        if not conv:
            return False, f"未收敛(Δp={pm:.4f}mm)"
        return True, "ok"

    def solve_pose(self, p_des, R_des, tau0=None, pos_tol_mm=50.0,
                   allow_infill=True, interp=True, refine=True, max_iter=6,
                   tol_pos_mm=0.05, tol_rot_mrad=0.5, lam=1e-4, verbose=None,
                   refine_n=None, pod=None, verify=None):
        """★ (p*, R*) → τ.

        有表  : 查表定位 (规避多平衡态) → 模型精化;
        无表  : 直接纯模型牛顿 (冷启动, 仅在浅弯区可靠)。

        tau0 显式给出时**跳过查表**, 直接以它为初值做模型牛顿。
        返回 TendonCouplingResult: tau, dL_mm, pos_res_mm, rot_res_mrad,
                hit, branch, interpolated, source, sol
        """
        v = self.verbose if verbose is None else verbose
        p_des = np.asarray(p_des, float).reshape(3)
        R_des = np.asarray(R_des, float).reshape(3, 3)

        # ── 显式初值: 不走查表 ──
        if tau0 is not None:
            out = self._solve_refine(p_des, R_des, tau0, max_iter, tol_pos_mm,
                                     tol_rot_mrad, lam, v, refine_n, pod, verify)
            return TendonCouplingResult(**dict(out), hit=False, branch=False,
                                        interpolated=False, source="tau0")

        # ── 查表 (有表时) ──
        q = None
        if self.has_table:
            q = self.table.query((p_des, R_des), pos_tol_mm=pos_tol_mm,
                                 allow_infill=allow_infill, interp=interp,
                                 verbose=v)
        if q is None or not q.get("reachable", False) or not q.get("hit", False):
            if v and self.has_table:
                print("  [查表] 不可达/未覆盖 → 退回纯模型逆解 (冷启动)")
            t0c = None
            if self.has_table:
                try:
                    t0c = np.asarray(self.table.nearest(p_des, R_des)[0]["tau"], float)
                except Exception:
                    t0c = None
            out = self._solve_refine(p_des, R_des, t0c, max_iter, tol_pos_mm,
                                     tol_rot_mrad, lam, v, refine_n, pod, verify)
            return TendonCouplingResult(**dict(out), hit=False, branch=False,
                                        interpolated=False,
                                        source="model-only" if self.has_table else "model")

        tau_t = np.asarray(q["tau"], float)
        if v:
            print(f"  [查表] hit={q['hit']} interp={q['interpolated']} "
                  f"branch={q['branch']} Δp={q.get('pos_res_mm', float('nan')):.2f} mm")
        if not refine:
            s = self.model.shape(tau_t)
            return TendonCouplingResult(tau=tau_t, dL_mm=np.asarray(q["dL_mm"], float),
                                        pos_res_mm=q.get("pos_res_mm", np.nan),
                                        rot_res_mrad=q.get("rot_res_mrad", np.nan),
                                        p_tip=np.asarray(q["p_tip"], float),
                                        R_tip=np.asarray(q["R_tip"], float),
                                        sol=s, converged=False, n_iter=0, hist=[],
                                        hit=True, branch=bool(q["branch"]),
                                        interpolated=bool(q["interpolated"]),
                                        source="table")
        out = self._solve_refine(p_des, R_des, tau_t, max_iter, tol_pos_mm,
                                 tol_rot_mrad, lam, v, refine_n, pod, verify)
        res = TendonCouplingResult(**dict(out), hit=True,
                                   branch=bool(q["branch"]),
                                   interpolated=bool(q["interpolated"]),
                                   source="table+model", tau_table=tau_t)
        solved, why = self.verdict(res)
        res["solved"] = bool(solved)
        res["fail_reason"] = why
        res["dL_valid"] = bool(solved)          # ΔL 是否可下发
        if v:
            print(f"  [精化] Δp={res['pos_res_mm']:.4f} mm  "
                  f"Δθ={res['rot_res_mrad']:.3f} mrad  n_it={res['n_iter']}")
        return res

    # ══════════════════════════════════ 表相关
    def _check_version(self):
        """版本键一致性: 表与模型必须同参 (否则 τ↔pose 映射不匹配).

        ★ 比的是**建表时写进 npz 的键** (`table.stored_version`) vs 当前模型现算的键。
        旧表没有持久化键 → 无法判定, 返回 True + 警告项, 不要读成"一致"。
        """
        ref = table_version_key(self.model.params, F_tip=self.model.F_tip)
        stored = getattr(self.table, "stored_version", None)
        if stored is None:
            return True, {"__no_stored_key__":
                          ("表内无版本键 (旧表)", "重存后可校验")}
        diff = {k: (stored.get(k), ref.get(k)) for k in ref
                if stored.get(k) != ref.get(k)}
        return (len(diff) == 0), diff

    def verify(self, n=5, idx=None, seed=0):
        """从表里抽 n 条, 用正解模型重解其 τ, 对比 tip/R/ΔL 是否一致."""
        if not self.has_table:
            return []
        recs = self.table.recs
        if idx is None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(recs), size=min(int(n), len(recs)), replace=False)
        rows = []
        for i in idx:
            r = recs[int(i)]
            s = self.model.shape(r["tau"])
            dp = float(np.linalg.norm(np.asarray(s["tip"]) - r["p_tip"]) * 1000)
            dth = float(np.linalg.norm(
                so3_log(r["R_tip"].T @ np.asarray(s["R_tip"]))) * 1000)
            dL = self.model.delta_L_mm(r["tau"], sol=s)
            ddL = float(np.linalg.norm(dL - np.asarray(r["dL_mm"])))
            rows.append(dict(i=int(i), tau=np.asarray(r["tau"]).copy(),
                             dp_mm=dp, dth_mrad=dth, ddL_mm=ddL))
        return rows

    # ══════════════════════════════════ 诊断
    def describe(self):
        m = self.model
        print("=" * 66)
        print("肌腱逆解模型 (TIM) — 位姿/ΔL → τ")
        print("=" * 66)
        if self.has_table:
            print(f"  查表: {len(self.table.recs)} 条, 度量 σ_p={self.table.metric.sp} mm, "
                  f"覆盖半径默认 50 mm")
            print(f"  版本键一致: {self.version_ok}"
                  + ("" if self.version_ok else f"  ⚠ 差异: {self.version_diff}"))
            print("  模式: 查表定初值 + 模型精化")
        else:
            print("  查表: 未加载 → 纯模型逆解 (冷启动)")
        print(f"  正解内核: EI={m.params.EI:.4g}, k_rod={m.params.k_rod:.4g}, "
              f"n_p={m.params.n_p}, n_d={m.params.n_d}, lattice={m.lattice}")
        print(f"  晶胞    : α={np.degrees(m.cell.alpha):.2f}°, EI0={m.cell.EI0:.4f} N·m²")
        print(f"  张力上限: [0, {self.tau_max:.0f}] N")
        print("=" * 66)

    def self_test(self, n=3, seed=1, verbose=True):
        """自检: ③ ΔL→τ 往返 ④ 位姿雅可比(Richardson) ⑤ 行程/松弛判定
               ⑥ 查表→逆解 往返 (有表时)."""
        say = print if verbose else (lambda *a, **k: None)
        M = self.model
        ok_all = True

        # ③ ΔL → τ 往返
        say("── ③ ΔL → τ 往返 ──")
        tau_t = np.array([6.0, 0.0, 6.0, 0.0, 6.0, 3.0])
        dL_t = M.delta_L_mm(tau_t)
        say(f"  目标 τ = {np.round(tau_t, 2)}   正向 ΔL = {np.round(dL_t, 3)} mm")
        out = self.solve_dL(dL_t, tau0=np.zeros(6), max_iter=25)
        say(f"  反解 τ = {np.round(out['tau'], 2)}  err={out['err_mm']:.4f} mm "
            f"conv={out['converged']} n_it={out['n_iter']}")
        ok1 = out["err_mm"] < 0.05 and out["converged"]
        ok_all &= ok1
        say(f"  → {'PASS ✅' if ok1 else 'FAIL ❌'}")

        # ④ 位姿雅可比: Richardson (h vs h/2). 中心差分是 2 阶, 两者应几乎相同;
        #    不能用"中心差分 vs 前向差分"比 —— 那差的是 O(h) 截断, 不是误差。
        say("── ④ 位姿雅可比 J=∂[p;rotvec]/∂τ (Richardson 校验) ──")
        tau0 = np.array([5.0, 0.0, 5.0, 0.0, 5.0, 2.0])
        J1 = self.pose_jacobian(tau0, h=1e-3)
        J2 = self.pose_jacobian(tau0, h=5e-4)
        rel = float(np.linalg.norm(J1 - J2) / np.linalg.norm(J1))
        # 线性预测一致性 (纯正解, 不经差分)
        p0, R0 = M.pose(tau0)
        v = np.array([1., 0., -1., 0., 1., 0.]); v /= np.linalg.norm(v)
        hh = 1e-3
        p1, R1 = M.pose(tau0 + hh * v)
        act = np.concatenate([(p1 - p0) * 1000.0, so3_log(R0.T @ R1) * 1000.0])
        pred = J1 @ (hh * v)
        lpe = float(np.linalg.norm(pred - act) / np.linalg.norm(act))
        ok2 = rel < 1e-8 and lpe < 1e-4
        ok_all &= ok2
        say(f"  ‖J‖={np.linalg.norm(J1):.3f}  Richardson(h vs h/2) 相对偏差={rel:.2e}")
        say(f"  线性预测相对误差={lpe:.2e}")
        say(f"  → {'PASS ✅' if ok2 else 'FAIL ❌'}")

        # ⑤ 行程/松弛: 全丝受拉的可达目标必须 ok; 不可达(要"推"丝)必须报 slack
        say("── ⑤ 行程 / 预紧松弛判定 ──")
        tau_pos = np.array([6.0, 3.0, 6.0, 3.0, 6.0, 3.0])   # 六丝均受拉 (> 预紧下限)
        dL_pos = M.delta_L_mm(tau_pos)
        good = self.solve_dL(dL_pos, tau0=np.zeros(6), max_iter=25, safe=True)
        bad = self.solve_dL([-5, 0, -5, 0, -5, 2], tau0=np.zeros(6),
                            max_iter=8, safe=True)
        ok_a = bool(good["safety"]["ok"])
        ok_b = (not bad["safety"]["ok"]) and len(bad["safety"]["slack"]) > 0
        ok_all &= (ok_a and ok_b)
        say(f"  可达(τ={np.round(tau_pos,0)}) 反解 τ={np.round(good['tau'],2)} "
            f"err={good['err_mm']:.4f} mm  ok={good['safety']['ok']} "
            f"clipped={good['safety']['clipped']} slack={good['safety']['slack']}  "
            f"{'✓' if ok_a else '✗'}")
        say(f"  不可达([-5,0,-5,0,-5,2]) → err={bad['err_mm']:.2f} mm "
            f"ok={bad['safety']['ok']} slack={bad['safety']['slack']} "
            f"(丝只能拉 ⇒ 必须报松弛)  {'✓' if ok_b else '✗'}")
        say(f"  → {'PASS ✅' if (ok_a and ok_b) else 'FAIL ❌'}")

        # ⑥ 查表 → 逆解 往返 (有表时)
        if self.has_table:
            recs = self.table.recs
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(recs), size=min(int(n), len(recs)), replace=False)
            say("── ⑥ 查表 → 逆解 往返 (以表点为靶) ──")
            ok6 = True
            for i in idx:
                r = recs[int(i)]
                o = self.solve_pose(r["p_tip"], r["R_tip"], refine=True)
                dtau = float(np.linalg.norm(o["tau"] - np.asarray(r["tau"])))
                ok = (o["pos_res_mm"] < 0.1) and (o["rot_res_mrad"] < 1.0)
                ok6 &= ok
                say(f"  #{int(i)}: Δp={o['pos_res_mm']:.4f} mm  "
                    f"Δθ={o['rot_res_mrad']:.3f} mrad  |Δτ|={dtau:.3f} N  "
                    f"src={o['source']} {'✓' if ok else '✗'}")
            ok_all &= ok6
        return bool(ok_all)


# ══════════════════════════════════════════════════════════════════════
# POD 降阶模型 (Proper Orthogonal Decomposition)  — 原 pod_solver.py
# ══════════════════════════════════════════════════════════════════════
# 用途: 全阶 Cosserat 单次 2.9s → 进不了控制环; 降成 r 维模态系数求解。
#
#   1. 离线: τ 采样 → 全阶 VC 解 → 快照 {κ(s)} → POD → 均值 κ̄ + 基 Φ=[φ₁..φ_r]
#   2. 在线: 解 κ(s) = κ̄(s) + Σ a_j φ_j(s) 的系数 a
#      约束 = 6 根丝的几何长匹配给定位移 ΔL:   L_w(a) = L0_w + ΔL_w
#
# ⚠ 线性子空间是平衡流形的近似 —— 这正是 r=8 时 ~0.5mm 残差的来源;
#   也是 r<6 时误差巨大的原因 (子空间张不出那个方向)。
# ⚠ 范围: POD 只在**建基用的 τ 范围**内有效, 外推不可靠。
# ⚠ 基只含 κ(s) 不含线应变 v(s): reduced_forward 用 v=e₃ 重建会偏长 40~58mm。
#
# 归属: 降阶模型主要服务于**在线逆解/控制**, 故并入逆解模型;
#       纯几何积分 integrate_shape* 已下沉到 sdm_vc.py。
# ══════════════════════════════════════════════════════════════════════
# ==================== 离线: 建 POD 基 ====================

def build_basis(records, r=8, params=None, want_curve_fn=None):
    """由建表记录(或 VC 解)提取 κ(s) 快照, 做 POD.

    records: ik_table 记录列表 (需含 tau / x)
    返回 dict(mean (nS,3), modes (r,nS,3), S (奇异值), r, S_grid, params)
    """
    from tendon_coupling import forward_batch
    params = params or default_params()
    taus = np.array([rec["tau"] for rec in records])
    xs = np.array([rec["x"] for rec in records])
    _, _, cv = forward_batch(xs, taus, params, want_curve=True)
    S = cv[1]; Uc = cv[2]                          # (N,nS,3)
    N, nS, _ = Uc.shape
    Snap = Uc.reshape(N, -1)
    mean = Snap.mean(axis=0)
    X = Snap - mean
    U, Sv, Vt = np.linalg.svd(X, full_matrices=False)
    r = int(min(r, len(Sv)))
    modes = Vt[:r].reshape(r, nS, 3)
    return {"mean": mean.reshape(nS, 3), "modes": modes, "S": Sv,
            "r": r, "S_grid": S, "params": params,
            "energy": float((Sv[:r] ** 2).sum() / (Sv ** 2).sum())}


def kappa_field(a, basis):
    """模态系数 a(r) → κ(s) 场 (nS,3)."""
    return basis["mean"] + np.tensordot(np.asarray(a, float), basis["modes"], axes=(0, 0))

def kappa_field_batch(A, basis):
    """A (N,r) → κ(s) 场 (N,nS,3)."""
    return basis["mean"][None] + np.tensordot(A, basis["modes"], axes=(1, 0))


def reduced_forward_batch(A, basis, nsub=60):
    """A (N,r) → tips (N,3), R_tips (N,3,3), dL (N,6)  (全批量).

    ⚠ **降阶路径的已知缺陷**: POD 基只含 κ(s), 不含线应变 v(s)。而骨架是
    `p'=R·v`, 臂体轴向刚度 `k_rod` 有限时 v_z≠1 (实测 0.85~0.90), 故本品用
    `v=e₃` 重建的骨架会比真解长 40~58mm, 由此算出的 dL 同样不可信。
    要修需把 v(s) 一并纳入 POD 基 (堆叠 [κ; v] 或另建一套基)。
    """
    kf = kappa_field_batch(A, basis)
    ps, Rs = integrate_shape_batch(kf, basis["S_grid"], nsub, v_batch=None)
    Lgeo = _wire_geom_len_batch(ps, Rs)
    return ps[:, -1], Rs[:, -1], (Lgeo - _L0_wire()) * 1000.0


def reduced_forward(a, basis, nsub=60):
    """a(r) → (p_tip, R_tip, dL_mm[6], 骨架)."""
    p, R, dL = reduced_forward_batch(np.asarray(a, float)[None], basis, nsub)
    kf = kappa_field(a, basis)
    _, Rs = integrate_shape_batch(kf[None], basis["S_grid"], nsub)
    return p[0], R[0], dL[0], (None, Rs[0])


# ==================== 在线: ΔL → a (牛顿) ====================

def solve_dL_to_shape(dL_mm, basis, a0=None, max_iter=30, tol_mm=1e-3,
                      lam0=1e-3, verbose=False):
    """给定位移 ΔL[6] → 模态系数 a → 形状 (静态 POD-Galerkin 求解).

    r=6 时 6 方程 6 未知(方阵); r>6 时超定(最小二乘)。用 Levenberg-Marquardt
    (自适应阻尼 + 接受判据), 不做硬钳制 —— 那会把大步长截断导致不收敛。
    返回 dict(a, p_tip, R_tip, dL, ps, Rs, converged, n_iter, res_mm)
    """
    dL = np.asarray(dL_mm, float)
    r = basis["r"]
    a = np.zeros(r) if a0 is None else np.array(a0, float)

    def resid(aa):
        p_tip, R_tip, dL_calc, shape = reduced_forward(aa, basis)
        return dL_calc - dL, shape, p_tip, R_tip

    res, shape, p_tip, R_tip = resid(a)
    n_old = np.linalg.norm(res)
    lam = lam0
    converged = False
    it = 0
    for it in range(max_iter):
        J = np.zeros((6, r)); h = 1e-6
        for c in range(r):
            ap = a.copy(); am = a.copy(); ap[c] += h; am[c] -= h
            J[:, c] = (resid(ap)[0] - resid(am)[0]) / (2 * h)
        A = J.T @ J + lam * np.eye(r)
        b = J.T @ res
        try:
            da = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            da = -np.linalg.pinv(A) @ b
        if not np.isfinite(da).all():
            break
        a_new = a + da
        res_new, shape_new, p_new, R_new = resid(a_new)
        n_new = np.linalg.norm(res_new)
        if n_new < n_old:
            a, res, shape, p_tip, R_tip = a_new, res_new, shape_new, p_new, R_new
            lam = max(lam / 5.0, 1e-12)
            n_old = n_new
            if n_new < tol_mm:
                converged = True
                break
        else:
            lam = min(lam * 10.0, 1e9)
            if lam >= 1e8:
                break
    return {"a": a, "p_tip": p_tip, "R_tip": R_tip, "dL": dL,
            "ps": shape[0], "Rs": shape[1], "converged": converged,
            "n_iter": it + 1, "res_mm": float(n_old)}


# ==================== τ 预测 (POD 系数 → 张力) ====================

def fit_tau_predictor(records, basis, order=2):
    """由建表数据拟合 a → τ 的预测器 (kT 多项式最小二乘).

    用于把降阶解映射回张力估计, 兼作融合求解的第二个观测。
    """
    from itertools import combinations_with_replacement as cwr
    from tendon_coupling import forward_batch
    params = basis["params"]
    taus = np.array([r["tau"] for r in records])
    xs = np.array([r["x"] for r in records])
    _, _, cv = forward_batch(xs, taus, params, want_curve=True)
    A = _modal_coeff(cv[2], basis)
    F = _feat(A, order, basis["r"])
    W, *_ = np.linalg.lstsq(F, taus, rcond=None)
    pred = F @ W
    return {"W": W, "order": order, "r": basis["r"],
            "fit_err": float(np.median(np.linalg.norm(pred - taus, axis=1)))}


def _modal_coeff(Uc, basis):
    N = Uc.shape[0]
    return (Uc.reshape(N, -1) - basis["mean"].reshape(-1)) @ \
        basis["modes"].reshape(basis["r"], -1).T


def _feat(A, order, r):
    A = np.atleast_2d(A)
    cols = [np.ones((len(A), 1)), A]
    if order >= 2:
        from itertools import combinations_with_replacement as cwr
        cols += [(A[:, i] * A[:, j])[:, None] for i, j in cwr(range(r), 2)]
    return np.hstack(cols)


def predict_tau(a, pred):
    return _feat(a, pred["order"], pred["r"])[0] @ pred["W"]


# ==================== 融合求解 (§ 张力作第二观测) ====================

def solve_fused(dL_mm, tau_meas, basis, tau_pred, a0=None,
                sigma_dL=0.2, sigma_tau=0.5, max_iter=40, tol=1e-3,
                tau_weight=1.0):
    """ΔL + τ 融合求解: 12 维观测 → 模态系数 a.

    残差 (逐项按噪声归一):
        r(a) = [ (ΔL(a) − ΔL_meas) / σ_dL ;  √w·(τ̂(a) − τ_meas) / σ_τ ]

    σ_dL / σ_tau: 两路观测的噪声(std)。σ_tau → ∞ 退化为只用 ΔL。
    tau_weight: τ 路的额外权重 (0 = 关闭, 1 = 按 1/σ_τ 正常加权)。
    返回同 solve_dL_to_shape, 另含 res_dL_mm / res_tau_N 分项。
    """
    dL = np.asarray(dL_mm, float)
    tau_m = np.asarray(tau_meas, float)
    r = basis["r"]
    a = np.zeros(r) if a0 is None else np.array(a0, float)
    sw = np.sqrt(max(tau_weight, 0.0))
    nsub = 60
    h = 1e-6
    nt = 6 if tau_weight > 0 else 0
    no = 6 + nt

    def eval_batch(A):
        """A (M,r) → (E (M,no), dL (M,6), tau_hat (M,6), tip (M,3), R_tip (M,3,3))."""
        A = np.atleast_2d(A)
        tips, Rtips, dLs = reduced_forward_batch(A, basis, nsub)
        if nt:
            th = _feat(A, tau_pred["order"], r) @ tau_pred["W"]
        else:
            th = np.zeros((len(A), 6))
        E = np.concatenate([(dLs - dL) / sigma_dL, sw * (th - tau_m) / sigma_tau],
                           axis=1)
        return E, dLs, th, tips, Rtips

    E0, dL_calc, tau_hat, p_tip, R_tip = eval_batch(a[None])
    res = E0[0]; dL_calc = dL_calc[0]; tau_hat = tau_hat[0]
    p_tip = p_tip[0]; R_tip = R_tip[0]
    n_old = np.linalg.norm(res)
    lam = 1e-3
    converged = False
    it = 0
    for it in range(max_iter):
        # 批量雅可比: 2r+1 个扰动点一次解出
        P = np.vstack([a, a + h * np.eye(r), a - h * np.eye(r)])       # (2r+1,r)
        E, _, _, _, _ = eval_batch(P)
        J = np.zeros((no, r))
        for c in range(r):
            J[:, c] = (E[1 + c] - E[1 + r + c]) / (2 * h)
        A_ = J.T @ J + lam * np.eye(r)
        b_ = J.T @ res
        try:
            da = -np.linalg.solve(A_, b_)
        except np.linalg.LinAlgError:
            da = -np.linalg.pinv(A_) @ b_
        if not np.isfinite(da).all():
            break
        E1, dLs1, th1, tip1, R1 = eval_batch((a + da)[None])
        n_new = np.linalg.norm(E1[0])
        if n_new < n_old:
            a = a + da
            res, dL_calc, tau_hat = E1[0], dLs1[0], th1[0]
            p_tip, R_tip = tip1[0], R1[0]
            n_old = n_new
            lam = max(lam / 5.0, 1e-12)
            if n_new < tol:
                converged = True
                break
        else:
            lam = min(lam * 10.0, 1e9)
            if lam >= 1e8:
                break
    return {"a": a, "p_tip": p_tip, "R_tip": R_tip, "converged": converged,
            "n_iter": it + 1, "res": float(n_old),
            "res_dL_mm": float(np.linalg.norm(dL_calc - dL)),
            "res_tau_N": float(np.linalg.norm(tau_hat - tau_m)),
            "dL_calc": dL_calc, "tau_hat": tau_hat, "ps": None, "Rs": None}


# ==================== 自检 ====================

def fit_tau_to_shape(records, basis, order=3):
    """拟合 **τ → 模态系数** 映射 (确定性, 高精度).

    ⚠️ 方向很关键 (实测, 见笔记第六.6 节):
      * τ → a 是**确定性**映射 —— 三次多项式即可到 0.60 mm (留出, 样本内 0.61,
        无过拟合), 已逼近 POD 基底噪 (r=8 时 0.46 mm)。
      * a → τ **不是函数** —— 同样形状可由不同张力实现, 样本内/留出误差
        都在 ~3.1 N (信息本身不存在), 换任何回归器都打不破。
      ⇒ 融合求解应**以 τ 为未知量** (见 solve_fused_tau), 而非 a。
    """
    taus = np.array([r["tau"] for r in records])
    from tendon_coupling import forward_batch
    xs = np.array([r["x"] for r in records])
    _, _, cv = forward_batch(xs, taus, basis["params"], want_curve=True)
    A = _modal_coeff(cv[2], basis)
    F = _feat(taus, order, 6)
    W, *_ = np.linalg.lstsq(F, A, rcond=None)
    return {"W": W, "order": order, "r": basis["r"], "basis": basis}


def shape_of_tau(tau_batch, tsp, nsub=60):
    """τ (N,6) → tips (N,3), R_tips (N,3,3), dL (N,6), a (N,r). 确定性前向."""
    A = _feat(tau_batch, tsp["order"], 6) @ tsp["W"]
    kf = kappa_field_batch(A, tsp["basis"])
    ps, Rs = integrate_shape_batch(kf, tsp["basis"]["S_grid"], nsub)
    dL = (_wire_geom_len_batch(ps, Rs) - _L0_wire()) * 1000.0
    return ps[:, -1], Rs[:, -1], dL, A


def solve_fused_tau(dL_mm, tau_meas, tsp, sigma_dL=0.2, sigma_tau=0.5,
                    max_iter=15, tol=1e-3, tau0=None):
    """★ 融合求解 (推荐): **以 τ 为未知量**, 用三次 â(τ) 做前向.

        r(τ) = [ (ΔL(τ) − ΔL_meas)/σ_dL ;  (τ − τ_meas)/σ_τ ]

    比 a-参数化的 solve_fused 精度高 ~3× (σ_τ=0.5 N 时 1.16 vs 3.96 mm),
    因为它避开了 "a → τ 不是函数" 这个不可逾越的地板。
    """
    dL_m = np.asarray(dL_mm, float)
    tau_m = np.asarray(tau_meas, float)
    t_ = np.zeros(6) if tau0 is None else np.array(tau0, float)
    h = 1e-3
    nsub = 60

    def err(tb):
        _, _, dL_c, _ = shape_of_tau(np.atleast_2d(tb), tsp, nsub)
        return dL_c                                   # (M,6)

    converged = False
    it = 0
    dL_c = err(t_)[0]                                 # (6,)
    for it in range(max_iter):
        r = np.concatenate([(dL_c - dL_m) / sigma_dL,
                            (t_ - tau_m) / sigma_tau])
        dLp = err(t_ + h * np.eye(6))                 # (6,6)
        J = np.zeros((12, 6))
        J[:6, :] = ((dLp - dL_c[None]) / h).T / sigma_dL
        J[6:, :] = np.eye(6) / sigma_tau
        lam = 1e-4
        try:
            dt = -np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ r)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(dt).all():
            break
        t_ = np.clip(t_ + dt, 0.0, None)              # τ ≥ 0
        dL_c = err(t_)[0]
        if np.linalg.norm(r) < tol:
            converged = True
            break
    p_tip, R_tip, dL_c2, A = shape_of_tau(t_[None], tsp, nsub)
    return {"tau": t_, "a": A[0], "p_tip": p_tip[0], "R_tip": R_tip[0],
            "dL_calc": dL_c2[0], "converged": converged, "n_iter": it + 1,
            "res_dL_mm": float(np.linalg.norm(dL_c2[0] - dL_m)),
            "res_tau_N": float(np.linalg.norm(t_ - tau_m))}


# ══════════════════════════════════════════════════════════════════════
# τ → 模态系数 a 的局部线性插值 (表内 kNN, 比全局多项式准得多)
# ══════════════════════════════════════════════════════════════════════
def _a_interp_w(q_tau, TAU, A, tree=None, k=8, ridge=1e-6):
    """k 近邻**局部线性**插值 → (A_hat (M,r), dAdq (M,r,6)).

    以**查询点**为原点做线性拟合 ⇒ Wc[0]=插值值, Wc[1:]=梯度 ∂a/∂τ (解析, 无差分噪声)。
    这一点很关键: POD 自身有 ~1 mm 偏置, 若用 τ-空间有限差分求雅可比会被噪声淹没。
    """
    q_tau = np.atleast_2d(np.asarray(q_tau, float))
    K = int(min(max(k, 4), len(TAU)))
    if tree is None:
        d = np.linalg.norm(TAU[None, :, :] - q_tau[:, None, :], axis=2)
        idx = np.argsort(d, axis=1)[:, :K]
    else:
        _, idx = tree.query(q_tau, k=K)
    idx = np.atleast_2d(idx)
    M = len(q_tau); r = A.shape[1]
    Ahat = np.empty((M, r)); dAdq = np.empty((M, r, 6))
    for i in range(M):
        j = idx[i]
        dT = TAU[j] - q_tau[i]
        F = np.hstack([np.ones((K, 1)), dT])
        G = F.T @ F + ridge * np.eye(F.shape[1])
        Wc = np.linalg.solve(G, F.T @ A[j])
        Ahat[i] = Wc[0]
        dAdq[i] = Wc[1:].T
    return Ahat, dAdq


def _a_interp(q_tau, TAU, A, tree=None, k=8, ridge=1e-6):
    return _a_interp_w(q_tau, TAU, A, tree, k, ridge)[0]


# ══════════════════════════════════════════════════════════════════════
# POD 降阶模型的统一入口 (把上面这些函数收成一个"模型")
# ══════════════════════════════════════════════════════════════════════
class PODReducedOrderModel:
    """POD-Galerkin 降阶模型: 离线建基 → 在线反解 (服务逆解/实时控制).

    用法
    ----
        rom = PODReducedOrderModel(train_records, r=8, params=P)   # 离线建基
        rom.fit_tau(train_records, order=2)                        # a → τ 预测器
        rom.fit_tau_shape(train_records, order=3)                  # τ → a 前向 (推荐)
        out = rom.solve_dL(dL_mm)                                  # ΔL → 形状
        out = rom.solve_dL_fused_tau(dL_mm, tau_meas)              # ΔL+τ 融合

    ⚠ r=8 是工程取值 (末端 0.46mm, 已达积分器底噪); **不要按能量占比选阶**
      —— r=5 时能量已 98.81%, 末端误差仍有 14.5mm。
    ⚠ 只在**建基用的 τ 范围**内有效, 外推不可靠。
    """

    def __init__(self, records=None, r=8, params=None, basis=None):
        self.params = params if params is not None else default_params()
        self.r = int(r)
        self.basis = basis
        if self.basis is None and records is not None:
            self.basis = build_basis(records, r=self.r, params=self.params)
        self.tau_pred = None      # a → τ (二次)
        self.tsp = None           # τ → a (三次, 推荐)
        if self.basis is not None:
            self.n_rec = len(records) if records is not None else None

    # ── 离线: 拟合映射 ──
    def fit_tau(self, records, order=2):
        """拟合 a → τ 预测器 (作融合求解的第二观测)."""
        self.tau_pred = fit_tau_predictor(records, self.basis, order=order)
        return self.tau_pred

    def fit_tau_shape(self, records, order=3):
        """拟合 τ → a (确定性, 高精度) —— 融合求解推荐用这个方向."""
        self.tsp = fit_tau_to_shape(records, self.basis, order=order)
        return self.tsp

    # ── 在线: 反解 ──
    def solve_dL(self, dL_mm, **kw):
        """ΔL → 形状 (模态系数牛顿解)."""
        return solve_dL_to_shape(dL_mm, self.basis, **kw)

    def solve_dL_fused(self, dL_mm, tau_meas, **kw):
        """ΔL + τ (12 维观测) → 形态系数 a."""
        if self.tau_pred is None:
            raise RuntimeError("先调用 fit_tau(records)")
        return solve_fused(dL_mm, tau_meas, self.basis, self.tau_pred, **kw)

    def solve_dL_fused_tau(self, dL_mm, tau_meas, **kw):
        """★ 推荐: 以 τ 为未知量的融合求解 (用三次 â(τ) 前向)."""
        if self.tsp is None:
            raise RuntimeError("先调用 fit_tau_shape(records)")
        return solve_fused_tau(dL_mm, tau_meas, self.tsp, **kw)

    def shape_of_tau(self, tau, nsub=60):
        """τ → 形状 (确定性前向, 无需迭代)."""
        if self.tsp is None:
            raise RuntimeError("先调用 fit_tau_shape(records)")
        return shape_of_tau(tau, self.tsp, nsub=nsub)

    # ── 诊断 ──
    def info(self):
        b = self.basis
        if b is None:
            return dict(r=self.r, built=False)
        err = None
        if isinstance(self.tau_pred, dict):
            err = self.tau_pred.get("fit_err")
        return dict(built=True, r=b["r"], energy=b["energy"],
                    nS=len(b["S_grid"]), EI0=self.params.EI,
                    k_rod=self.params.k_rod, tau_pred=err,
                    has_tsp=self.tsp is not None)

    def describe(self):
        i = self.info()
        print("=" * 66)
        print("POD 降阶模型 (POD-Galerkin ROM)")
        print("=" * 66)
        if not i.get("built"):
            print("  未建基 (传 records= 或 basis=)")
        else:
            print(f"  阶数 r={i['r']}   能量占比={i['energy']*100:.3f}%   "
                  f"快照网格点数 nS={i['nS']}")
            if i["tau_pred"] is not None:
                print(f"  τ̂(a) 拟合残差中位 = {i['tau_pred']:.3f} N")
            print(f"  τ→a 三次前向: {'已拟合' if i['has_tsp'] else '未拟合'}")
        print("=" * 66)



# ══════════════════════════════════════════════════════════════════════
# 函数式入口
# ══════════════════════════════════════════════════════════════════════
def solve_tendon_forces(dL_target_mm, tau0=None, **kw):
    """快捷: ΔL → τ (逆解模型)."""
    return LookupInverseSolver(**kw).solve_dL(dL_target_mm, tau0=tau0, safe=True)


def make_inverse_solver(path=None, table=None, **kw):
    """快捷: 构造逆解模型 (给 path/table 则启用查表定位)."""
    return LookupInverseSolver(path=path, table=table, **kw)


def solve_pose(p_des, R_des, path=None, table=None, tau0=None, **kw):
    """快捷: 位姿 → τ. 有表 → 查表定位+精化; 无表 → 纯模型冷启动."""
    return LookupInverseSolver(path=path, table=table, **kw).solve_pose(
        p_des, R_des, tau0=tau0)


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    tbl = None
    if "--table" in argv:
        i = argv.index("--table")
        tbl = argv[i + 1] if i + 1 < len(argv) else "vc_table_40.npz"

    S = LookupInverseSolver(path=tbl)
    S.describe()
    print()
    ok = S.self_test()
    if tbl:
        print()
        print("── 表 ↔ 模型 一致性抽检 ──")
        for r in S.verify(n=3, seed=1):
            print(f"  表点 #{r['i']}: 正解复算 Δp={r['dp_mm']:.4f} mm, "
                  f"Δθ={r['dth_mrad']:.3f} mrad, ΔΔL={r['ddL_mm']:.4f} mm")
    print()
    print(f"{'ALL PASS ✅' if ok else 'FAIL ❌'}")
    sys.exit(0 if ok else 1)