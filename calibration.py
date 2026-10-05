# -*- coding: utf-8 -*-
"""标定工具集 (ΔL 仿射标定: 拟合 / 应用)

本文件由 calib_fit.py + calib_apply.py 合并而来 (原文件已删除)。

用法
----
    python calibration.py fit [CSV] [选项]    # 拟合标定并保存参数
    python calibration.py show                # 查看 Calib 用法

- `Calib` 类保持在**模块级**, 可直接 `from calibration import Calib` 复用;
- `fit` 子命令保留原 calib_fit.py 的全部 argparse 逻辑。
"""
import sys

__all__ = ["Calib", "SUBCOMMANDS", "main"]


# ══════════════════════════════════════════════════════════════════════
# Calib — ΔL 仿射标定 (原 calib_apply.py, 模块级保留可 import)
# ══════════════════════════════════════════════════════════════════════
import os
import numpy as np


class Calib:
    """ΔL 仿射标定。"""

    def __init__(self, A=None, b=None, meta=None):
        self.A = np.eye(6) if A is None else np.asarray(A, float)
        self.b = np.zeros(6) if b is None else np.asarray(b, float)
        self.meta = meta or {"identity": True}
        self._Ainv = None

    # ---- 构造 ----
    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def load(cls, path):
        """载入 calib_*.npz；文件不存在时返回恒等标定（不报错，便于部署）。"""
        if not path or not os.path.exists(path):
            c = cls.identity()
            c.meta = {"identity": True, "missing": path}
            return c
        z = np.load(path)
        meta = {k: (z[k].item() if z[k].shape == () else z[k].tolist())
                for k in z.files if k not in ("A", "b", "Ainv")}
        c = cls(z["A"], z["b"], meta)
        if "Ainv" in z.files and np.isfinite(z["Ainv"]).all():
            c._Ainv = z["Ainv"]
        return c

    # ---- 用 ----
    def command(self, dL_model_mm):
        """模型 ΔL → 下发 ΔL。"""
        x = np.atleast_2d(np.asarray(dL_model_mm, float))
        y = x @ self.A.T + self.b
        return y[0] if np.ndim(dL_model_mm) == 1 else y

    def display(self, dL_cmd_mm):
        """下发 ΔL → 模型口径（上位机显示/存档用）。A 奇异时原样返回。"""
        if self._Ainv is None:
            try:
                self._Ainv = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                return np.asarray(dL_cmd_mm, float)
        x = np.atleast_2d(np.asarray(dL_cmd_mm, float))
        y = (x - self.b) @ self._Ainv.T
        return y[0] if np.ndim(dL_cmd_mm) == 1 else y

    def __repr__(self):
        if self.meta.get("identity"):
            return "<Calib 恒等 (未标定)>"
        return (f"<Calib n={self.meta.get('n_used','?')} "
                f"train={self.meta.get('train_rmse',float('nan')):.2f}mm "
                f"cv={self.meta.get('cv_med',float('nan')):.2f}mm>")


# ══════════════════════════════════════════════════════════════════════
# 子命令 `fit` — 拟合标定 (原 calib_fit.py)
# ══════════════════════════════════════════════════════════════════════
def cmd_fit():
    """拟合 ΔL 仿射标定  ← 原 calib_fit.py"""
    # -*- coding: utf-8 -*-
    """
    calib_fit.py — ΔL 仿射标定 (实测 CSV → 标定矩阵)
    ====================================================

    ## 为什么需要

    模型给的 ΔL_sim 与实机需要的 ΔL_cmd 之间差在四件事上，模型里都没有：

      1. **零点**     表基准 = 零张力直臂；实机基准 = 装配/预紧态
                       —— 这正是"ΔL 为什么全是负值"的来源
      2. **丝弹性**   丝不是刚性的，张力下自身会伸长，模型按刚性丝算
      3. **Capstan**  丝过孔/绕轮的摩擦，收丝侧与放丝侧不对称
      4. **卷径变化** 电机绕线盘上丝越绕越多，同样的电机转角对应不同的丝位移

    用一个仿射映射一次吃掉：

            ΔL_cmd = A · ΔL_sim + b

      A (6×6) 吸收 2/3/4（耦合与增益），b (6) 吸收 1（零点）。

    ## 输入

    上位机录的 CSV，列名沿用 `export_csv.py` 的约定：

      dL0..dL5        实测位移 (mm)，符号与模型一致：**负 = 收丝**
                      —— 即电机侧换算后的实际丝位移
      tau0..tau5      实测张力 (N)          [强烈建议有：用于剔除松弛丝]
      x_x,x_y,x_z     实测末端位置 (m)      [可选：剔除 ε 过大的点]
      p_x.. / R00..   实测末端位姿          [可选]

    至少要有一组 (dL, tau) 或 (dL, pose)。

    ## 模型侧 ΔL_sim 怎么来

    标定不需要"已知理论位移"。对每一行，用**实测张力 τ** 反查模型（`shooting`，
    用最近表节点热启动），得到模型在该张力下的形状 → ΔL_sim。这才是模型认为
    "同样张力下应该给的位移"，两者之差就是标定要修的量。

    ## 用法

        python calib_fit.py calib_20260922.csv --tag a12_60
        python calib_fit.py calib.csv --table vc_table_60.npz --diag

    输出 `calib_<tag>.npz`：A(6,6), b(6), Ainv(6,6), meta, 训练/留出残差。
    """
    import argparse
    import os
    import sys
    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8")
    from ik_table import load_table, default_params, _clone_params, so3_log
    from tendon_coupling import shooting, L1, L, SEG_A


    # ==================== CSV 读取 ====================

    def load_csv(path):
        """读实测 CSV。返回 dict，键为列名，值为 (N,) float 数组。"""
        import csv as _csv
        with open(path, newline="", encoding="utf-8-sig") as f:
            rd = _csv.DictReader(f)
            cols = {}
            for name in rd.fieldnames or []:
                cols[name.strip()] = []
            for row in rd:
                for k in cols:
                    v = (row.get(k) or "").strip()
                    try:
                        cols[k].append(float(v))
                    except ValueError:
                        cols[k].append(np.nan)
        return {k: np.asarray(v, float) for k, v in cols.items()}


    def _pick(cols, *names):
        """按候选名取列（大小写不敏感，允许 0/1 或 1/6 编号习惯）。"""
        low = {k.lower(): k for k in cols}
        for n in names:
            if n.lower() in low:
                return cols[low[n.lower()]]
        return None


    def assemble(rows):
        """把 CSV 列整理成 (N,6) 的 dL_cmd / tau / (N,3) 的 pos。缺列返回 None。

        列名对齐 SoftUI `session.rs` 的实际导出:

            timestamp_ms, sequence, device_id,
            motor_{1..6}_pos_mm / _vel_mmps / _acc_mmps2,
            bend_s1_angle_deg, bend_s2_angle_deg,
            tau_0..tau_5          实测张力 (N)
            dl_0..dl_5            实测丝位移 (mm), 负 = 收丝
            kappa_x_0, kappa_y_0 ... kappa_x_19, kappa_y_19   沿臂 20 点曲率 (1/m)
        """
        N = len(next(iter(rows.values())))

        def stack(prefix):
            """按 dl_0 / dL0 / dl0 / dl_1 等写法取 6 列。"""
            out = np.full((N, 6), np.nan)
            for i in range(6):
                c = _pick(rows,
                          f"{prefix}_{i}", f"{prefix}{i}",
                          f"{prefix}_{i+1}", f"{prefix}{i+1}")
                if c is not None:
                    out[:, i] = c
            return out

        dL = stack("dl")
        if not np.isfinite(dL).any():          # 回退旧命名
            dL = stack("dL")
        tau = stack("tau")
        # 也接受 motor_i_pos_mm 当作位移 (两者在 SoftUI 里是同一个量)
        if not np.isfinite(dL).any():
            for i in range(6):
                c = _pick(rows, f"motor_{i+1}_pos_mm")
                if c is not None:
                    dL[:, i] = c
        px = _pick(rows, "x_x", "p_x", "x", "px")
        py = _pick(rows, "x_y", "p_y", "y", "py")
        pz = _pick(rows, "x_z", "p_z", "z", "pz")
        pos = (np.stack([px, py, pz], 1) if all(v is not None for v in (px, py, pz))
               else None)
        return dL, tau, pos


    def assemble_kappa(rows, n_pts=20):
        """取沿臂曲率 (N, n_pts, 2) —— 列名 kappa_x_i / kappa_y_i。缺则 None。"""
        N = len(next(iter(rows.values())))
        kx = np.full((N, n_pts), np.nan)
        ky = np.full((N, n_pts), np.nan)
        got = False
        for i in range(n_pts):
            cx = _pick(rows, f"kappa_x_{i}", f"kx_{i}", f"kx{i}")
            cy = _pick(rows, f"kappa_y_{i}", f"ky_{i}", f"ky{i}")
            if cx is not None and cy is not None:
                kx[:, i] = cx; ky[:, i] = cy; got = True
        if not got:
            return None
        return np.stack([kx, ky], axis=2)


    # ==================== 模型侧：τ → ΔL_sim ====================

    def model_dL_from_tau(tau_rows, table, params=None, verbose=True):
        """对每行实测张力反查模型 → ΔL_sim (N,6) mm。

        用最近表节点的打靶未知量 x 热启动，每点通常 1~3 次迭代。
        τ 全 0 的行（完全松弛）模型无定义，标记为 NaN。
        """
        P = params or (table.params if table is not None else default_params())
        L0 = np.array([L1 if w in SEG_A else L for w in range(6)])
        from ik_table import _wire_geom_len_batch   # noqa: E402

        recs = table.recs if table is not None else []
        TAU_tab = (np.array([r["tau"] for r in recs]) if recs else np.zeros((0, 6)))
        X_tab = (np.array([r["x"] for r in recs]) if recs else np.zeros((0, 6)))

        out = np.full((len(tau_rows), 6), np.nan)
        n_bad = 0
        for i, tau in enumerate(tau_rows):
            if not np.isfinite(tau).all() or np.linalg.norm(tau) < 1e-9:
                n_bad += 1
                continue
            x0 = None
            if len(TAU_tab):
                x0 = X_tab[int(np.argmin(np.linalg.norm(TAU_tab - tau, axis=1)))]
            s = shooting(np.clip(tau, 0, None), P, g_world=(0, 0, -9.81),
                         tol=1e-9, max_iter=40, x0=x0)
            if not s["converged"] or not np.isfinite(s["p_curve"][-1]).all():
                n_bad += 1
                continue
            Pc = s["p_curve"][None]
            Rc = s["R_curve"][None]
            Lg = _wire_geom_len_batch(Pc, Rc)[0]
            out[i] = (Lg - L0) * 1000.0
        if verbose:
            print(f"  模型反查: {len(tau_rows)-n_bad}/{len(tau_rows)} 成功 "
                  f"({n_bad} 行跳过: τ≈0 或无解)")
        return out


    # ==================== 仿射拟合 ====================

    def fit_affine(X, Y, ridge=1e-6, diag=False):
        """拟合 Y ≈ X·Aᵀ + b，X,Y 均为 (N,6)。

        diag=True 时只拟合对角增益 + 偏置（12 参数，样本少时更稳）；
        diag=False 全 6×6（42 参数，需 N ≫ 42）。

        返回 dict(A (6,6), b (6), train_rmse, n_used)。
        """
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        X, Y = X[ok], Y[ok]
        N = len(X)
        if N < (12 if diag else 42):
            print(f"  ⚠ 有效样本只有 {N} 条"
                  f"（{'对角' if diag else '全阵'} 需要 ≥{12 if diag else 42}），"
                  f"结果仅供参考")

        if diag:
            # 逐丝独立拟合：ΔL_cmd[i] = A[i,i]·ΔL_sim[i] + b[i]
            A = np.eye(6); b = np.zeros(6)
            for i in range(6):
                x, y = X[:, i], Y[:, i]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 2:
                    continue
                Amat = np.stack([x[m], np.ones(m.sum())], 1)
                sol, *_ = np.linalg.lstsq(Amat, y[m], rcond=None)
                A[i, i], b[i] = sol[0], sol[1]
            pred = X @ A.T + b
        else:
            # 全仿射：未知 [A(36), b(6)]，每行 6 个方程
            N = len(X)
            M = np.zeros((6 * N, 42))
            t = np.zeros(6 * N)
            for i in range(N):
                for r in range(6):
                    M[6 * i + r, 6 * r:6 * r + 6] = X[i]
                    M[6 * i + r, 36 + r] = 1.0
                    t[6 * i + r] = Y[i, r]
            reg = ridge * np.eye(42)
            sol = np.linalg.solve(M.T @ M + reg, M.T @ t)
            A = sol[:36].reshape(6, 6); b = sol[36:]
            pred = X @ A.T + b
        rmse = float(np.sqrt(np.mean((pred - Y) ** 2)))
        return {"A": A, "b": b, "train_rmse": rmse, "n_used": len(X)}


    def cross_validate(X, Y, k=5, **kw):
        """k 折交叉验证：报告泛化 ΔL 残差（中位/90%），这才是可采信的数字。"""
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        X, Y = X[ok], Y[ok]
        N = len(X)
        if N < 10:
            return {"cv_med": np.nan, "cv_p90": np.nan, "n": N}
        idx = np.arange(N)                      # 不 shuffle，保持采集时序分块
        folds = np.array_split(idx, k)
        errs = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            if len(tr) < 6:
                continue
            r = fit_affine(X[tr], Y[tr], **kw)
            pred = X[f] @ r["A"].T + r["b"]
            errs.append(np.linalg.norm(pred - Y[f], axis=1))
        if not errs:
            return {"cv_med": np.nan, "cv_p90": np.nan, "n": N}
        e = np.concatenate(errs)
        return {"cv_med": float(np.median(e)), "cv_p90": float(np.percentile(e, 90)),
                "n": N}


    # ==================== 主流程 ====================

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("csv", help="实测 CSV (见模块 docstring 的列约定)")
        ap.add_argument("--table", default="vc_table_60.npz", help="模型查表")
        ap.add_argument("--tag", default="default", help="输出标签 -> calib_<tag>.npz")
        ap.add_argument("--diag", action="store_true",
                        help="只拟合对角增益+偏置 (12 参数, 样本少时更稳)")
        ap.add_argument("--ridge", type=float, default=1e-6, help="全阵拟合的岭正则")
        ap.add_argument("--out", default=None, help="输出路径, 默认 calib_<tag>.npz")
        args = ap.parse_args()

        print(f"[1] 读实测 CSV: {args.csv}")
        cols = load_csv(args.csv)
        dL_cmd, tau, pos = assemble(cols)
        kappa = assemble_kappa(cols)
        N = dL_cmd.shape[0]
        print(f"    {N} 行;  dL {'有' if np.isfinite(dL_cmd).any() else '缺'};  "
              f"tau {'有' if np.isfinite(tau).any() else '缺'};  "
              f"pos {'有' if pos is not None else '缺'};  "
              f"kappa20 {'有' if kappa is not None else '缺'}")
        if not np.isfinite(dL_cmd).all():
            raise SystemExit("CSV 缺位移列 (dl_0..dl_5 / motor_1..6_pos_mm) —— "
                             "实测位移是标定的目标量, 必须有。")
        if not np.isfinite(tau).all():
            raise SystemExit("CSV 缺 tau_0..tau_5 —— 模型侧 ΔL_sim 靠它反查, 必须有。")

        print(f"[2] 载入模型表: {args.table}")
        T = load_table(args.table, params=None) if os.path.exists(args.table) else None
        if T is None:
            print("    ⚠ 表不存在 → 用 default_params() 冷启动模型 (慢)")

        print("[3] 模型反查 ΔL_sim (实测 τ → 模型位移)")
        dL_sim = model_dL_from_tau(tau, T)

        print("[4] 拟合 ΔL_cmd = A·ΔL_sim + b")
        fit = fit_affine(dL_sim, dL_cmd, ridge=args.ridge, diag=args.diag)
        cv = cross_validate(dL_sim, dL_cmd, k=5, ridge=args.ridge, diag=args.diag)
        print(f"    训练 RMSE = {fit['train_rmse']:.3f} mm  (n={fit['n_used']})")
        print(f"    5折留出  中位 = {cv['cv_med']:.3f} mm  90% = {cv['cv_p90']:.3f} mm")
        if args.diag:
            print(f"    对角增益 = {np.round(np.diag(fit['A']), 4)}")
            print(f"    偏置 b   = {np.round(fit['b'], 3)} mm")

        # 未标定 vs 已标定：拿"模型原始 ΔL"与实测的差作对照
        raw = np.linalg.norm(dL_sim - dL_cmd, axis=1)
        ok = np.isfinite(raw)
        print(f"\n    标定前 模型 vs 实测 残差 中位 = {np.median(raw[ok]):.3f} mm")
        pred = dL_sim @ fit["A"].T + fit["b"]
        aft = np.linalg.norm(pred - dL_cmd, axis=1)
        print(f"    标定后            残差 中位 = {np.median(aft[ok]):.3f} mm"
              f"   -> 改善 {np.median(raw[ok])/max(np.median(aft[ok]),1e-9):.1f}x")

        A = fit["A"]
        try:
            Ainv = np.linalg.inv(A)
            print(f"    A 条件数 = {np.linalg.cond(A):.2f}  (可逆, 显示时可反算)")
        except np.linalg.LinAlgError:
            Ainv = None
            print("    ⚠ A 奇异, 不存 Ainv")

        out = args.out or f"calib_{args.tag}.npz"
        np.savez_compressed(
            out, A=A, b=fit["b"],
            Ainv=(Ainv if Ainv is not None else np.full((6, 6), np.nan)),
            diag=np.array(args.diag),
            train_rmse=np.array(fit["train_rmse"]),
            cv_med=np.array(cv["cv_med"]), cv_p90=np.array(cv["cv_p90"]),
            n_used=np.array(fit["n_used"]), src=np.array(os.path.basename(args.csv)),
            table=np.array(os.path.basename(args.table)),
        )
        print(f"\n[5] 已存 {out}")


    if True:  # 原 __main__ 守卫
        main()


def cmd_show():
    """显示 Calib 用法"""
    print(Calib.__doc__ or "")
    print("用法:  from calibration import Calib")
    print("       c = Calib(A, b); dL_cal = c.apply(dL_raw)")
    return 0


SUBCOMMANDS = {
    "fit": cmd_fit,      # 拟合标定
    "show": cmd_show,    # 查看 Calib 用法
}

_USAGE = """标定工具集
  fit [CSV] [选项]   拟合 ΔL 仿射标定 (见 python calibration.py fit --help)
  show               查看 Calib 用法
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in SUBCOMMANDS:
        print(f"未知子命令: {cmd}\n")
        print(_USAGE)
        return 2
    fn = SUBCOMMANDS[cmd]
    if cmd == "show":
        return fn()
    # fit 内部自带 argparse → 临时替换 sys.argv
    old = sys.argv
    sys.argv = [old[0]] + rest
    try:
        return fn()
    finally:
        sys.argv = old


if __name__ == "__main__":
    sys.exit(main() or 0)
