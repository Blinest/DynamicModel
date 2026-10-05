# -*- coding: utf-8 -*-
"""
main.py — 项目统一入口 (建表 / 报表 / 绘图 / 表检查)
====================================================
把散在各处的入口收成一条命令; 建表的核心参数 **tau_max 是形参**, 直接写在命令行上。

用法
----
    python main.py build 40                  # 建 40N 表
    python main.py build 30 40 60            # 一次建多张 (串行)
    python main.py build 60 --n 1500 --refine 300 --batch 200
    python main.py build 40 --force           # 覆盖已存在的表
    python main.py build 40 --no-save         # 只跑不存盘 (试算)

    python main.py info                      # 列出所有表 + 版本键校验
    python main.py report 40                 # 统计报表 (→ stats_table.csv)
    python main.py csv 40                    # 导出 vc_table_wide/key.csv
    python main.py plot 40 --dpi 300         # 可达空间图
    python main.py plot 40 --compare 40      # 张力上限对比图

说明
----
* 工作目录需为 tools/ (脚本与表都在此处);
* 建表是长任务 (40N ≈ 33 min 建表 + 7 min refine), 建议后台跑;
* `main.py info` 会用 `LookupInverseSolver` 校验每张表的版本键 —— **version_ok=False 的表不可用于查表逆解**。
"""
import argparse
import glob
import os
import sys
import time

__version__ = "2026-09-26"


# ══════════════════════════════════════════════════════════════════════
def cmd_build(a):
    """按 τmax 建表 (可多个)."""
    from ik_table import build_table_by_tau_max
    print(f"== 建表任务: τmax = {a.tau} ==", flush=True)
    t_all = time.time()
    for t in a.tau:
        t0 = time.time()
        fn, recs, Pf = build_table_by_tau_max(
            float(t), N=a.n, n_refine=a.refine, batch_size=a.batch,
            ladder_levels=a.ladder, force=a.force, save=not a.no_save)
        if recs is None:
            print(f"  [{t}N] 跳过 (已存在; 用 --force 覆盖)", flush=True)
        else:
            print(f"  [{t}N] 完成: {fn}  {len(recs)} 点  "
                  f"用时 {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"\n全部完成, 总用时 {(time.time()-t_all)/60:.1f} min", flush=True)
    return 0


def cmd_info(a):
    """列出所有表并校验版本键 (是否可用于查表逆解)."""
    from inverse_solver import LookupInverseSolver
    files = sorted(glob.glob("vc_table_*.npz"))
    if not files:
        print("(当前目录无 vc_table_*.npz)")
        return 1
    print(f"{'文件':26s} {'N':>6} {'版本键':>9} {'段A EI0':>9} {'段B EI0':>9}  {'修改时间':>19}")
    print("-" * 90)
    bad = 0
    unver = 0
    for f in files:
        try:
            L = LookupInverseSolver(path=f)
            ok = L.version_ok
            bad += (not ok)
            cs = getattr(L.table.params, "cells", None) or []
            ea = f"{cs[0].EI0:.4f}" if cs else "—"
            eb = f"{cs[1].EI0:.4f}" if len(cs) > 1 else "—"
            mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f)))
            print(f"{f:26s} {len(L.table.recs):>6} {str(ok):>9} {ea:>9} {eb:>9}  {mt:>19}")
            if getattr(L.table, "stored_version", None) is None:
                unver += 1
                print("    ⚠ 表内无持久化版本键 (旧表) → 无法判定是否过期, 请用 "
                      "--force 重建或重存后校验")
            if not ok:
                for k, v in (L.version_diff or {}).items():
                    print(f"    ⚠ 版本差异 {k}: 表={v[0]}  模型={v[1]}")
        except Exception as e:
            bad += 1
            print(f"{f:26s}  读取失败: {e}")
    print("-" * 90)
    if bad:
        print(f"⚠ {bad} 张表版本键不一致 → **不可用于查表逆解**")
    elif unver:
        print(f"⚠ {unver} 张表无持久化版本键 → 无法判定是否过期, 请 --force 重建")
    else:
        print("✅ 所有表版本键与建表时一致")
    return 0 if bad == 0 else 2


def _tbl(tag):
    return f"vc_table_{tag}.npz"


def cmd_report(a):
    """统计报表 (复用 ik_table 的 stats 子命令)."""
    import ik_table
    return ik_table.main(["stats", _tbl(a.tag)])


def cmd_csv(a):
    """导出 CSV (复用 ik_table 的 csv 子命令)."""
    import ik_table
    return ik_table.main(["csv", _tbl(a.tag)])


def _rpy_to_R(rpy_deg):
    """roll-pitch-yaw (deg, 内旋 ZYX: R = Rz@Ry@Rx) → 3×3 旋转矩阵."""
    import numpy as np
    r, p, y = np.radians(np.asarray(rpy_deg, float))
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def cmd_solve(a):
    """位姿 → ΔL (查表定初值 + 精化)."""
    import numpy as np
    from inverse_solver import LookupInverseSolver
    from ik_table import apply_safety
    L = LookupInverseSolver(path=a.table, verbose=a.verbose)
    p_des = np.asarray(a.pos, float) / 1000.0                 # mm → m
    R_des = _rpy_to_R(a.rpy) if a.rpy else np.eye(3)
    t0 = time.time()
    o = L.solve_pose(p_des, R_des, pod=a.pod, verify=(not a.no_verify),
                     max_iter=a.max_iter, verbose=a.verbose,
                     refine_n=(int(a.refine_n) or None))
    dt = time.time() - t0
    # ── ★ 发散防护: 结果不可信时**拒绝下发 ΔL** ──
    #   (可达目标实验 #4 曾返回 Δp=217.9mm / ΔL 误差 205.9mm 的指令, 危险)
    if not bool(o.get("solved", True)):
        print(f"\n❌ 求解失败 —— 结果不可信, **ΔL 不予下发**")
        print(f"   原因 : {o.get('fail_reason', '未知')}")
        print(f"   残差 : Δp = {o['pos_res_mm']:.3f} mm   Δθ = {o['rot_res_mrad']:.3f} mrad"
              f"   迭代 {o['n_iter']} 次   用时 {dt:.2f} s")
        print(f"   参考 : τ = {np.round(o['tau'], 3)} N   (仅供参考, 未经验证)")
        print("   建议 : ① 加 --reach 判断目标是否在可达集内"
              " ② 提高 --max-iter / 放宽 tol  ③ 换更近的位姿")
        return 2
    dLs, fl = apply_safety(np.asarray(o["dL_mm"], float), tau=o["tau"])
    print(f"\n目标位姿: p = {np.round(a.pos, 2)} mm   RPY = {a.rpy or [0, 0, 0]} (deg)")
    print(f"来源 = {o['source']}   refine = {o.get('refine', 'full')}   "
          f"迭代 {o['n_iter']} 次   用时 {dt:.2f} s")
    print(f"残差: Δp = {o['pos_res_mm']:.4f} mm   Δθ = {o['rot_res_mrad']:.3f} mrad")

    # ── ★ 可达性判据 (见 §11.19) ──
    rc = None
    sol = o.get("sol") if hasattr(o, "get") else None
    if isinstance(sol, dict):
        rc = sol.get("reachability")
    if a.reach:
        feas = o.get("feasible", None)
        proj = o.get("projected", None)
        if rc is None:
            print("\n── 可达性 ──")
            print("  (未计算: 该路径未走全阶牛顿 —— POD 路径不产出可达判据)")
        if rc is not None:
            print("\n── 可达性 ──")
            print(f"  目标是否在可达集内 : {'✅ 是' if rc['feasible'] else '❌ 否'}"
                  f"   (可行步长比 {rc['step_frac']:.3f})")
            print(f"  消除残差所需 τ 步   : {rc['need_N']:.1f} N "
                  f"(τ_max = {L.tau_max:.1f} N -> {rc['need_N'] / L.tau_max:.1f} 倍)")
            print(f"  距最近边界裕度      : {rc['room_N']:.2f} N")
            print(f"  顶满的丝            : {rc['saturated'] or '无'}")
            print(f"  残差地板(线性预估)  : {rc['floor']:.4f} mm")
            if rc['saturated']:
                tp = np.asarray(o.get('tau_proj', o['tau']), float)
                for k in rc['saturated']:
                    tv = float(tp[k]) if tp.ndim else float(tp)
                    side = "τ_max" if tv > L.tau_max * 0.5 else "0"
                    print(f"    · 丝 {k}: **限制步长**(最先顶到边界), 投影后 τ = {tv:.3f} N"
                          f"  → 顶到 {side}")
        if proj:
            print(f"  ⇒ 已返回**投影解**(最近可达 τ): {np.round(o.get('tau_proj', o['tau']), 3)} N")
        if feas is False:
            print("  ⚠ 目标超出可达集: 残差存在正地板, 继续迭代无望(已提前退出)。"
                  "\n     → 若需可达位姿, 请用投影解; 若必须达到该位姿, 需提高 τ_max(本例需远超)。")

    # ── 🛡 发散防护: 求解不可信时**不要**当下发指令 (见 §11.20 待办①) ──
    usable = o.get("usable", None)
    diverged = o.get("diverged", None)
    res0 = o.get("res_init_mm", None)
    if usable is False:
        print("\n🛡 ── 求解不可信 ──")
        if diverged:
            print(f"  ⚠ **检测到发散**: 优化器结果比表热启动更差 → 已自动回退到热启动 τ")
        if res0 is not None:
            print(f"  热启动残差 {res0:.3f} mm  →  当前残差 {o['pos_res_mm']:.3f} mm")
        print(f"  ⚠ 残差 {o['pos_res_mm']:.3f} mm 超出可用阈值 ⇒ **该 ΔL 仅供参考, 请勿直接下发**")
        print(f"\n⚠ 参考(不可用) ΔL = {np.round(dLs, 3)} mm")
        print(f"  对应 τ  = {np.round(o['tau'], 3)} N")
        return 0
    print(f"\n★ 下发 ΔL = {np.round(dLs, 3)} mm")
    print(f"  对应 τ  = {np.round(o['tau'], 3)} N")
    print(f"  安全: ok={fl['ok']}  clipped={fl['clipped']}  slack={fl['slack']}")
    return 0


def cmd_plot(a):
    """可达空间图 / 张力上限对比图."""
    from plotting import plot_workspace, plot_tau_compare
    if a.compare:
        tags = list(a.compare)
        pal = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
        cmap = {t: pal[i % len(pal)] for i, t in enumerate(tags)}
        r = plot_tau_compare(tuple(tags), cmap=cmap, dpi=a.dpi)
        print("对比图:", r["png"], "已用表:", r["have"])
    else:
        r = plot_workspace(str(a.tags[0]), dpi=a.dpi)
        print("主图:", r["main_png"], "| 副图:", r["orient_png"])
        print(f"  N={r['N']}  r∈[{r['r_min_mm']:.0f},{r['r_max_mm']:.0f}]mm  "
              f"倾角max={r['tilt_max_deg']:.1f}°")
    return 0


# ══════════════════════════════════════════════════════════════════════
def build_parser():
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="TCR 连续体机器人 — 统一入口 (建表 / 报表 / 绘图 / 表检查)")
    ap.add_argument("--version", action="version", version=f"main.py {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="按 τmax 建表 (可多个)")
    b.add_argument("tau", nargs="+", help="张力上限, 如: 40 或 30 40 60")
    b.add_argument("--n", type=int, default=None, help="采样点数 (默认按 τmax 自动: 20→977, 30/40→1200)")
    b.add_argument("--refine", type=int, default=300, help="refine 目标网格 n_p=n_d (默认 300)")
    b.add_argument("--batch", type=int, default=200, help="批量打靶样本数 (默认 200)")
    b.add_argument("--ladder", type=int, default=16, help="延拓阶梯级数 (默认 16, 粗了会不收敛)")
    b.add_argument("--force", action="store_true", help="覆盖已存在的表")
    b.add_argument("--no-save", action="store_true", help="只跑不存盘")
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("info", help="列出所有表 + 版本键校验")
    i.set_defaults(func=cmd_info)

    r = sub.add_parser("report", help="统计报表")
    r.add_argument("tag", nargs="?", default="40", help="如 40 → vc_table_40.npz")
    r.set_defaults(func=cmd_report)

    c = sub.add_parser("csv", help="导出 CSV")
    c.add_argument("tag", nargs="?", default="40")
    c.set_defaults(func=cmd_csv)

    sv = sub.add_parser("solve", help="位姿 → ΔL (查表 + 精化)")
    sv.add_argument("--pos", nargs=3, type=float, required=True,
                    metavar=("X", "Y", "Z"), help="目标末端位置 (mm)")
    sv.add_argument("--rpy", nargs=3, type=float, default=None,
                    metavar=("R", "P", "Y"), help="目标姿态 roll/pitch/yaw (deg, 默认单位阵)")
    sv.add_argument("--table", default="vc_table_60.npz", help="表文件 (默认 vc_table_60.npz)")
    sv.add_argument("--max-iter", type=int, default=12)
    sv.add_argument("--pod", action="store_true",
                    help="用 POD 降阶精化 (快, 但其自评残差严重乐观 127~744x, "
                         "必须配 verify; 默认关)")
    sv.add_argument("--no-verify", action="store_true", help="POD 精化后不做全阶复核 (更快)")
    sv.add_argument("--refine-n", type=int, default=0,
                    help="精化网格 n_p=n_d; 0 = 用表自带分辨率 (默认, 推荐)。"
                         "⚠ 2026-09-27 实测: n=60 反而**更慢更差** —— 118.9 s / 残差 2.28 mm "
                         "(6 轮迭代) vs n=300 的 47.9 s / 0.024 mm (1 轮)。原因是粗网格自己"
                         "带来 mm 级离散误差, 牛顿追不到底、白烧迭代; 与 _refine_model 里"
                         "'n=60 快 25x' 的旧注释相反。")
    sv.add_argument("--verbose", action="store_true")
    sv.add_argument("--reach", action="store_true",
                    help="显示可达性判据 (目标是否在可达集/残差地板/投影解)")
    sv.set_defaults(func=cmd_solve)

    pl = sub.add_parser("plot", help="绘图")
    pl.add_argument("tags", nargs="*", default=["40"], help="表标签, 如 40")
    pl.add_argument("--dpi", type=int, default=150)
    pl.add_argument("--compare", nargs="*", default=None,
                    help="改为画张力上限对比图, 可指定多个 tag")
    pl.set_defaults(func=cmd_plot)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    return a.func(a) or 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
