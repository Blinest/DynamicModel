# -*- coding: utf-8 -*-
"""
ik_table.py — §10 位姿驱动逆解与查表 (离线建表 + 在线查询 + 惰性补洞)
=====================================================================
目标: 给定末端位姿 (r, R) → 输出 6 根丝位移 ΔL[6] (下发下位机)
      + 曲率分布 κ(s) / 骨架数据 (上位机显示)。

三层结构 (§10.1):
  离线建表:  τ 采样 → 内层 VC 打靶 → pose → (ΔL, κ 系数) 入库
  在线查询:  查表/插值 (可 1~2 步牛顿精化) → ΔL 下发
  空洞补充:  查询未命中 → 现场跑外层牛顿 (pose → τ) → 解入库 (惰性)

复用 tools/sdm_vc.py 的 shooting(τ→骨架/姿态/曲率/内力);
本文件新增: 建表(build_table)、外层位姿牛顿(solve_pose_multi)、
查询(PoseTable)、κ(s) 节点存储(_kappa_nodes)、κ→骨架重建(reconstruct_shape)。

坐标系: 世界系 Z 轴朝上, 无重力时臂沿 +Z 伸直 (r(L)=[0,0,L])。
本构 (§7): C=Ha·κ ⇒ κ=Ha⁻¹C; 段A(两管叠) Ha=diag(2EI,2EI,2GJ), 段B 单管。

依赖: numpy, scipy(cKDTree 可选)。无第三方 Hilbert/LHS 包 — 自实现。
"""
import json
import sys
import numpy as np

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:                       # 退化: 暴力半径/最近邻 (慢, 仅小表)
    HAVE_SCIPY = False
    cKDTree = None

from tendon_coupling import (SDMParams, shooting, L1, L2, L, SEG_A, SEG_B, hat, _E3,
                    LatticeCell, N_CIRC_SEG_A, N_CIRC_SEG_B,
                    ALPHA_DEG, C0_DEFAULT, L_STRUT_DEFAULT, EI0_DEFAULT,
                    K_ROD_DEFAULT, R_DISK_DEFAULT, GJ_DEFAULT,
                    tendon_path_length,      # 几何丝长 → ΔL
                    _L0_wire, _wire_geom_len_batch)  # 几何辅助
from tendon_coupling import (shooting_batch, shooting_batch_retry,
                          forward_batch)   # 建表加速 (批量打靶)


# ==================== SO(3) 指数/对数映射 ====================

def so3_exp(w):
    """旋转向量 w(3) → 旋转矩阵 (Rodrigues)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    W = hat(w)
    if th < 1e-12:
        return np.eye(3) + W + 0.5 * (W @ W)
    a = w / th
    W = hat(a)
    return np.eye(3) + np.sin(th) * W + (1.0 - np.cos(th)) * (W @ W)


def so3_log(R):
    """旋转矩阵 → 旋转向量 w, ‖w‖∈[0,π] (连续, 无 ±π 回绕)."""
    R = np.asarray(R, float)
    c = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    th = np.arccos(c)
    if th < 1e-9:                                    # 小角度
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if th > np.pi - 1e-6:                            # 近 π: 由 (R+I)/2 取轴
        A = (R + np.eye(3)) / 2.0
        d = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(d))
        if d[k] < 1e-12:
            return np.zeros(3)
        ax = A[:, k] / d[k]
        ax = ax / np.linalg.norm(ax)
        return th * ax
    return 0.5 * th / np.sin(th) * np.array([R[2, 1] - R[1, 2],
                                             R[0, 2] - R[2, 0],
                                             R[1, 0] - R[0, 1]])


def rpy_from_R(R):
    """R → (roll, pitch, yaw), ZYX 内旋 = Rz(y)Ry(p)Rx(r). 仅作索引键/显示."""
    R = np.asarray(R, float)
    sp = min(1.0, max(-1.0, -R[2, 0]))
    pitch = np.arcsin(sp)
    if abs(sp) < 0.9999:
        return (np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0]))
    return (0.0, pitch, np.arctan2(-R[0, 1], R[1, 1]))   # 万向锁: roll=0


def rotvec_mrad(R):
    """R → 旋转向量(mrad). 白化度量/树的切分维度都用它(连续无回绕)."""
    return so3_log(R) * 1000.0


# ==================== 采样 (§10.2) ====================

def lhs(n, dim, seed=0):
    """拉丁超立方 → [0,1]^dim."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, dim))
    for d in range(dim):
        pts[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return pts


def _morton_key(pt_int, n_bits, dim):
    """交织位: dim 维 n_bits 位整数 → 单整数 (Z 序/Morton)."""
    s = 0
    for b in range(n_bits):
        for j in range(dim):
            s |= ((int(pt_int[j]) >> b) & 1) << (b * dim + j)
    return s


def space_filling_order(points_unit, n_bits=10):
    """把 [0,1]^d 点按空间填充曲线排序 → 排序索引 (§10.2 热启动次序).

    用 Z 序(Morton)而非严格 Hilbert: 二者都保证"索引空间相邻 ⇒ τ 空间相邻",
    热启动有效性一致, 而 Morton 实现简单、无第三方依赖。需要严格 Hilbert 时
    替换本函数即可(接口不变)。
    """
    pts = np.asarray(points_unit, float)
    n = pts.shape[0]
    N = 1 << n_bits
    idx = np.clip((pts * N).astype(np.int64), 0, N - 1)
    keys = np.array([_morton_key(idx[i], n_bits, pts.shape[1]) for i in range(n)])
    return np.argsort(keys)


# 兼容 §10.2 的命名
hilbert_order = space_filling_order


# ==================== κ(s) 节点存储 (§10.4) ====================
# 早期版本把 κ(s) 拟合成"6 阶/24 点"分段多项式。在 20N 表的大曲率下多项式追不上
# 段A基座处的曲率尖峰, 额外引入 ~1mm 末端误差 (实测 3.87 → 2.9 mm, 见 §10 诊断)。
# 现直接存打靶曲线的 κ 节点: kcoef = {"S": (m,), "K": (m,3)}, S 严格递增,
# K 逐节点按所属段取 κ = Ha⁻¹C。存整条曲线 (m = n_p+n_d+1), 不再降阶。

def _kappa_nodes(s_curve, C_curve, params, v_curve=None):
    """由打靶曲线 (s, C[, v]) 生成节点表. 段界 L1 处 κ 有物理跃变, 节点保留原值.

    ⚠ **必须同时存 v(s)**: 旧模型把弧长当不可伸 (p'=R·e₃), 但中心杆/臂体轴向刚度
    `k_rod` 有限时 `v_z = 1+ν_z` 沿弧长显著偏离 1 (实测 0.85~0.90, 即压缩
    10~15%)。只存 κ 会让重建骨架凭空长 40~58mm (实测 1200 条, 中位 46mm)。
    存全 3 分量 v 与只存 v_z 代价相同, 且能带出剪切分量, 故整存。
    """
    s = np.asarray(s_curve, float)
    C = np.asarray(C_curve, float)
    K = np.empty_like(C)
    # ★ 分段晶胞 + **正弦本构的精确逆变换**
    #   ① 分段: 刚度取本段晶胞 (不再硬编码"段A ×2"; 旧版段A 少乘1倍、段B 多乘1倍)
    #   ② 本构: 模型弯矩 C = C0·sin(κ·L_strut) ⇒ κ = arcsin(C/C0)/L_strut
    #          (直接用 κ=C/EI 是**线性近似**, 大曲率处偏 ~1e-2, 实测 θ³/6 量级)
    #         抗扭分量不随曲率变: κ_z = C_z/GJ
    # ⚠ 段界判据必须带**容差**: 解网格上恰有一个节点落在 s≈L1, 浮点误差会把它
    #   判到段B ⇒ 用错刚度(段A/段B EI0 差 2 倍) ⇒ κ 反算直接错 2 倍 (实测)
    _tol = max(1e-9, 1e-9 * abs(L))
    mskA = s <= L1 + _tol
    for k, msk in ((0, mskA), (1, ~mskA)):
        if not np.any(msk):
            continue
        ck = params.cells[k] if getattr(params, "cells", None) else None
        C0k = ck.C0 if ck is not None else params.C0
        Ls = ck.L_strut if ck is not None else params.L_strut
        GJk = ck.GJ if ck is not None else params.GJ
        eu = ck.EI0 if ck is not None else params.EI
        Cxy = C[msk][:, :2]
        if getattr(params, "lattice", False) and Ls > 0 and C0k > 0:
            # |C_xy| = C0·sin(θ),  θ = |κ_xy|·L_strut
            nC = np.linalg.norm(Cxy, axis=1)
            th = np.arcsin(np.clip(nC / C0k, -1.0, 1.0))
            # κ_xy = C_xy·θ/|C_xy|  (不能逐分量 arcsin!); |C|→0 时退化为 1/C0
            # κ = (θ/L_strut)·û  —— 别漏了 /L_strut!
            scale = np.where(nC > 1e-12, (th / Ls) / np.where(nC > 1e-12, nC, 1.0),
                             1.0 / (C0k * Ls))
            K[msk, :2] = Cxy * scale[:, None]
        else:
            K[msk, :2] = Cxy / eu
        K[msk, 2] = C[msk][:, 2] / GJk
    out = {"S": s, "K": K}
    if v_curve is not None:
        V = np.asarray(v_curve, float)
        if V.shape[0] == len(s):                 # 与节点一一对应
            out["V"] = V.copy()
    return out


def vz_at_batch(kcoef, s_arr):
    """按弧长插值 v(s) (线应变, 含轴向 ν_z 与剪切 ν_x,ν_y).

    旧表没有 "V" 字段 → 退化为不可伸的 [0,0,1] (即旧的 p'=R·e₃ 行为)。
    """
    s = np.atleast_1d(np.asarray(s_arr, float))
    if "V" not in kcoef:
        v = np.zeros((len(s), 3)); v[:, 2] = 1.0
        return v
    S = kcoef["S"]; V = kcoef["V"]
    out = np.empty((len(s), 3))
    for c in range(3):
        out[:, c] = np.interp(s, S, V[:, c])
    return out


# 兼容旧名 (曾返回多项式系数)
_kappa_coeff = _kappa_nodes


def kappa_at_batch(kcoef, s_arr):
    """批量 κ(s): s_arr (n,) → (n,3). 节点线性插值."""
    S = np.asarray(kcoef["S"], float)
    K = np.asarray(kcoef["K"], float)
    s = np.asarray(s_arr, float)
    return np.stack([np.interp(s, S, K[:, k]) for k in range(3)], axis=1)


def kappa_at(kcoef, si, params=None):
    """指定弧长处体坐标曲率 κ(s) (§7 本构 κ=Ha⁻¹C). params 保留以便旧调用."""
    return kappa_at_batch(kcoef, np.atleast_1d(np.asarray(si, float)))[0]


# ==================== 单样本: 内层打靶 → 入库记录 ====================

# _L0_wire / _wire_geom_len_batch 已下沉到 tendon_coupling (纯几何), 在此导入引用:
_L0_wire = _L0_wire
_wire_geom_len_batch = _wire_geom_len_batch


def rec_from_sol(tau, sol, params):
    """由一次 VC 解打包成入库记录 (§10.2 schema)."""
    Lgeo = np.array([tendon_path_length(sol, w) for w in range(6)])
    return {
        "tau": np.asarray(tau, float),
        "dL_mm": (Lgeo - _L0_wire()) * 1000.0,
        "p_tip": np.array(sol["p_curve"][-1], float),
        "R_tip": np.array(sol["R_curve"][-1], float),
        "kcoef": _kappa_nodes(sol["s_curve"], sol["m_curve"], params,
                                  sol.get("v_curve")),
        "res": float(np.linalg.norm(sol["res"])),
        "converged": bool(sol["converged"]),
        "x": np.array(sol["x"], float),               # 打靶未知 [v0;u0] 供热启动
    }


def sample_one(tau, params, x0=None):
    sol = shooting(tau, params, g_world=(0, 0, -9.81), tol=1e-8,
                   max_iter=60, x0=x0)
    return rec_from_sol(tau, sol, params)


# ==================== 建表 (§10.2) ====================

def default_params():
    """默认参数: 臂体菱形晶胞骨架 (无中心杆) + 割线弯曲刚度.

    臂体**没有中心杆**, 承载与弯曲全由菱形晶胞承担, 故 EI0 与 k_rod 由
    **同一套晶胞材料/几何参数**推出 (`sdm_vc.cell_stiffness`), 改一处即同步:

        k_unit = 2·E·A·sin²α/L_strut          单胞轴向刚度
        EI0    = k_unit·l_ax·Σy²              l_ax=2·L_strut·sinα, Σy²=2·N_circ·r_i²
        k_rod  = k_unit/N_long                 N_long 个胞串联
        C0     = EI0/L_strut                  弯矩常数

    弯曲本构用**割线刚度** `EI_eq(κ) = EI0·sinc(κ·L_strut)`, κ→0 时 → EI0
    (有限非零), 符合"低曲率下需要很大力才变形"。

    ⚠ **EI0 不是自由参数**: 旧代码取 0.064 是配合 `theta_seed=π/6` 补丁凑的
    占位值 (0.128×sin(π/6)); 笔记 §5 的 "C↔Python 对拍" 只验证两套实现一致,
    从未验证 EI 的物理值。实证: 0.064 下 6 丝各 20N 把 450mm 臂压短 155mm (34%),
    荒谬; 按材料参数推出的 EI0≈0.3745 只压短 26mm (5.8%)。
    """
    # ── 分段晶胞: 段A(近基座) N_circ=6 铺满 L1; 段B N_circ=3 铺满 L2 ──
    ca = LatticeCell(N_circ=N_CIRC_SEG_A, L=L1)
    cb = LatticeCell(N_circ=N_CIRC_SEG_B, L=L2)
    p = ca.sd_params(GA=1e3, rhoA=0.0, disk_mass=0.0,
                     n_p=30, n_d=30, lattice=True)
    p.cells = (ca, cb)
    return p


def _clone_params(params, n_p=None, n_d=None):
    """由已有 params 复制一份 VCParams, **保留全部刚度/模型字段**.

    ⚠ 这是一处既存 bug 的修复: refine_table 与 load_table 原来手写关键字重建
    VCParams, 只带了 EI/GA/EA/GJ/r_disk/n_p/n_d —— 悄悄丢掉 r_bias/lattice,
    导致"表与代码不自洽但不报错" (ik_table.py 原 docstring 自己警告过这一点)。
    新增 k_rod/L_strut 后必须同一处收口, 否则 bug 会再扩散。
    """
    p = SDMParams(EI=params.EI, GA=params.GA, k_rod=params.k_rod,
                  GJ=params.GJ, r_disk=params.r_disk, rhoA=params.rhoA,
                  disk_mass=params.disk_mass,
                  n_p=(params.n_p if n_p is None else n_p),
                  n_d=(params.n_d if n_d is None else n_d),
                  rod_free_len=params.rod_free_len,
                  rod_constraint=params.rod_constraint,
                  lattice=params.lattice, C0=params.C0,
                  L_strut=params.L_strut)
    # ⚠ **分段晶胞必须一起带走**: 漏掉 cells 会让 refine_table / load_table 悄悄
    #   退化成旧的"段A ×2"规则 (段B 误用段A 刚度), 表与模型不自洽 —— 2026-09-26 踩过。
    cells = getattr(params, "cells", None)
    if cells is not None:
        p.cells = tuple(cells)
    return p


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def canonical_version_key(key) -> str:
    """版本键 → 规范 JSON 字符串 (写入 npz / 跨会话比对的唯一口径).

    ⚠ 只用 `dict == dict` 靠不住: 版本键里存的是 tuple, 经 JSON 往返会变成
    list, `tuple != list` 恒真 → 假报警. 故一律走字符串比较.
    """
    return json.dumps(key, sort_keys=True, default=_json_default)


def table_version_key(params, gravity_on=True, F_tip=(0, 0, 0)):
    """§10.6 版本键 =(F_tip, g, r_disk, 刚度, 网格, 中心杆弹簧/晶胞). 任一变化须重建表.

    网格分辨率必须进版本键: 同一 τ 在不同 n_p/n_d 下解出的平衡点不同 (~mm 级),
    不带分辨率的旧键会把粗网格表误判成同版本。
    中心杆轴向刚度 k_rod / 晶胞 L_strut 同理 —— 同 τ 下位形完全不同。

    ⚠ 容器一律用 **list** (不用 tuple): 该键要 JSON 序列化后写进 npz, tuple 往返
    会退化成 list, 直接 dict 比较会假报警。旧表没有持久化键 → 见 load_table。
    """
    return {"EI": params.EI, "GJ": params.GJ, "GA": params.GA,
            "k_rod": params.k_rod, "r_disk": params.r_disk,
            "F_tip": list(np.ravel(F_tip)),
            "n_p": params.n_p, "n_d": params.n_d,
            "lattice": bool(params.lattice),
            "C0": params.C0, "L_strut": params.L_strut,
            "gravity": [0, 0, -9.81] if gravity_on else [0, 0, 0],
            # 分段晶胞: 每段 (N_circ, L, EI0, C0, k_rod) —— 任一段改参数都要重建表
            "seg": [[c.N_circ, round(c.L_arm, 6), round(c.EI0, 12),
                     round(c.C0, 12), round(c.k_rod, 9)]
                    for c in (getattr(params, "cells", None) or [])]}


def refine_table(records, params=None, n_p=300, n_d=300, verbose=True):
    """§10.2b 网格细化: 用记录里的 x 作热启动, 在细网格上重解并重算全部派生量.

    背景: 建表默认 n_p=n_d=30 (单元 7.5mm), RK2 中点法在 20N 大曲率下离散误差
    ~8mm (实测)。把同一 x 挪到细网格残差立刻变 3.4e-2 —— 粗网格解只在粗网格上
    成立。细化后收敛率 100%、迭代中位 4 次 (x 是极好的初值)。

    重算顺序严格照 build_table 的纪律: 曲线必须在【最终 x】之后重算, 否则
    p_tip/R_tip/kcoef 会与 x 不自洽 (2026-09-15 修过的同类 bug)。
    """
    params = params or default_params()
    Pf = _clone_params(params, n_p=n_p, n_d=n_d)
    taus = np.array([r["tau"] for r in records])
    xs = np.array([r["x"] for r in records])
    out = shooting_batch(taus, Pf, x0=xs, tol=1e-9, max_iter=60)
    conv = out["converged"]
    if verbose:
        print(f"  细化到 n_p=n_d={n_p}: 收敛 {conv.sum()}/{len(records)} "
              f"({conv.mean()*100:.1f}%), 迭代中位 {int(np.median(out['n_iter']))}")
    _, _, curves = forward_batch(out["x"], taus, Pf, want_curve=True)
    Pc, S, Uc, Cc, Rc, Nc, Vv = curves
    Lgeo = _wire_geom_len_batch(Pc, Rc)
    dL = (Lgeo - _L0_wire()) * 1000.0
    n_drop = 0
    for i, r in enumerate(records):
        if not conv[i] or not np.isfinite(Pc[i, -1]).all():
            n_drop += 1
            continue
        r["x"] = out["x"][i].copy()
        r["p_tip"] = Pc[i, -1].copy()
        r["R_tip"] = Rc[i, -1].copy()
        r["kcoef"] = _kappa_nodes(S, Cc[i], Pf, Vv[i])
        r["res"] = float(np.linalg.norm(out["res"][i]))
        r["dL_mm"] = dL[i]
        r["converged"] = True
    if verbose and n_drop:
        print(f"  ⚠ {n_drop} 条未收敛/非有限, 保持原值")
    return records, Pf


def record_from_batch(i, tau, params, out, curves):
    """由批量解的第 i 个样本打包记录 (含 κ 系数, 不做几何长积分)."""
    Pc, S, Uc, Cc, Rc, Nc, Vv = curves
    return {
        "tau": np.asarray(tau, float),
        "dL_mm": None,                       # 建表后由 _fill_dL 统一算
        "p_tip": Pc[i, -1].copy(),
        "R_tip": Rc[i, -1].copy(),
        "kcoef": _kappa_nodes(S, Cc[i], params, Vv[i]),
        "res": float(np.linalg.norm(out["res"][i])),
        "converged": bool(out["converged"][i]),
        "x": out["x"][i].copy(),
    }


def build_table(N=2000, tau_max=8.0, params=None, seed=0,
                verbose=True, log_every=None, batch_size=256,
                continuation=None, taus=None):
    """§10.2 建表: τ 空间 LHS 采样 + 批量 VC 打靶 + 曲线自洽 ΔL.

    continuation: 如 (0.5, 0.75, 1.0) —— **延拓法**. 高 τmax 时批量打靶
        从直臂初值收敛率会崩 (τmax>20 后 <15%): 多平衡态 + 刚性, 牛顿找不到
        弯曲分支。先把同一组采样方向按比例缩小求解, 再逐级放大并用上一级的
        x 热启动。
        ★ 阶梯**粒度**比级数更关键: 实测 τmax=30 时 4 级只有 23% 收敛, 16 级到
        87.5%, 而耗时几乎不变 (246s→389s/120点)。高 τmax 用 tuple(linspace(0.1,1.0,16))。

    taus: (M,6) 直接给定要解的 τ, 跳过 LHS 采样。用于定向加密 (如深弯区补点)。

    未收敛样本走 batch 多起点重试, 仍失败的丢弃。
    """
    params = params or default_params()
    if taus is not None:                          # 直接给定 τ (定向加密)
        tau_all = np.asarray(taus, float)
        N = len(tau_all)
    else:
        u = lhs(N, 6, seed=seed)                  # 方向 ∈[0,1]^6
        tau_all = u * tau_max
    # 注: space_filling_order(u) 排序此处**不适用** —— 建表每批独立打靶, 批内顺序
    # 不影响结果。Z 序只在"逐点热启动链"里才有用 (见 space_filling_order 注释)。

    records = [None] * N
    if log_every is None:
        log_every = max(1, N // 10)
    n_done = 0
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        idx = np.arange(s, e)
        tb = tau_all[idx]
        # ---- 延拓 ----
        # continuation=None 时不做阶梯放大, 直接解 tb —— 配合 taus= 给出的
        # "已延拓过的 τ" 用 (调用方自己滚过阶梯, 每点都带着热启动)。
        if continuation:
            x_hot = None
            for sc in continuation:
                out = shooting_batch(tb * sc, params, tol=1e-8,
                                     max_iter=(15 if sc < 1.0 else 30), x0=x_hot)
                x_hot = out["x"].copy()
        else:
            out = shooting_batch(tb, params, tol=1e-8, max_iter=40)
        # ---- 批量多起点兜底 ----
        # 未收敛样本换初值【并行】重跑 (shooting_batch_retry), 不用标量 shooting 逐点兜。
        # 实测 τmax=40 时逐点标量兜底占了大头: 80 点 16.3 s/点 (其中 59 点走兜底);
        # 换成批量重试后同样的活按批摊薄。仍未收敛的丢给下面的 `keep` 过滤。
        n_cold = n_drop = 0
        keep = out["converged"].copy()
        if not keep.all():
            bad = np.where(~keep)[0]
            sub = shooting_batch_retry(tb[bad], params, tol=1e-8,
                                       max_iter=40, n_try=4)
            for jj, j in enumerate(bad):
                if sub["converged"][jj]:
                    out["x"][j] = sub["x"][jj]
                    out["res"][j] = sub["res"][jj]
                    out["converged"][j] = True
                    keep[j] = True
                    n_cold += 1
                else:
                    n_drop += 1
        # ⚠️ 曲线必须在兜底【之后】用最终 x 重算 —— 否则 p_tip/R_tip/kcoef
        # 会与 x 不一致 (2026-09-15 修过一次这类 bug, 影响 13% 记录)
        out["x"][~keep] = 0.0                     # 丢弃样本给无害值, 防 NaN 传播
        _, _, curves = forward_batch(out["x"], tb, params, want_curve=True)
        for j, i in enumerate(idx):
            if not keep[j]:
                continue
            rec = record_from_batch(j, tau_all[i], params, out, curves)
            rec["converged"] = True
            if not np.isfinite(rec["p_tip"]).all():
                n_drop += 1
                continue
            records[i] = rec
        n_done += len(idx)
        if verbose and (n_done % log_every < len(idx) or n_done == N):
            done = [r for r in records if r is not None]
            tips = np.array([r["p_tip"] for r in done])
            print(f"  [{n_done:>5}/{N}] 入库 {len(done)} 兜底 {n_cold} 丢弃 {n_drop}"
                  f" | tip x∈[{tips[:,0].min():+.3f},{tips[:,0].max():+.3f}] "
                  f"z∈[{tips[:,2].min():+.3f},{tips[:,2].max():+.3f}]")
    records = [r for r in records if r is not None]
    _fill_dL(records, params)
    return records


def repair_records(records, params=None):
    """★ 原地修复 p_tip/R_tip/kcoef, 使之与记录的 x 自洽.

    背景: 早期 build_table 里冷启动兜底替换了 out["x"][j] 但复用了替换前
    算的 curves ⇒ 约 11% 记录的 p_tip/R_tip/κ系数 与其 x 不符 (dL/res 是对的,
    因为它们后来从 x 重算)。x 本身正确, 所以只需按 x 重算形状量, 不必重建表。
    """
    from tendon_coupling import forward_batch
    params = params or default_params()
    TAU = np.array([r["tau"] for r in records])
    X = np.array([r["x"] for r in records])
    _, _, cv = forward_batch(X, TAU, params, want_curve=True)
    Pc, S, Uc, Cc, Rc, Nc, Vv = cv
    n_fix = 0
    for i, r in enumerate(records):
        dp = np.linalg.norm(r["p_tip"] - Pc[i, -1])
        if dp > 1e-6:                       # 与 x 不自洽 → 修复
            n_fix += 1
        r["p_tip"] = Pc[i, -1].copy()
        r["R_tip"] = Rc[i, -1].copy()
        r["kcoef"] = _kappa_nodes(S, Cc[i], params, Vv[i])
    return n_fix


def _fill_dL(records, params):
    """由记录存的打靶解重解曲线, 积分几何丝长 → ΔL_mm (同源, 全批量)."""
    taus = np.array([r["tau"] for r in records])
    xs = np.array([r["x"] for r in records])
    _, _, curves = forward_batch(xs, taus, params, want_curve=True)
    Lgeo = _wire_geom_len_batch(curves[0], curves[4])       # (N,6)
    dL = (Lgeo - _L0_wire()) * 1000.0
    for i, r in enumerate(records):
        r["dL_mm"] = dL[i]


def _wire_geom_len_batch(Pc, Rc):
    """批量算 6 丝空间路径长 (与 sdm_vc_disp.tendon_path_length 一致).

    Pc (N, n+1, 3), Rc (N, n+1, 3, 3) → (N, 6)。
    """
    N, n1, _ = Pc.shape
    nsub = n1 - 1
    nA = int(round(L1 / (L / nsub)))
    a = np.radians(ALPHA_DEG)
    o = np.stack([0.030 * np.cos(a), 0.030 * np.sin(a), np.zeros(6)], axis=1)  # (6,3)
    off = np.einsum('nsij,wj->nswi', Rc, o)        # (N, n+1, 6, 3) 截面偏移随姿态转
    q = Pc[:, :, None, :] + off                    # (N, n+1, 6, 3) 丝节点空间位置
    seg = np.linalg.norm(np.diff(q, axis=1), axis=3)   # (N, n, 6) 每小段长
    out = np.zeros((N, 6))
    for w in range(6):
        end = nA if w in SEG_A else nsub
        out[:, w] = seg[:, :end, w].sum(axis=1)
    return out


def _wire_geom_len(p_curve, R_curve):
    """单条 (保留作参考/对拍用)."""
    return _wire_geom_len_batch(p_curve[None], R_curve[None])[0]


# ==================== 6D 白化度量 (§10.5 要点2) ====================

class Metric:
    """q=[σ_p·p(mm), w_f·rotvec(mrad)] 的 6 维白化度量 (§10.5).

    位置用 mm、姿态用 mrad; w_f 由"位置误差 ≃ 姿态误差"等效关系定
    (默认 1 mm ≃ 0.5 mrad ⇒ w_f = 2 mm/mrad)。
    """

    def __init__(self, sigma_p_mm=5.0, mm_per_mrad=0.5):
        self.sp = float(sigma_p_mm)
        self.wf = float(sigma_p_mm) / float(mm_per_mrad)

    def encode(self, p_tip_m, R_tip):
        p_mm = np.asarray(p_tip_m, float) * 1000.0
        return np.concatenate([self.sp * p_mm, self.wf * rotvec_mrad(R_tip)])


# ==================== 表 + 查询 (§10.5) ====================

class PoseTable:
    """建表结果 + 索引 + 查询 (k近邻 + 插值 + 分支上报 + 可选精化 + 惰性补洞)."""

    def __init__(self, records, params=None, metric=None, F_tip=(0, 0, 0),
                 stored_version=None):
        self.recs = list(records)
        self.params = params or default_params()
        self.metric = metric or Metric()
        self.F_tip = tuple(np.ravel(F_tip))
        self.version = table_version_key(self.params, F_tip=self.F_tip)
        # 建表时写进 npz 的版本键 (旧表为 None)。校验必须拿它比, 不能拿上面的
        # self.version —— 那是"用当前代码现算", 自己跟自己比, 恒等 (踩过)。
        self.stored_version = stored_version
        self.Q = np.array([self.metric.encode(r["p_tip"], r["R_tip"])
                           for r in self.recs])
        self.tree = cKDTree(self.Q) if HAVE_SCIPY else None
        self._dL_prev = None                      # 连续性优先的上一帧指令
        self._q_prev = None                       # 上一帧查询(白化空间), 用于判断是否"逐帧"
        self.cont_radius = 3.0                    # 连续性仅在 |Δq|<该值时启用 (≈15mm)
        self._x_prev = None                       # 惰性补洞的内层热启动
        self.n_infill = 0
        self.n_query = 0

    # ---- 选解 (§10.3): 连续性优先 → min‖τ‖ ----
    def _select(self, idxs, q=None):
        """从 k 近邻中选一个解.

        ⚠ 关键: 无历史连续性时**应取最近点**, 而不是 min‖τ‖ ——
          后者会挑到邻域里另一个记录, 其 τ/dΔL 与查询位姿无关
          (实测: 精确表点查询却给出 6.79 mm 的热启动偏差, 直接拖慢精化)。
        """
        cand = list(idxs)
        # ⚠ 连续性分支**只能用于真正的"逐帧跟踪"**: 必须同时满足
        #   ① 有上一帧 ΔL ② 有上一帧查询位置 ③ 本次查询离上一帧足够近。
        #   否则(随机访问/大跳变)会挑到邻域里 ΔL 最接近的**另一个分支**,
        #   实测偏差可达 42 N(τ) —— 这正是"精确表点查询却给 6.79mm 热启动"的根因。
        cont = (self._dL_prev is not None and q is not None
                and self._q_prev is not None
                and float(np.linalg.norm(q - self._q_prev)) < self.cont_radius)
        if cont:
            return min((self.recs[i] for i in cand),
                       key=lambda r: np.linalg.norm(r["dL_mm"] - self._dL_prev))
        if q is not None:                       # 最近点优先
            return self.recs[min(cand, key=lambda i: float(np.linalg.norm(self.Q[i] - q)))]
        return min((self.recs[i] for i in cand), key=lambda r: np.linalg.norm(r["tau"]))

    @staticmethod
    def _spread(idxs, recs):
        if len(idxs) < 2:
            return 0.0
        return float(np.max(np.std(np.array([recs[i]["dL_mm"] for i in idxs]), axis=0)))

    # ---- ΔL 的局部线性插值 (§10.5 步骤4') ----
    def _interp_dL(self, q, idxs, k=8, ridge=1e-6):
        """在最近 k 个表点上把 **ΔL(q')** 局部线性拟合, 取常数项作为 q 处的 ΔL.

        为什么插值 ΔL 而不是 τ: τ 是**输入**, 位姿是输出, 映射多对一 —— 相邻位姿
        的 τ 可以差很多, 插值 τ 病态 (实测只比最近邻好 3%)。ΔL 是下发量、随位姿
        光滑变化, 插值它才有效。

        ⚠ **必须是"过点插值"**: 纯局部线性拟合在表点处也偏 3~7 mm (实测)
          ⇒ 取拟合值后用 **Shepard(IDW) 残差修正**, 保证穿过表点。

        实验对比 (977 点, 半表留出, k=12):
            0阶 最近邻      ΔL 残差中位 5.11 mm
            1阶 插值 ΔL      ΔL 残差中位 **1.12 mm**   ← 4.6x, 不增加任何表点
            2阶 二次        崩 (29.5mm): 邻域交叉项病态。**别上二阶。**
        """
        cand = sorted(idxs, key=lambda i: np.linalg.norm(self.Q[i] - q))[:k]
        if not cand:
            return None
        if len(cand) < 4:
            return np.asarray(self.recs[cand[0]]["dL_mm"], float)
        dq = self.Q[cand] - q                                   # (k,6)
        B = np.array([self.recs[i]["dL_mm"] for i in cand])     # (k,6)
        nrm = np.linalg.norm(dq, axis=1)                        # 到各候选的加权距离
        # ── ① 精确命中: 最近点距离≈0 ⇒ 直接返回该点 ──
        if nrm.min() < 1e-9:
            return np.asarray(self.recs[cand[int(np.argmin(nrm))]]["dL_mm"], float)
        A = np.hstack([np.ones((len(cand), 1)), dq])            # (k,7)
        w = 1.0 / (nrm + 1.0)                                   # 近点权重
        try:
            coef, *_ = np.linalg.lstsq(A * w[:, None], B * w[:, None], rcond=ridge)
        except np.linalg.LinAlgError:
            return np.asarray(self.recs[cand[0]]["dL_mm"], float)
        d = np.asarray(coef[0], float)
        if not np.isfinite(d).all():
            return np.asarray(self.recs[cand[0]]["dL_mm"], float)
        # ── ② Shepard 残差修正: 让插值**穿过数据点** ──
        #   原实现只取局部线性拟合的常数项 = 邻域折中值, **即使查询点恰是表点
        #   也会偏 3~7 mm** (实测 #300 9.96mm / #500 3.34mm)。加 IDW 残差修正后
        #   在表点上误差 → 0, 且远离表点时平滑退化为原线性模型。
        pred = coef[0][None, :] + dq @ coef[1:]                 # (k,6) 线性预测
        r = B - pred                                            # 残差
        ww = 1.0 / (nrm ** 2 + 1e-12)                           # Shepard 权重 (p=2)
        d = d + (ww[:, None] * r).sum(axis=0) / ww.sum()
        return d

    # ---- 查询主入口 ----
    def query(self, pose, pos_tol_mm=50.0, branch_tol_mm=0.5, allow_infill=True,
              verbose=False, interp=True, k_interp=8, refine=0):
        """§10.5 查询. pose=(p_tip_m(3), R_tip(3x3)).

        **覆盖半径**: pos_tol_mm 是"最近表点允许的最大位置距离"(mm), 不是原来的
        半径搜索容差。原先的半径搜索按 pos_tol=5mm 展开到 150σ, 但本表 NN 中位
        是 619σ(≈44mm), 永远命中不了 → 每个查询都退化成惰性补洞(慢)。改为
        **k 近邻检索**: 取最近的 k 个点, 若最近点超出覆盖半径则判未覆盖。

        interp=True: 用这 k 个点的 **ΔL** 做局部线性插值 (1阶, 实测比最近邻选点好
            ~4.3x; 注意插 ΔL 不是插 τ —— τ 是输入, 插值病态)。
        interp=False: 退回最近邻选点 (0阶)。
        refine>0: 插值后追加 refine 步外层牛顿 (默认关, 实测收益很小)。
        超出覆盖半径时统一走 `infill()` 惰性补洞 (受 allow_infill 控制)。
        """
        self.n_query += 1
        p_tip, R_tip = pose
        q = self.metric.encode(p_tip, R_tip)
        K = max(int(k_interp), 4)
        n_jac = int(refine)

        if self.tree is not None:
            d, idx = self.tree.query(q, k=K)
        else:
            d = np.linalg.norm(self.Q - q, axis=1)
            idx = np.argsort(d)[:K]
            d = d[idx]
        idxs = [int(i) for i in np.atleast_1d(idx)]
        d = np.atleast_1d(d)
        nn_mm = float(np.linalg.norm(
            self.recs[idxs[0]]["p_tip"] - np.asarray(p_tip)) * 1000)

        if nn_mm > pos_tol_mm:                       # 超出覆盖半径
            if allow_infill:
                if verbose:
                    print(f"    最近点 {nn_mm:.1f}mm > 覆盖半径 → 现场牛顿补洞")
                rec, ok = self.infill(p_tip, R_tip)
                if not ok:
                    return self._unreachable()
                return self._pack(rec, hit=False, branch=False, refined=False,
                                  n_eval=0, target=(p_tip, R_tip))
            return self._unreachable()

        spread = self._spread(idxs, self.recs)             # 分支判据 (仅上报)
        branch = spread > branch_tol_mm
        base = self._select(idxs, q)                          # 选解 (回退/形状用)
        refined = False
        # 注: 分支判据**不**能拿来跳过插值 —— 本表 NN 中位 619σ, k 邻域的 dL 离散度
        # 天然就 > branch_tol, 一旦拿它当闸门插值永远不触发 (实测退化成 0 阶)。
        # 分支只作为信息上报, 由上层决定是否采信。
        if interp:
            # 步骤4': 邻域 ΔL 线性插值 (1阶) —— 精度主贡献, 不增加表点
            d_i = self._interp_dL(q, idxs, k=K)
            if d_i is not None:
                base = dict(base)                          # 不污染表内记录
                base["dL_mm"] = d_i
                base["interpolated"] = True
        if n_jac > 0 and not branch:                       # 可选: 再牛顿精化
            tau, sol, info = solve_pose_multi(
                p_tip, R_tip, self.params, F_tip=self.F_tip,
                tau0=base["tau"], x0=base["x"], max_iter=n_jac)
            if info["converged"] and sol is not None and \
                    np.isfinite(sol["p_curve"][-1]).all():
                base = rec_from_sol(tau, sol, self.params)
                refined = True
        self._dL_prev = np.asarray(base["dL_mm"]).copy()
        self._q_prev = np.asarray(q, float).copy()
        nn_rec, _ = self.nearest(p_tip, R_tip)
        return self._pack(base, hit=True, branch=branch, refined=refined,
                          n_eval=len(idxs), target=(p_tip, R_tip), ref=nn_rec)

    # ---- 惰性补洞 (§10.1/§10.5 步骤3) ----
    def nearest(self, p_tip, R_tip):
        """最近表点 (不受半径限制) → (记录, 白化距离). 用于选解/补洞热启动."""
        q = self.metric.encode(p_tip, R_tip)
        if self.tree is not None:
            d, i = self.tree.query(q, k=1)
            return self.recs[int(i)], float(d)
        d = np.linalg.norm(self.Q - q, axis=1)
        i = int(np.argmin(d))
        return self.recs[i], float(d[i])

    def infill(self, p_tip, R_tip):
        """现场外层牛顿补洞. 关键: 用最近表点的 τ 与 x 热启动 (冷启动 τ=0 不收敛)."""
        near, _ = self.nearest(p_tip, R_tip)
        tau, sol, info = solve_pose_multi(p_tip, R_tip, self.params,
                                          F_tip=self.F_tip,
                                          tau0=near["tau"], x0=near["x"])
        if sol is None or not np.isfinite(sol["p_curve"][-1]).all():
            return None, False
        rec = rec_from_sol(tau, sol, self.params)
        rec["infilled"] = True
        rec["pos_res_mm"] = info.get("pos_res_mm", np.nan)   # 到目标位姿的距离
        rec["exact"] = bool(info["converged"])               # 是否精确命中目标
        self._insert(rec)
        self.n_infill += 1
        self._dL_prev = rec["dL_mm"].copy()
        return rec, True                                     # ok=流形最近点也可达

    def _insert(self, rec):
        self.recs.append(rec)
        self._x_prev = rec["x"]
        qv = self.metric.encode(rec["p_tip"], rec["R_tip"])[None, :]
        self.Q = np.vstack([self.Q, qv])
        if HAVE_SCIPY:
            self.tree = cKDTree(self.Q)                     # 懒插入: 重建索引

    @staticmethod
    def _pack(rec, hit, branch, refined, n_eval, target=None, ref=None):
        out = {"hit": hit, "branch": branch, "refined": refined,
               "reachable": True, "n_eval": n_eval,
               "interpolated": bool(rec.get("interpolated", False)),
               "dL_mm": np.asarray(rec["dL_mm"]).copy(),
               "tau": np.asarray(rec["tau"]).copy(),
               "p_tip": np.asarray(rec["p_tip"]),
               "R_tip": np.asarray(rec["R_tip"]),
               "kcoef": rec["kcoef"], "res": rec.get("res", np.nan),
               "exact": bool(rec.get("exact", hit))}
        if target is not None:
            # ⚠ 残差必须用**最近表点**(ref)量, 不能用按连续性挑出来的 rec ——
            #   后者可能离查询位姿很远, 会把"覆盖距离"误报成"解误差"(实测虚高 ~14mm)。
            r0 = ref if ref is not None else rec
            p0 = np.asarray(r0["p_tip"]); R0 = np.asarray(r0["R_tip"])
            tp, tR = target
            out["nn_pos_mm"] = float(np.linalg.norm(p0 - np.asarray(tp)) * 1000)
            out["nn_rot_mrad"] = float(np.linalg.norm(
                so3_log(np.asarray(tR).T @ R0)) * 1000)
            out["pos_res_mm"] = out["nn_pos_mm"]          # 覆盖距离 (诚实口径)
            out["rot_res_mrad"] = out["nn_rot_mrad"]
            out["p_tip"] = p0
            out["R_tip"] = R0
        return out

    @staticmethod
    def _unreachable():
        """§10.5 步骤7: 不可达/超行程 —— 返回统一 schema 的空结果."""
        return {"hit": False, "branch": False, "refined": False,
                "reachable": False, "exact": False, "n_eval": 0,
                "dL_mm": np.zeros(6), "tau": np.zeros(6),
                "p_tip": np.full(3, np.nan), "R_tip": np.eye(3),
                "kcoef": None, "res": np.nan,
                "pos_res_mm": np.inf, "rot_res_mrad": np.inf}


# ==================== 外层位姿牛顿 (§10.1, 惰性补洞) ====================

def _pose_err(tips, RL, p_des, R_des):
    """(N,3) tip + (N,3,3) R(L) → (N,6) 位姿残差 [Δr, log(Rᵀ_des R(L))]."""
    N = len(tips)
    E = np.zeros((N, 6))
    E[:, :3] = tips - np.asarray(p_des, float)
    for i in range(N):
        E[i, 3:] = so3_log(np.asarray(R_des, float).T @ RL[i])
    return E


def _pose_err_batch(tips, RL, p_des, R_des):
    """批量版: tips (N,3), RL (N,3,3), p_des (N,3), R_des (N,3,3) → (N,6).

    姿态残差用 so3_log(R_desᵀ·R_L)。p_des/R_des 既可是 (N,...) 也可广播单条。
    """
    tips = np.asarray(tips, float)
    RL = np.asarray(RL, float)
    p_des = np.asarray(p_des, float)
    R_des = np.asarray(R_des, float)
    N = len(tips)
    if p_des.ndim == 1:
        p_des = np.broadcast_to(p_des, (N, 3))
    if R_des.ndim == 2:
        R_des = np.broadcast_to(R_des, (N, 3, 3))
    M = np.einsum('nji,njk->nik', R_des, RL)          # R_desᵀ @ R_L
    E = np.empty((N, 6))
    E[:, :3] = tips - p_des
    for i in range(N):
        E[i, 3:] = so3_log(M[i])
    return E


def solve_pose_multi(p_des, R_des, params, F_tip=(0, 0, 0), tau0=None, x0=None,
                     max_iter=15, tol_pos_mm=0.2, tol_rot_mrad=1.0, h=0.05,
                     ret_sol=True):
    """§10.1 外层牛顿: pose → τ∈R⁶ (内层 VC 打靶). 仅用于惰性补洞.

    e(τ)=[r(L)−p*, log(R*ᵀR(L))]=0 (6 方程 6 未知). 雅可比用**批量打靶**:
    一步把 13 个 τ(中心 + 6×±h)并行解, 而非 13 次标量求解 (~13× 加速)。

    收敛判据用**物理量纲**(不是混合范数): 位置 < tol_pos_mm 且姿态 < tol_rot_mrad;
    另接受"步长已停滞 + 残差不大"的解 —— 目标位姿可能落在可达流形之外(如两表点
    中点), 此时返回流形上最近解而非判不可达。
    """
    p_des = np.asarray(p_des, float)
    R_des = np.asarray(R_des, float)
    tau = np.zeros(6) if tau0 is None else np.array(tau0, float)
    xh = None if x0 is None else np.array(x0, float)
    converged = False
    it = 0
    e = np.full(6, np.inf)
    tpos = tol_pos_mm * 1e-3                    # m
    trot = tol_rot_mrad * 1e-3                  # rad
    for it in range(max_iter):
        T = np.vstack([tau, tau + h * np.eye(6), tau - h * np.eye(6)])
        T = np.clip(T, 0.0, 200.0)
        x0b = None if xh is None else np.tile(xh, (len(T), 1))
        out = shooting_batch(T, params, x0=x0b, tol=1e-9, max_iter=20)
        _, _, curves = forward_batch(out["x"], T, params, want_curve=True)
        E = _pose_err(out["tip"], curves[4][:, -1], p_des, R_des)
        e = E[0]
        if xh is None:
            xh = out["x"][0].copy()
        J = np.zeros((6, 6))
        for c in range(6):
            J[:, c] = (E[1 + c] - E[7 + c]) / (2 * h)
        J = np.where(np.isfinite(J), J, 0.0)
        lam = 1e-3
        try:
            dy = -np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ e)
        except np.linalg.LinAlgError:
            dy = -np.linalg.pinv(J.T @ J + lam * np.eye(6)) @ (J.T @ e)
        if not np.isfinite(dy).all():
            dy = np.zeros(6)
        if np.max(np.abs(dy)) > 1.0:
            dy *= 1.0 / np.max(np.abs(dy))
        # 物理收敛判据
        if np.linalg.norm(e[:3]) < tpos and np.linalg.norm(e[3:]) < trot:
            converged = True
            break
        if np.max(np.abs(dy)) < 1e-4:           # 停滞: 已在流形最近点
            break
        tau = np.clip(tau + dy, 0.0, 200.0)     # §10.5 步骤7: 非负
    sol = None
    if ret_sol:
        try:
            sol = shooting(tau, params, F_tip=F_tip, g_world=(0, 0, -9.81),
                           tol=1e-8, max_iter=60, x0=xh)
        except Exception:
            sol = None
    # 流形最近点也视为可达 (残差随结果返回, 由上层判断)
    ok = sol is not None and np.isfinite(sol["p_curve"][-1]).all()
    return tau, sol, {"converged": converged, "n_iter": it + 1,
                      "resid": float(np.linalg.norm(e)),
                      "pos_res_mm": float(np.linalg.norm(e[:3]) * 1000),
                      "rot_res_mrad": float(np.linalg.norm(e[3:]) * 1000),
                      "ok": bool(ok)}


def solve_pose_batch(p_des, R_des, params, tau0=None, x0=None, max_iter=12,
                     tol_pos_mm=0.2, tol_rot_mrad=1.0, h=0.05, chunk=500,
                     verbose=False):
    """§10.1 批量外层牛顿: N 个目标位姿 **同时** 反解 τ∈R⁶.

    `solve_pose_multi` 是单姿态版, 每点 >100s (n=300), 按位姿采样时不可用
    (20k 点 = 23 天)。这里把外层牛顿也批量化:
      每轮对 N 个样本各构造 13 个 τ (中心 + 6×±h) → 一次 shooting_batch,
      再按样本切出 6×6 雅可比 → 批量解 Levenberg 步。
    内层打靶本身已是批量/向量化的, 所以整体摊薄成本随 N 下降。

    热启动至关重要: 给最近邻表点的 (tau0, x0), 从 τ=0 冷启动不收敛。

    返回 dict: tau(N,6), x(N,6), tip(N,3), converged(N,), n_iter(N,),
               pos_res_mm(N,), rot_res_mrad(N,), tip_all(用于调试)
    """
    p_des = np.atleast_2d(np.asarray(p_des, float))
    R_des = np.asarray(R_des, float)
    if R_des.ndim == 2:
        R_des = R_des[None]
    N = len(p_des)
    tpos = tol_pos_mm * 1e-3
    trot = tol_rot_mrad * 1e-3

    tau = np.zeros((N, 6)) if tau0 is None else np.array(tau0, float).copy()
    xh = None if x0 is None else np.array(x0, float).copy()
    conv = np.zeros(N, bool)
    n_it = np.zeros(N, int)
    pos_res = np.full(N, np.inf)
    rot_res = np.full(N, np.inf)
    tip_out = np.full((N, 3), np.nan)
    R_out = np.tile(np.eye(3), (N, 1, 1))
    alive = np.ones(N, bool)

    off = np.array([0] + [1 + c for c in range(6)] + [7 + c for c in range(6)])
    J_ = np.arange(6)

    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        sl = slice(c0, c1)
        m = c1 - c0
        tau_c = tau[sl]
        xh_c = None if xh is None else xh[sl]
        p_c = p_des[sl]
        R_c = R_des[sl]
        done = np.zeros(m, bool)
        for it in range(max_iter):
            # 13 个 τ 行/样本, 顺序 [中心, +h*e0..5, -h*e0..5]
            T = np.concatenate(
                [tau_c[:, None, :],
                 tau_c[:, None, :] + h * np.eye(6)[None, :, :],
                 tau_c[:, None, :] - h * np.eye(6)[None, :, :]], axis=1
            ).reshape(-1, 6)
            T = np.clip(T, 0.0, 200.0)
            x0b = None if xh_c is None else np.repeat(xh_c, 13, axis=0)
            out = shooting_batch(T, params, x0=x0b, tol=1e-9, max_iter=15)
            _, _, curves = forward_batch(out["x"], T, params, want_curve=True)
            pr = np.repeat(p_c, 13, axis=0)
            Rr = np.repeat(R_c, 13, axis=0)
            E = _pose_err_batch(out["tip"], curves[4][:, -1], pr, Rr).reshape(m, 13, 6)
            e0 = E[:, 0, :]
            J = (E[:, 1:7, :] - E[:, 7:13, :]) / (2 * h)      # (m,6,6) [dE_c/dτ_c]
            J = np.transpose(J, (0, 2, 1))
            J = np.where(np.isfinite(J), J, 0.0)
            if xh_c is None:
                xh_c = out["x"].reshape(m, 13, 6)[:, 0, :].copy()
            else:
                xh_c = out["x"].reshape(m, 13, 6)[:, 0, :].copy()
            # 记录中心点的物理残差
            pos_res[c0:c1] = np.linalg.norm(e0[:, :3], axis=1) * 1000
            rot_res[c0:c1] = np.linalg.norm(e0[:, 3:], axis=1) * 1000
            tip_out[c0:c1] = out["tip"].reshape(m, 13, 3)[:, 0, :]
            R_out[c0:c1] = curves[4][:, -1].reshape(m, 13, 3, 3)[:, 0, :, :]
            now_conv = (pos_res[c0:c1] < tol_pos_mm) & (rot_res[c0:c1] < tol_rot_mrad)
            newc = now_conv & ~done
            n_it[c0:c1] = np.where(newc, it, n_it[c0:c1])
            done = done | now_conv
            if done.all():
                break
            # 批量 Levenberg 步
            lam = 1e-3
            H = np.einsum('nki,nkj->nij', J, J) + lam * np.eye(6)[None]
            g = np.einsum('nki,nk->ni', J, e0)
            try:
                dy = -np.linalg.solve(H, g[..., None])[..., 0]
            except np.linalg.LinAlgError:
                dy = -np.einsum('nij,nj->ni', np.linalg.pinv(H), g)
            dy = np.where(np.isfinite(dy), dy, 0.0)
            why = np.linalg.norm(dy, axis=1)
            dy[why > 1.0] *= (1.0 / why[why > 1.0])[:, None]
            stalled = why < 1e-4
            donow = done & ~stalled                       # 停滞视为到位 (流形最近点)
            dy[donow] = 0.0
            tau_c = np.clip(tau_c + dy, 0.0, 200.0)
            n_it[c0:c1] = np.where(stalled & ~done, it, n_it[c0:c1])
            done = done | stalled
            if done.all():
                break
        tau[sl] = tau_c
        if xh is not None:
            xh[sl] = xh_c
        conv[sl] = done
        if verbose:
            print(f"    批量牛顿 [{c1:>6}/{N}] 命中 {conv.sum()} "
                  f"pos残差中位 {np.median(pos_res[c0:c1]):.2f} mm")

    return {"tau": tau, "x": xh if xh is not None else np.zeros((N, 6)),
            "tip": tip_out, "R_tip": R_out, "converged": conv, "n_iter": n_it,
            "pos_res_mm": pos_res, "rot_res_mrad": rot_res}


# ==================== 显示: 由 κ(s) 积分重建骨架 (§10.4/§10.8) ====================

def reconstruct_shape(kcoef, params, s=None, n=600):
    """由 κ(s)/v(s) 节点表 RK2 积分重建骨架 p(s),R(s) (显示/下发用).

    p' = R·v, 其中 v 是线应变场 (v_z=1+ν_z 含轴向压缩)。**必须用 v, 不能当 e₃**:
    臂体轴向刚度 `k_rod` 有限时 v_z 沿弧长实测 0.85~0.90, 只用 κ 会让骨架
    凭空长 40~58mm。旧表无 "V" 字段 → 自动退化为 v=e₃ (旧行为)。

    n 默认 600 (步长 0.75mm): 与打靶网格 ~7.5mm 相比足够细, 积分本身不再引入
    毫米级误差。κ 在细网格上批量插值, 避免逐点 np.interp 的 Python 开销。
    """
    if s is None:
        s = np.linspace(0, L, n)
    ss = np.asarray(s, float)
    sm = 0.5 * (ss[1:] + ss[:-1])                       # 中点
    KM = kappa_at_batch(kcoef, sm)                      # (n-1, 3)
    VM = vz_at_batch(kcoef, sm)                         # (n-1, 3) 线应变 (旧表=e₃)
    ds = np.diff(ss)
    p = np.zeros(3); R = np.eye(3)
    ps = [p.copy()]; Rs = [R.copy()]
    for i in range(len(ds)):
        km = KM[i]
        p = p + ds[i] * ((R @ so3_exp(0.5 * ds[i] * km)) @ VM[i])
        R = R @ so3_exp(ds[i] * km)
        U, _, Vt = np.linalg.svd(R); R = U @ Vt
        ps.append(p.copy()); Rs.append(R.copy())
    return np.array(ps), np.array(Rs)


# ==================== 下发前安全检查 (§10.7) ====================

def apply_safety(dL_mm, tau=None, dL_max_mm=60.0, dL_min_mm=-60.0,
                 tau_min_N=0.1):
    """§10.7 下发前: 行程限位 + 预紧下限校验, 越界截断并上报.

    返回 (dL_safe, flags)：flags dict 含 clipped(截断的丝), slack(张力→0 松弛的丝),
    ok(全部合规)。丝松弛 (τ≤tau_min) 会让该丝失去约束, 需上报而非静默下发。
    """
    dL = np.asarray(dL_mm, float).copy()
    flags = {"clipped": [], "slack": [], "ok": True}
    hi = dL > dL_max_mm
    lo = dL < dL_min_mm
    if hi.any() or lo.any():
        flags["clipped"] = sorted(set(np.where(hi)[0].tolist() +
                                      np.where(lo)[0].tolist()))
        dL = np.clip(dL, dL_min_mm, dL_max_mm)
        flags["ok"] = False
    if tau is not None:
        slack = np.where(np.asarray(tau, float) <= tau_min_N)[0]
        if len(slack):
            flags["slack"] = slack.tolist()
            flags["ok"] = False
    return dL, flags


# ==================== 存取 (§10.9 规模) ====================

def save_table(path, records, params=None, meta=None):
    K = np.array([r["kcoef"]["K"] for r in records])          # (N, m, 3)
    S = np.array([r["kcoef"]["S"] for r in records])          # (N, m)
    # v(s) 线应变场: 重建骨架时 p'=R·v 必须用它 (k_rod 有限时 v_z≠1)
    has_v = all("V" in r["kcoef"] for r in records)
    extra = {}
    if has_v:
        extra["kV"] = np.array([r["kcoef"]["V"] for r in records])   # (N, m, 3)
    np.savez_compressed(
        path,
        tau=np.array([r["tau"] for r in records]),
        dL_mm=np.array([r["dL_mm"] for r in records]),
        p_tip=np.array([r["p_tip"] for r in records]),
        R_tip=np.array([r["R_tip"] for r in records]),
        kS=S, kK=K,
        res=np.array([r["res"] for r in records]),
        x=np.array([r["x"] for r in records]),
        n_p=np.array((params or default_params()).n_p),
        n_d=np.array((params or default_params()).n_d),
        # ★ 版本键持久化 (键名 `ver`, 别和 `kV` 混: 后者是 v(s) 线应变场)
        #   没有它, load_table 只能拿"当前代码"现算, version_ok 恒为 True (踩过)
        ver=np.array(canonical_version_key(
            table_version_key(params or default_params()))),
        **extra,
    )
    return path


def _kappa_nodes_from_poly(ka, kb, params, m=61):
    """旧表兼容: 由分段多项式系数 (kcoef_a/kcoef_b, (3,7), normalized polyval 序)
    在 m 个节点上求值成节点表. 仅为把旧表读进来好做 refine; refine 后会用新曲线覆盖."""
    S = np.linspace(0.0, L, m)
    K = np.empty((m, 3))
    for i, s in enumerate(S):
        if s <= L1:
            x = s / L1
            C = np.array([np.polyval(ka[k], x) for k in range(3)])
            Ha = np.diag(params.Kbt(0))
        else:
            x = (s - L1) / (L - L1)
            C = np.array([np.polyval(kb[k], x) for k in range(3)])
            Ha = np.diag(params.Kbt(1))
        K[i] = C / Ha
    return {"S": S, "K": K}


def load_table(path, params=None):
    """载入 npz 表.

    ⚠ 必须**整块**取出数组再逐条索引 —— numpy 的 NpzFile 每次 __getitem__ 都会
    重新解压整个数组 (不缓存), `z["kK"][i]` 写进循环等于把 kK 解压 N 遍:
    2556 条 × 36MB 的 kK ≈ 30 GB 解压量, 直接 MemoryError (踩过)。
    """
    z = np.load(path)
    if params is None:                      # 从文件恢复网格分辨率 (版本键一致性)
        base = default_params()
        n_p = int(z["n_p"]) if "n_p" in z.files else base.n_p
        n_d = int(z["n_d"]) if "n_d" in z.files else base.n_d
        params = _clone_params(base, n_p=n_p, n_d=n_d)
    # ★ 建表时写入的版本键 (旧表没有 → None, 由调用方判"无法校验"而非"一致")
    stored_version = None
    if "ver" in z.files:
        try:
            stored_version = json.loads(str(z["ver"].item()))
        except (ValueError, AttributeError):
            stored_version = None
    node_fmt = "kS" in z.files                 # 新节点格式 vs 旧多项式格式
    TAU = z["tau"]; DL = z["dL_mm"]; PT = z["p_tip"]; RT = z["R_tip"]
    RES = z["res"]; X = z["x"]
    if node_fmt:
        KS = z["kS"]; KK = z["kK"]
        KV = z["kV"] if "kV" in z.files else None       # 新表才有 v(s)
        if KV is not None:
            kc = lambda i: {"S": KS[i], "K": KK[i], "V": KV[i]}
        else:
            kc = lambda i: {"S": KS[i], "K": KK[i]}
    else:
        KA = z["kcoef_a"]; KB = z["kcoef_b"]
        kc = lambda i: _kappa_nodes_from_poly(KA[i], KB[i], params)
    recs = [{"tau": TAU[i], "dL_mm": DL[i], "p_tip": PT[i], "R_tip": RT[i],
             "kcoef": kc(i), "res": float(RES[i]), "converged": True, "x": X[i]}
            for i in range(len(TAU))]
    z.close()
    return PoseTable(recs, params=params, stored_version=stored_version)


# ══════════════════════════════════════════════════════════════════════
# 库级建表 API: 以 **tau_max 为形参** (供 main.py / 外部脚本直接调用)
# ══════════════════════════════════════════════════════════════════════
_DEFAULT_N_BY_TAUMAX = {20: 977, 30: 1200, 40: 1200}


def build_table_by_tau_max(tau_max, N=None, tag=None, params=None,
                           batch_size=200, n_refine=300, ladder_levels=16,
                           seed=0, save=True, force=False, verbose=True):
    """★ 按张力上限 τmax 建**一张表**并存盘 —— tau_max 是形参(不是 tag 字符串).

    流程: 延拓建表(ladder_levels 级) → refine 到 n=n_refine → 自检 → 存 vc_table_<tag>.npz

    参数
    ----
    tau_max      : float            张力上限 (N)          ← 核心形参
    N            : int | None       采样点数; None → 按 τmax 取默认 (20→977, 30/40→1200, 其他→1200)
    tag          : str | None       文件名标签; None → f"{tau_max:g}"
    params       : SDMParams | None None → default_params() (分段双晶胞: 段A N_circ=6 / 段B 3)
    batch_size   : int              批量打靶样本数
    n_refine     : int              refine 目标网格 (n_p=n_d=n_refine)
    ladder_levels: int              延拓阶梯级数 (τmax≳20 时细阶梯才能收敛)
    save / force : bool            是否存盘 / 已存在时是否覆盖(False=跳过)
    返回 (path, recs, Pf); 跳过时 (path, None, None)
    """
    import os
    import time
    import numpy as np

    tau_max = float(tau_max)
    if N is None:
        N = _DEFAULT_N_BY_TAUMAX.get(int(tau_max) if tau_max == int(tau_max) else -1, 1200)
    if tag is None:
        tag = f"{tau_max:g}"
    fn = f"vc_table_{tag}.npz"
    if verbose:
        print(f"\n{'='*66}\nτmax={tau_max:.0f}N -> {fn}  (N={N})", flush=True)
    if os.path.exists(fn) and not force:
        if verbose:
            print("  已存在, 跳过 (force=True 可覆盖)", flush=True)
        return fn, None, None

    P = params if params is not None else default_params()

    def mk(nn):
        return _clone_params(P, n_p=nn, n_d=nn)

    # 细延拓: τmax≳20 时粗阶梯收敛率崩 (4 级 23% vs 16 级 87.5%), 耗时几乎不变
    ladder = tuple(np.linspace(0.1, 1.0, int(ladder_levels)))
    t0 = time.time()
    recs = build_table(N=N, tau_max=tau_max, params=mk(50), seed=seed,
                       verbose=verbose, batch_size=batch_size,
                       continuation=ladder)
    dt = time.time() - t0
    if verbose:
        print(f"  建表入库 {len(recs)}/{N}  用时 {dt/60:.1f} min "
              f"({dt/max(len(recs),1):.1f} s/点)", flush=True)

    t0 = time.time()
    recs, Pf = refine_table(recs, mk(50), n_p=n_refine, n_d=n_refine,
                            verbose=verbose)
    if verbose:
        print(f"  refine 用时 {(time.time()-t0)/60:.1f} min", flush=True)

    if verbose:
        TAU = np.array([r["tau"] for r in recs])
        X = np.array([r["x"] for r in recs])
        res, tip = forward_batch(X, TAU, Pf)
        Pt = np.array([r["p_tip"] for r in recs])
        dp = np.linalg.norm(tip - Pt, axis=1) * 1000
        rn = np.linalg.norm(res, axis=1)
        r = np.linalg.norm(Pt, axis=1) * 1000
        tt = np.degrees(np.arccos(np.clip(
            np.array([x["R_tip"] for x in recs])[:, :, 2] @ _E3, -1, 1)))
        print(f"  自洽: p_tip vs x max={dp.max():.2e} mm | ||res|| 中位={np.median(rn):.2e}")
        print(f"  半径 r∈[{r.min():.0f},{r.max():.0f}] 中位={np.median(r):.0f} mm")
        print(f"  倾角 ∈[{tt.min():.1f},{tt.max():.1f}]° 中位={np.median(tt):.1f}°")
        print(f"  r<300mm 占比 = {(r<300).mean()*100:.1f}%", flush=True)

    if save:
        save_table(fn, recs, params=Pf)
        if verbose:
            print(f"  已存 {fn} ({os.path.getsize(fn)/1e6:.1f} MB)", flush=True)
    return fn, recs, Pf


# ══════════════════════════════════════════════════════════════════════
# CLI 入口  (原 table_build.py + table_report.py, 已并入本模块)
# ══════════════════════════════════════════════════════════════════════
#   build / lhs   LHS 采样建表              (原 build_table.py)
#   tau           按 τmax 批量建表           (原 build_tau_tables.py)
#   densify       深弯区定向加密              (原 densify_deep.py)
#   csv           npz → CSV                 (原 export_csv.py)
#   stats         统计报表                   (原 stats_table.py)
#
# 实现: 每个子命令把原脚本的模块级代码整体缩进成函数体, 调用期间临时替换
#       sys.argv —— 各脚本原本的 argparse / sys.argv 逻辑**一行未改**。
# 独立执行: python ik_table.py <子命令> [参数]
# ══════════════════════════════════════════════════════════════════════
class _Argv:
    """临时替换 sys.argv (让被包裹的脚本原样读取参数)."""

    def __init__(self, argv):
        self.argv = list(argv or [])

    def __enter__(self):
        self.old = sys.argv
        sys.argv = [self.old[0]] + self.argv
        return self

    def __exit__(self, *exc):
        sys.argv = self.old
        return False


def _wrap(fn):
    def wrapper(argv=None):
        with _Argv(argv):
            return fn()
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper

# ── 子命令 `lhs`: LHS 采样建表  (原 build_table.py) ──
def cmd_lhs():
    """LHS 采样建表  ← 原 build_table.py"""
    # -*- coding: utf-8 -*-
    """
    build_table.py — §10 位姿驱动逆解: 建表 → 存盘 → 查询端到端演示
    ==================================================================
    流程 (§10.1 三层结构):
      1. 离线建表  build_table(N, tau_max)          — 批量 VC 打靶 (sdm_vc_batch)
      2. 存盘/载入  save_table / load_table          — npz, 内存≈240 B/点
      3. 在线查询  PoseTable.query(pose)             — k近邻 + ΔL线性插值 + 分支上报
      4. 惰性补洞  未命中 → 现场外层牛顿 (最近表点热启动) → 入库
      5. 显示      κ(s) 节点 → reconstruct_shape      — 上位机渲染用

    用法:
      python build_table.py                 # 小表演示 (N=256, τmax=12)
      python build_table.py --N 4000 --tau-max 15 --save vc_table.npz

    验收指标 (§10.9): 留出位姿插值误差 / 形状积分误差 / 查询时延 / 可达区覆盖率。
    """
    import argparse
    import os
    import sys
    import time
    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8")
    from ik_table import (build_table, PoseTable, Metric, save_table, load_table,
                          solve_pose_multi, reconstruct_shape, apply_safety,
                          so3_exp, so3_log, default_params, L)


    def coverage_report(T):
        """§10.9 可达区覆盖率: 最近邻距离分布 (白化空间)."""
        d, _ = T.tree.query(T.Q, k=2)
        nn = d[:, 1]                                  # 到第二近的白化距离
        print(f"  覆盖率: 最近邻白化距离 中位={np.median(nn):.2f} "
              f"90%={np.percentile(nn,90):.2f} 最大={nn.max():.2f}"
              f"  (σ_p={T.metric.sp}mm 单位)")


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--N", type=int, default=256)
        ap.add_argument("--tau-max", type=float, default=12.0)
        ap.add_argument("--save", type=str, default="vc_table.npz")
        ap.add_argument("--load", type=str, default="")
        args = ap.parse_args()

        P = default_params()
        if args.load and os.path.exists(args.load):
            print(f"[1] 载入表: {args.load}")
            T = load_table(args.load, params=P)
            print(f"    节点 {len(T.recs)}")
        else:
            print(f"[1] 离线建表: τ∈[0,{args.tau_max}]^6, N={args.N}")
            t0 = time.time()
            recs = build_table(N=args.N, tau_max=args.tau_max, params=P,
                               seed=0, verbose=True, log_every=max(1, args.N // 8))
            print(f"    入库 {len(recs)} 点, 用时 {time.time()-t0:.1f}s "
                  f"({(time.time()-t0)/max(len(recs),1)*1000:.0f} ms/点)")
            T = PoseTable(recs, params=P)
            save_table(args.save, recs, params=P)
            print(f"    已存 {args.save} ({os.path.getsize(args.save)/1e6:.2f} MB)")

        tips = np.array([r["p_tip"] for r in T.recs])
        print(f"    tip 范围 x∈[{tips[:,0].min():+.3f},{tips[:,0].max():+.3f}] "
              f"y∈[{tips[:,1].min():+.3f},{tips[:,1].max():+.3f}] "
              f"z∈[{tips[:,2].min():+.3f},{tips[:,2].max():+.3f}]")
        coverage_report(T)

        # ---------- [2] 留出集验证 + 查询时延 (§10.9) ----------
        # pos_tol_mm 现在是**覆盖半径**(最近表点允许的最大位置距离), 不是半径搜索容差。
        print("[2] 留出样本验证: 半数节点建索引, 查询另一半 (真·插值误差)")
        T_half = PoseTable(T.recs[::2], params=P)
        held = T.recs[1::2]
        for interp in (True, False):
            errs_dL, errs_pos, t_q, n_hit = [], [], [], 0
            for r in held:
                t0 = time.perf_counter()
                o = T_half.query((r["p_tip"], r["R_tip"]), pos_tol_mm=200.0,
                                 allow_infill=False, interp=interp)
                t_q.append(time.perf_counter() - t0)
                if o["reachable"] and o["hit"]:
                    n_hit += 1
                    errs_dL.append(np.linalg.norm(o["dL_mm"] - r["dL_mm"]))
                    errs_pos.append(np.linalg.norm(o["p_tip"] - r["p_tip"]) * 1000)
            tag = "插值ΔL(1阶)" if interp else "最近邻选点(0阶)"
            print(f"    {tag}: 命中 {n_hit}/{len(held)}"
                  f" | ΔL 误差 中位={np.median(errs_dL):.2f} 90%={np.percentile(errs_dL,90):.2f} mm"
                  f" | 位置残差 中位={np.median(errs_pos):.1f} mm")
        print(f"    查询时延 中位={np.median(t_q)*1000:.2f} ms")
        # 全表回查 (确定性): 查询点就是表内点, ΔL 应≈0 (插值在 k 邻域里退化为自身)
        errs_full = [np.linalg.norm(
            T.query((r["p_tip"], r["R_tip"]), pos_tol_mm=200.0)["dL_mm"] - r["dL_mm"])
            for r in T.recs[::17]]
        print(f"    全表回查: ΔL 误差 max={max(errs_full):.4f} mm (应≈0)")

        # 相邻节点 ΔL 连续性 (τ 距离 → ΔL 距离应大致 Lipschitz)
        d_tau, d_dL = [], []
        for i in range(0, len(T.recs) - 1, 5):
            d_tau.append(np.linalg.norm(T.recs[i]["tau"] - T.recs[i+1]["tau"]))
            d_dL.append(np.linalg.norm(T.recs[i]["dL_mm"] - T.recs[i+1]["dL_mm"]))
        ratio = np.array(d_dL) / np.maximum(np.array(d_tau), 1e-9)
        print(f"    ΔL 连续性: |ΔdL|/|Δτ| 中位={np.median(ratio):.2f} "
              f"90%={np.percentile(ratio,90):.2f} mm/N")

        # ---------- [3] 未知位姿: 命中 / 补洞 ----------
        print("[3] 未知位姿查询")
        # 找一个"邻近"样本对 (τ 距离最近), 其中点才是真·表内空洞
        i_best, d_best = 0, np.inf
        for i in range(len(T.recs) - 1):
            d = np.linalg.norm(T.recs[i]["tau"] - T.recs[i + 1]["tau"])
            if d < d_best:
                d_best, i_best = d, i
        mid_p = 0.5 * (T.recs[i_best]["p_tip"] + T.recs[i_best + 1]["p_tip"])
        mid_R = T.recs[i_best]["R_tip"]
        tests = [
            # (名称, 位姿, 覆盖半径mm) —— 半径内走 k近邻+ΔL插值, 半径外触发惰性补洞
            ("邻近样本中点(真空洞)", mid_p, mid_R, 60.0),
            ("轻微偏移", T.recs[100]["p_tip"] + np.array([0.004, -0.003, 0.002]),
             T.recs[100]["R_tip"], 60.0),
            ("不可达(超长)", np.array([0.0, 0.0, 0.90]), np.eye(3), 60.0),
        ]
        for name, pd, Rd, tol in tests:
            o = T.query((pd, Rd), pos_tol_mm=tol, verbose=True)
            if o["reachable"]:
                print(f"    [{name}] hit={o['hit']} branch={o['branch']} "
                      f"interp={o.get('interpolated')} "
                      f"pos残差={o['pos_res_mm']:.2f}mm rot残差={o['rot_res_mrad']:.2f}mrad "
                      f"ΔL={np.round(o['dL_mm'],2)}")
            else:
                print(f"    [{name}] 不可达 (reachable=False)")
        print(f"    补洞次数={T.n_infill}, 表大小={len(T.recs)}")

        # ---------- [4] 形状显示: κ(s) 节点 → 骨架 ----------
        print("[4] 由 κ(s) 节点积分重建骨架 (上位机显示)")
        pd = T.recs[100]["p_tip"] + np.array([0.003, 0.0, -0.002])
        o = T.query((pd, T.recs[100]["R_tip"]), pos_tol_mm=60.0)
        if o["reachable"] and o["kcoef"] is not None:
            pk, Rk = reconstruct_shape(o["kcoef"], P)
            e_tip = np.linalg.norm(pk[-1] - o["p_tip"]) * 1000
            e_R = np.linalg.norm(so3_log(o["R_tip"].T @ Rk[-1]))
            print(f"    κ 重建 vs 解: tip 差={e_tip:.3f} mm, R(L) 差={np.rad2deg(e_R):.3f}°")

        # ---------- [5] 下发前安全检查 (§10.7) ----------
        print("[5] 下发前安全检查 (行程限位 + 预紧下限)")
        for name, dL in [("正常", o["dL_mm"]),
                         ("超行程", o["dL_mm"] + 80.0),
                         ("单丝松弛", np.concatenate([o["dL_mm"][:3], [-70.0, 0, 0]]))]:
            safe, flags = apply_safety(dL, tau=o["tau"], dL_max_mm=60, dL_min_mm=-60)
            print(f"    [{name}] ok={flags['ok']} 截断丝={flags['clipped']} "
                  f"松弛丝={flags['slack']} ΔL_out={np.round(safe,1)}")
        print("完成.")


    if True:  # 原 __main__ 守卫
        main()

# ── 子命令 `tau`: 按 τmax 批量建表  (原 build_tau_tables.py) ──
def cmd_tau():
    """按 τmax 建表 (可一次多个)  ← 原 build_tau_tables.py

    用法: python ik_table.py tau [40] [60] ...
    实际逻辑见库函数 `build_table_by_tau_max(tau_max, ...)`。
    """
    import sys
    from ik_table import build_table_by_tau_max

    if True:  # 原 __main__ 守卫
        args = sys.argv[1:] or ["30", "40"]
        for a in args:
            build_table_by_tau_max(float(a))
        print("\n全部完成", flush=True)


# ── 子命令 `densify`: 深弯区定向加密  (原 densify_deep.py) ──
def cmd_densify():
    """深弯区定向加密  ← 原 densify_deep.py"""
    # -*- coding: utf-8 -*-
    """
    densify_deep.py — 深弯区定向加密 (射线扫描版).

    前车之鉴: 我上一版用"已有深弯点的 x"给低-τ 的延拓起点做初值 —— 那是**错的分支**
    (深弯解在低 τ 下根本不对称), 收敛率 3/1350。

    正确做法 (射线扫描法, 实测 100% 收敛):
      从直臂 (v0=E3) 起步, 沿每个"集中驱动"方向把 τ 从 0 逐步拉到 τmax 并逐级热启动。
      这天然是延拓, 且每步都从正确的分支长出来。

    把它产出的点并入表, 即可补全深弯区的**包络**与**密度**。

    用法: python densify_deep.py <tag>
    """
    import os
    import sys
    import time
    import shutil
    import numpy as np
    sys.stdout.reconfigure(encoding="utf-8")
    from ik_table import (load_table, refine_table, save_table, default_params,
                          record_from_batch, _fill_dL, _clone_params)
    from ik_table import _clone_params
    from tendon_coupling import _E3, SEG_A, SEG_B
    from tendon_coupling import shooting_batch, forward_batch
    from scipy.spatial import cKDTree

    TAG = sys.argv[1] if len(sys.argv) > 1 else "30"
    TMAX = float(TAG)
    FN = f"vc_table_{TAG}.npz"
    P = default_params()
    P50 = _clone_params(P, n_p=50, n_d=50)

    # ---- 方向集 ----
    # ★ 深弯方位要靠 **2 丝组合 × 张力比** 才能铺匀。
    # 深弯由段B 丝(1/3/5, 锚在 L=450mm)提供, 单丝时方位锁在 60/180/300°。
    # 段A 丝(0/2/4, 锚在 L1=225mm) 虽弯不深, 但通过**杆的连续性** (段界 L1 处弯矩
    # 传给段B) 把弯的方向转走: 段B丝1(60°)+段A丝0 在 α=0.25 时转到 106.5°, 而
    # 半径 174 vs 173mm 几乎不变。
    # 实测 (scan_all): 只扫单丝/同段对 -> r<200mm 方位集中 (<5°占比 56%);
    #   扫全 **15 个 2-丝组合 × 15 个张力比** -> r<200mm 方位覆盖 12/12 桶,
    #   <5°占比 23.8% ≈ 均匀值 16.7%。所以必须用全组合, 不能只用段A×段B。
    dirs = []
    for w in range(6):                                               # 单丝
        v = np.zeros(6); v[w] = 1; dirs.append(v)
    COMBOS = [(i, j) for i in range(6) for j in range(i + 1, 6)]     # 全 15 对
    RATIOS = np.concatenate([np.linspace(0.05, 1.0, 8),
                             1.0 / np.linspace(0.05, 1.0, 8)[:-1]])  # 0.05..20
    for (i, j) in COMBOS:
        for rt in RATIOS:
            v = np.zeros(6); v[i] = rt; v[j] = 1.0; dirs.append(v)
    rng = np.random.default_rng(7)
    for _ in range(24):                                              # 稀疏随机 (偏集中)
        v = np.zeros(6); idx = rng.choice(6, rng.integers(1, 3), replace=False)
        v[idx] = 1.0; v += rng.random(6) * 0.25
        dirs.append(v)
    U = np.array(dirs); U = U / np.linalg.norm(U, axis=1, keepdims=True)
    ND = len(U)
    NSTEP = 30
    SS = np.linspace(0.05, 1.0, NSTEP)

    print(f"=== {TAG}N 深弯加密 (射线扫描版) ===", flush=True)
    T = load_table(FN)
    recs = T.recs
    Pt0 = np.array([r["p_tip"] for r in recs])
    r0 = np.linalg.norm(Pt0, axis=1) * 1000
    print(f"原表 N={len(recs)}  r_min={r0.min():.0f}  r<300 占比={(r0<300).mean()*100:.1f}%",
          flush=True)

    # ---- 射线扫描: 所有方向 × 所有步一次性批量, 逐步热启动从直臂长出 ----
    t0 = time.time()
    X = np.tile([0., 0., 1., 0., 0., 0.], (ND, 1))
    TAU_list, X_list, KEEP = [], [], []
    for si, s in enumerate(SS):
        TAU = U * (s * TMAX)
        out = shooting_batch(TAU, P50, x0=X, tol=1e-8,
                             max_iter=(20 if si == 0 else 40))
        # 每一步存 **本步的解** (不是被携带的热启动 —— 非收敛样本的 x 不是解)
        TAU_list.append(TAU)
        X_list.append(out["x"].copy())
        KEEP.append(out["converged"])
        X = np.where(out["converged"][:, None], out["x"], X)   # 只把收敛的带到下一步
        if si % 6 == 0:
            print(f"  s={s:.2f}: 收敛 {out['converged'].sum()}/{ND}", flush=True)

    TAU_all = np.vstack(TAU_list)
    X_all = np.vstack(X_list)
    ok = np.concatenate(KEEP)
    print(f"扫描 {len(TAU_all)}: 收敛 {ok.sum()} ({ok.mean()*100:.0f}%)  "
          f"{time.time()-t0:.0f}s", flush=True)
    # 去重: 同一网格步的相同 τ 只留一个 (不同方向可能重合)
    TU, idxu = np.unique(np.round(TAU_all[ok], 6), axis=0, return_index=True)
    ii = np.where(ok)[0][idxu]
    T_ok, X_ok = TAU_all[ii], X_all[ii]
    print(f"去重后 {len(T_ok)}", flush=True)

    # ---- 打包 -> refine ----
    res_all, _ = forward_batch(X_ok, T_ok, P50)
    _, _, cv = forward_batch(X_ok, T_ok, P50, want_curve=True)
    new = []
    for i in range(len(T_ok)):
        rec = record_from_batch(i, T_ok[i], P50,
                                {"x": X_ok, "res": res_all,
                                 "converged": np.ones(len(T_ok), bool)}, cv)
        if rec is not None and np.isfinite(rec["p_tip"]).all():
            rec["res"] = float(np.linalg.norm(res_all[i]))
            new.append(rec)
    _fill_dL(new, P50)
    print(f"打包 {len(new)} 条", flush=True)
    new, Pf = refine_table(new, P50, n_p=300, n_d=300, verbose=True)

    # ---- 去重合并 ----
    M = T.metric
    Q_old = np.array([M.encode(x["p_tip"], x["R_tip"]) for x in recs])
    Q_new = np.array([M.encode(x["p_tip"], x["R_tip"]) for x in new])
    d, _ = cKDTree(Q_old).query(Q_new, k=1)
    fresh = d > 1.0
    print(f"去重: {fresh.sum()}/{len(new)} 条是新的", flush=True)
    merged = recs + [new[i] for i in range(len(new)) if fresh[i]]
    print(f"合并后 N={len(merged)}", flush=True)

    if not os.path.exists(FN + ".bak2"):
        shutil.copy2(FN, FN + ".bak2"); print(f"已备份 → {FN}.bak2", flush=True)
    save_table(FN, merged, params=Pf)
    print(f"已存 {FN} ({os.path.getsize(FN)/1e6:.1f} MB)", flush=True)

    # ---- 自检 ----
    TA = np.array([x["tau"] for x in merged]); XA = np.array([x["x"] for x in merged])
    Pt2 = np.array([x["p_tip"] for x in merged])
    rr_ = np.linalg.norm(Pt2, axis=1) * 1000
    Rall = np.array([x["R_tip"] for x in merged])
    tt_ = np.degrees(np.arccos(np.clip(Rall[:, :, 2] @ _E3, -1, 1)))
    res, tip = forward_batch(XA, TA, Pf)
    print(f"\n自洽: max={np.linalg.norm(tip-Pt2,axis=1).max():.2e} mm | "
          f"||res|| 中位={np.median(np.linalg.norm(res,axis=1)):.2e}")
    print(f"半径 r∈[{rr_.min():.0f},{rr_.max():.0f}] 中位={np.median(rr_):.0f} mm")
    print(f"倾角 ∈[{tt_.min():.1f},{tt_.max():.1f}]°")
    print(f"r<300 占比 {(rr_<300).mean()*100:.1f}%   r<250 占比 {(rr_<250).mean()*100:.1f}%")

# ── 子命令 `csv`: 导出 CSV (wide + key)  (原 export_csv.py) ──
def cmd_csv():
    """导出 CSV (wide + key)  ← 原 export_csv.py"""
    # -*- coding: utf-8 -*-
    """
    export_csv.py — 把查表 npz 导出成人类可读 CSV (给 MATLAB / Excel / 人工核对用)

    输出两份:
      vc_table_wide.csv  — 每行一个节点 (tau/dL/p_tip/R_tip/κ摘要/res; κ 全量在 npz 的 kS/kK)
      vc_table_key.csv   — 精简版: 只有索引键与下发量 pose + dL (6+6+6=18 列)

    用法: python export_csv.py [vc_table_40.npz]
    """
    import sys
    import csv
    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8")

    SRC = sys.argv[1] if len(sys.argv) > 1 else "vc_table_40.npz"
    WIDE = "vc_table_wide.csv"
    KEY = "vc_table_key.csv"

    z = np.load(SRC)
    N = z["tau"].shape[0]


    def rpy_from_R(R):
        sp = min(1.0, max(-1.0, -R[2, 0]))
        p = np.arcsin(sp)
        if abs(sp) < 0.9999:
            return np.arctan2(R[2, 1], R[2, 2]), p, np.arctan2(R[1, 0], R[0, 0])
        return 0.0, p, np.arctan2(-R[0, 1], R[1, 1])


    # ---------- 1) 精简表: 位姿 + 下发量 ----------
    hdr_key = (["idx"]
               + [f"x_{a}" for a in ("x", "y", "z")]
               + [f"rpy_{a}" for a in ("roll", "pitch", "yaw")]
               + [f"dL{w}" for w in range(6)]
               + [f"tau{w}" for w in range(6)])
    with open(KEY, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(hdr_key)
        for i in range(N):
            p = z["p_tip"][i]
            r, pi, ya = rpy_from_R(z["R_tip"][i])
            row = ([i] + [f"{v:.6f}" for v in p]
                   + [f"{v:.6f}" for v in (r, pi, ya)]
                   + [f"{v:.4f}" for v in z["dL_mm"][i]]
                   + [f"{v:.4f}" for v in z["tau"][i]])
            w.writerow(row)

    # ---------- 2) 全量表 ----------
    # κ(s) 有两种存法: 新格式按节点存 (kS/kK, 节点数随网格变, 平铺会爆列 → 只给摘要);
    # 旧格式按分段多项式存 (kcoef_a/kcoef_b, (3,7) → 平铺成列)。按文件实际内容选。
    NEW_K = "kK" in z.files
    if NEW_K:
        hdr = (["idx"]
               + [f"p_{a}" for a in ("x", "y", "z")]
               + [f"R{r}{c}" for r in range(3) for c in range(3)]
               + [f"dL{w}" for w in range(6)]
               + [f"tau{w}" for w in range(6)]
               + ["kappa_summary"]
               + [f"x{k}" for k in range(6)]
               + ["res"])
    else:
        hdr = (["idx"]
               + [f"p_{a}" for a in ("x", "y", "z")]
               + [f"R{r}{c}" for r in range(3) for c in range(3)]
               + [f"dL{w}" for w in range(6)]
               + [f"tau{w}" for w in range(6)]
               + [f"kA{k}c{j}" for k in range(3) for j in range(z["kcoef_a"].shape[2])]
               + [f"kB{k}c{j}" for k in range(3) for j in range(z["kcoef_b"].shape[2])]
               + [f"x{k}" for k in range(6)]
               + ["res"])
    with open(WIDE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for i in range(N):
            head = ([i]
                    + [f"{v:.6f}" for v in z["p_tip"][i]]
                    + [f"{v:.6f}" for v in z["R_tip"][i].ravel()]
                    + [f"{v:.4f}" for v in z["dL_mm"][i]]
                    + [f"{v:.4f}" for v in z["tau"][i]])
            if NEW_K:
                # 节点格式: 只给平均 |κ| 摘要 (全量在 npz 的 kS/kK)
                kmid = ([f"{np.linalg.norm(z['kK'][i], axis=1).mean():.6f}"])
            else:
                kmid = ([f"{v:.6e}" for v in z["kcoef_a"][i].ravel()]
                        + [f"{v:.6e}" for v in z["kcoef_b"][i].ravel()])
            row = head + kmid + [f"{v:.6f}" for v in z["x"][i]] + [f"{z['res'][i]:.3e}"]
            w.writerow(row)

    print(f"{SRC} → {N} 节点")
    print(f"  {KEY}   ({len(hdr_key)} 列, 精简: 位姿+RPY+ΔL+τ)")
    print(f"  {WIDE}  ({len(hdr)} 列, 全量: +R矩阵 +κ +打靶未知量 +残差)")
    if NEW_K:
        print(f"  注: κ(s) 按节点存于 npz 的 kS/kK ({z['kK'].shape[1]} 节点/样本), CSV 只给平均|κ|")
    else:
        print(f"  注: 旧格式, κ(s) 为分段多项式系数 kcoef_a/kcoef_b")

# ── 子命令 `stats`: 统计报表 ASCII  (原 stats_table.py) ──
def cmd_stats():
    """统计报表 ASCII  ← 原 stats_table.py"""
    # -*- coding: utf-8 -*-
    """
    stats_table.py — 建表数据统计表 (可达空间 / 张力 / 曲率 / 覆盖)

    输出:
      stats_table.csv   机器可读
      控制台打印的 markdown 表 (可直接贴进思源)
    """
    import sys
    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8")
    from ik_table import load_table
    from tendon_coupling import forward_batch
    from inverse_solver import build_basis   # POD 降阶模型已并入逆解模型
    from tendon_coupling import _E3

    import sys
    T = load_table(sys.argv[1] if len(sys.argv) > 1 else "vc_table_40.npz")
    recs = T.recs
    N = len(recs)
    P = np.array([r["p_tip"] for r in recs])
    R = np.array([r["R_tip"] for r in recs])
    TAU = np.array([r["tau"] for r in recs])
    DL = np.array([r["dL_mm"] for r in recs])
    RES = np.array([r["res"] for r in recs])

    basis = build_basis(recs, r=8, params=T.params)   # 用表自身的网格, 否则退化到 30 单元
    _, _, cv = forward_batch(np.array([r["x"] for r in recs]), TAU, basis["params"],
                             want_curve=True)
    bend = np.linalg.norm(cv[2], axis=2).mean(axis=1)     # 平均曲率
    rad = np.linalg.norm(P, axis=1)                        # 到基座距离
    tilt = np.degrees(np.arccos(np.clip((R[:, :, 2] @ _E3), -1, 1)))

    Q = [0, 5, 25, 50, 75, 90, 95, 99, 100]


    def row(name, arr, unit, fmt="{:.3f}"):
        ps = [np.percentile(arr, q) for q in Q]
        return (name, unit, *[fmt.format(v) for v in ps],
                fmt.format(arr.mean()), fmt.format(arr.std()))


    def sub_row(name, vectors, unit, fmt="{:.3f}"):
        """对 N×k 数组, 先求每样本的范数再统计."""
        v = np.linalg.norm(vectors, axis=1)
        return row(name, v, unit, fmt)


    QH = ["指标", "单位", "最小", "5%", "25%", "中位", "75%", "90%", "95%", "99%", "最大", "均值", "标准差"]

    rows = [
        row("末端 x", P[:, 0] * 1000, "mm", "{:.1f}"),
        row("末端 y", P[:, 1] * 1000, "mm", "{:.1f}"),
        row("末端 z", P[:, 2] * 1000, "mm", "{:.1f}"),
        row("到基座距离 r", rad * 1000, "mm", "{:.1f}"),
        row("末端切线倾角", tilt, "°", "{:.1f}"),
        row("平均曲率 |κ|", bend, "1/m", "{:.3f}"),
        sub_row("张力范数 ||τ||", TAU, "N", "{:.2f}"),
        row("单丝张力 τ (全体)", TAU.ravel(), "N", "{:.2f}"),
        sub_row("丝位移范数 ||ΔL||", DL, "mm", "{:.2f}"),
        row("打靶残差", RES, "-", "{:.2e}"),
    ]

    # ---------- 控制台 markdown ----------
    print(f"### 建表数据统计 (N = {N}, τ∈[0,12]⁶ 均匀采样, 全阶 VC 打靶)\n")
    print("| " + " | ".join(QH) + " |")
    print("|" + "|".join([":-:"] * len(QH)) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")

    # ---------- 半径分布 ----------
    print(f"\n### 末端半径分布 (臂长 450 mm)\n")
    edges = list(range(100, 461, 20))
    h, _ = np.histogram(rad * 1000, bins=edges)
    print("| 半径区间 (mm) | 样本数 | 占比 |")
    print("|:-:|:-:|:-:|")
    for i, c in enumerate(h):
        print(f"| {edges[i]}–{edges[i+1]} | {c} | {c/N*100:.1f}% |")

    # ---------- 覆盖 ----------
    from scipy.spatial import cKDTree
    dd, _ = cKDTree(P).query(P, k=2)
    nn = dd[:, 1] * 1000
    print(f"\n### 位置空间覆盖\n")
    print(f"| 指标 | 单位 | 中位 | 90% | 最大 |")
    print(f"|:-:|:-:|:-:|:-:|:-:|")
    print(f"| 最近邻距 | mm | {np.median(nn):.1f} | {np.percentile(nn,90):.1f} | {nn.max():.1f} |")

    # ---------- CSV ----------
    import csv
    with open("stats_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(QH)
        for r in rows:
            w.writerow(r)
        w.writerow([])
        w.writerow(["半径区间(mm)", "样本数", "占比"])
        for i, c in enumerate(h):
            w.writerow([f"{edges[i]}-{edges[i+1]}", c, f"{c/N*100:.1f}%"])
    print("\nsaved: stats_table.csv")


SUBCOMMANDS = {
    "build": _wrap(cmd_lhs),       # LHS 采样建表
    "lhs": _wrap(cmd_lhs),         # 同上 (别名)
    "tau": _wrap(cmd_tau),         # 按 τmax 批量建表
    "densify": _wrap(cmd_densify), # 深弯区定向加密
    "csv": _wrap(cmd_csv),         # npz → CSV
    "stats": _wrap(cmd_stats),     # 统计报表
}

_USAGE = """ik_table — 表工具包 (库 + CLI)

  建表
    build | lhs       LHS 采样建表            (--N --tau-max --save)
    tau  [30 40 ...]  按张力上限批量建表 (延拓+refine)
    densify <tag>     深弯区定向加密

  看表
    csv   [table.npz] 导出 vc_table_wide.csv / vc_table_key.csv
    stats [table.npz] 统计报表 → stats_table.csv + 控制台

  直接跑:  python ik_table.py build --N 4000 --tau-max 15 --save vc_table.npz
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
    return SUBCOMMANDS[cmd](rest)


if __name__ == "__main__":
    sys.exit(main() or 0)
