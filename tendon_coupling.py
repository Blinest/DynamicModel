# -*- coding: utf-8 -*-
"""
tendon_coupling.py — 肌腱力耦合模型 (Tendon–Force Coupling Model, TFCM)
=======================================================================
**只做正解** (τ → 位姿/ΔL)。逆解 (位姿/ΔL → τ) 见 `inverse_solver.py`。

把原先散在几个文件里的东西**整合成一个模型**:

    菱形晶胞骨架 (唯一刚度来源)            ← sdm_vc.py §晶胞
    ├─ 几何:  L_strut=π·r_i/(N_circ·cosα), l_ax=2·L_strut·sinα, N_long=L/l_ax
    │         分段: 段A N_circ=6 / 段B N_circ=3 ⇒ L_strut_A = L_strut_B/2
    ├─ 刚度:  k_unit=2EA·sin²α/L_strut
    │         EI0 = 4·E·A·sin³α·Σy²,  k_rod = N_circ·4EA·sin³α/L
    │         C0 = EI0/L_strut,  GJ = EI0
    ├─ 恒等:  EI0/k_rod = 2·r_i²·L   (α,E,A,N_circ,N_long 全约掉)
    └─ 非线性: C(κ)=C0·sin(κ·L_strut), EI_eq(κ)=EI0·sinc(κ·L_strut)
                                          ＋
    全阶 Cosserat 梁静力平衡 (VC 式)      ← sdm_vc.py / sdm_vc_batch.py
    ├─ 本构:  N = Kse·(v−E3),  C = Kbt·u
    ├─ 平衡:  [Kse+A  G ; B  Kbt+H]·[v'; u'] = [d; c]
    ├─ 肌腱分布项 (A,B,G,H,c,d)           ← intermedquant()
    ├─ 锚接点集中力/矩 (段界 & 末端)        ← anchor_loads()
    └─ 打靶 (Levenberg 超定最小二乘)       ← shooting()
                                          ＋
    肌腱几何与力 → 长度的耦合                ← sdm_vc / ik_table
    ├─ 丝孔位偏移  o_w = r_disk·[cosα_w, sinα_w, 0]
    ├─ 丝空间路径长 L_w = Σ‖Δ(p_j + R_j·o_w)‖
    └─ ΔL_w = L_w − L0_w        (L0 = L1 若 w∈SEG_A, 否则 L)

核心产出 = **肌腱力 ↔ 肌腱长度 的 6×6 耦合矩阵**
    J[i,j] = ∂ΔL_i / ∂τ_j          (mm/N)
    ▸ 描述"拉哪根丝、哪些丝跟着变"，即**耦合**；
    ▸ 它是**正解灵敏度** (对正向映射求导), 本模块只提供它;
      用它做反解 (伪逆 → ΔL) 的迭代在 `inverse_solver.py`。

数学约定
--------
    变分坐标  x = [v0; u0] ∈ R⁶      v=E3 为不可伸直臂
    段 A: s∈[0,L1], 丝 {0,2,4}, 锚止于 s=L1
    段 B: s∈[L1,L], 丝 {1,3,5}, 锚止于末端
    残余残差  7 维 = [力(3), 矩(3), 中心杆弧长(1)]   未知仅 6 ⇒ 超定

用法
----
    from tendon_coupling import TendonCouplingModel
    M = TendonCouplingModel()
    sol = M.shape([2,2,2,2,2,2])             # 全阶 Cosserat 解
    L   = M.tendon_lengths(sol)              # 6 根丝几何长
    dL  = M.delta_L_mm([8,0,0,0,0,0])        # 6 维 ΔL (mm)
    J   = M.coupling_matrix([2,2,2,2,2,2])   # 6×6 耦合矩阵
    tip, Rtip, dL = M.tip_from_tau([6,0,6,0,6,3])   # τ → 位姿(正解)
    PS, RS = M.build_shapes_batch(taus)[:2]  # 批量细骨架 (绘图)

    # 反解 (另一模块):
    from inverse_solver import TendonInverseSolver
    S = TendonInverseSolver(model=M);  S.solve_dL(dL_target)

作者: 由 sdm_vc / sdm_vc_batch / ik_table 整合
"""
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# 内核 (原 sdm_vc.py, 已并入本模型)
#   • 物理:      全阶 Cosserat 静力平衡 (VC 式)
#   • 几何:      6 丝分布 / 几何丝长 / 骨架积分
#   • 本构:      菱形晶胞 (唯一刚度来源)
# ══════════════════════════════════════════════════════════════════════
import numpy as np

# ================= 几何 =================
# ⚠ 臂长 L1 / L2 / L 由**菱形晶胞结构推出**(不硬编码), 定义见下方「晶胞参数」段:
#      单胞轴向长  l_ax = 2·L_strut·sinα = 2·r_i·tanα
#      段长        L_g  = N_long,g · l_ax      (N_long,g 个胞沿轴向串联)
#      总长        L    = L1 + L2
#   这样"胞数 × 胞长 = 段长"严格自洽 (N_long 不再是取整近似)。
SEG_A = [0, 2, 4]
SEG_B = [1, 3, 5]
ALPHA_DEG = [0, 60, 120, 180, 240, 300]
ALPHA = np.radians(ALPHA_DEG)
_E3 = np.array([0., 0., 1.])


def wire_r(r_disk, i):
    a = np.radians(ALPHA_DEG[i])
    return np.array([r_disk * np.cos(a), r_disk * np.sin(a), 0.0])


# ================= 中心杆: 轴向可压缩串联弹簧 =================
# 臂体的轴向约束由**中心杆**承担, 驱动丝只提供"把杆压短"的力:
#   拉丝 → 杆缩短 δ → 丝几何长随之变短 ΔL ≈ -δ。
# 因此杆不是"给丝撑开半径的刚体", 而是一个轴向刚度 k_rod 的弹簧。
# 杆的轴向柔度由 Kse 的轴向分量承载 (v_z = 1+ν_z, δ = ∫ν_z ds), 无需扩状态。
#
# ⚠ EA 同时扛"抗剪"和"抗轴向"两个角色, 直接调 EA 会把剪切柔度一起放大。
# 故轴向刚度改用独立参数 k_rod, GA 单独标定为不变量 (见 VCParams.Kse)。

# 菱形晶胞等效弯曲刚度 (笔记《曲率分布函数κ(s)》§4.2 **修正式**)
#
#   单胞轴向刚度   k_unit = 2·E·A·sin²α / L_strut
#   截面弯矩       M = Σ_j k_unit·κ·l_ax·y_j²
#                 l_ax = 2·L_strut·sinα,  Σ_j y_j² = 2·N_circ·r_i²   (**不含 N_long**)
#   ⇒ 弯矩本构    C(κ) = EI0·sin(κ·L_strut),  EI0 = k_unit·l_ax·Σy²
#   ⇒ 割线刚度    EI_eq(κ) = C(κ)/κ = EI0·sinc(κ·L_strut)
#   ⇒ 切向刚度    dC/dκ = EI0·cos(κ·L_strut)
#
# 两种刚度在 κ→0 都 → EI0 (有限非零) —— 符合"低曲率下需要很大力才变形"。
#
# ⚠ 旧代码把 **弯矩 C(κ)** 当成了刚度, 写成 `EI = C0·|sin(κ·L_strut)|`, κ→0 时
# EI→0 (直臂零抗弯刚度, 物理上错), 且让 Cosserat 6×6 奇异、打靶不收敛。旧代码
# 再用 `theta_seed=π/6` + `EI_FLOOR_SIN=0.35` 两个补丁把 EI(0) 硬抬到 0.064
# (sin(π/6)=0.5, 0.128×0.5=0.064)。修正后 seed/floor 全部不需要。
# ---- 菱形晶胞材料/几何参数 (臂体唯一的承载结构: 无中心杆) ----
# 臂体**没有中心杆**, 承载与弯曲全由菱形晶胞骨架承担。所以弯曲刚度 EI0 与
# 轴向刚度 k_rod 必须由**同一套**晶胞参数推出, 不能各调各的:
#
#     单胞轴向刚度  k_unit = 2·E·A·sin²α / L_strut
#     胞轴向长      l_ax   = 2·L_strut·sinα
#     截面 Σy²       = 2·N_circ·r_i²              (不含 N_long)
#     ⇒ 弯曲刚度    EI0 = k_unit·l_ax·Σy²        (小角极限)
#     ⇒ 轴向刚度    k_rod = k_unit / N_long      (N_long 个胞串联)
#     ⇒ 弯矩常数    C0 = EI0 / L_strut
#
# ⚠ **EI0 不是自由参数**。旧代码取 EI0=0.064 当常量, 但那是为了配合
# `theta_seed=π/6` 的补丁 (0.128×sin(π/6)=0.064) 凑出来的占位值 ——
# 笔记 §5 那条 "C↔Python 对拍 ≤0.5mm" **只验证两套实现一致, 从未验证 EI 的物理值**。
# 实证: EI0=0.064 时 6 丝各 20N 会把 450mm 臂压短 155mm (34%), 荒谬;
#       EI0=0.3745 (下述材料参数) 时压短 26mm (5.8%), 才符合"轴向难以压缩"。
E_MAT_DEFAULT   = 3e9       # Pa, 尼龙晶胞杆材料模量
A_STRUT_DEFAULT = 1e-6      # m², 单根斜杆截面积 (1mm²)
N_CIRC_DEFAULT  = 3         # 环向胞数 (单段基准)
N_CIRC_SEG_A    = 6         # 段A (近基座, s∈[0,L1]) 周向菱形胞数
N_CIRC_SEG_B    = 3         # 段B (s∈[L1,L])       周向菱形胞数
R_DISK_DEFAULT  = 0.020     # m, 驱动丝分布半径 r_i
ALPHA_CELL_DEG  = 12.0      # deg, 胞斜杆倾角 (设计选定)

# ── 臂长由晶胞结构推出 (唯一来源: N_long, α, r_i) ──────────────────────
# ⚠ 两段 N_circ 不同 (6/3) ⇒ 斜杆长 L_strut **不是全局单值**:
#     L_strut ∝ 1/N_circ ⇒ L_strut_A = L_strut_B/2, l_ax_A = l_ax_B/2
N_LONG_SEG_A = 50           # 段A 纵向胞数 (225mm ÷ l_ax_A 4.452mm = 50.54 → 50)
N_LONG_SEG_B = 25           # 段B 纵向胞数 (225mm ÷ l_ax_B 8.904mm = 25.27 → 25)
_ALPHA_RAD = float(np.radians(ALPHA_CELL_DEG))
# 单根斜杆长: 环向一圈 N_circ 个胞, 每胞分到弧长 2π·r_i/N_circ, 半宽 = L_strut·cosα
_L_STRUT_A = np.pi * R_DISK_DEFAULT / (N_CIRC_SEG_A * np.cos(_ALPHA_RAD))
_L_STRUT_B = np.pi * R_DISK_DEFAULT / (N_CIRC_SEG_B * np.cos(_ALPHA_RAD))
_LAX_A = 2.0 * _L_STRUT_A * np.sin(_ALPHA_RAD)                 # 段A 单胞轴向长
_LAX_B = 2.0 * _L_STRUT_B * np.sin(_ALPHA_RAD)                 # 段B 单胞轴向长
L1 = N_LONG_SEG_A * _LAX_A  # 段A 长 = 胞数 × 胞长
L2 = N_LONG_SEG_B * _LAX_B  # 段B 长
L_TOTAL = L1 + L2           # 臂长 (下文 L 同名别名)
L  = L1 + L2


def cell_alpha(L=None, N_long=None, r_i=R_DISK_DEFAULT, alpha_deg=ALPHA_CELL_DEG,
               N_circ=N_CIRC_DEFAULT):
    """胞斜杆倾角 α (rad).

    默认返回**设计选定值** `ALPHA_CELL_DEG`; 若显式要求几何自洽 (alpha_deg=None),
    由 `tan α = N_circ·L/(2π·N_long·r_i)` 反解 —— 该式来自两个约束:
      ① 胞横向半宽 = 环向每胞分到的弧长之半:  b = L_strut·cosα = π·r_i/N_circ
      ② N_long 个胞铺满臂长:   L = N_long·l_ax,  l_ax = 2·L_strut·sinα

    ⚠ α 越**小**胞越扁、刚度 `∝sin³α` 越小越能弯; α→0 时 l_ax→0 (菱形退化成
    一条竖线), 需要 N_long→∞, 故 α 有几何下界。
    """
    if alpha_deg is not None:
        return float(np.radians(float(alpha_deg)))
    Ltot = L_TOTAL if L is None else float(L)
    return float(np.arctan(float(N_circ) * Ltot
                           / (2.0 * np.pi * float(N_long) * float(r_i))))


def cell_n_long(r_i=R_DISK_DEFAULT, alpha_deg=ALPHA_CELL_DEG, L=None,
                N_circ=N_CIRC_DEFAULT):
    """铺满臂长 L 所需的纵向胞数 N_long = L / l_ax, l_ax = 2π·r_i·tanα/N_circ.

    α 选得越小 ⇒ l_ax 越小 ⇒ 需要的胞数越多 (与"几何下界"同一件事)。
    """
    Ltot = L_TOTAL if L is None else float(L)
    al = float(np.radians(float(alpha_deg)))
    if np.tan(al) < 1e-12:
        return np.inf
    return float(N_circ) * Ltot / (2.0 * np.pi * float(r_i) * np.tan(al))


def cell_stiffness(E=E_MAT_DEFAULT, A=A_STRUT_DEFAULT, N_circ=N_CIRC_DEFAULT,
                   N_long=None, r_i=R_DISK_DEFAULT, L=None,
                   alpha_deg=ALPHA_CELL_DEG):
    """由晶胞材料/几何参数推出全套刚度 (k_unit, alpha, EI0, C0, k_rod).

    这是模型的**唯一**刚度来源 —— 弯曲与轴向同源, 改一处即同步。

    alpha_deg=None 时按 `cell_alpha()` 几何自洽反解; 默认用设计值 ALPHA_CELL_DEG。
    N_long=None 时按 `cell_n_long()` 取"刚好铺满臂长"的胞数 (取整)。

    ⚠ **环向并联**: 截面一圈有 N_circ 个胞, 轴向是**并联** (刚度相加); 纵向
    N_long 个胞是**串联** (刚度 ÷N_long)。故 `k_rod = N_circ·k_unit/N_long`。

    ★ **EI0 与 k_rod 都只含 sin³α** (L_strut/N_long/r_i 全约掉):

        EI0   = 4·E·A·sin³α · Σy²,   Σy² = 2·N_circ·r_i²
        k_rod = N_circ·4·E·A·sin³α / L

    所以 **α 是唯一的刚度旋钮**。且二者之比

        EI0 / k_rod = 2·r_i²·L     (alpha, E, A, N_circ, N_long **全部约掉**)

    即 **弯曲/轴向的相对软硬由 (r_i, L) 唯一决定**; 等价 `φ/ε = L/(2·r_i)`。
    实测随机参数 8 组逐位一致。
    """
    Ltot = L_TOTAL if L is None else float(L)
    al = cell_alpha(Ltot, N_long, r_i, alpha_deg, N_circ)
    if N_long is None:
        N_long = int(round(cell_n_long(r_i, np.degrees(al), Ltot, N_circ)))
    s, c = np.sin(al), np.cos(al)
    L_strut = np.pi * float(r_i) / (float(N_circ) * c) if c > 1e-12 else float(Ltot)
    k_unit = 2.0 * float(E) * float(A) * s ** 2 / L_strut
    l_ax = 2.0 * L_strut * s
    sum_y2 = 2.0 * float(N_circ) * float(r_i) ** 2
    ei0 = k_unit * l_ax * sum_y2
    c0 = ei0 / L_strut
    k_rod = float(N_circ) * k_unit / float(N_long)
    return dict(k_unit=k_unit, alpha=al, alpha_deg=np.degrees(al),
                L_strut=L_strut, l_ax=l_ax, sum_y2=sum_y2, N_long=N_long,
                EI0=ei0, C0=c0, k_rod=k_rod,
                eps_lock=(1.0 - s) / s if s > 1e-12 else np.inf)


_CELL = cell_stiffness()
EI0_DEFAULT = _CELL["EI0"]          # N·m², 小角弯曲刚度 (由材料参数推出)
C0_DEFAULT = _CELL["C0"]            # N·m², 弯矩常数 C0 = EI0/L_strut
K_ROD_DEFAULT = _CELL["k_rod"]      # N/m,  臂体轴向刚度
L_STRUT_DEFAULT = _CELL["L_strut"]  # m,    晶胞斜杆长 (= π·r_i/(N_circ·cosα), 由几何定)
GJ_DEFAULT = EI0_DEFAULT            # N·m², 抗扭 (圆截面基准与弯曲同值)


def EI_lattice(kappa_xy, L_strut=L_STRUT_DEFAULT, C0=C0_DEFAULT):
    """菱形晶胞等效弯曲刚度 — **割线刚度** EI_eq(κ) = C(κ)/κ.

        C(κ) = C0·sin(κ·L_strut)                (弯矩本构)
        EI_eq(κ) = EI0·sinc(κ·L_strut),  sinc(x)=sin(x)/x,  EI0 = C0·L_strut

    数值: 实际臂体 |κ|≤30 1/m、L_strut=5mm ⇒ θ≤0.15 rad, sinc(0.15)=0.9983 ——
    **修正式在工况内几乎就是常量 EI0**, 与已验证的常量 EI=0.064 模型自然衔接;
    仅 κ≳100 1/m 才明显软化 (κ=100 时 EI 降 8%, κ=300 时降 53%)。
    """
    theta = np.hypot(kappa_xy[0], kappa_xy[1]) * float(L_strut)
    ei0 = float(C0) * float(L_strut)            # EI0 = C0·L_strut
    if theta < 1e-9:
        return ei0                              # sinc(0) = 1
    return ei0 * np.sin(theta) / theta


def hat(v):
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])


class SDMParams:
    """SDM 模型参数. 刚度按**角色**分开:

      Kbt = diag(EI, EI, GJ)     抗弯 EI / 抗扭 GJ
      Kse = diag(GA, GA, k_rod)  抗剪 GA / **抗轴压 k_rod**

    k_rod 是中心杆(臂体)的轴向刚度 N/m, 不是材料模量。段A 两根杆并联(同曲率
    同应变 ⇒ 刚度相加), 段B 单杆。默认 2e4 N/m 相当于 450mm 杆在 20N 下压
    缩 0.45mm —— "轴向难以压缩"的量级。

    旧参数 EA 作废: 它同时进剪切与轴向两个方向, 调它会连带改变剪切柔度,
    把"绳只会收缩"的约束漏掉。保留 `EA` 关键字仅为兼容旧调用点, 会被映射到
    k_rod 且 GA 不受影响。
    """
    def __init__(self, EI=EI0_DEFAULT, GA=1e3, k_rod=K_ROD_DEFAULT, GJ=None,
                 r_disk=R_DISK_DEFAULT, rhoA=0.0, disk_mass=0.0, n_p=30, n_d=30,
                 rod_free_len=None, rod_constraint=False,
                 lattice=True, C0=C0_DEFAULT, L_strut=L_STRUT_DEFAULT,
                 EA=None, cells=None):
        if EA is not None:                      # 旧调用点兼容: EA → k_rod
            k_rod = float(EA)
        if GJ is None:                          # 抗扭: 圆截面基准取与弯曲同值
            GJ = EI0_DEFAULT
        self.EI = EI; self.GA = GA; self.GJ = GJ
        self.k_rod = float(k_rod)
        # 无应力杆长 (直臂): 驱动丝给 ΔL<0 时杆被压短。默认 L (直臂几何长)。
        self.rod_free_len = float(L if rod_free_len is None else rod_free_len)
        # rod_constraint: 预留开关 (当前恒 6 残差; 杆约束已由本构隐式满足)。
        self.rod_constraint = bool(rod_constraint)
        self.r_disk = r_disk; self.rhoA = rhoA; self.disk_mass = disk_mass
        self.n_p = n_p; self.n_d = n_d
        # 菱形晶胞非线性弯曲刚度: 开启后 EI 随曲率变化 (笔记原式, 无 seed/floor)
        self.lattice = lattice; self.C0 = C0; self.L_strut = L_strut
        # ── 分段晶胞: 段A / 段B 各一种菱形胞(只差周向胞数) ──
        # 给了 cells 就用它决定各段 Kbt/Kse (取代旧的"段A 双管 ×2"近似)
        self.cells = tuple(cells) if cells is not None else None
        if self.cells is not None:
            c0 = self.cells[0]
            self.EI = c0.EI0; self.C0 = c0.C0; self.L_strut = c0.L_strut
            self.GJ = c0.GJ; self.k_rod = c0.k_rod

    def _ei(self, k, kappa_xy):
        if not self.lattice or kappa_xy is None:
            return self.EI
        return EI_lattice(kappa_xy, self.L_strut, self.C0)

    def _seg_cell(self, k):
        """本段所用晶胞 (无分段配置时返回 None)."""
        if self.cells is None:
            return None
        return self.cells[0] if k == 0 else self.cells[-1]

    def Kbt(self, k, kappa_xy=None):
        """弯曲刚度矩阵. lattice=True 时按**本段菱形晶胞**随曲率变化 (EI=EI(κ))."""
        c = self._seg_cell(k)
        if c is not None:                      # 分段晶胞路径
            ei = c.EI_secant(kappa_xy) if (self.lattice and kappa_xy is not None) else c.EI0
            return np.diag([ei, ei, c.GJ])
        ei = self._ei(k, kappa_xy)             # 旧单胞路径 (保留兼容)
        base = np.diag([ei, ei, self.GJ])
        return 2.0 * base if k == 0 else base   # 旧行为: 段A 两管叠加

    def Kse(self, k):
        """拉伸/剪切刚度. 分段时取本段晶胞的 k_rod; 否则沿用旧"段A 并联"规则."""
        c = self._seg_cell(k)
        if c is not None:
            return np.diag([self.GA, self.GA, c.k_rod])
        kr = self.k_rod * (2.0 if k == 0 else 1.0)
        ga = self.GA * (2.0 if k == 0 else 1.0)
        return np.diag([ga, ga, kr])

    def active_wires(self, k):
        return list(range(6)) if k == 0 else SEG_B


def intermedquant(u, v, Kse, Kbt, wires, tau, r_disk, fe_b, le_b):
    """丝分布几何项 (照搬 intermedquant2, 推广 6 丝)."""
    A = np.zeros((3, 3)); B = np.zeros((3, 3))
    G = np.zeros((3, 3)); H = np.zeros((3, 3))
    a = np.zeros(3); b = np.zeros(3)
    for i in wires:
        ri = wire_r(r_disk, i)
        pb = hat(u) @ ri + v
        nrm = np.linalg.norm(pb)
        if nrm < 1e-12:
            continue
        pbhat = hat(pb)
        Ai = -(tau[i] / nrm ** 3) * (pbhat @ pbhat)
        rih = hat(ri)
        Bi = rih @ Ai
        A += Ai; B += Bi
        G -= Ai @ rih; H -= Bi @ rih
        ai = Ai @ (hat(u) @ pb)
        a += ai
        b += rih @ ai
    c = -(hat(u) @ Kbt @ u) - (hat(v) @ Kse @ (v - _E3)) - le_b - b
    d = -(hat(u) @ Kse @ (v - _E3)) - fe_b - a
    return A, B, G, H, c, d


def anchor_loads(u, v, R, anchor, tau, r_disk):
    """锚丝终止集中力/矩(空间系, 照搬 boundcond F_sigma/L_sigma 段)."""
    F = np.zeros(3); Lm = np.zeros(3)
    for i in anchor:
        ri = wire_r(r_disk, i)
        p_sp = R @ (hat(u) @ ri + v)          # 空间丝切向
        nrm = np.linalg.norm(p_sp)
        if nrm < 1e-12:
            continue
        F -= tau[i] * p_sp / nrm
        Lm -= tau[i] * (hat(R @ ri) @ p_sp) / nrm
    return F, Lm


def forward(x, tau, F_tip, g_world, R_mount, params, want_curve=False):
    """从基座推进两段到末端. x=[v0;u0] ∈R6.
    返回 (末端残差, 曲线信息). 曲线 = 每子段起点 p 与 s.

    末端残差 6 元 = [力残差(3), 矩残差(3)] + **中心杆弧长约束(标量)**,
    故实际是 7 元 (见 shooting 的非方阵最小二乘)。
    """
    v = np.array(x[:3], float); u = np.array(x[3:], float)
    R = np.array(R_mount, float).reshape(3, 3)
    p = np.zeros(3)
    Kse_end = None; Kbt_end = None
    # 中心杆压缩量 δ = ∫ν_z ds = ∫(v_z−1)ds; 推进时同步累加 (RK2 中点法)
    Lz = 0.0
    Kbt0 = params.Kbt(0, kappa_xy=u[:2])
    # s=0 体坐标内力/内力矩: N=Kse·(v−E3), C=Kbt·u (段A刚度=两管叠加)
    curve_p = [p.copy()]; curve_s = [0.0]
    curve_u = [np.array(u, float)]; curve_v = [np.array(v, float)]
    curve_m = [Kbt0 @ np.array(u, float)]
    curve_n = [params.Kse(0) @ (np.array(v, float) - _E3)]
    curve_R = [np.array(R, float)]
    if not want_curve:
        curve_p = curve_s = curve_u = curve_m = curve_v = curve_n = None

    for k in (0, 1):
        Kse = params.Kse(k)
        Lseg = L1 if k == 0 else L2
        nsub = params.n_p if k == 0 else params.n_d
        ds = Lseg / nsub
        wires = params.active_wires(k)
        # 每段分布重力(世界)转体坐标: fe_world = [0,0,−ρg]−盘重均摊
        for j in range(nsub):
            fe_w = np.zeros(3)
            fe_w[2] = -params.rhoA * 9.81
            fe_w[2] -= (params.disk_mass * 9.81) / L
            fe_b = R.T @ fe_w
            # 菱形晶胞刚度 EI(|κ|) 随当前曲率变化
            Kbt = params.Kbt(k, kappa_xy=u[:2])
            A_, B_, G_, H_, c_, d_ = intermedquant(
                u, v, Kse, Kbt, wires, tau, params.r_disk, fe_b, np.zeros(3))
            M = np.block([[Kse + A_, G_], [B_, Kbt + H_]])
            rhs = np.concatenate([d_, c_])
            dvdu = np.linalg.solve(M, rhs)
            dv = dvdu[:3]; du = dvdu[3:]
            # RK2 (midpoint) 更稳
            v_m = v + 0.5 * ds * dv; u_m = u + 0.5 * ds * du
            # 重新估中点右端(用 u_m,v_m) —— EI 也用中点曲率
            Kbt_m = params.Kbt(k, kappa_xy=u_m[:2])
            A2, B2, G2, H2, c2, d2 = intermedquant(
                u_m, v_m, Kse, Kbt_m, wires, tau, params.r_disk,
                R.T @ fe_w, np.zeros(3))
            M2 = np.block([[Kse + A2, G2], [B2, Kbt_m + H2]])
            vu2 = np.linalg.solve(M2, np.concatenate([d2, c2]))
            # 中心杆压缩: RK2 中点 (v_m 已含半步更新)
            Lz += ds * v_m[2]
            v = v + ds * vu2[:3]; u = u + ds * vu2[3:]
            # 刚性发散防护 (与 sdm_vc_batch 一致): 晶胞模型下牛顿试探步
            # 可能冲出物理量级, 不钳位会累积成非有限值 → SVD 不收敛。
            v = np.clip(v, -3.0, 3.0)
            u = np.clip(u, -80.0, 80.0)
            # p,R 更新
            Rmid = R @ (np.eye(3) + 0.5 * ds * hat(u_m))
            p = p + ds * (Rmid @ v_m)
            R = R @ (np.eye(3) + ds * hat(u))
            if not np.isfinite(R).all():              # 非有限 → 停止推进
                return np.full(7, np.inf), None
            U, S_, Vt = np.linalg.svd(R); R = U @ Vt
            if want_curve:
                curve_p.append(p.copy())
                curve_s.append(curve_s[-1] + ds)
                # 本子段末体坐标: 曲率 u, 内力矩 m=Kbt·u, 内力 n=Kse·(v−E3)
                curve_u.append(np.array(u, float))
                # ⚠ 与 curve_u 取同一节点(端点 u); 用中点 u_m 会引入 O(ds·κ') 不一致
                curve_m.append(params.Kbt(k, kappa_xy=u[:2]) @ np.array(u, float))
                curve_n.append(Kse @ (np.array(v, float) - _E3))  # 体坐标内力 N
                curve_v.append(np.array(v, float))
                curve_R.append(np.array(R, float))
        Kse_end = Kse; Kbt_end = params.Kbt(k, kappa_xy=u[:2])
        # 段界跳(仅 k=0, 段A 锚丝止于 L1)
        if k == 0:
            Fsig, Lsig = anchor_loads(u, v, R, SEG_A, tau, params.r_disk)
            v = v - np.linalg.solve(Kse, R.T @ Fsig)
            u = u - np.linalg.solve(Kbt_end, R.T @ Lsig)
    # 末端残差: 力/矩 (6 维, 方阵打靶)
    n_end = R @ (Kse_end @ (v - _E3))
    m_end = R @ (Kbt_end @ u)
    Fsig, Lsig = anchor_loads(u, v, R, SEG_B, tau, params.r_disk)
    res = np.concatenate([n_end - Fsig - np.array(F_tip), m_end - Lsig])
    if want_curve:
        return res, (np.array(curve_p), np.array(curve_s),
                     np.array(curve_u), np.array(curve_m), np.array(curve_R),
                     np.array(curve_n), np.array(curve_v))
    return res, None


def shooting(tau, params, F_tip=(0, 0, 0), g_world=(0, 0, -9.81),
             R_mount=None, x0=None, tol=1e-9, max_iter=60, lam=1e-3):
    """打靶: 求 x=[v0;u0] 使 forward 末端残差→0.

    残差是 7 维 (力3+矩3+杆弧长1), 未知只有 6 维 ⇒ **超定**, 用最小二乘。
    杆弧长残差对 x 的灵敏度远高于力/矩残差量级不同, 故先按各分量尺度归一化,
    否则 LM 会被大尺度的力残差主导而忽略杆约束。
    """
    R_mount = np.eye(3) if R_mount is None else np.array(R_mount, float)
    x = np.array([0., 0., 1., 0., 0., 0.]) if x0 is None else np.array(x0, float)
    tau = np.array(tau, float); F_tip = np.array(F_tip, float)
    g_world = np.array(g_world, float)
    lsq = bool(getattr(params, "rod_constraint", False))
    nx = 7 if lsq else 6
    # 各残差分量的量级: 力~N, 矩~N·m, 杆弧长残差~N·m (k_rod·δ) — 同量级, 不额外白化
    converged = False
    res, _ = forward(x, tau, F_tip, g_world, R_mount, params)
    for it in range(max_iter):
        # 数值雅可比 nx×6
        J = np.zeros((nx, 6)); h = 1e-6
        for c_ in range(6):
            xp = x.copy(); xm = x.copy(); xp[c_] += h; xm[c_] -= h
            rp, _ = forward(xp, tau, F_tip, g_world, R_mount, params)
            rm, _ = forward(xm, tau, F_tip, g_world, R_mount, params)
            J[:, c_] = (rp - rm) / (2 * h)
        # Levenberg (超定时 JᵀJ 仍是 6×6 法方程)
        H = J.T @ J + lam * np.eye(6)
        g = J.T @ res
        try:
            dx = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dx = -np.linalg.pinv(H) @ g
        # 线搜索: 减残差
        x_new = x + dx
        res_new, _ = forward(x_new, tau, F_tip, g_world, R_mount, params)
        if np.linalg.norm(res_new) >= np.linalg.norm(res) and lam < 1e6:
            lam *= 10; continue
        lam = max(lam / 5, 1e-10)
        x = x_new; res = res_new
        if np.linalg.norm(res, np.inf) < tol:
            converged = True; break
    res, curve = forward(x, tau, F_tip, g_world, R_mount, params, want_curve=True)
    # curve = (p_arr, s_arr, u_arr, m_arr, R_arr, n_arr, v_arr)
    return {"x": x, "v0": x[:3], "u0": x[3:], "res": res,
            "converged": converged, "n_iter": it + 1,
            "p_curve": curve[0], "s_curve": curve[1],
            "u_curve": curve[2], "m_curve": curve[3], "R_curve": curve[4],
            "n_curve": curve[5], "v_curve": curve[6]}


# ================= 肌腱几何长 (原 sdm_vc_disp.py, 已并入; 纯 numpy) =================
# ⚡ 向量化内核开关 (numba JIT; 设 TC_NO_JIT=1 可强制回退 numpy 版)
# ★ 2026-09-28 修: 原来只看环境变量 ⇒ TC_NO_JIT 没设就恒为 True。
#   若解释器**没有 numba**, forward_batch 里的 `except Exception: pass` 会静默吞掉
#   ModuleNotFoundError, 全部掉进纯 numpy 参考实现 (**慢 10x 以上**), 在 n=300 网格上
#   表现为"卡死"。实测同一批位姿: 无 numba 3/3 超时 >45s; 有 numba 3/3 在 35~47s 完成。
#   现在导入时**真实探测** numba 与 _jit_integrate, 失败就置 False 并显式告警。
import os as _os
import warnings as _warnings

_JIT_IMPORT_ERROR = None
_USE_JIT = _os.environ.get("TC_NO_JIT", "0") != "1"
# 打靶雅可比差分方式:
#   "forward" (默认) — 前向差分, 6 次正解/轮, 截断误差 O(h)
#   "central"        — 中心差分, 12 次正解/轮, 截断误差 O(h²)
# 实测 (纯 numpy, N=48, n_p=n_d=20, tol=1e-6):
#   热启动 x0 取自表记录 : 样本-次 1322→734 (−44%), 墙钟 3.3s→1.5s, 收敛同为 48/48,
#                          中位残差 5.82e-8 → 6.08e-8 (同量级, 非逐位相同)
#   冷启动 (直臂初值)    : 收敛 10/48 → 18/48, 中位残差 24.9 → 3.21, 墙钟 11.7s→8.0s
# 想要与旧版逐位一致的对拍, 设 `TC_JAC_MODE=central`。
_JAC_MODE = _os.environ.get("TC_JAC_MODE", "forward").strip().lower()
if not _USE_JIT:
    _JIT_IMPORT_ERROR = "TC_NO_JIT=1 (用户显式关闭)"
else:
    try:
        import numba as _numba                                  # noqa: F401
        import _jit_integrate as _jit_probe                     # noqa: F401
        del _numba, _jit_probe
    except Exception as _e:
        _USE_JIT = False
        _JIT_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
        _warnings.warn(
            "[tendon_coupling] numba JIT 不可用 → 回退纯 numpy 参考实现, 正解约慢 "
            f"10 倍以上。原因: {_JIT_IMPORT_ERROR}。换用装了 numba 的解释器即可恢复。",
            RuntimeWarning, stacklevel=2)


def tendon_path_length(shape, w, mode="geodesic"):
    """由 VC 解(骨架 p, R) 算丝 w 的空间路径长.

    mode="geodesic" (唯一实现): 沿孔位折线 q_j = p_j + R_j·o_w 逐段累加 —— 丝被
        孔约束在半径 r_disk 上, 每走 ds 跟随截面转动。

    ⚠ **"拉伸侧的丝变长"是本模型的固有结果, 不是 bug**。该几何长等价于
        L_w = ∫|e₃ + ν + κ×o_w| ds ≈ L + ∫(κ×o_w)·e₃ ds,
    弯曲时远离曲率中心那一侧的丝 (o_w·κ̂<0) 必然 L_w > L。实测单丝 20N 时
    ΔL 满足 ΔL_w ≈ −L·κ·r·cos(α_w − φ), 与 C 侧 `_calculate_L` 的经典公式逐字一致:
    被拉丝 (cos=1) 得 −74.5mm, 相距 120° 的丝 (cos=−0.5) 得 +36.8mm —— 比值 0.49。
    因此**只靠改路径口径消除不了拉伸侧**: 需要"丝把臂体拉向自己"的耦合机制
    (双路径/绞盘), 或直接承认 ΔL 是双向的。
    """
    if mode != "geodesic":
        raise ValueError(f"unsupported tendon path mode: {mode!r} "
                         "(只保留 geodesic; 见上方说明)")
    p = shape["p_curve"]; R = shape["R_curve"]
    n = len(p) - 1                 # 子段数
    nA = int(round(L1 / (L / n)))  # 段界所在节点
    end = nA if w in SEG_A else n
    o = np.array([0.030 * np.cos(ALPHA[w]), 0.030 * np.sin(ALPHA[w]), 0.0])
    total = 0.0
    q_prev = p[0] + R[0] @ o
    for j in range(1, end + 1):
        q = p[j] + R[j] @ o
        total += np.linalg.norm(q - q_prev)
        q_prev = q
    return total


# ============ 骨架几何积分 (κ,v) → p,R  (原 pod_solver.py, 已下沉到内核) ============
# 纯运动学: p' = R·v, R' = R·hat(κ); 与刚度/张力无关, 故归内核。
# ROM 离线建基与在线重建、绘图细骨架 都复用这里。
# ==================== 在线: 由 κ(s) 积分骨架 ====================

def integrate_shape(kappa_s, S_grid, nsub=60, v_grid=None):
    """已知 κ(s) 场 (nS,3) 与网格 S_grid → 骨架 p(s), R(s).

    每步只做 R'=R·hat(κ), p'=R·v, 无 6×6 解、无肌腱几何。
    与 sdm_vc.forward 的中点位姿积分同构。

    ⚠ **v 不能硬当 e₃**: 臂体轴向刚度 `k_rod` 有限时 `v_z = 1+ν_z` 沿弧长偏离 1
    (实测 0.85~0.90)。不传 v_grid 会让重建骨架凭空长 40~58mm。

    ⚠ nsub 决定步长 L/nsub。表网格已到 0.75mm, nsub 取小 (如默认 60=7.5mm) 会把
    细解的几何重新粗化 —— 必须与调用方的解网格匹配。
    """
    ds = L / nsub
    p = np.zeros(3); R = np.eye(3)
    ps = [p.copy()]; Rs = [R.copy()]
    for j in range(nsub):
        k = _interp_kappa(kappa_s, S_grid, (j + 0.5) * ds)
        vm = _interp_v(v_grid, S_grid, (j + 0.5) * ds)
        Rm = R @ (np.eye(3) + 0.5 * ds * hat(k))
        p = p + ds * (Rm @ vm)
        R = R @ (np.eye(3) + ds * hat(k))
        Uu, _, Vv = np.linalg.svd(R); R = Uu @ Vv
        ps.append(p.copy()); Rs.append(R.copy())
    return np.array(ps), np.array(Rs)


def _interp_v(v_grid, S_grid, s):
    """v(s) (线应变, 含轴向与剪切) 在 s 处线性插值; None → 不可伸 e₃."""
    if v_grid is None:
        return np.array([0., 0., 1.])
    j = int(np.clip(np.searchsorted(S_grid, s) - 1, 0, len(S_grid) - 2))
    w = (s - S_grid[j]) / (S_grid[j + 1] - S_grid[j])
    return (1 - w) * v_grid[j] + w * v_grid[j + 1]


def _interp_kappa(kappa_s, S_grid, s):
    """κ 在 s 处线性插值 (与 ik_table.kappa_at 一致; 旧版是最近邻, 有 O(ds·|κ'|) 误差)."""
    j = int(np.clip(np.searchsorted(S_grid, s) - 1, 0, len(S_grid) - 2))
    w = (s - S_grid[j]) / (S_grid[j + 1] - S_grid[j])
    return (1 - w) * kappa_s[j] + w * kappa_s[j + 1]


def integrate_shape_batch(kappa_batch, S_grid, nsub=600, v_batch=None):
    """批量积分骨架: kappa_batch (N,nS,3) → p(s) (N,nsub+1,3), R(s) (N,nsub+1,3,3).

    每步只做 R'=R·hat(κ), p'=R·v, 无 6×6 解、无肌腱几何 —— 这是降阶的核心加速点。
    nsub 默认 600 (=0.75mm 步长) 与建表网格一致; κ 用线性插值而非最近邻。

    ⚠ **v_batch 不能省**: 臂体轴向刚度有限时 v_z 沿弧长偏离 1 (实测 0.85~0.90),
    只用 κ 重建会让骨架凭空长 40~58mm。v_batch (N,nS,3) 同网格; None → v=e₃ (旧行为)。
    """
    N = kappa_batch.shape[0]
    ds = L / nsub
    p = np.zeros((N, 3))
    R = np.tile(np.eye(3), (N, 1, 1))
    # 每个积分子步的中点 → 在 S_grid 上线性插值取 κ (与 v)
    sm = (np.arange(nsub) + 0.5) * ds
    jj = np.clip(np.searchsorted(S_grid, sm) - 1, 0, len(S_grid) - 2)
    w = (sm - S_grid[jj]) / (S_grid[jj + 1] - S_grid[jj])       # (nsub,)
    K = ((1 - w)[None, :, None] * kappa_batch[:, jj, :]
         + w[None, :, None] * kappa_batch[:, jj + 1, :])         # (N, nsub, 3)
    if v_batch is None:
        V = np.zeros((N, nsub, 3)); V[:, :, 2] = 1.0
    else:
        V = ((1 - w)[None, :, None] * v_batch[:, jj, :]
             + w[None, :, None] * v_batch[:, jj + 1, :])
    ps = np.zeros((N, nsub + 1, 3)); ps[:, 0] = p
    Rs = np.zeros((N, nsub + 1, 3, 3)); Rs[:, 0] = R
    I3 = np.eye(3)
    for k in range(nsub):
        Kk = K[:, k, :]                              # (N,3)
        hK = _hat_b(Kk)                              # (N,3,3)
        Rm = R @ (I3[None] + 0.5 * ds * hK)
        p = p + ds * (Rm @ V[:, k, :][..., None])[..., 0]
        R = R @ (I3[None] + ds * hK)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        ps[:, k + 1] = p; Rs[:, k + 1] = R
    return ps, Rs


def _hat_b(V):
    H = np.zeros(V.shape + (3,))
    H[..., 0, 1] = -V[..., 2]; H[..., 0, 2] = V[..., 1]
    H[..., 1, 0] = V[..., 2];  H[..., 1, 2] = -V[..., 0]
    H[..., 2, 0] = -V[..., 1]; H[..., 2, 1] = V[..., 0]
    return H


# ── 几何辅助 (原 ik_table, 纯几何故下沉; 解 tendon_coupling→ik_table 的环) ──
def _L0_wire():
    return np.array([L1 if w in SEG_A else L for w in range(6)])

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


# ══════════════════════════════════════════════════════════════════════
# 批量 (向量化) 内核  (原 sdm_vc_batch.py, 已并入本模型)
#   与上面标量内核**数学完全一致** (RK2 中点法 / 段界锚丝跳跃 / Levenberg 打靶),
#   只是向量化; 供 shape_batch / 建表 / POD 建基 复用。
# ══════════════════════════════════════════════════════════════════════

# ---------------- 批量化小工具 ----------------

_USE_VEC_INTERMED = True      # C 优化: 向量化 _intermed_b (数值等价)


def hat_b(V):
    """V (N,3) → (N,3,3) 反对称矩阵."""
    N = V.shape[0]
    H = np.zeros((N, 3, 3))
    H[:, 0, 1] = -V[:, 2]; H[:, 0, 2] = V[:, 1]
    H[:, 1, 0] = V[:, 2];  H[:, 1, 2] = -V[:, 0]
    H[:, 2, 0] = -V[:, 1]; H[:, 2, 1] = V[:, 0]
    return H


def _wire_r_vec(r_disk):
    """6 根丝的截面半径向量 (6,3)."""
    a = np.radians(ALPHA_DEG)
    return np.stack([r_disk * np.cos(a), r_disk * np.sin(a), np.zeros(6)], axis=1)


def _safe_solve6(M, rhs):
    """批量解 6×6; 奇异/非有限样本退回单位增量 (不炸整批)."""
    N = M.shape[0]
    good = np.isfinite(M).all(axis=(1, 2)) & np.isfinite(rhs).all(axis=1)
    out = np.zeros((N, rhs.shape[1]))
    if good.any():
        try:
            out[good] = np.linalg.solve(M[good], rhs[good][..., None])[..., 0]
        except np.linalg.LinAlgError:
            for i in np.where(good)[0]:
                try:
                    out[i] = np.linalg.solve(M[i], rhs[i])
                except np.linalg.LinAlgError:
                    out[i] = 0.0
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, -1e3, 1e3)


def _orthonormalize_b(R):
    """R (N,3,3) 逐样本正交化. 用代数 (Gram-Schmidt), 不用 SVD —— SVD 是批量路径
    的主要开销 (每子段一次 3×3 SVD × N 样本), 代数化后快一个量级。
    """
    a0 = R[:, :, 0]
    a1 = R[:, :, 1]
    n0 = np.linalg.norm(a0, axis=1, keepdims=True)
    n0 = np.where(n0 < 1e-12, 1.0, n0)
    b0 = a0 / n0
    b1 = a1 - (np.sum(a1 * b0, axis=1, keepdims=True)) * b0
    n1 = np.linalg.norm(b1, axis=1, keepdims=True)
    n1 = np.where(n1 < 1e-12, 1.0, n1)
    b1 = b1 / n1
    b2 = np.cross(b0, b1)
    out = np.stack([b0, b1, b2], axis=2)
    # 非有限样本退回单位阵 (防刚性发散)
    good = np.isfinite(out).all(axis=(1, 2))
    if not good.all():
        out[~good] = np.eye(3)
    return out


# ---------------- 分布几何项 (批量) ----------------

def _intermed_b(u, v, Kse, Kbt, wires, tau, ri_all, fe_b):
    """照搬 sdm_vc.intermedquant, 批量. Kse/Kbt:(N,3,3), tau:(N,6), fe_b:(N,3)."""
    N = v.shape[0]
    A = np.zeros((N, 3, 3)); B = np.zeros((N, 3, 3))
    G = np.zeros((N, 3, 3)); H = np.zeros((N, 3, 3))
    a = np.zeros((N, 3)); b = np.zeros((N, 3))
    uh = hat_b(u)                                   # (N,3,3)
    for i in wires:
        ri = np.broadcast_to(ri_all[i], (N, 3))
        rih = np.broadcast_to(_hat3(ri_all[i]), (N, 3, 3))
        uri = (uh @ ri[..., None])[..., 0]           # (N,3)
        pb = uri + v
        nrm = np.linalg.norm(pb, axis=1)
        nrm = np.maximum(nrm, 1e-12)
        pbhat = hat_b(pb)
        pbhat2 = pbhat @ pbhat
        coef = -(tau[:, i] / nrm ** 3)
        Ai = coef[:, None, None] * pbhat2
        Bi = rih @ Ai
        A += Ai; B += Bi
        G -= Ai @ rih
        H -= Bi @ rih
        ai = (Ai @ (uh @ pb[..., None]))[..., 0]
        a += ai
        b += (rih @ ai[..., None])[..., 0]
    vv = v - _E3[None, :]
    uhKbtu = (uh @ (Kbt @ u[..., None]))[..., 0]
    vhKsev = (hat_b(v) @ (Kse @ vv[..., None]))[..., 0]
    uhKsev = (uh @ (Kse @ vv[..., None]))[..., 0]
    c = -uhKbtu - vhKsev - fe_b - b
    d = -uhKsev - fe_b - a
    return A, B, G, H, c, d


def _hat_all(V):
    """V (...,3) → (...,3,3) 反对称矩阵 (批量, 无 python 循环)."""
    V = np.asarray(V, float)
    H = np.zeros(V.shape + (3,))
    H[..., 0, 1] = -V[..., 2]; H[..., 0, 2] = V[..., 1]
    H[..., 1, 0] = V[..., 2];  H[..., 1, 2] = -V[..., 0]
    H[..., 2, 0] = -V[..., 1]; H[..., 2, 1] = V[..., 0]
    return H


def _intermed_b_vec(u, v, Kse, Kbt, wires, tau, ri_all, fe_b):
    """`_intermed_b` 的**向量化版** (丝循环 → (N,W) 批量).

    ⚡ 性能: 原版对 6 根丝逐丝做 ~15 次小 numpy 运算 (剖析显示占 `forward_batch`
      累计耗时 **60%**, 且引发 32k 次 broadcast_to / 27k 次 hat_b 调用)。
    本版把丝维折叠成数组轴 (einsum/批量 matmul), 数学**完全一致**。

    ⚠ 形参序与 `_intermed_b` **一致**: (u, v, ...)。
    与原版的唯一差别是浮点求和次序 (误差 ~1e-15)。
    """
    RI = np.asarray(ri_all, float)[list(wires)]              # (W,3)
    RIH = _hat_all(RI)                                       # (W,3,3) — 只算一次
    uh = hat_b(u)                                            # (N,3,3)
    uri = np.einsum('nij,wj->nwi', uh, RI)                   # (N,W,3)
    pb = uri + v[:, None, :]                                 # (N,W,3)
    nrm = np.maximum(np.linalg.norm(pb, axis=2), 1e-12)      # (N,W)
    P = _hat_all(pb)                                         # (N,W,3,3)
    P2 = P @ P                                               # (N,W,3,3)
    coef = -(np.asarray(tau, float)[:, list(wires)] / nrm ** 3)
    Ai = coef[:, :, None, None] * P2                         # (N,W,3,3)
    Bi = np.einsum('wij,nwjk->nwik', RIH, Ai)                # rih @ Ai
    A = Ai.sum(axis=1); B = Bi.sum(axis=1)
    G = -np.einsum('nwij,wjk->nik', Ai, RIH)
    H = -np.einsum('nwij,wjk->nik', Bi, RIH)
    uhpb = np.einsum('nij,nwj->nwi', uh, pb)                 # (N,W,3)
    ai = np.einsum('nwij,nwj->nwi', Ai, uhpb)                # (N,W,3)
    a = ai.sum(axis=1)
    b = np.einsum('wij,nwj->nwi', RIH, ai).sum(axis=1)
    vv = v - _E3[None, :]
    uhKbtu = (uh @ (Kbt @ u[..., None]))[..., 0]
    vhKsev = (hat_b(v) @ (Kse @ vv[..., None]))[..., 0]
    uhKsev = (uh @ (Kse @ vv[..., None]))[..., 0]
    c = -uhKbtu - vhKsev - fe_b - b
    d = -uhKsev - fe_b - a
    return A, B, G, H, c, d


def _hat3(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def _anchor_b(u, v, R, anchor, tau, ri_all):
    """锚丝终止集中力/矩(空间系), 批量 → F(N,3), Lm(N,3)."""
    N = v.shape[0]
    F = np.zeros((N, 3)); Lm = np.zeros((N, 3))
    uh = hat_b(u)
    for i in anchor:
        ri = np.broadcast_to(ri_all[i], (N, 3))
        pb = (uh @ ri[..., None])[..., 0] + v
        psp = (R @ pb[..., None])[..., 0]
        nrm = np.maximum(np.linalg.norm(psp, axis=1), 1e-12)
        Rri = (R @ ri[..., None])[..., 0]
        Rrih = hat_b(Rri)
        Rrih_psp = (Rrih @ psp[..., None])[..., 0]
        F -= tau[:, i][:, None] * psp / nrm[:, None]
        Lm -= tau[:, i][:, None] * Rrih_psp / nrm[:, None]
    return F, Lm


# ---------------- 批量前向 ----------------

def forward_batch(x, tau, params, F_tip=None, want_curve=False):
    """x (N,6) = [v0,u0]; tau (N,6). 返回 res (N,6 或 N,7), tip (N,3) [, 曲线].

    ⚡ 默认走 **numba JIT 版** (`_jit_integrate.integrate`), 实测 **12.1×** 提速
      (4.1 ms vs 49.4 ms @ nsub=61; S2 对拍 8 个量 max|Δ| ≤ 1.7e-13)。
    以下情况自动回退到 numpy 参考实现 `forward_batch_ref`:
      · `F_tip is not None` (JIT 版暂不支持端部外力)
      · JIT/import 出错, 或设环境变量 `TC_NO_JIT=1`
    """
    global _JIT_IMPORT_ERROR, _USE_JIT
    if _USE_JIT and F_tip is None:
        try:
            from _jit_integrate import integrate          # 延迟导入 (避免循环依赖)
            return integrate(x, tau, params, want_curve=want_curve)
        except Exception as _e:
            # ★ 2026-09-28: 原来这里是 `pass` —— 静默掉 10x 以上, 极难排查。
            _JIT_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
            _USE_JIT = False
            _warnings.warn(
                "[tendon_coupling] JIT 调用失败, 本次起回退纯 numpy 参考实现 "
                f"(慢 10x 以上)。原因: {_JIT_IMPORT_ERROR}",
                RuntimeWarning, stacklevel=2)
    return forward_batch_ref(x, tau, params, F_tip=F_tip, want_curve=want_curve)


def forward_batch_ref(x, tau, params, F_tip=None, want_curve=False):
    """numpy 参考实现 (原 `forward_batch`), 供对拍与回退用.

    params.rod_constraint=True 时 res 为 (N,7): 末列是中心杆弧长约束残差。
    """
    x = np.asarray(x, float); tau = np.asarray(tau, float)
    if x.ndim == 1:
        x = x[None, :]
    if tau.ndim == 1:
        tau = tau[None, :]
    N = x.shape[0]
    v = x[:, :3].copy(); u = x[:, 3:].copy()
    R = np.tile(np.eye(3), (N, 1, 1))
    p = np.zeros((N, 3))
    ri_all = _wire_r_vec(params.r_disk)
    rho_total = params.rhoA + params.disk_mass / L
    fe_w = np.zeros((N, 3)); fe_w[:, 2] = -rho_total * 9.81
    rod_on = bool(getattr(params, "rod_constraint", False))
    Lz = np.zeros(N)                                  # 中心杆累计弧长 ∫v_z ds
    lat = getattr(params, "lattice", False)           # 菱形晶胞 EI(κ)
    L_strut = getattr(params, "L_strut", L_STRUT_DEFAULT)
    C0 = getattr(params, "C0", C0_DEFAULT)
    # ── 分段晶胞 (段A/段B 各一种菱形胞): 每段刚度取本段晶胞;
    #    无 cells 配置时退回旧的"段A 并联 ×2"规则 (向后兼容) ──
    _cells = getattr(params, "cells", None)

    def _seg(cn):
        """第 cn 段 (k=0 → 段A) 的刚度来源: (EI0, C0, L_strut, GJ, k_rod, GA)."""
        c = (_cells[0] if cn == 0 else _cells[-1]) if _cells else None
        if c is not None:
            return (c.EI0, c.C0, c.L_strut, c.GJ, c.k_rod, params.GA)
        m = 2.0 if cn == 0 else 1.0
        return (m * params.EI, C0, L_strut, m * params.GJ,
                m * params.k_rod, m * params.GA)

    def _ei_of(uu):
        """菱形晶胞割线刚度 (N,) —— EI_eq=EI0·sinc(|κ|·L_strut), κ→0 时 → EI0."""
        if not lat:
            return np.full(N, params.EI)
        th = np.linalg.norm(uu[:, :2], axis=1) * L_strut
        ei0 = C0 * L_strut
        return ei0 * np.where(th < 1e-9, 1.0, np.sin(th) / np.where(th < 1e-9, 1.0, th))

    if want_curve:
        nsub_tot = params.n_p + params.n_d
        S = np.linspace(0, L, nsub_tot + 1)
        Pc = np.zeros((N, nsub_tot + 1, 3)); Pc[:, 0] = 0
        Uc = np.zeros((N, nsub_tot + 1, 3)); Uc[:, 0] = u
        Cc = np.zeros((N, nsub_tot + 1, 3)); Nc = np.zeros((N, nsub_tot + 1, 3))
        Vvf = np.zeros((N, nsub_tot + 1, 3)); Vvf[:, 0] = v
        Rc = np.zeros((N, nsub_tot + 1, 3, 3)); Rc[:, 0] = R
        # s=0 处在段A: 补齐首节点内力矩/内力 (否则是零哨兵, 会被下游误当真实曲线起点)
        # ⚠ 刚度必须与 _kbt 同源 (lattice 时 EI 随曲率变), 用常量会与后续节点不一致
        aEI, aC0, aLs, aGJ, aKR, aGA = _seg(0)
        if lat:
            th0 = np.linalg.norm(u[:, :2], axis=1) * aLs
            e0 = np.where(th0 < 1e-9, aEI,
                          aEI * np.sin(th0) / np.where(th0 < 1e-9, 1.0, th0))
        else:
            e0 = np.full(N, aEI)
        Cc[:, 0] = np.stack([e0 * u[:, 0], e0 * u[:, 1], aGJ * u[:, 2]], axis=1)
        Nc[:, 0] = np.stack([aGA * v[:, 0], aGA * v[:, 1],
                             aKR * (v[:, 2] - 1.0)], axis=1)
        idx = 1

    for k in (0, 1):
        Lseg = L1 if k == 0 else L2
        nsub = params.n_p if k == 0 else params.n_d
        sEI, sC0, sLs, sGJ, sKR, sGA = _seg(k)       # 本段刚度来源
        Kse = np.tile(np.diag([sGA, sGA, sKR]), (N, 1, 1))
        wires = list(range(6)) if k == 0 else list(SEG_B)
        ds = Lseg / nsub

        def _kbt(uu):
            """按当前曲率算弯曲刚度矩阵 (N,3,3). lattice=False 时退化为常量。"""
            if lat:
                th = np.linalg.norm(uu[:, :2], axis=1) * sLs
                e = np.where(th < 1e-9, sEI,
                             sEI * np.sin(th) / np.where(th < 1e-9, 1.0, th))
            else:
                e = np.full(N, sEI)
            M = np.zeros((N, 3, 3))
            M[:, 0, 0] = M[:, 1, 1] = e
            M[:, 2, 2] = sGJ
            return M

        for _ in range(nsub):
            Kbt = _kbt(u)
            fe_b = (np.einsum('nji,nj->ni', R, fe_w))
            _im = _intermed_b_vec if _USE_VEC_INTERMED else _intermed_b
            A, B, G, H, c, d = _im(u, v, Kse, Kbt, wires, tau, ri_all, fe_b)
            M = np.zeros((N, 6, 6))
            M[:, :3, :3] = Kse + A; M[:, :3, 3:] = G
            M[:, 3:, :3] = B;       M[:, 3:, 3:] = Kbt + H
            rhs = np.concatenate([d, c], axis=1)
            vu = _safe_solve6(M, rhs)
            dv, du = vu[:, :3], vu[:, 3:]
            v_m = v + 0.5 * ds * dv; u_m = u + 0.5 * ds * du
            Kbt_m = _kbt(u_m)
            fe_b2 = (np.einsum('nji,nj->ni', R, fe_w))
            A2, B2, G2, H2, c2, d2 = _im(u_m, v_m, Kse, Kbt_m, wires, tau,
                                          ri_all, fe_b2)
            M2 = np.zeros((N, 6, 6))
            M2[:, :3, :3] = Kse + A2; M2[:, :3, 3:] = G2
            M2[:, 3:, :3] = B2;       M2[:, 3:, 3:] = Kbt_m + H2
            rhs2 = np.concatenate([d2, c2], axis=1)
            vu2 = _safe_solve6(M2, rhs2)
            Lz += ds * v_m[:, 2]                      # 中心杆弧长 (RK2 中点)
            v = v + ds * vu2[:, :3]
            u = u + ds * vu2[:, 3:]
            v = np.clip(v, -3.0, 3.0)                 # 刚性发散防护
            u = np.clip(u, -80.0, 80.0)
            # p, R 更新
            Rmid = R @ (np.eye(3) + 0.5 * ds * hat_b(u_m))
            p = p + ds * (Rmid @ v_m[..., None])[..., 0]
            R = _orthonormalize_b(R @ (np.eye(3) + ds * hat_b(u)))
            if want_curve:
                Pc[:, idx] = p; Uc[:, idx] = u; Rc[:, idx] = R
                # ⚠ C 必须与 Uc 取**同一节点**(端点 u), 用中点 u_m 会引入
                #   O(ds·κ') ≈ 1e-2 的不一致 ⇒ 下游 kcoef 的 κ 反算偏差
                Cc[:, idx] = (_kbt(u) @ u[..., None])[..., 0]
                Nc[:, idx] = (Kse @ (v - _E3[None, :])[..., None])[..., 0]
                Vvf[:, idx] = v
                idx += 1
        if k == 0:
            Fs, Ls = _anchor_b(u, v, R, SEG_A, tau, ri_all)
            RFs = np.einsum('nji,nj->ni', R, Fs)
            RLs = np.einsum('nji,nj->ni', R, Ls)
            v = v - np.linalg.solve(Kse, RFs[..., None])[..., 0]
            u = u - np.linalg.solve(_kbt(u), RLs[..., None])[..., 0]

    n_e = (R @ (Kse @ (v - _E3[None, :])[..., None]))[..., 0]
    m_e = (R @ (_kbt(u) @ u[..., None]))[..., 0]
    Fs, Ls = _anchor_b(u, v, R, SEG_B, tau, ri_all)
    ft = np.zeros((N, 3)) if F_tip is None else np.tile(np.asarray(F_tip, float), (N, 1))
    res = np.concatenate([n_e - Fs - ft, m_e - Ls], axis=1)
    if rod_on:
        # 中心杆串联弹簧: 轴力沿全长恒定 ⇒ 弧长 Lz 应精确等于无应力长 L_rod。
        res = np.concatenate(
            [res, (Lz / params.rod_free_len - 1.0)[:, None]], axis=1)
    if want_curve:
        return res, p, (Pc, S, Uc, Cc, Rc, Nc, Vvf)
    return res, p


# ---------------- 批量 Levenberg 打靶 ----------------

def shooting_batch(taus, params, x0=None, tol=1e-8, max_iter=15, lam0=1e-3,
                   v_bound=3.0, u_bound=60.0):
    """批量打靶. taus (N,6) → 每样本 {x=[v0,u0], res, converged, n_iter, tip}.

    刚性系统下部分 τ 组合牛顿步会爆掉: 对状态 x 做盒约束 (v 近 E3, u 曲率量级),
    越界只做钳位不判死 —— 只要残差下降就接受; 仅 NaN/Inf 才标 failed。
    """
    taus = np.atleast_2d(np.asarray(taus, float))
    N = taus.shape[0]
    lo = -np.array([v_bound]*3 + [u_bound]*3)
    hi = np.array([v_bound]*3 + [u_bound]*3)
    x = (np.tile(np.array([0., 0., 1., 0., 0., 0.]), (N, 1)) if x0 is None
         else np.atleast_2d(np.asarray(x0, float)).copy())
    x = np.clip(x, lo, hi)
    lam = np.full(N, lam0)
    res, tip = forward_batch(x, taus, params)
    converged = np.zeros(N, bool)
    n_iter = np.zeros(N, int)
    failed = ~np.isfinite(res).all(axis=1)          # 初始就非有限 → 直接失败
    for it in range(max_iter):
        active = ~converged & ~failed
        if not active.any():
            break
        # ★ 2026-10-05: 雅可比只对**活跃行**求 (已收敛/已失败的行不再参与迭代, 算它们是纯浪费)。
        #   每轮原本对全部 N 行 × 6 分量做 ±h 中心差分 = 12 次全量正解;
        #   现在只对 na 行做 ⇒ 12 次**子集**正解。数值与原实现逐位一致:
        #   同一行的 J/dx/improved 只依赖该行自身的 x/res/lam, 与其它行无关。
        idx = np.flatnonzero(active)
        na = idx.size
        xa = x[idx]; ta = taus[idx]; ra = res[idx]
        nx = res.shape[1]
        J = np.zeros((na, nx, 6)); h = 1e-6
        if _JAC_MODE == "forward":
            # 前向差分: 6 次正解 (中心差分要 12 次)。雅可比精度降到 O(h),
            # 但 LM 只用它定下降方向 ⇒ 实测收敛结果一致, 迭代次数略增或无变化。
            xp = xa.copy()
            for c in range(6):
                xp[:, c] = xa[:, c] + h
                rp, _ = forward_batch(xp, ta, params)
                J[:, :, c] = (rp - ra) / h
                xp[:, c] = xa[:, c]          # 复位该列 (复用缓冲)
        else:
            xp = xa.copy(); xm = xa.copy()               # 复用缓冲, 不再每分量 copy 两次
            for c in range(6):
                xp[:, c] = xa[:, c] + h
                xm[:, c] = xa[:, c] - h
                rp, _ = forward_batch(xp, ta, params)
                rm, _ = forward_batch(xm, ta, params)
                J[:, :, c] = (rp - rm) / (2 * h)
                # ⚠ 必须复位该列, 否则下一分量会带上本轮扰动 (复用缓冲的经典坑)
                xp[:, c] = xa[:, c]
                xm[:, c] = xa[:, c]
        J = np.where(np.isfinite(J), J, 0.0)
        H = np.einsum('nki,nkj->nij', J, J)
        H += lam[idx][:, None, None] * np.eye(6)[None]
        g = np.einsum('nki,nk->ni', J, ra)
        try:
            dx = -np.linalg.solve(H, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            dx = -np.einsum('nij,nj->ni', np.linalg.pinv(H), g)
        dx = np.where(np.isfinite(dx), dx, 0.0)
        mx = np.max(np.abs(dx), axis=1)
        scale = np.where(mx > 0.5, 0.5 / np.maximum(mx, 1e-30), 1.0)
        x_new = np.clip(xa + dx * scale[:, None], lo, hi)
        res_new, tip_new = forward_batch(x_new, ta, params)
        bad = ~np.isfinite(res_new).all(axis=1)
        n_old = np.linalg.norm(ra, axis=1)
        n_new = np.linalg.norm(res_new, axis=1)
        improved = (n_new < n_old) & ~bad
        lam[idx] = np.where(improved, np.maximum(lam[idx] / 5.0, 1e-10),
                            np.minimum(lam[idx] * 10.0, 1e9))
        x[idx[improved]] = x_new[improved]
        res[idx[improved]] = res_new[improved]
        tip[idx[improved]] = tip_new[improved]
        failed[idx[bad]] = True
        # 残差列数随 rod_constraint 变 (6 或 7); 用前 6 列的力/矩范数判收敛,
        # 否则杆弧长残差 (量纲不同) 会把收敛判据整体拉偏。
        n_now = np.linalg.norm(np.where(improved[:, None], res_new, ra)[:, :6],
                               axis=1)
        conv_now = n_now < tol
        # n_iter 语义: 首次判收敛时所花的迭代次数 (it 从 0 起 ⇒ it+1)
        n_iter[idx[conv_now & ~converged[idx]]] = it + 1
        converged[idx] = converged[idx] | conv_now
    inf = np.linalg.norm(np.where(np.isfinite(res), res, np.inf), axis=1)
    return {"x": x, "res": res, "tip": tip, "converged": converged,
            "n_iter": n_iter, "inf": inf, "failed": failed}


def shooting_batch_retry(taus, params, tol=1e-8, max_iter=15, n_try=3):
    """批量打靶 + 多起点重试: 对未收敛样本换初值再批量跑 (比标量兜底快得多).

    起点 0: 直臂 v0=E3; 起点 1/2: 由 τ 幅值粗估的 v0 (小张力近直臂, 大张力强弯)。
    """
    out = shooting_batch(taus, params, tol=tol, max_iter=max_iter)
    for t in range(1, n_try):
        bad = ~out["converged"]
        if not bad.any():
            break
        idx = np.where(bad)[0]
        x0 = np.tile(np.array([0., 0., 1., 0., 0., 0.]), (len(idx), 1))
        # 粗估: 弯曲越大 v0 越偏离 E3
        mag = np.linalg.norm(taus[idx], axis=1)
        tilt = np.clip(mag * 0.05, 0.0, 0.6) * (t / n_try)
        x0[:, 0] = tilt * np.sign(taus[idx][:, 0] - taus[idx][:, 3])
        x0[:, 1] = tilt * np.sign(taus[idx][:, 1] - taus[idx][:, 4])
        sub = shooting_batch(taus[idx], params, x0=x0, tol=tol, max_iter=max_iter)
        out["x"][idx] = sub["x"]; out["res"][idx] = sub["res"]
        out["tip"][idx] = sub["tip"]; out["converged"][idx] = sub["converged"]
        out["inf"][idx] = sub["inf"]; out["failed"][idx] = sub["failed"]
    return out


__all__ = [
    "TendonCouplingModel", "TendonCouplingResult", "LatticeCell",
    "SDMParams",
    "L1", "L2", "L", "SEG_A", "SEG_B", "ALPHA_DEG",
    "wire_r", "intermedquant", "anchor_loads", "forward", "shooting",
    "tendon_path_length",
    "cell_stiffness", "cell_alpha", "cell_n_long", "EI_lattice",
    "E_MAT_DEFAULT", "A_STRUT_DEFAULT", "N_CIRC_DEFAULT", "ALPHA_CELL_DEG",
    "_E3", "hat", "integrate_shape", "integrate_shape_batch",
    "_L0_wire", "_wire_geom_len_batch",
    "forward_batch", "shooting_batch", "shooting_batch_retry",
]


# ══════════════════════════════════════════════════════════════════════
# §0  结果容器
# ══════════════════════════════════════════════════════════════════════
class TendonCouplingResult(dict):
    """带属性访问的 dict, 便于 out.tip / out.dL 直读."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __repr__(self):
        keys = ", ".join(self.keys())
        return f"TendonCouplingResult({keys})"


# N_CIRC_SEG_A / N_CIRC_SEG_B 定义见文件头部「菱形晶胞几何参数」(两段的 L_strut 不同)
#
# ⚠ 分段后 N_long 的取整误差 (方案 A: 保持设计值 α=12°, 取整接受):
#     段A: 225mm ÷ l_ax_A 4.452mm = 50.542 → N_long = 50
#     段B: 225mm ÷ l_ax_B 8.904mm = 25.271 → N_long = 25
#     ⇒ 胞阵长 L1 = 50·4.452 = 222.59mm, L2 = 25·8.904 = 222.59mm (各少 2.41mm, 1.07%)
#     ⇒ k_rod 相应偏低 1.07% (恒等式仍用 L_eff, 故 EI0/k_rod = 2r²L_eff 精确成立)
#   若要严格几何自洽, 可传 alpha_deg=None 由约束②(N_long·l_ax = L) 反解,
#   代价是 α 偏离设计值且 EI0 ∝ sin³α 随之变化。
#   注: 臂长现由晶胞推出 (L_g = N_long·l_ax), 故 L_eff ≡ L_g, 取整缺口已被消除;
#       本注释保留以说明"若强行用固定段长 225mm"会引入 1.07% 误差。

# ══════════════════════════════════════════════════════════════════════
# §0.5  菱形晶胞骨架 (Diamond Lattice Cell) — 模型**唯一**的刚度来源
# ══════════════════════════════════════════════════════════════════════
class LatticeCell:
    """菱形晶胞: 材料/几何 → 全套刚度. 弯曲与轴向**同源**, 改一处即同步.

    几何
    ----
        斜杆长    L_strut = π·r_i / (N_circ·cosα)
        胞轴向长  l_ax    = 2·L_strut·sinα
        纵向胞数  N_long  = L / l_ax              (串联)
        环向胞数  N_circ                          (并联)
        截面 Σy²  = 2·N_circ·r_i²                 (不含 N_long)

    刚度 (推导见 sdm_vc 头部注释 / 笔记《曲率分布函数κ(s)》§4.2)
    ----
        单胞轴向  k_unit = 2·E·A·sin²α / L_strut
        弯曲      EI0    = k_unit·l_ax·Σy² = 4·E·A·sin³α·Σy²
        轴向      k_rod  = N_circ·k_unit/N_long = N_circ·4·E·A·sin³α/L
        弯矩常数  C0     = EI0 / L_strut
        抗扭      GJ     = EI0 (圆截面基准)

    ★ 关键恒等式
    ----
        EI0 / k_rod = 2·r_i²·L      (α, E, A, N_circ, N_long **全部约掉**)
      ⇒ 弯曲/轴向的**相对软硬**只由 (r_i, L) 决定, 等价 φ/ε = L/(2·r_i)。
      ⇒ α 是**唯一**的整体刚度旋钮 (EI0 ∝ sin³α)。

    弯曲非线性 (割线刚度)
    ----
        弯矩本构  C(κ)     = C0·sin(κ·L_strut)
        割线刚度  EI_eq(κ) = C(κ)/κ = EI0·sinc(κ·L_strut)
        切向刚度  dC/dκ    = EI0·cos(κ·L_strut)
        κ→0 时两者都 → EI0 (有限非零: 直臂仍有抗弯刚度)。
      本模型实测工况 κ_max≈0.8 1/m ⇒ θ≈0.016 ⇒ EI 只降 0.004% (等于常量)。
    """

    def __init__(self, E=E_MAT_DEFAULT, A=A_STRUT_DEFAULT, N_circ=N_CIRC_DEFAULT,
                 r_i=R_DISK_DEFAULT, alpha_deg=ALPHA_CELL_DEG, N_long=None,
                 L=None):
        self.E = float(E); self.A = float(A)
        self.N_circ = int(N_circ); self.r_i = float(r_i)
        self.alpha_deg = alpha_deg
        self.L_arm = float(L_TOTAL if L is None else L)
        self._N_long_in = N_long
        self._recompute()

    def _recompute(self):
        d = cell_stiffness(E=self.E, A=self.A, N_circ=self.N_circ,
                           N_long=self._N_long_in, r_i=self.r_i,
                           L=self.L_arm, alpha_deg=self.alpha_deg)
        self.k_unit = d["k_unit"]; self.alpha = d["alpha"]
        self.L_strut = d["L_strut"]; self.l_ax = d["l_ax"]
        self.sum_y2 = d["sum_y2"]; self.N_long = d["N_long"]
        self.EI0 = d["EI0"]; self.C0 = d["C0"]; self.k_rod = d["k_rod"]
        self.GJ = self.EI0
        self.eps_lock = d["eps_lock"]

    # ── 派生关系 ──────────────────────────────────────────────
    @property
    def ratio_EI_krod(self):
        return self.EI0 / self.k_rod

    @property
    def L_eff(self):
        """有效臂长 = N_long·l_ax (与 L 的差别仅来自 N_long 取整)."""
        return float(self.N_long) * float(self.l_ax)

    @property
    def ratio_expected(self):
        """EI0/k_rod 的解析值 = 2·r_i²·L_eff.

        ⚠ 严格恒等式 `EI0/k_rod = 2·r_i²·L` 只在 `N_long = L/l_ax` **整取**时成立;
        实际 N_long 取整后应换成 L_eff = N_long·l_ax。
        """
        return 2.0 * self.r_i ** 2 * self.L_eff

    @property
    def ratio_expected_ideal(self):
        """理想值 2·r_i²·L (未计 N_long 取整)."""
        return 2.0 * self.r_i ** 2 * self.L_arm

    @property
    def phi_over_eps(self):
        """弯曲/轴向柔度比 = L/(2·r_i)."""
        return self.L_arm / (2.0 * self.r_i)

    # ── 弯曲本构 ──────────────────────────────────────────────
    def _kxy(self, kappa):
        a = np.ravel(np.asarray(kappa, float))
        return float(np.hypot(a[0], a[1])) if a.size > 1 else float(a[0])

    def moment(self, kappa):
        """弯矩 C(κ) = C0·sin(κ·L_strut)."""
        return self.C0 * np.sin(self._kxy(kappa) * self.L_strut)

    def EI_secant(self, kappa):
        """割线刚度 EI_eq(κ) = EI0·sinc(κ·L_strut)."""
        th = self._kxy(kappa) * self.L_strut
        return self.EI0 if th < 1e-9 else self.EI0 * np.sin(th) / th

    def EI_tangent(self, kappa):
        """切向刚度 dC/dκ = EI0·cos(κ·L_strut)."""
        return self.EI0 * np.cos(self._kxy(kappa) * self.L_strut)

    def EI_curve(self, kappa_max=40.0, n=101):
        """(κ_grid, EI_eq) 曲线, 便于画图/查表."""
        k = np.linspace(0.0, float(kappa_max), int(n))
        return k, self.EI0 * np.sinc(k * self.L_strut / np.pi)

    def kappa_soften(self, drop=0.05):
        """EI 相对 EI0 下降 drop 时的曲率 (线性域边界)."""
        return float(np.arccos(1.0 - float(drop)) / self.L_strut)

    # ── 与 SDMParams 接口 ─────────────────────────────────────
    def sd_params(self, EI=None, GA=1e3, rhoA=0.0, disk_mass=0.0,
                  n_p=30, n_d=30, lattice=True, **kw):
        """→ SDMParams. EI 缺省取 EI0 (lattice=True 时按当前曲率修正)."""
        return SDMParams(EI=self.EI0 if EI is None else EI, GA=GA,
                         k_rod=self.k_rod, GJ=self.GJ,
                         r_disk=self.r_i, rhoA=rhoA, disk_mass=disk_mass,
                         n_p=n_p, n_d=n_d, lattice=lattice,
                         C0=self.C0, L_strut=self.L_strut, **kw)

    def with_alpha(self, alpha_deg):
        """按新 α 重建 —— α 是唯一刚度旋钮 (EI0, k_rod ∝ sin³α)."""
        return LatticeCell(self.E, self.A, self.N_circ, self.r_i,
                           alpha_deg, self._N_long_in, self.L_arm)

    def with_params(self, **kw):
        p = dict(E=self.E, A=self.A, N_circ=self.N_circ, r_i=self.r_i,
                 alpha_deg=self.alpha_deg, N_long=self._N_long_in, L=self.L_arm)
        p.update(kw)
        return LatticeCell(**p)

    def stiffness_scaling(self, alpha_grid_deg=(6, 9, 12, 15, 18, 24, 30), ref_deg=12.0):
        """α 扫描: EI0 / k_rod / 二者比值 随 α 的变化 (验证 ∝sin³α)."""
        ref = self.with_alpha(ref_deg)
        rows = []
        for ad in alpha_grid_deg:
            c = self.with_alpha(ad)
            rows.append(dict(alpha_deg=ad, EI0=c.EI0, k_rod=c.k_rod,
                             EI_krod=c.ratio_EI_krod,
                             EI0_norm=c.EI0 / ref.EI0,
                             sin3=np.sin(c.alpha) ** 3 / np.sin(ref.alpha) ** 3))
        return rows

    def info(self):
        return dict(E=self.E, A=self.A, N_circ=self.N_circ, r_i=self.r_i,
                    alpha_deg=np.degrees(self.alpha), L_strut=self.L_strut,
                    l_ax=self.l_ax, N_long=self.N_long, sum_y2=self.sum_y2,
                    k_unit=self.k_unit, EI0=self.EI0, C0=self.C0,
                    k_rod=self.k_rod, GJ=self.GJ, eps_lock=self.eps_lock,
                    ratio_EI_krod=self.ratio_EI_krod,
                    ratio_expected=self.ratio_expected,
                    phi_over_eps=self.phi_over_eps,
                    kappa_5pct=self.kappa_soften(0.05))

    def report(self):
        o = self.info()
        print("─" * 66)
        print("菱形晶胞骨架 (Diamond Lattice Cell) — 唯一刚度来源")
        print("─" * 66)
        print(f"  材料/几何 : E={o['E']:.4g} Pa, A={o['A']:.3g} m², "
              f"N_circ={o['N_circ']}, r_i={o['r_i']*1e3:.1f} mm, L={self.L_arm*1e3:.1f} mm")
        print(f"  胞几何    : α={o['alpha_deg']:.2f}°, L_strut={o['L_strut']*1e3:.3f} mm, "
              f"l_ax={o['l_ax']*1e3:.3f} mm, N_long={o['N_long']}")
        print(f"  刚度      : k_unit={o['k_unit']:.4g} N/m")
        print(f"              EI0={o['EI0']:.4f} N·m²   C0={o['C0']:.4f} N·m²   "
              f"k_rod={o['k_rod']:.4g} N/m   GJ={o['GJ']:.4f} N·m²")
        print(f"  恒等式    : EI0/k_rod={o['ratio_EI_krod']:.6g}  "
              f"vs 2·r_i²·L_eff={o['ratio_expected']:.6g}   (φ/ε={o['phi_over_eps']:.3f})")
        print(f"  非线性    : EI_eq(κ)=EI0·sinc(κ·L_strut),  κ(降5%)={o['kappa_5pct']:.1f} 1/m")
        print(f"  锁死应变  : ε_lock={o['eps_lock']*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════
# §1  肌腱力耦合模型
# ══════════════════════════════════════════════════════════════════════
class TendonCouplingModel:
    """全阶 Cosserat + 肌腱几何 的统一模型.

    参数
    ----
    params : SDMParams | None      结构刚度/几何; None → 按 cell 推出 (或 ik_table.default_params())
    cell   : LatticeCell | None    菱形晶胞; 给了它就**完全由晶胞决定刚度**
    F_tip  : (3,)                  末端外载 (空间系, N)
    g_world: (3,)                  重力 (世界系); 传 (0,0,0) 关掉
    R_mount: (3,3)                 基座姿态
    lattice: bool                  弯曲刚度是否随曲率按晶胞非线性修正
    n_p/n_d: int                   段A/段B 积分子步数
    rhoA   : float                 线密度 (kg/m), 用于重力
    disk_mass : float              均布盘质量 (kg)
    h_tau  : float                 耦合矩阵数值差分步长 (N)
    tol    : float                 打靶收敛容差
    max_iter : int                 打靶迭代上限
    cache  : bool                  缓存 (tau → 解), 建耦合矩阵时省一半计算
    """

    def __init__(self, params=None, F_tip=(0, 0, 0), g_world=(0, 0, -9.81),
                 R_mount=None, h_tau=1e-3, tol=1e-9, max_iter=60, cache=True,
                 cell=None, cells=None, lattice=True, n_p=30, n_d=30,
                 rhoA=0.0, disk_mass=0.0, GA=1e3):
        # ── 分段晶胞: 段A(近基座, 铺满 L1) 与 段B(铺满 L2) 各一种菱形胞 ──
        if cells is not None:
            self.cells = tuple(cells)
        elif cell is not None:
            self.cells = (cell, cell)
        else:
            self.cells = (LatticeCell(N_circ=N_CIRC_SEG_A, L=L1),
                          LatticeCell(N_circ=N_CIRC_SEG_B, L=L2))
        self.cell = self.cells[0]
        if params is not None:
            self.params = params
            if getattr(params, "cells", None) is None:
                params.cells = self.cells
        else:
            self.params = self.cell.sd_params(GA=GA, rhoA=rhoA,
                                              disk_mass=disk_mass,
                                              n_p=n_p, n_d=n_d, lattice=lattice)
            self.params.cells = self.cells
        self.lattice = bool(lattice)
        self.F_tip = np.asarray(F_tip, float).reshape(3)
        self.g_world = np.asarray(g_world, float).reshape(3)
        self.R_mount = np.eye(3) if R_mount is None else np.asarray(R_mount, float).reshape(3, 3)
        self.h_tau = float(h_tau)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.L0 = np.asarray(_L0_wire(), float)      # (6,) 未变形丝长
        self.cache = bool(cache)
        self.tau_max = 60.0
        self._c = {}

    # ───────────────────────────────── 全阶 Cosserat 部分
    def shape(self, tau, x0=None):
        """τ(6) → 全阶 Cosserat 静力平衡解.

        返回 TendonCouplingResult:
            tip(3) R_tip(3,3) p_curve(j,3) R_curve(j,3,3) s_curve(j)
            u_curve m_curve n_curve v_curve  res(6|7) converged x(6) n_iter
        """
        tau = np.asarray(tau, float).reshape(6)
        key = None
        if self.cache:
            key = tau.round(9).tobytes()
            if key in self._c:
                return self._c[key]
        sol = shooting(tau, self.params, F_tip=self.F_tip, g_world=self.g_world,
                       R_mount=self.R_mount, x0=x0,
                       tol=self.tol, max_iter=self.max_iter)
        out = TendonCouplingResult(
            tau=tau, x=sol["x"], v0=sol["v0"], u0=sol["u0"],
            tip=np.asarray(sol["p_curve"][-1], float),
            R_tip=np.asarray(sol["R_curve"][-1], float),
            p_curve=sol["p_curve"], R_curve=sol["R_curve"], s_curve=sol["s_curve"],
            u_curve=sol["u_curve"], m_curve=sol["m_curve"],
            n_curve=sol["n_curve"], v_curve=sol["v_curve"],
            res=sol["res"], converged=bool(sol["converged"]),
            n_iter=sol["n_iter"], res_inf=float(np.linalg.norm(sol["res"], np.inf)),
        )
        if key is not None and out["converged"]:
            self._c[key] = out
        return out

    def anchor_forces(self, tau, x0=None):
        """肌腱作用在梁上的集中力/矩 (段 B 锚接点, 空间系) — 耦合的"源项".

        返回 (F(3), Lm(3))
        """
        s = self.shape(tau, x0=x0)
        u = np.asarray(s["u_curve"][-1], float)
        v = np.asarray(s["v_curve"][-1], float)
        R = np.asarray(s["R_curve"][-1], float)
        return anchor_loads(u, v, R, SEG_B, np.asarray(tau, float), self.params.r_disk)

    # ───────────────────────────────── 肌腱几何部分
    def tendon_lengths(self, sol):
        """由一次解算曲线得到 6 根丝的几何长度 L_w (m)."""
        if not isinstance(sol, dict) or "p_curve" not in sol:
            sol = self.shape(sol)
        return np.array([tendon_path_length(sol, w) for w in range(6)], float)

    def delta_L_mm(self, tau, x0=None, sol=None):
        """τ → ΔL (mm, 6 维).  ΔL_w = L_w(τ) − L0_w.

        ΔL_w < 0 表示该丝几何上变短(被"压"), > 0 表示被拉长.
        """
        s = sol if sol is not None else self.shape(tau, x0=x0)
        return (self.tendon_lengths(s) - self.L0) * 1000.0

    # ───────────────────────────────── 批量接口 (向量化, 供绘图/建表)
    def shape_batch(self, taus, want_curve=True, retry=True, max_iter=15,
                    tol=1e-8, x0=None):
        """★ 批量: τ(N,6) → N 条全阶 Cosserat 平衡解.

        与标量 `shape()` **同一套方程** (sdm_vc 的 VC 式), 只是向量化 ——
        走 `shooting_batch[_retry]` + `forward_batch`, 不做标量循环。

        返回 dict:
            taus (N,6)   x (N,6)   tip (N,3)
            converged (N,) bool    n_iter (N,)   res (N,6 或 7)
            n_failed int           S (nS,)       [曲线网格弧长]
            curve: want_curve=True 时 = (Pc, S, Uc, Cc, Rc, Nc, Vv)
                   (N,nS,3)/(N,nS,3,3) —— **解网格** (n_p/n_d 步), 非细骨架
        """
        taus = np.asarray(taus, float).reshape(-1, 6)
        if retry:
            out = shooting_batch_retry(taus, self.params, tol=tol, max_iter=max_iter)
        else:
            out = shooting_batch(taus, self.params, x0=x0, tol=tol, max_iter=max_iter)
        conv = np.asarray(out["converged"], bool)
        d = dict(taus=taus, x=np.asarray(out["x"], float), tip=np.asarray(out["tip"], float),
                 converged=conv, n_iter=np.asarray(out["n_iter"]),
                 res=np.asarray(out["res"], float), n_failed=int((~conv).sum()))
        if want_curve:
            _, _, cv = forward_batch(d["x"], taus, self.params,
                                     F_tip=self.F_tip, want_curve=True)
            d["curve"] = cv
            d["S"] = np.asarray(cv[1], float)
        return d

    def curve_batch(self, taus=None, x=None, want_curve=True, retry=True,
                    max_iter=15):
        """批量曲线场: 给定 x (打靶解) 或只给 τ.

        返回 (cv, sb):
            cv = (Pc, S, Uc, Cc, Rc, Nc, Vv)
                 Pc/Rc (N,nS,3)/(N,nS,3,3) 位置/姿态; Uc 曲率;
                 Cc 内力矩; Nc 内力; Vv 线应变  —— 均为**解网格** (n_p/n_d 步)
                 S (nS,) 该网格弧长
            sb = shape_batch 诊断 dict (自己解时非 None; 传入 x 时为 None)
        """
        if x is None:
            sb = self.shape_batch(taus, want_curve=True, retry=retry,
                                  max_iter=max_iter)
            return sb["curve"], sb
        taus = np.asarray(taus, float).reshape(-1, 6)
        x = np.asarray(x, float).reshape(-1, 6)
        _, _, cv = forward_batch(x, taus, self.params,
                                 F_tip=self.F_tip, want_curve=True)
        return cv, None

    def build_shapes_batch(self, taus, nsub=600, x=None, curve=None,
                           want_curve=True, retry=True, max_iter=15):
        """★ 批量重建**细骨架** (绘图用): τ(N,6) → PS(N,nsub+1,3), RS(N,nsub+1,3,3).

        x     给定则直接用 (如建表存下的打靶解, 免重解);
        curve 给定则复用已有曲线场 (免重复 forward_batch)。
        nsub  默认 600 → 步长 0.75mm (与表内 kS/kK 的 601 节点一致)。

        ⚠ 必须带 **v 场** (p'=R·v): k_rod 有限时 v_z 实测 0.85~0.90 (轴向压缩
          10~15%), 只用 κ 会让骨架凭空长 40~58mm。

        返回 (PS, RS, info) —— info 含 converged/n_failed/x/S 等诊断。
        """
        taus = np.asarray(taus, float).reshape(-1, 6)
        if curve is not None:
            cv = curve
            info = dict(x=None if x is None else np.asarray(x, float),
                        converged=None, n_failed=None,
                        S=np.asarray(cv[1], float))
        elif x is None:
            sb = self.shape_batch(taus, want_curve=True, retry=retry,
                                  max_iter=max_iter)
            x = sb["x"]; cv = sb["curve"]
            info = dict(x=x, converged=sb["converged"],
                        n_failed=sb["n_failed"], S=sb["S"])
        else:
            x = np.asarray(x, float).reshape(-1, 6)
            cv, _ = self.curve_batch(taus=taus, x=x)
            info = dict(x=x, converged=None, n_failed=None,
                        S=np.asarray(cv[1], float))
        PS, RS = integrate_shape_batch(cv[2], cv[1], nsub=nsub, v_batch=cv[6])
        info["PS"] = PS; info["RS"] = RS; info["curve"] = cv
        return PS, RS, info

    def tendon_lengths_batch(self, taus=None, x=None, curve=None, nsub=600,
                             fine=False, PS=None, RS=None):
        """批量 6 丝几何长 (N,6) m.

        fine=False (默认): 用**解网格**曲线 (n_p/n_d 步) —— 与标量
            `tendon_lengths()` / 表内 `dL_mm` **同口径** (逐位一致)。
        fine=True: 用 nsub 细网格重建 —— 折线更逼近真实弧长, 会比表内 dL
            长 0.05~0.1mm (450mm 上约 2e-4 相对量), 属**网格差异不是模型差异**。
        """
        if fine:
            if PS is None or RS is None:
                PS, RS, _ = self.build_shapes_batch(taus, nsub=nsub, x=x, curve=curve)
            return _wire_geom_len_batch(PS, RS)
        if curve is None:
            curve, _ = self.curve_batch(taus=taus, x=x)
        return _wire_geom_len_batch(curve[0], curve[4])

    def delta_L_batch(self, taus=None, x=None, curve=None, nsub=600,
                      fine=False, PS=None, RS=None):
        """★ 批量 ΔL(mm): τ(N,6) → (N,6).

        口径与标量 `delta_L_mm` 一致 (默认解网格; fine=True 走细网格重建)。
        """
        Lg = self.tendon_lengths_batch(taus=taus, x=x, curve=curve, nsub=nsub,
                                       fine=fine, PS=PS, RS=RS)
        return (Lg - self.L0[None, :]) * 1000.0

    def pose_batch(self, taus, **kw):
        """批量末端位姿: τ(N,6) → tip(N,3), R_tip(N,3,3), info."""
        sb = self.shape_batch(taus, want_curve=True, **kw)
        cv = sb["curve"]
        return (np.asarray(cv[0][:, -1], float),      # Pc 末节点
                np.asarray(cv[4][:, -1], float),      # Rc 末节点
                sb)

    def verify_batch(self, taus, n=10, nsub=600, seed=0, verbose=True):
        """★ 标量 vs 批量 一致性抽检: 逐条比较 ΔL / tip.

        逐条用标量 `delta_L_mm` (全阶打靶 + tendon_path_length) 对拍批量路径。
        """
        taus = np.asarray(taus, float).reshape(-1, 6)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(taus), size=min(int(n), len(taus)), replace=False)
        sb = self.shape_batch(taus, want_curve=True)
        # 解网格口径 (与标量 shape/delta_L_mm 逐位对齐)
        dL_b = self.delta_L_batch(curve=sb["curve"])
        # 细网格口径 (绘图骨架; 折线更逼近弧长, 会略长)
        PS, RS, _ = self.build_shapes_batch(taus, nsub=nsub, x=sb["x"],
                                            curve=sb["curve"])
        dL_f = self.delta_L_batch(fine=True, PS=PS, RS=RS)
        rows = []
        for i in idx:
            s = self.shape(taus[i])
            dL_s = self.delta_L_mm(taus[i], sol=s)
            d_tip = float(np.linalg.norm(np.asarray(s["tip"]) - sb["tip"][i]) * 1000)
            d_dL = float(np.linalg.norm(dL_s - dL_b[i]))
            d_dLf = float(np.linalg.norm(dL_s - dL_f[i]))
            rows.append(dict(i=int(i), tau=taus[i].copy(), d_tip_mm=d_tip,
                             d_dL_mm=d_dL, d_dL_fine_mm=d_dLf,
                             dL_scalar=dL_s, dL_batch=dL_b[i], dL_fine=dL_f[i],
                             L_err_mm=float(np.abs(dL_f[i] - dL_s).max())))
        if verbose:
            print("── 标量 vs 批量 一致性 ──")
            for r in rows:
                print(f"  τ={np.round(r['tau'],2)}  Δtip={r['d_tip_mm']:.2e} mm  "
                      f"Δ(ΔL)解网格={r['d_dL_mm']:.2e}  Δ(ΔL)细网格={r['d_dL_fine_mm']:.2e} mm")
            print(f"  最大: Δtip={max(r['d_tip_mm'] for r in rows):.2e} mm, "
                  f"Δ(ΔL)解网格={max(r['d_dL_mm'] for r in rows):.2e} mm, "
                  f"Δ(ΔL)细网格={max(r['d_dL_fine_mm'] for r in rows):.2e} mm "
                  f"(纯网格离散差)")
        # 判据: 两条路径解同一组方程, 差别只能来自**求解容差**
        # (标量 tol=1e-9 + SVD 正交化; 批量 tol=1e-8 + 代数正交化)
        # ⇒ 允许 1e-2 mm 级; 若超差说明两路径方程不一致 (真 bug)。
        ok = all(r["d_tip_mm"] < 1e-2 and r["d_dL_mm"] < 1e-3 for r in rows)
        return ok, rows

    # ───────────────────────────────── ★ 肌腱力耦合矩阵
    def coupling_matrix(self, tau, h=None, x0=None, verbose=False):
        """★ 6×6 肌腱力耦合矩阵  J[i,j] = ∂ΔL_i / ∂τ_j   (mm/N).

        中心差分; 每列 2 次全阶打靶 (共 12 次). 用前一次解热启动加速。

        返回 TendonCouplingResult(J, tau, dL0, n_fail)
            J      : (6,6)
            dL0    : 工作点处 ΔL (mm)
            n_fail : 未收敛的打靶次数 (J 可信度参考)
        """
        tau = np.asarray(tau, float).reshape(6)
        h = self.h_tau if h is None else float(h)
        base = self.shape(tau, x0=x0)
        dL0 = self.delta_L_mm(tau, sol=base)
        J = np.zeros((6, 6))
        n_fail = 0
        warm = base["x"]
        for j in range(6):
            tp = tau.copy(); tp[j] += h
            tm = tau.copy(); tm[j] -= h
            sp = self.shape(tp, x0=warm)
            sm = self.shape(tm, x0=warm)
            if not sp["converged"]:
                n_fail += 1
            if not sm["converged"]:
                n_fail += 1
            dp = self.delta_L_mm(tp, sol=sp)
            dm = self.delta_L_mm(tm, sol=sm)
            J[:, j] = (dp - dm) / (2.0 * h)
            if verbose:
                print(f"  τ{j}: 收敛 {sp['converged']}/{sm['converged']}")
        return TendonCouplingResult(J=J, tau=tau, dL0=dL0, n_fail=n_fail,
                                    h=h)

    # ───────────────────────────────── 便捷: 张力 → 位姿
    def tip_from_tau(self, tau):
        """τ → 末端位置(m)/姿态, ΔL(mm) 一步到位."""
        s = self.shape(tau)
        return s["tip"], s["R_tip"], self.delta_L_mm(tau, sol=s)

    # ───────────────────────────────── 位姿出口 (正解)
    def pose(self, tau, sol=None):
        """τ → (p_tip(m), R_tip(3,3))."""
        s = sol if sol is not None else self.shape(tau)
        return np.asarray(s["tip"], float), np.asarray(s["R_tip"], float)

    # ───────────────────────────────── 诊断
    def describe(self):
        """打印模型摘要 (整合了哪些理论/函数)."""
        p = self.params
        print("=" * 66)
        print("肌腱力耦合模型 (TFCM) — 菱形晶胞 × 全阶 Cosserat × 肌腱几何")
        print("=" * 66)
        if self.cells is not None:
            for i, c in enumerate(self.cells):
                print(f"\n【段{'AB'[i]}{'(近基座, s∈[0,L1])' if i == 0 else '(s∈[L1,L])'} 晶胞】")
                c.report()
        print("─" * 66)
        print(f"  臂长 L = {L*1000:.1f} mm  (段A {L1*1000:.1f} + 段B {L2*1000:.1f})")
        print(f"  丝半径 r_disk = {p.r_disk*1000:.1f} mm,  丝角 α = {ALPHA_DEG}°")
        print(f"  段A 丝 {SEG_A} (锚止 s=L1) | 段B 丝 {SEG_B} (锚止末端)")
        print(f"  L0 = {np.round(self.L0*1000,1)} mm")
        print(f"  刚度: EI={getattr(p,'EI',float('nan')):.4g}  GA={p.GA:.4g}  "
              f"GJ={p.GJ:.4g}  k_rod={p.k_rod:.4g}   lattice={self.lattice}")
        print(f"  离散: n_p={p.n_p}, n_d={p.n_d}")
        print(f"  打靶: tol={self.tol:g}, max_iter={self.max_iter}")
        print("=" * 66)

    def lattice_effect(self, tau, sol=None):
        """给定 τ, 量化**菱形晶胞非线性**沿臂的影响: 曲率范围 vs EI 折减.

        返回 dict(kappa_max, kappa_med, theta_max, EI_ratio_min, EI_ratio_med, kappa_5pct)
        """
        if self.cell is None:
            return None
        s = sol if sol is not None else self.shape(tau)
        u = np.asarray(s["u_curve"], float)
        kap = np.hypot(u[:, 0], u[:, 1])
        th = kap * self.cell.L_strut
        ratio = np.ones_like(th)
        nz = th > 1e-9
        ratio[nz] = np.sin(th[nz]) / th[nz]
        return dict(kappa_max=float(kap.max()), kappa_med=float(np.median(kap)),
                    theta_max=float(th.max()),
                    EI_ratio_min=float(ratio.min()),
                    EI_ratio_med=float(np.median(ratio)),
                    kappa_5pct=self.cell.kappa_soften(0.05))

    def lattice_test(self, verbose=True):
        """晶胞自检: ①恒等式 ②∝sin³α ③非线性折减 ④与 params 一致性."""
        say = print if verbose else (lambda *a, **k: None)
        c = self.cell
        ok = True

        say("── ⓪ 菱形晶胞 ──")
        say(f"  α={np.degrees(c.alpha):.3f}°  L_strut={c.L_strut*1e3:.4f} mm  "
            f"l_ax={c.l_ax*1e3:.4f} mm  N_long={c.N_long}")
        say(f"  EI0={c.EI0:.5f} N·m²  k_rod={c.k_rod:.4g} N/m  C0={c.C0:.5f} N·m²")

        r1, r2 = c.ratio_EI_krod, c.ratio_expected
        e1 = abs(r1 - r2) / r2
        ok &= e1 < 1e-12
        say("  ①b 分段恒等式 (每段各自成立, N_circ 仍约掉)")
        for i, cg in enumerate(self.cells):
            rr1, rr2 = cg.ratio_EI_krod, cg.ratio_expected
            ee = abs(rr1 - rr2) / rr2
            ok &= ee < 1e-12
            say(f"      段{'AB'[i]}: N_circ={cg.N_circ}  L={cg.L_arm*1e3:.0f}mm  "
                f"EI0={cg.EI0:.4f}  k_rod={cg.k_rod:.1f}  "
                f"EI0/k_rod={rr1:.6g} vs 2r²L={rr2:.6g}  误差={ee:.1e} "
                f"{'✓' if ee < 1e-12 else '✗'}")
        say(f"  ① EI0/k_rod = {r1:.8g}")
        say(f"     2·r_i²·L_eff = {r2:.8g}   相对误差={e1:.2e} "
            f"{'✓ 精确' if e1 < 1e-12 else '✗'}")
        say(f"     (2·r_i²·L = {c.ratio_expected_ideal:.8g}, "
            f"N_long 取整 {c.N_long} vs L/l_ax={c.L_arm/c.l_ax:.3f})")

        rows = c.stiffness_scaling()
        ei_ok = all(abs(r["EI0_norm"] - r["sin3"]) < 1e-12 for r in rows)
        ok &= ei_ok
        say("  ② α 扫描 (EI0 ∝ sin³α, 以 α=12° 归一):")
        for r in rows:
            say(f"       α={r['alpha_deg']:>4.0f}°  EI0={r['EI0']:.4f}  "
                f"k_rod={r['k_rod']:>8.1f}  EI0/k_rod={r['EI_krod']:.6g}  "
                f"EI0/EI0(12°)={r['EI0_norm']:.4f}  sin³比={r['sin3']:.4f}")
        say(f"       {'✓ 逐点吻合' if ei_ok else '✗ 不吻合'}")

        kap = c.kappa_soften(0.05)
        e_pct = (1 - c.EI_secant(30.0) / c.EI0) * 100
        say(f"  ③ 非线性: κ=30 1/m 时 EI 降 {e_pct:.3f}%  (5% 折减点 κ={kap:.1f} 1/m)")
        say(f"     实测工况 κ_max≈0.8 1/m ⇒ θ≈0.016 ⇒ EI 降 "
            f"{(1-c.EI_secant(0.8)/c.EI0)*100:.4f}% → 工况内可视作常量 ✓")

        e2 = abs(self.params.k_rod - c.k_rod) / c.k_rod
        say(f"  ④ params 一致性: k_rod 相对误差={e2:.2e}  EI={self.params.EI:.5f}")
        return bool(ok)

    def lattice_sweep(self, tau_list=None):
        """扫一组 τ, 汇总晶胞非线性对 EI 的影响 (验证工况内是否近似常量)."""
        if tau_list is None:
            tau_list = [[2]*6, [6]*6, [10, 0, 10, 0, 10, 5], [20, 0, 20, 0, 20, 10]]
        rows = []
        for t in tau_list:
            e = self.lattice_effect(t)
            if e:
                rows.append((tuple(float(x) for x in np.ravel(t)), e))
        return rows

    def self_test(self, verbose=True):
        """自检 (只覆盖**正解**): 直臂/对称/单丝 三组工况 + 耦合矩阵.

        逆解的自检在 `inverse_solver.TendonInverseSolver.self_test()`。
        """
        log = [] if not verbose else None

        def say(t):
            if verbose:
                print(t)
            if log is not None:
                log.append(t)

        say("── ① 直臂 / 对称 / 单丝 ──")
        for name, tau in [("全 0", [0]*6), ("对称 2N", [2]*6), ("单丝0=8N", [8,0,0,0,0,0])]:
            s = self.shape(tau)
            dL = self.delta_L_mm(tau, sol=s)
            tip = s["tip"]
            say(f"  {name:10s} conv={s['converged']} res∞={s['res_inf']:.1e} "
                f"tip=({tip[0]*1e3:+.2f},{tip[1]*1e3:+.2f},{tip[2]*1e3:+.2f}) mm "
                f"ΔL={np.round(dL,3)}")

        tau0 = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        say("── ② 耦合矩阵 J=∂ΔL/∂τ (mm/N) @ τ=2N×6 ──")
        cm = self.coupling_matrix(tau0, verbose=False)
        J = cm["J"]
        say(f"  对角  : {np.round(np.diag(J),4)}")
        off = J - np.diag(np.diag(J))
        sv = np.linalg.svd(J, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
        say(f"  非对角: |max|={np.abs(off).max():.4f}  (非零 ⇒ 存在耦合)")
        say(f"  cond(J) = {cond:.3g}   未收敛打靶 {cm['n_fail']}/12")
        ok = bool(np.abs(off).max() > 0 and cm["n_fail"] == 0)
        say(f"  → {'PASS ✅' if ok else 'FAIL ❌'}")
        return ok


# ══════════════════════════════════════════════════════════════════════
# §2  函数式入口 (不想用类时)
# ══════════════════════════════════════════════════════════════════════


def coupling_matrix(tau, **kw):
    """快捷: τ → J=∂ΔL/∂τ."""
    return TendonCouplingModel(**kw).coupling_matrix(tau)


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    M = TendonCouplingModel()
    M.describe()
    print()
    ok0 = M.lattice_test()
    print()
    ok = M.self_test()
    print()
    print("── ③ 晶胞非线性对实际弯曲构型的影响 ──")
    for t, e in M.lattice_sweep():
        print(f"  τ={t}  κ_max={e['kappa_max']:6.2f} 1/m  θ_max={e['theta_max']:.4f}  "
              f"EI/EI0 min={e['EI_ratio_min']:.6f}")
    print()

    print("── ④ 标量 vs 批量 一致性 ──")
    rng = np.random.default_rng(0)
    TT = rng.uniform(0.5, 8.0, size=(12, 6))
    import time as _t
    t0 = _t.time()
    ok_b, _ = M.verify_batch(TT, n=6, nsub=600, seed=0)
    print(f"  12 条批量重建耗时 {_t.time()-t0:.2f}s")
    print()
    print("  (逆解自检见: python inverse_solver.py [--table vc_table_40.npz])")

    allok = bool(ok0 and ok and ok_b)
    print()
    print(f"{'ALL PASS ✅' if allok else 'FAIL ❌'}")
    sys.exit(0 if allok else 1)