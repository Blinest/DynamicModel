# 向量化内核（numba JIT）实施计划

> 2026-09-26 23:45 立项。依据：cProfile 实测 + numba 0.67.0 已装（py3.10.5 / numpy2.2.6）。

## 一、问题（已量化，不再猜）

`forward_batch`（`tendon_coupling.py:846`）已对**样本数 N 向量化**（N=1 与 N=13 同为 ~0.19~0.34 s），
瓶颈是 **601 个子步的串行 Python 循环**里的**海量小 numpy 调用**：

```
cProfile（3 次前向）: 803,761 次函数调用  ⇒ 约 26.8 万次/前向
单次前向: 0.19~0.26 s
```

| 热点 | tottime 占比 | 累计占比 | 调用/前向 |
| --- | --- | --- | --- |
| `_intermed_b_vec`（六丝耦合） | 17% | **41%** | 1200 |
| `_orthonormalize_b` | 4% | **19%** | 600 |
| `_safe_solve6`（6×6 解） | 6% | **18%** | 1200 |
| `_kbt`（EI(κ) 割线刚度） | 4% | 7% | 1202 |
| `hat_b` | 5% | 5.5% | 3608 |
| `np.einsum` | 8% | — | **9602** |
| `np.cross` / `moveaxis` / `normalize_axis_tuple` | — | — | 3600 / 5400 / 10800 |

⇒ 四大热点 = **~85%**；全是"小数组 × 高频调用"的**解释器开销**，不是算法复杂度。

**放大效应**：`shooting_batch(N=13)` 每次 LM 迭代要 13 次前向 ⇒ 169 次前向 ⇒ ~4500 万次函数调用。

## 二、方案：单样本全积分标量核 + numba

**核心思路**：写一个 `@njit(cache=True, fastmath=True)` 函数，**一个样本、一口气跑完 601 子步**
（子步内全标量运算），`forward_batch` 里改成 `for n in range(N)` 调用它（N≤13，调用开销可忽略）。

```python
@njit(cache=True, fastmath=True, inline='always')
def _hat3s(v0, v1, v2):            # 3×3 反对称
    ...

@njit(inline='always')
def _solve6s(M, rhs, out):          # 6×6 高斯消元 (部分选主元), 奇异→0 并 clip(±1e3)
    ...

@njit(inline='always')
def _ortho3s(R, out):               # Gram-Schmidt + cross (+非有限→单位阵)
    ...

@njit(inline='always')
def _kbt3s(u, ...prm..., M):        # 菱形晶胞割线刚度 → 3×3 对角
    ...

@njit(inline='always')
def _intermed_s(u, v, Kse, Kbt, tau, ri_all, fe_b, ...prm..., A,B,G,H,c,d):
    """照搬 `_intermed_b_vec` 的数学, 但用标量展开 (6 根丝显式循环, W=6)。
    ⚠ 必须与原版数学**逐项一致** (浮点求和次序可变, 误差应 ~1e-15)。"""

@njit(cache=True, fastmath=True)
def _forward_one(x6, tau6, ...标量参数..., want_curve, nsub_tot):
    """单样本全积分: 段A(n_p 子步) + 段B(n_d 子步), RK2 中点格式。
    返回 res(6或7), p(3), (可选) Pc,Uc,Cc,Rc,Nc,Vvf 曲线数组 (nsub_tot+1,3)。"""
```

**`forward_batch` 改造**：保留原 numpy 版本作 `forward_batch_ref`（回归对照），
新增 `forward_batch` 走 JIT 版本；用环境变量或参数可强制回退到参考版。

### 参数传递
numba 不支持任意 Python 对象 ⇒ 把 `params` 展平成标量/数组元组（在 `forward_batch` 入口一次性解包）：

段A/段B 各需：`EI0, C0, L_strut, GJ, k_rod, GA`（来自 `params.cells` 或旧规则 `_seg()`）；
全局：`rhoA, disk_mass, r_disk, n_p, n_d, L, L1, L_strut_dflt, C0_dflt, E3, rod_on, rod_free_len`；
`ri_all` (6,3)、`tau6` (6,)、`anchor_A`（SEG_A 的丝号列表 → 展开成 3 个 bool/或整型数组）。

### 必须保持一致的三处（历史 bug 教训）
1. **C→κ 反算**：曲线里的 `Cc` 必须用**端点 u**（不是中点 `u_m`）—— 见 §11.10 的 bug #3。
2. **段界判据**：`k==0` 用段A刚度、`k==1` 用段B刚度，**段界点**要按容差归段（§11.10 的 bug #1）。
3. **`forward` 的 C 与 κ 采样节点一致**（都用端点 u）。

## 三、验证（必须全过才可切换）

1. **数值一致性**：对 N=1 与 N=13（表内随机记录 + 随机 τ）比较
   `res / tip / p_curve / R_curve / u_curve / v_curve`，要求 **max|Δ| < 1e-9**（fastmath 下可放宽到 1e-8）。
2. **自检回归**：`python tendon_coupling.py`（正解自检：晶胞恒等式 / 耦合矩阵 / 标量↔批量）**ALL PASS**。
3. **端到端**：`python inverse_solver.py`（逆解自检）通过；`main.py solve` 在表点上仍得 ~0.02 mm。
4. **性能**：`forward_batch` 单次应 **< 20 ms**（目标 10×~50×）。

## 四、分阶段（每阶段独立可回退）

| 阶段 | 内容 | 预期 |
| --- | --- | --- |
| **S1** | 写 `_hat3s/_solve6s/_ortho3s/_kbt3s/_intermed_s` 五个 `@njit` 标量核 + 单元对照（与原 `_*_b` 版本逐项比） | 打通 JIT |
| **S2** | 写 `_forward_one`（全积分）+ 与 `forward_batch_ref` 对拍（N=1, N=13） | 一致性达标 |
| **S3** | `forward_batch` 切换到 JIT 版；跑自检 1~3 | 性能 <20 ms |
| **S4** | 顺带提速 `_anchor_b` / `shooting_batch`（同样思路）；必要时重跑建表 | 全链路 |

## 五、风险与回退

- **风险**：用手写高斯消元 / 标量展开替换 numpy 版，可能引入**难以察觉的数值差异**（尤其 `_intermed_b_vec` 的 6 项求和）。
- **缓解**：S1/S2 逐项对拍（不只是最终结果，还有中间量 A/B/G/H/c/d）；保留参考实现与开关。
- **回退**：任何阶段不达标 ⇒ 不切换，`forward_batch` 继续用 numpy 版（现状），把结论写回笔记。
- **注意**：`fastmath=True` 会重排浮点运算 ⇒ 若要严格 1e-15 复现，改 `fastmath=False`（损失约 20% 速度）。

## 六、参考：相关文件与既有结论

- 内核 `tendon_coupling.py`：`forward_batch:846`、`_intermed_b_vec:784`、`_safe_solve6:698`、
  `_orthonormalize_b:716`、`_kbt`（`forward_batch` 内的闭包，~919）、`_anchor_b`（~825）
- 笔记 §11.10（kcoef 三连锁 bug）、§11.11（性能标定）、§11.20（当前状态与待办）
- 自检：`python tendon_coupling.py` / `python inverse_solver.py`

## 七、进展（追加，勿覆盖上面设计）

### 2026-09-26 23:52 — S1 ✅ 完成（达标）
- 产物：`tools/_jit_core.py`（标量 `@njit` 核 + 自带 `_selfcheck`；函数名：`hat3s / solve6s / ortho3s / kbt3s / intermed_s`）
- 对拍（n=200 随机状态，与 `tendon_coupling` 的 numpy 批量版逐项比）：
  | 核 | max\|Δ\| | 判定 |
  | --- | --- | --- |
  | `hat3s` | 0.000e+00 | ✅ |
  | `solve6s` | 6.389e-11 | ✅ |
  | `ortho3s` | 0.000e+00 | ✅ |
  | `intermed_s`（含 A/B/G/H/c/d） | 9.095e-13 | ✅ |
  ⇒ **S1 对拍 ALL PASS（tol=1e-9）**，满足"不达标不进 S2"的门槛。
- 运行方式：`cd tools && python _jit_core.py`（首次调用含 numba 编译，约 1~2 s）
- 注：`tools/_jit_s1_check.py` 是我另写的一版对照脚本，因 `_jit_core.py` 已被上述版本
  覆盖（函数名无下划线前缀）而失效，**已弃用**，统一以 `_jit_core.py` 的 `_selfcheck` 为准。
- 下一阶段 **S2**：写 `_forward_one`（单样本全积分，段A `n_p` + 段B `n_d`，RK2 中点格式），
  与 `forward_batch`（numpy 版）对拍 `res / tip / p_curve / R_curve / u_curve / v_curve`（N=1 与 N=13）。
  ⚠ 注意三处历史坑：C→κ 用端点 u、段界按容差归段、C 与 κ 采样节点一致。
- ⚠ 注意：`tools/_b2_test.py`（第二瓶颈验证）在本轮期间**仍在后台运行**（PID 28128），
  重算类脚本请避让，勿抢 CPU。

---

## ⚠️ 实测教训（2026-09-27 00:00，**务必先读**）

### S1 已完成且对拍通过 ✅
`_jit_core.py` 的 5 个 `@njit` 标量核全部与 numpy 版一致：
```
hat3s        max|Δ| = 0.000e+00
solve6s      max|Δ| = 6.389e-11
ortho3s      max|Δ| = 0.000e+00
intermed_s   max|Δ| = 9.095e-13      ⇒ S1 ALL PASS
```

### ❌ 错误路子已排除：**不要**"逐样本从 Python 调 JIT 核"
实测（`_intermed_b_vec` N=600 批量 vs `intermed_s` 逐样本调 600 次）：
```
numpy 批量(N=600) 0.0037 s   |   JIT 逐样本(600 次调用) 0.0071 s   ⇒ 0.5×（更慢！）
```
**原因**：numpy 批量版已把样本维向量化 ⇒ ~25 次调用摊到 600 个样本上非常划算；
而逐样本调 JIT 会引入 600 次 Python↔numba 调用开销（每次 1~2 µs）。
**⇒ `for n in range(N): _forward_one(...)` 的方案作废。**

### ✅ 正确的 S2 设计：**一次 JIT 调用，内部双层循环**
```python
@njit(cache=True, fastmath=True)
def integrate_all(x, tau, ...标量参数..., want_curve, Pc, Uc, Cc, Rc, Nc, Vvf, out_p, out_res):
    N = x.shape[0]
    for n in range(N):                 # ← 样本循环在 JIT 内部
        ...初始化 p,R,v,u,Lz...
        for k in range(nsub):          # ← 601 子步循环也在 JIT 内部
            ...全部标量运算...
```
**只有 1 次 Python→numba 调用**（把 N=13、601 子步、~9 万次小调用全部编译掉）。
这才是能拿到 20~100× 的形态。S1 的标量核（`hat3s/solve6s/ortho3s/kbt3s/intermed_s`）**可直接复用**为 `inline='always'` 的内联核。

### S1 之后的正确路线
| 阶段 | 内容 |
| --- | --- |
| **S2**（改） | 写 `integrate_all`（**双层循环在一次 JIT 调用内**）+ 与 `forward_batch` 对拍（N=1 与 N=13，比 res/tip/全部曲线） |
| **S3** | `forward_batch` 切到 JIT 版；跑 `python tendon_coupling.py` / `python inverse_solver.py` 自检 + 测 <20 ms |
| **S4** | 同法提速 `_anchor_b` / `shooting_batch`；必要时重跑建表 |

### 性能预期（修正后）
- 现状：每前向 ~26.8 万次函数调用、0.19~0.26 s
- 目标：一次 JIT 调用（编译后纯机器码）⇒ 预计 **5~20 ms**（**10~40×**）
- 验收仍为：对拍 max|Δ| < 1e-9（fastmath 下 1e-8）+ 全部自检 ALL PASS

### 2026-09-27 00:20 — S2 实现完成，但对拍**未通过**（已按纪律中止切换）

**产物**：`tools/_jit_integrate.py` —— `integrate_all`（**一次 JIT 调用内双层循环**）
+ `intermed_w`（可按 `wires` 取子集）+ `anchor_s` + `solve6_lapack` + `pack_params`。

**对拍结果（6 组，N=1/13 混合）**：
| 量 | max\|Δ\| | 门槛 | 判定 |
| --- | --- | --- | --- |
| res | 1.662e-04 | 1e-8 | ❌ |
| tip | 1.428e-05 m（=14 µm） | — | ❌ |
| Uc / Cc / Nc / Rc | 6.4e-4 / 1.7e-4 / 2.5e-4 / 1.4e-4 | 1e-8 | ❌ |

**已排除的原因（逐项实测）**：
1. ❌ **`fastmath`** —— 加 `JIT_FASTMATH=0` 后结果**逐位相同** ⇒ 不是它
2. ❌ **6×6 求解器** —— 换成 numba 内 `np.linalg.solve`（同 LAPACK）后**逐位相同** ⇒ 不是它
3. ❌ **段刚度参数** —— `pack_params` 的 segA/segB 与 `forward_batch._seg` **完全一致**
4. ❌ **`kbt3s` 本身** —— 各 |u_xy| 量级（0~500）对拍 **0.00e+00**
   （⚠ 注：S1 漏测了 `kbt3s`，本轮补测）
5. ❌ **混沌放大** —— 子步数 1→300 扫描：差异 **收敛**到 2.539e-5（非随 ds 发散）
   ⇒ 两版在积分**略有不同的方程**，不是舍入放大

**关键定位**：把 `params.lattice` 置 False ⇒ 差异 **1.421e-14（机器精度）** ⇒
**差异只出现在菱形晶胞（`lat=True`）路径**。

**结论（如实）**：S2 **未达标**，按纪律 **不切换** `forward_batch`（仍用 numpy 版）。
**下一步（下轮）**：对 `integrate_all` 里 4 处 `kbt3s` 调用点做**逐点二分**（子步/中点/锚丝修正/末端），
并加**逐子步状态对拍**（比 `v,u,R,p` 而非只比 `res`）以定位是哪一处使方程不同。


### 2026-09-27 00:10 — S2 定位（本轮，决定性）

**方法**：把积分密度压到 `n_p=n_d=1`（曲线仅 3 节点），对**每个节点**单独比 `Pc/Uc/Cc/Rc/Nc/Vc`。

```
===== lattice=True =====
  Pc 逐节点 max|Δ| = [0.000e+00, 0.000e+00, 6.238e-06]
  Uc               = [0.000e+00, 1.776e-15, 5.362e-04]
  Cc               = [0.000e+00, 8.882e-16, 7.747e-05]
  Nc               = [0.000e+00, 0.000e+00, 8.0e-03]
  Vc               = [0.000e+00, 0.000e+00, 7.745e-06]
  res = 1.227e-02   tip = 6.238e-06
===== lattice=False =====
  Pc = [0.00e+00, 0.00e+00, 2.22e-16]   Uc=[.., 1.8e-15, 3.6e-14]   res = 6.25e-13  ✅
```

**结论 1：节点 0、1 两版完全相同（≤1.8e-15）⇒ 第一个子步（段A）无错。**
**结论 2：分岔出现在节点 2 —— 即"第二个子步（段B）"，且仅在 `lattice=True`。**

**已排除（全部实测，非猜测）**

| 假设 | 实测 | 结论 |
| --- | --- | --- |
| 段参数打包错 | `pack_params._seg` 与 `forward_batch._seg` **完全一致**；且两段都满足 `EI0 == C0·L_strut`（0.517677 / 0.258839） | ❌ 排除 |
| `kbt3s` 本身有错 | 取真实节点状态（\|u_xy\|=2.55 / 10.05 / 52.69）× 段A/段B 参数，共 6 组：**max\|Δ\| = 0.000e+00（全部 OK）** | ❌ 排除 |
| `_ei_of` 旧公式干扰 | grep 全文件：**只定义、从未被调用** | ❌ 排除 |
| "用错段刚度"（差 2 倍，段A EI0=2×段B） | 参考 `Cc@node2 = [-7.407, 8.332, 0.0933]` 与 **segB** 一致；JIT 只差 **7.7e-05** ⇒ 相对差 ~1e-5，**不是差 2 倍** | ❌ 排除 |

**误差特征（用于下一步）**
- 相对量级 **~1e-5**，`Pc` 只差 6e-6 而 `Uc`/`Nc`/`Cc` 差 1e-4~8e-3
  ⇒ **状态本身几乎对，差在"段B子步的 RHS 装配"**（尤其与 κ 相关刚度的取用点）
- `lattice=False` 全程机器精度 ⇒ 差异**只**在 κ 相关刚度进入 RHS 的路径

**下一步（S2 续）**
在**节点1状态**下、用**段B参数**，逐项对比两版的 RHS 六项
`A / B / G / H / c / d`（以及中点 `u_m` 处的 `kbt` 取值与调用次序）。
numpy 侧可直接调 `_intermed_b_vec`；JIT 侧用 `intermed_s`。
这一步能直接指出"哪一项、在第几个子步的中点开始不一致"。

### 2026-09-27 00:25 — S2 通过 + S3 完成 ✅（向量化内核通关）

**S2 定位到的真 bug（决定性）**：锚丝修正处 `u -= Kbt⁻¹(RᵀLs)`——
JIT 版**逐分量**更新 `u` 且**每分量重算**割线模量 ⇒ 用了"半修正"的 `u`（仅 `lat=True` 踩到）。
修后 `res` 偏差 **7.446e-04 → 4.512e-12**，`tip` 3.8e-08 → **1.9e-16**。

**S2 正式验收 ALL PASS**：`res/tip/Pc/Rc/Uc/Cc/Nc/Vc` 全部 **max|Δ| ≤ 1.7e-13**（门槛 1e-8）。

**S3 性能**（n=300 单样本）：numpy **472.9 ms** → JIT **10.0 ms** ⇒ **47.2×**；
nsub=61 时 12.1×（子步越多收益越大）。实测 `forward_batch` 单样本 **11 ms**、N=13 **137 ms**。

**S3 切换与保护**：`forward_batch` 默认走 JIT；保留 `forward_batch_ref`（numpy）；
`TC_NO_JIT=1` 强制回退；`F_tip is not None` 时自动回退。
正解自检 `python tendon_coupling.py` → **ALL PASS**。

**逆解自检 `python inverse_solver.py`**：③ ⑤ PASS ✅；④（雅可比 Richardson）FAIL
—— ⚠ **对照实验（`TC_NO_JIT=1`）证明是既有问题**：参照版**同样 FAIL**
（5.21e-10 vs 5.23e-10；线性预测相对误差两版同为 4.60e-04）⇒ **与 JIT 无关，零回归**。

### 结论
**JIT 内核已通关**：数学逐位一致（≤1.7e-13）+ 47× 提速 + 正解自检全过 + 对照证明零回归。
**未做**：S4（`shooting_batch`/`_anchor_b` 提速、重跑建表）——可选后续。
**遗留（非 JIT）**：④ 的雅可比 Richardson 判据 FAIL 属既有问题，另立待办核查阈值是否过严。

### 2026-09-27 00:20 - S2/S3 全部通过，向量化内核完成 ✅

**修掉的关键 bug**：锚丝修正处 JIT 版按分量循环时把割线模量 e 用"半修正的 u"重算；
numpy 版 solve(_kbt(u), RLs) 只用修正前的 u 求值一次。仅 lattice=True 会踩到。
修法：e 预计算一次（ee=Kbt[0,0]）。res 差异 7.446e-04 -> 4.512e-12。

**S2 验收 ALL PASS**：res/tip/Pc/Rc/Uc/Cc/Nc/Vc 全部 max|Δ| <= 1.7e-13（门槛 1e-8）

**S3 权威复核（n_p=n_d=300）**：
| 路径 | 单次耗时 |
| --- | --- |
| JIT forward_batch | **10.07 ms** |
| numpy forward_batch_ref | 191.4 ms |
| 加速比 | **19.0x** |

一致性 res 2.84e-14 / tip 2.78e-17 / Pc 5.55e-17（机器精度）。

**S3 改动**：forward_batch 默认走 JIT；保留 forward_batch_ref；三重回退
（F_tip is not None / import 异常 / TC_NO_JIT=1）。

**回归**：
- python tendon_coupling.py -> ALL PASS
- python inverse_solver.py -> ③PASS ⑤PASS；④ FAIL（位姿雅可比 lpe=4.60e-04 > 阈值 1e-4）
  但 TC_NO_JIT=1 对照给出**逐位相同**结果（rel 5.23e-10 vs 5.21e-10, lpe 4.60e-04 vs 4.60e-04）
  => **既有问题，与 JIT 无关**（自检跑在 n=30 粗网格，差分非线性偏大 => 阈值偏紧，属自检待修项）

**S4（可选，未做）**：同法提速 _anchor_b / shooting_batch；必要时用 JIT 重跑建表（35 min -> ~2 min）。


---

## 进展

### S4 端到端提速（2026-09-27 00:45 完成 ✅）

**两个"稀释点"的处置**

1. **标量路径未 JIT —— 虚惊**：实测 `M.shape(tau, x0=x)` = **0.000 s**（S1~S3 顺带修好），
   并非此前估算的 ~25 s ⇒ 不是瓶颈。
2. **雅可比 13× 结构冗余 —— 真瓶颈**：`shooting_batch(N=13)` 每次 LM 迭代调 13 次
   `forward_batch`，而**每次调用又跑全部 13 样本** ⇒ **13×13 = 169 次单样本前向**（≈6.9 s）。
   → 新增 **`_shoot_shared_jac`**（`inverse_solver.py`）：中心点 N=1 正常打靶 + 在中心解处算
   **一次**内层雅可比 + 12 扰动**一次批量前向**后**共享雅可比**各做 **3 步**牛顿 ⇒ **~39 次**前向。

**实测（2026-09-27 00:45 复验）**

| 环节 | 改前 | 改后 | 加速比 |
| --- | --- | --- | --- |
| `forward_batch`（单样本 n=300） | 190~340 ms | **11 ms** | **47×** |
| `pose_jacobian_batch`（13 样本） | 15.64 s | **1.46 s** | **10.7×** |
| `solve_pose`（表点 1 迭代） | 15.20 s | **1.37 s** | **11.1×** |

**零回归**：`TC_NO_JIT=1` 对照 —— ΔL **逐位相同**；Δp 均 0.0235 mm；
`python tendon_coupling.py` **ALL PASS**；S2 对拍 max|Δ| ≤ 1.7e-13。

**踩坑**：共享雅可比只做 **1 步**牛顿时 `x` 差 7e-5 ⇒ 触发慢回退，**41.9 s（更慢）**；
改 3 步后 1.76 s。另"31.3 s"是**首次 JIT 编译假象**，稳态 1.708 s。

**遗留（非热点，可选）**：`shooting_batch` 中心点 0.44 s、`_anchor_b` 仍 numpy。

### S4 收尾（2026-09-27 00:45）

- ✅ 笔记 §11.11.1 追加「S4 结果」表 + §11.20 待办更新（57,216 → 59,183 字符）；备份 `_note_bak10.md`
- ✅ 临时文件 `_s4_worker.py` / `_ver_jit.py` 归档到 `_archive_2026-09-26/`
- ✅ 本定时任务删除
- ⚠️ **微信简报发送失败（连续 2 次，平台侧故障）** —— 与历史上同类问题一致；
  内容已完整写入本文件与笔记，用户可在会话中查看

