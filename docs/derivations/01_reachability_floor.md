# ① 线性化可达判据与残差地板：推导与验证

> 2026-10-05 ｜ 对象：`inverse_solver.LookupInverseSolver.reachability()`
> 验证脚本：`tools/check_reachability_floor.py`（只需 numpy，`python tools/check_reachability_floor.py`）

---

## 1. 记号与线性化模型

末端位姿误差 $e\in\mathbb{R}^{6}$（前 3 位位置 mm、后 3 位姿态 × 0.1 折算），
$\tau\in\mathbb{R}^6$ 为六根肌腱张力，箱约束 $0\le\tau_k\le\tau_{\max}$（本机构 60 N）。

在 $\tau$ 处取一阶展开（$J=\partial e/\partial\tau\in\mathbb{R}^{6\times6}$，即 `pose_jacobian_batch` 的输出）：

$$
e(\tau+\Delta) \;\approx\; e + J\,\Delta,\qquad \Delta \in \mathcal{B}(\tau):=\{\,d\;:\;-\tau \le d \le \tau_{\max}-\tau\,\}
$$

**线性化可达性问题**：箱约束下能达到的最小残差

$$
\text{floor}^\* \;=\; \min_{\Delta\in\mathcal{B}(\tau)} \bigl\| e + J\Delta \bigr\|_2
\tag{B}
$$

$\text{floor}^\*=0$ 当且仅当 $e \in J\,\mathcal{B}(\tau)$（线性意义下精确可达）；
误差方向落在 $J$ 的弱奇异方向之外时，无论怎么调 $\tau$ 都消不掉。

---

## 2. 判据 A（当前实现）的推导

当前实现走"最小范数方向 + 等比缩放"：

**(a) 最小范数解。** 取满足 $J\Delta=-e$ 的解中欧氏范数最小者

$$
\Delta_{\min} \;=\; -J^{\top}\bigl(JJ^{\top}+\epsilon I\bigr)^{-1} e \;=\; -J^{+}e
$$

（代码用 $10^{-10}$ 的正则化代替 SVD 截断；$\epsilon\to0$ 时即 Moore–Penrose 伪逆。）

**(b) 关键恒等式。** 因为 $J\Delta_{\min}=-e$ 精确成立，沿该方向取任意缩放 $s$ 时

$$
\bigl\| e + J(s\Delta_{\min}) \bigr\| \;=\; \|(1-s)e\|\;=\;(1-s)\,\|e\|
\tag{1}
$$

**残差是 $s$ 的线性函数** —— 这是"用 $s$ 直接读出地板"的全部依据。

**(c) 箱约束下的最大等比步长。** 对每个分量独立求 $s_k$ 使 $\tau_k+s_k \Delta_k$ 恰好触界：

$$
s_k = \begin{cases}
(\tau_{\max}-\tau_k)/\Delta_k, & \Delta_k > 0\\[2pt]
\tau_k/(-\Delta_k), & \Delta_k < 0\\[2pt]
+\infty, & \Delta_k = 0
\end{cases}
\qquad
s^\*=\min\Bigl(1,\ \min_k s_k\Bigr)
\tag{2}
$$

$$
\boxed{\ \text{floor}_A \;=\; (1-s^\*)\,\|e\|\ }
\tag{3}
$$

**(d) 为什么必须等比缩放而不是 `clip`。** 若直接用 $\mathrm{clip}(\tau+\Delta_{\min})$，
当 $\|\Delta_{\min}\|$ 远大于行程（实测 need=311 N vs 行程 25 N）时六个分量会**同时**被削，
方向被完全扭曲，一阶预测反而给出 $\text{floor}>\|e\|$ 的荒谬值（实测 #309：floor 79.7 > now 7.07）。
等比缩放保方向，才使 (1) 式可用。

---

## 3. 判据 A 的真实性质：它是 (B) 的**可行点**，因而只是上界

$s^\*\Delta_{\min}$ 定义上满足箱约束，即它是 (B) 的**一个可行点**。由 (B) 是最小化问题：

$$
\text{floor}^\* \;\le\; \bigl\|e+J(s^\*\Delta_{\min})\bigr\| \;=\; \text{floor}_A
\tag{4}
$$

> **命题 1（保守性）**：$\text{floor}_A \ge \text{floor}^\*$ 恒成立，等号当且仅当
> 最小范数方向的缩放解本身已是箱约束最小二乘的最优解。

因此判据 A **只会高估地板**，绝不会把不可达说成可达；代价是可能把可达说成不可达。
后者已被 `reachability()` 的注释定性为"局部一阶判据误判"，本推导给出其**定量下界**（§5）。

---

## 4. 判据 B 的精确解（KKT + 活跃集）

(B) 是 6 维箱约束最小二乘（严格凸 QP）。最优点的每个分量 $d_k$ 只有三种状态：
自由、贴下界 $l_k=-\tau_k$、贴上界 $u_k=\tau_{\max}-\tau_k$。先验指标集 $\mathcal{A}$ 固定后是等式约束 LS：

$$
d_k=l_k \text{ 或 } u_k\ (k\in\mathcal{A}),\qquad
\Delta_{\mathcal{F}}=-J_{\mathcal{F}}^{+}\bigl(e+J_{\mathcal{A}}d_{\mathcal{A}}\bigr)
\tag{5}
$$

KKT 号条件（$\mathcal{F}$ 为自由集，$g=J^{\top}(e+Jd)$）：

$$
g_k = 0\ (k\in\mathcal{F}),\qquad
d_k=l_k \Rightarrow g_k \ge 0,\qquad
d_k=u_k \Rightarrow g_k \le 0
\tag{6}
$$

6 维下活跃集模式只有 $3^6=729$ 种，**全部枚举 + 用 (6) 过滤**即得精确解（无需迭代），
代价约 $729$ 次小规模 LS（微秒级）。`tools/check_reachability_floor.py` 即按此实现，并用投影梯度下降独立复核。

**特例（1 维）**：$J\in\mathbb{R}^{1\times1}$ 时

$$
\text{floor}^\*=\max\bigl(0,\ \|e\|-|J|\cdot \text{room}\bigr),\qquad
\text{room}=\begin{cases}\tau_{\max}-\tau,& J e<0\\ \tau, & Je>0\end{cases}
$$

与直觉一致：**能改多少残差 = 灵敏度 × 到边界的行程**。

---

## 5. 数值验证结果（`tools/check_reachability_floor.py`）

300 组随机用例（其中 1/3 强制谱为实测的 $\sigma=[23.4,\,21.8,\,1.51,\,1.12,\,0.146,\,0.0082]$，
cond≈2850；$e$ 幅值跨 3 个数量级；$\tau$ 在箱内均匀）：

| 检查项 | 结果 |
|--------|------|
| ① 判据 A 是保守上界（命题 1） | ✅ 恒成立，300/300 无例外 |
| ② $s^\*$ 闭式 (2) vs 一维暴力网格扫描 | ✅ 全部一致（误差 < 1e-4） |
| ③ box-QP 真值：活跃集枚举 vs 投影梯度 | ✅ 全部一致（12/12，误差 < 1e-4·floor） |
| ④ A 与 B 有差距的用例 | 182/300（60.7%），相对差距最大 $0.895\,\|e\|$ |
| ⑤ **A 判"不可达"而 B 判"可达"** | **88/300（29.3%）** ← 这就是保守性代价 |

⑤ 是对 `reachability()` 注释里那个"必然可达却被判死"反例的量化：**约三成**。

---

## 6. 落地建议

1. **地板换成箱-QP 真值**（判据 B）。成本：729 次 6 维小 LS，相对一次打靶（~20 s）可忽略；
   收益：消除 §5⑤ 的三成误判。若不想枚举，20 次投影梯度下降也能到 1e-6 精度。
2. **`blocked`（$s^\*\le10^{-9}$）继续作为唯一的提前退出条件** —— 它要求最小范数方向在局部被**完全**挡住，
   与命题 1 无关，是安全的紧判据。
3. **`feasible` 语义改为 $\text{floor}_B \le 0.5\|e\|$**，并在结果里同时输出两种地板，
   便于回看历史日志时对比口径变化。
4. **薄流形结论的定量形式**：沿 $J$ 的第 $i$ 个左奇异方向把末端推 $\delta$，需要
   $\|\Delta\|\ge \delta/\sigma_i$；本机构 $\sigma_{\min}=0.0082$ 对应约 $122\ \text{N/mm}$，
   故 5 mm 偏移需 $\gtrsim600\ \text{N}\approx10\,\tau_{\max}$ —— 与实测 732 N / 311 N 同量级，
   **"算法问题"的旧定性应改为"执行器行程不足"**（与 `_TODO.md §1.5` 的结论一致）。
