# ③ 打靶雅可比的解析化（变分方程）—— 阶段 1：∂f/∂y 已解析并验证

> 2026-10-13 ｜ 代码：`tools/varjac/varjac.py`、`tools/varjac/check_stage_jac.py`
> 目标：把 `shooting_batch` 里靠**有限差分**求的 `J = ∂res/∂x₀` 换成解析式。
> 现状：**最难的 ∂(A,B,G,H,c,d)/∂(u,v) 与 ∂f/∂y 已解析并逐元素对拍通过**；
> 变分传播（∂res/∂x₀、∂tip/∂x₀）与锚点项的雅可比是下一步。

---

## 1. 为什么值得做

`shooting_batch` 每个 LM 迭代要重求一次 `J = ∂res/∂x₀`，现在用 6 次（前向差分）
或 12 次（中心差分）**完整正解**（601 子步）。`docs/derivations/README.md` 统计它
占端到端耗时约 90% ⇒ 解析化是唯一能把建表/逆解再提速一个量级的推导。

## 2. 结构

打靶未知量 $y=[v;u]\in\mathbb{R}^6$，状态方程

$$
y'(s)=f(y)=M(y)^{-1}r(y),\qquad
M=\begin{bmatrix}K_{se}+A & G\\ B & K_{bt}+H\end{bmatrix},\quad
r=\begin{bmatrix}d\\ c\end{bmatrix}
$$

$A,B,G,H,c,d$ 由 `intermed_w` 的代数给出（$O(W)$ 次 3×3 运算，与状态数无关）。
敏感度矩阵 $\Phi=\partial y(s)/\partial y(0)$ 满足 $\Phi'=(\partial f/\partial y)\Phi$，$\Phi(0)=I$。
RK2 中点格式下

$$
\Phi_{n+1}=\Bigl[I+ds\,(\partial f/\partial y)_{\text{mid}}\Bigr]
\Bigl[I+\tfrac12 ds\,(\partial f/\partial y)\Bigr]\Phi_n
$$

关键恒等式：

$$
\boxed{\ \frac{\partial f}{\partial y}=M^{-1}\Bigl(\frac{\partial r}{\partial y}
-\frac{\partial M}{\partial y}\,f\Bigr)\ }
\tag{1}
$$

（$f=M^{-1}r$，故 $df=M^{-1}(dr-dM\,f)$。）

## 3. 逐丝导数的闭式

单根丝 $w$ 贡献（$\hat p$ 为反对称阵，$\tilde r_w$ 为孔位偏移）：

$$
p_b=\hat u\,\tilde r_w+v,\qquad n=|p_b|,\qquad
A_i=-\frac{\tau_w}{n^3}\hat p_b^2,\qquad B_i=\hat{\tilde r}_w A_i
$$

$$
A=\textstyle\sum A_i,\quad B=\sum B_i,\quad
G=-\sum A_i\hat{\tilde r}_w,\quad H=-\sum B_i\hat{\tilde r}_w
$$

$$
a=\sum A_i\,\hat u\,p_b,\qquad b=\sum \hat{\tilde r}_w a_i
$$

$$
c=-\hat u\,K_{bt}u-\hat v\,K_{se}(v-E_3)-f_{e,b}-b,\qquad
d=-\hat u\,K_{se}(v-E_3)-f_{e,b}-a
$$

三条基础导数（全部是**闭式**，无差分）：

$$
\frac{\partial p_b}{\partial u}=-\hat{\tilde r}_w,\qquad
\frac{\partial p_b}{\partial v}=I
\tag{2}
$$

$$
\frac{\partial \hat p^{\,2}}{\partial p_m}=e_m p^\top+p\,e_m^\top-2p_m I
\qquad\bigl(\hat p^{\,2}=pp^\top-|p|^2 I\bigr)
\tag{3}
$$

$$
\frac{\partial}{\partial p}\Bigl(-\frac{\tau}{n^3}\Bigr)=\frac{3\tau}{n^4}\,\frac{p}{n}
\tag{4}
$$

再由 $\partial A_i/\partial y=\hat p_b^2\otimes\partial_y c_{\text{oef}}+c_{\text{oef}}\cdot
(\partial\hat p_b^2/\partial p_b)\cdot\partial_y p_b$ 逐层链式即得。

**κ 的割线刚度**（`lat=True`）是唯一非线性的刚度项：

$$
e(u)=EI_0\frac{\sin\theta}{\theta},\quad \theta=|u_{xy}|\,L_{\text{strut}}
\;\Longrightarrow\;
\frac{\partial e}{\partial|u_{xy}|}=EI_0\frac{\theta\cos\theta-\sin\theta}{\theta^2}L_{\text{strut}}
\tag{5}
$$

## 4. 验证（`tools/varjac/check_stage_jac.py`）

四级闭环，5 组用例（含 $u_z\neq0$ 以检验"割线刚度只对 $u_{xy}$ 有效"）：

| 检查 | 对拍对象 | max 相对差 |
| --- | --- | --- |
| ① `intermed()` 数学搬移 | `tendon_coupling._intermed_b_vec` | 9.5e-16 |
| ② $\partial(A,B,G,H,c,d)/\partial y$ | 中心差分 | 1.5e-9 |
| ③ $\partial f/\partial y$（含式 1） | 中心差分 | **8.1e-12** |
| ④ $\partial K_{bt}/\partial u$（式 5） | 中心差分 | 1.4e-10 |

跑法：`TC_NO_JIT=1 python tools/varjac/check_stage_jac.py`（纯 numpy，0.3 秒）

### 4.1 两个踩过的坑（值得记）

1. **`_jit_integrate.py` 顶层无条件 `from numba import njit`** ⇒ 无 numba 环境
   `import` 直接失败。检查脚本不能复用它，须自带一份 `pack_params` 子集。
2. **归一化选错会造出假警报**：$\partial K_{bt}/\partial u$ 只有 ~1e-5 量级，
   而中心差分的噪声底是 $\varepsilon|K_{bt}|/(2h)\approx5.7\times10^{-11}$。
   若用它**自身**的 max 归一化，会报出 1.95e-5 的"相对误差"；改用 $K_{bt}$ 的尺度
   归一化后是 1.4e-10。**导数对拍要按被求导函数本身的尺度归一化。**

## 5. 尚未做（下一步）

1. **变分传播**：由 $\partial f/\partial y$ 积分出 $\Phi$，得 $\partial(v,u)/\partial x_0$
2. **$\partial R/\partial x_0$**：$R\leftarrow R(I+ds\,\hat u)$ 后还要 `ortho3s` 正交化；
   正交化的雅可比按 $O(ds^2)$ 忽略会引入误差（$ds\approx7.4\text{e-}3\Rightarrow$
   误差上界量级 1e-2），需要实测或补上
3. **锚点项雅可比**：段界修正 $v\mathrel{-}=K_{se}^{-1}R^\top F_s$、$u\mathrel{-}=K_{bt}^{-1}R^\top L_s$
   与末端残差 $res=R\,(\cdot)-F_s/L_s$ 都要对 $(u,v,R)$ 求导
4. **µ≠0**：此时 $\tau_w$ 依赖累积包角 $\varphi(u)$，$\partial\tau/\partial y\neq0$
5. **批量版**：现在的 `varjac.py` 是单样本；要接进 `shooting_batch` 需向量化
6. 端到端收益要实测（预期 3–6× 于雅可比本身；不是与状态数无关的免费加速）

## 6. 诚实备注

解析雅可比**不能**因为"状态数少"就自动比差分快：$\partial f/\partial y$ 的传播需要
$6\times6$ 的矩阵链，单样本下与 6 次正解同阶。真正的收益来自 ① 一次积分解出全部
6 列（差分要 6 趟）、② 省掉尾部的 12 次正解。**在没做批量与实测之前，不宣称加速倍数。**
