<p align="right"><a href="../en/guide.md">English</a> · <a href="../zh/guide.md">中文</a></p>

# 用户指南：二维地幔对流真实算例

本文描述 `examples/case_realistic_2d.jl` 搭起来的算例：物理、数值方法、每个旋钮、每个输出文件、重启，以及线程和内存行为。这里的每个参数卡值都是该脚本里的字面量，或是 `src/driver/run.jl` 中 `run_semquad_simulation` 的默认值。第 7 节指向工程记录。

算例是 2.5 × 1 的不可压流体盒，四壁自由滑移，顶部温度 0、底部温度 1，内部加热 `H = 10`，名义 Rayleigh 数 `Ra = 1e5`，复合扩散-位错流变，两个相变并带潜热。屈服和应变弱化关闭。

## 1. 物理问题

### 1.1 区域与边界条件

区域是 $[0,2.5]\times[0,1]$，由 5 × 2 个基础单元在四叉树上细分而成，叶是正方形。$x$ 水平，$y$ 竖直，$y=0$ 在底部、$y=1$ 在顶部；深度是 $1-y$。

四壁都是自由滑移，也就是法向无流动、切向自由。实现方式是固定法向分量：

$$
u_x = 0 \quad (x=0,\,2.5),
\qquad
u_y = 0 \quad (y=0,\,1),
$$

其中 $u_x$ 和 $u_y$ 是速度的两个分量。温度在顶壁为 $T=0$、在底壁为 $T=1$，侧壁绝热。

### 1.2 方程

无量纲 Boussinesq 动量、连续性与能量方程：

$$
-\nabla\cdot\left(2\eta\,\boldsymbol\varepsilon(\mathbf u)\right) + \nabla p = \mathbf f,
\qquad
\nabla\cdot\mathbf u = 0,
$$

$$
\frac{\partial\theta}{\partial t} + \mathbf u\cdot\nabla\theta
= \nabla^2\theta - u_y\frac{\mathrm d\bar T}{\mathrm dy},
\qquad
f_y = \mathrm{Ra}\,\theta - \mathrm{Ra}\sum_i B_i\left[\Gamma_i(y,T)-\Gamma_i(y,\bar T)\right],
\qquad
f_x = 0.
$$

| 符号 | 含义 |
|---|---|
| $\mathbf u=(u_x,u_y)$ | 速度。 |
| $p$ | 动压力。 |
| $\eta$ | 黏度。 |
| $\boldsymbol\varepsilon(\mathbf u)$ | 应变率张量 $\left(\nabla\mathbf u+\nabla\mathbf u^{\mathsf T}\right)/2$。 |
| $\theta=T-\bar T$ | 温度异常。 |
| $\bar T(y)$ | 1.3 节的传导基态。 |
| $\Gamma_i$ | 第 $i$ 个相变的相分数。 |
| $B_i$ | 第 $i$ 个相变的浮力数，致密相为正，所以过量相分数把物质往下推。 |
| $\mathrm{Ra}$ | Rayleigh 数。 |
| $\mathbf f=(0,f_y)$ | 体力，只有竖直分量。 |

`Ra = 1e5` 乘热浮力，黏度以参考黏度为单位。1.4 节的前因子让 $T=0.5$ 时 $\eta=1$，这一点落在冷盖里，所以对流内部在 $T\approx1.3$ 处运行，$\eta$ 在 $10^{-3}$ 到 $10^{-2}$ 之间，内部 Rayleigh 数在 $10^7$ 到 $10^8$ 量级。一次翻转时间 $1/\mathrm{Vrms}$ 约 $10^{-3}$。

### 1.3 基态、加热与 Nusselt 数

$H$ 是能量方程 $\partial T/\partial t + \mathbf u\cdot\nabla T = \nabla^2 T + H$ 中的体积加热项。参数卡演化的是围绕传导剖面的异常，该剖面恰好带走这份加热：

$$
\bar T(y) = (1-y) + \frac H2\,y(1-y),
\qquad
\nabla^2\bar T = -H,
\qquad
\frac{\mathrm d\bar T}{\mathrm dy} = -1 + \frac H2(1-2y).
$$

把 $T=\bar T+\theta$ 代入会消掉源项，只留下 $-u_y\,\mathrm d\bar T/\mathrm dy$，所以组装里没有显式加热项。$H=10$ 时，$\bar T(0.5)=1.75$，基态的体平均是 $0.5+H/12=1.3333$，传导壁面梯度在顶部是 $-6$、底部是 $+4$。两个 Nusselt 数用这些梯度作参考：

$$
\mathrm{Nu}_{top} = 1 + \frac H2 - \left\langle\frac{\partial\theta}{\partial y}\right\rangle_{top},
\qquad
\mathrm{Nu}_{bot} = 1 - \frac H2 - \left\langle\frac{\partial\theta}{\partial y}\right\rangle_{bot}.
$$

这里 $\mathrm{Nu}_{top}$ 和 $\mathrm{Nu}_{bot}$ 是顶壁与底壁的 Nusselt 数，$\theta$ 是温度异常，$H$ 是内部加热，尖括号是沿 $x$ 的平均，导数取壁面单侧值。传导态下两者分别为 6 和 -4。稳态对流满足

$$
\mathrm{Nu}_{top} - \mathrm{Nu}_{bot} = H;
$$

偏离这个恒等式的量就是 `closure`（4.2 节）。

### 1.4 流变

`eta_law = :composite`，`comp_rule = :isostress`，即 ASPECT 用的分配方式。两种机制同时变形；温度 $T$、高度 $y$ 和应变率不变量 $\dot\varepsilon$ 处，两者的逆黏度是

$$
\eta_{\mathrm{diff}}^{-1}
= A_d\exp\left(-\frac{E_d + V_d(1-y)}{T_{nd}}\right),
\qquad
\eta_{\mathrm{disl}}^{-1}
= A_s\exp\left(-\frac{E_s + V_s(1-y)}{n\,T_{nd}}\right)\dot\varepsilon^{\frac{n-1}{n}}.
$$

| 符号 | 含义 |
|---|---|
| $\eta_{\mathrm{diff}}$、$\eta_{\mathrm{disl}}$ | 扩散蠕变与位错蠕变的等效黏度。 |
| $T_{nd}=T+\eta_{T0}$ | 以温度降为单位的绝对温度，参数卡取 $\eta_{T0}=0.5$，所以地表在半个温度降处。 |
| $n$ | 位错蠕变的应力指数。 |
| $A_d$、$A_s$ | 两种机制的前因子。 |
| $E_d$、$E_s$ | 除以 $R\Delta T$ 的活化能，$R$ 是气体常数。 |
| $V_d$、$V_s$ | 活化体积，本卡都取 0。 |
| $\dot\varepsilon$ | $\max(\varepsilon_{II},\dot\varepsilon_{\min})$，$\varepsilon_{II}=\sqrt{\tfrac12\,\boldsymbol\varepsilon:\boldsymbol\varepsilon}$，$\dot\varepsilon_{\min}=10^{-12}$。 |

等应力规则让两种机制承受同一应力，应变率在两者之间分配。令 $x=\eta/\eta_{\mathrm{disl}}$，分配满足

$$
k\,x + x^n = 1,
\qquad
k = \frac{\eta_{\mathrm{disl}}}{\eta_{\mathrm{diff}}},
\qquad
\eta = x\,\eta_{\mathrm{disl}}.
$$

其中 $k$ 是两种机制的无量纲黏度比，$x$ 是等应力解给出的位错分支份额，$\eta$ 是合成黏度。$k>0$ 时根唯一；$k\to0$ 是纯位错极限，$k\to\infty$ 是纯扩散极限。参数卡把 $\eta$ 截到 $[10^{-4},10^4]$，这是求解器的安全网，也决定条件数；实测黏度对比约 $10^7$。

常数是 $A_d=1.6240183119\times10^5$、$A_s=2.6796419238\times10^{-1}$、$E_d=12$、$E_s=17$、$n=3$、$\eta_{T0}=0.5$。它们在 $y=0.5$ 处解两个条件：$\eta(T=0.5,\dot\varepsilon_{\mathrm{ref}})=1$ 定共同尺度，$\eta(T=0.1)/\eta(T=0.9)=10^4$ 在 $\dot\varepsilon_{\mathrm{ref}}$ 下定分支比。参考应变率 $\dot\varepsilon_{\mathrm{ref}}=1.65\times10^3$ 来自同一算例的常黏度前驱。

### 1.5 两个相变

记深度为 $1-y$，第 $i$ 个相变的相分数是

$$
\Gamma_i(y,T) = \frac12\left[
1 + \tanh\left(
\frac{(z_{0,i}-y) - \mathrm{slope}_i\,(T-T_{\mathrm{ref},i})}{w_i}
\right)
\right].
$$

| 符号 | 含义 |
|---|---|
| $\Gamma_i$ | 第 $i$ 个相变的相分数。 |
| $z_{0,i}$ | 参考高度。 |
| $w_i$ | 相变宽度，以高度计。 |
| $\mathrm{slope}_i = \mathrm dz/\mathrm dT$ | Clapeyron 斜率，即每单位温度的高度偏移。 |
| $T_{\mathrm{ref},i}$ | 参考温度。 |

热上涌在正斜率下把相变往下推，在负斜率下往上推。

| 相变 | 410 | 660 |
|---|---|---|
| $z_0$ | 0.87 | 0.78 |
| $w$ | 0.025 | 0.025 |
| $\mathrm{slope}$ | +0.032 | -0.032 |
| $T_{\mathrm{ref}}$ | 0.5 | 0.5 |
| $\Delta\rho$（kg/m³） | 180 | 300 |
| $B$ | 1.8182 | 3.0303 |

浮力数是

$$
B = \frac{\Delta\rho}{\rho_0\alpha_0\Delta T},
$$

其中 $\Delta\rho$ 是密度跳变，$\rho_0=3300$ kg/m³ 是参考密度，$\alpha_0=3\times10^{-5}$ 1/K 是热膨胀系数，$\Delta T=1000$ K 是温度降。上涌在 410 处获得浮力、在 660 处失去浮力，这就是放热相变与吸热相变的行为。

### 1.6 潜热

每个系数由各自相变导出：

$$
\gamma = \frac{\mathrm{slope}\,\rho_0 g d}{\Delta T},
\qquad
\Delta S_A = \frac{\gamma\,\Delta\rho}{\rho_0^2},
\qquad
L = -\frac{T_{nd}\,\Delta S_A}{c_p}.
$$

| 符号 | 含义 |
|---|---|
| $\gamma$ | Clapeyron 斜率，单位 Pa/K。 |
| $\rho_0$ | 参考密度。 |
| $g=9.81$ m/s² | 重力加速度。 |
| $d=2.9\times10^6$ m | 盒深。 |
| $\Delta T=1000$ K | 温度降。 |
| $\Delta S_A$ | 比熵变，单位 J/(kg K)。 |
| $L$ | 无量纲潜热系数。 |
| $T_{nd}$ | 相变处基态温度（以 $\Delta T$ 为单位）加 $T_{\mathrm{surface}}=0.5$。 |
| $c_p=1000$ J/(kg K) | 比热。 |

| 相变 | $\Delta S_A$ | $L$ | 读法 |
|---|---|---|---|
| 410 | +49.66 J/(kg K) | -0.059364 | 加热下沉流 |
| 660 | -82.76 J/(kg K) | +0.130596 | 冷却下沉流 |

能量方程把它们拆成两个因子：

$$
C_{\mathrm{lat}} = 1+\sum_i L_i\frac{\partial\Gamma_i}{\partial T},
\qquad
S_{\mathrm{lat}} = -\sum_i L_i\frac{\partial\Gamma_i}{\partial y}u_y,
$$

其中 $C_{\mathrm{lat}}$ 是表观热容因子，$S_{\mathrm{lat}}$ 是移流源，$L_i$ 是第 $i$ 个相变的潜热系数，$\Gamma_i$ 是相分数，$u_y$ 是竖直速度。容量因子留在 $[0.88,1.12]$ 内，因为

$$
\left|\frac{\partial\Gamma_i}{\partial T}\right| \le \frac{|\mathrm{slope}_i|}{2w_i}=0.64.
$$

`CASE_LATENT=global` 复现 2026-11-08 之前的运行：两个相变共用一个 `latent_L = 0.11`。这个值在 410 处符号相反，只是回放开关。

### 1.7 初始条件

$$
\theta_0(x,y) = 0.05\sin(\pi y)\cos\left(\frac{2\pi x}{L_x}\right)
+ 0.02\sin(2\pi y)\cos\left(\frac{4\pi x}{L_x}\right),
$$

其中 $L_x=2.5$ 是盒宽，$\theta_0$ 是初始温度异常。它在顶壁和底壁为零，这是异常必须满足的；它关于 $x=L_x/2$ 对称。没有随机数：参数卡、种子和固定步数让一次运行确定。

关闭的开关如下：

| 开关 | 值 | 作用 |
|---|---|---|
| `yield_c`、`yield_taumax` | `Inf` | 关闭屈服。 |
| `yield_phi` | 0 | 关闭摩擦角。 |
| `weak_eps_ref` | 0 | 关闭应变弱化。 |
| `rho_heating`、`rho_cp` | `false` | 关闭密度加权加热。 |
| `Di` | 0 | 关闭耗散与绝热加热。 |
| `refstate` | `nothing` | 不用参考态。 |
| `deviatoric` | `false` | 不做偏量投影。 |
| `mass_form` | `:boussinesq` | Boussinesq 质量形式。 |

## 2. 数值方法

### 2.1 网格

网格是四叉树。叶是 $(\ell,i_x,i_y)$，树保持 2:1 平衡：共面邻居最多差一级。加密把一个四兄弟拆开，粗化把四个合成一个。参数卡把深度限制在 `max_level = 7`，所以最细叶高是盒高的 $1/(2\cdot2^{7})=1/256$；叶数上限是 `max_leaves = 10000`。

指示子是内部面上的梯度跳变：

$$
I_K = \frac h2\int_{\partial K}
[\![\mathbf u\cdot\mathbf n]\!]^2\,\mathrm ds,
$$

顶底壁再加一个单侧项。其中 $I_K$ 是叶 $K$ 的指示子，$h$ 是叶高，$\mathbf u$ 是速度，$\mathbf n$ 是面法向，$[\![\cdot]\!]$ 是跨面跳变。标记用 Dörfler 策略：`refine_frac = 0.2` 是要分辨的指示子总量份额，所以需要多少单元由指示子场决定。

平衡循环的各机制如下：

| 机制 | 参数 | 作用 |
|---|---|---|
| 粗化 | `coarsen_frac * total` | 在一个完整四兄弟上，从低于相对阈值的部分粗化。 |
| 预算交换 | `max_leaves` | 通过合并误差最低的合格四兄弟来支付加密。 |
| 迟滞 | `amr_hysteresis = 1` | 刚合并过的四兄弟不立即再拆。 |
| 2:1 预检 | `amr_coarsen_gate = true` | 拒绝会迫使下一步撤销的合并。 |
| 壁面规则 | `wall_strip = 0.1`、`wall_min_level = 5` | 顶底壁条带内和壁面行至少保持 level 5，之后执行叶预算。 |
| 场传递 | `transfer_mode = :nodal` | 网格之间用节点插值搬场。 |

### 2.2 空间与悬挂节点

Taylor-Hood 对：速度在 $Q_p$，压力在 $Q_{p-1}$，本卡取 $p=2$（$Q_2$ 速度、$Q_1$ 压力）。节点是参考正方形上的 Gauss-Legendre-Lobatto 点，速度每个方向 $p+1$ 个，压力每个方向 $p$ 个；单元积分用 $(p,p-1)$ 阶 Gauss-Legendre 求积。

粗叶面上的细节点是悬挂节点。它的值取粗面 GLL 插值给出的粗侧迹，写成稀疏约束矩阵 $C$（自由行上是单位阵）。系统按

$$
C^{\mathsf T} A\,C
$$

组装并求解，其中 $A$ 是未加约束的组装矩阵，$C$ 是约束矩阵，$C^{\mathsf T}$ 是它的转置。这样保持对称性，对线性约束是精确的。边界值进入同一个仿射映射，另外固定一个压力自由度以去掉常数零空间。

### 2.3 时间积分

流动是准静态的：每步解一次 Stokes。能量方程用 BDF2，第一步用后向 Euler，变步长比

$$
w = \frac{\Delta t}{\Delta t_{\mathrm{prev}}}
$$

进入 BDF2 系数。这里 $\Delta t$ 是当前步长，$\Delta t_{\mathrm{prev}}$ 是上一步步长。参数卡取 `maxpic = 1`，黏度滞后：一步中唯一的流动求解用步初温度，所以该步在黏度耦合上一阶，并且是确定的。

步长按

$$
\Delta t = \mathrm{CFL}\,\frac{h_{\min}}{|\mathbf u|_{\max}}
$$

更新，其中 $\mathrm{CFL}=0.5$，$h_{\min}$ 是最小叶高，$|\mathbf u|_{\max}$ 是最大速度模。步长每步最多增长 5%，每步最多缩小 5%，除非硬 CFL 切断触发；它被夹在 $[\Delta t_{\min},\Delta t_{\max}]=[10^{-9},5\times10^{-5}]$，第一步从 $\Delta t_0=10^{-6}$ 出发并切到 CFL 值。

### 2.4 熵黏性

振荡由加到能量算子上的逐单元熵黏性控制，每个叶一个值：

$$
\nu_e = \frac{c_{ev}\,h_e^2 R}{\mathcal E},
\qquad
\nu_{\max} = c_{\max}\,|\mathbf u|\,h_e,
\qquad
\nu = \min(\nu_{\max},\nu_e).
$$

| 符号 | 含义 |
|---|---|
| $\nu$ | 该叶实际加入的熵黏性。 |
| $R$ | 单元内 $\theta$ 方程残差的最大绝对值。 |
| $h_e=\sqrt2\,h$ | 熵黏性的长度尺度，$h$ 是叶尺寸。 |
| $\mathcal E$ | 熵残差的全局散布。 |
| $\lvert\mathbf u\rvert$ | 单元内最大速度模。 |
| $c_{ev}$ | 参数卡的 `ev`，取 0.11。 |
| $c_{\max}$ | 参数卡的 `ev_cmax`，取 0.052。 |

参数卡取 `ev_alpha = 2`，所以用 $h_e^2$ 形式。黏性在壁面行上为零，每步重算；它存进检查点，因为没有它就无法重建上一步的能量算子。

### 2.5 两条线性路径与 `:auto`

直接路径形成约束和边界消去后的约化鞍点

$$
\begin{bmatrix}
K_{\mathrm{red}} & B^{\mathsf T}\\
B_s & 0
\end{bmatrix}
\begin{bmatrix}\mathbf u\\ p\end{bmatrix}
=
\begin{bmatrix}\mathbf f\\ 0\end{bmatrix},
$$

其中 $K_{\mathrm{red}}$ 是约化速度矩阵，$B$ 是梯度，$B_s$ 是散度，$p$ 是压力。它用 UMFPACK 分解，并检查真正的约化残差。壁面加密的自由滑移树会让无平移分解不稳定，所以该路径走平移阶梯：相对对角平移 $10^{-12}$、$10^{-10}$、$10^{-8}$，最后一级对无平移算子做若干步迭代精化。`stokes_scaling = :equilibrated` 先试对称平衡 $S\to DSD$，其中 $S$ 是鞍点矩阵、$D$ 是对角缩放矩阵，把接近零的压力模态抬到舍入下限之上，只在本来会进阶梯的求解上生效。直接路径是唯一逐位可复现的路径。

多重网格路径是 flexible GMRES 的块三角预条件：速度块用几何 V 循环（每级至少 1.5 倍 h 粗化、阻尼 Jacobi 光滑、每级由叶局部无矩阵 apply 构建），压力 Schur 补用三因子 w-BFBT 近似。FGMRES 每 100 次重启，上限 500 次，按真正的约化鞍点残差对 `mg_tol` 停止。参数卡传 `mg_tol = 1e-8`，比它的 `stokes_tol = 1e-3` 低五个数量级；库默认是 `1e-10`。

`linear = :auto` 在组装前按当前鞍点的两个性质定路径，鞍点自由度按

$$
N_{\mathrm{saddle}} = 2N_u + N_p
$$

计，其中 $N_u$ 是速度自由度数，$N_p$ 是压力自由度数。规则是：

| 条件 | 选择 |
|---|---|
| $N_{\mathrm{saddle}} \ge 20\,000$ | 多重网格，任何黏度对比。 |
| 黏度对比不超过 100，且 $N_{\mathrm{saddle}} \ge 2\,000$ | 多重网格。 |
| 其余 | 直接路径，也是逐位路径，所以平局归它。 |

黏度对比在固定的 33 × 33 网格上采样，样本取自本次求解实际用的黏度场。`:auto` 选中多重网格时还会强制 `mg_operator = :matrixfree` 和 `mg_threads = min(8, Threads.nthreads())`。该次求解没过 `stokes_tol` 或抛异常时，同一次求解改走直接路径并计入 `LAST_AUTO_FALLBACK`；运行继续，计数在末尾打成 `auto_fallbacks`。显式 `linear = :mg` 没有这层保护网，失败就报错。本卡的状态在 8 188 叶时鞍点是 78 720 个自由度，在 10 000 叶时是 95 308 个，都高于 20 000，所以新运行解析到多重网格。

## 3. 参数参考

### 3.1 参数卡的环境变量

每个旋钮都是脚本启动时读的 `CASE_*` 环境变量，第二列是参数卡的默认值。

| 变量 | 默认值 | 允许值 | 作用 |
|---|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | 任意路径 | 输出根目录。运行写出的所有东西都在它下面。 |
| `CASE_STEPS` | 0 | 整数 | 步数预算，与绝对步号比较。0 选模式默认：smoke 10，short 150，spinup 10000，freeze 20000，resume 和 wallrefine 2000。 |
| `CASE_FRAME` | 0 | 整数 | 帧间隔，单位步。0 用参数卡的 5。 |
| `CASE_MAXLEV` | 7 | 整数 | 四叉树最大层级，所以最细叶高是盒高的 $1/(2\cdot2^{\ell})$。 |
| `CASE_MAXLEAVES` | 10000 | 整数 | 硬叶预算，每个 AMR 周期之后执行。 |
| `CASE_AMR_GATE` | 1 | 0、1 | 平衡循环的 2:1 粗化预检。0 关闭。 |
| `CASE_HYST` | 1 | 整数 | AMR 周期计的迟滞长度：冷却时间，也是平衡循环的死区倍数。 |
| `CASE_COARSEN` | 0.05 | 浮点 | 粗化份额：平衡循环的相对阈值和每周期单元上限。 |
| `CASE_WALLMIN` | 5 | 整数 | 壁面条带内和壁面行的最小层级。0 关闭壁面规则。 |
| `CASE_WALLSTRIP` | 0.1 | $(0,0.5)$ 内的浮点 | 壁面条带宽度，从每个壁面向内占盒高的比例。 |
| `CASE_SYMMETRY` | `none` | `none`、`x` | AMR 各趟的 x 镜像闭合：`x` 让镜像伙伴一起加密和合并，用于对称基准。 |
| `CASE_LINEAR` | `auto` | `auto`、`direct`、`mg`、`schur_cg`、`lagged` | 线性求解器，见 2.5 节。 |
| `CASE_EV` | 0.11 | 浮点 | 熵黏性系数。 |
| `CASE_LATENT` | `per_phase` | `per_phase`、`global` | 潜热系数：逐个相变，或两个相变共用历史值 `latent_L = 0.11`。其他字符串会在构造 `Params` 时被拒绝。 |
| `CASE_STOKES_SCALING` | `none` | `none`、`equilibrated` | 直接鞍点的平衡。只在本来会失败的求解上生效，健康的求解不变。 |
| `CASE_EXTDIAG` | 0 | 0、1 | 把精确能量恒等式残差 `resid_A` 追加到 `history.csv`。它改变列数，所以只在新输出目录里接受。 |

`main()` 在做任何事之前拒绝未知的 `CASE_AMR_GATE`、`CASE_SYMMETRY` 或 `CASE_STOKES_SCALING`。以下设置是字面量，不属于旋钮：

| 字面量 | 值 |
|---|---|
| `Ra` | `1e5` |
| `H` | `10` |
| `p` | 2 |
| `stokes_tol` | `1e-3` |
| `mg_tol` | `1e-8` |
| `amr_every` | 5 |
| `refine_frac` | 0.2 |
| `dt0` | `1e-6` |
| `dtmax` | `5e-5` |
| `dtmin` | `1e-9` |
| `cfl` | 0.5 |
| `dt_growth` | 1.05 |
| `maxpic` | 1 |
| `pic_tol` | `1e-6` |
| `ev_cmax` | 0.052 |
| `ev_alpha` | 2 |
| `checkpoint_every` | 200 |
| 帧网格 | 256 × 256 |

七个模式是：

| 模式 | 作用 |
|---|---|
| `smoke` | 10 步冒烟。 |
| `short` | 150 步稳定性检查。 |
| `spinup` | 打开 AMR 的生产预积分。 |
| `freeze <ckpt>` | 冻结网格续算，把 `amr_every` 改成 0。 |
| `wallrefine <ckpt>` | 换一套壁面规则后续算。 |
| `resume <ckpt>` | 带 AMR 续算。 |
| `symseed <ckpt>` | 换一个 `CASE_SYMMETRY` 后续算。 |

三个种子模式复制检查点，只改写下一阶段要改的配置字符串，所以重启守卫仍然检查其余一切。

### 3.2 `run_semquad_simulation` 的关键字

默认值是库默认；参数卡一列是 `case_realistic_2d.jl` 传的值。

| 关键字 | 默认值 | 参数卡 | 作用 |
|---|---|---|---|
| `p` | 2 | 2 | 多项式阶：速度 $Q_p$，压力 $Q_{p-1}$。 |
| `maxsteps` | 10000 | `CASE_STEPS` | 停止的绝对步号。 |
| `tstop` | 1e-3 | `Inf` | 在这个无量纲时间停止。 |
| `dt0`、`dtmax`、`dtmin`、`dt_growth` | 1e-4、`Inf`、1e-14、1.05 | 1e-6、5e-5、1e-9、1.05 | 步长及其上下界。 |
| `cfl` | 0 | 0.5 | 步长更新的 CFL 数；0 把 `dt` 冻结在 `dt0`。 |
| `amr_every` | 0 | 5 | AMR 周期之间的步数；0 冻结网格。 |
| `amr_hysteresis`、`amr_coarsen_gate` | 1、true | `CASE_HYST`、`CASE_AMR_GATE == 1` | 平衡循环的迟滞长度和 2:1 闸门。 |
| `refine_frac`、`coarsen_frac` | 0.2、0.0 | 0.2、`CASE_COARSEN` | 平衡循环的累积误差份额（从上方和下方做的 Dörfler 标记）。 |
| `max_level`、`max_leaves` | 无上限 | 7、10000 | 深度上限和硬叶预算。 |
| `wall_min_level`、`wall_strip` | 0、0.1 | 5、0.1 | 壁面规则；0 关闭。 |
| `symmetry`、`transfer_mode` | `:none`、`:nodal` | `CASE_SYMMETRY`、`:nodal` | 镜像闭合与 AMR 传递。 |
| `advection` | `:galerkin` | `:skew` | 能量方程的对流形式。 |
| `ev`、`ev_cmax`、`ev_alpha` | 0、0.052、2 | 0.11、0.052、2 | 熵黏性，见 2.4 节；`ev = 0` 关闭。 |
| `maxpic`、`pic_tol`、`pic_relax`、`pic_mode` | 1、1e-6、1.0、`:picard` | 相同 | 步内非线性迭代；Newton 要求 `linear=:direct`。 |
| `linear` | `:auto` | `CASE_LINEAR` | 线性求解器，见 2.5 节。 |
| `stokes_tol` | `Inf` | 1e-3 | 真正残差超过它时中止该步；也是 `:auto` 回退和直接路径的接受容差。 |
| `mg_tol`、`mg_threads` | 1e-10、1 | 1e-8、1（`auto`：`min(8, nthreads)`） | 多重网格的外层容差和线程数。 |
| `mg_operator`、`mg_pressure_eta_tol` | `:matrixfree`、1.0 | 相同 | 多重网格几何，以及缓存压力块的重建阈值。 |
| `stokes_scaling` | `:none` | `CASE_STOKES_SCALING` | `:none` 或 `:equilibrated`，只对直接路径有效。 |
| `mass_form` | `:anelastic` | `:boussinesq` | 质量守恒形式。 |
| `phases`、`latent_scheme` | `[]`、`:implicit` | 两个相变、`:implicit` | 相变与潜热耦合；`maxpic = 1` 时两种格式一致。 |
| `restart_from` | `""` | 依模式 | 从哪个检查点重启。 |
| `checkpoint_path`、`checkpoint_every` | `""`、0 | `$CASE_OUT/spinup.h5`、200 | 检查点写到哪里、多久写一次。 |
| `callback`、`verbose`、`profile_steps` | `nothing`、false、false | 帧写入器、true、false | 步钩子、每步输出、每步时间剖分。 |
| `extdiag` | false | `CASE_EXTDIAG` | 增加 `resid_A` 历史列。 |
| `initial_refine` | 0 | 0 | 第一步之前对初值做的 AMR 周期数。 |
| `tavg_start`、`steady_tol`、`cfl_tol` | `Inf` | `Inf` | 时间平均窗口、稳态停止、CFL 中止保护。 |

函数返回 `theta`、`t`、`steps`、`history`、`history_extra`、`qt`、`ss`、`ux`、`uz`、`pv`、`avg`、`steady`、`weak`、`chi`、`chi_prev`。`history` 每实际跑一步一行，列见 4.2 节；`history_extra` 命名固定 18 列之后追加的列。

## 4. 输出格式

所有东西都写在 `CASE_OUT` 下。程序自己创建目录；只有 shell 重定向进该目录时才需要它事先存在。

### 4.1 `run.log`

程序不创建这个文件：它是 stdout 和 stderr 的 shell 重定向。里面有四行头部、每步一行（`verbose=true`）、每 5 步一行 AMR（带加密、粗化、合并计数和平衡循环的计数器）、每 100 步一行参数卡回调的进度，以及收尾的

```
[case] DONE mode=spinup steps=2000 t=<t> wall=<s> s (<s> s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=<v> Nu_bot=<v> Vrms=<v> meanT=<v> closure=<v> leaves=<n> N=<n>
```

stdout 是块缓冲的，回调每 100 步 flush 一次，所以重定向的日志最多滞后 100 步。错误走 stderr，不缓冲。

### 4.2 `history.csv`

运行返回时写一次，每实际跑一步一行：

```
t,dt,Nu_top,Nu_bot,Vrms,meanT,N,leaves,div_max,hmin,umax,cfl,closure,usrms_surf,umax_surf,W,U,balance
```

本卡因为潜热项开着，再追加一列。

| 列 | 含义 |
|---|---|
| `t` | 步末的无量纲时间。 |
| `dt` | 刚走的步长。 |
| `Nu_top`、`Nu_bot` | 壁面 Nusselt 数，见 1.3 节。 |
| `Vrms` | 速度模的体均方根。 |
| `meanT` | 完整温度 $\bar T+\theta$ 的体积平均。 |
| `N` | 网格上标量空间的节点数。 |
| `leaves` | 四叉树叶数。 |
| `div_max` | GLL 节点上 $\lvert\nabla\cdot\mathbf u\rvert$ 的最大逐点值。 |
| `hmin`、`umax` | 最小叶高和最大速度模。 |
| `cfl` | 该步的 CFL 数 $\lvert\mathbf u\rvert_{\max}\Delta t/h_{\min}$。 |
| `closure` | $\mathrm{Nu}_{top}-\mathrm{Nu}_{bot}-(H-I_{base})$ 加上潜热项在面积上的时间导数，$I_{base}=\int u_y\,\mathrm d\bar T/\mathrm dy$。稳态下趋于零，瞬态下是焓储存率加记账项。 |
| `usrms_surf`、`umax_surf` | 顶壁上速度模的均方根和最大值。 |
| `W`、`U` | 浮力功 $W=\langle\rho T u_y\rangle$ 和黏性耗散 $U=\langle2\eta\,\boldsymbol\varepsilon:\boldsymbol\varepsilon\rangle$ 的体积平均。 |
| `balance` | $(\mathrm{Ra}\,W-U)/\max(\lvert\mathrm{Ra}\,W\rvert,\lvert U\rvert)$，力学平衡的相对残差。 |
| `closure_nolatent` | 去掉潜热项的 `closure`。本卡有潜热，所以存在；没有相变的运行写 18 列。 |

`CASE_EXTDIAG=1` 时追加一列 `resid_A`：引用的逐点 Nusselt 对与格式能量恒等式之间的精确残差，按单位面积计，所以对时间步通量为零，对逐点通量是 $O(h)$。

### 4.3 帧

每 `CASE_FRAME` 步一帧，偏移到每个 AMR 周期之后的一步：AMR 那一趟会把速度清零，所以在 AMR 步上写的帧会带零速度和没有意义的 Nusselt 数。最后总会补一帧。

`frames/frame_NNNNN_T.bin`：完整温度 $T=\bar T+\theta$，在 256 × 256 网格上，小端 Float64，512 KB。$x$ 变化最快，然后 $y$；在 numpy 里就是 `T = np.fromfile(path).reshape(256, 256)`，`T[j, i]` 是 $(x_i,y_j)$ 处的值，画图用 `origin="lower"`。

`frames/frame_NNNNN_mesh.csv`：每个叶一行，按叶序（先 level，再 `ix`，再 `iy`），列为 `level,x0,y0,h`，17 位有效数字。$h$ 是叶高，$x_0$、$y_0$ 是左下角。

`frames/frame_meta.csv` 以追加方式打开，每帧一行：

```
frame,step,t,dt,N,leaves,nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac,wall_s,ng
```

`nlev` 是不同层级的个数，`eta_clip_frac` 是黏度贴住边界的速度节点比例，`wall_s` 是从帧写入器构造至今的秒数。

### 4.4 检查点

一个 HDF5 文件，每 `checkpoint_every` 步写一次，结束时再写一次，总是写到同一路径（`$CASE_OUT/spinup.h5`、`frozen.h5`、`resume.h5` 等）。它保存继续轨迹所需的完整状态：

| 内容 | 形状 |
|---|---|
| `leaves` | `3 × nleaves` Int32：每个叶的 level、ix、iy，已排序。 |
| `theta`、`theta_prev` | 标量节点上的两个 BDF2 温度层。 |
| `ux`、`uz` | 标量节点上的两个速度分量。 |
| `pv` | 压力空间上的压力。 |
| `ev` | 可选，每个叶一个熵黏性值，按 `leaves` 顺序。 |
| `weak`、`weak_prev` | 可选，两个层的应变弱化场。本卡没有。 |
| `cooldown_no_refine`、`cooldown_no_coarsen` | 可选，平衡循环的迟滞状态，每个叶键一列。 |
| 属性 | `t`、`dt`、`dt_prev`、`step`、`p`、`maxlevel`、`nx0`、`ny0`、`Lx`、`cooldown_cycle`、`velocity_stale`、自由格式元数据，以及 `config`：影响轨迹的设置，按换行分隔的 `key=value` 字符串。 |

`velocity_stale` 标记最后一步是 AMR 步的收尾检查点：存下的速度是诊断性重解，重启时会清零。

## 5. 重启与配置守卫

`restart_from` 在第一步之前用检查点内容替换网格、`p` 和初始状态，`checkpoint_path` 把续算写到自己的文件。`maxsteps` 和 `tstop` 保持绝对。守卫逐键比较检查点存的 `config` 与当前调用的设置，有任何差异就拒绝启动：

```
ArgumentError: restart configuration mismatch in /path/resume.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

比较的键是影响轨迹的那些：

| 类别 | 键 |
|---|---|
| AMR | 节奏、各趟、份额、叶预算、层级上限、壁面规则、热点与顶壁规则、对称、传递模式、迟滞与闸门。 |
| 离散 | 对流形式、熵黏性系数。 |
| 非线性与时间步 | `maxpic`、`pic_tol`、`pic_relax`、`pic_mode`、`pic_reject`、`cfl`、`dtmax`、`dtmin`。 |
| 质量与投影 | `mass_form`、`deviatoric`。 |
| 加热 | 加热开关和 `H`。 |
| 流变 | 当前定律的参数，本卡是 `comp_*` 和 `eta_*`，Tosi 定律是 `tosi_*` 块。 |
| 屈服与弱化 | 屈服和弱化参数。 |
| 潜热 | `latent_L`、`latent_mode`、`latent_scheme` 和相变列表。 |
| 参考态 | 参考态设置。 |
| 缩放 | `stokes_scaling`。 |

比较按检查点的键迭代，所以晚于检查点出现的键不比较。`stokes_scaling` 有定义好的默认值：没有这个键的文件按 `none` 读，给这种链打开该选项会被拒绝。

属于状态、不作为配置比较的内容：`p` 和网格从文件恢复，`:balanced` 冷却也随它们恢复。`dt0`、`dt_growth`、`initial_refine`、`linear`、`mg_tol`、`mg_threads` 和求解器的其他设置不存储、不比较，所以同一条轨迹可以用不同的线性路径或检查点节奏继续。

续算被拒时，重复存下的值，或用种子模式。种子模式复制检查点，只改写冲突的那个字符串：`freeze` 把 `amr_every` 设为 0，`wallrefine` 设置 `wall_min_level` 和 `wall_strip` 并加密壁面，`symseed` 设置 `symmetry`。源文件永远不会被修改。

## 6. 线程与内存

用 `--threads=N` 启动 Julia。本卡唯一使用 Julia 线程池的地方是 `linear = :auto` 选中的无矩阵 V 循环，它被限制在八个线程（`mg_threads = min(8, Threads.nthreads())`），所以超过八个对 auto 路径没有增益。单线程时 auto 路径警告一次并串行运行。显式 `linear = :mg` 保持 `mg_threads = 1`，除非调用者设置它；直接分解在任何情况下都是串行的。

生产状态的实测内存：在 10 000 叶预算下，八线程多重网格路径的峰值常驻约 1.3 GB，直接分解约 2.4 GB。两个都是进程峰值，包含 Julia 运行时和已加载状态。默认运行预算 2 GB 空闲内存，直接路径预算 3 GB。

磁盘由帧主导：每帧约 0.8 MB（512 KB 的 `T.bin` 加几百 kB 的网格 CSV），所以 20 000 步、间隔 5 的一轮约 3 GB；一个检查点约 2 MB。

## 7. 工程记录

默认值背后的测量、一路上发现的缺陷、优化战役，以及对基准和 ASPECT 的验证，都保存在[工程记录](../dev/)，也就是 `docs/dev/`。
