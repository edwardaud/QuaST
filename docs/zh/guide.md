<p align="right"><a href="../en/guide.md">English</a> · <a href="../zh/guide.md">中文</a></p>

# 用户指南：二维地幔对流真实算例

本页讲 `examples/case_realistic_2d.jl` 这个算例：它求解的物理问题、所用的数值方法、每一个可调项、每一个输出文件的格式、重启规则，以及线程和内存表现。文中出现的每个取值，要么是该脚本里的字面量，要么是 `src/driver/run.jl` 中 `run_semquad_simulation` 的默认值。第 7 节说明这些结论的原始记录在哪里。

算例本身：2.5 × 1 的不可压流体盒，四壁自由滑移，顶部温度 0、底部温度 1，内部加热 `H = 10`，名义 Rayleigh 数 `Ra = 1e5`，扩散与位错两种蠕变同时生效，含两个相变并带潜热。屈服和应变弱化都关掉了。

<details>
<summary><b>目录</b></summary>

- [1. 物理问题](#1-物理问题)
- [2. 数值方法](#2-数值方法)
- [3. 参数参考](#3-参数参考)
- [4. 输出格式](#4-输出格式)
- [5. 重启与配置守卫](#5-重启与配置守卫)
- [6. 线程与内存](#6-线程与内存)
- [7. 工程记录](#7-工程记录)

</details>

## 1. 物理问题

### 1.1 区域与边界条件

计算域是 $[0,2.5]\times[0,1]$ ，由 5 × 2 个基础单元在四叉树上不断细分得到，每个叶都是正方形。 $x$ 是水平方向， $y$ 是竖直方向， $y=0$ 在底、 $y=1$ 在顶，深度即 $1-y$ 。

四壁自由滑移：法向不能流动，切向不受约束。实现上就是限制法向分量：

$$
u_x = 0 \quad (x=0,\,2.5),
\qquad
u_y = 0 \quad (y=0,\,1),
$$

其中 $u\_x$ 、 $u\_y$ 是速度的两个分量。温度在顶壁取 $T=0$ 、在底壁取 $T=1$ ，侧壁绝热。

边界条件可以逐壁设置。`bc = :free`（默认）表示四壁自由滑移；`bc = :rigid` 表示四壁无滑移，此时切向分量也被固定，取值来自壁面速度闭合 $u\_{Dx}(x,y)$ 与 $u\_{Dy}(x,y)$ 。`wall_bc` 单独覆盖某一面壁，顺序为 `(xminus, xplus, yminus, yplus)`，取值 `:same`（跟随 `bc`）、`:free` 或 `:rigid`。卡片里对应 `CASE_WALL_BC=rigid,rigid,free,free`。

壁面速度闭合本身就是位置的函数，而四面壁处在不同的 $y$ （或 $x$ ）上，所以空间变化的壁面速度不需要额外接口：取 $u\_{Dx}=Uy$ 并设 `bc = :rigid`，就是顶盖以 $U$ 拖动、底部静止，解析解 $u=(Uy,0)$ 可复现到 $10^{-10}$ ，由门禁 G238 钉住。单一闭合唯一无法表达的是角点上两壁速度不一致，而那里连续条件本来就不允许不一致。

### 1.2 方程

无量纲 Boussinesq 方程组，动量、连续性、能量：

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
| $\mathbf u=(u\_x,u\_y)$ | 速度。 |
| $p$ | 动压力。 |
| $\eta$ | 黏度。 |
| $\boldsymbol\varepsilon(\mathbf u)$ | 应变率张量 $\left(\nabla\mathbf u+\nabla\mathbf u^{\mathsf T}\right)/2$ 。 |
| $\theta=T-\bar T$ | 温度异常。 |
| $\bar T(y)$ | 1.3 节的传导基态。 |
| $\Gamma\_i$ | 第 $i$ 个相变的相分数。 |
| $B\_i$ | 该相变的浮力数。致密相取正，因此多余的相分数把物质往下压。 |
| $\mathrm{Ra}$ | Rayleigh 数。 |
| $\mathbf f=(0,f\_y)$ | 体力，只有竖直分量。 |

`Ra = 1e5` 乘的是热浮力，黏度以参考黏度为单位。1.4 节的前因子让 $\eta=1$ 对应 $T=0.5$ ，而这一温度落在冷盖内部；对流区在 $T\approx1.3$ 处，黏度只有 $10^{-3}$ 到 $10^{-2}$ ，换算成内部 Rayleigh 数是 $10^7$ 到 $10^8$ 量级。翻一遍的时间尺度 $1/\mathrm{Vrms}$ 约为 $10^{-3}$ 。

### 1.3 基态、加热与 Nusselt 数

$H$ 是能量方程 $\partial T/\partial t + \mathbf u\cdot\nabla T = \nabla^2 T + H$ 里的体积加热率。参数卡演化的对象是相对传导剖面的温度异常，而那个剖面正好带走这份加热：

$$
\bar T(y) = (1-y) + \frac H2\,y(1-y),
\qquad
\nabla^2\bar T = -H,
\qquad
\frac{\mathrm d\bar T}{\mathrm dy} = -1 + \frac H2(1-2y).
$$

代入 $T=\bar T+\theta$ ，源项被消掉，只剩 $-u\_y\,\mathrm d\bar T/\mathrm dy$ ，所以组装里不再出现显式加热项。取 $H=10$ 时， $\bar T(0.5)=1.75$ ，基态的体平均是 $0.5+H/12=1.3333$ ，顶壁与底壁的传导梯度分别是 $-6$ 和 $+4$ 。两个 Nusselt 数以这两个梯度为基准：

$$
\mathrm{Nu}_{top} = 1 + \frac H2 - \left\langle\frac{\partial\theta}{\partial y}\right\rangle_{top},
\qquad
\mathrm{Nu}_{bot} = 1 - \frac H2 - \left\langle\frac{\partial\theta}{\partial y}\right\rangle_{bot}.
$$

其中 $\mathrm{Nu}\_{top}$ 、 $\mathrm{Nu}\_{bot}$ 是顶壁与底壁的 Nusselt 数， $\theta$ 是温度异常， $H$ 是内部加热，尖括号表示沿 $x$ 平均，导数取壁面单侧值。纯传导时两者分别是 6 和 −4。进入稳态对流后应满足

$$
\mathrm{Nu}_{top} - \mathrm{Nu}_{bot} = H;
$$

实际偏离这个恒等式的量就是 `closure`，含义见 4.2 节。

### 1.4 流变

参数卡取 `eta_law = :composite`、`comp_rule = :isostress`，也就是 ASPECT 采用的那种组合方式。两种蠕变机制同时参与变形。给定温度 $T$ 、高度 $y$ 和应变率不变量 $\dot\varepsilon$ ，两者的逆黏度分别是

$$
\eta_{\mathrm{diff}}^{-1}
= A_d\exp\left(-\frac{E_d + V_d(1-y)}{T_{nd}}\right),
\qquad
\eta_{\mathrm{disl}}^{-1}
= A_s\exp\left(-\frac{E_s + V_s(1-y)}{n\,T_{nd}}\right)\dot\varepsilon^{\frac{n-1}{n}}.
$$

| 符号 | 含义 |
|---|---|
| $\eta\_{\mathrm{diff}}$ 、 $\eta\_{\mathrm{disl}}$ | 扩散蠕变与位错蠕变的等效黏度。 |
| $T\_{nd}=T+\eta\_{T0}$ | 以温度降为单位的绝对温度。参数卡取 $\eta\_{T0}=0.5$ ，即地表位于半个温度降处。 |
| $n$ | 位错蠕变的应力指数。 |
| $A\_d$ 、 $A\_s$ | 两种机制的前因子。 |
| $E\_d$ 、 $E\_s$ | 活化能，已除以 $R\Delta T$ ， $R$ 是气体常数。 |
| $V\_d$ 、 $V\_s$ | 活化体积，本卡均为 0。 |
| $\dot\varepsilon$ | $\max(\varepsilon\_{II},\dot\varepsilon\_{\min})$ ，其中 $\varepsilon\_{II}=\sqrt{\tfrac12\,\boldsymbol\varepsilon:\boldsymbol\varepsilon}$ ， $\dot\varepsilon\_{\min}=10^{-12}$ 。 |

等应力规则要求两种机制承受同一应力，应变率在两者之间分配。令 $x=\eta/\eta\_{\mathrm{disl}}$ ，分配满足

$$
k\,x + x^n = 1,
\qquad
k = \frac{\eta_{\mathrm{disl}}}{\eta_{\mathrm{diff}}},
\qquad
\eta = x\,\eta_{\mathrm{disl}}.
$$

$k$ 是两种机制的无量纲黏度比， $x$ 是等应力解中位错分支所占份额， $\eta$ 是合成黏度。 $k>0$ 时方程有唯一根； $k\to0$ 退化为纯位错， $k\to\infty$ 退化为纯扩散。参数卡把 $\eta$ 截断在 $[10^{-4},10^4]$ ，这既是求解器的保护，也直接决定条件数。实测黏度对比约为 $10^7$ 。

常数取 $A\_d=1.6240183119\times10^5$ 、 $A\_s=2.6796419238\times10^{-1}$ 、 $E\_d=12$ 、 $E\_s=17$ 、 $n=3$ 、 $\eta\_{T0}=0.5$ 。它们由 $y=0.5$ 处的两个条件定出： $\eta(T=0.5,\dot\varepsilon\_{\mathrm{ref}})=1$ 定整体尺度， $\eta(T=0.1)/\eta(T=0.9)=10^4$ 在 $\dot\varepsilon\_{\mathrm{ref}}$ 下定两种机制的比例。参考应变率 $\dot\varepsilon\_{\mathrm{ref}}=1.65\times10^3$ 来自同一算例的常黏度前驱。

### 1.5 两个相变

记深度为 $1-y$ ，第 $i$ 个相变的相分数是

$$
\Gamma_i(y,T) = \frac12\left[
1 + \tanh\left(
\frac{(z_{0,i}-y) - \mathrm{slope}_i\,(T-T_{\mathrm{ref},i})}{w_i}
\right)
\right].
$$

| 符号 | 含义 |
|---|---|
| $\Gamma\_i$ | 该相变的相分数。 |
| $z\_{0,i}$ | 参考高度。 |
| $w\_i$ | 相变宽度，以高度计。 |
| $\mathrm{slope}\_i = \mathrm dz/\mathrm dT$ | Clapeyron 斜率，即每单位温度引起的高度偏移。 |
| $T\_{\mathrm{ref},i}$ | 参考温度。 |

斜率为正时，热上涌把相变界面往下推；斜率为负时往上推。

| 相变 | 410 | 660 |
|---|---|---|
| $z\_0$ | 0.87 | 0.78 |
| $w$ | 0.025 | 0.025 |
| $\mathrm{slope}$ | +0.032 | −0.032 |
| $T\_{\mathrm{ref}}$ | 0.5 | 0.5 |
| $\Delta\rho$ （kg/m³） | 180 | 300 |
| $B$ | 1.8182 | 3.0303 |

浮力数按

$$
B = \frac{\Delta\rho}{\rho_0\alpha_0\Delta T}
$$

计算，其中 $\Delta\rho$ 是密度跳变， $\rho\_0=3300$ kg/m³ 是参考密度， $\alpha\_0=3\times10^{-5}$ 1/K 是热膨胀系数， $\Delta T=1000$ K 是温度降。上涌物质在 410 处获得浮力、在 660 处失去浮力，这正是放热相变与吸热相变该有的表现。

### 1.6 潜热

每个系数都由对应相变推出：

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
| $\rho\_0$ | 参考密度。 |
| $g=9.81$ m/s² | 重力加速度。 |
| $d=2.9\times10^6$ m | 盒深。 |
| $\Delta T=1000$ K | 温度降。 |
| $\Delta S\_A$ | 比熵变，单位 J/(kg K)。 |
| $L$ | 无量纲潜热系数。 |
| $T\_{nd}$ | 相变处的基态温度（以 $\Delta T$ 为单位）加 $T\_{\mathrm{surface}}=0.5$ 。 |
| $c\_p=1000$ J/(kg K) | 比热。 |

| 相变 | $\Delta S\_A$ | $L$ | 效果 |
|---|---|---|---|
| 410 | +49.66 J/(kg K) | −0.059364 | 给下沉流加热 |
| 660 | −82.76 J/(kg K) | +0.130596 | 给下沉流降温 |

能量方程把潜热拆成两个因子：

$$
C_{\mathrm{lat}} = 1+\sum_i L_i\frac{\partial\Gamma_i}{\partial T},
\qquad
S_{\mathrm{lat}} = -\sum_i L_i\frac{\partial\Gamma_i}{\partial y}u_y,
$$

$C\_{\mathrm{lat}}$ 是表观热容因子， $S\_{\mathrm{lat}}$ 是移流源项， $L\_i$ 是第 $i$ 个相变的潜热系数， $\Gamma\_i$ 是相分数， $u\_y$ 是竖直速度。热容因子始终落在 $[0.88,1.12]$ 之内，因为

$$
\left|\frac{\partial\Gamma_i}{\partial T}\right| \le \frac{|\mathrm{slope}_i|}{2w_i}=0.64.
$$

`CASE_LATENT=global` 用来复现 2026-11-08 之前的运行：两个相变共用一个 `latent_L = 0.11`。这个值在 410 处连符号都是错的，所以它只是一个回放开关。

### 1.7 初始条件

$$
\theta_0(x,y) = 0.05\sin(\pi y)\cos\left(\frac{2\pi x}{L_x}\right) + 0.02\sin(2\pi y)\cos\left(\frac{4\pi x}{L_x}\right),
$$

其中 $L\_x=2.5$ 是盒宽， $\theta\_0$ 是初始温度异常。它在顶壁和底壁为零，这是温度异常必须满足的条件；分布关于 $x=L\_x/2$ 对称。整个过程没有随机数：参数卡、种子和固定步数决定了每次运行都可重复。

以下开关处于关闭状态：

| 开关 | 取值 | 作用 |
|---|---|---|
| `yield_c`、`yield_taumax` | `Inf` | 关闭屈服。 |
| `yield_phi` | 0 | 关闭摩擦角。 |
| `weak_eps_ref` | 0 | 关闭应变弱化。 |
| `rho_heating`、`rho_cp` | `false` | 关闭密度加权加热。 |
| `side_thermal`、`Twall_left`、`Twall_right` | `:insulating` | 侧壁绝热（法向通量为零）。取 `:fixed` 时两侧壁分别固定为给定物理温度，约定与顶/底壁相同：`Twall - Tbar`。 |
| `H_half_life` | `Inf` | 放射性生热保持恒定。取有限值时按 `H·2^(-t/H_half_life)` 衰减，时间单位与代码一致；只能用 Picard 时间推进。 |
| `Di` | 0 | 关闭耗散加热与绝热加热。 |
| `refstate` | `nothing` | 不使用参考态。 |
| `deviatoric` | `false` | 不做偏量投影。 |
| `mass_form` | `:boussinesq` | Boussinesq 质量守恒形式。 |

### 1.8 黏弹性

Maxwell 黏弹性默认关闭。 `Params.shear_modulus` 是无量纲弹性剪切模量 $G$ ，参数卡把它暴露为 `CASE_SHEAR_MODULUS`（不设置或留空表示 0，即关闭）。取正值时它做两件事：携带偏应力历史，并把流场求解实际看到的黏度换成有效黏度

$$
\eta_{\mathrm{eff}} = \frac{1}{1/\eta + 1/(G\,\Delta t)},
\qquad
\tau^{n+1} = \frac{\tau^{n} + 2G\,\Delta t\,\hat{\boldsymbol\varepsilon}^{\,n+1}}{1 + \Delta t/\tau_{\mathrm M}},
\qquad
\tau_{\mathrm M} = \eta/G .
$$

其中 $\tau$ 是偏应力， $\hat{\boldsymbol\varepsilon}$ 是它的偏应变率， $\tau\_{\mathrm M}$ 是 Maxwell 时间， $\Delta t$ 是该步实际走过的步长。上一步的应力以滞后载荷 $\nabla\cdot(a\tau^n)$ 进入动量方程，其中 $a=\eta/(\eta+G\Delta t)$ ，正是这一项让平衡后的黏弹流动回到**材料**黏度对应的流动。取 `shear_modulus = 0` 时上述各项全部不存在，不采样任何一个量，运行结果与纯黏性版本逐位相同。该值属于 `run_config` ，因此重启无法悄悄改变它。

两个极限构成了全部行为：

- $G\Delta t \gg \eta$ ：快速松弛区，步长比 Maxwell 时间长。应力在一步之内就松弛掉， $\eta\_{\mathrm{eff}} \approx \eta$ ，流动就是纯黏性流动，记忆不可见。
- $G\Delta t \ll \eta$ ：弹性区， $\Delta t \ll \tau\_{\mathrm M}$ 。此时 $\eta\_{\mathrm{eff}} \approx G\Delta t$ ，黏度被弹性值封顶，最初几步的应力按 $2G\Delta t\,\hat{\varepsilon}$ 增长。

**物理区域会改变。** 这个上限本身是正确的物理，但它意味着黏弹卡不等于同一条卡加上记忆：凡是远大于 $G\Delta t$ 的黏度都被压向 $G\Delta t$ ，于是材料对比度中较硬的一端被削掉，求解看到的对比度大约按最硬材料超出 $G\Delta t$ 的倍数塌缩。两个实测状态：

- 演化成熟的生产状态（第 20 000 步， $\Delta t = 4.182\times10^{-7}$ ， $G\Delta t = 1.485$ ）：材料黏度对比 $3.3\times10^6$ ，实际有效对比约 $5\times10^2$ ，去掉了三到四个数量级；同一原因下实测一轮算例便宜了 33%。
- 起步段 $\Delta t = 3.6\times10^{-8}$ 、 $G\Delta t = 0.129$ ：对比度从材料 $1.0\times10^7$ 塌到有效 $1.3\times10^2$ 。

所以黏弹卡是一个**更软、也不同**的问题，它的诊断量不是它 `G = 0` 孪生卡的那一套。只要 `shear_modulus > 0` ，驱动会在第一个推进应力的步和运行结束时各报告一次：材料黏度范围与对比度、该步 `dt` 下的有效范围与对比度、 $G\Delta t/\eta$ 的范围，并用一行说明由此产生的对比度塌缩。 `shear_modulus = 0` 时什么都不打印。读 $G > 0$ 的算例之前，先登记预期的有效对比度。

**代价。** 黏弹机制本身很便宜。在生产参数卡上实测约占一步的 1%（36.2 ms：节点上重新采样材料黏度 33.5 ms，应力更新 2.65 ms，弹性载荷 0.08 ms）；检查点因三个应力场精确增加 $3\times8\times N$ 字节，从 1.730 MB 到 2.679 MB，即 1.55 倍。总步时仍可能下降，上面那 33% 就是例子：那不是这个功能的代价，那是换了一个问题。

**共转带。** Jaumann 旋转按整步的线性化 $1+\Delta t\,ad\_W$ 施加，而不是指数。它只在 $2|W|\min(\Delta t,\tau\_{\mathrm M})$ 很小时有效， `MAXWELL_COROTATION_BOUND` = 0.1 把「很小」量化。驱动每次运行报告 $2|W|\Delta t$ 、 $2\tau\_{\mathrm M}|W|$ 与逐步旋转的运行最大值，并在第一个越界的步警告一次；这条报告与上面的区域报告相互独立。运行不会中止：这是一个精度带，不是稳定性限制，把 `dt`（`dtmax`/`cfl`）调小即可回到带内。

**应力历史不是什么。** 它是网格上携带的张量状态，不是自由表面，也不是地形。本代码没有运动的上边界：黏性空气密度钩子（`rho_multiplier`）让低密度顶层能够流动，但其界面停在 Euler 网格给定的位置，因此不会产生地表起伏。黏弹卡不是一个地形观测量。

## 2. 数值方法

### 2.1 网格

网格是四叉树。每个叶用 $(\ell,i\_x,i\_y)$ 标记，树始终保持 2:1 平衡，即相邻叶最多相差一级。加密把一个四兄弟拆成四个，粗化把四个合回一个。参数卡把深度限制在 `max_level = 7` ，所以最细的叶高是盒高的 $1/(2\cdot2^{7})=1/256$ ；叶数上限是 `max_leaves = 10000` 。

加密指示子取内部面上的梯度跳变：

$$
I_K = \frac h2\int_{\partial K}
[\![\mathbf u\cdot\mathbf n]\!]^2\,\mathrm ds,
$$

顶壁和底壁再额外加一个单侧项。其中 $I\_K$ 是叶 $K$ 的指示子， $h$ 是叶高， $\mathbf u$ 是速度， $\mathbf n$ 是面法向， $[\![\cdot]\!]$ 表示跨面跳变。标记采用 Dörfler 策略：`refine_frac = 0.2` 指需要分辨的指示子总量占比，因此加密多少单元由指示子场自己决定。

平衡循环包含以下几部分：

| 机制 | 参数 | 作用 |
|---|---|---|
| 粗化 | `coarsen_frac * total` | 在完整四兄弟上，把相对阈值以下的部分粗化掉。 |
| 配额交换 | `max_leaves` | 合并误差最小的合格四兄弟，腾出叶数给要加密的地方。 |
| 迟滞 | `amr_hysteresis = 1` | 刚合并过的四兄弟不马上再拆开。 |
| 2:1 校验 | `amr_coarsen_gate = true` | 拒绝会导致一步之后又得撤销的合并。 |
| 壁面规则 | `wall_strip = 0.1`、`wall_min_level = 5` | 顶底壁条带内以及壁面行至少保持 level 5，之后再执行叶数上限。 |
| 场传递 | `transfer_mode = :nodal` | 网格变化时用节点插值搬运场变量。 |

### 2.2 空间离散与悬挂节点

采用 Taylor-Hood 单元：速度取 $Q\_p$ ，压力取 $Q\_{p-1}$ ，本卡取 $p=2$ ，即 $Q\_2$ 速度、 $Q\_1$ 压力。节点是参考正方形上的 Gauss-Legendre-Lobatto 点，速度每方向 $p+1$ 个，压力每方向 $p$ 个；单元积分用 $(p,p-1)$ 阶 Gauss-Legendre 求积。

粗叶面上存在细节点，即悬挂节点。它的取值由粗面 GLL 插值给出的迹决定，写成稀疏约束矩阵 $C$ ，自由行上是单位阵。系统按

$$
C^{\mathsf T} A\,C
$$

组装并求解，其中 $A$ 是未加约束的组装矩阵， $C$ 是约束矩阵， $C^{\mathsf T}$ 是其转置。这样对称性得以保留，对线性约束而言是精确的。边界值走同一个仿射映射，另外固定一个压力自由度，消掉常数零空间。

### 2.3 时间积分

流场是准静态的，每步解一次 Stokes。能量方程用 BDF2，第一步退化为后向 Euler，变步长比

$$
w = \frac{\Delta t}{\Delta t_{\mathrm{prev}}}
$$

进入 BDF2 系数。 $\Delta t$ 是当前步长， $\Delta t\_{\mathrm{prev}}$ 是上一步步长。参数卡取 `maxpic = 1` ，即黏度滞后一步：一步之内唯一的流动求解使用步初温度，因此该步在黏度耦合上只有一阶精度，但结果完全确定。

步长按

$$
\Delta t = \mathrm{CFL}\,\frac{h_{\min}}{|\mathbf u|_{\max}}
$$

更新，其中 $\mathrm{CFL}=0.5$ ， $h\_{\min}$ 是最小叶高， $|\mathbf u|\_{\max}$ 是最大速度模。每步最多放大 5%、最多缩小 5%，除非触发硬 CFL 截断；步长被限制在 $[\Delta t\_{\min},\Delta t\_{\max}]=[10^{-9},5\times10^{-5}]$ 内，第一步从 $\Delta t\_0=10^{-6}$ 出发并立即切到 CFL 允许值。

### 2.4 熵黏性

振荡通过逐单元熵黏性压制，黏性项加到能量算子上，每个叶一个值：

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
| $h\_e=\sqrt2\,h$ | 熵黏性的长度尺度， $h$ 是叶尺寸。 |
| $\mathcal E$ | 熵残差的全局散布。 |
| $\lvert\mathbf u\rvert$ | 单元内最大速度模。 |
| $c\_{ev}$ | 参数卡的 `ev`，取 0.11。 |
| $c\_{\max}$ | 参数卡的 `ev_cmax`，取 0.052。 |

参数卡取 `ev_alpha = 2` ，所以用 $h\_e^2$ 形式。壁面行上的黏性为零。每个时间步都重算，并写入检查点，因为没有它就无法重建上一步的能量算子。

### 2.5 两条线性求解路径与 `:auto`

直接路径求解的是消去约束与边界值之后的约化鞍点

$$
K_{\mathrm{red}}\,\mathbf u + B^{\mathsf T} p = \mathbf f,
\qquad
B_s\,\mathbf u = 0,
$$

也就是下面这个 2 × 2 块系统：

| 块 | 尺寸 | 是什么 |
|---|---|---|
| $K\_{\mathrm{red}}$ | 速度为行、速度为列 | 消去约束与边界值后的约化速度矩阵 |
| $B^{\mathsf T}$ | 速度为行、压力为列 | 梯度 |
| $B\_s$ | 压力为行、速度为列 | 散度 |
| $0$ | 压力为行、压力为列 | 零块，鞍点结构就来自它 |

其中 $K\_{\mathrm{red}}$ 是约化速度矩阵， $B$ 是梯度， $B\_s$ 是散度， $p$ 是压力。分解用 UMFPACK，并对真正的约化残差做检查。壁面加密后的自由滑移四叉树会让无平移分解失稳，因此这条路径采用平移阶梯：相对对角平移依次取 $10^{-12}$ 、 $10^{-10}$ 、 $10^{-8}$ ，最后一级对无平移算子做若干步迭代精化。`stokes_scaling = :equilibrated` 会先尝试对称平衡 $S\to DSD$ ，其中 $S$ 是鞍点矩阵、 $D$ 是对角缩放矩阵，作用是把接近零的压力模态抬到舍入误差之上；它只在本来会进入阶梯的那些求解上生效。直接路径是唯一逐位可复现的路径。

多重网格路径用 flexible GMRES 配块三角预条件：速度块用几何 V 循环，每级至少按 1.5 倍 h 粗化，采用阻尼 Jacobi 光滑，各层算子由叶局部无矩阵 apply 构建；压力 Schur 补用三因子 w-BFBT 近似。FGMRES 每 100 次重启，总上限 500 次，按真正的约化鞍点残差对 `mg_tol` 判断收敛。参数卡传 `mg_tol = 1e-8` ，比它的 `stokes_tol = 1e-3` 低五个数量级；库默认值是 `1e-10` 。

`linear = :auto` 在组装之前根据当前鞍点的两个属性决定路径。鞍点自由度按

$$
N_{\mathrm{saddle}} = 2N_u + N_p
$$

计算， $N\_u$ 是速度自由度数， $N\_p$ 是压力自由度数。判断规则是：

| 条件 | 选择 |
|---|---|
| $N\_{\mathrm{saddle}} \ge 20\,000$ | 多重网格，无论黏度对比多大。 |
| 黏度对比不超过 100，且 $N\_{\mathrm{saddle}} \ge 2\,000$ | 多重网格。 |
| 其余情况 | 直接路径，它也是逐位路径，所以平局时归它。 |

黏度对比在固定的 33 × 33 网格上采样，样本取自本次求解实际使用的黏度场。 `:auto` 一旦选中多重网格，还会强制 `mg_operator = :matrixfree` 和 `mg_threads = min(8, Threads.nthreads())` 。如果这次求解没过 `stokes_tol` 或抛出异常，同一次求解会改走直接路径并记入 `LAST_AUTO_FALLBACK` ，运行继续，计数最后打在 `auto_fallbacks` 里。显式指定 `linear = :mg` 没有这层保护，失败就是失败。本卡在 8 188 叶时鞍点为 78 720 个自由度，在 10 000 叶时为 95 308 个，都超过 20 000，所以新运行都会解析到多重网格。

## 3. 参数参考

### 3.1 参数卡的环境变量

每个可调项都是脚本启动时读取的 `CASE_*` 环境变量，第二列是参数卡给出的默认值。

| 变量 | 默认值 | 允许值 | 作用 |
|---|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | 任意路径 | 输出根目录，运行产生的全部文件都在它下面。 |
| `CASE_STEPS` | 0 | 整数 | 步数上限，与绝对步号比较。取 0 表示用模式默认值：smoke 10，short 150，spinup 10000，freeze 20000，resume 与 wallrefine 2000。 |
| `CASE_FRAME` | 0 | 整数 | 帧间隔，单位是步。取 0 表示用参数卡的 5。 |
| `CASE_MAXLEV` | 7 | 整数 | 四叉树最大层级，最细叶高为盒高的 $1/(2\cdot2^{\ell})$ 。 |
| `CASE_MAXLEAVES` | 10000 | 整数 | 叶数硬上限，每个 AMR 周期之后执行。 |
| `CASE_AMR_GATE` | 1 | 0、1 | 平衡循环中的 2:1 粗化校验，取 0 关闭。 |
| `CASE_HYST` | 1 | 整数 | 迟滞长度，以 AMR 周期计：既是冷却时长，也是平衡循环死区的倍数。 |
| `CASE_COARSEN` | 0.05 | 浮点 | 粗化配额：平衡循环的相对阈值与每周期单元数上限。 |
| `CASE_WALLMIN` | 5 | 整数 | 壁面条带内以及壁面行的最小层级，取 0 关闭壁面规则。 |
| `CASE_WALLSTRIP` | 0.1 | $(0,0.5)$ 内的浮点 | 壁面条带宽度，从每个壁面向内占盒高的比例。 |
| `CASE_SYMMETRY` | `none` | `none`、`x` | AMR 各轮的 x 镜像闭合：取 `x` 时镜像伙伴一起加密、一起合并，用于对称基准。 |
| `CASE_LINEAR` | `auto` | `auto`、`direct`、`mg`、`schur_cg`、`lagged` | 线性求解器，见 2.5 节。 |
| `CASE_EV` | 0.11 | 浮点 | 熵黏性系数。 |
| `CASE_LATENT` | `per_phase` | `per_phase`、`global` | 潜热系数：逐个相变计算，或两个相变共用历史值 `latent_L = 0.11` 。传入其他字符串会在构造 `Params` 时被拒绝。 |
| `CASE_STOKES_SCALING` | `none` | `none`、`equilibrated` | 直接鞍点的平衡开关，只对本来会失败的求解生效，健康的求解不受影响。 |
| `CASE_SHEAR_MODULUS` | 空（关闭） | 大于等于 0 的浮点 | Maxwell 黏弹性：无量纲弹性剪切模量 $G$ 。留空或取 0 表示关闭，运行结果逐位不变；取正值会改变有效黏度与物理区域，见 1.8 节。 |
| `CASE_EXTDIAG` | 0 | 0、1 | 把精确能量恒等式残差 `resid_A` 追加到 `history.csv` 。它会改变列数，所以只接受全新的输出目录。 |

`main()` 在做任何事之前就会拒绝无法识别的 `CASE_AMR_GATE`、`CASE_SYMMETRY` 或 `CASE_STOKES_SCALING` 。下面这些是字面量，不是可调项：

| 字面量 | 取值 |
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

共有七种模式：

| 模式 | 作用 |
|---|---|
| `smoke` | 跑 10 步冒烟。 |
| `short` | 跑 150 步做稳定性检查。 |
| `spinup` | 打开 AMR 的生产预积分。 |
| `freeze <ckpt>` | 冻结网格续算，把 `amr_every` 设为 0。 |
| `wallrefine <ckpt>` | 换一套壁面参数后续算。 |
| `resume <ckpt>` | 带 AMR 续算。 |
| `symseed <ckpt>` | 换一个 `CASE_SYMMETRY` 后续算。 |

三个种子模式都是复制检查点、只改写下一阶段要变的那条配置字符串，所以重启守卫对其余各项依然照查。

### 3.2 `run_semquad_simulation` 的关键字

「默认值」一列是库的默认，「参数卡」一列是 `case_realistic_2d.jl` 传入的值。

| 关键字 | 默认值 | 参数卡 | 作用 |
|---|---|---|---|
| `p` | 2 | 2 | 多项式阶：速度 $Q\_p$ ，压力 $Q\_{p-1}$ 。 |
| `maxsteps` | 10000 | `CASE_STEPS` | 停止的绝对步号。 |
| `tstop` | 1e-3 | `Inf` | 达到该无量纲时间即停止。 |
| `dt0`、`dtmax`、`dtmin`、`dt_growth` | 1e-4、`Inf`、1e-14、1.05 | 1e-6、5e-5、1e-9、1.05 | 步长及其上下界。 |
| `cfl` | 0 | 0.5 | 步长更新所用 CFL 数；取 0 表示把 `dt` 冻结在 `dt0` 。 |
| `amr_every` | 0 | 5 | 两次 AMR 之间的步数；取 0 冻结网格。 |
| `amr_hysteresis`、`amr_coarsen_gate` | 1、true | `CASE_HYST`、`CASE_AMR_GATE == 1` | 平衡循环的迟滞长度与 2:1 校验。 |
| `refine_frac`、`coarsen_frac` | 0.2、0.0 | 0.2、`CASE_COARSEN` | 平衡循环的误差份额，分别从上方和下方做 Dörfler 标记。 |
| `max_level`、`max_leaves` | 无上限 | 7、10000 | 深度上限与叶数硬上限。 |
| `wall_min_level`、`wall_strip` | 0、0.1 | 5、0.1 | 壁面规则，取 0 关闭。 |
| `symmetry`、`transfer_mode` | `:none`、`:nodal` | `CASE_SYMMETRY`、`:nodal` | 镜像闭合与 AMR 场传递方式。 |
| `advection` | `:galerkin` | `:skew` | 能量方程的对流离散形式。 |
| `ev`、`ev_cmax`、`ev_alpha` | 0、0.052、2 | 0.11、0.052、2 | 熵黏性，见 2.4 节；`ev = 0` 关闭。 |
| `maxpic`、`pic_tol`、`pic_relax`、`pic_mode` | 1、1e-6、1.0、`:picard` | 相同 | 步内非线性迭代；用 Newton 时要求 `linear=:direct` 。 |
| `linear` | `:auto` | `CASE_LINEAR` | 线性求解器，见 2.5 节。 |
| `stokes_tol` | `Inf` | 1e-3 | 真正残差超过它就中止该步；同时也是 `:auto` 回退和直接路径的接受容差。 |
| `mg_tol`、`mg_threads` | 1e-10、1 | 1e-8、1（`auto`：`min(8, nthreads)`） | 多重网格的外层容差与线程数。 |
| `mg_operator`、`mg_pressure_eta_tol` | `:matrixfree`、1.0 | 相同 | 多重网格的几何形式，以及压力块缓存的重建阈值。 |
| `stokes_scaling` | `:none` | `CASE_STOKES_SCALING` | `:none` 或 `:equilibrated` ，只对直接路径有效。 |
| `corotation_guard` | true | 默认 | 报告共转项的适用范围，越界时警告一次； `shear_modulus = 0` 时既不采样也不打印（见 1.8 节）。 |
| `regime_report` | true | 默认 | 在第一个推进应力的步与运行结束时各报告一次 Maxwell 区域； `shear_modulus = 0` 时既不计算也不打印（见 1.8 节）。 |
| `mass_form` | `:anelastic` | `:boussinesq` | 质量守恒形式。 |
| `phases`、`latent_scheme` | `[]`、`:implicit` | 两个相变、`:implicit` | 相变与潜热耦合；`maxpic = 1` 时两种格式结果相同。 |
| `restart_from` | `""` | 取决于模式 | 从哪个检查点重启。 |
| `checkpoint_path`、`checkpoint_every` | `""`、0 | `$CASE_OUT/spinup.h5`、200 | 检查点的写入位置与间隔。 |
| `callback`、`verbose`、`profile_steps` | `nothing`、false、false | 帧写入器、true、false | 步钩子、逐步输出、逐步计时。 |
| `extdiag` | false | `CASE_EXTDIAG` | 增加 `resid_A` 历史列。 |
| `initial_refine` | 0 | 0 | 第一步之前在初值上做的 AMR 轮数。 |
| `tavg_start`、`steady_tol`、`cfl_tol` | `Inf` | `Inf` | 时间平均窗口、稳态判据、CFL 中止保护。 |

函数返回 `theta`、`t`、`steps`、`history`、`history_extra`、`qt`、`ss`、`ux`、`uz`、`pv`、`avg`、`steady`、`weak`、`chi`、`chi_prev` 。 `history` 每实际跑一步记一行，列的含义见 4.2 节； `history_extra` 存放固定 18 列之后追加的列。

## 4. 输出格式

所有文件都写在 `CASE_OUT` 下。目录由程序创建；只有要用 shell 重定向把日志写进该目录时，才需要它事先存在。

### 4.1 `run.log`

这个文件不是程序创建的，而是 stdout 与 stderr 的 shell 重定向。内容包括四行头部、每步一行（`verbose=true` 时）、每 5 步一行 AMR（带加密、粗化、合并计数以及平衡循环的计数器）、每 100 步一行参数卡回调的进度，以及最后两行：

```text
[case] DONE mode=spinup steps=2000 t=<t> wall=<s> s (<s> s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=<v> Nu_bot=<v> Vrms=<v> meanT=<v> closure=<v> leaves=<n> N=<n>
```

stdout 是块缓冲的，回调每 100 步 flush 一次，因此重定向出来的日志最多滞后 100 步。错误走 stderr，不缓冲。

### 4.2 `history.csv`

在运行返回时一次写出，每实际跑一步一行：

```text
t,dt,Nu_top,Nu_bot,Vrms,meanT,N,leaves,div_max,hmin,umax,cfl,closure,usrms_surf,umax_surf,W,U,balance
```

本卡因为开了潜热，会再追加一列。

| 列 | 含义 |
|---|---|
| `t` | 步末的无量纲时间。 |
| `dt` | 刚走过的步长。 |
| `Nu_top`、`Nu_bot` | 壁面 Nusselt 数，见 1.3 节。 |
| `Vrms` | 速度模的体积均方根。 |
| `meanT` | 完整温度 $\bar T+\theta$ 的体积平均。 |
| `N` | 标量空间在网格上的节点数。 |
| `leaves` | 四叉树叶数。 |
| `div_max` | GLL 节点上 $\lvert\nabla\cdot\mathbf u\rvert$ 的最大逐点值。 |
| `hmin`、`umax` | 最小叶高与最大速度模。 |
| `cfl` | 该步的 CFL 数 $\lvert\mathbf u\rvert\_{\max}\Delta t/h\_{\min}$ 。 |
| `closure` | $\mathrm{Nu}\_{top}-\mathrm{Nu}\_{bot}-(H-I\_{base})$ 加上潜热项在面积上的时间导数，其中 $I\_{base}=\int u\_y\,\mathrm d\bar T/\mathrm dy$ 。稳态趋于零，瞬态下等于焓储存率加记账项。 |
| `usrms_surf`、`umax_surf` | 顶壁上速度模的均方根与最大值。 |
| `W`、`U` | 浮力功 $W=\langle\rho T u\_y\rangle$ 与黏性耗散 $U=\langle2\eta\,\boldsymbol\varepsilon:\boldsymbol\varepsilon\rangle$ 的体积平均。 |
| `balance` | $(\mathrm{Ra}\,W-U)/\max(\lvert\mathrm{Ra}\,W\rvert,\lvert U\rvert)$ ，力学平衡的相对残差。 |
| `closure_nolatent` | 去掉潜热项后的 `closure` 。本卡有潜热，所以这一列存在；没有相变的运行只写 18 列。 |

设 `CASE_EXTDIAG=1` 会追加一列 `resid_A` ：所引用的逐点 Nusselt 数与格式能量恒等式之间的精确残差，按单位面积计，因此对时间步通量为零，对逐点通量为 $O(h)$ 。

### 4.3 帧文件

每 `CASE_FRAME` 步存一帧，并错开每个 AMR 周期的第一步：AMR 那一轮会把速度清零，如果在 AMR 步上写帧，得到的是零速度和没有意义的 Nusselt 数。最后一帧总会补上。

`frames/frame_NNNNN_T.bin`：完整温度 $T=\bar T+\theta$ ，256 × 256 网格，小端 Float64，512 KB。 $x$ 变化最快，然后 $y$ ；在 numpy 里就是 `T = np.fromfile(path).reshape(256, 256)` ， `T[j, i]` 是 $(x\_i,y\_j)$ 处的值，画图时用 `origin="lower"` 。

`frames/frame_NNNNN_mesh.csv`：每个叶一行，按叶序排列（先 level，再 `ix`，再 `iy`），列为 `level,x0,y0,h` ，17 位有效数字。 $h$ 是叶高， $x\_0$ 、 $y\_0$ 是左下角坐标。

`frames/frame_meta.csv` 以追加方式打开，每帧一行：

```text
frame,step,t,dt,N,leaves,nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac,wall_s,ng
```

`nlev` 是不同层级的个数，`eta_clip_frac` 是黏度贴住上下限的速度节点比例，`wall_s` 是写帧器构造至今的秒数。

### 4.4 检查点

单个 HDF5 文件，每 `checkpoint_every` 步写一次，结束时再写一次，始终写到同一路径（`$CASE_OUT/spinup.h5`、`frozen.h5`、`resume.h5` 等）。它保存把轨迹接下去所需的全部状态：

| 内容 | 形状 |
|---|---|
| `leaves` | `3 × nleaves` Int32：每个叶的 level、ix、iy，已排序。 |
| `theta`、`theta_prev` | 标量节点上的两个 BDF2 温度层。 |
| `ux`、`uz` | 标量节点上的两个速度分量。 |
| `pv` | 压力空间上的压力。 |
| `ev` | 可选，每个叶一个熵黏性值，按 `leaves` 的顺序。 |
| `weak`、`weak_prev` | 可选，两个层的应变弱化场，本卡没有。 |
| `cooldown_no_refine`、`cooldown_no_coarsen` | 可选，平衡循环的迟滞状态，每个叶键一列。 |
| 属性 | `t`、`dt`、`dt_prev`、`step`、`p`、`maxlevel`、`nx0`、`ny0`、`Lx`、`cooldown_cycle`、`velocity_stale`、自由格式元数据，以及 `config`：影响轨迹的设置，写成按换行分隔的 `key=value` 字符串。 |

`velocity_stale` 标记最后一步是 AMR 步的收尾检查点：里面存的速度只是诊断性重解，重启时会被清零。

## 5. 重启与配置守卫

`restart_from` 会在第一步之前用检查点内容替换网格、`p` 和初始状态；`checkpoint_path` 让续算写到自己的文件。`maxsteps` 与 `tstop` 始终按绝对量解释。守卫把检查点里存的 `config` 与本次调用的设置逐键比较，只要有一项不同就拒绝启动：

```text
ArgumentError: restart configuration mismatch in /path/resume.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

参与比较的是会影响轨迹的那些键：

| 类别 | 键 |
|---|---|
| AMR | 节奏、各轮设置、份额、叶数上限、层级上限、壁面规则、热点与顶壁规则、对称、传递模式、迟滞与校验。 |
| 离散 | 对流形式、熵黏性系数。 |
| 非线性与时间步 | `maxpic`、`pic_tol`、`pic_relax`、`pic_mode`、`pic_reject`、`cfl`、`dtmax`、`dtmin`。 |
| 质量与投影 | `mass_form`、`deviatoric`。 |
| 加热 | 加热开关与 `H`。 |
| 流变 | 当前定律的参数；本卡是 `comp_*` 与 `eta_*` ，Tosi 定律则是 `tosi_*` 块。 |
| 屈服与弱化 | 屈服、弱化参数。 |
| 潜热 | `latent_L`、`latent_mode`、`latent_scheme` 与相变列表。 |
| 参考态 | 参考态相关设置。 |
| 缩放 | `stokes_scaling`。 |

比较以检查点里的键为准迭代，所以比检查点更晚出现的键不参与比较。`stokes_scaling` 有明确默认值：缺这个键的文件按 `none` 读取，而给这类轨迹打开该选项会被拒绝。

属于状态、不参与配置比较的内容：`p` 与网格从文件恢复，平衡循环的冷却状态也随之恢复。`dt0`、`dt_growth`、`initial_refine`、`linear`、`mg_tol`、`mg_threads` 以及其他求解器设置既不存储也不比较，因此同一条轨迹可以用不同的线性路径或不同的检查点节奏继续。

续算被拒时，把存下的值原样重设一遍，或者改用种子模式。种子模式复制检查点，只改写冲突的那一个字符串：`freeze` 把 `amr_every` 设为 0，`wallrefine` 设置 `wall_min_level` 与 `wall_strip` 并加密壁面，`symseed` 设置 `symmetry` 。源文件始终不被修改。

## 6. 线程与内存

用 `--threads=N` 启动 Julia。本卡唯一用到 Julia 线程池的地方是 `linear = :auto` 选中的无矩阵 V 循环，它被限制在八个线程（`mg_threads = min(8, Threads.nthreads())`），所以开超过八个对 `auto` 路径没有收益。单线程时 `auto` 会警告一次并串行运行。显式指定 `linear = :mg` 时 `mg_threads` 保持为 1，除非调用者自己设置；直接分解在任何情况下都是串行的。

生产状态下的实测内存：在 10 000 叶上限下，八线程多重网格路径峰值常驻约 1.3 GB，直接分解约 2.4 GB。两者都是进程峰值，包含 Julia 运行时和已加载的状态。默认运行应预留 2 GB 空闲内存，直接路径预留 3 GB。

磁盘几乎全被帧吃掉：每帧约 0.8 MB（512 KB 的 `T.bin` 加几百 kB 的网格 CSV），所以 20 000 步、间隔 5 的一轮约 3 GB；一个检查点约 2 MB。

## 7. 工程记录

默认值背后的测量、开发中发现的缺陷、性能优化过程，以及对基准和 ASPECT 的验证，都保存在工程记录，也就是 `docs/dev/` 目录。
