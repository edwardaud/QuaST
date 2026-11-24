<p align="right"><a href="../en/guide.md">English</a> · <a href="../zh/guide.md">中文</a></p>

# 用户指南：二维地幔对流真实算例

本页是参考页：讲 `examples/case_realistic_2d.jl` 求解的物理问题、边界条件、数值方法、自适应策略、线性求解器、全部参数、输出格式、重启规则，以及线程与内存表现。要按步骤操作，见[快速开始](quickstart.md)；要跑现成的算例，见[算例](examples.md)。

算例本身： $2.5 \times 1$ 的不可压流体盒，四壁自由滑移，顶部温度 0、底部温度 1，内部加热 `H = 10`，名义 Rayleigh 数 `Ra = 1e5`；扩散与位错两种蠕变同时生效，含两个相变并带潜热。屈服和应变弱化都关掉了。本页出现的每个取值，要么是脚本里的字面量，要么是 `src/driver/run.jl` 中 `run_semquad_simulation` 的默认值，要么是仓库内的实测记录。

<details>
<summary><b>目录</b></summary>

- [1. 物理问题](#1-物理问题)
- [2. 边界条件参考](#2-边界条件参考)
- [3. 离散](#3-离散)
- [4. 自适应策略](#4-自适应策略)
- [5. 线性求解器与 `:auto` 规则](#5-线性求解器与-auto-规则)
- [6. 参数参考](#6-参数参考)
- [7. 输出格式](#7-输出格式)
- [8. 重启与配置守卫](#8-重启与配置守卫)
- [9. 线程与内存](#9-线程与内存)
- [10. 下一步读什么](#10-下一步读什么)

</details>

## 1. 物理问题

### 1.1 区域与默认边界

计算域是 $[0,2.5]\times[0,1]$ ，由 5 × 2 个基础单元在四叉树上不断细分得到，每个叶都是正方形。 $x$ 是水平方向， $y$ 是竖直方向， $y=0$ 在底、 $y=1$ 在顶，深度即 $1-y$ 。

默认四壁自由滑移：法向不能流动，切向不受约束。实现上就是限制法向分量：

$$
u_x = 0 \quad (x=0,\,2.5),
\qquad
u_y = 0 \quad (y=0,\,1),
$$

其中 $u\_x$ 、 $u\_y$ 是速度的两个分量。温度在顶壁取 $T=0$ 、在底壁取 $T=1$ ，侧壁绝热。

四壁条件可以逐壁覆盖： `bc` 给默认壁型， `wall_bc` 逐壁覆盖， `wall_u` 、 `wall_un` 、 `wall_t` 、 `wall_tn` 给逐壁数据。四种壁型、四组数据的约定、压力基准、角点归属与每一种被拒绝的配置都由第 2 节拥有；本页其余各节只按需引用它。

### 1.2 控制方程

无量纲 Boussinesq 方程组，动量、连续性与能量：

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
| $\boldsymbol\varepsilon(\mathbf u)$ | 应变率张量 $(\nabla\mathbf u+\nabla\mathbf u^{\mathsf T})/2$ 。 |
| $\theta=T-\bar T$ | 温度异常。 |
| $\bar T(y)$ | 1.3 节的传导基态。 |
| $\Gamma\_i$ | 第 $i$ 个相变的相分数。 |
| $B\_i$ | 该相变的浮力数。致密相取正，因此多余的相分数把物质往下压。 |
| $\mathrm{Ra}$ | Rayleigh 数。 |
| $\mathbf f=(0,f\_y)$ | 体力，只有竖直分量。 |

`Ra = 1e5` 乘的是热浮力，黏度以参考黏度 `eta_ref` 为单位（见 1.9 节）。1.4 节的前因子让 $\eta=1$ 对应 $T=0.5$ ，而这一温度落在冷盖内部；对流区在 $T\approx1.3$ 处，黏度只有 $10^{-3}$ 到 $10^{-2}$ ，换算成内部 Rayleigh 数是 $10^7$ 到 $10^8$ 量级。翻转一次的时间尺度 $1/\mathrm{Vrms}$ 约为 $10^{-3}$ 。

### 1.3 基态、内部加热与 Nusselt 数

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

实际偏离这个恒等式的量就是 `closure`，含义见 7.2 节。

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

`eta_law` 的其余取值： `:expT` 用 `deta_T`； `:expTz` 用 `eta_b` 与 `eta_c`； `:arrhenius` 用 `eta_E` 与 `eta_T0`； `:tosi` 用 `TosiLaw`，它自带塑性分支； `:composite` 用 `comp_*`。 `run_semquad_simulation` 只接受这五种，`:const` 只存在于 `Params`；常黏度用 `deta_T = 1` 的 `:expT` 表示。参数卡没覆盖的定律可以传 `eta_custom=(x, z) -> ...`，它在 Stokes 组装时接管温度定律。

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

两个系数绝对值之和为 0.18996，因此表观热容因子不会越出 $[0.88,1.12]$ 。若装配时算出的因子会变成负数，代码直接报错，因为那等于让能量方程在时间上反向推进。

`CASE_LATENT=global` 让两个相变共用一个历史值 `latent_L = 0.11`，只用于回放早期运行。这个值在 410 处连符号都是错的，所以它不是一个可用的物理选项。

### 1.7 初始条件

$$
\theta_0(x,y) = 0.05\sin(\pi y)\cos\left(\frac{2\pi x}{L_x}\right) + 0.02\sin(2\pi y)\cos\left(\frac{4\pi x}{L_x}\right),
$$

其中 $L\_x=2.5$ 是盒宽， $\theta\_0$ 是初始温度异常。它在顶壁和底壁为零，这是温度异常必须满足的条件；分布关于 $x=L\_x/2$ 对称。整个过程没有随机数：初值、参数卡与固定步数决定了每次运行都可重复。

### 1.8 黏弹性

Maxwell 黏弹性默认关闭。 `Params.shear_modulus` 是无量纲弹性剪切模量 $G$ ，默认值 `0` ；参数卡把它暴露为 `CASE_SHEAR_MODULUS`（不设置或留空同样表示 0，即关闭）。取正值时它做两件事：携带偏应力历史，并把流场求解实际看到的黏度换成有效黏度

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

这两份报告都可以关闭：把 `regime_report` 或 `corotation_guard`（默认都是 `true`）设为 `false`，输出即可保持安静，运行本身不受影响。

应力更新用该步实际走过的 `dt`，并且在变步长序列上具有一阶一致性，因此没有固定弹性步长这个选项： `dt` 照常按 CFL 规则更新，无需额外设置。

**应力历史不是什么。** 它是网格上携带的张量状态，不是自由表面，也不是地形。本代码没有运动的上边界：黏性空气密度钩子（`rho_multiplier`）让低密度顶层能够流动，但其界面停在 Euler 网格给定的位置，因此不会产生地表起伏。黏弹卡不是一个地形观测量。

### 1.9 参考黏度与 Rayleigh 数

1.2 节的无量纲方程里藏着一个容易忽略的标度选择： `Ra` 是相对一个参考黏度 `eta_ref` 定义的，

$$
\mathrm{Ra} = \frac{\rho g \alpha \Delta T d^3}{\kappa\,\eta_{\mathrm{ref}}},
$$

流场求解中凡出现材料定律给出的黏度的地方都用 `eta_law/eta_ref` ，包括 `eta_multiplier` 与 1.8 节的 Maxwell 有效黏度闭合。 `eta_ref` 就是 `Params.eta_ref` ，默认值 `1.0` ，单位是材料定律返回值的单位。参数卡通过 `CASE_ETA_REF` 暴露它（默认也是 `1.0`），把它打印在头部，并写入检查点元数据。

只有把被缩放的定律和同样的 `eta_ref` 缩放配对，这个约定才是自洽的。把 `comp_Ad` 与 `comp_As` 同乘 `1/c` ，定律返回的每个黏度都乘 $c$ ；此时取 `eta_ref = c` 流动不变，因为比值 `eta/eta_ref` 没变，而取 `eta_ref = 1` 时同一个 `Ra` 描述的是另一个物理 Rayleigh 数。截断上下限是绝对量： `eta_min` 与 `eta_max` 以定律的单位给出，并且在归一化之前施加，所以定律乘 $c$ 时上下限也要乘 $c$ 。默认值 `1.0` 让所有已有运行逐位不变。

当 `eta_ref != 1` 时，驱动会在第一次流动求解时和运行结束时各报告一次这个约定： `Ra` 、 `eta_ref` 、归一化场 `eta/eta_ref` 的范围与对比度、 `eta_interior` （在 `0.25 <= z <= 0.75` 的内部条带内对 `eta/eta_ref` 取算术体平均，两侧壁层都排除在外）、 `Ra_interior = Ra/eta_interior` ；若参数卡提供了尺度组，还会报告以 Pa s 为单位的 `eta_0` 。 `pic_mode = :newton` 直接用原始定律构造黏度，因此会拒绝 `eta_ref != 1` ，而不是悄悄求解另一个问题。

生产参数卡的默认值是 `eta_ref = 1` ，但它的材料定律在盒内跨越多个数量级，所以内部 Rayleigh 数远高于名义值。报告只在 `eta_ref != 1` 时打印；用同一套约定算出的该卡成熟状态数值是 `eta_interior = 4.280635e-3` 、 `Ra_interior = 2.336102e7` 和 `eta_0 = 2.368635e23 Pa s` （ `Ra` 定义所对应的物理黏度），由此得到物理内部黏度 `1.013926e21 Pa s` 。在标定所用的热内部点（ `T = 1.75` ， `edot_ref` ）上，同一套约定给出 `Ra = 7.84e7` 与 `3.02e20 Pa s` 的黏度。所以 `Ra = 1e5` 这个标签描述的是标定点，而不是内部：引用时要和 `Ra_interior` 一起给出。

## 2. 边界条件参考

本节是边界条件的唯一参考：壁型、四组数据、牵引的符号约定、派生压力基准、角点归属、每一种被拒绝的配置，以及重启守卫记录了什么。参数名与默认值见 6.1 节，本节不再重复它们；把几种壁型组合成区域入流/出流问题的配方见[算例](examples.md)的区域通道一节。

### 2.1 四种壁型

`bc` 是默认壁型，取 `:free` 或 `:rigid`，默认 `:free`。 `wall_bc` 按顺序 `(xminus, xplus, yminus, yplus)` 逐壁覆盖，每一面取 `:same`（跟随 `bc`）、`:free`、`:rigid` 或 `:open`。库接口对应 `Params.wall_bc`，参数卡对应 `CASE_WALL_BC`。

| 壁型 | 法向分量 | 切向分量 |
|---|---|---|
| `:free` | Dirichlet，默认零 | 自由，自然条件为零切向牵引 |
| `:rigid` | Dirichlet，默认零 | Dirichlet，默认零，可由 `wall_u` 给定 |
| `:open` | 自由 | 自由 |
| `:same` | 跟随 `bc` | 跟随 `bc` |

`:free` 是自由滑移， `:rigid` 是无滑移， `:open` 两个分量都没有 Dirichlet 条件：适用自然（零）牵引，数据由 `wall_t` 与 `wall_tn` 给出。逐壁条件只作用于计算域的外边界，内部叶面不受影响。

### 2.2 四组数据

四组数据都按 `(xminus, xplus, yminus, yplus)` 逐壁给出。库接口接受位置的函数 `(x, y) -> Float64`，参数卡只接受每面壁一个常量（空项表示 `nothing`）。取值为 `nothing` 时：速度数据沿用全局闭合 `uDx`/`uDz`（驱动把这两个闭合设为零），牵引数据表示零牵引。

| 数据 | 物理含义 | 允许的壁 | 约定 |
|---|---|---|---|
| `wall_u` | 切向速度 | 仅 `:rigid` | 全局分量，沿正坐标方向：竖直壁上为 $u\_y$ ，水平壁上为 $u\_x$ 。 |
| `wall_un` | 法向速度 | `:free`、`:rigid` | 全局分量，沿正坐标方向：竖直壁上为 $u\_x$ ，水平壁上为 $u\_y$ ；不是外法向通量 $u\cdot n$ 。 |
| `wall_t` | 总 Cauchy 牵引的切向分量 | `:free`、`:open` | 全局分量，沿正坐标方向：竖直壁上为 $t\_z$ ，水平壁上为 $t\_x$ 。 |
| `wall_tn` | 总 Cauchy 牵引的法向分量 | 仅 `:open` | 全局分量，沿正坐标方向：竖直壁上为 $t\_x$ ，水平壁上为 $t\_z$ 。 |

参数卡暴露四组数据中的三组： `CASE_WALL_UN`、`CASE_WALL_T`、`CASE_WALL_TN`； `wall_u` 只能从库接口给出，参数卡没有对应的环境变量。四组数据都用求解器的无量纲单位：速度与 `wall_un` 同一个速度单位，牵引与压力、应力同一个单位。

### 2.3 牵引是总 Cauchy 应力

对动量方程分部积分，边界项以正号落在右端：

$$
\int_\Omega 2\eta\,\boldsymbol\varepsilon(\mathbf u):\boldsymbol\varepsilon(\mathbf v)\,\mathrm dV \;- \int_\Omega p\,\nabla\cdot\mathbf v\,\mathrm dV
= \int_\Omega \mathbf f\cdot\mathbf v\,\mathrm dV + \int_{\partial\Omega} \mathbf t\cdot\mathbf v\,\mathrm dS,
\qquad
\mathbf t = \boldsymbol\sigma\cdot\mathbf n = -p\,\mathbf n + 2\eta\,\boldsymbol\varepsilon(\mathbf u)\cdot\mathbf n,
$$

其中 $\boldsymbol\sigma$ 是总 Cauchy 应力， $\mathbf n$ 是外法向。 `wall_t` 与 `wall_tn` 就是上式中的 $\mathbf t$ 的两个分量，作为边界载荷加到**未约化**的载荷向量上，再由约束矩阵投影到容许检验空间。它们是自由侧的数据项，不是 Dirichlet 值。

在轴对齐壁上，牵引的切向分量不含压力： $x$ 壁上 $t\_z=\sigma\_{zy}$ ， $y$ 壁上 $t\_x=\sigma\_{xy}$ ，所以 `wall_t` 就是 $2\eta\,\boldsymbol\varepsilon(\mathbf u)\cdot\mathbf n$ 的切向分量，且在常黏度下与 $\eta$ 无关。法向分量携带压力：

$$
t_n = -p + 2\eta\,\varepsilon_{nn},
$$

所以 `wall_tn` 是确定压力水平的数据项。因为 $\mathbf t$ 使用外法向，给定压力 $p\_0$ 时：

| 壁 | 外法向 $\mathbf n$ | `wall_tn` |
|---|---|---|
| 顶壁 $y=1$ | $+\mathbf e\_y$ | $-p\_0$ |
| 底壁 $y=0$ | $-\mathbf e\_y$ | $+p\_0$ |
| `xminus` $x=0$ | $-\mathbf e\_x$ | $+p\_0$ |
| `xplus` $x=L\_x$ | $+\mathbf e\_x$ | $-p\_0$ |

等价的写法是 `wall_tn` 取 $-p\_0$ 乘以外法向在该坐标方向上的分量。严格地说 `wall_tn` 固定的是总法向牵引；压力恰为 $p\_0$ 还需要 $2\eta\,\varepsilon\_{nn}=0$ 。 `wall_tn` 给的是求解器无量纲压力单位下的**动态**压力牵引，不是岩石静压：求解器的 $p$ 不含基态，浮力里已经把基态解析地减掉，给一个静水压等于把整根水柱当成牵引加载。

### 2.4 压力基准

节点压力钉值只在**没有** `:open` 壁时施加。四壁都保留 Dirichlet 法向分量时，常数压力位于约化梯度的零空间内，被丢掉的那一行连续性方程是其余行的线性组合，钉一个压力自由度因此是精确的。

有 `:open` 壁时，常数压力不在这个零空间里：存在法向迹非零的边界检验函数。此时钉一个压力自由度等于把一条连续性方程换成 $p\_a=0$ ，那是另一个问题。所以求解器不钉值，压力水平由 `wall_tn` 数据决定；求解器把决定记录在 `LAST_SEMSTOKES_PRESSURE_GAUGE` 中（`:pin` 或 `:none`）。不钉值的求解在 `linear=:direct`（含平移阶梯）、`:mg` 与 `:auto` 上验证过；`:schur_cg` 与 `:lagged` 依赖闭边界的规范，在开壁配置上被拒绝（见 2.6）。

### 2.5 角点归属

四个方向按 `(xminus, xplus, yminus, yplus)` 的顺序写入，后写的壁覆盖先写的壁，所以角点分量有确定的归属：

| 角点分量 | 归属 |
|---|---|
| $u\_x$ | `x` 壁不是 `:open` 时取它的法向值；若该角点的 `y` 壁是 `:rigid` ，`y` 壁的切向 `wall_u` 值后写并覆盖它。 |
| $u\_y$ | 该角点的 `y` 壁不是 `:open` 时，归 `y` 壁的法向值；`y` 壁是 `:open` 时不写，`x` 壁（若为 `:rigid`）的切向值保留。 |

`:open` 壁不写任何分量。后果之一是：刚性顶壁或底壁占有其两个角点的 $u\_x$ ，侧壁的入流数据在角点被它覆盖。

### 2.6 被拒绝的配置

下列配置在构造 `Params` 时即被拒绝；直接调用 `solve_semquad_stokes_var` 时求解器重复同样的检查，因此库调用也绕不过去。

| 配置 | 拒绝原因 |
|---|---|
| 全局 `bc = :open` | 默认壁型只接受 `:free` 或 `:rigid`； `:open` 只能逐壁给出。 |
| 四面全 `:open` | 纯牵引问题：速度只确定到刚体运动，数据还必须满足净力与净力矩相容；离散鞍点奇异，压力不被边界确定。 |
| 两面 `x` 壁全 `:open` | 留下 $x$ 平移。求解器用 Dirichlet 自由度上 3 × 3 刚体模态 Gram 矩阵的秩测试确认，报错信息给出实测特征值与可容许运动。 |
| 两面 `y` 壁全 `:open` | 留下 $y$ 平移，判据同上。 |
| `wall_u` 给在解析为 `:free` 或 `:open` 的壁 | 该壁切向分量自由（ `:free` ）或根本没有 Dirichlet 分量（ `:open` ），值会被静默忽略。 |
| `wall_un` 给在 `:open` 壁 | 法向速度在那里是未知量，值会被静默忽略。 |
| `wall_t` 给在 `:rigid` 壁 | 切向分量是 Dirichlet 条件，载荷会被约束矩阵丢掉。 |
| `wall_tn` 给在 `:free` 或 `:rigid` 壁 | 法向分量是 Dirichlet 条件，载荷会被丢掉。 |
| 无 `:open` 壁时，指定的法向速度净外通量不为零 | 四壁法向都是 Dirichlet 时，离散连续性方程只在 $\int\_{\partial\Omega}\mathbf u\cdot\mathbf n\,\mathrm dS=0$ 时相容。阈值是 $10^{-12}\max(u\_{\max}\cdot P,1)$ ，其中 $u\_{\max}$ 是给定的法向值的最大绝对值， $P$ 是边界周长。绝对下限是必需的：制造解测试把精确场作为壁面数据，其数学净通量为零，但浮点求值不为零（例如壁上的 $\sin(\pi)=1.2\times10^{-16}$ ）。 |
| 有 `:open` 壁时使用 `:schur_cg` 或 `:lagged` | 这两条路径假设闭边界的压力规范；开壁配置没有节点钉值。 |
| `pic_mode = :newton` 且任一逐壁字段非默认 | 每步 Newton 路径只用 `bc` 解 Stokes，逐壁数据会被忽略，开壁会被当成闭壁。 |
| 未知的壁型字符串 | 只接受 `:same`、`:free`、`:rigid`、`:open`。 |

有 `:open` 壁时净通量检查被跳过：开壁的自由法向速度按构造吸收不平衡量。

### 2.7 重启标记

边界条件属于 `run_config` ： `wall_bc` 以逗号连接的字符串记录； `wall_u`、`wall_un`、`wall_t`、`wall_tn` 每面壁记一个 `none`/`custom` 标记。守卫拒绝标记改变的续算。函数没有可比较的取值，所以把一个函数换成另一个函数不会被检出：这类改动应从头重算。旧检查点缺少某个键时跳过该键的比较，规则见第 8 节。

## 3. 离散

### 3.1 网格与空间

网格是四叉树。每个叶用 $(\ell,i\_x,i\_y)$ 标记，树始终保持 2:1 平衡，即相邻叶最多相差一级。加密把一个四兄弟拆成四个，粗化把四个合回一个。参数卡把深度限制在 `max_level = 7` ，所以最细的叶高是盒高的 $1/(2\cdot2^{7})=1/256$ 。初始树由 5 × 2 个基础单元各均匀细分三次得到，即均匀的 level 3、共 640 个叶。

空间离散是 Taylor–Hood 元：速度取 $Q\_p$ ，压力取 $Q\_{p-1}$ ，本卡取 $p=2$ ，即 $Q\_2$ 速度、 $Q\_1$ 压力。节点是参考正方形上的 Gauss–Legendre–Lobatto 点，每个单元上的基函数是这些节点上的 Lagrange 基：速度每方向 $p+1$ 个节点，压力每方向 $p$ 个。单元矩阵用 Gauss–Legendre 求积，每方向取 `p+2` 个点，即 `make_fedata` 的默认 `nq = p + 2` 。这是过积分：它精确积分 $Q\_p$ 的质量矩阵与刚度矩阵。代码不做质量集中，质量矩阵按完整求积组装成一致矩阵而不是对角阵，所以这里没有用 GLL 求积换对角质量矩阵的余地。

### 3.2 悬挂节点与约束

粗叶面上存在细节点，即悬挂节点。它的取值由粗面 GLL 插值给出的迹决定，写成稀疏约束矩阵 $C$ ，自由行上是单位阵。系统按

$$
C^{\mathsf T} A\,C
$$

组装并求解，其中 $A$ 是未加约束的组装矩阵， $C$ 是约束矩阵， $C\^{\mathsf T}$ 是其转置。这样对称性得以保留，对线性约束而言是精确的。边界值走同一个仿射映射；压力基准按 2.4 节的规则处理：没有 `:open` 壁时钉住一个压力自由度，有 `:open` 壁时不钉值。

### 3.3 时间积分

流场是准静态的，每步解一次 Stokes。能量方程用 BDF2，第一步退化为后向 Euler。记变步长比为

$$
w = \frac{\Delta t}{\Delta t_{\mathrm{prev}}},
$$

则一步的更新写成

$$
\frac{a}{\Delta t}\theta^{n+1} + \mathbf u\cdot\nabla\theta^{n+1} - \nabla^2\theta^{n+1}
= \frac{b}{\Delta t}\theta^{n} - \frac{c}{\Delta t}\theta^{n-1},
$$

$$
a = \frac{2w+1}{1+w}, \qquad b = 1+w, \qquad c = \frac{w^2}{1+w}.
$$

| 符号 | 含义 |
|---|---|
| $\theta^n$ | 步初的温度异常。 |
| $\theta^{n-1}$ | 再往前一步的温度异常。 |
| $\Delta t$ | 当前步长； $\Delta t\_{\mathrm{prev}}$ 是上一步步长。 |
| $a, b, c$ | BDF2 的质量系数；第一步取 $a=b=1/\Delta t$ 、 $c=0$ 。 |

参数卡取 `maxpic = 1` ，即黏度滞后一步：一步之内唯一的流动求解使用步初温度，因此该步在黏度耦合上只有一阶精度，但结果完全确定。

步长按

$$
\Delta t = \mathrm{CFL}\,\frac{h_{\min}}{|\mathbf u|_{\max}}
$$

更新，其中 $\mathrm{CFL}=0.5$ ， $h\_{\min}$ 是最小叶高， $|\mathbf u|\_{\max}$ 是最大速度模。每步最多放大 5%、最多缩小 5%，除非触发硬 CFL 截断；步长被限制在 $[\Delta t\_{\min},\Delta t\_{\max}]=[10^{-9},5\times10^{-5}]$ 内，第一步从 $\Delta t\_0=10^{-6}$ 出发并立即切到 CFL 允许值。

### 3.4 熵黏性

振荡通过逐单元熵黏性压制，黏性项加到能量算子上，每个叶一个值：

$$
\nu_e = \frac{c_{ev}\,h_e^2 R}{\sigma},
\qquad
\nu_{\max} = c_{\max}\,|\mathbf u|\,h_e,
\qquad
\nu = \min(\nu_{\max},\nu_e).
$$

| 符号 | 含义 |
|---|---|
| $\nu$ | 该叶实际加入的熵黏性。 |
| $R$ | 单元内 $\theta$ 方程残差的最大绝对值（ `ev_alpha = 2` 时再乘以 $\lvert T-\langle T\rangle\rvert$ ）。 |
| $h\_e=\sqrt2\,h$ | 熵黏性的长度尺度， $h$ 是叶尺寸。 |
| $\sigma$ | 熵残差的全局散布：对网格上的 $(T-\langle T\rangle)^2$ 取 $\max(e\_{\max}-\bar e,\ \bar e-e\_{\min})$ ，其中 $\langle T\rangle=(T\_{\min}+T\_{\max})/2$ 是温度的中点， $\bar e$ 是 $(T-\langle T\rangle)^2$ 的面积平均。 |
| $\lvert\mathbf u\rvert$ | 单元内最大速度模。 |
| $c\_{ev}$ | 参数卡的 `ev`，取 0.11。 |
| $c\_{\max}$ | 参数卡的 `ev_cmax`，取 0.052。 |

参数卡取 `ev_alpha = 2` ，所以用 $h\_e^2$ 形式。壁面行上的黏性为零。每个时间步都重算；最初两步以及任何变差为零时，代码直接取 $\nu=\nu\_{\max}$ 。熵黏性场写入检查点，因为没有它就无法重建上一步的能量算子。

## 4. 自适应策略

### 4.1 指示子

加密指示子默认取温度异常梯度在内部面上的跳变：

$$
I_K = \frac h2\sum_{F\subset\partial K}\int_F [\![\nabla\theta\cdot\mathbf n]\!]^2\,\mathrm ds,
$$

左右两侧壁绝热，对指示子没有任何贡献；顶壁和底壁再额外加一个单侧法向导数项，正是这一项让热边界层得到分辨。其中 $I\_K$ 是叶 $K$ 的指示子， $h$ 是叶高， $\theta$ 是温度异常， $\mathbf n$ 是面法向， $[\![\cdot]\!]$ 表示跨面跳变。

加密指示子作用在哪个场，由 `amr_indicator` 选择，参数卡用 `CASE_INDICATOR` 设置。默认的 `:theta` 就是上面这个温度异常场； `:velocity` 把同样的估计分别作用在两个速度分量上再相加； `:both` 把两个指示子各自按自己的总量归一化后相加：

$$
j_{\mathrm{both}} = j_\theta + \frac{\sum_K j_{\theta,K}}{\sum_K j_{v,K}}\,j_{v,K},
$$

其中 $j\_\theta$ 与 $j\_v$ 分别是两个场的逐叶指示子。这样两个场都不会主导加密；静止时速度为零， `:both` 精确退化为 `:theta` 。

下面的比较在一个成熟的二维状态上以相同成本测得：10 000 叶、43 步、同样的固定 `dt`，参照解是 40 000 叶的 `:theta` 解。 `Nu_bot` 的相对误差， `:theta` 为 1.10e-4， `:velocity` 为 1.01e-5；体积平均温度分别 1.75e-4 与 1.15e-4。精确壁面通量残差 `resid_A/Nu_top` 则是 `:theta` 更小，约 7e-4% 对 5e-3%。机制在于细叶落在哪里： `:theta` 把 85% 的细叶放在壁面条带， `:velocity` 只放 3.6%，转而加密内部。因此选择取决于观测量，而不是给两种指示子排名。这个比较只覆盖一个状态、一个 Rayleigh 数、43 步（时间轴的 0.2%）和二维，所以没有理由时保持默认，并把上面的数字看作该状态的属性，而不是其他流动的保证。

### 4.2 标记、粗化与叶数上限

标记采用 Dörfler 策略： `refine_frac = 0.2` 指需要分辨的指示子总量占比，因此加密多少单元由指示子场自己决定。

平衡循环包含以下几部分：

| 机制 | 参数 | 作用 |
|---|---|---|
| 粗化 | `coarsen_frac * total`、`round(coarsen_frac * n)` | 在完整四兄弟上，把相对阈值以下的部分粗化掉；每个周期丢掉的叶数上限是 `round(coarsen_frac * n)` ，`n` 是周期开始时的叶数。 |
| 配额交换 | `max_leaves` | 合并误差最小的合格四兄弟，腾出叶数给要加密的地方。 |
| 2:1 校验 | `amr_coarsen_gate = true` | 拒绝会导致一步之后又得撤销的合并。 |
| 场传递 | `transfer_mode = :nodal` | 网格变化时用节点插值搬运场变量。 |

叶数上限是 `max_leaves = 10000` 。本卡取 `amr_passes = 1` ，也就是每个 AMR 周期只跑一轮指示子、加密与场传递。

### 4.3 迟滞与壁面规则

平衡循环还有三条固定规则：

| 机制 | 参数 | 作用 |
|---|---|---|
| 迟滞 | `amr_hysteresis = 1` | 刚合并过的四兄弟不马上再拆开。 |
| 死区 | `0.5` | 平衡循环的粗化阈值取 `tau_h = min(tau_c, 0.5 tau_r)` ，这个倍数固定为 0.5。 |
| 壁面规则 | `wall_strip = 0.1`、`wall_min_level = 5` | 顶底壁条带内以及壁面行至少保持 level 5，之后再执行叶数上限。 |

壁面规则存在的原因：单靠指示子时，顶壁停留在 level 3 或 4，而内部已经到 level 5 或 6；这时逐点 `Nu_top` 与壁面加密后的值相差 1.2%，一致边界通量相差 9.6%。

## 5. 线性求解器与 `:auto` 规则

### 5.1 直接路径

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
| $B\^{\mathsf T}$ | 速度为行、压力为列 | 梯度 |
| $B\_s$ | 压力为行、速度为列 | 散度 |
| $0$ | 压力为行、压力为列 | 零块，鞍点结构就来自它 |

其中 $K\_{\mathrm{red}}$ 是约化速度矩阵， $B$ 是梯度， $B\_s$ 是散度， $p$ 是压力。分解用 UMFPACK，并对真正的约化残差做检查。壁面加密后的自由滑移四叉树会让无平移分解失稳，因此这条路径采用平移阶梯：相对对角平移依次取 $10^{-12}$ 、 $10^{-10}$ 、 $10^{-8}$ ；若没有一级满足接受容差，就以最好的那级为预条件，对无平移算子做若干步 GMRES 迭代精化。另有两级更小的平移 $10^{-14}$ 与 $10^{-13}$ ：当无平移分解已经完全丢掉解时先试这两级，阶梯与精化都失败、该步即将中止前再作为最后手段试一次。 `stokes_scaling = :equilibrated` 会先尝试对称平衡 $S\to DSD$ ，其中 $S$ 是鞍点矩阵、 $D$ 是对角缩放矩阵，作用是把接近零的压力模态抬到舍入误差之上；它只在本来会进入阶梯的那些求解上生效。直接路径是唯一逐位可复现的路径。

### 5.2 多重网格路径

多重网格路径用 flexible GMRES 配块三角预条件：速度块用几何 V 循环，每级至少按 1.5 倍 h 粗化，采用阻尼 Jacobi 光滑，各层算子由叶局部无矩阵 apply 构建；压力 Schur 补用三因子 w-BFBT 近似。FGMRES 每 100 次重启，总上限 500 次，按真正的约化鞍点残差对 `mg_tol` 判断收敛。参数卡传 `mg_tol = 1e-8` ，比它的 `stokes_tol = 1e-3` 低五个数量级；库默认值是 `1e-10` 。

### 5.3 `:auto` 规则

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

## 6. 参数参考

### 6.1 参数卡的环境变量

下表的可调项通过脚本启动时读取的 `CASE_*` 环境变量设置；第二列是参数卡给出的默认值。逐壁边界条件的四个变量只列名字与默认值：取值、数据约定与拒绝规则由第 2 节拥有。

| 变量 | 默认值 | 允许值 | 作用 |
|---|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | 任意路径 | 输出根目录，运行产生的全部文件都在它下面。 |
| `CASE_STEPS` | 0 | 整数 | 步数上限，与绝对步号比较。取 0 表示用模式默认值：smoke 10，short 150，spinup 10000，freeze 20000，resume 与 wallrefine 2000。 |
| `CASE_FRAME` | 0 | 整数 | 帧间隔，单位是步。取 0 表示用参数卡的 5。 |
| `CASE_LX` | 2.5 | 浮点 | 盒宽。必须保持 `CASE_LX/CASE_CX == 1/CASE_CY` ；`Quadtree` 构造时校验，不一致直接报错。 |
| `CASE_CX`、`CASE_CY` | 5、2 | 整数 | x 与 y 方向的基础单元数。 |
| `CASE_MAXLEV` | 7 | 整数 | 四叉树最大层级，最细叶高为盒高的 $1/(2\cdot2^{\ell})$ 。 |
| `CASE_MAXLEAVES` | 10000 | 整数 | 叶数硬上限，每个 AMR 周期之后执行。 |
| `CASE_AMR_GATE` | 1 | 0、1 | 平衡循环中的 2:1 粗化校验，取 0 关闭。 |
| `CASE_HYST` | 1 | 整数 | 迟滞冷却，以 AMR 周期计；取 0 关闭。 |
| `CASE_COARSEN` | 0.05 | 浮点 | 粗化配额：平衡循环的相对阈值与每周期单元数上限。 |
| `CASE_WALLMIN` | 5 | 整数 | 壁面条带内以及壁面行的最小层级，取 0 关闭壁面规则。 |
| `CASE_WALLSTRIP` | 0.1 | $(0,0.5)$ 内的浮点 | 壁面条带宽度，从每个壁面向内占盒高的比例。 |
| `CASE_WALL_BC` | 空（全部 `same`） | 见 2.1 节 | 逐壁速度条件。 |
| `CASE_WALL_UN` | 空（全部 `nothing`） | 见 2.2 节 | 逐壁法向速度，常量四元组。 |
| `CASE_WALL_T` | 空（全部 `nothing`） | 见 2.2 节 | 逐壁切向牵引，常量四元组。 |
| `CASE_WALL_TN` | 空（全部 `nothing`） | 见 2.2 节 | 逐壁法向牵引，常量四元组。 |
| `CASE_SYMMETRY` | `none` | `none`、`x` | AMR 各轮的 x 镜像闭合：取 `x` 时镜像伙伴一起加密、一起合并，用于对称基准。 |
| `CASE_INDICATOR` | `theta` | `theta`、`velocity`、`both` | 加密指示子作用的场，见 4.1 节。 |
| `CASE_LINEAR` | `auto` | `auto`、`direct`、`mg`、`schur_cg`、`lagged` | 线性求解器，见第 5 节。 |
| `CASE_EV` | 0.11 | 浮点 | 熵黏性系数。 |
| `CASE_LATENT` | `per_phase` | `per_phase`、`global` | 潜热系数：逐个相变计算，或两个相变共用历史值 `latent_L = 0.11` 。传入其他字符串会在构造 `Params` 时被拒绝。 |
| `CASE_COMP_ED`、`CASE_COMP_ES` | 12.0、17.0 | 浮点 | 复合定律扩散分支与位错分支的活化能 `E_d`、`E_s`（见 1.4 节）。 |
| `CASE_COMP_AD`、`CASE_COMP_AS` | 1.6240183119e5、2.6796419238e-1 | 浮点 | 两种机制的前因子，单位是 $1/\eta\_0$ 。在另一个 `Ra` 下运行时按 `Ra / Ra_new` 缩放，以保持有量纲的材料常数不变。 |
| `CASE_DT0` | 1.0e-6 | 浮点 | 初始步长，第一步会立即切到 CFL 值。 |
| `CASE_DTMAX`、`CASE_DTMIN` | 5.0e-5、1.0e-9 | 浮点 | 自适应步长的上下界。 |
| `CASE_RA` | 1.0e5 | 浮点 | 参数卡的 Rayleigh 数。 |
| `CASE_ETA_MIN`、`CASE_ETA_MAX` | 1.0e-4、1.0e4 | 浮点 | 黏度的绝对上下界，以定律的单位给出，并在 `eta_ref` 归一化之前施加（见 1.9 节）。 |
| `CASE_STOKES_SCALING` | `none` | `none`、`equilibrated` | 直接鞍点的平衡开关，只对本来会失败的求解生效，健康的求解不受影响。 |
| `CASE_ETA_REF` | 1.0 | 大于 0 的浮点 | 以材料定律单位给出的 Rayleigh 数参考黏度： `Ra` 用它定义，流场求解使用 `eta/eta_ref` 。头部会打印它，检查点也会记录（见 1.9 节）。 |
| `CASE_SHEAR_MODULUS` | 空（关闭） | 大于等于 0 的浮点 | Maxwell 黏弹性：无量纲弹性剪切模量 $G$ 。留空或取 0 表示关闭，运行结果逐位不变；取正值会改变有效黏度与物理区域，见 1.8 节。 |
| `CASE_EXTDIAG` | 0 | 0、1 | 把精确能量恒等式残差 `resid_A` 追加到 `history.csv` 。它会改变列数，所以只接受全新的输出目录。 |

`main()` 在做任何事之前就会拒绝无法识别的 `CASE_AMR_GATE`、`CASE_SYMMETRY` 或 `CASE_STOKES_SCALING` ，并校验 `CASE_WALL_BC`、`CASE_WALL_UN`、`CASE_WALL_T` 与 `CASE_WALL_TN` 各自的四元组以及每个数值是否有限；逐壁组合是否适定在构造 `Params` 时检查（见 2.6 节）。下面这些是参数卡传给驱动的取值：

| 取值 | 参数卡里的值 |
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
| `maxpic` | 1。`pic_mode = :newton` 是研究选项：需要 `linear=:direct`，拒绝参考态、可压缩质量形式与 `nu_cbf`，在实测算例上比 Picard + 多重网格更慢。 |
| `pic_tol` | `1e-6` |
| `ev_cmax` | 0.052 |
| `ev_alpha` | 2 |
| `checkpoint_every` | 200 |
| 帧网格 | 256 × 256 |

共有七种运行模式：

| 模式 | 作用 |
|---|---|
| `smoke` | 跑 10 步冒烟。 |
| `short` | 跑 150 步做稳定性检查。 |
| `spinup` | 打开 AMR 的生产预积分。 |
| `freeze <ckpt>` | 冻结网格续算，把 `amr_every` 设为 0。 |
| `wallrefine <ckpt>` | 换一套壁面参数后续算。 |
| `resume <ckpt>` | 带 AMR 续算。 |
| `symseed <ckpt>` | 换一个 `CASE_SYMMETRY` 后续算。 |

种子模式只复制检查点并改写下一阶段要变的那条配置字符串，其余各项照旧参与重启守卫的比较，规则见第 8 节。

### 6.2 run_semquad_simulation 的关键字

「默认值」一列是库的默认，「参数卡」一列是 `case_realistic_2d.jl` 传入的值。逐壁边界数据不是这里的关键字：它们属于 `Params` 的字段，见第 2 节。

| 关键字 | 默认值 | 参数卡 | 作用 |
|---|---|---|---|
| `p` | 2 | 2 | 多项式阶：速度 $Q\_p$ ，压力 $Q\_{p-1}$ 。 |
| `maxsteps` | 10000 | `CASE_STEPS` | 停止的绝对步号。 |
| `tstop` | 1e-3 | `Inf` | 达到该无量纲时间即停止。 |
| `dt0`、`dtmax`、`dtmin`、`dt_growth` | 1e-4、`Inf`、1e-14、1.05 | 1e-6、5e-5、1e-9、1.05 | 步长及其上下界。 |
| `cfl` | 0 | 0.5 | 步长更新所用 CFL 数；取 0 表示把 `dt` 冻结在 `dt0` 。 |
| `amr_every` | 0 | 5 | 两次 AMR 之间的步数；取 0 冻结网格。 |
| `amr_indicator` | `:theta` | `CASE_INDICATOR` | 加密指示子作用的场：`:theta`、`:velocity` 或 `:both`（见 4.1 节）。 |
| `amr_passes` | 1 | 1 | 每个 AMR 周期里指示子、加密与场传递的轮数。 |
| `amr_hysteresis`、`amr_coarsen_gate` | 1、true | `CASE_HYST`、`CASE_AMR_GATE == 1` | 平衡循环的迟滞长度与 2:1 校验。 |
| `refine_frac`、`coarsen_frac` | 0.2、0.0 | 0.2、`CASE_COARSEN` | 平衡循环的误差份额，分别从上方和下方做 Dörfler 标记。 |
| `max_level`、`max_leaves` | 无上限 | 7、10000 | 深度上限与叶数硬上限。 |
| `wall_min_level`、`wall_strip` | 0、0.1 | 5、0.1 | 壁面规则，取 0 关闭。 |
| `symmetry`、`transfer_mode` | `:none`、`:nodal` | `CASE_SYMMETRY`、`:nodal` | 镜像闭合与 AMR 场传递方式。 |
| `hotspot_weight`、`hotspot_frac`、`hotspot_min_level`、`hotspot_mode`、`hotspot_zmin` | 0、0.5、0、`:theta`、0 | 关闭 | 热点加密规则。 |
| `top_wall_level`、`top_wall_frac` | 0、0.5 | 关闭 | 顶壁加密规则。 |
| `zero_side_eta`、`side_strip` | false、0.05 | false、0.05 | 侧墙条带的黏度清零。 |
| `advection` | `:galerkin` | `:skew` | 能量方程的对流离散形式。 |
| `ev`、`ev_cmax`、`ev_alpha` | 0、0.052、2 | 0.11、0.052、2 | 熵黏性，见 3.4 节；`ev = 0` 关闭。 |
| `maxpic`、`pic_tol`、`pic_relax`、`pic_mode` | 1、1e-6、1.0、`:picard` | 相同 | 步内非线性迭代；用 Newton 时要求 `linear=:direct` 。 |
| `pic_reject`、`pic_reject_clamp`、`pic_reject_res`、`pic_reject_max` | false、0.25、10、4 | 默认值 | 当贴限比例或残差变大时拒绝一次 Picard 试步，并用减半的松弛重做。 |
| `pic_adapt`、`pic_relax_min`、`pic_eta_relax`、`pic_u_relax` | false、0.05、1.0、1.0 | 默认值 | Picard 循环的松弛自适应、应变率平均与速度欠松弛。 |
| `pic_div_window`、`pic_div_factor` | 0、10 | 默认值 | Picard 循环的散度检测器。 |
| `newton_maxit`、`newton_krylov`、`newton_restart`、`newton_ls`、`newton_geo_safety` | 15、80、30、8、0 | 默认值 | Newton 控制项，`pic_mode = :picard` 时都不起作用。 |
| `linear` | `:auto` | `CASE_LINEAR` | 线性求解器，见第 5 节。 |
| `stokes_tol` | `Inf` | 1e-3 | 真正残差超过它就中止该步；同时也是 `:auto` 回退和直接路径的接受容差。 |
| `schur_maxiter`、`schur_precond` | 200、`:mass` | 默认值 | `linear = :schur_cg` 的迭代上限与预条件。 |
| `mg_tol`、`mg_threads` | 1e-10、1 | 1e-8、1（`auto`：`min(8, nthreads)`） | 多重网格的外层容差与线程数。 |
| `mg_operator`、`mg_pressure_eta_tol` | `:matrixfree`、1.0 | 相同 | 多重网格的几何形式，以及压力块缓存的重建阈值。 |
| `mg_vel_solver`、`mg_smoothing`、`mg_omega` | `:vcycle`、`:auto`、`nothing` | 默认值 | 多重网格的速度求解器与 V 循环调度。 |
| `stokes_scaling` | `:none` | `CASE_STOKES_SCALING` | `:none` 或 `:equilibrated` ，只对直接路径有效。 |
| `corotation_guard` | true | 默认 | 报告共转项的适用范围，越界时警告一次； `shear_modulus = 0` 时既不采样也不打印（见 1.8 节）。 |
| `regime_report` | true | 默认 | 在第一个推进应力的步与运行结束时各报告一次 Maxwell 区域； `shear_modulus = 0` 时既不计算也不打印（见 1.8 节）。 |
| `eta_ref_scale` | `nothing` | 参数卡的 `(rho, g, alpha, dT, d, kappa)` 尺度组 | `eta_ref` 报告所用的尺度组；给出后报告还会打印以 Pa s 为单位的 `eta_0`（见 1.9 节）。 |
| `mass_form` | `:anelastic` | `:boussinesq` | 质量守恒形式。 |
| `phases`、`latent_scheme` | `[]`、`:implicit` | 两个相变、`:implicit` | 相变与潜热耦合；`maxpic = 1` 时两种格式结果相同。 |
| `weak_chi0` | 0 | 0 | 初始应变弱化场；只有参数卡设置了弱化参数时才起作用。 |
| `nu_cbf` | false | false | 把一致边界通量的 Nusselt 对追加到 history 里（见 7.2 节）。 |
| `restart_from` | `""` | 取决于模式 | 从哪个检查点重启。 |
| `checkpoint_path`、`checkpoint_every` | `""`、0 | `$CASE_OUT/spinup.h5`、200 | 检查点的写入位置与间隔。 |
| `checkpoint_meta`、`callback`、`verbose`、`profile_steps` | 空、`nothing`、false、false | 参数卡的元数据、帧写入器、true、false | 检查点元数据、步钩子、逐步输出、逐步计时。 |
| `pic_diag`、`extdiag` | false、false | false、`CASE_EXTDIAG` | 非线性诊断，以及 `resid_A` 历史列。 |
| `initial_refine`、`theta_ic` | 0、`nothing` | 0、`nothing` | 第一步之前在初值上做的 AMR 轮数；`theta_ic` 在每次加密后的树上重新采样解析初值。 |
| `tavg_start`、`steady_tol`、`cfl_tol`、`tmin`、`steady_window` | `Inf`、`Inf`、`Inf`、0、5 | `Inf`、`Inf`、`Inf`、0、5 | `tavg_start` 是时间平均窗口； `steady_tol`、`steady_window` 与 `tmin` 是稳态停止判据，在 `t > tmin` 之后，相对温度与相对 `Nu` 的变化在 `steady_window` 步内都低于 `steady_tol` 即停止； `cfl_tol` 是壁面 CFL 中止保护。 |

函数返回 `theta`、`t`、`steps`、`history`、`history_extra`、`qt`、`ss`、`ux`、`uz`、`pv`、`avg`、`weak`、`chi`、`chi_prev` ，以及 Maxwell 应力历史的 `txx`、`tyy`、`txy`（`shear_modulus = 0` 时为 `nothing`）。 `history` 每实际跑一步记一行，列的含义见 7.2 节； `history_extra` 存放固定 18 列之后追加的列。

### 6.3 参数卡

`case_realistic_2d.jl` 里的 `make_params()` 设置下列字段； `Params` 的其余字段保持 `src/physics/params.jl` 中列出的默认值。表中既有显式关闭的开关，也有为交代取值而列出的关键字段；物理背景见 1.1 至 1.9 节，逐壁字段的语义见第 2 节。

| 字段 | 取值 | 含义 |
|---|---|---|
| `comp_Tcut` | 0（关闭） | 复合定律里绝对温度的下限取 0，即关闭此项。 |
| `yield_c`、`yield_taumax` | `Inf` | 关闭屈服。 |
| `yield_phi` | 0 | 关闭摩擦角。 |
| `yield_c_grad`、`yield_mu_grad` | 0、0 | 关闭屈服面的深度依赖：内聚力与摩擦系数随深度的梯度都取 0。 |
| `weak_f` | 1 | 残余弱化因子取 1，即不弱化。 |
| `weak_eps_ref` | 0 | 关闭应变弱化。 |
| `rho_heating`、`rho_cp` | `false` | 关闭密度加权加热。 |
| `side_thermal`、`Twall_left`、`Twall_right` | `:insulating`；两个温度字段保持默认 0 | 侧壁绝热（法向通量为零）。取 `:fixed` 时两侧壁分别固定为给定物理温度，约定与顶/底壁相同：`Twall - Tbar`。 |
| `Twall_top`、`Twall_bot` | 0、1 | 顶壁与底壁温度；参数卡不显式设置，保持 `Params` 的默认值。 |
| `H_half_life` | `Inf` | 放射性生热保持恒定。取有限值时按 `H·2^(-t/H_half_life)` 衰减，时间单位与代码一致；只能用 Picard 时间推进。 |
| `Di` | 0 | 关闭耗散加热与绝热加热。 |
| `heating_adiabatic`、`heating_dissipation` | `false`、`false` | 绝热加热与耗散加热的开关；`Di = 0` 时两项都不产生源项。 |
| `refstate` | `nothing` | 不使用参考态。 |
| `deviatoric` | `false` | 不做偏量投影。 |
| `bc`、`wall_bc`、`wall_un`、`wall_t`、`wall_tn` | `:free`；逐壁字段来自 `CASE_WALL_*` | 四壁自由滑移；逐壁覆盖与四组数据的语义见第 2 节，变量名与默认值见 6.1 节。 |

`Params` 还会校验黏度定律。 `:composite` 拒绝 Drucker–Prager 屈服面； `:tosi` 自带塑性分支，因此拒绝 `yield_*` ； `:arrhenius` 要求 `eta_T0 > 0` 。

## 7. 输出格式

所有文件都写在 `CASE_OUT` 下。目录由程序创建；只有要用 shell 重定向把日志写进该目录时，才需要它事先存在。

### 7.1 run.log

这个文件不是程序创建的，而是 stdout 与 stderr 的 shell 重定向。内容包括四行头部、每步一行（`verbose=true` 时）、每 5 步一行 AMR（带加密、粗化、合并计数以及平衡循环的计数器）、每 100 步一行参数卡回调的进度，以及最后两行：

```text
[case] DONE mode=spinup steps=2000 t=<t> wall=<s> s (<s> s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=<v> Nu_bot=<v> Vrms=<v> meanT=<v> closure=<v> leaves=<n> N=<n>
```

stdout 是块缓冲的，回调每 100 步 flush 一次，因此重定向出来的日志最多滞后 100 步。错误走 stderr，不缓冲。

### 7.2 history.csv

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

设 `CASE_EXTDIAG=1` 会追加一列 `resid_A` ：所引用的逐点 Nusselt 数与格式能量恒等式之间的精确残差，按单位面积计，因此对时间步通量为零，对逐点通量为 $O(h)$ 。设 `nu_cbf=true` 会追加一致边界通量的一对 Nusselt 数 `Nu_cbf_top` 与 `Nu_cbf_bot` 。两项可以同时打开；在本卡上把这两项都打开时，追加的列依次是 `closure_nolatent`、`Nu_cbf_top`、`Nu_cbf_bot`、`resid_A` 。

### 7.3 帧文件

每 `CASE_FRAME` 步存一帧，并错开每个 AMR 周期的第一步：AMR 那一轮会把速度清零，如果在 AMR 步上写帧，得到的是零速度和没有意义的 Nusselt 数。最后一帧总会补上。

每个帧步写下列文件：

| 文件 | 内容 |
|---|---|
| `frames/frame_NNNNN_T.bin` | 完整温度 $T=\bar T+\theta$ ，256 × 256 网格，小端 Float64，512 KB。 |
| `frames/frame_NNNNN_mesh.csv` | 每个叶一行，按叶序排列（先 level，再 `ix`，再 `iy`），列为 `level,x0,y0,h` ，17 位有效数字。 |
| `frames/frame_NNNNN_eta.bin` | 每个速度节点一个单精度值，为 $\log\_{10}(\eta/\eta\_{\mathrm{ref}})$ ，供渲染器使用。 |
| `frames/frame_NNNNN_speed.bin` | 每个速度节点一个单精度值，为速度模。 |
| `frames/frame_NNNNN_nodes.bin` | 每个速度节点两个单精度值，先 $x$ 后 $y$ 。 |
| `frames/frame_meta.csv` | 追加一行元数据，见下。 |

`T.bin` 的 $x$ 变化最快，然后 $y$ ；在 numpy 里就是 `T = np.fromfile(path).reshape(256, 256)` ， `T[j, i]` 是 $(x\_i,y\_j)$ 处的值，画图时用 `origin="lower"` 。 `mesh.csv` 里的 $h$ 是叶高， $x\_0$ 、 $y\_0$ 是左下角坐标。

`frame_meta.csv` 以追加方式打开，每帧一行：

```text
frame,step,t,dt,N,leaves,nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac,wall_s,ng
```

`nlev` 是不同层级的个数，`eta_clip_frac` 是黏度贴住上下限的速度节点比例，`wall_s` 是写帧器构造至今的秒数。

### 7.4 检查点

单个 HDF5 文件，每 `checkpoint_every` 步写一次，结束时再写一次，始终写到同一路径（`$CASE_OUT/spinup.h5`、`frozen.h5`、`resume.h5` 等）。它保存把轨迹接下去所需的全部状态：

| 内容 | 形状 |
|---|---|
| `leaves` | `3 × nleaves` Int32：每个叶的 level、ix、iy，已排序。 |
| `theta`、`theta_prev` | 标量节点上的两个 BDF2 温度层。 |
| `ux`、`uz` | 标量节点上的两个速度分量。 |
| `pv` | 压力空间上的压力。 |
| `ev` | 可选，每个叶一个熵黏性值，按 `leaves` 的顺序。 |
| `weak`、`weak_prev` | 可选，两个层的应变弱化场，本卡没有。 |
| `txx`、`tyy`、`txy` | 可选，三个 Maxwell 偏应力分量； `shear_modulus = 0` 时不存在（见 1.8 节）。 |
| `cooldown_no_refine`、`cooldown_no_coarsen` | 可选，平衡循环的迟滞状态，每个叶键一列。 |
| 属性 | `t`、`dt`、`dt_prev`、`step`、`p`、`maxlevel`、`nx0`、`ny0`、`Lx`、`cooldown_cycle`、`velocity_stale`、自由格式元数据，以及 `config`：影响轨迹的设置，写成按换行分隔的 `key=value` 字符串。 |

`velocity_stale` 标记最后一步是 AMR 步的收尾检查点：里面存的速度只是诊断性重解，重启时会被清零。

## 8. 重启与配置守卫

`restart_from` 会在第一步之前用检查点内容替换网格、`p` 和初始状态；`checkpoint_path` 让续算写到自己的文件。`maxsteps` 与 `tstop` 始终按绝对量解释。守卫把检查点里存的 `config` 与本次调用的设置逐键比较，只要有一项不同就拒绝启动，并在错误信息里逐行列出 `键: checkpoint=..., current=...` 。完整报错样例与处理步骤见[快速开始](quickstart.md)的「出问题怎么办」。

参与比较的是会影响轨迹的那些键：

| 类别 | 键 |
|---|---|
| AMR | 节奏、各轮设置、份额、叶数上限、层级上限、壁面规则、热点与顶壁规则、对称、传递模式、迟滞与校验。 |
| 壁面运动与边界数据 | `wall_bc`；`wall_u`、`wall_un`、`wall_t` 与 `wall_tn` 按每面壁记成 `none`/`custom` 标记：函数没有可比较的取值，若「是否有自定义值」不变而函数本身被替换，守卫无法检出，应从头重算（见 2.7 节）。 |
| 离散 | 对流形式、熵黏性系数。 |
| 非线性与时间步 | `maxpic`、`pic_tol`、`pic_relax`、`pic_mode`、`pic_reject`、`cfl`、`dtmax`、`dtmin`。 |
| 质量与投影 | `mass_form`、`deviatoric`。 |
| 加热 | 加热开关与 `H`。 |
| 流变 | `:composite` 记 `comp_*`、`comp_rule`、`eta_T0`、`eta_min`、`eta_max`；`:tosi` 记 `tosi_*` 块。`:expT`、`:expTz`、`:arrhenius` 的定律参数不写入 `config`，续算前要自己确认一致。 |
| 屈服与弱化 | 屈服、弱化参数。 |
| 潜热 | `latent_L`、`latent_mode`、`latent_scheme` 与相变列表。 |
| 参考态 | 参考态相关设置。 |
| 缩放 | `stokes_scaling`。 |

`physics_revision` 是物理代码的属性，而不是某个参数的取值：驱动拒绝在另一个 revision 下写出的检查点；没有这个标记的文件会在警告后续算，警告里会说明该文件是否带有 Maxwell 应力历史。

比较以检查点里的键为准迭代，所以比检查点更晚出现的键不参与比较。 `stokes_scaling` 有明确默认值：缺这个键的文件按 `none` 读取，而给这类轨迹打开该选项会被拒绝。

属于状态、不参与配置比较的内容：`p` 与网格从文件恢复，平衡循环的冷却状态也随之恢复。 `dt0`、`dt_growth`、`initial_refine`、`linear`、`mg_tol`、`mg_threads` 以及其他求解器设置既不存储也不比较，因此同一条轨迹可以用不同的线性路径或不同的检查点节奏继续。

续算被拒时，把存下的值原样重设一遍，或者改用种子模式。种子模式复制检查点，只改写冲突的那一个字符串：`freeze` 把 `amr_every` 设为 0，`wallrefine` 设置 `wall_min_level` 与 `wall_strip` 并加密壁面，`symseed` 设置 `symmetry` 。源文件始终不被修改。

## 9. 线程与内存

用 `--threads=N` 启动 Julia。本卡唯一用到 Julia 线程池的地方是 `linear = :auto` 选中的无矩阵 V 循环，它被限制在八个线程（`mg_threads = min(8, Threads.nthreads())`），所以开超过八个对 `auto` 路径没有收益。单线程时 `auto` 会警告一次并串行运行。显式指定 `linear = :mg` 时 `mg_threads` 保持为 1，除非调用者自己设置；直接分解在任何情况下都是串行的。

实测线程收益：八线程下，生产规模 Stokes 的热启动求解耗时 1.36 s，直接求解 3.37 s；完整求解相对单线程的加速是 1.30 到 1.42 倍，V 循环是 1.77 到 2.05 倍。需要逐位可复现时固定用 `linear = :direct` 。

生产状态下的实测内存：在 10 000 叶上限下，八线程多重网格路径峰值常驻约 1.3 GB，直接分解约 2.4 GB。两者都是进程峰值，包含 Julia 运行时和已加载的状态。默认运行应预留 2 GB 空闲内存，直接路径预留 3 GB。

磁盘几乎全被帧吃掉。7.3 节列出每帧写的文件，大小随树变化；一个检查点约 2 MB。

## 10. 下一步读什么

想核对本页这些默认值，见[验证](validation.md)；上手之后想改某一处，见[常见问题](faq.md)。

下一步：可运行的算例、每个的命令与用来核对的数见[算例](examples.md)。



