<p align="right"><a href="../en/examples.md">English</a> · <a href="../zh/examples.md">中文</a></p>

# 算例

六个可以直接跑的例子。每个都给出命令、物理设置、八线程实测墙钟，以及一个用来核对的数。命令都在仓库根目录执行，第一条只需跑一次：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

下面的时间都是八线程墙钟，不是 CPU 时间。直接求解器在进程内是串行的，所以直接路径用不上另外七个线程。全新的 Julia 进程里第一次 `run_semquad_simulation` 主要花在编译驱动上（约 70 s，是整个仓库共同的首次开销）；仿真本身都很短。各数的验证依据与适用范围见[验证](validation.md)。

| 算例 | 命令 | 八线程墙钟 | 核对点 |
|---|---|---|---|
| Blankenbach 2b（稳态） | `benchmark_cases.jl 2b 3 0.2` | 约 45 分钟 | `Nu_top = Nu_bot` |
| blob | `run_blob_kelly_1024.jl` | 141 s | `Tmax25` 与两个摘要 |
| 真实算例 smoke | `case_realistic_2d.jl smoke` | 首次约 2 分钟 | `Nu_top = 5.999881` |
| 区域通道 | `channel_flow_2d.jl` | 首次约 70 s，仿真不到 1 s | 剖面误差低于 `1e-10` |
| 区域俯冲型算例 | `subduction_2d.jl rheology` | 首次约 8 分钟 | `Nu_top = 31.383`、`delta(0.8) = 0.0522` |
| 程序化运行 | `build_and_run.jl` | 首次约 84 s | `Nu_top = Nu_bot = 1.000783` |

## Blankenbach 2b

2b 基准是 2.5 × 1 盒子，四壁自由滑移， `Ra = 1e4` ，没有内部加热，黏度服从温度与深度的联合指数律

$$
\eta = \exp(-bT + c(1-z)),
\qquad
b = \ln 16384,
\qquad
c = \ln 64,
$$

其中 $T$ 是温度， $z$ 是高度， $\eta$ 是黏度； $b$ 与 $c$ 的数值分别是 9.70406 与 4.15888。参考值为 `Nu = 6.9299` 、 `Vrms = 171.755` （Blankenbach et al. 1989，表 9）。

### 跑到稳态

`examples/benchmark_cases.jl` 是现在推荐的入口。它从 ASPECT 自带的温度场起步——那份初值已经接近稳态，所以只需几百到几千步，而不是几万步：

```bash
julia --project=. --threads=8 examples/benchmark_cases.jl 2b 3 0.2
```

参数依次是 `算例 层级 tstop` 。level 3 把 5 × 2 基础网格细分三次，得到 40 × 16 个方格叶， `p = 2` ，走直接求解器。跑完打印

```text
[2b] steps=5492 t=2.0001e-01 wall=20530 s  Nu_top=7.248805 Nu_bot=7.250974 Vrms=170.213063  leaves=640
[2b] reference Nu=6.929900 (4.60%)  Vrms=171.755000 (-0.90%)
```

两个壁面热流相差 0.03%，说明已经进稳态； `Nu` 偏高 4.60%，是这个网格上壁面欠解析的结果，不是实现错误。验收标准就是上下壁相等——无内热时稳态必须相等。

### 为什么壁面要单独加密

`Nu` 取的是边界层里的单侧梯度，而均匀加密把算力摊在无关紧要的内部。证据是实测的：

| 2b 状态 | `Nu` 相对参考 |
|---|---:|
| 均匀 level 3（40 × 16） | +4.60% |
| 均匀 level 4（80 × 32） | +5.31%（未收敛即中止） |
| 壁面加密到 level 4 | +0.26% |

均匀网格从 L3 到 L4 不但没有改善，反而略升；而壁面加密用远少的单元就把误差压到 0.26%。启用壁面规则：

```bash
CASE_WALLMIN=5 CASE_INITREF=1 julia --project=. --threads=8 examples/benchmark_cases.jl 2b 4 0.2
```

`CASE_WALLMIN` 指定壁面条带内与壁面行的最小层级， `CASE_INITREF=1` 让初值先做一轮自适应。壁面加密态上三种 `Nu` 定义（壁面梯度、稳态 CBF、能量族）分别是 6.9480、6.9325、6.9330，散布 0.22%。

## blob 回归

blob 是 `Ra = 1e8` 的自适应路径回归。盒子 1 × 1，四壁自由滑移，内部加热 `H = 5` ，黏度用同一个双指数律，初值是传导基态叠一个高斯：

$$
\theta_0(x,z) = 0.12\exp\left[-\left(\frac{x-0.5}{0.12}\right)^2-\left(\frac{z-0.35}{0.10}\right)^2\right]\sin(\pi z),
$$

其中 $\theta\_0$ 是初始温度异常， $x$ 、 $z$ 是水平与竖直坐标。

```bash
julia --project=. --threads=8 examples/run_blob_kelly_1024.jl
```

脚本用 `p = 4` 、直接求解器、每 5 步一次 Kelly 自适应、1024 叶硬上限，每 `4e-10` 存一帧到 `t = 1e-8` ，帧网格 256 × 256。启动时它会删掉自己的输出目录，换路径请设 `BLOB_OUT` 。最终停在 898 叶、25 帧。本机八线程跑完 126 步用 141.3 s，含 Julia 启动约 2.5 分钟。

最后两行是

```text
done t=1.008e-08 steps=126 leaves=898 N=15361 frames=25 in 141.3 s
FINAL frames=25 end_theta_sha256=a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda end_u_sha256=1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51 Tmax25=1.264329738
```

`end_theta_sha256` 是逐位校验，`Tmax25` 是拿来比对的标量：与 ASPECT 同一帧结果的对照见[验证](validation.md)的跨代码对照一节。场级差 2.87% 是经典分支的测量结果，没有在当前自适应分支上重测，所以不要把默认运行说成 2.87%。摘要只在固定的 Julia、BLAS 和线程栈上逐位成立，`Tmax25` 则是平台容差内的检查。

## 真实算例

旗舰算例是 2.5 × 1 盒子， `Ra = 1e5` 、 `H = 10` ，等应力复合流变， `z = 0.87` 与 `0.78` 两个相变并带逐相变潜热， `p = 2` ， `max_level = 7` 的四叉树，10000 叶上限，每 5 步一次自适应。初值、边界条件与全部开关见[用户指南](guide.md)第 1 节。本节命令的输出目录都放在 `out_julia/` 下的临时目录里；`CASE_OUT` 的取法见[快速开始](quickstart.md)的「三个必须弄明白的旋钮」。

冒烟是完整参数卡跑 10 步。首次约两分钟，主要是 Julia 编译和第一次 Stokes 求解；本机预热后 47.6 s。命令与输出文件见[快速开始](quickstart.md)的十步冒烟一节；收尾必须是 `[case] DONE mode=smoke steps=10` 加上下面这行：

```text
[case] final: Nu_top=5.999881 Nu_bot=-4.029590 Vrms=6.808435e+03 meanT=1.333328 closure=-8.794e-04 leaves=2644 N=11233
```

顶壁通量与传导值的相对误差是 **2e-5**；底壁比它的传导值低 0.74%，因为底边界层还在形成。传导值见[用户指南](guide.md) 1.3 节。

两千步预积分是同一张参数卡打开自适应：

```bash
CASE_OUT=out_julia/demo CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

树在前 100 步从 640 叶长到 10000 叶上限，每 200 步和退出时写 `spinup.h5` 。本页没有跑满这两千步。按 4.13 s/步估算约 **2.3 h**，这是上界，因为早期树小、每步更便宜。这张参数卡没有解析参考：两千步时仍是启动瞬态。用同一张卡的另一段两千步运行来比，不要比两万步的终点（`Nu_top = 18.2892` 、 `Nu_bot = -2.7636`）。

### 打开黏弹性

Maxwell 黏弹性默认关闭；本节之前的命令跑的都是纯黏性问题，除非加上参数卡旋钮 `CASE_SHEAR_MODULUS` ，也就是无量纲剪切模量 $G$ ，物理定义见[用户指南](guide.md) 1.8 节。十步冒烟是最快的打开方式：

```bash
CASE_OUT=out_julia/demo_ve CASE_STEPS=10 CASE_SHEAR_MODULUS=3.55e6 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

$G$ 是无量纲的：它以参数卡的参考黏度与时间单位为单位，所以同一个物理剪切模量在 `Ra` 、参考黏度或尺度集改变后是不同的数值。这次运行的头部行末尾是 `eta_ref=1 shear_modulus=3.55e+06` 。 `shear_modulus > 0` 时驱动还会额外打印两次 Maxwell 区域报告，第一次在第一个推进应力的步，第二次在运行结束时；报告的形状、判读与代价见[用户指南](guide.md) 1.8 节。

如果参数卡的材料律被重标度，同样的缩放要作用到参考黏度上，否则同一个 `Ra` 描述的是另一个物理 Rayleigh 数；约定见[用户指南](guide.md) 1.9 节。

### 冻结与续算

`resume` 与 `freeze` 是接同一个检查点的两条路。 `resume` 保留自适应， `freeze` 只把 `amr_every` 改成 0，在冻结网格上继续：

```bash
CASE_OUT=out_julia/demo_frozen CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl freeze \
  out_julia/demo/spinup.h5

CASE_OUT=out_julia/demo_resume CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

两条都从第 2 000 步再走 2 000 步到第 4 000 步，各约 **2.3 h**，本页没有重跑。 `CASE_OUT` 要分开：各自写自己的 `frames/` 与 `history.csv` 。对冻结后的检查点用 `resume` 会被守卫拒绝，因为该模式传的是 `amr_every = 5` ；这一支要用 `freeze` 接着跑 `frozen.h5` 。重启规则与绝对步号的含义见[用户指南](guide.md)第 5 节和[快速开始](quickstart.md)的「停下与续算」。

## 区域通道

`examples/channel_flow_2d.jl` 是边界条件功能的组合演示：平面通道的上游壁给定入流法向速度，下游壁是带精确总牵引的 `:open` 壁，上下壁无滑移。参数卡就是普通的 `Params`，通过 `run_semquad_simulation` 跑一步：

```bash
julia --project=. --threads=8 examples/channel_flow_2d.jl
```

通道是 2 × 1，常黏度，由压力梯度 `dp/dx = -G` 驱动， `G = 2` （驱动里的体力是浮力，没有水平分量）。开出的出流给定动压力 `p_out = 2` ，精确解是平面 Poiseuille：

$$
u_x = \frac{G\,y\,(H-y)}{2\eta},
\qquad
p = p_{\text{out}} + G\,(L-x),
$$

因此参数卡里是 `wall_un = y(1-y)` 、 `wall_t = 1 - 2y` （剪切牵引 `eta du/dy`）与 `wall_tn = -2` （沿外法向的出流动压力）。组合配方见用户指南 1.10 节。运行末尾输出

```text
[channel] steps=1 leaves=128 err_u=1.832e-14 err_uz=3.289e-14 err_p=4.567e-13 max|u|=0.250000
```

`err_u` 是最大剖面误差除以中心速度 `G H^2/(8 eta) = 0.25` ， `err_uz` 用同一尺度， `err_p` 是绝对误差。Q2 速度与 Q1 压力精确包含这个解，所以这几个 `1e-14` 是线性求解的误差而不是离散误差。驱动编译好之后，运行本身只要 0.04 s；全新进程约 70 s 花在首次编译上，`julia -O0 --project=. examples/channel_flow_2d.jl` 把启动时间压到约 55 s，在这个精度下答案不变。

通道的门禁 G264 还会跑同一个解在常数体力驱动下的形式（此时 `p` 为常数、 `wall_tn = -p0`），以及自由滑移出流变体（ `wall_t = nothing` ）：后者是另一个有界的流动，因为精确的 Poiseuille 出流本身带有非零剪切，自由滑移解与 Poiseuille 的差恰好是被扣掉那份牵引的响应（这里为中心速度的 18%）。门禁在进程内直接调用本脚本，所以这一页不是只写在文档里的。

## 区域俯冲型算例

`examples/subduction_2d.jl` 是[用户指南](guide.md) 1.10 节区域通道配方的生产用法。2 × 1 盒子（2 000 × 1 000 km）里，热软流圈从远场壁 `x = 0` 流入，在仰冲板块下方穿过，从海沟侧的开出流壁 `x = 2` 流出；冷表面随物质向海沟运动长出板块冷却边界层。这是运动学热模型：`Ra = 0`，热浮力与相变浮力都关闭，两个相变只通过潜热起作用。

五个模式就是 `docs/dev/REGIONAL_DESIGN.md` 的分阶段计划：

```bash
julia --project=. --threads=8 examples/subduction_2d.jl validate
julia --project=. --threads=8 examples/subduction_2d.jl rheology
julia --project=. --threads=8 examples/subduction_2d.jl phases
julia --project=. --threads=8 examples/subduction_2d.jl full
julia --project=. --threads=8 examples/subduction_2d.jl converge
```

`validate` 是解析阶段：黏度恒定，精确解是运动板块与无滑移底面之间的 Couette-Poiseuille 流动，

$$
u_x = u_{\text{ref}}(y) = U_p\,y + G\,y(1-y)/2, \qquad u_y = 0,
\qquad p = p_{\text{out}} + G\,(L_x-x),
$$

其中 `U_p = 1.6e3`、`G = 1e3`、`p_out = 1e3`、`L_x = 2`、黏度 1。上游壁给定法向速度 `u_ref(y)`，板块壁给定 `U_p`，开出流壁给定远场总牵引 `wall_tn = -p_out` 与 `wall_t = tau_ref(y) = U_p + G(1/2-y)`。Q2 速度与 Q1 压力精确包含该解，所以打印的误差是直接线性求解而不是离散误差。该模式以

```text
[regional] check  1 u-field        max|ux-uref|/Up    measured=1.4211e-14 ... PASS
[regional] check  2 p-field        max|p-pref|/pscale measured=1.2738e-12 ... PASS
[regional] check  3 flux-net       |sum Q|/Qref       measured=9.0091e-16 ... PASS
[regional] validate: 13 checks (13 gated): 13 PASS, 0 FAIL, 0 DIAG, 0 SKIP
[regional] validate: PASS
```

结束。`rheology` 加入 10^3 温度依赖黏度并把热场积分到发育的板块冷却状态。本机上它收尾于 `Nu_top = 31.383`、`Nu_bot = 0.212`、`max|u| = 1.600e3 = U_p` ，两站给出 `delta(0.8) = 0.0522`（半空间 0.0519）与 `delta(1.6) = 0.0739`（0.0734），表面异常热流 24.28 与 16.55 对比 25.25 与 17.87——重跑后拿这些数核对。`phases` 加入 410 km 与 660 km 两个相变及其潜热，并打印两个系数（实测 `-0.051193` 与 `+0.105288`）；`full` 跑扩散-位错复合流变，写帧并带 `nu_cbf`/`extdiag` 额外列；`converge` 是设计 6.6 节的网格分辨率序列。每个模式逐项打印检查，并在 `out_julia/regional_channel/<mode>/` 下写 `checks.csv`（模式、检查号、物理量、实测值、容差、PASS/FAIL/DIAG/SKIP）与 `history.csv`。

检查分类是明确的。精确类：阶段 (i) 的场、壁面通量与 `deta_T = 1` 延拓（1e-10，线性求解）。守恒类：逐壁质量通量与净通量为零的恒等式。渐近类：相对半空间解 `T = T_inf erf((1-y)/(2 sqrt(x/U_p)))` 的板块冷却剖面，温度与边界层厚度 10%、表面热流 15%。设计 6.3 节的开盒热收支与 6.7 节的潜热焓恒等式（检查 10）都只作为诊断量报告，不作为门禁；原因见下面的局限。

诚实的局限就是设计第 7、9 节列的那些。没有倾斜俯冲板片界面：海沟侧是竖直开出流壁，板片在盒子之外，算例建模的是"进入的板块加上其下的地幔通道"，不是楔形地幔。没有自由表面、没有地形、没有成分场、没有自洽的板片拉曳。`Ra = 0` 让速度由壁面给定、热演化是运动学的，因此相变只贡献潜热。侧壁温度是标量，无法施加随深度变化的板片顶温度剖面，发表的板片顶温度曲线不是本温度场的目标。因此本算例不能与 van Keken 一类发表的楔形流数字比较；它对应角流的平板块极限，也就是精确的 Couette-Poiseuille。

还有两个读者必须知道的实现事实。壁面数据是参考通道的远场运动学，所以开出流切向牵引是 `wall_t = tau_ref(y) * eta_far(y)/eta_ref`，`eta_far` 取参数卡热侧壁温度下的远场黏度；设计字面上的等黏度*应力*（`CASE_WALL_T=stress`）在 10^3 反差下实测会困住一层 `max|u| ~ 1.7e5` 的出流剪切层，所以不是默认。其次，用单侧壁面梯度拼出的开盒热收支对"侧壁固定温度"的参数卡不闭合：侧壁换热是 Dirichlet 反力，不是已解析层的一侧导数，而库只在上下壁实现了一致边界通量（`docs/dev/VALIDATION.md` 4.1 节）。这两件事与实测数字都记在 `docs/dev/REGIONAL_PROGRESS.md`。

## 程序化调用

`run_semquad_simulation` 接收一棵树、一张 `Params` 参数卡和一个初始温度异常。把下面这段存成 `build_and_run.jl` ，它是完整的，每步打印一行历史。

```julia
using QuaST, Printf

# 1. 树：单位盒，2 × 2 基础网格，均匀加密三次（2*2*4^3 = 256 叶，2:1 平衡）
qt = Quadtree(3; Lx=1.0, cells_x=2, cells_y=2)
for _ in 1:3, lf in collect(qt.leaves)
    refine!(qt, lf)
end
balance!(qt)

# 2. 参数卡：Boussinesq、常黏度、自由滑移。
#    :expT 取 deta_T = 1 就是常黏度极限。
pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
            Twall_top=0.0, Twall_bot=1.0, bc=:free)

# 3. 初始温度异常，两个壁面上为零。
ss = make_semstokes(qt, 2)
theta0 = [0.01 * cospi(x) * sinpi(y) for (x, y) in ss.su.coords]

# 4. 走 20 步，再把历史读回来。
res = run_semquad_simulation(qt, pr, theta0; p=2, tstop=Inf, maxsteps=20,
                             dt0=1.0e-4, dtmax=1.0e-4, cfl=0.0,
                             linear=:direct, verbose=false)
@printf("steps=%d t=%.6e leaves=%d N=%d\n",
        res.steps, res.t, length(res.qt.leaves), res.ss.su.N)
@printf("%3s %12s %10s %10s %10s\n", "i", "t", "Nu_top", "Nu_bot", "Vrms")
for (i, row) in enumerate(res.history)
    @printf("%3d %12.5e %10.5f %10.5f %10.5f\n", i, row[1], row[3], row[4], row[5])
end
@printf("last row: Nu_top=%.6f Nu_bot=%.6f Vrms=%.6f\n",
        res.history[end][3], res.history[end][4], res.history[end][5])
```

```bash
julia --project=. --threads=8 build_and_run.jl
```

首次约 84 s，几乎全是驱动第一次调用时的编译。历史每行是 `[t, dt, Nu_top, Nu_bot, Vrms, meanT, N, leaves, ...]` ，所以 `row[3]` 、 `row[4]` 是壁面 Nusselt 数， `row[5]` 是 `Vrms` 。输出以

```text
steps=20 t=2.000000e-03 leaves=256 N=1089
last row: Nu_top=1.000783 Nu_bot=1.000783 Vrms=2.762497
```

输出以这两行收尾。最严格的检查是对称性：流场关于中轴反射对称， `Nu_top` 与 `Nu_bot` 必须打印出同一个数；差超过 `1e-6` 就说明 Stokes 或能量求解被干扰了。相对传导值 1 的 0.000783 偏移是 `t = 2e-3` 时的物理瞬态，不是求解误差。换一套 BLAS 重跑， `Nu` 与 `Vrms` 应在 `1e-5` 内一致；直接求解器的舍入远小于这个量级。

下一步：这些数字的验证依据与适用范围见[验证](validation.md)。
