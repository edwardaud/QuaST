<p align="right"><a href="../en/examples.md">English</a> · <a href="../zh/examples.md">中文</a></p>

# 算例

七个可以直接跑的例子：真实算例的几种模式、Blankenbach 2b、blob 回归、区域通道、区域俯冲型算例、二维圆环基准，以及程序化调用。每个都给出命令、八线程实测墙钟和一个用来核对的数。命令都在仓库根目录执行，第一条只需跑一次：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

下面的时间都是八线程墙钟，不是 CPU 时间。直接求解器在进程内是串行的，所以直接路径用不上另外七个线程。全新的 Julia 进程里第一次调用 `run_semquad_simulation` 主要花在编译驱动上（约 70 s），除特别注明外，表里给的是模式自身的墙钟，不含这次编译。各数的验证依据与适用范围见[验证](validation.md)。

| 算例 | 命令 | 八线程墙钟 | 核对点 |
|---|---|---|---|
| 真实算例 | `case_realistic_2d.jl smoke` | 首次约 2 分钟；预热后 47.6 s | `Nu_top = 5.999881` |
| Blankenbach 2b（稳态） | `benchmark_cases.jl 2b 3 0.2` | 20 530 s（约 5.7 h） | `Nu_top = Nu_bot` |
| blob 回归 | `run_blob_kelly_1024.jl` | 141 s | `Tmax25 = 1.264329738` |
| 区域通道 | `channel_flow_2d.jl` | 仿真 0.04 s；首次约 1 分钟（编译） | 剖面误差 1.8e-14 |
| 区域俯冲型算例 | `subduction_2d.jl validate` | 15 s | 13 项检查全部 PASS |
| 二维圆环基准，`Ra = 1e5` | `ANN_CFL=0 ANN_DT0=1e3 ANN_MAXPIC=25 ANN_TMIN=0 ANN_WINDOW=3 ANN_TSTOP=1e12 examples/annulus_benchmark.jl 32x384` | 驱动 193 s，端到端约 4 分钟 | `steady=1` 且 `Nu_T = 9.89994` |
| 程序化调用 | `build_and_run.jl` | 首次约 84 s | `Nu_top = Nu_bot = 1.000783` |

## 真实算例（`case_realistic_2d.jl`）

生产算例是 2.5 × 1 盒子， `Ra = 1e5` 、 `H = 10` ，等应力复合流变， `z = 0.87` 与 `0.78` 两个相变并带逐相变潜热， `p = 2` ， `max_level = 7` 的四叉树，10000 叶上限，每 5 步一次自适应。初值、默认边界条件与全部开关见[用户指南](guide.md)第 1 节；边界条件的逐壁语义见指南第 2 节。本节的输出目录都放在 `out_julia/` 下；`CASE_OUT` 的取法见[快速开始](quickstart.md)的「三个必须弄明白的旋钮」。

### 模式与步数

七种模式都由 `CASE_STEPS` 设定步数上限，默认值如下。

| 模式 | 默认步数 | 作用 |
|---|---|---|
| `smoke` | 10 | 完整参数卡跑十步，检查整条链路。 |
| `short` | 150 | 稳定性检查。 |
| `spinup` | 10 000 | 打开 AMR 的生产预积分。 |
| `freeze <ckpt>` | 20 000 | 冻结网格续算（`amr_every = 0`）。 |
| `resume <ckpt>` | 2 000 | 带 AMR 续算。 |
| `wallrefine <ckpt>` | 2 000 | 换一套壁面参数后续算，网格继续自适应。 |
| `symseed <ckpt>` | 不跑步 | 复制检查点，只改写 `CASE_SYMMETRY` 一个字符串，供下一段续算。 |

### 十步冒烟

```bash
CASE_OUT=out_julia/demo_smoke julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

首次约两分钟，主要是 Julia 编译和第一次 Stokes 求解；本机预热后 47.6 s。 `smoke` 不写检查点，输出目录里只有 `history.csv`（表头加 10 行）、`frames/` 下第 1、6、10 步各一帧，以及 3 行的 `frame_meta.csv`；文件格式见指南 7.2 与 7.3 节。收尾必须是 `[case] DONE mode=smoke steps=10` 加上下面这行：

```text
[case] final: Nu_top=5.999881 Nu_bot=-4.029590 Vrms=6.808435e+03 meanT=1.333328 closure=-8.794e-04 leaves=2644 N=11233
```

顶壁通量与传导值的相对误差是 2e-5；底壁比它的传导值低 0.74%，因为底边界层还在形成。传导值见指南 1.3 节。

### 预积分、冻结与续算

两千步预积分是同一张参数卡打开自适应：

```bash
CASE_OUT=out_julia/demo CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

每 200 步和退出时写 `spinup.h5` ；树在这个阶段从 640 叶长到 10000 叶上限。本页没有跑满这两千步；按 4.13 s/步估算约 **2.3 h**，这是上界，因为早期树小、每步更便宜。这张参数卡没有解析参考：两千步时仍是启动瞬态，比较时用同一张卡的另一段两千步运行，不要比两万步的终点（`Nu_top = 18.2892` 、 `Nu_bot = -2.7636`）。

`resume` 与 `freeze` 是接同一个检查点的两条路。 `resume` 保留自适应， `freeze` 只把 `amr_every` 改成 0，在冻结网格上继续：

```bash
CASE_OUT=out_julia/demo_frozen CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl freeze \
  out_julia/demo/spinup.h5

CASE_OUT=out_julia/demo_resume CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

两条都从第 2000 步再走 2000 步到第 4000 步，各约 2.3 h，本页没有重跑。 `CASE_OUT` 要分开：各自写自己的 `frames/` 与 `history.csv` 。对冻结后的检查点用 `resume` 会被守卫拒绝，因为该模式传的是 `amr_every = 5` ；这一支要用 `freeze` 接着跑 `frozen.h5` 。重启规则与绝对步号的含义见指南第 8 节和[快速开始](quickstart.md)的「停下与续算」。

### 打开黏弹性

Maxwell 黏弹性默认关闭，物理定义见指南 1.8 节。十步冒烟是最快的打开方式：

```bash
CASE_OUT=out_julia/demo_ve CASE_STEPS=10 CASE_SHEAR_MODULUS=3.55e6 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

$G$ 是无量纲的：它以参数卡的参考黏度与时间单位为单位，所以同一个物理剪切模量在 `Ra` 、参考黏度或尺度集改变后是不同的数值。这次运行的头部行末尾是 `eta_ref=1 shear_modulus=3.55e+06` 。 `shear_modulus > 0` 时驱动还会额外打印两次 Maxwell 区域报告，第一次在第一个推进应力的步，第二次在运行结束时；报告的形状、判读与代价见指南 1.8 节。

如果参数卡的材料律被重标度，同样的缩放要作用到参考黏度上，否则同一个 `Ra` 描述的是另一个物理 Rayleigh 数；约定见指南 1.9 节。

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

`examples/benchmark_cases.jl` 是现在推荐的入口。它从 ASPECT 自带的温度场起步——那份初值已经接近稳态，所以只需几百到几千步，而不是几万步：

```bash
julia --project=. --threads=8 examples/benchmark_cases.jl 2b 3 0.2
```

参数依次是 `算例 层级 tstop` 。level 3 把 5 × 2 基础网格细分三次，得到 40 × 16 个方格叶， `p = 2` ；`:auto` 在这个规模上解析为直接路径。实测 5 492 步、墙钟 20 530 s（约 5.7 h），跑完打印

```text
[2b] steps=5492 t=2.0001e-01 wall=20530 s  Nu_top=7.248805 Nu_bot=7.250974 Vrms=170.213063  leaves=640
[2b] reference Nu=6.929900 (4.60%)  Vrms=171.755000 (-0.90%)
```

两个壁面热流相差 0.03%，上下壁相等是无内热稳态的验收标准；两行末尾打印的 `Nu` +4.60% 与 `Vrms` −0.90% 就是相对参考值的偏差。逐级收敛记录见[验证](validation.md)的稳态基准一节。

`Nu` 取的是边界层里的单侧梯度，均匀加密把算力摊在无关紧要的内部；壁面规则（见指南 4.3 节）是解决之处。用参数卡的两个旋钮打开：

```bash
CASE_WALLMIN=5 CASE_INITREF=1 julia --project=. --threads=8 examples/benchmark_cases.jl 2b 4 0.2
```

`CASE_WALLMIN` 设置壁面条带内与壁面行的最小层级，`CASE_INITREF=1` 在初值上先做一轮自适应；两者都是 `benchmark_cases.jl` 的旋钮，不是真实算例的。壁面规则带来的实测变化（均匀 level 4 停在 +5.31%，壁面加密后 +0.90%，以及更细状态上三种 `Nu` 定义的散布）见[验证](validation.md)的稳态基准与「2b 的三种热流族」两节。

## 二维圆环基准

`examples/annulus_benchmark.jl` 是 Davies 等（2022）与 ASPECT 文档的二维圆柱壳对流基准：等黏度 Boussinesq，`Ra = 1e5`，两圆面自由滑移，`R0 = 1.22`、`R1 = 2.22`（间隙是长度单位），无内部加热，论文 Eq. 41 初值，闭合周期圆环。[验证](validation.md)给出发表值、容差与分辨率记录，本节是命令。

已发表分辨率是 `128x1536`（ASPECT 级）与 `512x2048`（Firedrake 级），本机最大能跑前者（后者按实测每叶内存需要约 66–72 GB）。稳态用基准脚本文档化的松弛模式达到：取很大的固定步长，时间导数可忽略，于是每次能量求解都是稳态对流–扩散问题，步序列成为耦合稳态的阻尼 Picard 迭代。`32x384` 这一级实测驱动墙钟 193 s（连同启动与编译约四分钟）：

```bash
ANN_CFL=0 ANN_DT0=1e3 ANN_MAXPIC=25 ANN_PIC_RELAX=0.5 ANN_TMIN=0 \
ANN_WINDOW=3 ANN_TSTOP=1e12 ANN_CKPT_DIR=out_julia/ann_ckpt \
julia --project=. --threads=8 examples/annulus_benchmark.jl 32x384
```

要核对的是最后两行，结果行与它相对发表值的偏差：

```text
ANNULUS_RESULT L5 nx=32 ny=384 ... steps=6 ... NuT=9.899941 NuB=9.943853 Vrms=193.25842 meanT=0.382464 ... steady=1 ...
ANNULUS_DEV NuT_vs_aspect=3.6860% ... Vrms_vs_aspect=-0.0582% ... dNu_rel=-0.4426%
```

`steady=1` 表示写明的停止判据触发（相对 `theta`、`Nu_T` 与 `V_RMS` 的变化连续三步低于 `1e-6`）；要对照的数是 `Nu_T = 9.89994`、`Nu_B = 9.94385`、`V_RMS = 193.258`、`<T> = 0.382464`。再往上是 `64x768`（实测驱动墙钟 424 s）与 `128x1536`（1 391 s、峰值内存 13.7 GB、`Nu_T = 9.56743`，相对 ASPECT +0.20%）；被中断的运行会从 `out_julia/ann_ckpt/` 续跑，重跑同一条命令即可。不加 `ANN_*` 设置时脚本走时间精确的 `CFL = 1` 模式，那是物理瞬态，在这个分辨率下需要数千步；脚本头部文档化了每个旋钮与稳态判据。

## blob 回归

blob 是 `Ra = 1e8` 的自适应路径回归。盒子 1 × 1，四壁自由滑移，内部加热 `H = 5` ，黏度用同一个双指数律，初值是传导基态叠一个高斯：

$$
\theta_0(x,z) = 0.12\exp\left[-\left(\frac{x-0.5}{0.12}\right)^2-\left(\frac{z-0.35}{0.10}\right)^2\right]\sin(\pi z),
$$

其中 $\theta\_0$ 是初始温度异常， $x$ 、 $z$ 是水平与竖直坐标。

```bash
julia --project=. --threads=8 examples/run_blob_kelly_1024.jl
```

脚本用 `p = 4` 、直接求解器、每 5 步一次 Kelly 自适应、1024 叶硬上限，每 `4e-10` 存一帧到 `t = 1e-8` ，帧网格 256 × 256。启动时它会删掉自己的输出目录，换路径请设 `BLOB_OUT` 。最终停在 898 叶、25 帧。本机八线程跑完 126 步用 141.3 s，含 Julia 启动约 2.5 分钟。最后两行是

```text
done t=1.008e-08 steps=126 leaves=898 N=15361 frames=25 in 141.3 s
FINAL frames=25 end_theta_sha256=a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda end_u_sha256=1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51 Tmax25=1.264329738
```

`end_theta_sha256` 是逐位校验，`Tmax25` 是拿来比对的标量：与 ASPECT 同一帧结果的对照见[验证](validation.md)的跨代码对照一节。场级差 2.87% 是经典分支的测量结果，没有在当前自适应分支上重测，所以不要把默认运行说成 2.87%。摘要只在固定的 Julia、BLAS 和线程栈上逐位成立，`Tmax25` 则是平台容差内的检查。

## 区域通道（`channel_flow_2d.jl`）

`examples/channel_flow_2d.jl` 是三种逐壁机制的组合演示：平面通道的上游壁给定入流法向速度，下游壁是带精确总牵引的 `:open` 壁，上下壁无滑移。三种壁型的语义见[用户指南](guide.md)第 2 节；本节给的是把三者组合成区域入流/出流问题的配方。

通道参数卡按 `(xminus, xplus, yminus, yplus)` 的顺序长这样：

| 壁面 | `wall_bc` | 数据 |
|---|---|---|
| 上游 x 壁 | `:rigid` | `wall_un` = 入流剖面的法向速度 $u\_x(y)$ ；`wall_u` = 切向速度（纯法向入流时为零） |
| 下游 x 壁 | `:open` | `wall_tn` = $-p\_{\mathrm{out}}$ （出流的动压力）以及 `wall_t = \eta\,\mathrm du\_x/\mathrm dy` （剪切牵引） |
| 顶壁与底壁 | `:rigid` 或 `:free` | 不设数据；自由滑移时也可显式给 `wall_un = 0` |

入流壁必须是 `:rigid`： `:free` 壁的切向牵引为零，撑不住带剪切剖面在入口处的 $\eta\,\mathrm du\_x/\mathrm dy$ 。把同样的法向速度给在 `:free` 壁上得到的是另一个流动，在平面通道上实测为中心速度的 5.6%；所以通道入流一律用 `:rigid` ，切向部分交给 `wall_u` 。

三个相容条件让这个组合适定：

- **质量。** 连续性固定了通量。入流数据的积分就是出流必须带走的总量，离散连续性方程保证这个平衡。有开壁时净通量检查被跳过（见指南 2.6 节），入口与出流的不平衡量只会改变出流速度。
- **压力水平。** 开壁配置的压力水平由 `wall_tn` 数据确定（见指南 2.4 节）：它就是出流的动压力 $p\_{\mathrm{out}}$ ，符号随外法向。它是动压力而不是岩石静压：求解器的 `p` 不含基态，所以给一个静水压等于把整根水柱当成牵引加载。
- **适定。** 至少一面 x 壁和一面 y 壁保留 Dirichlet 法向分量（见指南 2.6 节）；通道用刚性入流、两个通道壁和开出的出流满足它。

高度 $H$ 、长度 $L$ 、压力梯度 $G$ 、常黏度 $\eta$ 的通道，精确解是平面 Poiseuille：

$$
u_x=\frac{G\,y\,(H-y)}{2\eta},\qquad u_y=0,\qquad p=p_{\mathrm{out}}+G\,(L-x),
$$

而出流处外法向为 $+e\_x$ 的总牵引是

$$
\boldsymbol\sigma\cdot\mathbf n=\left(-p,\ \eta\frac{\mathrm du\_x}{\mathrm dy}\right)
=\left(-p_{\mathrm{out}},\ G\left(\frac H2-y\right)\right).
$$

也就是 `wall_tn = -p_out` 、 `wall_t = G(H/2 - y)` ；常黏度下切向部分与 $\eta$ 无关。同一个剖面也是常数体力 $f\_x=G$ 、无压力梯度时的精确解，只是此时压力为常数 $p=p\_0$ ，出流法向牵引为 $-p\_0$ 。

$Q\_2$ 速度与 $Q\_1$ 压力精确包含这个解，牵引的面求积对多项式数据也是精确的，所以跑一次通道量到的是线性求解的误差而不是离散误差。演示脚本的命令、输出与墙钟：

```bash
julia --project=. --threads=8 examples/channel_flow_2d.jl
```

```text
[channel] steps=1 leaves=128 err_u=1.832e-14 err_uz=3.289e-14 err_p=4.567e-13 max|u|=0.250000
```

通道是 2 × 1，常黏度，由压力梯度驱动， `G = 2` （驱动里的体力是浮力，没有水平分量）， `p_out = 2` ； `err_u` 是最大剖面误差除以中心速度 $GH^2/(8\eta)=0.25$ ， `err_uz` 用同一尺度， `err_p` 是绝对误差。驱动编译好之后运行本身只要 0.04 s；全新进程约 74 s，其中约 71 s 是首次编译，`julia -O0 --project=. examples/channel_flow_2d.jl` 把启动时间压到约 55 s，精度不变。所以这个算例是「首次编译约一分钟」，不是「运行约一分钟」。

若省略出流切向牵引（ `wall_t = nothing` ），开壁就是自由滑移出口。此时 Poiseuille 剖面**不是**精确解：它的出流剪切 $G(H/2-y)$ 违反零切向牵引条件。自由滑移解保持同样的入流数据与质量通量，与 Poiseuille 的差恰好是被扣掉那份剪切牵引的响应；这个差有界且局域：平面通道上它是中心速度的 18%，其应变能是推导尺度 $L\,\lVert\eta u'\rVert^2/(2\eta^2)$ 的 2.4%。

## 区域俯冲型算例（`subduction_2d.jl`）

`examples/subduction_2d.jl` 是区域通道配方的生产用法。2 × 1 盒子（2 000 × 1 000 km）里，热软流圈从远场壁 `x = 0` 流入，在仰冲板块下方穿过，从海沟侧的开出流壁 `x = 2` 流出；冷表面随物质向海沟运动长出板块冷却边界层。这是运动学热模型：`Ra = 0`，热浮力与相变浮力都关闭，两个相变只通过潜热起作用。

五个模式是分阶段的检查：

```bash
julia --project=. --threads=8 examples/subduction_2d.jl validate
julia --project=. --threads=8 examples/subduction_2d.jl rheology
julia --project=. --threads=8 examples/subduction_2d.jl phases
julia --project=. --threads=8 examples/subduction_2d.jl full
julia --project=. --threads=8 examples/subduction_2d.jl converge
```

| 模式 | 内容 | 八线程墙钟 | 核对点 |
|---|---|---|---|
| `validate` | 等黏度解析阶段，uniform level 3（512 叶），`linear=:direct` | 15 s | 13 项检查全部 PASS；速度 1.4e-14、压力 1.3e-12、净通量 9.0e-16 |
| `rheology` | `deta_T = 1e3` 的温度依赖黏度，积分到板块冷却状态 | 409 s | `Nu_top = 31.383` 、 `Nu_bot = 0.212` 、 `max|u| = 1.600e3 = U_p` 、 `delta(0.8) = 0.0522`（半空间 0.0519）、`delta(1.6) = 0.0739`（0.0734） |
| `phases` | 加入 410 km 与 660 km 两个相变及其潜热 | 463 s | 潜热系数 `-0.051193` 与 `+0.105288` |
| `full` | 扩散-位错复合流变，写帧并带 `nu_cbf`/`extdiag` 额外列 | 154 s | `Nu_top = 31.374` 、 `Nu_bot = 0.283` 、7 帧 |
| `converge` | 手动分辨率序列；60 步的功能性运行只验证三臂能跑 | 149 s（60 步） | 分辨率比较需要各臂跑到稳态 |

表里的墙钟是模式自身的计时；首次调用驱动还要加约 70 s 编译。每个模式逐项打印检查，并在 `out_julia/regional_channel/<mode>/` 下写 `checks.csv`（列是 `mode,id,name,quantity,measured,reference,tolerance,class,status`）与 `history.csv`。

检查分类是明确的。精确类：解析阶段的场、壁面通量与 `deta_T = 1` 延拓，1e-10 相对，量级是线性求解的舍入。守恒类：逐壁质量通量与净通量为零的恒等式。渐近类：相对半空间解 $T = T\_{\inf}\operatorname{erf}((1-y)/(2\sqrt{x/U\_p}))$ 的板块冷却剖面，温度与边界层厚度容差 10%、表面热流 15%。开盒热收支与潜热焓恒等式只作为诊断量报告，不作为验收判据：侧壁的固定温度是 Dirichlet 反力，而库的一致性边界通量只实现上/下壁，所以这两项在这个卡上不是闭合恒等式。

出流切向牵引默认是 `CASE_WALL_T=rate`，即参考通道的剪切率乘以远场黏度；另一种用法 `CASE_WALL_T=stress` 直接把参考通道的等黏度应力当作牵引，在 10^3 反差下它会困住一层 `max|u|` 约 1.7e5 的出流剪切层，所以不是默认。本算例没有倾斜俯冲板片界面、没有自由表面、没有组分场，`Ra = 0` 使它只是运动学热模型。它对应平板块极限的 Couette–Poiseuille 流动，因此不能与 van Keken 一类楔形地幔流的发表数字直接比较。

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

这两行收尾。最严格的检查是对称性：流场关于中轴反射对称， `Nu_top` 与 `Nu_bot` 必须打印出同一个数；差超过 `1e-6` 就说明 Stokes 或能量求解被干扰了。相对传导值 1 的 0.000783 偏移是 `t = 2e-3` 时的物理瞬态，不是求解误差。换一套 BLAS 重跑， `Nu` 与 `Vrms` 应在 `1e-5` 内一致；直接求解器的舍入远小于这个量级。

下一步：这些数字的验证依据与适用范围见[验证](validation.md)。
