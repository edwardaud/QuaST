<p align="right"><a href="../en/examples.md">English</a> · <a href="../zh/examples.md">中文</a></p>

# 算例

四个可以直接跑的例子。每个都给出命令、物理设置、八线程实测墙钟，以及一个用来核对的数。命令都在仓库根目录执行，第一条只需跑一次：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

下面的时间都是八线程墙钟，不是 CPU 时间。直接求解器在进程内是串行的，所以直接路径用不上另外七个线程。测量记录与条件见[验证卷](../dev/VALIDATION.md)和[真实算例卷](../dev/CASE_REALISTIC.md)。

| 算例 | 命令 | 八线程墙钟 | 核对点 |
|---|---|---|---|
| Blankenbach 2b（稳态） | `benchmark_cases.jl 2b 3 0.2` | 约 45 分钟 | `Nu_top = Nu_bot` |
| blob | `run_blob_kelly_1024.jl` | 141 s | `Tmax25` 与两个摘要 |
| 真实算例 smoke | `case_realistic_2d.jl smoke` | 首次约 2 分钟 | `Nu_top = 5.999881` |
| 程序化运行 | `build_and_run.jl` | 首次约 84 s | `Nu_top = Nu_bot = 1.000783` |

## 1. Blankenbach 2b

2b 基准是 2.5 × 1 盒子，四壁自由滑移，`Ra = 1e4`，没有内部加热，黏度服从温度与深度的联合指数律

$$
\eta = \exp(-bT + c(1-z)),
\qquad
b = \ln 16384,
\qquad
c = \ln 64,
$$

其中 $T$ 是温度， $z$ 是高度， $\eta$ 是黏度。参考值为 `Nu = 6.9299`、`Vrms = 171.755`（Blankenbach et al. 1989，表 9）。

### 跑到稳态

`examples/benchmark_cases.jl` 是现在推荐的入口。它从 ASPECT 自带的温度场起步——那份初值已经接近稳态，所以只需几百到几千步，而不是几万步：

```bash
julia --project=. --threads=8 examples/benchmark_cases.jl 2b 3 0.2
```

参数依次是 `算例 层级 tstop`。level 3 把 5 × 2 基础网格细分三次，得到 40 × 16 个方格叶，`p = 2`，走直接求解器。跑完打印

```text
[2b] steps=5492 t=2.0001e-01 wall=20530 s  Nu_top=7.248805 Nu_bot=7.250974 Vrms=170.213063  leaves=640
[2b] reference Nu=6.929900 (4.60%)  Vrms=171.755000 (-0.90%)
```

两个壁面热流相差 0.03%，说明已经进稳态；`Nu` 偏高 4.60%，是这个网格上壁面欠解析的结果，不是实现错误。验收标准就是上下壁相等——无内热时稳态必须相等。

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

`CASE_WALLMIN` 指定壁面条带内与壁面行的最小层级，`CASE_INITREF=1` 让初值先做一轮自适应。壁面加密态上三种 `Nu` 定义（壁面梯度、稳态 CBF、能量族）分别是 6.9480、6.9325、6.9330，散布 0.22%。

## 2. blob 回归

blob 是 `Ra = 1e8` 的自适应路径回归。盒子 1 × 1，四壁自由滑移，内部加热 `H = 5`，黏度用同一个双指数律，初值是传导基态叠一个高斯：

$$
\theta_0(x,z) = 0.12\exp\left[-\left(\frac{x-0.5}{0.12}\right)^2-\left(\frac{z-0.35}{0.10}\right)^2\right]\sin(\pi z),
$$

其中 $\theta\_0$ 是初始温度异常， $x$ 、 $z$ 是水平与竖直坐标。

```bash
julia --project=. --threads=8 examples/run_blob_kelly_1024.jl
```

脚本用 `p = 4`、直接求解器、每 5 步一次 Kelly 自适应、1024 叶硬上限，每 `4e-10` 存一帧到 `t = 1e-8`，帧网格 256 × 256。启动时它会删掉自己的输出目录，换路径请设 `BLOB_OUT` 。最终停在 898 叶、25 帧。本机八线程跑完 126 步用 141.3 s，含 Julia 启动约 2.5 分钟。

最后两行是

```text
done t=1.008e-08 steps=126 leaves=898 N=15361 frames=25 in 141.3 s
FINAL frames=25 end_theta_sha256=a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda end_u_sha256=1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51 Tmax25=1.264329738
```

`end_theta_sha256` 是逐位校验，`Tmax25` 是拿来比对的标量。ASPECT 同一帧的 `Tmax = 1.2531`，这里 1.264330，高 **0.9%**。场级差 2.87% 是经典分支的测量结果，没有在当前自适应分支上重测，所以不要把默认运行说成 2.87%。摘要只在固定的 Julia、BLAS 和线程栈上逐位成立，`Tmax25` 则是平台容差内的检查。

## 3. 真实算例

旗舰算例是 2.5 × 1 盒子，`Ra = 1e5`、`H = 10`，等应力复合流变，`z = 0.87` 与 `0.78` 两个相变并带逐相变潜热，`p = 2`，`max_level = 7` 的四叉树，10000 叶上限，每 5 步一次自适应。初始异常为

$$
\theta_0(x,y) = 0.05\sin(\pi y)\cos\left(\frac{2\pi x}{2.5}\right) + 0.02\sin(2\pi y)\cos\left(\frac{4\pi x}{2.5}\right),
$$

其中 $x$ 、 $y$ 是水平与竖直坐标。 `CASE_OUT` 一定要设到临时目录，因为默认路径可能残留着生产运行留下的文件。

冒烟是完整参数卡跑 10 步：

```bash
CASE_OUT=out_julia/demo_smoke CASE_STEPS=10 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

首次约两分钟，主要是 Julia 编译和第一次 Stokes 求解；本机预热后 47.6 s。收尾必须是 `[case] DONE mode=smoke steps=10` 加上下面这行：

```text
[case] final: Nu_top=5.999881 Nu_bot=-4.029590 Vrms=6.808435e+03 meanT=1.333328 closure=-8.794e-04 leaves=2644 N=11233
```

这个基态的传导参考是 `Nu_top = 6`，所以十步之后顶壁通量的相对误差是 **2e-5**；底壁比它的 `-4` 参考低 0.74%，因为底边界层还在形成。

两千步预积分是同一张参数卡打开自适应：

```bash
CASE_OUT=out_julia/demo CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

树在前 100 步从 640 叶长到 10000 叶上限，每 200 步和退出时写 `spinup.h5`。本页没有跑满这两千步。按项目记录的 4.13 s/步估算约 **2.3 h**，这是上界，因为早期树小、每步更便宜。这张参数卡没有解析参考：两千步时仍是启动瞬态。用同一张卡的另一段两千步运行来比，不要比两万步的终点（`Nu_top = 18.2892`、`Nu_bot = -2.7636`）。

`resume` 与 `freeze` 是接同一个检查点的两条路。`resume` 保留自适应，`freeze` 只把 `amr_every` 改成 0，在冻结网格上继续：

```bash
CASE_OUT=out_julia/demo_frozen CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl freeze \
  out_julia/demo/spinup.h5

CASE_OUT=out_julia/demo_resume CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

两条都从第 2 000 步再走 2 000 步到第 4 000 步，各约 **2.3 h**，本页没有重跑。 `CASE_OUT` 要分开：各自写自己的 `frames/` 与 `history.csv` 。 `CASE_STEPS` 是绝对步号，不大于检查点步号时驱动一步不跑。对冻结后的检查点用 `resume` 会被守卫拒绝，因为该模式传的是 `amr_every = 5` ；这一支要用 `freeze` 接着跑 `frozen.h5` 。

## 4. 程序化调用

`run_semquad_simulation` 接收一棵树、一张 `Params` 参数卡和一个初始温度异常。把下面这段存成 `build_and_run.jl`，它是完整的，每步打印一行历史。

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

首次约 84 s，几乎全是驱动第一次调用时的编译。历史每行是 `[t, dt, Nu_top, Nu_bot, Vrms, meanT, N, leaves, ...]` ，所以 `row[3]` 、 `row[4]` 是壁面 Nusselt 数，`row[5]` 是 `Vrms` 。输出以

```text
steps=20 t=2.000000e-03 leaves=256 N=1089
last row: Nu_top=1.000783 Nu_bot=1.000783 Vrms=2.762497
```

收尾。最严格的检查是对称性：流场关于中轴反射对称，`Nu_top` 与 `Nu_bot` 必须打印出同一个数；差超过 `1e-6` 就说明 Stokes 或能量求解被干扰了。相对传导值 1 的 0.000783 偏移是 `t = 2e-3` 时的物理瞬态，不是求解误差。换一套 BLAS 重跑，`Nu` 与 `Vrms` 应在 `1e-5` 内一致；直接求解器的舍入远小于这个量级。
