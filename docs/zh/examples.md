<p align="right"><a href="../en/examples.md">English</a> · <a href="../zh/examples.md">中文</a></p>

# 算例

四个可以直接跑的例子。每个都给出命令、物理设置、八线程上的实测墙钟和一个用于核对的数。所有命令都在仓库根目录执行；第一条命令只需要跑一次：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

下面的时间都是八线程墙钟，不含 CPU 时间。直接求解器在进程内是串行的，所以直接路径用不到另外七个线程。测量记录和条件在[验证卷](../dev/VALIDATION.md)与[真实算例卷](../dev/CASE_REALISTIC.md)。

| 算例 | 命令 | 八线程墙钟 | 检查 |
|---|---|---|---|
| Blankenbach 2b | `blankenbach_2b.jl 3 0.1` | 329 s | 时间相关瞬态输出 |
| blob | `run_blob_kelly_1024.jl` | 141 s | `Tmax25` 与两个摘要 |
| 真实算例 smoke | `case_realistic_2d.jl smoke` | 首次约 2 分钟 | `Nu_top = 5.999881` |
| 程序化运行 | `build_and_run.jl` | 首次约 84 s | `Nu_top = Nu_bot = 1.000783` |

## 1. Blankenbach 2b

2b 基准是 2.5 × 1 盒子，四壁自由滑移，`Ra = 1e4`，没有内部加热，黏度是联合指数律

$$
\eta = \exp(-bT + c(1-z)),
\qquad
b = \ln 16384,
\qquad
c = \ln 64,
$$

其中 $T$ 是温度，$z$ 是高度，$\eta$ 是黏度。参考值是 `Nu = 6.9299`、`Vrms = 171.755`（Blankenbach et al. 1989，表 9）。

```bash
julia --project=. --threads=8 examples/blankenbach_2b.jl 3 0.1
```

参数是 `[level] [tstop] [ev]`，默认 `level = 3`、`tstop = 0.1`、`ev = 0.11`。level 3 把 5 × 2 基础网格加密三次，得到 40 × 16 个方格叶，`p = 2`，走直接求解器。初值是近传导的

$$
T_0(x,z) = (1-z) + 0.01\cos\left(\frac{\pi x}{2.5}\right)\sin(\pi z),
$$

其中 $x$ 是水平坐标。驱动用 `steady_tol = 1e-5`、`steady_window = 20`、`tmin = 0.02` 判稳态，每 500 步写一次检查点。本机八线程跑 2 550 步、329 s，最后一行是

```
level 3: cells=640 nodes=2673 t=0.100 steps=2550  Nu_top=7.9039 (+14.06%)  Vrms=143.64 (-16.37%)  Nu_bot=7.9960  umax=588.5  [329s]
```

这是一次时间相关运行，不做稳态基准。`t = 0.1` 时状态仍在启动瞬变里，`Nu_top` 比参考高 14.06%，`Vrms` 低 16.37%，两个壁面通量都还没进入稳态关系。它适合检查算例能跑、直接路径可用、输出格式正确，不适合引用百分比精度。

历史 level-3 稳态记录是壁面逐点值 `Nu_top = 6.942798`、`Nu_bot = 6.938484`、`Vrms = 170.962139`，相对参考 +0.19% / -0.46%；它来自阶段延拓流程，当前脚本不复现。更早的能量族记录是 `Nu = 6.9874`（+0.83%）、`Vrms = 171.18`（-0.33%）；在这个分辨率上壁面族收敛最慢。壁面加密的 level-4 状态给出壁面、CBF、能量三族 `6.9480`、`6.9325`、`6.9330`，散布 0.22%，`Vrms = 172.010`（+0.15%）；要百分比级的 `Nu` 就读这一行。逐位检查是 `t_g150_regression_snapshot`：5 × 2、level-2 网格、EV 加 CFL、从 ASPECT 初值走 20 步，钉住 `Nu_top = 7.029996422723808`、`Nu_bot = 5.585292640587811`、`Vrms = 125.9039928721316`，相对容差 `1e-6`；没有 gate 钉 `171.18`。

## 2. blob 回归

blob 是钉住自适应路径的 `Ra = 1e8` 回归。盒子是 1 × 1，四壁自由滑移，内部加热 `H = 5`，黏度用同一个双指数律，初值是传导基态加一个高斯：

$$
\theta_0(x,z) = 0.12\exp\left[-\left(\frac{x-0.5}{0.12}\right)^2-\left(\frac{z-0.35}{0.10}\right)^2\right]\sin(\pi z),
$$

其中 $\theta_0$ 是初始温度异常，$x$ 和 $z$ 是水平与竖直坐标。

```bash
julia --project=. --threads=8 examples/run_blob_kelly_1024.jl
```

这个脚本用 `p = 4`、直接求解器、每 5 步一次 Kelly AMR、1024 叶硬预算，每 `4e-10` 写一帧到 `t = 1e-8`，帧网格 256 × 256。它启动时删除自己的输出目录，所以换路径要设 `BLOB_OUT`。平衡 AMR 是当前唯一的自适应策略，最终停在 898 叶、25 帧。本机八线程用 141.3 s 跑完 126 步；含 Julia 启动约 2.5 分钟。

最后两行是

```
done t=1.008e-08 steps=126 leaves=898 N=15361 frames=25 in 141.3 s
FINAL frames=25 end_theta_sha256=a4fb8248558a5c8fb7c776a7d3364a981374f27533528ce56ea105542b8cdeda end_u_sha256=1cfcb5c63a476ed2e9a62e9700917ae718bc3165d9a485a9d54609a4844b0b51 Tmax25=1.264329738
```

`end_theta_sha256` 是逐位钉，`Tmax25` 是拿来比较的标量。ASPECT 在同一帧的 `Tmax = 1.2531`，计算值 1.264330 比它高 **0.9%**。场级差 2.87% 是经典分支的测量，没有在平衡分支上重测，所以不要把这个默认运行说成 2.87%。经典分支记录过的摘要是 `868a79d92a78bcfe89cfacd4774a0301c0eeb0a865284ab005155f6f458afe88`，`Tmax25 = 1.264427445`；那条路径已删，不再可复现。摘要只在固定的 Julia、BLAS 和线程栈上逐位成立，`Tmax25` 是平台容差内的检查。

## 3. 真实算例

旗舰算例是 2.5 × 1 盒子，`Ra = 1e5`、`H = 10`，复合等应力流变，`z = 0.87` 和 `0.78` 两个相变并带逐相变潜热，`p = 2`，`max_level = 7` 的四叉树，10000 叶预算，每 5 步 AMR。初始异常是

$$
\theta_0(x,y) = 0.05\sin(\pi y)\cos\left(\frac{2\pi x}{2.5}\right)
+ 0.02\sin(2\pi y)\cos\left(\frac{4\pi x}{2.5}\right),
$$

其中 $x$ 和 $y$ 是水平与竖直坐标。始终把 `CASE_OUT` 设到临时目录；回退路径可能是生产运行用过的目录。

冒烟是完整参数卡跑 10 步：

```bash
CASE_OUT=out_julia/demo_smoke CASE_STEPS=10 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl smoke
```

首次约两分钟，大部分是 Julia 编译和第一次 Stokes 求解；本机预热后 47.6 s。它必须以 `[case] DONE mode=smoke steps=10` 和下面这行收尾：

```
[case] final: Nu_top=5.999881 Nu_bot=-4.029590 Vrms=6.808435e+03 meanT=1.333328 closure=-8.794e-04 leaves=2644 N=11233
```

这个基态的传导参考是 `Nu_top = 6`，所以 10 步后顶壁通量相对误差 **2e-5**；底壁比它的 `-4` 参考低 0.74%，因为底边界层还在形成。

2 000 步预积分是同一张参数卡打开 AMR：

```bash
CASE_OUT=out_julia/demo CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

树从 640 叶在前 100 步长到 10000 叶上限，每 200 步和退出时写 `spinup.h5`。这一页没有跑满 2 000 步。项目记录是 10000 叶上限上 4.13 s/步，所以 2 000 步约 **2.3 h**；早期树小、步更便宜，2.3 h 是上界。这张卡没有解析参考：2 000 步的状态仍是启动瞬态。两段历史生产运行在共同早期窗口里 `Nu_top` 一致到 0.01%，到 `t = 4.34e-3` 差 5.1%。用同一张卡的另一段 2 000 步运行来比，不要比 20000 步 E 端点（`Nu_top = 18.2892`、`Nu_bot = -2.7636`）。

`resume` 和 `freeze` 是继续同一个检查点的两条路。`resume` 保持 AMR，`freeze` 只把 `amr_every` 改写成 0，在冻结网格上继续：

```bash
CASE_OUT=out_julia/demo_frozen CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl freeze \
  out_julia/demo/spinup.h5

CASE_OUT=out_julia/demo_resume CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

两条都从第 2 000 步再跑 2 000 步到第 4 000 步。按项目记录的 4.13 s/步，各约 **2.3 h**；这一页没有重跑任一条。用分开的 `CASE_OUT`：各自写自己的 `frames/` 和 `history.csv`。`CASE_STEPS` 是绝对步号，预算小于等于检查点步号时驱动跑零步、什么都不写。对冻结检查点用 `resume` 会被守卫拒绝，因为该模式传 `amr_every = 5`；这一支要用 `freeze` 接着跑 `frozen.h5`。

## 4. 程序化运行

`run_semquad_simulation` 接收一棵树、一张 `Params` 参数卡和一个初始异常。把下面这段存成 `build_and_run.jl`，它是完整的，每步打印一行历史。

```julia
using QuaST, Printf

# 1. 树：单位盒，2 × 2 基础网格，均匀加密三次
#    （2*2*4^3 = 256 叶，2:1 平衡）。
qt = Quadtree(3; Lx=1.0, cells_x=2, cells_y=2)
for _ in 1:3, lf in collect(qt.leaves)
    refine!(qt, lf)
end
balance!(qt)

# 2. 参数卡：Boussinesq、常黏度、自由滑移。
#    :expT 取 deta_T = 1 是 T 律的常黏度极限。
pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
            Twall_top=0.0, Twall_bot=1.0, bc=:free)

# 3. 初始温度异常，两个壁面为零。
ss = make_semstokes(qt, 2)
theta0 = [0.01 * cospi(x) * sinpi(y) for (x, y) in ss.su.coords]

# 4. 前进 20 步，再读回历史。
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

首次运行约 84 s，几乎全是驱动第一次调用时的 Julia 编译。历史行是 `[t, dt, Nu_top, Nu_bot, Vrms, meanT, N, leaves, ...]`，所以 `row[3]` 和 `row[4]` 是壁面 Nusselt 数，`row[5]` 是 `Vrms`。输出以

```
steps=20 t=2.000000e-03 leaves=256 N=1089
last row: Nu_top=1.000783 Nu_bot=1.000783 Vrms=2.762497
```

收尾。严格检查是对称性：流动关于中轴反射对称，`Nu_top` 和 `Nu_bot` 必须打印同一个值；差超过 `1e-6` 说明 Stokes 或能量求解被打扰。相对传导值 1 的 0.000783 偏移是 `t = 2e-3` 的物理瞬态，不属于求解误差。换一套 BLAS 重跑，`Nu` 和 `Vrms` 应在 `1e-5` 内一致；直接求解器的舍入远低于这个量级。
