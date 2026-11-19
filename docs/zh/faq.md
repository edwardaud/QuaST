<p align="right"><a href="../en/faq.md">English</a> · <a href="../zh/faq.md">中文</a></p>

# 常见问题

## 怎么改 Rayleigh 数和内部加热？

在库调用里传 `Params(Ra=..., H=...)`：`Ra` 是 Rayleigh 数，`H` 是体积内部加热。传导基态为

$$
\bar T(z) = (1-z) + \frac H2\, z(1-z),
$$

其中 $z$ 是高度， $z=0$ 在底、 $z=1$ 在顶。因此纯传导盒子的壁面 Nusselt 数是

$$
\mathrm{Nu}_{top} = 1 + \frac H2,
\qquad
\mathrm{Nu}_{bot} = 1 - \frac H2,
$$

`H = 10` 时分别为 6 和 −4。真实算例要改的是 `examples/case_realistic_2d.jl` 里的常量 `RA` 和 `HHEAT`，这两个都没有对应的环境变量。

## 怎么加一个相变？

先构造 `PhaseTransition(z0, width, slope, Tref, drho, B)` ，把需要的相变放进一个向量，再用 `phases=ph` 传给 `run_semquad_simulation` 。`z0` 是高度（0 在底、1 在顶），`slope` 是 $\mathrm dz/\mathrm dT$ ，浮力数按

$$
B = \frac{\Delta\rho}{\rho_0\alpha_0\Delta T}
$$

计算，其中 $\Delta\rho$ 是密度跳变， $\rho\_0$ 是参考密度， $\alpha\_0$ 是热膨胀系数， $\Delta T$ 是温度降。

潜热用 `latent_mode=:per_phase` 打开，系数由 `per_phase_latent(pr, ph; rho_ref=..., cp=..., dT=..., g=..., d=..., T_surface=...)` 算出。`examples/case_realistic_2d.jl` 的 `phases()` 里已经有一对配好的 410 与 660，可以直接抄。

## 怎么换流变定律？

在 `Params` 上指定 `eta_law` ：

| `eta_law` | 相关参数 | 说明 |
|---|---|---|
| `:expT` | `deta_T` | 只依赖温度的指数律。 |
| `:expTz` | `eta_b`、`eta_c` | 温度与高度联合的指数律。 |
| `:arrhenius` | `eta_E`、`eta_T0` | Arrhenius 形式。 |
| `:tosi` | `TosiLaw` | Tosi 等人提出的黏塑性定律。 |
| `:composite` | `comp_Ad`、`comp_As`、`comp_Ed`、`comp_Es`、`comp_n`、`comp_rule=:isostress` | ASPECT 使用的扩散与位错复合律。 |

参数卡没覆盖的定律可以传 `eta_custom=(x, z) -> ...` ，它在 Stokes 组装时接管温度定律。例如 2b 用的定律是

$$
\eta = \exp(-9.70406\,T + 4.15888\,(1-z)),
$$

其中 $T$ 是温度、 $z$ 是高度。真实算例的复合参数卡在 `examples/case_realistic_2d.jl` 的 `make_params()` 里。

## 能算三维吗？

不能。网格是二维四叉树，单元是二维 GLL 张量积单元，Stokes 求解器只装配两个速度分量和一个压力；仓库里没有三维实现，也没有做过三维测量。工程记录的性能卷第 8 节的三维列写的就是 "nothing measured" 。量级上也别类推： $p=2$ 的六面体叶有 27 个节点，二维只有 9 个，所以实测阈值不能搬到三维。

## 线程该开多少？

用 `julia --project=. --threads=8` ，并让 `linear=:auto`（默认）或 `linear=:mg` 保持原样。真正吃线程的是无矩阵多重网格， `:auto` 在生产规模会设 `mg_threads = min(8, Threads.nthreads())` 。八线程下，生产规模 Stokes 的热启动求解耗时 1.36 s，直接求解 3.37 s；完整求解相对单线程的加速是 1.30 到 1.42 倍，V 循环是 1.77 到 2.05 倍。稀疏直接路径是串行 UMFPACK，加线程不会变快；需要逐位可复现时固定用 `linear=:direct` 。

## 为什么大部分时间都花在 Stokes 求解上？

每一步都要解一个变黏度鞍点问题，而能量方程只是一个标量对流-扩散系统。生产参数卡在八线程上的热步耗时 2.72 s，其中 Stokes 2.21 s（81%）、能量组装与求解 0.31 s、功积分 0.14 s、诊断 0.04 s。想压低 Stokes 成本，可以给多重网格更大的叶数上限、在驱动的 `stokes_tol` 之内放松 `mg_tol` 、或者减小 `max_leaves` ；改能量方程的容差基本不会改变总时间。

## 网格为什么不再加密了？

先看 `frames/frame_meta.csv` ：`leaves` 列是否顶到 `CASE_MAXLEAVES`（默认 10000），`nlev` 列是否顶到 `CASE_MAXLEV`（默认 7），就知道撞的是哪个上限。`leaves` 到顶就调高 `CASE_MAXLEAVES`（10000 叶约 1.3 GB 内存），或者调低 `CASE_COARSEN` ，让平衡循环里的配额交换少释放一些单元；`nlev` 到 7 就调高 `CASE_MAXLEV` ，每加一级叶尺寸减半。两个上限都没到，再查 `CASE_WALLMIN=5` 与 `CASE_WALLSTRIP=0.1` 指定的壁面条带；另外 `freeze` 模式把 `amr_every` 设为 0，本来就不加密。

## 怎么重启？

用检查点。参数卡每 200 步写一次，退出时再写一次。`resume` 带着 AMR 接着跑，`freeze` 只把 `amr_every` 改成 0，在冻结的网格上继续：

```bash
CASE_OUT=out_julia/demo CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

`CASE_STEPS` 是绝对步号，必须大于检查点里的步号，否则驱动一步也不跑、什么都不写。之前用过的每个存储设置都要原样重设（`CASE_SYMMETRY`、`CASE_AMR_GATE`、`CASE_WALLMIN`、`CASE_LINEAR`）；不一致时重启守卫会中止，并把两边的值都打印出来。

## 怎么和 ASPECT 对照？

用 `aspect_runs/realistic/` 里的配置，以 `realistic_composite` 插件跑 ASPECT 2.5.0-pre，再按工程记录的验证卷 5.1 节的尺度组与流变映射做转换。先比 `Nu_top` ：在 $t\_{nd} = 10^{-3}$ 到 $5\times10^{-3}$ 区间内一致到 0.005 至 0.9 个百分点（ $t\_{nd}$ 为无量纲时间）。 `Nu_bot` 必须带时间窗口引用，因为 $10^{-6}$ 到 $10^{-3}$ 段的差是 −6.11%， $10^{-3}$ 到 $5\times10^{-3}$ 段是 +6.05%；不带窗口的均值连符号都是错的。若要做状态级检查，用 `examples/import_aspect_state.jl` 导入 ASPECT 检查点，两个算子在底壁一致到 0.003%。

## 怎么引用？

[`CITATION.cff`](../../CITATION.cff) 里有发布元数据：仓库地址、版本 0.1.0、MIT 许可。目前没有 DOI。引用时写明你克隆的仓库地址与代码提交，再加上所用算例的原始论文：2b 用 Blankenbach et al. (1989)，Tosi 的 Case 1 与 5a 用 Tosi et al. (2015)，跨代码对照用 ASPECT 论文。引用任何具体数字时，一并说明提交、Julia 版本和线程数——逐位可复现依赖这三者，而基准容差不依赖。
