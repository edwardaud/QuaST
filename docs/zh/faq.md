<p align="right"><a href="../en/faq.md">English</a> · <a href="../zh/faq.md">中文</a></p>

# 常见问题

## 怎么改 Rayleigh 数和加热？

库调用传 `Params(Ra=..., H=...)`：`Ra` 是 Rayleigh 数，`H` 是体积内部加热。传导基态是

$$
\bar T(z) = (1-z) + \frac H2\, z(1-z),
$$

其中 $z$ 是高度， $z=0$ 在底部、 $z=1$ 在顶部。所以传导盒的壁面 Nusselt 数是

$$
\mathrm{Nu}_{top} = 1 + \frac H2,
\qquad
\mathrm{Nu}_{bot} = 1 - \frac H2,
$$

在 `H = 10` 时分别是 6 和 -4。真实算例改 `examples/case_realistic_2d.jl` 里的常量 `RA` 和 `HHEAT`；两者都没有环境变量。

## 怎么加一个相变？

构造 `PhaseTransition(z0, width, slope, Tref, drho, B)`，放进向量，作为 `phases=ph` 传给 `run_semquad_simulation`。`z0` 是高度（0 底、1 顶），`slope` 是 $\mathrm dz/\mathrm dT$ ，浮力数是

$$
B = \frac{\Delta\rho}{\rho_0\alpha_0\Delta T},
$$

其中 $\Delta\rho$ 是密度跳变， $\rho_0$ 是参考密度， $\alpha_0$ 是热膨胀系数， $\Delta T$ 是温度降。

潜热用 `latent_mode=:per_phase`，系数用 `per_phase_latent(pr, ph; rho_ref=..., cp=..., dT=..., g=..., d=..., T_surface=...)` 填。现成的一对在 `examples/case_realistic_2d.jl` 的 `phases()` 里，可以直接抄。

## 怎么换流变定律？

在 `Params` 上选 `eta_law`：

| `eta_law` | 参数 | 说明 |
|---|---|---|
| `:expT` | `deta_T` | 温度比的指数律。 |
| `:expTz` | `eta_b`、`eta_c` | 温度加高度的联合指数律。 |
| `:arrhenius` | `eta_E`、`eta_T0` | Arrhenius 形式。 |
| `:tosi` | `TosiLaw` | Tosi 等人的定律。 |
| `:composite` | `comp_Ad`、`comp_As`、`comp_Ed`、`comp_Es`、`comp_n`、`comp_rule=:isostress` | ASPECT 的扩散加位错复合律。 |

参数卡没有的定律，传 `eta_custom=(x, z) -> ...`，它在 Stokes 组装里覆盖温度定律。2b 的定律是

$$
\eta = \exp(-9.70406\,T + 4.15888\,(1-z)),
$$

其中 $T$ 是温度、 $z$ 是高度。真实算例的复合卡在 `examples/case_realistic_2d.jl` 的 `make_params()` 里。

## 能跑三维吗？

不能。网格是二维四叉树，单元是二维 GLL 张量单元，Stokes 求解器只装两个速度分量和一个压力；仓库没有三维路径，也没有三维测量。[工程记录的性能卷](../dev/PERFORMANCE.md)第 8 节的三维列明确写着 "nothing measured"； $p=2$ 的六面体叶有 27 个节点，二维只有 9 个，所以测得的阈值不能搬过去。

## 应该开多少线程？

用 `julia --project=. --threads=8`，让 `linear=:auto`（默认）或 `linear=:mg` 保持原样。无矩阵多重网格是走线程的路径，`:auto` 在生产规模设 `mg_threads = min(8, Threads.nthreads())`。八线程下生产 Stokes 的热启动求解是 1.36 s，直接求解是 3.37 s；完整求解相对单线程的增益是 1.30 到 1.42 倍，V 循环是 1.77 到 2.05 倍。稀疏直接路径是串行 UMFPACK，加线程不会更快；需要逐位路径时固定 `linear=:direct`。

## 为什么墙钟主要花在 Stokes 求解上？

每一步都要解一个变黏度鞍点，能量方程只是标量对流-扩散系统。生产卡在八线程上的热步是 2.72 s：Stokes 2.21 s（81%），能量组装与求解 0.31 s，功积分 0.14 s，诊断 0.04 s。降低 Stokes 成本靠更大的多重网格叶预算、在驱动的 `stokes_tol` 内放松 `mg_tol`、或减小 `max_leaves`；改能量容差不会明显改变总时间。

## 网格不再加密怎么办？

读 `frames/frame_meta.csv`：`leaves` 列对 `CASE_MAXLEAVES`（默认 10000），`nlev` 列对 `CASE_MAXLEV`（默认 7），就知道撞了哪个上限。`leaves` 到顶就提高 `CASE_MAXLEAVES`（10000 叶约 1.3 GB 内存），或降低 `CASE_COARSEN`，让平衡预算交换少释放一些单元；`nlev` 到 7 就提高 `CASE_MAXLEV`，每升一级叶尺寸减半。两个都没到顶，就查 `CASE_WALLMIN=5` 和 `CASE_WALLSTRIP=0.1` 的壁面条带；`freeze` 运行的 `amr_every=0`，按设计不加密。

## 怎么重启？

用检查点；参数卡每 200 步和退出时各写一个。`resume` 带 AMR 继续，`freeze` 只把 `amr_every` 改写成 0，在冻结网格上继续：

```bash
CASE_OUT=out_julia/demo CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  out_julia/demo/spinup.h5
```

`CASE_STEPS` 是绝对步号，必须大于检查点的步号，否则驱动跑零步、什么都不写。重复运行用过的每个存储设置（`CASE_SYMMETRY`、`CASE_AMR_GATE`、`CASE_WALLMIN`、`CASE_LINEAR`）；不匹配时重启守卫中止并打印两边的值。

## 怎么和 ASPECT 对照？

用 `aspect_runs/realistic/`，以 `realistic_composite` 插件跑 ASPECT 2.5.0-pre，再按[工程记录的验证卷](../dev/VALIDATION.md) 5.1 节的尺度组和流变映射转换。先比 `Nu_top`：在 $t_{nd} = 10^{-3}$ 到 $5\times10^{-3}$ 内一致到 0.005 到 0.9 个百分点，其中 $t_{nd}$ 是无量纲时间。`Nu_bot` 必须带窗口引用，因为 $10^{-6}$ 到 $10^{-3}$ 的差是 -6.11%， $10^{-3}$ 到 $5\times10^{-3}$ 是 +6.05%；不带窗口的均值在大小和符号上都错。要做状态级检查，用 `examples/import_aspect_state.jl` 导入 ASPECT 检查点；两个算子在底壁一致到 0.003%。

## 怎么引用？

[`CITATION.cff`](../../CITATION.cff) 带发布元数据：仓库 URL、版本 0.1.0、MIT 许可。没有 DOI。引用你克隆的仓库 URL 和代码提交，再加上所用算例的基准论文：2b 用 Blankenbach et al. (1989)，Tosi 的 Case 1 和 5a 用 Tosi et al. (2015)，跨代码对照用 ASPECT 论文。引用任何数字时都写明提交、Julia 版本和线程数；逐位钉依赖这三者，基准容差不依赖。
