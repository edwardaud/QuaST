<p align="right"><a href="../en/faq.md">English</a> · <a href="../zh/faq.md">中文</a></p>

# 常见问题

这些问题通常出现在读完[快速开始](quickstart.md)之后。每个答案只给结论和出处；公式、默认值与完整说明都在指向的页面里，不在这里重复。

## 怎么改 Rayleigh 数和内部加热？

库调用里传 `Params(Ra=..., H=...)`：`Ra` 是 Rayleigh 数，`H` 是体积内部加热。真实算例要改的是 `examples/case_realistic_2d.jl` 里的常量 `RA` 和 `HHEAT` 。传导基态与 Nusselt 数的定义见[用户指南](guide.md) 1.3 节。

## 为什么底部 Nusselt 数是负的？

符号来自传导基态的梯度：内部加热足够强时，纯传导的底壁 Nusselt 数就是负的（本卡的具体数值见[用户指南](guide.md) 1.3 节）。所以负值不是错误。引用底壁通量时必须带上时间窗口，因为与 ASPECT 的底壁之差会随时间窗口变号，见[验证](validation.md)的跨代码对照一节。

## 怎么加一个相变？

先构造 `PhaseTransition(z0, width, slope, Tref, drho, B)` ，把需要的相变放进一个向量，再用 `phases=ph` 传给 `run_semquad_simulation` 。 `z0` 是高度（0 在底、1 在顶），`slope` 是 Clapeyron 斜率 $\mathrm dz/\mathrm dT$ ，浮力数按指南 1.5 节的式子算。潜热用 `latent_mode=:per_phase` 打开，系数由 `per_phase_latent(pr, ph; rho_ref=..., cp=..., dT=..., g=..., d=..., T_surface=...)` 算出。 `examples/case_realistic_2d.jl` 的 `phases()` 里已经有一对配好的 410 与 660，可以直接抄。

## 怎么换流变定律？

在 `Params` 上指定 `eta_law` ：

| `eta_law` | 相关参数 | 说明 |
|---|---|---|
| `:expT` | `deta_T` | 只依赖温度的指数律。 |
| `:expTz` | `eta_b`、`eta_c` | 温度与高度联合的指数律。 |
| `:arrhenius` | `eta_E`、`eta_T0` | Arrhenius 形式。 |
| `:tosi` | `TosiLaw` | Tosi 等人提出的黏塑性定律。 |
| `:composite` | `comp_Ad`、`comp_As`、`comp_Ed`、`comp_Es`、`comp_n`、`comp_rule=:isostress` | ASPECT 使用的扩散与位错复合律。 |

参数卡没覆盖的定律可以传 `eta_custom=(x, z) -> ...` ，它在 Stokes 组装时接管温度定律，例如 2b 基准用的温度-深度指数律（见[算例](examples.md)的 Blankenbach 2b 一节）。真实算例的复合参数卡在 `examples/case_realistic_2d.jl` 的 `make_params()` 里。

## eta_ref 是什么，什么时候要改？

`eta_ref` 就是 `Params.eta_ref` ，Rayleigh 数定义所用的参考黏度，默认值 `1.0` ，单位是材料定律返回值的单位。真实算例默认保持 `1.0` 即可：它的标定就是按 `Ra = 1e5` 与 `eta_ref = 1` 定义的。要改的情况是自己缩放了材料律：定律整体乘 $c$ 时，同一个物理问题需要 `eta_ref = c` ；上下限也要跟着缩放，因为 `eta_min` 与 `eta_max` 是以定律单位给出的绝对量。完整约定与报告内容见[用户指南](guide.md) 1.9 节。

## 为什么报告的 Ra 不是内部 Rayleigh 数？

`Ra` 用单一的参考黏度 `eta_ref` 归一化，而材料的实际黏度在盒内跨越多个数量级，所以名义 `Ra` 描述的是标定点，不是盒子内部。内部 Rayleigh 数由 `Ra_interior = Ra/eta_interior` 给出，具体数值见[用户指南](guide.md) 1.9 节。引用时把名义值和 `Ra_interior` 一起给出，不要把它当作内部 Rayleigh 数。

## 能算三维吗？

不能。网格是二维四叉树，单元是二维 GLL 张量积单元，Stokes 求解器只装配两个速度分量和一个压力；仓库里没有三维实现，也没有做过三维测量。量级上也别类推： $p=2$ 的六面体叶有 27 个节点，二维只有 9 个，所以实测阈值不能搬到三维。

## 线程该开多少？

用 `julia --project=. --threads=8` ，保持 `linear=:auto`（默认）即可。真正吃线程的是多重网格 V 循环， `:auto` 在生产规模会设 `mg_threads = min(8, Threads.nthreads())` ；直接路径是串行 UMFPACK，加线程不会变快。实测加速见[用户指南](guide.md)第 6 节；需要逐位可复现时固定用 `linear=:direct` 。

## 为什么大部分时间都花在 Stokes 求解上？

每一步都要解一个变黏度鞍点问题，而能量方程只是一个标量对流-扩散系统。生产参数卡在八线程上的热步耗时 2.72 s，其中 Stokes 2.21 s（81%）、能量组装与求解 0.31 s、功积分 0.14 s、诊断 0.04 s。想压低 Stokes 成本，可以给多重网格更大的叶数上限、在驱动的 `stokes_tol` 之内放松 `mg_tol` 、或者减小 `max_leaves` ；改能量方程的容差基本不会改变总时间。

## 为什么黏弹运行可能比它的 shear_modulus = 0 孪生运行更便宜？

因为 `G` 改变的是问题本身，而不只是增加开销：有效黏度被 $G\Delta t$ 从上方封顶，远高于它的材料黏度都被压下来，求解器看到的对比度因此变小。黏弹卡是一个不同的问题，它的诊断量不是它 `G = 0` 孪生卡的；实测数字与判读见[用户指南](guide.md) 1.8 节。

## 网格为什么不再加密了？

先看 `frames/frame_meta.csv` ：`leaves` 列是否顶到 `CASE_MAXLEAVES` ，`nlev` 列是否顶到 `CASE_MAXLEV` ，就知道撞的是哪个上限（默认值见[用户指南](guide.md) 3.1 节）。`leaves` 到顶就调高 `CASE_MAXLEAVES` ，或者调低 `CASE_COARSEN` ，让平衡循环里的配额交换少释放一些单元；`nlev` 到顶就调高 `CASE_MAXLEV` ，每加一级叶尺寸减半。两个上限都没到，再查 `CASE_WALLMIN` 与 `CASE_WALLSTRIP` 指定的壁面条带；另外 `freeze` 模式把 `amr_every` 设为 0，本来就不加密。平衡循环的机制见指南 2.1 节。

## 该选哪个加密指示子？

`amr_indicator` 选择加密指示子作用的场，参数卡用 `CASE_INDICATOR` 设置： `:theta`（默认）、`:velocity`，或 `:both` 。一组实测对比与它们的适用条件见[用户指南](guide.md) 2.1 节：哪个更好取决于观测量，而这个比较只覆盖一个状态、一个 Rayleigh 数、43 步和二维；没有理由时保持默认。

## history.csv 里的 closure 是什么？

它是历史列 `closure` 的公式在瞬态状态下与稳态通量平衡的偏离，读作焓的变化率加记账项，不是热流误差；状态还在演化时，没有任何 Nusselt 定义能让它闭合。公式、各列含义与 `CASE_EXTDIAG=1` 追加的 `resid_A` 见[用户指南](guide.md) 4.2 节。

## 怎么重启？

用检查点续算。 `resume` 带着 AMR 接着跑，`freeze` 只把 `amr_every` 改成 0，在冻结的网格上继续；影响轨迹的设置要与检查点一致，否则重启守卫会中止。命令与处理步骤见[快速开始](quickstart.md)的「停下与续算」，守卫比较的范围见[用户指南](guide.md)第 5 节。

## 怎么和 ASPECT 对照？

用 `aspect_runs/realistic/` 里的配置，以 `realistic_composite` 插件跑 ASPECT 2.5.0-pre。顶壁通量的对照窗口、底壁通量随时间窗口变号的数据，以及状态级检验的差异量都在[验证](validation.md)的跨代码对照一节。若要做状态级检查，用 `examples/import_aspect_state.jl` 导入 ASPECT 检查点，两个算子在底壁一致到 0.003%。

## 怎么引用？

[`CITATION.cff`](../../CITATION.cff) 里有发布元数据：仓库地址、版本 0.1.0、MIT 许可。目前没有 DOI。引用时写明你克隆的仓库地址与代码提交，再加上所用算例的原始论文：2b 用 Blankenbach et al. (1989)，Tosi 的 Case 1 与 5a 用 Tosi et al. (2015)，跨代码对照用 ASPECT 论文。引用任何具体数字时，一并说明提交、Julia 版本和线程数——逐位可复现依赖这三者，而基准容差不依赖。

下一步：全部文档的索引见[中文文档](README.md)。
