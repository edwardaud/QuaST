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

## 为什么底部 Nusselt 数是负的？

符号来自底态的梯度。由于 `Nu_bot = (1 - H/2) - <dtheta/dy>_bot` 且 `H = 10`，传导值就是 `1 - 5 = -4`。这么强的内部加热让底态在底部向下输运热量（该处 `dTbar/dy = +4`），所以负的底部 Nusselt 数是本算例的常态；第 20 000 步端点是 `Nu_bot = -2.7636`。引用时务必带上时间窗口，因为与 ASPECT 的底部通量之差在 `t_nd = 1.42e-3` 变号（见[验证](validation.md)）。

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

## `eta_ref` 是什么，什么时候要改？

`eta_ref` 就是 `Params.eta_ref` ，Rayleigh 数定义所用的参考黏度：

$$
\mathrm{Ra} = \frac{\rho g \alpha \Delta T d^3}{\kappa\,\eta_{\mathrm{ref}}},
$$

默认值 `1.0` ，单位是材料定律返回值的单位；真实算例通过 `CASE_ETA_REF` 暴露它，默认值同样是 `1.0` 。流场求解中凡出现定律黏度的地方都用 `eta_law/eta_ref` 。真实算例保持默认即可：它的标定就是按 `Ra = 1e5` 与 `eta_ref = 1` 定义的。要改的情况是自己缩放了定律：把 `comp_Ad` 与 `comp_As` 同乘 `1/c` 会让每个黏度都乘 $c$ ，同一个物理问题需要 `eta_ref = c` ；若仍取 `eta_ref = 1` ，名义 `Ra` 不变而物理 Rayleigh 数变了。上下限也要跟着缩放，因为 `eta_min` 与 `eta_max` 是以定律单位给出的绝对量。取非 1 值时驱动会在第一次流动求解和运行结束时各报告一次该约定； `pic_mode = :newton` 会拒绝它，因为它直接用原始定律构造黏度。

## 为什么报告的 `Ra` 不是内部 Rayleigh 数？

`Ra` 是用单一的参考黏度 `eta_ref` 归一化的，而材料的实际黏度在盒内跨越多个数量级。取非 1 值的运行因此还会报告 `eta_interior` （在 `0.25 <= z <= 0.75` 内对 `eta/eta_ref` 取的算术体平均）和 `Ra_interior = Ra/eta_interior` 。在生产参数卡的成熟状态上，这些量是 `eta_interior = 4.280635e-3` 与 `Ra_interior = 2.336102e7` ，所以该卡的名义 `Ra = 1e5` 与内部 Rayleigh 数是两个不同的数； `eta_0 = 2.368635e23 Pa s` 对应物理内部黏度 `1.013926e21 Pa s` 。引用时把名义值和 `Ra_interior` 一起给出，不要把它当作内部 Rayleigh 数。

## 能算三维吗？

不能。网格是二维四叉树，单元是二维 GLL 张量积单元，Stokes 求解器只装配两个速度分量和一个压力；仓库里没有三维实现，也没有做过三维测量。工程记录的性能卷第 8 节的三维列写的就是 "nothing measured" 。量级上也别类推： $p=2$ 的六面体叶有 27 个节点，二维只有 9 个，所以实测阈值不能搬到三维。

## 线程该开多少？

用 `julia --project=. --threads=8` ，并让 `linear=:auto`（默认）或 `linear=:mg` 保持原样。真正吃线程的是无矩阵多重网格， `:auto` 在生产规模会设 `mg_threads = min(8, Threads.nthreads())` 。八线程下，生产规模 Stokes 的热启动求解耗时 1.36 s，直接求解 3.37 s；完整求解相对单线程的加速是 1.30 到 1.42 倍，V 循环是 1.77 到 2.05 倍。稀疏直接路径是串行 UMFPACK，加线程不会变快；需要逐位可复现时固定用 `linear=:direct` 。

## 为什么大部分时间都花在 Stokes 求解上？

每一步都要解一个变黏度鞍点问题，而能量方程只是一个标量对流-扩散系统。生产参数卡在八线程上的热步耗时 2.72 s，其中 Stokes 2.21 s（81%）、能量组装与求解 0.31 s、功积分 0.14 s、诊断 0.04 s。想压低 Stokes 成本，可以给多重网格更大的叶数上限、在驱动的 `stokes_tol` 之内放松 `mg_tol` 、或者减小 `max_leaves` ；改能量方程的容差基本不会改变总时间。

## 为什么黏弹运行可能比它的 `shear_modulus = 0` 孪生运行更便宜？

因为 `G` 改变的是问题本身，而不只是增加开销。有效黏度是 `eta_eff = 1/(1/eta + 1/(G*dt))` ，所以凡是远高于 `G*dt` 的黏度都被压到 `G*dt` ：在生产状态（`G*dt = 1.485`），材料对比 `3.3e6` 在求解器眼里只有约 `5e2` ，这正是黏弹运行可能比它的孪生运行便宜的原因，尽管黏弹机制本身只占一步的约 1%。黏弹卡是一个不同的问题，它的诊断量不是它 `G = 0` 孪生卡的；解释黏弹运行之前先记下预期的有效对比。

## 网格为什么不再加密了？

先看 `frames/frame_meta.csv` ：`leaves` 列是否顶到 `CASE_MAXLEAVES`（默认 10000），`nlev` 列是否顶到 `CASE_MAXLEV`（默认 7），就知道撞的是哪个上限。`leaves` 到顶就调高 `CASE_MAXLEAVES`（10000 叶约 1.3 GB 内存），或者调低 `CASE_COARSEN` ，让平衡循环里的配额交换少释放一些单元；`nlev` 到 7 就调高 `CASE_MAXLEV` ，每加一级叶尺寸减半。两个上限都没到，再查 `CASE_WALLMIN=5` 与 `CASE_WALLSTRIP=0.1` 指定的壁面条带；另外 `freeze` 模式把 `amr_every` 设为 0，本来就不加密。

## 该选哪个加密指示子？

`amr_indicator` 选择加密指示子作用的场，参数卡用 `CASE_INDICATOR` 设置： `:theta`（默认）、`:velocity`，或 `:both` ——后者把两个指示子各自按总量归一化后相加，两个场都不会主导。在一个成熟的二维状态上、相同成本下（10 000 叶、43 步、同样的固定 `dt`，参照解为 40 000 叶的 `:theta` 解），`:velocity` 的 `Nu_bot` 误差更小（1.01e-5 对 1.10e-4），体积平均温度误差也更小（1.15e-4 对 1.75e-4）；而 `:theta` 的精确壁面通量残差 `resid_A/Nu_top` 更小（约 7e-4% 对 5e-3%）。`:theta` 把 85% 的细叶放在壁面条带，`:velocity` 只放 3.6%，转而加密内部。所以哪个更好取决于观测量，而这个比较只覆盖一个状态、一个 Rayleigh 数、43 步（时间轴的 0.2%）和二维；没有理由时保持默认，并把上面的数字看作该状态的属性而不是排名。

## `history.csv` 里的 `closure` 是什么？

`closure` 等于 `Nu_top - Nu_bot - (H - Ibase)` 再加上潜热项对面积的体积分的时间导数，其中 `Ibase = int u_y dTbar/dy`。它衡量两侧壁面通量离定常平衡有多远，因此在瞬变态它等于焓的储存速率加若干记账项。状态还在演化时，没有任何 Nusselt 定义能让它闭合，所以运行早期的值量级为 `H` 属正常；应把它读作焓的变化率。设置 `CASE_EXTDIAG=1` 会额外输出 `resid_A`，即所引用的逐点对这一组恒等式的精确残差。

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
