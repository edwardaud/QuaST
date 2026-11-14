# QuaST

[English](README.md) · **中文**

二维地幔对流谱元求解器，网格是自适应加密的四叉树，Stokes 用无矩阵多重网格。

QuaST 求解不可压 Boussinesq 或滞弹性（anelastic）的 Stokes 与能量方程，单元是
$Q\_p/Q\_{p-1}$ Taylor–Hood，节点取 Gauss–Legendre–Lobatto 点。网格是保持 2:1 平衡的
四叉树，悬挂节点用**精确约束**消去而非罚函数，时间推进是 BDF2，网格按“误差对固定叶
子预算”的策略自适应。Stokes 鞍点可走稀疏直接分解，也可走无矩阵几何多重网格，由规模
与黏度对比自动选择。

这是一个只做二维的研究代码。它不做三维、板块边界、黏弹性或可压缩 MHD。

## 安装

Julia 1.12。

```bash
git clone git@github.com:edwardaud/QuaST.git
cd QuaST
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`instantiate` 需要五到十分钟、不到 2 GB。

## 第一次运行

真实算例跑十步，含编译约两分钟：

```bash
CASE_OUT=/tmp/quast_smoke julia --project=. examples/case_realistic_2d.jl smoke
```

**每次都要设 `CASE_OUT`**。不设时它写进 `out_julia/case_realistic`，那里可能存着你
在意的运行结果。

生产级 spin-up，八线程约两小时：

```bash
mkdir -p out_julia/run1
CASE_OUT=out_julia/run1 CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

进度写入 `run.log`，逐步诊断累积在 `history.csv`，`frames/` 存温度场供出图。

## 能力

- 四叉树自适应网格：平衡加密策略、2:1 合法性闸门、局部再平衡；壁面边界层可保持最小层级。
- $Q\_p$ 速度、 $Q\_{p-1}$ 压力，GLL 节点，悬挂节点用精确约束消去。
- 扩散–位错的复合流变、随深度变化的 Drucker–Prager 屈服与应变弱化、逐相变潜热、
  内部加热、Adams–Williamson 或 Murnaghan 参考态。
- 稀疏直接与无矩阵多重网格两条 Stokes 路径，由 `linear` 一个开关选择，`:auto` 按鞍点
  规模与黏度对比自动决定。
- 逐位可续算：同一检查点与同一配置重启，轨迹逐位重现。

## 验证

Blankenbach 2b 基准复现到印刷位数；blob 回归复现参考摘要（逐位）；Tosi 各例与复合流变
与 ASPECT 的本构卡一致；从同一初始条件在 ASPECT 中跑同一算例，顶端热流在窗口内一致到
百分之一以内。底端热流差 6–9%，随窗口而变；原因已有记录，但未完全分解。

## 文档

全部文档带中英切换，入口在 [docs/](docs/README.md)。

| 文档 | 内容 |
|---|---|
| [快速开始](docs/zh/quickstart.md) | 安装、第一次运行、停止与续算、新手会撞的失败 |
| [用户指南](docs/zh/guide.md) | 物理问题、数值方法、全部算例变量与关键字、输出格式 |
| [算例](docs/zh/examples.md) | 四个可直接跑的例子，附运行时间与应有的精度 |
| [验证](docs/zh/validation.md) | 基准、ASPECT 对照、两种 Nu 约定、验证边界 |
| [常见问题](docs/zh/faq.md) | 改 Rayleigh 数、加相变、线程、重启、引用 |
| [代码结构](docs/architecture.md) | 每个文件在哪、为什么 |
| [工程记录](docs/dev/README.md) | 推导、性能史、缺陷账本。维护者用 |

## 测试

```bash
julia --project=. test/runtests.jl                             # 全量，约 22 分钟
julia --project=. test/run_gates.jl t_g229_default_balanced.jl # 单个 gate
```

套件是 156 个 gate 文件、131 851 条断言，覆盖逐位回归钉、基准值、不变量与精度类别。
CI 跑全部。

## ASPECT 对照

`aspect_runs/realistic/` 里有 `.prm` 输入、把复合流变喂给 ASPECT 2.5 的小型 C++ 插件，
以及分析脚本。跑它需要 ASPECT 镜像，镜像不在本仓库。

## 许可

MIT，见 [LICENSE](LICENSE)。
