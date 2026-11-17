# QuaST

<p align="center">
  <b>二维地幔对流谱元求解器</b><br>
  自适应四叉树网格 · 无矩阵多重网格 Stokes 求解
</p>

<p align="center">
  <a href="docs/zh/README.md">文档</a> ·
  <a href="docs/zh/quickstart.md">快速开始</a> ·
  <a href="docs/zh/examples.md">算例</a> ·
  <a href="docs/zh/validation.md">验证</a> ·
  <a href="README.md">English</a>
</p>

<p align="center">
  <a href="https://github.com/edwardaud/QuaST/actions/workflows/gates.yml"><img alt="Gates" src="https://github.com/edwardaud/QuaST/actions/workflows/gates.yml/badge.svg"></a>
  <img alt="Julia" src="https://img.shields.io/badge/Julia-1.12-9558B2?logo=julia&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-131%20851%20assertions-brightgreen">
  <img alt="Docs" src="https://img.shields.io/badge/docs-English%20%7C%20中文-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

QuaST 求解不可压 Boussinesq 或滞弹性（anelastic）形式的 Stokes 方程与能量方程。单元是 $Q\_p/Q\_{p-1}$ Taylor–Hood，节点取 Gauss–Legendre–Lobatto 点。网格是保持 2:1 平衡的四叉树，悬挂节点用精确约束消去而非罚函数。时间推进用 BDF2，网格按“误差对固定叶数上限”的策略自适应。线性求解由规模与黏度对比自动选路。

这是一个只做二维的研究代码：没有三维，没有板块边界，没有黏弹性，也没有可压缩 MHD。

## 安装

需要 Julia 1.12。

```bash
git clone git@github.com:edwardaud/QuaST.git
cd QuaST
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`instantiate` 需要五到十分钟，占用不到 2 GB 磁盘。

## 第一次运行

真实算例跑十步，含编译约两分钟：

```bash
CASE_OUT=/tmp/quast_smoke julia --project=. examples/case_realistic_2d.jl smoke
```

> [!WARNING]
> 每次都要设 `CASE_OUT`。不设的时候结果会写进 `out_julia/case_realistic`，那里可能存着你在意的运行数据。

生产规模的预积分，八线程约两小时：

```bash
mkdir -p out_julia/run1
CASE_OUT=out_julia/run1 CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

进度写进 `run.log`，逐步诊断累积在 `history.csv`，`frames/` 存温度场供出图。

## 功能

| | |
|---|---|
| **网格** | 四叉树自适应：平衡加密、2:1 合法性校验、局部再平衡；壁面边界层可强制保持最小层级 |
| **离散** | 速度取 $Q\_p$ 、压力取 $Q\_{p-1}$ ，GLL 节点，悬挂节点用精确约束消去 |
| **物理** | 扩散–位错复合流变、随深度变化的 Drucker–Prager 屈服与应变弱化、逐相变潜热、内部加热、Adams–Williamson 与 Murnaghan 两种参考态 |
| **线性代数** | 稀疏直接与无矩阵多重网格两条 Stokes 路径，由 `linear` 一个开关选择；`:auto` 按鞍点规模与黏度对比自动决定 |
| **重启** | 逐位可复现：同一检查点配同一套配置重启，轨迹逐位重现 |

## 验证

**表 1.** 已发表基准与本求解器的对照。`我们` 一列是每个算例最好的收敛运行；`ASPECT` 一列是同
一个量在 ASPECT 随基准卡片发布的参考统计中的值（Blankenbach 各例），或 Tosi 等（2015）
Table 2 中 ASPECT 那一列。最后两列分别是相对发表值与相对 ASPECT 的差。破折号表示该算例没有这个量。

| 算例 | 量 | 发表值 | ASPECT | 我们 | 相对发表值 | 相对 ASPECT |
|:-----------------|:-------------|-------------:|-------------:|-------------:|-------------:|-------------:|
| Blankenbach 1a | `Nu` | 4.884409 | 4.884409 | **4.888163** | +0.08% | +0.08% |
| | `Vrms` | 42.864947 | 42.864956 | **42.8724** | +0.02% | +0.02% |
| Blankenbach 1b | `Nu` | 10.534095 | 10.533912 | **10.568610** | +0.33% | +0.33% |
| | `Vrms` | 193.21454 | 193.22175 | **193.2137** | −0.00% | −0.00% |
| Blankenbach 1c | `Nu` | 21.972465 | 21.972523 | **22.315960** | +1.56% | +1.56% |
| | `Vrms` | 833.98977 | 833.99083 | **834.0082** | +0.00% | +0.00% |
| Blankenbach 2a | `Nu` | 10.066 | 10.065793 | **10.124814** | +0.58% | +0.59% |
| | `Vrms` | 480.4334 | 480.43579 | **479.695** | −0.15% | −0.15% |
| Blankenbach 2b | `Nu` | 6.9299 | 6.929727 | **6.992620** | +0.91% | +0.91% |
| | `Vrms` | 171.755 | 171.75354 | **171.634** | −0.07% | −0.07% |
| Tosi 1 | `Nu_top` | 3.4298 | 3.4305 | **3.4216** | −0.24% | −0.26% |
| | `u_rms` | 251.80 | 251.3069 | **249.53** | −0.90% | −0.71% |
| Tosi 5a | 周期 | 0.0792 | — | **0.0783** | −1.14% | — |
| | `Nu_top` 最小 | 2.6950 | — | **2.6924** | −0.10% | — |
| | `Nu_top` 最大 | 7.2792 | — | **7.4889** | +2.88% | — |
| | 平均 `T` 最小 | 0.6543 | — | **0.6505** | −0.58% | — |
| | 平均 `T` 最大 | 0.6722 | — | **0.6686** | −0.54% | — |
| blob，`Ra = 1e8`，第 25 帧 | `Tmax` | — | 1.2531 | **1.2644** | — | +0.90% |
| 真实算例，导入 ASPECT 状态 | `Nu_top` | — | 5.990543 | **5.990543** | — | +0.000% |
| | `Nu_bot` | — | −3.075076 | **−3.074986** | — | +0.003% |

最后三行比的是 ASPECT 而不是发表值：blob 是 25 帧的瞬态；真实算例那两行从导入的 ASPECT
状态出发，差别只能来自离散——底壁一致到 0.003%。每一行背后的条件见[验证页](docs/zh/validation.md)。

![同一个算例分别由 QuaST 与 ASPECT 求解](docs/assets/quast_vs_aspect_realistic.gif)

**同一个算例，两个代码。** 这段动图把同一个算例跑了两遍：2.5 × 1 的真实算例卡片，含扩散–位错复合流变、两个带潜热的相变、内部加热与自适应网格。上排是 QuaST 的矩阵无关多重网格，下排是 ASPECT。两者从同一状态出发，覆盖 $t\_{nd} = 10^{-6}$ 到 $3.2 \times 10^{-4}$ ，色标在整段中固定不动。

从左到右依次是温度、相对水平平均的温度异常、对数黏度、以及按加密层级着色的网格。第三列值得停一下：这个算例的物理就在黏度里，而温度场看不出它。两个代码把硬盖、软流内部、以及羽流柱上的窄通道放在同样的位置；我们的通道更细——9 976 个叶子跟着通道走，ASPECT 是 27 460 个均匀单元。每个面板下方印的是该帧自己的读数。

动图是 120 帧渲染，取自我们 485 帧、ASPECT 940 帧。

## 文档

全部文档都带中英切换，入口在 [docs/](docs/README.md)。

| 文档 | 内容 |
|---|---|
| [快速开始](docs/zh/quickstart.md) | 安装、第一次运行、中途停下与续算、新手常踩的坑 |
| [用户指南](docs/zh/guide.md) | 方程与边界条件、离散与线性求解、全部可调项、输出格式 |
| [算例](docs/zh/examples.md) | 四个可直接运行的算例，附实测墙钟与供核对的数值 |
| [验证](docs/zh/validation.md) | 基准收敛结果、ASPECT 对照、Nusselt 数的两种约定 |
| [常见问题](docs/zh/faq.md) | 改 Rayleigh 数、加相变、换流变、线程、重启、引用 |
| [代码结构](docs/architecture.md) | 每个文件放在哪里、为什么 |
| [工程记录](docs/dev/README.md) | 推导、性能史、缺陷账本，供维护者查阅 |

## 测试

```bash
julia --project=. test/runtests.jl                             # 全量，约 22 分钟
julia --project=. test/run_gates.jl t_g229_default_balanced.jl # 单个 gate
```

套件包含 156 个 gate 文件、131 851 条断言，覆盖逐位回归、基准值、不变量与精度类别。CI 跑全量。

## 与 ASPECT 对照

`aspect_runs/realistic/` 里放着 `.prm` 输入、把复合流变接入 ASPECT 2.5 的小型 C++ 插件，以及分析脚本。运行它们需要 ASPECT 镜像，镜像不在本仓库内。

## 许可

MIT，见 [LICENSE](LICENSE)。
