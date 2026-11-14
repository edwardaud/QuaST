# QuaST

[English](README.md) · **中文**

二维地幔对流谱元求解器。网格是自适应加密的四叉树，Stokes 鞍点可走稀疏直接分解，也可走无矩阵多重网格。

QuaST 求解不可压 Boussinesq 或滞弹性（anelastic）形式的 Stokes 方程与能量方程。单元是 $Q\_p/Q\_{p-1}$ Taylor–Hood，节点取 Gauss–Legendre–Lobatto 点。网格是保持 2:1 平衡的四叉树，悬挂节点用**精确约束**消去而非罚函数。时间推进用 BDF2，网格按“误差对固定叶数上限”的策略自适应。线性求解由规模与黏度对比自动选路。

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

**每次都要设 `CASE_OUT`**。不设的时候结果会写进 `out_julia/case_realistic`，那里可能存着你在意的运行数据。

生产规模的预积分，八线程约两小时：

```bash
mkdir -p out_julia/run1
CASE_OUT=out_julia/run1 CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup
```

进度写进 `run.log`，逐步诊断累积在 `history.csv`，`frames/` 存温度场供出图。

## 功能

- 四叉树自适应网格：平衡加密、2:1 合法性校验、局部再平衡；壁面边界层可强制保持最小层级。
- 速度取 $Q\_p$ 、压力取 $Q\_{p-1}$ ，GLL 节点，悬挂节点用精确约束消去。
- 扩散–位错复合流变、随深度变化的 Drucker–Prager 屈服与应变弱化、逐相变潜热、内部加热、Adams–Williamson 与 Murnaghan 两种参考态。
- 稀疏直接与无矩阵多重网格两条 Stokes 路径，由 `linear` 一个开关选择；`:auto` 按鞍点规模与黏度对比自动决定。
- 可逐位续算：同一检查点配同一套配置重启，轨迹逐位重现。

## 验证

已发表基准已在多个网格层级上跑完，结果列成收敛表而非单个数字。case 1a 中（ $Ra=10^4$ ），`Nu` 误差从 level 3 的 +1.08% 降到 level 4 的 +0.28%，再到 level 5 的 +0.08%；收敛比 3.9 与 3.5，二阶单元的理论值是 4。同一批运行里 `Vrms` 始终在 0.02% 以内。

2b 的壁面加密态把三种 `Nu` 定义（壁面梯度、稳态 CBF、能量族）收到 0.22% 的散布内，与发表值相差 0.04% 到 0.26%。

与 ASPECT 的对照分两层：同一初始条件下的跨代码比较，以及导入 ASPECT 检查点后的状态级比较——后者两个算子在底壁一致到 0.003%。底壁热流在瞬态窗口内的差随窗口变化（−6.11% 到 +6.05%），原因已有记录但未完全分解，[验证](docs/zh/validation.md)页如实写明了这一点。

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
