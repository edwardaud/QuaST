<p align="right"><a href="../en/quickstart.md">English</a> · <a href="../zh/quickstart.md">中文</a></p>

# 快速开始

本页从零跑到第一次正式运行：装 Julia、跑十步冒烟、跑两千步预积分、中途停下再续算。算例是 `examples/case_realistic_2d.jl`，物理设定见[用户指南](guide.md)第 1 节；每个算例应打印的数字见[算例](examples.md)。命令都在仓库根目录执行。

## 安装

本项目需要 Julia 1.12。 `Project.toml` 里写的是 `julia = "1.10"`，因此 1.10 以后的版本也能用；本页的命令在 1.12 上验证过。

```bash
cd /path/to/QuaST
julia --version                      # 应为 1.12.x
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using QuaST; println("ok")'
```

`--project=.` 让后面的命令都使用当前目录的工程。 `Pkg.instantiate()` 按 `Manifest.toml` 装好依赖，并下载 HDF5 的二进制包。最后一条命令会编译一次求解器并打印 `ok`，之后调用就能立即返回。

预留 5 到 10 分钟和不到 2 GB 磁盘空间：Julia 本体约 1 GB， `~/.julia` 下的包与预编译缓存约 0.5 GB。

## 十步冒烟

```bash
CASE_OUT=/tmp/quast_smoke julia --project=. examples/case_realistic_2d.jl smoke
```

第一次运行约两分钟，时间基本花在编译上；再跑一次只要几秒。 `smoke` 从参数卡自带的初值出发走十步，在第 5、10 步调整网格，在第 1、6、10 步各存一帧。输出先是四行头部信息，然后每步一行，每个网格周期一行 AMR，最后两行收尾：

```text
[case] mode=smoke Ra=1.0e+05 H=10.0 p=2 max_level=7 max_leaves=10000 amr_every=5 steps=10 ...
[case] comp_Ad=1.6240183119e+05 comp_As=2.6796419238e-01 Ed=12.0 Es=17.0 n=3.0 T0=0.50 ...
[case] phases: z0=0.87 w=0.025 slope=+0.032 B=1.8182 | z0=0.78 ... latent_mode=per_phase
[case] latent coefficients: 410 -0.059364 660 +0.130596 (negative = heats a downwelling)
  step    1 t=... dt=... Nu=.../... Vrms=... N=... leaves=... closure=...
    AMR: N=... leaves=... maxjump=<j> refined=... coarsened=... merged=...
[case] DONE mode=smoke steps=10 t=... wall=... s (... s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=... Nu_bot=... Vrms=... meanT=... closure=... leaves=... N=...
```

数字会随步数变化，行的格式不变。两个 Nusselt 数从传导值出发，十步之内都停在附近；定义与传导值见[用户指南](guide.md) 1.3 节，用作核对的数值见[算例](examples.md#真实算例)。

`smoke` 不写检查点，所以输出目录里只有这些：

```text
/tmp/quast_smoke/history.csv                  表头加 10 行
/tmp/quast_smoke/frames/                      第 1、6、10 步各一帧
/tmp/quast_smoke/frames/frame_meta.csv        3 行加一个表头
```

每个帧步在 `frames/` 下写 `frame_NNNNN_T.bin`（512 KB）、`frame_NNNNN_mesh.csv`，以及节点场文件 `frame_NNNNN_eta.bin`、`frame_NNNNN_speed.bin`、`frame_NNNNN_nodes.bin`；各类文件的格式见[用户指南](guide.md) 6.3 节。 `wc -l /tmp/quast_smoke/history.csv` 得到 11，`ls /tmp/quast_smoke/frames/*_T.bin | wc -l` 得到 3。本卡写出 19 列，`history.csv` 的列含义见指南 6.2 节。

## 第一次正式运行

用默认参数卡跑两千步预积分，开八个线程：

```bash
mkdir -p "$HOME/quast/run1"
CASE_OUT="$HOME/quast/run1" CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup \
  > "$HOME/quast/run1/run.log" 2>&1
```

`mkdir` 要放在前面：shell 打开 `run.log` 时目录必须已经存在，程序自己也会 `mkpath(CASE_OUT)` ，但已经晚了一步。 `spinup` 每 200 步写一个检查点到 `$CASE_OUT/spinup.h5` ，结束时再写一个；每 5 步存一帧，末尾补一帧。

八线程下大约 1.5 到 2.5 小时。从零开始的每一步，在 `CASE_LINEAR=auto` 为这个规模选中的多重网格路径上约 2.8 秒，`CASE_LINEAR=direct` 约 4.4 秒。两千步大约占 0.3 GB 帧文件（401 个），加一个 2 MB 的检查点、两千行 `history.csv` 和一份日志，磁盘几乎全被帧吃掉。

### 看运行状态

`run.log` 是实时窗口。里面有求解器逐步输出、每 5 步一条 AMR 记录、每 100 步一行参数卡回调的进度：

```text
  step <n> t=<t> dt=<dt> Nu=<Nu_top>/<Nu_bot> Vrms=<Vrms> N=<nodes> leaves=<leaves> closure=<closure>
    AMR: N=<nodes> leaves=<leaves> maxjump=<j> refined=<r> coarsened=<c> merged=<m>
  [case] step=<n> t=<t> leaves=<leaves> N=<nodes> auto_fallbacks=<count>
```

回调会主动 flush，所以重定向出来的日志最多滞后 100 步。

**`auto_fallbacks` 上升不是故障。** 每次回退只是把一次 Stokes 求解换成直接法，运行照常继续。

`frames/frame_meta.csv` 是第二个实时窗口，每存完一帧就 flush，所以每 5 步前进一次，列含义见[用户指南](guide.md) 6.3 节。 `history.csv` 只在运行结束后写一次，运行期间不存在；列含义与 `closure` 的读法见指南 6.2 节。

## 停下与续算

按 Ctrl-C 或直接杀掉进程。最多丢 199 步，也就是最后一次检查点之后的步数，因为参数卡每 200 步检查点一次。

```bash
CASE_OUT="$HOME/quast/run1" CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  "$HOME/quast/run1/spinup.h5" >> "$HOME/quast/run1/run.log" 2>&1
```

有三点要注意：

- `CASE_STEPS` 是绝对步号，不是增量。续算时若 `CASE_STEPS` 不大于检查点里的步号，程序一步也不跑，打印 `[case] DONE mode=resume steps=... (no steps were run: maxsteps=... <= checkpoint step); nothing written` 后退出。
- `resume` 把结果写到 `$CASE_OUT/resume.h5` 。所以下一次续算要么显式指定这个路径，要么又回到旧的 `spinup.h5` 。不带参数时它读 `$CASE_OUT/spinup.h5` 。
- 续算要求当前设置与检查点一致。常规续算不必再设任何变量，因为脚本默认值与检查点相同；影响轨迹的键（AMR、壁面规则、对称、物理参数等）如果之前改过，这次必须设成同样的值，否则被配置守卫拒绝； `CASE_LINEAR` 、步长初值、检查点节奏这类求解器设置不在比较之列，可以更换。比较范围见[用户指南](guide.md)第 7 节，报错的处理办法见本页「出问题怎么办」。

**`CASE_OUT` 一定要设，而且用绝对路径。** 默认值是仓库内的 `out_julia/case_realistic` ，可能残留着更早一次预积分的文件：写进去会覆盖 `history.csv` 、覆盖步号相同的帧、向 `frame_meta.csv` 追加行。建议每次运行用一个独立目录，冒烟和测试用临时路径。

## 三个必须弄明白的旋钮

| 旋钮 | 默认值 | 作用 |
|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | 输出根目录。每次调用都给，且用绝对路径；运行产生的所有文件都在它下面。 |
| `CASE_STEPS` | 0（按模式取默认，各模式的步数见指南 5.1 节） | 步数上限，与绝对步号比较，所以续算时它就是停下的那一步。 |
| `CASE_LINEAR` | `auto` | 线性求解器。默认的 `auto` 按鞍点规模与黏度对比自动选择多重网格或直接分解，规则见指南 4.3 节。 `direct` 强制走直接路径，也就是逐位可复现的那条，墙钟和内存都约翻倍。 `mg` 强制走无矩阵多重网格，且只能串行，因为参数卡没有传 `mg_threads` ；只有 `auto` 会用到多线程 V 循环。 |

其余 `CASE_*` 变量与 `run_semquad_simulation` 的关键字见[用户指南](guide.md)第 5 节。定义 `Ra` 的参考黏度是 `CASE_ETA_REF`（默认 `1.0`），约定见指南 1.9 节。

## 出问题怎么办

### 内存不够

现象是 Julia 抛 `ERROR: OutOfMemoryError()` 并附栈回溯，或者进程直接消失、`run.log` 断在半截。WSL 下的 OOM killer 会用退出码 137 结束进程，不留下任何消息。

默认参数卡在 10 000 叶的上限下，峰值常驻内存见[用户指南](guide.md)第 8 节。优先保留 `CASE_LINEAR=auto` ，或改用 `CASE_LINEAR=mg` ，后者在这个规模上只占直接分解一半左右的内存；也可以把上限调低，例如 `CASE_MAXLEAVES=5000` 或 `CASE_MAXLEV=6` ；启动前先看一眼 `free -g` 。WSL 虚拟机的内存上限在 Windows 侧的 `.wslconfig` 里调。网格相关的键属于重启配置，所以缩小预算只能新开一次运行，不能续算。

### 重启被配置守卫拒绝

驱动会把检查点里的设置与当前运行逐键比对，只要有一项不同就拒绝继续。错误信息会给出文件名和每个不一致的键：

```text
ERROR: ArgumentError: restart configuration mismatch in /home/you/quast/run1/spinup.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

值是该设置的规范字符串：浮点写成 `5.0` ，符号写成 `x` 或 `none` 。把这些键按检查点里的值重设一遍，或者用下面的命令读回来：

```bash
julia --project=. -e 'using HDF5; h5open(ARGS[1]) do f; println(HDF5.attrs(f)["config"]); end' \
  "$HOME/quast/run1/spinup.h5"
```

确实要改某一项，就用种子模式：它们复制检查点，只改写那一个字符串，规则见[用户指南](guide.md)第 5 节。 `freeze` 冻结网格（把 `amr_every` 设为 0），`wallrefine` 换一套壁面参数，`symseed` 换一个新的 `CASE_SYMMETRY` 。源检查点不会被改动。

### Stokes 残差没达标

运行会向 stderr 打两行，然后中止这一步：

```text
[case] STOKES-FAIL step=<n> res=<r> (vel=<rv> div=<rd>) stokes_tol=1.0e-03 shift=<s> refine=<k> leaves=<nl> N=<N> saddle=<nd> eta=[<lo>,<hi>] eta_clip=<pct>%
ERROR: Stokes residual <r> exceeds stokes_tol=0.001 at step <n>; the solve auto-selected mg; pass linear=:direct explicitly to force the direct shift ladder (...)
```

`vel` 与 `div` 把残差拆成动量块和连续性块；`eta=[lo,hi]` 与 `eta_clip` 说明黏度有没有贴住上下限。

- 设 `CASE_LINEAR=direct` 。直接路径会走对角平移阶梯并做迭代精化，是能把困难步救回来的那条路。
- 再加 `CASE_STOKES_SCALING=equilibrated` ，它对直接鞍点做平衡，只在本来就会失败的那些求解上生效，对多重网格路径没有作用。
- 直接法也过不去，说明这个分辨率下的状态本身就难：换小一点的预算新开一段运行，不要接着续这个检查点。

运行已经中止，所以检查点最多旧 200 步。改好旋钮后从它重启；万一提示「零步」，说明 `CASE_STEPS` 设小了。

下一步：物理设定、全部开关与输出格式见[用户指南](guide.md)。
