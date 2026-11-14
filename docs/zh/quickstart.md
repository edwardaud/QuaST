<p align="right"><a href="../en/quickstart.md">English</a> · <a href="../zh/quickstart.md">中文</a></p>

# 快速开始

从一份干净的检出到跑起来的地幔对流算例：安装 Julia，跑十步冒烟测试，启动 2 000 步预积分，观察、停止、续算。算例是 `examples/case_realistic_2d.jl`：2.5 × 1 的盒子，四壁自由滑移，内部加热，扩散-位错复合流变，两个相变并带潜热。物理、方法和全部旋钮见[用户指南](guide.md)。

## 1. 安装

需要 Julia 1.12。`Project.toml` 的 compat 是 `julia = "1.10"`，所以 1.10 及以后的版本都能解析；本卡在 1.12 上运行。

```bash
cd /path/to/QuaST
julia --version                      # 应为 1.12.x
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using QuaST; println("ok")'
```

`--project=.` 让本页所有命令都使用当前目录的工程。`Pkg.instantiate()` 安装 `Manifest.toml` 锁定的版本，并下载 HDF5 的 artifact。最后一行编译一次求解器并打印 `ok`；之后的调用立即返回。

预算 5 到 10 分钟、不到 2 GB 磁盘：Julia 本身约 1 GB，`~/.julia` 下的 depot 约 0.5 GB，存放包和预编译缓存。

## 2. 十步冒烟

```bash
CASE_OUT=/tmp/quast_smoke julia --project=. examples/case_realistic_2d.jl smoke
```

首次运行约两分钟，大部分时间在编译；第二次只需几秒。`smoke` 在参数卡自带的初值上前进十步，在第 5、10 步自适应网格，在第 1、6、10 步各写一帧。它打印四行头部信息、每步一行、每个网格周期一行 AMR，以及两行收尾：

```
[case] mode=smoke Ra=1.0e+05 H=10.0 p=2 max_level=7 max_leaves=10000 amr_every=5 steps=10 ...
[case] comp_Ad=1.6240183119e+05 comp_As=2.6796419238e-01 Ed=12.0 Es=17.0 n=3.0 T0=0.50 ...
[case] phases: z0=0.87 w=0.025 slope=+0.032 B=1.8182 | z0=0.78 ... latent_mode=per_phase
[case] latent coefficients: 410 -0.059364 660 +0.130596 (negative = heats a downwelling)
  step    1 t=... dt=... Nu=.../... Vrms=... N=... leaves=... closure=...
    AMR: N=... leaves=... maxjump=<j> refined=... coarsened=... merged=...
[case] DONE mode=smoke steps=10 t=... wall=... s (... s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=... Nu_bot=... Vrms=... meanT=... closure=... leaves=... N=...
```

数字随步数变化，行的形状不变。两个 Nusselt 数从 `H = 10` 的传导值出发：

$$
\mathrm{Nu}_{top} = 1 + \frac H2 = 6,
\qquad
\mathrm{Nu}_{bot} = 1 - \frac H2 = -4,
$$

其中 $H$ 是内部加热数，$\mathrm{Nu}_{top}$ 和 $\mathrm{Nu}_{bot}$ 分别是顶壁和底壁的 Nusselt 数。十步内两者都保持在附近。

`smoke` 不传检查点，所以输出目录里没有检查点：

```
/tmp/quast_smoke/history.csv                   表头加 10 行
/tmp/quast_smoke/frames/frame_00001_T.bin      每个 512 KB，第 1、6、10 步
/tmp/quast_smoke/frames/frame_meta.csv         3 行加一个表头
```

`wc -l /tmp/quast_smoke/history.csv` 打印 11，`ls /tmp/quast_smoke/frames/*_T.bin | wc -l` 打印 3。`history.csv` 第 1 列是无量纲时间，第 3、4 列是 `Nu_top` 和 `Nu_bot`，第 8 列是叶数；19 列的全部含义见指南 4.2 节。

## 3. 第一次正式运行

用默认参数卡跑 2 000 步预积分，开八线程：

```bash
mkdir -p "$HOME/quast/run1"
CASE_OUT="$HOME/quast/run1" CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup \
  > "$HOME/quast/run1/run.log" 2>&1
```

`mkdir` 必须放在前面，因为 shell 在程序创建 `CASE_OUT` 之前就要打开 `run.log`；程序自己也会调用 `mkpath(CASE_OUT)`。`spinup` 每 200 步写一个检查点到 `$CASE_OUT/spinup.h5`，结束时再写一个；每 5 步写一帧，末尾补一帧。

八线程上预计 1.5 到 2.5 小时。从零开始的一步在 `CASE_LINEAR=auto` 于本规模选中的多重网格路径上约 2.8 s，`CASE_LINEAR=direct` 约 4.4 s。2 000 步的磁盘占用约 0.3 GB 帧（401 个文件，每个约 0.8 MB）、一个约 2 MB 的检查点、2 000 行的 `history.csv` 和日志。帧流占大头。

### 观察运行

`run.log` 是实时视图。它逐步携带 `verbose=true` 的求解器输出，每 5 步一条 AMR 记录，每 100 步一行参数卡回调的进度：

```
  step <n> t=<t> dt=<dt> Nu=<Nu_top>/<Nu_bot> Vrms=<Vrms> N=<nodes> leaves=<leaves> closure=<closure>
    AMR: N=<nodes> leaves=<leaves> maxjump=<j> refined=<r> coarsened=<c> merged=<m>
  [case] step=<n> t=<t> leaves=<leaves> N=<nodes> auto_fallbacks=<count>
```

回调会 flush stdout，所以重定向的日志最多滞后 100 步。`auto_fallbacks` 上升不代表失败：每次回退把一次 Stokes 求解改走直接路径，运行继续。

`frames/frame_meta.csv` 是第二个实时视图，每帧之后 flush，因此每 5 步前进一次。它每帧一行，列为

```
frame,step,t,dt,N,leaves,nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac,wall_s,ng
```

其中 `wall_s` 是从帧写入器启动至今的秒数。`history.csv` 只在运行返回后写一次，运行期间不存在。它的列为 `t, dt, Nu_top, Nu_bot, Vrms, meanT, N, leaves, div_max, hmin, umax, cfl, closure, usrms_surf, umax_surf, W, U, balance, closure_nolatent`；最后一列存在，是因为本卡开了潜热。`closure` 度量壁面通量偏离稳态平衡

$$
\mathrm{Nu}_{top} - \mathrm{Nu}_{bot} = H
$$

的程度。瞬态下它是储能率，所以运行早期出现与 $H$ 同阶的值很正常。

## 4. 停止与续算

用 Ctrl-C 停止，或杀掉进程。最多丢 199 步，也就是最后一个检查点之后的步数，因为参数卡每 200 步检查点一次。

```bash
CASE_OUT="$HOME/quast/run1" CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  "$HOME/quast/run1/spinup.h5" >> "$HOME/quast/run1/run.log" 2>&1
```

三条规则：

- `CASE_STEPS` 是绝对步号，不按增量解释。续算的 `CASE_STEPS` 不超过检查点的步号时，运行零步并打印 `[case] DONE mode=resume steps=... (no steps were run: maxsteps=... <= checkpoint step); nothing written`。
- `resume` 把自己的输出写到 `$CASE_OUT/resume.h5`，所以下一次续算要么显式给它这个路径，要么又从旧的 `spinup.h5` 开始。不带参数时它读 `$CASE_OUT/spinup.h5`。
- 续算沿用检查点里存下的设置。默认运行不需要额外变量；改过某个 `CASE_*` 旋钮的运行必须重设同一个值，否则重启被拒（第 6 节）。

始终设置 `CASE_OUT`，并给绝对路径。回退值是仓库内的 `out_julia/case_realistic`，可能是更早一次预积分用过的目录：写进去会覆盖 `history.csv`，覆盖步号相同的帧，并向 `frame_meta.csv` 追加。每次运行用一个目录，每次冒烟和测试用一个临时路径。

## 5. 三个必须知道的旋钮

| 旋钮 | 默认值 | 作用 |
|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | 输出根目录。每次调用都设，且用绝对路径。运行写出的所有东西都在它下面。 |
| `CASE_STEPS` | 0（模式默认：smoke 10，short 150，spinup 10000，freeze 20000，resume 和 wallrefine 2000） | 步数预算，与绝对步号比较，所以在续算时它是停止的步号。 |
| `CASE_LINEAR` | `auto` | 线性求解器。`auto` 在鞍点自由度超过 20 000 时选多重网格，或在自由度超过 2 000 且黏度对比不超过 100 时选多重网格，其余情况选稀疏直接分解。`direct` 强制直接路径，也就是逐位可复现的那条，墙钟约两倍、内存约两倍。`mg` 强制无矩阵多重网格，串行，因为参数卡不传 `mg_threads`；只有 `auto` 拿到多线程 V 循环。 |

## 6. 故障排查

### 内存不足

Julia 报 `ERROR: OutOfMemoryError()` 并带栈回溯，或者进程直接消失、`run.log` 停在半截；在 WSL 下 OOM killer 会以退出码 137 结束它，不给消息。

默认参数卡在 10 000 叶预算下，多重网格路径的峰值常驻内存约 1.3 GB，`CASE_LINEAR=direct` 约 2.4 GB。保持 `CASE_LINEAR=auto`，或设 `CASE_LINEAR=mg`，后者在本规模只用直接分解约一半的内存；用 `CASE_MAXLEAVES=5000` 或 `CASE_MAXLEV=6` 降低预算；启动前看 `free -g`。WSL 给虚拟机单独的内存上限，在 Windows 侧的 `.wslconfig` 里调高。两个网格键都是重启配置键，所以缩小预算意味着新运行，不能续算。

### 重启被配置守卫拒绝

驱动逐键比较检查点里存的设置与当前运行的设置，有任何差异就拒绝继续。消息会指出文件和每个不同的键：

```
ERROR: ArgumentError: restart configuration mismatch in /home/you/quast/run1/spinup.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

值是该设置的规范字符串：浮点打印成 `5.0`，符号打印成 `x` 或 `none`。重复检查点写入时的整组设置，或者用下面的命令读回来：

```bash
julia --project=. -e 'using HDF5; h5open(ARGS[1]) do f; println(HDF5.attrs(f)["config"]); end' \
  "$HOME/quast/run1/spinup.h5"
```

要故意改一个键，用种子模式：它们复制检查点，只改写那一个字符串。`freeze` 冻结网格，把 `amr_every` 改成 0；`wallrefine` 换一套壁面规则；`symseed` 换一个新的 `CASE_SYMMETRY`。源检查点永远不会被修改。

### Stokes 残差没过容差

运行会向 stderr 打印两行并中止该步：

```
[case] STOKES-FAIL step=<n> res=<r> (vel=<rv> div=<rd>) stokes_tol=1.0e-03 shift=<s> refine=<k> leaves=<nl> N=<N> saddle=<nd> eta=[<lo>,<hi>] eta_clip=<pct>%
ERROR: Stokes residual <r> exceeds stokes_tol=0.001 at step <n>; the solve auto-selected mg; pass linear=:direct explicitly to force the direct shift ladder (...)
```

`vel` 和 `div` 把残差拆成动量块与连续性块；`eta=[lo,hi]` 和 `eta_clip` 说明黏度是否贴住了边界。

- 设 `CASE_LINEAR=direct`。直接路径走对角平移阶梯并做迭代精化，是能救回困难步的那条路。
- 加 `CASE_STOKES_SCALING=equilibrated`，它平衡直接鞍点，只在本来会失败的求解上生效，对多重网格路径无效。
- 直接求解仍然不过，说明该分辨率下的状态确实困难：用更小的预算新开一段运行，不要续算这个检查点。

运行已经中止，所以检查点最多旧 200 步。用改正后的旋钮从它重启；零步消息会告诉你 `CASE_STEPS` 是否太小。
