<p align="right"><a href="../en/quickstart.md">English</a> · <a href="../zh/quickstart.md">中文</a></p>

# Quick start

This page takes you from nothing to a running mantle-convection case: install Julia, run a ten-step smoke test, start a two-thousand-step spin-up, stop it, and resume it. The case is `examples/case_realistic_2d.jl`, the 2.5 x 1 realistic case documented in the [guide](guide.md). The guide has the physics and every knob; this page has the commands and the failures a new user meets.

## Install

You need Julia 1.12. `Project.toml` declares `julia = "1.10"`, so anything from 1.10 on resolves; the commands here were checked on 1.12.

```bash
cd /path/to/QuaST
julia --version                      # expect 1.12.x
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using QuaST; println("ok")'
```

`--project=.` points every later command at the project in this directory. `Pkg.instantiate()` installs what `Manifest.toml` pins and downloads the HDF5 binaries. The last command compiles the solver once and prints `ok`; after that, calls return immediately.

Set aside five to ten minutes and under 2 GB of disk. Julia itself is about 1 GB, and the depot in `~/.julia`, which holds the packages and the precompile cache, is about 0.5 GB.

## The ten-step smoke test

```bash
CASE_OUT=/tmp/quast_smoke julia --project=. examples/case_realistic_2d.jl smoke
```

The first run takes about two minutes, almost all of it compilation. Run it again and it finishes in seconds. `smoke` advances ten steps from the initial condition stored in the case, adapts the mesh at steps 5 and 10, and writes a frame at steps 1, 6 and 10. Expect four header lines, then a step record and a Picard line per step, then AMR records at each mesh cycle, then two closing lines:

```text
[case] mode=smoke Ra=1.0e+05 H=10.0 p=2 max_level=7 max_leaves=10000 amr_every=5 steps=10 ...
[case] comp_Ad=1.6240183119e+05 comp_As=2.6796419238e-01 Ed=12.0 Es=17.0 n=3.0 T0=0.50 ...
[case] phases: z0=0.87 w=0.025 slope=+0.032 B=1.8182 | z0=0.78 ... latent_mode=per_phase
[case] latent coefficients: 410 -0.059364 660 +0.130596 (negative = heats a downwelling)
  step    1 t=... dt=... Nu=.../... Vrms=... N=... leaves=... closure=...
    Picard it=1 change=...
    AMR: N=... leaves=... maxjump=<j> refined=... coarsened=... merged=...
      balanced: gate_refused=... extended=... recirc=... post=... tau_h=... eligible=...
[case] DONE mode=smoke steps=10 t=... wall=... s (... s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=... Nu_bot=... Vrms=... meanT=... closure=... leaves=... N=...
```

The numbers move with the step count; the shape of each line does not. Both Nusselt numbers start at the conductive values and stay near them for ten steps; section 1.3 of the [guide](guide.md) defines them and states the conductive values.

The smoke test writes no checkpoint, so the output directory holds only the history and the frames:

```text
/tmp/quast_smoke/history.csv                   header plus 10 rows
/tmp/quast_smoke/frames/frame_00001_T.bin      512 KB each, steps 1, 6, 10
/tmp/quast_smoke/frames/frame_00001_mesh.csv   one line per leaf, same steps
/tmp/quast_smoke/frames/frame_00001_eta.bin    node arrays, same steps
/tmp/quast_smoke/frames/frame_00001_speed.bin  node arrays, same steps
/tmp/quast_smoke/frames/frame_00001_nodes.bin  node arrays, same steps
/tmp/quast_smoke/frames/frame_meta.csv         one row per frame plus a header
```

`wc -l /tmp/quast_smoke/history.csv` prints 11, and `ls /tmp/quast_smoke/frames/*_T.bin | wc -l` prints 3. In the history, column 1 is nondimensional time, columns 3 and 4 are `Nu_top` and `Nu_bot`, and column 8 is the leaf count. The [guide](guide.md) lists all the columns, the files of a frame and the checkpoint contents.

## The first real run

A two-thousand-step spin-up with the default case, on eight threads:

```bash
mkdir -p "$HOME/quast/run1"
CASE_OUT="$HOME/quast/run1" CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup \
  > "$HOME/quast/run1/run.log" 2>&1
```

Put `mkdir` first. The shell opens `run.log` before the program gets a chance to create `CASE_OUT`; the program does call `mkpath(CASE_OUT)`, but that comes too late. `spinup` writes a checkpoint every 200 steps and one at the end, both to `$CASE_OUT/spinup.h5`, plus a frame every 5 steps and a final frame.

Plan on 1.5 to 2.5 hours on eight threads. A from-scratch step costs about 2.8 s on the multigrid path that `CASE_LINEAR=auto` selects at this size, and about 4.4 s with `CASE_LINEAR=direct`. Frames dominate the disk use; each frame writes the files listed in section 7.3 of the [guide](guide.md), and two thousand steps at the default stride write 401 of them.

### Watching the run

`run.log` is your live view. It carries a step record and a Picard line per step, two AMR records every 5 steps, and a progress line from the case callback every 100 steps:

```text
  step <n> t=<t> dt=<dt> Nu=<Nu_top>/<Nu_bot> Vrms=<Vrms> N=<nodes> leaves=<leaves> closure=<closure>
    Picard it=<i> change=<c>
    AMR: N=<nodes> leaves=<leaves> maxjump=<j> refined=<r> coarsened=<c> merged=<m>
      balanced: gate_refused=<g> extended=<e> recirc=<r> post=<p> tau_h=<t> eligible=<n>
  [case] step=<n> t=<t> leaves=<leaves> N=<nodes> auto_fallbacks=<count>
```

The callback flushes stdout, so a redirected log lags by at most 100 steps.

**A rising `auto_fallbacks` count is not a failure.** Each fallback redoes one Stokes solve on the direct path, and the run keeps going.

`frames/frame_meta.csv` gives you a second live view. It is flushed after every frame, so it advances every 5 steps, one row per frame; section 7.3 of the [guide](guide.md) names the columns and defines `wall_s`. `history.csv` is different: it is written once, after the run returns, so it does not exist while the run is in progress. Its columns are in the [guide](guide.md), together with the meaning of `closure`. Read `closure` as an enthalpy rate, not as a flux error.

## Stopping and resuming

Press Ctrl-C, or kill the process. You lose at most 199 steps, the ones since the last checkpoint, because the case checkpoints every 200.

```bash
CASE_OUT="$HOME/quast/run1" CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  "$HOME/quast/run1/spinup.h5" >> "$HOME/quast/run1/run.log" 2>&1
```

Three things to know:

- `CASE_STEPS` is an absolute step number, not an increment. If it does not exceed the checkpoint's step, the driver runs nothing, prints `[case] DONE mode=resume steps=... (no steps were run: maxsteps=... <= checkpoint step); nothing written`, and exits.
- `resume` writes to `$CASE_OUT/resume.h5`, so the next resume either takes that path explicitly or starts again from the older `spinup.h5`. With no argument it reads `$CASE_OUT/spinup.h5`.
- A resume inherits the settings stored in the checkpoint. A plain run needs no extra variables. A run that changed a restart key must repeat the same value, or the configuration guard refuses it; section 8 of the [guide](guide.md) lists the keys that are compared.

**Always set `CASE_OUT`, and give it an absolute path.** Without it, the run writes into a directory inside the repository that may still hold files from an earlier spin-up. Writing there overwrites `history.csv`, overwrites frames with the same step numbers, and appends rows to `frame_meta.csv`. Use one directory per run, and a scratch path for every smoke test and test run.

## The three things to change

| Knob | What to do |
|---|---|
| `CASE_OUT` | Set it on every call, make it absolute, and use one directory per run. Everything the run produces lives under it. |
| `CASE_STEPS` | The run stops at this absolute step number, so on a resume it is the step to stop at, not the number of steps to add. The mode defaults are in section 6.1 of the [guide](guide.md). |
| `CASE_LINEAR` | Leave it at `auto` unless you have a reason. `direct` is the bitwise reproducible path and is slower; `mg` forces the matrix-free multigrid and is serial unless the caller sets `mg_threads`. Sections 5.1 and 5.3 of the [guide](guide.md) give the paths and the rule. |

The [guide](guide.md) lists the other `CASE_*` variables and every keyword of `run_semquad_simulation`, including the reference viscosity `CASE_ETA_REF` that defines `Ra` (section 1.9 there). The card also reads four per-wall boundary knobs, `CASE_WALL_BC`, `CASE_WALL_UN`, `CASE_WALL_T` and `CASE_WALL_TN`. Every one is empty by default, which is the free-slip closed box this page runs; [guide section 2](guide.md#2-boundary-conditions) is their reference.

## When something goes wrong

### Out of memory

Julia throws `ERROR: OutOfMemoryError()` with a stack trace, or the process simply disappears and `run.log` stops in the middle of a step. Under WSL the OOM killer ends it with exit code 137 and no message at all.

At the 10 000-leaf cap the multigrid path needs about half the memory of the direct factorization; section 9 of the [guide](guide.md) gives the measured peaks and the memory plan. Stay on `CASE_LINEAR=auto`, or switch to `CASE_LINEAR=mg`. You can also lower the cap with `CASE_MAXLEAVES=5000` or `CASE_MAXLEV=6`. Check `free -g` before launching. WSL gives the virtual machine its own memory limit, raised in `.wslconfig` on the Windows side. Both mesh keys are restart configuration keys, so a smaller cap means a new run rather than a resume.

### A restart refused by the configuration guard

The driver compares the settings stored in the checkpoint against the settings of the live run, key by key, and refuses to continue on any difference. The message names the file and every key that disagrees, with the stored value next to the live one; section 8 of the [guide](guide.md) shows its shape.

Each value is the canonical string of that setting: floats print as `5.0`, symbols as `x` or `none`. Repeat every setting the checkpoint was written with, or read them back:

```bash
julia --project=. -e 'using HDF5; h5open(ARGS[1]) do f; println(HDF5.attrs(f)["config"]); end' \
  "$HOME/quast/run1/spinup.h5"
```

If you really do mean to change one key, use a seed mode. Each one copies the checkpoint and rewrites exactly the conflicting strings; section 8 of the [guide](guide.md) lists what `freeze`, `wallrefine` and `symseed` rewrite. The source checkpoint is never touched.

### A Stokes residual that misses the tolerance

The run prints two lines to stderr and aborts the step:

```text
[case] STOKES-FAIL step=<n> res=<r> (vel=<rv> div=<rd>) stokes_tol=1.0e-03 shift=<s> refine=<k> leaves=<nl> N=<N> saddle=<nd> eta=[<lo>,<hi>] eta_clip=<pct>%
ERROR: Stokes residual <r> exceeds stokes_tol=0.001 at step <n>; the solve auto-selected mg; pass linear=:direct explicitly to force the direct shift ladder (...)
```

`vel` and `div` split the residual into its momentum and continuity parts, and `eta=[lo,hi]` with `eta_clip` tells you whether the viscosity sat against its bounds.

- Set `CASE_LINEAR=direct`. The direct path walks a diagonal-shift ladder and finishes with iterative refinement, and it is the one that recovers hard steps.
- Add `CASE_STOKES_SCALING=equilibrated`. It equilibrates the direct saddle and engages only on solves that would otherwise fail; it does nothing for the multigrid path.
- If a direct solve still misses, the state really is hard at this resolution. Start a new run with a smaller cap instead of resuming this checkpoint.

The run aborted, so the checkpoint is up to 200 steps old. Restart from it with the corrected knobs. If you get the zero-step message, `CASE_STEPS` is too small.

**Next:** [Examples](examples.md) — six runs with their commands and the numbers to check.
