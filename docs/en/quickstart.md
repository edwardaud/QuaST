<p align="right"><a href="../en/quickstart.md">English</a> · <a href="../zh/quickstart.md">中文</a></p>

# Quick start

This page takes you from nothing to a running mantle-convection case: install Julia, run a ten-step smoke test, start a two-thousand-step spin-up, stop it, and resume it. The case is `examples/case_realistic_2d.jl` — a 2.5 x 1 box with free-slip walls, internal heating, composite diffusion–dislocation rheology, and two phase transitions with latent heat. For the physics and every knob, see the [guide](guide.md).

## 1. Install

You need Julia 1.12. `Project.toml` declares `julia = "1.10"`, so anything from 1.10 on resolves; the commands here were checked on 1.12.

```bash
cd /path/to/QuaST
julia --version                      # expect 1.12.x
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using QuaST; println("ok")'
```

`--project=.` points every later command at the project in this directory. `Pkg.instantiate()` installs what `Manifest.toml` pins and downloads the HDF5 binaries. The last command compiles the solver once and prints `ok`; after that, calls return immediately.

Set aside five to ten minutes and under 2 GB of disk. Julia itself is about 1 GB, and the depot in `~/.julia` — packages plus precompile cache — is about 0.5 GB.

## 2. The ten-step smoke test

```bash
CASE_OUT=/tmp/quast_smoke julia --project=. examples/case_realistic_2d.jl smoke
```

The first run takes about two minutes, almost all of it compilation. Run it again and it finishes in seconds. `smoke` advances ten steps from the initial condition stored in the case, adapts the mesh at steps 5 and 10, and writes a frame at steps 1, 6 and 10. Expect four header lines, then one line per step, then AMR records at each mesh cycle, then two closing lines:

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

The numbers move with the step count; the shape of each line does not. Both Nusselt numbers start at the conductive values for `H = 10` and stay near them for ten steps:

$$
\mathrm{Nu}_{top} = 1 + \frac H2 = 6,
\qquad
\mathrm{Nu}_{bot} = 1 - \frac H2 = -4
$$

Here $H$ is the internal heating number, and $\mathrm{Nu}\_{top}$ and $\mathrm{Nu}\_{bot}$ are the top and bottom Nusselt numbers.

The smoke test writes no checkpoint, so the output directory holds only this:

```text
/tmp/quast_smoke/history.csv                   header plus 10 rows
/tmp/quast_smoke/frames/frame_00001_T.bin      512 KB each, steps 1, 6, 10
/tmp/quast_smoke/frames/frame_00001_mesh.csv   one line per leaf, same steps
/tmp/quast_smoke/frames/frame_meta.csv         3 rows plus a header
```

`wc -l /tmp/quast_smoke/history.csv` prints 11, and `ls /tmp/quast_smoke/frames/*_T.bin | wc -l` prints 3. In the history, column 1 is nondimensional time, columns 3 and 4 are `Nu_top` and `Nu_bot`, and column 8 is the leaf count. The guide lists all nineteen columns in section 4.2.

## 3. The first real run

A two-thousand-step spin-up with the default case, on eight threads:

```bash
mkdir -p "$HOME/quast/run1"
CASE_OUT="$HOME/quast/run1" CASE_STEPS=2000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl spinup \
  > "$HOME/quast/run1/run.log" 2>&1
```

Put `mkdir` first. The shell opens `run.log` before the program gets a chance to create `CASE_OUT`; the program does call `mkpath(CASE_OUT)`, but that comes too late. `spinup` writes a checkpoint every 200 steps and one at the end, both to `$CASE_OUT/spinup.h5`, plus a frame every 5 steps and a final frame.

Plan on 1.5 to 2.5 hours on eight threads. A from-scratch step costs about 2.8 s on the multigrid path that `CASE_LINEAR=auto` selects at this size, and about 4.4 s with `CASE_LINEAR=direct`. Disk for two thousand steps comes to about 0.3 GB of frames (401 files at roughly 0.8 MB), a 2 MB checkpoint, a `history.csv` of two thousand rows, and the log. Frames eat almost all of it.

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

`frames/frame_meta.csv` gives you a second live view. It is flushed after every frame, so it advances every 5 steps, one row per frame:

```text
frame,step,t,dt,N,leaves,nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac,wall_s,ng
```

`wall_s` is the number of seconds since the frame writer started. `history.csv` is different: it is written once, after the run returns, so it does not exist while the run is in progress. Its columns are `t`, `dt`, `Nu_top`, `Nu_bot`, `Vrms`, `meanT`, `N`, `leaves`, `div_max`, `hmin`, `umax`, `cfl`, `closure`, `usrms_surf`, `umax_surf`, `W`, `U`, `balance` and `closure_nolatent`. The last one is present because this case has latent heat on. `closure` measures how far the wall fluxes sit from the steady balance

$$
\mathrm{Nu}_{top} - \mathrm{Nu}_{bot} = H
$$

At a transient state it equals the storage rate, so a value of order $H$ early in a run is exactly what you should see.

## 4. Stopping and resuming

Press Ctrl-C, or kill the process. You lose at most 199 steps, the ones since the last checkpoint, because the case checkpoints every 200.

```bash
CASE_OUT="$HOME/quast/run1" CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  "$HOME/quast/run1/spinup.h5" >> "$HOME/quast/run1/run.log" 2>&1
```

Three things to know:

- `CASE_STEPS` is an absolute step number, not an increment. If it does not exceed the checkpoint's step, the driver runs nothing, prints `[case] DONE mode=resume steps=... (no steps were run: maxsteps=... <= checkpoint step); nothing written`, and exits.
- `resume` writes to `$CASE_OUT/resume.h5`, so the next resume either takes that path explicitly or starts again from the older `spinup.h5`. With no argument it reads `$CASE_OUT/spinup.h5`.
- A resume inherits the settings stored in the checkpoint. A plain run needs no extra variables. A run that changed a `CASE_*` knob must repeat the same value, or the configuration guard refuses it (section 6).

**Always set `CASE_OUT`, and give it an absolute path.** The fallback is `out_julia/case_realistic` inside the repository, which may still hold files from an earlier spin-up. Writing there overwrites `history.csv`, overwrites frames with the same step numbers, and appends rows to `frame_meta.csv`. Use one directory per run, and a scratch path for every smoke test and test run.

## 5. The three knobs that matter

| Knob | Default | What it does |
|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | Output root. Set it on every call, and make it absolute. Everything the run produces lives under it. |
| `CASE_STEPS` | 0 (mode default: smoke 10, short 150, spinup 10000, freeze 20000, resume and wallrefine 2000) | Step limit, compared against the absolute step number, so on a resume it is the step to stop at. |
| `CASE_LINEAR` | `auto` | Linear solver. `auto` picks multigrid at 20 000 saddle degrees of freedom or more, or at 2 000 or more when the viscosity contrast stays at or below 100, and the sparse direct factorization otherwise. `direct` forces the bitwise reproducible path, at roughly twice the wall time and twice the memory. `mg` forces the matrix-free multigrid, serial, because the case passes no `mg_threads`; only `auto` gets the threaded V-cycle. |

The guide lists the other `CASE_*` variables and every keyword of `run_semquad_simulation`; for the reference viscosity that defines `Ra`, `CASE_ETA_REF` (default `1.0`), see the guide's section on the Rayleigh-number reference viscosity.

## 6. When something goes wrong

### Out of memory

Julia throws `ERROR: OutOfMemoryError()` with a stack trace, or the process simply disappears and `run.log` stops in the middle of a step. Under WSL the OOM killer ends it with exit code 137 and no message at all.

At the 10 000-leaf cap, the default case peaks at about 1.3 GB resident on the multigrid path and about 2.4 GB with `CASE_LINEAR=direct`. Stay on `CASE_LINEAR=auto`, or switch to `CASE_LINEAR=mg`, which needs about half the memory of the direct factorization at this size. You can also lower the cap with `CASE_MAXLEAVES=5000` or `CASE_MAXLEV=6`. Check `free -g` before launching. WSL gives the virtual machine its own memory limit, raised in `.wslconfig` on the Windows side. Both mesh keys are restart configuration keys, so a smaller cap means a new run rather than a resume.

### A restart refused by the configuration guard

The driver compares the settings stored in the checkpoint against the settings of the live run, key by key, and refuses to continue on any difference. The message names the file and every key that disagrees:

```text
ERROR: ArgumentError: restart configuration mismatch in /home/you/quast/run1/spinup.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

Each value is the canonical string of that setting: floats print as `5.0`, symbols as `x` or `none`. Repeat every setting the checkpoint was written with, or read them back:

```bash
julia --project=. -e 'using HDF5; h5open(ARGS[1]) do f; println(HDF5.attrs(f)["config"]); end' \
  "$HOME/quast/run1/spinup.h5"
```

If you really do mean to change one key, use a seed mode. Each one copies the checkpoint and rewrites exactly that one string: `freeze` freezes the mesh by setting `amr_every` to 0, `wallrefine` installs a new wall rule, and `symseed` installs a new `CASE_SYMMETRY`. The source checkpoint is never touched.

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
