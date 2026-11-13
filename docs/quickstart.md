# Quickstart

From a clean checkout to a running mantle-convection case: install Julia, run the
ten-step smoke test, start a 2 000-step spin-up, watch it, stop it, resume it. The
case is `julia/scripts/case_realistic_2d.jl`, a 2.5 x 1 box with free-slip walls,
internal heating, composite diffusion-dislocation rheology, two phase transitions
and latent heat. `docs/guide.md` has the physics, the method and every knob.

## 1. Install

You need Julia 1.12. The compat bound in `julia/Project.toml` is `julia = "1.10"`,
so any 1.10 or later release resolves; 1.12 is what the card is run with.

```bash
cd /path/to/QuaST
julia --version                      # expect 1.12.x
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia -e 'using MantleSEM; println("ok")'
```

`--project=julia` points every command on this page at the project in `julia/`.
`Pkg.instantiate()` installs the versions pinned in `julia/Manifest.toml` and
downloads the HDF5 artifacts. The third line compiles the solver once and prints
`ok`; later calls return immediately.

Budget 5 to 10 minutes and under 2 GB of disk: about 1 GB for Julia itself and
about 0.5 GB for the depot in `~/.julia`, which holds the packages and their
precompile cache.

## 2. The ten-step smoke

```bash
CASE_OUT=/tmp/quast_smoke julia --project=julia julia/scripts/case_realistic_2d.jl smoke
```

The first run takes about two minutes, most of it compilation; a second run takes
seconds. `smoke` advances ten steps on the card's own initial condition, adapts
the mesh at steps 5 and 10, and writes a frame at steps 1, 6 and 10. It prints
four header lines that record the card, one line per step, an AMR line at each
mesh cycle, and two closing lines:

```
[case] mode=smoke Ra=1.0e+05 H=10.0 p=2 max_level=7 max_leaves=10000 amr_every=5 steps=10 ...
[case] comp_Ad=1.6240183119e+05 comp_As=2.6796419238e-01 Ed=12.0 Es=17.0 n=3.0 T0=0.50 ...
[case] latent coefficients: 410 -0.059364 660 +0.130596 (negative = heats a downwelling)
  step    1 t=... dt=... Nu=.../... Vrms=... N=... leaves=... closure=...
    AMR: N=... leaves=... maxjump=1 refined=... coarsened=... merged=...
[case] DONE mode=smoke steps=10 t=... wall=... s (... s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=... Nu_bot=... Vrms=... meanT=... closure=... leaves=... N=...
```

The numbers depend on the step; the shapes do not. The two Nusselt numbers start
at the conductive values for `H = 10`, 6 at the top and -4 at the bottom, and stay
near them for ten steps.

The output directory holds no checkpoint, because the smoke mode passes none:

```
/tmp/quast_smoke/history.csv                   header plus 10 rows
/tmp/quast_smoke/frames/frame_00001_T.bin      512 KB each, steps 1, 6, 10
/tmp/quast_smoke/frames/frame_meta.csv         3 rows plus a header
```

`wc -l /tmp/quast_smoke/history.csv` prints 11 and
`ls /tmp/quast_smoke/frames/*_T.bin | wc -l` prints 3. In the history, column 1
is the nondimensional time, columns 3 and 4 are `Nu_top` and `Nu_bot` and column 8
is the leaf count; section 4.2 of the guide lists all nineteen columns.

## 3. The first real run

A 2 000-step spin-up with the default card, on eight threads:

```bash
mkdir -p "$HOME/quast/run1"
CASE_OUT="$HOME/quast/run1" CASE_STEPS=2000 \
  julia --project=julia --threads=8 julia/scripts/case_realistic_2d.jl spinup \
  > "$HOME/quast/run1/run.log" 2>&1
```

The `mkdir` comes first because the shell opens `run.log` before the program can
create `CASE_OUT`; the program also calls `mkpath(CASE_OUT)`. `spinup` writes a
checkpoint every 200 steps and one at the end, to `$CASE_OUT/spinup.h5`, and one
frame every 5 steps plus a final frame.

Expect 1.5 to 2.5 hours on eight threads: a from-scratch run costs about
2.8 s/step on the multigrid path that `CASE_LINEAR=auto` selects at this size, and
about 4.4 s/step with `CASE_LINEAR=direct`. Disk for the 2 000 steps is about
0.3 GB of frames (401 files of roughly 0.8 MB), one checkpoint of about 2 MB, a
`history.csv` of 2 000 rows and the log. The frame stream dominates.

### Watching it

`run.log` is the live view. It carries, per step, the line the solver prints with
`verbose=true`, every 5 steps an AMR record, and every 100 steps a progress line
from the card's callback:

```
  step <n> t=<t> dt=<dt> Nu=<Nu_top>/<Nu_bot> Vrms=<Vrms> N=<nodes> leaves=<leaves> closure=<closure>
    AMR: N=<nodes> leaves=<leaves> maxjump=<j> refined=<r> coarsened=<c> merged=<m>
  [case] step=<n> t=<t> leaves=<leaves> N=<nodes> auto_fallbacks=<count>
```

The callback line flushes stdout, so a redirected log lags by at most 100 steps. A
rising `auto_fallbacks` count is not an error: each fallback redoes one Stokes
solve on the direct path and the run continues.

`frames/frame_meta.csv` is the second live view, flushed after every frame, so it
advances every 5 steps. It holds one row per frame: `frame,step,t,dt,N,leaves,
nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac,wall_s,
ng`, where `wall_s` is the seconds since the frame writer started. `history.csv`
is written once, after the last step, when the run returns: it does not exist
while the run is in progress. Its columns are `t, dt, Nu_top, Nu_bot, Vrms, meanT,
N, leaves, div_max, hmin, umax, cfl, closure, usrms_surf, umax_surf, W, U, balance,
closure_nolatent`; the last is present because this card has latent heat on.
`closure` is the departure of the wall fluxes from the steady balance
`Nu_top - Nu_bot = H`, a storage rate at a transient state, so a value of order
`H` early in a run is normal.

## 4. Stopping and resuming

Stop with Ctrl-C, or kill the process. You lose at most 199 steps, the steps since
the last checkpoint, because the card checkpoints every 200.

```bash
CASE_OUT="$HOME/quast/run1" CASE_STEPS=4000 \
  julia --project=julia --threads=8 julia/scripts/case_realistic_2d.jl resume \
  "$HOME/quast/run1/spinup.h5" >> "$HOME/quast/run1/run.log" 2>&1
```

Three rules:

- `CASE_STEPS` is an absolute step number, not a count. A resume whose
  `CASE_STEPS` does not exceed the checkpoint's step runs zero steps and prints
  `[case] DONE mode=resume steps=... (no steps were run: maxsteps=... <= checkpoint step); nothing written`.
- `resume` writes its own output to `$CASE_OUT/resume.h5`, so the next resume
  needs that path explicitly, or it starts again from the older `spinup.h5`. With
  no argument it reads `$CASE_OUT/spinup.h5`.
- A resume repeats the settings the checkpoint stored. A default run needs no
  extra variables; a run that changed a `CASE_*` knob must repeat it, or the
  restart is refused (section 6).

Always set `CASE_OUT`, and give it an absolute path. The fallback is
`out_julia/case_realistic` inside the repository, which is the directory an
earlier spin-up may have used: writing there overwrites `history.csv`, overwrites
the frames with the same step numbers and appends to `frame_meta.csv`. Use one
directory per run, and a scratch path for every smoke and test.

## 5. The three knobs you need

| knob | default | what it does |
|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | Output root. Set it on every invocation, absolute. Everything the run writes lives under it. |
| `CASE_STEPS` | 0 (mode default: 10 smoke, 150 short, 10000 spinup, 20000 freeze, 2000 resume and wallrefine) | Step budget, compared against the absolute step number, so on a resume it is the step to stop at. |
| `CASE_LINEAR` | `auto` | Linear solver. `auto` picks the multigrid above 20 000 saddle dofs, or above 2 000 dofs with a viscosity contrast at most 100, and the sparse direct factorisation otherwise. `direct` forces the direct path, the bitwise reproducible one, at roughly twice the wall time and twice the memory. `mg` forces the matrix-free multigrid, serial, because the card passes no `mg_threads`; only `auto` gets the threaded V-cycle. |

## 6. Troubleshooting

### Out of memory

Julia reports `ERROR: OutOfMemoryError()` with a stack trace, or the process
disappears and `run.log` simply stops; under WSL the OOM killer ends it with exit
code 137 and no message.

The default card peaks at about 1.3 GB resident with the multigrid path and about
2.4 GB with `CASE_LINEAR=direct`, at the 10 000-leaf budget. Keep
`CASE_LINEAR=auto` or set `CASE_LINEAR=mg`, which uses about half the memory of
the direct factorisation at this size; lower the budget with
`CASE_MAXLEAVES=5000` or `CASE_MAXLEV=6`; check `free -g` before launching. WSL
gives the virtual machine its own memory limit, raised in `.wslconfig` on the
Windows side. The two mesh keys are restart configuration keys, so a smaller
budget means a new run, not a resume.

### A restart refused by the configuration guard

The driver compares the settings stored in a checkpoint with the settings of the
live run and refuses to continue on any difference. The message names the file and
every key that differs:

```
ERROR: ArgumentError: restart configuration mismatch in /home/you/quast/run1/spinup.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

The values are the canonical strings of each setting: floats print as `5.0`,
symbols as `x` or `none`. Repeat the whole set the checkpoint was written with, or
read it back with

```bash
julia --project=julia -e 'using HDF5; h5open(ARGS[1]) do f; println(HDF5.attrs(f)["config"]); end' \
  "$HOME/quast/run1/spinup.h5"
```

For a deliberate change of one key there are seed modes that copy the checkpoint
and rewrite exactly that string: `freeze` (mesh frozen, `amr_every` becomes 0),
`wallrefine` (a new wall rule) and `symseed` (a new `CASE_SYMMETRY`). The source
checkpoint is never modified.

### A Stokes residual that misses the tolerance

The run prints two lines to stderr and aborts the step:

```
[case] STOKES-FAIL step=<n> res=<r> (vel=<rv> div=<rd>) stokes_tol=1.0e-03 shift=<s> refine=<k> leaves=<nl> N=<N> saddle=<nd> eta=[<lo>,<hi>] eta_clip=<pct>%
ERROR: Stokes residual <r> exceeds stokes_tol=0.001 at step <n>; the solve auto-selected mg; pass linear=:direct explicitly to force the direct shift ladder (...)
```

`vel` and `div` split the residual into the momentum and continuity blocks, and
`eta=[lo,hi]` with `eta_clip` says whether the viscosity sat at its bounds.

- Set `CASE_LINEAR=direct`. The direct path runs a diagonal-shift ladder with
  iterative refinement, and it is the path that recovers the hard steps.
- Add `CASE_STOKES_SCALING=equilibrated`, which equilibrates the direct saddle and
  engages only on a solve that would otherwise fail. It has no effect on the
  multigrid path.
- If a direct solve still misses, the state is genuinely hard at this resolution:
  start a new run with a smaller budget rather than resuming this checkpoint.

The run aborted, so the checkpoint is up to 200 steps old. Restart from it with
the corrected knobs; the zero-step message tells you if `CASE_STEPS` is too small.
