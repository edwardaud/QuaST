# Handoff: the Julia quadtree SEM at e790092

Read this first. It is the entry point to the project at HEAD `e790092` (2026-11-13), after
the matrix-free campaign, the `linear` and `amr_policy` default flips, and the three-arm
verification. It replaces the 511-line note written at `015b6f7`. The runbook and the rules
of engagement survive; the state, the numbers and the open list are new. Every number below
is a configuration value or is attributed to the document, gate or log that recorded it.
Numbers nobody measured are absent.

Environment: Julia 1.12.7. The tests and the benchmarks were run under WSL Ubuntu-24.04
with the Linux Julia; the Windows build cannot run the project because the Manifest and
HDF5 do not resolve there. Commands below assume you start at the repository root.

**Run artifacts.** The `out_julia/...` paths in this document name case directories, logs
and checkpoints from the development tree. They are cited so that a measured number can be
traced to the record that produced it; they are not part of this repository. The commands
that write into them are the recorded protocol — any `CASE_OUT` of your own works instead.

## 1. Where the project stands

`QuaST` is a two-dimensional Cartesian box mantle-convection solver in Julia.
The live code line is Route C, the quadtree adaptive spectral element method under `julia/`,
and it is what this repository contains. The frozen PETSc linear-backend prototype (Route D)
and the Route A archive are not part of this repository.

### 1.1 The three layers

The defaults now depend on which layer you call. The table is the whole story.

| layer | linear | AMR |
|---|---|---|
| library `run_semquad_simulation` | `linear=:auto` since `2453f16` | `amr_policy=:balanced` since `4584e85` |
| realistic card `case_realistic_2d.jl` | `CASE_LINEAR=auto`; resolves to MG at production size | `:balanced`, 2:1 gate on, `symmetry=:none`, 10 000 leaves, wall 5/0.1 |
| watchdog `watchdog_case.sh` | follows the checkpoint | follows the checkpoint; the three AMR keys default to the checkpoint-era values |

`linear=:auto` runs before assembly from two quantities: `saddle_dofs = 2*Nu + Np` and the
viscosity contrast sampled on a fixed 33 x 33 grid. The rule, in order:

```julia
if saddle_dofs >= 20_000                          return :mg
elseif contrast <= 100 && saddle_dofs >= 2_000    return :mg
else                                               return :direct
```

A selected MG solve that misses `auto_fallback_tol` is redone with `:direct` and counted in
`LAST_AUTO_FALLBACK`. At the two production checkpoints the selector's own convention puts
both states above 20 000: 78 720 on `F` (8 188 leaves) and 95 308 on `P1` (10 000 leaves),
so a bare driver call and the card both resolve to MG. `:direct` remains the only bitwise
path, and `pic_mode=:newton` requires it explicitly.

### 1.2 The document map

Each volume owns one subject. Start with the one closest to the question.

| volume | what it owns |
|---|---|
| SOLVER | [SOLVER.md](SOLVER.md): what the code solves, the Taylor-Hood discretization, the four linear paths and the `:auto` rule, the Newton exception, and the physics feature inventory |
| PHYSICS | [PHYSICS.md](PHYSICS.md): the energy operator, the rheology laws, latent heat per transition, ALA versus TALA, the EOS reference state, and the validation of each |
| AMR | [AMR.md](AMR.md): the quadtree, the balanced policy and its defaults, the 2:1 gate, the local rebalance, the hot-path ledger, and the re-baseline that made `:balanced` the default |
| MG | [MG.md](MG.md): the matrix-free multigrid solver, the ten stages with their numbers, the spectrum and Schur studies, the crossover with direct, and the rejected levers |
| MG_LITERATURE | [MG_LITERATURE.md](MG_LITERATURE.md): what mature codes do, with sources: per-cycle costs, smoothers, Schur approximations, triangular solves, coarsening, vectorization |
| PERFORMANCE | [PERFORMANCE.md](PERFORMANCE.md): where a warm step's time goes, the direct-versus-MG verdict, the re-calibrated crossover, allocations, scaling and threading |
| BUG_LEDGER | [BUG_LEDGER.md](BUG_LEDGER.md): every serious defect class, with symptom, mechanism, measurement, fix and the gate that pins it. Read this first when something breaks |
| VALIDATION | [VALIDATION.md](VALIDATION.md): the benchmarks, the Nusselt conventions and their error terms, the energy-budget identity, the blob digests and the ASPECT comparison |
| CASE_REALISTIC | [CASE_REALISTIC.md](CASE_REALISTIC.md): the realistic card and its defaults, the launch protocol, the E and G runs, and what the case does not establish |
| VIDEOS | [VIDEOS.md](VIDEOS.md): the three-row renderer, the shared colour scale, the time alignment, the mesh-level colouring and the clip catalogue |
| HANDOFF | this file: the current state, the limits, the commands and the rules |

The project-level review (`ROUTE_C_REVIEW.md`) and the frozen route-D prototype are not part of this repository;
this file is the shipped entry point to the state they reviewed. The `ROUTE_C_REVIEW §n` citations in §2.3 name
that development-tree review.

### 1.3 What the code does now

At `e790092` the live code solves the infinite-Prandtl Boussinesq system with Q_p velocity
and Q_{p-1} pressure on a 2:1 balanced quadtree, hanging-node constraints through `C'AC`,
BDF2 time stepping with CFL control, entropy viscosity, and a Picard or damped-Newton
nonlinear solve. The mesh side has Kelly and DWR indicators, the `:balanced` policy with the
2:1 coarsening gate, local rebalance, cooldowns that survive restarts, wall refinement and
an `:x` mirror. The linear side has four paths: bitwise direct (`:direct`), matrix-free
geometric multigrid with three-factor w-BFBT (`:mg`), lagged (`:lagged`), and the
size-and-contrast selector (`:auto`, the library default). The physics stack carries
internal heating through a conductive base state, composite diffusion-plus-dislocation creep
with the isostress partition, per-transition latent heat, ALA and TALA mass forms,
Adams-Williamson and Murnaghan reference states, depth-dependent Drucker-Prager friction
with strain weakening, and phase buoyancy with Clapeyron-derived latent terms. Diagnostics
cover pointwise wall Nusselt numbers, consistent boundary flux and a heat budget; IO covers
bitwise HDF5 checkpoints, VTK and a live renderer. Five benchmark families have comparison
tables: Blankenbach 2b, its wall-refined continuation, Tosi 2015 Cases 1 and 5a, and King
2010's compressible ALA case. The suite has 217 `t_g*.jl` files, 211 of them registered in
`julia/test/runtests.jl`.

The production case is mid-flight. The G arm (`:balanced`, gate, `symmetry=:none`,
`wall_min_level=5`, 10 000-leaf budget) reached step 15 195 at `t = 4.340e-3`, with
`Nu_top = 16.597`, `Nu_bot = -2.720`, `Vrms = 1.113e3` and `N = 43 185`
(`out_julia/case_realistic_G.log`). No case process and no watchdog are running, and the log
has no `[case] DONE`, so treat that arm as stopped rather than finished. The E arm finished
20 000 steps at `t = 8.5598e-3`, with `Nu_top = 18.289`, `Nu_bot = -2.7636`, `Vrms = 913.7`
and `meanT = 1.2620`. The frozen-mesh statistical phase has no run in this tree; the ASPECT
comparison used E out to `t_nd = 5e-3`, a comparison window rather than production
statistics.

## 2. What works and is verified

### 2.1 The matrix-free Stokes solve

The production measurement is one warm `F` solve at eight threads: **1.356 s and 71 FGMRES
iterations**, true residual 6.75e-7, `|du|/|u| = 7.96e-7` and gauge-removed
`|dp|/|p| = 2.27e-7` against the equilibrated direct solve. That direct solve is **3.372 s
warm**. The 2.49x pairing is cross-session, and the same-session floor for the same cell is
2.31x. The range over the four cells (one and eight threads, cold and warm) is 1.30 to 2.31x.
At equal thread counts it is 1.96x on `F` and 1.50x on `P1`.

The direct number is a solve that actually reaches the 1e-6 tolerance. The unscaled direct
ladder stops at 1.317e-3 on `F`, 1 300x above the target, so a comparison against it would
measure two different problems. The reference is `linear=:direct,
stokes_scaling=:equilibrated`, whose own residual is 9.97e-11 on `F` and 1.84e-9 on `P1`.

The step follows. One warm production step is 2.72 s, split as below (the shares are the
deliverable; the absolute sum decomposes the 500-step mean to a few percent).

| phase | s | share | contents |
|---|---|---|---|
| Stokes solve (MG, `mg_tol=1e-8`) | 2.21 | 81.2 % | reduced saddle assembly, warm hierarchy, cached w-BFBT, 91 FGMRES iterations on `P1`, unpack |
| energy | 0.31 | 11.4 % | energy operator assembly and BDF2 solve |
| work | 0.14 | 5.1 % | viscous dissipation integral |
| diagnostics | 0.04 | 1.5 % | Nu, Vrms, mean temperature, divergence, surface flux |
| AMR | 0.025 | 0.9 % | one pass per five steps at 0.122 to 0.134 s per cycle |
| total | 2.72 | 100.1 % | rounded one decimal at a time |

The end-to-end check is the verify100 pair: 500 steps from `t = 0` on the production card,
eight threads, frame writer off. The MG arm takes 1 398.9 s (**2.798 s/step**) and the
equilibrated direct arm 2 201.3 s (**4.403 s/step**), a **1.57x** wall-time ratio, with zero
`auto_fallbacks` on both. The pre-MG 3.58 s step is now a labelled baseline, not a number to
quote as current.

### 2.2 Ten stages, and the gates that pin them

Ten stages shipped: nine changes plus the `ldiv!` one-liner that came with the ninth. The
campaign kept the bitwise contract wherever the arithmetic was supposed to be inert. Eight
of the ten are bitwise; the two that move the preconditioner, stage 1 (coarsening factor and
smoothing schedule) and stage 9 (exact leaf-local BFBT diagonal), are gated on `|du|/|u|`
and gauge-removed `|dp|/|p|` against `linear=:direct, stokes_scaling=:equilibrated` in the
same process. A converged residual is not accepted as evidence for those two.

| gate | what it pins |
|---|---|
| G219 (27 assertions) | the matrix-free defaults: coarsening factor 1.5, `:auto` smoothing |
| G220 (37) | the reduced matrix-free assembly and the finest-`Cu` dedup |
| G221 (57) | the per-level hierarchy cache |
| G222 (18) | the 32-lane and zero-allocation applies are bitwise |
| G223 (39 and 71) | the region V-cycle is bitwise |
| G224 (93 and 180) | the transfer and split-`Cu` phases are bitwise |
| G225 (129 and 129) | the eta lane table is bitwise |
| G226 (28) | the exact restricted BFBT diagonal |
| G227 (15) | the `S1` inner-solve switch reproduces `:chol` bitwise |
| G228 | the card's `mg_tol = 1e-8` declarations and forwarding |
| G229 (13) | the library default `amr_policy = :balanced` |
| G230 (27) | the library default `linear = :auto`, including the Newton rejection |

The gate files exist; registration is uneven. `runtests.jl` registers G222, G225, G226,
G228, G229 and G230; the CI `gates.yml` also runs G219, G220, G221 and G223, and does not run
G224 or G227. Run a missing one explicitly with `run_gates.jl`. Six of the 217 gate files are
unregistered in `runtests.jl`, and they are exactly those six, so the 211 includes and the 217
files are not a contradiction. Registration is a coverage gap, not a numerics gap. The last recorded
full-suite run is Stage 0 of the linear flip, `42ac0f8`: **133 637 pass, 2 fail, 2 broken over
211 includes**. The two failures are G64's known self-convergence pair; the two broken are
the pre-existing skips in G209 and G214. No full-suite count is recorded at `e790092`, so do
not fill one in by arithmetic over per-gate counts.

### 2.3 The mesh and the physics

The `:balanced` policy is the default on both layers. The policy is the same; the card is
what supplies the production numbers: the 10 000-leaf budget, the 2:1 gate, `symmetry=:none`,
the wall rule and `coarsen_frac=0.05`. A bare driver call gets the policy, not the production
configuration. Measured on the 10 000-leaf state over 12 cycles: the AMR pass goes **0.4412 to
0.1198 s/cycle (3.7x)**, the hard budget pass **0.3206 to 0.0113 s/cycle (28x)**, allocation
per four cycles **282.47 to 30.50 MB (9.3x)**, and the fraction of cycles that change the mesh
goes from 0 to 100 percent. Local rebalance costs **7.2 us per merge against 22.9 ms** for a
full `balance!`. The `h/delta` ratio of the wall layer drops from 0.213/0.126 to
**0.045/0.024**. The cooldown is checkpointed as state: `cooldown_no_refine`,
`cooldown_no_coarsen` and a cycle counter are HDF5 datasets, an old file without them loads as
an empty cooldown, and a `:balanced` resume is bitwise against the uninterrupted run (G213, 71
assertions).

The benchmark table is the project's record. The numbers are the 2026-11-05 measurement; they
were not re-run for the 2026-11-08 S8 changes or since, so read the rows as the recorded
result, not as a fresh measurement at `e790092`. The physical path did not change, and the
default path is pinned by the FNV hashes.

| benchmark | mesh | recorded result | evidence |
|---|---|---|---|
| Blankenbach 2b | L3, 40 x 16 | Vrms 171.18 (-0.33 %), energy Nu 6.9874 (+0.83 %), wall/CBF mean 6.9014 (-0.41 %); reference 171.755 / 6.9299 | ROUTE_C_REVIEW §2.5, §4.8 |
| 2b wall-refined | interior 80 x 32, wall level 6, 10 240 leaves | Vrms 172.010 (+0.15 %, EV=0.11) and 171.823 (+0.04 %, EV=0); wall 6.9480, CBF 6.9325, energy 6.9330 agree to 0.04 % | ROUTE_C_REVIEW §2.5, §4.8 |
| Tosi 2015 Case 1 | 64 x 64, 4 096 elements | eight diagnostics inside the Table 2 spread, within 0.6 % of the ASPECT 64 x 64 column; pointwise Nu_bot 3.5410 falls outside | ROUTE_C_REVIEW §2.5, §4.9 |
| Tosi 5a and King ALA | L6; L5 32 x 32 | Tosi 5a: period 0.0790 against ASPECT 0.07614, u_RMS extremum -2.4 %, dissipation within 5 % of ASPECT. King: six diagnostics within 0.5 %, wall budget difference 0.006 | ROUTE_C_REVIEW §2.5 |

Two cautions belong with the table. The 2b initial condition is ASPECT's `ascii data` file
(`julia/scripts/data/initial_temperature_case2b.txt`, an already-convecting state); the
analytic `Function` in the prm was never used by ASPECT, and a near-conductive analytic
initial condition with 16 384x contrast produces the transient that made 2b look
unreachable. Nu is a wall functional and converges more slowly than Vrms: the 2b L3 CBF
deviation of -5.31 % becomes +0.47 % at L4 with the same formula, so low-resolution runs
quote the wall/CBF mean or the L4 value.

The physics stack is verified in the same time step and the same nonlinear solve. The S8
switches default to off, and the default path is pinned bitwise by the 2b and blob FNV
hashes (`d22fdab099ae84e0`, `45ac8acd099adadc`), so turning any of them on is explicit.

| area | what is verified | evidence |
|---|---|---|
| internal heating | conductive state has `Nu_top = Nu_bot = 1`; steady balance `Nu_top - Nu_bot = H` | PHYSICS §1.4; CASE_REALISTIC §1.1 |
| composite creep | branch law matches an independent ASPECT algorithm to 9.0e-15 on 495 points; tangents to 4.8e-9 / 1.9e-8 | PHYSICS §2.1, §5.1; G192, G200, G201 |
| latent heat | apparent heat capacity in [0.88, 1.12]; one-step boundary reaction closes to 1.6e-7; lagged coupling first order (1.01), implicit second order (2.0) | PHYSICS §4.2, §4.4; G194, G199 |
| ALA, TALA, EOS | EOS reference state reproduces the closed form to 9.7e-12; depth-dependent alpha breaks the old formula by 24 %; TALA versus ALA changes Vrms by -7.56 % (-6.88 % after the isostress switch) | PHYSICS §3.1, §3.2, §3.4; G197, G198 |
| Drucker-Prager | `Kp` carries 19.6 % of the pressure-direction Jacobian norm; the 1-D chi steady state closes to 8.5e-8; no analytic strain localization produced | PHYSICS §2.5, §2.6; G195, G196 |
| phase transitions | Clapeyron-derived latent terms have the textbook signs; `latent_i = -0.0594 / +0.1306` | CASE_REALISTIC §1.3; PHYSICS §4.2 |

Two negative results belong with this stack. No analytic shear localisation was obtained: the
plastic strain-rate max/mean ratio is 8.78 without weakening and 8.46 to 11.60 with it, and
chi does not correlate with plastic strain (-0.03 to 0.13) because chi transport has no
diffusion and the element Peclet number is about 1e5. The five-step convective budget does
not close on the transient composite case: 54 % residual without the ALA term and -41 % with
it, while the `u=0` closure of 0.9348 is the meaningful measurement (ROUTE_C_REVIEW §4.13).

## 3. What is limited or broken

1. **MG still loses below the 20 000-dof threshold on high-contrast trees at the library
   tolerance.** At the library `mg_tol = 1e-10` the two trajectory frames at 17 135 and 19 413
   selector dofs are **0.81 to 0.82x** warm, and 9 351 dofs is 1.09x cold. At 1e-6 the same
   25-frame trajectory wins on every frame (mean 1.55x), which is why the selector quote is at
   the accuracy direct reaches and why the 20 000 constant stays. The rule keeps a 1.5x
   margin; a wrong pick costs a fallback, not a failed run. Every row of the crossover table
   is in PERFORMANCE §4.1.

2. **The bottom-flux cross-code gap is unexplained.** Against ASPECT 2.5 on the same card,
   `Nu_top` agrees to 0.005 to 0.9 %, and `Nu_bot` differs by **+/-6 to 9 % with the sign
   flipping with the window**: -6.11 % (CBF) and -8.44 % (gradient) on `[1e-6, 1e-3]`,
   +6.05 % and +5.83 % on `[1e-3, 5e-3]`, +8.5 % in the saturated segment, crossing zero at
   `t_nd = 1.42e-3`. The same-state experiment rules out discretization and phase: importing
   ASPECT's Q2 state reproduces its diagnostics to **0.003 %**, and over 0.84 overturn times
   `Nu_bot` separates by at most 0.6 % while `Vrms` swings +/-15 %. The cause sits in the
   bottom boundary layer's mean structure; the mechanism is open after five candidates were
   excluded. ASPECT's own `Nu_bot` variability is **12 to 16x ours** (sd 4.36 to 5.82 %
   against 0.34 to 0.37 %), and refining its bottom wall makes the sd larger, 4.36 to
   5.82 %. Quote `Nu_bot` with the window and the sign.

3. **Newton requires an explicit `:direct` and is wall-expensive.** `pic_mode=:newton`
   throws at argument validation with the fix in the message: pass `linear=:direct`. The
   reason is structural: each Newton iteration assembles a Jacobian and needs one direct
   saddle factorization, while a Krylov path answers to a tolerance. The cost is measured.
   At 64 leaves a per-step Newton is **1.87x** a Picard-to-tolerance step and **5.15x** the
   production `maxpic=1` step; at L5 those become **12.3x and 60.5x**, and **26.9x and
   131x** when the Krylov budget is raised to 300. Per iteration **96 % of the time is
   GMRES**: 45 % JVP, 17 % preconditioner, 34 % bookkeeping, with the Krylov vector count
   growing from 8 to 80 because the block preconditioner drops the theta-u coupling. The
   default stays `maxpic=1`; Newton is opt-in for small problems that need iteration
   accuracy.

4. **The two `chol(S1)` solves are at their measured floor.** `S1` is 7 424 x 7 424 with
   198 490 nonzeros on `F`; the factor has 649 814 nonzeros, 3.27x fill, 7.8 MB. One solve
   is **0.436 ms** hot, **0.948 ms** in the outer iteration and the pair is **1.90 ms per
   iteration, 13.2 %** of the eight-thread iteration. The pair does not scale: CHOLMOD's
   `cholmod_solve` is serial, 1.84 ms at one thread and 1.90 at eight. The pair's two-pass
   DRAM floor is 1.16 ms, so the recoverable remainder is **0.74 ms, 5.1 %** of the
   iteration. Both routes that aimed at it measured slower than the exact factor: the
   level-scheduled solve **0.43x**, the inexact CG or IC(0) solve **0.30 to 0.34x**. An
   inexact solve would have to finish in 1 to 2 iterations to break even at 0.32 ms.

5. **Mixed precision is unmeasured.** The external evidence is direct: Kronbichler and
   Ljungkvist measure up to +83 % from a single-precision V-cycle with a double-precision
   outer correction. The expectation here is **1.2 to 1.8x on the apply**, which is
   **5 to 12 % of the step**. It cannot be bitwise, so the decision is whether to carry two
   verification bases: a bitwise one for the direct path and an accuracy one for the
   matrix-free path. No number from this repository exists yet.

6. **The pressure near-null band is tamed, not removed.** It is an element-level,
   viscosity-weighted checkerboard in the stiff lid, with relative eigenvalue
   `|lambda|/max|S|` of 3.3e-13 at L4, 6.8e-15 at L5 and 5.4e-18 at L6, so it gets worse
   with refinement and level-by-level patches cannot work. Solver-side symmetric
   equilibration, applied only when a health check fails, brings the failing residuals to
   5.96e-10, 3.65e-8 and 3.64e-7 and is bitwise inert on healthy solves. It is opt-in
   (`stokes_scaling=:equilibrated`, default `:none`). The L6 margin is 2.7x under the 1e-6
   goal; L7 is unmeasured.

7. **AMR allocates about 30 MB per four cycles, and that part is structural.** Of the
   residual, the new mesh's `make_semquad` arrays are 15.3 MB, the cycle's sort and collect
   3.1 MB, the face-table pool 2.6 MB, the transfer output 2.6 MB, the pair list 1.8 MB and
   the wall collect 0.9 MB. The pass time is **0.011 s/step, 0.2 %**, because the AMR pass
   is 2 to 3 % of a step. Do not reopen this as a performance project.

## 4. The verification machinery

`julia/scripts/verify100_case.sh` launches one arm of the verification run from scratch at
`t = 0`, not from the G resume checkpoint. It runs the production G card for 500 steps,
writes one `history.csv`, one checkpoint and the card's cadence frames, and refuses to start
if the output directory exists or if another `case_realistic_2d` spinup is already running.
`CASE_OUT` is always explicit. The script accepts `mg`, `direct` or `auto`; the comparison
video uses `mg`, `direct` and ASPECT 2.5.

The run that exists is the 500-step, three-arm comparison: direct, matrix-free MG and
ASPECT 2.5, all on the same card and the same start time.

| arm | wall | s/step | end `t` | `Nu_top` | `Nu_bot` | `Vrms` |
|---|---|---|---|---|---|---|
| matrix-free MG | 1 398.9 s | 2.798 | 3.288644e-5 | 5.999914 | -7.915330 | 9 712.7 |
| sparse direct | 2 201.3 s | 4.403 | 3.289298e-5 | 5.999914 | -7.911161 | 9 716.8 |
| ASPECT 2.5 | 397 s | n/a | 3.2886e-5 | 6.000008 | -7.7439 | 8 906.7 |

The two Julia arms agree to 0.05 % in `Nu_bot` and 0.04 % in `Vrms` at the final step, which
is the point of the A/B: the linear path does not move the early trajectory. The ASPECT row
is not an A/B. Its 9 553 cells and 90 918 Stokes dofs are its own mesh, its wall time is a
different code, and the bottom flux is still in the startup oscillation. Read the top flux
agreement (0.002 %) and the picture, not the bottom row.

The video pipeline is `scripts/three_row_video.py`, documented in
[VIDEOS.md](VIDEOS.md) (the VIDEOS volume). It reads either a run's `frames/` or
an ASPECT `solution/` directory, stacks N rows at 1920 x (N*720), forces one shared colour
scale over all rows, picks the nearest frame per target time and never synthesizes a frame
in the one-column mode. The 500-step three-arm master is
`out_julia/verify100_render/verify100_3row.mp4`, 1920 x 2160 at 10 fps over
`[1e-6, 3.288644e-5]`. The one-row MG video and the two-column MG-versus-ASPECT video come
from the same script.

The caveat is the window. **500 steps from `t = 0` end at `t = 3.29e-5`.** The production E
run ends at `t = 8.56e-3`, 260x later, and `Nu_top` climbs from its conductive value 6 to
18.3 over that span. At the end of the three-arm window `Nu_top` is still 6.0. The video
therefore verifies plumbing, the selector, the two linear paths and the renderer on the
startup transient. It does not verify a statistical state, and it is not a substitute for
the E-versus-G comparison or the ASPECT window in VALIDATION §6.

## 5. How to run things

Everything runs from WSL. Define the interpreter once:

```bash
J=/home/szh/opt/julia-1.12.7/bin/julia
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=2   # timings; CI uses this
```

From Windows, wrap a command as `wsl.exe -d Ubuntu-24.04 --cd <repo> bash -lc '<cmd>'`, with
`<repo>` the directory where you cloned the repository. Production runs use `--threads=8` where the script
takes it; the timing protocol uses one Julia thread and `BLAS=1`.

Smoke and short runs. **Always set `CASE_OUT` to a scratch path.** The card writes into
`out_julia/case_realistic` when the variable is unset, which is the directory an earlier
spin-up used; a two-step smoke launched without the variable propagating overwrote that
directory's `history.csv`, its first frames and its frame metadata on 2026-11-12.

```bash
CASE_OUT=out_julia/case_realistic/smoke CASE_STEPS=10 $J --project=julia \
  julia/scripts/case_realistic_2d.jl smoke
CASE_OUT=out_julia/case_realistic/short CASE_STEPS=150 $J --project=julia \
  julia/scripts/case_realistic_2d.jl short
```

The driver has seven modes: `smoke`, `short`, `spinup`, `resume`, `freeze`, `wallrefine` and
`symseed`; the last three take a checkpoint path as the second argument. Background runs
need `setsid`; a plain `&` inside `wsl -e bash -lc` is killed when the launching shell
exits, which was measured. Since 2026-11-13 the production policy is the default on the
card: `:balanced`, gate on, `symmetry=:none`, `coarsen_frac=0.05`, `coarsen_theta_tol=1e-2`,
`wall_min_level=5`, `max_leaves=10000`. For a production spinup, only `CASE_OUT`,
`CASE_STEPS` and, for the bitwise arm, `CASE_LINEAR=direct` still need setting. A checkpoint
written before that date needs its own `CASE_AMR`/`CASE_SYMMETRY`/`CASE_AMR_GATE` back.

```bash
# production spin-up, checkpoint every 200 steps
CASE_OUT=out_julia/case_realistic_G CASE_STEPS=20000 setsid nohup $J --project=julia \
  julia/scripts/case_realistic_2d.jl spinup \
  > out_julia/case_realistic_G.log 2>&1 < /dev/null &

# the bitwise arm: add CASE_LINEAR=direct
CASE_OUT=out_julia/case_realistic_E CASE_STEPS=20000 CASE_LINEAR=direct \
CASE_AMR=classic CASE_SYMMETRY=x CASE_AMR_GATE=0 setsid nohup $J --project=julia \
  julia/scripts/case_realistic_2d.jl spinup \
  > out_julia/case_realistic_E.log 2>&1 < /dev/null &

# continue any checkpoint, or freeze the mesh for the statistics phase
CASE_OUT=out_julia/case_realistic_G $J --project=julia \
  julia/scripts/case_realistic_2d.jl resume out_julia/case_realistic_G/spinup.h5
CASE_OUT=out_julia/case_realistic_G $J --project=julia \
  julia/scripts/case_realistic_2d.jl freeze out_julia/case_realistic_G/spinup.h5
```

`resume` keeps the checkpoint's AMR cadence; `freeze` rewrites only `amr_every` to 0 and
continues. Neither rewrites the policy, the gate or the coarsening fractions, so the
environment must repeat every stored knob that differs from the current default or the guard
refuses the restart. A resume whose `CASE_STEPS` does not exceed the checkpoint step runs
zero steps; the driver reports that and writes nothing.

Renderer and watchdog, each in its own process:

```bash
setsid nohup /home/szh/dedalus-env/bin/python scripts/live_render.py \
  --dir out_julia/case_realistic_G --interval 3 --gif-every 12 --max-per-poll 20 \
  > out_julia/case_realistic_G/render.log 2>&1 < /dev/null &

setsid nohup bash scripts/watchdog_case.sh out_julia/case_realistic_G \
  >> out_julia/case_realistic_G/watchdog.out 2>&1 < /dev/null &
```

The watchdog checks every 120 s whether the case process is alive, whether the log ends with
the driver's `DONE` record, and whether the newest readable checkpoint is more than 400 steps
behind the log or frames. When all three checks say restart, it copies the checkpoint to
`.bak` under `flock` and relaunches `resume` (or `freeze` when the checkpoint has
`amr_every=0`) with the same `CASE_*` knobs. It refuses a second instance per case
directory. It forwards thirteen knobs, including `CASE_WALLMIN=5`, `CASE_WALLSTRIP=0.1`, the
coarsening pair `CASE_COARSEN`/`CASE_THETATOL`, and the **three AMR keys
`CASE_AMR`, `CASE_SYMMETRY`, `CASE_AMR_GATE`**. Those three default to `classic / x / 0`,
which is the checkpoint-era card, so an E restart needs no extra environment. A run started
under the new defaults must export all three. A G checkpoint must export
`CASE_AMR=balanced CASE_AMR_GATE=1 CASE_SYMMETRY=none`; F's pre-gate checkpoint needs
`CASE_AMR=balanced` and keeps the other two defaults, because its config has no gate key. If
the three do not match what the checkpoint records, the restart guard refuses the resume.

The restart guard is a provenance check with a structural blind spot. On resume it compares
every key in the checkpoint's own `run_config` against the current one and skips a key that
is absent from the file. **It compares only the keys a checkpoint carries.** A checkpoint
written before `amr_policy` and `amr_coarsen_gate` existed does not carry them and cannot be
compared on them. `stokes_scaling` is the one absent-means-`none` special case. The balanced
cooldown is state, not configuration: it is persisted and restores bitwise, and an old
checkpoint without it loads an empty cooldown. `freeze` rewrites only `amr_every`;
`wallrefine` rewrites only `wall_min_level` and `wall_strip` and redistributes leaves;
`symseed` rewrites only `symmetry`. The source checkpoint is never modified. The final
checkpoint carries `velocity_stale` when its velocity is the diagnostic re-solve rather than
the trajectory state. Keep a checkpoint past any step that failed: relaunching from the same
one is a treadmill.

Disk: `/mnt/e` is 1.4 T with **361 G free (74 % used)**. A 10 000-leaf checkpoint is about
1.7 MB, but frames, renders and case directories accumulate for weeks. The failure mode
after a 12-hour run is a full disk, and it looks like a solver error in the log.

## 6. What to do next

The matrix-free path is now the faster linear solver at the production size. It beats a
direct solve that actually reaches the 1e-6 tolerance by 1.30 to 2.31x across the measured
cells, and by 2.49x for the shipped 71-iteration row against the eight-thread direct column.
That changes the shape of the remaining work.

1. **Mixed precision inside the apply.** The only unmeasured lever with double-digit
   potential: 1.2 to 1.8x on the apply, which is 5 to 12 % of the step. It cannot be bitwise,
   so the decision is whether to carry two verification bases. Estimate three to five days.

2. **A better-conditioned `S1`.** The two `chol(S1)` solves are 1.90 ms per iteration, 13.2 %
   of the eight-thread iteration, against a 1.16 ms two-pass DRAM floor. The recoverable
   0.74 ms is 5.1 % of the iteration. An additive product would also let standard GMRES drop
   the `Z` basis (0.5 to 0.6 % of the iteration, 45 MB). This is a redesign of the Schur
   block, not a knob: the cheap routes measured 0.43x and 0.30 to 0.34x.

3. **The 3-D question.** Everything measured here is 2-D at 69 000 to 87 000 saddle dofs. A
   p=2 hexahedral leaf has 27 nodes against 9, so the working set, the 20 480-dof serial
   threshold, the level dof counts and the `S1` conclusion (7 424 rows here against ASPECT's
   A22 at 1e5 to 1e6 rows) are all 2-D numbers. Measure, do not extrapolate.

4. **A larger leaf budget.** The premise is that the margin grows with size. The two measured
   states do not order that way: 2.66x on `F` at 8 188 leaves and 2.06x on `P1` at 10 000.
   Test 40 000 leaves with `bench_direct_vs_mg.jl` before quoting a scaling law.

The AMR-aware coarse space is a later item, two to six weeks. Galerkin coarsening takes `F`
from 69 to 61 iterations and `P1` from 65 to 46, the plateau is 40 and 27 iterations (0.58
and 0.56 s, 39 % and 31 % of the warm solve), and a perfect coarse space bounds the warm
solve at 1.93x and 1.65x. A conservative 20 % iteration cut is 0.23 and 0.27 s. It only pays
after the first four.

The closed list, each with the number that closed it:

| option | number that closed it |
|---|---|
| LU reordering | 0.1 to 0.3 s (inference), 4 to 11 % of the step; below the 0.3 to 0.5 s bar |
| assembly rewrite | 0.15 to 0.25 s, 5 to 9 %; below the bar, and the 6.65 s premise was a cold-start subtraction |
| Chebyshev smoothing | diverges: `F` 0.97, `P1` 0.99 after 500 iterations |
| slice-wise cache blocking | -2.0 to +1.0 % |
| small-mode deflation | 83 to 79 iterations at best, at most 5 %; the `k=16` wall clock is worse, 1.230 against 1.164 s |
| `S1`-based large-mode deflation | 2.6 to 3.2x the whole solve; subspace defect 5 to 100; best row 11 % slower |
| standard GMRES | 196 iterations on `F`, `|du|/|u| = 3.58e-6` |
| level-scheduled `S1` solve | 0.43x; 1 005 forward levels, the widest 0.9 us of work against a 7.7 us phase |
| inexact CG or IC(0) `S1` solve | 0.30 to 0.34x; break-even 0.32 ms, IC(0) is 17.2 to 22.5 ms per solve |
| `bfbt_s1=:mass` | 500 iterations at `relres` 0.99 |

## 7. Rules of engagement

These are the habits this project paid for. They are cheap to follow and expensive to
relearn.

1. **Same-session A/B only.** Absolute seconds move 5 to 12 % between runs and 2.6 % between
   sessions at the same commit. A speed claim needs both arms in one session on one machine,
   and the deliverable is the ratio.

2. **Verify bitwise against a git archive.** Unpack the baseline with `git archive HEAD` into
   `/home/szh/ms_base`, copy `julia/Manifest.toml`, run the same script under both projects,
   and compare FNV-1a over `theta`, `ux`, `uz`, `pv` and the 2b/blob hashes. A refactor that
   does not change arithmetic has to reproduce them bit for bit. The pins are
   `d22fdab099ae84e0` for 2b and `45ac8acd099adadc` for blob.

3. **Small commits, explicit pathspecs.** Stage the one file with `git add -- docs/HANDOFF.md`
   and commit it with the same pathspec, `git commit -- docs/HANDOFF.md`. The tree carries
   untracked scripts, large untracked data files and case directories; `git commit -a` or
   `git add -A` sweeps them in.

4. **Do not "fix" load-bearing historical quirks in a bitwise stage.** The AMR budget pass
   pushes blocked child keys in a way that reads like a bug; deleting that push changes the
   trajectory in 6 of 60 random cases. Keep the quirk, pin it with a gate, and change it only
   in a stage that is allowed to change results.

5. **Documents quote measured numbers only.** The "5.6 s assembly" survived several documents
   before a warm re-measurement showed 0.35 to 0.45 s, and the composite 20-step runaway could
   not be reproduced and is now marked unverified. The full-suite pass count is another number
   with a recorded run but no measurement at this HEAD. If a number is needed, measure it or
   say it is unknown.

6. **Check the disk before long runs.** A 10 000-leaf checkpoint is small; the frame stream is
   not. A 20 000-step run at the production cadence costs about 4 GB. Check free space and
   prune `render/` before the next arm.

7. **A converged residual does not certify a preconditioner change.** A deflated `S1` reached
   `relres` 7.2e-7 with a pressure error 20 to 90 times the baseline, 1.0e-4 against
   1.2e-6. Gate every preconditioner change on `|du|/|u|` and the gauge-removed `|dp|/|p|`
   against `linear=:direct, stokes_scaling=:equilibrated` in the same process. Never on
   `relres`.

8. **A default flip is staged.** Pin the dependents, flip the default, run the full suite
   twice, then document. The AMR flip did exactly that: Stage 0 pinned 28 callers in 20 gates
   and 8 scripts (65 insertions against 8 deletions), Stage 1 flipped the driver kwarg and
   added G229, Stage 2 re-baselined the one canonical baseline that runs AMR, Stage 3 closed
   the guard gaps. The suite read 133 566 pass / 2 fail / 2 broken before and 133 579 pass /
   2 fail / 2 broken after, the difference being G229's 13 assertions. The linear flip
   followed the same shape: Stage 0 pinned the direct-path numbers (`42ac0f8`), Stage 1 moved
   the kwarg (`2453f16`), G230 pins the default. A flip that skips the pinning stage moves
   numbers nobody can attribute.
