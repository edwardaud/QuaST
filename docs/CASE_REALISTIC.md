# CASE_REALISTIC.md: a composite-rheology convection case

The case is a 2.5 x 1 box of fluid with free-slip walls, T = 0 at the top and
T = 1 at the bottom, internal heating H = 10, a nominal Rayleigh number of
1e5, a composite diffusion-dislocation rheology, and two phase transitions
with latent heat.  It is the first case in this repository built to look like a
piece of mantle convection instead of a benchmark: the viscosity spans seven
orders of magnitude, the lid is stiff and cold, the interior is hot and weak,
and the two transitions move heat in opposite directions.  Yielding and strain
weakening are off.

Two production runs exist.  The E run used the classic AMR policy and finished
20000 steps at t = 8.5598e-3 with Nu_top = 18.289, Nu_bot = -2.7636,
Vrms = 913.7, meanT = 1.2620; its mesh had been effectively frozen since step
15815.  The G run used the balanced policy with `symmetry=none` and the
coarsening gate on, and was stopped at step 15195, t = 4.340e-3, with
Nu_top = 16.597, Nu_bot = -2.7195, Vrms = 1113.2; its wall rows had reached
level 7 and its Nu_top was still rising.  Neither run is a converged
statistical steady state, and section 6 says what that rules out.

This document is the case record: the card, the protocol, the measured results,
the diagnostics that matter, and the operating notes.  The long derivations and
the failure histories live in the companion documents: `docs/AMR.md` for the
policy internals (sections 3 to 6), `docs/PERFORMANCE.md` for what a step costs
(sections 1 and 2), `docs/BUG_LEDGER.md` for the defects and their fixes
(sections 5, 8, 9 and 10), `docs/VALIDATION.md` for the benchmarks, the Nusselt
definitions and the gates (sections 2 to 8), and `docs/PHYSICS.md` for the
equations and the rheology (sections 1, 2 and 4).

| file | what it is |
|---|---|
| `julia/scripts/case_realistic_2d.jl` | the card, the frame writer, and the run modes `smoke`, `short`, `spinup`, `freeze`, `wallrefine`, `resume`, `symseed` |
| `julia/scripts/calib_realistic_2d.jl` | the derivation of the two composite prefactors (section 1.2) |
| `scripts/live_render.py` | the sidecar that turns frames into PNG panels, `live.gif` and `live.html` |
| `scripts/watchdog_case.sh` | the supervisor that restarts a dead long run from its newest checkpoint |
| `docs/CASE_REALISTIC.md` | this file |

**Run artifacts.** The `out_julia/...` paths in this document name run directories,
logs, checkpoints and frame dumps from the development tree, and the
`aspect_runs/realistic/output-*` directories and `run_*.log` files name ASPECT runs
from the same tree. They are cited so that a measured number can be traced to the
record that produced it; they are not part of this repository, which ships the
card, the scripts and the `aspect_runs/realistic/` parameter files, analysis
scripts and plugin source. Where a command below names an `out_julia/...` path,
any `CASE_OUT` of your own works instead.

## 1. The card

Every number in this section is a literal in `case_realistic_2d.jl`.  The run
does not read the calibration script, so the card is self-contained.

### 1.1 Geometry, walls, heating

The domain is `[0, 2.5] x [0, 1]`, built from 5 x 2 base cells on a quadtree
with square leaves.  All four walls are free slip: no normal flow, free
tangential flow, no external shear.  The temperatures are fixed at `T_top = 0`
and `T_bot = 1`, and the initial temperature is the conductive base state plus a
small anomaly.

The heating enters through that base state.  Write `z` for height and `H` for
the volumetric heating number; the conductive profile is

```
Tbar(z) = (1 - z) + (H/2) z (1 - z),
dTbar/dz = -1 + (H/2)(1 - 2z),
lap(Tbar) = -H.
```

The first term is the linear profile from 1 at the bottom to 0 at the top, and
the second is the parabolic bump that internal heating produces.  The physical
temperature solves `dT/dt + u.grad T = lap T + H`.  The code evolves the anomaly
`theta = T - Tbar`; substituting `T = Tbar + theta` and using `lap Tbar = -H`
leaves

```
dtheta/dt + u.grad theta = lap theta - u_z dTbar/dz,
```

so the heating source and the base curvature cancel exactly and the assembly
carries no explicit source term.  With `H = 10` the mid-depth base temperature
is `Tbar(0.5) = 1.75`, the volume mean of the base state is
`1 + H/12 = 1.3333`, and the conductive top gradient is `-(1 + H/2) = -6`.

The Nusselt numbers use the same base gradient as their scale:

```
Nu_top = -(dT/dy)|top / (1 + H/2),      Nu_bot = -(dT/dy)|bot / (1 - H/2).
```

Both are 1 for the purely conductive state, which is what makes them
dimensionless fluxes rather than raw gradients.  A steady convecting state
satisfies `Nu_top - Nu_bot = H`; the departure from that identity is the
enthalpy storage discussed in section 4.3.

### 1.2 Rheology and calibration

The law is `eta_law=:composite` with `comp_rule=:isostress`, the partition
ASPECT uses.  Two mechanisms deform the material: diffusion creep, which is
linear in stress, and dislocation creep, which is power-law.  In inverse form
the branch viscosities are

```
inv_d = A_d exp(-E_d / T_nd),
inv_s = A_s exp(-E_s / (n T_nd)) edot^((n-1)/n),
```

where `T_nd = T + 0.5` is the absolute temperature in units of the temperature
drop (the surface sits at half a drop), `edot = max(edot_II, 1e-12)` is the
second invariant of the strain-rate tensor with a floor, `E_d = 12` and
`E_s = 17` are the activation energies divided by `R dT` (their ratio 1.42 is
the ASPECT 530/375 kJ/mol pair), `n = 3` is the dislocation stress exponent, and
`A_d`, `A_s` are the two prefactors that the calibration produces.  The
isostress rule makes both mechanisms carry the same stress, so the strain rate
splits between them.  With `x = eta / eta_s` the split solves the scalar
equation

```
k x + x^n = 1,     k = inv_d / inv_s = eta_s / eta_d,
```

and the effective viscosity is `eta = x eta_s`.  The root is unique for
`k > 0`, and the two limits are the pure mechanisms: `k -> 0` gives `x -> 1`
(dislocation only), `k -> infinity` gives `x -> 0` (diffusion only).  The card
clamps `eta` to `[1e-4, 1e4]`; the clamp is a solver safety net (section 1.5),
not part of the constitutive law.

The prefactors solve two conditions:

1. `eta(T = 0.5, z = 0.5, edot_ref) = 1`, which fixes the common scale;
2. `eta(T = 0.1) / eta(T = 0.9) = 1e4` at the same point, which fixes the branch
   ratio `r = eta_d / eta_s`.

Both need a strain rate, and `edot_ref` is measured rather than guessed: a
constant-viscosity precursor of the same case (same box, tree, seed and heating,
`Ra = 1e5`, 120 steps, `eta = 1`) gives `Vrms = 6.96e2`, `max|u| = 1.66e3`,
`Nu_top = 37.2`, and a volume RMS second invariant of `edot_ref = 1.65e3`
(median 1.10e3, maximum 7.20e3).  A constant-viscosity precursor keeps
`edot_ref` independent of the law being calibrated.  Bisection in `log r` gives
`r = 0.129720`, then

```
comp_As = exp(E_s/n) edot_ref^(-(n-1)/n) x = 2.6796419238e-1,
comp_Ad = exp(E_d) eta_s(0.5) / r           = 1.6240183119e5.
```

Checked against the public evaluator, `eta(0.5, z=0.5, edot_ref)` comes out as
`1.000000000000000` and the ratio as `9.999999999999991e3`, a relative deviation
of 9.1e-16.  At the calibration point the dislocation branch carries about 7.7x
the diffusion viscosity, so the law is genuinely mixed rather than sitting in a
one-mechanism limit.  The rate dependence adds a second factor on top of the
thermal contrast: at `edot_ref/100` the viscosity ratio over `[0.1, 0.9]` is
8.7e4, and at `1000 edot_ref` it is 2.6e2.  A clamp census at `z = 0.5` over
`T` in `[0, 2.6]` clips 0 of 261 points with `[1e-4, 1e4]` and 75 of 261 with
`[1e-3, 1e5]`, which is why the wider window is not used.

One consequence deserves stating up front.  The calibration point is
`T = 0.5`, so `eta = 1` sits in the cold lid, while the convecting interior runs
at `T` near 1.3 and has `eta` around `1e-3` to `1e-2`.  The nominal `Ra = 1e5`
therefore describes the reference viscosity, and the interior Rayleigh number is
of order `1e7` to `1e8`.  The flow is fast (`Vrms` around 1e3), the time step is
CFL-limited to a few times `1e-7`, and an overturn time `1/Vrms` is about
`1e-3`.  This was a deliberate reading of the brief, which specified the
`T = 0.5, z = 0.5` point; section 6 states what it means for the numbers.

### 1.3 Phase transitions and latent heat

Two horizontal transitions represent the 410 and 660 km discontinuities.  With
`depth = 1 - y`, a reference density `rho0 = 3300 kg/m^3`, a thermal expansion
`alpha0 = 3e-5 1/K`, a temperature drop `dT = 1000 K` and a box depth
`H_box = 2900 km`, the buoyancy number of a density jump is

```
B = drho / (rho0 alpha0 dT) = drho / 99 kg/m^3,
```

and the nondimensional Clapeyron slope is

```
slope = (dP/dT) / (rho0 g H_box) * dT.
```

For the 410, `dP/dT = +3.0 MPa/K` gives `slope = +0.032`; the 660 uses the same
magnitude with the opposite sign.

| | 410 | 660 |
|---|---|---|
| `z0` | 0.87 | 0.78 |
| `width` | 0.025 (72 km) | 0.025 (72 km) |
| `slope` | +0.032 | -0.032 |
| `drho` | +180 kg/m^3 | +300 kg/m^3 |
| `B` | 1.8182 | 3.0303 |
| `Tref` | 0.5 | 0.5 |

The phase fraction is a tanh of the deviation,
`Gamma = 0.5 (1 + tanh(((z0 - z) - slope (T - Tref)) / width))`, so a hot
upwelling displaces the transition downward at the 410 and upward at the 660.
The buoyancy term is `-Ra B (Gamma(T) - Gamma(Tad))`.  With those signs an
upwelling gains buoyancy at the 410 and loses it at the 660, the textbook
behaviour of an exothermic and an endothermic transition.

The latent heat is derived from the same two numbers, not fitted.  In the code
the Clapeyron slope is first converted back to SI,
`gamma = slope rho0 g H_box / dT` in Pa/K, then

```
dS_A = gamma drho / rho0^2                 (J/(kg K)),
latent_i = -T_nd,i dS_A,i / cp,
```

where `T_nd,i` is the base-state absolute temperature at the transition in
units of `dT`, and `cp = 1000 J/(kg K)`.  The minus sign is the convention of
this code: it writes the term as `-latent DGamma/Dt` where ASPECT writes
`+T dS_A DGamma/Dt`.  Evaluating the base state at the two depths gives
`Tbar(0.87) = 0.6955` and `Tbar(0.78) = 1.078`, so `T_nd = 1.1955` and 1.578,
and the coefficients are

| | `dS_A` | `latent_i` | physical reading |
|---|---|---|---|
| 410 | +49.66 J/(kg K) | **-0.059364** | heats a downwelling |
| 660 | -82.76 J/(kg K) | **+0.130596** | cools a downwelling |

The apparent heat capacity is `1 + sum_i latent_i dGamma_i/dT`.  The derivative
`dGamma/dT = -(slope / (2 width)) (1 - tanh^2)` has a maximum magnitude
`|slope| / (2 width) = 0.64`, so the factor stays in `[0.88, 1.12]` even if both
transitions act at full strength, and it never approaches zero.  The optional
`CASE_LATENT=global` reproduces the pre-2026-11-08 runs, which used one
`latent_L = 0.11` for both transitions; the derivation above is the default.

### 1.4 Initial condition

The seed is

```
theta0(x, y) = 0.05 sin(pi y) cos(2 pi x / Lx) + 0.02 sin(2 pi y) cos(4 pi x / Lx).
```

It vanishes on the top and bottom walls, as required for an anomaly, and it is
symmetric about `x = Lx/2`.  There are no random numbers anywhere: the card, the
seed and a fixed step count make a run a deterministic sequence.  The evolved
field does not stay symmetric, and the G run does not impose symmetry (the
production card uses `symmetry = :none`, section 1.5).

### 1.5 Numerics

| | card default (production policy) |
|---|---|
| polynomial order | `p = 2` (Q2 velocity, Q1 pressure) |
| tree | `max_level = 7`, leaf `1/256`; `max_leaves = 10000` (the script default) |
| AMR cadence | `amr_every = 5` |
| marking | `:error_fraction`, `refine_frac = 0.2` |
| coarsening | `coarsen_frac = 0.05`, `coarsen_theta_tol = 1e-2` |
| policy | `amr_policy = :balanced`, `hysteresis = 1`, `symmetry = :none` |
| coarsening gate | `amr_coarsen_gate = true` |
| wall rule | `wall_min_level = 5`, `wall_strip = 0.1` |
| transfer | `:nodal` |
| time step | `cfl = 0.5`, `dt0 = 1e-6`, `dtmax = 5e-5`, `dtmin = 1e-9`, growth 1.05 |
| advection | `:skew`, `ev = 0.11`, `ev_cmax = 0.052`, `ev_alpha = 2` |
| nonlinear | `:picard`, `maxpic = 1`, `pic_tol = 1e-6` |
| linear | `:auto` (`:direct` for G), `stokes_tol = 1e-3`, `mg_tol = 1e-8` |
| stokes scaling | `:equilibrated` in the production and verification cards; `:none` in the bare script (§2.5, `docs/SOLVER.md` §3.6) |
| mass form | `:boussinesq` |
| latent | `:per_phase`, implicit |

The policy, coarsening, gate, wall, symmetry and leaf-budget rows are the
script's defaults since 2026-11-13, so a new `spinup` runs the production policy
with no `CASE_*` variable at all (`julia/scripts/case_realistic_2d.jl`). The
solver scaling is the exception: the script keeps `:none`, and the production
and verification cards export `CASE_STOKES_SCALING=equilibrated` (§2.5). Since
the same date the library default of `run_semquad_simulation` is also
`:balanced` (commit `4584e85`); `semquad_amr_passes!` keeps its own `:classic`
default, so the unit-level primitive and G208's tree pins are unchanged.  The
classic path is still available as `amr_policy=:classic` and is what reproduces
E.  `linear = :auto` is likewise the library default of
`run_semquad_simulation` since 2026-11-13 (commit `2453f16`), so the card's
`CASE_LINEAR=auto` is the same policy it already passed explicitly; `:direct`
stays available as the only bitwise path and `pic_mode=:newton` requires it
explicitly.  E ran
the same mesh budget, wall rule, time step and linear path with the historical
policy rows: `amr_policy = :classic`, `symmetry = :x`, no coarsening gate,
`coarsen_frac = 0.05`, `coarsen_theta_tol = 1e-2`.

Reproducing or continuing an E/G leg therefore means pinning those rows
explicitly.  The `run_config` guard compares `amr_policy`, `amr_coarsen_gate`,
`amr_hysteresis`, `symmetry`, `coarsen_frac`, `coarsen_theta_tol`,
`refine_thresh`, `coarsen_thresh`, `wall_min_level` and `wall_strip`
on restart (§2.2): a checkpoint written under the old defaults needs the
matching `CASE_*` values, and a pre-gate checkpoint needs an explicit
`CASE_AMR_GATE=0` because its stored config has no `amr_coarsen_gate` key and
the guard iterates over the checkpoint's keys (an absent key is not compared).
The three coarsening/threshold keys were added to `run_config` on 2026-11-13,
after the survey found `CASE_THETATOL` silently unguarded; old checkpoints
without them skip the check, new ones cannot change them on restart.
`freeze`, `wallrefine` and `symseed` rewrite one conflicting string each; they
do not rewrite the policy, the gate or the coarsening fractions.

The transition width is what fixes the tree.  A `tanh` with `width = 0.025` has
its largest derivative `1 / (2 width) = 20` in the temperature equation, and at
`max_level = 7` the leaf is `1/256 = 0.0039`, so the transition spans 6.4
leaves.  At `max_level = 6` it spans 3.2, which is marginal for the same
derivative.  At a fixed leaf budget, `p = 2` costs 9 velocity nodes per leaf
and `p = 4` costs 25, so the lower order buys roughly 2.8x the degrees of
freedom for the same cost; the wall rule then spends the saving where the
boundary layers are.  This is the same reasoning as the blob production card in
`docs/SOLVER.md`, and it is why the card is `p=2, max_level=7`.

The E and G arms chose `:direct` because the matrix-free multigrid path did not
converge on this non-uniform tree with the `eta_max` lid at the time; the
measured failure is 0.97 relative residual, recorded in `docs/PERFORMANCE.md`
section 3 and `docs/BUG_LEDGER.md` section 11.  The clamp `eta in [1e-4, 1e4]` is a
safety net that also sets the conditioning: with `eta_max = 1e6` the initial
contrast is 1.6e8 and the direct solve's residual is 4.1e-4 at step 0, while
`1e4` gives a contrast of 1.0e7 and a residual of 8.5e-10.  The tolerance
`1e-3` therefore guards against a real failure, not against round-off.

After the exact constrained Jacobi and leaf-local BFBT diagonals the current
default `CASE_LINEAR=auto` resolves to MG at this tree size -- 78 720
selector dofs on `F`, 95 308 on `P1`, both far above the 20 000 rule (the
threshold table was re-calibrated at `c7e0f55`; `docs/SOLVER.md` §4.4) -- and the card
passes `mg_tol=1e-8`, so the auto solve reaches `stokes_tol=1e-3` with five
orders of residual margin and zero fallbacks (§1.6, `docs/SOLVER.md` §4.4,
`julia/scripts/case_realistic_2d.jl`).  A new production run therefore does not
need `CASE_LINEAR=direct` unless the bitwise direct path is required.

### 1.6 The MG outer tolerance

The driver accepts a Stokes solve at `stokes_tol = 1e-3` and fails the step
above it; `linear=:auto` also uses that number as its fallback gate.  The
library default `mg_tol = 1e-10` is therefore seven orders below the residual
the driver accepts, and the measured production solve spends its last 12-20 %
of warm time on that excess.  The card passes `mg_tol = 1e-8` explicitly
(`MG_TOL` in `julia/scripts/case_realistic_2d.jl`), five orders below the gate;
the library `mg_tol` default is unchanged (the `linear` default flip does not
touch it).

Read-only copies of the three states (`F`, `P1`, and the 8 192-leaf-budget
`C`-arm checkpoint, 8 260 leaves after the wall rule), the card's solver kwargs
(`linear=:auto`, `auto_fallback_tol = stokes_accept_tol = 1e-3`), eight threads,
`BLAS=1`, warm best of three after a cold build.  The last two columns are
against the same solve at `mg_tol = 1e-10` (the reference row), pressure gauge
removed.  All rows were accepted by the driver (`res <= 1e-3`) with zero
`LAST_AUTO_FALLBACK` events and an auto choice of MG.

| state (leaves, selector dofs) | `mg_tol` | outer it | true res | warm s | du/u | gauge dp/p |
|---|---|---|---|---|---|---|
| `F` (8 188, 78 720) | `1e-10` (default) | 92 | 7.85e-11 | 1.608 | — | — |
| `F` | **`1e-8` (card)** | **81** | **8.61e-9** | **1.414** | **1.98e-8** | **3.26e-9** |
| `F` | `1e-6` | 71 | 6.75e-7 | 1.260 | 7.95e-7 | 2.27e-7 |
| `F` | `1e-5` | 66 | 8.56e-6 | 1.182 | 1.28e-5 | 1.49e-6 |
| `P1` (10 000, 95 308) | `1e-10` (default) | 108 | 8.73e-11 | 2.681 | — | — |
| `P1` | **`1e-8` (card)** | **91** | **9.44e-9** | **2.208** | **1.83e-8** | **7.86e-9** |
| `P1` | `1e-6` | 65 | 6.62e-7 | 1.703 | 9.84e-7 | 1.28e-6 |
| `P1` | `1e-5` | 59 | 8.83e-6 | 1.584 | 3.17e-5 | 2.09e-5 |
| `C` (8 260, 79 308) | `1e-10` (default) | 90 | 8.14e-11 | 1.524 | — | — |
| `C` | **`1e-8` (card)** | **67** | **6.99e-9** | **1.215** | **2.71e-8** | **4.07e-9** |
| `C` | `1e-6` | 57 | 6.62e-7 | 1.084 | 9.41e-7 | 2.65e-7 |
| `C` | `1e-5` | 51 | 7.82e-6 | 0.980 | 1.03e-5 | 3.37e-6 |

The `1e-8` row is the card's choice.  It keeps the true residual 5.0-5.2
orders below the gate and moves the velocity by at most 2.7e-8 and the
gauge-removed pressure by at most 7.9e-9, i.e. 78x and 111x inside the
documented MG accuracy class (`|du|/|u| <= 2.1e-6`, gauge-removed
`|dp|/|p| <= 8.8e-7`; `MG.md` §0).  The warm solve is 12-20 %
shorter than at the default: 0.19 s on `F`, 0.47 s on `P1` and 0.31 s on `C`.

The `1e-6` row is faster still -- 71/65/57 iterations and 22-37 % less warm
time -- but it moves `P1`'s gauge-removed pressure by 1.28e-6, above the
documented 8.8e-7 class, so it is not the card's tolerance.  `1e-5` fails the
velocity class on all three states (1.0e-5 to 3.2e-5) and is rejected too.  The
accuracy class is the one the existing `1e-6` benchmark rows were accepted at
(`MG.md` §0), not a new gate.

Reproduce with

```bash
julia --project=julia --threads=8 julia/scripts/bench_card_mg_tol.jl F 3
```

and the same command for `P1` and `C`.  The runtime spread between sessions is
the 4-6 % noted in `MG.md`; the iteration counts, true residuals and
field differences are deterministic.  G228 pins the card's declarations
(`STOKES_TOL`, `MG_TOL` and their forwarding), the same choice through the
`linear=:auto` path on a uniform L4 tree, and the accuracy class on a
hanging-node tree with a 1e4 viscosity contrast.  `mg_tol`, like `linear`, is a
solver setting and is not part of `run_config`, so existing checkpoints resume;
a run should still keep one setting for its whole trajectory.

## 2. Protocol

### 2.1 Spin-up, the frozen phase, and the statistics window

The intended protocol has two stages: a spin-up with AMR, then a statistics
window on a frozen mesh so that the mesh does not change under the averages.
`freeze <ckpt>` implements the second stage by rewriting the checkpoint's
`amr_every` to 0 and continuing.  The two production runs are both spin-ups:
E's classic policy froze its own mesh: 4000 AMR cycles, 725 with any change
(18.1%), the leaf count between 9982 and 9994 after step 12000 and exactly 9994
from step 15815 on.  The seven late cycles that fire are two-quartet merges
whose leaves are re-created in the same cycle, so the net mesh does not move.
G was stopped while its mesh was still trading leaves at the 10000 cap.

The window itself is defined in overturn times.  With `t_ov = 1/Vrms` and the
integrated `int Vrms dt`, E covers 6.9 overturns over its 20000 steps and G
covers 5.2 over 15195 steps.  A usable window needs block means of 100 steps
whose drift is below 2% over the last five blocks, and at least five overturn
times.  E's Nu_top satisfies the drift part by the end (about 1.5% over the
last 2000 steps), and its mean temperature does not: see section 4.3.
G was still climbing when it stopped.  So the statistics phase is specified,
implemented, and not yet occupied by either run.

### 2.2 Restarts and the guard

Every stage is a restart from a checkpoint, and the driver compares the stored
`run_config` string by string.  A mismatch aborts with the two values printed:

```
restart configuration mismatch in <file>:
  amr_every: checkpoint=5.0, current=0.0
  symmetry: checkpoint=x, current=none
  wall_min_level: checkpoint=0.0, current=5.0
```

The guard is the reproducibility mechanism.  Changing the physics, the AMR
policy, the mirror rule or the wall rule between stages requires a seed copy
that rewrites exactly the conflicting strings and copies the state verbatim:
`freeze` rewrites `amr_every`, `wallrefine` rewrites `wall_min_level` and
`wall_strip` and also refines the wall, and `symseed` rewrites `symmetry`.  The
source checkpoint is never modified.

Two pieces of state are not configuration and are therefore not compared.  The
`:balanced` cooldown is written as two integer tables (`cooldown_no_refine`,
`cooldown_no_coarsen`) plus a cycle attribute, and a resume restores it, so a
`:balanced` restart continues the same bitwise trajectory (gate G213).
`freeze` and `symseed` carry the cooldown;
`wallrefine` starts it empty because the leaf set changes.  The classic path
still has one caveat: a checkpoint written on an AMR step stores the zeroed
velocity, because the AMR pass clears `ux`/`uz` just before the checkpoint
statement, so a classic restart differs from the uninterrupted run at the 1e-5
level.  A checkpoint cadence that is not a multiple of `amr_every` (201, not
200) makes that restart bitwise.  The final checkpoint carries the
`velocity_stale` marker and the driver zeroes the restored velocity, which
removes the diagnostic re-solve from the restart state.

A checkpoint is written every 200 steps and at exit, so a killed run loses at
most 200 steps.  G lost 41 steps at its step-7641 failure and 2 steps at its
second restart.

### 2.3 Commands

```bash
JL=/home/szh/opt/julia-1.12.7/bin/julia

# production spin-up: the AMR policy, coarsening, gate, wall, symmetry and
# leaf-budget rows are the card defaults; the two solver rows below are the G
# arm's non-defaults (bitwise direct solver, equilibrated saddle)
CASE_OUT=out_julia/case_realistic_G CASE_STEPS=20000 \
  CASE_LINEAR=direct CASE_STOKES_SCALING=equilibrated \
  setsid nohup $JL --project=julia julia/scripts/case_realistic_2d.jl spinup \
  > out_julia/case_realistic_G.log 2>&1 < /dev/null &

# continue any checkpoint: repeat the knobs it stored.  The defaults have been
# the production set since 2026-11-13, so a checkpoint written before that date
# needs its own rows back, e.g. an E checkpoint needs
#   CASE_AMR=classic CASE_SYMMETRY=x CASE_AMR_GATE=0 CASE_LINEAR=direct
# and a G checkpoint from the last leg needs CASE_STOKES_SCALING=equilibrated
$JL --project=julia julia/scripts/case_realistic_2d.jl resume \
  out_julia/case_realistic_G/spinup.h5

# freeze the mesh and run the statistics phase
$JL --project=julia julia/scripts/case_realistic_2d.jl freeze \
  out_julia/case_realistic_G/spinup.h5
```

`setsid` is required for a background launch: a plain `&` under
`wsl -e bash -lc` is killed when the launching shell exits (the first attempt
left no log file at all), while a `systemd-run --user` transient unit works.
`resume` keeps the checkpoint's AMR cadence and `freeze` rewrites it to 0;
neither changes the card, so the environment must repeat every stored knob that
differs from the current default or the guard refuses the restart.

### 2.4 Watchdog

`scripts/watchdog_case.sh <case_dir>` checks every 120 s and appends one line to
`<case_dir>/watchdog.log`.  It scans `/proc` for a julia process whose command
line contains `case_realistic_2d.jl` and the case directory (or whose environment
carries `CASE_OUT=<case>`), ignores zombies, and reports `running`.  When the
process is gone it looks for a `DONE` line in the log, then picks the newest
readable checkpoint (`.h5` or `.h5.bak`, by the HDF5 `step` attribute and mtime,
requiring `step`, `theta` and `leaves`) and refuses to resume one that is more
than 400 steps behind the newest log or frame line.  A restart is taken under
`flock`, after a 90 s launch grace, with the checkpoint copied to `.bak` first,
and a second watchdog on the same directory is refused by an instance file.  It
forwards thirteen `CASE_*` knobs (steps, leaves, coarsen, theta tolerance,
linear, latent mode, max level, wall minimum, wall strip, output, and the three
AMR keys `CASE_AMR`, `CASE_SYMMETRY`, `CASE_AMR_GATE`), and any other card key
travels in the watchdog's own environment.  The three AMR keys default to the
pre-2026-11-13 values (`classic / x / 0`) on purpose: the watchdog resumes
existing checkpoints, not new runs, so an E/classic checkpoint restarts with no
extra environment, while a run started under the new card defaults must export
all three (`CASE_AMR=balanced CASE_SYMMETRY=none CASE_AMR_GATE=1` for G; F's
pre-gate `:balanced` checkpoint needs `CASE_AMR=balanced` and keeps the other
two defaults) or the restart guard refuses the resume.  `CASE_STOKES_SCALING` is
not one of the thirteen forwarded keys and is part of `run_config`: a G
checkpoint written with `equilibrated` needs it in the watchdog's own
environment, or the resume is refused.  `CASE_THETATOL` is
forwarded next to `CASE_COARSEN`; since `coarsen_theta_tol`, `refine_thresh` and
`coarsen_thresh` are in `run_config` (2026-11-13), a restart that changes the
tolerance is now refused instead of silently changing the classic protect set.
The policy rule is provenance, not preference: never let the watchdog switch
the policy of an existing trajectory.  An E checkpoint restarts as `:classic`
(the defaults above); F and G checkpoints carry `:balanced` and must be
restarted with their recorded gate and symmetry keys.  A balanced continuation
of E is a **new arm** -- new output directory, new frame stream -- not a
resume: it reproduces neither E's step-20000 numbers nor a native balanced
run's.

The G run is the case study.  It ran 17.45 h across three processes.  The
watchdog restarted it at 20:52 from the step-7600 checkpoint after the
step-7641 `STOKES-FAIL` (the shift-ladder failure in `docs/BUG_LEDGER.md`
section 10), and again at 00:19 from step 10800.  The first restart overwrote
frames 7601 to 7641 (their `frame_meta.csv` rows survive); the run continued to
15195.  To stop a supervised run, kill the watchdog first and the julia second,
so the watchdog cannot relaunch it.

### 2.5 The three-arm verification

`julia/scripts/verify100_case.sh <mg|direct|auto> [steps]` runs one arm of the from-scratch comparison.  It starts at
`t = 0` on the card's own initial condition, not from a G checkpoint, and writes every AMR knob explicitly so the arms
cannot drift apart: `CASE_AMR=balanced`, `CASE_AMR_GATE=1`, `CASE_COARSEN=0.05`, `CASE_THETATOL=1e-2`,
`CASE_WALLMIN=5`, `CASE_SYMMETRY=none`, `CASE_MAXLEAVES=10000`, `CASE_HYST=1`, `CASE_EV=0.11` and
`CASE_STOKES_SCALING=equilibrated`.  `RUN_ID` (default `verify100`) names the output root
`out_julia/${RUN_ID}_<arm>` and the log; `CASE_FRAME` sets the frame stride, and its default 0 means the card's 5.  The
script refuses an output directory that already exists and refuses to start while a `case_realistic_2d.jl spinup` is
running.

The 500-step arms write 101 frames: steps 1, 6, ..., 496 and the final state at 500.  Measured wall clocks, 8 threads,
one arm at a time:

| arm | wall | s/step | end `t_nd` | `Nu_top` | `Nu_bot` | `Vrms` |
|---|---|---|---|---|---|---|
| matrix-free MG | 1398.9 s | 2.798 | 3.2886e-5 | 5.999914 | -7.915330 | 9605.85 |
| sparse direct | 2201.3 s | 4.403 | 3.2893e-5 | 5.999914 | -7.911161 | 9609.90 |
| ASPECT 2.5 | 397 s (646 timesteps) | n/a | 3.2893e-5 | 6.00000 | -8.32288 (gradient) / -7.743888 (CBF) | 8906.68 |

The walls are the `[case] DONE` lines of `out_julia/verify100_mg.log` and `out_julia/verify100_direct.log`; the
`Nu` and `Vrms` values are frame 100 of each `frames/frame_meta.csv`. The ASPECT clock is in
`aspect_runs/realistic/run_verify100.log`, and its statistics are `output-verify100/statistics`.

The two Julia arms agree to 3.7e-8 relative in `Nu_top`, 5.3e-4 in `Nu_bot` and 4.2e-4 in `Vrms`; the 256x256
fields differ by rms 2.7e-4, 1.9e-3 of the anomaly's rms 0.142.  `docs/VALIDATION.md` §5.5 has the mesh-divergence
counts and the family caveats.  Against ASPECT, `Nu_top` agrees to 1.4e-5, `Nu_bot` is -5.2 percent in the gradient
family and `Vrms` is -7.3 percent.

The window is the startup transient, and that limits what the run can say.  500 steps from `t = 0` reach
`t_nd = 3.3e-5`.  The E run reaches 8.56e-3, 260 times later, and `Nu_top` climbs from 6.0 to 18.3 over that span
(section 3.1).  At frame 100 all three arms still read `Nu_top = 6.0`.  The run verifies the card wiring, the two
linear paths and the frame writer; it does not verify a statistical state.

The direct arm's 4.403 s/step is one core's time: UMFPACK is serial.  The MG arm uses the eight threads, and ASPECT
uses 6 MPI ranks.  The 1.57x direct/MG ratio is a wall-clock ratio on this card, not a solver rating.  The direct log
counts `scaled=365`, the MG log `scaled=0`: `CASE_STOKES_SCALING=equilibrated` engages only on the direct path, where
the unscaled health check fails.  The longer ASPECT arm reaches `t_nd = 3.2e-4` in 2362 timesteps at 6 ranks and
10 618 s (`aspect_runs/realistic/output-verify7000`), with `Nu_top` still 6.0.

`CASE_OUT` is always explicit in the script: it derives the path from `RUN_ID` and refuses an existing directory.  The
general rule of section 5 still applies to hand-run smokes: set `CASE_OUT` to a scratch path, and never to a directory
that holds a production history or frames.

## 3. Results

### 3.1 The E run

E is the classic-policy run: `max_leaves = 10000`, `coarsen_frac = 0.05`,
`coarsen_theta_tol = 1e-2`, `wall_min_level = 5`, `linear = :direct`, and
`amr_policy = :classic` with `symmetry = :x` (the card defaults before
2026-11-13).  It finished 20000 steps in
80878.9 s (4.044 s/step) with no Stokes-gate failure (`auto_fallbacks = 0`).
A watchdog restart of E must keep `CASE_AMR=classic CASE_AMR_GATE=0
CASE_SYMMETRY=x` (the watchdog's defaults do); a balanced continuation of E is
a new arm, not a resume.

| quantity | E, step 20000 |
|---|---|
| `t` | 8.5598e-3 |
| `Nu_top` | **18.2892** |
| `Nu_bot` | **-2.7636** |
| `Vrms` | **913.7** |
| `meanT` | **1.2620** |
| `Tmax` | 1.4338 |
| leaves / `N` | 9994 / 42177 |
| `eta` clipped | 0.76% |
| top / bottom wall level | L5 / L5 |
| `h_top` / `delta_top` | 1/64 / 0.0732 |
| `h_bot` / `delta_bot` | 1/64 / 0.1226 |

The early transient is not the late run.  `Nu_bot` averages -7.56 over the
first 2000 steps and settles near -2.8; `Nu_top` sits on the conductive value 6
until about `t = 1.5e-3`, rises through 7.9 at `2e-3`, 11.4 over steps
8001-10000, and peaks at 18.93 in the block 16001-16500 before drifting down to
18.27 over the last 500 steps.  The mean temperature falls monotonically from 1.3333
to 1.2620: the box is still cooling at the end, which is what section 4.3
measures.

### 3.2 The G run

G is the production policy: `:balanced`, `symmetry = :none`,
`amr_coarsen_gate = true`, `hysteresis = 1`, same 10000-leaf budget and wall
rule, `linear = :direct`.  It was stopped at step 15195 after 17.45 h
(4.13 s/step including the two restarts), with no `DONE` line.  G is unaffected
by the library-default flip (it already ran `:balanced`); its restarts need the
recorded `CASE_AMR=balanced CASE_AMR_GATE=1 CASE_SYMMETRY=none`.

| quantity | G, step 15195 |
|---|---|
| `t` | 4.340e-3 |
| `Nu_top` | **16.597** |
| `Nu_bot` | **-2.7195** |
| `Vrms` | **1113.2** |
| `meanT` | 1.3315 |
| `Tmax` | 1.5386 |
| leaves / `N` | 10000 / 43185 |
| `eta` clipped | 2.9% |
| top / bottom wall level | L7 / L7 |
| `h_top` / `delta_top` | 1/256 / 0.0866 |
| `h_bot` / `delta_bot` | 1/256 / 0.1607 |

The mesh is the story here.  Of 3080 AMR cycles, 3058 changed the mesh (99.3%),
and every cycle in the last thousand changed it; after the growth phase the
leaf count sits at the 10000 cap.  The policy pays for the wall with the interior: 752714 leaf
refinements and 757766 quartet coarsenings over the run, with no face-level jump
above 1 in any cycle.  The balancer re-splits 0 to 6 quartets on a typical cycle
(mean 4.4 over the last thousand; the maximum over the run is 20), and the
coarsening gate refuses 2 to 29 quartets per cycle while the supply rule
extends past the deadband on 142 to 410 quartets.  Both wall rows reach L7 by
`t = 4e-3`; at step 7641 the top row was still L5.

### 3.3 E against G in the common early window

Both runs start from the same seed and part at the first AMR cycle (step 5), so
they are different trajectories.  Over `[0, 1.089e-3]`, sampled at 201 uniform
times by linear interpolation of the frame series, they agree closely:

| quantity | E | G | G - E |
|---|---|---|---|
| `Nu_top` | 6.0024 | 6.0016 | **-0.01%** |
| `Nu_bot` | -3.3721 | -3.4204 | **-1.43%** |
| `Vrms` | 1589.5 | 1607.7 | **+1.12%** |
| `Tmax` | 1.7842 | 1.7986 | **+0.81%** |

The Vrms number is the sensitive one: the window mean samples a decaying
transient, and a different sample grid moves it by a few tenths of a percentage
point.
The reading is that the AMR policy changes the mesh without biasing the early
solution at the 1% level.  The agreement does not survive: at `t = 4.338e-3` the
G run is 5.1% lower in `Nu_top` (16.592 against 17.483) and 54% higher in `Vrms`
(1135.9 against 737.7), with `Tmax` 2.0% higher.  The early window is the only
window in which the two mesh histories can be compared cleanly.

### 3.4 Cost

| run | steps | wall time | per step | state |
|---|---|---|---|---|
| E (classic) | 20000 | 80878.9 s (22.47 h) | 4.044 s | 9994 leaves, `N = 42177` |
| G (balanced, gate) | 15195 | about 17.45 h | 4.13 s | 10000 leaves, `N = 43185` |

The 200-step A/B arms at the same 10000-leaf card cost 5.37 to 5.62 s/step when
two arms ran concurrently, so the production figures above are the favourable
ones.  The warm step at this size is about 3.6 s, of which 3.1 s is the Stokes
solve, 0.31 s the energy assembly and solve, 0.14 s the work and dissipation
terms, and 0.04 s diagnostics (`docs/PERFORMANCE.md` section 1); the rest of the
production per-step time is the AMR pass, the frame diagnostics and checkpoint
I/O.  The gate itself added 0.03 s/step (0.6%) in the measured 200-step arm; the
AMR pass as a whole is a few tenths of a second per cycle.

## 4. Diagnostics that matter

### 4.1 Boundary layers and h/delta

The boundary-layer thickness used here is the x-averaged tangent construction:
with `T_mid` the horizontally averaged temperature at mid-depth and the wall
gradient taken from a second-order one-sided difference, `delta_top = |T_mid -
T_top| / |dT/dy|_top` and `delta_bot` the same at the bottom.  The mesh number is
the level of the wall leaf, `h = 1/(2 * 2^level)`.  At the end of each run:

| run | wall | level | `h` | `delta` | `h/delta` |
|---|---|---|---|---|---|
| E | top | L5 | 1/64 | 0.0732 | **0.213** |
| E | bottom | L5 | 1/64 | 0.1226 | **0.128** |
| G | top | L7 | 1/256 | 0.0866 | **0.045** |
| G | bottom | L7 | 1/256 | 0.1607 | **0.024** |

The working criterion is `h/delta <= 0.25`: E passes at the top only just
(0.213) and at the bottom with a factor of two (0.128), while G passes both with
room (0.045 and 0.024).  The pointwise wall flux is insensitive to the wall
element while `h/delta` stays near or below 0.2 (`docs/VALIDATION.md` section
3), so E's numbers sit at the edge of the converged range; the value of G's
finer wall is that it keeps them converged as the layers keep thinning.

### 4.2 Level histogram

| leaves | L2 | L3 | L4 | L5 | L6 | L7 | total |
|---|---|---|---|---|---|---|---|
| E, step 20000 | 0 | 90 | 544 | 5752 | 3448 | 160 | 9994 |
| G, step 15195 | 7 | 212 | 689 | 2603 | 3389 | 3100 | 10000 |

E's tree is the frozen classic state: the interior sits at L5 and L6, the top
wall is L5, and only 160 leaves reach L7.  G's tree has nearly twenty times as
many L7 leaves, both wall rows are L7, and the interior paid for it (L5 5752 to
2603, L6 3448 to 3389, and 7 leaves at L2 that E never has).  The histogram is
the mechanism behind the `h/delta` table above.

### 4.3 Energy budget residual

`history.csv` column 13 is the historical `closure`:

```
closure = Nu_top - Nu_bot - (H - Ibase) + (latent_src + latent_adv) / A,
```

where `Ibase = int u_z dTbar/dz` is the discrete violation of the divergence
identity, the two latent terms are `int sum_i latent_i dGamma_i/dz u_z` and
`int sum_i latent_i dGamma_i/dT u.grad(theta)`, and `A = 2.5` is the box area.
At a statistical steady state it vanishes.  It does not vanish here; the value
tracks enthalpy storage:

| | E, last 1000 steps | G, last 1000 steps |
|---|---|---|
| `closure` (mean) | **11.00** (sd 0.47) | **8.83** |
| `Nu_top - Nu_bot - H` | 11.04 | 8.83 |
| `-d(meanT)/dt` | 10.63 | 5.44 |

The residual is of order `H` in both runs and follows the mean-temperature
drift, which is the enthalpy term `-d(meanT)/dt` of the scheme's identity plus
the bookkeeping corrections in `docs/BUG_LEDGER.md` section 8.  The exact
residual of the quoted pointwise pair against the scheme's own identity is the
opt-in `resid_A` column, requested with `CASE_EXTDIAG=1` and documented in
`docs/VALIDATION.md` section 4; column 13 is an enthalpy-budget term and should
not be read as a heat-flux error.

### 4.4 The top boundary layer thins

As `Nu_top` climbs from 6 to 18, the top boundary layer thins by a factor of
3.8 and the fixed L5 wall becomes progressively under-resolved:

| E, `t` | `Nu_top` | `delta_top` | `h/delta` at L5 |
|---|---|---|---|
| 1.0e-4 | 6.00 | 0.280 | 0.056 |
| 2.0e-3 | 7.91 | 0.184 | 0.085 |
| 2.5e-3 | 10.22 | 0.141 | 0.111 |
| 4.0e-3 | 16.55 | 0.085 | 0.183 |
| 8.0e-3 | 18.31 | 0.073 | 0.213 |

G shows the same thinning (0.280 at `t = 1e-4` to 0.087 at `t = 4.3e-3`), but its
wall rule and balanced policy had taken both wall rows to L7 by `t = 4e-3`, so
its `h/delta_top` stays at 0.045.  This is the accuracy argument for the wall
rule: the quantity that agrees with ASPECT to about 1% is the one whose
resolution is being lost.

## 5. Operation notes

### Always set `CASE_OUT` for a smoke run

The card writes into `out_julia/case_realistic` when `CASE_OUT` is unset, which is the
same directory an earlier spin-up used. A two-step smoke launched without the variable
propagating through the shell wrote over that directory's `history.csv`, its first frames
and its frame metadata (2026-11-12). The production run's own record lives in its
`CASE_OUT` directory (`out_julia/case_realistic_E` holds the 20 000-step E history and is
intact), so a smoke is only dangerous if it lands on a directory that holds a record you
care about. Set `CASE_OUT` to a scratch path for every smoke, resume and wall-refinement
seed.


### 5.1 Environment variables

The case script is configured entirely through `CASE_*` variables.  The default
column is the card's value since 2026-11-13 (the production policy); the E/G
column is what the two production runs used, and the rows where the two differ
are the explicit pins an old checkpoint needs.

| variable | default | E/G | meaning |
|---|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | `out_julia/case_realistic_{E,G}` | output root |
| `CASE_STEPS` | 0 (mode default) | 20000 | step budget; 0 uses 10/150/10000/20000/2000 by mode |
| `CASE_MAXLEV` | 7 | 7 | maximum tree level |
| `CASE_MAXLEAVES` | 10000 | 10000 | hard leaf budget |
| `CASE_COARSEN` | **0.05** | 0.05 | coarsen fraction: relative-error share and, under `:balanced`, the per-cycle cell cap |
| `CASE_THETATOL` | **1e-2** | 1e-2 | absolute per-leaf theta-range protect tolerance (`:classic` only; in `run_config` since 2026-11-13) |
| `CASE_WALLMIN` | **5** | 5 | minimum level inside the wall strip (0 = off) |
| `CASE_WALLSTRIP` | 0.1 | 0.1 | wall strip width |
| `CASE_AMR` | **`balanced`** | `balanced` (G), `classic` (E) | AMR policy; the library default is `balanced` since 2026-11-13 |
| `CASE_HYST` | 1 | 1 | balanced-policy cooldown in cycles |
| `CASE_SYMMETRY` | **`none`** | `none` (G), `x` (E) | x-mirror closure of the AMR passes |
| `CASE_AMR_GATE` | **1** | 1 (G), 0 (E, and F before the key) | 2:1 coarsening pre-check |
| `CASE_LINEAR` | `auto` | `direct` | linear solver; the library default of `run_semquad_simulation` is `auto` since 2026-11-13, `direct` is the bitwise path |
| `CASE_LATENT` | `per_phase` | `per_phase` | latent coefficients |
| `CASE_STOKES_SCALING` | `equilibrated`† | `none` (E), `equilibrated` (last G leg) | solver-side saddle equilibration |
| `CASE_EXTDIAG` | 0 | 0 | append the exact `resid_A` column; new directory only |
| `CASE_FRAME` | 0 (5) | 5 | frame stride in steps |
| `CASE_EV` | 0.11 | 0.11 | entropy-viscosity coefficient |

The six bold rows are the balanced policy's production set: a new `spinup` gets
`:balanced`, the 2:1 gate, `symmetry = :none`, the balanced coarsening pair and
the wall rule with no variables at all.  The library default of
`run_semquad_simulation` is `:balanced` since 2026-11-13 as well
(`semquad_amr_passes!` still defaults to `:classic`), and its `linear` default
is `:auto` since the same date (`2453f16`), so `CASE_LINEAR=auto` and a bare
driver call are now the same policy.  A classic replay (E) pins
`CASE_AMR=classic CASE_AMR_GATE=0 CASE_SYMMETRY=x CASE_COARSEN=0.05
CASE_THETATOL=1e-2` and, for the bitwise linear path, `CASE_LINEAR=direct`; the
pre-2026-11-13 card defaults were
`0.005 / 1e-3 / 0 / x / classic / 0`.

† `CASE_STOKES_SCALING` is the one row whose script default is still `none`; the production G card and the
verification script export `equilibrated`, while the E arm ran `none` (`docs/SOLVER.md` §3.6).

`CASE_EXTDIAG=1` changes the history layout, so `main()` refuses an output
directory that already holds `history.csv`, a checkpoint or `frames/`.  The
solver-side knobs (`CASE_STOKES_SCALING`) are part of `run_config` and are
compared on restart; an old checkpoint without the key reads as `none`.

### 5.2 The zero-step guard

A resume whose `CASE_STEPS` does not exceed the checkpoint's step number runs
zero steps, and the per-step history comes back empty.  The script used to
index `res.history[end]` in its final-frame path and threw a `BoundsError` after
the solve.  It now reports and returns before writing anything:

```
[case] DONE mode=resume steps=11400 (no steps were run: maxsteps=11400 <= checkpoint step); nothing written
```

The checkpoint and frame stream stay untouched, and a mistyped budget gives a
one-line message instead of a stack trace.  Verified on the G step-11400 seed.

### 5.3 Renderer

`scripts/live_render.py --dir <case> --interval 3 --gif-every 12 --max-per-poll 20`
polls `frames/frame_meta.csv`, renders only frames it has not seen, and skips
any `T.bin` whose size is not exactly `256*256*8` (a partially written frame is
picked up on the next poll).  It writes one panel per frame (temperature with
the mesh, next to `Nu(t)` and `Vrms(t)`), rewrites `live.html` atomically
(`.tmp` plus `os.replace`) and rebuilds `live.gif` from at most the last 120
panels.  `--once` renders everything available and exits.  The renderer is a
pure reader: it never writes into `frames/`, so it cannot disturb a run.  The
frames themselves are taken on `step % 5 == 1`, one step past every AMR cycle,
because the AMR pass zeroes the velocity just before the callback and a frame
written on that step would carry `Vrms = 0` and a meaningless `Nu`.

### 5.4 Disk space

The frame stream is the large artefact.  Each frame writes a 512 KB `T.bin`, a
0.28 MB mesh CSV at 10k leaves, and a 0.16 MB panel PNG, about 0.96 MB per five
steps.  Measured: E 4.22 GB in 16011 files (frames 2.99 GB, render 1.22 GB), G
3.29 GB in 12169 files (frames 2.31 GB), so a 20000-step run at this cadence
costs about 4 GB.  The volume had 388 GB free; a sweep of arms at this cadence
would not fit so easily.  Check free space and prune `render/` before the next
arm: `frames/*.bin` and `*_mesh.csv` plus `history.csv` are the replay state,
and the PNGs regenerate with `--once`.

### 5.5 Reading a log

Per-step and AMR lines go to stdout (block buffered); `[case] STOKES-FAIL` and
`ERROR` go to stderr (unbuffered), so a dead run shows the error before the step
lines that led to it and the log can end on a clean AMR record.  Search for both
strings explicitly; the watchdog does the same.

## 6. What this case is not

**It is not a converged statistical steady state.**  `Nu_top` is not stationary:
G was stopped while it was rising about 6% per `3e-4` of time, and E peaked near
18.9 and drifted to 18.3 over its last 4000 steps while its mean temperature
kept falling at 10.6 per unit time.  The enthalpy residual is of order `H` in
both runs (section 4.3).  Any average here is a window mean over 6.9 or 5.2
overturn times, inside a transient.

**It is not a strain-localisation case.**  Yielding and strain weakening are
off, and the composite law is viscous: it has no plastic branch and no strain
memory.  The viscosity structure comes from temperature and strain rate, so
there are no shear bands, no weakening feedback and no lithospheric
localisation to measure.  Round 2 would need either the Drucker-Prager branch,
which `Params` refuses to combine with the composite law, or a different law;
the hooks and the gate references are in `docs/PHYSICS.md` section 2.

**The 2-D box limits what the numbers mean.**  The domain is 2.5 x 1 with
free-slip walls, so there is no toroidal motion, no 3-D planform selection, and
no through-going plate.  The transitions are horizontal sheets with no
depth-dependent rheology beyond temperature.  `Nu_top`, `Nu_bot`, `Vrms` and the
boundary-layer thicknesses are box averages; the cross-code comparison with
ASPECT is a 2-D comparison and inherits the same limits.  The bottom flux is the
least transferable number: its sign and magnitude depend on the bottom boundary
layer and its window (`docs/VALIDATION.md` section 6).  The interior Rayleigh
number is `1e7` to `1e8` at the running temperature, so the flow is not in the
`Ra = 1e5` regime that the nominal number suggests.

## 7. Reproducibility

There are no random numbers and no timestamps in the physics.  `maxpic = 1`
makes a run a deterministic sequence from the card, the seed and the step count.
The two prefactors are reproduced by `julia/scripts/calib_realistic_2d.jl`
(which also prints the precursor's `edot_ref`), and the card's values are its
output rounded to ten significant digits.  The state is the checkpoint plus
`history.csv`; the frames and panels are a visualisation of that state.

This rewrite changes one file.  `julia/src` is untouched, so the default path
and every solver gate stay where they were; the validation evidence is in
`docs/VALIDATION.md` sections 2 to 8, and the defect history in
`docs/BUG_LEDGER.md`.  The two production directories, `out_julia/case_realistic_E`
and `out_julia/case_realistic_G`, are the data behind every number in sections 3
and 4.
