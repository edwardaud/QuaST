# Guide: the realistic 2-D mantle-convection card

This document describes the case that `examples/case_realistic_2d.jl` sets
up: the physics, the numerical method, every knob, every output file, restarting,
and the thread and memory behaviour. Every card value here is a literal in that
script or a default of `run_semquad_simulation` in
`src/driver/run.jl`. Section 7 points to the engineering record.

The case is a 2.5 x 1 box of incompressible fluid with free-slip walls,
temperature 0 at the top and 1 at the bottom, internal heating `H = 10`, a nominal
Rayleigh number `Ra = 1e5`, composite diffusion-dislocation rheology, and two
phase transitions with latent heat. Yielding and strain weakening are off.

## 1. The physical problem

### 1.1 Domain and boundary conditions

`[0, 2.5] x [0, 1]`, built from 5 x 2 base cells on a quadtree with square leaves.
`x` is horizontal and `y` is height, `y = 0` at the bottom and `y = 1` at the top;
depth is `1 - y`. All four walls are free slip, that is no normal flow and free
tangential flow, implemented by fixing the normal component: `ux = 0` on the left
and right walls, `uy = 0` on the bottom and top walls. The temperature is `T = 0`
on the top wall and `T = 1` on the bottom wall; the side walls are insulating.

### 1.2 The equations

Nondimensional Boussinesq momentum, continuity and energy:

```
-div(2 eta eps(u)) + grad(p) = f
div(u) = 0
dtheta/dt + u.grad(theta) = lap(theta) - u_y dTbar/dy
f_y = Ra theta - Ra sum_i B_i [Gamma_i(y,T) - Gamma_i(y,Tbar)]
```

`u = (ux, uy)` is the velocity, `p` the dynamic pressure, `eta` the viscosity,
`eps(u) = (grad u + grad u^T)/2` the strain rate, `theta = T - Tbar` the
temperature anomaly, `Tbar(y)` the base state of 1.3, `Gamma_i` the phase fraction
of 1.5, `B_i` its buoyancy number (positive for a dense phase, so an excess phase
fraction pushes material down), and `Ra` the Rayleigh number: `Ra = 1e5`
multiplies the thermal buoyancy, with the viscosity in units of the reference
viscosity. The prefactors of 1.4 make `eta = 1` at `T = 0.5`, which sits in the
cold lid, so the convecting interior runs at `T` near 1.3 with `eta` around `1e-3`
to `1e-2` and an interior Rayleigh number of order `1e7` to `1e8`. An overturn
time `1/Vrms` is about `1e-3`.

### 1.3 Base state, heating and Nusselt numbers

`H` is the volumetric heating of `dT/dt + u.grad T = lap T + H`. The card evolves
the anomaly about the conductive profile that carries it:

```
Tbar(y) = (1 - y) + (H/2) y (1 - y),   lap(Tbar) = -H,
dTbar/dy = -1 + (H/2)(1 - 2y).
```

Substituting `T = Tbar + theta` cancels the source and leaves the `-u_y dTbar/dy`
term, so the assembly carries no explicit heating term. With `H = 10`:
`Tbar(0.5) = 1.75`, the volume mean of the base state is `0.5 + H/12 = 1.3333`,
and the conductive wall gradients are `-6` at the top and `+4` at the bottom. The
two Nusselt numbers use those gradients as their reference,

```
Nu_top = (1 + H/2) - <dtheta/dy>_top,   Nu_bot = (1 - H/2) - <dtheta/dy>_bot,
```

where the brackets are the x-average of the one-sided wall derivative. At the
conductive state they read 6 and -4. A steady convecting state satisfies
`Nu_top - Nu_bot = H`; the departure from that identity is `closure` (4.2).

### 1.4 Rheology

`eta_law = :composite` with `comp_rule = :isostress`, the partition ASPECT uses.
Two mechanisms deform the material; in inverse form their viscosities at
temperature `T`, height `y` and strain-rate invariant `edot` are

```
inv_diff = A_d exp(-(E_d + V_d (1 - y)) / T_nd)
inv_disl = A_s exp(-(E_s + V_s (1 - y)) / (n T_nd)) edot^((n-1)/n)
```

`T_nd = T + eta_T0` is the absolute temperature in units of the temperature drop
(`eta_T0 = 0.5`, so the surface sits at half a drop), `n` the dislocation stress
exponent, `A_d`, `A_s` the prefactors, `E_d`, `E_s` the activation energies
divided by `R dT`, `V_d`, `V_s` the activation volumes (both zero here), and
`edot = max(eps_II, comp_eps_min)` with `eps_II = sqrt(0.5 edot:edot)` and
`comp_eps_min = 1e-12`.

The isostress rule gives both mechanisms the same stress, so the strain rate
splits between them. With `x = eta / eta_disl` the split solves

```
k x + x^n = 1,   k = eta_disl / eta_diff,   eta = x eta_disl.
```

The root is unique for `k > 0`; `k -> 0` is the dislocation-only limit and
`k -> infinity` the diffusion-only one. The card clips `eta` to `[1e-4, 1e4]`, a
solver safety net that also sets the conditioning; the measured viscosity contrast
is about `1e7`.

The constants are `A_d = 1.6240183119e5`, `A_s = 2.6796419238e-1`, `E_d = 12`,
`E_s = 17`, `n = 3`, `eta_T0 = 0.5`. They solve two conditions at `y = 0.5`:
`eta(T = 0.5, edot_ref) = 1`, fixing the common scale, and
`eta(T = 0.1) / eta(T = 0.9) = 1e4` at `edot_ref`, fixing the branch ratio. The
reference strain rate `edot_ref = 1.65e3` comes from a constant-viscosity precursor
of the same case.

### 1.5 The two phase transitions

With `depth = 1 - y`, the phase fraction of transition `i` is

```
Gamma_i(y, T) = 0.5 (1 + tanh(((z0 - y) - slope (T - Tref)) / width)).
```

`slope = dz/dT` is the Clapeyron slope as a height shift per unit temperature,
`width` the transition width in height, `z0` the reference height and `Tref` the
reference temperature. A hot upwelling displaces the transition downward at a
positive slope and upward at a negative one.

| | 410 | 660 |
|---|---|---|
| `z0` | 0.87 | 0.78 |
| `width` | 0.025 | 0.025 |
| `slope` | +0.032 | -0.032 |
| `Tref` | 0.5 | 0.5 |
| `drho` (kg/m^3) | 180 | 300 |
| `B` | 1.8182 | 3.0303 |

`B = drho / (rho0 alpha0 dT)` is the nondimensional buoyancy number of the density
jump, with `rho0 = 3300 kg/m^3`, `alpha0 = 3e-5 1/K` and `dT = 1000 K`. An
upwelling gains buoyancy at the 410 and loses it at the 660, the behaviour of an
exothermic and an endothermic transition.

### 1.6 Latent heat

Each coefficient is derived from its own transition:

```
gamma  = slope rho0 g d / dT          (Clapeyron slope in Pa/K)
dS_A   = gamma drho / rho0^2          (entropy change, J/(kg K))
latent = -T_nd dS_A / cp
```

`cp = 1000 J/(kg K)`, `g = 9.81 m/s^2`, `d = 2.9e6 m` is the box depth, and `T_nd`
is the absolute base-state temperature at the transition in units of `dT`, offset
by `T_surface = 0.5`. The minus sign is this code's convention: it writes the term
as `-latent DGamma/Dt` where ASPECT writes `+T dS_A DGamma/Dt`.

| | `dS_A` | `latent` | reading |
|---|---|---|---|
| 410 | +49.66 J/(kg K) | -0.059364 | heats a downwelling |
| 660 | -82.76 J/(kg K) | +0.130596 | cools a downwelling |

The energy equation carries them as an apparent heat capacity,
`(1 + sum_i latent_i dGamma_i/dT) Dtheta/Dt`, plus the advective source
`-sum_i latent_i dGamma_i/dy u_y`. The capacity factor stays inside
`[0.88, 1.12]` because `|dGamma/dT| <= |slope| / (2 width) = 0.64`.
`CASE_LATENT=global` reproduces the runs before 2026-11-08, which used the single
`latent_L = 0.11` for both transitions; that value has the wrong sign at the 410
and is a replay switch.

### 1.7 Initial condition

```
theta0(x, y) = 0.05 sin(pi y) cos(2 pi x / Lx) + 0.02 sin(2 pi y) cos(4 pi x / Lx)
```

It vanishes on the top and bottom walls, as an anomaly must, and is symmetric
about `x = Lx/2`. There are no random numbers: the card, the seed and a fixed step
count make a run deterministic. Yielding and strain weakening are off
(`yield_c = yield_taumax = Inf`, `yield_phi = 0`, `weak_eps_ref = 0`), as are
density-weighted heating (`rho_heating = rho_cp = false`), dissipation and
adiabatic heating (`Di = 0`), the reference state (`refstate = nothing`), the
deviatoric projection (`deviatoric = false`) and the anelastic mass form
(`mass_form = :boussinesq`).

## 2. The numerical method

### 2.1 Mesh

A quadtree. A leaf is `(level, ix, iy)`, and the tree is kept 2:1 balanced: face
neighbours differ by at most one level. Refinement splits a quartet of siblings,
coarsening merges one. The card caps the depth at `max_level = 7`, so the finest
leaf is `1/256` of the box height, and the leaf count at `max_leaves = 10000`.

The indicator is the gradient jump `(h/2) int_face [du.n]^2 ds` over interior
faces plus a one-sided term on the top and bottom walls. Marking is
`:error_fraction`, that is Dörfler marking: `refine_frac = 0.2` is the share of the
total indicator to resolve, so the number of cells it takes follows from the
indicator field. The production policy
(`amr_policy = :balanced`) coarsens from below with the relative threshold
`coarsen_frac * total` on a complete quartet, pays for refinement by merging the
lowest-error eligible quartet (budget exchange), and holds a hysteresis cooldown
so a quartet that was just merged is not re-split; `amr_coarsen_gate` refuses a
merge it would have to undo. The wall rule forces every leaf within
`wall_strip = 0.1` of the top or bottom wall, and the wall rows themselves, back to
at least `wall_min_level = 5` after every cycle, then enforces the leaf budget.
Fields move between meshes by nodal interpolation (`transfer_mode = :nodal`).

### 2.2 Spaces and hanging nodes

Taylor-Hood pair: velocity in `Q_p`, pressure in `Q_{p-1}`, with `p = 2` on this
card (`Q2` velocity, `Q1` pressure). Nodes are the Gauss-Legendre-Lobatto points of
the reference square, `p+1` per direction for the velocity and `p` for the
pressure; element integrals use Gauss-Legendre quadrature of order `(p, p-1)`.

A node on the face of a coarser leaf is a hanging node. Its value is the coarse
face's GLL interpolation of the coarse trace, applied as a sparse constraint
matrix `C` (identity on free rows). The system is assembled and solved as
`C' A C`, which keeps symmetry and is exact for the linear constraints. Boundary
values enter the same affine map, and one pressure degree of freedom is pinned to
remove the constant null space.

### 2.3 Time integration

The flow is quasi-static: one Stokes solve per step. The energy equation is
integrated with BDF2, backward Euler on the first step, with the variable step
ratio `w = dt/dt_prev` in the BDF2 coefficients. With `maxpic = 1` (the card) the
viscosity is lagged: the single flow solve of a step uses the start-of-step
temperature, so the step is first order in the viscosity coupling and
deterministic.

The step follows `dt = cfl h_min / |u|_max` with `cfl = 0.5` and `h_min` the
smallest leaf height. It grows by at most 5% per step, shrinks by at most 5% per
step unless the hard CFL cut fires, and is clamped to
`[dtmin, dtmax] = [1e-9, 5e-5]`; the first step starts from `dt0 = 1e-6` and is
cut to the CFL value.

### 2.4 Entropy viscosity

Oscillations are controlled by an elementwise entropy viscosity added to the
energy operator, one value per leaf:

```
nu_e   = ev h_e^2 R / entropy_variation
nu_max = ev_cmax |u| h_e
nu     = min(nu_max, nu_e)
```

`R` is the residual of the theta equation, `h` the leaf size, `h_e = sqrt(2) h`,
and `entropy_variation` the global spread of the entropy residual. The card sets
`ev = 0.11` and `ev_cmax = 0.052`. The viscosity is zero on the wall rows and is
recomputed every step; it is stored in the checkpoint because the last step's
energy operator cannot be reconstructed without it.

### 2.5 The two linear solvers, and `:auto`

The direct path forms the reduced saddle `[Kred Bt; Bs 0]` (velocity and pressure
after the constraint and boundary elimination), factorises it with UMFPACK and
checks the true reduced residual. A wall-refined free-slip tree makes the unshifted
factorisation unstable, so the path runs a shift ladder: a relative diagonal shift
of `1e-12`, then `1e-10`, then `1e-8`, and a few iterative refinement steps against
the unshifted operator as the last rung. With `stokes_scaling = :equilibrated` it
first tries the symmetric equilibration `S -> D S D`, which lifts the near-null
pressure modes above the round-off floor and engages only on a solve that would
otherwise enter the ladder. The direct path is the only bitwise reproducible one.

The multigrid path is a block-triangular preconditioner for flexible GMRES: a
geometric V-cycle for the velocity block (h-coarsening by at least 1.5 per level,
damped-Jacobi smoothing, every level built from the leaf-local matrix-free apply)
and a three-factor w-BFBT approximation of the pressure Schur complement. FGMRES
restarts every 100 iterations, is capped at 500, and stops on the true
reduced-saddle residual against `mg_tol`. The card passes `mg_tol = 1e-8`, five
orders below its `stokes_tol = 1e-3`; the library default is `1e-10`.

`linear = :auto` resolves the path before the assembly, from two properties of the
current saddle:

1. `saddle_dofs = 2*Nu + Np >= 20 000` selects the multigrid, at any contrast.
2. Otherwise, a sampled viscosity contrast at most 100 together with
   `saddle_dofs >= 2 000` selects the multigrid.
3. Everything else selects the direct path, which is also the bitwise path, so
   ties go there.

The contrast is sampled on a fixed 33 x 33 grid of the viscosity field the solve
uses. An `:auto` decision for the multigrid also forces `mg_operator =
:matrixfree` and `mg_threads = min(8, Threads.nthreads())`. If that solve misses
`stokes_tol`, or throws, the same solve is redone on the direct path and counted
in `LAST_AUTO_FALLBACK`; the run continues and the count is printed as
`auto_fallbacks` at the end. An explicit `linear = :mg` has no such safety net and
fails loudly. At the card's states the saddle is 78 720 dofs at 8 188 leaves and
95 308 at 10 000 leaves, both above the 20 000 rule, so a new run resolves to the
multigrid.

## 3. Parameter reference

### 3.1 Environment variables of the card

Every knob is a `CASE_*` environment variable read when the script starts, with
the card's default in the second column.

| variable | default | allowed | effect |
|---|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | any path | Output root. Everything the run writes goes here. |
| `CASE_STEPS` | 0 | integer | Step budget, compared with the absolute step number. 0 selects the mode default: 10 smoke, 150 short, 10000 spinup, 20000 freeze, 2000 resume and wallrefine. |
| `CASE_FRAME` | 0 | integer | Frame stride in steps. 0 uses the card's 5. |
| `CASE_MAXLEV` | 7 | integer | Maximum quadtree level, so the finest leaf is `1/(2*2^level)` of the box height. |
| `CASE_MAXLEAVES` | 10000 | integer | Hard leaf budget, enforced after every AMR cycle. |
| `CASE_AMR` | `balanced` | `balanced`, `classic` | AMR policy. `balanced` is the production policy of 2.1; `classic` is the historical fraction-plus-protect policy. |
| `CASE_AMR_GATE` | 1 | 0, 1 | The 2:1 coarsening pre-check of the balanced policy. 0 turns it off. |
| `CASE_HYST` | 1 | integer | Hysteresis length in AMR cycles: the cooldown, and the deadband multiplier of the balanced policy. |
| `CASE_COARSEN` | 0.05 | float | Coarsening share: relative threshold and per-cycle cell cap under `balanced`, error fraction under `classic`. |
| `CASE_THETATOL` | 1e-2 | float | Absolute per-leaf theta-range protect tolerance. Only `classic` reads it, but it is stored and compared on restart. |
| `CASE_WALLMIN` | 5 | integer | Minimum level inside the wall strip and on the wall rows. 0 turns the wall rule off. |
| `CASE_WALLSTRIP` | 0.1 | float in (0, 0.5) | Width of the wall strip, as a fraction of the box height from each wall. |
| `CASE_SYMMETRY` | `none` | `none`, `x` | x-mirror closure of the AMR passes: `x` refines and merges mirror partners together, for symmetric benchmarks. |
| `CASE_LINEAR` | `auto` | `auto`, `direct`, `mg`, `schur_cg`, `lagged` | Linear solver, as in 2.5. |
| `CASE_EV` | 0.11 | float | Entropy-viscosity coefficient. |
| `CASE_LATENT` | `per_phase` | `per_phase`, `global` | Latent coefficients: one per transition, or the single historical `latent_L = 0.11` for both. Any other string behaves as `global`. |
| `CASE_STOKES_SCALING` | `none` | `none`, `equilibrated` | Equilibration of the direct saddle. It engages only on a solve that would otherwise fail, so a healthy solve is unchanged. |
| `CASE_EXTDIAG` | 0 | 0, 1 | Append the exact energy-identity residual `resid_A` to `history.csv`. It changes the column count, so it is accepted only in a new output directory. |

`main()` rejects an unknown `CASE_AMR`, `CASE_AMR_GATE`, `CASE_SYMMETRY` or
`CASE_STOKES_SCALING` before it does any work. Settings that are literals, not
knobs: `Ra = 1e5`, `H = 10`, `p = 2`, `stokes_tol = 1e-3`, `mg_tol = 1e-8`,
`amr_every = 5`, `refine_frac = 0.2`, `dt0 = 1e-6`, `dtmax = 5e-5`,
`dtmin = 1e-9`, `cfl = 0.5`, `dt_growth = 1.05`, `maxpic = 1`, `pic_tol = 1e-6`,
`ev_cmax = 0.052`, `ev_alpha = 2`, `checkpoint_every = 200`, and the 256 x 256
frame grid.

The seven modes are `smoke`, `short`, `spinup`, `freeze <ckpt>`, `wallrefine
<ckpt>`, `resume <ckpt>` and `symseed <ckpt>`. The three seed modes copy a
checkpoint and rewrite exactly the configuration strings the next stage changes,
so the restart guard still checks everything else.

### 3.2 Keywords of `run_semquad_simulation`

The defaults are the library defaults; the card column is the value
`case_realistic_2d.jl` passes.

| keyword | default | card | effect |
|---|---|---|---|
| `p` | 2 | 2 | Polynomial order: velocity `Q_p`, pressure `Q_{p-1}`. |
| `maxsteps` | 10000 | `CASE_STEPS` | Absolute step number to stop at. |
| `tstop` | 1e-3 | `Inf` | Stop at this nondimensional time. |
| `dt0`, `dtmax`, `dtmin`, `dt_growth` | 1e-4, `Inf`, 1e-14, 1.05 | 1e-6, 5e-5, 1e-9, 1.05 | Step size and its bounds. |
| `cfl` | 0 | 0.5 | CFL number of the step update; 0 freezes `dt` at `dt0`. |
| `amr_every` | 0 | 5 | Steps between AMR cycles; 0 freezes the mesh. |
| `amr_policy`, `amr_hysteresis`, `amr_coarsen_gate` | `:balanced`, 1, true | `:balanced`, `CASE_HYST`, `CASE_AMR_GATE == 1` | Policy, hysteresis length and the 2:1 gate. |
| `marking` | `:global` | `:error_fraction` | `:global`, `:level` or Dörfler `:error_fraction`. |
| `refine_frac`, `coarsen_frac`, `coarsen_theta_tol` | 0.2, 0.0, `Inf` | 0.2, `CASE_COARSEN`, `CASE_THETATOL` | Marking shares and the `:classic` protect tolerance. |
| `max_level`, `max_leaves` | no cap | 7, 10000 | Depth cap and hard leaf budget. |
| `wall_min_level`, `wall_strip` | 0, 0.1 | 5, 0.1 | The wall rule; 0 turns it off. |
| `symmetry`, `transfer_mode` | `:none`, `:nodal` | `CASE_SYMMETRY`, `:nodal` | Mirror closure and the AMR transfer. |
| `advection` | `:galerkin` | `:skew` | Advection form of the energy equation. |
| `ev`, `ev_cmax`, `ev_alpha` | 0, 0.052, 2 | 0.11, 0.052, 2 | Entropy viscosity, as in 2.4; `ev = 0` switches it off. |
| `maxpic`, `pic_tol`, `pic_relax`, `pic_mode` | 1, 1e-6, 1.0, `:picard` | same | Nonlinear iteration inside a step; Newton requires `linear=:direct`. |
| `linear` | `:auto` | `CASE_LINEAR` | Linear solver, as in 2.5. |
| `stokes_tol` | `Inf` | 1e-3 | Abort the step above this true residual; also the `:auto` fallback and the direct path's acceptance tolerance. |
| `mg_tol`, `mg_threads` | 1e-10, 1 | 1e-8, 1 (`auto`: `min(8, nthreads)`) | Outer tolerance and threads of the multigrid. |
| `mg_operator`, `mg_pressure_eta_tol` | `:matrixfree`, 1.0 | same | MG geometry and the rebuild threshold of the cached pressure block. |
| `stokes_scaling` | `:none` | `CASE_STOKES_SCALING` | `:none` or `:equilibrated`, direct path only. |
| `mass_form` | `:anelastic` | `:boussinesq` | Mass conservation form. |
| `phases`, `latent_scheme` | `[]`, `:implicit` | two transitions, `:implicit` | Phase transitions and the latent coupling; the two schemes agree when `maxpic = 1`. |
| `restart_from` | `""` | mode dependent | Checkpoint to restart from. |
| `checkpoint_path`, `checkpoint_every` | `""`, 0 | `$CASE_OUT/spinup.h5`, 200 | Where and how often to checkpoint. |
| `callback`, `verbose`, `profile_steps` | `nothing`, false, false | frame writer, true, false | Step hook, per-step lines, and the per-step time profile. |
| `extdiag` | false | `CASE_EXTDIAG` | Add the `resid_A` history column. |
| `initial_refine` | 0 | 0 | AMR cycles on the initial condition before the first step. |
| `tavg_start`, `steady_tol`, `cfl_tol` | `Inf` | `Inf` | Time-average window, steady-state stop, and CFL abort guard. |

The function returns `theta`, `t`, `steps`, `history`, `history_extra`, `qt`,
`ss`, `ux`, `uz`, `pv`, `avg`, `steady`, `weak`, `chi`, `chi_prev`. `history`
holds one row per step actually run, with the columns of 4.2, and
`history_extra` names the columns appended after the fixed 18.

## 4. Output formats

Everything is written under `CASE_OUT`. The program creates the directory; only a
shell redirect into it needs it to exist beforehand.

### 4.1 `run.log`

The program does not create this file: it is the shell redirect of stdout and
stderr. It holds the four header lines, a per-step line (`verbose=true`), an AMR
line every 5 steps with the refined, coarsened and merged counts and the balanced
policy's counters, a progress line every 100 steps from the card's callback, and
at the end:

```
[case] DONE mode=spinup steps=2000 t=<t> wall=<s> s (<s> s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=<v> Nu_bot=<v> Vrms=<v> meanT=<v> closure=<v> leaves=<n> N=<n>
```

stdout is block buffered and the callback flushes it every 100 steps, so a
redirected log lags by at most 100 steps. Errors go to stderr unbuffered.

### 4.2 `history.csv`

Written once, when the run returns, with one row per step actually run:

```
t,dt,Nu_top,Nu_bot,Vrms,meanT,N,leaves,div_max,hmin,umax,cfl,closure,usrms_surf,umax_surf,W,U,balance
```

This card appends one more column because its latent terms are active.

| column | meaning |
|---|---|
| `t` | Nondimensional time at the end of the step. |
| `dt` | The step just taken. |
| `Nu_top`, `Nu_bot` | Wall Nusselt numbers, as in 1.3. |
| `Vrms` | Volume RMS of the velocity magnitude. |
| `meanT` | Volume mean of the full temperature `Tbar + theta`. |
| `N` | Number of scalar-space nodes on the mesh. |
| `leaves` | Number of quadtree leaves. |
| `div_max` | Maximum pointwise `|div u|` at the GLL nodes. |
| `hmin`, `umax` | Smallest leaf height and maximum velocity magnitude. |
| `cfl` | `umax * dt / hmin`, the CFL number of the step. |
| `closure` | `Nu_top - Nu_bot - (H - Ibase)` plus the latent time derivative over the area, with `Ibase = int u_y dTbar/dy`. It goes to zero at a steady state and is the enthalpy storage rate plus bookkeeping terms at a transient one. |
| `usrms_surf`, `umax_surf` | RMS and maximum velocity magnitude on the top wall. |
| `W`, `U` | Volume means of the buoyancy work `int rho T u_y` and the viscous dissipation `int 2 eta edot:edot`. |
| `balance` | `(Ra W - U) / max(|Ra W|, |U|)`, the relative residual of the mechanical balance. |
| `closure_nolatent` | `closure` without the latent term. Present because this card has latent heat; a run without phases writes 18 columns. |

With `CASE_EXTDIAG=1` a column `resid_A` follows: the exact residual of the quoted
pointwise Nusselt pair against the scheme's energy identity, per unit area, so it
is zero for the time-step flux and `O(h)` for the pointwise one.

### 4.3 Frames

One frame per `CASE_FRAME` steps, offset by one step past every AMR cycle: the AMR
pass zeroes the velocity, so a frame written on an AMR step would carry a zero
velocity and a meaningless Nusselt number. A final frame is always written.

`frames/frame_NNNNN_T.bin`: the full temperature `T = Tbar + theta` on a
256 x 256 grid, little-endian Float64, 512 KB. `x` varies fastest, then `y`; in
numpy that is `T = np.fromfile(path).reshape(256, 256)`, with `T[j, i]` the value
at `(x_i, y_j)`, drawn with `origin="lower"`.

`frames/frame_NNNNN_mesh.csv`: one line per leaf, in leaf order (level, then
`ix`, then `iy`), with the columns `level,x0,y0,h` at 17 significant digits. `h`
is the leaf height and `x0`, `y0` the lower-left corner.

`frames/frame_meta.csv` is opened in append mode and gets one row per frame:

```
frame,step,t,dt,N,leaves,nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac,wall_s,ng
```

`nlev` is the number of distinct levels, `eta_clip_frac` the fraction of velocity
nodes whose viscosity hit a bound, and `wall_s` the seconds since the frame writer
was constructed.

### 4.4 Checkpoints

An HDF5 file, written every `checkpoint_every` steps and once at the end, always
to the same path (`$CASE_OUT/spinup.h5`, `frozen.h5`, `resume.h5`, and so on). It
holds the full state needed to continue the trajectory:

| content | shape |
|---|---|
| `leaves` | `3 x nleaves` Int32: level, ix, iy of every leaf, sorted. |
| `theta`, `theta_prev` | The two BDF2 temperature levels on the scalar nodes. |
| `ux`, `uz` | Velocity components on the scalar nodes. |
| `pv` | Pressure on the pressure space. |
| `ev` | Optional, one entropy-viscosity value per leaf, in `leaves` order. |
| `weak`, `weak_prev` | Optional, the strain-weakening field at its two levels. Absent on this card. |
| `cooldown_no_refine`, `cooldown_no_coarsen` | Optional, the balanced policy's hysteresis state, one column per leaf key. |
| attributes | `t`, `dt`, `dt_prev`, `step`, `p`, `maxlevel`, `nx0`, `ny0`, `Lx`, `cooldown_cycle`, `velocity_stale`, the free-form metadata, and `config`: the trajectory-affecting settings as newline-separated `key=value` strings. |

`velocity_stale` marks a final checkpoint whose last step was an AMR step: the
stored velocity is a diagnostic re-solve, and a restart zeroes it.

## 5. Restarting and the configuration guard

`restart_from` replaces the mesh, `p` and the initial state with the checkpoint
contents before the first step, and `checkpoint_path` writes the continued run to
its own file. `maxsteps` and `tstop` stay absolute. The guard compares the
checkpoint's stored `config` with the settings of the live call, key by key, and
refuses to start on any difference:

```
ArgumentError: restart configuration mismatch in /path/resume.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

The compared keys are the trajectory-affecting ones: the AMR cadence, strategy,
passes, marking, fractions, thresholds, leaf budget, level cap, wall rule, hotspot
and top-wall rules, symmetry, transfer mode, policy, hysteresis and gate;
advection and the entropy-viscosity coefficients; the Picard and time-step
controls (`maxpic`, `pic_tol`, `pic_relax`, `pic_mode`, `pic_reject`, `cfl`,
`dtmax`, `dtmin`); `mass_form` and `deviatoric`; the heating flags and `H`; the
rheology parameters of the active law (`comp_*` and `eta_*` for this card, the
`tosi_*` block for that law); the yield and weakening parameters; `latent_L`,
`latent_mode`, `latent_scheme` and the phase list; the reference state; and
`stokes_scaling`. The comparison iterates over the checkpoint's keys, so a key
that postdates a checkpoint is not compared. `stokes_scaling` has a defined
default: a file without it is read as `none`, and switching the option on for such
a chain is refused.

State rather than configuration, and therefore not compared: `p` and the mesh are
restored from the file, and the `:balanced` cooldown is restored with them.
`dt0`, `dt_growth`, `initial_refine`, `linear`, `mg_tol`, `mg_threads` and the
solver's other settings are not stored and not compared, so the same trajectory
can be continued with a different linear path or checkpoint cadence.

When a resume is refused, repeat the stored values, or use a seed mode, which
copies the checkpoint and rewrites exactly the conflicting strings: `freeze` sets
`amr_every = 0`, `wallrefine` sets `wall_min_level` and `wall_strip` and refines
the wall, `symseed` sets `symmetry`. The source file is never modified.

## 6. Threads and memory

Start Julia with `--threads=N`. The only part of this card that uses the Julia
thread pool is the matrix-free V-cycle that `linear = :auto` selects, and it is
capped at eight threads (`mg_threads = min(8, Threads.nthreads())`), so more than
eight adds nothing on the auto path. With one thread the auto path warns once and
runs serial. An explicit `linear = :mg` keeps `mg_threads = 1` unless the caller
sets it, and the direct factorisation is serial in every case.

The measured memory of the production state is a peak resident set of about 1.3 GB
with the multigrid path at eight threads and about 2.4 GB with the direct
factorisation, at the 10 000-leaf budget. Both figures are process peaks, so they
include the Julia runtime and the loaded state. Budget 2 GB of free memory for a
default run and 3 GB with the direct path.

Disk is dominated by the frames: about 0.8 MB per frame (a 512 KB `T.bin` plus a
mesh CSV of a few hundred kB), so a 20 000-step run at stride 5 costs about 3 GB,
and a checkpoint is about 2 MB.

## 7. The engineering record

The measurements behind the defaults, the defects found on the way, the
optimisation campaign and the validation against benchmarks and ASPECT are kept in
[docs/dev/](dev/), the engineering record.
