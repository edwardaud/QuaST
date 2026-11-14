<p align="right"><a href="../en/guide.md">English</a> · <a href="../zh/guide.md">中文</a></p>

# Guide: the realistic 2-D mantle-convection case

This document describes the case that `examples/case_realistic_2d.jl` sets up: the physics, the
numerical method, every knob, every output file, restarting, and the thread and memory
behaviour. Every case value here is a literal in that script or a default of
`run_semquad_simulation` in `src/driver/run.jl`. Section 9 points to the engineering record.

The case is a 2.5 x 1 box of incompressible fluid with free-slip walls, temperature 0 at the top
and 1 at the bottom, internal heating `H = 10`, a nominal Rayleigh number `Ra = 1e5`, composite
diffusion-dislocation rheology, and two phase transitions with latent heat. Yielding and strain
weakening are off.

## 1. The physical problem

### 1.1 Domain and boundary conditions

`[0, 2.5] x [0, 1]`, built from 5 x 2 base cells on a quadtree with square leaves. `x` is
horizontal and `y` is height, `y = 0` at the bottom and `y = 1` at the top; depth is `1 - y`.
All four walls are free slip, that is no normal flow and free tangential flow, implemented by
fixing the normal component: `ux = 0` on the left and right walls, `uy = 0` on the bottom and
top walls. The temperature is `T = 0` on the top wall and `T = 1` on the bottom wall; the side
walls are insulating.

### 1.2 The equations

Nondimensional Boussinesq momentum, continuity and energy:

$$
-\nabla\cdot\left(2\eta\,\varepsilon(\mathbf{u})\right) + \nabla p = \mathbf{f}, \qquad \nabla\cdot\mathbf{u} = 0,
$$

$$
\varepsilon(\mathbf{u}) = \frac{1}{2}\left(\nabla\mathbf{u} + \nabla\mathbf{u}^{\mathsf T}\right), \qquad
\frac{\partial\theta}{\partial t} + \mathbf{u}\cdot\nabla\theta = \nabla^2\theta -
u_y\frac{d\bar T}{dy}.
$$

| symbol | meaning |
|---|---|
| $\mathbf{u} = (u_x, u_y)$ | velocity; $y$ is height, so $u_y$ is the vertical component (the code calls it `uz`) |
| $p$ | dynamic pressure |
| $\eta$ | effective viscosity, from section 1.4 |
| $\varepsilon(\mathbf{u})$ | strain-rate tensor |
| $\mathbf{f}$ | body force |
| $\theta = T - \bar T$ | temperature anomaly about the conductive base state |
| $\bar T(y)$ | base state of section 1.3 |

The body force is vertical and carries the thermal buoyancy, the phase buoyancy and `Ra`:

$$
f_y = Ra\,T - Ra\sum_i B_i\left[\Gamma_i(y,T) - \Gamma_i(y,\bar T)\right], \qquad T = \bar T + \theta.
$$

| symbol | meaning |
|---|---|
| $Ra$ | Rayleigh number, `1e5` on this case; it multiplies temperature, with the viscosity in units of the reference viscosity |
| $B_i$ | nondimensional buoyancy number of transition $i$, positive for a dense phase |
| $\Gamma_i(y,T)$ | phase fraction of transition $i$, section 1.5 |
| $T$ | full temperature; the library evaluates $\bar T + \theta$ |

The $\bar T(y)$ part of $Ra\,T$ is a function of height alone and is balanced by the discrete
pressure up to the pressure space's interpolation error, so the flow is driven by the anomaly and
the phase contrast. The energy equation carries no explicit heating term: substituting
$T = \bar T + \theta$ cancels the source `H` against $\nabla^2\bar T = -H$ (section 1.3) and
leaves $-u_y\,d\bar T/dy$.

### 1.3 Base state, heating and Nusselt numbers

`H` is the volumetric heating of `dT/dt + u.grad T = lap T + H`. The case evolves the anomaly
about the conductive profile that carries it:

$$
\bar T(y) = (1-y) + \frac{H}{2}y(1-y), \qquad
\frac{d\bar T}{dy} = -1 + \frac{H}{2}(1-2y), \qquad \nabla^2\bar T = -H.
$$

| symbol | meaning |
|---|---|
| $H$ | internal-heating number, `10` on this case |
| $y$ | height, 0 at the bottom and 1 at the top |

With `H = 10`: `Tbar(0.5) = 1.75`, the volume mean of the base state is `0.5 + H/12 = 1.3333`,
and the conductive wall gradients are `-6` at the top and `+4` at the bottom. The two Nusselt
numbers use those gradients as their reference:

$$
\mathrm{Nu}_{\mathrm{top}} = \left(1 + \frac{H}{2}\right) - \left\langle\frac{\partial\theta}{\partial y}\right\rangle_{\mathrm{top}}, \qquad
\mathrm{Nu}_{\mathrm{bot}} = \left(1 - \frac{H}{2}\right) -
\left\langle\frac{\partial\theta}{\partial y}\right\rangle_{\mathrm{bot}}.
$$

The angle brackets are the $x$ -average of the one-sided GLL derivative of $\theta$ at the wall.
At the conductive state the two values are 6 and -4. A steady convecting state satisfies
$\mathrm{Nu}_{\mathrm{top}} - \mathrm{Nu}_{\mathrm{bot}} = H$; the departure from that identity
is the `closure` column of the history (section 6.2).

### 1.4 Rheology

The case sets `eta_law = :composite` with `comp_rule = :isostress`, the partition ASPECT uses.
Two mechanisms deform the material. In inverse form their viscosities at temperature `T`, height
`y` and strain-rate invariant `edot` are

$$
\eta_{\mathrm{diff}}^{-1} = A_d\exp\left(-\frac{E_d + V_d(1-y)}{T_{\mathrm{nd}}}\right), \qquad
\eta_{\mathrm{disl}}^{-1} = A_s\exp\left(-\frac{E_s +
V_s(1-y)}{n\,T_{\mathrm{nd}}}\right)\dot\varepsilon^{\frac{n-1}{n}}.
$$

| symbol | meaning |
|---|---|
| $A_d$, $A_s$ | diffusion and dislocation prefactors, in units of $1/\eta_0$ |
| $E_d$, $E_s$ | activation energies divided by $R\,\Delta T$ |
| $V_d$, $V_s$ | activation volumes as depth factors; both are zero on this case |
| $n$ | dislocation stress exponent, `3` |
| $T_{\mathrm{nd}} = \max(T + \eta_{T0}, T_{\mathrm{cut}})$ | absolute temperature in units of the temperature drop; `eta_T0 = 0.5`, `comp_Tcut = 0` (off) |
| $\dot\varepsilon = \max(\varepsilon_{II}, \dot\varepsilon_{\min})$ | floored second invariant, $\varepsilon_{II} = \sqrt{\tfrac12\varepsilon:\varepsilon}$, `comp_eps_min = 1e-12` |

The isostress rule gives both mechanisms the same stress, so the strain rate splits between
them:

$$
\frac{\tau}{2\eta_{\mathrm{diff}}} + A_s\exp\left(-\frac{E_s+V_s(1-y)}{n\,T_{\mathrm{nd}}}\right)\tau^n = \dot\varepsilon, \qquad \eta = \frac{\tau}{2\dot\varepsilon}.
$$

The code solves the equivalent scalar equation for $x = \eta/\eta_{\mathrm{disl}}$:

$$
k\,x + x^n = 1, \qquad k = \frac{\eta_{\mathrm{disl}}}{\eta_{\mathrm{diff}}}, \qquad \eta = x\,\eta_{\mathrm{disl}}.
$$

| symbol | meaning |
|---|---|
| $\tau$ | deviatoric stress |
| $x$ | fraction of the effective viscosity carried by the dislocation branch, in $(0,1]$ |
| $k$ | ratio of the two branch viscosities at the total strain rate |

The root is unique for $k > 0$; $k \to 0$ is the dislocation-only limit and $k \to \infty$ the
diffusion-only one. The solve is Newton with a bisection bracket on $[0,1]$, tolerance `1e-15`
on $kx + x^n - 1$, cap 100 iterations; the measured worst case is 6 iterations. The case clips
$\eta$ once, after the combination, to `[1e-4, 1e4]`, a solver safety net that also sets the
conditioning; the measured viscosity contrast is about `1e7`.

The constants are `A_d = 1.6240183119e5`, `A_s = 2.6796419238e-1`, `E_d = 12`, `E_s = 17`,
`n = 3`, `eta_T0 = 0.5`. They solve two conditions at `y = 0.5`: `eta(T = 0.5, edot_ref) = 1`,
fixing the common scale, and `eta(T = 0.1) / eta(T = 0.9) = 1e4` at `edot_ref`, fixing the
branch ratio. The reference strain rate `edot_ref = 1.65e3` comes from a constant-viscosity
precursor of the same case. At $T = 0.5$ and $\dot\varepsilon = \dot\varepsilon_{\mathrm{ref}}$
the law gives $\eta = 1$; the volume mean of the base state is $1.3333$, and where the interior
runs above $T = 1$ the composite law gives $\eta$ between `1e-3` and `1e-2`, an effective
Rayleigh number of order `1e7` to `1e8`.

The library also provides the simpler laws; the case uses none of them, and
`run_semquad_simulation` rejects `eta_law = :const` even though `Params` accepts it (use `:expT`
with `deta_T = 1` for constant viscosity):

| `eta_law` | formula | parameters |
|---|---|---|
| `:expT` | $\eta = \exp[-\ln(\Delta\eta_T)\,T]$ | `deta_T` is the viscosity ratio: $\eta(0) = 1$, $\eta(1) = 1/\Delta\eta_T$ |
| `:expTz` | $\eta = \exp(-bT + c(1-y))$ | `eta_b` = b, `eta_c` = c; 2b uses `b = ln 16384`, `c = ln 64` |
| `:arrhenius` | $\eta = \exp\{E\,[1/(T+T_0) - 1/(1+T_0)]\}$ | `eta_E` = E, `eta_T0` = T_0 |
| `:composite` | the two displays above | `comp_*`, `eta_T0`, `comp_rule` |
| `:tosi` | Tosi viscoplastic law in absolute units | `tosi = TosiLaw(...)`, which carries its own plastic branch |

### 1.5 The two phase transitions

With `y` the height, the phase fraction of transition `i` is

$$
\Gamma_i(y,T) = \frac{1}{2}\left[1 + \tanh\left(\frac{(z_{0,i}-y) - s_i\,(T-T_{\mathrm{ref},i})}{w_i}\right)\right].
$$

| symbol | meaning |
|---|---|
| $z_{0,i}$ | reference height of the transition |
| $w_i$ | transition width in height |
| $s_i = dz/dT$ | Clapeyron slope as a height shift per unit temperature |
| $T_{\mathrm{ref},i}$ | reference temperature |

A hot upwelling displaces the transition downward at a positive slope and upward at a negative
one.

| | 410 | 660 |
|---|---|---|
| `z0` | 0.87 | 0.78 |
| `width` | 0.025 | 0.025 |
| `slope` | +0.032 | -0.032 |
| `Tref` | 0.5 | 0.5 |
| `drho` (kg/m^3) | 180 | 300 |
| `B` | 1.8182 | 3.0303 |

`B = drho / (rho0 alpha0 dT)` is the nondimensional buoyancy number of the density jump, with
`rho0 = 3300 kg/m^3`, `alpha0 = 3e-5 1/K` and `dT = 1000 K`. An upwelling gains buoyancy at the
410 and loses it at the 660, the behaviour of an exothermic and an endothermic transition.

### 1.6 Latent heat

Each coefficient is derived from its own transition:

$$
\gamma_i = \frac{s_i\,\rho_0\,g\,d}{\Delta T}, \qquad
\Delta S_{A,i} = \frac{\gamma_i\,\Delta\rho_i}{\rho_0^2}, \qquad \lambda_i =
-\frac{T_{\mathrm{abs},i}\,\Delta S_{A,i}}{c_p}.
$$

| symbol | meaning |
|---|---|
| $\gamma_i$ | Clapeyron slope in Pa/K |
| $\Delta S_{A,i}$ | entropy change of the transition, J/(kg K) |
| $\lambda_i$ | latent coefficient in the energy equation, nondimensional |
| $T_{\mathrm{abs},i}$ | absolute base-state temperature at the transition, in units of $\Delta T$, including the surface offset |
| $\rho_0$, $c_p$, $g$, $d$, $\Delta T$ | scale set: `3300 kg/m^3`, `1000 J/(kg K)`, `9.81 m/s^2`, `2.9e6 m`, `1000 K` |

The minus sign is this code's convention: it writes the term as $-\lambda_i\,D\Gamma_i/Dt$ where
ASPECT writes $+T\,\Delta S_{A}\,D\Gamma_i/Dt$.

| | `dS_A` | `latent` | reading |
|---|---|---|---|
| 410 | +49.66 J/(kg K) | -0.059364 | heats a downwelling |
| 660 | -82.76 J/(kg K) | +0.130596 | cools a downwelling |

The energy equation carries them as an apparent heat capacity plus an advective source:

$$
\left(1+\sum_i\lambda_i\frac{\partial\Gamma_i}{\partial T}\right)\left(\frac{\partial\theta}{\partial t} + \mathbf{u}\cdot\nabla\theta\right) = \nabla^2\theta - u_y\frac{d\bar T}{dy} - \sum_i\lambda_i\frac{\partial\Gamma_i}{\partial y}u_y.
$$

Here $\lambda_i$ is the coefficient of the table above, $\Gamma_i$ the phase fraction of section
1.5, and $\theta$, $\mathbf{u}$ and $\bar T$ are the fields of section 1.2.

The capacity factor stays inside `[0.88, 1.12]`: the sum of the two coefficient magnitudes,
0.18996, times the bound `|dGamma/dT| <= |slope| / (2 width) = 0.64`, is 0.122. The assembly
throws if the factor goes negative, which would reverse the energy equation in time.
`CASE_LATENT=global` reproduces the runs before 2026-11-08, which used the single
`latent_L = 0.11` for both transitions; that value has the wrong sign at the 410 and is a replay
switch.

### 1.7 Initial condition

$$
\theta_0(x,y) = 0.05\sin(\pi y)\cos(2\pi x/L_x) + 0.02\sin(2\pi y)\cos(4\pi x/L_x).
$$

It vanishes on the top and bottom walls, as an anomaly must, and is symmetric about `x = Lx/2`.
There are no random numbers: the case, the seed and a fixed step count make a run deterministic.
Yielding and strain weakening are off (`yield_c = yield_taumax = Inf`, `yield_phi = 0`,
`weak_eps_ref = 0`), as are density-weighted heating (`rho_heating = rho_cp = false`),
dissipation and adiabatic heating (`Di = 0`), the reference state (`refstate = nothing`), the
deviatoric projection (`deviatoric = false`) and the anelastic mass form
(`mass_form = :boussinesq`).

## 2. The discretisation

### 2.1 Mesh and spaces

The mesh is a quadtree. A leaf is `(level, ix, iy)`, and the tree is kept 2:1 balanced: face
neighbours differ by at most one level. Refinement splits a quartet of siblings, coarsening
merges one. The case caps the depth at `max_level = 7`, so the finest leaf is `1/256` of the box
height, and the leaf count at `max_leaves = 10000`. The initial tree is uniform level 3, 640
leaves.

The velocity and pressure pair is Taylor-Hood: velocity in $Q_p$, pressure in $Q_{p-1}$, with
`p = 2` on this case ($Q_2$ velocity, $Q_1$ pressure). Nodes are the Gauss-Legendre-Lobatto
points of the reference square, `p+1` per direction for the velocity and `p` for the pressure.
The element matrices use Gauss-Legendre quadrature with `p+2` points per direction, the default
`nq = p + 2` of `make_fedata`, which integrates the $Q_p$ mass and stiffness matrices exactly.

### 2.2 Hanging nodes and constraints

A node on the face of a coarser leaf is a hanging node. Its value is the coarse face's GLL
interpolation of the coarse trace, applied as a sparse constraint matrix `C` (identity on free
rows). The system is assembled and solved as `C' A C`, which keeps symmetry and is exact for the
linear constraints. Boundary values enter the same affine map, and one pressure degree of
freedom is pinned to remove the constant null space.

### 2.3 Time stepping

The flow is quasi-static: one Stokes solve per step. The energy equation is integrated with
BDF2, backward Euler on the first step, with the variable step ratio `w = dt/dt_prev`:

$$
\frac{a}{\Delta t}\theta^{n+1} + \mathbf{u}\cdot\nabla\theta^{n+1} - \nabla^2\theta^{n+1}
= \frac{b}{\Delta t}\theta^{n} - \frac{c}{\Delta t}\theta^{n-1},
$$

$$
a = \frac{2w+1}{1+w}, \qquad b = 1+w, \qquad c = \frac{w^2}{1+w}, \qquad w = \frac{\Delta t}{\Delta t_{\mathrm{prev}}}.
$$

| symbol | meaning |
|---|---|
| $\theta^n$ | anomaly at the start of the step |
| $\theta^{n-1}$ | anomaly one step earlier |
| $\Delta t$ | current step; $\Delta t_{\mathrm{prev}}$ the previous one |
| $a, b, c$ | BDF2 mass coefficients; the first step uses $a = b = 1/\Delta t$, $c = 0$ |

With `maxpic = 1` (the case) the viscosity is lagged: the single flow solve of a step uses the
start-of-step temperature, so the step is first order in the viscosity coupling and
deterministic.

The step follows

$$
\Delta t^{n+1} = \mathrm{CFL}\,\frac{h_{\min}}{|u|_{\max}},
$$

with `cfl = 0.5` and `h_min` the smallest leaf height. It grows by at most 5% per step, shrinks
by at most 5% per step unless the hard CFL cut fires, and is clamped to
`[dtmin, dtmax] = [1e-9, 5e-5]`; the first step starts from `dt0 = 1e-6` and is cut to the CFL
value. The update uses the velocity and mesh of the step's first flow solve, so the AMR pass
that zeroes the velocity cannot freeze `dt`.

### 2.4 Entropy viscosity

Oscillations are controlled by an elementwise entropy viscosity added to the energy operator,
one value per leaf:

$$
\nu_e = c_{\mathrm{ev}}\frac{h_e^2 R_e}{\sigma}, \qquad
\nu_{\max} = c_{\max}|u|\,h_e, \qquad \nu = \min(\nu_e, \nu_{\max}), \qquad h_e = \sqrt{2}\,h.
$$

| symbol | meaning |
|---|---|
| $c_{\mathrm{ev}}$ | entropy-viscosity coefficient, `ev = 0.11` |
| $c_{\max}$ | maximum-viscosity coefficient, `ev_cmax = 0.052` |
| $h$ | leaf size; $h_e$ is its diagonal scale |
| $R_e$ | maximum over the leaf's quadrature points of the residual of the $\theta$ equation, multiplied by $\lvert T - \langle T\rangle\rvert$ when `ev_alpha = 2` |
| $\sigma$ | global spread of the entropy residual, $\max(e_{\max}-\bar e, \bar e-e_{\min})$ of $(T-\langle T\rangle)^2$ over the mesh, with $\langle T\rangle = (T_{\min}+T_{\max})/2$ and $\bar e$ the area mean of $(T-\langle T\rangle)^2$ |
| $\lvert u\rvert$ | maximum velocity magnitude in the leaf |

The viscosity is zero on the wall rows and is recomputed every step. For the first two steps,
and whenever a variation vanishes, the code uses $\nu = \nu_{\max}$ instead. It is stored in the
checkpoint because the last step's energy operator cannot be reconstructed without it.

## 3. The adaptive policy

### 3.1 Indicator

The indicator is the temperature-gradient jump over the leaf faces, with a one-sided term on the
Dirichlet walls:

$$
J_\ell = \frac{h_\ell}{2}\sum_{F\subset\partial\ell}\int_F [\![\nabla\theta\cdot\mathbf{n}]\!]^2\,ds.
$$

| symbol | meaning |
|---|---|
| $h_\ell$ | leaf height; the $(h/2)$ scale uses the face-normal leaf size |
| $F$ | a face of the leaf |
| $[\![\cdot]\!]$ | jump of the normal derivative between the two sides |
| $\mathbf{n}$ | face normal |

Left and right walls are insulating and contribute nothing; the top and bottom walls contribute
the one-sided normal derivative, which is what resolves a thermal boundary layer.

### 3.2 Marking, coarsening and the leaf budget

Marking is Dörfler marking: `refine_frac = 0.2` is the share of the total indicator to resolve,
so the number of cells it takes follows from the indicator field. The balanced cycle coarsens
from below with the relative threshold `coarsen_frac * total` on a complete quartet, pays for
refinement by merging the lowest-error eligible quartet (budget exchange), and caps the leaves
dropped per cycle at `round(coarsen_frac * n)`. `coarsen_frac = 0.05` on this case.

The cycle enforces two hard limits: the depth cap `max_level` and the leaf budget `max_leaves`,
both applied after every cycle. `amr_coarsen_gate` refuses a merge it would have to undo, the
2:1 balance pre-check of the balanced policy. Fields move between meshes by nodal interpolation
(`transfer_mode = :nodal`). `amr_passes = 1` on this case.

### 3.3 Hysteresis and the wall rule

The hysteresis cooldown (`amr_hysteresis = 1`, `CASE_HYST`) keeps a quartet that was just merged
from being re-split for that many cycles; `0` disables the cooldown. The deadband multiplier of
the balanced cycle is the fixed `0.5`, applied as `tau_h = min(tau_c, 0.5 tau_r)`.

The wall rule forces every leaf within `wall_strip = 0.1` of the top or bottom wall, and the
wall rows themselves, back to at least `wall_min_level = 5` after every cycle, then enforces the
leaf budget. `CASE_WALLMIN=0` turns the rule off. The rule exists because the indicator alone
leaves the top wall at level 3 or 4 while the interior reaches L5/L6; the pointwise `Nu_top`
then differs from the wall-refined value by 1.2 percent and the consistent boundary flux by 9.6
percent.

## 4. The linear solvers and the `:auto` rule

### 4.1 The direct path

The direct path forms the reduced saddle `[Kred Bt; Bs 0]` (velocity and pressure after the
constraint and boundary elimination), factorises it with UMFPACK and checks the true reduced
residual. A wall-refined free-slip tree makes the unshifted factorisation unstable, so the path
runs a shift ladder: relative diagonal shifts of `1e-12`, then `1e-10`, then `1e-8`. If no rung
meets the acceptance tolerance, a few GMRES refinement iterations against the unshifted
operator, preconditioned by the best rung, polish the iterate. Two smaller shifts, `1e-14` and
`1e-13`, are tried first when the unshifted factorisation has lost the solution outright, and
again as a last resort before the step fails. With `stokes_scaling = :equilibrated` it first
tries the symmetric equilibration `S -> D S D`, which lifts the near-null pressure modes above
the round-off floor and engages only on a solve that would otherwise enter the ladder. The
direct path is the only bitwise reproducible one, so a bitwise reference pins it explicitly.

### 4.2 The multigrid path

The multigrid path is a block-triangular preconditioner for flexible GMRES: a geometric V-cycle
for the velocity block (h-coarsening by at least 1.5 per level, damped-Jacobi smoothing, every
level built from the leaf-local matrix-free apply) and a three-factor w-BFBT approximation of
the pressure Schur complement. FGMRES restarts every 100 iterations, is capped at 500, and stops
on the true reduced-saddle residual against `mg_tol`. The case passes `mg_tol = 1e-8`, five
orders below its `stokes_tol = 1e-3`; the library default is `1e-10`. The measured iteration
counts and timings behind the constants are in [../dev/SOLVER.md](../dev/SOLVER.md) section 4.

### 4.3 The `:auto` rule

`linear = :auto` resolves the path before the assembly, from two properties of the current
saddle:

1. `saddle_dofs = 2*Nu + Np >= 20 000` selects the multigrid, at any contrast.
2. Otherwise, a sampled viscosity contrast at most 100 together with `saddle_dofs >= 2 000`
   selects the multigrid.
3. Everything else selects the direct path, which is also the bitwise path, so ties go there.

The contrast is sampled on a fixed 33 x 33 grid of the viscosity field the solve uses, 1 089
evaluations at a fixed cost. An `:auto` decision for the multigrid also forces
`mg_operator = :matrixfree` and `mg_threads = min(8, Threads.nthreads())`. If that solve misses
`stokes_tol`, or throws, the same solve is redone on the direct path and counted in
`LAST_AUTO_FALLBACK`; the run continues and the count is printed as `auto_fallbacks` at the end.
An explicit `linear = :mg` has no such safety net and fails loudly. At the case's states the
saddle is 78 720 dofs at 8 188 leaves and 95 308 at 10 000 leaves, both above the 20 000 rule,
so a new run resolves to the multigrid. The library also keeps two historical paths,
`:schur_cg` (CG on the Schur complement, with `schur_maxiter` and `schur_precond`) and `:lagged`
(a reused direct factorisation); this case uses neither.

## 5. Parameter reference

### 5.1 Environment variables of the case

Every knob is a `CASE_*` environment variable read when the script starts, with the case's
default in the second column.

| variable | default | allowed | effect |
|---|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | any path | Output root. Everything the run writes goes here. |
| `CASE_STEPS` | 0 | integer | Step budget, compared with the absolute step number. 0 selects the mode default: 10 smoke, 150 short, 10000 spinup, 20000 freeze, 2000 resume and wallrefine. |
| `CASE_FRAME` | 0 | integer | Frame stride in steps. 0 uses the case's 5; frames are written when `step % stride == 1`, one step past every AMR cycle. |
| `CASE_MAXLEV` | 7 | integer | Maximum quadtree level, so the finest leaf is `1/(2*2^level)` of the box height. |
| `CASE_MAXLEAVES` | 10000 | integer | Hard leaf budget, enforced after every AMR cycle. |
| `CASE_AMR_GATE` | 1 | 0, 1 | The 2:1 coarsening pre-check of the balanced policy. 0 turns it off. |
| `CASE_HYST` | 1 | integer | Hysteresis cooldown in AMR cycles. 0 disables the cooldown. |
| `CASE_COARSEN` | 0.05 | float | Coarsening share: relative threshold and per-cycle cell cap of the balanced policy. |
| `CASE_WALLMIN` | 5 | integer | Minimum level inside the wall strip and on the wall rows. 0 turns the wall rule off. |
| `CASE_WALLSTRIP` | 0.1 | float in (0, 0.5) | Width of the wall strip, as a fraction of the box height from each wall. |
| `CASE_SYMMETRY` | `none` | `none`, `x` | x-mirror closure of the AMR passes: `x` refines and merges mirror partners together, for symmetric benchmarks. |
| `CASE_LINEAR` | `auto` | `auto`, `direct`, `mg`, `schur_cg`, `lagged` | Linear solver, as in section 4. |
| `CASE_EV` | 0.11 | float | Entropy-viscosity coefficient. |
| `CASE_LATENT` | `per_phase` | `per_phase`, `global` | Latent coefficients: one per transition, or the single historical `latent_L = 0.11` for both. Any other value is rejected by `Params`. |
| `CASE_STOKES_SCALING` | `none` | `none`, `equilibrated` | Equilibration of the direct saddle. It engages only on a solve that would otherwise fail, so a healthy solve is unchanged. |
| `CASE_EXTDIAG` | 0 | 0, 1 (also `true`, `yes`, `on`) | Append the exact energy-identity residual `resid_A` to `history.csv`. It changes the column count, so it is accepted only in a new output directory. |

`main()` rejects an unknown `CASE_AMR_GATE`, `CASE_SYMMETRY` or `CASE_STOKES_SCALING` before it
does any work. Settings that are literals: `Ra = 1e5`, `H = 10`, `p = 2`,
`stokes_tol = 1e-3`, `mg_tol = 1e-8`, `amr_every = 5`, `refine_frac = 0.2`, `dt0 = 1e-6`,
`dtmax = 5e-5`, `dtmin = 1e-9`, `cfl = 0.5`, `dt_growth = 1.05`, `maxpic = 1`, `pic_tol = 1e-6`,
`ev_cmax = 0.052`, `ev_alpha = 2`, `checkpoint_every = 200`, and the 256 x 256 frame grid.

With no argument the script runs `short`; otherwise the first argument selects one of seven
modes:

| mode | default steps | what it does |
|---|---|---|
| `smoke` | 10 | Ten steps on the initial condition, AMR at steps 5 and 10, frames at 1, 6 and 10, no checkpoint. |
| `short` | 150 | Stability check; writes `short.h5` every 200 steps and at the end. |
| `spinup` | 10000 | Production spin-up with AMR; writes `spinup.h5` every 200 steps and at the end. |
| `freeze <ckpt>` | 20000 | Copies the checkpoint to `frozen.h5` with `amr_every = 0`, then continues on the frozen mesh. |
| `wallrefine <ckpt>` | 2000 | Writes a wall-refined seed under `seed/`, then continues it with AMR to `resume.h5`. |
| `resume <ckpt>` | 2000 | Continues any checkpoint with AMR to `resume.h5`. |
| `symseed <ckpt>` | none | Writes a copy of the checkpoint with `symmetry` rewritten under `seed/`; runs no steps. |

The four restart modes read `$CASE_OUT/spinup.h5` when the checkpoint argument is omitted. The
three seed modes copy a checkpoint and rewrite exactly the configuration strings the next stage
changes, so the restart guard still checks everything else.

### 5.2 Keywords of `run_semquad_simulation`

The defaults are the library defaults; the case column is the value `case_realistic_2d.jl`
passes.

| keyword | default | case | effect |
|---|---|---|---|
| `p` | 2 | 2 | Polynomial order: velocity $Q_p$, pressure $Q_{p-1}$. |
| `maxsteps` | 10000 | mode default, `CASE_STEPS` overrides | Absolute step number to stop at. |
| `tstop` | 1e-3 | `Inf` | Stop at this nondimensional time. |
| `dt0`, `dtmax`, `dtmin`, `dt_growth` | 1e-4, `Inf`, 1e-14, 1.05 | 1e-6, 5e-5, 1e-9, 1.05 | Step size and its bounds. |
| `cfl` | 0 | 0.5 | CFL number of the step update; 0 freezes `dt` at `dt0`. |
| `cfl_tol` | `Inf` | `Inf` | Abort the run if `umax*dt/hmin` exceeds this wall-CFL guard. |
| `tmin`, `steady_tol`, `steady_window` | 0, `Inf`, 5 | 0, `Inf`, 5 | Steady-state stop: after `t > tmin`, stop when the relative theta change and the `Nu` change stay below `steady_tol` for `steady_window` steps. |
| `tavg_start` | `Inf` | `Inf` | Start of the running time average returned as `avg`; `Inf` disables it. |
| `amr_every` | 0 | 5 | Steps between AMR cycles; 0 freezes the mesh. |
| `amr_passes` | 1 | 1 | Indicator/refine/transfer passes per cycle. |
| `refine_frac`, `coarsen_frac` | 0.2, 0.0 | 0.2, `CASE_COARSEN` | Cumulative-error shares of the balanced cycle (Dörfler marking from above and from below). |
| `amr_hysteresis`, `amr_coarsen_gate` | 1, true | `CASE_HYST`, `CASE_AMR_GATE == 1` | Hysteresis cooldown and the 2:1 gate of the balanced cycle. |
| `max_level`, `max_leaves` | no cap | 7, 10000 | Depth cap and hard leaf budget. |
| `wall_min_level`, `wall_strip` | 0, 0.1 | 5, 0.1 | The wall rule; 0 turns it off. |
| `symmetry`, `transfer_mode` | `:none`, `:nodal` | `CASE_SYMMETRY`, `:nodal` | Mirror closure and the AMR transfer. |
| `initial_refine`, `theta_ic` | 0, `nothing` | 0, `nothing` | AMR cycles on the initial condition before the first step; `theta_ic` re-samples an analytic initial condition on each refined tree. |
| `hotspot_weight`, `hotspot_frac`, `hotspot_min_level`, `hotspot_mode`, `hotspot_zmin` | 0, 0.5, 0, `:theta`, 0 | off | Hotspot refinement rule. |
| `top_wall_level`, `top_wall_frac` | 0, 0.5 | off | Top-wall refinement rule. |
| `zero_side_eta`, `side_strip` | false, 0.05 | false, 0.05 | Side-strip viscosity zeroing. |
| `advection` | `:galerkin` | `:skew` | Advection form of the energy equation. |
| `ev`, `ev_cmax`, `ev_alpha` | 0, 0.052, 2 | 0.11, 0.052, 2 | Entropy viscosity, as in section 2.4; `ev = 0` switches it off. |
| `maxpic`, `pic_tol`, `pic_relax`, `pic_mode` | 1, 1e-6, 1.0, `:picard` | same | Nonlinear iteration inside a step; Newton requires `linear=:direct`. |
| `pic_reject`, `pic_reject_clamp`, `pic_reject_res`, `pic_reject_max` | false, 0.25, 10, 4 | defaults | Reject a Picard trial step when the clamp fraction or the residual grows, and redo it with a halved relaxation. |
| `pic_adapt`, `pic_relax_min`, `pic_eta_relax`, `pic_u_relax` | false, 0.05, 1.0, 1.0 | defaults | Relaxation adaptation, strain-rate averaging and velocity under-relaxation of the Picard loop. |
| `pic_div_window`, `pic_div_factor` | 0, 10 | defaults | Divergence detector of the Picard loop. |
| `newton_maxit`, `newton_krylov`, `newton_restart`, `newton_ls`, `newton_geo_safety` | 15, 80, 30, 8, 0 | defaults | Newton controls; inert with `pic_mode = :picard`. |
| `linear` | `:auto` | `CASE_LINEAR` | Linear solver, as in section 4. |
| `stokes_tol` | `Inf` | 1e-3 | Abort the step above this true residual; also the `:auto` fallback and the direct path's acceptance tolerance. |
| `schur_maxiter`, `schur_precond` | 200, `:mass` | defaults | Iteration cap and preconditioner of `linear = :schur_cg`. |
| `mg_tol`, `mg_threads` | 1e-10, 1 | 1e-8, 1 (`auto`: `min(8, nthreads)`) | Outer tolerance and threads of the multigrid. |
| `mg_operator`, `mg_vel_solver`, `mg_smoothing`, `mg_omega` | `:matrixfree`, `:vcycle`, `:auto`, `nothing` | defaults | MG geometry, velocity solver and V-cycle schedule. |
| `mg_pressure_eta_tol` | 1.0 | 1.0 | Rebuild threshold of the cached pressure block when the sampled viscosity drifts. |
| `stokes_scaling` | `:none` | `CASE_STOKES_SCALING` | `:none` or `:equilibrated`, direct path only. |
| `mass_form` | `:anelastic` | `:boussinesq` | Mass conservation form. |
| `phases`, `latent_scheme` | `[]`, `:implicit` | two transitions, `:implicit` | Phase transitions and the latent coupling; the two schemes agree when `maxpic = 1`. |
| `weak_chi0` | 0 | 0 | Initial strain-weakening field; active only when the `Params` weakening parameters are set. |
| `nu_cbf` | false | false | Append the consistent-boundary-flux Nusselt pair to the history. |
| `restart_from` | `""` | mode dependent | Checkpoint to restart from. |
| `checkpoint_path`, `checkpoint_every` | `""`, 0 | mode dependent (`spinup.h5`, `short.h5`, `frozen.h5`, `resume.h5`), 200 | Where and how often to checkpoint. |
| `checkpoint_meta`, `callback`, `verbose`, `profile_steps` | empty, `nothing`, false, false | case metadata, frame writer, true, false | Checkpoint metadata, step hook, per-step lines, and the per-step time profile. |
| `pic_diag`, `extdiag` | false, false | false, `CASE_EXTDIAG` | Nonlinear diagnostics, and the `resid_A` history column. |
| `steady_newton` | `:none` | `:none` | Optional steady Newton solve on the final state; `:auto`, `:map`, `:tosi` or `:dp`. |

The function returns `theta`, `t`, `steps`, `history`, `history_extra`, `qt`, `ss`, `ux`, `uz`,
`pv`, `avg`, `steady`, `weak`, `chi`, `chi_prev`. `history` holds one row per step actually run,
with the columns of section 6.2, and `history_extra` names the columns appended after the fixed
18.

### 5.3 The `Params` card

`make_params()` in the case script sets these fields; the rest of `Params` takes its defaults,
listed in `src/physics/params.jl`.

| field | value | meaning |
|---|---|---|
| `Ra`, `H` | 1e5, 10 | Rayleigh number and internal heating |
| `eta_law`, `comp_rule` | `:composite`, `:isostress` | Viscosity law and combination rule |
| `comp_Ad`, `comp_As` | 1.6240183119e5, 2.6796419238e-1 | Diffusion and dislocation prefactors |
| `comp_Ed`, `comp_Es` | 12, 17 | Activation energies |
| `comp_Vd`, `comp_Vs` | 0, 0 | Activation volumes |
| `comp_n`, `comp_Tcut`, `comp_eps_min` | 3, 0 (off), 1e-12 | Stress exponent, exponent floor, strain-rate floor |
| `eta_T0`, `eta_min`, `eta_max` | 0.5, 1e-4, 1e4 | Surface offset and viscosity bounds |
| `bc` | `:free` | Free-slip walls; `:rigid` is no-slip |
| `latent_L`, `latent_mode` | 0.11, `:per_phase` | Historical single coefficient (unused here) and the per-transition mode |
| `yield_c`, `yield_phi`, `yield_taumax` | `Inf`, 0, `Inf` | Yield surface off |
| `weak_f`, `weak_eps_ref` | 1, 0 | Strain weakening off |
| `rho_heating`, `rho_cp` | false, false | Density-weighted heating off |
| `Di`, `heating_adiabatic`, `heating_dissipation` | 0, false, false | Dissipation and adiabatic heating off |
| `refstate`, `deviatoric` | `nothing`, false | Conductive base state, full strain rate |
| `Twall_top`, `Twall_bot` | 0, 1 | Wall temperatures |

`Params` also validates the law: `:composite` refuses a Drucker-Prager yield surface, `:tosi`
refuses `yield_*` because it carries its own plastic branch, and `:arrhenius` needs `eta_T0 > 0`.

## 6. Output formats

Everything is written under `CASE_OUT`. The program creates the directory; only a shell redirect
into it needs it to exist beforehand.

### 6.1 `run.log`

The program does not create this file: it is the shell redirect of stdout and stderr. It holds
the four header lines, two lines per step (`verbose=true` prints the step record and the Picard
update), two lines every 5 steps (the `AMR:` record and the `balanced:` counters), a progress
line every 100 steps from the case's callback, and at the end:

```text
[case] DONE mode=spinup steps=2000 t=<t> wall=<s> s (<s> s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=<v> Nu_bot=<v> Vrms=<v> meanT=<v> closure=<v> leaves=<n> N=<n>
```

stdout is block buffered and the callback flushes it every 100 steps, so a redirected log lags
by at most 100 steps. Errors go to stderr unbuffered.

### 6.2 `history.csv`

Written once, when the run returns, with one row per step actually run:

```text
t,dt,Nu_top,Nu_bot,Vrms,meanT,N,leaves,div_max,hmin,umax,cfl,closure,usrms_surf,umax_surf,W,U,balance
```

This case appends one more column because its latent terms are active.

| column | meaning |
|---|---|
| `t` | Nondimensional time at the end of the step. |
| `dt` | The step just taken. |
| `Nu_top`, `Nu_bot` | Wall Nusselt numbers, as in section 1.3. |
| `Vrms` | Volume RMS of the velocity magnitude. |
| `meanT` | Volume mean of the full temperature `Tbar + theta`. |
| `N` | Number of scalar-space nodes on the mesh. |
| `leaves` | Number of quadtree leaves. |
| `div_max` | Maximum pointwise divergence of the velocity at the GLL nodes. |
| `hmin`, `umax` | Smallest leaf height and maximum velocity magnitude. |
| `cfl` | `umax * dt / hmin`, the CFL number of the step. |
| `closure` | `Nu_top - Nu_bot - (H - Ibase)` plus the latent time derivative over the area, with `Ibase = int u_y dTbar/dy`. It goes to zero at a steady state and is the enthalpy storage rate plus bookkeeping terms at a transient one. |
| `usrms_surf`, `umax_surf` | RMS and maximum velocity magnitude on the top wall. |
| `W`, `U` | Volume means of the buoyancy work `int rho T u_y` and the viscous dissipation `int 2 eta edot:edot`. |
| `balance` | `(Ra W - U) / max(abs(Ra W), abs(U))`, the relative residual of the mechanical balance. |
| `closure_nolatent` | `closure` without the latent term. Present because this case has latent heat; a run without phases writes 18 columns. |

With `CASE_EXTDIAG=1` a column `resid_A` follows: the exact residual of the quoted pointwise
Nusselt pair against the scheme's energy identity, per unit area, so it is zero for the
time-step flux and `O(h)` for the pointwise one.

### 6.3 Frames

One frame per `CASE_FRAME` steps, offset by one step past every AMR cycle: the AMR pass zeroes
the velocity, so a frame written on an AMR step would carry a zero velocity and a meaningless
Nusselt number. A final frame is always written.

`frames/frame_NNNNN_T.bin`: the full temperature `T = Tbar + theta` on a 256 x 256 grid,
little-endian Float64, 512 KB. `x` varies fastest, then `y`; in numpy that is
`T = np.fromfile(path).reshape(256, 256)`, with `T[j, i]` the value at `(x_i, y_j)`, drawn with
`origin="lower"`.

`frames/frame_NNNNN_mesh.csv`: one line per leaf, in leaf order (level, then `ix`, then `iy`),
with the columns `level,x0,y0,h` at 17 significant digits. `h` is the leaf height and `x0`, `y0`
the lower-left corner.

`frames/frame_meta.csv` is opened in append mode and gets one row per frame:

```text
frame,step,t,dt,N,leaves,nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac,wall_s,ng
```

`nlev` is the number of distinct levels, `eta_clip_frac` the fraction of velocity nodes whose
viscosity hit a bound, and `wall_s` the seconds since the frame writer was constructed.

### 6.4 Checkpoints

An HDF5 file, written every `checkpoint_every` steps and once at the end, always to the same
path (`$CASE_OUT/spinup.h5`, `frozen.h5`, `resume.h5`, and so on). It holds the full state
needed to continue the trajectory:

| content | shape |
|---|---|
| `leaves` | `3 x nleaves` Int32: level, ix, iy of every leaf, sorted. |
| `theta`, `theta_prev` | The two BDF2 temperature levels on the scalar nodes. |
| `ux`, `uz` | Velocity components on the scalar nodes. |
| `pv` | Pressure on the pressure space. |
| `ev` | Optional, one entropy-viscosity value per leaf, in `leaves` order. |
| `weak`, `weak_prev` | Optional, the strain-weakening field at its two levels. Absent on this case. |
| `cooldown_no_refine`, `cooldown_no_coarsen` | Optional, the balanced policy's hysteresis state, one column per leaf key. |
| attributes | `t`, `dt`, `dt_prev`, `step`, `p`, `maxlevel`, `nx0`, `ny0`, `Lx`, `cooldown_cycle`, `velocity_stale`, the free-form metadata, and `config`: the trajectory-affecting settings as newline-separated `key=value` strings. |

`velocity_stale` marks a final checkpoint whose last step was an AMR step: the stored velocity
is a diagnostic re-solve, and a restart zeroes it. A checkpoint is about 2 MB at the production
state.

## 7. Restarting and the configuration guard

`restart_from` replaces the mesh, `p` and the initial state with the checkpoint contents before
the first step, and `checkpoint_path` writes the continued run to its own file. `maxsteps` and
`tstop` stay absolute. The case's resume command is

```bash
CASE_OUT="$HOME/quast/run1" CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  "$HOME/quast/run1/spinup.h5"
```

The guard compares the checkpoint's stored `config` with the settings of
the live call, key by key, and refuses to start on any difference:

```text
ArgumentError: restart configuration mismatch in /path/resume.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

The compared keys are the trajectory-affecting ones: the AMR cadence, passes, fractions, leaf
budget, level cap, wall rule, hotspot and top-wall rules, symmetry, transfer mode, hysteresis
and gate; advection and the entropy-viscosity coefficients; the Picard and time-step controls
`maxpic`, `pic_tol`, `pic_relax`, `pic_mode`, `pic_reject`, `cfl`, `dtmax` and `dtmin`;
`mass_form` and `deviatoric`; the heating flags and `H`; the rheology parameters of the active
law (`comp_*` and `eta_*` for this case, the `tosi_*` block for that law); the yield and
weakening parameters; `latent_L`, `latent_mode`, `latent_scheme` and the phase list; the
reference state; and `stokes_scaling`. The comparison iterates over the checkpoint's keys, so a
key that postdates a checkpoint is not compared. `stokes_scaling` has a defined default: a file
without it is read as `none`, and switching the option on for such a chain is refused.

These are state, so they are not compared: `p` and the mesh are restored from the file, and the
`:balanced` cooldown is restored with them. `dt0`, `dt_growth`, `initial_refine`, `linear`,
`mg_tol`, `mg_threads` and the solver's other settings are not stored and not compared, so the
same trajectory can be continued with a different linear path or checkpoint cadence.

When a resume is refused, repeat the stored values, or use a seed mode, which copies the
checkpoint and rewrites exactly the conflicting strings: `freeze` sets `amr_every = 0`,
`wallrefine` sets `wall_min_level`, `wall_strip`, `max_leaves`, `ev` and `symmetry` and refines
the wall, `symseed` sets `symmetry`. The source file is never modified.

## 8. Threads and memory

Start Julia with `--threads=N`. The only part of this case that uses the Julia thread pool is
the matrix-free V-cycle that `linear = :auto` selects, and it is capped at eight threads,
`mg_threads = min(8, Threads.nthreads())`, so more than eight adds nothing on the auto path.
With one thread the auto path warns once and runs serial. An explicit `linear = :mg` keeps
`mg_threads = 1` unless the caller sets it, and the direct factorisation is serial in every
case.

The measured memory of the production state is a peak resident set of about 1.3 GB with the
multigrid path at eight threads and about 2.4 GB with the direct factorisation, at the
10 000-leaf budget. Both figures are process peaks, so they include the Julia runtime and the
loaded state. Budget 2 GB of free memory for a default run and 3 GB with the direct path.

Disk is dominated by the frames: about 0.8 MB per frame (a 512 KB `T.bin` plus a mesh CSV of a
few hundred kB), so a 20 000-step run at stride 5 costs about 3 GB, and a checkpoint is about 2
MB.

## 9. The engineering record

The measurements behind the defaults, the defects found on the way, the optimisation campaign
and the validation against benchmarks and ASPECT are kept in [../dev/](../dev/), the engineering
record. Start with [../dev/SOLVER.md](../dev/SOLVER.md) for the solver measurements,
[../dev/AMR.md](../dev/AMR.md) for the adaptive policy, and
[../dev/VALIDATION.md](../dev/VALIDATION.md) for the validation record.
