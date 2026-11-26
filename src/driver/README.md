# Driver

The time loop and the nonlinear solvers that advance one step. `run.jl` is
`run_semquad_simulation`, the production entry point; `maxwell_state.jl`,
`maxwell_reports.jl` and `eta_ref_report.jl` hold the viscoelastic driver state
and the two viscosity diagnostics it reports; `picard.jl` holds the lagging,
averaging and relaxation helpers; `step_newton.jl` and `newton_tangent.jl`
implement the optional per-step Newton (a research path: it needs
`linear = :direct` and is slower than the production Picard + multigrid; see the
header of `step_newton.jl`). Physics and linear algebra are elsewhere: the driver
only composes them.

| File | What it is |
|---|---|
| `amr_step.jl` | The AMR block of the time loop, extracted from `run.jl` as `_amr_step!`. |
| `diagnostics_row.jl` | The per-step diagnostics row of the run history. |
| `eta_ref_report.jl` | The `eta_ref` convention report: normalised field, interior mean, `Ra_interior`, physical units. |
| `mass.jl` | The TALA/EOS density functions, the compressibility block and the eta-range diagnostic. |
| `maxwell_reports.jl` | The Maxwell diagnostics: the co-rotation validity band and the viscosity regime records. |
| `maxwell_state.jl` | The driver-side Maxwell stress state: nodal material viscosity and the one-step objective update. |
| `newton_tangent.jl` | The matrix-free tangent-vector product of the per-step Newton. |
| `picard.jl` | Lagged viscosity, strain-rate averaging and adaptive Picard relaxation. |
| `picard_control.jl` | The field blend, the divergence test and the adaptive relaxation. |
| `strain_rate.jl` | The nodal strain-rate and spin fields of the SEM velocity: historical box bodies plus cut-6 mapped bodies (physical gradient by the chain rule) for the strain-rate-dependent laws, yielding, weakening and the Maxwell advance. |
| `restart_guard.jl` | The `run_config` builder and the restart guard of `run_semquad_simulation`. The guard compares a checkpoint's stored `geometry` against the live config and **refuses** a resume whose config lacks the key (AUDIT_BUGS R5; the mapped checkpoint no longer resumes silently as a box). Since G290/G291 the `run_config` also records the physics parameter values that were missing (`Ra`, `eta_law`, `deta_T`, `eta_b`/`eta_c`, `eta_E`, `eta_T0`, `eta_const`, `eta_min`/`eta_max`, `bc`, `Twall_top`/`Twall_bot`, `eta_custom`, `eta_soft`, `rho_ref_weight`, `compress_p`/`compress_gamma`, `supg`, `div_penalty`); `_RESTART_PHYSICS_KEYS` separates them from the numerics, and a differing physics key is refused by default or continued with `restart_physics=:allow`. |
| `run.jl` | `run_semquad_simulation`: the time loop, output history and per-step profile. Entry validations added in remediation batch P0 part 1: `dt0 > 0`, `dtmax >= dt0`, `tstop > 0` (except the build-only `tstop = 0, maxsteps = 0` sentinel), `dtmin > 0`, `maxsteps >= 0` (AUDIT_VV F8), and a named non-finite-theta refusal at the driver entry and at every per-step entry (AUDIT_VV F5). |
| `run_diagnostics.jl` | The driver-owned snapshot of the solver and AMR diagnostic out-parameters. |
| `step_newton.jl` | Globalization of the per-time-step nonlinear solve. |

Stage 0 cut 4 lifted the mapped-geometry refusal for the fixed mesh: with cuts
1–4 the assembled Stokes operator, the energy assembly and every per-step
diagnostics path are routed through the geometry, and cut 5 added the mapped
matrix-free/MG solver.  Cut 6 routes the adaptive layer: the mapped indicator,
transfer, balanced cycle, wall strips, level limits and the strain-rate
reconstructions of `strain_rate.jl` are geometry-aware, so a mapped card may run
`initial_refine > 0`, `amr_every > 0`, `amr_indicator=:theta/:velocity/:both`,
`transfer_mode=:nodal/:l2`, `wall_min_level`/`wall_strip`, and the Tosi /
composite / yielding / weakening law paths.  What is still refused up front is
refused because it is not correct yet, not for missing plumbing: the
`symmetry=:x` mirror (logical x is not a physical symmetry under a map; the
annulus must run `:none`), the Cartesian-y AMR policies (`top_wall_level`,
`hotspot_*`, `zero_side_eta`), the per-step Newton
path, the Maxwell stress history (the advance is mapped but its elastic force
load into the Stokes right-hand side is still the cut-2 box-only kernel), and
the Cartesian-y reference-state / density-hook physics wiring.  `:schur_cg`/`:lagged` are refused as a **gate-coverage restriction**, not a
known defect: the stage-0.5 fix campaign's scratch-worktree measurement found
both mapped paths solving the annulus ring to the same state as `:direct`
(1e-12 relative for `:lagged`, whose stale-LU factor is the *mapped* assembled
saddle keyed by `_saddle_key(qt, geom, ...)`; `:schur_cg` to its iteration
tolerance).  The history row of a
mapped run is the mapped diagnostics: G270 runs the driver on both validation
geometries with heating off/on, the CBF and exact-residual columns on, and the
latent-heat closure path, and closes the time-step CBF energy budget identity to
the solver's floor; G272 runs the mapped
adaptive driver end to end (several steps, `initial_refine` and `amr_every` both
on, checkpoint and bit-identical resume) and the strain-rate-dependent law paths.

**Stage 0.5b1: the annulus card and the radial physics.**  `run.jl` resolves the
geometry knob `geom::Union{Symbol,AbstractGeometry}` (`:annulus` builds
`PolarAnnulus(annulus_R0, annulus_R1, annulus_Phi)`; any geometry object passes
through) and dispatches the implementation body on the **concrete** geometry
type, so every geometry accessor in the physics closures is statically
dispatched and the box closures keep their concrete types and allocation.  The
buoyancy load is `+Ra rho alpha theta grad h` (`_semquad_buoyancy_load`): the
box branch is the historical vertical closure verbatim with `fx = 0`, the
mapped branch projects the same upward scalar on `grad h`, so the annulus gets
the two-component radial load `Ra theta e_r`; phase anomalies and the density
hook ride in the same scalar.  The driver's eta closures evaluate the base
temperature and the depth argument through the geometry
(`semquad_base_T(pr,geom,x,y)`/`height(geom,x,y)`), so the annulus uses the
radial log/H base state and `r - R0`.  The annulus card requires
`R1 - R0 = 1` (the gap is the physics length unit), `side_thermal=:insulating`
(the wedge cuts are symmetry cuts, not Cartesian side walls) and keeps the
`nu_cbf`/`extdiag` **history columns** out of the annulus run as a
definition/units choice (both diagnostics are mapped now — see
`src/diagnostics/README.md` — but the published annulus Nusselt definition is
the per-wall pointwise pair the annulus history carries), and, through the
mapped-geometry block, the reference state, the density hook, `symmetry=:x`,
per-step Newton and `linear=:schur_cg`/`:lagged` (a gate-coverage restriction;
a scratch-worktree measurement found those two paths correct on the annulus
ring).  The **periodic seam** is *not* refused:
stage 0.5c1 implemented the closed ring and it is accepted (see below).
Stage 0.5b1 ran `:rigid` walls only; **stage 0.5b2 lifts
that**: a `:free` wall is the general curved-wall row `n.u = 0` (zero normal
velocity, natural zero tangential traction), a `:rigid` wall takes the
wall-frame data `wall_un = u.n`/`wall_u = u.tau`, and `:open` walls use the
natural traction `wall_t`/`wall_tn` in the same `(n, tau)` frame with the
derived (unpinned) pressure gauge.  The default `Params` is `bc = :free`, so a
plain `geom=:annulus` card now runs free-slip circles and free-slip radial
cuts.  The annular initial condition is a knob: `annulus_ic = :firedrake`
(Davies et al. 2022 Eq. (41)) or `:aspect` (the ASPECT spherical hexagonal
perturbation, angular mode 4) sets the paper's/ASPECT's documented **total**
temperature, converted by the driver to the perturbation `theta = T_ic - Tbar(r)`
through the annulus base state and re-sampled by the existing `theta_ic` hook
after every AMR cycle; it needs `geom=:annulus` and cannot be combined with a
user `theta_ic` or with `restart_from`.  **Stage 0.5c1** adds the closed ring: `annulus_seam = :periodic` (with
`annulus_Phi = 2*pi` and a tree built with `Quadtree(...; seam = :periodic)`)
makes the seam a periodic seam -- `seam=:wall` stays the wedge and the cut
ring -- and the driver checks the tree/geometry seam agreement before setup
(`check_seam_compat`: a ring geometry on a bounded tree, or a periodic tree on
a wedge, is refused instead of running a half-glued topology).  The ring has
only the two circles as walls, so `bc`/`wall_bc` entries 3-4 are inert, and the
free-slip ring engages the rotation gauge and the pressure pin together.  G274 is the
stage-0.5b1 gate (conduction fixed point, H-inclusive flux identity, hydrostatic
balance, the end-to-end rigid run with checkpoint/resume and the remaining
refusals, depth-dependent closure wiring); G275 is the stage-0.5b2 gate
(free-slip rows, Couette/manufactured analytic gates, `:open` traction, rotation
gauge, annular IC and the free-slip end-to-end driver run).

**The undefined Nusselt pair on a strongly heated ring (`nu_undefined`, G294).**
The per-wall Nusselt normalisation needs the H-inclusive base conductive flux
`-dTbar/dh` to point outward at both walls, so it is defined on the ring only
for `H < H_crit = 4/((R1^2-R0^2) - 2 R0^2 ln(R1/R0))` (= 2.4127 for the
1.22/2.22 ring; the previously quoted `4/(R1^2-R0^2) = 1.163` is the stricter
condition that the *logarithmic part alone* stays decreasing, i.e. sufficient
but not necessary).  Above the limit `semquad_nu_walls` throws by design.  The
driver keyword `nu_undefined` (default `:refuse`) keeps that refusal verbatim;
the opt-in `:nan` fills the two Nu history columns with `NaN`, emits one `@warn`
at the start of the run naming the card, `H` and the two wall references, and
tears down the steady-state criterion for that run (a NaN Nu never settles).
It is diagnostic only (no trajectory, state or `run_config` entry; not in the
restart guard), is bit-identical on the box and on mapped cards inside the
limit, and exists so the demonstration card
`examples/annulus_realistic_demo.jl` (the realistic composite-rheology card with
`H = 10` on the ring) can adapt and integrate while its Nusselt pair stays
undefined; G294 pins the default refusal, the `:nan` run and the box-invariance.

**Physics-aware state continuation (G290/G291).**  A checkpoint stores the full
state (`theta`, `theta_prev`, `u`, `p`, `t`, `dt`, `dt_prev`, `step`, the AMR
cooldown, the EV/weak/stress fields) *and* the `run_config`, so a same-card
restart is bitwise identical to the uninterrupted run (G290: annulus ring,
`linear=:direct`, plus an MG-vs-direct sanity arm).  The `run_config` records
the physics parameter values of `Params` too (the list above; the set is
`_RESTART_PHYSICS_KEYS`).  The restart guard then applies one of two protocols:

* `restart_physics=:refuse` (the default) -- an `ArgumentError` that lists every
  differing physics key with its checkpoint/current value and says to pass
  `restart_physics=:allow` for an intentional continuation.  A same-physics
  restart never reaches this path.
* `restart_physics=:allow` -- one `@warn` with the same list, then continue.
  The driver resets the first resumed step to `dt0` (`dt = dt_prev = dt0`,
  keeping `theta_prev` and every other field): the checkpoint's `dt` was
  CFL-limited for the *checkpoint's* flow, and carrying it into stronger physics
  is a measured blow-up.  The CFL block then re-grows the step from `dt0`.

Numerics/trajectory settings (AMR, advection, EV, Picard, CFL, the dt bounds),
`geometry` and `physics_revision` are refused in **both** modes, and a
checkpoint written before a key existed does not carry it and is resumed as
before (historical silent behaviour).  The documented case-2.1 -> case-2.2
annulus continuation is the intended use: run case 2.1 (`Ra = 1e4`) with
checkpointing, then continue its checkpoint with `Ra = 1e5`,
`restart_physics=:allow` and a small `dt0`; see
`examples/annulus_benchmark.jl`'s `ANN_RESTART`/`ANN_RESTART_PHYSICS` knobs and
`docs/dev/BUG_FIX_LOG.md` §20.

