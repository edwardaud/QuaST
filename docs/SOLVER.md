# SOLVER.md: what this code solves, and how

This is the solver volume. It states the equations, the element and the
constraints, the time step, the five linear solution paths, the physics that is
wired in, and what is missing. Numbers are measured on this checkout unless
labelled otherwise, and the two solver tables name the commit and the session
they came from. Where a number is absent from the repository it is omitted
rather than estimated.

**Run artifacts.** The `out_julia/...` path in this document names the stored blob
trajectory in the development tree, cited so that the per-frame numbers can be traced to
it. It is not part of this repository.

Three defaults moved on 2026-11-13, and this revision follows them. The library
driver `run_semquad_simulation` now defaults to `linear = :auto` (`2453f16`)
and to `amr_policy = :balanced` (`4584e85`), and the `:auto` thresholds were
re-calibrated at `c7e0f55` after the multigrid campaign: 20 000 saddle dofs at
any contrast, 2 000 at contrast <= 100, and no contrast-10 rung. Those three
changes land in §2.1, §3.6 and §4.4. The depth behind the physics subsections
is in `docs/PHYSICS.md`, `docs/CASE_REALISTIC.md` and `docs/PERFORMANCE.md`.

---

## 1. What the code solves

### 1.1 The system

Symbol by symbol. `u` is velocity in units of `κ/H`, with `H` the box height and
`κ` the thermal diffusivity, so `|u| = 1` is one box height per diffusion time.
`ε(u) = (∇u + ∇uᵀ)/2` is the strain rate; an incompressible Newtonian stress is
`2η ε(u)`, so momentum balances the divergence of a viscous stress against a
pressure gradient and buoyancy. `η` is viscosity over the reference `η₀`, a field
of temperature, strain rate, pressure and weakening. `p` is dynamic pressure in
units of `η₀κ/H²`, with the hydrostatic part removed analytically (§1.2).
`θ = T - T̄(z)` is the anomaly, `T` in units of the drop `ΔT` (bottom wall 1, top
0); subtracting a base state is what makes buoyancy linear in the unknown and the
wall data homogeneous. `z` is height, `0` at the bottom and `1` at the top, so
gravity points along `-ê_z`, and `y` and `z` are used interchangeably.
`Ra = ρ₀ g α₀ ΔT H³/(η₀ κ)` measures buoyancy drive against viscous and diffusive
damping; the runs here use `1e4` (Blankenbach 2b), `1e2` to `1e8` (Tosi), and
`1e5` nominal for the realistic card.

`T̄(z) = (1-z) + (H/2) z(1-z)` is the conductive base state that carries internal
heating, with `dT̄/dz = -1 + (H/2)(1-2z)` and `∇²T̄ = -H`. This is why `H` never
appears as a source term: the `+H` of the physical equation is the `-∇²T̄` of the
anomaly form. `Tbar` and `dTbar` live in `julia/src/physics.jl`. `S` collects the
optional sources: shear heating, the adiabatic term with a reference state, and
latent heat with phases. All three are off by default.

### 1.2 Compressibility and the reference state

Three continuity equations are selectable through `mass_form`:

* `:boussinesq` (TALA): `∇·u = 0`. Buoyancy still carries the reference density
  and expansivity, `Ra ρ(y) α(y) θ`, so density enters the force balance and not
  the mass balance.
* `:anelastic` (ALA, the default): `∇·(ρ u) = 0`, discretised as
  `∫ q (ρ ∇·u + u_z dρ/dz) dV = 0`, so mass flux is conserved.
* `:rho_div`: `∫ q ρ ∇·u = 0`, the historical variant. It is mathematically
  inconsistent and kept for regression only.

With a reference state (`refstate=SemRefState(...; eos=:adams_williamson)` or
`:murnaghan`) the unknown becomes `θ = T - T_ad`, with
`T_ad(z) = T_surf exp(α g (1-z)/c_p)` and `dT_ad/dz = -(α g/c_p) T_ad`. The
energy equation then uses `-u_z dT_ad/dz` as base advection and gains the source
`+∇²T_ad = (α g/c_p)² T_ad`. Buoyancy keeps only the anomaly, and at `θ = 0` the
load vector is identically zero (measured `max|fv| = 0.0`), so the discrete
hydrostatic balance holds to the last bit at any resolution. The dimensional
reading is a choice: the card takes `dT = 1000 K`, `ρ₀ = 3300 kg/m³`,
`α₀ = 3e-5 1/K` and `H_box = 2900 km`.

## 2. Discretization

### 2.1 The quadtree and the 2:1 balance

The mesh is a quadtree of square leaves. A leaf at level `l` has side
`h = L_x/(cells_x · 2^l)`, and the tree is kept 2:1 balanced: no two face
neighbours differ by more than one level. `balance!` in `julia/src/quadtree.jl`
splits every leaf that has a neighbour two levels finer, one wave at a time, with
a fixed-point guard (`wave == prev && break`) and a worklist so later waves only
re-examine the children of split leaves and their old face neighbours. The
worklist equals the full sweep (140 random trees, zero differences in the leaf
set, all satisfying `max_face_level_jump <= 1`) and is 6.0x faster on a tree with
many simultaneous violations. Face queries are cached per tree version in
`QuadFaceCache`, so the indicator, the constraints and the assembly share one
neighbour table.

Refinement uses the Kelly face-jump indicator on `θ` over the leaf faces and
Dörfler marking by error fraction. `max_leaves` is a hard invariant in both
policies: a cycle that ends above the budget is trimmed by merging the parent
quartet with the smallest indicator sum (G173). Under `:balanced` the budget is
also a target, and the balanced cycle refuses to refine before it has paid for
the refinement by merging.

`amr_policy` selects the coarsening rule. The library default is `:balanced`
since 2026-11-13 (`4584e85`), the case card has passed `:balanced` for longer,
and `:classic` stays selectable. Four things differ. The balanced cycle merges
a quartet only when its worst child is at or below `tau_h = min(tau_c,
hysteresis*tau_r)`, a relative Dörfler threshold, where `tau_r` and `tau_c` are
the refine and coarsen thresholds and `hysteresis` is `amr_hysteresis`; the
classic cycle protects every leaf whose `theta` range exceeds the absolute
`coarsen_theta_tol`. The balanced cycle funds refinement by budget exchange:
with `n0` leaves at cycle entry, `want` leaves that the Dörfler prefix would
add and `capcells = round(coarsen_frac*n0)`, it merges `need = min(max(0, n0 +
want - max_leaves), capcells)` quartets before the refinement loop runs. The
balanced wall protect set is the first and last rows; the classic set is the
whole `wall_strip` band. And an `AmrCooldown` deadband (`amr_hysteresis = 1`)
keeps a merged parent from being split again in the next cycle. The 2:1-aware
`coarsening_allowed` pre-check (`amr_coarsen_gate = true` by default) belongs to
the balanced path.

The measurement that justified the flip is the E run. It sits at 9994 of 10000
leaves, and under `:classic` its absolute guard is a blanket freeze: the
per-leaf `theta` range has median `3.092e-2` against the `1e-2` tolerance, so
9892 leaves (99.0 %) enter the protect set, the mirror closure adds 69
partners, 14 leaves stay coarsenable, and no quartet exists. The 200-step A/B
from step 18800 is the report card: the classic arm changes the mesh in 0 of 40
cycles, the balanced arm in 40 of 40. The top wall row moves from L5 to L6,
`h/delta_top` from 0.213 to 0.106, `Nu_top` by 0.065 %, and the step costs
4.3 % more, of which 0.25 % is the policy. `:classic` remains the reproducible
E path: `CASE_AMR=classic` with `CASE_AMR_GATE=0 CASE_SYMMETRY=x
CASE_COARSEN=0.05 CASE_THETATOL=1e-2` reproduces E. F and G are balanced arms
and keep `CASE_AMR=balanced`, and `run_config` compares the policy on restart so
a resume cannot switch it silently. The unit-level `semquad_amr_passes!` keeps
its own `:classic` default, which is what the direct tree pins in G55 and G208
still test; t_g229 pins the new driver default.

### 2.2 Q_p velocity and Q_{p-1} pressure on GLL nodes

Velocity is `Q_p` and pressure is `Q_{p-1}`, the Taylor-Hood pair. Both are
tensor products of one-dimensional Lagrange bases on Gauss-Legendre-Lobatto
nodes, `p+1` points for velocity and `p` for pressure, so a `p=4` leaf carries 25
velocity nodes and a `p=2` leaf carries 9. The pair is LBB stable, which is what
keeps the pressure free of the checkerboard mode a `Q_p/Q_p` pair would produce.

`p` is a runtime parameter, and `p >= 2` is required because the pair is
`Q_p/Q_{p-1}`: measured, `make_semstokes` throws for `p = 1`. Two settings are
used, and the split is deliberate.

`p = 4` serves benchmarks and the blob (2b, Tosi, the `Ra = 1e8` blob). Accuracy
per node is what a spectral element buys, and the wall flux is the functional
that punishes a low-order element most. `p = 2` serves the realistic card at
`max_level = 7`, `max_leaves = 10000`, where the binding constraint is the
phase-transition width `0.025` (72 km). At `max_level = 7` the leaf is
`1/256 = 0.0039`, i.e. 6.4 leaves per transition width, against 3.2 at
`max_level = 6`, which aliases a `tanh` whose derivative `1/(2·width)` is the
largest term in the problem. At the same leaf budget `p = 2` costs 9 velocity
nodes per leaf against 25, so the finer `h` is affordable: `N = 8645` measured,
against the blob's `N = 17609` at 1024 leaves and `p = 4`. Element tables are
built once per tree and keyed on its version, so AMR invalidates them.

### 2.3 Hanging nodes are linear constraints, and the system stays symmetric

A fine leaf face that meets a coarse leaf contributes nodes with no partner on
the coarse side. Continuity is imposed exactly, one interpolation per coarse-fine
face:

```
u_fine = Σ_{a=0..p} φ_a(ξ*) u_coarse,a
```

`ξ*` is the fine node's position in the coarse element's reference coordinates,
`φ_a` the coarse Lagrange basis, and `u_coarse,a` the coarse nodal values.
`semquad_constraints` walks every coarse-fine face and `resolve_constraints`
chains multi-level constraints so one application of `C` suffices. As a matrix,
`û = C û_f + u₀`, with `û` the full nodal vector, `û_f` the free unknowns, and
`u₀` the wall Dirichlet data.

Substituting into the Stokes weak form

```
∫ 2η ε(u):ε(v) dV - ∫ p ∇·v dV = ∫ Ra ρ α θ v_z dV,     ∫ q ∇·u dV = 0
```

gives the reduced saddle system

```
[ Cᵀ K C   Cᵀ Bᵀ ] [ û_f ]   [ Cᵀ (f - K u₀) ]
[ B C      0    ] [ p̂_f ] = [ -B u₀          ]
```

`K` is the viscous stiffness `∫ 2η ε(φ_i):ε(φ_j)`, `B` the divergence matrix
`∫ φ_a ∇·φ_j`, and `f` the buoyancy load. The constraint enters as a congruence
`Cᵀ A C`, so the reduced operator is symmetric whenever `A` is, which the steady
Newton, the multigrid transfers and the checkpoint all rely on. The constraint
costs 2.6 to 4 percent of a warm direct solve.

### 2.4 Assembly

Stokes and energy operators are assembled leaf by leaf as COO triplets and
handed to `sparse`. The leaf kernel evaluates `η` and the buoyancy closure at the
leaf's quadrature points, forms the dense local blocks, and pushes the triplets.
The COO buffers carry exact `sizehint`s (`4 n_v² n_leaf` for `K`,
`2 n_pq n_v n_leaf` for `B`), and the measured lengths equal the hints. There is
one assembly per driver step: the shift ladder re-factorises shifted copies and
never reassembles, and under `mass_form=:boussinesq` the density-weighted `Bc` is
not rebuilt because `Bc = Bref`.

Warm assembly on the three production states is 0.353, 0.420 and 0.452 s (8188,
10 000 and 10 000 leaves, `p = 2`), 15.5 percent of the solve on the largest
state, of which the leaf kernel is 60 percent and closure sampling 40. The figure
of 5.6 s per assembly that once motivated a rewrite came from subtracting a warm
`lu` from a cold first solve; corrected, the assembly is 0.80 to 1.03 times the
historical 0.44 s. `assemble_threads` splits the leaves into contiguous blocks
with per-block COO buffers concatenated in block order: the leaf kernel alone is
4.2x faster on 8 threads while the whole assembly is 1.65x, because the sparse
products that follow are serial. `assemble_threads = 1` stays bitwise.

## 3. Time stepping

### 3.1 BDF2 for the energy equation

With `w = Δt/Δt_prev`, the energy step is

```
(a/Δt) M θ^{n+1} + K θ^{n+1} + C(u) θ^{n+1}
    = (b/Δt) M θ^n - (c/Δt) M θ^{n-1} + F
a = (2w+1)/(1+w),   b = 1+w,   c = w²/(1+w)
```

`M` is the temperature mass matrix, `K` the diffusion matrix including the
entropy-viscosity augmentation, `C(u)` the advection operator (skew-symmetric in
production), and `F` the source vector: internal heating through the base state,
shear heating, adiabatic heating, latent heat, wall flux. `θ^n` and `θ^{n-1}`
are the two previous levels, `Δt` the current step and `Δt_prev` the one before.
At constant `Δt` (`w = 1`) the coefficients fall back to `3/2, 2, 1/2`. The first
step after a start or a restart has no `θ^{n-1}` and degenerates to backward
Euler.

The variable-step form is what makes the CFL controller usable: `Δt` may
change every step without dropping the order, provided the same `w` enters
both the mass term and the right-hand side. One defect lived here: the Picard
loop passed the running iterate as `θ^n`, and since `a - b = -c` for BDF2 that
converges to a different scheme, which `maxpic = 1` hid because the two
coincide there.

### 3.2 The Picard/Newton loop

Each step solves the coupled `(u, p, θ)` system by fixed-point iteration.
`maxpic` caps the iterations, `pic_tol` (default `1e-6`) is the relative change
in `θ` that stops it, `pic_relax` under-relaxes `θ`, and `pic_eta_relax` averages
the nodal strain-rate field used by the rate-dependent laws. Production runs
`maxpic = 1`, one Stokes solve plus one energy solve per step. That is adequate
at moderate contrast because the lagged viscosity is a good enough approximation
over one step, and every benchmark and blob run uses it.

`pic_mode = :newton` replaces the sweep with a Newton iteration on the per-step
residual `(r_S, r_C, r_E)`, plus the weakening block when it is on, with the
analytic Jacobian: `K(η)` and its derivatives `dη/dT`, `dη/dε̇`, `dη/dp`,
`dη/dχ`, the geometric tangent `K_geo`, and the pressure tangent `Kp_red`.
Measured on a 64-leaf composite case, `Ra = 1e6`, `dt = 1e-5`:

| step | Picard iterations | Newton iterations |
|---|---|---|
| 0 | 8 | 3 |
| 1 | 5 | 2 |
| 2 | 5 | 2 |
| total | 18 | 7 |

The Newton residual history at step 0 is `2.5e-1 → 3.7e-2 → 1.1e-3 → 1.0e-6 →
9.3e-13`, the quadratic tail. A converged Newton iteration costs 51.5 ms against
10.9 ms for a Picard sweep on that case, so the Newton is 1.87x slower in wall
time for the same tolerance; at L5 the per-iteration factor grows to 20 to 60x
and the wall-time disadvantage to 12x. The Newton buys iteration count and
robustness; it does not buy time. It also needs one direct saddle factorisation
per iteration, so the driver refuses `pic_mode = :newton` unless `linear` is
passed as `:direct`; §4.6 quotes the guard. It never aborts a step either: when it fails, the driver
falls back to the Picard loop from the start of the step and `pic_diag` records
`fallback = true`.

### 3.3 The CFL controller

The step is `Δt_CFL = cfl · h_min/|u|_max`, with `cfl = 0.5` in the realistic
card. `Δt` may grow by at most `dt_growth = 1.05` per step, may shrink freely,
and is hard-clamped to `[dtmin, dtmax]` (`1e-9` to `5e-5` in the card). The
update happens inside the first Picard iteration, before AMR, because the AMR
pass rebuilds the mesh and zeroes the velocity; computing the CFL afterwards
would read a zero velocity field and take the largest allowed step.

### 3.4 Entropy viscosity

The diffusion matrix is replaced by `(1+ν_e)∇θ`, with one `ν_e` per leaf:

```
ν_e   = min( ν_max , c_E h_e² R_e / max(ē - e_min, e_max - ē) )
ν_max = c_max |u|_max,e h_e
R_e   = max_{x∈e} | ∂T/∂t + u·∇T - ∇²T | · |T - T_avg|
```

`h` is the geometric mean leaf size `√(h_x h_y)` and `h_e = √2 h` the length
scale in the viscosity; `T = T̄ + θ` is the absolute temperature and
`T_avg = (min T + max T)/2`; `ē` is the volume average of `(T - T_avg)²` with
`e_min`, `e_max` its extremes; `c_E` is the user's `ev` and `c_max` the `ev_cmax`
cap. `R_e` is the part of the residual the discretisation does not resolve,
weighted by the local deviation from the mean, so it is large only where the
solution is under-resolved. Since `ν_e ∝ h_e² R_e`, refining or smoothing drives
the artificial diffusion to zero and the scheme returns to spectral accuracy;
`ν_max` is a safety cap. Wall leaves force `ν_e = 0`, because a boundary layer is
exactly where the artificial term would corrupt the wall flux. The 2b production
values are `ev = 0.11`, `ev_cmax = 0.052`, `ev_alpha = 2`. G11 pins the formula
and the boundary treatment; `t_g149_ev_cbf.jl` pins the EV/CBF interaction.

### 3.5 The steady entry

A steady solver exists, as a separate entry point rather than a mode of the time
loop. `semquad_steady_newton` dispatches on the rheology:

* `:map` (`semquad_newton_steady`): the fixed point of one backward-Euler energy
  step, `G(θ) = θ` with `u = Stokes(η(θ))`. The Jacobian is analytic:
  `δη = dη/dT δθ`, `δK u = K(δη) u`, a linearised Stokes solve for `(δu, δp)`,
  then a linearised energy solve, with `J_F v = δθ⁺ - v`.
* `:tosi` (`semquad_tosi_newton`): the strain-rate law with an inner Picard
  iteration and the geometric tangent `K_geo` (G161).
* `:dp` (`semquad_dp_newton`): Drucker-Prager with an inner `(u, p)` Newton and
  both `K_geo` and `Kp` (G168).

`kind = :auto` picks `:tosi` when `pr.tosi` is set, `:dp` when a yield surface is
active, and `:map` otherwise. On 2b at L3 the map Newton takes three steps from a
developed state to `‖F‖ = 5.9e-13`, and 8 to 9 steps from an ascii cold start
(9 steps, 25.9 s); time stepping needs thousands of steps for the same state.
The time driver reaches this through `steady_newton = :auto | :map | :tosi | :dp`,
which runs the solve on the state the time loop returns. `run_semquad_simulation`
still time-steps, so the steady solve is an extra pass.

### 3.6 The realistic card, as the card holds it

`julia/scripts/case_realistic_2d.jl` is the production configuration, and its
defaults are read from the source rather than remembered.

| group | defaults |
|---|---|
| AMR, lines 146-234 | `amr_policy = :balanced`, `amr_coarsen_gate = true`, `amr_hysteresis = 1`, `coarsen_frac = 0.05`, `coarsen_theta_tol = 1e-2`, `wall_min_level = 5`, `wall_strip = 0.1`, `symmetry = :none`, `max_leaves = 10 000`, `max_level = 7` |
| solver, lines 148-294 | `p = 2`, `linear = :auto`, `stokes_tol = 1e-3`, `mg_tol = 1e-8`, `stokes_scaling = :none` |
| stepping, lines 469-486 | `cfl = 0.5`, `dt0 = 1e-6`, `dtmax = 5e-5`, `dtmin = 1e-9`, `dt_growth = 1.05`, `maxpic = 1`, `pic_tol = 1e-6`, `pic_mode = :picard`, `amr_every = 5`, `refine_frac = 0.2`, `marking = :error_fraction`, `advection = :skew`, `ev = 0.11`, `ev_cmax = 0.052`, `ev_alpha = 2` |

Two entries need a sentence. `coarsen_theta_tol` is forwarded and stored, and
the balanced cycle never reads it, so the historical `1e-3` of the classic A/B
pair does not matter to a balanced arm. `stokes_scaling` is `:none` in the card
script, the historical unscaled direct path; the production G arm writes
`CASE_STOKES_SCALING=equilibrated` explicitly
(`julia/scripts/verify100_case.sh:58`), and every direct number in §4 is
measured with that setting. The fixed physics is the subject of §5: the box is
2.5 x 1 on 5 x 2 base cells, `Ra = 1e5`, four free-slip walls, `T_top = 0`,
`T_bot = 1`, `H = 10.0`, `mass_form = :boussinesq` (TALA) with no reference
state, the composite isostress law of §5.4, and the two transitions of §5.6.

## 4. Linear solvers

`linear` takes `:direct`, `:schur_cg`, `:mg`, `:lagged` and `:auto`. The
library default of `run_semquad_simulation` is `:auto` since 2026-11-13
(commit `2453f16`): each Stokes solve resolves by the size/contrast rule of
§4.4, so a driver call gets the measured faster path without passing a keyword.
`:direct` stays selectable, is the only bitwise reproducible path, and is what
every bitwise reference pins explicitly; `:mg`, `:lagged` and `:schur_cg` stay
selectable too. The realistic case card passes `CASE_LINEAR=auto` explicitly
(unchanged behaviour), which at the production size resolves to `:mg` (§4.4).
`:schur_cg` is the fifth path and the legacy one; §4.5 gives its scope. One
constraint crosses the paths: `pic_mode = :newton` requires `linear = :direct`
and rejects every other setting at argument validation (§4.6).

### 4.1 `:direct`, the reference path

The reduced saddle is factorised with UMFPACK. The system is
`S = [Kred Bt; Bs 0]` with `Kred = CᵀKC`, `Bs = -Cpᵀ Bc Cu` and `Bt = Bsᵀ`. The
factorisation is followed by a true-residual recomputation, and if the residual
is too large a shift ladder runs: the matrix is re-factorised with a diagonal
shift, the shift is refined, and a last-resort rule applies tiny shifts (`1e-12`
and its refinement) when the finite unshifted residual falls in the band where
the historical rungs sit inside the wall-strip near-null cluster. The ladder
never reassembles; it factorises shifted copies of the same matrix. G207 pins the
rungs and G212 the tiny-shift fallback (62/62), including the production state
that missed the old trigger by 3.5 percent.

Symmetric equilibration (`stokes_scaling = :equilibrated`, off by default) applies
`D = diag(1/√diag(Kred), 2 max|Kred|/max(|Bt|,|Bs|))` as `S → D S D`,
`rhs → D rhs`, `x = D y`. It is an exact congruence, so the solution is unchanged
and only the LU arithmetic changes; on the G-7641 tree it takes the failing
unshifted solve from a true residual of `4.20e2` to `5.96e-10`, where the `1e-14`
shift stopgap gives `3.72e-7`. No level number, threshold or wall constant enters
`D`, and it is off only because it changes the bitwise path.

Warm numbers, best of three, one process per state, `--threads=1`, three
production checkpoints at `p = 2`:

| state | leaves | saddle dofs | assembly | LU pair | warm solve (median) | min |
|---|---|---|---|---|---|---|
| F | 8 188 | 69 388 | 0.353 s | 1.581 s | 2.092 s | 1.983 s |
| P1 | 10 000 | 86 544 | 0.420 s | 1.991 s | 2.713 s | 2.632 s |
| P2 | 10 000 | 82 951 | 0.452 s | 1.923 s | 2.534 s | 2.327 s |

The LU pair is 73 to 83 percent of the solve and the assembly 15 to 18 percent.
The pair is not two equal factorisations: the unshifted LU is 0.906 to 1.058 s
and the tiny-shift LU 0.56 to 0.60 s, because a full diagonal lifts UMFPACK's
diagonal fraction above its 0.9 rule and the fill drops from 31.3M to 12.0M
nonzeros on F. Pricing the ladder as `2 × lu(S)` overestimates it by about half.
A warm driver step on P1 is 3.58 s (Stokes 3.08, energy 0.31, work 0.14,
diagnostics 0.04) and the first step in a process 20.4 s, of which Stokes 16.4,
from JIT and first touch. Allocations are 1.7 to 2.0 GB per solve, and the
UMFPACK factors alone are 0.88 GB at 1024 leaves with `p = 4`.

Direct is the reference path at 10k leaves (the library default is now
`:auto`, which resolves to MG at that size, so §4.4 is where a bitwise
reference has to pin `linear=:direct` explicitly), and its cost is dominated by a
sparse LU over which no cheap lever remains. The reordering is COLAMD, the
assembly is 15 percent, a total assembly rewrite was measured at 0.15 to
0.25 s per step, and the only multi-second item is the factorisation. A
symmetric-indefinite backend is projected at 2 to 4x on the numeric
factorisation, i.e. 0.6 to 1.0 s for the pair, and it would break the bitwise
contract that makes direct the reference: G150 asserts 12/12 after every
change in this area, and every MG measurement is validated against it.

### 4.2 `:mg`, the matrix-free multigrid path

`mg_operator = :matrixfree` (the default for `:mg`) builds an h-multigrid
hierarchy by merging each leaf quartet one level at a time and restoring the 2:1
balance after each merge. The velocity operator is never assembled: the leaf
kernel applies `K(η)v` as a tensor-product contraction with `η` sampled at the
quadrature points, wrapped in `Cu`/`Cuᵀ` so the hanging constraints are applied
without forming `CᵀAC`. The pressure block is the three-factor w-BFBT

```
S̃⁻¹ = S₁⁻¹ S₂ S₁⁻¹,   S₁ = B D⁻¹ Bᵀ,   S₂ = B D⁻¹ K D⁻¹ Bᵀ,   D = diag(M_u(√η))
```

with `S₁` a sparse Cholesky and `D` the velocity mass diagonal. Smoothing is
damped Jacobi on a per-level schedule: `:auto` selects `:taper`
(`1,2,2,4,4,8,8` pre and post from the finest level) with `ω = 0.55` when the
hierarchy is top-heavy (`nfs[end-1] > 0.5 nfs[end]`), and the uniform 2+2 with
`ω = 2/3` otherwise. The `:taper` shape takes the frame-19 1024-leaf V-cycle
from 11.53 to 9.29 ms at 2.6 percent more outer iterations, and a single solve on
that tree from 1.221 to 1.058 s.

The measured limits are specific. Coarsening two layers at once drops the
V-cycle 25 to 34 percent but raises outer iterations 37 to 61 percent, so the
total does not move (425 to 467 ms against 429): the 18k-dof level is the one
that still resolves the viscosity structure. A coarse direct solve is not worth
it, and pHMG was rejected because the p-level applies alone total 8.9 ms, the
entire h-V-cycle. The pressure block is where a cheaper choice fails outright:
`:mass` at 16384x contrast diverges, `1/η`-weighted mass needs 68 outer
iterations against BFBT's 39, PCD degenerates to the mass inverse because
momentum has no advection term, and LSC is 6x worse. w-BFBT is the best
available option, and the block is only 17 percent of an outer iteration, so
making it free buys about 14 percent.

**Re-measured at `c7e0f55` (2026-11-13), after the MG campaign of
[MG.md](MG.md).** Same-session A/B in one process per state, both
paths on the same tree and the same closure objects, `BLAS=1`, direct
`:equilibrated` (the accuracy reference: it reaches `9.97e-11` on `F`, where the
unscaled ladder of §4.1 stops at `1.317e-3`), `:mg` with the shipped defaults at
`mg_tol = 1e-6` (the direct reference's accuracy), `t_min`; ratio = `t_direct /
t_mg`, so **>1 means MG is faster**. That is the convention of §4.4 and of the
selector comment in `semquadstokes.jl`. Instrument:
`julia/scripts/bench_auto_crossover.jl`. Cold clears the MG solver caches (it
pays the hierarchy and the pressure block), warm is the production in-tree
solve reusing both.

| state (leaves, p) | saddle dofs | contrast | direct cold / warm | `:mg` cold / warm | ratio cold / warm | iters |
|---|---|---|---|---|---|---|
| uniform L2 (16, p=4) | 747 | 1 | 0.0084 / 0.0083 | 0.0072 / 0.0049 | 1.17 / 1.71 | 17 |
| uniform L3 (64, p=4) | 2 803 | 1 | 0.042 / 0.043 | 0.024 / 0.016 | 1.74 / 2.75 | 18 |
| uniform L4 at p=2 (256) | 2 467 | 1 | 0.019 / 0.018 | 0.013 / 0.009 | 1.48 / 1.91 | 15 |
| uniform L4 (256, p=4) | 10 851 | 1 | 0.222 / 0.262 | 0.094 / 0.068 | 2.36 / 3.86 | 19 |
| uniform L5 (1 024, p=4) | 42 691 | 1 | 1.516 / 1.555 | 0.469 / 0.211 | 3.23 / 7.36 | 11 |
| uniform L6 (4 096, p=4) | 169 347 | 1 | 8.334 / 8.477 | 2.338 / 1.216 | 3.56 / 6.97 | 11 |
| stratified 50x L3 (64, p=4) | 2 803 | 50 | 0.042 / 0.041 | 0.024 / 0.017 | 1.74 / 2.39 | 21 |
| stratified 50x L4 (256, p=4) | 10 851 | 50 | 0.226 / 0.274 | 0.103 / 0.068 | 2.20 / 4.04 | 22 |
| blob traj frame 1 (214) | 9 351 | 3 024 | 0.147 / 0.162 | 0.134 / 0.109 | 1.09 / 1.48 | 27 |
| blob traj frame 14 (628) | 28 135 | 5 087 | 0.910 / 0.778 | 0.691 / 0.387 | 1.32 / 2.01 | 29 |
| blob f19 (1 024) | 45 475 | 5 087 | 1.235 / 1.254 | 1.086 / 0.703 | 1.14 / 1.78 | 27 |
| blob f19x4 (2 344) | 99 559 | 5 087 | 3.882 / 3.992 | 2.437 / 1.656 | 1.59 / 2.41 | 27 |
| blob f19x16 (4 096) | 174 915 | 5 087 | 7.692 / 7.569 | 3.920 / 2.891 | 1.96 / 2.62 | 25 |
| production `F` (8 188, p=2) | 78 720 | 5.7e6 | 3.751 / 3.729 | 2.082 / 1.644 | 1.80 / 2.27 | 71 |
| production `P1` (10 000, p=2) | 95 308 | 3.3e6 | 4.112 / 3.912 | 2.962 / 2.449 | 1.39 / 1.60 | 65 |

At eight threads (`--threads=8`, `mg_threads = 8`; the direct path is serial, so
this is threaded MG against serial direct, the production configuration) the
ratios are, cold / warm: L2 0.91 / 1.09, L3 1.74 / 2.60, L4 p=2 1.22 / 1.81,
L4 2.41 / 3.94, L5 3.97 / 7.21, L6 4.38 / 7.59, stratified-50 L3 1.51 / 2.33,
stratified-50 L4 2.15 / 3.48, f19 1.17 / 1.55, f19x4 1.86 / 2.44, f19x16
2.07 / 2.77, `F` 2.33 / 2.66, `P1` 1.89 / 2.06. **No cell of either table is
below 1.0 in the warm protocol**, and the only sub-1.0 cell in the whole
campaign is the 747-dof L2 tree cold at eight threads (0.91x, a 17 ms solve).

The campaign's own close, at `6c20284` and in the `S1_SOLVE_COST` session, is
the warm `F` step at eight threads: **1.356 s, 71 outer iterations**, true
residual 6.75e-7, `|du|/|u| = 7.96e-7` and gauge-removed `|dp|/|p| = 2.27e-7`
against the equilibrated direct solve, whose own residual is 9.97e-11. The
direct arm of that row is 3.372 s warm, so the ratio is 2.49x, and 1.96x at
equal threads (3.773 against 1.929 s). Absolute seconds are session currency:
the same row reads 1.324 to 1.334 s in the session behind `S1_VARIANTS_PROBE`,
with the identical iteration count and velocity difference. The ratio and the
iteration count are the claim. The cost structure of that solve is what the
campaign bought: at 81 iterations and 1.489 s, setup is 0.325 s (21.8 %) and
the iteration 1.164 s (78.2 %, 14.37 ms each), split into V-cycle 5.095 ms
(35.5 %), w-BFBT block 3.635 ms (25.3 %), outer `K` apply 2.023 ms (14.1 %) and
FGMRES own work 3.033 ms (21.1 %). Almost half of the eight-thread iteration is
serial: the two `S1` solves, `Bt`, `Bs`, the `Z`/`W` copies, MGS and the true
residual are 6.6 of 14.4 ms on `F` (46 %) and 7.7 of 20.6 ms on `P1` (37 %).

The 25-frame blob trajectory (the canonical `run_blob_kelly_1024.jl` state, the
one whose old row read "MG 1.38x slower") is measured frame by frame on the
stored `frame_<k>_mesh.csv` + `frame_<k>_T.bin` of
`out_julia/fresh_blob_kelly_1024`: 214-1 024 leaves, 9 351-45 475 saddle dofs,
contrast 3 024-5 793. Per-frame warm ratio mean **1.55** (1 thread) and **1.51**
(8 threads), time-weighted **1.60** / **1.53**, per-frame range 1.24-1.99 at one
thread and 1.11-1.91 at eight. The row
flipped: **every one of the 25 frames is faster with MG** in the warm protocol
(min 1.24 at one thread, 1.11 at eight), so the trajectory as a whole is an MG
win, worth roughly 1.4x on the whole 25-frame run (Stokes is 79 % of it:
138.97 s of 163.77 s in the old MG arm, 94.6 s inferred in the direct arm).

At the library default `mg_tol = 1e-10` the same sweep is still a win overall but
no longer on every frame: per-frame mean 1.17, time-weighted 1.21, min 0.81
(17 135 and 19 413 dofs, the two frames the 20 000 constant keeps on the direct
side), max 1.61. That is the tolerance case the constant is set to protect
against, and it is why the flip is quoted at `1e-6` (the accuracy direct
reaches) rather than at the library default.

**How much of this is the campaign and how much is the tolerance.** The
pre-campaign measurement was made at the library default `mg_tol = 1e-10`; the
rows above are at `mg_tol = 1e-6`, matched to what direct actually reaches.
Repeating two rows at `1e-10` separates the two: uniform L4 is 2.03 cold /
3.11 warm against 3.04x before, and blob f19 is 1.18 cold / 1.39 warm against
1.03x before. The uniform rows barely moved at equal tolerance, so they were
calibrated correctly, while **f19 flips on the code change alone** (1.03 ->
1.39) and the 1e-6 protocol adds a further 1.28x on the MG side. The FNV-level
accuracy at `1e-10` is unchanged (`|du|/|u| = 2.2e-11` on L4, 2.9e-12 on f19).

**The pre-campaign record**, kept for provenance. Peak memory at 132k dof was
2 597 MB for MG against 5 139 MB for direct, and solve-only memory 923 against
3 328 MB. At the pre-tuning production size (8.2k and 10k leaves, `p = 2`)
direct was 6.65 s and the best repaired MG about 11 s, 1.6x slower, with the
uncorrected-failure path at 3.8x, and at `max_leaves = 10000` the crossover was
not reached. The row-level numbers, including the 25-frame run where direct was
119.40 s against MG 163.78 s, are in [PERFORMANCE.md](PERFORMANCE.md) §5.3.

That reading is history. The current head-to-head is the re-measured table above
(at `c7e0f55`) and [MG.md](MG.md) §14 (at `6c20284`): MG is faster
than the accuracy-matched equilibrated direct solve at every cell, 1.30-1.96x at
one thread and 1.67-2.31x at eight threads on `F` and `P1`.

The largest defect was the matrix-free damped-Jacobi diagonal: `Cuᵀ diag(K)`
is not `diag(Cuᵀ K Cu)` on hanging rows, where the negative constraint weights
cancelled to ≤ 0 and a `1e-12 d_max` clip turned them into a `1e12` amplifier
per sweep. The step-6130 tree stalled at `9.718e-01`; with the assembled
diagonal it reaches `6.07e-07` in 70 iterations (G218).

### 4.3 `:lagged`

`linear = :lagged` factorises once and then iterates a correction against the
frozen factorisation, re-factorising only when the correction stops converging
(G174). It wins where conditioning is mild: 2b at L3, 640 leaves, `p = 2`, 16384x
contrast takes 0.293 s against direct's 0.661 s, a factor 2.25. It loses on the
blob, at 178.58 s for the 25-frame run against 119.40 s, with 27 refactors and
585 iteration corrections, and the highest peak memory of the three paths
(3 029 MB). It is a tool for one regime. The blob A/B is the direct-era run:
that 119.40 s direct arm is now the slower path on the same trajectory, where
the matrix-free solve is a 1.55x per-frame mean (§4.2), so the 1.50x deficit
understates the distance to the production choice.

### 4.4 `:auto`

`linear = :auto` is the library default of `run_semquad_simulation` since
2026-11-13, and it may be passed explicitly (the realistic card does). It
resolves to `:direct` or `:mg` before assembly, from two
quantities: the saddle size `2 Nu + Np`, a node count, and the viscosity contrast
measured on the eta closure the solve is about to use, sampled on a fixed
`33 × 33` grid (1 089 evaluations, 0.02 ms on the production blob closure). The
rule, in order:

```
if saddle_dofs >= 20_000                         return :mg
elseif contrast <= 100 && saddle_dofs >= 2_000    return :mg
else                                              return :direct
```

The selector runs per Stokes solve, before the assembly, on the tree and on the
eta field that solve is about to use. An AMR cycle that changes the tree can
therefore change the path mid-run, and `LAST_SEMSTOKES_AUTO_CHOICE` records each
decision with its two inputs and a reason string.

The constants are calibrated against measured crossings, re-measured at
`c7e0f55` (§4.2 above; the previous 60 000 / 20 000 / 8 000 ladder, with a
contrast-10 rung, was calibrated on the pre-campaign MG solve). The ladder moved
down 3-4x because only the MG side got faster, and it lost its middle contrast
rung: the `10 < contrast <= 100` band had no measurement behind its 20 000
constant, and the two `stratified 50x` rows now show that band costs the
V-cycle nothing measurable against contrast 1. The only contrast boundary left
is 100, which is where the MG pressure block switches from the mass inverse to
w-BFBT, a property of the MG configuration and not a measured crossover.

* **20 000** is the any-contrast point. Every measured high-contrast state at or
  above it wins: 19 413 dofs 1.24x (`8` threads), 28 135 dofs 1.32x cold /
  2.01x warm, 45 475 dofs 1.14x cold / 1.78x warm, 99 559 dofs 1.59x cold. Below
  it the high-contrast rows are ties or losses (9 351 dofs 1.09x cold; 17 135
  and 19 413 dofs are 0.81-0.82x warm at the library default `mg_tol = 1e-10`),
  so the constant also keeps the frames that lose at the *library* tolerance on
  the direct side. It sits 2.1x above the last measured high-contrast tie; the
  old 60 000 sat 1.32x above its tie (45 475).
* **2 000** is the low/mid-contrast point. 747 dofs is the last measured
  tie/loss (1.17x cold at one thread, 0.91x at eight), 2 467 dofs (p=2) is
  1.22-1.48x cold / 1.91x warm, 2 803 dofs (p=4) is 1.74x cold at both thread
  counts, and the contrast-50 tree of the same size is 1.51-1.74x cold. The old
  8 000 was never measured below 10 851 dofs; this one is bracketed.

When the rule selects MG it also sets `mg_operator = :matrixfree` and
`mg_threads = min(8, Threads.nthreads())`. Every measured case lands on the
intended side (ratio = `t_direct`/`t_mg`, warm, 1 thread / 8 threads):

| case | saddle dofs | contrast | measured ratio | choice |
|---|---|---|---|---|
| uniform L2 | 747 | 1 | 1.71 / 1.09 | direct |
| uniform L3 | 2 803 | 1 | 2.75 / 2.60 | MG |
| uniform L4 at p=2 | 2 467 | 1 | 1.91 / 1.81 | MG |
| uniform L4 | 10 851 | 1 | 3.86 / 3.94 | MG |
| uniform L5 | 42 691 | 1 | 7.36 / 7.21 | MG |
| uniform L6 | 169 347 | 1 | 6.97 / 7.59 | MG |
| stratified 50x, L3 | 2 803 | 50 | 2.39 / 2.33 | MG |
| stratified 50x, L4 | 10 851 | 50 | 4.04 / 3.48 | MG |
| blob traj frame 1 | 9 351 | 3 024 | 1.48 / 1.11 | direct |
| blob traj frame 14 | 28 135 | 5 087 | 2.01 / 1.52 | MG |
| blob f19 | 45 475 | 5 087 | 1.78 / 1.55 | MG |
| blob f19×4 | 99 559 | 5 087 | 2.41 / 2.44 | MG |
| blob f19×16 | 174 915 | 5 087 | 2.62 / 2.77 | MG |
| production `F` | 78 720 | 5.7e6 | 2.27 / 2.66 | MG |
| production `P1` | 95 308 | 3.3e6 | 1.60 / 2.06 | MG |

Two rows flipped against the previous table: blob f19 was 1.03x and the
25-frame trajectory was 0.72x, and both are now MG wins: the trajectory on
every one of its 25 frames at `mg_tol = 1e-6` (mean 1.55 at one thread, §4.2).
The selector keeps a 1.5x margin: it only picks MG where the warm measurement is
at least 1.5x, and only keeps `:direct` where the conservative cold measurement
is a tie (`<= 1.25x`, ties going to the bitwise path). The rows between those
bands (f19: 1.14x cold, 1.78x warm; frame 1: 1.09x cold, 1.48x warm) are
resolved by the 20 000 constant, and the `linear=:auto` safety net
(`LAST_AUTO_FALLBACK`) turns a wrong MG pick into time lost, not a failed run.

At the two production checkpoints the selector's own convention puts both states
above the 20 000 rule: `saddle_dofs = 2*Nu + Np` is 78 720 on `F` (8 188
leaves) and 95 308 on `P1` (10 000 leaves), so the realistic card's default
`linear = :auto` resolves to MG. (The reduced `nf + nfp` after Dirichlet and
hanging-node elimination is 69 388 and 86 544, the convention of the §4.1
table.) On read-only copies with the driver's kwargs
(`stokes_tol = 1e-3` forwarded as `auto_fallback_tol` and `stokes_accept_tol`,
and the card's `mg_tol = 1e-8`) the auto solve measured 81 FGMRES iterations /
8.6e-9 on `F` and 91 / 9.4e-9 on `P1`, with zero `LAST_AUTO_FALLBACK` events;
the warm best of three was 1.41 s and 2.21 s against 3.47 s and 3.92 s for
`linear=:direct` with `stokes_scaling=:equilibrated`.  At the library default
`mg_tol = 1e-10` the same solve is 92 / 108 iterations, 7.8e-11 / 8.7e-11 and
1.61 s / 2.68 s warm; the card's five-orders-below-the-gate choice and its
accuracy cost are quantified in `docs/CASE_REALISTIC.md` §1.6.  The 8192-leaf
C-arm checkpoint (8 260 leaves, 79 308 selector dofs) also resolves to MG and
converges: 67 iterations / 7.0e-9 at the card's tolerance, 90 / 8.1e-11 at the
library default, zero fallbacks.  The historical 4.33 failure at 7 498 leaves is
removed by the exact constrained Jacobi diagonal (9250a3c); the exact
leaf-local BFBT diagonal (503fd87) then cut the iteration count and improved
the accuracy. The card comment at
`julia/scripts/case_realistic_2d.jl` carries the long form.

The contrast column is what the sampler sees, not the `16384x` of the viscosity
law parameter. Node-subset sampling was rejected because on the real frame-19
field it underestimates the contrast by up to a factor 5 and returns three
different answers on three trees (985, 5 489, 995) for one physical field, while
the fixed grid returns 5 086.6 on all of them. When `:auto` selects direct it is
bitwise identical to an explicit `:direct` (G191 asserts `==` on `u`, `p` and the
residual), and when it selects MG it agrees with an explicit `:mg` to `0.0` at
one thread. **`:auto` is the library default of `run_semquad_simulation` since
2026-11-13** (`2453f16`); `:direct` and `:mg` remain selectable, an explicit
`:direct` call is bitwise unchanged, and `CASE_LINEAR=direct` still forces it on
the card, which at the production size otherwise takes the MG path above. A
selected MG solve that
misses `auto_fallback_tol` (the driver passes its `stokes_tol`) is redone with
`:direct` and counted in `LAST_AUTO_FALLBACK`; the production rows above had
zero events.

### 4.5 `:schur_cg`, the legacy Krylov path

`linear = :schur_cg` solves the reduced saddle as a Schur-complement projected
CG: `Kred u + Bt p = fr` and `Bs u = gr`, with `Kred = CᵀKC` the constrained
viscous block and `S = Bs Kred⁻¹ Bt` the Schur complement, applied through one
factorisation of `Kred` reused for the whole solve. `Kred` is regularised by a
`1e-12` relative diagonal shift, since a constrained or hanging velocity mode
can leave a zero eigenvalue after AMR, and row/column equilibrated by
`d_i = √(max(Kred_ii, 1e-14 max_j Kred_jj))` as `Kred → D⁻¹ Kred D⁻¹` with
`D = diag(d)`. The preconditioner is the pressure mass matrix, spectrally
equivalent to `S`; `schur_maxiter` (default 200) caps the CG, and the loop stops
when the squared Schur residual falls below `1e-18` of its initial value.
`LAST_SEMSTOKES_SCHUR_ITERS` records the count and `LAST_SEMSTOKES_RESIDUAL` the
true reduced-saddle residual.

Its verdict is a scope statement. It is the default of the legacy driver
(`julia/src/production.jl:51`), it has no row in the §4.2 or §4.4 tables, and
the production card does not ask for it. The constructor rejects it wherever
the saddle is nonsymmetric: with a reference state and `mass_form = :anelastic`
the density-weighted `Bc` breaks `Bm = Bcᵀ`, and the call throws

```
mass_form=:anelastic makes the saddle system nonsymmetric (Bm != Bc'); use linear=:direct
```

Without a reference state `Bc = Bref`, the saddle is symmetric and the path is
available. Everything the modern paths bought (the equilibrated direct ladder,
the matrix-free hierarchy, the exact leaf-local BFBT diagonal) is outside it, so
it stays as the historical Krylov reference.

### 4.6 The Newton exception

`pic_mode = :newton` is the one setting that constrains the linear path from
outside the solver, and it constrains it at argument validation, before the
first step:

```
pic_mode=:newton requires linear=:direct explicitly: the library default is
:auto, which may resolve to :mg, while Newton needs one direct saddle
factorisation per iteration.  Pass linear=:direct.
```

The reason is structural: each Newton iteration assembles a Jacobian whose
`(1,1)` block is `K(η(T, ε̇, p, χ))` plus its tangents and needs one
factorisation of that saddle, while a Krylov path answers to a tolerance and
returns an approximate correction. The same block rejects a reference state,
the compressible momentum term and `nu_cbf`, with the messages
`pic_mode=:newton does not support a reference state (TALA/ALA buoyancy)`,
`... the compressible momentum term` and `... nu_cbf (the EV field of the step
is internal)`; each adds a term the analytic Jacobian does not linearise, so the
guard keeps the Newton out of a silent Picard iteration. A production Newton
run passes `linear = :direct`, and the `:auto` default is what makes that pin
necessary.

## 5. Physics features

Everything in this section is off by default, and every default-path change is
gated bitwise.

### 5.1 Internal heating `H`

`H` enters through `T̄(z) = (1-z) + (H/2) z(1-z)`, so the energy equation is
solved for the anomaly and the physical `+H` is the `-∇²T̄` of the form. The wall
fluxes use the same base gradient, `Nu_top = -(dT/dy)|_top/(1+H/2)` and
`Nu_bot = -(dT/dy)|_0/(1-H/2)`, so a purely conductive state gives
`Nu_top = Nu_bot = 1` and the steady balance is `Nu_top - Nu_bot = H` (the
`closure` column in `history.csv`). The realistic card uses `H = 10`, the low end
of the requested 10 to 20 band, which puts the conductive mid-depth temperature
at `T̄(0.5) = 1.75`. Verified by `t_g125_heating.jl`, with the `rho_heating`
variant pinned by `t_g83`.

### 5.2 Compressibility: ALA, TALA, and the EOS reference state

The three mass forms are in §1.2. TALA is the anomaly buoyancy with `∇·u = 0`;
ALA is the same buoyancy with `∇·(ρu) = 0`. The difference is measured: at
`Ra = 1e6` with `ρ = exp(1-y)`, ALA against TALA gives
`dVrms/Vrms = -6.9 percent` and `dNu/Nu = -1.6e-7`. Both are checked by
`t_g197_tala_consistency.jl` (43 assertions), including the exact zero load at
`θ = 0`, the `h²` residual and `h⁴` spurious-flow orders of a resting conductive
column, and the sign convention of the discrete pressure (`p'(y) = fz(y)`, the
hydrostatic pressure increasing downward).

The EOS reference state integrates the self-compacting adiabat downward, with
`dT/dy = -ΛT`, `dρ/dy = -ρ g/φ` and `dp/dy = -ρ g`. For
`eos = :adams_williamson` the closure `K_S = ρ c_p/α` gives `dlnρ/dy = -Λ`, the
classical Adams-Williamson relation; for `:murnaghan` the bulk modulus is linear
in pressure, which couples the density ODE to the hydrostatic pressure. The
profile is tabulated on `nz` uniform nodes by RK4 (8 substeps per interval) and
interpolated with cubic Hermite whose node derivatives are the ODE right-hand
sides. The derivatives the physics needs (`dT_ad/dz = -ΛT_ad`,
`∇²T_ad = (Λ² - dΛ/dy)T_ad`, `dρ/dz = -ρ g/φ`) are evaluated from the EOS, not
differenced from the table, so they are derivatives of the same adiabat. At
`nz = 129` the closed-form reproduction is `9.7e-12`, fourth order in `nz`.
Verified by `t_g198_eos_refstate.jl` (81 assertions).

### 5.3 Per-transition latent heat

The energy equation is written in apparent-heat-capacity form,

```
(1 + Σ_i L_i Γ_T,i) (∂θ/∂t + u·∇θ) = ∇²θ - u_z dT̄/dz - Σ_i L_i Γ_z,i u_z
```

`Γ_i(y, T)` is the phase function of transition `i` (§5.6), `Γ_T` and `Γ_z` its
derivatives, and `L_i` its dimensionless latent heat. The capacity factor
multiplies the mass term, so latent heat is implicit in the BDF2 step with no
extra lag. Applying it to both the mass term and the advection term is what
makes the discrete enthalpy `h = θ + Σ L_i (Γ_i(y,T) - Γ_i(y,T̄))`
conservative: `d/dt ∫h = ∫_wall ∂θ/∂n`, with the base part of `Γ_z` moved into
the base advection so it integrates to zero. The guard
`1 + Σ L_i Γ_T,i >= 0` is reproduced from ASPECT and throws rather than
running.

The coefficient per transition is derived from the Clapeyron slope, not fitted:
`dS_A = (dP/dT) dρ/ρ²` and `L_i = -T_i dS_A,i/(c_p dT)`, with `T_i` the
reference temperature at the phase depth. The minus sign is the difference
between conventions: the code writes `-L DΓ/Dt` where ASPECT writes
`+T dS_A DΓ/Dt`. For the realistic card this gives `-0.0594` at the 410
(exothermic, it heats a downwelling) and `+0.1306` at the 660 (endothermic, it
cools it). The pre-2026-11-08 code applied one positive `0.11` to both, which is
wrong-signed at the 410 and right by accident at the 660. Apparent capacity stays
within `[0.88, 1.12]` in the worst case, far from the guard. Verified by `t_g194`
(102 assertions), `t_g206` (68, the sign) and `t_g199` (27, the time order).

### 5.4 Composite isostress rheology

Two creep mechanisms share the applied stress, so their strain rates add:

```
η_diff = A_d⁻¹ exp((E_d + V_d(1-z))/T_nd)
η_disl = A_s⁻¹ exp((E_s + V_s(1-z))/(n T_nd)) ε̇_II^((1-n)/n)
T_nd   = max(T + T0, T_cut)
```

`T` is the code temperature, `z` the height so `1-z` is depth, `ε̇_II` the second
invariant of the strain rate with the `√(½ ε̇:ε̇)` convention, `n` the stress
exponent, `E` the activation energy, `V` the activation volume, and `T0` the
surface-temperature offset shared with the `:arrhenius` law. For `:isostress`
(the default since 2026-11-08, ASPECT's rule) the effective viscosity solves
`τ/(2η_diff) + A_s e^{-(E_s+V_s(1-z))/(n T_nd)} τ^n = ε̇_II`, i.e.
`η_eff = τ/(2 ε̇_II)`. For `:isostrain` (the legacy rule) it is the harmonic mean
`1/(1/η_diff + 1/η_disl)` with `η_disl` at the total strain rate.

The two rules agree exactly in the one-mechanism limits and at `n = 1`, and
differ by up to 38.5 percent (`n = 3`) or 43.3 percent (`n = 3.5`) where both
mechanisms contribute. The switch changes the solution: on the 64-leaf composite
case `dVrms/Vrms = -35.28 percent`, `dNu/Nu = +1.7e-8`, `|dθ|/|θ| = 6.8e-3`, and
the converged viscosity is larger by a median factor 1.3592. The analytic
derivatives agree with central differences to `4.8e-9` (`dη/dT`) and `1.9e-8`
(`dη/dε̇`), three orders inside the gate's `1e-6`, and at the `r = 1` crossover
they are 27.2 and 59.1 percent away from the legacy ones, which `t_g201` asserts
so the finite-difference check cannot pass with the wrong derivative. The
constitutive card against ASPECT reproduces ASPECT's algorithm to
`6.6e-15 .. 8.9e-15` on all four parameter sets. Gates: `t_g192` (148, the legacy
rule), `t_g201` (202, the new default), `t_g200` (39, the card).

### 5.5 Depth-dependent Drucker-Prager and strain weakening

The yield stress carries linear depth profiles on cohesion and on the friction
coefficient:

```
c(d) = c0 + c_grad d,     μ(d) = sin(φ0) + μ_grad d
τ_y  = min( f(χ) ( c(d) cos φ0 + μ(d) p ), τ_max )
```

`d = 1-y` is depth, `p` the dynamic pressure, `φ0` the surface friction angle,
and `f(χ)` the weakening factor, which is 1 until weakening is switched on. The
profile is imposed on `μ = sin φ` rather than on `φ` because `μ` is the object in
`τ_y = c + μp` and the quantity the literature profiles. Both gradients default
to `0.0`, and the validator rejects a profile that would make `c(d)` or `μ(d)`
negative anywhere in `d ∈ [0,1]`; in the zero-gradient limit the bodies are
bitwise identical to the pre-change versions across 500 parameter combinations
and five functions. Gates: `t_g195_depth_dp.jl` (2 791 assertions, the profiles
at every Gauss point and the tangents at two depths), plus `t_g162`, `t_g163`,
`t_g168` and `t_g170` for the pressure tangent, the plastic damper and the DP
Newton.

Weakening transports a scalar `χ` with the BDF2 coefficients of the energy
equation:

```
dχ/dt + u·∇χ = (1-χ) ε̇_pl/ε̇_ref - χ/τ_heal      (:rate)
dχ/dt + u·∇χ = ε̇_pl/ε̇_ref                       (:linear)
f(χ)         = 1 - (1-w_f) clamp(χ,0,1)          ∈ [w_f, 1]
```

`ε̇_pl` is the second invariant of the total strain rate while the plastic branch
is active, and zero otherwise, which is what ASPECT's `visco_plastic`
accumulates. `ε̇_ref` is the reference end strain, `w_f` the residual strength
factor, and `τ_heal` the healing time (`Inf` by default). The raw solution is not
clamped inside the solve, because clamping makes the Picard fixed point differ
from the Newton root by a measured 14 percent; the driver clamps the stored field
at the end of each step. Gate: `t_g196` (73 assertions).

### 5.6 Phase transitions and their Clapeyron slopes

`phase.jl` implements ASPECT's phase function

```
Γ(y,T) = ½ (1 + tanh( ((z₀ - z) - slope (T - T_ref)) / width ))
```

`z₀` is the transition depth in code units, `width` its half-width, `T_ref` the
reference temperature, and `slope = dz/dT` the Clapeyron slope in code units.
Buoyancy enters as `-Ra B (Γ(T) - Γ(T_ad))`, with `B = dρ/(ρ α dT)` the
dimensionless density jump; subtracting `Γ(T_ad)` keeps a state on the base
adiabat neutrally buoyant. The realistic card's two transitions are the 410
(`z₀ = 0.87`, `width = 0.025`, `slope = +0.032`, `B = 1.818`) and the 660
(`z₀ = 0.78`, `slope = -0.032`, `B = 3.030`), so an upwelling gains buoyancy at
the 410 and loses it at the 660, the textbook sign. The same `Γ` and its analytic
derivatives `Γ_T`, `Γ_z`, `Γ_TT`, `Γ_zT` feed the phase buoyancy, the latent
capacity of §5.3 and the Newton tangents; the four derivatives match central
differences to `2.6e-9`, `5.6e-10`, `4.6e-9` and `4.3e-9`. Gates: `t_g139` and
`t_g140`.

## 6. What is missing, and what is known to be limited

* **Two dimensions only, Cartesian only.** The tree is a quadtree, the elements
  are tensor products of two one-dimensional bases, and the velocity is
  `(u_x, u_z)`. There is no 3-D and no spherical shell, so no global-mantle
  geometry.
* **No TALA switch in ASPECT, so no cross-code TALA comparison.** The statement is
  measured: ASPECT 2.5.0-pre accepts only `isentropic compression | custom |
  anelastic liquid approximation | Boussinesq`, and `anelastic liquid
  approximation` hard-wires `mass_conservation = reference_density_profile`. A
  TALA run needs a pressure-independent density, which removes the
  depth-dependent reference density the ALA run has, so the difference would
  confound the mass form with the buoyancy; the TALA-equivalent configuration is
  rejected before the first step with an `is_compressible()` error.
* **The near-null pressure band at a free-slip wall.** On the G-7641 tree the
  reduced saddle has a relative eigenvalue near `1e-14`: an element-level,
  viscosity-weighted pressure checkerboard in the stiff lid rather than the
  constant mode or a constraint artefact. The band does not close with refinement
  (the uniform L5 run continues it to `k = 12`), so it is handled by solver-side
  means, namely the shift ladder, the tiny-shift fallback and optionally
  `stokes_scaling = :equilibrated`. Equilibration lifts the pressure family, but
  the exact solution's near-null component still grows with level, so `|dp|/|p|`
  is `4.4e-2` between the exact and shift-regularised representatives; the
  velocity, and therefore `Nu`, is protected (its share is `3e-7` to `4e-6`).
* **The AMR-aware coarse space for MG.** The rediscretised coarse operators are
  2 to 32x too soft because coarse Gauss points do not see the boundary layer,
  and the coarse-grid correction is 31, 412 and 842 times worse than the optimal
  correction in the same coarse space on the three coarsest levels. Semicoarsening
  that keeps the wall layer, wall-aware agglomeration, or a strong-connection AMG
  is a new component, estimated at 2 to 6 weeks.
* **MG has crossed over; the limit is one thread and one size.** After
  [MG.md](MG.md) §11-§15 the MG solve beats the accuracy-matched
  equilibrated direct solve at both production states: 1.93-2.31x at eight
  threads and 1.63-1.96x at one thread on `F`, 1.67-2.10x and 1.30-1.50x on `P1`
  (§14). The eight-thread margin is threaded MG against serial direct; the
  equal-thread rows are 1.96x on `F` and 1.50x on `P1`. The pre-tuning "11 s
  against 6.65 s" and "crossover beyond `max_leaves = 10000`" statements are
  superseded. What remains is the one-thread iteration count, the coarse-space
  work above, the serial half of the eight-thread iteration (46 % on `F`), and
  the 2-D, two-state scope: the larger-leaf-budget premise is untested, and the
  next levers are mixed precision, an augmented Lagrangian and three dimensions
  ([MG.md](MG.md) §6).
* **Latent heat is first order in time.** The enthalpy budget closes at first
  order for both the `:implicit` and the `:lagged` schemes; the apparent-capacity
  form puts the term in the mass matrix with no extra lag, and that does not buy
  second order for the coupling.
* **Internal heating and an EOS reference state cannot be combined today.** With
  a reference state, `semquad_base_T` returns `T_ad`, so `T̄(z,H)` and with it the
  only route by which `H` enters the energy equation are unused; `H` would only
  reappear through `rho_heating`, which is a different model. The realistic card
  therefore runs with `H = 10` and no reference state, and the omitted
  compressibility is the measured 6.9 percent velocity effect above.
* **`:composite` cannot carry the plastic branch**, there is no grain-size field,
  and the `:lsc` preconditioner does not exist on the matrix-free path (it throws
  an `ArgumentError` because it needs an assembled `Kred`).
* **`:auto`'s mid-contrast band has two synthetic rows and no rugged one.** The
  band `10 < contrast <= 100` used to earn MG only above 20 000 dofs, with no
  measurement behind that rung. The 2026-11-13 re-measurement added the two
  `stratified 50x` rows (smooth `eta = exp(ln50·y)` on a uniform `p=4` tree,
  2 803 and 10 851 dofs, 1.51-2.20x cold and 2.33-4.04x warm), which say the band
  costs the V-cycle nothing measurable against contrast 1. That is why the middle
  rung and the contrast-10 boundary are gone, and why the only contrast boundary
  left is 100, where the MG pressure block switches to w-BFBT (§4.4). A *rugged*
  mid-contrast AMR tree at 5-20k dofs is still unmeasured, and the 2 000 constant
  is the hedge against it. The direct path is bitwise only through UMFPACK's
  COLAMD and threshold pivoting, so any change of ordering, pivots or
  factorisation breaks the contract the gates rely on.

## 7. Where this lives in `julia/src`

`physics.jl` holds `Params`, `Tbar`/`dTbar`, `eta_at`/`deta_dT_at`, the composite
law and its isostress solve, and the heating closures. `phase.jl` holds the phase
function `Γ`, its four derivatives and the per-phase latent coefficient.
`rheology.jl` and `semquad_rheology.jl` hold the SI creep and yield pieces, the
depth-dependent yield surface, the plastic viscosity, `deta_dp`, `deta_dp_eps`,
`deta_chi` and strain-rate sampling. `semquad_refstate.jl` and `refstate.jl` hold
the EOS reference state with `T_ad`, `dT_ad/dz`, `ρ`, `α` and `c_p`.

`quadtree.jl` holds the tree, `refine!`/`coarsen!`, `neighbors`, `balance!`, the
face cache and the local 2:1 rebalance. `dgquad.jl` holds the AMR cycle, the
Kelly indicator, Dörfler marking, the leaf budget and the transfer.
`amr_balanced.jl`, `amr_advanced.jl` and `amr_dwr.jl` hold the `:balanced` policy
with its coarsening gate plus the DWR indicator. `gll.jl`, `mesh.jl` and `fe.jl`
hold the GLL nodes and weights, the Lagrange bases with their cached rules, and
the structured mesh and element tables.

`semquad.jl` holds `SemQuad`, the hanging constraints, the energy assembly, the
EV field and point evaluation. `semquadstokes.jl` holds the Stokes assembly, the
direct LU with its shift ladder, equilibration and tiny-shift fallback, `:lagged`
and the `:auto` selector. `semquad_mg.jl`, `mg.jl` and `gmres.jl` hold the
matrix-free hierarchy and V-cycle, the w-BFBT block, the contrast sampler, the
older sparse MG, and the restarted FGMRES with its true-residual stop.
`semquadenergy.jl` holds the BDF2 energy step and its latent option.

`semquadconvection.jl` holds the time driver, the CFL update, the Picard loop,
AMR scheduling, the `steady_newton` hook and checkpoint writing.
`semquad_picard.jl` holds the relaxation helpers and `semquad_step_newton.jl` the
per-step `(u,p,θ,χ)` Newton with its analytic Jacobian. `semquad_newton.jl`,
`semquad_tosi_steady.jl`, `semquad_dp_steady.jl`, `semquad_coupled.jl` and
`semquad_steady_entry.jl` hold the steady Newtons and the dispatch across them.
`semquad_weaken.jl` and `semquad_checkpoint.jl` hold the `χ` transport and the
HDF5 checkpointing. `semquaddiagnostics.jl`, `semquad_flux.jl` and `cbf.jl` hold
the Nusselt definitions, Vrms, divergence, work dissipation, the heat budget and
the CBF wall flux. `timestep.jl`, `direct.jl`, `streamfunction.jl` and
`psi_stokes.jl` hold the legacy BDF2 scaffolding, the sparse direct solve with
its true-residual contract, and the divergence-free velocity spaces.
