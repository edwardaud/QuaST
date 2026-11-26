# Gates

One file per gate: `t_g<N>_<slug>.jl`. Numbers are historical, not contiguous:
retired gates left gaps and split gates kept their number. Each is a standalone
test script that starts with `using`; most wrap one `@testset`.

Recorded full-suite totals: **146 101 total / 146 099 passed / 0 failed / 2 broken**, 36 m 32 s
(2026-11-26, 8 threads / 16 cores, final tree of the annulus AMR demonstration batch
(G293); log `out_julia/agent/demo_suite.log`).  The tree has 212 gate files and `runtests.jl` includes each of them exactly
once; `scripts/check_counts.jl`
(run by hand, not part of the suite) checks those counts against this record and
the other documented surfaces (`CONTRIBUTING.md`; the top-level `README.md` is
checked as a warning-only surface).  This file is the one committed record of a
full-suite run: the public pages link here for the totals, and the two surfaces
that must quote them (`README.md` and `CONTRIBUTING.md`) are checked against
this record.  The two `broken` entries are the explicit, documented
`@test_skip` environment gates in G209 and G214 -- the realistic-state
checkpoints they wait for are data-dependent states of 8-27 MB produced by long
runs and are not committed (the decision and the alternatives are recorded in
the commit that made the change); they are no longer literal tautologies.  The
batch-P0 part-1 campaign record (code guards, order-gate thresholds, test holes,
before/after evidence) lives in the commit messages of that batch.

History: the earlier cuts' recorded totals were 134 086 / 134 084 / 0 / 2 on 185
files (cut 1), then 136 980 / 136 978 / 0 / 2 on 194 files (stage 0.5b2), with
G267/G268 (+2 files), G269 (+151 assertions), G270 (+302), G271 (+136), G272
(+245), G273 (+1 565), G274 (+158) and G275 (+160) added along the way, then
137 619 / 137 617 / 0 / 2 on 199 files at `f373126` (the campaign baseline).

G266 is the geometry/mapping-layer gate of stage 0 cut 1: it rebuilds the box
coordinates, node ids, leaf/face metrics, nodal transfer, AMR indicator and the
checkpoint geometry record from the geometry layer and compares them, in
process, with the historical implementations carried in the gate (no recorded
cross-machine SHA literal), and it checks the map identities of the two
validation geometries `AffineBox`/`CurvedBox` (roundtrip, analytic `det J`,
outward normals, face/volume measures, patch test) plus the checkpoint
geometry record/rejection. G267 is the cut-2 affine exactness gate: a general
`AffineBox` (rotation + shear + anisotropic scale + shift) reproduces a
polynomial Stokes solution, a linear-shear traction solution and the
mapping-independence statement to round-off, because an affine element contains
the physical P2/P1 spaces and the Gauss rule is exact. G268 is the cut-2
cross-path gate: the mapped Stokes kernel fed the box map reproduces the
untouched box kernel entry by entry (K/Kred/B/Bm'/pressure mass and stiffness,
anelastic mass, compressibility block, traction load, loads/state pieces,
perimeter) and the direct-solve result, on a dyadic tree, a non-dyadic tree and
a hanging tree; both gates state why their assertions are not tautological in
their headers. G269 is the cut-3 energy gate: a polynomial temperature field on
an `AffineBox` and the P2 Stokes velocity are reproduced by the mapped energy
operator to solver round-off (an affine element contains P2 and the Gauss rule
is exact), a `CurvedBox` manufactured Stokes/advection-diffusion solution
converges at the design's orders measured over four levels (velocity L2 3 / H1 2,
pressure 2, temperature L2 3 / H1 2 — with the asymptotic-parallelogram caveat
recorded, not tuned), the mapped energy kernel fed the box map reproduces the
untouched box `M`/`K`/`C`/`F`, the entropy-viscosity coefficient, a full-physics
assembly and one BDF2 energy step (EV on, and with a flux wall) on a dyadic, a
non-dyadic and a hanging tree, and the map-curvature second derivatives are
checked against exact coordinate-Laplacian identities, a fourth-order
finite-difference Laplacian reference and a `height_laplacian` finite
difference — with the dropped curvature/cross-metric terms asserted to fail at
O(1) (measured, printed) rather than assumed impossible. G270 is the cut-4
diagnostics gate: on a rotated/translated `AffineBox` every per-step diagnostic
(wall Nu pair, CBF pair, Vrms, mean T, energy Nu, W/U, base advection, theta²
div u, max divergence, surface velocities, the heat budget, the exact residual,
the hmin/CFL sizes and the eta_ref interior band) equals the untouched box
bodies' value for the same physical problem at ≤2.1e-15; on a general affine map
the volume integrals and wall fluxes equal an independent high-order quadrature
of the analytic fields at ≤2.4e-15; the mapped bodies fed the box map reproduce
the box bodies on a dyadic, a non-dyadic and a hanging tree at ≤5.5e-15 (all
direct diagnostics; the cancellation-amplified CBF flux is bounded at 1e-13
with the measured 1.5e-14); a wrong mapped normal and a wrong face Jacobian are
caught with measured 22.2 % / 25.0 % relative failures; and the mapped driver
runs end to end on `AffineBox`/`CurvedBox` (heating off/on, latent, CBF and
exact-residual columns) with the mass-flux identities and the time-step CBF
energy-budget identity (ID1) closing at ≤6.6e-14. G271 is the cut-5 mapped
multigrid/matrix-free gate: the mapped sparse `K` equals the mapped
matrix-free apply to ≤3.5e-16 (round-off), mapped `linear=:mg` equals mapped
`linear=:direct` to ≤1.3e-9 on a uniform and a hanging tree (both at their own
true reduced residuals), two geometries on one tree give two operators, cache
entries, hierarchies and `_saddle_key`s with a **zero-allocation** warm mapped
apply while the *pre-cut* key is shown (reconstructed in the gate) to collide and
the collision's failure is measured (the box operator on the mapped system is
solved silently to round-off at 14 % wrong velocity; a box hierarchy is a 4.1×
worse preconditioner), the coarse levels are the box hierarchy's logical
coarsenings with mapped coarse operators and bitwise-equal prolongation, the
threaded mapped V-cycle is bitwise the serial one, and the mapped driver runs
`:direct`/`:mg`/`:auto` (auto resolves to `:mg` above the selector crossover)
with `:schur_cg`/`:lagged` still refused with their box-assumption message.
G272 is the cut-6 mapped transfer/indicator/AMR gate: a logical P2 is transferred exactly (max error 0.0) by the nodal transfer on an `AffineBox` and a `CurvedBox` (refinement and coarsening) and by the mapped L2 transfer (3-4e-15), the mapped nodal transfer on the box map is bit-identical to the historical box body, a deliberately wrong ``box-formula'' evaluation point on `CurvedBox` is caught with a measured 4.4e-1 error (correct: 0.0), and a smooth coarsening shows L2 beating nodal by four orders of magnitude on the G80 wall-graded trees.  The mapped indicator equals an independent analytic-map oracle (closed-form map, Jacobian, face normals and normal sizes written in the gate) to 3.7e-16, is invariant under a rigid rotation of geometry+field (7e-15), is non-negative, reproduces a corrected box oracle bitwise on the box map -- and differs from the **historical** box body by 9.96e-1, because the pre-cut box body computes the fine-side horizontal-face derivative in the x index (a pre-existing defect, kept for the nine-hash bitwise invariant and recorded as the cut's main finding); the linear-field probe measures the spurious interior term (6.25e-4) and the mapped body's zero.  The physical-vs-logical decision is measured on a `CurvedBox` whose layer-direction physical cell size varies by the annulus's `(R0+1)/R0 = 2.818`: the per-leaf `eta_phys/eta_log` equals the analytic map factor (7.9e-11 ... 8.4e-16 at levels 4-6), the compressed/tall span converges 2.592 -> 2.716 -> 2.770 towards 2.818, and on a similarity (box) map the two estimators coincide exactly -- the physical choice is justified by a number, not an opinion.  End to end, a mapped driver run with `initial_refine = 2` and `amr_every = 2` completes four steps, evolves 16 -> 28 leaves, writes 4x18 history rows, keeps every state finite, and checkpoints/resumes bit-identically to the uninterrupted run (the geometry record is in the checkpoint config); the Tosi / yielding law paths and the velocity indicator run mapped; the still-refused mapped paths (`symmetry=:x`, the Cartesian-y AMR policies, Newton, the Maxwell elastic load, `:schur_cg`) name their missing piece.  The box half (158 assertions) pins the indicator, nodal transfer, wall-strip/protect predicates, the box-default cycle/wall refinement/`wall_graded_tree` and the box-map transfer against the historical bodies carried in the gate.  G265 is the gate of
G273 is the stage-0.5a polar-annulus geometry gate: it constructs `PolarAnnulus` for five fixtures (the published-unit-gap quarter wedge, the published 1.22/2.22 half wedge whose Float64 gap is `1+2^-52`, a three-quarter wedge, a full-turn cut ring, and a thick annulus with gap 2) on three trees and pins the map and its analytic inverse geometrically (radius, angle, collinearity, node counts), `det J` against the closed form and against a fourth-order finite difference of the public map, `J^T J^{-T} = I` and the `(xi_x, eta_x, xi_y, eta_y)` convention, the map Hessian against closed form and finite differences, the plane-annulus measures (`total_measure` and a quadrature-only `sum(detJ*w)` equal `Phi/2 (R1^2-R0^2)` to 4e-16, wall lengths `Phi R0` / `Phi R1` / `R1-R0`, perimeter `Phi(R0+R1)+2(R1-R0)`, leaf min/vertical/normal sizes), the registry and the face metric at arbitrary face parameters (inner `-e_r`, outer `+e_r`, radial sides `-/+e_phi`, tangents `+e_phi`/`+e_r`, analytic face Jacobians, unit length and orthogonality), `height = r-R0`, `|grad h| = 1`, `lap h = 1/r` (also against a finite difference), `gravity = -e_r` with unit magnitude, `nu_wall_sides = (:xminus, :xplus)`, the constructor refusals and the seam decision (`:periodic` is refused with an actionable topology message; a `Phi = 2*pi`, `seam = :wall` cut ring is constructible and its perimeter counts the seam).  The second-derivative block is the annulus analogue of G269 (d): the exact physical Laplacians of `S, T, S^2, T^2, S*T` come out at 2.0e-12 while the deliberately dropped map-curvature term fails by 15.07, and the FD Laplacian of a smooth nodal field through `semquad_eval` agrees to 2.9e-10 (dropped-C 2.4e-1) on 256 interior quadrature points.  The canonical identity pins the exact checkpoint string including the seam, each parameter moves the identity and the cut-5 cache stamp, and the annulus stamp cannot alias the box's.  1 565 assertions in 5.6 s; the box path is untouched (nine state-dump hashes unchanged).

**Remediation batch P0 part 1 (2026-11-25): strengthened old gates and the
closed test holes.**  G24, G28, G36 and G37 were re-measured with a *quadrature
norm* on four levels (G24 keeps its three mixed-tree levels) and now assert the
observed convergence order of every consecutive pair with G269-style margins:
G24 Poisson L2 >= 2.7 and H1 >= 1.8 (measured 3.075/3.040 and 2.19/2.08); G28
and G37 velocity L2 >= 2.7, velocity H1 >= 1.8, pressure L2 >= 1.7 (measured
3.197/3.061/3.022, 2.139/2.026/2.015, 2.419/2.220/2.079 for G28; 3.070/2.979/
2.998, 2.154/2.017/2.010, 2.133/2.065/2.050 for G37); G36 BDF2 >= 1.7 on every
dt rung (measured 2.060/2.158/2.310) with a **first-order backward-Euler
control** asserted <= 1.4 (measured 0.986/0.995/1.002), and its old single ratio
threshold of 1.5 is gone.  Nothing was loosened: the old thresholds (2.0 / 2.0 /
1.0 / 1.5) pass implementations the new ones reject; the old and new thresholds
and the measured tables are recorded in the gate headers.
G281 (23 assertions) is the restart-guard geometry comparison (R5: a direct
caller's config without the `geometry` key must not resume an annulus
checkpoint), with the reachable driver restart still bitwise.  G282 (43) is the
named non-finite-input battery (F5: Poisson/Stokes body forces, viscosity, theta
at assembly, driver entry and per-step entry), with valid controls unchanged.
G283 (18) is the positive-viscosity rule for the raw solve API (F6), with the
known-answer homogeneity rescaling as the "valid field unchanged" control.
G284 (39) is the `p >= 2` space-constructor rule and the internal Q_1 pressure
builder (F7), bitwise against the pre-fix body for p = 2, 3, 4.  G285 (22) is
the time-window/step-card validation (F8).  G286 (4 257) is a known-answer test
of the entropy-viscosity operator on a uniform-gradient steady state: analytic
globals, per-leaf `Rmax`/`nu`, the assembly `ev_out` field, and the exact
`K(nu=1) == 2 K(nu=0)` scaling; the audit's m03 mutation makes it red.  G287
(25) pins the curved-wall `:rigid` hanging-node pass white-box, after measuring
that the audit's proposed physical fixture is structurally unreachable on a
balanced quadtree (0 such nodes over clustered ring/wedge trees -- recorded);
m16b makes it red.  G288 (68) discriminates the 1e-10 constraint tolerance with
a near-coincident-but-distinct node fixture (both face orientations, box and
mapped bodies); m17/m17b make it red.  The four tautology/skip sites were
cleaned: G202 and G216 got real assertions, and the two G209/G214 skips became
explicit documented skips with a reason (the broken count stays 2, by the
recorded decision).  See the commit messages of the batch for the mutation logs,
the box nine-hash/allocation evidence and every mistake found on the way.
`examples/subduction_2d.jl` (the
regional subduction-style case). It pins only the checks the design classifies
as exact identities or as the mass-flux conservation identity, with their
stated tolerances. The example's open-box heat budget, plate-cooling profile
and resolution sequence are `rheology` / `phases` / `full` / `converge` mode
outputs, not gate pins: they are asymptotic or discretisation-level comparisons
and cost minutes each, which is stated in the gate header as well.

| File | What it is |
|---|---|
| `runtests.jl` | Includes every `t_g*.jl` in order inside `@testset "QuaST gates"`; this is the suite CI runs. |
| `run_gates.jl` | Includes the gate files named on the command line, in one process. |

From the repository root, one gate:

    julia --project=. test/run_gates.jl t_g229_default_balanced.jl

and the whole suite:

    julia --project=. --threads=8 test/runtests.jl

A kernel gate such as `t_g0.jl` takes about 2 s warm; the heavy
checkpoint/regression gates run to minutes. The full suite is 30 m 45 s on eight
threads (CI allows 90 at four); the campaign baseline at `f373126` measured
30 m 49 s.
