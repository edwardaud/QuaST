# Gates

One file per gate: `t_g<N>_<slug>.jl`. Numbers are historical, not contiguous:
retired gates left gaps and split gates kept their number. Each is a standalone
test script that starts with `using`; most wrap one `@testset`. The 194 files
hold the suite's 136 980 assertions (136 978 passed, 2
broken, 0 failed) — the count of the stage-0.5b2 tree, recorded with its
evidence in `docs/dev/GEOM_PROGRESS.md`. (Cut 1 recorded 134 086 / 134 084 / 0 / 2
on 185 files; cut 2 adds G267 and G268; cut 3 adds G269, +151 assertions; cut 4
adds G270, +302 assertions; cut 5 adds G271, +136 assertions; cut 6 adds G272,
+245 assertions; stage 0.5a adds G273, +1 565 assertions; stage 0.5b1 adds G274, +158 assertions; stage 0.5b2 adds G275, +160 assertions.) The top-level README
still quotes the pre-cut-1 total
(133 716); the top-level markdown was outside the cut write scopes.

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
checkpoint/regression gates run to minutes. The full suite is about 22 min on
eight threads (CI allows 90 at four).
