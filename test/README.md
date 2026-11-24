# Gates

One file per gate: `t_g<N>_<slug>.jl`. Numbers are historical, not contiguous:
retired gates left gaps and split gates kept their number. Each is a standalone
test script that starts with `using`; most wrap one `@testset`. The 190 files
hold the suite's 134 852 assertions (134 850 passed, 2 broken, 0 failed) — the
count of the stage-0 cut-5 tree, recorded with its evidence in
`docs/dev/GEOM_PROGRESS.md`. (Cut 1 recorded 134 086 / 134 084 / 0 / 2 on 185
files; cut 2 adds G267 and G268; cut 3 adds G269, +151 assertions; cut 4 adds
G270, +302 assertions; cut 5 adds G271, +136 assertions.) The top-level README
still quotes the pre-cut-1 total (133 716); the top-level markdown was outside
the cut write scopes.

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
G265 is the gate of
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
