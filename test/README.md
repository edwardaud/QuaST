# Gates

One file per gate: `t_g<N>_<slug>.jl`. Numbers are historical, not contiguous:
retired gates left gaps and split gates kept their number. Each is a standalone
test script that starts with `using`; most wrap one `@testset`. The 188 files
hold the suite's 134 414 assertions (134 412 passed, 2 broken, 0 failed) — the
count of the stage-0 cut-3 tree, recorded with its evidence in
`docs/dev/GEOM_PROGRESS.md`. (Cut 1 recorded 134 086 / 134 084 / 0 / 2 on 185
files; cut 2 adds G267 and G268; cut 3 adds G269, +151 assertions.) The
top-level README still quotes the pre-cut-1 total (133 716); the top-level
markdown was outside the cut write scopes.

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
O(1) (measured, printed) rather than assumed impossible. G265 is the gate of
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
