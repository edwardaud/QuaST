# Core

The numerical substrate every other layer stands on: GLL nodes and
Gauss–Legendre quadrature, the reference element, the 2:1-balanced quadtree,
the geometry/mapping layer the discretisation lives on, the continuous Q_p
space with exact hanging-node constraints, point evaluation, solution transfer
between meshes, and the threading helpers for leaf-blocked assembly.
`poisson.jl` is a verification-only SIPG/Nitsche operator, not part of the
mantle solve. There is no viscosity, buoyancy, Stokes or AMR policy here;
those live in `physics/`, `stokes/` and `amr/`.

| File | What it is |
|---|---|
| `constraints.jl` | Hanging-node constraints: a fine face node is the coarse face's GLL interpolation (historical box body plus a logical mapped method). |
| `eval.jl` | Point evaluation of a quadtree SEM field: leaf search, the logical-point evaluation and gradient primitives the mapped transfer/indicator use, and a reusable Lagrange workspace. |
| `fe.jl` | Reference-element data: GLL nodes for the basis, Gauss–Legendre points for quadrature. |
| `geometry.jl` | Geometry/mapping layer: `CartesianBox` (production), `AffineBox`/`CurvedBox` (validation), `PolarAnnulus` (stage 0.5a, the 2-D polar annulus), point map and inverse, Jacobian/metric and face normals, the map-curvature second derivatives, logical boundary registry, height/depth/gravity, physical measures and normal leaf size, checkpoint identity. |
| `gll.jl` | GLL nodes and weights, the barycentric Lagrange basis, Gauss–Legendre quadrature. |
| `guards.jl` | Named, allocation-free input guards (remediation batch P0 part 1): the non-finite-field scan of the assembly/driver entry points (`_require_finite!`, `_require_step_theta!`), the non-finite load sample (`_require_load!`) and the positive-viscosity rule (`_require_eta!`, AUDIT_VV F5/F6). The success path is one comparison per element and adds 0 B to the box hot path; the throw path names the field, the value and (where known) the coordinate. |
| `parallel.jl` | Threading helpers shared by the leaf-blocked assembly loops. |
| `poisson.jl` | SIPG/Nitsche Poisson operator: a verification-only assembler and its constrained solve. |
| `quadtree.jl` | The 2:1-balanced quadtree: refine, coarsen, neighbours, balance and wall grading (the cut-6 height-based wall grading of `wall_graded_tree`); the stage-0.5c1 `seam` topology (`:none`/`:periodic`), the wrapped cross-seam pairs and the periodic top-level pair list. |
| `quadtree_balance.jl` | The x-mirror symmetry and the 2:1 balancing policy. |
| `quadtree_table.jl` | The flat face table the quadtree queries are answered from; on a periodic tree the two seam columns are cross-face neighbours. |
| `semquad.jl` | The continuous Q_p space on the quadtree, its mass matrix and constraints; node identity across a periodic seam and the tree/geometry seam agreement. The public `make_semquad` requires the velocity/temperature degree `p >= 2` (AUDIT_VV F7); the `Q_{p-1}` pressure space is built by the internal `_make_semquad_space` that `make_semstokes` calls. |
| `transfer.jl` | Solution transfer between quadtree SEM meshes: nodal restriction and L2 projection; the mapped method evaluates the old field at the new node's logical position and uses the mapped mass; a ring transfer keeps the seam trace exact. |

`quadtree.jl`, `quadtree_table.jl` and `quadtree_balance.jl` are three views of
one object: the refinement primitives maintain the flat face table, the table
build asks the balancer whether a leaf set is legal, and the balancer splits
leaves.  The call graph between the three is therefore cyclic *by construction*;
they are split for readability, and a change to any of them is a change to the
tree itself.

`geometry.jl` is the layer the discretisation is expressed on: the
`CartesianBox` methods are the historical box arithmetic (they must stay
bitwise identical), the two validation geometries
exercise the general mapped path, and the logical topology helpers
(`global_logical`, `leaf_ref`, `locate_leaf_logical`, `face_pairs_logical`) keep
node identity and face pairing independent of the metric.  `logical_map(g, S, T)`
takes the **global logical** `(S,T) in [0,1]^2`; each geometry applies its own
physical scaling internally (the box's `x = Lx*S`), so a caller must never pass
a physical-like `Lx*S` (the stage-0.5 audit's D1 defect in
`mg/pressure.jl`'s contrast sampler).

Two measures are easy to confuse: `total_area(qt)` (`quadtree.jl`) is the
logical E13 box-tree measure, `sum leaf_hx*leaf_hy` — it has no geometry, and
since the stage-0.5 fix campaign it **refuses a periodic-seam tree** with an
`ArgumentError` pointing at `total_measure(g, lay)`; the physical normaliser of
a mapped domain is `total_measure(g, lay)` (stage-0.5 audit defect B4).  A
bounded mapped tree (a wedge) is topologically indistinguishable from a box tree
(the tree stores no geometry), so a caller of that form must pass its geometry
explicitly; there is no default that can be right.

`PolarAnnulus` (stage 0.5a) is the first
production-shaped curved geometry: one section at the end of the file defines
the map `r = R0 + (R1-R0)*S`, `phi = Phi*T`, its analytic inverse, the leaf
Jacobian `det J = Phi*r*(R1-R0)/(4*nx*ny)` and the map Hessian, the radial
height/gravity (`h = r - R0`, `grad h = e_r`, `lap h = 1/r`, `g = -e_r`), the
circle/seam boundary registry (`:inner`/`:outer` as `:wall`, the seam sides as
`:seam_lo`/`:seam_hi`, `:periodic` for the ring) and `nu_wall_sides =
(:xminus, :xplus)`.  It consumes the generic
mapped leaf/face/metric/measure/cache bodies unchanged -- that is the stage-0
layer's acceptance test -- and G273 pins the map, Jacobian, inverse,
second-derivative surface, normals and the plane-annulus measures against
closed forms.  The radial physics wiring is stage 0.5b.  One layer-level
consequence of adding the fourth concrete subtype is recorded in the section
itself: `SemQuad.geom` is a non-parametric `AbstractGeometry` field, so the
geometry API's abstract-typed call sites now see four candidate methods; the
default inference budget (`max_methods = 3`) returns `Any` for them, which
boxed the cut-5 cache key (+128 B per warm matrix-free apply, G271) and widened
`@inferred` box diagnostics (G188).  `geometry.jl` raises the budget to 4 for
the 14 one-method-per-subtype API functions with
`Base.Experimental.@max_methods` declarations (a compiler budget, no body or
value change), restoring the three-subtype inference exactly; a fifth geometry
must raise them to 5 and re-check G188/G271.

The **map curvature** (`element_hessian`, `map_curvature`, `lap_phi`, cut 3) is
the second-derivative surface the energy path needs: the physical Laplacian of a
reference-basis function is `sum M[j,k] f,xi_j xi_k + sum C[j] f,xi_j`, with
`M = J^{-T} J^{-T}` the metric tensor and
`C[j] = -sum_p Jinv[p,j] tr(M H_p)` the map-Hessian term `H_p` that vanishes for
an affine map.  The mapped method evaluates it from `jacobian_inverse` and
`element_hessian` (implemented analytically for `CurvedBox`; a new curved
geometry must implement `element_hessian`); the `CartesianBox` method returns
the **historical** entropy-viscosity coefficients `(2.0/h)^2` in both directions
with `h = leaf_hy` and zero cross/curvature terms, because the historical box EV
body assumes square leaves (`GEOM_DESIGN` §1.6 item 1/F4) and stays bitwise.
`upward_velocity` is the `u . grad h` accessor the mapped directional physics
terms use (the box method is the historical `u_z`).

The **physical measures** (cut 4, `GEOM_DESIGN` §2.3) are the diagnostics' entry
to the metric: `leaf_measure`, `total_measure`, `boundary_measure`,
`leaf_min_size`, `leaf_vertical_size`, the tree-wide `minimum_leaf_size` /
`minimum_leaf_height`, and `nu_wall_sides` (which logical sides the Nusselt pair
integrates).  Their box methods are the historical expressions bit for bit
(`total_area = sum(leaf_hx*leaf_hy)` E13, the E8 GLL wall weights,
`min(leaf_hx,leaf_hy)` / `leaf_hy` E15); the mapped methods integrate
`quad_measure`, the face Jacobian at the wall GLL nodes and the physical extents.
The wall-integrating diagnostics use the mapped face metric at the **GLL** nodes
(`_mapped_face_metric` with `sq.xi`) because the box wall bodies use the GLL
wall rule (E8), so the box-map cross-check in G270 is a round-off comparison.
The measures are cold code: they are never called inside an operator kernel.

The **metric / curvature / face cache** (cut 5, `GEOM_DESIGN` §2.5) makes the
warm mapped path allocation-free: `leaf_metric`, `map_curvature` and
`_mapped_face_metric` read per-leaf tables cached in `SEMQUAD_GEOM_CACHE`,
keyed exactly by `(geometry_stamp(g), _leafset_hash(leaves), nleaf, nx0, ny0,
Lx, p, nq)` (`GeometryCacheKey`).  `geometry_stamp` is the 64-bit deterministic
stamp of `geometry_identity`'s parameter set (raw `Float64` bits + type tag), so
a box and a mapped geometry on the same tree can never share an entry; the entry
is a pure function of its key (never invalidated: a new geometry/tree/degree/
rule is a new key; `empty_semquad_geometry_cache!()` is for memory management and
gates only).  The three tables are lazy slots on one entry: the volume metric,
the map curvature and the per-side face metric are built only when a kernel asks
for them, so a face-only wall loop builds no volume table.  The box path never
touches the cache (its methods are the historical seven-scalar expressions and
the 4-argument `leaf_metric` on a `CartesianBox` returns the 3-argument box
metric).  `semquad_geometry_cache_stats()` reports the live entries.

The **logical face-pair list** is cached the same way
(`LOGICAL_FACE_PAIR_CACHE`, keyed by leaf-set fingerprint + layout in
`quadtree.jl`): it is topology and must not see the geometry (`GEOM_DESIGN`
§2.1 rule 4), so its key carries no geometry stamp and a geometry change never
invalidates it.  Before cut 5 every mapped `semquad_constraints` rebuilt the
list (~1.5 MB per call at 1024 leaves, cut-2 record); the mapped constraints and
assemblies now reuse it, with the same read-only contract as the box
`face_pairs` cache.

The **transfer and the AMR indicator** are the cut-6 consumers of that logical
identity.  `transfer_semquad` dispatches on the two spaces' geometries: the
`CartesianBox` pair runs the historical body verbatim (bit for bit), and a
mapped pair evaluates the old field at the new node's **logical** position
(`global_logical` + `locate_leaf_logical` + `leaf_ref`) while the node's
coordinate is the map image — the map is a global diffeomorphism, so the
transfer is exact for a logical polynomial of the space's degree.  The L2 arm
uses the mapped `semquad_mass` (the historical box body plus a mapped body whose
weight is the cached `quad_measure`).  The gradient-jump indicator's mapped body
is the **physical** estimator `sum (h/2) int [du/dn]^2 ds` with the true
one-sided physical normal derivative, the physical face Jacobian and the
physical normal leaf size `leaf_normal_size`; its box body is the historical
arithmetic, which contains an index defect on the fine side of a horizontal face
kept for the bitwise invariant.
`_semquad_eval_ref!` and `_semquad_grad_ref` are the logical-point evaluation
and physical-gradient primitives both use; `leaf_normal_size` (box: the
historical `leaf_hx`/`leaf_hy`; mapped: `2/|grad xi|` at the face midpoint)
completes the §2.3 measure API.

**Stage 0.5c1: the periodic ring.**  The annulus's logical `T = 0` and `T = 1`
sides are one physical curve, and that is a quadtree *topology* feature, not a
mapping one.  `Quadtree(...; seam = :periodic)` closes the logical `y`
direction: the flat face table wraps the last column's `:yplus` face to the
first column and the first column's `:yminus` face to the last (a substitution
of the column index with an in-range one, so the hot path stays index
arithmetic), the ancestor walk then finds a cross-seam coarse/fine neighbour
with the same code as an interior face, `balance!` therefore sees the seam and
enforces the 2:1 condition across it, and `_logical_node_key` canonicalises the
`T = 1` numerator (`2*ny0*2^maxlevel`) to `T = 0`, so the two seam images are
one node with one physical coordinate.  The top-level pair list
(`face_pairs(qt, geom)` / `face_pairs_logical`) makes its two `y` direction
tests circular and its cache key carries the `seam`; the bounded-box
`face_pairs(qt)` refuses a periodic tree rather than silently dropping the
seam.  `PolarAnnulus(...; seam = :periodic)` is accepted only for `Phi == 2*pi`
(exactly: a gap is not a ring, and the angle is part of the geometry
identity/stamp) and its registry marks the two seam sides `:periodic`;
`check_seam_compat` (called by `make_semquad`, so every space -- including the
MG levels -- is checked, and by the driver before setup) refuses a ring
geometry on a bounded tree and a periodic tree on a bounded geometry.  The
hanging-node constraints (the cross-seam pair is a normal coarse/fine pair, so
the seam trace is the coarse polynomial exactly), the transfer (the matrix of
a mapped transfer is unchanged; the seam nodes are already one dof), the MG
hierarchy (`_mg_copy_tree` keeps the flag, the prolongation evaluates the
coarse field at the fine node's logical position, and the hierarchy/level-cache
keys carry the seam), the balanced/AMR cycle (the indicator wraps the coarse
abscissa of a cross-seam pair into the coarse leaf) and the checkpoint (the
HDF5 attribute `seam`; absent in old files = bounded) all consume the wrapped
pairs unchanged.  G276 pins the node identity, the table-vs-scan oracle
agreement, the balance mirror, the space-level C0 seam, the periodic transfer,
the Couette ring gate, the ring MG, the rotation-gauge/pressure-pin composition
and an end-to-end driver run.
