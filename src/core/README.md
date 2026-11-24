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
| `eval.jl` | Point evaluation of a quadtree SEM field: leaf search and a reusable Lagrange workspace. |
| `fe.jl` | Reference-element data: GLL nodes for the basis, Gauss–Legendre points for quadrature. |
| `geometry.jl` | Geometry/mapping layer: `CartesianBox` (production), `AffineBox`/`CurvedBox` (validation), point map and inverse, Jacobian/metric and face normals, the map-curvature second derivatives, logical boundary registry, height/depth/gravity, checkpoint identity. |
| `gll.jl` | GLL nodes and weights, the barycentric Lagrange basis, Gauss–Legendre quadrature. |
| `parallel.jl` | Threading helpers shared by the leaf-blocked assembly loops. |
| `poisson.jl` | SIPG/Nitsche Poisson operator: a verification-only assembler and its constrained solve. |
| `quadtree.jl` | The 2:1-balanced quadtree: refine, coarsen, neighbours, balance and wall grading. |
| `quadtree_balance.jl` | The x-mirror symmetry and the 2:1 balancing policy. |
| `quadtree_table.jl` | The flat face table the quadtree queries are answered from. |
| `semquad.jl` | The continuous Q_p space on the quadtree, its mass matrix and constraints. |
| `transfer.jl` | Solution transfer between quadtree SEM meshes: nodal restriction and L2 projection. |

`quadtree.jl`, `quadtree_table.jl` and `quadtree_balance.jl` are three views of
one object: the refinement primitives maintain the flat face table, the table
build asks the balancer whether a leaf set is legal, and the balancer splits
leaves.  The call graph between the three is therefore cyclic *by construction*;
they are split for readability, and a change to any of them is a change to the
tree itself.

`geometry.jl` is the layer the discretisation is expressed on: the
`CartesianBox` methods are the historical box arithmetic (they must stay
bitwise identical, `docs/dev/GEOM_DESIGN.md` §4), the two validation geometries
exercise the general mapped path, and the logical topology helpers
(`global_logical`, `leaf_ref`, `locate_leaf_logical`, `face_pairs_logical`) keep
node identity and face pairing independent of the metric.

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

