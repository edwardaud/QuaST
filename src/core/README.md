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
