# Core

The numerical substrate every other layer stands on: GLL nodes and
Gauss–Legendre quadrature, the reference element, the 2:1-balanced quadtree,
the continuous Q_p space with exact hanging-node constraints, point evaluation,
solution transfer between meshes, and the threading helpers for leaf-blocked
assembly. `poisson.jl` is a verification-only SIPG/Nitsche operator, not part of
the mantle solve. There is no viscosity, buoyancy, Stokes or AMR policy here;
those live in `physics/`, `stokes/` and `amr/`.

| File | What it is |
|---|---|
| `constraints.jl` | Hanging-node constraints: a fine face node is the coarse face's GLL interpolation. |
| `eval.jl` | Point evaluation of a quadtree SEM field: leaf search and a reusable Lagrange workspace. |
| `fe.jl` | Reference-element data: GLL nodes for the basis, Gauss–Legendre points for quadrature. |
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
