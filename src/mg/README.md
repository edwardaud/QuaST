# Multigrid

The geometric multigrid behind `linear=:mg`: coarsen the quadtree level by
level, assemble or rediscretize the velocity operator on each level, cache the
levels that survive an AMR cycle, smooth with Jacobi, solve the coarsest level
directly, and precondition the pressure block with a mass/LSC/BFBT inverse
inside flexible GMRES. The Stokes residual and the mesh come from `stokes/`.
This directory holds the solver machinery only: no physics, and no direct
factorisation of the fine saddle.

| File | What it is |
|---|---|
| `bfbt.jl` | The leaf-local exact BFBT pressure diagonal and its per-block scratch. |
| `build.jl` | Build the hierarchy: coarsen the tree, assemble every level and factor the coarsest. |
| `gmres.jl` | Restarted GMRES and flexible GMRES with optional right preconditioning. |
| `hierarchy.jl` | The multigrid hierarchy types and per-level data, plus the MG profile record. |
| `levelcache.jl` | Reuse the coarse levels whose leaf set did not change across an AMR cycle. |
| `pressure.jl` | The cached w-BFBT pressure-block approximate inverse. |
| `s1solve.jl` | The inner `chol(S1)` solves of the w-BFBT pressure block. |
| `saddle.jl` | CG on the velocity block and flexible outer GMRES on the reduced saddle. |
| `vcycle.jl` | The threaded element-wise V-cycle helpers, bitwise in the thread count. |

The V-cycle needs the operator's per-level kernels, and the operator's apply
needs the V-cycle as its preconditioner: the two directories are mutually
recursive by design.  `src/stokes/README.md` states the rule that keeps the
recursion from spreading.
