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

Stage 0 cut 5 (`docs/dev/GEOM_PROGRESS.md`) makes the hierarchy work on a mapped
geometry.  The coarsening is **logical**: `_mg_coarsen_tree`/`_mg_hierarchy_trees`
use leaf-index arithmetic only, so a box and a mapped geometry on one tree get the
same per-level leaf sets (`G271 (e)` checks this level by level).  What changes is
the operator data: every level's `SemStokes` space is built with the caller's
geometry (`make_semstokes(q, p; geom=ss.su.geom)`), the mapped matrix-free
coarse operators are the mapped `SemQuadKOpMapped` rediscretisations (applied by
the `stokes/apply.jl` mapped kernels, with the mapped Jacobi diagonal from
`stokes/diag.jl`), the coarsest dense operator comes from the mapped apply, and
the prolongation `_mg_prolongation_full` uses **logical** coordinates for a
mapped geometry (fine node logical position, O(maxlevel) logical descent,
`leaf_ref` on the coarse leaf; the map is a global diffeomorphism so the weights
are the same 1-D GLL interpolation weights as the box's — G271 (e) measures them
equal).  The pressure block is built from the mapped pressure mass / w-BFBT
diagonal (`pressure.jl`/`bfbt.jl` mapped bodies), and the driver's contrast
sampler covers the mapped domain (`semquad_eta_contrast` mapped method).

Every cut-5 cache key carries the geometry identity (`geometry_stamp`):
`SEMQUAD_MG_CACHE` (per tree), `SemQuadMGLevelKey` (per coarse level),
`SEMQUAD_KOP_CACHE` (per operator) and the direct path's `_saddle_key` (the
pressure-cache / stale-LU fingerprint).  The keys are extended, never re-laid
out: the *level* cache still stores topology only (`docs/dev/MG.md` policy), and
an entry is a pure function of its key, so a geometry change is a new key and a
stale hit is impossible.
