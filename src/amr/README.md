# AMR

The balanced AMR policy and the loop that drives it: the gradient-jump
indicator marks leaves, the quartet gate and plan decide what may merge,
cooldown hysteresis and the leaf budget control the exchange, `leaf_ops.jl`
holds the pair primitives and the hard budget, and `hooks.jl` holds the AMR
pass and wall refinement that the driver calls. The quadtree itself is in
`core/`; this layer assembles nothing.

| File | What it is |
|---|---|
| `amr_pass.jl` | One adaptive pass: indicator, marking, coarsening, balance, transfer. |
| `wall_refine.jl` | Refining the wall strips to a minimum level before a run. |
| `coarsen_gate.jl` | The 2:1-aware `coarsening_allowed` gate (deal.II port). |
| `coarsen_plan.jl` | Quartet geometry, the coarsening plan and the live-gated funding merge. |
| `cooldown.jl` | Hysteresis state of the balanced cycle (`AmrCooldown`). |
| `cycle.jl` | The balanced cycle: thresholds, plan execution, budget exchange and report. |
| `hooks.jl` | The AMR pass (`semquad_amr_passes!`) and wall refinement that the driver calls. |
| `indicator.jl` | The gradient-jump error indicator of the continuous SEM field. |
| `leaf_ops.jl` | Leaf refine/coarsen primitives, wall refinement, the hard leaf budget and the cut-6 physical wall-strip / protect predicates. |

`coarsen_gate.jl` and `coarsen_plan.jl` are two halves of one policy and call
each other: the plan asks the gate whether a pair may merge, and the gate reads
the plan's ledger to decide.  They are separate files because the gate is the
part that can be switched off from the card (`amr_coarsen_gate`), not because
they are independent.

Stage 0 cut 6 routes the adaptive layer through the geometry.  The
**gradient-jump indicator** (`indicator.jl`) dispatches on the space's geometry:
the box body is the historical arithmetic verbatim; a mapped body computes the
**physical** estimator `eta(leaf) = sum_faces (h/2) int_face [du/dn_phys]^2 ds`
with the true one-sided physical normal derivative, the cached physical face
metric (`_mapped_face_metric`) and the physical normal leaf size
(`leaf_normal_size`), and the coarse side of a hanging face is evaluated at the
fine node's logical position on the coarse face (no physical-coordinate test).
The design's §6.8 physical-vs-logical decision is measured in G272 (c).  The
**wall strip** of the balanced cycle, the AMR pass's wall protect set and
`refine_wall_to_min!` are physical-height criteria on a mapped geometry
(`_amr_in_wall_strip` / `_amr_wall_leaf` in `leaf_ops.jl`, using `height` and
`nu_wall_sides`; the box methods are the historical Cartesian-`y` predicates).
They consume the physical indicator, so the strip and the protect set are where
the estimator's units matter.  The cycle, wall refinement and
`wall_graded_tree` take an optional `geom` (default the box) and their box call
sites are the historical ones, so the box path is bitwise and allocates exactly
what it did before the cut (G272 (e), the dump, the A/B).
