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
| `leaf_ops.jl` | Leaf refine/coarsen primitives, wall refinement and the hard leaf budget. |

`coarsen_gate.jl` and `coarsen_plan.jl` are two halves of one policy and call
each other: the plan asks the gate whether a pair may merge, and the gate reads
the plan's ledger to decide.  They are separate files because the gate is the
part that can be switched off from the card (`amr_coarsen_gate`), not because
they are independent.
