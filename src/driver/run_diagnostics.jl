# The driver-owned snapshot of the Stokes and AMR diagnostic out-parameters.
#
# `LAST_SEMSTOKES_*` (`stokes/state.jl`, `stokes/selector.jl`) and `LAST_AMR_*`
# (`amr/amr_pass.jl`, `amr/cycle.jl`) are module-level `Ref`s: the Stokes
# solvers and the AMR cycle write them as out-parameters, and the driver logs
# them.  `RunDiagnostics` is the snapshot the driver owns, and this file is the
# only place `src/driver` reads those Refs (stage 1 of the `RunState` refactor;
# the producer side is unchanged and the Refs stay where they are).

# Field types are the value types of the mirrored Refs.  For the named tuples
# the producers define, deriving the type with `typeof` keeps the snapshot
# exactly as wide as the Ref instead of restating a 27-field literal.
const _RunDiagResidSplit = typeof(LAST_SEMSTOKES_RESID_SPLIT[])
const _RunDiagAutoChoice = typeof(LAST_SEMSTOKES_AUTO_CHOICE[])
const _RunDiagAmrStats = typeof(LAST_AMR_STATS[])
const _RunDiagAmrBalanced = typeof(LAST_AMR_BALANCED[])

# The values the mirrored Refs have when this file is included, i.e. before any
# run can write one.  `RunDiagnostics()` starts from them, so a fresh snapshot
# is deterministic even in a process that has already run a simulation.
const _RUN_DIAGNOSTICS_INITIAL = (
    residual=LAST_SEMSTOKES_RESIDUAL[],
    resid_split=LAST_SEMSTOKES_RESID_SPLIT[],
    shift=LAST_SEMSTOKES_SHIFT[],
    refine_iters=LAST_SEMSTOKES_REFINE_ITERS[],
    ladder=LAST_SEMSTOKES_LADDER[],
    schur_iters=LAST_SEMSTOKES_SCHUR_ITERS[],
    scaled=LAST_SEMSTOKES_SCALED[],
    auto_choice=LAST_SEMSTOKES_AUTO_CHOICE[],
    amr_stats=LAST_AMR_STATS[],
    amr_balanced=LAST_AMR_BALANCED[])

"""
    RunDiagnostics()
    snapshot!(d::RunDiagnostics) -> d

The driver-owned snapshot of the diagnostics the Stokes solvers and the AMR
cycle publish through module-level `Ref`s.

The `LAST_SEMSTOKES_*` and `LAST_AMR_*` Refs remain the out-parameter contract
the producers write, and are not moved by this file.  `RunDiagnostics` is a
snapshot the driver owns, and this file is the only place `src/driver` reads
those Refs: `snapshot!` copies their current values into the fields, and the
driver reads only the fields.

A snapshot is valid only until the next producer call.  `run.jl` re-takes it
immediately after every `solve_flow!`, after the per-step Newton (the last
solver-side call before its residual is read) and after every AMR pass; reusing
a snapshot across one of those calls would log a stale value.

`SEMQUAD_AMR_COOLDOWN` is deliberately not a field: it is driver state, not a
producer diagnostic, and stage 1 of the refactor leaves it in `amr/cooldown.jl`.
"""
mutable struct RunDiagnostics
    residual::Float64                 # LAST_SEMSTOKES_RESIDUAL
    resid_split::_RunDiagResidSplit   # LAST_SEMSTOKES_RESID_SPLIT
    shift::Float64                    # LAST_SEMSTOKES_SHIFT
    refine_iters::Int                 # LAST_SEMSTOKES_REFINE_ITERS
    ladder::Any                       # LAST_SEMSTOKES_LADDER: nothing or the rungs
    schur_iters::Int                  # LAST_SEMSTOKES_SCHUR_ITERS
    scaled::Int                       # LAST_SEMSTOKES_SCALED
    auto_choice::_RunDiagAutoChoice   # LAST_SEMSTOKES_AUTO_CHOICE
    amr_stats::_RunDiagAmrStats       # LAST_AMR_STATS
    amr_balanced::_RunDiagAmrBalanced # LAST_AMR_BALANCED
end

RunDiagnostics() = RunDiagnostics(values(_RUN_DIAGNOSTICS_INITIAL)...)

"""
    snapshot!(d::RunDiagnostics) -> d

Refresh `d` from the module Refs the Stokes solvers and the AMR cycle write.
This is the only read of those Refs in `src/driver`; call it immediately after
a producer call (`solve_flow!`, the per-step Newton, an AMR pass), never once
per step, because the producers overwrite the Refs in place.
"""
function snapshot!(d::RunDiagnostics)
    d.residual = LAST_SEMSTOKES_RESIDUAL[]
    d.resid_split = LAST_SEMSTOKES_RESID_SPLIT[]
    d.shift = LAST_SEMSTOKES_SHIFT[]
    d.refine_iters = LAST_SEMSTOKES_REFINE_ITERS[]
    d.ladder = LAST_SEMSTOKES_LADDER[]
    d.schur_iters = LAST_SEMSTOKES_SCHUR_ITERS[]
    d.scaled = LAST_SEMSTOKES_SCALED[]
    d.auto_choice = LAST_SEMSTOKES_AUTO_CHOICE[]
    d.amr_stats = LAST_AMR_STATS[]
    d.amr_balanced = LAST_AMR_BALANCED[]
    return d
end
