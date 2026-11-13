# Experimental AMR strategies, opt-in only.  The production path in
# semquadconvection.jl remains unchanged; pass amr_strategy=:threshold to use
# this file.  It replaces the fixed refine/coarsen fractions by thresholds
# relative to the current maximum indicator, with a dead band
# (coarsen_thresh < eta/eta_max < refine_thresh) that suppresses churn.
#
# The `amr_policy = :balanced` policy lives in `amr_balanced.jl` and is the
# default of `run_semquad_simulation` since 2026-11-13.  `amr_policy=:classic`
# is the historical path (and the `semquad_amr_passes!` default).

"""
    amr_cycle_hysteresis!(qt, eta; refine_thresh=0.5, coarsen_thresh=0.1, ...)

Experimental threshold/hysteresis AMR cycle.  Cells above
`refine_thresh * max(eta)` are refined, cells below
`coarsen_thresh * max(eta)` are coarsened, and the dead band is left
untouched.  `max_leaves` defaults to `typemax(Int)` (no hard budget); this is
deliberately different from the legacy fraction+cap path.
"""
function amr_cycle_hysteresis!(qt::Quadtree, eta::Dict{NTuple{3,Int},Float64};
                               refine_thresh::Float64=0.5,
                               coarsen_thresh::Float64=0.1,
                               max_level::Int=typemax(Int),
                               max_leaves::Int=typemax(Int),
                               symmetry::Symbol=:none,
                               protect::Set{NTuple{3,Int}}=Set{NTuple{3,Int}}())
    leaves = collect(qt.leaves)
    isempty(leaves) && return 0, 0
    emax = maximum(values(eta))
    emax <= 0.0 && return 0, 0
    nref = count(lf -> eta[lf] > refine_thresh * emax, leaves)
    ncoa = count(lf -> eta[lf] < coarsen_thresh * emax, leaves)
    n = length(leaves)
    return amr_cycle!(qt, eta;
                      refine_frac=min(1.0, nref / n),
                      coarsen_frac=min(1.0, ncoa / n),
                      marking=:error_fraction, max_level=max_level,
                      max_leaves=max_leaves, symmetry=symmetry,
                      protect=protect)
end
