# Saturation (time-to-steady) protocol for route C.
#
# Route C deliberately has no separate steady-state solver (see
# STOKES_DECISION.md section 0: route B's solver converged to the conductive,
# pseudo-weak and nonphysical branches, and the steady problem has more than one
# self-consistent discrete solution).  Benchmarks are therefore reached by time
# integration, and the question "is it saturated?" is answered from the
# diagnostic history:
#
#   * take the last `window` steps;
#   * for each monitored quantity compute the spread (max - min) over the window;
#   * call it saturated when every spread, relative to that quantity's mean, is
#     below `rtol`;
#   * report half the spread as the error bar.
#
# This is the same "window average + drift" protocol ASPECT's benchmarks use, and
# it is deliberately a *report*, not an automatic stop: a periodic solution never
# satisfies it (Tosi Case 5a is reported over one full period instead).

const SEMQUAD_SAT_COLUMNS = (3 => :Nu_top, 4 => :Nu_bot, 5 => :Vrms, 6 => :mean_T)

"""
    semquad_saturation(hist; window=5, rtol=1.0e-3, columns=SEMQUAD_SAT_COLUMNS)

Decide whether the diagnostic history `hist` (the `Vector{Vector{Float64}}`
returned by `run_semquad_simulation`) is saturated, and give the error bars.

Returns a named tuple
`(saturated, t_from, t_to, drift, err, n)` where `drift` and `err` are named
tuples keyed by the monitored quantities; `err` is half the window spread.
"""
function semquad_saturation(hist::Vector{Vector{Float64}};
                            window::Int=5, rtol::Float64=1.0e-3,
                            columns=SEMQUAD_SAT_COLUMNS)
    window >= 2 || throw(ArgumentError("window must be >= 2"))
    rtol > 0.0 || throw(ArgumentError("rtol must be > 0"))
    n = length(hist)
    n < window && return (saturated=false, t_from=NaN, t_to=NaN,
                          drift=NamedTuple{map(last, columns)}(fill(NaN, length(columns))),
                          err=NamedTuple{map(last, columns)}(fill(NaN, length(columns))),
                          n=n)
    rows = hist[(n - window + 1):n]
    drift_vals = Float64[]
    err_vals = Float64[]
    for (col, _) in columns
        vals = [r[col] for r in rows]
        lo, hi = extrema(vals)
        mu = sum(vals) / length(vals)
        push!(drift_vals, (hi - lo) / max(abs(mu), 1.0e-30))
        push!(err_vals, 0.5 * (hi - lo))
    end
    names = map(last, columns)
    drift = NamedTuple{names}(Tuple(drift_vals))
    err = NamedTuple{names}(Tuple(err_vals))
    saturated = all(d -> d < rtol, drift_vals)
    return (saturated=saturated, t_from=rows[1][1], t_to=rows[end][1],
            drift=drift, err=err, n=n)
end
