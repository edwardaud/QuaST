# The `linear = :auto` safety net.
#
# If the multigrid solve it selected is unacceptable on its own gate, the public
# entry point redoes it with the explicit direct path -- same arguments, same
# state, so the answer is the direct answer -- and counts the fallback.  The
# counter is reported by the driver and pinned by the solver gates.

# (same arguments, same state, so the answer is the direct answer) and count it.
function _solve_semquad_stokes_var_fallback(ss::SemStokes, qt::Quadtree,
                                            eta_fun::Function, fx::Function,
                                            fz::Function, uDx::Function,
                                            uDz::Function, core, ch, why::String)
    prev = LAST_AUTO_FALLBACK[]
    LAST_AUTO_FALLBACK[] = (count=prev.count + 1, last_reason=why)
    @warn "linear=:auto resolved to :mg but that solve was not acceptable; " *
          "redoing it with linear=:direct ($why).  Explicit linear=:mg is not " *
          "affected; LAST_AUTO_FALLBACK[] counts these solves." maxlog=1
    direct = (; (k => v for (k, v) in pairs(core) if k !== :linear)...,
                 linear=:direct)
    res = _solve_semquad_stokes_var_core(ss, qt, eta_fun, fx, fz, uDx, uDz;
                                         direct...)
    # Report the selector's decision; the counter records which path answered.
    LAST_SEMSTOKES_AUTO_CHOICE[] = ch
    return res
end
