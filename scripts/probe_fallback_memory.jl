#!/usr/bin/env julia
# Does the multigrid fall back to a direct factorisation at high leaf counts, and
# is that where the memory goes?
#
#   julia --project=. --threads=8 scripts/probe_fallback_memory.jl [nleaf] [steps]
#
# The hierarchy at 163 840 leaves costs 653 MB (measure_mg_memory.jl), yet the lev-9
# run died at 25 GB.  The auto-fallback path in `linear=:mg` is the remaining
# candidate: a UMFPACK factorisation of a million-dof block needs tens of GB.  This
# prints the fallback counter, the refine-iteration counter and the RSS delta for a
# short run, so the attribution is measured rather than assumed.

using QuaST, Printf

rss_mb() = begin
    for l in split(read("/proc/self/status", String), '\n')
        startswith(l, "VmRSS:") && return parse(Int, split(l)[2]) / 1024.0
    end
    0.0
end

function main()
    nleaf = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 80_000
    steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
    maxlev = 9

    qt = Quadtree(maxlev; Lx=2.5, cells_x=5, cells_y=2)
    while length(qt.leaves) < nleaf
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
    end
    ss = make_semstokes(qt, 2)
    sq = ss.su
    theta0 = [0.05 * sinpi(z) * cospi(2.0 * x / 2.5) +
              0.02 * sinpi(2.0 * z) * cospi(4.0 * x / 2.5) for (x, z) in sq.coords]
    pr = Params(Ra=1.0e5, H=10.0, eta_law=:composite,
                comp_Ad=1.6240183119e5, comp_As=2.6796419238e-1,
                comp_Ed=12.0, comp_Es=17.0, comp_n=3.0, eta_T0=0.5,
                eta_min=1.0e-4, eta_max=1.0e4, bc=:free)

    @printf("leaves=%d dofs=%d rss_before=%.0f MB\n", length(qt.leaves), 2 * sq.N,
            rss_mb())
    fb0 = LAST_AUTO_FALLBACK[].count
    t0 = time()
    res = run_semquad_simulation(qt, pr, theta0; p=2, maxsteps=steps,
                                 dt0=1.0e-6, dtmax=5.0e-5, dtmin=1.0e-9,
                                 amr_every=0, linear=:mg, stokes_tol=1.0e-3,
                                 mg_tol=1.0e-8, wall_min_level=5, wall_strip=0.1,
                                 max_level=maxlev, max_leaves=nleaf + 10_000,
                                 ev=0.11, ev_cmax=0.052, verbose=false)
    @printf("steps=%d wall=%.1f s  rss_after=%.0f MB  delta=%.0f MB\n",
            res.steps, time() - t0, rss_mb(), rss_mb() - 0.0)
    @printf("auto_fallback count=%d reason=%s\n", LAST_AUTO_FALLBACK[].count - fb0,
            LAST_AUTO_FALLBACK[].last_reason)
    @printf("direct-ladder refine iters=%d  shift=%.3e\n",
            LAST_SEMSTOKES_REFINE_ITERS[], LAST_SEMSTOKES_SHIFT[])
end

main()
