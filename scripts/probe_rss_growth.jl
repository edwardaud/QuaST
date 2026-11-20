#!/usr/bin/env julia
# Does RSS grow across steps, and is it retained or lagging garbage?
#
#   julia --project=. --threads=8 scripts/probe_rss_growth.jl [nleaf] [steps]
#
# The one-step probe (scripts/probe_fallback_memory.jl) shows 9.2 GB at 163 840
# leaves with no multigrid fallback, so the 25 GB crash of the lev-9 run must come
# from something that builds up over steps.  This walks the same state for many
# steps, prints RSS every step, and calls `GC.gc()` on every fifth step: RSS that
# falls back after a collection is garbage, RSS that stays is retained.

using QuaST, Printf

rss_mb() = begin
    for l in split(read("/proc/self/status", String), '\n')
        startswith(l, "VmRSS:") && return parse(Int, split(l)[2]) / 1024.0
    end
    0.0
end

function main()
    nleaf = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 40_000
    steps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50

    qt = Quadtree(9; Lx=2.5, cells_x=5, cells_y=2)
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

    cb = s -> begin
        g = s.step % 5 == 0
        g && GC.gc()
        @printf("step %3d leaves %6d cache %5d rss %8.0f MB%s\n", s.step,
                length(s.qt.leaves), length(QuaST.SEMQUAD_MG_CACHE),
                rss_mb(), g ? "  (after gc)" : "")
        flush(stdout)
    end

    @printf("start: leaves=%d rss=%.0f MB\n", length(qt.leaves), rss_mb())
    res = run_semquad_simulation(qt, pr, theta0; p=2, maxsteps=steps,
                                 dt0=1.0e-6, dtmax=5.0e-5, dtmin=1.0e-9,
                                 amr_every=5, refine_frac=0.2, coarsen_frac=0.05,
                                 linear=:mg, stokes_tol=1.0e-3, mg_tol=1.0e-8,
                                 wall_min_level=5, wall_strip=0.1,
                                 max_level=9, max_leaves=nleaf * 2,
                                 ev=0.11, ev_cmax=0.052, verbose=false,
                                 callback=cb)
    GC.gc()
    @printf("end: steps=%d leaves=%d rss=%.0f MB (after gc)\n",
            res.steps, length(res.qt.leaves), rss_mb())
end

main()
