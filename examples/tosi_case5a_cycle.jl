# Tosi et al. (2015) case 5a: the periodic benchmark, integrated with the
# viscosity converged *inside* each step.
#
# Why this script exists.  The time-stepping driver in `src/driver/run.jl` solves
# one Stokes problem per step with a lagged viscosity (`maxpic = 1`), and it
# refuses to iterate inside a step for the Tosi law.  That lag is harmless when
# the flow is already moving, but case 5a starts from a 1 % perturbation: the
# first Stokes solve sees the uniform `eta_init` and produces a strain rate small
# enough that the perturbation decays before the viscosity can soften.  Run that
# way, the case decays to pure conduction and never reaches its limit cycle.
#
# ASPECT avoids this with 30 nonlinear iterations per step.  This script does the
# same thing without touching the solver: `semquad_tosi_inner` iterates
# Stokes <-> eta(eps) at frozen temperature, then one backward-Euler energy step
# advances the temperature with that velocity.  It is still backward Euler in
# time, so the trajectory is a legitimate integration, not a pseudo-time path.
#
#     julia --project=. --threads=4 examples/tosi_case5a_cycle.jl [level] [tstop]
#
# Reference values, Tosi et al. (2015) Table 2 (median of eleven codes):
#   period 0.0792, Nu_top 2.6950 to 7.2792, mean T 0.6543 to 0.6722.

using QuaST, Printf

const REF_PERIOD = 0.0792
const REF_NU = (2.6950, 7.2792)
const REF_T = (0.6543, 0.6722)

"""
    run_case5a(level; tstop=1.0, cfl=0.5, dt0=1e-4, dtmax=1e-2, tol=1e-3, maxit=20)

Integrate case 5a and return the time series `(t, Nu_top, mean T)`.
"""
function run_case5a(level::Int; tstop::Float64=1.0, cfl::Float64=0.5,
                    dt0::Float64=1.0e-4, dtmax::Float64=1.0e-2, dtmin::Float64=1.0e-9,
                    tol::Float64=1.0e-3, maxit::Int=20, advection::Symbol=:galerkin)
    qt = Quadtree(level; cells_x=4, cells_y=4, Lx=1.0)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    law = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                  eta_init=1.0e-1, eta0=1.0e-1, eta_min=1.0e-6, eta_max=10.0,
                  z_is_depth=true)
    # The paper's Ra = 100 is defined with the reference viscosity, so the value
    # this solver needs depends on how the law is made nondimensional.  CASE_RA
    # lets a short run check that question without touching the library.
    ra = parse(Float64, get(ENV, "CASE_RA", "100.0"))
    pr = Params(Ra=ra, H=0.0, eta_law=:tosi, tosi=law,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    theta = [0.01 * cospi(x) * sinpi(y) for (x, y) in ss.su.coords]
    for (g, (_, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end

    series = Tuple{Float64,Float64,Float64}[]
    t = 0.0
    dt = dt0
    u = nothing
    step = 0
    hmin = minimum(QuaST.leaf_hy(qt, lf[1]) for lf in qt.leaves)
    while t < tstop
        inner = semquad_tosi_inner(ss.su, qt, ss, pr, theta;
                                   tol=tol, maxit=maxit, u_init=u, linear=:direct)
        u = inner.u
        umax = maximum(hypot(u[i], u[sq_n(ss) + i]) for i in 1:sq_n(ss))
        # CFL step, grown and cut the way the driver does it
        target = umax > 0 ? cfl * hmin / umax : dtmax
        dt = clamp(min(max(target, 0.95 * dt), 1.05 * dt), dtmin, dtmax)
        t + dt > tstop && (dt = tstop - t)
        parts = QuaST.semquad_steady_parts(ss.su, qt, pr, theta, inner.u;
                                           dt=dt, advection=advection)
        ur = parts.Ar \ (parts.Cfree' * (parts.rhs - parts.A * parts.u0))
        theta = parts.u0 + parts.Cfree * ur
        t += dt
        step += 1
        nt, _ = semquad_nu_walls(ss.su, qt, theta, pr)
        push!(series, (t, nt, semquad_mean_T(ss.su, qt, theta, pr)))
        if step % parse(Int, get(ENV, "CASE_PRINT", "25")) == 0
            @printf("  step %5d t=%.4e dt=%.2e Nu_top=%.4f Vrms=%.3f eta=[%.1e,%.1e] inner_it=%d%s\n",
                    step, t, dt, nt,
                    semquad_vrms(ss.su, qt, u[1:sq_n(ss)], u[(sq_n(ss) + 1):(2sq_n(ss))]),
                    minimum(inner.eta), maximum(inner.eta),
                    inner.iters, inner.converged ? "" : " (inner stalled)")
            flush(stdout)
        end
    end
    return series
end

sq_n(ss) = length(ss.su.coords)

"""Period and min/max pair from the last two maxima of Nu(t)."""
function report_cycle(series)
    nu = [s[2] for s in series]
    tt = [s[1] for s in series]
    first = max(1, length(nu) ÷ 3)
    peaks = [i for i in (first + 1):(length(nu) - 1) if nu[i] > nu[i - 1] && nu[i] >= nu[i + 1]]
    if length(peaks) < 2
        @printf("no limit cycle: %d samples, Nu range %.4f to %.4f, Vrms never settled\n",
                length(series), minimum(nu), maximum(nu))
        return
    end
    i1, i2 = peaks[end - 1], peaks[end]
    period = tt[i2] - tt[i1]
    nlo, nhi = minimum(nu[i1:i2]), maximum(nu[i1:i2])
    tlo = minimum(s[3] for s in series[i1:i2])
    thi = maximum(s[3] for s in series[i1:i2])
    @printf("\n[tosi5a] period=%.4f   Nu_top=%.4f to %.4f   mean T=%.4f to %.4f\n",
            period, nlo, nhi, tlo, thi)
    @printf("[tosi5a] reference: period=%.4f   Nu_top=%.4f to %.4f   mean T=%.4f to %.4f\n",
            REF_PERIOD, REF_NU[1], REF_NU[2], REF_T[1], REF_T[2])
    @printf("[tosi5a] differences: period %+.1f%%, Nu +%.1f%%/%.1f%%, T +%.1f%%/%.1f%%\n",
            100 * (period - REF_PERIOD) / REF_PERIOD,
            100 * (nlo - REF_NU[1]) / REF_NU[1], 100 * (nhi - REF_NU[2]) / REF_NU[2],
            100 * (tlo - REF_T[1]) / REF_T[1], 100 * (thi - REF_T[2]) / REF_T[2])
end

function main(args)
    level = length(args) >= 1 ? parse(Int, args[1]) : 3
    tstop = length(args) >= 2 ? parse(Float64, args[2]) : 1.0
    @printf("[tosi5a] level %d, tstop %.3f, in-step eta(eps) convergence\n", level, tstop)
    series = run_case5a(level; tstop=tstop)
    report_cycle(series)
end

main(ARGS)
