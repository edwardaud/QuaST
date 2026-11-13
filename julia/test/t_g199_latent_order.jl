# G199 -- time accuracy of the latent-heat coupling (docs/RHEOLOGY.md §12).
#
# The apparent-heat-capacity latent term is evaluated either at the current
# iterate of the step (`latent_scheme=:implicit`) or at the start-of-step
# temperature (`:lagged`).  With the default `maxpic = 1` there is no iterate, so
# the step is a single lagged sweep.  This gate pins the measured convergence
# orders on the 1-D conductive column of docs/RHEOLOGY.md §7, where the velocity
# is exactly zero and the only coupling under test is the latent capacity:
#
#   (1) `latent_scheme=:implicit` with the iterated step (the G193 fixed
#       iteration) is SECOND order in dt: the theta error against a dt/8
#       reference has an observed order >= 1.8.
#   (2) the lagged step (the default at maxpic=1) is FIRST order in the latent
#       coupling: the difference between the lagged and the implicit solution at
#       the same dt has an observed order of 1 (0.85 .. 1.15).
#   (3) with a phase transition sharp enough that the lag error dominates the
#       second-order BDF2 error, the lagged scheme's own error is first order
#       too (0.85 .. 1.3); with the §7 transition the lag error is subdominant
#       and the lagged total error stays second order (measured 2.08) -- that
#       case is asserted to be second order for the implicit scheme only.
#   (4) the enthalpy budget (exact enthalpy change against the consistent
#       boundary reaction of the step operator) closes at O(dt) for BOTH
#       schemes: the apparent-capacity form is not the conservative variable,
#       so a second-order solution does not give a second-order budget.
#   (5) the code default at maxpic=1 (`latent_scheme` left at its default
#       `:implicit`) is bitwise the explicit `:lagged` scheme.
#
# Measured values (2026-11-08, 64-leaf column, tstop=1e-2, reference dt=6.25e-5):
#   §7 transition:   order lagged 2.08, implicit 2.05, lag-imp diff 1.00
#   sharp transition: order lagged 1.06, implicit 2.07, lag-imp diff 1.01
#   budget order:    1.00 / 1.00 for both transitions and both schemes
using Test, MantleSEM, LinearAlgebra

const G199_PT_A = PhaseTransition(0.6, 0.1, 0.05, 0.5, 0.1, 1.0)  # §7 transition
const G199_PT_B = PhaseTransition(0.6, 0.05, 0.2, 0.5, 0.1, 1.0)  # sharp in T

function g199_tree(nref::Int=3)
    qt = Quadtree(4)
    for _ in 1:nref, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    return qt
end

function g199_order(dts, errs)
    x = log.(collect(Float64, dts)); y = log.(collect(Float64, errs))
    m = length(x); xm = sum(x) / m; ym = sum(y) / m
    return sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
end

# Per-step enthalpy budget: the exact enthalpy change against the consistent
# boundary reaction of the step operator, normalised by the total magnitude of
# the Dirichlet reactions (`scale`).
function g199_step_budget(sq, qt, pr, phases, th_new, th_n, th_nm1, ux, uz,
                          dt, dt_prev, implicit::Bool)
    parts = solve_semquad_energy(sq, qt, pr, th_n, th_nm1, ux, uz, dt, dt_prev,
                                 false; advection=:galerkin, phases=phases,
                                 latent_theta=(implicit ? th_new : nothing),
                                 return_parts=true)
    r = parts.A * th_new - parts.rhs
    dir = findall(==(0.0), diag(parts.Cfree * parts.Cfree'))
    Qc = sum(r[dir])
    b0 = semquad_heat_budget(sq, qt, pr, th_n; phases=phases, ux=ux, uz=uz)
    b1 = semquad_heat_budget(sq, qt, pr, th_new; phases=phases, ux=ux, uz=uz)
    dE = b1.enthalpy - b0.enthalpy
    return abs(dE / dt - Qc) / max(sum(abs, r[dir]), 1.0e-300)
end

function g199_column(; nsteps, dt, lat, pt, amp, scheme=:implicit, maxpic=1,
                     pic_tol=1.0e-9, nref=3, budget::Bool=false)
    qt = g199_tree(nref)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    th0 = [amp * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end
    pr = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, latent_L=lat)
    rec = Any[]
    cb = st -> push!(rec, (dt=st.dt, theta=copy(st.theta), ux=copy(st.ux),
                           uz=copy(st.uz)))
    r = run_semquad_simulation(qt, pr, th0; p=2, tstop=nsteps * dt, dt0=dt,
                               dtmax=dt, dtmin=1.0e-18, maxsteps=nsteps,
                               amr_every=0, ev=0.0, advection=:galerkin,
                               linear=:direct, maxpic=maxpic, pic_tol=pic_tol,
                               phases=[pt], latent_scheme=scheme, callback=cb,
                               verbose=false)
    worst = 0.0
    if budget
        implicit = (scheme === :implicit && maxpic > 1)
        for k in 2:length(rec)
            d = g199_step_budget(sq, qt, pr, [pt], rec[k].theta,
                                 rec[k - 1].theta,
                                 k >= 3 ? rec[k - 2].theta : rec[k - 1].theta,
                                 rec[k].ux, rec[k].uz, rec[k].dt,
                                 k >= 3 ? rec[k - 1].dt : rec[k].dt, implicit)
            worst = max(worst, d)
        end
    end
    return (theta=r.theta, worst=worst, steps=r.steps)
end

# One convergence study: returns (dts, err_lagged, err_implicit, diff, budget...)
function g199_scan(pt, lat, amp; tstop=1.0e-2, ref_dt=6.25e-5,
                   dts=(1.0e-3, 5.0e-4, 2.5e-4))
    nref = round(Int, tstop / ref_dt)
    ref = g199_column(nsteps=nref, dt=ref_dt, lat=lat, pt=pt, amp=amp,
                      scheme=:implicit, maxpic=80, pic_tol=1.0e-11)
    out = Any[]
    for dt in dts
        n = round(Int, tstop / dt)
        lg = g199_column(nsteps=n, dt=dt, lat=lat, pt=pt, amp=amp,
                         scheme=:lagged, maxpic=1, budget=true)
        im = g199_column(nsteps=n, dt=dt, lat=lat, pt=pt, amp=amp,
                         scheme=:implicit, maxpic=80, pic_tol=1.0e-9,
                         budget=true)
        @test lg.steps == n && im.steps == n
        push!(out, (dt=dt, n=n, el=norm(lg.theta - ref.theta) / norm(ref.theta),
                    ei=norm(im.theta - ref.theta) / norm(ref.theta),
                    dl=norm(lg.theta - im.theta) / norm(im.theta),
                    wl=lg.worst, wi=im.worst))
    end
    return out
end

@testset "G199 latent-heat time order" begin
    # ---- (1)(2) the §7 transition: implicit 2nd order, latent lag 1st order --
    let
        s = g199_scan(G199_PT_A, 2.5, 0.1)
        d = [x.dt for x in s]
        o_imp = g199_order(d, [x.ei for x in s])
        o_lag = g199_order(d, [x.el for x in s])
        o_dif = g199_order(d, [x.dl for x in s])
        # measured: 2.05 / 2.08 / 1.00
        @test o_imp >= 1.8
        @test o_imp <= 2.4
        @test 0.85 <= o_dif <= 1.15          # the latent coupling is first order
        @test o_lag >= 1.8                   # lag error subdominant here
        # both schemes converge to the reference; at the finest dt the lagged
        # error is *smaller* (the first-order lag partially cancels the BDF2
        # error) -- that cancellation is why case B below exists
        @test s[end].ei < 2.0e-5
        @test s[end].el < 2.0e-5
        # the difference is a first-order signal: it halves per dt halving
        @test s[1].dl / s[end].dl > 3.0
        @test s[1].dl / s[end].dl < 5.0
        # budget closure: O(dt) for both schemes (measured 1.00/1.00)
        o_bud_l = g199_order(d, [x.wl for x in s])
        o_bud_i = g199_order(d, [x.wi for x in s])
        @test 0.8 <= o_bud_l <= 1.2
        @test 0.8 <= o_bud_i <= 1.2
        @test maximum(x.wl for x in s) < 2.0e-2
        @test maximum(x.wi for x in s) < 2.0e-2
    end

    # ---- (3) sharp transition: the lagged error itself is first order --------
    let
        s = g199_scan(G199_PT_B, 0.45, 0.3)
        d = [x.dt for x in s]
        o_imp = g199_order(d, [x.ei for x in s])
        o_lag = g199_order(d, [x.el for x in s])
        o_dif = g199_order(d, [x.dl for x in s])
        # measured: 2.07 / 1.06 / 1.01
        @test o_imp >= 1.8
        @test 0.85 <= o_lag <= 1.3           # lagged total error is first order
        @test 0.85 <= o_dif <= 1.15
        @test s[end].el > 5.0 * s[end].ei    # measured ~10x
        @test s[end].ei < 2.0e-5
        # the lagged error halves per dt halving (first order)
        @test 3.2 < s[1].el / s[end].el < 5.2
    end

    # ---- (5) the default at maxpic=1 is the lagged scheme, bitwise ----------
    let
        dt = 1.0e-3
        n = 10
        kw = (; p=2, tstop=n * dt, dt0=dt, dtmax=dt, dtmin=1.0e-18, maxsteps=n,
              amr_every=0, ev=0.0, advection=:galerkin, linear=:direct,
              maxpic=1, pic_tol=1.0e-9, phases=[G199_PT_A], verbose=false)
        qt = g199_tree(3)
        ss = make_semstokes(qt, 2)
        pr = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, latent_L=2.5)
        th0 = [0.1 * sinpi(y) for (x, y) in ss.su.coords]
        for (i, (x, y)) in enumerate(ss.su.coords)
            (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
        end
        a = run_semquad_simulation(qt, pr, th0; kw...)
        b = run_semquad_simulation(qt, pr, th0; kw..., latent_scheme=:lagged)
        c = run_semquad_simulation(qt, pr, th0; kw..., latent_scheme=:implicit)
        @test a.theta == b.theta             # default is bitwise :lagged
        @test a.theta == c.theta             # and bitwise :implicit at maxpic=1
    end

    # ---- (4) a Newton step is the same implicit fixed point -----------------
    let
        # the per-step Newton and the iterated Picard step solve the same
        # implicit equation: measured |dtheta|/|theta| ~ 1e-9 on this column
        dt = 1.0e-3
        n = 10
        imp = g199_column(nsteps=n, dt=dt, lat=0.45, pt=G199_PT_B, amp=0.3,
                          scheme=:implicit, maxpic=80, pic_tol=1.0e-10)
        qt = g199_tree(3)
        ss = make_semstokes(qt, 2)
        pr = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, latent_L=0.45)
        th0 = [0.3 * sinpi(y) for (x, y) in ss.su.coords]
        for (i, (x, y)) in enumerate(ss.su.coords)
            (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
        end
        nw = run_semquad_simulation(qt, pr, th0; p=2, tstop=n * dt, dt0=dt,
                                    dtmax=dt, dtmin=1.0e-18, maxsteps=n,
                                    amr_every=0, ev=0.0, advection=:galerkin,
                                    linear=:direct, maxpic=20, pic_tol=1.0e-9,
                                    phases=[G199_PT_B], pic_mode=:newton,
                                    newton_maxit=12, verbose=false)
        @test norm(nw.theta - imp.theta) / norm(imp.theta) < 1.0e-6
    end
end
