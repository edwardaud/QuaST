using MantleSEM, Printf

# Blankenbach (1989) 2b, **time-dependent**, on the square-cell grid (5x2 base).
#
# 2b is run with the transient driver only (no steady solver): the viscosity
# contrast is reached by *continuation in time* -- a chain of time-dependent runs
# with b, c scaled by s = 0.25, 0.5, 0.75, 1.0, each starting from the previous
# temperature field.  This keeps the same time-stepping machinery throughout and
# avoids the violent first transient of the full contrast
# (ROUTE_C_STATUS 4.14: the analytic IC with eta_min = 6e-5 gives u ~ 4e3 at t=0).
#
#   eta = exp(-b T + c (1 - z)), b = ln(16384), c = ln(64)
#   IC  = (1 - z) + 0.01 cos(pi x / 2.5) sin(pi z)
#   reference: Nu = 6.9299, Vrms = 171.755
const B = 9.70406052783923433184
const C = 4.1588830833596718565
const REF_NU = 6.9299
const REF_VR = 171.755
const LX = 2.5

function chain(level::Int; stages=(0.25, 0.5, 0.75, 1.0), tstage=0.05,
               ev=0.11, ev_cmax=0.052, dt0=1.0e-4, cfl=0.5)
    qt = Quadtree(level; cells_x=5, cells_y=2)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    theta = [0.01 * cos(pi * x / LX) * sin(pi * y) for (x, y) in ss.su.coords]
    @printf("# 2b time-dependent, level %d, cells=%d, nodes=%d\n", level,
            length(qt.leaves), ss.su.N)
    for s in stages
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=s * B, eta_c=s * C,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
        cb = (st) -> begin
            st.step % 500 == 0 || return nothing
            @printf("  s=%.2f step %6d t=%.3e dt=%.2e Vrms=%.2f\n", s, st.step,
                    st.t, st.dt, semquad_vrms(ss.su, st.qt, st.ux, st.uz))
            flush(stdout)
            return nothing
        end
        st = try
            # Benchmark reference path: pin the bitwise direct solver, not the default.
            run_semquad_simulation(qt, pr, theta; p=2, tstop=tstage, dt0=dt0,
                                   linear=:direct,
                                   cfl=cfl, maxsteps=200000, verbose=false,
                                   ev=ev, ev_cmax=ev_cmax, callback=cb,
                                   steady_tol=1.0e-5, steady_window=20,
                                   tmin=0.01)
        catch err
            @printf("  s=%.2f FAILED: %s\n", s,
                    first(split(sprint(showerror, err), '\n')))
            return
        end
        theta = st.theta
        nt, nb = semquad_nu_walls(ss.su, st.qt, theta, pr)
        vr = semquad_vrms(ss.su, st.qt, st.ux, st.uz)
        @printf("STAGE s=%.2f  t=%.3f steps=%d  Nu_top=%.4f (%+.2f%%)  Vrms=%.2f (%+.2f%%)  umax=%.1f\n",
                s, st.t, st.steps, nt, 100 * (nt / REF_NU - 1), vr,
                100 * (vr / REF_VR - 1), maximum(hypot.(st.ux, st.uz)))
        flush(stdout)
        s > 0 && nt > 50 && (println("  (diverged, stopping)"); break)
    end
end

main() = begin
    lev = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
    tstage = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.05
    chain(lev; tstage=tstage)
end
main()
