using QuaST, Printf

# Blankenbach 1989 case 2b, ASPECT's setup (benchmarks/blankenbach/base_case2b.prm):
#   box 2.5 x 1, Ra = 1e4, Boussinesq, free slip, T = 1 bottom / 0 top,
#   eta = exp(-b T + c (1-z)), b = 9.70406052783923433184, c = 4.1588830833596718565,
#   T0 = (1-z) + 0.01 cos(pi x / 2.5) sin(pi z).
# The box is covered by 5x2 base cells so every leaf is a SQUARE (stage A).
# ASPECT's steady reference (case2b_reference.stat): Nu_top = 6.9299, Vrms = 171.755.
#
# Usage: julia --project=. examples/blankenbach_2b.jl <level> <tstop> [ev]
const B = 9.70406052783923433184
const C = 4.1588830833596718565
const REF_NU = 6.9299
const REF_VR = 171.755
const LX = 2.5

function run2b(level::Int; tstop::Float64=0.1, ev::Float64=0.11,
               ev_cmax::Float64=0.052, dt0::Float64=1.0e-4)
    qt = Quadtree(level; cells_x=5, cells_y=2)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=B, eta_c=C,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th0 = [0.01 * cos(pi * x / LX) * sin(pi * y) for (x, y) in ss.su.coords]
    t0 = time()
    # progressive output: a killed run must still leave data behind
    cb = (st) -> begin
        st.step % 200 == 0 || return nothing
        nt, _ = semquad_nu_walls(ss.su, st.qt, st.theta, pr)
        @printf("  level %d step %5d t=%.3e dt=%.2e Nu_top=%.4f Vrms=%.2f umax=%.1f\n",
                level, st.step, st.t, st.dt, nt,
                semquad_vrms(ss.su, st.qt, st.ux, st.uz),
                maximum(hypot.(st.ux, st.uz)))
        flush(stdout)
        return nothing
    end
    st = try
        # Benchmark reference path: pin the bitwise direct solver, not the default.
        run_semquad_simulation(qt, pr, th0; p=2, tstop=tstop, dt0=dt0,
                               linear=:direct,
                               cfl=0.5, maxsteps=200000, verbose=false,
                               ev=ev, ev_cmax=ev_cmax,
                               steady_tol=1.0e-5, steady_window=20, tmin=0.02,
                               checkpoint_path=joinpath(@__DIR__, "..", "..", "out_julia", "bb2b_L4_ck.h5"),
                               checkpoint_every=500,
                               callback=cb)
    catch err
        @printf("level %d FAILED: %s\n", level,
                first(split(sprint(showerror, err), '\n')))
        return nothing
    end
    ss2 = make_semstokes(st.qt, 2)
    nt, nb = semquad_nu_walls(ss2.su, st.qt, st.theta, pr)
    vr = semquad_vrms(ss2.su, st.qt, st.ux, st.uz)
    umax = maximum(hypot.(st.ux, st.uz))
    @printf("level %d: cells=%d nodes=%d t=%.3f steps=%d  Nu_top=%.4f (%+.2f%%)  Vrms=%.2f (%+.2f%%)  Nu_bot=%.4f  umax=%.1f  [%.0fs]\n",
            level, length(st.qt.leaves), ss2.su.N, st.t, st.steps, nt,
            100 * (nt / REF_NU - 1), vr, 100 * (vr / REF_VR - 1), nb, umax,
            time() - t0)
    return (nt, vr)
end

main() = begin
    lev = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
    tstop = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.1
    ev = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.11
    run2b(lev; tstop=tstop, ev=ev)
end
main()
