using QuaST, Printf

# Blankenbach 2b on route C with **wall refinement** instead of uniform h-refinement:
# base level 4 (80x32 cells) plus a band of strip = 0.1 near both walls refined to
# level 6, i.e. the thermal boundary layers see 320x128 while the interior stays
# coarse.  This is the point of the quadtree: Nu is a wall-gradient diagnostic.
#
# Usage: julia --project=. examples/blankenbach_2b_wall.jl [base] [wall] [tstop]
const B = 9.70406052783923433184
const C = 4.1588830833596718565
const REF_NU = 6.9299
const REF_VR = 171.755
const LX = 2.5

function main()
    base = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
    wall = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
    tstop = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.08

    qt = Quadtree(max(base, wall); cells_x=5, cells_y=2)
    for _ in 1:base, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    n0 = length(qt.leaves)
    n1 = refine_wall_to_min!(qt, wall; strip=0.1)
    ss = make_semstokes(qt, 2)
    @printf("# 2b wall-refined: base %d -> +%d leaves (%d), nodes=%d\n",
            base, n1, length(qt.leaves), ss.su.N)
    flush(stdout)

    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=B, eta_c=C,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th0 = [0.01 * cos(pi * x / LX) * sin(pi * y) for (x, y) in ss.su.coords]
    cb = (st) -> begin
        st.step % 100 == 0 || return nothing
        @printf("  step %5d t=%.4e dt=%.2e Nu_top=%.4f Vrms=%.2f umax=%.1f\n",
                st.step, st.t, st.dt,
                semquad_nu_walls(ss.su, st.qt, st.theta, pr)[1],
                semquad_vrms(ss.su, st.qt, st.ux, st.uz),
                maximum(hypot.(st.ux, st.uz)))
        flush(stdout)
        return nothing
    end
    # Benchmark reference path: pin the bitwise direct solver, not the default.
    st = run_semquad_simulation(qt, pr, th0; p=2, tstop=tstop, dt0=1.0e-4,
                                linear=:direct,
                                cfl=0.5, maxsteps=200000, verbose=false,
                                ev=0.11, ev_cmax=0.052, callback=cb,
                                checkpoint_path=joinpath(@__DIR__, "..", "..",
                                                         "out_julia", "bb2b_wall_ck.h5"),
                                checkpoint_every=250)
    ss2 = make_semstokes(st.qt, 2)
    nt, nb = semquad_nu_walls(ss2.su, st.qt, st.theta, pr)
    vr = semquad_vrms(ss2.su, st.qt, st.ux, st.uz)
    @printf("FINAL t=%.4f steps=%d  Nu_top=%.4f (%+.2f%%)  Nu_bot=%.4f  Vrms=%.2f (%+.2f%%)\n",
            st.t, st.steps, nt, 100 * (nt / REF_NU - 1), nb, vr,
            100 * (vr / REF_VR - 1))
end
main()
