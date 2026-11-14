using QuaST, Printf

# Steady benchmarks on the quadtree solver. One entry point per case, so a case
# that is still unrun in the documentation can be produced with one command:
#
#     julia --project=. examples/benchmark_cases.jl <case> [level] [tstop]
#
# `case` is one of 1a 1b 1c 2a 2b tosi1 tosi5a.  The script prints the wall
# Nu numbers and Vrms once the run reaches its steady criterion or `tstop`.
# Reference values are the ones the documentation quotes:
#
#   1a  Nu = 4.884409   Vrms = 42.864947    (Blankenbach et al. 1989, Table 2)
#   1b  Nu = 10.534095  Vrms = 193.21454
#   1c  Nu = 21.972465  Vrms = 833.98977
#   2a  Nu = 10.066     Vrms = 480.4334
#   2b  Nu = 6.9299     Vrms = 171.755     (Table 9)

const REF = Dict(
    "1a" => (4.884409, 42.864947),
    "1b" => (10.534095, 193.21454),
    "1c" => (21.972465, 833.98977),
    "2a" => (10.066, 480.4334),
    "2b" => (6.9299, 171.755),
)

"""Conductive profile plus one roll, the perturbation every case starts from."""
roll(x, y, Lx, a) = (1.0 - y) + a * cos(pi * x / Lx) * sin(pi * y)

struct Case
    name::String
    Lx::Float64
    cells_x::Int
    cells_y::Int
    pr::Params
    amp::Float64
    tstop::Float64
end

function make_case(name::String)
    if name == "1a"
        # Ra = 1e4, no heating, constant viscosity
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 0.2)
    elseif name == "1b"
        # Ra = 1e4, internal heating H = 10
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e4, H=10.0, eta_law=:expT, deta_T=1.0,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 0.2)
    elseif name == "1c"
        # Ra = 1e4, eta = exp(-b T) with Delta eta = 1000
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 0.2)
    elseif name == "2a"
        # Ra = 1e5, H = 10, constant viscosity
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e5, H=10.0, eta_law=:expT, deta_T=1.0,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 0.2)
    elseif name == "2b"
        # Ra = 1e4, joint temperature and depth dependence, 2.5 x 1 box
        B = 9.70406052783923433184
        C = 4.1588830833596718565
        return Case(name, 2.5, 5, 2,
                    Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=B, eta_c=C,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.01, 0.5)
    elseif name == "tosi1"
        # Tosi et al. (2015) case 1: temperature dependent only
        law = TosiLaw(etaT=1.0e5, etaZ=1.0, eta_star=0.0, sigma_y=0.0,
                      eta_init=1.0e-1, eta0=1.0, eta_min=1.0e-6, eta_max=10.0,
                      z_is_depth=true)
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e4, H=0.0, eta_law=:tosi, tosi=law,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 0.2)
    elseif name == "tosi5a"
        # Tosi et al. (2015) case 5a: Ra = 1e2, the full viscoplastic law
        law = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                      eta_init=1.0e-1, eta0=1.0e-1, eta_min=1.0e-6, eta_max=10.0,
                      z_is_depth=true)
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e2, H=0.0, eta_law=:tosi, tosi=law,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 0.2)
    end
    error("unknown case $name; use 1a, 1b, 1c, 2a, 2b, tosi1 or tosi5a")
end

function run_case(c::Case; level::Int=5, tstop::Float64=c.tstop, out::String="")
    qt = Quadtree(level; cells_x=c.cells_x, cells_y=c.cells_y, Lx=c.Lx)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    th0 = [roll(x, y, c.Lx, c.amp) for (x, y) in ss.su.coords]
    # pin the walls to the boundary values the roll already satisfies
    for (g, (_, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[g] = 0.0)
    end
    t0 = time()
    cb = (st) -> begin
        st.step % 200 == 0 || return nothing
        nt, _ = semquad_nu_walls(ss.su, st.qt, st.theta, c.pr)
        @printf("  %s step %5d t=%.3e dt=%.2e Nu_top=%.5f Vrms=%.5f leaves=%d\n",
                c.name, st.step, st.t, st.dt, nt,
                semquad_vrms(ss.su, st.qt, st.ux, st.uz), length(st.qt.leaves))
        flush(stdout)
        return nothing
    end
    # CFL-controlled time step: a fixed dt0 would need tens of thousands of steps
    # to reach the steady state of a Ra = 1e4 case.
    st = run_semquad_simulation(qt, c.pr, th0; p=2, tstop=tstop, dt0=1.0e-4,
                                cfl=0.5, dtmax=1.0e-2, dt_growth=1.05,
                                maxsteps=200_000, steady_tol=1.0e-6,
                                steady_window=20, tmin=0.05, callback=cb,
                                linear=:auto, amr_every=0, max_level=level,
                                checkpoint_path=isempty(out) ? "" : joinpath(out, "$(c.name).h5"),
                                checkpoint_every=500, verbose=true)
    nt, nb = semquad_nu_walls(ss.su, st.qt, st.theta, c.pr)
    vr = semquad_vrms(ss.su, st.qt, st.ux, st.uz)
    wall = time() - t0
    @printf("\n[%s] steps=%d t=%.4e wall=%.0f s  Nu_top=%.6f Nu_bot=%.6f Vrms=%.6f  leaves=%d\n",
            c.name, st.steps, st.t, wall, nt, nb, vr, length(st.qt.leaves))
    if haskey(REF, c.name)
        rn, rv = REF[c.name]
        @printf("[%s] reference Nu=%.6f (%.2f%%)  Vrms=%.6f (%.2f%%)\n",
                c.name, rn, 100 * (nt - rn) / rn, rv, 100 * (vr - rv) / rv)
    end
    return st
end

function main(args)
    isempty(args) && error("usage: benchmark_cases.jl <case> [level] [tstop]")
    name = args[1]
    level = length(args) >= 2 ? parse(Int, args[2]) : 5
    c = make_case(name)
    tstop = length(args) >= 3 ? parse(Float64, args[3]) : c.tstop
    run_case(c; level=level, tstop=tstop)
end

main(ARGS)
