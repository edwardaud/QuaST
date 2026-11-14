using QuaST, Printf

# Steady benchmarks on this solver. One entry point per case, so a case
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
    ic_file::String          # ASPECT's ascii initial condition, "" for the analytic roll
    periodic::Bool           # true for a limit cycle such as Tosi case 5a
end

# a steady case does not pass the flag
Case(name, Lx, cx, cy, pr, amp, tstop, ic) = Case(name, Lx, cx, cy, pr, amp, tstop, ic, false)

"""Read an ASPECT `ascii data` temperature file: `x y T` rows, `#` comments."""
function read_ic(path::String)
    xs = Float64[]; ys = Float64[]; ts = Float64[]
    for line in eachline(path)
        t = strip(line)
        (isempty(t) || startswith(t, "#")) && continue
        parts = split(t)
        length(parts) < 3 && continue
        push!(xs, parse(Float64, parts[1]))
        push!(ys, parse(Float64, parts[2]))
        push!(ts, parse(Float64, parts[3]))
    end
    ux = sort(unique(round.(xs, digits = 9)))
    uy = sort(unique(round.(ys, digits = 9)))
    T = fill(NaN, length(ux), length(uy))
    for k in eachindex(xs)
        i = searchsortedfirst(ux, round(xs[k], digits = 9))
        j = searchsortedfirst(uy, round(ys[k], digits = 9))
        T[i, j] = ts[k]
    end
    return ux, uy, T
end

"""Bilinear value of an ascii field at (x, y); used to start from ASPECT's state."""
function ic_value(ux, uy, T, x, y)
    i = clamp(searchsortedlast(ux, x), 1, length(ux) - 1)
    j = clamp(searchsortedlast(uy, y), 1, length(uy) - 1)
    tx = clamp((x - ux[i]) / (ux[i + 1] - ux[i]), 0.0, 1.0)
    ty = clamp((y - uy[j]) / (uy[j + 1] - uy[j]), 0.0, 1.0)
    return (1 - tx) * (1 - ty) * T[i, j] + tx * (1 - ty) * T[i + 1, j] +
           (1 - tx) * ty * T[i, j + 1] + tx * ty * T[i + 1, j + 1]
end

const DATADIR = normpath(joinpath(@__DIR__, "data"))

function make_case(name::String)
    ic(name) = joinpath(DATADIR, "initial_temperature_case$(name).txt")
    if name == "1a"
        # Ra = 1e4, no heating, constant viscosity
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 1.0, ic("1a"))
    elseif name == "1b"
        # Ra = 1e5, no heating, constant viscosity
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e5, H=0.0, eta_law=:expT, deta_T=1.0,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 1.0, ic("1b"))
    elseif name == "1c"
        # Ra = 1e6, no heating, constant viscosity
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e6, H=0.0, eta_law=:expT, deta_T=1.0,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 1.0, ic("1c"))
    elseif name == "2a"
        # Ra = 1e4, no heating, eta = exp(-ln(1000) T)
        return Case(name, 1.0, 4, 4,
                    Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.1, 1.0, ic("2a"))
    elseif name == "2b"
        # Ra = 1e4, joint temperature and depth dependence, 2.5 x 1 box
        B = 9.70406052783923433184
        C = 4.1588830833596718565
        return Case(name, 2.5, 5, 2,
                    Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=B, eta_c=C,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.01, 1.0, ic("2b"))
    elseif name == "tosi1"
        # ASPECT Tosi_Case1.prm: surface Ra = 100, eta_T = 1e5, no plastic branch,
        # eta_z = 1, eta in [1e-6, 10], initial guess 0.1, refinement 5 (32 x 32).
        # ASPECT makes its law nondimensional with eta0 = 0.1 (its "Initial
        # viscosity"), so the same choice is available here through CASE_ETA0.
        # ASPECT's Ra = 100 is defined with eta0 = 0.1, while this solver takes Ra
        # with a reference viscosity of one.  The two describe the same problem
        # only when Ra / (eta_abs / eta0) matches, i.e. Ra_ours = 10 / eta0:
        # (eta0 = 1, Ra = 10) and (eta0 = 0.1, Ra = 100) are the consistent pairs.
        eta0 = parse(Float64, get(ENV, "CASE_ETA0", "0.1"))
        law = TosiLaw(etaT=1.0e5, etaZ=1.0, eta_star=0.0, sigma_y=0.0,
                      eta_init=1.0e-1, eta0=eta0,
                      eta_min=1.0e-6, eta_max=10.0, z_is_depth=true)
        return Case(name, 1.0, 4, 4,
                    Params(Ra=10.0 / eta0, H=0.0, eta_law=:tosi, tosi=law,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.01, 1.0, "")
    elseif name == "tosi5a"
        # ASPECT Tosi_Case5.prm: eta_z = 10, yield stress 4, eta* = 1e-3,
        # refinement 6 (64 x 64).  This case is a limit cycle, not a steady state.
        eta0 = parse(Float64, get(ENV, "CASE_ETA0", "0.1"))
        law = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                      eta_init=1.0e-1, eta0=eta0,
                      eta_min=1.0e-6, eta_max=10.0, z_is_depth=true)
        return Case(name, 1.0, 4, 4,
                    Params(Ra=10.0 / eta0, H=0.0, eta_law=:tosi, tosi=law,
                           Twall_top=0.0, Twall_bot=1.0, bc=:free), 0.01, 1.0, "", true)
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
    # Start from ASPECT's own initial condition when it ships with the case:
    # those fields are already close to the steady state, which is why the
    # reference runs need hundreds of steps rather than tens of thousands.  The
    # stored field is T; the solver wants the anomaly about the conductive base.
    if !isempty(c.ic_file) && isfile(c.ic_file)
        ux, uy, Tic = read_ic(c.ic_file)
        th0 = [ic_value(ux, uy, Tic, x, y) - (1.0 - y) for (x, y) in ss.su.coords]
    else
        th0 = [roll(x, y, c.Lx, c.amp) for (x, y) in ss.su.coords]
    end
    for (g, (_, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[g] = 0.0)
    end
    t0 = time()
    # Nu_top and the mean temperature against time; a limit-cycle case is
    # described by these two series, not by a single converged number.
    series = Tuple{Float64,Float64,Float64}[]
    every = c.periodic ? 1 : 25
    cb = (st) -> begin
        push!(series, (st.t, semquad_nu_walls(ss.su, st.qt, st.theta, c.pr)[1],
                       semquad_mean_T(ss.su, st.qt, st.theta, c.pr)))
        st.step % every == 0 || return nothing
        nt, _ = semquad_nu_walls(ss.su, st.qt, st.theta, c.pr)
        @printf("  %s step %5d t=%.3e dt=%.2e Nu_top=%.5f Vrms=%.5f leaves=%d\n",
                c.name, st.step, st.t, st.dt, nt,
                semquad_vrms(ss.su, st.qt, st.ux, st.uz), length(st.qt.leaves))
        flush(stdout)
        return nothing
    end
    # CFL-controlled time step: a fixed dt0 would need tens of thousands of steps
    # to reach the steady state of a Ra = 1e4 case.
    # A steady benchmark does not depend on the time step, so a larger CFL only
    # shortens the transient.  ASPECT runs these cases with CFL = 5.
    cfl = parse(Float64, get(ENV, "CASE_CFL", "0.5"))
    # ASPECT solves Tosi with up to 30 nonlinear iterations per step to 1e-5;
    # one Picard iteration per step is too lagged for a 1e5 viscosity contrast.
    maxpic = parse(Int, get(ENV, "CASE_MAXPIC", "1"))
    st = run_semquad_simulation(qt, c.pr, th0; p=2, tstop=tstop, dt0=1.0e-4,
                                cfl=cfl, dtmax=1.0e-2, dt_growth=1.05,
                                maxpic=maxpic, pic_tol=1.0e-5,
                                maxsteps=200_000, steady_tol=1.0e-6,
                                steady_window=20, tmin=0.05, callback=cb,
                                linear=:auto, max_level=level,
                                # Wall refinement: the production rule forces
                                # every leaf within `strip` of a wall, and the
                                # wall rows, back to `wall_min`. Nu is a
                                # one-sided gradient inside that layer, so this
                                # is where the error lives (docs/dev/AMR.md).
                                amr_every=parse(Int, get(ENV, "CASE_AMR_EVERY", "0")),
                                initial_refine=parse(Int, get(ENV, "CASE_INITREF", "0")),
                                wall_min_level=parse(Int, get(ENV, "CASE_WALLMIN", "0")),
                                wall_strip=parse(Float64, get(ENV, "CASE_WALLSTRIP", "0.1")),
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
    if c.periodic && length(series) > 40
        report_cycle(c.name, series)
    end
    return st
end

"""Period and min/max pair of a limit cycle, from the last two maxima of Nu(t)."""
function report_cycle(name::String, series)
    nu = [p[2] for p in series]
    tt = [p[1] for p in series]
    # local maxima, ignoring the first third of the record (still transient)
    first = max(1, length(nu) ÷ 3)
    peaks = Int[]
    for i in (first + 1):(length(nu) - 1)
        nu[i] > nu[i - 1] && nu[i] >= nu[i + 1] && push!(peaks, i)
    end
    if length(peaks) < 2
        println("[", name, "] no limit cycle detected in ", length(series), " samples")
        return
    end
    i1, i2 = peaks[end - 1], peaks[end]
    period = tt[i2] - tt[i1]
    lo, hi = minmax(nu[i1:i2]...)
    tlo = minimum(series[i1:i2][:, 3]); thi = maximum(series[i1:i2][:, 3])
    @printf("[%s] limit cycle: period=%.4f  Nu range=%.4f to %.4f  mean T range=%.4f to %.4f\n",
            name, period, lo, hi, tlo, thi)
    @printf("[%s] reference (Tosi Table 2): period=0.0792  Nu_top 2.6950 to 7.2792  mean T 0.6543 to 0.6722\n",
            name)
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
