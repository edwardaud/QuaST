# High-Ra / high-viscosity-contrast / internal-heating runner.
#
# Case definition: EXTREME_CASE.md.  Ra is the ASPECT nondimensional parameter
# referenced to T = 0; the hot-boundary local Rayleigh number is Ra * Delta eta.
# The literal target is Ra = 1e8, Delta eta = 1e3 (local 1e11), with optional
# constant internal heating H (EXTREME_CASE.md uses H = 5).
#
# Usage: julia --project=julia julia/scripts/highra_run.jl [nxe] [Ra] [deta] [tstop] [zbeta] [tag] [cfl] [cfl_basis] [ev_cR] [ev_beta] [nze] [p] [linear] [init_h5] [cfl_mode] [dt0] [H] [advection] [maxpic] [save_dt]
using QuaST
using Printf
using HDF5

function parse_args(args)
    n = length(args) >= 1 ? parse(Int, args[1]) : 32
    Ra = length(args) >= 2 ? parse(Float64, args[2]) : 1.0e5
    deta = length(args) >= 3 ? parse(Float64, args[3]) : 1.0e3
    tstop = length(args) >= 4 ? parse(Float64, args[4]) : 1.0e-2
    zbeta = length(args) >= 5 ? parse(Float64, args[5]) : 3.0
    tag = length(args) >= 6 ? args[6] : "n$(n)"
    cfl = length(args) >= 7 ? parse(Float64, args[7]) : 2.0
    cfl_basis = length(args) >= 8 ? Symbol(args[8]) : :node
    ev_cR = length(args) >= 9 ? parse(Float64, args[9]) : 0.0
    ev_beta = length(args) >= 10 ? parse(Float64, args[10]) : 0.0
    nz = length(args) >= 11 ? parse(Int, args[11]) : n
    p = length(args) >= 12 ? parse(Int, args[12]) : 4
    linear = length(args) >= 13 ? Symbol(args[13]) : :direct
    init = length(args) >= 14 ? args[14] : ""
    cfl_mode = length(args) >= 15 ? Symbol(args[15]) : :global
    dt0arg = length(args) >= 16 ? parse(Float64, args[16]) : 1.0e-9
    H = length(args) >= 17 ? parse(Float64, args[17]) : 0.0
    adv = length(args) >= 18 ? Symbol(args[18]) : :skew
    maxpic = length(args) >= 19 ? parse(Int, args[19]) : 1
    save_dt = length(args) >= 20 ? parse(Float64, args[20]) : 0.0

    adv in (:skew, :galerkin, :conservative) ||
        throw(ArgumentError("advection must be skew, galerkin or conservative, got $adv"))
    maxpic >= 1 || throw(ArgumentError("maxpic must be >= 1"))
    H >= 0.0 || throw(ArgumentError("H must be >= 0"))
    save_dt >= 0.0 || throw(ArgumentError("save_dt must be >= 0"))
    return (; n, Ra, deta, tstop, zbeta, tag, cfl, cfl_basis, ev_cR, ev_beta,
            nz, p, linear, init, cfl_mode, dt0arg, H, adv, maxpic, save_dt)
end

function main()
    a = parse_args(ARGS)
    m = make_mesh(1.0, a.n, a.nz, a.p; zgrading=:tanh, zbeta=a.zbeta)
    fd = make_fedata(a.p, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    theta0 = if a.init != "" && isfile(a.init)
        th = Float64[]
        h5open(a.init, "r") do f
            th = Vector{Float64}(read(f["theta"]))
        end
        length(th) == m.Nscalar || error("restart theta has $(length(th)) entries, mesh needs $(m.Nscalar)")
        @printf("# restart from %s (%d scalars)\n", a.init, length(th))
        th
    else
        [0.01 * cospi(xs[mod(k-1,m.nvx)+1]) * sinpi(zs[div(k-1,m.nvx)+1])
         for k in 1:m.Nscalar]
    end
    pr = Params(Ra=a.Ra, H=a.H, eta_law=:expT, deta_T=a.deta, bc=:free, supg=0.0)
    mkpath("out_julia")
    @printf("# highra %s nxe=%d nze=%d p=%d hzmin=%.3e Ra=%.1e Ra_int=%.1e deta=%.1e zbeta=%.1f tstop=%.1e cfl=%.1f basis=%s ev=%.3f/%.3f H=%.2f adv=%s maxpic=%d save_dt=%.1e\n",
            a.tag, a.n, a.nz, a.p, z_min_size(m), a.Ra, a.Ra * a.deta, a.deta,
            a.zbeta, a.tstop, a.cfl, a.cfl_basis, a.ev_cR, a.ev_beta, a.H,
            a.adv, a.maxpic, a.save_dt)
    t0 = time()
    res = run_simulation(m, fd, pr; theta0=theta0, tstop=a.tstop, dt0=a.dt0arg,
                         dtmax=1e-3, cfl=a.cfl, cfl_basis=a.cfl_basis, maxsteps=200000,
                         ev=a.ev_cR, ev_cmax=a.ev_beta, ev_alpha=2, maxpic=a.maxpic,
                         pic_tol=1e-9, steady_tol=0.0, report_every=50, linear=a.linear,
                         cfl_mode=a.cfl_mode, advection=a.adv,
                         save_dt=a.save_dt,
                         save_path="out_julia/highra_$(a.tag)_frames.h5",
                         verbose=true, log_path="out_julia/highra_$(a.tag).log",
                         log_every=10)
    wall = time() - t0
    # history CSV: t Nu_top Nu_bot Vrms (layout fixed by run_simulation)
    open("out_julia/highra_$(a.tag)_hist.csv", "w") do io
        println(io, "t,Nu_top,Nu_bot,Vrms,Tavg,gamma,Phi")
        for h in res.history
            length(h) >= 13 || continue
            @printf(io, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                    h[1], h[3], h[4], h[5], h[11], h[12], h[13])
        end
    end
    h5open("out_julia/highra_$(a.tag).h5", "w") do f
        f["theta"] = res.theta
        f["ux"] = res.ux
        f["uz"] = res.uz
        attributes(f)["nxe"] = a.n
        attributes(f)["nze"] = a.nz
        attributes(f)["p"] = a.p
        attributes(f)["Ra"] = a.Ra
        attributes(f)["deta"] = a.deta
        attributes(f)["H"] = a.H
        attributes(f)["zbeta"] = a.zbeta
        attributes(f)["dt0"] = a.dt0arg
        attributes(f)["cfl"] = a.cfl
        attributes(f)["cfl_basis"] = String(a.cfl_basis)
        attributes(f)["cfl_mode"] = String(a.cfl_mode)
        attributes(f)["linear"] = String(a.linear)
        attributes(f)["advection"] = String(a.adv)
        attributes(f)["maxpic"] = a.maxpic
        attributes(f)["ev_cR"] = a.ev_cR
        attributes(f)["ev_beta"] = a.ev_beta
        attributes(f)["t"] = res.t
        attributes(f)["Nu_top"] = res.Nu_top
        attributes(f)["Nu_bot"] = res.Nu_bot
        attributes(f)["Vrms"] = res.Vrms
    end
    @printf("DONE %s steps=%d t=%.4e wall=%.0fs Vrms=%.4f Nu_top=%.4f Nu_bot=%.4f H=%.2f\n",
            a.tag, res.steps, res.t, wall, res.Vrms, res.Nu_top, res.Nu_bot, a.H)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
