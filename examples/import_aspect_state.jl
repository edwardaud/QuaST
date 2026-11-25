#!/usr/bin/env julia
# =============================================================================
# import_aspect_state.jl — build a QuaST checkpoint from an ASPECT state.
#
#   julia --project=. examples/import_aspect_state.jl \
#       --state  /tmp/mspec/asp_cont_f17.h5 \
#       --template out_julia/case_realistic_E/spinup.h5 \
#       --out    out_julia/case_realistic_E_aspimport/seed/from_aspect_f17.h5
#
# `--state` is written by `aspect_runs/realistic/extract_aspect_state.py`, which
# reconstructs the *exact* Q2 field of one ASPECT VTU frame (its nine nodal
# values per cell) instead of the bilinear patches the VTU literally contains.
#
# What this script does:
#
#  1. converts the frame with the section-2 scale set
#         t_nd = t/(d^2/kappa),  T_nd = T/1000 - 0.5,  u_nd = u d/kappa,
#     and builds the perturbation `theta = T_nd - Tbar(y)`,
#         Tbar(y) = 1 - y + (H/2) y (1-y)          (H = 10, the card),
#     by evaluating ASPECT's Q2 polynomial at every QuaST velocity node
#     (point location by descending the logical quadtree, then the
#     tensor-product Q2 Lagrange basis);
#  2. builds the production-card tree from that field: the card's 5x2 base at
#     level 3, then repeated `amr_cycle_balanced!` (0.2 / 0.05, theta
#     gradient-jump indicator) + `refine_wall_to_min!` (5 / 0.1) +
#     `mirror_close!`/`balance!(symmetry=:x)` + `semquad_enforce_leaf_budget!`
#     until the leaf budget is reached — i.e. the card's own AMR rule, driven by
#     the imported field, not a hand-picked mesh;
#  3. re-interpolates onto the final tree (one direct ASPECT->tree evaluation,
#     so no transfer error accumulates), zeroes theta on the walls (it is
#     identically zero there: ASPECT's Dirichlet nodes are exactly 1500/500 K),
#     and writes a checkpoint with `save_semquad_state`, taking the `run_config`
#     from `--template` (an existing checkpoint of the same card) so the restart
#     guard accepts the resume;
#  4. prints the diagnostics of the imported state — `Nu_top`, `Nu_bot`, `Vrms`,
#     `Tmax`, `Tmean` — which are the interpolation check: they must reproduce
#     ASPECT's own numbers for that frame (Vrms/Tmax/Tmean to the definition,
#     Nu to the wall-functional family floor).
#
# Pressure is not imported: it is not an initial condition of the Boussinesq
# system (the first Stokes solve recomputes it), so `pv` is written as zeros.
# `theta_prev` is set equal to `theta` and `step = 0`, so the first step is the
# driver's backward-Euler first step: the import has no time history, and a
# faked BDF2 history would be a larger error than the O(dt) first-step truncation
# it replaces (measured in §16.4: the first-step Nu_bot change is 5.6e-4, against
# the 1.1e-2 separation the experiment measures).
#
# `--probe-level L` instead builds a uniform level-L tree over the 5x2 base and
# reports the same diagnostics without writing anything: it measures how much of
# the imported-state diagnostic is tree resolution (the interpolation error of
# step 1) rather than ASPECT's field.
# =============================================================================

using QuaST, Printf
import HDF5

# ---- the card (mirrors examples/case_realistic_2d.jl) ------------------
const LX, CX, CY = 2.5, 5, 2
const PORD       = 2
const RA         = parse(Float64, get(ENV, "IMPORT_RA", "1.0e5"))
const HHEAT      = 10.0
const MAXLEV     = parse(Int, get(ENV, "IMPORT_MAXLEV", "7"))
const COMP_AD    = 1.6240183119e5
const COMP_AS    = 2.6796419238e-1
const COMP_ED    = 12.0
const COMP_ES    = 17.0
const COMP_N     = 3.0
const ETA_T0     = 0.5
const COMP_EPS_MIN = 1.0e-12
const ETA_MIN    = 1.0e-4
const ETA_MAX    = 1.0e4
const SLOPE410   = 0.032
const SLOPE660   = -0.032
const WIDTH_PT   = 0.025
const B410       = 180.0 / (3300.0 * 3.0e-5 * 1000.0)
const B660       = 300.0 / (3300.0 * 3.0e-5 * 1000.0)
const LATENT_L   = 0.11
const LATENT_SCALE = (rho_ref=3300.0, cp=1000.0, dT=1000.0, g=9.81, d=2.9e6,
                      T_surface=0.5)
# scale set
const D_SCALE = 2.9e6            # d, m
const KAPPA   = 1.0e-6           # m^2/s
const T_SCALE = 1000.0           # dT, K
const T_OFF   = 0.5              # T_nd = T/T_SCALE - T_OFF

function make_params()
    return Params(Ra=RA, H=HHEAT,
                  eta_law=:composite, comp_rule=:isostress,
                  comp_Ad=COMP_AD, comp_As=COMP_AS,
                  comp_Ed=COMP_ED, comp_Es=COMP_ES,
                  comp_Vd=0.0, comp_Vs=0.0, comp_n=COMP_N,
                  comp_Tcut=0.0, comp_eps_min=COMP_EPS_MIN,
                  eta_T0=ETA_T0, eta_min=ETA_MIN, eta_max=ETA_MAX,
                  bc=:free, latent_L=LATENT_L, latent_mode=:per_phase,
                  yield_c=Inf, yield_phi=0.0, yield_taumax=Inf,
                  yield_c_grad=0.0, yield_mu_grad=0.0,
                  weak_f=1.0, weak_eps_ref=0.0,
                  rho_heating=false, rho_cp=false, Di=0.0,
                  heating_adiabatic=false, heating_dissipation=false,
                  refstate=nothing, deviatoric=false)
end

function make_phases()
    ph = [PhaseTransition(0.87, WIDTH_PT, SLOPE410, 0.5, 180.0, B410),
          PhaseTransition(0.78, WIDTH_PT, SLOPE660, 0.5, 300.0, B660)]
    return per_phase_latent(Params(Ra=RA, H=HHEAT), ph; LATENT_SCALE...)
end

# ---- the ASPECT Q2 field ---------------------------------------------------
struct AspField
    idx::Dict{NTuple{3,Int},Int}     # (level, ix, iy) -> cell index
    lmin::Int
    lmax::Int
    x0::Vector{Float64}
    y0::Vector{Float64}
    hx::Vector{Float64}
    hy::Vector{Float64}
    T::Matrix{Float64}               # (ncells, 9), node m = 3b + a
    ux::Matrix{Float64}
    uz::Matrix{Float64}
    t::Float64
end

function read_asp(path::AbstractString)
    HDF5.h5open(path, "r") do f
        g = f["cells"]
        lev = Int.(read(g["level"]))
        ix = Int.(read(g["ix"]))
        iy = Int.(read(g["iy"]))
        idx = Dict{NTuple{3,Int},Int}()
        for k in eachindex(lev)
            idx[(lev[k], ix[k], iy[k])] = k
        end
        nc = length(lev)
        # h5py writes C-ordered (ncells, 9); HDF5.jl reads the same bytes as
        # column-major, so the fast axis may arrive first.  Accept both.
        read_mat(name) = begin
            A = read(g[name])
            size(A) == (nc, 9) && return Matrix{Float64}(A)
            size(A) == (9, nc) && return Matrix{Float64}(permutedims(A))
            error("$name is $(size(A)), expected ($nc, 9) or (9, $nc)")
        end
        # Convert to the solver's nondimensional units here, once, so everything
        # downstream (point location, Q2 evaluation, diagnostics) is in the
        # units of QuaST's box [0, 2.5] x [0, 1]:
        #   geometry  m       -> nd        (divide by d)
        #   T         K       -> T_nd      (T/dT - 0.5: 1500 K -> 1, 500 K -> 0)
        #   velocity  m/s     -> u_nd      (u d / kappa)
        x0 = Float64.(read(g["x0"])) ./ D_SCALE
        y0 = Float64.(read(g["y0"])) ./ D_SCALE
        hx = Float64.(read(g["hx"])) ./ D_SCALE
        hy = Float64.(read(g["hy"])) ./ D_SCALE
        Tnd = read_mat("T") ./ T_SCALE .- T_OFF
        uscale = D_SCALE / KAPPA
        ux = read_mat("ux") .* uscale
        uz = read_mat("uz") .* uscale
        return AspField(idx, minimum(lev), maximum(lev), x0, y0, hx, hy,
                        Tnd, ux, uz, Float64(HDF5.attrs(f)["time"]))
    end
end

"""Index of the ASPECT Q2 cell containing `(x, y)` (nondimensional; 0 if none)."""
@inline function locate(a::AspField, x::Float64, y::Float64)
    for l in a.lmin:a.lmax
        q = 1 << l
        jx = clamp(floor(Int, x * q / LX), 0, q - 1)
        jy = clamp(floor(Int, y * q), 0, q - 1)
        c = get(a.idx, (l, jx, jy), 0)
        c != 0 && return c
    end
    return 0
end

@inline function _lag(t::Float64, i::Int)     # nodes at -1, 0, 1
    i == 0 && return 0.5 * t * (t - 1.0)
    i == 1 && return 1.0 - t * t
    return 0.5 * t * (t + 1.0)
end

"""Evaluate T_nd, u_nd of the ASPECT field at `(x, y)` (nondimensional)."""
@inline function eval_asp(a::AspField, x::Float64, y::Float64)
    c = locate(a, x, y)
    c == 0 && error("point ($x, $y) is not inside the ASPECT mesh")
    xi = 2.0 * (x - a.x0[c]) / a.hx[c] - 1.0
    et = 2.0 * (y - a.y0[c]) / a.hy[c] - 1.0
    sT = 0.0
    sx = 0.0
    sy = 0.0
    @inbounds for b in 0:2
        mb = _lag(et, b)
        for aa in 0:2
            w = _lag(xi, aa) * mb
            m = 3 * b + aa + 1
            sT += w * a.T[c, m]
            sx += w * a.ux[c, m]
            sy += w * a.uz[c, m]
        end
    end
    return sT, sx, sy
end

# ---- QuaST tree --------------------------------------------------------
"""
Fill `theta` on `sq` from the ASPECT frame, and (optionally) the velocity.

`theta = T_nd - Tbar(y)` with `T_nd` and the velocity already nondimensional
(`read_asp`); the top/bottom wall nodes are set to exactly 0 (ASPECT's Dirichlet
values make them 0 to round-off, but the 1e-12 driver guard would otherwise see
1e-16..1e-9 residuals).
"""
function interp_state!(theta::Vector{Float64}, ux::Union{Nothing,Vector{Float64}},
                       uz::Union{Nothing,Vector{Float64}}, sq::SemQuad,
                       a::AspField)
    nwall = 0
    for g in eachindex(sq.coords)
        x, y = sq.coords[g]
        Tnd, vx, vz = eval_asp(a, x, y)
        theta[g] = Tnd - Tbar(y, HHEAT)
        if ux !== nothing
            ux[g] = vx
            uz[g] = vz
        end
        if y <= 1.0e-14 || y >= 1.0 - 1.0e-14
            abs(theta[g]) > 1.0e-8 && (nwall += 1)
            theta[g] = 0.0
            uz !== nothing && (uz[g] = 0.0)
        end
    end
    return nwall
end

function make_initial_tree()
    qt = Quadtree(MAXLEV; Lx=LX, cells_x=CX, cells_y=CY)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

"""Leaves of the wall strip (the set `semquad_enforce_leaf_budget!` protects)."""
function wall_protect(qt::Quadtree, strip::Float64)
    p = Set{NTuple{3,Int}}()
    for lf in qt.leaves
        h = QuaST.leaf_hy(qt, lf[1])
        zc = (lf[3] + 0.5) * h
        if lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1 ||
           zc <= strip || zc >= 1.0 - strip
            push!(p, lf)
        end
    end
    return p
end

"""
    build_tree(a; max_leaves, wall_min, wall_strip, refine_frac, coarsen_frac,
                 coarsen_tol, max_passes, verbose)

The production card's AMR rule, driven by the imported field, from the card's
initial tree to the leaf budget.  Returns `(qt, theta, ux, uz)` — the last
interpolation is the one on the final tree, so nothing is transferred twice.
"""
function build_tree(a::AspField; max_leaves::Int=10000, wall_min::Int=5,
                    wall_strip::Float64=0.1, refine_frac::Float64=0.2,
                    coarsen_frac::Float64=0.05, coarsen_tol::Float64=1.0e-2,
                    max_passes::Int=40, amr_max_level::Int=MAXLEV,
                    symmetry::Symbol=:none, verbose::Bool=true)
    qt = make_initial_tree()
    nl = length(qt.leaves)
    hist = Int[]
    for pass in 1:max_passes
        ss = make_semstokes(qt, PORD)
        sq = ss.su
        theta = zeros(Float64, sq.N)
        interp_state!(theta, nothing, nothing, sq, a)
        eta = semquad_gradient_jump(sq, qt, theta)
        amr_cycle_balanced!(qt, eta; geom=CartesianBox(qt), refine_frac=refine_frac,
                            coarsen_frac=coarsen_frac,
                            max_level=amr_max_level,
                            max_leaves=max_leaves, symmetry=symmetry)
        refine_wall_to_min!(qt, wall_min; geom=CartesianBox(qt), strip=wall_strip, symmetry=symmetry)
        symmetry === :x && mirror_close!(qt; max_level=amr_max_level)
        balance!(qt; symmetry=symmetry)
        eta_cap = copy(eta)
        for lf in qt.leaves
            get!(eta_cap, lf, 0.0)
        end
        merged, ok = semquad_enforce_leaf_budget!(qt, eta_cap;
                        max_leaves=max_leaves,
                        protect=wall_protect(qt, wall_strip),
                        symmetry=symmetry)
        push!(hist, length(qt.leaves))
        verbose && @printf("  [import] AMR pass %2d: %d -> %d leaves (merged %d, budget=%s)\n",
                           pass, nl, length(qt.leaves), merged, ok)
        nl = length(qt.leaves)
        if nl >= max_leaves || (pass > 3 && hist[end] == hist[end-1])
            break
        end
    end
    ss = make_semstokes(qt, PORD)
    sq = ss.su
    theta = zeros(Float64, sq.N)
    ux = zeros(Float64, sq.N)
    uz = zeros(Float64, sq.N)
    nwall = interp_state!(theta, ux, uz, sq, a)
    return qt, theta, ux, uz, nwall
end

"""Level census string, e.g. `L3=112 L4=684 ...`."""
function level_census(qt::Quadtree)
    d = Dict{Int,Int}()
    for lf in qt.leaves
        d[lf[1]] = get(d, lf[1], 0) + 1
    end
    return join(["L$(l)=$(d[l])" for l in sort(collect(keys(d)))], " ")
end

function diagnostics(qt::Quadtree, theta::Vector{Float64}, ux, uz, pr)
    ss = make_semstokes(qt, PORD)
    sq = ss.su
    Nut, Nub = semquad_nu_walls(sq, qt, theta, pr)
    Vr = semquad_vrms(sq, qt, ux, uz)
    Tm = semquad_mean_T(sq, qt, theta, pr)
    Tmax = maximum(Tbar(y, pr.H) + theta[g] for (g, (x, y)) in enumerate(sq.coords))
    return (Nu_top=Nut, Nu_bot=Nub, Vrms=Vr, Tmean=Tm, Tmax=Tmax)
end

"""Rebuild the quadtree from a `frames/frame_NNNNN_mesh.csv` of a QuaST run.

The file is `level,x0,y0,h` per leaf (`write_frame!` in the driver).  Using the
tree the production card actually had at the same physical time removes the mesh
as a variable from the comparison; `--amr-max-level` and the AMR build are then
skipped.
"""
function tree_from_mesh(path::AbstractString; max_leaves::Int=10000)
    leaves = NTuple{3,Int}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        p = split(line, ",")
        length(p) == 4 || continue
        l = parse(Int, p[1])
        x0 = parse(Float64, p[2]); y0 = parse(Float64, p[3]); h = parse(Float64, p[4])
        hx = leaf_hx_probe(l)                    # 2.5/(5*2^l)
        abs(h - hx) <= 1e-9 * hx ||
            error("leaf height $h does not match level $l (expected $hx)")
        push!(leaves, (l, round(Int, x0 / h), round(Int, y0 / h)))
    end
    isempty(leaves) && error("no leaves parsed from $path")
    length(leaves) <= max_leaves ||
        error("$(length(leaves)) leaves > max_leaves=$max_leaves")
    qt = Quadtree(MAXLEV; Lx=LX, cells_x=CX, cells_y=CY)
    for (l, ix, iy) in leaves
        node = (0, ix >> l, iy >> l)
        for lev in 1:l
            node in qt.leaves && refine!(qt, node)
            node = (lev, ix >> (l - lev), iy >> (l - lev))
        end
    end
    Set(qt.leaves) == Set(leaves) ||
        error("the mesh file is not a partition of the $(CX)x$(CY) box")
    return qt
end

@inline leaf_hx_probe(l::Int) = LX / (CX * (1 << l))

# ---- main ------------------------------------------------------------------
function parse_args(argv)
    d = Dict{String,String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("unexpected argument `$a`")
        k = a[3:end]
        if k in ("report-only", "quiet", "debug")
            d[k] = "true"
            i += 1
        else
            i + 1 <= length(argv) || error("missing value for `$a`")
            d[k] = argv[i+1]
            i += 2
        end
    end
    return d
end

function main()
    args = parse_args(ARGS)
    haskey(args, "state") || error("--state <asp_state.h5> is required")
    if haskey(args, "probe-level")
        a = read_asp(args["state"])
        L = parse(Int, args["probe-level"])
        qt = Quadtree(MAXLEV; Lx=LX, cells_x=CX, cells_y=CY)
        for _ in 1:L, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        ss = make_semstokes(qt, PORD)
        sq = ss.su
        theta = zeros(Float64, sq.N)
        ux = zeros(Float64, sq.N)
        uz = zeros(Float64, sq.N)
        interp_state!(theta, ux, uz, sq, a)
        di = diagnostics(qt, theta, ux, uz, make_params())
        @printf("[probe] uniform L%d: %d leaves, %d dofs: Nu_top=%.6f Nu_bot=%.6f Vrms=%.4f Tmax=%.6f Tmean=%.6f\n",
                L, length(qt.leaves), sq.N, di.Nu_top, di.Nu_bot, di.Vrms,
                di.Tmax, di.Tmean)
        return
    end
    for k in ("template", "out")
        haskey(args, k) || error("--$k is required")
    end
    pr = make_params()
    ph = make_phases()
    a = read_asp(args["state"])
    if haskey(args, "debug")
        for (x, y) in ((0.0, 0.0), (0.0, 0.02), (1.25, 0.5), (2.5, 1.0),
                       (1.25, 0.99), (0.5, 0.0005))
            c = locate(a, x, y)
            if c == 0
                @printf("[dbg] (%g, %g) -> outside\n", x, y)
                continue
            end
            T, vx, vz = eval_asp(a, x, y)
            @printf("[dbg] (%g, %g) -> cell %d: T_nd=%.6f (%.2f K)  ux=%.4e  uz=%.4e\n",
                    x, y, c, T, (T + T_OFF) * T_SCALE, vx, vz)
        end
    end
    t_nd = a.t / (D_SCALE^2 / KAPPA)
    @printf("[import] ASPECT frame: t = %.6e s -> t_nd = %.6e; %d Q2 cells, levels %d..%d\n",
            a.t, t_nd, length(a.idx), a.lmin, a.lmax)

    # the frame's own gradient-family diagnostics, for the interpolation check
    @printf("[import] ASPECT field itself (Q2, extracted): T in [%.2f, %.2f] K\n",
            minimum(a.T), maximum(a.T))

    max_leaves = haskey(args, "max-leaves") ? parse(Int, args["max-leaves"]) : 10000
    wall_min = haskey(args, "wall-min") ? parse(Int, args["wall-min"]) : 5
    wall_strip = haskey(args, "wall-strip") ? parse(Float64, args["wall-strip"]) : 0.1
    verbose = !haskey(args, "quiet")
    if haskey(args, "tree-from-mesh")
        qt = tree_from_mesh(args["tree-from-mesh"]; max_leaves=max_leaves)
        ss = make_semstokes(qt, PORD)
        sq = ss.su
        theta = zeros(Float64, sq.N)
        ux = zeros(Float64, sq.N)
        uz = zeros(Float64, sq.N)
        nwall = interp_state!(theta, ux, uz, sq, a)
        @printf("[import] tree from %s: %d leaves\n", args["tree-from-mesh"],
                length(qt.leaves))
    else
        qt, theta, ux, uz, nwall = build_tree(a; max_leaves=max_leaves,
            wall_min=wall_min, wall_strip=wall_strip,
            symmetry=Symbol(get(ENV, "IMPORT_SYMMETRY", "none")),
            coarsen_tol=haskey(args, "coarsen-tol") ? parse(Float64, args["coarsen-tol"]) : 1.0e-2,
            max_passes=haskey(args, "max-passes") ? parse(Int, args["max-passes"]) : 40,
            amr_max_level=haskey(args, "amr-max-level") ? parse(Int, args["amr-max-level"]) : MAXLEV,
            verbose=verbose)
    end
    @printf("[import] tree: %d leaves, %s, %d dofs; %d wall nodes had |theta| > 1e-8 before zeroing\n",
            length(qt.leaves), level_census(qt), length(theta), nwall)

    di = diagnostics(qt, theta, ux, uz, pr)
    @printf("[import] imported state: Nu_top = %.6f  Nu_bot = %.6f  Vrms = %.4f  Tmax = %.6f  Tmean = %.6f\n",
            di.Nu_top, di.Nu_bot, di.Vrms, di.Tmax, di.Tmean)
    @printf("[import]   (in ASPECT units: Tmax = %.3f K, Tmean = %.3f K, Vrms = %.4e m/s)\n",
            (di.Tmax + T_OFF) * T_SCALE, (di.Tmean + T_OFF) * T_SCALE,
            di.Vrms * KAPPA / D_SCALE)

    haskey(args, "report-only") && return
    st = load_semquad_state(args["template"])
    cfg = copy(st.config)
    dt = haskey(args, "dt") ? parse(Float64, args["dt"]) : 3.7e-7
    step = haskey(args, "step") ? parse(Int, args["step"]) : 0
    pv = zeros(Float64, make_semstokes(qt, PORD).sp.N)
    mkpath(dirname(abspath(args["out"])))
    save_semquad_state(args["out"], qt, theta, copy(theta), ux, uz, pv;
                       t=t_nd, dt=dt, dt_prev=dt, step=step, p=PORD,
                       config=cfg,
                       meta=Dict("imported_from" => abspath(args["state"]),
                                 "asp_t_s" => a.t, "t_nd" => t_nd,
                                 "asp_x0" => get(st.config, "x0", "n/a")))
    @printf("[import] wrote %s  (t_nd = %.6e, dt = %.3e, step = %d, %d leaves)\n",
            args["out"], t_nd, dt, step, length(qt.leaves))
    @printf("[import] the restart guard will compare %d run_config keys against the template\n",
            length(cfg))
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
