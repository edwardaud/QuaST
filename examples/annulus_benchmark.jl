using QuaST, Printf
const ANN_MS = QuaST

# The published 2-D cylindrical-annulus convection benchmark
# ==========================================================
#
# One entry point for the Davies et al. (2022) / ASPECT Ra = 1e5 cylindrical
# shell case on the closed (periodic) ring.  This is the Davies et al. (2022)
# and ASPECT documentation benchmark; the published validation record is in
# `docs/en/validation.md`:
#
#     julia --project=. examples/annulus_benchmark.jl <nx>x<ny> [tstop]
#
# e.g. `256x3072`, or the two published resolutions `128x1536` (ASPECT-like)
# and `512x2048` (Firedrake-like).  `<nx>x<ny>` is radial x angular cells; both
# must be composable from the quadtree's `(cells_x=1, cells_y=ny/nx)` base
# tessellation, i.e. `nx` must be a power of two and `ny` a multiple of `nx`.
#
# The card (all from ANNULUS_REFS):
#
#   * isoviscous Boussinesq, Ra = rho g alpha dT d^3 / (eta_ref kappa) = 1e5,
#     d = R1 - R0 = 1 (the gap is the length unit), eta = 1, k = cp = rho = 1,
#     no internal heating, radial inward constant-magnitude gravity (the mapped
#     `grad h = +e_r` buoyancy `Ra theta e_r`), T = 1 at r = R0, T = 0 at r = R1;
#   * free-slip on both circles, closed (periodic) ring in the angular
#     direction, solid-body-rotation gauge (V_RMS is only comparable after the
#     net rotation is removed, ANNULUS_REFS §10.7) and the nodal pressure pin;
#   * the paper's initial condition, Eq. (41) of Davies et al. (2022):
#     T = (R1 - r) + 0.02 cos(4 atan2(y,x)) sin(pi (r - R0))  (`annulus_ic =
#     :firedrake`); `ANN_IC=aspect` selects ASPECT's `spherical hexagonal
#     perturbation` instead for the record.  The transient is not a published
#     target; the steady state is.
#   * the paper's discretisation is Q2-Q1 velocity-pressure and Q2 temperature
#     on a uniform mesh; this solver is Q2-Q1-Q2 on the quadtree, advection
#     `:galerkin` (the paper's element pair, no stabilisation) by default, and
#     no entropy viscosity (`ANN_EV`, default 0).  The benchmark's targets are
#     steady scalars, so the transport stabilisation is a stated variant, not a
#     hidden knob; `ANN_ADVECTION=skew` provides the conservative variant.
#
# Steady-state criterion (stated, not guessed): after `t > ANN_TMIN`, the run
# stops once, for `ANN_WINDOW` consecutive steps,
#
#     |dtheta|/|theta| < ANN_TOL,  |dNu_T| < ANN_TOL*max(1,|Nu_T|),
#     |dV_RMS| < ANN_TOL*max(1,|V_RMS|)
#
# (implemented by `semquad_steady_step` in `src/driver/run.jl`; this card sets
# `steady_vrms=true`).  The default `ANN_TOL=1e-6` is far below the 4-6
# significant digits the published values carry; `ANN_WINDOW=20`.  A run that
# reaches `tstop` without firing is reported with `steady=0` and its last-window
# drifts, never as converged.
#
# Two time-stepping modes reach that criterion:
#
#   * **relaxation (the mode the published-resolution runs use).**  With
#     `ANN_CFL=0` and a large fixed `ANN_DT0` (e.g. `1e3`), the BDF2 time
#     derivative is negligible in every step, so each energy solve is the
#     *steady* advection-diffusion equation for the current velocity and the
#     sequence of steps is the classical Stokes <-> energy Picard iteration for
#     the coupled steady Boussinesq problem.  The iteration solves the slow
#     thermal relaxation (which the published codes follow to `t = 2..3`)
#     directly instead of resolving it in time; it typically settles in tens of
#     iterations at any resolution.  The fixed point is the same discrete steady
#     state for every dt (the time derivative vanishes identically there), and
#     the record verifies it against a time-accurate march at a coarse level.
#   * **time-accurate**, `ANN_CFL>0`: the physical trajectory; at the published
#     resolution it needs thousands of CFL-limited steps to `t ~ 1`, which is
#     the honest reason the relaxation mode is used for the finest meshes.
#
# Diagnostics: Nu_T (outer/top wall), Nu_B (inner/bottom wall) from the mapped
# one-sided GLL radial derivative with the *logarithmic* conduction
# normalisation `Nu_wall = R_wall |ln f| <dT/dr>/dT` (ANNULUS_REFS §6.2; the
# Firedrake multipliers 1.3290170684486309 / 0.7303607313096079 are
# `R1|ln f|` / `R0|ln f|` for this geometry), V_RMS = sqrt(int|u|^2 dA / A),
# <T> = area mean of the full T = Tbar(r) + theta, and the `Nu_T - Nu_B` split,
# which is zero in the continuum at steady state with no heating and is the
# resolution diagnostic the benchmark publishes (0.38% for ASPECT at
# 1536x128).
#
# Published values this script compares against (printed, not gated):
#
#                                       Nu_T      Nu_B     V_RMS    <T>
#   Davies et al. (2022) Table A1       9.541             193.26
#   ASPECT docs, 1536x128               9.548     9.584    193.371  0.382
#
# The tolerances ANNULUS_REFS §11 motivates are 0.5%/0.2% (Nu/V_RMS) at
# ASPECT-like resolution and 0.1% at Firedrake-like resolution, the 0.07%
# spread between those two reference codes.
#
# ENV knobs: ANN_TSTOP (3.0; use a huge value in relaxation mode), ANN_MAXSTEPS
# (200000), ANN_DT0 (1e-5; the relaxation mode passes e.g. 1e3), ANN_DTMAX
# (1e-2), ANN_CFL (1.0; 0 = fixed ANN_DT0), ANN_TOL (1e-6), ANN_WINDOW (20),
# ANN_TMIN (0.02), ANN_MAXPIC (1), ANN_PIC_RELAX (1.0), ANN_PIC_TOL (1e-6),
# ANN_ADVECTION (galerkin; `skew` is the conservative variant), ANN_EV (0.0),
# ANN_IC (firedrake; `aspect` is the ASPECT initial condition), ANN_REPORT (25),
# ANN_CKPT_DIR (""), ANN_CKPT_EVERY (200), ANN_LINEAR (auto), ANN_MG_THREADS
# (8), ANN_TAG (""), ANN_RA (1.0e5; the published card is Ra = 1e5, the
# ASPECT/Davies case 2.1 cross-code comparison runs the same card at Ra = 1e4).
#
# Continuation knobs (G291; a physics-aware restart of a checkpoint):
#
#   ANN_RESTART          checkpoint file to resume from ("" = no explicit
#                        restart; an existing `ANN_CKPT_DIR/ann_<nx>x<ny>.h5` is
#                        resumed automatically as before).  A restart carries
#                        its own theta, so it is mutually exclusive with
#                        `ANN_IC` (the harness drops the IC knob on resume,
#                        exactly as it already does for the automatic resume).
#   ANN_RESTART_PHYSICS  `refuse` (default) or `allow`; forwarded to the
#                        driver's `restart_physics`.  `allow` is the documented
#                        case-2.1 (Ra = 1e4) -> case-2.2 (Ra = 1e5) protocol: it
#                        warns once with the differing physics keys and resets
#                        the first resumed step to `ANN_DT0` so the CFL ramp
#                        restarts safely (pass a small ANN_DT0, e.g. 1e-4).
#
# Both knobs only take effect on a restart; with neither set (and no existing
# checkpoint in `ANN_CKPT_DIR`) the run is the historical card byte for byte.
#
# Frame dumping (off unless ANN_FRAMES names a directory):
#
#   ANN_FRAMES      directory for the per-frame state ("" = off).  The directory
#                   is recreated: a stale frame_*.bin / frame_meta.csv from an
#                   earlier run is removed, so a renderer can never mix runs.
#   ANN_FRAME_DT    dump cadence in time units (0.05).  The first state the
#                   driver exposes is written as frame 0 (after step 1), then one
#                   frame at the first step whose t crosses every multiple of
#                   ANN_FRAME_DT, plus the final state if it crosses one more.
#   ANN_FRAME_NR    / ANN_FRAME_NPHI  (200 / 1024) the *analysis* polar grid
#                   (r x phi cell centres) recorded in the meta header; the
#                   renderer evaluates both codes on this same grid for the
#                   angular-mean anomaly theta = T - <T>_phi(r), so the two rows
#                   share one definition.  No field is evaluated on it here.
#
# Frame format (one small binary per frame, atomic temp+rename):
#
#   <ANN_FRAMES>/frame_%05d.bin
#       int64 nnodes, then five Float64 arrays of length nnodes, in this order:
#       x, y, theta, T, speed = |u|.  The node cloud is the Q2 velocity /
#       temperature space itself (unique nodes, `sq.coords`), T = Tbar(r) +
#       theta is the *total* temperature on those nodes.
#   <ANN_FRAMES>/frame_meta.csv
#       '#' header lines: format version, the uniform benchmark mesh
#       (nx, ny, R0, R1, Phi), the analysis polar grid and the array layout;
#       then a CSV table with one row per frame:
#       frame,step,t,Nu_T,Nu_B,Vrms,meanT,Tmin,Tmax,nnodes
#       where the diagnostics are the harness definitions above
#       (`semquad_nu_walls` family A, `semquad_vrms`, `semquad_mean_T`).
#
# The frame writer is a pure diagnostic: it changes nothing in the solve, and a
# run without ANN_FRAMES prints exactly the pre-frame-dump log.

const ANN_R0 = 1.22
const ANN_R1 = 2.22
const ANN_PHI = 2.0 * pi

const ANN_REF = (Firedrake_Nu=9.541, Firedrake_Vrms=193.26,
                 Aspect_NuT=9.548, Aspect_NuB=9.584,
                 Aspect_Vrms=193.371, Aspect_meanT=0.382)

env_f(k, d) = parse(Float64, get(ENV, k, string(d)))
env_i(k, d) = parse(Int, get(ENV, k, string(d)))
env_s(k, d) = get(ENV, k, string(d))

"""The benchmark's material/physics card (isoviscous Boussinesq, Ra = 1e5).

`ANN_RA` overrides the Rayleigh number (the published cards are Ra = 1e5 for
case 2.2; the same-resolution ASPECT cross-code case 2.1 runs at Ra = 1e4).
Without the environment variable this returns the historical card unchanged.
"""
function annulus_benchmark_params()
    return Params(Ra=env_f("ANN_RA", 1.0e5), H=0.0,
                  # :expT with deta_T = 1 is constant viscosity eta = 1
                  # (eta = exp(-log(deta_T) T)); the benchmark's eta_ref = 1.
                  eta_law=:expT, deta_T=1.0,
                  Twall_top=0.0, Twall_bot=1.0,
                  bc=:free, wall_bc=(:free, :free, :free, :free))
end

"""The benchmark geometry: the closed 2-D annulus (periodic ring)."""
annulus_benchmark_geometry() = PolarAnnulus(ANN_R0, ANN_R1, ANN_PHI;
                                            seam=:periodic)

"""
    annulus_benchmark_tree(nx, ny) -> Quadtree

Uniform ring tree with `nx` radial and `ny` angular cells.  `nx` must be a
power of two (radial base cells = 1) and `ny` a multiple of `nx`; the angular
base-cell count is `ny/nx`.  The published meshes are `128x1536` and
`512x2048`, i.e. `(cells_y, level) = (12, 7)` and `(4, 9)`.
"""
function annulus_benchmark_tree(nx::Int, ny::Int)
    nx >= 1 && ispow2(nx) ||
        throw(ArgumentError("annulus_benchmark_tree: nx=$nx must be a power " *
                            "of two (the quadtree's radial base is one cell)"))
    level = trailing_zeros(nx)
    ny % nx == 0 ||
        throw(ArgumentError("annulus_benchmark_tree: ny=$ny must be a " *
                            "multiple of nx=$nx (nx*2^level angular cells)"))
    cells_y = ny ÷ nx
    cells_y >= 2 ||
        throw(ArgumentError("annulus_benchmark_tree: cells_y=$cells_y must be " *
                            ">= 2 for a periodic seam"))
    qt = Quadtree(level; cells_x=1, cells_y=cells_y, seam=:periodic)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

"""
    annulus_benchmark_dofs(nx, ny; p=2) -> NamedTuple

The DOF counts of the uniform ring mesh, from the space definitions (Q2 nodal
temperature/velocity on the ring's `(p*nx+1) * (p*ny)` nodes -- the ring shares
the seam nodes, G276 (b1) -- and Q1 nodal pressure on `(nx+1) * ny` nodes).
"""
function annulus_benchmark_dofs(nx::Int, ny::Int; p::Int=2)
    n_vel = (p * nx + 1) * (p * ny)
    n_p = (nx + 1) * ny
    return (leaves=nx * ny, T=n_vel, u=2 * n_vel, p=n_p,
            saddle=2 * n_vel + n_p)
end

rss_mb(path="/proc/self/status") = begin
    s = read(path, String)
    m = match(r"VmRSS:\s+(\d+) kB", s)
    m === nothing ? NaN : parse(Float64, m.captures[1]) / 1024.0
end

peak_mb(path="/proc/self/status") = begin
    s = read(path, String)
    m = match(r"VmHWM:\s+(\d+) kB", s)
    m === nothing ? NaN : parse(Float64, m.captures[1]) / 1024.0
end

# ------------------------------------------------------------- frame writer ---
# The renderer's input: one binary per frame with the state on the solver's own
# node cloud, plus a CSV row of the harness diagnostics.  The frames are written
# from the driver callback, so they are the states the trajectory actually
# visits -- nothing is interpolated in time and no field is re-evaluated on a
# grid (the theta anomaly is computed by the renderer on a common polar grid so
# both codes share one definition; ANN_FRAME_NR/NPHI record that grid here).
mutable struct AnnulusFrameWriter
    dir::String
    meta::IO
    frame::Int
    dt::Float64
    nr::Int
    nphi::Int
    next_t::Float64
    t0::Float64
end

"""
    AnnulusFrameWriter(root, nx, ny, g, pr; dt=0.05, nr=200, nphi=1024)

Open the frame stream in `root` (recreated: stale `frame_*.bin` and
`frame_meta.csv` are removed so two runs can never be mixed).  The `'#'` header
lines carry the uniform benchmark mesh `(nx, ny, R0, R1, Phi)` and the analysis
polar grid `(nr, nphi)`, so the renderer can reconstruct the element boundaries
and use one interpolation/theta grid for both codes.
"""
function AnnulusFrameWriter(root::String, nx::Int, ny::Int, g::PolarAnnulus,
                            pr::Params; dt::Float64=0.05, nr::Int=200,
                            nphi::Int=1024)
    dt > 0.0 || throw(ArgumentError("ANN_FRAME_DT must be > 0 (got $dt)"))
    nr >= 2 && nphi >= 4 ||
        throw(ArgumentError("ANN_FRAME_NR/NPHI must be >= 2/4 (got $nr/$nphi)"))
    mkpath(root)
    for old in readdir(root)
        (occursin(r"^frame_\d{5}\.bin$", old) ||
         old == "frame_meta.csv") && rm(joinpath(root, old); force=true)
    end
    metapath = joinpath(root, "frame_meta.csv")
    io = open(metapath, "w")
    println(io, "# annulus_frames_v1")
    println(io, "# mesh nx=$(nx) ny=$(ny) R0=$(repr(g.R0)) R1=$(repr(g.R1)) ",
            "Phi=$(repr(g.Φ))")
    println(io, "# polar_grid nr=$nr nphi=$nphi")
    println(io, "# arrays x,y,theta,T,speed float64; file frame_%05d.bin = ",
            "int64 nnodes then the five arrays, each nnodes long")
    println(io, "frame,step,t,Nu_T,Nu_B,Vrms,meanT,Tmin,Tmax,nnodes")
    flush(io)
    return AnnulusFrameWriter(root, io, 0, dt, nr, nphi, 0.0, time())
end

"""
    write_annulus_frame!(fw, st, pr, g)

Write the current callback state (`st.theta`, `st.ux`, `st.uz` on `st.sq`) as
the next frame, atomically: the binary is written to a temporary name in the
same directory and renamed, then the meta row is appended and flushed.  A live
renderer therefore sees a frame only once its file and its meta row are both
complete.
"""
function write_annulus_frame!(fw::AnnulusFrameWriter, st, pr::Params,
                              g::PolarAnnulus)
    s = st.sq
    n = s.N
    # the diagnostics on exactly the state the frame shows (the harness
    # definitions: family-A wall gradient, volume RMS, area mean of T)
    nut, nub = semquad_nu_walls(s, st.qt, st.theta, pr)
    vr = semquad_vrms(s, st.qt, st.ux, st.uz)
    tm = semquad_mean_T(s, st.qt, st.theta, pr)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    th = Vector{Float64}(undef, n)
    T = Vector{Float64}(undef, n)
    sp = Vector{Float64}(undef, n)
    tmin = Inf
    tmax = -Inf
    @inbounds for i in 1:n
        xi, yi = s.coords[i]
        x[i] = xi
        y[i] = yi
        th[i] = st.theta[i]
        Ti = semquad_base_T(pr, g, xi, yi) + st.theta[i]
        T[i] = Ti
        tmin = min(tmin, Ti)
        tmax = max(tmax, Ti)
        sp[i] = hypot(st.ux[i], st.uz[i])
    end
    tag = @sprintf("frame_%05d", fw.frame)
    binpath = joinpath(fw.dir, tag * ".bin")
    tmppath = binpath * ".tmp"
    open(tmppath, "w") do io
        write(io, Int64(n))
        write(io, x)
        write(io, y)
        write(io, th)
        write(io, T)
        write(io, sp)
    end
    mv(tmppath, binpath; force=true)
    @printf(fw.meta,
            "%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d\n",
            fw.frame, st.step, st.t, nut, nub, vr, tm, tmin, tmax, n)
    flush(fw.meta)
    fw.frame += 1
    return nothing
end

"""
    annulus_frame_due!(fw, st, pr, g) -> Bool

Write one frame if the state is the first state the driver exposes (`frame ==
0`) or if `st.t` has reached the next multiple of the frame cadence.  Returns
`true` when a frame was written.
"""
function annulus_frame_due!(fw::AnnulusFrameWriter, st, pr::Params,
                            g::PolarAnnulus)
    if fw.frame == 0
        write_annulus_frame!(fw, st, pr, g)
        fw.next_t = fw.dt
        while st.t >= fw.next_t - 1.0e-12
            fw.next_t += fw.dt
        end
        return true
    end
    st.t >= fw.next_t - 1.0e-12 || return false
    write_annulus_frame!(fw, st, pr, g)
    while st.t >= fw.next_t - 1.0e-12
        fw.next_t += fw.dt
    end
    return true
end

"""
    annulus_div_volume(g, sq, ux, uz) -> Float64

`int_Omega div u dA` on the mapped ring (Gauss quadrature through the leaf
metrics).  With the nodal pressure pin this is the *dropped* continuity row: it
is zero when the constant pressure is exactly in the discrete null space and
`O(h^(p+1))` when only the nodal normal velocity is Dirichlet on the curved
free-slip circles (stage 0.5b2 design entry 2).  It is the pressure-pin caveat's
direct measure; the benchmark record reports it per resolution.
"""
function annulus_div_volume(g, sq, ux, uz)
    p = sq.p
    nu = sq.N
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = ANN_MS.quad_measure(m, fd, qx, qy)
            dux = 0.0
            duz = 0.0
            for j in 1:p+1, i in 1:p+1
                gx, gy = basis_grad(m, fd, qx, qy, i, j)
                k = (j - 1) * (p + 1) + i
                dux += ux[ids[k]] * gx
                duz += uz[ids[k]] * gy
            end
            s += w * (dux + duz)
        end
    end
    return s
end

"""`dTbar/dr`-weighted normal-flux defect: the base-advection integral `int u.grad Tbar`."""
function annulus_base_advection(g, sq, pr, ux, uz)
    p = sq.p
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = ANN_MS.quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0; uzq = 0.0
            for j in 1:p+1, i in 1:p+1
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                k = (j - 1) * (p + 1) + i
                uxq += ux[ids[k]] * b
                uzq += uz[ids[k]] * b
            end
            s += w * upward_velocity(g, x, y, uxq, uzq) *
                 ANN_MS.semquad_base_gradient(pr, g, x, y)
        end
    end
    return s
end

"""Parse `NXxNY` (or a single `NX`, with `NY = 12*NX`)."""
function parse_resolution(spec::AbstractString)
    if occursin('x', spec)
        parts = split(spec, 'x')
        return parse(Int, parts[1]), parse(Int, parts[2])
    end
    nx = parse(Int, spec)
    return nx, 12 * nx
end

"""The card line's Rayleigh-number label: the historical literal at the default.

The published cards are Ra = 1e5 and the log has always said `Ra=1e5`; a run
with `ANN_RA` set prints the actual value in the same short scientific form, so
the default log stays byte-identical.
"""
function annulus_ra_label(ra::Float64)
    ra == 1.0e5 && return "1e5"
    if isfinite(ra) && ra > 0.0
        e = round(Int, log10(ra))
        10.0^e == ra && return "1e$e"
    end
    return @sprintf("%g", ra)
end

function main(args)
    isempty(args) && error("usage: annulus_benchmark.jl <nx>x<ny> [tstop]")
    nx, ny = parse_resolution(args[1])
    tstop = length(args) >= 2 ? parse(Float64, args[2]) : env_f("ANN_TSTOP", 3.0)
    dofs = annulus_benchmark_dofs(nx, ny)
    g = annulus_benchmark_geometry()
    qt = annulus_benchmark_tree(nx, ny)
    ss = make_semstokes(qt, 2; geom=g)
    pr = annulus_benchmark_params()
    # DOF arithmetic against the space actually built.
    @assert length(qt.leaves) == dofs.leaves
    @assert ss.su.N == dofs.T
    @assert ss.sp.N == dofs.p
    @assert 2 * ss.su.N + ss.sp.N == dofs.saddle
    adv = Symbol(env_s("ANN_ADVECTION", "galerkin"))
    ev = env_f("ANN_EV", 0.0)
    ic = Symbol(env_s("ANN_IC", "firedrake"))
    tol = env_f("ANN_TOL", 1.0e-6)
    window = env_i("ANN_WINDOW", 20)
    tmin = env_f("ANN_TMIN", 0.02)
    cfl = env_f("ANN_CFL", 1.0)
    dt0 = env_f("ANN_DT0", 1.0e-5)
    maxpic = env_i("ANN_MAXPIC", 1)
    pic_relax = env_f("ANN_PIC_RELAX", 1.0)
    report = env_i("ANN_REPORT", 25)
    ckdir = env_s("ANN_CKPT_DIR", "")
    ckevery = env_i("ANN_CKPT_EVERY", 200)
    # G291 continuation knobs: an explicit checkpoint to resume from and the
    # driver's physics-change protocol for it.  `refuse` is the default and a
    # same-physics restart never triggers the guard; `allow` is the documented
    # case-2.1 -> case-2.2 protocol (warn once, reset the first resumed step).
    restart_path = env_s("ANN_RESTART", "")
    restart_physics = Symbol(env_s("ANN_RESTART_PHYSICS", "refuse"))
    restart_physics in (:refuse, :allow) ||
        error("ANN_RESTART_PHYSICS must be refuse or allow (got " *
              "$(env_s("ANN_RESTART_PHYSICS", "")))")
    frames_dir = env_s("ANN_FRAMES", "")
    frame_dt = env_f("ANN_FRAME_DT", 0.05)
    frame_nr = env_i("ANN_FRAME_NR", 200)
    frame_nphi = env_i("ANN_FRAME_NPHI", 1024)
    frames_on = !isempty(frames_dir)
    ckpt = isempty(ckdir) ? "" : joinpath(ckdir, "ann_$(nx)x$(ny).h5")
    # `ANN_RESTART` is an explicit restart file and wins over the automatic
    # resume of an existing `ANN_CKPT_DIR` checkpoint; both are mutual
    # exclusive with `annulus_ic` (a restart carries its own theta, so the IC
    # knob is dropped below, exactly as the automatic resume always did).
    resume = isempty(ckpt) ? false : isfile(ckpt)
    restart_from = ckpt
    if !isempty(restart_path)
        isfile(restart_path) ||
            error("ANN_RESTART=$restart_path is not an existing checkpoint file")
        resume = true
        restart_from = restart_path
    end
    if !resume && restart_physics !== :refuse
        error("ANN_RESTART_PHYSICS=$(env_s("ANN_RESTART_PHYSICS", "")) needs a " *
              "restart: set ANN_RESTART=<checkpoint> or put a checkpoint in " *
              "ANN_CKPT_DIR")
    end
    !isempty(ckdir) && !isnothing(mkpath(ckdir))
    println("# annulus benchmark nx=$nx ny=$ny level=$(trailing_zeros(nx)) " *
            "cells_y=$(ny÷nx) leaves=$(dofs.leaves) dofs=$(dofs.saddle) " *
            "(u=$(dofs.u) p=$(dofs.p) T=$(dofs.T))")
    println("# card: Ra=$(annulus_ra_label(pr.Ra)) isoviscous H=0 free-slip ring R0=$(ANN_R0) " *
            "R1=$(ANN_R1) phi=2pi; advection=$adv ev=$ev ic=$ic")
    println("# criterion: t > $tmin, change_t/|dNu_T|/|dV_RMS| < $tol for " *
            "$window consecutive steps (steady_vrms=true); tstop=$tstop " *
            "maxsteps=$(env_i("ANN_MAXSTEPS", 200000)) ckpt=$(resume ? "resume" : isempty(ckpt) ? "off" : "new")")
    if resume
        println("# resume: file=$(restart_from) restart_physics=$restart_physics" *
                (restart_physics === :allow ?
                 " (intentional physics change: one warning, first resumed dt " *
                 "reset to dt0=$dt0)" : " (same-physics resume expected)"))
    end
    println("# mode: " *
            (cfl > 0 ? "time-accurate cfl=$cfl dt0=$dt0 dtmax=$(env_f("ANN_DTMAX", 1.0e-2))" :
                       "steady relaxation dt=$dt0 (CFL off; each step is a steady energy solve)") *
            " maxpic=$maxpic pic_relax=$pic_relax")
    if frames_on
        println("# frames: dir=$frames_dir every=$frame_dt analysis_grid=" *
                "$(frame_nr)x$(frame_nphi) (node-cloud states; the renderer " *
                "computes theta = T - <T>_phi(r) on the analysis grid)")
    end
    flush(stdout)
    t_run = time()
    n_report = 0
    fw = frames_on ?
         AnnulusFrameWriter(frames_dir, nx, ny, g, pr; dt=frame_dt,
                            nr=frame_nr, nphi=frame_nphi) : nothing
    cb = (st) -> begin
        if st.step % report == 0
            nt, nb = semquad_nu_walls(st.sq, st.qt, st.theta, pr)
            @printf("  step %6d t=%.6e dt=%.3e Nu_T=%.6f Nu_B=%.6f dNu=%.2e Vrms=%.5f meanT=%.6f rss=%.0fMB\n",
                    st.step, st.t, st.dt, nt, nb, nt - nb,
                    semquad_vrms(st.sq, st.qt, st.ux, st.uz),
                    semquad_mean_T(st.sq, st.qt, st.theta, pr), rss_mb())
            flush(stdout)
        end
        n_report += 1
        frames_on && annulus_frame_due!(fw, st, pr, g)
        return nothing
    end
    kw = Dict{Symbol,Any}(
        :geom => :annulus, :annulus_R0 => ANN_R0, :annulus_R1 => ANN_R1,
        :annulus_Phi => ANN_PHI, :annulus_seam => :periodic, :p => 2,
        :tstop => tstop, :dt0 => dt0,
        :dtmax => (cfl > 0 ? env_f("ANN_DTMAX", 1.0e-2) : dt0),
        :dtmin => 1.0e-16,
        :cfl => cfl, :maxsteps => env_i("ANN_MAXSTEPS", 200000),
        :linear => Symbol(env_s("ANN_LINEAR", "auto")),
        :advection => adv, :ev => ev, :maxpic => maxpic,
        :pic_relax => pic_relax, :pic_tol => env_f("ANN_PIC_TOL", 1.0e-6),
        :steady_tol => tol, :steady_window => window, :steady_vrms => true,
        :tmin => tmin, :annulus_ic => ic, :profile_steps => true,
        :verbose => false, :callback => cb,
        :checkpoint_path => ckpt, :checkpoint_every => ckevery)
    if Symbol(env_s("ANN_LINEAR", "auto")) === :mg
        kw[:mg_operator] = :matrixfree
        kw[:mg_threads] = env_i("ANN_MG_THREADS", 8)
    end
    if resume
        kw[:restart_from] = restart_from
        kw[:restart_physics] = restart_physics
        delete!(kw, :annulus_ic)  # a restart carries its own theta
    end
    st = run_semquad_simulation(qt, pr, zeros(Float64, ss.su.N); kw...)
    wall = time() - t_run
    nut, nub = semquad_nu_walls(st.ss.su, st.qt, st.theta, pr)
    vr = semquad_vrms(st.ss.su, st.qt, st.ux, st.uz)
    tm = semquad_mean_T(st.ss.su, st.qt, st.theta, pr)
    divv = annulus_div_volume(st.ss.su.geom, st.ss.su, st.ux, st.uz)
    ibase = annulus_base_advection(st.ss.su.geom, st.ss.su, pr, st.ux, st.uz)
    area = total_measure(st.ss.su.geom, st.ss.su)
    umax = maximum(hypot.(st.ux, st.uz))
    # last-window drift of the three criterion quantities
    h = st.history
    w = min(window, length(h))
    nut_drift = w > 1 ? abs(h[end][3] - h[end - w + 1][3]) : NaN
    vr_drift = w > 1 ? abs(h[end][5] - h[end - w + 1][5]) : NaN
    nrows = length(h)
    # The loop stops on the criterion, so a stop short of both caps is the
    # criterion firing; a run that hit its cap reports steady=0 with drifts.
    steady = tmin < st.t && nrows >= window && st.t < tstop &&
             st.steps < env_i("ANN_MAXSTEPS", 200000)
    # was the run stopped by the criterion (not maxsteps/tstop)?
    tprof = SEMQUAD_STEP_PROFILE[]
    per_step = isempty(tprof) ? NaN : sum(r.total for r in tprof) / length(tprof)
    stokes_s = isempty(tprof) ? NaN : sum(r.stokes for r in tprof) / length(tprof)
    # warm per-step cost: the first steps carry the one-off JIT/hierarchy build,
    # so the average over steps after the first three is the production number
    warm = length(tprof) > 3 ? tprof[4:end] : tprof
    s_step_warm = isempty(warm) ? NaN :
                  sum(r.total for r in warm) / length(warm)
    stokes_warm = isempty(warm) ? NaN :
                  sum(r.stokes for r in warm) / length(warm)
    @printf("ANNULUS_RESULT %s nx=%d ny=%d level=%d leaves=%d u_dofs=%d p_dofs=%d T_dofs=%d saddle_dofs=%d steps=%d t=%.6e dt=%.3e wall=%.1f s_per_step=%.4f s_step_warm=%.4f stokes=%.4f stokes_warm=%.4f NuT=%.6f NuB=%.6f dNu=%.3e Vrms=%.5f meanT=%.6f Umax=%.3f div_vol=%.3e div_rel=%.2e Ibase=%.3e rss=%.0f peak=%.0f steady=%d window=%d nut_drift=%.2e vr_drift=%.2e adv=%s ev=%.3g ic=%s gauge=%s pgauge=%s\n",
            isempty(env_s("ANN_TAG", "")) ? "-" : env_s("ANN_TAG", ""),
            nx, ny, trailing_zeros(nx), length(st.qt.leaves), 2 * st.ss.su.N,
            st.ss.sp.N, st.ss.su.N, 2 * st.ss.su.N + st.ss.sp.N, st.steps,
            st.t, h[end][2], wall, per_step, s_step_warm, stokes_s,
            stokes_warm, nut, nub, nut - nub, vr,
            tm, umax, divv, abs(divv) / max(umax * area, 1.0e-300),
            ibase, rss_mb(), peak_mb(), steady ? 1 : 0, window, nut_drift,
            vr_drift, adv, ev, ic,
            string(ANN_MS.LAST_SEMSTOKES_ROTATION_GAUGE[]),
            string(ANN_MS.LAST_SEMSTOKES_PRESSURE_GAUGE[]))
    # deviation from the two published codes (the numbers the record quotes)
    @printf("ANNULUS_DEV NuT_vs_aspect=%.4f%% NuT_vs_firedrake=%.4f%% NuB_vs_aspect=%.4f%% Vrms_vs_aspect=%.4f%% Vrms_vs_firedrake=%.4f%% meanT_vs_aspect=%.4f%% dNu_rel=%.4f%%\n",
            100 * (nut - ANN_REF.Aspect_NuT) / ANN_REF.Aspect_NuT,
            100 * (nut - ANN_REF.Firedrake_Nu) / ANN_REF.Firedrake_Nu,
            100 * (nub - ANN_REF.Aspect_NuB) / ANN_REF.Aspect_NuB,
            100 * (vr - ANN_REF.Aspect_Vrms) / ANN_REF.Aspect_Vrms,
            100 * (vr - ANN_REF.Firedrake_Vrms) / ANN_REF.Firedrake_Vrms,
            100 * (tm - ANN_REF.Aspect_meanT) / ANN_REF.Aspect_meanT,
            100 * (nut - nub) / ((nut + nub) / 2))
    if frames_on
        close(fw.meta)
        @printf("ANNULUS_FRAMES n=%d dir=%s every=%.6g analysis_grid=%dx%d wall=%.1f\n",
                fw.frame, frames_dir, frame_dt, frame_nr, frame_nphi, wall)
    end
    flush(stdout)
    return st
end

# Run only as a script: the gate `test/t_g277_annulus_benchmark.jl` includes
# this file (in an isolated module) to pin the card and the diagnostics without
# launching a benchmark.
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
