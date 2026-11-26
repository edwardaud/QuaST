# G292 — the ASPECT -> QuaST annulus state import (the annulus twin of the box
# import path).
#
# What this gate pins (synthetic HDF5 states built here — no external ASPECT
# files, so the suite stays hermetic):
#
#   (a) exactness arm.  State grid = the ring's own Q2 node cloud (17 x 192
#       for the cheap 8 x 96 periodic ring), total T = Tbar(r) + R(r) with
#       R(r) = A s(1-s), s = (r-R0)/(R1-R0) — a global quadratic, hence exactly
#       representable by the state's Q2 field, and zero on both walls.  The
#       imported theta = T - Tbar must equal R at every node to round-off
#       (measured relative bound 1e-12), the walls must be exactly zero,
#       V_rms = 0 (zero velocity), the mean T must equal the closed-form radial
#       quadrature, and the family-A Nu pair must equal the *analytic* value
#       Nu_top = 1 - R'(R1) R1 lnr, Nu_bot = 1 - R'(R0) R0 lnr to round-off
#       (the Q2 one-sided wall derivative is exact for a quadratic).  Note on
#       the campaign brief's "Nu = 1 with R'(walls) = 0, e.g. R = A s(1-s)":
#       a nonzero quadratic that vanishes on both walls has R'(walls) = +-A,
#       so Nu = 1 is unattainable that way; the exact analytic pair is pinned
#       instead, and sub-arm (a2) with the quartic R = A (s(1-s))^2 (which DOES
#       have R'(walls) = 0) pins Nu = 1 to its measured discrete value — see
#       the campaign record for the algebra.
#   (a-wall) the wall-zeroing guard: the same state with a deliberate +0.01
#       total-temperature offset on the wall rows imports with theta == 0
#       exactly at the walls (the driver's 1e-12 wall guard), and the interior
#       still equals R.  Dropping the zeroing makes this arm red (mutation
#       campaign, `out_julia/agent/`).
#   (b) smooth-angular arm.  State grid 17 x 144 (radial nodes = the ring's,
#       coarse angular), T = Tbar + 0.2 s(1-s) cos(4 phi) — the documented
#       ASPECT/benchmark initial-perturbation form — so the transfer really
#       interpolates: the measured L-infinity error into the ring's GLL nodes
#       must stay below the stated bound, and the periodic seam trace must be
#       continuous (the importer checks phi = 0 vs phi = 2pi; the gate also
#       measures the theta jump at the seam nodes).
#   (c) checkpoint round trip: the written checkpoint loads, carries the
#       template's run_config (every key, same count), the geometry record and
#       the state's t; `theta_prev == theta`, step 0; a same-physics restart
#       from it advances one step and stays finite.
#   (d) refusal arms, each with an actionable message: a non-finite T field, a
#       state grid whose R0/R1 disagree with the target annulus, a
#       non-periodic angular grid (duplicate phi = 2pi column / odd nphi), a
#       state whose wall temperatures are not the benchmark's 1/0, and a
#       template whose run_config records another geometry.
#
# The real ASPECT frames (Davies et al. 2022 case 2.2 ref 4) are NOT in the
# gate: the import record is `benchmarks/results/aspc-davies-case22_annulus-
# import-same-state.md`; this gate is the cheap regression of the code path.
using Test
using QuaST
using HDF5
using Printf

const MS292 = QuaST

# `@printf` needs a literal format string; the gate's messages are assembled
# from adjacent literals across lines for readability, so use the runtime
# formatter (same output; negligible cost in a gate).
iprintf(fmt::AbstractString, args...) =
    print(Printf.format(Printf.Format(fmt), args...))

# The shipped importer, included in an isolated module so this gate pins the
# *same* code `examples/import_annulus_aspect_state.jl` runs.  Its `main_import`
# is guarded by `abspath(PROGRAM_FILE) == @__FILE__` and does not execute on
# include; the importer in turn includes `examples/annulus_benchmark.jl` for the
# card, geometry and uniform ring rule.
module G292Importer
using QuaST
include(joinpath(@__DIR__, "..", "examples", "import_annulus_aspect_state.jl"))
end
const IMP = G292Importer

# ---------------------------------------------------------------------------
# synthetic state files (written with HDF5.jl inside the gate)
# ---------------------------------------------------------------------------
"""
    g292_write_state(path; nr, nphi, R0, R1, T, ux, uy, t, frame, attrs)

Write one synthetic ASPECT-style state HDF5 file: datasets `r`, `phi`, `T`,
`ux`, `uy` and the extraction attributes (`R0`, `R1`, `cells_r`, `cells_t`,
`t`, `frame`, `source_vtu_sha256`).  `T`, `ux`, `uy` are (nr, nphi) node
matrices; `attrs` overrides/adds attributes.
"""
function g292_write_state(path::AbstractString; nr::Int, nphi::Int,
                          R0::Float64, R1::Float64, T::Matrix{Float64},
                          ux::Matrix{Float64}, uy::Matrix{Float64},
                          t::Float64, frame::Int=41,
                          attrs::Dict{String,Any}=Dict{String,Any}())
    size(T) == (nr, nphi) || error("T has size $(size(T)), expected ($nr, $nphi)")
    r = [R0 + (i - 1) * (R1 - R0) / (nr - 1) for i in 1:nr]
    phi = [(j - 1) * 2.0 * pi / nphi for j in 1:nphi]
    h5open(path, "w") do f
        f["r"] = r
        f["phi"] = phi
        f["T"] = T
        f["ux"] = ux
        f["uy"] = uy
        HDF5.attrs(f)["R0"] = R0
        HDF5.attrs(f)["R1"] = R1
        HDF5.attrs(f)["cells_r"] = (nr - 1) ÷ 2
        HDF5.attrs(f)["cells_t"] = nphi ÷ 2
        HDF5.attrs(f)["t"] = t
        HDF5.attrs(f)["frame"] = frame
        HDF5.attrs(f)["source_vtu_sha256"] = "g292-synthetic"
        for (k, v) in attrs
            HDF5.attrs(f)[k] = v
        end
    end
    return path
end

"""Radial base state of the annulus card (H = 0): `ln(R1/r)/ln(R1/R0)`."""
g292_Tbar(g, r) = log(g.R1 / r) / log(g.R1 / g.R0)

"""`s = (r - R0)/(R1 - R0)` (the height from the inner wall, gap = 1)."""
g292_s(g, r) = (r - g.R0) / (g.R1 - g.R0)

"""Area mean of a radial field `T(r)` by 1-D Gauss-Legendre quadrature of
`2 r T(r)/(R1^2 - R0^2) dr` (computed directly: a 64-point Gauss rule on each of
four radial panels, far beyond the accuracy of the field representations)."""
function g292_radial_mean(g, f; npanel::Int=4, nq::Int=64)
    # Gauss-Legendre nodes/weights on [-1, 1] by Newton iteration
    x = zeros(nq); w = zeros(nq)
    for i in 1:nq
        z = cos(pi * (i - 0.25) / (nq + 0.5))
        for _ in 1:100
            p0, p1 = 1.0, z
            for k in 2:nq
                p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
            end
            dp = nq * (z * p1 - p0) / (z * z - 1.0)
            dz = -p1 / dp
            z += dz
            abs(dz) < 1.0e-16 && break
        end
        p0, p1 = 1.0, z
        for k in 2:nq
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        dp = nq * (z * p1 - p0) / (z * z - 1.0)
        x[i] = z
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp)
    end
    edges = range(g.R0, g.R1; length=npanel + 1)
    s = 0.0
    for k in 1:npanel
        a, b = edges[k], edges[k + 1]
        c, h = 0.5 * (a + b), 0.5 * (b - a)
        for q in 1:nq
            r = c + h * x[q]
            s += w[q] * h * 2.0 * r * f(r)
        end
    end
    return s / (g.R1^2 - g.R0^2)
end

@testset "G292 annulus ASPECT state import" begin
    R0, R1 = IMP.ANN_R0, IMP.ANN_R1
    g = IMP.annulus_benchmark_geometry()
    mktempdir() do dir
        # ---- the cheap target ring and a template checkpoint of its card -----
        qt = IMP.annulus_benchmark_tree(8, 96)
        ss = make_semstokes(qt, 2; geom=g)
        n0 = ss.su.N
        pr = IMP.annulus_benchmark_params()
        function run_card(; maxsteps::Int, ic::Symbol=:none, ckpt::String="",
                          restart::String="")
            return run_semquad_simulation(qt, pr, zeros(Float64, n0);
                geom=:annulus, annulus_R0=R0, annulus_R1=R1, annulus_Phi=2.0 * pi,
                annulus_seam=:periodic, p=2, tstop=Inf, dt0=1.0e-3,
                dtmax=1.0e-2, dtmin=1.0e-16, cfl=10.0, maxsteps=maxsteps,
                linear=:direct, amr_every=0, advection=:galerkin, maxpic=1,
                annulus_ic=ic, checkpoint_path=ckpt, checkpoint_every=0,
                restart_from=restart, verbose=false)
        end
        tpl_path = joinpath(dir, "g292_template.h5")
        run_card(maxsteps=1, ic=:aspect, ckpt=tpl_path)
        @test isfile(tpl_path)
        tpl = load_semquad_state(tpl_path)
        @test haskey(tpl.config, "geometry")
        @test tpl.config["geometry"] == geometry_identity(g)
        @test tpl.config["Ra"] == "100000.0"
        @test length(tpl.config) > 50

        # state grid = the ring's own Q2 node cloud
        nr, nphi = 2 * 8 + 1, 2 * 96
        r = [R0 + (i - 1) * (R1 - R0) / (nr - 1) for i in 1:nr]
        phi = [(j - 1) * 2.0 * pi / nphi for j in 1:nphi]

        # ==================================================================
        # (a) exactness arm: total T = Tbar + R, R = A s(1-s), quadratic
        # ==================================================================
        @testset "(a) quadratic R is imported to round-off" begin
            A = 0.1
            Rf = rr -> begin
                s = g292_s(g, rr)
                A * s * (1.0 - s)
            end
            T = [g292_Tbar(g, r[i]) + Rf(r[i]) for i in 1:nr, j in 1:nphi]
            Z = zeros(nr, nphi)
            st_path = joinpath(dir, "g292_a_state.h5")
            g292_write_state(st_path; nr=nr, nphi=nphi, R0=R0, R1=R1, T=T,
                             ux=Z, uy=copy(Z), t=2.0)
            ck = joinpath(dir, "g292_a_ckpt.h5")
            res = IMP.import_annulus_state(state=st_path, template=tpl_path,
                                           out=ck, nx=8, ny=96, verbose=false)
            # the field really is nonzero and quadratic (non-vacuity)
            @test maximum(abs, res.theta) > 0.01
            # theta == R at every ring node to round-off
            err = 0.0
            Rscale = maximum(abs.(Rf.(res.state.r)))
            sq2 = make_semstokes(res.qt, 2; geom=g).su
            for (gi, (x, y)) in enumerate(sq2.coords)
                err = max(err, abs(res.theta[gi] - Rf(hypot(x, y))))
            end
            iprintf("G292 (a): max|theta_import - R| / max|R| = %.3e\n",
                    err / Rscale)
            @test err / Rscale <= 1.0e-12
            # walls are exactly zero (R vanishes there, and the guard enforces it)
            wall = [gi for (gi, (x, y)) in enumerate(sq2.coords)
                    if hypot(x, y) <= R0 + 1.0e-14 ||
                       hypot(x, y) >= R1 - 1.0e-14]
            @test length(wall) == 4 * 96      # 2 walls x 2*ny angular nodes
            @test all(res.theta[gi] == 0.0 for gi in wall)
            @test res.interp.nwall == 0        # nothing needed zeroing: R(walls)=0
            # V_rms = 0 for the zero velocity
            @test res.diag.Vrms == 0.0
            # mean T against the direct radial quadrature
            Tfull = rr -> g292_Tbar(g, rr) + Rf(rr)
            direct = g292_radial_mean(g, Tfull)
            iprintf("G292 (a): meanT imported = %.15f  direct quadrature = " *
                    "%.15f  rel = %.3e\n", res.diag.Tmean, direct,
                    (res.diag.Tmean - direct) / abs(direct))
            @test abs(res.diag.Tmean - direct) / abs(direct) <= 1.0e-8
            # Nu pair equals the analytic family-A value of a quadratic R
            lnr = log(R1 / R0)
            gap = R1 - R0
            dRdr_in = A / gap          # R'(R0) = +A/gap
            dRdr_out = -A / gap        # R'(R1) = -A/gap
            nu_bot_an = 1.0 - dRdr_in * R0 * lnr
            nu_top_an = 1.0 - dRdr_out * R1 * lnr
            iprintf("G292 (a): Nu_top = %.12f (analytic %.12f)  Nu_bot = " *
                    "%.12f (analytic %.12f)\n", res.diag.Nu_top, nu_top_an,
                    res.diag.Nu_bot, nu_bot_an)
            @test abs(res.diag.Nu_top - nu_top_an) <= 1.0e-12
            @test abs(res.diag.Nu_bot - nu_bot_an) <= 1.0e-12
            # the imported checkpoint exists and reloads with the state
            chk = load_semquad_state(ck)
            @test chk.t == 2.0
            @test chk.step == 0
            @test length(chk.theta) == sq2.N
            @test chk.theta == res.theta
            @test chk.theta_prev == res.theta
        end

        # ==================================================================
        # (a2) zero-wall-derivative profile: Nu = 1 to the discrete value
        # ==================================================================
        @testset "(a2) quartic profile pins Nu = 1" begin
            A = 0.1
            Qf = rr -> begin
                s = g292_s(g, rr)
                A * (s * (1.0 - s))^2
            end
            T = [g292_Tbar(g, r[i]) + Qf(r[i]) for i in 1:nr, j in 1:nphi]
            Z = zeros(nr, nphi)
            st_path = joinpath(dir, "g292_a2_state.h5")
            g292_write_state(st_path; nr=nr, nphi=nphi, R0=R0, R1=R1, T=T,
                             ux=Z, uy=copy(Z), t=2.0)
            res = IMP.import_annulus_state(state=st_path, template=tpl_path,
                                           nx=8, ny=96, verbose=false)
            sq2 = make_semstokes(res.qt, 2; geom=g).su
            Rscale = maximum(abs.(Qf.(res.state.r)))
            err = maximum(abs(res.theta[gi] - Qf(hypot(x, y)))
                          for (gi, (x, y)) in enumerate(sq2.coords))
            iprintf("G292 (a2): max|theta_import - Q| / max|Q| = %.3e\n",
                    err / Rscale)
            @test err / Rscale <= 1.0e-12
            # analytic family-A value of the Q2 interpolant's wall derivative
            # (the one-sided formula is exact for the per-cell quadratic)
            lnr = log(R1 / R0)
            gap = R1 - R0
            h = gap / 8.0                       # ring radial cell (8 cells)
            d_in = (-3.0 * 0.0 + 4.0 * Qf(R0 + h / 2) - Qf(R0 + h)) / h
            d_out = (3.0 * 0.0 - 4.0 * Qf(R1 - h / 2) + Qf(R1 - h)) / h
            nu_bot_dis = 1.0 - d_in * R0 * lnr
            nu_top_dis = 1.0 - d_out * R1 * lnr
            iprintf("G292 (a2): Nu_top = %.12f (discrete %.12f, |Nu-1| = " *
                    "%.2e)  Nu_bot = %.12f (discrete %.12f, |Nu-1| = %.2e)\n",
                    res.diag.Nu_top, nu_top_dis, abs(res.diag.Nu_top - 1.0),
                    res.diag.Nu_bot, nu_bot_dis,
                    abs(res.diag.Nu_bot - 1.0))
            @test abs(res.diag.Nu_top - nu_top_dis) <= 1.0e-12
            @test abs(res.diag.Nu_bot - nu_bot_dis) <= 1.0e-12
            @test abs(res.diag.Nu_top - 1.0) <= 2.0e-3
            @test abs(res.diag.Nu_bot - 1.0) <= 2.0e-3
            # the Q2 one-sided derivative is only O(h^2) accurate at the wall,
            # so the residual is dominated by that h^2 term, not by the import
            @test abs(res.diag.Nu_bot - 1.0) > 0.0
        end

        # ==================================================================
        # (a-wall) the wall zeroing guard
        # ==================================================================
        @testset "(a-wall) wall theta is zeroed" begin
            A = 0.1
            Rf = rr -> A * g292_s(g, rr) * (1.0 - g292_s(g, rr))
            T = [g292_Tbar(g, r[i]) + Rf(r[i]) for i in 1:nr, j in 1:nphi]
            # sub-tolerance wall noise (5e-7 < the importer's 1e-6 wall-T
            # consistency check, > the 1e-8 zeroing report threshold): the
            # guard must set these nodes to exactly 0 before the driver's
            # 1e-12 wall check sees them.
            T[1, :] .+= 5.0e-7
            T[nr, :] .+= -5.0e-7
            Z = zeros(nr, nphi)
            st_path = joinpath(dir, "g292_wall_state.h5")
            g292_write_state(st_path; nr=nr, nphi=nphi, R0=R0, R1=R1, T=T,
                             ux=Z, uy=copy(Z), t=2.0)
            res = IMP.import_annulus_state(state=st_path, template=tpl_path,
                                           nx=8, ny=96, verbose=false)
            sq2 = make_semstokes(res.qt, 2; geom=g).su
            wall = [gi for (gi, (x, y)) in enumerate(sq2.coords)
                    if hypot(x, y) <= R0 + 1.0e-14 ||
                       hypot(x, y) >= R1 - 1.0e-14]
            interior = setdiff(1:sq2.N, wall)
            iprintf("G292 (a-wall): %d wall nodes had |theta| > 1e-8 before " *
                    "zeroing (max %.3e); after: max|theta_wall| = %.3e\n",
                    res.interp.nwall, res.interp.max_wall,
                    maximum(abs.(res.theta[wall])))
            @test res.interp.nwall == length(wall)
            @test all(res.theta[gi] == 0.0 for gi in wall)
            @test maximum(abs, res.theta[interior]) > 0.01
        end

        # ==================================================================
        # (b) smooth-angular arm: the transfer really interpolates
        # ==================================================================
        @testset "(b) smooth angular field: interpolation bound + seam" begin
            nr_b, nphi_b = 2 * 8 + 1, 144      # radial nodes = the ring's
            rb = [R0 + (i - 1) * (R1 - R0) / (nr_b - 1) for i in 1:nr_b]
            phib = [(j - 1) * 2.0 * pi / nphi_b for j in 1:nphi_b]
            pert = (rr, pp) -> begin
                s = g292_s(g, rr)
                0.2 * s * (1.0 - s) * cos(4.0 * pp)
            end
            T = [g292_Tbar(g, rb[i]) + pert(rb[i], phib[j])
                 for i in 1:nr_b, j in 1:nphi_b]
            Z = zeros(nr_b, nphi_b)
            st_path = joinpath(dir, "g292_b_state.h5")
            g292_write_state(st_path; nr=nr_b, nphi=nphi_b, R0=R0, R1=R1, T=T,
                             ux=Z, uy=copy(Z), t=2.0)
            res = IMP.import_annulus_state(state=st_path, template=tpl_path,
                                           nx=8, ny=96, verbose=false)
            sq2 = make_semstokes(res.qt, 2; geom=g).su
            thex = [pert(hypot(x, y), mod(atan(y, x), 2.0 * pi))
                    for (x, y) in sq2.coords]
            fscale = maximum(abs.(thex))
            err = maximum(abs(res.theta[gi] - thex[gi])
                          for gi in eachindex(thex))
            iprintf("G292 (b): theta scale = %.6f, max interpolation error = " *
                    "%.3e (relative %.3e), bound 1e-3 relative\n",
                    fscale, err, err / fscale)
            @test err / fscale <= 1.0e-3
            @test err > 0.0        # this arm is not a nodal copy: it interpolates
            # the seam trace: phi = 0 and phi = 2pi agree at the seam nodes
            seam = [gi for (gi, (x, y)) in enumerate(sq2.coords)
                    if mod(atan(y, x), 2.0 * pi) == 0.0]
            @test length(seam) == 2 * 8 + 1
            yseam = [pert(hypot(sq2.coords[gi][1], sq2.coords[gi][2]), 0.0)
                     for gi in seam]
            @test maximum(abs.(res.theta[seam] .- yseam)) / fscale <= 1.0e-3
            @test res.interp.max_seam <= 1.0e-12
        end

        # ==================================================================
        # (c) checkpoint round trip and a same-physics restart
        # ==================================================================
        @testset "(c) checkpoint round trip + restart" begin
            A = 0.1
            Rf = rr -> A * g292_s(g, rr) * (1.0 - g292_s(g, rr))
            T = [g292_Tbar(g, r[i]) + Rf(r[i]) for i in 1:nr, j in 1:nphi]
            Z = zeros(nr, nphi)
            st_path = joinpath(dir, "g292_c_state.h5")
            g292_write_state(st_path; nr=nr, nphi=nphi, R0=R0, R1=R1, T=T,
                             ux=Z, uy=copy(Z), t=2.003)
            ck = joinpath(dir, "g292_c_ckpt.h5")
            res = IMP.import_annulus_state(state=st_path, template=tpl_path,
                                           out=ck, nx=8, ny=96, verbose=false)
            chk = load_semquad_state(ck)
            @test chk.config == tpl.config
            @test length(chk.config) == length(tpl.config)
            @test chk.config["geometry"] == geometry_identity(g)
            @test chk.t == 2.003
            @test chk.step == 0
            @test chk.dt == tpl.dt
            @test chk.dt_prev == tpl.dt_prev
            @test chk.velocity_stale == false
            @test chk.theta == res.theta
            # same-physics restart: one step, finite, advances t
            restart = run_card(maxsteps=2, restart=ck)
            @test restart.steps == 2
            @test restart.t > 2.003
            @test all(isfinite, restart.theta)
            @test all(isfinite, restart.ux)
            @test maximum(abs, restart.theta) < 10.0
            iprintf("G292 (c): restarted from the imported checkpoint: steps " *
                    "= %d, t = %.6f, max|theta| = %.3f\n", restart.steps,
                    restart.t, maximum(abs, restart.theta))
        end

        # ==================================================================
        # (d) refusals
        # ==================================================================
        @testset "(d) refusal arms" begin
            A = 0.1
            Rf = rr -> A * g292_s(g, rr) * (1.0 - g292_s(g, rr))
            Tgood = [g292_Tbar(g, r[i]) + Rf(r[i]) for i in 1:nr, j in 1:nphi]
            Z = zeros(nr, nphi)
            # (d1) non-finite T
            Tbad = copy(Tgood)
            Tbad[5, 7] = NaN
            p1 = joinpath(dir, "g292_badnan.h5")
            g292_write_state(p1; nr=nr, nphi=nphi, R0=R0, R1=R1, T=Tbad,
                             ux=Z, uy=copy(Z), t=2.0)
            e1 = try
                IMP.import_annulus_state(state=p1, template=tpl_path,
                                         nx=8, ny=96, verbose=false)
                nothing
            catch e
                e
            end
            @test e1 isa ArgumentError
            m1 = e1 === nothing ? "" : sprint(showerror, e1)
            @test occursin("non-finite", m1)
            iprintf("G292 (d1) refusal: %s\n", first(m1, 150))
            # (d2) a self-consistent state of ANOTHER annulus (shifted radii):
            # its own grid and attributes agree, but not with the target card
            p2 = joinpath(dir, "g292_badr.h5")
            g292_write_state(p2; nr=nr, nphi=nphi, R0=R0 + 0.08, R1=R1 + 0.08,
                             T=Tgood, ux=Z, uy=copy(Z), t=2.0)
            e2 = try
                IMP.import_annulus_state(state=p2, template=tpl_path,
                                         nx=8, ny=96, verbose=false)
                nothing
            catch e
                e
            end
            @test e2 isa ArgumentError
            m2 = e2 === nothing ? "" : sprint(showerror, e2)
            @test occursin("R0", m2) && occursin("target annulus geometry", m2)
            iprintf("G292 (d2) refusal: %s\n", first(m2, 150))
            # (d3) non-periodic angular grid: a duplicate phi = 2pi column
            nphi_bad = 2 * 96 + 1
            # uniform on the circle but with an ODD node count: only the
            # nphi == 2*cells_t (seam counted once) guard can catch this one
            phib = [(j - 1) * 2.0 * pi / nphi_bad for j in 1:nphi_bad]
            Tbad3 = [g292_Tbar(g, r[i]) + Rf(r[i]) for i in 1:nr, j in 1:nphi_bad]
            p3 = joinpath(dir, "g292_badseam.h5")
            h5open(p3, "w") do f
                f["r"] = r
                f["phi"] = phib
                f["T"] = Tbad3
                f["ux"] = zeros(nr, nphi_bad)
                f["uy"] = zeros(nr, nphi_bad)
                HDF5.attrs(f)["R0"] = R0
                HDF5.attrs(f)["R1"] = R1
                HDF5.attrs(f)["cells_r"] = 8
                HDF5.attrs(f)["cells_t"] = 96
                HDF5.attrs(f)["t"] = 2.0
                HDF5.attrs(f)["frame"] = 0
                HDF5.attrs(f)["source_vtu_sha256"] = "g292-synthetic"
            end
            e3 = try
                IMP.import_annulus_state(state=p3, template=tpl_path,
                                         nx=8, ny=96, verbose=false)
                nothing
            catch e
                e
            end
            @test e3 isa ArgumentError
            m3 = e3 === nothing ? "" : sprint(showerror, e3)
            @test occursin("seam", m3) && occursin("2*cells_t", m3)
            iprintf("G292 (d3) refusal: %s\n", first(m3, 180))
            # (d4) wall temperatures that are not the benchmark's 1/0
            p4 = joinpath(dir, "g292_badwall.h5")
            Twall = copy(Tgood)
            Twall[1, :] .+= 0.5
            g292_write_state(p4; nr=nr, nphi=nphi, R0=R0, R1=R1, T=Twall,
                             ux=Z, uy=copy(Z), t=2.0)
            e4 = try
                IMP.import_annulus_state(state=p4, template=tpl_path,
                                         nx=8, ny=96, verbose=false)
                nothing
            catch e
                e
            end
            @test e4 isa ArgumentError
            m4 = e4 === nothing ? "" : sprint(showerror, e4)
            @test occursin("wall temperatures", m4)
            iprintf("G292 (d4) refusal: %s\n", first(m4, 180))
            # (d5) a template whose geometry record is not the annulus
            wedge = PolarAnnulus(R0, R1, 0.5 * pi; seam=:wall)
            ck_wedge = joinpath(dir, "g292_template_wedge.h5")
            save_semquad_state(ck_wedge, tpl.qt, tpl.theta, tpl.theta_prev,
                               tpl.ux, tpl.uz, tpl.pv; t=tpl.t, dt=tpl.dt,
                               dt_prev=tpl.dt_prev, step=tpl.step, p=tpl.p,
                               config=merge(copy(tpl.config),
                                   Dict{String,Any}("geometry" =>
                                       geometry_identity(wedge))))
            p5 = joinpath(dir, "g292_d5state.h5")
            g292_write_state(p5; nr=nr, nphi=nphi, R0=R0, R1=R1, T=Tgood,
                             ux=Z, uy=copy(Z), t=2.0)
            e5 = try
                IMP.import_annulus_state(state=p5, template=ck_wedge,
                                         nx=8, ny=96, verbose=false)
                nothing
            catch e
                e
            end
            @test e5 isa ArgumentError
            m5 = e5 === nothing ? "" : sprint(showerror, e5)
            @test occursin("not the annulus target card", m5)
            @test occursin("PolarAnnulus", m5)
            iprintf("G292 (d5) refusal: %s\n", first(m5, 200))
        end
    end
end
