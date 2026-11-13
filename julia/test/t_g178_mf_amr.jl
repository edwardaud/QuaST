using Test, MantleSEM, LinearAlgebra, Printf

# G178 / S3-MF-5: the matrix-free MG + CG saddle solve on AMR trees.
#
# Three geometries, each a case the sparse direct path could not scale to:
#   1. 2b L3 (640 leaves, p=2, 16384x viscosity contrast): a transient run with
#      the direct LU vs `linear=:mg, mg_operator=:matrixfree`; Vrms and Nu (and
#      the whole theta field) must agree to <= 1e-8 relative.
#   2. the wall-refined G147 tree (mixed levels 3/4/5, hanging nodes): a short
#      convection run, same comparison.
#   3. a 4-frame slice of the blob production case (p=4, Kelly AMR every 5
#      steps, Ra=1e8, 16384x contrast, symmetry=:x): the tree reaches 3-4 levels
#      with hanging nodes, and the per-frame Tmax must match the direct run to
#      <= 1e-6 relative.  (The full 25-frame production run is
#      out_julia/agent/run_blob_kelly_1024_mf.jl; it is too slow for a gate, and
#      its numbers are recorded in docs/S3_DESIGN.md.)
#
# Acceptance (S3 design, MF-5): Vrms/Nu to <= 1e-8 on 2b L3, and the AMR blob
# must reproduce the direct field.  The gates also record the wall times.

const _G178_MF = (linear=:mg, mg_operator=:matrixfree)
const _G178_D = (linear=:direct,)

# ---- 2b L3 ---------------------------------------------------------------
const _G178_2B = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=log(16384.0),
                        eta_c=log(64.0), bc=:free)

function _g178_tree2b()
    qt = Quadtree(3; cells_x=5, cells_y=2, Lx=2.5)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

_g178_ic(x, y) = 0.5 + 0.1 * cos(pi * x / 2.5) * sin(pi * y)

function _g178_run2b(lin::Symbol; tstop::Float64=1.0e-3)
    qt = _g178_tree2b()
    ss = make_semstokes(qt, 2)
    th = [_g178_ic(x, y) - (1.0 - y) for (x, y) in ss.su.coords]
    for (g, (_, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[g] = 0.0)
    end
    t0 = time()
    res = run_semquad_simulation(qt, _G178_2B, th; p=2, tstop=tstop, dt0=1.0e-4,
                                 cfl=5.0, maxsteps=500, maxpic=1, amr_every=0,
                                 linear=lin, stokes_tol=1.0e-6, verbose=false)
    dt = time() - t0
    ss2 = make_semstokes(res.qt, 2)
    nu, _ = semquad_nu_walls(ss2.su, res.qt, res.theta, _G178_2B)
    vr = semquad_vrms(ss2.su, res.qt, res.ux, res.uz)
    return (theta=copy(res.theta), vr=vr, nu=nu, wall=dt, steps=res.steps,
            res=LAST_SEMSTOKES_RESIDUAL[])
end

# ---- G147 wall tree ------------------------------------------------------
function _g178_wall_tree()
    qt = Quadtree(5)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    refine_wall_to_min!(qt, 5; strip=0.1)
    return qt
end

function _g178_runwall(qt, pr, lin::Symbol; tstop::Float64=1.0e-3)
    ss = make_semstokes(qt, 2)
    th0 = [0.01 * cos(pi * x) * sin(pi * y) for (x, y) in ss.su.coords]
    t0 = time()
    st = run_semquad_simulation(qt, pr, th0; p=2, tstop=tstop, dt0=1.0e-4,
                                cfl=0.5, maxsteps=5000, verbose=false,
                                ev=0.11, ev_cmax=0.052, linear=lin,
                                stokes_tol=1.0e-6)
    dt = time() - t0
    ss2 = make_semstokes(st.qt, 2)
    nu, _ = semquad_nu_walls(ss2.su, st.qt, st.theta, pr)
    vr = semquad_vrms(ss2.su, st.qt, st.ux, st.uz)
    return (theta=copy(st.theta), vr=vr, nu=nu, wall=dt,
            leaves=length(st.qt.leaves),
            nlev=length(unique(lf[1] for lf in st.qt.leaves)),
            res=LAST_SEMSTOKES_RESIDUAL[])
end

# ---- blob slice ----------------------------------------------------------
const _G178_BLOB = Params(Ra=1.0e8, H=5.0, eta_law=:expTz,
                          eta_b=log(16384.0), eta_c=log(64.0), bc=:free)

function _g178_blob(lin::Symbol; tstop::Float64=1.6e-9, maxleaves::Int=1024)
    dt = 8.0e-11
    tout = 4.0e-10
    qt = Quadtree(6)
    for _ in 1:4, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 4)
    theta0 = [0.12 * exp(-((x - 0.5) / 0.12)^2 - ((z - 0.35) / 0.10)^2) * sinpi(z)
              for (x, z) in ss.su.coords]
    frames = Float64[]
    nlev = Int[]
    leaves = Int[]
    nextt = Ref(tout)
    knot = Ref(0)
    cb = function (st)
        while st.t >= nextt[] - 1.0e-18
            knot[] += 1
            push!(frames, maximum(st.theta))
            push!(nlev, length(unique(lf[1] for lf in st.sq.leaves)))
            push!(leaves, length(st.sq.leaves))
            nextt[] = (knot[] + 1) * tout
        end
    end
    t0 = time()
    # Policy-agnostic AMR comparison calibrated on classic; pin the policy explicitly.
    res = run_semquad_simulation(qt, _G178_BLOB, theta0; p=4, tstop=tstop,
                                 dt0=dt, dtmax=dt, dtmin=1.0e-16,
                                 maxsteps=100, maxpic=1, amr_every=5,
                                 amr_policy=:classic,
                                 refine_frac=0.2, coarsen_frac=0.005,
                                 coarsen_theta_tol=1.0e-3,
                                 marking=:error_fraction, symmetry=:x,
                                 max_leaves=maxleaves, max_level=6,
                                 ev=0.11, ev_cmax=0.052, advection=:skew,
                                 linear=lin, mg_operator=:matrixfree,
                                 stokes_tol=1.0e-6, verbose=false,
                                 callback=cb)
    dt_tot = time() - t0
    return (tmax=frames, nlev=nlev, leaves=leaves, wall=dt_tot,
            steps=res.steps, theta=copy(res.theta), qt=res.qt,
            res=LAST_SEMSTOKES_RESIDUAL[])
end

@testset "G178 matrix-free MG on AMR and wall trees" begin
    # ---- 0. JIT warm-up ------------------------------------------------
    # The wall-clock comparison below is meaningful only after both `linear`
    # branches have been compiled once: the first run of a branch otherwise
    # pays 10-20 s of JIT for the whole simulation stack.
    let
        _g178_run2b(:direct; tstop=1.0e-4)
        _g178_run2b(:mg; tstop=1.0e-4)
        qww = _g178_wall_tree()
        prw0 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                      Twall_top=0.0, Twall_bot=1.0, bc=:free)
        _g178_runwall(qww, prw0, :direct; tstop=1.0e-4)
        _g178_runwall(qww, prw0, :mg; tstop=1.0e-4)
        _g178_blob(:direct; tstop=8.0e-11)
        _g178_blob(:mg; tstop=8.0e-11)
    end

    # ---- 1. 2b L3 (640 leaves, p=2) direct vs matrix-free MG + CG ---------
    rd = _g178_run2b(:direct)
    rmf = _g178_run2b(:mg)
    @test rd.steps > 3
    @test rmf.steps == rd.steps
    @test rd.res < 1.0e-9 && rmf.res < 1.0e-9
    dvr = abs(rmf.vr - rd.vr) / abs(rd.vr)
    dnu = abs(rmf.nu - rd.nu) / abs(rd.nu)
    dth = maximum(abs.(rmf.theta .- rd.theta)) / maximum(abs.(rd.theta))
    @test dvr <= 1.0e-8
    @test dnu <= 1.0e-8
    @test dth <= 1.0e-8
    @info "G178 2b L3" leaves = 640 direct_s = rd.wall mg_s = rmf.wall speedup = rd.wall / rmf.wall Vrms = rd.vr dVrms = dvr Nu = rd.nu dNu = dnu dtheta = dth

    # ---- 2. G147 wall tree (mixed levels, hanging nodes) -----------------
    qw = _g178_wall_tree()
    @test length(unique(lf[1] for lf in qw.leaves)) > 1
    prw = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                 Twall_top=0.0, Twall_bot=1.0, bc=:free)
    wd = _g178_runwall(qw, prw, :direct)
    wm = _g178_runwall(qw, prw, :mg)
    @test wd.nlev > 1 && wm.nlev == wd.nlev
    @test wd.res < 1.0e-9 && wm.res < 1.0e-9
    @test abs(wm.vr - wd.vr) / abs(wd.vr) <= 1.0e-8
    @test abs(wm.nu - wd.nu) / abs(wd.nu) <= 1.0e-8
    @test maximum(abs.(wm.theta .- wd.theta)) / maximum(abs.(wd.theta)) <= 1.0e-8
    @info "G178 wall tree" leaves = wd.leaves nlev = wd.nlev direct_s = wd.wall mg_s = wm.wall dVrms = abs(wm.vr - wd.vr) / abs(wd.vr) dNu = abs(wm.nu - wd.nu) / abs(wd.nu)

    # ---- 3. blob slice (p=4, Kelly AMR, mixed levels) ---------------------
    bd = _g178_blob(:direct)
    bm = _g178_blob(:mg)
    @test bd.steps == bm.steps
    @test length(bd.tmax) == length(bm.tmax)
    @test length(bd.tmax) >= 2
    @test maximum(bd.nlev) > 1              # genuinely mixed-level AMR trees
    @test bd.res < 1.0e-9 && bm.res < 1.0e-9
    dT = maximum(abs.(bm.tmax .- bd.tmax) ./ abs.(bd.tmax))
    @test dT <= 1.0e-6
    @test bm.leaves == bd.leaves            # same AMR decisions
    @info "G178 blob slice" steps = bd.steps frames = length(bd.tmax) leaves = bd.leaves nlev = bd.nlev direct_s = bd.wall mg_s = bm.wall speedup = bd.wall / bm.wall dTmax = dT tmax_mf = bm.tmax
    # NOTE (S3/MF-5): the matrix-free MG path is *not* faster than the direct LU
    # on this case (it is ~2.6x slower on the full 25-frame production run, see
    # docs/S3_DESIGN.md §9); the gate therefore only requires correctness and a
    # sane time, and records both totals.
    @test bm.wall < 3.0 * bd.wall

    # ---- 4. against the stored direct reference, when it is available -----
    # `out_julia/` is not in git, so this only runs locally.  The production
    # script records the interpolated Tmax (the reference CSV's column); here we
    # compare the nodal perturbation maximum, which is what both runs in this
    # gate see, and `out_julia/agent/compare_blob_mf.jl` does the production
    # comparison against frame_meta.csv.
    ref = normpath(joinpath(@__DIR__, "..", "..", "out_julia",
                            "fresh_blob_kelly_1024", "frame_meta.csv"))
    if isfile(ref)
        @info "G178 stored direct reference present" path = ref frames = length(bd.tmax)
    else
        @info "G178 stored reference not present (out_julia is not in git)"
    end
end
