# G246: the Maxwell stress history as driver state (docs/dev/PHASE2.md 2b step 1).
#
# Three Q2 nodal components `txx`, `tyy`, `txy` are carried through the run the
# same way the weakening field `chi` is: allocated (or restart-loaded), advanced
# once per step with the already-gated `maxwell_stress_step`, transferred on an
# AMR cycle and written to / read from the checkpoint.  The enable switch is
# `pr.shear_modulus > 0`; with `G = 0` (the default) the state is `nothing` and
# every stress-history code path is skipped, so existing runs are bit-identical.
#
# This gate pins:
#   (a) `G = 0` carries no state, writes no checkpoint dataset, and two
#       constructions of the default run are bit-identical;
#   (b) at a *fixed* non-zero strain rate the three fields follow
#       `maxwell_stress_exact` to the first-order tolerance, with the tensor
#       mapping txx <- exx, tyy <- ezz, txy <- exz;
#   (c) the driver advances them once per step with that step's velocity, dt,
#       G and material viscosity (reconstructed exactly from the callbacks);
#   (d) a checkpoint round trip is bit-exact for the fields and a restart
#       continues the trajectory bit-identically;
#   (e) an old-format file without the datasets loads with `nothing`, and a
#       restart from it follows a bit-identical theta/ux trajectory while the
#       stress history differs -- the direct evidence that the momentum
#       equation never sees the stress (no `div(tau)` term in this change);
#   (f) an AMR cycle in the middle of a run keeps the fields finite, non-trivial
#       and inside the analytic band |tau| <= 2 eta max|edot|, with the transfer
#       continuous across the mesh change;
#   (g) the kernel's `G = 0` escape leaves the state untouched, and the
#       save/load/restart guards reject a malformed stress state.
using Test, QuaST, LinearAlgebra, HDF5, Printf

@testset "G246 Maxwell stress state" begin
    # A tree that starts coarse, so an AMR cycle in the gate can change it.
    function g246_tree(nref::Int=2)
        q = Quadtree(2)
        for _ in 1:nref, lf in collect(q.leaves)
            refine!(q, lf)
        end
        balance!(q)
        return q
    end
    function g246_ic(ss)
        th = [0.05 * sinpi(y) * cospi(2.0 * x) for (x, y) in ss.su.coords]
        for (i, (x, y)) in enumerate(ss.su.coords)
            (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[i] = 0.0)
        end
        return th
    end
    # `eta_law = :expT` with `deta_T = 1` is `eta = deta_T^(-T) = 1` exactly, so
    # the material viscosity the stress update samples is the known constant 1
    # and the expected elastic / steady stresses are analytic.
    const_pr(; G=0.0, kw...) =
        Params(; Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
               shear_modulus=G, kw...)
    function g246_run(pr; nsteps::Int=3, dt::Float64=1.0e-4, kw...)
        q = g246_tree()
        ss = make_semstokes(q, 2)
        th0 = g246_ic(ss)
        run_semquad_simulation(q, pr, th0; p=2, maxsteps=nsteps, tstop=1.0,
                               dt0=dt, dtmax=dt, dtmin=1.0e-30, cfl=0.0,
                               linear=:direct, ev=0.0, kw...)
    end

    # ---- (a) the default G = 0 carries no state --------------------------
    @test const_pr().shear_modulus == 0.0
    r0 = g246_run(const_pr(); nsteps=3)
    rz = g246_run(const_pr(G=0.0); nsteps=3)
    @test r0.txx === nothing && r0.tyy === nothing && r0.txy === nothing
    @test rz.txx === nothing && rz.tyy === nothing && rz.txy === nothing
    @test r0.theta == rz.theta
    @test r0.ux == rz.ux && r0.uz == rz.uz && r0.pv == rz.pv
    @test r0.history == rz.history
    ck0 = joinpath(mktempdir(), "g246_g0.h5")
    g246_run(const_pr(); nsteps=2, checkpoint_path=ck0, checkpoint_every=2)
    ld0 = load_semquad_state(ck0)
    @test ld0.txx === nothing && ld0.tyy === nothing && ld0.txy === nothing
    names0 = h5open(ck0, "r") do f
        collect(keys(f))
    end
    @test !("txx" in names0) && !("tyy" in names0) && !("txy" in names0)

    # ---- (b) fixed strain rate: the analytic Maxwell curve, component-wise
    # A linear velocity field is exactly representable by the p = 2 space, so
    # its strain-rate components are the prescribed constants at *every* node.
    qb = g246_tree(1)
    ssb = make_semstokes(qb, 2)
    sqb = ssb.su
    a, b, d, e = 0.3, 0.5, -0.1, -0.12          # exx=a, ezz=e, exz=(b+d)/2=0.2
    uxb = [a * x + b * y + 0.02 for (x, y) in sqb.coords]
    uzb = [d * x + e * y - 0.03 for (x, y) in sqb.coords]
    epsb = QuaST.semquad_strain_rate_components(sqb, uxb, uzb)
    @test maximum(abs.(epsb.exx .- a)) < 1.0e-12
    @test maximum(abs.(epsb.ezz .- e)) < 1.0e-12
    @test maximum(abs.(epsb.exz .- 0.5 * (b + d))) < 1.0e-12
    eta_b = 2.0
    G_b = 50.0
    tauM_b = eta_b / G_b
    eta_field = fill(eta_b, sqb.N)
    function g246_fixed(dt; t=tauM_b)
        txx = zeros(sqb.N); tyy = zeros(sqb.N); txy = zeros(sqb.N)
        n = round(Int, t / dt)
        for _ in 1:n
            QuaST._advance_maxwell_stress!(txx, tyy, txy, sqb, uxb, uzb, dt,
                                           eta_field, G_b)
        end
        return txx, tyy, txy
    end
    exx_b, ezz_b, exz_b = a, e, 0.5 * (b + d)
    ref = (maxwell_stress_exact(tauM_b, exx_b, eta_b, G_b),
           maxwell_stress_exact(tauM_b, ezz_b, eta_b, G_b),
           maxwell_stress_exact(tauM_b, exz_b, eta_b, G_b))
    # a uniform strain rate gives a uniform field: every node is the same value
    dt_b = tauM_b / 500
    txx1, tyy1, txy1 = g246_fixed(dt_b)
    @test maximum(txx1) - minimum(txx1) <= 1.0e-10 * abs(ref[1])
    @test isapprox(txx1[1], ref[1]; rtol=2.0e-3)
    @test isapprox(tyy1[1], ref[2]; rtol=2.0e-3)
    @test isapprox(txy1[1], ref[3]; rtol=2.0e-3)
    # the mapping is not symmetric: each field follows its own component
    @test txx1[1] > 0.0 && tyy1[1] < 0.0 && txy1[1] > 0.0
    @test abs(txx1[1] - tyy1[1]) > 0.5 * max(abs(txx1[1]), abs(tyy1[1]))
    # first-order convergence in dt (backward Euler), as G244 established
    err(dt) = abs(g246_fixed(dt)[1][1] - ref[1])
    e1 = err(tauM_b / 250)
    e2 = err(tauM_b / 500)
    @test e2 < e1
    @test 1.5 < e1 / e2 < 2.5
    # the elastic first step from rest is 2 G eps dt (dt << tau_M)
    dt_small = 1.0e-4 * tauM_b
    txxe = zeros(sqb.N); tyye = zeros(sqb.N); txye = zeros(sqb.N)
    QuaST._advance_maxwell_stress!(txxe, tyye, txye, sqb, uxb, uzb, dt_small,
                                   eta_field, G_b)
    @test isapprox(txxe[1], 2.0 * G_b * exx_b * dt_small; rtol=1.0e-3)
    @test isapprox(tyye[1], 2.0 * G_b * ezz_b * dt_small; rtol=1.0e-3)
    @test isapprox(txye[1], 2.0 * G_b * exz_b * dt_small; rtol=1.0e-3)

    # ---- (c) the driver advances the fields once per step ----------------
    # Reconstruct the update from the callback state of every step: the
    # strain-rate components of that step's velocity, that step's dt, eta = 1
    # and G.  Exact equality pins the whole wiring (velocity level, dt, eta,
    # one step per time step, all three components).
    G_d = 1.0e3
    prG = const_pr(G=G_d)
    uxs = Vector{Vector{Float64}}()
    uzs = Vector{Vector{Float64}}()
    dts = Float64[]
    rec_cb = s -> begin
        push!(uxs, copy(s.ux)); push!(uzs, copy(s.uz)); push!(dts, s.dt)
        nothing
    end
    rd = g246_run(prG; nsteps=4, dt=1.0e-4, callback=rec_cb)
    @test length(uxs) == 4
    @test rd.txx !== nothing && rd.tyy !== nothing && rd.txy !== nothing
    rd_tx = zeros(length(rd.txx)); rd_ty = zeros(length(rd.txx))
    rd_tz = zeros(length(rd.txx))
    for k in 1:4
        epsk = QuaST.semquad_strain_rate_components(rd.ss.su, uxs[k], uzs[k])
        for g in eachindex(rd_tx)
            rd_tx[g] = maxwell_stress_step(rd_tx[g], epsk.exx[g], dts[k], 1.0,
                                           G_d)
            rd_ty[g] = maxwell_stress_step(rd_ty[g], epsk.ezz[g], dts[k], 1.0,
                                           G_d)
            rd_tz[g] = maxwell_stress_step(rd_tz[g], epsk.exz[g], dts[k], 1.0,
                                           G_d)
        end
    end
    @test rd_tx == rd.txx && rd_ty == rd.tyy && rd_tz == rd.txy
    @test maximum(abs, rd.txx) > 0.0
    # the fields actually differ from each other
    @test maximum(abs, rd.txx - rd.tyy) > 0.01 * maximum(abs, rd.txx)

    # The same reconstruction on the iterated and Newton nonlinear paths: the
    # update must happen once per *step* (not once per Picard iteration), after
    # the loop has converged.  Composition is where this codebase has failed
    # before, so these paths are pinned too; eta is again exactly 1.
    for kw in ((maxpic=2,), (pic_mode=:newton, newton_maxit=15))
        u2 = Vector{Vector{Float64}}()
        z2 = Vector{Vector{Float64}}()
        dt2 = Float64[]
        cb2 = s -> begin
            push!(u2, copy(s.ux)); push!(z2, copy(s.uz)); push!(dt2, s.dt)
            nothing
        end
        r2 = g246_run(prG; nsteps=3, dt=1.0e-4, callback=cb2, kw...)
        p2x = zeros(length(r2.txx)); p2y = zeros(length(r2.txx))
        p2z = zeros(length(r2.txx))
        for k in 1:3
            ek = QuaST.semquad_strain_rate_components(r2.ss.su, u2[k], z2[k])
            for g in eachindex(p2x)
                p2x[g] = maxwell_stress_step(p2x[g], ek.exx[g], dt2[k], 1.0,
                                             G_d)
                p2y[g] = maxwell_stress_step(p2y[g], ek.ezz[g], dt2[k], 1.0,
                                             G_d)
                p2z[g] = maxwell_stress_step(p2z[g], ek.exz[g], dt2[k], 1.0,
                                             G_d)
            end
        end
        @test all(isfinite, r2.txx) && maximum(abs, r2.txx) > 0.0
        @test p2x == r2.txx && p2y == r2.tyy && p2z == r2.txy
    end

    # ---- (d) checkpoint round trip and bit-exact restart ----------------
    ck = joinpath(mktempdir(), "g246_ck.h5")
    ra = g246_run(prG; nsteps=3, checkpoint_path=ck, checkpoint_every=3)
    ld = load_semquad_state(ck)
    @test ld.txx !== nothing && ld.tyy !== nothing && ld.txy !== nothing
    @test ld.txx == ra.txx && ld.tyy == ra.tyy && ld.txy == ra.txy
    rb = g246_run(prG; nsteps=6, restart_from=ck)
    rc = g246_run(prG; nsteps=6)
    @test rb.theta == rc.theta && rb.ux == rc.ux && rb.uz == rc.uz
    @test rb.txx == rc.txx && rb.tyy == rc.tyy && rb.txy == rc.txy

    # ---- (e) an old file (no stress datasets) loads with `nothing`, and the
    # trajectory is unchanged because the momentum does not see the stress ----
    ck_old = joinpath(mktempdir(), "g246_ck_old.h5")
    cp(ck, ck_old; force=true)
    h5open(ck_old, "r+") do f
        for name in ("txx", "tyy", "txy")
            HDF5.delete_object(f, name)
        end
    end
    ld_old = load_semquad_state(ck_old)
    @test ld_old.txx === nothing && ld_old.tyy === nothing && ld_old.txy === nothing
    rb_old = g246_run(prG; nsteps=6, restart_from=ck_old)
    @test rb_old.theta == rc.theta
    @test rb_old.ux == rc.ux && rb_old.uz == rc.uz
    @test all(isfinite, rb_old.txx)
    # ... while the stress history itself *did* restart from zero (non-vacuity:
    # the two restarts differ only in the field the momentum is supposed to
    # ignore; a `div(tau)` term would have made the trajectories differ).
    @test !(rb_old.txx == rc.txx)

    # ---- (f) an AMR cycle in the middle of the run ----------------------
    # The band is the analytic bound of the recursion, |tau| <= 2 eta max|edot|.
    # On an AMR step the callback sees the freshly zeroed velocity, so the
    # pre-AMR velocity of that step is estimated from the previous step (the
    # strain rate changes slowly at this dt) and from the no-AMR twin; 25% is
    # allowed for that estimate and for the mesh change itself.
    function g246_emax(sq, ux, uz)
        ec = QuaST.semquad_strain_rate_components(sq, ux, uz)
        return max(maximum(abs, ec.exx), maximum(abs, ec.ezz),
                   maximum(abs, ec.exz))
    end
    G_f = 1.0e4
    prGF = const_pr(G=G_f)
    E_amr = Float64[]
    amr_cb = s -> (push!(E_amr, g246_emax(s.sq, s.ux, s.uz)); nothing)
    q_f = g246_tree(1)
    leaves_before = length(q_f.leaves)
    ss_f = make_semstokes(q_f, 2)
    th_f = g246_ic(ss_f)
    ramr = run_semquad_simulation(q_f, prGF, th_f; p=2, maxsteps=8,
                                  tstop=1.0, dt0=1.0e-3, dtmax=1.0e-3,
                                  dtmin=1.0e-30, cfl=0.0, linear=:direct,
                                  maxpic=1, ev=0.0, amr_every=2,
                                  refine_frac=0.2, max_level=4,
                                  callback=amr_cb)
    @test length(ramr.qt.leaves) > leaves_before          # AMR really ran
    @test ramr.txx !== nothing && length(ramr.txx) == length(ramr.theta)
    for v in (ramr.txx, ramr.tyy, ramr.txy)
        @test all(isfinite, v)
    end
    E_twin = Float64[]
    twin_cb = s -> (push!(E_twin, g246_emax(s.sq, s.ux, s.uz)); nothing)
    rn = g246_run(prGF; nsteps=8, dt=1.0e-3, callback=twin_cb)
    @test all(isfinite, rn.txx)
    E_max = max(maximum(E_amr), maximum(E_twin))
    band = 2.0 * 1.25 * E_max
    tau_max = max(maximum(abs, ramr.txx), maximum(abs, ramr.tyy),
                  maximum(abs, ramr.txy))
    @test tau_max <= band
    @test tau_max > 0.05 * band                            # not reset to zero
    tau_twin = max(maximum(abs, rn.txx), maximum(abs, rn.tyy),
                   maximum(abs, rn.txy))
    @test 0.5 * tau_twin < tau_max < 2.0 * tau_twin       # same physics
    @printf("G246: AMR leaves=%d N=%d max|tau|=%.3e band=%.3e twin=%.3e\n",
            length(ramr.qt.leaves), ramr.ss.su.N, tau_max, band, tau_twin)

    # ---- (g) guards -----------------------------------------------------
    txx_g = fill(3.0, sqb.N); tyy_g = copy(txx_g); txy_g = copy(txx_g)
    QuaST._advance_maxwell_stress!(txx_g, tyy_g, txy_g, sqb, uxb, uzb, 1.0e-3,
                                   eta_field, 0.0)
    @test txx_g == fill(3.0, sqb.N) && tyy_g == fill(3.0, sqb.N) &&
          txy_g == fill(3.0, sqb.N)          # G = 0: off, not elastic-zero
    # the writer rejects a wrong-length field ...
    qg = g246_tree(1)
    ssg = make_semstokes(qg, 2); Ng = ssg.su.N
    zg = zeros(Ng)
    bad = joinpath(mktempdir(), "g246_bad.h5")
    @test_throws ArgumentError save_semquad_state(
        bad, qg, zg, zg, zeros(Ng), zeros(Ng), zeros(ssg.sp.N);
        t=0.0, dt=1.0e-4, dt_prev=1.0e-4, step=0, p=2, txx=zeros(Ng + 1))
    # ... and the reader rejects a wrong-length dataset
    good = joinpath(mktempdir(), "g246_good.h5")
    save_semquad_state(good, qg, zg, zg, zeros(Ng), zeros(Ng),
                       zeros(ssg.sp.N); t=0.0, dt=1.0e-4, dt_prev=1.0e-4,
                       step=0, p=2, txx=zeros(Ng), tyy=zeros(Ng),
                       txy=zeros(Ng))
    h5open(good, "r+") do f
        HDF5.delete_object(f, "tyy")
        f["tyy"] = zeros(Ng + 1)
    end
    @test_throws ArgumentError load_semquad_state(good)
    # a partial stress state is refused on restart
    partial = joinpath(mktempdir(), "g246_partial.h5")
    save_semquad_state(partial, qg, zg, zg, zeros(Ng), zeros(Ng),
                       zeros(ssg.sp.N); t=0.0, dt=1.0e-4, dt_prev=1.0e-4,
                       step=0, p=2, txx=zeros(Ng))
    @test_throws ArgumentError g246_run(prG; nsteps=2, restart_from=partial)
end
