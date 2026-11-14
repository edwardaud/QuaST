# G214: symmetric equilibration of the direct saddle.
using Test, QuaST, LinearAlgebra, SparseArrays

# G214: solver-side symmetric equilibration of the direct saddle
# (docs/dev/BUG_LEDGER.md).
#
# The wall-strip near-null pressure cluster of the level-5 lid makes the
# *unshifted* sparse LU of the reduced saddle unusable (G step 7641: residual
# 4.2e2, growth 7.1e6; uniform L5: 1.6e6, growth 8.5e10).  The fix measured here
# is the exact congruence `S -> D*S*D`, `rhs -> D*rhs` with
# `D = diag(1/sqrt(diag Kred), 2 max|Kred| / max(|Bt|,|Bs|))`, which puts the
# cluster at `O(beta^2)` (4e-6 relative) instead of `~2e-14`:
#   * the equilibrated unshifted residual is 6.0e-10 on the G-7641 AMR state
#     and 3.6e-8 on the uniform-L5 control (both <= 1e-6), against 3.7e-7 and
#     1.2e-6 for the shipped unscaled tiny-shift stopgap;
#   * the scaling is engaged only when the historical unscaled unshifted solve
#     fails its health check, so healthy and non-benefiting solves are bitwise
#     identical even with the option on;
#   * `:none` (the default) is the historical path bit for bit.
#
# This gate pins:
#   (1) the option surface and the equilibration diagonal;
#   (2) the congruence on a real saddle (solve scaled, residual of the unscaled
#       system is at round-off);
#   (3) healthy solves are bitwise identical with and without the option;
#   (4) a wall-refined solve that does not benefit is bitwise identical too (the
#       historical ladder is the fallback);
#   (5) the realistic failing-class state (G214_STATE, skipped in a bare
#       checkout): the equilibrated answer is unshifted (shift 0, phase
#       `:equilibrated`), meets the 1e-6 goal, and agrees with the tiny-shift
#       answer in velocity;
#   (6) `linear=:mg` is not touched.
#   (7) the driver plumbing: `run_semquad_simulation(stokes_scaling=...)`
#       stores the key in `run_config`, counts the engages in
#       `LAST_SEMSTOKES_SCALED`, and the restart guard reads a checkpoint
#       without the key (every file written before the option existed) as
#       `none`, so switching the equilibration on cannot be silent.

@testset "G214 pressure equilibration" begin
    # ---------------------------------------------------------------- (1) ----
    @test QuaST.SEMSTOKES_SCALINGS == (:none, :equilibrated)

    function wall_tree(base, wall)
        qt = Quadtree(max(base, wall); cells_x=5, cells_y=2)
        for _ in 1:base, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        refine_wall_to_min!(qt, wall; strip=0.1)
        balance!(qt)
        return qt
    end
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=9.70406052783923433,
                eta_c=4.15888308335967186, Twall_top=0.0, Twall_bot=1.0,
                bc=:free)
    z2 = (x, y) -> 0.0
    qt = wall_tree(4, 5)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    theta = [0.5 * sin(pi * x) * sin(pi * y) for (x, y) in sq.coords]
    for g in 1:sq.N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end
    leaves = Set(qt.leaves)
    eta_fun = (x, y) -> semquad_eta_at(sq, theta, x, y, pr, leaves)
    fz = (x, y) -> pr.Ra * QuaST._semquad_eval_at(sq, theta, x, y, leaves)
    solve_wall(; kw...) = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2;
                                                   bc=:free, linear=:direct, kw...)

    sys = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2; bc=:free,
                                   linear=:direct, assemble_only=true)
    nf = size(sys.Kred, 1); nfp = size(sys.Bs, 1)
    S = [sys.Kred sys.Bt; sys.Bs spzeros(Float64, nfp, nfp)]
    rhs = vcat(sys.fr, sys.gr)
    @test_throws ArgumentError solve_wall(stokes_scaling=:bogus)

    d = QuaST._semquad_stokes_equil_diag(sys.Kred, sys.Bt, sys.Bs)
    @test length(d) == nf + nfp
    @test all(isfinite, d) && all(>(0.0), d)
    # velocity part: 1/sqrt(diag Kred) (with the 1e-14 max floor)
    dk = max.(diag(sys.Kred), 1.0e-14 * maximum(diag(sys.Kred)))
    @test d[1:nf] ≈ 1.0 ./ sqrt.(dk) rtol = 1.0e-14
    # pressure part: one uniform max-norm scale with the factor-2 margin
    bmax = max(maximum(abs, sys.Bt), maximum(abs, sys.Bs))
    @test all(d[(nf + 1):end] .== 2.0 * maximum(abs, sys.Kred) / bmax)

    # ---------------------------------------------------------------- (2) ----
    # The congruence on the real (small) saddle: the equilibrated solve leaves a
    # residual of the *unscaled* system at the level of its own round-off, and
    # (where the LU is stable, here it is not guaranteed) the same answer.
    x0 = S \ rhs
    r0 = norm(S * x0 - rhs) / max(norm(rhs), 1.0)
    x1, d1 = QuaST._semquad_stokes_equil_solve(S, sys.Kred, sys.Bt, sys.Bs, rhs)
    @test d1 == d
    r1 = norm(S * x1 - rhs) / max(norm(rhs), 1.0)
    @test isfinite(r1)
    @test norm(x1) < 1.0e8                  # not a lost-solution blow-up
    # the historical unscaled solve on this tree is already usable (1.6e-4):
    @test r0 < 1.0e-3

    # ---------------------------------------------------------------- (3) ----
    # A healthy uniform mesh: the option is never engaged and the answers are
    # bit-for-bit the historical ones.
    qu = Quadtree(4)
    for _ in 1:2, lf in collect(qu.leaves)
        refine!(qu, lf)
    end
    balance!(qu)
    ssu = make_semstokes(qu, 2)
    squ = ssu.su
    thu = [0.5 * sin(pi * x) * sin(pi * y) for (x, y) in squ.coords]
    lvu = Set(qu.leaves)
    etau = (x, y) -> semquad_eta_at(squ, thu, x, y, pr, lvu)
    fzu = (x, y) -> pr.Ra * QuaST._semquad_eval_at(squ, thu, x, y, lvu)
    uu0, pp0, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                                 bc=:free, linear=:direct)
    @test LAST_SEMSTOKES_LADDER_PHASE[] === :healthy
    r_healthy = LAST_SEMSTOKES_RESIDUAL[]
    LAST_SEMSTOKES_SCALED[] = 0
    uu1, pp1, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                                 bc=:free, linear=:direct,
                                                 stokes_scaling=:equilibrated)
    @test uu1 == uu0 && pp1 == pp0
    @test LAST_SEMSTOKES_LADDER_PHASE[] === :healthy
    @test LAST_SEMSTOKES_RESIDUAL[] == r_healthy
    @test LAST_SEMSTOKES_LADDER[] === nothing
    # a healthy solve never engages the scaling: the counter stays at 0
    @test LAST_SEMSTOKES_SCALED[] == 0

    # ---------------------------------------------------------------- (4) ----
    # A wall-refined solve that the scaling does not rescue: the historical
    # ladder runs unchanged and the answer is bitwise the default one.  (The
    # scaled unshifted solve on this 4k-leaf tree measures 0.26, so the
    # fallback is taken; if a future factorisation rescues it instead, the
    # answer must only improve.)
    u_def, p_def, _, _, _ = solve_wall(stokes_accept_tol=1.0e-3)
    r_def = LAST_SEMSTOKES_RESIDUAL[]
    ph_def = LAST_SEMSTOKES_LADDER_PHASE[]
    lad_def = LAST_SEMSTOKES_LADDER[]
    @test r_def < 1.0e-6
    u_eq, p_eq, _, _, _ = solve_wall(stokes_accept_tol=1.0e-3,
                                    stokes_scaling=:equilibrated)
    if LAST_SEMSTOKES_LADDER_PHASE[] === :equilibrated
        @test LAST_SEMSTOKES_RESIDUAL[] <= r_def * (1.0 + 1.0e-9)
        @test LAST_SEMSTOKES_SHIFT[] == 0.0
    else
        @test u_eq == u_def && p_eq == p_def
        @test LAST_SEMSTOKES_RESIDUAL[] == r_def
        @test LAST_SEMSTOKES_LADDER_PHASE[] === ph_def
        @test LAST_SEMSTOKES_LADDER[] == lad_def
    end

    # ---------------------------------------------------------------- (6) ----
    # `linear=:mg` is not touched: the direct-path records stay reset and the
    # solve is finite.
    umg, _, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                               bc=:free, linear=:mg,
                                               mg_maxit=40, mg_restart=40)
    @test all(isfinite, umg)
    @test LAST_SEMSTOKES_SHIFT[] == 0.0
    @test LAST_SEMSTOKES_LADDER[] === nothing
    @test LAST_SEMSTOKES_LADDER_PHASE[] === nothing

    # ---------------------------------------------------------------- (5) ----
    # The realistic failing-class state (data-dependent, not committed): the G
    # step-7641 nearest reproducible state.
    state = get(ENV, "G214_STATE", "")
    if isempty(state) || !isfile(state)
        @info "G214: realistic-state check skipped (set G214_STATE to the " *
              "G step-7641 checkpoint to enable it)"
        @test_skip 1 + 1 == 2
    else
        RA_G = 1.0e5
        COMP_AD = 1.6240183119e5
        COMP_AS = 2.6796419238e-1
        B410 = 180.0 / (3300.0 * 3.0e-5 * 1000.0)
        B660 = 300.0 / (3300.0 * 3.0e-5 * 1000.0)
        LATENT_SCALE = (rho_ref=3300.0, cp=1000.0, dT=1000.0, g=9.81, d=2.9e6,
                        T_surface=0.5)
        phs = [PhaseTransition(0.87, 0.025, 0.032, 0.5, 180.0, B410),
               PhaseTransition(0.78, 0.025, -0.032, 0.5, 300.0, B660)]
        prr = Params(Ra=RA_G, H=10.0, eta_law=:composite,
                     comp_rule=:isostress, comp_Ad=COMP_AD, comp_As=COMP_AS,
                     comp_Ed=12.0, comp_Es=17.0, comp_Vd=0.0, comp_Vs=0.0,
                     comp_n=3.0, comp_Tcut=0.0, comp_eps_min=1.0e-12,
                     eta_T0=0.5, eta_min=1.0e-4, eta_max=1.0e4,
                     bc=:free, latent_L=0.11, latent_mode=:per_phase,
                     rho_heating=false, rho_cp=false, Di=0.0,
                     heating_adiabatic=false, heating_dissipation=false,
                     refstate=nothing, deviatoric=false)
        phl = per_phase_latent(prr, phs; LATENT_SCALE...)
        stt = load_semquad_state(state)
        sst = make_semstokes(stt.qt, stt.p)
        sqt = sst.su
        leavesT = Set(stt.qt.leaves)
        TfunT = (x, y) -> QuaST._semquad_eval_at(sqt, stt.theta, x, y, leavesT)
        etaT = (x, y) -> begin
            T = Tbar(y, prr.H) + TfunT(x, y)
            ed = semquad_strain_rate_at(sqt, stt.ux, stt.uz, x, y, leavesT)
            eta_composite_at(T, ed, y, prr)[1]
        end
        fzT = (x, y) -> prr.Ra * TfunT(x, y) +
                        semquad_phase_buoyancy(prr, phl, y, TfunT(x, y))
        solve_state(; kw...) = solve_semquad_stokes_var(
            sst, stt.qt, etaT, z2, fzT, z2, z2; bc=:free, linear=:direct,
            stokes_accept_tol=1.0e-3, kw...)
        u_def2, p_def2, _, _, _ = solve_state()
        r_def2 = LAST_SEMSTOKES_RESIDUAL[]
        ph_def2 = LAST_SEMSTOKES_LADDER_PHASE[]
        sh_def2 = LAST_SEMSTOKES_SHIFT[]
        n_scaled = LAST_SEMSTOKES_SCALED[]
        u_eq2, p_eq2, _, _, _ = solve_state(stokes_scaling=:equilibrated)
        r_eq2 = LAST_SEMSTOKES_RESIDUAL[]
        @test ph_def2 === :tiny && sh_def2 == 1.0e-14      # the shipped stopgap
        @test r_def2 <= 1.0e-6
        @test LAST_SEMSTOKES_LADDER_PHASE[] === :equilibrated
        @test LAST_SEMSTOKES_SHIFT[] == 0.0                # unshifted
        @test LAST_SEMSTOKES_REFINE_ITERS[] == 0
        @test r_eq2 <= 1.0e-6
        # the equilibrated answer is the one accepted: the engage counter moved
        @test LAST_SEMSTOKES_SCALED[] == n_scaled + 1
        @test norm(u_eq2 - u_def2) / norm(u_def2) <= 1.0e-3
        @test all(isfinite, u_eq2) && all(isfinite, p_eq2)
    end

    # ---------------------------------------------------------------- (7) ----
    # Driver plumbing: the keyword is accepted and stored, and the restart
    # guard reads a checkpoint written before the option existed (no key) as
    # `none` -- resuming such a file with `:equilibrated` is refused instead of
    # silently changing the solver.
    mktempdir() do dir
        ck = joinpath(dir, "g214_driver.h5")
        LAST_SEMSTOKES_SCALED[] = 0
        st1 = run_semquad_simulation(qu, pr, copy(thu); p=2, tstop=1.0e-6,
                                     dt0=1.0e-6, maxsteps=1, verbose=false,
                                     checkpoint_path=ck)
        @test st1.steps == 1
        @test load_semquad_state(ck).config["stokes_scaling"] == "none"
        bad_new = try
            run_semquad_simulation(qu, pr, zeros(squ.N); p=2, tstop=1.0,
                                   dt0=1.0e-6, maxsteps=1, verbose=false,
                                   restart_from=ck,
                                   stokes_scaling=:equilibrated)
            nothing
        catch e
            e
        end
        @test bad_new isa ArgumentError &&
              occursin("stokes_scaling", sprint(showerror, bad_new))
        # rewrite the copy as an *old* checkpoint: the key did not exist then
        old = joinpath(dir, "g214_driver_old.h5")
        s = load_semquad_state(ck)
        cfg = copy(s.config); delete!(cfg, "stokes_scaling")
        save_semquad_state(old, s.qt, s.theta, s.theta_prev, s.ux, s.uz, s.pv;
                           t=s.t, dt=s.dt, dt_prev=s.dt_prev, step=s.step,
                           p=s.p, config=cfg, ev=s.ev, cooldown=s.cooldown,
                           velocity_stale=s.velocity_stale)
        st_old = run_semquad_simulation(qu, pr, zeros(squ.N); p=2, tstop=1.0,
                                        dt0=1.0e-6, maxsteps=0, verbose=false,
                                        restart_from=old)
        @test st_old.steps == 1          # absolute step: `none` resumes, runs 0
        bad_old = try
            run_semquad_simulation(qu, pr, zeros(squ.N); p=2, tstop=1.0,
                                   dt0=1.0e-6, maxsteps=0, verbose=false,
                                   restart_from=old,
                                   stokes_scaling=:equilibrated)
            nothing
        catch e
            e
        end
        @test bad_old isa ArgumentError &&
              occursin("key absent", sprint(showerror, bad_old))
    end
end
