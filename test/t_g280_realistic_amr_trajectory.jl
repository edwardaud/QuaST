# G280: the AMR trajectory regression gate for the box gradient-jump indicator.
#
# The nine-hash process dump (`out_julia/agent/state_dump_run.jl` / `b_rs.txt`)
# is inert in AMR: its tree is `Quadtree(2)` refined twice, every leaf is at the
# tree's `maxlevel`, the coarsening fraction is zero, and no cycle decision is
# consumed.  Its hashes are therefore invariant under the 2026-11-25 correction
# of the box indicator's y-face index (measured) -- they never pinned it.  The
# AMR-active dump card (`out_julia/agent/amr_state_dump_run.jl` /
# `amr_rs.txt`) pins the process; this gate pins the *trajectory* with an
# independent, in-process run of the shipped realistic card
# (`examples/case_realistic_2d.jl`, the audit's smoke fixture): its AMR decision
# is exactly what the indicator feeds.
#
# The card is replicated with the shipped defaults at `eeb6726` (Ra = 1e5,
# H = 10, p = 2, 5 x 2 base cells on 2.5 x 1, initial uniform level 3, composite
# viscosity, per-phase latent heat, `linear = :auto`, `mg_tol = 1e-8`,
# `stokes_tol = 1e-3`, `amr_every = 5`, `refine_frac = 0.2`,
# `coarsen_frac = 0.05`, `max_level = 7`, `max_leaves = 10000`,
# `wall_min_level = 5`, `wall_strip = 0.1`, coarsening gate on,
# `symmetry = :none`).  The only omissions are the render stream (it writes no
# state) and the opt-in `extdiag` residual column (a diagnostic; the audit's
# card used it, and the state is unchanged by it).
#
# The pinned values are the audit's measured corrected arm
# (docs/dev/BUG_AUDIT_INDICATOR.md §3.2), and every integer is accompanied by
# the historical value it replaces:
#
#   first cycle (step 5): corrected refines **7** leaves -> 2560 leaves,
#     N = 10853; the pre-fix body refined **28** -> 2644 leaves, N = 11233
#     (the spurious interior y-face mass pushed extra leaves over `tau_r`).
#   second cycle (step 10): refines 16 -> 2608 leaves, N = 11069 (pre-cycle
#     history row: 2560 / 10853).
#   step-10 history row: Nu_top = 5.999881326, Nu_bot = -4.029590902,
#     Vrms = 6.808272e+03.  The pre-fix arm's step-10 Vrms was 6.808435e+03
#     (2.44e-5 relative away), so the Vrms bound below is tight enough to fail on
#     the historical trajectory; the leaf counts fail it by construction
#     (2560 vs 2644, 10853 vs 11233).
#
# The scalars are pinned at rtol 1e-6 (Nu) / 5e-6 (Vrms): the trajectory is
# solved by the matrix-free MG whose thread count (`mg_threads =
# min(8, nthreads)`) changes the last bits of the solve, and the measured
# across-thread drift at step 10 is 4.4e-7 relative in Vrms (2-thread
# measurement 6.808272e+03 against the audit's 6-thread 6.808269e+03).  The
# integer pins are thread-independent (both measurements give the same counts),
# and they are the discriminating assertion.
using Test, QuaST, Printf

const G280_LX = 2.5

function g280_params()
    return Params(Ra=1.0e5, H=10.0,
                  eta_law=:composite, comp_rule=:isostress,
                  comp_Ad=1.6240183119e5, comp_As=2.6796419238e-1,
                  comp_Ed=12.0, comp_Es=17.0,
                  comp_Vd=0.0, comp_Vs=0.0, comp_n=3.0,
                  comp_Tcut=0.0, comp_eps_min=1.0e-12,
                  eta_T0=0.5, eta_min=1.0e-4, eta_max=1.0e4,
                  eta_ref=1.0, bc=:free,
                  wall_bc=(:same, :same, :same, :same),
                  wall_un=(nothing, nothing, nothing, nothing),
                  wall_t=(nothing, nothing, nothing, nothing),
                  wall_tn=(nothing, nothing, nothing, nothing),
                  latent_L=0.11, latent_mode=:per_phase,
                  shear_modulus=0.0,
                  yield_c=Inf, yield_phi=0.0, yield_taumax=Inf,
                  yield_c_grad=0.0, yield_mu_grad=0.0,
                  weak_f=1.0, weak_eps_ref=0.0,
                  rho_heating=false, rho_cp=false, Di=0.0,
                  heating_adiabatic=false, heating_dissipation=false,
                  refstate=nothing, deviatoric=false)
end

function g280_phases()
    ph = [PhaseTransition(0.87, 0.025, 0.032, 0.5, 180.0, 1.818),
          PhaseTransition(0.78, 0.025, -0.032, 0.5, 300.0, 3.030)]
    return per_phase_latent(Params(Ra=1.0e5, H=10.0), ph;
                            rho_ref=3300.0, cp=1000.0, dT=1000.0, g=9.81,
                            d=2.9e6, T_surface=0.5)
end

function g280_tree()
    qt = Quadtree(7; Lx=G280_LX, cells_x=5, cells_y=2)
    for _ in 1:3, lf in collect(qt.leaves)      # the card's initial level 3
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

g280_seed(x, y) = 0.05 * sinpi(y) * cospi(2.0 * x / G280_LX) +
                  0.02 * sinpi(2.0 * y) * cospi(4.0 * x / G280_LX)

@testset "G280 realistic-card AMR trajectory" begin
    qt = g280_tree()
    ss = make_semstokes(qt, 2)
    theta0 = [g280_seed(x, y) for (x, y) in ss.su.coords]
    pr = g280_params()
    ph = g280_phases()
    scale_set = (rho=3300.0, g=9.81, alpha=3.0e-5, dT=1000.0, d=2.9e6,
                 kappa=1.0e-6)
    cycles = NamedTuple[]
    lastrec = Ref{Any}(nothing)
    rec = function (st)
        # AMR runs every `amr_every = 5` steps, so only such a step can be a new
        # cycle record.  (LAST_AMR_BALANCED is a **process global**: in the full
        # suite a previous gate leaves its last cycle in it, and the first
        # version of this callback recorded that leftover at step 1.  Caught by
        # the suite run; the AMR cadence filter is the fix.)
        st.step % 5 == 0 || return nothing
        d = QuaST.LAST_AMR_BALANCED[]
        if d.refined + d.coarsened + d.merged > 0
            # Duplicates from the non-AMR steps of the same cycle cannot occur
            # behind the cadence filter, but keep the guard: a cycle that made
            # no decision is not a record.
            key = (d.refined, d.coarsened, d.merged, d.post, d.tau_r, d.tau_c,
                   length(st.qt.leaves), st.sq.N)
            if key != lastrec[]
                lastrec[] = key
                push!(cycles, (step=st.step, leaves=length(st.qt.leaves),
                               N=st.sq.N, refined=d.refined,
                               coarsened=d.coarsened, merged=d.merged))
            end
        end
        return nothing
    end
    res = run_semquad_simulation(qt, pr, theta0;
        p=2, tstop=Inf, maxsteps=10,
        dt0=1.0e-6, dtmax=5.0e-5, dtmin=1.0e-9, cfl=0.5, dt_growth=1.05,
        maxpic=1, pic_tol=1.0e-6, pic_mode=:picard,
        amr_every=5,
        refine_frac=0.2, coarsen_frac=0.05,
        max_leaves=10000, max_level=7, symmetry=:none,
        wall_min_level=5, wall_strip=0.1,
        amr_hysteresis=1, amr_indicator=:theta,
        amr_coarsen_gate=true,
        transfer_mode=:nodal,
        advection=:skew, ev=0.11, ev_cmax=0.052, ev_alpha=2,
        linear=:auto, stokes_tol=1.0e-3, mg_tol=1.0e-8,
        stokes_scaling=:none,
        mass_form=:boussinesq, phases=ph, latent_scheme=:implicit,
        eta_ref_scale=scale_set,
        callback=rec, verbose=false, profile_steps=false)

    @test res.steps == 10
    @test length(res.history) == 10
    # the trajectory up to the first cycle is bit-identical to the shipped card
    for k in 1:5
        @test res.history[k][8] == 640          # leaves
        @test res.history[k][7] == 2673         # N
    end
    # The first AMR cycle (end of step 5): the corrected body refines 7 leaves.
    # Historical (buggy indicator) value: 28 leaves -> 2644 leaves / N 11233.
    @test length(cycles) == 2
    @test cycles[1].step == 5
    @test cycles[1].refined == 7
    @test cycles[1].coarsened == 0
    @test cycles[1].merged == 0
    @test cycles[1].leaves == 2560
    @test cycles[1].N == 10853
    # the first post-cycle history row and the rest of the measured smoke table
    for k in 6:10
        @test res.history[k][8] == 2560         # leaves (pre-cycle row at k=10)
        @test res.history[k][7] == 10853        # N
    end
    # second AMR cycle (end of step 10): 16 refined -> 2608 leaves / N 11069
    @test length(cycles) == 2
    @test cycles[2].step == 10
    @test cycles[2].refined == 16
    @test cycles[2].leaves == 2608
    @test cycles[2].N == 11069
    # the returned tree is the post-second-cycle one; the last history row is the
    # pre-cycle state (the driver pushes the row before the AMR block)
    @test length(res.qt.leaves) == 2608
    @test res.ss.su.N == 11069
    # scalars of the step-10 history row.  Nu bounds are a regression pin; the
    # Vrms bound is tight enough to fail on the historical arm (2.44e-5 away)
    # while surviving the measured MG thread-count drift (4.4e-7).
    lastrow = res.history[end]
    @printf("G280 step10: leaves=%d N=%d Nu_top=%.9e Nu_bot=%.9e Vrms=%.9e\n",
            lastrow[8], lastrow[7], lastrow[3], lastrow[4], lastrow[5])
    @printf("G280 cycles: %s\n", [(c.step, c.refined, c.leaves, c.N)
                                  for c in cycles[1:2]])
    @test lastrow[3] ≈ 5.999881326 rtol = 1.0e-6
    @test lastrow[4] ≈ -4.029590902 rtol = 1.0e-6
    @test lastrow[5] ≈ 6.808272e+03 rtol = 5.0e-6
    @test maximum(abs, res.theta) < 1.0
    @test all(isfinite, res.theta)
    @test all(isfinite, res.pv)
    @test max_face_level_jump(res.qt) <= 1
end
