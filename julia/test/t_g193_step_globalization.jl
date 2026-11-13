# G193: globalization of the *per-time-step* nonlinear solve.
#
# The driver's step is the root of the coupled residual
#
#     r_S = Cu'(fv - K(eta(T, eps(u))) u - Bm'p)      momentum
#     r_C = gr - Bs uhat                             continuity
#     r_E = Cfree'(A_E(u) theta - rhs)               BDF2 / backward Euler
#
# with `A_E = (a/dt)M + K + C(u)` and `rhs` built from the **start-of-step**
# temperature `theta^n` (see `solve_semquad_energy`).  This gate checks:
#
#   (1) `pic_mode = :picard` (the default) is untouched: the explicit and the
#       default call are bit-identical, and so is the `pic_diag` instrumentation;
#   (2) the fixed point of the `maxpic > 1` loop is a genuine time step, i.e.
#       `theta^n` is used and not the running iterate (the historical bug: with
#       the iterate the mass terms cancel and the loop converges to the *steady*
#       energy equation);
#   (3) the analytic Jacobian of `semquad_step_newton` matches a central
#       difference of the residual (Kgeo + deta/dT wired in, rel <= 1e-6);
#   (4) on the strong composite case the Newton reaches `pic_tol` in fewer
#       iterations than Picard;
#   (5) both reach the *same* fixed point (to the per-step tolerance);
#   (6) when the Newton cannot solve a step it reports `converged == false` and
#       the driver falls back to Picard, so the run still completes;
#   (7) the trial-state rejection fires on a clamp-dominated step and its
#       relaxation is reduced (and it does not fire on a healthy one);
#   (8) the argument guards of `pic_mode = :newton`.
using Test, MantleSEM, LinearAlgebra

const G193_ED = 9.210340371976184          # ln(1e4): 2.15e5 thermal contrast
const G193_ES = 12.0
const G193_N = 3.0
const G193_T0 = 0.5

# `asc` scales the dislocation prefactor: 1e-2 keeps the viscosity away from the
# clamps (the Newton converges), 1.0 is the clamp-dominated, strongly weakening
# case in which the coupled linearisation cannot be solved (the fallback runs).
function g193_params(; asc=1.0e-2, Ra=1.0e6, eta_min=1.0e-6, eta_max=1.0e4)
    Params(; Ra=Ra, H=0.0, eta_law=:composite,
           comp_Ad=exp(G193_ED / (1.0 + G193_T0)), comp_Ed=G193_ED,
           comp_Vd=0.0,
           comp_As=asc * exp(G193_ES / (G193_N * (1.0 + G193_T0))),
           comp_Es=G193_ES, comp_Vs=0.0, comp_n=G193_N, eta_T0=G193_T0,
           eta_min=eta_min, eta_max=eta_max)
end

function g193_tree(nref::Int=3)
    qt = Quadtree(5)
    for _ in 1:nref, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    return qt
end

function g193_run(pr, th0; nsteps=3, dt=1.0e-5, kw...)
    qt = g193_tree()
    r = run_semquad_simulation(qt, pr, th0; p=2, tstop=nsteps * dt, dt0=dt,
                               dtmax=dt, dtmin=1.0e-16, amr_every=0, ev=0.0,
                               advection=:skew, linear=:direct, maxpic=1,
                               pic_tol=1.0e-6, pic_diag=true, verbose=false,
                               kw...)
    # SEMQUAD_PIC_DIAG is a module-level Ref overwritten by every run, so the
    # per-step record must be copied out before the next call.
    return (r=r, diag=copy(SEMQUAD_PIC_DIAG[]))
end

@testset "G193 per-step globalization" begin
    pr = g193_params()
    qt0 = g193_tree()
    ss0 = make_semstokes(qt0, 2)
    th0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss0.su.coords]

    # ---- (1) the default Picard path is untouched -------------------------
    r_def = g193_run(pr, th0).r
    r_pic = g193_run(pr, th0, pic_mode=:picard, pic_reject=false).r
    @test r_def.theta == r_pic.theta
    @test r_def.history == r_pic.history
    # the `pic_diag` counters must not change a single bit of the state
    r_nodiag = run_semquad_simulation(g193_tree(), pr, th0; p=2, tstop=3.0e-5,
                                      dt0=1.0e-5, dtmax=1.0e-5, dtmin=1.0e-16,
                                      amr_every=0, ev=0.0, advection=:skew,
                                      linear=:direct, maxpic=1,
                                      pic_tol=1.0e-6, verbose=false)
    @test r_nodiag.theta == r_def.theta
    @test r_nodiag.history == r_def.history

    # ---- (2) `maxpic > 1` converges to a genuine time step -----------------
    # A linear-diffusion-dominated case (constant viscosity, no advection heat):
    # the fixed point of the iterated step must satisfy the backward-Euler
    # equation with theta_n = theta^0.  The historical code used the *iterate*
    # as theta_n; then the mass term cancels and the iteration converges to the
    # steady energy equation instead, which this test would reject.
    prv = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0)
    qtv = g193_tree(2)
    ssv = make_semstokes(qtv, 2)
    Nv = ssv.su.N
    thv = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ssv.su.coords]
    dtv = 1.0e-3
    rv = run_semquad_simulation(qtv, prv, thv; p=2, tstop=dtv, dt0=dtv,
                                dtmax=dtv, dtmin=1.0e-18, amr_every=0,
                                ev=0.0, advection=:galerkin, linear=:direct,
                                maxpic=40, pic_tol=1.0e-13, pic_relax=1.0,
                                pic_diag=true, verbose=false)
    dv = SEMQUAD_PIC_DIAG[]
    @test last(dv[1].changes) < 1.0e-11     # the iteration really converged
    # one full step: theta^0 = thv (the run's initial state), theta^1 = rv.theta
    leaves_v = Set(rv.qt.leaves)
    uxv, uzv = rv.ux, rv.uz
    eta_v = (x, y) -> eta_at(Tbar(y, prv.H) +
                             MantleSEM._semquad_eval_at(ssv.su, rv.theta, x, y,
                                                        leaves_v), y, prv)[1]
    parts = solve_semquad_energy(ssv.su, rv.qt, prv, thv, thv, uxv, uzv, dtv,
                                 dtv, true; advection=:galerkin,
                                 eta_fun=eta_v, return_parts=true)
    res_be = norm(parts.Cfree' * (parts.A * rv.theta - parts.rhs)) /
             norm(parts.Cfree' * parts.rhs)
    @test res_be < 1.0e-8                 # the step is a real backward-Euler step
    # ... and the *steady* energy equation does NOT hold at that state (the
    # historical fixed point), so the check above is not vacuous
    _, _, Cst, Fst = assemble_semquad_energy(ssv.su, rv.qt, prv, uxv, uzv;
                                             advection=:galerkin,
                                             theta=rv.theta)
    res_st = norm(parts.Cfree' * (parts.K * rv.theta + Cst * rv.theta - Fst)) /
             norm(parts.Cfree' * Fst)
    @test res_st > 1.0e-3
    # ... and the step is a small change of theta^0 (a genuine time step),
    # whereas the steady solution (the historical fixed point) is far away
    @test norm(rv.theta - thv) < 0.5 * norm(thv)

    # ---- (3) the analytic per-step Jacobian vs central differences --------
    qj = g193_tree(2)
    ssj = make_semstokes(qj, 2)
    sqj = ssj.su
    Nj = sqj.N
    leaves_j = Set(qj.leaves)
    thj = [0.05 * sinpi(x) * sinpi(z) for (x, z) in sqj.coords]
    # Mechanical update (docs/RHEOLOGY.md §8.4): the `make_eta` contract gained
    # the pressure argument for the Drucker-Prager yield stress; this gate's
    # composite law does not use it.
    make_eta_j = function (theta, ufull, _pfull=nothing)
        uxv = ufull[1:Nj]; uzv = ufull[(Nj + 1):(2Nj)]
        T_at(x, y) = Tbar(y, pr.H) + MantleSEM._semquad_eval_at(sqj, theta, x,
                                                               y, leaves_j)
        ed_at(x, y) = semquad_strain_rate_at(sqj, uxv, uzv, x, y, leaves_j)
        ((x, y) -> eta_composite_at(T_at(x, y), ed_at(x, y), y, pr)[1],
         (x, y) -> deta_composite_at(T_at(x, y), ed_at(x, y), y, pr)[1],
         (x, y, ed) -> deta_composite_at(T_at(x, y), ed, y, pr)[2])
    end
    z2 = (x, y) -> 0.0
    # a state away from the root (three lagged Picard solves)
    ufull = zeros(2Nj)
    for _ in 1:3
        ef, _, _ = make_eta_j(thj, ufull)
        u, _, _, _, _ = solve_semquad_stokes_var(
            ssj, qj, ef, z2,
            (x, y) -> pr.Ra * MantleSEM._semquad_eval_at(sqj, thj, x, y, leaves_j),
            z2, z2; bc=pr.bc, linear=:direct)
        ufull = 0.7 * ufull + 0.3 * u
    end
    pfull = zeros(ssj.sp.N)
    ctxj = MantleSEM._semquad_step_residual_ctx(ssj, qj, pr, thj, thj, thj,
                                               ufull, pfull, make_eta_j,
                                               leaves_j; dt=1.0e-5,
                                               dt_prev=1.0e-5,
                                               firststep=false)
    Mvj = assemble_semquad_pressure_mass(sqj)
    nfj, nfpj, ntj = ctxj.nf, ctxj.nfp, ctxj.nt
    resid_of = function (x)
        thx = thj + ctxj.parts.Cfree * x[(nfj + nfpj + 1):end]
        ufx = ufull + ctxj.sys.Cu * x[1:nfj]
        pfx = pfull + ctxj.sys.Cp * x[(nfj + 1):(nfj + nfpj)]
        c = MantleSEM._semquad_step_residual_ctx(ssj, qj, pr, thj, thj, thx, ufx,
                                                pfx, make_eta_j, leaves_j;
                                                dt=1.0e-5, dt_prev=1.0e-5,
                                                firststep=false)
        return MantleSEM.semquad_step_residual(c, thx)
    end
    x = [sin(0.7 * k) for k in 1:(nfj + nfpj + ntj)]
    x ./= norm(x)
    Jx = MantleSEM.semquad_step_jvp(ctxj, thj, ufull, x, Mvj; N=Nj, sq=sqj,
                                   q=ssj.q, pr=pr)
    for h in (1.0e-6, 1.0e-7)
        fd = (resid_of(h * x) - resid_of(-h * x)) / (2h)
        @test norm(Jx - fd) / norm(Jx) <= 1.0e-6
    end

    # ---- (4)-(5) Newton vs Picard on the strong composite case ------------
    rp_res = g193_run(pr, th0; maxpic=20, pic_relax=1.0)
    rn_res = g193_run(pr, th0; pic_mode=:newton, newton_maxit=10)
    r_p, dp = rp_res.r, rp_res.diag
    r_n, dn = rn_res.r, rn_res.diag
    @test length(dp) == length(dn) == 3
    @test all(row -> last(row.changes) < 1.0e-6, dp)   # Picard reaches pic_tol
    @test all(row -> last(row.changes) < 1.0e-6, dn)   # Newton reaches pic_tol
    @test all(row -> row.mode === :newton, dn)
    @test sum(row.iters for row in dn) < sum(row.iters for row in dp)
    # fewer iterations at every step
    @test all(dn[k].iters < dp[k].iters for k in eachindex(dp))
    # the Newton's residual history is (super)linear: the last change is far
    # below the previous one
    @test dn[1].iters <= 5
    # both methods converge to the same fixed point to the per-step tolerance
    @test norm(r_p.theta - r_n.theta) / norm(r_p.theta) < 1.0e-5
    @test r_p.history[end][3] ≈ r_n.history[end][3] rtol = 1.0e-8
    @test all(isfinite, r_n.theta)

    # ---- (6) fallback: the clamp-dominated case ---------------------------
    prs = g193_params(asc=1.0)
    rv_res = g193_run(prs, th0; pic_mode=:newton, maxpic=1, newton_maxit=6)
    r_v, dv = rv_res.r, rv_res.diag
    @test any(row -> row.fallback, dv)      # the Newton gave up
    @test all(row -> row.mode === :picard, dv)   # ... and Picard produced the step
    @test all(isfinite, r_v.theta) && all(isfinite, vcat(r_v.history...))

    # ---- (7) trial-state rejection ---------------------------------------
    # On the clamp-dominated case with several iterations the eta field hits
    # `eta_max` over a growing fraction and the rejection must fire; on the
    # well-behaved case it must not.
    r_rj, drj = let o = g193_run(prs, th0; maxpic=12, pic_relax=1.0,
                                 pic_reject=true, pic_reject_clamp=0.05,
                                 pic_reject_res=3.0, pic_reject_max=6)
        (o.r, o.diag)
    end
    @test any(row -> row.rejects > 0, drj)
    @test all(row -> row.relax < 1.0, filter(row -> row.rejects > 0, drj))
    @test all(isfinite, r_rj.theta)
    r_no, dno = let o = g193_run(pr, th0; maxpic=12, pic_relax=1.0,
                                 pic_reject=true)
        (o.r, o.diag)
    end
    @test all(row -> row.rejects == 0, dno)
    @test r_no.theta == g193_run(pr, th0; maxpic=12, pic_relax=1.0).r.theta

    # ---- (8) guards -------------------------------------------------------
    @test_throws ArgumentError g193_run(pr, th0; pic_mode=:newton,
                                        linear=:schur_cg)
    # The Drucker-Prager plastic branch used to be rejected here; it is now
    # linearised exactly (Kp block and the d eta/d chi block,
    # docs/RHEOLOGY.md §8.4/§9.4) and covered by t_g195/t_g196.  The guard
    # checked here is the CBF one, which the Newton still does not linearise.
    @test_throws ArgumentError g193_run(pr, th0; pic_mode=:newton, nu_cbf=true)
    @test_throws ArgumentError g193_run(pr, th0; pic_mode=:bogus)
    @test_throws ArgumentError g193_run(pr, th0; pic_reject_clamp=1.5)
    @test_throws ArgumentError g193_run(pr, th0; pic_reject_res=0.5)
end
