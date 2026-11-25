# G196: strain weakening.
#
#     dchi/dt + u.grad chi = (1-chi)*eps_pl/eps_ref - chi/tau_heal   (:rate)
#     dchi/dt + u.grad chi = eps_pl/eps_ref                          (:linear)
#     f(chi) = 1 - (1-weak_f)*clamp(chi,0,1),   tau_y <- f*tau_y
#
# This gate checks:
#   (1) f is monotone, bounded in [weak_f, 1], and is the factor the yield
#       stress, the plastic viscosity and deta/dchi actually use;
#   (2) the 1-D analytic steady state and transient of the transport (uniform
#       shear, no advection of a uniform field): exact steady state and
#       dt-convergence to the analytic transient, plus the `:linear` ramp;
#   (3) the per-step Newton Jacobian with the chi block (d r_S/dchi, d r_W/dchi,
#       d r_W/du) against central differences, and the d r_S/dchi non-vacuity;
#   (4) a 2-D time-stepping case: stable, bounded, checkpointed and bitwise
#       restartable; Picard vs Newton agreement in the weak-coupling regime and
#       the measured non-uniqueness at strong coupling (recorded, not hidden);
#   (5) weakening-off is bitwise identical to the pre-weakening default;
#   (6) parameter guards.
using Test, QuaST, LinearAlgebra, Statistics, Printf

# ---- (1) the weakening factor -------------------------------------------
@testset "G196 strain weakening" begin
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:const, eta_const=1.0,
                yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                yield_eps_min=1.0e-8, eta_min=1.0e-4, eta_max=10.0,
                bc=:free, weak_f=0.2, weak_eps_ref=0.5)
    @test weak_active(pr)
    @test !weak_active(Params(yield_c=1.0, yield_phi=0.3, eta_min=1e-4))
    # monotone non-increasing, bounded, endpoints
    chis = collect(-0.5:0.05:1.5)
    fs = [semquad_weak_factor(c, pr) for c in chis]
    @test all(fs .<= 1.0 .+ 1e-15) && all(fs .>= 0.2 - 1e-15)
    @test all(diff(fs) .<= 0.0)
    @test semquad_weak_factor(0.0, pr) == 1.0
    @test semquad_weak_factor(1.0, pr) == 1.0 - (1.0 - 0.2)
    @test semquad_weak_factor(-10.0, pr) == 1.0
    @test semquad_weak_factor(10.0, pr) == 1.0 - (1.0 - 0.2)
    # the yield stress is f * the unweakened stress (the one source of truth)
    y, p = 0.4, 0.7
    tau0 = semquad_yield_stress(p, y, pr)
    for chi in (0.0, 0.3, 0.75, 1.0)
        f = 1.0 - (1.0 - pr.weak_f) * chi
        @test semquad_yield_stress(p, y, chi, pr) ≈ min(f * tau0, pr.yield_taumax) rtol=1e-14
    end
    # ... and the plastic viscosity / deta_dchi use it: FD in chi
    T, ed = 0.5, 0.8
    for chi in (0.1, 0.5)
        h = 1.0e-6
        fd = (semquad_visco_plastic_eta(T, y, ed, p, chi + h, pr) -
              semquad_visco_plastic_eta(T, y, ed, p, chi - h, pr)) / (2h)
        an = semquad_deta_chi(T, y, ed, p, chi, pr)
        @test an < 0.0                       # weakening lowers the viscosity
        @test abs(an - fd) <= 1.0e-6 * abs(fd) + 1.0e-12
    end

    # ---- (2) 1-D uniform-shear transport --------------------------------
    # u = (y, 0) -> edot = 0.5 everywhere; a uniform chi has u.grad chi = 0,
    # so the discrete transport reduces to the ODE at every node.
    q = Quadtree(2; cells_x=2, cells_y=2)
    for _ in 1:1, lf in collect(q.leaves); refine!(q, lf); end
    balance!(q)
    ss = make_semstokes(q, 2)
    sq = ss.su; sp = ss.sp
    ux = [y for (x, y) in sq.coords]
    uz = zeros(sq.N)
    th = zeros(sq.N)
    pv = zeros(sp.N)
    zp = zeros(sq.N)
    # constant-viscosity law so the yielded branch is eta_v = 1 everywhere
    pr1 = Params(Ra=1.0e4, H=0.0, eta_law=:const, eta_const=1.0,
                 yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                 yield_eps_min=1.0e-8, yield_damper=0.0,
                 eta_min=1.0e-4, eta_max=10.0, bc=:free,
                 weak_f=0.2, weak_eps_ref=0.02, weak_heal_time=0.5,
                 weak_mode=:rate)
    g = 0.5 / pr1.weak_eps_ref
    lam = g + 1.0 / pr1.weak_heal_time
    chi_ss = g / lam
    chi = copy(zp)
    chi_prev = copy(zp)
    dt = 1.0e-3
    nstep = 600              # t = 0.6 = 16 healing/weakening times
    for k in 1:nstep
        chi_new = solve_semquad_weakening(sq, sp, q, pr1, chi, chi_prev, th,
                                          ux, uz, pv, dt, dt, k == 1;
                                          advection=:galerkin, chi_src=chi)
        chi_prev = chi
        chi = chi_new
        t = k * dt
        ana = chi_ss * (1.0 - exp(-lam * t))
        if k in (1, 2, 10, 40, nstep)
            @test maximum(abs.(chi .- ana)) <= 5.0e-3
            @test maximum(abs.(chi .- chi[1])) < 1.0e-10   # spatially uniform
        end
    end
    @test maximum(abs.(chi .- chi_ss)) < 1.0e-6   # the analytic steady state
    # (lambda*t = 16.2, i.e. the residual is chi_ss*exp(-16.2) = 8.5e-8)
    # dt halving reduces the transient error (first order: the first step is BE)
    function g196_transport_err(dt)
        c = zeros(length(zp)); prev = copy(c)
        n = round(Int, 0.02 / dt)
        for k in 1:n
            cnew = solve_semquad_weakening(sq, sp, q, pr1, c, prev, th, ux, uz,
                                           pv, dt, dt, k == 1; advection=:galerkin,
                                           chi_src=c)
            prev = c
            c = cnew
        end
        return maximum(abs.(c .- chi_ss * (1.0 - exp(-lam * n * dt))))
    end
    e1 = g196_transport_err(1.0e-3)
    e2 = g196_transport_err(5.0e-4)
    @test e1 > 0.0
    @test e1 / e2 > 1.7            # ~2 for a first-order start, not a plateau
    # :linear accumulation: chi = g*t exactly for a constant source (BDF2 is
    # exact for a linear-in-t solution once the first BE step is exact)
    prl = Params(Ra=1.0e4, H=0.0, eta_law=:const, eta_const=1.0,
                 yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                 yield_eps_min=1.0e-8, yield_damper=0.0,
                 eta_min=1.0e-4, eta_max=10.0, bc=:free,
                 weak_f=0.2, weak_eps_ref=0.02, weak_mode=:linear)
    cl = copy(zp); prevl = copy(zp)
    for k in 1:6
        cnew = solve_semquad_weakening(sq, sp, q, prl, cl, prevl, th, ux, uz,
                                       pv, dt, dt, k == 1; advection=:galerkin,
                                       chi_src=cl)
        prevl = cl
        cl = cnew
        @test maximum(abs.(cl .- g * k * dt)) <= 1.0e-10
    end

    # ---- (3) per-step Newton Jacobian with the chi block ---------------
    prn = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                 yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                 yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                 eta_min=1.0e-4, eta_max=10.0, bc=:free,
                 weak_f=0.4, weak_eps_ref=0.3, weak_mode=:rate)
    qj = Quadtree(2; cells_x=2, cells_y=2)
    for _ in 1:1, lf in collect(qj.leaves); refine!(qj, lf); end
    balance!(qj)
    ssj = make_semstokes(qj, 2)
    sqj = ssj.su; spj = ssj.sp
    Nj = sqj.N
    leaves_j = Set(qj.leaves)
    thj = [0.05 * cospi(x) * sinpi(z) for (x, z) in sqj.coords]
    for (i, (x, z)) in enumerate(sqj.coords)
        (z <= 1.0e-12 || z >= 1.0 - 1.0e-12) && (thj[i] = 0.0)
    end
    make_eta_j = function (theta, ufull, pfull, chi)
        uxv = ufull[1:Nj]; uzv = ufull[(Nj + 1):(2Nj)]
        T_at(x, y) = Tbar(y, prn.H) +
                     QuaST._semquad_eval_at(sqj, theta, x, y, leaves_j)
        ed_at(x, y) = semquad_strain_rate_at(sqj, uxv, uzv, x, y, leaves_j)
        p_at(x, y) = QuaST._semquad_eval_at(spj, pfull, x, y,
                                               Set(spj.leaves))
        c_at(x, y) = QuaST._semquad_eval_at(sqj, chi, x, y, leaves_j)
        ((x, y) -> semquad_visco_plastic_eta(T_at(x, y), y, ed_at(x, y),
                                            p_at(x, y), c_at(x, y), prn),
         (x, y) -> semquad_dp_deta_T(T_at(x, y), y, ed_at(x, y), p_at(x, y),
                                     c_at(x, y), prn),
         (x, y, ed) -> semquad_deta_dp_eps(T_at(x, y), y, ed, p_at(x, y),
                                           c_at(x, y), prn),
         (x, y, ed, pq) -> semquad_deta_dp(T_at(x, y), y, ed, pq,
                                           c_at(x, y), prn),
         (x, y, ed) -> semquad_deta_chi(T_at(x, y), y, ed, p_at(x, y),
                                        c_at(x, y), prn))
    end
    z2 = (x, y) -> 0.0
    ufull = zeros(2Nj); pfull = zeros(spj.N)
    chi_state = zeros(Nj)
    for _ in 1:3
        ef = make_eta_j(thj, ufull, pfull, chi_state)
        u, p, _, _, _ = solve_semquad_stokes_var(
            ssj, qj, ef[1], z2,
            (x, y) -> prn.Ra * QuaST._semquad_eval_at(sqj, thj, x, y, leaves_j),
            z2, z2; bc=prn.bc, linear=:direct)
        ufull = 0.7 * ufull + 0.3 * u
        pfull = 0.7 * pfull + 0.3 * p
    end
    # a non-trivial chi state so d r_S/d chi != 0
    chi_state = [0.4 * (1.0 - y) for (x, y) in sqj.coords]
    ctx = QuaST._semquad_step_residual_ctx(ssj, qj, prn, thj, thj, thj,
                                              ufull, pfull, make_eta_j,
                                              leaves_j; dt=1.0e-5,
                                              dt_prev=1.0e-5, firststep=false,
                                              chi=chi_state, chi_n=chi_state,
                                              chi_nm1=chi_state)
    @test ctx.weak && ctx.Kchi_red !== nothing
    @test size(ctx.Kchi_red) == (ctx.nf, ctx.nw)
    Mv = assemble_semquad_pressure_mass(sqj)
    nf, nfp, nt, nw = ctx.nf, ctx.nfp, ctx.nt, ctx.nw
    resid_of = function (x)
        thx = thj + ctx.parts.Cfree * x[(nf + nfp + 1):(nf + nfp + nt)]
        ufx = ufull + ctx.sys.Cu * x[1:nf]
        pfx = pfull + ctx.sys.Cp * x[(nf + 1):(nf + nfp)]
        chx = chi_state + ctx.partsW.Cfree * x[(nf + nfp + nt + 1):end]
        c = QuaST._semquad_step_residual_ctx(ssj, qj, prn, thj, thj, thx,
                                                ufx, pfx, make_eta_j, leaves_j;
                                                dt=1.0e-5, dt_prev=1.0e-5,
                                                firststep=false, chi=chx,
                                                chi_n=chi_state,
                                                chi_nm1=chi_state)
        return vcat(c.rS, c.rC, c.rE, c.rW)
    end
    x = [sin(0.7 * k) for k in 1:(nf + nfp + nt + nw)]
    x ./= norm(x)
    Jx = QuaST.semquad_step_jvp(ctx, thj, ufull, x, Mv; N=Nj, sq=sqj,
                                    q=ssj.q, pr=prn)
    @test length(Jx) == nf + nfp + nt + nw
    # The only inexact term is the derivative of the plastic-yielding indicator
    # that selects eps_pl in the chi source: it is zero a.e., but a finite
    # difference perturbation can cross the yield surface.  Measured 7.1e-5;
    # every block that is differentiable is exact to ~1e-10.
    for h in (1.0e-6, 1.0e-7)
        fd = (resid_of(h * x) - resid_of(-h * x)) / (2h)
        @test norm(Jx - fd) / norm(Jx) <= 1.0e-4
    end
    # the d r_S/d chi block is not vacuous: a chi-only direction
    xc = zeros(nf + nfp + nt + nw)
    xc[(nf + nfp + nt + 1):end] = ctx.partsW.Cfree' * chi_state
    xc[(nf + nfp + nt + 1):end] ./= norm(xc[(nf + nfp + nt + 1):end])
    Jc = QuaST.semquad_step_jvp(ctx, thj, ufull, xc, Mv; N=Nj, sq=sqj,
                                    q=ssj.q, pr=prn)
    fd = (resid_of(1.0e-7 * xc) - resid_of(-1.0e-7 * xc)) / 2.0e-7
    @test norm(Jc - fd) / norm(Jc) <= 1.0e-4
    ctxn = merge(ctx, (Kchi_red=nothing,))
    Jc_no = QuaST.semquad_step_jvp(ctxn, thj, ufull, xc, Mv; N=Nj, sq=sqj,
                                       q=ssj.q, pr=prn)
    # the missing block shows up in the *momentum rows*, not in the (larger)
    # chi-equation rows
    @test norm((Jc - Jc_no)[1:nf]) / max(norm(Jc[1:nf]), 1e-30) > 1.0e-4
    @printf("G196: d r_S/d chi share of the momentum rows = %.3e\n",
            norm((Jc - Jc_no)[1:nf]) / max(norm(Jc[1:nf]), 1e-30))

    # ---- (4) 2-D time stepping, checkpoint and restart ------------------
    function g196_tree(nref=2)
        qq = Quadtree(2; cells_x=2, cells_y=2)
        for _ in 1:nref, lf in collect(qq.leaves); refine!(qq, lf); end
        balance!(qq)
        return qq
    end
    pr2 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                 yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                 yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                 eta_min=1.0e-4, eta_max=10.0, bc=:free,
                 weak_f=0.3, weak_eps_ref=0.5, weak_mode=:rate)
    function g196_run(pru; nsteps=30, mode=:picard, maxpic=1, tol=1e-6, kw...)
        qq = g196_tree()
        ss2 = make_semstokes(qq, 2)
        th0 = [0.05 * cospi(x) * sinpi(y) for (x, y) in ss2.su.coords]
        for (i, (x, y)) in enumerate(ss2.su.coords)
            (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
        end
        kk = mode === :newton ? (pic_mode=:newton, newton_maxit=15) :
             (maxpic=maxpic,)
        # Restart gate; the balanced cycle is the only AMR policy.
        r = run_semquad_simulation(qq, pru, th0; p=2, tstop=10.0, dt0=1.0e-6,
                                   dtmax=1.0e-4, cfl=2.0, maxsteps=nsteps,
                                   verbose=false, ev=0.0, advection=:skew,
                                   linear=:direct, pic_diag=true, pic_tol=tol,
                                   kw..., kk...)
        (r=r, diag=copy(SEMQUAD_PIC_DIAG[]))
    end
    out = g196_run(pr2; nsteps=30)
    r = out.r
    @test r.steps == 30 && r.weak
    @test all(isfinite, r.chi)
    @test minimum(r.chi) >= 0.0 && maximum(r.chi) <= 1.0
    @test maximum(r.chi) > 0.0                     # the field actually evolved
    @test all(isfinite, vcat(r.history...))
    st = semquad_weak_stats(r.chi)
    @printf("G196: 2-D 30 steps: chi[min=%.4f max=%.4f mean=%.4f] Vrms=%.4e Nu=%.6f\n",
            st.min, st.max, st.mean,
            semquad_vrms(r.ss.su, r.qt, r.ux, r.uz), r.history[end][3])
    # Newton with weakening: converges and is finite
    outn = g196_run(pr2; nsteps=10, mode=:newton)
    @test all(row -> row.mode === :newton, outn.diag)
    @test all(row -> last(row.changes) < 1.0e-6, outn.diag)
    @test all(isfinite, outn.r.chi)
    @printf("G196: Newton 10 steps iters=%s\n", [d.iters for d in outn.diag])
    # weak coupling: Picard and Newton agree
    prw = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                 yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                 yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                 eta_min=1.0e-4, eta_max=10.0, bc=:free,
                 weak_f=0.9, weak_eps_ref=0.5, weak_mode=:rate)
    rpw = g196_run(prw; nsteps=5, maxpic=40, tol=1.0e-10)
    rnw = g196_run(prw; nsteps=5, mode=:newton, tol=1.0e-10)
    @test norm(rpw.r.theta - rnw.r.theta) / norm(rnw.r.theta) < 1.0e-3
    @test norm(rpw.r.chi - clamp.(rnw.r.chi, 0.0, 1.0)) / norm(rpw.r.chi) < 5.0e-3
    @printf("G196: weak coupling Picard-vs-Newton |dtheta|=%.2e |dchi|=%.2e\n",
            norm(rpw.r.theta - rnw.r.theta) / norm(rnw.r.theta),
            norm(rpw.r.chi - clamp.(rnw.r.chi, 0.0, 1.0)) / norm(rpw.r.chi))
    # strong coupling: the two iterations can reach different roots of the
    # nonsmooth (yielding-indicator) system; record, do not hide
    rps = g196_run(pr2; nsteps=5, maxpic=40, tol=1.0e-10)
    rns = g196_run(pr2; nsteps=5, mode=:newton, tol=1.0e-10)
    dchi_s = norm(rps.r.chi - clamp.(rns.r.chi, 0.0, 1.0)) /
             max(norm(rps.r.chi), 1e-30)
    @printf("G196: strong coupling |dchi|/chi=%.2e (nonsmooth non-uniqueness)\n",
            dchi_s)
    @test dchi_s < 0.5
    # checkpoint carries the field, restart is bit-exact
    ck = joinpath(mktempdir(), "g196_ck.h5")
    ra = g196_run(pr2; nsteps=3, checkpoint_path=ck, checkpoint_every=3,
                  checkpoint_meta=Dict{String,Any}())
    ld = load_semquad_state(ck)
    @test ld.weak !== nothing && ld.weak_prev !== nothing
    @test length(ld.weak) == length(ld.theta)
    @test maximum(abs.(ld.weak .- ra.r.chi)) == 0.0
    rb = g196_run(pr2; nsteps=6, restart_from=ck)
    rc = g196_run(pr2; nsteps=6)
    @test norm(rb.r.theta - rc.r.theta) == 0.0
    @test norm(rb.r.chi - rc.r.chi) == 0.0
    @printf("G196: restart bitwise: |dtheta|=%.1e |dchi|=%.1e\n",
            norm(rb.r.theta - rc.r.theta), norm(rb.r.chi - rc.r.chi))
    # AMR with weakening runs and keeps chi bounded
    ramr = g196_run(pr2; nsteps=12, amr_every=4, refine_frac=0.2)
    @test all(isfinite, ramr.r.chi)
    @test minimum(ramr.r.chi) >= 0.0 && maximum(ramr.r.chi) <= 1.0
    @printf("G196: AMR 12 steps N=%d leaves=%d chi max=%.4f\n", ramr.r.ss.su.N,
            length(ramr.r.qt.leaves), maximum(ramr.r.chi))

    # ---- (5) weakening off is the historical default ---------------------
    pr_off1 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                     yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                     yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                     eta_min=1.0e-4, eta_max=10.0, bc=:free)
    pr_off2 = Params(; Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                     yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                     yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                     eta_min=1.0e-4, eta_max=10.0, bc=:free,
                     weak_f=0.3, weak_eps_ref=0.0)          # eps_ref = 0: off
    pr_off3 = Params(; Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                     yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                     yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                     eta_min=1.0e-4, eta_max=10.0, bc=:free,
                     weak_f=1.0, weak_eps_ref=0.5)          # weak_f = 1: off
    r1 = g196_run(pr_off1; nsteps=4).r
    r2 = g196_run(pr_off2; nsteps=4).r
    r3 = g196_run(pr_off3; nsteps=4).r
    @test r2.weak == false && r3.weak == false
    @test r1.theta == r2.theta == r3.theta
    @test r1.ux == r2.ux == r3.ux
    @test r1.uz == r2.uz == r3.uz
    @test r1.pv == r2.pv == r3.pv
    @test r1.history == r2.history == r3.history
    @test r2.chi === nothing

    # ---- (6) guards ------------------------------------------------------
    @test_throws ArgumentError Params(yield_c=1.0, yield_phi=0.3,
                                      weak_f=1.5, weak_eps_ref=1.0,
                                      eta_min=1.0e-4)
    @test_throws ArgumentError Params(yield_c=1.0, yield_phi=0.3,
                                      weak_f=0.2, weak_eps_ref=-1.0,
                                      eta_min=1.0e-4)
    @test_throws ArgumentError Params(yield_c=1.0, yield_phi=0.3,
                                      weak_f=0.2, weak_eps_ref=1.0,
                                      weak_heal_time=0.0, eta_min=1.0e-4)
    @test_throws ArgumentError Params(yield_c=1.0, yield_phi=0.3,
                                      weak_f=0.2, weak_eps_ref=1.0,
                                      weak_mode=:bogus, eta_min=1.0e-4)
    @test_throws ArgumentError Params(eta_law=:const, eta_const=1.0,
                                      weak_f=0.2, weak_eps_ref=1.0,
                                      eta_min=1.0e-4)
    qq = g196_tree()
    ssq = make_semstokes(qq, 2)
    thq = zeros(ssq.su.N)
    @test_throws ArgumentError run_semquad_simulation(qq, pr2, thq;
                                                      weak_chi0=2.0)
end
