# G194: latent heat of the phase transitions in the time-stepping energy
# equation (apparent heat capacity, docs/RHEOLOGY.md §7).
#
# The time step now carries the phase transitions in the *energy* equation, not
# only in the buoyancy:
#
#     (1 + latent_L*Gamma_T) (dtheta/dt + u.grad theta + u_z*dTbar/dz)
#         + latent_L*Gamma_z*u_z = lap(theta)
#
# (`Gamma` = `phase_gamma` of `phase.jl`, i.e. the same phase function
# `semquad_phase_buoyancy` uses).  This gate checks:
#
#   (1) the phase derivatives (dGamma/dT, dGamma/dz, d2Gamma/dT2, d2Gamma/dzdT)
#       against central differences, and that both the buoyancy and the latent
#       coefficient consume those shared functions;
#   (2) the apparent-heat-capacity operator equals the explicit-source operator
#       (the algebraic form of the same equation) and the latent assembly
#       matches an independent quadrature;
#   (3) the default path is untouched: `phases` empty or `latent_L = 0` is
#       bit-identical, and `latent_scheme` does not change a run without phases;
#   (4) the per-step Newton Jacobian with phase buoyancy + latent heat matches
#       central differences of the residual (including the latent tangent);
#   (5) the energy budget of one step closes against the consistent boundary
#       reaction, and the multi-step budget against the physical wall flux
#       closes at the same order as the latent-free scheme;
#   (6) a 1-D conductive column reaches the analytic steady state T = 1-y
#       independently of latent_L, with the latent enthalpy active in the
#       transient; a negative apparent heat capacity is rejected;
#   (7) a composite-rheology case with a phase transition runs several steps
#       with Picard (maxpic > 1) and with the per-step Newton, both converge and
#       reach the same state.
using Test, MantleSEM, LinearAlgebra, Printf

const G194_PT = PhaseTransition(0.6, 0.1, 0.05, 0.5, 0.1, 1.0)
const G194_ED = 9.210340371976184          # ln(1e4)
const G194_ES = 12.0
const G194_N = 3.0
const G194_T0 = 0.5

function g194_tree(nref::Int=3; maxlevel::Int=5)
    qt = Quadtree(maxlevel)
    for _ in 1:nref, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    return qt
end

function g194_sin_state(qt; amp=0.05)
    ss = make_semstokes(qt, 2)
    return [amp * sinpi(x) * sinpi(y) for (x, y) in ss.su.coords], ss
end

function g194_composite_params(; latent_L=2.5, Ra=1.0e6)
    return Params(; Ra=Ra, H=0.0, eta_law=:composite,
                  comp_Ad=exp(G194_ED / (1.0 + G194_T0)), comp_Ed=G194_ED,
                  comp_Vd=0.0,
                  comp_As=1.0e-2 * exp(G194_ES / (G194_N * (1.0 + G194_T0))),
                  comp_Es=G194_ES, comp_Vs=0.0, comp_n=G194_N,
                  eta_T0=G194_T0, eta_min=1.0e-6, eta_max=1.0e4,
                  latent_L=latent_L)
end

# One-step column solve; returns the exact enthalpy rate, the consistent
# boundary reaction and the physical wall flux of the step.
function g194_onestep(; dt=1.0e-3, lat=2.5, phases=[G194_PT], nref=3)
    pr = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, latent_L=lat)
    qt = g194_tree(nref)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    th0 = [1.0e-3 * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end
    ux = zeros(N); uz = zeros(N)
    parts = solve_semquad_energy(sq, qt, pr, th0, th0, ux, uz, dt, dt, true;
                                 advection=:galerkin, phases=phases,
                                 latent_theta=th0, return_parts=true)
    th1 = parts.u0 + parts.Cfree * (parts.Ar \ (parts.Cfree' *
                                               (parts.rhs - parts.A * parts.u0)))
    r = parts.A * th1 - parts.rhs
    # the consistent boundary exchange is the full residual summed over the
    # Dirichlet dofs (its free-dof residual is zero to round-off)
    free = findall(!=(0.0), diag(parts.Cfree * parts.Cfree'))
    Qc = sum(r) - sum(r[free])
    b0 = semquad_heat_budget(sq, qt, pr, th0; phases=phases, ux=ux, uz=uz)
    b1 = semquad_heat_budget(sq, qt, pr, th1; phases=phases, ux=ux, uz=uz)
    dE = b1.enthalpy - b0.enthalpy
    Q_phys = 0.5 * (b0.flux + b1.flux)
    return (dE=dE, Qc=Qc, Q_phys=Q_phys, dt=dt,
            cons=(dE / dt - Qc) / abs(Qc),
            phys=(dE / dt - Q_phys) / abs(Q_phys),
            ares=norm(parts.Cfree' * r) / norm(parts.Cfree' * parts.rhs),
            th0=th0, th1=th1, b0=b0, b1=b1)
end

# Multi-step column run through the driver; returns the per-step budget records.
function g194_column(; nsteps=6, dt=1.0e-3, lat=2.5, phases_on=true)
    qt = g194_tree(3)
    ss = make_semstokes(qt, 2)
    th0 = [1.0e-3 * sinpi(y) for (x, y) in ss.su.coords]
    for (i, (x, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end
    ph = phases_on ? [G194_PT] : PhaseTransition[]
    pr = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, latent_L=lat)
    rec = Any[]
    cb = st -> push!(rec, (dt=st.dt,
                           b=semquad_heat_budget(st.sq, st.qt, pr, st.theta;
                                                 phases=ph, ux=st.ux, uz=st.uz)))
    r = run_semquad_simulation(qt, pr, th0; p=2, tstop=nsteps * dt, dt0=dt,
                               dtmax=dt, dtmin=1.0e-18, amr_every=0, ev=0.0,
                               advection=:galerkin, linear=:direct, maxpic=1,
                               pic_tol=1.0e-10, phases=ph, callback=cb,
                               verbose=false)
    return (r=r, rec=rec)
end

function g194_budget_worst(rec)
    worst = 0.0
    for k in 2:length(rec)
        r0, r1 = rec[k - 1], rec[k]
        dE = r1.b.enthalpy - r0.b.enthalpy
        Qm = 0.5 * (r0.b.flux + r1.b.flux)
        res = dE - Qm * r1.dt
        worst = max(worst, abs(res) / max(abs(Qm) * r1.dt, 1.0e-300))
    end
    return worst
end

function g194_smoke(; nsteps=5, mode=:picard, maxpic=8, newton_maxit=10,
                    latent_scheme=:implicit, phases=[G194_PT], latent_L=2.5)
    qt = g194_tree(3)
    ss = make_semstokes(qt, 2)
    th0 = [0.05 * sinpi(x) * sinpi(y) for (x, y) in ss.su.coords]
    pr = g194_composite_params(latent_L=latent_L)
    r = run_semquad_simulation(qt, pr, th0; p=2, tstop=nsteps * 1.0e-5,
                               dt0=1.0e-5, dtmax=1.0e-5, dtmin=1.0e-16,
                               amr_every=0, ev=0.0, advection=:galerkin,
                               linear=:direct, maxpic=maxpic, pic_tol=1.0e-6,
                               pic_diag=true, pic_mode=mode,
                               newton_maxit=newton_maxit, phases=phases,
                               latent_scheme=latent_scheme, verbose=false)
    return (r=r, diag=copy(SEMQUAD_PIC_DIAG[]))
end

@testset "G194 latent heat" begin
    # ---- (1) phase derivatives: one implementation, checked by FD ----------
    for (z, T) in ((0.6, 0.45), (0.62, 0.48), (0.55, 0.4), (0.7, 0.3), (0.5, 0.6))
        gT = phase_dgamma_dT(G194_PT, z, T)
        gz = phase_dgamma_dz(G194_PT, z, T)
        gTT = phase_d2gamma_dT2(G194_PT, z, T)
        gzT = phase_d2gamma_dzdT(G194_PT, z, T)
        for h in (1.0e-6, 1.0e-7)
            fdT = (phase_gamma(G194_PT, z, T + h) -
                   phase_gamma(G194_PT, z, T - h)) / (2h)
            fdz = (phase_gamma(G194_PT, z + h, T) -
                   phase_gamma(G194_PT, z - h, T)) / (2h)
            fdTT = (phase_dgamma_dT(G194_PT, z, T + h) -
                    phase_dgamma_dT(G194_PT, z, T - h)) / (2h)
            fdzT = (phase_dgamma_dz(G194_PT, z, T + h) -
                    phase_dgamma_dz(G194_PT, z, T - h)) / (2h)
            @test abs(gT - fdT) <= 1.0e-6 * abs(fdT)
            @test abs(gz - fdz) <= 1.0e-6 * abs(fdz)
            @test abs(gTT - fdTT) <= 1.0e-6 * abs(fdTT)
            @test abs(gzT - fdzT) <= 1.0e-6 * abs(fdzT)
        end
        # Gamma_T = slope * Gamma_z: the two derivatives are the same phase
        # function, so the buoyancy and the latent heat cannot drift apart
        @test gT ≈ G194_PT.slope * gz rtol = 1.0e-13
        @test gTT ≈ G194_PT.slope * gzT rtol = 1.0e-13
    end
    # the buoyancy's theta-derivative is -Ra*B*dGamma/dT (same function)
    let
        pr = Params(Ra=2.0, H=0.0)
        pt = PhaseTransition(0.6, 0.1, 0.05, 0.5, 0.1, 1.0)
        for (y, th) in ((0.6, 0.02), (0.62, -0.03), (0.55, 0.05))
            h = 1.0e-6
            fd = (semquad_phase_buoyancy(pr, [pt], y, th + h) -
                  semquad_phase_buoyancy(pr, [pt], y, th - h)) / (2h)
            ana = -pr.Ra * pt.B * phase_dgamma_dT(pt, y,
                                                  semquad_base_T(pr, y) + th)
            @test fd ≈ ana rtol = 1.0e-8
        end
    end
    # sharp transition: every derivative switches off, no NaN in the limit
    let pt0 = PhaseTransition(0.6, 0.0, 0.05, 0.5, 0.1, 1.0)
        @test phase_dgamma_dT(pt0, 0.5, 0.4) == 0.0
        @test phase_d2gamma_dT2(pt0, 0.5, 0.4) == 0.0
        @test phase_d2gamma_dzdT(pt0, 0.5, 0.4) == 0.0
    end

    # ---- (2) apparent heat capacity vs explicit source ---------------------
    let
        pr = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, latent_L=2.5)
        qt = g194_tree(2)
        sq = make_semquad(qt, 2)
        N = sq.N
        th = [0.3 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
        dth = [sin(0.4 * k) * sinpi(sq.coords[k][2]) for k in 1:N]
        ux = [0.1 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
        uz = [0.2 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
        M1, _, C1, F1 = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                                advection=:galerkin, theta=th,
                                                phases=[G194_PT])
        M0, _, C0, F0 = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                                advection=:galerkin, theta=th)
        r_cp = M1 * dth + C1 * th - F1
        r_src = M0 * dth + C0 * th - F0
        # independent quadrature of the explicit-source form of the same
        # equation: -L int (Gamma_T (dtheta/dt + u.grad theta) + Gamma_z u_z) phi
        fdq = make_fedata(sq.p, sq.p - 1)
        n = sq.p + 1
        src = zeros(N)
        for lf in sq.leaves
            hx = MantleSEM.leaf_hx(sq, lf[1])
            hy = MantleSEM.leaf_hy(sq, lf[1])
            x0 = lf[2] * hx; y0 = lf[3] * hy
            ids = sq.elem[lf]
            for qy in 1:fdq.nq, qx in 1:fdq.nq
                x = x0 + 0.5 * (fdq.gl_x[qx] + 1.0) * hx
                y = y0 + 0.5 * (fdq.gl_x[qy] + 1.0) * hy
                w = 0.25 * hx * hy * fdq.gl_w[qx] * fdq.gl_w[qy]
                thq = 0.0; uxq = 0.0; uzq = 0.0; dq = 0.0
                thxq = 0.0; thyq = 0.0
                for j in 1:n, i in 1:n
                    k = (j - 1) * n + i
                    b = fdq.phi_v[i, qx] * fdq.phi_v[j, qy]
                    thq += th[ids[k]] * b
                    dq += dth[ids[k]] * b
                    uxq += ux[ids[k]] * b
                    uzq += uz[ids[k]] * b
                    thxq += th[ids[k]] * fdq.dphi_v[i, qx] * fdq.phi_v[j, qy] *
                            (2.0 / hx)
                    thyq += th[ids[k]] * fdq.phi_v[i, qx] * fdq.dphi_v[j, qy] *
                            (2.0 / hy)
                end
                T = semquad_base_T(pr, y) + thq
                # residual form: the equation is
                #   theta_t + u.grad theta = lap(theta) - u_z dTbar/dz
                #       - L[Gamma_T (theta_t + u.grad theta) + Gamma_z u_z]
                # so moving the bracket to the left gives +L(...)
                s = pr.latent_L *
                    (phase_dgamma_dT(G194_PT, y, T) * (dq + uxq * thxq +
                                                       uzq * thyq) +
                     phase_dgamma_dz(G194_PT, y, T) * uzq)
                for j in 1:n, i in 1:n
                    k = (j - 1) * n + i
                    src[ids[k]] += w * s * fdq.phi_v[i, qx] * fdq.phi_v[j, qy]
                end
            end
        end
        @test norm(r_cp - r_src - src) <= 1.0e-10 * norm(src)
        @test norm(src) > 0.0
        # `theta_lat` is the temperature at which the latent coefficients are
        # evaluated: a perturbation of theta_lat alone changes M by exactly the
        # secant of the apparent capacity at that temperature
        th2 = th .* 1.5
        Ma, _, _, _ = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                              advection=:galerkin, theta=th,
                                              theta_lat=th2, phases=[G194_PT])
        Mb, _, _, _ = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                              advection=:galerkin, theta=th,
                                              theta_lat=th, phases=[G194_PT])
        @test norm(Ma - Mb) > 0.0
        @test_throws ArgumentError assemble_semquad_energy(sq, qt, pr, ux, uz;
                                                          theta=th,
                                                          theta_lat=zeros(N - 1),
                                                          phases=[G194_PT])
    end

    # ---- (3) the default path is untouched --------------------------------
    let
        pr0 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=10.0)
        prL = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=10.0, latent_L=2.5)
        qt = g194_tree(2)
        th0, _ = g194_sin_state(qt)
        base = run_semquad_simulation(qt, pr0, th0; p=2, tstop=3.0e-4,
                                      dt0=1.0e-4, dtmax=1.0e-4, dtmin=1e-16,
                                      amr_every=0, linear=:direct, maxpic=1,
                                      verbose=false)
        # latent_L without phases: nothing to apply
        nof = run_semquad_simulation(g194_tree(2), prL, th0; p=2,
                                     tstop=3.0e-4, dt0=1.0e-4, dtmax=1.0e-4,
                                     dtmin=1e-16, amr_every=0, linear=:direct,
                                     maxpic=1, verbose=false)
        @test nof.theta == base.theta
        @test nof.ux == base.ux && nof.uz == base.uz && nof.pv == base.pv
        @test nof.history == base.history
        # phases with a zero density jump and zero latent heat: same state
        zero_pt = PhaseTransition(0.6, 0.1, 0.05, 0.5, 0.1, 0.0)
        zz = run_semquad_simulation(g194_tree(2), pr0, th0; p=2,
                                    tstop=3.0e-4, dt0=1.0e-4, dtmax=1.0e-4,
                                    dtmin=1e-16, amr_every=0, linear=:direct,
                                    maxpic=1, phases=[zero_pt], verbose=false)
        @test zz.theta == base.theta && zz.history == base.history
        # the scheme selector is inert without phases
        lag = run_semquad_simulation(g194_tree(2), prL, th0; p=2,
                                     tstop=3.0e-4, dt0=1.0e-4, dtmax=1.0e-4,
                                     dtmin=1e-16, amr_every=0, linear=:direct,
                                     maxpic=1, latent_scheme=:lagged,
                                     verbose=false)
        @test lag.theta == base.theta && lag.history == base.history
        # the same latent problem with/without the phase list differs (the
        # latent term must actually be active)
        lat = run_semquad_simulation(g194_tree(2), prL, th0; p=2,
                                     tstop=3.0e-4, dt0=1.0e-4, dtmax=1.0e-4,
                                     dtmin=1e-16, amr_every=0, linear=:direct,
                                     maxpic=1, phases=[G194_PT], verbose=false)
        @test norm(lat.theta - base.theta) / norm(base.theta) > 1.0e-5
        @test_throws ArgumentError run_semquad_simulation(g194_tree(2), prL,
                                                          th0; p=2, tstop=1.0e-5,
                                                          dt0=1.0e-5,
                                                          phases=[G194_PT],
                                                          latent_scheme=:bogus)
    end

    # ---- (4) Newton Jacobian with phases + latent -------------------------
    let
        pr = g194_composite_params()
        qt = g194_tree(2)
        ss = make_semstokes(qt, 2)
        sq = ss.su
        N = sq.N
        leaves = Set(qt.leaves)
        th = [0.05 * sinpi(x) * sinpi(z) for (x, z) in sq.coords]
        phases = [G194_PT, PhaseTransition(0.3, 0.08, -0.01, 0.4, 0.05, 0.5)]
        # Mechanical update (docs/RHEOLOGY.md §8.4): the `make_eta` contract
        # gained the pressure argument for the Drucker-Prager yield stress.
        make_eta = function (theta, ufull, _pfull=nothing)
            ux = ufull[1:N]; uz = ufull[(N + 1):(2N)]
            T_at(x, y) = Tbar(y, pr.H) +
                         MantleSEM._semquad_eval_at(sq, theta, x, y, leaves)
            ed_at(x, y) = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
            ((x, y) -> eta_composite_at(T_at(x, y), ed_at(x, y), y, pr)[1],
             (x, y) -> deta_composite_at(T_at(x, y), ed_at(x, y), y, pr)[1],
             (x, y, ed) -> deta_composite_at(T_at(x, y), ed, y, pr)[2])
        end
        z2 = (x, y) -> 0.0
        ufull = zeros(2N)
        for _ in 1:3
            ef, _, _ = make_eta(th, ufull)
            u, _, _, _, _ = solve_semquad_stokes_var(
                ss, qt, ef, z2,
                (x, y) -> pr.Ra * MantleSEM._semquad_eval_at(sq, th, x, y, leaves),
                z2, z2; bc=pr.bc, linear=:direct)
            ufull = 0.7 * ufull + 0.3 * u
        end
        pfull = zeros(ss.sp.N)
        ctx = MantleSEM._semquad_step_residual_ctx(ss, qt, pr, th, th, th, ufull,
                                                  pfull, make_eta, leaves;
                                                  dt=1.0e-5, dt_prev=1.0e-5,
                                                  firststep=false, phases=phases)
        Mv = assemble_semquad_pressure_mass(sq)
        nf, nfp, nt = ctx.nf, ctx.nfp, ctx.nt
        resid_of = function (x)
            thx = th + ctx.parts.Cfree * x[(nf + nfp + 1):end]
            ufx = ufull + ctx.sys.Cu * x[1:nf]
            pfx = pfull + ctx.sys.Cp * x[(nf + 1):(nf + nfp)]
            c = MantleSEM._semquad_step_residual_ctx(ss, qt, pr, th, th, thx,
                                                    ufx, pfx, make_eta, leaves;
                                                    dt=1.0e-5, dt_prev=1.0e-5,
                                                    firststep=false,
                                                    phases=phases)
            return MantleSEM.semquad_step_residual(c, thx)
        end
        x = [sin(0.7 * k) for k in 1:(nf + nfp + nt)]
        x ./= norm(x)
        Jx = MantleSEM.semquad_step_jvp(ctx, th, ufull, x, Mv; N=N, sq=sq,
                                        q=ss.q, pr=pr, advection=:galerkin)
        for h in (1.0e-6, 1.0e-7)
            fd = (resid_of(h * x) - resid_of(-h * x)) / (2h)
            @test norm(Jx - fd) / norm(Jx) <= 1.0e-6
        end
        # the latent part is a real (small) term of the Jacobian, and dropping it
        # makes the Jacobian measurably worse
        Jx0 = MantleSEM.semquad_step_jvp(ctx, th, ufull, x, Mv; N=N, sq=sq,
                                         q=ss.q, pr=pr, advection=:galerkin,
                                         latent_tangent=false)
        @test norm(Jx - Jx0) > 0.0
        fd = (resid_of(1.0e-6 * x) - resid_of(-1.0e-6 * x)) / 2.0e-6
        @test norm(Jx0 - fd) / norm(Jx0) > norm(Jx - fd) / norm(Jx)
    end

    # ---- (5) energy budget -------------------------------------------------
    let
        # (a) one step: the exact enthalpy rate equals the consistent boundary
        # reaction to O(dt); the residual of that identity halves with dt's
        o1 = g194_onestep(dt=1.0e-3)
        o2 = g194_onestep(dt=5.0e-4)
        @test o1.ares < 1.0e-12            # the step is solved
        @test abs(o1.cons) <= 1.0e-5       # measured ~1e-6
        @test abs(o2.cons) <= 1.0e-5
        @test abs(o1.cons) / max(abs(o2.cons), 1.0e-30) > 1.5   # first order
        # the physical wall flux is the same budget to the wall-quadrature
        # term: a ~1% closure, unchanged by the latent heat
        @test abs(o1.phys) <= 5.0e-2
        o0 = g194_onestep(dt=1.0e-3, lat=0.0, phases=PhaseTransition[])
        @test abs(o1.phys) <= 2.0 * abs(o0.phys)

        # (b) multi-step driver budget: the latent enthalpy is active, and the
        # closure against the physical wall flux is the same order as the
        # latent-free scheme
        ol = g194_column(nsteps=6, dt=1.0e-3, lat=2.5, phases_on=true)
        o0 = g194_column(nsteps=6, dt=1.0e-3, lat=2.5, phases_on=false)
        wl = g194_budget_worst(ol.rec)
        w0 = g194_budget_worst(o0.rec)
        @test wl <= 3.0e-2                 # measured 1.4e-2
        @test w0 <= 3.0e-2
        @test wl <= 1.5 * w0               # measured ratio 1.01-1.10
        @test maximum(abs(x.b.latent_int) for x in ol.rec) > 1.0e-6
    end

    # ---- (6) analytic 1-D conductive column -------------------------------
    let
        # T = 1-y is the exact steady state of the latent-heat column: the
        # apparent capacity changes the transient, not the steady state
        st = Dict{Float64,Any}()
        for lat in (0.0, 2.5)
            qt = g194_tree(3)
            ss = make_semstokes(qt, 2)
            th0 = [1.0e-3 * sinpi(y) for (x, y) in ss.su.coords]
            for (i, (x, y)) in enumerate(ss.su.coords)
                (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
            end
            pr = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0,
                        latent_L=lat)
            rec = Any[]
            cb = s -> push!(rec, (b=semquad_heat_budget(s.sq, s.qt, pr, s.theta;
                                                        phases=[G194_PT]),
                                  theta=copy(s.theta)))
            r = run_semquad_simulation(qt, pr, th0; p=2, tstop=1.2,
                                       dt0=1.0e-2, dtmax=1.0e-2, dtmin=1e-18,
                                       maxsteps=120, amr_every=0, ev=0.0,
                                       advection=:galerkin, linear=:direct,
                                       maxpic=1, phases=[G194_PT],
                                       callback=cb, verbose=false)
            st[lat] = (r=r, rec=rec)
            @test maximum(abs.(r.theta)) < 1.0e-6        # analytic T = 1-y
            @test abs(rec[end].b.flux) < 1.0e-6
        end
        # the latent heat is active during the transient: at the first step the
        # latent enthalpy is non-zero and the solution differs from L = 0
        @test abs(st[2.5].rec[1].b.latent_int) > 1.0e-7
        @test norm(st[2.5].rec[1].theta - st[0.0].rec[1].theta) /
              norm(st[0.0].rec[1].theta) > 1.0e-3
        # a negative apparent heat capacity is rejected instead of blowing up
        qt = g194_tree(2)
        ss = make_semstokes(qt, 2)
        th_bad = [0.05 * sinpi(x) * sinpi(y) for (x, y) in ss.su.coords]
        prbad = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0,
                       latent_L=10.0)
        @test_throws ArgumentError run_semquad_simulation(
            g194_tree(2), prbad, th_bad; p=2, tstop=1.0e-4, dt0=1.0e-4,
            amr_every=0, linear=:direct, maxpic=1, phases=[G194_PT],
            verbose=false)
    end

    # ---- (7) composite rheology + phase transition, Picard and Newton ------
    let
        pc = g194_smoke(maxpic=20)
        nw = g194_smoke(mode=:newton)
        @test pc.r.steps == 5 && nw.r.steps == 5
        @test all(isfinite, pc.r.theta) && all(isfinite, nw.r.theta)
        @test all(isfinite, vcat(pc.r.history...)) && all(isfinite, vcat(nw.r.history...))
        @test all(row -> last(row.changes) < 1.0e-6, pc.diag)
        @test all(row -> last(row.changes) < 1.0e-6, nw.diag)
        @test all(row -> row.mode === :newton, nw.diag)
        @test !any(row -> row.fallback, nw.diag)
        @test sum(row.iters for row in nw.diag) < sum(row.iters for row in pc.diag)
        # both find the same fixed point of the step
        @test norm(pc.r.theta - nw.r.theta) / norm(pc.r.theta) < 1.0e-5
        @test pc.r.history[end][3] ≈ nw.r.history[end][3] rtol = 1.0e-8
        # the latent enthalpy of the final state is non-zero (the term is on)
        prc = g194_composite_params()
        b = semquad_heat_budget(nw.r.ss.su, nw.r.qt, prc, nw.r.theta;
                                phases=[G194_PT], ux=nw.r.ux, uz=nw.r.uz)
        @test abs(b.latent_int) > 0.0
        # the lagged variant is a first-order-in-time approximation: it runs and
        # stays close to the implicit one on five steps
        lg = g194_smoke(maxpic=20, latent_scheme=:lagged)
        @test all(isfinite, lg.r.theta)
        @test norm(lg.r.theta - pc.r.theta) / norm(pc.r.theta) < 1.0e-3
        @test_throws ArgumentError g194_smoke(mode=:newton, latent_scheme=:bad)
    end
end
