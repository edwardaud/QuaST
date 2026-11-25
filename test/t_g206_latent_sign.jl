# G206: the SIGN of the latent heat of a phase transition, per transition.
#
# The energy equation carries `-sum_i latent_i DGamma_i/Dt`, i.e. an apparent
# heat capacity `1 + sum_i latent_i Gamma_T,i` and an advective source
# `-sum_i latent_i Gamma_z,i u_z`.  The physics (ASPECT's `latent heat` material
# model + heating model, `dS_A = gamma drho/rho^2`, `+T dS_A DGamma/Dt`) fixes the
# sign: with `latent_i = -T_i dS_A,i/(cp dT)`,
#
#   * a POSITIVE Clapeyron slope (exothermic, our 410) has latent_i < 0 and
#     HEATS descending material (`-latent_i Gamma_z u_z > 0`);
#   * a NEGATIVE slope (endothermic, our 660) has latent_i > 0 and COOLS it.
#
# A single positive `latent_L` for both transitions (the historical `:global`
# mode) therefore has the 410 wrong-signed.
#
# This gate checks:
#   (1) the coefficient formula against ASPECT's numbers and against the
#       equivalent `-T_abs*slope*B*Di` form;
#   (2) the sign of the temperature anomaly of descending and ascending
#       material at an exothermic and at an endothermic transition (1-D column,
#       compared against the latent-free step);
#   (3) `:per_phase` and `:global` agree BITWISE when every transition has the
#       same coefficient (all gammas of one sign), and differ when they do not;
#   (4) the one-step and multi-step energy budget closes at the same order as
#       the global/latent-free scheme;
#   (5) the per-step Newton Jacobian with the per-phase coefficients matches
#       central differences of the residual;
#   (6) the guards: `:per_phase` without coefficients, a bad `latent_mode`, the
#       negative apparent heat capacity, and the `:global` default.
using Test, QuaST, LinearAlgebra, Printf

# ASPECT scale set (the realistic case)
const G206_RHO0 = 3300.0
const G206_CP = 1000.0
const G206_DT = 1000.0
const G206_G = 9.81
const G206_D = 2.9e6
const G206_ALPHA = 3.0e-5

function fnv(a)
    h = 0xcbf29ce484222325
    for b in reinterpret(UInt8, a)
        h ⊻= UInt64(b)
        h *= 0x100000001b3
    end
    return string(h, base=16, pad=16)
end
fnvall(xs...) = fnv(reinterpret(UInt8, reduce(vcat, [collect(Float64, x) for x in xs])))
fnv_sparse(A) = fnvall(collect(A.colptr), collect(A.rowval), A.nzval)

function g206_tree(nref::Int=3; maxlevel::Int=5)
    qt = Quadtree(maxlevel)
    for _ in 1:nref, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    return qt
end

# The two transitions of the realistic case, as `PhaseTransition`s (the
# `per_phase_latent` helper fills the coefficients from the scale set).
g206_realistic_phases() =
    [PhaseTransition(0.87, 0.025, +0.032, 0.5, 180.0,
                     180.0 / (G206_RHO0 * G206_ALPHA * G206_DT)),
     PhaseTransition(0.78, 0.025, -0.032, 0.5, 300.0,
                     300.0 / (G206_RHO0 * G206_ALPHA * G206_DT))]

g206_scale_kwargs() = (; T_surface=0.5, rho_ref=G206_RHO0, cp=G206_CP,
                       dT=G206_DT, g=G206_G, d=G206_D)

# Base state of the realistic case (H = 10) at rest.  `:expT` with `deta_T = 1`
# is the neutral rheology of the driver paths used here (`run_semquad_simulation`
# rejects `:const`); the column solves do not use eta.
g206_params(; latent_L=0.0, latent_mode=:per_phase) =
    Params(Ra=0.0, H=10.0, eta_law=:expT, deta_T=1.0, latent_L=latent_L,
           latent_mode=latent_mode)

# One energy step of a 1-D column with a prescribed vertical velocity `U`
# (`u_z = -U`, U > 0 = descending).  Returns the state after the step.
function g206_column_step(pt; pr=g206_params(), U=1.0, nref=3, dt=2.0e-3,
                          amp=1.0e-3, phases=nothing)
    qt = g206_tree(nref)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    th0 = [amp * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end
    ux = zeros(N); uz = fill(-U, N)
    ph = phases === nothing ? (pt === nothing ? PhaseTransition[] : [pt]) : phases
    parts = solve_semquad_energy(sq, qt, pr, th0, th0, ux, uz, dt, dt, true;
                                 advection=:galerkin, phases=ph,
                                 latent_theta=th0, return_parts=true)
    th1 = parts.u0 + parts.Cfree *
          (parts.Ar \ (parts.Cfree' * (parts.rhs - parts.A * parts.u0)))
    return (sq=sq, qt=qt, ss=ss, th0=th0, th1=th1, pr=pr, phases=ph,
            ux=ux, uz=uz, dt=dt)
end

# mean of `th1 - ref` in the band `z0 -+ 2*width`, on the column's own nodes
function g206_band_mean(r, ref, z0, width)
    idx = [i for (i, (x, y)) in enumerate(r.sq.coords)
           if abs(y - z0) <= 2.0 * width]
    d = r.th1 - ref.th1
    return sum(d[idx]) / length(idx), maximum(abs.(d))
end

# One-step enthalpy budget of the column (G194's `g194_onestep`, per-phase).
function g206_onestep(; dt=1.0e-3, phases, pr, nref=3)
    qt = g206_tree(nref)
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
    th1 = parts.u0 + parts.Cfree *
          (parts.Ar \ (parts.Cfree' * (parts.rhs - parts.A * parts.u0)))
    r = parts.A * th1 - parts.rhs
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
            b0=b0, b1=b1)
end

# Multi-step driver run (1-D column, no velocity: the latent capacity acts, the
# advective latent source is zero) with the per-step budget recorded.
function g206_column_run(; nsteps=6, dt=1.0e-3, phases, pr)
    qt = g206_tree(3)
    ss = make_semstokes(qt, 2)
    th0 = [1.0e-3 * sinpi(y) for (x, y) in ss.su.coords]
    for (i, (x, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end
    rec = Any[]
    cb = st -> push!(rec, (dt=st.dt,
                           b=semquad_heat_budget(st.sq, st.qt, pr, st.theta;
                                                 phases=phases, ux=st.ux,
                                                 uz=st.uz)))
    r = run_semquad_simulation(qt, pr, th0; p=2, tstop=nsteps * dt, dt0=dt,
                               dtmax=dt, dtmin=1.0e-18, amr_every=0, ev=0.0,
                               advection=:galerkin, linear=:direct, maxpic=1,
                               pic_tol=1.0e-10, phases=phases, callback=cb,
                               verbose=false)
    return (r=r, rec=rec)
end

function g206_budget_worst(rec)
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

@testset "G206 latent sign" begin
    # ---- (1) the coefficient formula --------------------------------------
    let
        pr = g206_params()
        ph = per_phase_latent(pr, g206_realistic_phases(); g206_scale_kwargs()...)
        L410, L660 = ph[1].latent, ph[2].latent
        # ASPECT's numbers: its
        # coefficient L_A = T dS_A/(cp dT) is +0.060 / -0.131; ours is -L_A.
        @test L410 < 0.0 && L660 > 0.0
        @test L410 ≈ -0.060 rtol = 2.0e-2        # -0.059364 (T = 1195.5 K)
        @test L660 ≈ +0.131 rtol = 2.0e-2        # +0.130596 (T = 1578.0 K)
        @test abs(L410 + 5.9364e-2) < 1.0e-5
        @test abs(L660 - 1.30596e-1) < 1.0e-4
        # T_abs = Tbar(z0) + T_surface is the temperature the coefficient uses
        @test semquad_base_T(pr, 0.87) + 0.5 ≈ 1.1955 rtol = 1.0e-12
        @test semquad_base_T(pr, 0.78) + 0.5 ≈ 1.578 rtol = 1.0e-12
        # the same number from the three algebraically equal forms:
        #   -T dS_A/cp,  -T gamma drho/(rho^2 cp),  -theta T slope B Di
        pt = ph[2]
        Tabs = semquad_base_T(pr, pt.z0) + 0.5
        gamma = pt.slope * G206_RHO0 * G206_G * G206_D / G206_DT
        manual1 = -Tabs * gamma * pt.drho / (G206_RHO0^2 * G206_CP)
        @test pt.latent ≈ manual1 rtol = 1.0e-13
        di = G206_ALPHA * G206_G * G206_D / G206_CP
        manual2 = -Tabs * pt.slope * pt.B * di
        @test pt.latent ≈ manual2 rtol = 1.0e-12
        # the scale set is what carries the units: with the code's own unit
        # system (all scale keywords 1) the coefficient is -theta*slope*drho
        pt1 = PhaseTransition(0.5, 0.1, 0.7, 0.5, 0.2, 1.0)
        @test phase_latent_coefficient(pt1, 0.9) ≈ -0.9 * 0.7 * 0.2 rtol = 1e-15
        # `per_phase_latent` only fills `latent`, nothing else moves
        @test ph[1].z0 == 0.87 && ph[1].width == 0.025 && ph[1].slope == 0.032
        @test ph[1].Tref == 0.5 && ph[1].drho == 180.0 && ph[1].B ==
              g206_realistic_phases()[1].B
    end

    # ---- (2) the SIGN of the temperature anomaly --------------------------
    let
        # the physically-derived coefficients of the realistic case, one
        # transition at a time, on a 1-D column with u_z = -+U
        pr = g206_params()
        ph = per_phase_latent(pr, g206_realistic_phases();
                              g206_scale_kwargs()...)
        exo, endo = ph[1], ph[2]          # z0 = 0.87 (slope>0), 0.78 (slope<0)
        @test exo.latent < 0.0
        @test endo.latent > 0.0
        # latent-free references, on the same tree and initial state
        r0d = g206_column_step(nothing; pr=g206_params(), U=+1.0,
                               phases=PhaseTransition[])     # descending
        r0u = g206_column_step(nothing; pr=g206_params(), U=-1.0,
                               phases=PhaseTransition[])     # ascending
        for pt in (exo, endo)
            rdown = g206_column_step(pt; pr=pr, U=+1.0)
            rup = g206_column_step(pt; pr=pr, U=-1.0)
            mdown, amax = g206_band_mean(rdown, r0d, pt.z0, pt.width)
            mup, _ = g206_band_mean(rup, r0u, pt.z0, pt.width)
            if pt === exo
                @test mdown > 0.0        # exothermic: descending HEATS
                @test mup < 0.0          # and ascending cools
            else
                @test mdown < 0.0        # endothermic: descending COOLS
                @test mup > 0.0
            end
            @test amax > 1.0e-8          # the latent term is really active
        end
        # the same run in `:global` mode (one positive latent_L) gets the
        # exothermic transition wrong: it cools the downwelling
        prg = Params(Ra=0.0, H=10.0, eta_law=:const, latent_L=0.11)
        rexo_g = g206_column_step(exo; pr=prg, U=+1.0)
        mdown_g, _ = g206_band_mean(rexo_g, r0d, exo.z0, exo.width)
        @test mdown_g < 0.0                        # the bug: cools the 410
        mdown_p, _ = g206_band_mean(g206_column_step(exo; pr=pr, U=+1.0),
                                    r0d, exo.z0, exo.width)
        @test mdown_p > 0.0
        @test sign(mdown_g) != sign(mdown_p)       # the fix flips it
    end

    # ---- (3) `:per_phase` == `:global` when the coefficients are equal ------
    let
        # all gammas of one sign, every per-phase coefficient equal to the
        # global one: the assembled operators and a whole time step must be
        # bitwise identical (the only difference would be a rounding change in
        # the order of the coefficient multiplication)
        prg = Params(Ra=1.0e6, H=0.0, eta_law=:expT, deta_T=1.0, latent_L=0.35)
        prp = Params(Ra=1.0e6, H=0.0, eta_law=:expT, deta_T=1.0,
                     latent_mode=:per_phase)
        phg = [PhaseTransition(0.6, 0.1, -0.05, 0.5, 0.1, 1.0),
               PhaseTransition(0.3, 0.08, -0.02, 0.4, 0.05, 0.5)]
        php = [PhaseTransition(0.6, 0.1, -0.05, 0.5, 0.1, 1.0, 0.35),
               PhaseTransition(0.3, 0.08, -0.02, 0.4, 0.05, 0.5, 0.35)]
        for (tag, prr, phs) in (("global", prg, phg), ("per_phase", prp, php))
            @test prr.latent_L == 0.35 || prr.latent_mode === :per_phase
        end
        qt = g206_tree(2)
        ss = make_semstokes(qt, 2)
        sq = ss.su
        th = [0.05 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
        N = sq.N
        ux = [0.3 * cos(0.5 * pi * x) * sin(pi * y) for (x, y) in sq.coords]
        uz = [-0.3 * sin(0.5 * pi * x) * cos(pi * y) for (x, y) in sq.coords]
        ops(prr, phs) = assemble_semquad_energy(sq, qt, prr, ux, uz;
                                               advection=:skew, theta=th,
                                               phases=phs)
        Mg, Kg, Cg, Fg = ops(prg, phg)
        Mp, Kp, Cp, Fp = ops(prp, php)
        @test fnv_sparse(Mg) == fnv_sparse(Mp)
        @test fnv_sparse(Kg) == fnv_sparse(Kp)
        @test fnv_sparse(Cg) == fnv_sparse(Cp)
        @test fnv(Fg) == fnv(Fp)
        @test any(!=(0.0), Fg)          # the latent source is really active
        # a whole step through the driver
        run1(prr, phs) = run_semquad_simulation(
            g206_tree(2), prr, th; p=2, tstop=4.0e-5, dt0=1.0e-5,
            dtmax=1.0e-5, dtmin=1.0e-16, amr_every=0, ev=0.0,
            advection=:galerkin, linear=:direct, maxpic=1, phases=phs,
            verbose=false)
        a = run1(prg, phg); b = run1(prp, php)
        @test fnv(a.theta) == fnv(b.theta)
        @test fnv(a.ux) == fnv(b.ux)
        @test fnv(a.uz) == fnv(b.uz)
        # mixed signs: the two forms no longer agree (that IS the bug fix)
        phm = [PhaseTransition(0.6, 0.1, +0.05, 0.5, 0.1, 1.0, -0.35),
               PhaseTransition(0.3, 0.08, -0.02, 0.4, 0.05, 0.5, 0.35)]
        prm = Params(Ra=1.0e6, H=0.0, eta_law=:expT, deta_T=1.0, latent_L=0.35,
                     latent_mode=:per_phase)
        c = run1(prm, phm)
        @test fnv(c.theta) != fnv(a.theta)
    end

    # ---- (4) the energy budget --------------------------------------------
    let
        pr = g206_params(latent_L=0.0)
        ph = per_phase_latent(pr, g206_realistic_phases();
                              g206_scale_kwargs()...)
        o1 = g206_onestep(dt=1.0e-3, phases=ph, pr=pr)
        o2 = g206_onestep(dt=5.0e-4, phases=ph, pr=pr)
        @test o1.ares < 1.0e-12                 # the step is solved
        @test abs(o1.cons) <= 1.0e-5            # measured ~1e-6 (G194: 1.6e-7)
        @test abs(o2.cons) <= 1.0e-5
        @test abs(o1.cons) > 0.0                # the latent term is active
        @test abs(o1.phys) <= 5.0e-2            # the wall-quadrature term
        o0 = g206_onestep(dt=1.0e-3, phases=PhaseTransition[],
                          pr=g206_params())
        @test abs(o1.phys) <= 2.0 * abs(o0.phys)
        # multi-step: same order as the latent-free scheme
        wl = g206_budget_worst(g206_column_run(phases=ph, pr=pr).rec)
        w0 = g206_budget_worst(g206_column_run(phases=PhaseTransition[],
                                               pr=g206_params()).rec)
        @test wl <= 3.0e-2
        @test w0 <= 3.0e-2
        @test wl <= 1.5 * w0
        run = g206_column_run(phases=ph, pr=pr)
        @test maximum(abs(x.b.latent_int) for x in run.rec) > 1.0e-6
        # the two transitions' contributions have opposite signs
        b = run.rec[end].b
        @test abs(b.latent_int) > 0.0
    end

    # ---- (5) the Newton Jacobian with the per-phase coefficients -----------
    let
        ED = 9.210340371976184; ES = 12.0; NN = 3.0; T0 = 0.5
        pr = Params(; Ra=1.0e6, H=0.0, eta_law=:composite,
                    comp_Ad=exp(ED / (1.0 + T0)), comp_Ed=ED, comp_Vd=0.0,
                    comp_As=1.0e-2 * exp(ES / (NN * (1.0 + T0))),
                    comp_Es=ES, comp_Vs=0.0, comp_n=NN,
                    eta_T0=T0, eta_min=1.0e-6, eta_max=1.0e4,
                    latent_mode=:per_phase)
        qt = g206_tree(2)
        ss = make_semstokes(qt, 2)
        sq = ss.su
        N = sq.N
        leaves = Set(qt.leaves)
        th = [0.05 * sinpi(x) * sinpi(z) for (x, z) in sq.coords]
        phases = [PhaseTransition(0.6, 0.1, +0.05, 0.5, 0.1, 1.0, -0.6),
                  PhaseTransition(0.3, 0.08, -0.01, 0.4, 0.05, 0.5, +0.4)]
        @test all(pt -> pt.latent !== nothing, phases)
        make_eta = function (theta, ufull, _pfull=nothing)
            ux = ufull[1:N]; uz = ufull[(N + 1):(2N)]
            T_at(x, y) = Tbar(y, pr.H) +
                         QuaST._semquad_eval_at(sq, theta, x, y, leaves)
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
                (x, y) -> pr.Ra * QuaST._semquad_eval_at(sq, th, x, y, leaves),
                z2, z2; bc=pr.bc, linear=:direct)
            ufull = 0.7 * ufull + 0.3 * u
        end
        pfull = zeros(ss.sp.N)
        ctx = QuaST._semquad_step_residual_ctx(ss, qt, pr, th, th, th, ufull,
                                                  pfull, make_eta, leaves;
                                                  dt=1.0e-5, dt_prev=1.0e-5,
                                                  firststep=false, phases=phases)
        @test ctx.latent_implicit                # the latent tangent is on
        Mv = assemble_semquad_pressure_mass(sq)
        nf, nfp, nt = ctx.nf, ctx.nfp, ctx.nt
        resid_of = function (x)
            thx = th + ctx.parts.Cfree * x[(nf + nfp + 1):end]
            ufx = ufull + ctx.sys.Cu * x[1:nf]
            pfx = pfull + ctx.sys.Cp * x[(nf + 1):(nf + nfp)]
            c = QuaST._semquad_step_residual_ctx(ss, qt, pr, th, th, thx,
                                                    ufx, pfx, make_eta, leaves;
                                                    dt=1.0e-5, dt_prev=1.0e-5,
                                                    firststep=false,
                                                    phases=phases)
            return QuaST.semquad_step_residual(c, thx)
        end
        x = [sin(0.7 * k) for k in 1:(nf + nfp + nt)]
        x ./= norm(x)
        Jx = QuaST.semquad_step_jvp(ctx, th, ufull, x, Mv; N=N, sq=sq,
                                        q=ss.q, pr=pr, advection=:galerkin)
        for h in (1.0e-6, 1.0e-7)
            fd = (resid_of(h * x) - resid_of(-h * x)) / (2h)
            @test norm(Jx - fd) / norm(Jx) <= 1.0e-6
        end
        Jx0 = QuaST.semquad_step_jvp(ctx, th, ufull, x, Mv; N=N, sq=sq,
                                         q=ss.q, pr=pr, advection=:galerkin,
                                         latent_tangent=false)
        @test norm(Jx - Jx0) > 0.0
        fd = (resid_of(1.0e-6 * x) - resid_of(-1.0e-6 * x)) / 2.0e-6
        @test norm(Jx0 - fd) / norm(Jx0) > norm(Jx - fd) / norm(Jx)
        # the latent tangent itself is right: its own FD check
        dth = ctx.parts.Cfree * x[(nf + nfp + 1):end]
        Md, Cd, Fd = assemble_semquad_energy_latent_jvp(
            sq, qt, pr, phases, dth, th, ufull[1:N], ufull[(N + 1):(2N)];
            advection=:galerkin)
        @test any(!=(0.0), Md.nzval)
        Mm = Md * th + Cd * th - Fd
        @test norm(Mm) > 0.0
    end

    # ---- (6) guards and the default ---------------------------------------
    let
        @test Params().latent_mode === :global            # the default
        @test_throws ArgumentError Params(latent_mode=:bogus)
        @test Params(latent_L=0.5).latent_mode === :global
        # `:per_phase` without coefficients fails loudly, in the assembly and
        # in the driver
        prp = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0,
                     latent_mode=:per_phase)
        pt = PhaseTransition(0.6, 0.1, 0.05, 0.5, 0.1, 1.0)
        @test pt.latent === nothing
        qt = g206_tree(2)
        ss = make_semstokes(qt, 2)
        sq = ss.su
        th = zeros(sq.N)
        @test_throws ArgumentError assemble_semquad_energy(
            sq, qt, prp, zeros(sq.N), zeros(sq.N); advection=:skew, theta=th,
            phases=[pt])
        @test_throws ArgumentError run_semquad_simulation(
            g206_tree(2), prp, th; p=2, tstop=1.0e-5, dt0=1.0e-5,
            phases=[pt], verbose=false)
        @test_throws ArgumentError semquad_heat_budget(sq, qt, prp, th;
                                                       phases=[pt])
        # `latent_mode` is ignored when there are no phases (bitwise default)
        @test !semquad_latent_active(prp, PhaseTransition[])
        @test !semquad_latent_active(Params(latent_L=0.0), [pt])
        @test semquad_latent_active(Params(latent_L=0.5), [pt])
        # all-zero per-phase coefficients switch the terms off
        pt0 = PhaseTransition(0.6, 0.1, 0.05, 0.5, 0.1, 1.0, 0.0)
        @test !semquad_latent_active(prp, [pt0])
        # the negative-apparent-capacity guard (ASPECT's advection.cc): a
        # positive coefficient on a positive-slope transition drives
        # 1 + L Gamma_T below zero
        ptneg = PhaseTransition(0.6, 0.1, 0.05, 0.5, 0.1, 1.0, 10.0)
        @test_throws ArgumentError assemble_semquad_energy(
            sq, qt, Params(latent_mode=:per_phase), zeros(sq.N), zeros(sq.N);
            advection=:skew, theta=[1.0e-3 * sinpi(y) for (x, y) in sq.coords],
            phases=[ptneg])
        # `semquad_latent_coeff` / `semquad_latent_coefficients` agree
        prg = Params(latent_L=0.25)
        php = [PhaseTransition(0.6, 0.1, 0.05, 0.5, 0.1, 1.0, -0.25),
               PhaseTransition(0.3, 0.1, 0.05, 0.5, 0.1, 1.0, 0.5)]
        @test semquad_latent_coefficients(prg, php) == [0.25, 0.25]
        @test semquad_latent_coefficients(prp, php) == [-0.25, 0.5]
        @test all(semquad_latent_coeff(prp, pt) == pt.latent for pt in php)
    end
end

@printf("G206: assertions passed\n")
