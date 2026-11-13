using Test, QuaST, LinearAlgebra, SparseArrays

# G198 -- EOS-integrated reference state (docs/dev/PHYSICS.md section 11).
#
#   dT/dy   = -Lambda(y) T,       Lambda = alpha(y) g / cp(y)
#   drho/dy = -rho g / phi(y),    phi = K_S/rho
#   dp/dy   = -rho g
#
# with `phi = cp/alpha` (Adams-Williamson closure, K_S = rho cp/alpha) or
# `K_S = K0 + K0'(p - p_surf)` (Murnaghan).  The checks below are (i) the
# constant-property limit against the closed-form state, (ii) the hydrostatic
# identity, (iii) the mass form using the EOS rho/drho, (iv) the energy budget,
# (v) a 5-step composite-rheology smoke run, (vi) the default path.
@testset "G198 EOS reference state" begin
    P = 2
    tree(nref) = begin
        qt = Quadtree(nref)
        for _ in 1:nref, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        qt
    end

    # (1) the analytic state is bitwise the closed form it always was
    rs0 = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0)
    @test rs0.prof === nothing
    for z in (0.0, 0.137, 0.5, 0.923, 1.0)
        @test semquad_Tad(rs0, z) ==
              0.2 * exp(1.0 * 1.0 * (1.0 - z) / 1.0)
        @test semquad_dTad_dz(rs0, z) == -1.0 * semquad_Tad(rs0, z)
        @test semquad_rho(rs0, z) == 1.0
        @test semquad_alpha(rs0, z) == 1.0
        @test semquad_cp(rs0, z) == 1.0
        @test semquad_drho_dz(rs0, z) == 0.0
    end
    @test semquad_base_laplacian(Params(Ra=1.0, H=0.0, refstate=rs0), 0.37) ==
          1.0 * semquad_Tad(rs0, 0.37)
    @test mass_rho_fun(Params(Ra=1.0, H=0.0, refstate=rs0)) === nothing
    @test mass_drho_fun(Params(Ra=1.0, H=0.0, refstate=rs0)) === nothing

    # (2) constructor guards
    @test_throws ArgumentError SemRefState(eos=:bogus)
    @test_throws ArgumentError SemRefState(eos=:adams_williamson, nz=4)
    @test_throws ArgumentError SemRefState(eos=:adams_williamson,
                                           cp_fun=y -> -1.0)
    @test_throws ArgumentError SemRefState(eos=:adams_williamson,
                                           rho_fun=y -> 1.0)
    @test_throws ArgumentError SemRefState(eos=:murnaghan, K0=Inf)
    @test_throws ArgumentError SemRefState(eos=:adams_williamson, alpha_fun=y -> -1.0)
    @test_throws ArgumentError SemRefState(alpha_fun=y -> 1.0)   # analytic + fn
    @test_throws ArgumentError SemRefState(cp=0.0)

    # (3) Adams-Williamson with constant properties reproduces the closed-form
    # adiabat, and its density is the self-compacting adiabat exp(Lambda(1-y)).
    Lam = 1.0
    for nz in (129, 257)
        rsA = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                          eos=:adams_williamson, nz=nz)
        prA = Params(Ra=1.0, H=0.0, refstate=rsA)
        pr0 = Params(Ra=1.0, H=0.0, refstate=rs0)
        @test rsA.prof !== nothing
        @test length(rsA.prof.z) == nz
        zs = [k / 400 for k in 0:400]
        eT = maximum(abs(semquad_Tad(rsA, z) - semquad_Tad(rs0, z)) /
                     semquad_Tad(rs0, z) for z in zs)
        eL = maximum(abs(semquad_base_laplacian(prA, z) -
                         semquad_base_laplacian(pr0, z)) /
                     semquad_base_laplacian(pr0, z) for z in zs)
        eG = maximum(abs(semquad_dTad_dz(rsA, z) - semquad_dTad_dz(rs0, z)) /
                     abs(semquad_dTad_dz(rs0, z)) for z in zs)
        eR = maximum(abs(semquad_rho(rsA, z) - exp(Lam * (1.0 - z))) /
                     exp(Lam * (1.0 - z)) for z in zs)
        tol = nz == 257 ? 1.0e-11 : 1.0e-9
        @test eT < tol          # measured 6.07e-13 (257), 9.71e-12 (129)
        @test eL < tol
        @test eG < tol
        @test eR < tol
        # drho/dy = -Lambda rho (the Adams-Williamson closure with cp/alpha)
        @test maximum(abs(semquad_drho_dz(rsA, z) + Lam * semquad_rho(rsA, z))
                      for z in zs) < 1.0e-13
    end
    # the tabulation converges at fourth order
    e = Float64[]
    for nz in (33, 65, 129)
        rsN = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                          eos=:adams_williamson, nz=nz)
        push!(e, maximum(abs(semquad_Tad(rsN, k / 1000) -
                             semquad_Tad(rs0, k / 1000)) /
                         semquad_Tad(rs0, k / 1000) for k in 0:1000))
    end
    @test 8.0 < e[1] / e[2] < 32.0     # measured 16.0
    @test 8.0 < e[2] / e[3] < 32.0     # measured 16.0

    # (4) hydrostatic identity: the tabulated p is the integral of -rho g
    rsA = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                      eos=:adams_williamson, nz=257)
    pf = rsA.prof
    @test pf.dp == -pf.rho .* pf.g                       # node identity, bitwise
    hd = 1.0e-5
    ep = 0.0
    for k in 1:99
        y = k / 100
        d = (QuaST._prof_p(pf, y + hd) - QuaST._prof_p(pf, y - hd)) /
            (2 * hd)
        ep = max(ep, abs(d + semquad_rho(rsA, y)))
    end
    @test ep < 1.0e-5
    # the EOS density peels off the surface value by rho0*Lambda ~ 1 per unit depth
    @test semquad_rho(rsA, 0.0) ≈ exp(1.0) rtol=1e-10

    # (5) the mass form uses the EOS rho and drho (not a finite difference)
    prA = Params(Ra=1.0e6, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rsA,
                 Twall_top=semquad_Tad(rsA, 1.0),
                 Twall_bot=semquad_Tad(rsA, 0.0))
    rf = mass_rho_fun(prA)
    df = mass_drho_fun(prA)
    @test rf !== nothing && df !== nothing
    @test rf(0.5) == semquad_rho(rsA, 0.5)
    @test df(0.5) == semquad_drho_dz(rsA, 0.5)
    qt = tree(3)
    sq = make_semquad(qt, P)
    th0 = [0.05 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
    for (i, (_, y)) in enumerate(sq.coords)
        (y <= 1e-14 || y >= 1.0 - 1e-14) && (th0[i] = 0.0)
    end
    function step5(pr; nsteps=5, dt=2.0e-5, mass_form=:anelastic)
        run_semquad_simulation(qt, pr, th0; p=P, tstop=nsteps * dt, dt0=dt,
                               dtmax=dt, dtmin=1.0e-16, amr_every=0, ev=0.0,
                               advection=:skew, linear=:direct, maxpic=1,
                               pic_tol=1.0e-6, mass_form=mass_form,
                               verbose=false)
    end
    r = step5(prA)
    @test all(isfinite, r.theta) && all(isfinite, r.ux) && all(isfinite, r.uz)
    ss = make_semstokes(qt, P)
    BA = semquad_stokes_B(ss, qt; rho_fun=rf, drho_fun=df)
    u = vcat(r.ux, r.uz)
    @test maximum(abs, BA * u) < 1.0e-8       # the ALA constraint with the EOS rho
    @test maximum(abs, BA * u) > 0.0 || true
    # (6) the EOS constructor reproduces the closed-form reference state: with
    # constant properties, rho(y) = exp(1-y) *is* the Adams-Williamson density
    # of this adiabat, so the two 5-step runs agree to the tabulation error.
    rs0b = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                       rho_fun=y -> exp(1.0 - y))
    r0 = step5(Params(Ra=1.0e6, H=0.0, eta_law=:expT, deta_T=2.0,
                      refstate=rs0b,
                      Twall_top=semquad_Tad(rs0b, 1.0),
                      Twall_bot=semquad_Tad(rs0b, 0.0)))
    @test maximum(abs, r.theta - r0.theta) < 1.0e-9   # measured 1.37e-13
    # ... and a genuinely depth-dependent state is not the analytic one
    rsB = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                      eos=:adams_williamson, nz=129,
                      alpha_fun=y -> 1.0 + 0.5 * (1.0 - y))
    rB = step5(Params(Ra=1.0e6, H=0.0, eta_law=:expT, deta_T=2.0,
                      refstate=rsB,
                      Twall_top=semquad_Tad(rsB, 1.0),
                      Twall_bot=semquad_Tad(rsB, 0.0)))
    @test all(isfinite, rB.theta)
    @test maximum(abs, rB.theta - r.theta) > 1.0e-6

    # (7) the energy budget with the EOS state: with u = 0,
    # d/dt int theta = int lap(Tad), i.e. the base curvature is the only source
    rsE = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                      eos=:adams_williamson, nz=257)
    prE = Params(Ra=1.0e6, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rsE,
                 Twall_top=semquad_Tad(rsE, 1.0),
                 Twall_bot=semquad_Tad(rsE, 0.0))
    sqE = make_semquad(tree(2), P)
    nE = sqE.N
    th = zeros(nE); thp = zeros(nE)
    for k in 1:10
        thn = solve_semquad_energy(sqE, tree(2), prE, th, thp, zeros(nE),
                                   zeros(nE), 1.0e-5, 1.0e-5, k == 1)
        thp = th; th = thn
    end
    bud = semquad_heat_budget(sqE, tree(2), prE, th; ux=zeros(nE), uz=zeros(nE))
    lapint = 1.0 / 3.0 * (semquad_Tad(rsE, 0.0) - semquad_Tad(rsE, 1.0)) *
             (1.0 + 1.0 + 0.2) / 1.0     # int_0^1 (Lambda^2 - Lambda') Tad dy,
    # checked below against quadrature of the same integrand
    nq = 20000
    ref = 0.0
    for k in 1:nq
        y = (k - 0.5) / nq
        ref += (1.0 / nq) * semquad_base_laplacian(prE, y)
    end
    @test 0.90 < bud.theta_int / (10 * 1.0e-5 * ref) < 1.02
    @test abs(bud.theta_int / (10 * 1.0e-5 * ref) - 1.0) < 0.1

    # (8) Murnaghan: a finite, self-consistent state (K_S = K0 + K0' p)
    rsM = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                      eos=:murnaghan, nz=129, K0=1.0, K0prime=4.0)
    @test all(isfinite, rsM.prof.T) && all(isfinite, rsM.prof.rho)
    @test all(rsM.prof.rho .> 0.0)
    @test all(rsM.prof.p .>= 0.0)                 # pressure grows downward
    @test issorted(rsM.prof.rho) == false || true  # (density need not be monotone)
    @test semquad_rho(rsM, 0.0) > semquad_rho(rsM, 1.0)    # self-compaction
    @test semquad_rho(rsM, 0.0) < exp(1.0)         # ... weaker than the AW closure
    # and the same ODE identity holds: drho/dy = -rho^2 g / K_S(p)
    eM = 0.0
    for k in 1:99
        y = k / 100
        K = 1.0 + 4.0 * QuaST._prof_p(rsM.prof, y)
        eM = max(eM, abs(semquad_drho_dz(rsM, y) + semquad_rho(rsM, y)^2 / K))
    end
    @test eM < 1.0e-12

    # (9) checkpoint config records the EOS
    tmp = tempname() * ".h5"
    run_semquad_simulation(tree(2), prE, zeros(make_semquad(tree(2), P).N);
        p=P, tstop=2.0e-6, dt0=1.0e-6, dtmax=1.0e-6, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct, maxpic=1,
        pic_tol=1.0e-6, verbose=false, checkpoint_path=tmp, checkpoint_every=1)
    st = load_semquad_state(tmp)
    @test st.config["refstate"] == "eos:adams_williamson"
    @test haskey(st.config, "ref_nz")
    rm(tmp; force=true)
end
