# G163: the Drucker-Prager Kp saddle block — assembled and checked against finite differences.
using Test, QuaST, LinearAlgebra, SparseArrays

# G163: Drucker-Prager pressure-derivative saddle block `Kp`.
#
# The DP viscosity depends on the dynamic pressure through the yield stress,
#     tau_y = min(c cos(phi) + p sin(phi), tau_max),  eta_pl = tau_y/(2 edot),
# so the momentum residual `R_m = K(eta(T,edot,p)) u - B'p - f` has a pressure
# derivative in the velocity rows,
#     (Kp)_{i,a} = int 2 (deta/dp) eps(u):eps(v_i) phi_a,
# and none in the constraint rows: the Newton saddle is non-symmetric.
#
# The gate verifies the assembled block against central differences of the
# production residual in two regimes:
#   * a constant `deta/dp = alpha` (any smooth p field): the derivative is
#     exactly alpha, so the check is an assembly/integration test;
#   * the real DP law in its plastic branch (linear in p, so central differences
#     are exact up to round-off): the check is a rheology-wiring test.
# It also pins the zero cases (phi = 0, viscous branch, tau_max cap, eta clip),
# argument validation, and that the resulting non-symmetric saddle factorises
# and solves with a direct LU.

@testset "G163 Drucker-Prager Kp block" begin

    function make_tree(uniform::Bool)
        qt = Quadtree(6; cells_x=2, cells_y=1)
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        if !uniform
            # refine two leaves to create 2:1 hanging nodes on both space
            lf1 = first(qt.leaves)
            refine!(qt, lf1)
            balance!(qt)
            cand = [lf for lf in qt.leaves if lf[1] == 2][1]
            refine!(qt, cand)
            balance!(qt)
        end
        return qt
    end

    # DP parameters in which the plastic branch is active and unclipped for
    # p in [1, 3] and edot = 0.5: eta_pl ~ 1.36..2.31 while eta_visc = 1e6.
    pr = Params(Ra=1.0, H=0.0, eta_law=:const, eta_const=1.0e6,
                yield_c=1.0, yield_phi=0.5, yield_taumax=Inf,
                eta_min=1.0e-6, eta_max=1.0e9)
    T0 = 0.5
    edot0 = 0.5

    for uniform in (true, false)
        qt = make_tree(uniform)
        ss = make_semstokes(qt, 2)
        sq = ss.su
        sp = ss.sp
        pleaves = Set(sp.leaves)
        z2 = (x, y) -> 0.0
        pateval(pf, x, y) = QuaST._semquad_eval_at(sp, pf, x, y, pleaves)

        # u = (y, 0) is a pure shear: edot = 0.5 exactly at every quadrature
        # point, so the viscosity below can use the constant edot0.
        u = zeros(Float64, 2 * sq.N)
        for (g, (x, y)) in enumerate(sq.coords)
            u[g] = y
        end
        pbase = [1.5 + 0.3 * sin(pi * x) * sin(pi * y) for (x, y) in sp.coords]
        dpfull = [0.5 * cos(pi * x) * y for (x, y) in sp.coords]

        etafun(pf) = (x, y) -> semquad_visco_plastic_eta(T0, y, edot0,
                                                        pateval(pf, x, y), pr)
        dp_coef = (x, y, ed, p) -> semquad_deta_dp(T0, y, ed, p, pr)

        sys = solve_semquad_stokes_var(ss, qt, etafun(pbase), z2, z2, z2, z2;
                                       bc=:free, deviatoric=false,
                                       assemble_only=true, dp_u=u,
                                       dp_pfield=pbase, dp_coef=dp_coef)
        @test sys.Kp !== nothing
        @test size(sys.Kp) == (2 * sq.N, sp.N)
        @test norm(sys.Kp) > 0.0
        @test sys.Kp_red !== nothing
        @test size(sys.Kp_red) == (size(sys.Bs, 2), size(sys.Cp, 2))

        # ---- (1) full block against central differences of K(eta(p)) u ----
        h = 1.0e-6
        sysp = solve_semquad_stokes_var(ss, qt, etafun(pbase .+ h .* dpfull),
                                        z2, z2, z2, z2; bc=:free,
                                        deviatoric=false, assemble_only=true)
        sysm = solve_semquad_stokes_var(ss, qt, etafun(pbase .- h .* dpfull),
                                        z2, z2, z2, z2; bc=:free,
                                        deviatoric=false, assemble_only=true)
        fd = (sysp.K * u - sysm.K * u) ./ (2h)
        an = sys.Kp * dpfull
        @test norm(fd - an) <= 1.0e-6 * norm(an)

        # ---- (2) reduced block: Cu'(Kp Cp - Bm'Cp) = Kp_red + Bt ----
        # A random reduced direction maps to a full pressure perturbation
        # through Cp (this keeps hanging pressure constraints consistent).
        dph = randn(size(sys.Cp, 2))
        dpfull2 = sys.Cp * dph
        sysp2 = solve_semquad_stokes_var(ss, qt,
                                         etafun(pbase .+ h .* dpfull2),
                                         z2, z2, z2, z2; bc=:free,
                                         deviatoric=false, assemble_only=true)
        sysm2 = solve_semquad_stokes_var(ss, qt,
                                         etafun(pbase .- h .* dpfull2),
                                         z2, z2, z2, z2; bc=:free,
                                         deviatoric=false, assemble_only=true)
        fdr = sys.Cu' * ((sysp2.K * u - sysm2.K * u) ./ (2h)) + sys.Bt * dph
        anr = (sys.Kp_red + sys.Bt) * dph
        @test norm(fdr - anr) <= 1.0e-6 * norm(anr)

        # ---- (3) constant deta/dp = alpha: pure assembly/integration check ----
        alpha = 0.7
        coef_a = (x, y, ed, p) -> alpha
        etafun_a(pf) = (x, y) -> 1.0 + alpha * pateval(pf, x, y)
        sysa = solve_semquad_stokes_var(ss, qt, etafun_a(pbase), z2, z2, z2, z2;
                                        bc=:free, deviatoric=false,
                                        assemble_only=true, dp_u=u,
                                        dp_pfield=pbase, dp_coef=coef_a)
        sysap = solve_semquad_stokes_var(ss, qt, etafun_a(pbase .+ h .* dpfull),
                                         z2, z2, z2, z2; bc=:free,
                                         deviatoric=false, assemble_only=true)
        sysam = solve_semquad_stokes_var(ss, qt, etafun_a(pbase .- h .* dpfull),
                                         z2, z2, z2, z2; bc=:free,
                                         deviatoric=false, assemble_only=true)
        fda = (sysap.K * u - sysam.K * u) ./ (2h)
        ana = sysa.Kp * dpfull
        @test norm(fda - ana) <= 1.0e-6 * norm(ana)

        # ---- (4) zero cases ----
        # phi = 0: no pressure dependence in the yield stress
        pr0 = Params(Ra=1.0, H=0.0, eta_law=:const, eta_const=1.0e6,
                     yield_c=1.0, yield_phi=0.0, yield_taumax=Inf,
                     eta_min=1.0e-6, eta_max=1.0e9)
        s0 = solve_semquad_stokes_var(ss, qt, etafun(pbase), z2, z2, z2, z2;
                                      bc=:free, deviatoric=false,
                                      assemble_only=true, dp_u=u,
                                      dp_pfield=pbase,
                                      dp_coef=(x, y, ed, p) ->
                                          semquad_deta_dp(T0, y, ed, p, pr0))
        @test norm(s0.Kp) == 0.0
        @test norm(s0.Kp_red) == 0.0
        # viscous branch: eta_visc = 1 < eta_pl, so deta/dp = 0
        prv = Params(Ra=1.0, H=0.0, eta_law=:const, eta_const=1.0,
                     yield_c=1.0, yield_phi=0.5, yield_taumax=Inf,
                     eta_min=1.0e-6, eta_max=1.0e9)
        sv = solve_semquad_stokes_var(ss, qt, etafun(pbase), z2, z2, z2, z2;
                                      bc=:free, deviatoric=false,
                                      assemble_only=true, dp_u=u,
                                      dp_pfield=pbase,
                                      dp_coef=(x, y, ed, p) ->
                                          semquad_deta_dp(T0, y, ed, p, prv))
        @test norm(sv.Kp) == 0.0
        # tau_max cap: yield stress is capped, so deta/dp = 0
        prc = Params(Ra=1.0, H=0.0, eta_law=:const, eta_const=1.0e6,
                     yield_c=1.0, yield_phi=0.5, yield_taumax=1.0,
                     eta_min=1.0e-6, eta_max=1.0e9)
        sc = solve_semquad_stokes_var(ss, qt, etafun(pbase), z2, z2, z2, z2;
                                      bc=:free, deviatoric=false,
                                      assemble_only=true, dp_u=u,
                                      dp_pfield=pbase,
                                      dp_coef=(x, y, ed, p) ->
                                          semquad_deta_dp(T0, y, ed, p, prc))
        @test norm(sc.Kp) == 0.0
        # eta clipped at eta_min: the clip sets the derivative to zero.  With
        # p=1.5, phi=0.5, edot=0.5 the plastic branch sits at eta_pl ~ 0.72, so
        # a floor of 1.0 clips it.
        prm = Params(Ra=1.0, H=0.0, eta_law=:const, eta_const=1.0e6,
                     yield_c=1.0e-9, yield_phi=0.5, yield_taumax=Inf,
                     eta_min=1.0, eta_max=1.0e9)
        @test semquad_deta_dp(T0, 0.5, edot0, 1.5, prm) == 0.0

        # ---- (5) non-symmetric saddle is factorisable by the direct LU ----
        nf = size(sys.Kred, 1)
        nfp = size(sys.Bs, 1)
        S = [sys.Kred (sys.Bt + sys.Kp_red); sys.Bs spzeros(Float64, nfp, nfp)]
        S0 = [sys.Kred sys.Bt; sys.Bs spzeros(Float64, nfp, nfp)]
        @test norm(S - S') > 100 * norm(S0 - S0')
        F = lu(S)
        b = randn(nf + nfp)
        xs = F \ b
        # The 2b saddle is ill-conditioned (the coupled-Newton prototype measured
        # cond(J) ~ 1e8), so demand a backward-stable residual, not 1e-14.
        @test norm(S * xs - b) <= 1.0e-7 * norm(b)
    end

    # ---- argument validation ----
    qt = make_tree(true)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    sp = ss.sp
    z2 = (x, y) -> 0.0
    etaf = (x, y) -> 1.0
    u = zeros(Float64, 2 * sq.N)
    p0 = zeros(Float64, sp.N)
    co = (x, y, ed, p) -> 1.0
    @test_throws ArgumentError solve_semquad_stokes_var(
        ss, qt, etaf, z2, z2, z2, z2; assemble_only=true, dp_u=u)
    @test_throws ArgumentError solve_semquad_stokes_var(
        ss, qt, etaf, z2, z2, z2, z2; assemble_only=true, dp_u=u,
        dp_pfield=zeros(Float64, sp.N + 1), dp_coef=co)
    @test_throws ArgumentError solve_semquad_stokes_var(
        ss, qt, etaf, z2, z2, z2, z2; assemble_only=true, dp_u=zeros(Float64, 1),
        dp_pfield=p0, dp_coef=co)
    @test_throws ArgumentError solve_semquad_stokes_var(
        ss, qt, etaf, z2, z2, z2, z2; assemble_only=true, dp_u=u,
        dp_pfield=p0, dp_coef=co, deviatoric=true)
end
