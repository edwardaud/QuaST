using Test, QuaST, LinearAlgebra, Printf

# G171: pseudo-transient continuation (PTC) for the DP Stokes Newton.
#
# Formulation (Mulder & van Leer 1985; Kelley & Keyes 1998; Coffey, Kelley &
# Keyes 2003): [J + (1/delta) M_u] dx = r with the *velocity* mass matrix only,
# delta updated by switched evolution relaxation delta <- delta*|r_k|/|r_{k+1}|
# (capped), and PTC switched off once the residual has dropped by `ptc_eta`.
#
# The gate locks the two measured facts:
#  (a) on the fully yielded cold start (no plastic damper) PTC alone does *not*
#      converge, but it tames the runaway: the final scaled residual and the
#      velocity are far below the no-PTC run (which diverges to Vrms ~ 1e5);
#  (b) combined with the plastic damper (G170) the cold start converges.

@testset "G171 pseudo-transient continuation" begin
    qt = Quadtree(2; cells_x=3, cells_y=3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    th = [0.1 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[i] = 0.0)
    end
    mk(damp) = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                      yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                      yield_eps_min=1.0e-8, yield_damper=damp,
                      eta_min=1.0e-4, eta_max=10.0,
                      Twall_top=0.0, Twall_bot=1.0, bc=:free)
    pr0 = mk(0.0)
    prd = mk(1.0e-2)
    s_no = semquad_dp_stokes_newton(sq, qt, ss, pr0, th; tol=1.0e-8,
                                    maxit=15)
    s_ptc = semquad_dp_stokes_newton(sq, qt, ss, pr0, th; tol=1.0e-8,
                                     maxit=15, ptc=true)
    v_no = semquad_vrms(sq, qt, s_no.u[1:sq.N], s_no.u[(sq.N + 1):(2sq.N)])
    v_ptc = semquad_vrms(sq, qt, s_ptc.u[1:sq.N], s_ptc.u[(sq.N + 1):(2sq.N)])
    @printf("G171: no damper, no ptc: resid=%.2e Vrms=%.1e\n", s_no.resid, v_no)
    @printf("G171: no damper, ptc:    resid=%.2e Vrms=%.1e\n", s_ptc.resid,
            v_ptc)
    # (a) PTC tames the runaway but does not converge on its own
    @test !s_ptc.converged
    @test s_ptc.resid < 0.5 * s_no.resid
    @test v_ptc < 0.2 * v_no
    @test v_ptc < 1.0e5
    # (b) damper + PTC converges
    s_both = semquad_dp_stokes_newton(sq, qt, ss, prd, th; tol=1.0e-8,
                                      maxit=15, ptc=true)
    s_damp = semquad_dp_stokes_newton(sq, qt, ss, prd, th; tol=1.0e-8,
                                      maxit=15)
    @printf("G171: damper + ptc: conv=%s it=%d resid=%.1e\n", s_both.converged,
            s_both.iters, s_both.resid)
    @test s_both.converged
    @test s_both.resid < 1.0e-8
    @test s_both.iters <= 10
    @test s_damp.converged
    # both paths reach the same yielded state
    @test abs(semquad_vrms(sq, qt, s_both.u[1:sq.N],
                           s_both.u[(sq.N + 1):(2sq.N)]) -
              semquad_vrms(sq, qt, s_damp.u[1:sq.N],
                           s_damp.u[(sq.N + 1):(2sq.N)])) /
          semquad_vrms(sq, qt, s_damp.u[1:sq.N],
                       s_damp.u[(sq.N + 1):(2sq.N)]) < 1.0e-6
end
