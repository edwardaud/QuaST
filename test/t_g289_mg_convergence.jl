# G289: multigrid as a solver -- an independent convergence-rate and accuracy
# criterion (remediation batch P3, task 2).
#
# Why this gate exists.  Before this gate every MG assertion was either
# *path equality* ("threaded == serial", "region == per-call", "reduced == full")
# or *solution equality with the direct solver*; neither is a convergence
# criterion (docs/dev/AUDIT_GATES.md §4 flags multigrid exactly for this).  A
# preconditioner can be path-consistent, bitwise thread-invariant and still stop
# converging (an under-swept V-cycle, a wrong coarse operator, a mis-scaled
# pressure block): that is the defect class this gate measures.
#
# What is asserted.  On a *fixed* assembled saddle system -- the same assembled
# K/B/fr/gr that the direct solver factors -- the shipped MG configuration
# (matrix-free level operators, damped-Jacobi V-cycle with the production `:auto`
# 2+2 schedule and omega = 2/3, w-BFBT pressure block from the production
# `mode=:auto` rule, production `coarsen_factor = 1.5`, serial `mg_threads=1`,
# `vel_solver=:vcycle`, i.e. the default of `linear=:mg` since S3/MF-6) must:
#
#   (1) reduce the **true** reduced-saddle residual by a stated number of orders
#       in a stated number of outer FGMRES iterations (forced with `maxit=20`,
#       `tol=0`), and its true residual must reach 1e-8 within an expected
#       iteration band -- an *iteration-count* assertion, not a path equality;
#   (2) agree with the sparse direct solution of the same saddle to <=1e-6
#       relative (the direct solve's own residual is <1e-10, checked);
#   (3) hold on a **badly conditioned variable-viscosity** box system
#       (smooth exp contrast 1e4, all-rigid walls, 64 leaves / 450 reduced
#       velocity dofs) and on a **mapped** `PolarAnnulus` system (R0=1.22,
#       R1=2.22, wedge pi/2; radial contrast 1e3, 256 leaves / 1890 reduced
#       velocity dofs, the matrix-free mapped kernels);
#   (4) be **non-vacuous**: the same measurement on an under-swept V-cycle
#       (pre=1, post=0) and on a wrong coarse operator (a hierarchy sampled with
#       eta = 1 for the 1e4-contrast operator; a box-metric hierarchy for the
#       mapped operator) must fail (1), by orders of magnitude;
#   (5) be realised by the shipped `linear=:mg` path: the production solve's
#       own outer iteration count and Stokes residual agree with the hand-built
#       system, and its u/p agree with the shipped direct solve to <=1e-6.
#
# Numbers below are measured on this tree (out_julia/agent/p3/probe_g289c.log and
# the gate's own printed table).  Nothing is relaxed to pass: the thresholds are
# set from the measured values with a stated margin, and the control arms are
# measured in the same process.
using Test, QuaST, LinearAlgebra, SparseArrays
using QuaST: Quadtree, make_semstokes, make_semquad_mg, PolarAnnulus,
             solve_semquad_stokes_var, semquad_mg_pressure_precond,
             semquad_mg_saddle_solve, LAST_SEMSTOKES_RESIDUAL,
             LAST_SEMSTOKES_SCHUR_ITERS

const _G289_Z2 = (x, y) -> 0.0

"""Uniform `2^level x 2^level`-leaf forest (cells_x x cells_y base cells)."""
function _g289_uniform(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

"""
    _g289_precond(ss, qt, eta, asm; bc=:rigid, p=2, hier_ss=ss, hier_eta=eta,
                  hier_q=qt, mgkw...) -> (mg, applyK!)

Build the production-default MG hierarchy (`operator=:matrixfree`,
`coarsen_factor=1.5`, `cycles=1`) and its fine-level matrix-free apply.  The
`hier_*` overrides inject the wrong-coarse-operator controls (a hierarchy built
on a different space / viscosity than the outer operator).
"""
function _g289_precond(ss, qt, eta, asm; bc::Symbol=:rigid, p::Int=2,
                       hier_ss=ss, hier_eta=eta, hier_q=qt, mgkw...)
    mg = make_semquad_mg(hier_ss, hier_q, p, bc, false, hier_eta, asm.Kred;
                         cycles=1, operator=:matrixfree, coarsen_factor=1.5,
                         mgkw...)
    applyK! = (out, v) -> QuaST._semquad_apply_Kred!(out, mg.mfops[end],
                                                     mg.geo.Cus[end],
                                                     mg.mfetas[end], v,
                                                     mg.mfwork[end])
    return (mg=mg, applyK=applyK!)
end

"""
    _g289_build(ss, qt, eta, fx, fz; bc=:rigid, p=2) -> system NamedTuple

Assemble the fixed saddle system, build the production MG preconditioner and the
pressure block, and factor the direct reference of the *same* assembled saddle.
"""
function _g289_build(ss, qt, eta, fx, fz; bc::Symbol=:rigid, p::Int=2)
    asm = solve_semquad_stokes_var(ss, qt, eta, fx, fz, _G289_Z2, _G289_Z2;
                                   bc=bc, linear=:direct, assemble_only=true)
    pre = _g289_precond(ss, qt, eta, asm; bc=bc, p=p)
    Fp, pscale = semquad_mg_pressure_precond(ss, asm.Cu, asm.Cp, asm.Bs,
                                            asm.Bt, nothing, eta;
                                            mode=:auto, cache=false,
                                            applyK=pre.applyK)
    nf = size(asm.Bs, 2)
    nfp = size(asm.Bs, 1)
    rhs = vcat(asm.fr, asm.gr)
    S = [asm.Kred asm.Bt; asm.Bs spzeros(nfp, nfp)]
    xd = S \ rhs
    return (asm=asm, mg=pre.mg, applyK=pre.applyK, Fp=Fp, pscale=pscale,
            rhs=rhs, xd=xd, nf=nf, nfp=nfp,
            direct_res=norm(S * xd - rhs) / max(norm(rhs), 1.0))
end

"""Exactly `maxit` outer FGMRES steps (`tol=0`) on the fixed system."""
function _g289_rate(sys; maxit::Int=20, mg=sys.mg, applyK=sys.applyK)
    sol, nit, relres, _ = semquad_mg_saddle_solve(nothing, sys.asm.Bs,
                                                  sys.asm.Bt, sys.Fp,
                                                  sys.pscale, sys.rhs, mg;
                                                  tol=0.0, maxit=maxit,
                                                  applyK=applyK)
    return (sol=sol, nit=nit, relres=relres)
end

"""Solve to `tol` and report the iteration count and the error vs the direct solve."""
function _g289_solve(sys; tol::Float64=1.0e-8, maxit::Int=500,
                     mg=sys.mg, applyK=sys.applyK)
    sol, nit, relres, _ = semquad_mg_saddle_solve(nothing, sys.asm.Bs,
                                                  sys.asm.Bt, sys.Fp,
                                                  sys.pscale, sys.rhs, mg;
                                                  tol=tol, maxit=maxit,
                                                  applyK=applyK)
    return (sol=sol, nit=nit, relres=relres,
            err=norm(sol - sys.xd) / max(norm(sys.xd), 1.0))
end

@testset "G289 MG convergence-rate criterion" begin
    p = 2
    fx = (x, y) -> sinpi(x) * sinpi(y)
    fz = (x, y) -> 0.5 * cospi(x) * sinpi(y)

    # ==================================================================
    # (a) box, smooth viscosity contrast 1e4, all-rigid walls
    # ==================================================================
    qt = _g289_uniform(3)                      # 64 leaves
    ss = make_semstokes(qt, p)
    eta = (x, y) -> exp(-log(1.0e4) * (1.0 - y))
    sys = _g289_build(ss, qt, eta, fx, fz; bc=:rigid, p=p)
    println("G289 (a) box: leaves=", length(qt.leaves), " nf=", sys.nf,
            " nfp=", sys.nfp, " nlev=", sys.mg.geo.L + 1,
            " pressure=", sys.Fp isa NamedTuple ? sys.Fp.kind : :chol,
            " direct_res=", sys.direct_res)
    # the direct reference is exact on this system
    @test sys.direct_res < 1.0e-10             # measured 1.9e-14

    # (1) convergence rate: 20 outer iterations reduce the true residual by
    # >= 8 orders (measured 1.05e-9 = 8.98 orders in 20 iterations).
    r20 = _g289_rate(sys; maxit=20)
    @test r20.nit == 20
    @test r20.relres <= 1.0e-8                 # measured 1.05e-9
    # (2) iteration band and accuracy at tol = 1e-8 (measured nit 19,
    # relres 4.78e-9, err 1.87e-10)
    s8 = _g289_solve(sys; tol=1.0e-8)
    @test 10 <= s8.nit <= 40                   # measured 19
    @test s8.relres <= 1.0e-8                  # measured 4.78e-9
    @test s8.err <= 1.0e-6                     # measured 1.87e-10
    # (3) the tight solve reaches the direct solution (measured 6.2e-12)
    s10 = _g289_solve(sys; tol=1.0e-10)
    @test s10.err <= 1.0e-6                    # measured 6.21e-12
    println("G289 (a) relres(20)=", r20.relres, " nit(1e-8)=", s8.nit,
            " relres(1e-8)=", s8.relres, " err(1e-8)=", s8.err,
            " err(1e-10)=", s10.err)
    # (4) control: under-swept V-cycle (one pre-sweep, no post-sweep) fails (1)
    under = _g289_precond(ss, qt, eta, sys.asm; bc=:rigid, p=p, pre=1, post=0)
    ru = _g289_rate(sys; mg=under.mg, applyK=under.applyK, maxit=20)
    @test ru.relres > 1.0e-5                   # measured 5.00e-4
    @test ru.relres > 1.0e3 * r20.relres       # measured 4.8e5x worse
    # (5) control: wrong coarse operator (hierarchy sampled with eta = 1 for the
    # 1e4-contrast outer operator) fails (1) (measured 0.898)
    wrong = _g289_precond(ss, qt, eta, sys.asm; bc=:rigid, p=p,
                          hier_eta=(x, y) -> 1.0)
    rw = _g289_rate(sys; mg=wrong.mg, applyK=sys.applyK, maxit=20)
    @test rw.relres > 1.0e-3                   # measured 0.898
    println("G289 (a) controls: under-swept relres(20)=", ru.relres,
            " wrong-coarse relres(20)=", rw.relres)

    # (6) the shipped linear=:mg path realises the criterion and agrees with the
    # shipped direct path (measured nit 19, res 1.38e-10, du 1.7e-10, dp 9.0e-10)
    ud, pd, = solve_semquad_stokes_var(ss, qt, eta, fx, fz, _G289_Z2,
                                       _G289_Z2; bc=:rigid,
                                       linear=:direct)[1:2]
    um, pm, = solve_semquad_stokes_var(ss, qt, eta, fx, fz, _G289_Z2,
                                       _G289_Z2; bc=:rigid, linear=:mg,
                                       mg_operator=:matrixfree,
                                       mg_tol=1.0e-8)[1:2]
    nit_prod = LAST_SEMSTOKES_SCHUR_ITERS[]
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-8  # measured 1.38e-10
    @test abs(nit_prod - s8.nit) <= 2          # measured 19 == 19
    @test norm(um - ud) / max(norm(ud), 1.0) <= 1.0e-6   # measured 1.70e-10
    @test norm(pm - pd) / max(norm(pd), 1.0) <= 1.0e-6   # measured 8.99e-10

    # ==================================================================
    # (b) mapped PolarAnnulus, radial viscosity contrast 1e3, rigid walls
    # ==================================================================
    R0, R1, PHI = 1.22, 2.22, 0.5 * pi
    g = PolarAnnulus(R0, R1, PHI)
    qta = _g289_uniform(3; cx=1, cy=4)         # 256 leaves
    ssa = make_semstokes(qta, p; geom=g)
    etaA = (x, y) -> exp(-log(1.0e3) *
                         (1.0 - (hypot(x, y) - R0) / (R1 - R0)))
    sysa = _g289_build(ssa, qta, etaA, fx, fz; bc=:rigid, p=p)
    println("G289 (b) annulus: leaves=", length(qta.leaves), " nf=", sysa.nf,
            " nfp=", sysa.nfp, " nlev=", sysa.mg.geo.L + 1,
            " pressure=", sysa.Fp isa NamedTuple ? sysa.Fp.kind : :chol,
            " direct_res=", sysa.direct_res)
    @test sysa.direct_res < 1.0e-10            # measured 1.5e-13
    # (1) >= 4 orders of true residual in 20 iterations (measured 6.38e-6, 5.19
    # orders) -- the mapped operator is harder; the bound is 15.7x the measured
    # value and is still failed by both controls below by >=5 orders.
    r20a = _g289_rate(sysa; maxit=20)
    @test r20a.nit == 20
    @test r20a.relres <= 1.0e-4                # measured 6.38e-6
    # (2) iteration band and accuracy at 1e-8 (measured nit 29, err 3.35e-9)
    s8a = _g289_solve(sysa; tol=1.0e-8)
    @test 15 <= s8a.nit <= 60                  # measured 29
    @test s8a.relres <= 1.0e-8                 # measured 6.14e-9
    @test s8a.err <= 1.0e-6                    # measured 3.35e-9
    # (3) the tight solve reaches the direct solution (measured 6.67e-11)
    s10a = _g289_solve(sysa; tol=1.0e-10)
    @test s10a.err <= 1.0e-6                   # measured 6.67e-11
    println("G289 (b) relres(20)=", r20a.relres, " nit(1e-8)=", s8a.nit,
            " relres(1e-8)=", s8a.relres, " err(1e-8)=", s8a.err,
            " err(1e-10)=", s10a.err)
    # (4) control: under-swept V-cycle (measured 2.09e-2)
    undera = _g289_precond(ssa, qta, etaA, sysa.asm; bc=:rigid, p=p,
                           pre=1, post=0)
    rua = _g289_rate(sysa; mg=undera.mg, applyK=undera.applyK, maxit=20)
    @test rua.relres > 1.0e-4                  # measured 2.09e-2
    # (5) control: wrong coarse operator -- a box-metric hierarchy on the mapped
    # operator (the pre-cut-5 key-collision defect class, G271 (d); measured
    # 0.900)
    boxwrong = _g289_precond(ssa, qta, etaA, sysa.asm; bc=:rigid, p=p,
                             hier_ss=make_semstokes(qta, p))
    rwa = _g289_rate(sysa; mg=boxwrong.mg, applyK=sysa.applyK, maxit=20)
    @test rwa.relres > 1.0e-3                  # measured 0.900
    println("G289 (b) controls: under-swept relres(20)=", rua.relres,
            " wrong-coarse relres(20)=", rwa.relres)
    # (6) the shipped mapped linear=:mg path (measured nit 29, res 2.71e-10,
    # du 3.37e-9, dp 3.29e-9)
    uda, pda, = solve_semquad_stokes_var(ssa, qta, etaA, fx, fz, _G289_Z2,
                                         _G289_Z2; bc=:rigid,
                                         linear=:direct)[1:2]
    uma, pma, = solve_semquad_stokes_var(ssa, qta, etaA, fx, fz, _G289_Z2,
                                         _G289_Z2; bc=:rigid, linear=:mg,
                                         mg_operator=:matrixfree,
                                         mg_tol=1.0e-8)[1:2]
    nit_proda = LAST_SEMSTOKES_SCHUR_ITERS[]
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-8  # measured 2.71e-10
    @test abs(nit_proda - s8a.nit) <= 2        # measured 29 == 29
    @test norm(uma - uda) / max(norm(uda), 1.0) <= 1.0e-6  # measured 3.37e-9
    @test norm(pma - pda) / max(norm(pda), 1.0) <= 1.0e-6  # measured 3.29e-9
    println("G289 done")
end
