using Test, MantleSEM, LinearAlgebra, Random

# G177 / S3-MF-4 (updated in MF-6): the `linear=:mg` saddle solve, its two
# velocity-block solvers, and the TRUE residual criterion.
#
# The `linear=:mg` saddle solve is the deal.II/ASPECT scheme:
#   * outer:  `fgmres_restarted` on the reduced saddle [Kred Bt; Bs 0],
#   * inner:  the fixed `mg_cycles` V-cycles (`mg_vel_solver=:vcycle`, the
#             DEFAULT since S3/MF-6), or PCG on `Kred` preconditioned by the
#             geometric MG hierarchy (`mg_vel_solver=:cg`, the MF-4 scheme,
#             still available and covered below),
#   * pressure: the same three-factor w-BFBT / mass Schur block as before,
#   * stopping: the **true** reduced-saddle residual, never the preconditioned
#             one.  Because the inner CG iteration count depends on the
#             right-hand side, the outer method must be *flexible* there:
#             standard GMRES would apply a stale preconditioner to the final
#             correction.
#
# MF-6 switched the default to `:vcycle` after measuring 40 outer iterations /
# 40 V-cycles against 25 outer / 84 inner-CG iterations on the 1024-leaf blob
# tree (33% faster per solve); this gate now asserts the default runs no inner
# CG and that the CG variant still works and agrees.
#
# Acceptance (S3 design, MF-4): a full Stokes solve at p=4 L4 (256 leaves,
# 8450 velocity dofs) with `linear=:mg, mg_operator=:matrixfree` must take
# < 0.44 s (the direct LU is 0.62 s on the design machine).

const _G177_Z2 = (x, y) -> 0.0
const _G177_ONE = (x, y) -> 1.0

function _g177_uniform(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# Interleaved timing: the machine this runs on also has other work, so take the
# best of `n` samples of every variant rather than one noisy number.
function _g177_times(fs::Vector{<:Function}, n::Int)
    for f in fs
        f()
    end
    ts = [Float64[] for _ in fs]
    for _ in 1:n, (i, f) in enumerate(fs)
        push!(ts[i], @elapsed f())
    end
    return ts
end

@testset "G177 MF-4 CG velocity block + true-residual FGMRES" begin
    Random.seed!(20251106)

    qt = _g177_uniform(4)
    ss = make_semstokes(qt, 4)
    Lx = qt.Lx
    fz = (x, y) -> sin(pi * x / Lx) * sin(pi * y)
    eta = _G177_ONE

    solvekw(kw) = solve_semquad_stokes_var(ss, qt, eta, _G177_Z2, fz, _G177_Z2,
                                           _G177_Z2; bc=:free, kw...)

    # ---- the direct path is still the reference ---------------------------
    uD, _, _, _, _ = solvekw((linear=:direct,))
    resD = LAST_SEMSTOKES_RESIDUAL[]
    @test resD < 1.0e-11
    @test LAST_SEMSTOKES_SCHUR_ITERS[] == 0
    @test LAST_SEMSTOKES_VEL_CG_ITERS[] == 0

    # ---- default = matrix-free + fixed V-cycle velocity solve -------------
    # (MF-6 changed the default from `mg_vel_solver=:cg` to `:vcycle`: 40 outer
    # iterations / 40 V-cycles instead of 25 outer / 84 inner-CG iterations,
    # 33% faster per solve on the 1024-leaf blob tree.  The CG+MG scheme of
    # MF-4 is still available, still flexible-FGMRES-safe, and is exercised
    # explicitly below.)
    uM, _, _, _, _ = solvekw((linear=:mg,))
    resM = LAST_SEMSTOKES_RESIDUAL[]
    itM = LAST_SEMSTOKES_SCHUR_ITERS[]
    cgM = LAST_SEMSTOKES_VEL_CG_ITERS[]
    @test resM < 1.0e-9                  # true reduced-saddle residual
    @test itM > 0
    @test cgM == 0                       # the default runs no inner CG
    @test norm(uM - uD) / norm(uD) < 1.0e-8
    @info "G177 defaults" outer_iters = itM inner_cg_iters = cgM true_res = resM

    # the default really is the matrix-free hierarchy with the vcycle solver
    uM2, _, _, _, _ = solvekw((linear=:mg, mg_operator=:matrixfree,
                               mg_vel_solver=:vcycle))
    @test uM2 == uM
    @test LAST_SEMSTOKES_SCHUR_ITERS[] == itM
    # ... and the MF-4 CG velocity solve is still available and equivalent
    uCg, _, _, _, _ = solvekw((linear=:mg, mg_vel_solver=:cg))
    itCg = LAST_SEMSTOKES_SCHUR_ITERS[]
    cgCg = LAST_SEMSTOKES_VEL_CG_ITERS[]
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    @test cgCg > 0                       # the inner CG really ran
    @test norm(uCg - uD) / norm(uD) < 1.0e-8
    @info "G177 :cg variant" outer_iters = itCg inner_cg_iters = cgCg
    # ... and the sparse hierarchy is still reachable and equivalent
    uS, _, _, _, _ = solvekw((linear=:mg, mg_operator=:sparse))
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    @test norm(uS - uD) / norm(uD) < 1.0e-8
    @test norm(uS - uM) / norm(uM) < 1.0e-8
    # ... and the historical fixed V-cycle velocity solve is the default
    uV, _, _, _, _ = solvekw((linear=:mg, mg_vel_solver=:vcycle))
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    @test LAST_SEMSTOKES_VEL_CG_ITERS[] == 0
    @test norm(uV - uD) / norm(uD) < 1.0e-8

    # ---- true residual, not the preconditioned one ------------------------
    # With a deliberately loose outer tolerance the solve must stop on the true
    # residual: the returned solution's reduced residual has to be below tol
    # (the old code stopped on |g| of the preconditioned system and could be
    # several digits off).  Also check the iteration cap really caps.
    uL, _, _, _, _ = solvekw((linear=:mg, mg_tol=1.0e-6))
    resL = LAST_SEMSTOKES_RESIDUAL[]
    itL = LAST_SEMSTOKES_SCHUR_ITERS[]
    @test resL < 1.0e-6
    @test resL > 0.0
    @test itL < itM                     # looser tolerance, fewer iterations
    uC, _, _, _, _ = solvekw((linear=:mg, mg_maxit=3))
    resC = LAST_SEMSTOKES_RESIDUAL[]
    @test LAST_SEMSTOKES_SCHUR_ITERS[] == 3
    # the true residual of the capped (unconverged) solve is honestly reported
    @test resC > 1.0e-6
    @info "G177 true-residual criterion" loose_iters = itL loose_res = resL capped_res = resC

    # ---- timing at p=4 L4 (256 leaves) -----------------------------------
    # acceptance: full solve (assembly included) < 0.44 s, direct is the
    # baseline.  Best of 6 interleaved samples.
    dkw = (linear=:direct,)
    mkw = (linear=:mg, mg_operator=:matrixfree)                       # vcycle
    vkw = (linear=:mg, mg_operator=:matrixfree, mg_vel_solver=:cg)
    ts = _g177_times([() -> solvekw(dkw), () -> solvekw(mkw),
                      () -> solvekw(vkw)], 6)
    tD = minimum(ts[1]); tM = minimum(ts[2]); tV = minimum(ts[3])
    _, _, _, _, _ = solvekw(mkw)
    itM = LAST_SEMSTOKES_SCHUR_ITERS[]
    cgM = LAST_SEMSTOKES_VEL_CG_ITERS[]
    resM = LAST_SEMSTOKES_RESIDUAL[]
    @info "G177 timing p=4 L4" direct_s = tD mg_default_s = tM mg_cg_s = tV ratio =
        tD / tM outer = itM inner_cg = cgM true_res = resM
    @test tM < 0.44
    # A regression guard, not the design criterion: the absolute number above is
    # the acceptance; this only catches a catastrophic slowdown of the MG path.
    @test tM < 1.5 * tD
    @test tV < 0.44

    # ---- SPD + CG on a strongly graded tree ------------------------------
    # The inner CG needs `Kred` SPD; on a wall-refined hanging-node tree with a
    # 16384x viscosity contrast the operator is only SPD up to the hanging
    # constraints, so this is the case where CG could stall.  It must converge
    # and give the same velocity as the direct solve.
    qtw = Quadtree(4; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(qtw.leaves)
        refine!(qtw, lf)
    end
    balance!(qtw)
    refine_wall_to_min!(qtw, 4; strip=0.1)
    balance!(qtw)
    @test length(unique(lf[1] for lf in qtw.leaves)) > 1
    ssw = make_semstokes(qtw, 2)
    etaw = (x, y) -> exp(-log(16384.0) * (1 - y))
    fzw = (x, y) -> sin(pi * x / qtw.Lx) * sin(pi * y)
    uwD, _, _, _, _ = solve_semquad_stokes_var(ssw, qtw, etaw, _G177_Z2, fzw,
                                               _G177_Z2, _G177_Z2;
                                               bc=:free, linear=:direct)
    uwM, _, _, _, _ = solve_semquad_stokes_var(ssw, qtw, etaw, _G177_Z2, fzw,
                                               _G177_Z2, _G177_Z2;
                                               bc=:free, linear=:mg)
    resw = LAST_SEMSTOKES_RESIDUAL[]
    itw = LAST_SEMSTOKES_SCHUR_ITERS[]
    cgw = LAST_SEMSTOKES_VEL_CG_ITERS[]
    @test resw < 1.0e-9
    @test itw < 200
    @test cgw == 0                       # default is the vcycle velocity solve
    @test norm(uwM - uwD) / norm(uwD) < 1.0e-8
    # ... and the CG velocity solve must not stall on the same tree
    uwC, _, _, _, _ = solve_semquad_stokes_var(ssw, qtw, etaw, _G177_Z2, fzw,
                                               _G177_Z2, _G177_Z2;
                                               bc=:free, linear=:mg,
                                               mg_vel_solver=:cg)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    @test LAST_SEMSTOKES_VEL_CG_ITERS[] > 0
    @test norm(uwC - uwD) / norm(uwD) < 1.0e-8
    @info "G177 wall tree" leaves = length(qtw.leaves) outer = itw inner_cg = cgw true_res =
        resw du = norm(uwM - uwD) / norm(uwD)

    # ---- the Newton MG saddle exposes mg_operator too --------------------
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=log(10.0),
                eta_c=log(2.0), Twall_top=0.0, Twall_bot=1.0, bc=:free)
    qtn = _g177_uniform(3)
    ssn = make_semstokes(qtn, 2)
    th = [0.05 * sin(pi * x / qtn.Lx) * sin(pi * y) for (x, y) in ssn.su.coords]
    for (i, (x, y)) in enumerate(ssn.su.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th[i] = 0.0)
    end
    step_d = semquad_steady_step(make_semquad(qtn, 2), qtn, ssn, pr, th;
                                 linear=:direct)
    step_m = semquad_steady_step(make_semquad(qtn, 2), qtn, ssn, pr, th;
                                 linear=:mg)
    # The Newton JVP is solved to 1e-8, so the MG saddle and the direct one
    # agree to that level, not to round-off.
    @test norm(step_m.u - step_d.u) / norm(step_d.u) < 1.0e-6
    @test norm(step_m.theta - step_d.theta) / norm(step_d.theta) < 1.0e-6
    @info "G177 newton saddle" du = norm(step_m.u - step_d.u) / norm(step_d.u)
end
