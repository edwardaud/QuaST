using Test, MantleSEM, LinearAlgebra, SparseArrays, Random

# G218: the matrix-free Jacobi diagonal of the *constrained* velocity operator.
#
# `_semquad_diag_reduced` used to return `Cu' * diag(K)`, which is `diag(Cu'KC)`
# only on a conforming tree.  On a hanging-node tree the Lagrange interpolation
# weights are negative, a hanging row can cancel to <= 0, and the `1e-12*dmax`
# positivity clip turns that row into a 1e12 amplifier per Jacobi sweep (the
# production failure in docs/MG_REALISTIC_DIAGNOSIS.md §5.2: one V-cycle maps
# ||b|| = 1e3 to ||z|| = 1.6e170 and `:cg` throws `SingularException`).
#
# This gate pins the fix on a small hanging-node tree:
#   (a) the matrix-free diagonal is `diag(Cu' K Cu)` to round-off at *every*
#       level (the sparse path's `diag(A[k])` is the assembled reference);
#   (b) when the assembled diagonal is positive, no row of the matrix-free
#       diagonal sits on the `1e-12*dmax` clip;
#   (c) one V-cycle on the hanging tree is finite and non-amplifying
#       (||b - A z|| / ||b|| < 1);
#   (d) the `mg_vel_solver=:cg` saddle solve runs to convergence without the
#       `SingularException` the defect used to trigger;
#   and that `mg_diag=:lumped` keeps the old path available for A/B.

const _G218_Z2 = (x, y) -> 0.0
const _G218_ONE = (x, y) -> 1.0

# 2x2 base cells, one uniform refinement, then the lower-left 2x2 block refined
# once more and 2:1-balanced: a small tree with hanging nodes on all four sides
# of a refined block.
function _g218_hanging()
    qt = Quadtree(3; cells_x=2, cells_y=2)
    for lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    for lf in collect(qt.leaves)
        lf[1] == 1 && lf[2] < 2 && lf[3] < 2 && refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

@testset "G218 constrained matrix-free Jacobi diagonal" begin
    Random.seed!(20261111)
    qt = _g218_hanging()
    @test length(unique(lf[1] for lf in qt.leaves)) > 1
    @test max_face_level_jump(qt) <= 1

    for p in (2, 3)
        ss = make_semstokes(qt, p)
        eta = (x, y) -> exp(-log(1000.0) * (1 - y))
        asm = solve_semquad_stokes_var(ss, qt, eta, _G218_Z2, _G218_Z2,
                                       _G218_Z2, _G218_Z2; bc=:free,
                                       deviatoric=false, linear=:direct,
                                       assemble_only=true,
                                       mass_form=:boussinesq)
        mgS = MantleSEM.make_semquad_mg(ss, qt, p, :free, false, eta, asm.Kred;
                                        operator=:sparse,
                                        coarse_operator=:rediscretize,
                                        smoothing=:auto)
        mgF = MantleSEM.make_semquad_mg(ss, qt, p, :free, false, eta, asm.Kred;
                                        operator=:matrixfree, smoothing=:auto)
        @test mgF.operator === :matrixfree
        @test length(mgF.D) == mgF.geo.L + 1
        for k in 1:(mgF.geo.L + 1)
            dF = mgF.D[k]
            dS = mgS.D[k]                       # diag(assembled Cu' K Cu)
            # (a) exact constrained diagonal, to round-off
            @test maximum(abs.(dF .- dS)) <= 1.0e-12 * maximum(abs, dS)
            # (b) the assembled diagonal is positive and the matrix-free one is
            # never on the 1e-12 clip
            @test minimum(dS) > 0.0
            @test minimum(dS) / maximum(dS) > 1.0e-8
            @test minimum(dF) / maximum(dF) > 1.0e-8
            @test count(==(1.0e-12 * maximum(dF)), dF) == 0
            @test all(isfinite, dF)
        end
        # the fine-level reference is the explicitly assembled reduced operator
        @test maximum(abs.(mgF.D[end] .- Vector(diag(asm.Kred)))) <=
              1.0e-12 * maximum(abs, diag(asm.Kred))
        @info "G218 diagonal" p = p levels = mgF.geo.L + 1 nf = length(mgF.D[end]) rel =
            maximum(abs.(mgF.D[end] .- Vector(diag(asm.Kred)))) /
            maximum(abs, diag(asm.Kred))

        # ---- the A/B path is still reachable and stays positive ----------
        mgL = MantleSEM.make_semquad_mg(ss, qt, p, :free, false, eta, asm.Kred;
                                        operator=:matrixfree,
                                        mg_diag=:lumped)
        @test mgL.D[end] != mgF.D[end]
        @test all(>(0.0), mgL.D[end])
        @test_throws ArgumentError MantleSEM.make_semquad_mg(
            ss, qt, p, :free, false, eta, asm.Kred; operator=:matrixfree,
            mg_diag=:bogus)
        @test_throws ArgumentError solve_semquad_stokes_var(
            ss, qt, eta, _G218_Z2, _G218_Z2, _G218_Z2, _G218_Z2;
            bc=:free, linear=:mg, mg_diag=:bogus)
    end

    # ---- (c)+(d) at p=2: V-cycle and the inner-CG saddle solve -----------
    p = 2
    ss = make_semstokes(qt, p)
    eta = (x, y) -> exp(-log(1000.0) * (1 - y))
    asm = solve_semquad_stokes_var(ss, qt, eta, _G218_Z2, _G218_Z2, _G218_Z2,
                                   _G218_Z2; bc=:free, linear=:direct,
                                   assemble_only=true, mass_form=:boussinesq)
    mgF = MantleSEM.make_semquad_mg(ss, qt, p, :free, false, eta, asm.Kred;
                                    operator=:matrixfree, smoothing=:auto)
    nf = size(asm.Kred, 1)
    b = randn(nf)
    z = MantleSEM.semquad_mg_apply(mgF, b)
    @test all(isfinite, z)
    Az = zeros(Float64, nf)
    MantleSEM._semquad_mg_apply!(Az, z, mgF, mgF.geo.L + 1)
    amp = norm(b .- Az) / norm(b)
    @test isfinite(amp)
    @test amp < 1.0
    # the defective lumped diagonal is a different preconditioner on the same
    # tree (this 28-leaf tree does not activate the 1e-12 clip; the production
    # 8188-leaf state does, which is what `amp` above excludes)
    mgL = MantleSEM.make_semquad_mg(ss, qt, p, :free, false, eta, asm.Kred;
                                    operator=:matrixfree, mg_diag=:lumped)
    zL = MantleSEM.semquad_mg_apply(mgL, b)
    AzL = zeros(Float64, nf)
    MantleSEM._semquad_mg_apply!(AzL, zL, mgL, mgL.geo.L + 1)
    ampL = norm(b .- AzL) / norm(b)
    @test all(isfinite, zL)
    @test zL != z
    @info "G218 V-cycle" amp = amp amp_lumped = ampL

    # (d) `:cg` must not throw (it threw SingularException with the clipped D)
    fz = (x, y) -> sin(pi * x / qt.Lx) * sin(pi * y)
    uM, pM, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, _G218_Z2, fz,
                                               _G218_Z2, _G218_Z2; bc=:free,
                                               linear=:mg,
                                               mg_vel_solver=:cg)
    resM = LAST_SEMSTOKES_RESIDUAL[]
    cgM = LAST_SEMSTOKES_VEL_CG_ITERS[]
    @test resM < 1.0e-8
    @test cgM > 0
    @test all(isfinite, uM) && all(isfinite, pM)
    uD, pD, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, _G218_Z2, fz,
                                               _G218_Z2, _G218_Z2; bc=:free,
                                               linear=:direct)
    @test norm(uM - uD) / norm(uD) < 1.0e-6
    @info "G218 :cg saddle" res = resM cg_iters = cgM du = norm(uM - uD) / norm(uD)
end
