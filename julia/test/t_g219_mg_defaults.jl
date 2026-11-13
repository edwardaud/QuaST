using Test, MantleSEM, LinearAlgebra, SparseArrays, Random

# G219: the production matrix-free MG defaults (task 1/2 of the MG optimisation
# pass) and the inert finest-level assembly removal.
#
# Pinned:
#   (a) the production default `mg_coarsen_factor=nothing` resolves to **1.5**
#       on the matrix-free path: on the 43-leaf graded tree below it removes the
#       226-dof near-duplicate level (5 -> 4 levels) and reaches the same
#       hierarchy as 2.0, while `1.0` restores the historical 5 levels and `2.0`
#       stays selectable;
#   (b) the outer iteration count of that default is 21 (+/- 1) at 1e-8 --
#       time is deliberately *not* gated (measured in docs/dev/MG.md);
#   (c) the `:auto` smoothing profile is uniform 2+2 on a non-top-heavy
#       hierarchy and uniform 3+3 on a top-heavy one (the measured winner on
#       P1: 65 outer iterations against 84 for 2+2 and 90 for the old taper);
#       `:taper` remains selectable and keeps omega=0.55;
#   (d) passing the caller's `Cu` into `_semquad_mg_geometry` is **bitwise
#       inert**: same levels, same Cus/Ps/D/mfetas, same V-cycle and the same
#       saddle solution;
#   (e) the direct path is untouched by the MG keywords, bitwise.
#
# The Chebyshev smoother was tried in this pass and rejected: it converges on a
# uniform low-contrast tree but diverges on F and P1 (the reduced operator has
# near-null modes outside the polynomial window), so it is not shipped.  The
# measurement is recorded in docs/dev/MG.md

const _G219_Z2 = (x, y) -> 0.0
const _G219_ONE = (x, y) -> 1.0

# 43-leaf graded tree (G183's blob construction at level 4): the production
# hierarchy has a 226-dof level over a 306-dof finest level, i.e. a 1.35x step.
function _g219_graded()
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for _ in 1:2
        for lf in collect(qt.leaves)
            lf[1] >= 4 && continue
            h = 1.0 / (1 << lf[1])
            x0 = lf[2] * h; y0 = lf[3] * h
            far = max((x0 - 0.35)^2 + (y0 - 0.35)^2,
                      (x0 + h - 0.35)^2 + (y0 - 0.35)^2,
                      (x0 - 0.35)^2 + (y0 + h - 0.35)^2,
                      (x0 + h - 0.35)^2 + (y0 + h - 0.35)^2)
            far < 0.30^2 && refine!(qt, lf)
        end
        balance!(qt)
    end
    return qt
end

# Mixed-level tree with levels {2,3}: its finest coarsening step keeps >50% of
# the dofs, so the auto smoothing profile is the top-heavy one.
function _g219_topheavy()
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for (l, ix, iy) in collect(qt.leaves)
        (l == 2 && ix % 2 == 0 && iy % 2 == 0) && refine!(qt, (l, ix, iy))
    end
    balance!(qt)
    return qt
end

@testset "G219 matrix-free MG defaults" begin
    Random.seed!(20261111)
    qt = _g219_graded()
    ss = make_semstokes(qt, 2)
    asm = solve_semquad_stokes_var(ss, qt, _G219_ONE, _G219_Z2, _G219_Z2,
                                   _G219_Z2, _G219_Z2; bc=:free,
                                   linear=:direct, assemble_only=true)
    Cu = SparseMatrixCSC{Float64,Int}(sparse(asm.Cu))

    # ---- (a) factor resolution: 1.0 / default 1.5 / 2.0 -------------------
    empty!(MantleSEM.SEMQUAD_MG_CACHE)
    g10 = MantleSEM._semquad_mg_geometry(ss, qt, 2, :free, false;
                                         coarsen_factor=1.0)
    g15 = MantleSEM._semquad_mg_geometry(ss, qt, 2, :free, false;
                                         coarsen_factor=1.5)
    g20 = MantleSEM._semquad_mg_geometry(ss, qt, 2, :free, false;
                                         coarsen_factor=2.0)
    @test g10.nfs == [6, 30, 126, 226, 306]
    @test g15.nfs == [6, 30, 126, 306]
    @test g20.nfs == [6, 30, 126, 306]
    @test length(g15.nfs) < length(g10.nfs)

    # production default (no mg_coarsen_factor) == explicit 1.5; the direct
    # path is not affected by the MG keywords at all.
    z2 = _G219_Z2
    fz = (x, y) -> sinpi(x) * sinpi(y)
    u0, p0, _, _, _ = solve_semquad_stokes_var(ss, qt, _G219_ONE, z2, fz, z2,
                                               z2; bc=:free, linear=:direct)
    u0b, p0b, _, _, _ = solve_semquad_stokes_var(ss, qt, _G219_ONE, z2, fz, z2,
                                                 z2; bc=:free, linear=:direct,
                                                 mg_coarsen_factor=1.0,
                                                 mg_smoothing=:taper,
                                                 mg_pre=1, mg_post=1)
    @test u0 == u0b && p0 == p0b
    u1, p1, _, _, _ = solve_semquad_stokes_var(ss, qt, _G219_ONE, z2, fz, z2,
                                               z2; bc=:free, linear=:mg,
                                               mg_tol=1.0e-8)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-6
    @test abs(LAST_SEMSTOKES_SCHUR_ITERS[] - 21) <= 1
    @test LAST_SEMSTOKES_MG_PROFILE[] == (:uniform, 2.0 / 3.0)
    @test norm(u1 - u0) / norm(u0) < 1.0e-6
    # explicit 1.0 still works and is the historical hierarchy
    empty!(MantleSEM.SEMQUAD_MG_CACHE); MantleSEM.SEMQUAD_PRESSURE_CACHE[] = nothing
    solve_semquad_stokes_var(ss, qt, _G219_ONE, z2, fz, z2, z2;
                             bc=:free, linear=:mg, mg_tol=1.0e-8,
                             mg_coarsen_factor=1.0)
    @test abs(LAST_SEMSTOKES_SCHUR_ITERS[] - 14) <= 1
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-6

    # ---- (b) smoothing profile: top-heavy -> uniform 3+3 ----------------
    qh = _g219_topheavy()
    ssh = make_semstokes(qh, 2)
    gh = MantleSEM._semquad_mg_geometry(ssh, qh, 2, :free, false;
                                        coarsen_factor=1.5)
    @test gh.nfs[end - 1] > 0.5 * gh.nfs[end]
    mgh = MantleSEM.make_semquad_mg(ssh, qh, 2, :free, false, _G219_ONE,
                                    nothing; cycles=1, operator=:matrixfree,
                                    coarsen_factor=1.5)
    @test mgh.pre == fill(3, gh.L + 1)
    @test mgh.omega == 2.0 / 3.0
    @test MantleSEM.LAST_SEMSTOKES_MG_PROFILE[] == (:uniform, 2.0 / 3.0)
    # the taper is still selectable and still pairs with omega = 0.55
    mgt = MantleSEM.make_semquad_mg(ssh, qh, 2, :free, false, _G219_ONE,
                                    nothing; cycles=1, operator=:matrixfree,
                                    coarsen_factor=1.5, smoothing=:taper)
    @test mgt.omega == 0.55
    @test mgt.pre == reverse([min(1 << (k ÷ 2), 8) for k in 1:(gh.L + 1)])

    # ---- (c) passing Cu is bitwise inert --------------------------------
    mgA = MantleSEM.make_semquad_mg(ss, qt, 2, :free, false, _G219_ONE,
                                    asm.Kred; cycles=1, operator=:matrixfree,
                                    coarsen_factor=1.5)
    mgB = MantleSEM.make_semquad_mg(ss, qt, 2, :free, false, _G219_ONE,
                                    asm.Kred; cycles=1, operator=:matrixfree,
                                    coarsen_factor=1.5, fine_Cu=Cu)
    @test mgA.geo.nfs == mgB.geo.nfs
    @test mgA.geo.Cus == mgB.geo.Cus
    @test mgA.geo.Ps == mgB.geo.Ps
    @test mgA.geo.free_rows == mgB.geo.free_rows
    @test mgA.D == mgB.D
    @test mgA.mfetas == mgB.mfetas
    @test mgA.pre == mgB.pre && mgA.post == mgB.post
    b = randn(mgA.geo.nfs[end])
    @test MantleSEM.semquad_mg_apply(mgA, b) ==
          MantleSEM.semquad_mg_apply(mgB, b)
    # ... and the whole saddle solve is bitwise
    rhs = vcat(asm.fr, asm.gr)
    mfapplyK = (out, v) -> MantleSEM._semquad_apply_Kred!(out,
                                                           mgB.mfops[end],
                                                           mgB.geo.Cus[end],
                                                           mgB.mfetas[end], v,
                                                           mgB.mfwork[end])
    Fp, pscale = MantleSEM.semquad_mg_pressure_precond(ss, asm.Cu, asm.Cp,
                                                       asm.Bs, asm.Bt, nothing,
                                                       _G219_ONE; mode=:auto,
                                                       cache=false,
                                                       applyK=mfapplyK)
    solA, nitA, resA, _ = MantleSEM.semquad_mg_saddle_solve(nothing, asm.Bs,
                                                            asm.Bt, Fp, pscale,
                                                            rhs, mgA;
                                                            tol=1.0e-10,
                                                            maxit=200,
                                                            applyK=mfapplyK)
    solB, nitB, resB, _ = MantleSEM.semquad_mg_saddle_solve(nothing, asm.Bs,
                                                            asm.Bt, Fp, pscale,
                                                            rhs, mgB;
                                                            tol=1.0e-10,
                                                            maxit=200,
                                                            applyK=mfapplyK)
    @test solA == solB
    @test nitA == nitB && resA == resB
end
