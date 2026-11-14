# G226: the exact restricted BFBT diagonal — sparse reference, positivity and solve accuracy.
using Test, QuaST, LinearAlgebra, SparseArrays, Random

# G226: the exact restricted BFBT diagonal `D = diag(Cu' M_u(sqrt(eta)) Cu)`.
#
# The historical BFBT weight was the *lumped* restriction `Cu' diag(M_u)`
# (`mg_pressure_bfbt_d=:leaf`): it restricts the leaf mass diagonal but misses
# the hanging-node cross terms of `diag(C' M C)`, and on the production `F`
# tree its raw diagonal has 16 non-positive entries which the `1e-12*max|d|`
# clip turns into a 1e12 contrast.  The exact diagonal removed the outliers
# (0 non-positive, contrast 5.9e5) and measured 81 -> 71 outer iterations on
# `F`; docs/dev/MG.md ships it on the matrix-free path as
# `mg_pressure_bfbt_d=:exact` (the `:leaf` A/B and the `:matrix` sparse
# reference stay selectable).
#
# This gate pins, on a small hanging-node tree with a 1000x viscosity contrast:
#   (a) the leaf-local exact diagonal equals the sparse reference
#       `diag(Cux' Mv Cux + Cuz' Mv Cuz)` to round-off (it is not bitwise: the
#       leaf-local sum has a different order);
#   (b) no entry is clipped and every entry is positive (the exact weight needs
#       no clip; the clip must stay inert);
#   (c) the exact and lumped diagonals differ -- the cross terms are real on a
#       hanging-node tree, so (a) is not vacuous;
#   (d) the construction is deterministic (two calls are bitwise equal);
#   (e) both weights run a saddle solve to `relres <= 1e-6`, and the exact one's
#       accuracy against the equilibrated direct solve stays inside the
#       documented MG class `|du|/|u| <= 2.1e-6`, gauge-removed
#       `|dp|/|p| <= 8.8e-7` (docs/dev/MG.md) -- `relres` alone does not
#       certify a preconditioner change (docs/dev/MG.md);
#   (f) the driver default resolves to `:exact` on the matrix-free path and
#       `:leaf` on the sparse-operator path (observable in the pressure cache),
#       and an invalid value is rejected.

const _G226_Z2 = (x, y) -> 0.0

# 28-leaf tree: 2x2 base cells, one uniform refinement, the lower-left 2x2
# block refined once more, 2:1-balanced (the G218 tree).
function _g226_hanging()
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

@testset "G226 exact restricted BFBT diagonal" begin
    Random.seed!(20261112)
    qt = _g226_hanging()
    @test length(unique(lf[1] for lf in qt.leaves)) > 1
    @test max_face_level_jump(qt) <= 1
    eta = (x, y) -> exp(-log(1000.0) * (1 - y))
    wfun = (x, y) -> sqrt(max(eta(x, y), 1.0e-300))

    for p in (2, 3)
        ss = make_semstokes(qt, p)
        sys = solve_semquad_stokes_var(ss, qt, eta, _G226_Z2, _G226_Z2,
                                       _G226_Z2, _G226_Z2; bc=:free,
                                       deviatoric=false, mass_form=:boussinesq,
                                       linear=:mg, mg_operator=:matrixfree,
                                       assemble_only=:mg)
        Cu = sys.Cu
        d_ex = QuaST._semquad_bfbt_diag(ss.su, Cu, eta, wfun; how=:exact)
        d_lf = QuaST._semquad_bfbt_diag(ss.su, Cu, eta, wfun; how=:leaf)
        d_sp = QuaST._semquad_bfbt_diag(ss.su, Cu, eta, wfun; how=:matrix)
        dsc = maximum(abs, d_sp)
        # (a) the sparse reference, to round-off (not bitwise)
        @test maximum(abs.(d_ex .- d_sp)) <= 1.0e-12 * dsc
        # (b) positive, unclipped
        @test all(>(0.0), d_ex)
        @test minimum(d_ex) / maximum(d_ex) > 1.0e-12
        @test count(==(1.0e-12 * maximum(d_ex)), d_ex) == 0
        # (c) the cross terms are not vacuous on this tree
        @test maximum(abs.(d_ex .- d_lf)) > 1.0e-6 * dsc
        # (d) deterministic
        d_ex2 = QuaST._semquad_bfbt_diag(ss.su, Cu, eta, wfun; how=:exact)
        @test reinterpret(UInt64, d_ex) == reinterpret(UInt64, d_ex2)
        @info "G226 diagonal" p = p nfp = length(d_ex) rel =
            maximum(abs.(d_ex .- d_sp)) / dsc rel_lump =
            maximum(abs.(d_lf .- d_sp)) / dsc contrast =
            maximum(d_ex) / minimum(d_ex)
    end

    # ---- (e) the solver: tolerance and accuracy ------------------------
    p = 2
    ss = make_semstokes(qt, p)
    fz = (x, y) -> sinpi(x) * sinpi(y)
    u_dir, p_dir = solve_semquad_stokes_var(ss, qt, eta, _G226_Z2, fz,
                                            _G226_Z2, _G226_Z2; bc=:free,
                                            deviatoric=false,
                                            mass_form=:boussinesq,
                                            linear=:direct,
                                            stokes_scaling=:equilibrated)
    @test norm(u_dir) > 1.0e-8
    for weight in (:leaf, :exact)
        empty!(QuaST.SEMQUAD_MG_CACHE)
        QuaST.SEMQUAD_PRESSURE_CACHE[] = nothing
        u, pv, = solve_semquad_stokes_var(ss, qt, eta, _G226_Z2, fz,
                                          _G226_Z2, _G226_Z2; bc=:free,
                                          deviatoric=false,
                                          mass_form=:boussinesq, linear=:mg,
                                          mg_operator=:matrixfree,
                                          mg_vel_solver=:vcycle,
                                          mg_tol=1.0e-6,
                                          mg_pressure_bfbt_d=weight)
        nit = LAST_SEMSTOKES_SCHUR_ITERS[]
        res = LAST_SEMSTOKES_RESIDUAL[]
        du = norm(u - u_dir) / norm(u_dir)
        pg = pv .- sum(pv) / length(pv)
        pdg = p_dir .- sum(p_dir) / length(p_dir)
        dpg = norm(pg - pdg) / norm(pdg)
        @test res <= 1.0e-6              # true residual, not relres alone
        @test nit < 100
        @test du <= 2.1e-6               # the documented MG accuracy class
        @test dpg <= 8.8e-7
        @info "G226 solve" weight = weight it = nit res = res du = du dpg = dpg
    end

    # ---- (f) the shipped default and the A/B switch --------------------
    empty!(QuaST.SEMQUAD_MG_CACHE)
    QuaST.SEMQUAD_PRESSURE_CACHE[] = nothing
    solve_semquad_stokes_var(ss, qt, eta, _G226_Z2, fz, _G226_Z2, _G226_Z2;
                             bc=:free, deviatoric=false,
                             mass_form=:boussinesq, linear=:mg,
                             mg_operator=:matrixfree, mg_tol=1.0e-6)
    @test QuaST.SEMQUAD_PRESSURE_CACHE[] !== nothing
    @test QuaST.SEMQUAD_PRESSURE_CACHE[].dweight === :exact
    empty!(QuaST.SEMQUAD_MG_CACHE)
    QuaST.SEMQUAD_PRESSURE_CACHE[] = nothing
    solve_semquad_stokes_var(ss, qt, eta, _G226_Z2, fz, _G226_Z2, _G226_Z2;
                             bc=:free, deviatoric=false,
                             mass_form=:boussinesq, linear=:mg,
                             mg_operator=:sparse, mg_tol=1.0e-6)
    @test QuaST.SEMQUAD_PRESSURE_CACHE[].dweight === :leaf
    @test_throws ArgumentError solve_semquad_stokes_var(
        ss, qt, eta, _G226_Z2, fz, _G226_Z2, _G226_Z2; bc=:free,
        linear=:mg, mg_pressure_bfbt_d=:bogus)
    # the lumped A/B is still reachable and gives a positive, finite D
    sys2 = solve_semquad_stokes_var(ss, qt, eta, _G226_Z2, _G226_Z2,
                                    _G226_Z2, _G226_Z2; bc=:free,
                                    deviatoric=false, mass_form=:boussinesq,
                                    linear=:mg, mg_operator=:matrixfree,
                                    assemble_only=:mg)
    Fp, _ = QuaST.semquad_mg_pressure_precond(
        ss, sys2.Cu, sys2.Cp, sys2.Bs, sys2.Bt, nothing, eta;
        mode=:bfbt, cache=false, bfbt_d=:leaf)
    @test all(>(0.0), 1.0 ./ Fp.Dinvv)
end
