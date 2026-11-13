using Test, MantleSEM, LinearAlgebra, SparseArrays, Random

# G220: the reduced matrix-free MG assembly (`assemble_only=:mg`).
#
# `linear=:mg, mg_operator=:matrixfree` never stores the sparse velocity
# stiffness `K` (S3 stage C: `K`/`Kred` are `nothing` and the leaf assembly skips
# the `K` triplets).  This gate pins the *assembly* side of that contract:
#
#   (a) `assemble_only=:mg` returns the same `Cu`/`Cp`/`B`/`Bc`/`Bm`/`Bs`/`Bt`/
#       `u0`/`p0`/`fv`/`gr` as the full `assemble_only=true`, with
#       `K === nothing` and `Kred === nothing`, and does not increment
#       `SEMSTOKES_K_ASSEMBLIES`; a plain `assemble_only=true` still forms `K`;
#   (b) one production matrix-free MG solve -- hierarchy included -- forms **no**
#       sparse velocity stiffness at all (the counter is unchanged), i.e. the
#       hierarchy's coarse-level `Cu` comes from the reduced assembly too;
#   (c) the coarse-level geometry assembly mode is bitwise inert: `:minimal`
#       (reduced) and `:full` (historical, K-forming) give the same levels,
#       `Cus`, `Ps`, `free_rows`, `D`, `mfetas`, `pre`/`post`, the same V-cycle
#       and the same saddle solution/iteration count/true residual;
#   (d) passing the caller's finest-level operator and eta field
#       (`fine_mfop`/`fine_mfeta`) is bitwise inert and a foreign operator is
#       rejected;
#   (e) the saddle solve is bitwise between (c)'s two hierarchies.
#
# The measured setup drop is in docs/dev/MG.md; time is not gated here.

const _G220_Z2 = (x, y) -> 0.0
const _G220_ONE = (x, y) -> 1.0

# 43-leaf graded tree (the same construction as G219, self-contained): a
# hierarchy with a near-duplicate finest step and hanging nodes.
function _g220_graded()
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

@testset "G220 reduced matrix-free MG assembly" begin
    Random.seed!(20261112)
    qt = _g220_graded()
    ss = make_semstokes(qt, 2)
    z2 = _G220_Z2
    fz = (x, y) -> sinpi(x) * sinpi(y)

    # ---- (a) the reduced assembly returns the same system, without K ------
    n0 = MantleSEM.SEMSTOKES_K_ASSEMBLIES[]
    full = solve_semquad_stokes_var(ss, qt, _G220_ONE, z2, fz, z2, z2;
                                    bc=:free, linear=:direct,
                                    assemble_only=true)
    @test MantleSEM.SEMSTOKES_K_ASSEMBLIES[] == n0 + 1
    @test full.K isa SparseMatrixCSC && full.Kred isa SparseMatrixCSC
    n1 = MantleSEM.SEMSTOKES_K_ASSEMBLIES[]
    red = solve_semquad_stokes_var(ss, qt, _G220_ONE, z2, fz, z2, z2;
                                   bc=:free, linear=:mg,
                                   mg_operator=:matrixfree,
                                   assemble_only=:mg)
    @test MantleSEM.SEMSTOKES_K_ASSEMBLIES[] == n1   # no sparse K formed
    @test red.K === nothing && red.Kred === nothing
    @test red.Cu == full.Cu && red.Cp == full.Cp
    @test red.B == full.B && red.Bc == full.Bc && red.Bm == full.Bm
    @test red.Bs == full.Bs && red.Bt == full.Bt
    @test red.u0 == full.u0 && red.p0 == full.p0
    @test red.fv == full.fv && red.gr == full.gr
    # `fr` is the one object that is *not* comparable: the reduced assembly uses
    # the matrix-free `K u0` (the same value the production solve uses) while
    # the full one forms the sparse `K u0`.  Same RHS to round-off (profile §1.1).
    @test isapprox(red.fr, full.fr, rtol=1.0e-12, atol=1.0e-14)

    # a plain `assemble_only=true` keeps the historical K-forming behaviour
    n2 = MantleSEM.SEMSTOKES_K_ASSEMBLIES[]
    plain = solve_semquad_stokes_var(ss, qt, _G220_ONE, z2, fz, z2, z2;
                                     bc=:free, linear=:mg,
                                     mg_operator=:matrixfree,
                                     assemble_only=true)
    @test MantleSEM.SEMSTOKES_K_ASSEMBLIES[] == n2 + 1
    @test plain.K == full.K && plain.Kred == full.Kred

    # the reduced mode is only defined for the matrix-free MG path
    @test_throws ArgumentError solve_semquad_stokes_var(
        ss, qt, _G220_ONE, z2, fz, z2, z2; bc=:free, linear=:direct,
        assemble_only=:mg)

    # ---- (b) one production MF solve forms no K at all -------------------
    empty!(MantleSEM.SEMQUAD_MG_CACHE)
    MantleSEM.SEMQUAD_PRESSURE_CACHE[] = nothing
    n3 = MantleSEM.SEMSTOKES_K_ASSEMBLIES[]
    u, p, _, _, _ = solve_semquad_stokes_var(ss, qt, _G220_ONE, z2, fz, z2, z2;
                                             bc=:free, linear=:mg,
                                             mg_tol=1.0e-8)
    @test MantleSEM.SEMSTOKES_K_ASSEMBLIES[] == n3
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-6
    @test norm(u) > 0.0 && norm(p) > 0.0

    # ---- (c) the geometry assembly mode is bitwise inert -----------------
    Cu = SparseMatrixCSC{Float64,Int}(sparse(full.Cu))
    empty!(MantleSEM.SEMQUAD_MG_CACHE)
    mgmin = MantleSEM.make_semquad_mg(ss, qt, 2, :free, false, _G220_ONE,
                                      nothing; cycles=1,
                                      operator=:matrixfree,
                                      coarsen_factor=1.5, fine_Cu=Cu,
                                      geom_assembly=:minimal)
    empty!(MantleSEM.SEMQUAD_MG_CACHE)
    mgfull = MantleSEM.make_semquad_mg(ss, qt, 2, :free, false, _G220_ONE,
                                       nothing; cycles=1,
                                       operator=:matrixfree,
                                       coarsen_factor=1.5, fine_Cu=Cu,
                                       geom_assembly=:full)
    @test mgmin.geo.L >= 2                     # several coarse levels
    @test mgmin.geo.nfs == mgfull.geo.nfs
    @test mgmin.geo.Cus == mgfull.geo.Cus
    @test mgmin.geo.Ps == mgfull.geo.Ps
    @test mgmin.geo.free_rows == mgfull.geo.free_rows
    @test mgmin.D == mgfull.D
    @test mgmin.mfetas == mgfull.mfetas
    @test mgmin.pre == mgfull.pre && mgmin.post == mgfull.post
    @test mgmin.cA.L == mgfull.cA.L         # same dense coarse factor
    b = randn(mgmin.geo.nfs[end])
    @test MantleSEM.semquad_mg_apply(mgmin, b) ==
          MantleSEM.semquad_mg_apply(mgfull, b)

    # ---- (d) the caller's finest operator / eta field are bitwise inert --
    op = semquad_kop(ss, qt; deviatoric=false)
    ev = semquad_eta_field(op, _G220_ONE)
    empty!(MantleSEM.SEMQUAD_MG_CACHE)
    mgNo = MantleSEM.make_semquad_mg(ss, qt, 2, :free, false, _G220_ONE,
                                     nothing; cycles=1, operator=:matrixfree,
                                     coarsen_factor=1.5)
    empty!(MantleSEM.SEMQUAD_MG_CACHE)
    mgYes = MantleSEM.make_semquad_mg(ss, qt, 2, :free, false, _G220_ONE,
                                      nothing; cycles=1, operator=:matrixfree,
                                      coarsen_factor=1.5, fine_Cu=Cu,
                                      fine_mfop=op, fine_mfeta=ev)
    @test mgYes.mfops[end] === op
    @test mgYes.geo.nfs == mgNo.geo.nfs
    @test mgYes.geo.Cus == mgNo.geo.Cus
    @test mgYes.mfetas == mgNo.mfetas
    @test mgYes.D == mgNo.D
    @test MantleSEM.semquad_mg_apply(mgYes, b) ==
          MantleSEM.semquad_mg_apply(mgNo, b)
    # a foreign finest operator is rejected
    qt3 = Quadtree(3)
    ss3 = make_semstokes(qt3, 2)
    op3 = semquad_kop(ss3, qt3; deviatoric=false)
    @test_throws ArgumentError MantleSEM.make_semquad_mg(
        ss, qt, 2, :free, false, _G220_ONE, nothing; operator=:matrixfree,
        fine_Cu=Cu, fine_mfop=op3, fine_mfeta=ev)
    @test_throws ArgumentError MantleSEM.make_semquad_mg(
        ss, qt, 2, :free, false, _G220_ONE, nothing; operator=:matrixfree,
        fine_Cu=Cu, fine_mfop=op, fine_mfeta=[1.0, 2.0])

    # ---- (e) the saddle solve is bitwise between the two hierarchies -----
    mfapplyK = (out, v) -> MantleSEM._semquad_apply_Kred!(out,
                                                          mgmin.mfops[end],
                                                          mgmin.geo.Cus[end],
                                                          mgmin.mfetas[end], v,
                                                          mgmin.mfwork[end])
    Fp, pscale = MantleSEM.semquad_mg_pressure_precond(ss, full.Cu, full.Cp,
                                                       full.Bs, full.Bt, nothing,
                                                       _G220_ONE; mode=:auto,
                                                       cache=false,
                                                       applyK=mfapplyK)
    rhs = vcat(full.fr, full.gr)
    sA, nA, rA, _ = MantleSEM.semquad_mg_saddle_solve(nothing, full.Bs, full.Bt,
                                                      Fp, pscale, rhs, mgmin;
                                                      tol=1.0e-10, maxit=200,
                                                      applyK=mfapplyK)
    sB, nB, rB, _ = MantleSEM.semquad_mg_saddle_solve(nothing, full.Bs, full.Bt,
                                                      Fp, pscale, rhs, mgfull;
                                                      tol=1.0e-10, maxit=200,
                                                      applyK=mfapplyK)
    @test sA == sB
    @test nA == nB && rA == rB
    @test nA > 0 && rA < 1.0e-9
end
