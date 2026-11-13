using Test, QuaST, LinearAlgebra, SparseArrays

# G164: the direct saddle LU must be reliable on free-slip wall-refined trees.
#
# On the 2b wall trees (interior level 3-4, wall strip level 5-6) the *unshifted*
# sparse LU of the reduced saddle has a growth factor ~1e15 and returns a
# solution whose true residual is ~1e3, while the same system solved with a
# 1e-12 relative diagonal shift is exact and gives the same velocity to 1e-6.
# `solve_semquad_stokes_var` now retries with that shift when the first solve
# leaves a bad residual; this gate pins the retry down, checks the equivalence
# with an explicitly shifted solve, and checks that well-conditioned meshes do
# not trigger it (so their results stay bit-identical).

@testset "G164 direct saddle regularization" begin
    function wall_tree(base, wall)
        qt = Quadtree(max(base, wall); cells_x=5, cells_y=2)
        for _ in 1:base, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        refine_wall_to_min!(qt, wall; strip=0.1)
        balance!(qt)
        return qt
    end

    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=9.70406052783923433,
                eta_c=4.15888308335967186, Twall_top=0.0, Twall_bot=1.0,
                bc=:free)
    z2 = (x, y) -> 0.0

    # (base=4, wall=5) is the smallest tree found that triggers the unstable
    # factorisation with a smooth temperature field.
    qt = wall_tree(4, 5)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    theta = [0.5 * sin(pi * x) * sin(pi * y) for (x, y) in sq.coords]
    for g in 1:sq.N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end
    leaves = Set(qt.leaves)
    eta_fun = (x, y) -> semquad_eta_at(sq, theta, x, y, pr, leaves)
    fz = (x, y) -> pr.Ra * QuaST._semquad_eval_at(sq, theta, x, y, leaves)

    u, p, fixed, cons2, cons_p = solve_semquad_stokes_var(
        ss, qt, eta_fun, z2, fz, z2, z2; bc=:free, linear=:direct)
    @test LAST_SEMSTOKES_REGULARIZED[]
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-5
    @test all(isfinite, u)
    # the physical state is the same as before the fallback existed: Vrms of
    # this smooth initial condition is ~1550 (anything near 1e3 or larger means
    # the solve returned a null-space component instead of the solution)
    @test 1000.0 < semquad_vrms(sq, qt, u[1:sq.N], u[(sq.N + 1):(2sq.N)]) < 2000.0

    # ... and it matches an explicitly shifted factorisation to tight tolerance
    sys = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2; bc=:free,
                                   linear=:direct, assemble_only=true)
    nf = size(sys.Kred, 1)
    nfp = size(sys.Bs, 1)
    S = [sys.Kred sys.Bt; sys.Bs spzeros(Float64, nfp, nfp)]
    dmax = maximum(abs, diag(S)) + 1.0
    sol = (S + (1.0e-12 * dmax) * I) \ vcat(sys.fr, sys.gr)
    ush = sys.u0 + sys.Cu * sol[1:nf]
    @test norm(u - ush) <= 1.0e-6 * norm(ush)

    # a well-conditioned uniform mesh takes the unshifted path (bit-identical
    # to the historical behaviour)
    qu = Quadtree(4)
    for _ in 1:2, lf in collect(qu.leaves)
        refine!(qu, lf)
    end
    balance!(qu)
    ssu = make_semstokes(qu, 2)
    squ = ssu.su
    thu = [0.5 * sin(pi * x) * sin(pi * y) for (x, y) in squ.coords]
    lvu = Set(qu.leaves)
    etau = (x, y) -> semquad_eta_at(squ, thu, x, y, pr, lvu)
    fzu = (x, y) -> pr.Ra * QuaST._semquad_eval_at(squ, thu, x, y, lvu)
    uu, _, _, _, _ = solve_semquad_stokes_var(ssu, qu, etau, z2, fzu, z2, z2;
                                              bc=:free, linear=:direct)
    @test !LAST_SEMSTOKES_REGULARIZED[]
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-10
    @test all(isfinite, uu)

    # the flag is reset by every direct solve (no stale true from the wall tree)
    @test !LAST_SEMSTOKES_REGULARIZED[]
end
