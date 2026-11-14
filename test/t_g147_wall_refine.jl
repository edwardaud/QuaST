# G147: refine_wall_to_min! — strip width, target level and the 2:1 balance.
using Test, QuaST, LinearAlgebra

# G147: refine_wall_to_min! (the wall-refinement path used by AMR).
#
# This path was silently broken by a refactor (it used an undefined `sq`) and no
# gate covered it, which is exactly the kind of regression a gate exists to catch.
# The gate pins down the geometry (leaf count, wall spacing, total area) and runs a
# short convection case on a wall-refined mesh, comparing it with the uniform mesh.

@testset "G147 wall refinement" begin
    function wall_tree()
        qt = Quadtree(5)
        for _ in 1:3, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        refine_wall_to_min!(qt, 5; strip=0.1)
        return qt
    end

    qw = Quadtree(5)
    for _ in 1:3, lf in collect(qw.leaves)
        refine!(qw, lf)
    end
    balance!(qw)
    n0 = length(qw.leaves)
    n1 = refine_wall_to_min!(qw, 5; strip=0.1)
    @test n1 > 0
    @test length(qw.leaves) > n0
    @test total_area(qw) ≈ 1.0 rtol = 1.0e-12
    for lf in qw.leaves
        h = QuaST.leaf_hy(qw, lf[1])
        zc = (lf[3] + 0.5) * h
        if zc <= 0.1 || zc >= 0.9
            @test lf[1] >= 5
        end
    end
    hy = sort(unique(round.([c[2] for c in make_semstokes(qw, 2).su.coords], digits=10)))
    @test hy[2] - hy[1] ≈ 0.5 * QuaST.leaf_hy(qw, 5) rtol = 1.0e-8

    q5 = Quadtree(6; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(q5.leaves)
        refine!(q5, lf)
    end
    balance!(q5)
    m0 = length(q5.leaves)
    refine_wall_to_min!(q5, 4; strip=0.1)
    @test length(q5.leaves) > m0
    @test total_area(q5) ≈ 2.5 rtol = 1.0e-12

    function short_run(qt)
        ss = make_semstokes(qt, 2)
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
        th0 = [0.01 * cos(pi * x) * sin(pi * y) for (x, y) in ss.su.coords]
        # The uniform tree is 256 leaves / 2 467 dofs, above the selector's
        # 2 000-dof threshold, and the rw/ru tolerances were calibrated on the
        # direct path; pin it so the wall-refinement comparison does not move.
        st = run_semquad_simulation(qt, pr, th0; p=2, tstop=2.0e-3, dt0=1.0e-4,
                                    cfl=0.5, maxsteps=5000, verbose=false,
                                    linear=:direct, ev=0.11, ev_cmax=0.052)
        nt, _ = semquad_nu_walls(ss.su, st.qt, st.theta, pr)
        return (nt, semquad_vrms(ss.su, st.qt, st.ux, st.uz),
                maximum(hypot.(st.ux, st.uz)))
    end
    qu = Quadtree(4)
    for _ in 1:4, lf in collect(qu.leaves)
        refine!(qu, lf)
    end
    balance!(qu)
    ru = short_run(qu)
    rw = short_run(wall_tree())
    @test rw[1] ≈ ru[1] rtol = 1.0e-3
    @test rw[2] ≈ ru[2] rtol = 1.0e-2
    @test rw[3] < 100.0
end