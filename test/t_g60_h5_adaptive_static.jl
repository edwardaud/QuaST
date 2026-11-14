# G60: H=5 short-time adaptive vs static at the same physical time.
#
# Same initial perturbation on both meshes, 10 steps, p=4.  The adaptive run
# starts at level 2 and refines every step with a max_leaves budget; the static
# run is uniform level 4.  The static theta is transferred onto the adaptive
# nodes and compared in relative L2.  This is the H=5 analogue of G48 and is a
# consistency gate, not a calibration of the AMR error constant.
using Test
using LinearAlgebra
using Printf
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

function h60_theta(sq)
    return [0.01 * sinpi(sq.coords[i][1]) * sinpi(sq.coords[i][2])
            for i in 1:sq.N]
end

@testset "G60 H=5 adaptive vs static short time" begin
    pr = Params(Ra=1.0e4, H=5.0, eta_law=:expT, deta_T=1.0, bc=:free)

    qta = uniform_tree(2)
    ssa = make_semquad(qta, 4)
    # Threshold calibrated on the balanced policy (docs/dev/AMR.md).
    ra = run_semquad_simulation(qta, pr, h60_theta(ssa);
                                p=4, tstop=1.0e-4, dt0=1.0e-5, cfl=0.0,
                                maxpic=1, amr_every=1, refine_frac=0.3,
                                coarsen_frac=0.0, max_leaves=300,
                                maxsteps=10,
                                linear=:direct, verbose=false)

    qts = uniform_tree(4)
    sss = make_semquad(qts, 4)
    rs = run_semquad_simulation(qts, pr, h60_theta(sss);
                                p=4, tstop=1.0e-4, dt0=1.0e-5, cfl=0.0,
                                maxpic=1, amr_every=0, maxsteps=10,
                                linear=:direct, verbose=false)

    @test ra.steps == 10
    @test rs.steps == 10
    @test rs.t ≈ ra.t rtol=1e-14
    @test ra.ss.su.N < rs.ss.su.N
    @test max_face_level_jump(ra.qt) <= 1

    th_static_on_adaptive = transfer_semquad(rs.ss.su, rs.theta, ra.ss.su)
    err = norm(th_static_on_adaptive - ra.theta) / norm(ra.theta)
    @printf("G60 err=%.6e N_adapt=%d N_static=%d\n",
            err, ra.ss.su.N, rs.ss.su.N)
    @test err < 0.05
end
