# G48: short-time adaptive vs static at the same physical time.
#
# The adaptive run (amr_every=1, refine_frac=0.3) and the uniformly refined
# static run are advanced to the same final t.  The static theta is transferred
# onto the adaptive nodes and compared in relative L2.  This is the local
# evidence for acceptance criterion 1 at small Ra; it is not a claim about
# Ra=1e8.
using Test
using LinearAlgebra
using MantleSEM

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

function h48_theta(sq)
    # theta is the perturbation T - Tbar: it must vanish on the top and bottom
    # walls, where solve_semquad_energy imposes homogeneous Dirichlet data.
    return [0.05 * sinpi(sq.coords[i][1]) * sinpi(sq.coords[i][2])
            for i in 1:sq.N]
end

function h48_poly(sq)
    # Q2-exact on every leaf and zero on all four walls: the L2 and L4 initial
    # projections are the same function, so this isolates the AMR path from the
    # initial-projection error of the sine case.
    return [0.05 * (4 * sq.coords[i][1] * (1 - sq.coords[i][1])) *
                  (4 * sq.coords[i][2] * (1 - sq.coords[i][2]))
            for i in 1:sq.N]
end

@testset "G48 adaptive vs static at the same time" begin
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e1, bc=:free)

    qta = uniform_tree(2)
    ssa = make_semquad(qta, 2)
    # Threshold calibrated on the classic policy; pin it (docs/dev/AMR.md).
    ra = run_semquad_simulation(qta, pr, h48_theta(ssa);
                                tstop=1.0e-5, dt0=1.0e-6, cfl=0.0,
                                maxpic=1, amr_every=1, refine_frac=0.3,
                                coarsen_frac=0.0, maxsteps=10,
                                amr_policy=:classic, linear=:direct, verbose=false)

    qts = uniform_tree(4)
    sss = make_semquad(qts, 2)
    rs = run_semquad_simulation(qts, pr, h48_theta(sss);
                                tstop=1.0e-5, dt0=1.0e-6, cfl=0.0,
                                maxpic=1, amr_every=0, maxsteps=10,
                                linear=:direct, verbose=false)

    @test ra.steps == 10
    @test rs.steps == 10
    @test rs.t ≈ ra.t rtol=1e-14
    @test ra.ss.su.N > ssa.N  # AMR refined beyond the base level-2 mesh
    @test ra.ss.su.N < rs.ss.su.N  # but stays below the uniform level-4 mesh

    th_static_on_adaptive = transfer_semquad(rs.ss.su, rs.theta, ra.ss.su)
    err = norm(th_static_on_adaptive - ra.theta) / norm(ra.theta)
    # With the boundary-consistent perturbation (the earlier 1-y+... initial
    # condition violated theta=0 at the wall and produced a spurious layer):
    #   adaptive N=943 vs uniform L4 N=1089: err = 0.0042
    #   uniform L2/L3/L4 vs L5:              0.00444 / 0.00263 / 0.00128
    # Of the 0.0042, 0.0039 is the L2 initial projection: a static run on the
    # final tree with the same L2 IC is within 0.00044 of the adaptive run, and
    # with the Q2-exact IC 0.05*16x(1-x)y(1-y) adaptive vs L4 is 0.00055.
    # The 5% threshold was declared with the test and is intentionally kept;
    # this is a consistency gate, not a calibration of the AMR error constant.
    @test err < 0.05

    # Regression for the old 38.74% artefact: the full temperature was passed as
    # theta0, so theta was nonzero on the bottom wall.  This must be rejected,
    # not silently turned into a spurious boundary layer.
    qt_bad = uniform_tree(2)
    sq_bad = make_semquad(qt_bad, 2)
    th_bad = [1.0 - sq_bad.coords[i][2] for i in 1:sq_bad.N]
    @test_throws ArgumentError run_semquad_simulation(qt_bad, pr, th_bad;
                                                      tstop=1.0e-5, dt0=1.0e-6,
                                                      maxsteps=1, linear=:direct)

    # Q2-exact initial perturbation: both meshes represent the initial
    # condition exactly, so this isolates the AMR path.  Measured err = 5.5e-4
    # (vs 4.2e-3 for the sine case, of which 3.9e-3 is the L2 projection).
    qtp = uniform_tree(2)
    ssp = make_semquad(qtp, 2)
    rp = run_semquad_simulation(qtp, pr, h48_poly(ssp);
                                tstop=1.0e-5, dt0=1.0e-6, cfl=0.0,
                                maxpic=1, amr_every=1, refine_frac=0.3,
                                coarsen_frac=0.0, maxsteps=10,
                                amr_policy=:classic, linear=:direct, verbose=false)
    qts4 = uniform_tree(4)
    ss4 = make_semquad(qts4, 2)
    rs4 = run_semquad_simulation(qts4, pr, h48_poly(ss4);
                                 tstop=1.0e-5, dt0=1.0e-6, cfl=0.0,
                                 maxpic=1, amr_every=0, maxsteps=10,
                                 linear=:direct, verbose=false)
    err_poly = norm(transfer_semquad(rs4.ss.su, rs4.theta, rp.ss.su) - rp.theta) /
               norm(rp.theta)
    @test rp.steps == 10
    @test rp.ss.su.N < rs4.ss.su.N
    @test err_poly < 0.05
end
