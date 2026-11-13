# G79: symmetry-preserving AMR marking.
#
# A mirror-symmetric IC and mesh must stay symmetric.  Default marking breaks
# ties asymmetrically; symmetry=:x marks the mirror leaf together.
using Test
using MantleSEM

function mirror_missing(qt)
    s = Set(qt.leaves)
    miss = 0
    for lf in qt.leaves
        ml = (lf[1], 2^lf[1] - 1 - lf[2], lf[3])
        ml in s || (miss += 1)
    end
    return miss
end

@testset "G79 symmetry-preserving AMR" begin
    qt = wall_graded_tree(2, 7; strip=0.005)
    sq = make_semquad(qt, 4)
    th = [0.12 * exp(-((x - 0.5) / 0.12)^2 - ((y - 0.35) / 0.10)^2) * sinpi(y)
          for (x, y) in sq.coords]
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expTz,
                eta_b=9.704060527839234, eta_c=4.158883083359672,
                bc=:free, eta_min=1.0e-6, eta_max=1.0e6)
    # Policy-agnostic gate calibrated on classic; pin the policy explicitly.
    res = run_semquad_simulation(deepcopy(qt), pr, th;
                                 p=4, tstop=1.0, dt0=1.0e-9,
                                 dtmax=1.0e-9, cfl=0.0, maxsteps=1,
                                 maxpic=1, initial_refine=0, amr_every=1,
                                 amr_policy=:classic,
                                 marking=:error_fraction, refine_frac=0.2,
                                 coarsen_frac=0.1, max_leaves=30000,
                                 max_level=7, wall_min_level=7,
                                 wall_strip=0.005, symmetry=:x,
                                 linear=:schur_cg, schur_maxiter=2000,
                                 stokes_tol=1.0e-6, advection=:skew,
                                 verbose=false)
    @test mirror_missing(res.qt) == 0
    @test res.steps == 1
    @test max_face_level_jump(res.qt) <= 1
end
