# G53: the AMR convection loop must support both viscosity laws.
#
# :expT  = exp(-ln(Delta eta) T)              (temperature only)
# :expTz = exp(-b T + c (1-z))                (ASPECT nondimensional joint law)
#
# The joint law is the one used by Blankenbach 2b with b=ln(16384), c=ln(64).
# This gate checks the formula at the nodes, that :expT is unchanged, that the
# driver accepts :expTz, and that the first Stokes solve matches an independent
# solve with the same eta field.
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

@testset "G53 AMR viscosity :expTz" begin
    qt = uniform_tree(2)
    sq = make_semquad(qt, 2)
    theta = [0.01 * cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2])
             for i in 1:sq.N]
    leaves = Set(sq.leaves)
    b = log(16384.0); c = log(64.0)
    pr = Params(Ra=1.0e4, H=5.0, eta_law=:expTz, eta_b=b, eta_c=c, bc=:free)

    for (x, y) in ((0.1, 0.2), (0.5, 0.5), (0.9, 0.8))
        T = Tbar(y, pr.H) + MantleSEM._semquad_eval_at(sq, theta, x, y, leaves)
        @test semquad_eta_at(sq, theta, x, y, pr, leaves) ≈
              exp(-b * T + c * (1.0 - y)) rtol=1.0e-12
    end

    prT = Params(Ra=1.0e4, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free)
    for (x, y) in ((0.2, 0.3), (0.7, 0.6))
        T = Tbar(y, prT.H) + MantleSEM._semquad_eval_at(sq, theta, x, y, leaves)
        @test semquad_eta_at(sq, theta, x, y, prT, leaves) ≈
              exp(-log(1.0e3) * T) rtol=1.0e-12
    end

    res = run_semquad_simulation(qt, pr, theta;
                                 tstop=1.0e-5, dt0=1.0e-6, cfl=0.0,
                                 maxpic=1, amr_every=0, maxsteps=1,
                                 linear=:direct, verbose=false)
    @test res.steps == 1
    @test all(isfinite, res.history[1])

    # Independent Stokes solve with the same eta field and body force.
    ss = make_semstokes(qt, 2)
    sq2 = ss.su
    leaves2 = Set(sq2.leaves)
    eta_fun = (x, y) -> semquad_eta_at(sq2, theta, x, y, pr, leaves2)
    ev = (x, y) -> MantleSEM._semquad_eval_at(sq2, theta, x, y, leaves2)
    u, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta_fun,
                                             (x, y) -> 0.0,
                                             (x, y) -> pr.Ra * ev(x, y),
                                             (x, y) -> 0.0, (x, y) -> 0.0;
                                             bc=:free, linear=:direct)
    nu = sq2.N
    vr = semquad_vrms(sq2, qt, u[1:nu], u[nu+1:end])
    @test res.history[1][5] ≈ vr rtol=1.0e-10
end
