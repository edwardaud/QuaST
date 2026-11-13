# AMR cycle: gradient-jump indicator on a localized manufactured solution,
# refinement concentrated on the feature, error reduction and invariants.
using Test
using LinearAlgebra
using MantleSEM

function l2err_amr(dg, u, exact)
    e = 0.0
    M = 20
    for k in 0:M, l in 0:M
        x = 1.0e-6 + (1 - 2.0e-6) * k / M
        y = 1.0e-6 + (1 - 2.0e-6) * l / M
        e += (dgquad_eval(dg, u, x, y) - exact(x, y))^2
    end
    return sqrt(e / (M + 1)^2)
end

@testset "G20 AMR cycle" begin
    sigma = 0.12
    r2(x, y) = (x - 0.5)^2 + (y - 0.5)^2
    exact(x, y) = exp(-r2(x, y) / sigma^2)
    rhs(x, y) = -exact(x, y) * (4 * r2(x, y) / sigma^4 - 4 / sigma^2)

    qt = Quadtree(8)
    refine!(qt, (0, 0, 0))
    for lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    @test length(qt.leaves) == 16

    dg = make_dgquad(qt, 2)
    A, b = assemble_dgquad_poisson(dg, qt, rhs, exact; sigma_C=50.0)
    u = A \ b
    e0 = l2err_amr(dg, u, exact)

    eta = dgquad_jump_indicator(dg, qt, u)
    @test all(eta[lf] >= 0.0 for lf in qt.leaves)
    nr, nc = amr_cycle!(qt, eta; refine_frac=0.3)
    @test nr > 0
    @test total_area(qt) ≈ 1.0
    @test max_face_level_jump(qt) <= 1

    dg2 = make_dgquad(qt, 2)
    A2, b2 = assemble_dgquad_poisson(dg2, qt, rhs, exact; sigma_C=50.0)
    u2 = A2 \ b2
    e1 = l2err_amr(dg2, u2, exact)
    @test e1 < 0.6 * e0

    # the largest indicator must sit on the Gaussian, and most refined leaves
    # (the others are 2:1 balance neighbours of the refined patch) must too
    best = argmax(eta)
    hb = 2.0^(-best[1])
    xb = (best[2] + 0.5) * hb; yb = (best[3] + 0.5) * hb
    @test (xb - 0.5)^2 + (yb - 0.5)^2 < 0.35^2
    ref = [lf for lf in qt.leaves if lf[1] > 2]
    near = 0
    for lf in ref
        h = 2.0^(-lf[1])
        xc = (lf[2] + 0.5) * h; yc = (lf[3] + 0.5) * h
        (xc - 0.5)^2 + (yc - 0.5)^2 < 0.35^2 && (near += 1)
    end
    @test near / length(ref) > 0.6
end
