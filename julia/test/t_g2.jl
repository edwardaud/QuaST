using Test
using SparseArrays
using LinearAlgebra
using MantleSEM

# G2: inf-sup / pressure null space, with a negative control.

function nullity(B::SparseMatrixCSC{Float64,Int})
    s = svdvals(Matrix(B))
    tol = 1e-10 * s[1]
    return count(x -> x < tol, s), s
end

@testset "G2: inf-sup and pressure null space" begin
    m1 = make_mesh(1.0, 4, 4, 2; q=1)
    fd1 = make_fedata(2, 1)
    dm1 = make_dofmap(m1, :free)
    B1 = assemble_B(m1, fd1, dm1)
    d1, s1 = nullity(B1)
    println("  Q2-Q1: nullity=$d1  sigma_min=$(minimum(s1))  sigma_max=$(maximum(s1))")
    @test d1 == 1

    m2 = make_mesh(1.0, 4, 4, 2; q=2)
    fd2 = make_fedata(2, 2)
    dm2 = make_dofmap(m2, :free)
    B2 = assemble_B(m2, fd2, dm2)
    d2, s2 = nullity(B2)
    println("  Q2-Q2: nullity=$d2  sigma_min=$(minimum(s2))  sigma_max=$(maximum(s2))")
    @test d2 > 1

    # hydrostatic state with full-T buoyancy: u = 0, p = Ra (z - z^2/2)
    # (the production default is the anomaly form, where this pressure is zero;
    # this gate keeps testing the full-T pressure solve explicitly)
    m = make_mesh(1.0, 8, 8, 4)
    fd = make_fedata(m.p, m.q)
    pr = Params(Ra=1e4, eta_law=:const, bc=:free)
    dm = make_dofmap(m, :free)
    theta = zeros(Float64, m.Nscalar)
    sol = solve_stokes(m, fd, pr, theta, dm; buoyancy_form=:full)
    println("  hydrostatic: max|u|=$(max(maximum(abs, sol.ux), maximum(abs, sol.uz)))  res=$(sol.res)")
    @test max(maximum(abs, sol.ux), maximum(abs, sol.uz)) < 1e-10
    @test sol.res < 1e-10

    rulep = gll_rule(m.q)
    xsp = zeros(Float64, m.npx)
    zsp = zeros(Float64, m.npz)
    for ix in 1:m.npx
        if ix == m.npx
            xsp[ix] = m.Lx
        else
            ex = (ix - 1) ÷ m.q
            a = mod(ix - 1, m.q) + 1
            xsp[ix] = ex * m.hx + 0.5 * (rulep.nodes[a] + 1.0) * m.hx
        end
    end
    for iz in 1:m.npz
        if iz == m.npz
            zsp[iz] = 1.0
        else
            ez = (iz - 1) ÷ m.q
            b = mod(iz - 1, m.q) + 1
            zsp[iz] = ez * m.hz + 0.5 * (rulep.nodes[b] + 1.0) * m.hz
        end
    end
    perr = 0.0
    for iz in 1:m.npz
        for ix in 1:m.npx
            g = (iz - 1) * m.npx + ix
            pex = pr.Ra * (zsp[iz] - 0.5 * zsp[iz]^2)
            perr = max(perr, abs(sol.p[g] - pex))
        end
    end
    println("  hydrostatic: max|p-p_exact|=$perr")
    @test perr < 1e-7
end
