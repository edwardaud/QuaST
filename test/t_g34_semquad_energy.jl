# G34: quadtree SEM energy operator against the uniform-mesh assembler.
#
# On a uniform tree the quadtree M, K, C and F must reproduce the uniform
# assemble_energy action for the same smooth theta, velocity and H.  This
# checks the physical quadrature, the base-state source -u_z dTbar/dz + H, and
# all three advection forms before any time-dependent AMR run uses them.
using Test
using LinearAlgebra
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 3)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G34 quadtree SEM energy operator" begin
    L = 3
    nxe = 1 << L
    qt = uniform_tree(L)
    sq = make_semquad(qt, 2)
    m = make_mesh(1.0, nxe, nxe, 2)
    fd = make_fedata(2, m.q)
    xs, zs = node_coords(m, fd.gll_v)

    uxf(x, y) = 1.0 + 0.1x + 0.2y
    uzf(x, y) = 0.3x - 0.2y
    thf(x, y) = 0.5 + 0.1x + 0.2y + 0.05x * y
    ux_u = [uxf(xs[mod(k - 1, m.nvx) + 1], zs[div(k - 1, m.nvx) + 1]) for k in 1:m.Nscalar]
    uz_u = [uzf(xs[mod(k - 1, m.nvx) + 1], zs[div(k - 1, m.nvx) + 1]) for k in 1:m.Nscalar]
    th_u = [thf(xs[mod(k - 1, m.nvx) + 1], zs[div(k - 1, m.nvx) + 1]) for k in 1:m.Nscalar]

    tol = 1.0e-12
    idx = Dict{Tuple{Int,Int},Int}()
    for g in 1:m.Nscalar
        x = xs[mod(g - 1, m.nvx) + 1]
        z = zs[div(g - 1, m.nvx) + 1]
        idx[(round(Int, x / tol), round(Int, z / tol))] = g
    end
    perm = [idx[(round(Int, sq.coords[i][1] / tol),
                round(Int, sq.coords[i][2] / tol))] for i in 1:sq.N]
    @test length(unique(perm)) == sq.N
    @test sq.N == m.Nscalar

    ux_q = ux_u[perm]
    uz_q = uz_u[perm]
    th_q = th_u[perm]
    dmt = ThetaMap(collect(1:m.Nscalar), m.Nscalar)

    for H in (0.0, 5.0), adv in (:galerkin, :skew, :conservative)
        pr = Params(Ra=1.0e4, H=H, eta_law=:const, bc=:free, supg=0.0)
        em = assemble_energy(m, fd, pr, ux_u, uz_u, dmt; advection=adv,
                             ev=0.0, theta_now=th_u, theta_prev=th_u, dt=1.0)
        Mq, Kq, Cq, Fq = assemble_semquad_energy(sq, qt, pr, ux_q, uz_q;
                                                 advection=adv)
        scale(x) = max(maximum(abs, x), 1.0)
        for (name, rq, ru) in (("M", Mq * th_q, em.M * th_u),
                               ("K", Kq * th_q, em.K * th_u),
                               ("C", Cq * th_q, em.C * th_u))
            @test maximum(abs, rq - ru[perm]) / scale(ru) < 1.0e-10
        end
        @test maximum(abs, Fq - em.F[perm]) / scale(em.F) < 1.0e-10
    end
end
