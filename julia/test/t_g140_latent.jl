using Test, MantleSEM, LinearAlgebra

# G140: latent heat as an apparent heat capacity.
#
# With  rho cp DT/Dt = div(k grad T) - rho L (Gamma_T DT/Dt + Gamma_z u_z)
# the assembly scales the mass and advection matrices by (1 + L Gamma_T) and adds
# -L Gamma_z u_z to the source.  The gate checks
#   1. L = 0 reproduces the matrices bit for bit;
#   2. the mass matrix with L != 0 equals an independent quadrature of
#      int (1 + L Gamma_T) phi_i phi_j  (same rule, separately coded);
#   3. 1' K = 0 with natural boundary conditions (the invariant behind the
#      conservation of int(theta + L Gamma)).

@testset "G140 latent heat" begin
    qt = Quadtree(3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    sq = make_semquad(qt, 2)
    n = sq.p + 1
    pt = PhaseTransition(0.7, 0.05, 0.02, 0.5, 0.1, 1.0)
    thv = [0.3 * sin(pi * x) * sin(pi * y) for (x, y) in sq.coords]
    z = zeros(sq.N)
    pr0 = Params(Ra=1.0, H=0.0, latent_L=0.0)
    prL = Params(Ra=1.0, H=0.0, latent_L=2.5)

    M0, K0, C0, F0 = assemble_semquad_energy(sq, qt, pr0, z, z; theta=thv,
                                             phases=[pt])
    Mref, Kref, Cref, Fref = assemble_semquad_energy(sq, qt, pr0, z, z;
                                                     theta=thv)
    # 1. L = 0 is bit identical with and without a phase list
    @test M0 == Mref
    @test K0 == Kref
    @test C0 == Cref
    @test F0 == Fref

    ML, KL, CL, FL = assemble_semquad_energy(sq, qt, prL, z, z; theta=thv,
                                             phases=[pt])
    @test ML != Mref
    # 3. the diffusion operator has no boundary rows, so its column sums vanish
    @test norm(ones(sq.N)' * KL) < 1.0e-10

    # 2. independent quadrature of int (1 + L Gamma_T) theta, same rule
    fd = make_fedata(sq.p, sq.p - 1)
    num = 0.0
    for lf in qt.leaves
        hx = MantleSEM.leaf_hx(sq, lf[1]); hy = MantleSEM.leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            th = 0.0
            for j in 1:n, i in 1:n
                th += thv[ids[(j - 1) * n + i]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
            end
            T = semquad_base_T(prL, y) + th
            lt = prL.latent_L * phase_dgamma_dT(pt, y, T)
            num += w * (1.0 + lt) * th
        end
    end
    @test sum(ML * thv) ≈ num rtol = 1.0e-10
    # the source term is the advective part of the phase change: it only appears
    # with a nonzero vertical velocity
    uzn = [0.2 * sin(pi * c[2]) for c in sq.coords]
    _, _, _, FL2 = assemble_semquad_energy(sq, qt, prL, z, uzn; theta=thv,
                                           phases=[pt])
    _, _, _, F02 = assemble_semquad_energy(sq, qt, pr0, z, uzn; theta=thv,
                                           phases=[pt])
    @test FL2 != F02

    # ... and its value must equal an independent quadrature of -L Gamma_z u_z
    src = zeros(sq.N)
    for lf in qt.leaves
        hx = MantleSEM.leaf_hx(sq, lf[1]); hy = MantleSEM.leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            th = 0.0; uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                th += thv[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uzq += uzn[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
            end
            T = semquad_base_T(prL, y) + th
            s = -prL.latent_L * phase_dgamma_dz(pt, y, T) * uzq
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                src[ids[k]] += w * s * fd.phi_v[i, qx] * fd.phi_v[j, qy]
            end
        end
    end
    @test maximum(abs.(FL2 .- F02 .- src)) < 1.0e-10 * max(1.0, maximum(abs.(src)))
    @test maximum(abs.(src)) > 0.0
end

