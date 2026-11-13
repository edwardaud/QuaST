using Test
using Random
using LinearAlgebra
using MantleSEM

@testset "G3: eta -> c eta scaling" begin
    m = make_mesh(1.0, 8, 8, 4)
    fd = make_fedata(m.p, m.q)
    dm = make_dofmap(m, :free)
    theta = seed_theta(m, fd.gll_v; amp=0.5)
    pr1 = Params(Ra=1e4, eta_law=:const, eta_const=1.0, bc=:free)
    base = solve_stokes(m, fd, pr1, theta, dm; buoyancy_form=:full)
    for c in (1e3, 0.1, 7.0)
        prc = Params(Ra=1e4, eta_law=:const, eta_const=c, bc=:free)
        sol = solve_stokes(m, fd, prc, theta, dm; buoyancy_form=:full)
        eu = maximum(abs, sol.ux .- base.ux ./ c) / max(maximum(abs, base.ux ./ c), 1e-30)
        ew = maximum(abs, sol.uz .- base.uz ./ c) / max(maximum(abs, base.uz ./ c), 1e-30)
        ep = maximum(abs, sol.p .- base.p) / max(maximum(abs, base.p), 1e-30)
        println("  c=$c  rel du=$(eu)  dw=$(ew)  dp=$(ep)")
        @test eu < 1e-10
        @test ew < 1e-10
        @test ep < 1e-10
    end
end

@testset "G4: independent bilinear-form check (constant and variable eta)" begin
    Random.seed!(20261019)
    m = make_mesh(1.0, 6, 6, 4)
    fd = make_fedata(m.p, m.q)
    dm = make_dofmap(m, :free)
    theta = seed_theta(m, fd.gll_v; amp=0.3)

    for (label, pr) in (("const", Params(Ra=0.0, eta_law=:const, eta_const=1.0, bc=:free)),
                        ("expT", Params(Ra=0.0, eta_law=:expT, deta_T=1000.0, bc=:free)))
        sys = assemble_stokes(m, fd, pr, theta, dm)
        ux = randn(m.Nscalar)
        uz = randn(m.Nscalar)
        vx = randn(m.Nscalar)
        vz = randn(m.Nscalar)
        for g in 1:m.Nscalar
            dm.red_ux[g] == 0 && (ux[g] = 0.0; vx[g] = 0.0)
            dm.red_uz[g] == 0 && (uz[g] = 0.0; vz[g] = 0.0)
        end
        ur = vcat(gather_reduced(dm.red_ux, ux, dm.nux),
                  gather_reduced(dm.red_uz, uz, dm.nuz))
        vr = vcat(gather_reduced(dm.red_ux, vx, dm.nux),
                  gather_reduced(dm.red_uz, vz, dm.nuz))
        nvel = dm.nux + dm.nuz
        val_mat = dot(vr, sys.A[1:nvel, 1:nvel] * ur)
        val_quad = bilinear_stokes(m, fd, pr, theta, ux, uz, vx, vz)
        rel = abs(val_mat - val_quad) / max(abs(val_quad), 1e-30)
        println("  $label: v'Au=$val_mat  quad=$val_quad  rel=$rel  eta=[$(sys.eta_min),$(sys.eta_max)] clip=$(sys.nclip)")
        @test rel < 1e-11
        if label == "expT"
            @test sys.eta_min >= 1e-3 * 0.999
            @test sys.eta_max <= 1.0 * 1.001
        end
    end
end
