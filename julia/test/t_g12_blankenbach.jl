# Blankenbach (1989) case 2b: joint temperature/depth viscosity law.
#
# Source: ASPECT benchmarks/blankenbach/base_case2b.prm and
# source/material_model/nondimensional.cc:73-75:
#   eta = (compressible ? Di : 1)/Ra * exp(-b (T - T_ad) + c depth)
# with b = ln(16384) = 9.70406052783923433184,
#      c = ln(64)    = 4.1588830833596718565.
# For the Boussinesq runs T_ad = 0, depth = 1 - z, and the 1/Ra prefactor is
# absorbed by the body-force scaling (as in the validated 2a case), so
#   eta = exp(-b T + c (1-z)).
#
# The full 2b steady benchmark is run by julia/scripts/run_2b.jl.  These gates
# are fast and check the law itself and that the Stokes assembly uses it.
using Test
using MantleSEM

@testset "G12 2b viscosity law" begin
    b = log(16384.0); c = log(64.0)
    pr = Params(Ra=1.0e4, eta_law=:expTz, eta_b=b, eta_c=c)
    @test eta_at(0.0, 1.0, pr)[1] ≈ 1.0 rtol=1e-14
    @test eta_at(0.0, 0.0, pr)[1] ≈ 64.0 rtol=1e-14
    @test eta_at(1.0, 1.0, pr)[1] ≈ 1.0 / 16384.0 rtol=1e-14
    @test eta_at(1.0, 0.0, pr)[1] ≈ exp(-b + c) rtol=1e-14
    @test eta_at(0.5, 0.5, pr)[1] ≈ exp(-0.5 * b + 0.5 * c) rtol=1e-14
    # algebraic contrast of the law over (T,z) in [0,1]^2: 64 / (1/16384)
    @test eta_at(0.0, 0.0, pr)[1] / eta_at(1.0, 1.0, pr)[1] ≈ 16384.0 * 64.0 rtol=1e-12
    # eta_of_T has no z and must refuse the joint law
    @test_throws ArgumentError eta_of_T(0.5, pr)
    # :expT still behaves as before
    pr2 = Params(eta_law=:expT, deta_T=1000.0)
    @test eta_at(1.0, 0.3, pr2)[1] ≈ 1.0e-3 rtol=1e-14
end

@testset "G12 2b viscosity enters the Stokes assembly" begin
    b = log(16384.0); c = log(64.0)
    m = make_mesh(2.5, 4, 4, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    dm = make_dofmap(m, :free)
    xs, zs = node_coords(m, fd.gll_v)
    theta = [0.01 * cospi(xs[mod(k-1,m.nvx)+1] / 2.5) * sinpi(zs[div(k-1,m.nvx)+1])
             for k in 1:m.Nscalar]
    pr = Params(Ra=1.0e4, eta_law=:expTz, eta_b=b, eta_c=c)
    st = solve_stokes(m, fd, pr, theta, dm; res_bar=1.0e-10)
    # independent evaluation of eta at the same Gauss points
    emin = Inf; emax = -Inf
    for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq, qx in 1:fd.nq
                z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                T = 0.0
                for j in 1:m.p+1, i in 1:m.p+1
                    kk = (j - 1) * (m.p + 1) + i
                    T += (Tbar(z, 0.0) + theta[g[kk]]) *
                         fd.phi_v[i, qx] * fd.phi_v[j, qy]
                end
                e = exp(-b * T + c * (1.0 - z))
                emin = min(emin, e); emax = max(emax, e)
            end
        end
    end
    @test st.eta_min ≈ emin rtol=1e-12
    @test st.eta_max ≈ emax rtol=1e-12
end

@testset "G12 1b/1c ASCII initial condition" begin
    m = make_mesh(1.0, 8, 8, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    for (case, ngrid, tmin, tmax) in (("1b", 64, 0.00591661, 0.994071),
                                      ("1c", 32, 0.0245622, 0.975438))
        path = joinpath(@__DIR__, "..", "data", "initial_temperature_case$(case).txt")
        g = read_temperature_ascii(path)
        @test length(g.x) == ngrid
        @test length(g.y) == ngrid
        @test minimum(g.T) ≈ tmin rtol=1e-5
        @test maximum(g.T) ≈ tmax rtol=1e-5
        theta = theta_from_ascii(m, fd.gll_v, path; blend=0.0)
        # homogeneous Dirichlet rows at z=0 and z=1
        @test all(theta[1:m.nvx] .== 0.0)
        @test all(theta[(m.nvz-1)*m.nvx+1:end] .== 0.0)
        @test all(isfinite, theta)
        @test maximum(abs, theta) < 1.2
    end
end
