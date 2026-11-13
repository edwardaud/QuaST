using Test
using QuaST

# G1: manufactured Stokes solution with homogeneous rigid walls.
#
# psi = X(x) Z(z),  X = x^2(1-x)^2,  Z = z^2(1-z)^2
# u   = (psi_z, -psi_x),  p = X Z,  eta = 1, Ra = 0
# f   = -Delta u + grad p

X(x) = x^2 - 2.0 * x^3 + x^4
Xp(x) = 2.0 * x - 6.0 * x^2 + 4.0 * x^3
Xpp(x) = 2.0 - 12.0 * x + 12.0 * x^2
Xppp(x) = -12.0 + 24.0 * x

Z(z) = z^2 - 2.0 * z^3 + z^4
Zp(z) = 2.0 * z - 6.0 * z^2 + 4.0 * z^3
Zpp(z) = 2.0 - 12.0 * z + 12.0 * z^2
Zppp(z) = -12.0 + 24.0 * z

exact_ux(x, z) = X(x) * Zp(z)
exact_uz(x, z) = -Xp(x) * Z(z)
exact_p(x, z) = X(x) * Z(z)

function exact_force(x, z)
    lap_ux = Xpp(x) * Zp(z) + X(x) * Zppp(z)
    lap_uz = -(Xppp(x) * Z(z) + Xp(x) * Zpp(z))
    fx = -lap_ux + Xp(x) * Z(z)
    fz = -lap_uz + X(x) * Zp(z)
    return fx, fz
end

force_x(x, z) = exact_force(x, z)[1]
force_z(x, z) = exact_force(x, z)[2]

function l2_errors(m, fd, ux, uz, p)
    pv = fd.phi_v
    pp = fd.phi_p
    eu2 = 0.0
    ep2 = 0.0
    @inbounds for ez in 0:m.nze-1
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            gp = element_pres_gids(m, ex, ez)
            for qy in 1:fd.nq
                z = ez * m.hz + 0.5 * (fd.gl_x[qy] + 1.0) * m.hz
                for qx in 1:fd.nq
                    x = ex * m.hx + 0.5 * (fd.gl_x[qx] + 1.0) * m.hx
                    uxq = 0.0; uzq = 0.0; pq = 0.0
                    k = 0
                    for j in 1:m.p+1
                        py = pv[j, qy]
                        for i in 1:m.p+1
                            k += 1
                            b = pv[i, qx] * py
                            uxq += ux[g[k]] * b
                            uzq += uz[g[k]] * b
                        end
                    end
                    l = 0
                    for b in 1:m.q+1
                        py = pp[b, qy]
                        for a in 1:m.q+1
                            l += 1
                            pq += p[gp[l]] * pp[a, qx] * py
                        end
                    end
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * m.hz / 4.0
                    eu2 += w * ((uxq - exact_ux(x, z))^2 + (uzq - exact_uz(x, z))^2)
                    ep2 += w * (pq - exact_p(x, z))^2
                end
            end
        end
    end
    return sqrt(eu2 / m.Lx), sqrt(ep2 / m.Lx)
end

@testset "G1: analytic Stokes convergence (rigid walls)" begin
    pr = Params(Ra=0.0, eta_law=:const, eta_const=1.0, bc=:rigid)
    errs_u = Float64[]
    errs_p = Float64[]
    hs = Float64[]
    for n in (4, 8, 16)
        println("  G1 mesh n=$n")
        m = make_mesh(1.0, n, n, 4)
        fd = make_fedata(m.p, m.q)
        dm = make_dofmap(m, :rigid)
        theta = zeros(Float64, m.Nscalar)
        sol = solve_stokes_general(m, fd, pr, theta, dm, force_x, force_z)
        eu, ep = l2_errors(m, fd, sol.ux, sol.uz, sol.p)
        push!(errs_u, eu)
        push!(errs_p, ep)
        push!(hs, m.hx)
        @test sol.res < 1e-9
        @test maximum(abs, sol.ux) < 1.0
        println("    err_u=$eu  err_p=$ep")
    end
    for i in 1:2
        ou = log(errs_u[i] / errs_u[i+1]) / log(hs[i] / hs[i+1])
        op = log(errs_p[i] / errs_p[i+1]) / log(hs[i] / hs[i+1])
        println("  order u=$(round(ou,digits=2))  p=$(round(op,digits=2))")
        @test ou > 4.5
        @test op > 3.5
    end
    @test errs_u[end] < errs_u[1]
    @test errs_p[end] < errs_p[1]
end
