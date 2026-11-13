using Test
using QuaST

# G4b: manufactured Stokes solution with a SMOOTH VARIABLE viscosity.
#
# psi = X(x) Z(z), u = (psi_z, -psi_x), p = X Z,
# eta = 1 + 0.5 x z,   f = -div(2 eta eps(u)) + grad p.
#
# This is the "non-degenerate" companion to G4: it exercises the eta-gradient
# terms of the weak form, which a constant-eta equivalence test cannot see.

vX(x) = x^2 - 2.0 * x^3 + x^4
vXp(x) = 2.0 * x - 6.0 * x^2 + 4.0 * x^3
vXpp(x) = 2.0 - 12.0 * x + 12.0 * x^2
vXppp(x) = -12.0 + 24.0 * x
vZ(z) = z^2 - 2.0 * z^3 + z^4
vZp(z) = 2.0 * z - 6.0 * z^2 + 4.0 * z^3
vZpp(z) = 2.0 - 12.0 * z + 12.0 * z^2
vZppp(z) = -12.0 + 24.0 * z

vv_ux(x, z) = vX(x) * vZp(z)
vv_uz(x, z) = -vXp(x) * vZ(z)
vv_p(x, z) = vX(x) * vZ(z)
vv_eta(x, z) = 1.0 + 0.5 * x * z

function vv_force(x, z)
    exx = vXp(x) * vZp(z)
    ezz = -vXp(x) * vZp(z)
    exz = 0.5 * (vX(x) * vZpp(z) - vXpp(x) * vZ(z))
    lap_ux = vXpp(x) * vZp(z) + vX(x) * vZppp(z)
    lap_uz = -(vXppp(x) * vZ(z) + vXp(x) * vZpp(z))
    eta = vv_eta(x, z)
    etax = 0.5 * z
    etaz = 0.5 * x
    fx = -(eta * lap_ux + 2.0 * (etax * exx + etaz * exz)) + vXp(x) * vZ(z)
    fz = -(eta * lap_uz + 2.0 * (etax * exz + etaz * ezz)) + vX(x) * vZp(z)
    return fx, fz
end

vv_fx(x, z) = vv_force(x, z)[1]
vv_fz(x, z) = vv_force(x, z)[2]

function vv_l2_errors(m, fd, ux, uz, p)
    pv = fd.phi_v
    pp = fd.phi_p
    eu2 = 0.0
    ep2 = 0.0
    for ez in 0:m.nze-1, ex in 0:m.nxe-1
        g = element_vel_gids(m, ex, ez)
        gp = element_pres_gids(m, ex, ez)
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        for qy in 1:fd.nq
            z = ze0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
            for qx in 1:fd.nq
                x = ex * m.hx + 0.5 * (fd.gl_x[qx] + 1.0) * m.hx
                uxq = 0.0; uzq = 0.0; pq = 0.0; k = 0
                for j in 1:m.p+1, i in 1:m.p+1
                    k += 1
                    b = pv[i, qx] * pv[j, qy]
                    uxq += ux[g[k]] * b
                    uzq += uz[g[k]] * b
                end
                l = 0
                for b in 1:m.q+1, a in 1:m.q+1
                    l += 1
                    pq += p[gp[l]] * pp[a, qx] * pp[b, qy]
                end
                w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                eu2 += w * ((uxq - vv_ux(x, z))^2 + (uzq - vv_uz(x, z))^2)
                ep2 += w * (pq - vv_p(x, z))^2
            end
        end
    end
    return sqrt(eu2 / m.Lx), sqrt(ep2 / m.Lx)
end

@testset "G4b: manufactured Stokes with variable eta" begin
    errs_u = Float64[]
    errs_p = Float64[]
    hs = Float64[]
    for n in (4, 8, 16)
        m = make_mesh(1.0, n, n, 4)
        fd = make_fedata(m.p, m.q)
        dm = make_dofmap(m, :rigid)
        pr = Params(Ra=0.0, bc=:rigid, eta_custom=vv_eta)
        sol = solve_stokes_general(m, fd, pr, zeros(m.Nscalar), dm,
                                   vv_fx, vv_fz)
        eu, ep = vv_l2_errors(m, fd, sol.ux, sol.uz, sol.p)
        push!(errs_u, eu); push!(errs_p, ep); push!(hs, m.hx)
        @test sol.res < 1e-9
        println("  n=$n  err_u=$eu  err_p=$ep  max|div u|=$(max_divergence(m, fd, sol.ux, sol.uz))")
    end
    for i in 1:2
        ou = log(errs_u[i] / errs_u[i+1]) / log(hs[i] / hs[i+1])
        op = log(errs_p[i] / errs_p[i+1]) / log(hs[i] / hs[i+1])
        println("  variable-eta order u=$(round(ou,digits=2)) p=$(round(op,digits=2))")
        @test ou > 4.5
        @test op > 3.5
    end
end
