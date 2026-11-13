# ALA / reference-state gates (King 2010 benchmark formulation).
#
# These are the first tests in the suite that exercise the anelastic liquid
# approximation path.  They encode two things that ASPECT's source pins down and
# that were previously wrong in this solver:
#
# 1. The discrete mass equation must be  div u = +Di u_z  for rho_ad = exp(Di*depth).
#    ASPECT: source/simulator/assemblers/stokes.cc:600 and :636-638
#        - div u - (1/rho) drho_d(depth) (ghat.u) = 0
#    with drho_d(depth) = +Di rho and ghat_z = -1.
#    The opposite sign gives div u = -Di u_z and therefore div(rho_ad u) =
#    -2 Di rho_ad u_z != 0, which destroys the global energy balance.
#
# 2. The ALA temperature equation uses the ADIABATIC REFERENCE DENSITY, not the
#    full material density, and is independent of pressure.
#    ASPECT: source/simulator/assembly.cc:896-904 (reference_density_profile for
#    the temperature equation, selected for ALA by parameters.cc:1591-1595).
using Test
using QuaST

function _ala_case(n; Di=1.0)
    m = make_mesh(1.0, n, n, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    dm = make_dofmap(m, :free)
    xs, zs = node_coords(m, fd.gll_v)
    theta = [0.01 * cospi(xs[mod(k-1, m.nvx)+1]) * sinpi(zs[div(k-1, m.nvx)+1])
             for k in 1:m.Nscalar]
    pr = Params(Ra=1.0e4, eta_const=Di)
    rs = build_refstate(; rho_fun=z -> exp(Di * (1 - z)), alpha_fun=z -> Di,
                        beta_fun=z -> 1.0, cp_fun=z -> 1.0, k_fun=z -> 1.0, g=1.0,
                        rho0=1.0, alpha0=Di, cp0=1.0, k0=1.0, Tsurf=0.091)
    return m, fd, dm, theta, pr, rs
end

"""L2 norms of (div u - Di uz) and (div u + Di uz) at the Gauss points."""
function _continuity_norms(m, fd, ux, uz, Di)
    pv = fd.phi_v; dpv = fd.dphi_v
    s_minus = 0.0; s_plus = 0.0; s_ref = 0.0; vol = 0.0
    for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq, qx in 1:fd.nq
                divu = 0.0; uzq = 0.0
                for j in 1:m.p+1, i in 1:m.p+1
                    k = (j - 1) * (m.p + 1) + i
                    divu += ux[g[k]] * dpv[i, qx] * pv[j, qy] * (2.0 / m.hx) +
                            uz[g[k]] * pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                    uzq += uz[g[k]] * pv[i, qx] * pv[j, qy]
                end
                w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                s_minus += w * (divu - Di * uzq)^2
                s_plus += w * (divu + Di * uzq)^2
                s_ref += w * (Di * uzq)^2
                vol += w
            end
        end
    end
    return sqrt(s_minus / vol), sqrt(s_plus / vol), sqrt(s_ref / vol)
end

@testset "G10 ALA mass conservation sign" begin
    m, fd, dm, theta, pr, rs = _ala_case(4)
    st = solve_stokes(m, fd, pr, theta, dm; refstate=rs, mass_form=:aspect,
                      buoyancy_form=:anomaly, res_bar=1.0e-10)
    minus, plus, ref = _continuity_norms(m, fd, st.ux, st.uz, 1.0)
    # The physically and source-correct constraint is the "minus" residual.
    @test minus < 1.0e-2 * ref
    # The opposite sign must be clearly violated: this fails if the sign is
    # flipped back, which is what the test guards against.
    @test plus > 0.1 * ref
end

@testset "G10 ALA temperature equation is pressure independent" begin
    m, fd, dm, theta, pr, rs = _ala_case(4)
    dmt = make_theta_map(m)
    st = solve_stokes(m, fd, pr, theta, dm; refstate=rs, mass_form=:aspect,
                      buoyancy_form=:anomaly, res_bar=1.0e-10)
    p_zero = zeros(Float64, m.Npres)
    p_big = fill(1.0e4, m.Npres)
    kw = (; refstate=rs, heating_adiabatic=true, heating_dissipation=true, Di=1.0,
          theta_now=theta, theta_prev=theta, dt=1.0e-3)
    em0 = assemble_energy(m, fd, pr, st.ux, st.uz, dmt; p_prev=p_zero, kw...)
    em1 = assemble_energy(m, fd, pr, st.ux, st.uz, dmt; p_prev=p_big, kw...)
    # rho*c_p must be rho_ad*cp, so neither the mass matrix nor the source
    # term may depend on the pressure (ASPECT replaces the density by the
    # adiabatic profile before assembling the temperature equation).
    @test em0.M == em1.M
    @test em0.F ≈ em1.F rtol=0.0 atol=0.0
end

@testset "G10 ALA reference density in rho*cp" begin
    m, fd, dm, theta, pr, rs = _ala_case(4)
    dmt = make_theta_map(m)
    st = solve_stokes(m, fd, pr, theta, dm; refstate=rs, mass_form=:aspect,
                      buoyancy_form=:anomaly, res_bar=1.0e-10)
    em = assemble_energy(m, fd, pr, st.ux, st.uz, dmt; refstate=rs,
                         theta_now=theta, theta_prev=theta, dt=1.0e-3)
    # Row sums of the mass matrix are the integrals of rho_ad*cp*phi_i; compare
    # against an independent Gauss quadrature of the same quantity.
    rows = vec(sum(em.M; dims=2))
    w = zeros(Float64, m.Nscalar)
    for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq, qx in 1:fd.nq
                z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                rho = QuaST.refstate_at(rs, z)[1] / rs.rho0
                wq = rho * fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                for j in 1:m.p+1, i in 1:m.p+1
                    kk = (j - 1) * (m.p + 1) + i
                    # only free theta DOFs are columns of the reduced matrix
                    dmt.red[g[kk]] > 0 || continue
                    w[g[kk]] += wq * fd.phi_v[i, qx] * fd.phi_v[j, qy]
                end
            end
        end
    end
    # Row sums of the mass matrix are int rho_ad*cp*phi_i only for nodes all of
    # whose neighbouring basis functions are free.  Elements touching z=0,1
    # carry fixed (Dirichlet) basis functions, so restrict the comparison to
    # rows at least one element away from the top and bottom boundaries.
    node_of_red = zeros(Int, dmt.n)
    for g in eachindex(dmt.red)
        dmt.red[g] > 0 && (node_of_red[dmt.red[g]] = g)
    end
    iz_of(r) = div(node_of_red[r] - 1, m.nvx) + 1
    keep = [r for r in 1:dmt.n
            if m.p + 2 <= iz_of(r) <= m.nvz - m.p - 1]
    @test length(keep) > 0
    @test rows[keep] ≈ w[dmt.red .> 0][keep] rtol=1.0e-12 atol=1.0e-12
end

@testset "G10 ALA adiabatic heating uses the full temperature" begin
    m, fd, dm, theta, pr, rs = _ala_case(4)
    dmt = make_theta_map(m)
    st = solve_stokes(m, fd, pr, theta, dm; refstate=rs, mass_form=:aspect,
                      buoyancy_form=:anomaly, res_bar=1.0e-10)
    zth = zeros(Float64, m.Nscalar)
    d = Dict{Symbol,Float64}()
    assemble_energy(m, fd, pr, st.ux, st.uz, dmt; refstate=rs,
                    theta_now=zth, theta_prev=zth, dt=1.0e-3,
                    heating_adiabatic=true, heating_dissipation=false,
                    Di=1.0, diag=d)
    # independent Gauss quadrature of -Di rho_ad alpha T u_z with
    # T = Tbar + Tad(top)  (the physical temperature) and, as a control,
    # T = Tbar (what the buggy version used)
    s_full = 0.0; s_noTad = 0.0
    for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq, qx in 1:fd.nq
                z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                uzq = 0.0
                for j in 1:m.p+1, i in 1:m.p+1
                    kk = (j - 1) * (m.p + 1) + i
                    uzq += st.uz[g[kk]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
                end
                rho = QuaST.refstate_at(rs, z)[1] / rs.rho0
                wq = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                s_full += (-1.0) * rho * (Tbar(z, 0.0) + rs.Tad[end]) * uzq * wq
                s_noTad += (-1.0) * rho * Tbar(z, 0.0) * uzq * wq
            end
        end
    end
    @test d[:gam] ≈ s_full rtol=1.0e-12 atol=1.0e-12
    # The Tad(top) offset is not optional: omitting it changes the heating by
    # several percent (9% of T for King), which is what biased the flow.
    @test abs(s_full - s_noTad) > 0.01 * abs(s_full)
    @test abs(d[:gam] - s_noTad) > 0.01 * abs(s_full)
end

