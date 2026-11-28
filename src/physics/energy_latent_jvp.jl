# Energy assembly: latent-heat directional derivative (Newton JVP) for the
# box and mapped paths.  Kept separate from the value assembly because it is
# the tangent of a different operator.

"""
    assemble_semquad_energy_latent_jvp(sq, qt, pr, phases, dtheta, theta, ux, uz;
                                       advection=:galerkin, rho_ref=nothing,
                                       rho_heating=false, rho_cp=false)

Directional derivative of the **latent-heat** part of
[`assemble_semquad_energy`](@ref) in the temperature direction `dtheta`, at the
temperature `theta` (the apparent heat capacity is
`1 + sum_i latent_i * Gamma_T,i` and the source is
`-sum_i latent_i * Gamma_z,i u_z`; `latent_i` is `pr.latent_L` in `:global`
mode or `PhaseTransition.latent` in `:per_phase` mode):

    Md_ij = int  L_i Gamma_TT,i(y,T) dtheta phi_i phi_j
    Cd_ij = int  L_i Gamma_TT,i(y,T) dtheta phi_i (u.grad phi_j)
    Fd_i  = int -L_i Gamma_zT,i(y,T) dtheta u_z phi_i

so the per-step Newton's energy tangent is
`Cfree' (A_E dtheta + ca*Md*theta - cb*Md*theta_n + cc*Md*theta_prev + Cd*theta - Fd)`
(the `ca/cb/cc` are the BDF2 coefficients of `solve_semquad_energy`).  The phase
derivatives are the shared `phase_d2gamma_dT2` / `phase_d2gamma_dzdT`; an empty
`phases` list or all-zero coefficients returns zero operators.  `Cd` is
transformed by the same `advection` rule as the assembled `C`
(`:skew` / `:conservative`).

As in the rest of the per-step Newton, the temperature derivative of the
`rho_heating`/`rho_cp` density factor is not differentiated (it is a separate,
pre-existing approximation).
"""
function assemble_semquad_energy_latent_jvp(sq::SemQuad, qt::Quadtree,
                                            pr::Params,
                                            phases::Vector{PhaseTransition},
                                            dtheta::Vector{Float64},
                                            theta::Vector{Float64},
                                            ux::Vector{Float64},
                                            uz::Vector{Float64};
                                            advection::Symbol=:galerkin,
                                            rho_ref::Union{Nothing,Function}=nothing,
                                            rho_heating::Bool=false,
                                            rho_cp::Bool=false)
    N = sq.N
    n = sq.p + 1
    nv = n * n
    if !semquad_latent_active(pr, phases)
        return (spzeros(Float64, N, N), spzeros(Float64, N, N),
                zeros(Float64, N))
    end
    # Cut 3: the box body below is the historical arithmetic; a mapped geometry
    # evaluates the phase coefficients at the physical height and the advective
    # latent source with `u . grad h` (its own method below).
    sq.geom isa CartesianBox ||
        return assemble_semquad_energy_latent_jvp(sq.geom, sq, qt, pr, phases,
                                                  dtheta, theta, ux, uz;
                                                  advection=advection,
                                                  rho_ref=rho_ref,
                                                  rho_heating=rho_heating,
                                                  rho_cp=rho_cp)
    lcoef = semquad_latent_coefficients(pr, phases)
    per_phase = pr.latent_mode === :per_phase
    fd = make_fedata(sq.p, sq.p - 1)
    IM = Int[]; JM = Int[]; VM = Float64[]
    sizehint!(IM, nv * nv * length(sq.leaves))
    sizehint!(JM, nv * nv * length(sq.leaves))
    sizehint!(VM, nv * nv * length(sq.leaves))
    IC = Int[]; JC = Int[]; VC = Float64[]
    sizehint!(IC, nv * nv * length(sq.leaves))
    sizehint!(JC, nv * nv * length(sq.leaves))
    sizehint!(VC, nv * nv * length(sq.leaves))
    F = zeros(Float64, N)
    phi = zeros(Float64, nv)
    gx = zeros(Float64, nv)
    gz = zeros(Float64, nv)
    Me = zeros(Float64, nv, nv)
    Ce = zeros(Float64, nv, nv)
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uxq = 0.0; uzq = 0.0; Tq = 0.0; dthq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                uxq += ux[ids[k]] * phi[k]
                uzq += uz[ids[k]] * phi[k]
                Tq += theta[ids[k]] * phi[k]
                dthq += dtheta[ids[k]] * phi[k]
            end
            Tq += semquad_base_T(pr, y)
            rho = (rho_ref === nothing ? 1.0 : rho_ref(y))::Float64
            (rho_heating || rho_cp) && (rho = max(1.0 - Tq, 1.0e-6))
            gTT = 0.0; gzT = 0.0
            # `:global` keeps the historical arithmetic exactly (one `latent_L`
            # applied *after* the sum of the second derivatives); `:per_phase`
            # weights each transition inside the sum, which is a different
            # rounding even when the coefficients happen to be equal.
            if per_phase
                for i in eachindex(phases)
                    Li = lcoef[i]
                    gTT += Li * phase_d2gamma_dT2(phases[i], y, Tq)
                    gzT += Li * phase_d2gamma_dzdT(phases[i], y, Tq)
                end
                cdir = dthq
            else
                for pt in phases
                    gTT += phase_d2gamma_dT2(pt, y, Tq)
                    gzT += phase_d2gamma_dzdT(pt, y, Tq)
                end
                cdir = pr.latent_L * dthq
            end
            ws = w * (cdir * gTT)
            cM = ws * rho
            for k in 1:nv
                F[ids[k]] += (-(cdir * gzT) * uzq) * w * phi[k]
            end
            for l in 1:nv
                pl = phi[l]
                cx = uxq * gx[l] + uzq * gz[l]
                @simd for k in 1:nv
                    pk = phi[k]
                    ck = cM * pk
                    Me[k, l] += ck * pl
                    Ce[k, l] += ck * cx
                end
            end
        end
        for l in 1:nv, k in 1:nv
            vm = Me[k, l]
            vm != 0.0 && (push!(IM, ids[k]); push!(JM, ids[l]); push!(VM, vm))
            vc = Ce[k, l]
            vc != 0.0 && (push!(IC, ids[k]); push!(JC, ids[l]); push!(VC, vc))
        end
        fill!(Me, 0.0)
        fill!(Ce, 0.0)
    end
    Md = sparse(IM, JM, VM, N, N)
    Cd = sparse(IC, JC, VC, N, N)
    if advection === :skew
        Cd = 0.5 * (Cd - Cd')
    elseif advection === :conservative
        Cd = -Cd'
    end
    return (Md, Cd, F)
end

"""
    assemble_semquad_energy_latent_jvp(g, sq, qt, pr, phases, dtheta, theta,
                                       ux, uz; ...)

Mapped method of [`assemble_semquad_energy_latent_jvp`](@ref) (cut 3): the same
directional derivative with the phase coefficients evaluated at the physical
height `h = height(g,x,y)`, the mapped quadrature/basis gradients, and the
source `-sum_i latent_i Gamma_zT,i u . grad h` (the mapped generalisation of
`u_z`, GEOM_DESIGN §6.4).  `Md`/`Cd` are assembled with the mapped measure; the
box body above is unchanged.
"""
function assemble_semquad_energy_latent_jvp(g::AbstractGeometry, sq::SemQuad,
                                            qt::Quadtree, pr::Params,
                                            phases::Vector{PhaseTransition},
                                            dtheta::Vector{Float64},
                                            theta::Vector{Float64},
                                            ux::Vector{Float64},
                                            uz::Vector{Float64};
                                            advection::Symbol=:galerkin,
                                            rho_ref::Union{Nothing,Function}=nothing,
                                            rho_heating::Bool=false,
                                            rho_cp::Bool=false)
    N = sq.N
    n = sq.p + 1
    nv = n * n
    lcoef = semquad_latent_coefficients(pr, phases)
    per_phase = pr.latent_mode === :per_phase
    fd = make_fedata(sq.p, sq.p - 1)
    IM = Int[]; JM = Int[]; VM = Float64[]
    sizehint!(IM, nv * nv * length(sq.leaves))
    sizehint!(JM, nv * nv * length(sq.leaves))
    sizehint!(VM, nv * nv * length(sq.leaves))
    IC = Int[]; JC = Int[]; VC = Float64[]
    sizehint!(IC, nv * nv * length(sq.leaves))
    sizehint!(JC, nv * nv * length(sq.leaves))
    sizehint!(VC, nv * nv * length(sq.leaves))
    F = zeros(Float64, N)
    phi = zeros(Float64, nv)
    gx = zeros(Float64, nv)
    gz = zeros(Float64, nv)
    Me = zeros(Float64, nv, nv)
    Ce = zeros(Float64, nv, nv)
    for lf in sq.leaves
        ids = sq.elem[lf]
        lm = leaf_metric(g, sq, lf, fd)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(lm, sq, lf, fd, qx, qy)
            w = quad_measure(lm, fd, qx, qy)
            h = height(g, x, y)
            uxq = 0.0; uzq = 0.0; Tq = 0.0; dthq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k], gz[k] = basis_grad(lm, fd, qx, qy, i, j)
                uxq += ux[ids[k]] * phi[k]
                uzq += uz[ids[k]] * phi[k]
                Tq += theta[ids[k]] * phi[k]
                dthq += dtheta[ids[k]] * phi[k]
            end
            ugh = upward_velocity(g, x, y, uxq, uzq)
            Tq += semquad_base_T(pr, h)
            rho = (rho_ref === nothing ? 1.0 : rho_ref(h))::Float64
            (rho_heating || rho_cp) && (rho = max(1.0 - Tq, 1.0e-6))
            gTT = 0.0; gzT = 0.0
            if per_phase
                for i in eachindex(phases)
                    Li = lcoef[i]
                    gTT += Li * phase_d2gamma_dT2(phases[i], h, Tq)
                    gzT += Li * phase_d2gamma_dzdT(phases[i], h, Tq)
                end
                cdir = dthq
            else
                for pt in phases
                    gTT += phase_d2gamma_dT2(pt, h, Tq)
                    gzT += phase_d2gamma_dzdT(pt, h, Tq)
                end
                cdir = pr.latent_L * dthq
            end
            ws = w * (cdir * gTT)
            cM = ws * rho
            for k in 1:nv
                F[ids[k]] += (-(cdir * gzT) * ugh) * w * phi[k]
            end
            for l in 1:nv
                pl = phi[l]
                cx = uxq * gx[l] + uzq * gz[l]
                @simd for k in 1:nv
                    pk = phi[k]
                    ck = cM * pk
                    Me[k, l] += ck * pl
                    Ce[k, l] += ck * cx
                end
            end
        end
        for l in 1:nv, k in 1:nv
            vm = Me[k, l]
            vm != 0.0 && (push!(IM, ids[k]); push!(JM, ids[l]); push!(VM, vm))
            vc = Ce[k, l]
            vc != 0.0 && (push!(IC, ids[k]); push!(JC, ids[l]); push!(VC, vc))
        end
        fill!(Me, 0.0)
        fill!(Ce, 0.0)
    end
    Md = sparse(IM, JM, VM, N, N)
    Cd = sparse(IC, JC, VC, N, N)
    if advection === :skew
        Cd = 0.5 * (Cd - Cd')
    elseif advection === :conservative
        Cd = -Cd'
    end
    return (Md, Cd, F)
end
