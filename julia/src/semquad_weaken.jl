# Strain weakening for the quadtree SEM path (route C), docs/RHEOLOGY.md §9.
#
# A scalar field chi(x,y,t) in [0,1] is advected with the flow and integrated
# with the same BDF2 machinery as the temperature:
#
#     dchi/dt + u.grad chi = source(eps_pl, chi)
#
# with the two mechanisms of §9.2 (selected by `pr.weak_mode`):
#
#   :rate    source = (1 - chi) * eps_pl/eps_ref - chi/tau_heal
#   :linear  source = eps_pl/eps_ref                      (ASPECT accumulation)
#
# and the single weakening factor of `semquad_weak_factor`
#
#     f(chi) = 1 - (1 - weak_f)*clamp(chi, 0, 1),
#
# which multiplies the Drucker-Prager yield stress (semquad_rheology.jl).  The
# `:rate` source is linear in chi, so its BDF2 step is implicit:
#
#     [ca*M + C(u) + M_g + (1/tau_heal)*M] chi = cb*M*chi_n - cc*M*chi_nm1 + M_g*1
#
# with `M_g` the mass matrix weighted by g = eps_pl/eps_ref.  The `:linear`
# source is the same equation without the `chi` terms.  The plastic strain rate
# `eps_pl` is `semquad_plastic_strain_rate` (the total strain rate while the
# plastic branch is active), exactly the quantity ASPECT accumulates.
#
# The transport is a *scalar* solve in its own right: no Dirichlet boundary
# conditions (the walls are natural, like the side walls of the energy
# equation), only the hanging-node constraints.  chi is clamped to [0,1] after
# every solve, so a small BDF2 overshoot cannot leave the admissible range; the
# yield stress clamps too (`semquad_weak_factor`), so the clamp is not load
# bearing in the momentum equation.

"""
    semquad_weakening_source_g(sq, sp, qt, pr, theta, ux, uz, pv, chi)

Return the closure `g(x,y) = eps_pl(x,y)/weak_eps_ref` of the state
`(theta, ux, uz, pv, chi)` at an arbitrary `(x,y)` (the yielding indicator uses
`chi`, so the caller passes the same field the momentum solve used).
`weak_active(pr)` must be true.
"""
function semquad_weakening_source_g(sq::SemQuad, sp::SemQuad, qt::Quadtree,
                                    pr::Params, theta::Vector{Float64},
                                    ux::Vector{Float64}, uz::Vector{Float64},
                                    pv::Vector{Float64}, chi::Vector{Float64})
    leaves = Set(qt.leaves)
    pleaves = Set(sp.leaves)
    inv_ref = 1.0 / pr.weak_eps_ref
    return (x, y) -> begin
        T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
        ed = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
        p = _semquad_eval_at(sp, pv, x, y, pleaves)
        c = _semquad_eval_at(sq, chi, x, y, leaves)
        semquad_plastic_strain_rate(T, y, ed, p, c, pr) * inv_ref
    end
end

"""
    semquad_weakening_parts(sq, sp, qt, pr, chi_n, chi_nm1, theta, ux, uz, pv,
                            dt, dt_prev, firststep; advection=:galerkin,
                            chi_src=chi_n)

Assemble the implicit BDF2 operator of one weakening step and return a named
tuple `(M, C, Mg, A, rhs, one, Cfree, u0, Ar, ca, cb, cc, firststep, g)`.

* `M`, `C` are the scalar mass and advection matrices of the **current**
  velocity `(ux, uz)` (assembled by `assemble_semquad_energy` with the heating
  and EV options explicitly off, so `theta` only labels the field and no source
  is added);
* `Mg` is the weighted mass matrix `int g phi_i phi_j`, `g = eps_pl/eps_ref`;
* `A = ca*M + C + Mg + (1/tau_heal)*M` for `weak_mode = :rate` and
  `A = ca*M + C` for `:linear`;
* `rhs = cb*M*chi_n - cc*M*chi_nm1 + Mg*1`;
* `Cfree` is the hanging-node constraint map (no Dirichlet dofs);
* `chi_src` is the field used for the yielding indicator of `g` (the current
  Picard/Newton iterate), while the *time levels* are always the start-of-step
  `chi_n` and `chi_nm1` — the same rule as the energy step (docs/RHEOLOGY.md §3).
"""
function semquad_weakening_parts(sq::SemQuad, sp::SemQuad, qt::Quadtree,
                                 pr::Params, chi_n::Vector{Float64},
                                 chi_nm1::Vector{Float64},
                                 theta::Vector{Float64},
                                 ux::Vector{Float64}, uz::Vector{Float64},
                                 pv::Vector{Float64},
                                 dt::Float64, dt_prev::Float64,
                                 firststep::Bool;
                                 advection::Symbol=:galerkin,
                                 chi_src::Vector{Float64}=chi_n)
    weak_active(pr) ||
        throw(ArgumentError("semquad_weakening_parts called with weakening off"))
    N = sq.N
    # Mass and advection of the current velocity.  The energy assembly is the
    # same scalar advection operator the temperature uses; the heating/EV/latent
    # terms are explicitly disabled so `F` is zero and `K` is unused.
    M, _, C, _ = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                         advection=advection, theta=chi_n,
                                         rho_heating=false, rho_cp=false,
                                         heating_adiabatic=false,
                                         heating_dissipation=false)
    g = semquad_weakening_source_g(sq, sp, qt, pr, theta, ux, uz, pv, chi_src)
    # Same quadrature as `M`: p+2 Gauss points (the pressure-mass default).
    Mg = assemble_semquad_pressure_mass(sq; weight=g, nq=sq.p + 2)
    one = ones(Float64, N)
    if firststep
        ca = 1.0 / dt
        cb = 1.0 / dt
        cc = 0.0
    else
        w = dt / dt_prev
        a = (2.0 * w + 1.0) / (1.0 + w)
        b = 1.0 + w
        c = w * w / (1.0 + w)
        ca = a / dt
        cb = b / dt
        cc = c / dt
    end
    heal = isfinite(pr.weak_heal_time) ? 1.0 / pr.weak_heal_time : 0.0
    if pr.weak_mode === :rate
        A = (ca + heal) * M + C + Mg
    else
        A = ca * M + C
    end
    rhs = cb * (M * chi_n) - cc * (M * chi_nm1) + Mg * one
    cons = semquad_constraints(sq, qt)
    u0, Cfree, _ = _affine_constraint(N, cons, Dict{Int,Float64}())
    Ar = Cfree' * A * Cfree
    return (M=M, C=C, Mg=Mg, A=A, rhs=rhs, one=one, Cfree=Cfree, u0=u0,
            Ar=Ar, ca=ca, cb=cb, cc=cc, firststep=firststep, g=g)
end

"""
    solve_semquad_weakening(sq, sp, qt, pr, chi_n, chi_nm1, theta, ux, uz, pv,
                            dt, dt_prev, firststep; advection=:galerkin,
                            chi_src=chi_n, return_parts=false)

One implicit BDF2 step of the strain-weakening field (docs/RHEOLOGY.md §9.2):
returns the clamped `chi` of the new time level, or the assembled parts when
`return_parts=true` (the per-step Newton needs them for its residual and its
`d r_W/d chi` block).
"""
function solve_semquad_weakening(sq::SemQuad, sp::SemQuad, qt::Quadtree,
                                 pr::Params, chi_n::Vector{Float64},
                                 chi_nm1::Vector{Float64},
                                 theta::Vector{Float64},
                                 ux::Vector{Float64}, uz::Vector{Float64},
                                 pv::Vector{Float64},
                                 dt::Float64, dt_prev::Float64,
                                 firststep::Bool;
                                 advection::Symbol=:galerkin,
                                 chi_src::Vector{Float64}=chi_n,
                                 return_parts::Bool=false)
    parts = semquad_weakening_parts(sq, sp, qt, pr, chi_n, chi_nm1, theta, ux,
                                    uz, pv, dt, dt_prev, firststep;
                                    advection=advection, chi_src=chi_src)
    return_parts && return parts
    br = parts.Cfree' * (parts.rhs - parts.A * parts.u0)
    ur = parts.Ar \ br
    # The raw solution is returned unclamped: the Galerkin advection operator
    # can undershoot slightly (13 of 81 nodes, min -0.097 in the measured
    # `Ra=1e4` case), and clamping *inside* the solve would make the Picard
    # fixed point `chi = clamp(A^{-1}rhs)` differ from the per-step Newton's
    # root of `A chi = rhs` (measured: 14% difference in chi).  The driver
    # clamps the *stored* field to [0,1] at the end of every step, and
    # `semquad_weak_factor` clamps its argument, so the yield stress never sees
    # an out-of-range value.
    return parts.u0 + parts.Cfree * ur
end

"""
    semquad_weakening_jvp(parts, sq, qt, pr, dux, duz, chi; advection=:galerkin)

Velocity-direction part of the weakening residual,
`Cfree' * C(du) * chi`, where `C(du)` is the scalar advection operator of the
velocity perturbation (`parts.Cfree' * A * chi` and the source `Mg*1` are
independent of the velocity, so this is the whole `d r_W/du` block up to the
lagged yielding indicator of `g`; docs/RHEOLOGY.md §9.4).
"""
function semquad_weakening_jvp(parts, sq::SemQuad, qt::Quadtree, pr::Params,
                               dux::Vector{Float64}, duz::Vector{Float64},
                               chi::Vector{Float64};
                               advection::Symbol=:galerkin)
    _, _, Cd, _ = assemble_semquad_energy(sq, qt, pr, dux, duz;
                                          advection=advection, theta=chi,
                                          rho_heating=false, rho_cp=false,
                                          heating_adiabatic=false,
                                          heating_dissipation=false)
    return parts.Cfree' * (Cd * chi)
end

"""
    assemble_semquad_deta_chi(sq, qt, coef, u)

Momentum-block derivative with respect to the weakening field,

    (Kchi)_{i,j} = int 2 (d eta/d chi) eps(u):eps(v_i) phi_j,     (2N x N)

with `coef(x,y,edot)` the pointwise `d eta/d chi` (a wrapper around
`semquad_deta_chi` carrying T and chi) and the strain rate taken from the
velocity `u` that enters the viscosity.  This is the exact `d r_S/d chi` block
of the per-step Newton (docs/RHEOLOGY.md §9.4): the chi basis is the scalar
`p`-degree space, the quadrature the Stokes one (`p+2` Gauss points), and the
layout is the `Kp` one with the pressure test function replaced by `phi_j`.
"""
function assemble_semquad_deta_chi(sq::SemQuad, qt::Quadtree, coef::Function,
                                   u::Vector{Float64})
    N = sq.N
    p = sq.p
    n = p + 1
    nv = n * n
    fd = make_fedata(p, p)
    phi = zeros(Float64, nv)
    gx = zeros(Float64, nv)
    gz = zeros(Float64, nv)
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 2 * nv * nv * length(sq.leaves))
    sizehint!(J, 2 * nv * nv * length(sq.leaves))
    sizehint!(V, 2 * nv * nv * length(sq.leaves))
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            duxdx = 0.0; duxdy = 0.0; duzdx = 0.0; duzdy = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                duxdx += u[ids[k]] * gx[k]
                duxdy += u[ids[k]] * gz[k]
                duzdx += u[N + ids[k]] * gx[k]
                duzdy += u[N + ids[k]] * gz[k]
            end
            exd = duxdx; ezd = duzdy
            exz = 0.5 * (duxdy + duzdx)
            edot = sqrt(max(0.5 * (exd * exd + ezd * ezd + 2.0 * exz * exz),
                            0.0))
            c = coef(x, y, edot)
            c == 0.0 && continue
            for j in 1:nv
                gj = ids[j]
                wc = 2.0 * c * w
                for k in 1:nv
                    ph = wc * phi[j]
                    g = ids[k]
                    push!(I, g); push!(J, gj); push!(V, ph * (exd * gx[k] + exz * gz[k]))
                    push!(I, N + g); push!(J, gj); push!(V, ph * (ezd * gz[k] + exz * gx[k]))
                end
            end
        end
    end
    return sparse(I, J, V, 2 * N, N)
end

"""
    semquad_weak_stats(chi)

Diagnostics of a weakening field: `(min, max, mean, frac)` with `frac` the
fraction of nodes with `chi > 0.5`.
"""
function semquad_weak_stats(chi::Vector{Float64})
    N = length(chi)
    N == 0 && return (min=0.0, max=0.0, mean=0.0, frac=0.0)
    return (min=minimum(chi), max=maximum(chi), mean=sum(chi) / N,
            frac=count(>(0.5), chi) / N)
end
