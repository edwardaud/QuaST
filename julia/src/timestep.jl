# Time stepping: implicit BDF2 energy, Picard coupling with eta(T), adaptive dt,
# steady-state detection.

struct StokesSolution
    ux::Vector{Float64}
    uz::Vector{Float64}
    p::Vector{Float64}
    res::Float64
    eta_min::Float64
    eta_max::Float64
    nclip::Int
    velq::Union{Nothing,QuadVelocity}
end

StokesSolution(ux::Vector{Float64}, uz::Vector{Float64}, p::Vector{Float64},
               res::Float64, eta_min::Float64, eta_max::Float64, nclip::Int) =
    StokesSolution(ux, uz, p, res, eta_min, eta_max, nclip, nothing)

"""Shift p so that the mean pressure on the top boundary is zero (ASPECT surface normalization)."""
function _normalize_surface_pressure(m::Mesh, p::Vector{Float64})
    q = m.q
    seen = Set{Int}()
    s = 0.0
    n = 0
    for ex in 0:m.nxe-1
        g = element_pres_gids(m, ex, m.nze - 1)
        for a in 1:q+1
            gid = g[q*(q+1) + a]
            if !(gid in seen)
                push!(seen, gid)
                s += p[gid]
                n += 1
            end
        end
    end
    n == 0 && return p
    c = s / n
    return p .- c
end
"""Assemble and solve the Stokes system for the current temperature field."""
function solve_stokes(m::Mesh, fd::FEData, pr::Params,
                      theta::Vector{Float64}, dm::DofMap;
                      res_bar::Float64=1.0e-8, linear::Symbol=:direct,
                      velocity_space::Symbol=:taylor_hood,
                      refstate::Union{Nothing,RefState}=nothing,
                      tala::Bool=true,
                      p_prev::Union{Nothing,Vector{Float64}}=nothing,
                      mass_form::Symbol=:rho,
                      buoyancy_form::Symbol=:anomaly,
                      phases::Vector{PhaseTransition}=PhaseTransition[],
                      rheology::Union{Nothing,RheologyModel}=nothing,
                      u_prev::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}=nothing,
                      tosi::Union{Nothing,TosiParams}=nothing,
                      T_offset::Float64=0.0, ala_test::Bool=false)
    if velocity_space === :psi
        return solve_stokes_psi(m, fd, pr, theta, dm; res_bar=res_bar, linear=linear)
    elseif velocity_space === :psi_omega
        return solve_stokes_psi_omega(m, fd, pr, theta, dm;
                                      res_bar=res_bar, linear=linear)
    elseif velocity_space === :rt0
        return solve_stokes_rt0(m, fd, pr, theta, dm;
                                res_bar=res_bar, linear=linear)
    elseif velocity_space === :rt1
        return solve_stokes_rt1(m, fd, pr, theta, dm;
                                res_bar=res_bar, linear=linear)
    end
    sys = assemble_stokes(m, fd, pr, theta, dm; refstate=refstate, tala=tala,
                          p_prev=p_prev, mass_form=mass_form, buoyancy_form=buoyancy_form, phases=phases,
                          rheology=rheology, u_prev=u_prev, tosi=tosi, T_offset=T_offset, ala_test=ala_test)

    if pr.div_penalty > 0
        D = assemble_div_penalty(m, fd, dm)
        sys = StokesSystem(sys.A + pr.div_penalty * D, sys.f,
                           sys.eta_min, sys.eta_max, sys.nclip)
    end
    if linear === :schur
        u, p, res, _ = solve_stokes_schur(sys, dm; tol=res_bar)
        x = vcat(u, p)
    elseif linear === :schur_cg
        u, p, res, _ = solve_stokes_schur_cg(sys, dm; tol=res_bar)
        x = vcat(u, p)
    elseif linear === :gmres
        u, p, res, _ = solve_stokes_gmres(sys, dm; tol=res_bar)
        x = vcat(u, p)
    elseif linear === :mg && rheology === nothing && tosi === nothing
        mg = build_mg(m, fd, pr, theta, dm, sys; refstate=refstate, tala=tala,
                      mass_form=mass_form, buoyancy_form=buoyancy_form,
                      phases=phases, rheology=rheology, tosi=tosi)
        if mg === nothing
            x, res = solve_linear(sys.A, sys.f; res_bar=res_bar, name="Stokes", equilibrate=true, refine=5)
        else
            Mp = assemble_pressure_mass_visc(m, fd, pr, theta)
            gp = findall(!iszero, dm.red_p)
            Ml = lu(Mp[gp, gp])
            Bm = sys.A[mg.nvel+1:end, 1:mg.nvel]
            pre = r -> apply_mg!(zeros(length(r)), r, mg, sys.A, Ml, Bm)
            x, gres, _ = gmres_restarted(v -> sys.A * v, sys.f; precond=pre, tol=res_bar, maxit=200, restart=50)
            res = gres
            if !isfinite(res) || res > 1e-7
                x, res = solve_linear(sys.A, sys.f; res_bar=res_bar, name="Stokes", equilibrate=true, refine=5)
            end
        end
        LAST_STOKES_RESIDUAL[] = res
    else
        x, res = solve_linear(sys.A, sys.f; res_bar=res_bar, name="Stokes", equilibrate=true, refine=5)
        LAST_STOKES_RESIDUAL[] = res
    end
    ux, uz, p = _scatter_stokes(m, dm, x)
    refstate !== nothing && (p = _normalize_surface_pressure(m, p))
    return StokesSolution(ux, uz, p, res, sys.eta_min, sys.eta_max, sys.nclip)
end

"""
    stokes_system_general(m, fd, pr, theta, dm, fx, fz; kwargs...)

Assemble the Stokes system including an extra body force and return the
`StokesSystem` (A, f), for Newton residual evaluation.
"""
function stokes_system_general(m::Mesh, fd::FEData, pr::Params,
                               theta::Vector{Float64}, dm::DofMap,
                               fx::Function, fz::Function; kwargs...)
    sys = assemble_stokes(m, fd, pr, theta, dm; kwargs...)
    sys.f .+= assemble_body_force(m, fd, dm, fx, fz)
    return sys
end
"""Stokes solve with an extra body force (manufactured-solution gate G1)."""
function solve_stokes_general(m::Mesh, fd::FEData, pr::Params,
                              theta::Vector{Float64}, dm::DofMap,
                              fx::Function, fz::Function;
                              res_bar::Float64=1.0e-8, linear::Symbol=:direct,
                              rheology=nothing, refstate=nothing,
                              u_prev=nothing, smooth_caps::Bool=false,
                              newton::Bool=false)
    sys = assemble_stokes(m, fd, pr, theta, dm; rheology=rheology,
                          refstate=refstate, u_prev=u_prev,
                          smooth_caps=smooth_caps, newton=newton)
    f = copy(sys.f)
    f .+= assemble_body_force(m, fd, dm, fx, fz)
    A = sys.A

    if pr.div_penalty > 0
        A = A + pr.div_penalty * assemble_div_penalty(m, fd, dm)
    end
    if linear === :schur
        sys2 = StokesSystem(A, f, sys.eta_min, sys.eta_max, sys.nclip)
        u, p, res, _ = solve_stokes_schur(sys2, dm; tol=res_bar)
        x = vcat(u, p)
    elseif linear === :schur_cg
        sys2 = StokesSystem(A, f, sys.eta_min, sys.eta_max, sys.nclip)
        u, p, res, _ = solve_stokes_schur_cg(sys2, dm; tol=res_bar)
        x = vcat(u, p)
    else
        x, res = solve_linear(A, f; res_bar=res_bar, name="Stokes", equilibrate=true, refine=5)
        LAST_STOKES_RESIDUAL[] = res
    end
    ux, uz, p = _scatter_stokes(m, dm, x)
    refstate !== nothing && (p = _normalize_surface_pressure(m, p))
    return StokesSolution(ux, uz, p, res, sys.eta_min, sys.eta_max, sys.nclip)
end

function _scatter_stokes(m::Mesh, dm::DofMap, x::Vector{Float64})
    ux = zeros(Float64, m.Nscalar)
    uz = zeros(Float64, m.Nscalar)
    p = zeros(Float64, m.Npres)
    @inbounds for g in 1:m.Nscalar
        r = dm.red_ux[g]
        r > 0 && (ux[g] = x[r])
        r = dm.red_uz[g]
        r > 0 && (uz[g] = x[dm.nux + r])
    end
    @inbounds for g in 1:m.Npres
        r = dm.red_p[g]
        r > 0 && (p[g] = x[dm.nux + dm.nuz + r])
    end
    return ux, uz, p
end

"""
One implicit BDF2 energy solve.  ``theta_n`` and ``theta_nm1`` are the two
previous time levels; ``firststep`` selects backward Euler.  The advection and
SUPG matrices use the current velocity ``ux,uz``.
"""
function solve_energy(m::Mesh, fd::FEData, pr::Params,
                      theta_n::Vector{Float64}, theta_nm1::Vector{Float64},
                      ux::Vector{Float64}, uz::Vector{Float64},
                      dmt::ThetaMap, dt::Float64, dt_prev::Float64,
                      firststep::Bool; res_bar::Float64=1.0e-8,
                      advection::Symbol=:skew,
                      velq::Union{Nothing,QuadVelocity}=nothing,
                      ev::Float64=0.0,
                      ev_cmax::Float64=0.5,
                      ev_alpha::Int=2,
                      ev_gamma::Float64=0.0,
                      ev_timestep::Int=0,
                      ev_dt_ratio::Float64=1.0,
                      ev_length_scale::Float64=0.0,
                      ux_ev::Union{Nothing,Vector{Float64}}=nothing,
                      uz_ev::Union{Nothing,Vector{Float64}}=nothing,
                      nu_const::Float64=0.0,
                      nu_fun::Union{Nothing,Function}=nothing,
                      ev_bnd_neighbor::Bool=false,
                      ev_bnd_own::Bool=false,
                      heating_adiabatic::Bool=false,
                      heating_dissipation::Bool=false,
                      Di::Float64=0.0,
                      refstate::Union{Nothing,RefState}=nothing,
                      phases::Vector{PhaseTransition}=PhaseTransition[],
                      latent_heat::Bool=true,
                      p_prev::Union{Nothing,Vector{Float64}}=nothing,
                      ev_diag::Union{Nothing,Vector{Float64}}=nothing)
    em = assemble_energy(m, fd, pr, ux, uz, dmt; advection=advection, velq=velq,
                         ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                         ev_gamma=ev_gamma, ev_timestep=ev_timestep,
                         ev_dt_ratio=ev_dt_ratio,
                         ev_length_scale=ev_length_scale,
                         theta_now=theta_n,
                         theta_prev=theta_nm1, dt=(firststep ? dt : dt_prev),
                         ux_ev=ux_ev, uz_ev=uz_ev, nu_const=nu_const,
                         nu_fun=nu_fun, ev_bnd_neighbor=ev_bnd_neighbor,
                         ev_bnd_own=ev_bnd_own,
                         heating_adiabatic=heating_adiabatic,
                         heating_dissipation=heating_dissipation, Di=Di,
                         refstate=refstate,
                         phases=phases,
                         latent_heat=latent_heat,
                         p_prev=p_prev,
                         ev_diag=ev_diag)
    tn = gather_reduced(dmt.red, theta_n, dmt.n)
    if firststep
        A = (1.0 / dt) * em.M + em.K + em.C + em.S
        rhs = (1.0 / dt) * (em.M * tn) + em.F
    else
        w = dt / dt_prev
        a = (2.0 * w + 1.0) / (1.0 + w)
        b = 1.0 + w
        c = w * w / (1.0 + w)
        tprev = gather_reduced(dmt.red, theta_nm1, dmt.n)
        A = (a / dt) * em.M + em.K + em.C + em.S
        rhs = (b / dt) * (em.M * tn) - (c / dt) * (em.M * tprev) + em.F
    end
    x, res = solve_linear(A, rhs; res_bar=res_bar, name="energy")
    LAST_ENERGY_RESIDUAL[] = A * x - rhs
    theta_new = scatter_reduced(dmt.red, x, m.Nscalar)
    return theta_new, res
end

const LAST_ENERGY_RESIDUAL = Ref{Vector{Float64}}(Float64[])
const LAST_STOKES_RESIDUAL = Ref(0.0)

function _picard_metrics(m::Mesh, fd::FEData, pr::Params,
                         th::Vector{Float64}, thp::Vector{Float64},
                         zs::Vector{Float64}, st, stp, ez::Int, ex::Int)
    g = element_vel_gids(m, ex, ez)
    du = 0.0; um = 0.0; de = 0.0; em = 0.0
    for kk in eachindex(g)
        iz = div(g[kk] - 1, m.nvx) + 1
        T = Tbar(zs[iz], pr.H) + th[g[kk]]
        Tp = Tbar(zs[iz], pr.H) + thp[g[kk]]
        eta = exp(-log(pr.deta_T) * T)
        etap = exp(-log(pr.deta_T) * Tp)
        du = max(du, hypot(st.ux[g[kk]] - stp.ux[g[kk]], st.uz[g[kk]] - stp.uz[g[kk]]))
        um = max(um, hypot(st.ux[g[kk]], st.uz[g[kk]]))
        de = max(de, abs(eta - etap))
        em = max(em, eta)
    end
    return du / max(um, 1.0e-30), de / max(em, 1.0e-30)
end

function _write_T_frame!(path::String, idx::Int, t::Float64, T::Vector{Float64})
    mode = idx == 0 ? "w" : "r+"
    h5open(path, mode) do f
        gname = "frame_" * lpad(string(idx), 5, '0')
        g = create_group(f, gname)
        g["T"] = T
        HDF5.attrs(g)["time"] = t
        HDF5.attrs(f)["n_frames"] = idx + 1
    end
    return nothing
end


struct SimulationResult    theta::Vector{Float64}
    theta_prev::Vector{Float64}
    theta_prev2::Vector{Float64}
    dt_prev::Float64
    dt_prevprev::Float64
    ux::Vector{Float64}
    uz::Vector{Float64}
    p::Vector{Float64}
    t::Float64
    steps::Int
    dt::Float64
    Nu_top::Float64
    Nu_bot::Float64
    Vrms::Float64
    Nu_energy::Float64
    Nu_top_avg::Float64
    Nu_bot_avg::Float64
    Vrms_avg::Float64
    n_avg::Int
    history::Vector{Vector{Float64}}
end

"""Seed ``theta = amp cos(pi x/Lx) sin(pi z)`` on the GLL nodes.

The horizontal parity matters: with insulating side walls the unstable linear
mode is ``cos(pi x/Lx)`` (zero x-derivative at the walls), not ``sin``.  The
``sin`` seed is symmetric about the box centre and converges to a metastable
two-cell branch with a lower Nusselt number.
"""
function seed_theta(m::Mesh, rule::GLLRule; amp::Float64=1.0e-3)
    xs, zs = node_coords(m, rule)
    theta = zeros(Float64, m.Nscalar)
    @inbounds for iz in 1:m.nvz
        for ix in 1:m.nvx
            g = (iz - 1) * m.nvx + ix
            theta[g] = amp * cos(pi * xs[ix] / m.Lx) * sin(pi * zs[iz])
        end
    end
    return theta
end

"""
Run a simulation to ``tstop`` or to a steady state.

The time step is adaptive and *accuracy*-limited (``cfl * hmin / umax``), not
stability-limited: the energy solve is fully implicit.  A failed Picard step
halves ``dt`` and retries; it never advances silently.
"""
function _volume_gamma(m::Mesh, fd::FEData, pr::Params, refstate, theta::Vector{Float64},
                       uz::Vector{Float64}, Di::Float64)
    p = m.p
    w = fd.gll_v.weights
    tot = 0.0
    refstate === nothing && return 0.0
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez); ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for j in 1:p+1, i in 1:p+1
                k = (j - 1) * (p + 1) + i
                z = ze0 + 0.5 * (fd.gll_v.nodes[j] + 1.0) * hze
                rq, aq, cq, _, _, _, _ = refstate_at(refstate, z)
                rho_n = rq / refstate.rho0
                alpha_n = aq / refstate.alpha0
                # Match the assembled source term: the ALA temperature equation
                # uses the adiabatic reference density (assembly.cc:896-904) and
                # the total temperature including the adiabatic surface value.
                T = Tbar(z, pr.H) + refstate.Tad[end] + theta[g[k]]
                tot += (-Di * rho_n * alpha_n * T * uz[g[k]]) *
                       w[i] * w[j] * (m.hx / 2) * (hze / 2)
            end
        end
    end
    return tot
end

"""
    _volume_phi(m, fd, pr, refstate, theta, ux, uz, Di)

Volume integral of the shear heating ``(Di/Ra) 2 eta eps:eps`` evaluated with the
same Gauss rule and the same 2-D deviatoric strain rate as the energy assembly,
so it can be compared directly with the boundary fluxes in the global energy
budget ``Nu_top - Nu_bot = int(gamma) + int(Phi)`` at steady state.
"""
function _volume_phi(m::Mesh, fd::FEData, pr::Params, refstate,
                     theta::Vector{Float64}, ux::Vector{Float64},
                     uz::Vector{Float64}, Di::Float64)
    pv = fd.phi_v; dpv = fd.dphi_v
    tot = 0.0
    for ez in 0:m.nze-1
        hze = z_size(m, ez); ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq, qx in 1:fd.nq
                z = ze0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                uxq = 0.0; uzq = 0.0
                duxdx = 0.0; duxdz = 0.0; duzdx = 0.0; duzdz = 0.0
                thn = 0.0
                for j in 1:m.p+1, i in 1:m.p+1
                    k = (j - 1) * (m.p + 1) + i
                    b = pv[i, qx] * pv[j, qy]
                    gx = dpv[i, qx] * pv[j, qy] * (2.0 / m.hx)
                    gz = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                    uxq += ux[g[k]] * b; uzq += uz[g[k]] * b
                    duxdx += ux[g[k]] * gx; duxdz += ux[g[k]] * gz
                    duzdx += uz[g[k]] * gx; duzdz += uz[g[k]] * gz
                    thn += theta[g[k]] * b
                end
                Tloc_h = Tbar(z, pr.H) + thn +
                         (refstate === nothing ? 0.0 : refstate.Tad[end])
                divu = duxdx + duzdz
                exz = 0.5 * (duxdz + duzdx)
                eps2 = refstate === nothing ?
                       (duxdx^2 + duzdz^2 + 0.5 * (duxdz + duzdx)^2) :
                       ((duxdx - divu / 3)^2 + (duzdz - divu / 3)^2 + 2 * exz^2)
                eta = eta_at(Tloc_h, z, pr)[1]
                w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                tot += (Di / pr.Ra) * 2.0 * eta * eps2 * w
            end
        end
    end
    return tot
end

function _volume_T(m::Mesh, fd::FEData, pr::Params, theta::Vector{Float64},
                   refstate::Union{Nothing,RefState}=nothing)
    p = m.p
    w = fd.gll_v.weights
    tot = 0.0
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez); ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for j in 1:p+1, i in 1:p+1
                k = (j - 1) * (p + 1) + i
                z = ze0 + 0.5 * (fd.gll_v.nodes[j] + 1.0) * hze
                rho_n = 1.0; cp_n = 1.0
                if refstate !== nothing
                    rq, _, cq, _, _, _, _ = refstate_at(refstate, z)
                    rho_n = rq / refstate.rho0
                    cp_n = cq / refstate.cp0
                end
                tot += rho_n * cp_n * (Tbar(z, pr.H) + theta[g[k]]) *
                       w[i] * w[j] * (m.hx / 2) * (hze / 2)
            end
        end
    end
    return tot
end
"""CFL length: minimum GLL node spacing (`:node`) or element diameter (`:element`)."""
function cfl_length(m::Mesh, fd::FEData, basis::Symbol)
    basis in (:node, :element) ||
        throw(ArgumentError("cfl_basis must be :node or :element"))
    xs, zs = node_coords(m, fd.gll_v)
    basis === :node && return min(minimum(diff(xs)), minimum(diff(zs)))
    return minimum(sqrt(m.hx^2 + z_size(m, ez)^2) for ez in 0:m.nze-1)
end

"""Local CFL ratio `min_e h_e / max_e|u|` (a single global dt, but each element's
advective CFL is bounded; unlike `hmin/umax` it does not let a tiny, slow wall
cell throttle the whole domain)."""
function cfl_local_dt(m::Mesh, ux::Vector{Float64}, uz::Vector{Float64})
    n = m.p + 1
    best = Inf
    @inbounds for ez in 0:m.nze-1, ex in 0:m.nxe-1
        gids = element_vel_gids(m, ex, ez)
        um = 0.0
        for k in gids
            um = max(um, hypot(ux[k], uz[k]))
        end
        he = sqrt(m.hx^2 + z_size(m, ez)^2)
        um > 0.0 && (best = min(best, he / um))
    end
    return isfinite(best) ? best : Inf
end

function run_simulation(m::Mesh, fd::FEData, pr::Params;
                        theta0::Union{Nothing,Vector{Float64}}=nothing,
                        tstop::Float64=1.0, dt0::Float64=1.0e-4,
                        dtmax::Float64=1.0e-2, dtmin::Float64=1.0e-12,
                        cfl::Float64=2.0, cfl_basis::Symbol=:node, cfl_mode::Symbol=:global,
                        r_max::Float64=1.05, r_min::Float64=0.95,
                        umax_window::Int=0,
                        maxsteps::Int=100000,
                        maxpic::Int=3, nonlinear::Symbol=:picard, pic_tol::Float64=1.0e-6,
                        pic_relax::Float64=1.0,
                        steady_tol::Float64=1.0e-9, steady_window::Int=5,
                        tmin::Float64=0.0, report_every::Int=1,
                        verbose::Bool=true, project::Bool=false,
                        advection::Symbol=:skew, linear::Symbol=:direct,
                        tavg_start::Float64=Inf,
                        energy_stream::Bool=false,
                        velocity_space::Symbol=:taylor_hood,
                        ev::Float64=0.0,
                        ev_cmax::Float64=0.5,
                        ev_alpha::Int=2,
                        ev_gamma::Float64=0.0,
                        ev_length_scale::Float64=0.0,
                        nu_const::Float64=0.0,
                        nu_fun::Union{Nothing,Function}=nothing,
                        cbf_diag::Bool=false,
                        tmax_diag::Bool=false,
                        pic_diag::Bool=false,
                        log_path::String="", log_every::Int=20,
                        ev_bnd_neighbor::Bool=false,
                        ev_bnd_own::Bool=true,
                        heating_adiabatic::Bool=false,
                        heating_dissipation::Bool=false,
                        Di::Float64=0.0,
                        refstate::Union{Nothing,RefState}=nothing,
                        tala::Bool=true,
                        beta_coupling::Bool=true,
                        mass_form::Symbol=:rho,
                        buoyancy_form::Symbol=:anomaly,
                        phases::Vector{PhaseTransition}=PhaseTransition[],
                        rheology::Union{Nothing,RheologyModel}=nothing,
                        tosi::Union{Nothing,TosiParams}=nothing,
                        latent_heat::Bool=true,
                        save_dt::Float64=0.0,
                        save_path::String="T_frames.h5",
                        res_bar::Float64=1.0e-8)
    dm = make_dofmap(m, pr.bc)
    dmt = make_theta_map(m)
    ev_diag = zeros(Float64, 5)
    theta_n = theta0 === nothing ? seed_theta(m, fd.gll_v) : copy(theta0)
    theta_nm1 = copy(theta_n)
    theta_nm2 = copy(theta_n)

    st = solve_stokes(m, fd, pr, theta_n, dm; res_bar=res_bar, linear=linear,
                      velocity_space=velocity_space, refstate=refstate, tala=tala,
                      mass_form=mass_form, buoyancy_form=buoyancy_form, phases=phases,
                      rheology=rheology, tosi=tosi)
    ux_prev = copy(st.ux); uz_prev = copy(st.uz)
    ux_prev2 = copy(st.ux); uz_prev2 = copy(st.uz)
    if st.velq !== nothing
        velq = st.velq
    elseif energy_stream
        velq, _ = streamfunction_quads(m, fd, dm, st.ux, st.uz; verbose=verbose)
    else
        velq = st.velq !== nothing ? st.velq :
           (project ? potential_correction(m, fd, dm, st.ux, st.uz; verbose=verbose)[1] : nothing)
    end
    Nu_top, Nu_bot = nu_walls(m, fd.gll_v, theta_n, pr)
    has_quad = st.velq !== nothing || energy_stream
    Vr = has_quad ? quad_stats(m, fd, velq)[1] : vrms(m, fd, st.ux, st.uz)
    umax0 = has_quad ? quad_stats(m, fd, velq)[2] : maximum(hypot.(st.ux, st.uz))
    Nu_e = nu_energy(m, fd, pr, theta_n, st.ux, st.uz)
    verbose && @printf("  initial: Nu=%.6f Vrms=%.6f eta=[%.3e,%.3e] clip=%d res=%.1e\n",
                       Nu_top, Vr, st.eta_min, st.eta_max, st.nclip, st.res)
    verbose && flush(stdout)
    hist = Vector{Vector{Float64}}()
    push!(hist, [0.0, 0.0, Nu_top, Nu_bot, Vr, Nu_e, maximum(abs, theta_n),
                 st.res, 0.0, umax0])
    save_idx = 0
    next_save = save_dt
    if save_dt > 0.0
        xs_s, zs_s = node_coords(m, fd.gll_v)
        Tf0 = similar(theta_n)
        for iz in eachindex(zs_s), ix in eachindex(xs_s)
            g = (iz - 1) * m.nvx + ix
            Tf0[g] = (refstate === nothing ? Tbar(zs_s[iz], pr.H) : refstate_at(refstate, zs_s[iz])[6]) + theta_n[g]
            zs_s[iz] <= 0.0 && (Tf0[g] = 1.0)
            zs_s[iz] >= 1.0 && (Tf0[g] = 0.0)
        end
        _write_T_frame!(save_path, 0, 0.0, Tf0)
        save_idx = 1
    end
    sum_Nu = 0.0
    sum_Nub = 0.0
    sum_V = 0.0
    n_avg = 0

    t = 0.0
    dt = min(dt0, dtmax)
    dt_prev = dt
    dt_prevprev = dt
    step = 0
    n_pic_fail = 0
    n_dt_cut = 0
    n_cfl_override = 0
    umax_hist = Float64[]
    steady_count = 0
    Nu_prev = Nu_top
    # CFL length.  `:node` (default) is the smallest GLL node spacing, which
    # keeps the nodal Courant number at `cfl`; with p4 that spacing is ~0.17 of
    # the element width.  `:element` uses the element diameter (deal.II's cell
    # diameter), the convention of ASPECT's `CFL number`; the nodal Courant is
    # then ~6x larger, which is stable for implicit BDF2 but must be checked
    # against `:node` for transient accuracy (see the high-Ra plan).
    hmin = cfl_length(m, fd, cfl_basis)
    xs_h, zs_h = node_coords(m, fd.gll_v)   # diagnostics (tmax_diag)
    if !isempty(log_path)
        open(log_path, "w") do io
            println(io, "# step t dt Courant Tmax Vrms")
        end
    end
    theta_next = copy(theta_n)
    st_next = st
    res_e = 0.0
    t0 = time()
    xs_d, zs_d = tmax_diag ? node_coords(m, fd.gll_v) : (Float64[], Float64[])
    zs_pic = pic_diag ? node_coords(m, fd.gll_v)[2] : Float64[]
    st_prev_time = st
    hot_du = Float64[]; hot_de = Float64[]
    ref_du = Float64[]; ref_de = Float64[]
    while step < maxsteps && t < tstop
        theta_iter = copy(theta_n)
        st_loc = st
        re = 0.0
        pic_iters = 0
        for it in 1:(nonlinear === :newton ? 1 : maxpic)
            pic_iters = it
            st_loc = if nonlinear === :newton
                rheology === nothing && error("nonlinear=:newton requires rheology")
                solve_stokes_newton(m, fd, pr, theta_iter, dm; rheology=rheology,
                                    refstate=refstate, st0=st_loc)
            else
                solve_stokes(m, fd, pr, theta_iter, dm; res_bar=res_bar,
                             linear=linear, velocity_space=velocity_space,
                             refstate=refstate, tala=tala,
                             p_prev=(beta_coupling ? st.p : nothing), mass_form=mass_form,
                             buoyancy_form=buoyancy_form, phases=phases,
                             rheology=rheology, u_prev=(st_loc.ux, st_loc.uz), tosi=tosi)
            end
            if pic_diag && it == 1
                Tf = similar(theta_iter)
                for iz in eachindex(zs_pic), ix in 1:m.nvx
                    g = (iz - 1) * m.nvx + ix
                    Tf[g] = Tbar(zs_pic[iz], pr.H) + theta_iter[g]
                end
                im = argmax(Tf)
                iz0 = div(im - 1, m.nvx); ix0 = mod(im - 1, m.nvx)
                ezh = div(iz0, m.p); exh = div(ix0, m.p)
                du, de = _picard_metrics(m, fd, pr, theta_n, theta_nm1, zs_pic,
                                         st_loc, st_prev_time, ezh, exh)
                push!(hot_du, du); push!(hot_de, de)
                du, de = _picard_metrics(m, fd, pr, theta_n, theta_nm1, zs_pic,
                                         st_loc, st_prev_time, m.nze ÷ 2, m.nxe ÷ 2)
                push!(ref_du, du); push!(ref_de, de)
            end
            if st_loc.velq !== nothing
                velq = st_loc.velq
            elseif energy_stream
                velq, _ = streamfunction_quads(m, fd, dm, st_loc.ux, st_loc.uz; verbose=verbose)
            else
                velq = project ? potential_correction(m, fd, dm, st_loc.ux, st_loc.uz; verbose=verbose)[1] : nothing
            end
            fill!(ev_diag, 0.0)
            ux_ev = 0.5 .* (ux_prev .+ ux_prev2)
            uz_ev = 0.5 .* (uz_prev .+ uz_prev2)
            th_new, re = solve_energy(m, fd, pr, theta_n, theta_nm1,
                                      st_loc.ux, st_loc.uz, dmt,
                                      dt, dt_prev, step == 0; res_bar=res_bar,
                                      advection=advection, velq=velq, ev=ev,
                                      ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                                      ev_gamma=ev_gamma,
                                      ev_length_scale=ev_length_scale,
                                      ev_timestep=step,
                                      ev_dt_ratio=(step <= 1 ? 0.0 : dt / dt_prev),
                                      ux_ev=ux_ev, uz_ev=uz_ev,
                                      nu_const=nu_const, nu_fun=nu_fun,
                                      ev_bnd_neighbor=ev_bnd_neighbor,
                                      ev_bnd_own=ev_bnd_own,
                                      heating_adiabatic=heating_adiabatic,
                                      heating_dissipation=heating_dissipation,
                                      Di=Di,
                                      refstate=refstate,
                                      phases=phases,
                                      latent_heat=latent_heat,
                                      p_prev=st.p,
                                      ev_diag=ev_diag)
            change = norm(th_new - theta_iter) / max(norm(th_new), 1.0e-30)
            if !isfinite(change)
                n_pic_fail += 1
                break
            end
            theta_iter = (1.0 - pic_relax) * theta_iter + pic_relax * th_new
            if change < pic_tol
                break
            end
        end
        theta_next = theta_iter
        st_next = st_loc
        st_prev_time = st_loc
        res_e = re
        ux_prev2 = ux_prev; uz_prev2 = uz_prev
        ux_prev = copy(st_next.ux); uz_prev = copy(st_next.uz)

        # advance
        theta_nm2 = theta_nm1
        theta_nm1 = theta_n
        theta_n = theta_next
        dt_prevprev = dt_prev
        dt_prev = dt
        t += dt
        step += 1

        if save_dt > 0.0 && t >= next_save - 1.0e-15
            Tfr = similar(theta_n)
            for iz in eachindex(zs_s), ix in eachindex(xs_s)
                g = (iz - 1) * m.nvx + ix
                Tfr[g] = (refstate === nothing ? Tbar(zs_s[iz], pr.H) : refstate_at(refstate, zs_s[iz])[6]) + theta_n[g]
                zs_s[iz] <= 0.0 && (Tfr[g] = 1.0)
                zs_s[iz] >= 1.0 && (Tfr[g] = 0.0)
            end
            _write_T_frame!(save_path, save_idx, t, Tfr)
            save_idx += 1
            next_save += save_dt
        end

        Nu_top, Nu_bot = nu_walls(m, fd.gll_v, theta_n, pr)
        if st_next.velq !== nothing || energy_stream
            Vr, umax = quad_stats(m, fd, velq)
        else
            Vr = vrms(m, fd, st_next.ux, st_next.uz)
            umax = maximum(hypot.(st_next.ux, st_next.uz))
        end
        Nu_e = nu_energy(m, fd, pr, theta_n, st_next.ux, st_next.uz)
        Nu_cbf_bot, Nu_cbf_top = 0.0, 0.0
        if cbf_diag
            Nu_cbf_bot, Nu_cbf_top = cbf_wall_flux(m, fd, pr, theta_n, theta_nm1,
                                                   st_next.ux, st_next.uz, dt_prev;
                                                   theta_prev2=theta_nm2,
                                                   dt_prev=dt_prevprev,
                                                   ev=ev, ev_cmax=ev_cmax)
        end
        amp = maximum(abs, theta_n)
        push!(hist, [t, dt, Nu_top, Nu_bot, Vr, Nu_e, amp,
                     st_next.res, res_e, umax, _volume_T(m, fd, pr, theta_n, refstate),
                     _volume_gamma(m, fd, pr, refstate, theta_n, st_next.uz, Di),
                     _volume_phi(m, fd, pr, refstate, theta_n, st_next.ux,
                                 st_next.uz, Di)])
        if !isempty(log_path) && step % log_every == 0
            Tmax_now = 0.0
            for iz in eachindex(zs_h), ix in eachindex(xs_h)
                g = (iz - 1) * m.nvx + ix
                Tbase = refstate === nothing ? Tbar(zs_h[iz], pr.H) :
                        refstate_at(refstate, zs_h[iz])[6]
                Tmax_now = max(Tmax_now, Tbase + theta_n[g])
            end
            open(log_path, "a") do io
                @printf(io, "%6d %.6e %.3e %.3f %.6f %.6e\n",
                        step, t, dt, umax * dt / hmin, Tmax_now, Vr)
            end
        end
        if t >= tavg_start
            sum_Nu += Nu_top
            sum_Nub += Nu_bot
            sum_V += Vr
            n_avg += 1
        end

        if verbose && (step % report_every == 0)
            @printf("  step %6d t=%.4e dt=%.2e Nu=%.8f/%.8f Vrms=%.6f amp=%.3e res=%.1e/%.1e eta=[%.2e,%.2e] clip=%d pic=%d\n",
                    step, t, dt, Nu_top, Nu_bot, Vr, amp, st_next.res, res_e,
                    st_next.eta_min, st_next.eta_max, st_next.nclip, pic_iters)
            if ev > 0.0 && ev_diag[5] > 0.0
                @printf("     EV nu_e_max=%.3e nu_max_max=%.3e nu_h_max=%.3e nu_e_mean=%.3e\n",
                        ev_diag[1], ev_diag[2], ev_diag[3], ev_diag[4] / ev_diag[5])
            end
            if cbf_diag
                @printf("     CBF bottom=%.4e top=%.4e\n", Nu_cbf_bot, Nu_cbf_top)
            end
            if tmax_diag
                Tf = similar(theta_n)
                for iz in eachindex(zs_d), ix in eachindex(xs_d)
                    g = (iz - 1) * m.nvx + ix
                    Tf[g] = Tbar(zs_d[iz], pr.H) + theta_n[g]
                end
                im = argmax(Tf)
                iz0 = div(im - 1, m.nvx); ix0 = mod(im - 1, m.nvx)
                ezd = div(iz0, m.p); exd = div(ix0, m.p)
                evg = _ev_globals(m, fd, pr, refstate, theta_nm1, theta_nm2,
                                  st_next.ux, st_next.uz, 0.0, 0.0)
                Rmax, umaxd, max_pref, _ = _ev_cell_stats(
                    m, fd, pr, refstate, theta_nm1, theta_nm2,
                    st_next.ux, st_next.uz, dt_prevprev, ezd, exd, evg;
                    T_offset=0.0, Di=Di,
                    heating_adiabatic=heating_adiabatic,
                    heating_dissipation=heating_dissipation,
                    alpha=ev_alpha, gamma_stab=ev_gamma)
                he = sqrt(m.hx^2 + z_size(m, ezd)^2)
                nu_max_d = ev_cmax * max_pref * umaxd * he
                nu_e_d = evg.entropy_variation > 0.0 ?
                         ev * he * he * Rmax / evg.entropy_variation : 0.0
                nuh = min(nu_max_d, nu_e_d)
                (ezd == 0 || ezd == m.nze - 1) && (nuh = 0.0)
                @printf("     TMAX=%.4f z=%.4f x=%.4f ez=%d ex=%d uz=%.3e Rmax=%.3e nu=%.3e eta_min=%.3e entvar=%.3e\n",
                        Tf[im], zs_d[iz0+1], xs_d[ix0+1], ezd, exd, st_next.uz[im], Rmax, nuh,
                        st_next.eta_min, evg.entropy_variation)
            end
            flush(stdout)
        end

        # steady-state detection
        change_t = norm(theta_n - theta_nm1) / max(norm(theta_n), 1.0e-30)
        dnu = abs(Nu_top - Nu_prev)
        if t > tmin && change_t < steady_tol && dnu < steady_tol * max(1.0, abs(Nu_top))
            steady_count += 1
            steady_count >= steady_window && break
        else
            steady_count = 0
        end
        Nu_prev = Nu_top

        # adapt dt (accuracy, not stability)
        push!(umax_hist, umax)
        if umax_window > 0 && length(umax_hist) > umax_window
            popfirst!(umax_hist)
        end
        umax_cfl = umax_window > 0 ? maximum(umax_hist) : umax
        if cfl_mode === :local
            dt_local = cfl_local_dt(m, st_next.ux, st_next.uz)
            dt_cfl = isfinite(dt_local) ? cfl * dt_local : dtmax
        elseif umax_cfl > 0.0
            dt_cfl = cfl * hmin / umax_cfl
        else
            dt_cfl = dtmax
        end
        dt_target = min(dtmax, dt_cfl)
        if step == 0
            dt_new = dt_target
        else
            dt_lo = r_min * dt
            dt_hi = r_max * dt
            dt_new = min(max(dt_target, dt_lo), dt_hi)
            if dt_new > dt_cfl
                dt_new = dt_cfl
                n_cfl_override += 1
            end
        end
        dt_new = min(dtmax, max(dtmin, dt_new))
        (step > 0 && dt_new < 0.9 * dt) && (n_dt_cut += 1)
        dt = dt_new
    end

    Nu_top, Nu_bot = nu_walls(m, fd.gll_v, theta_n, pr)
    Vr = vrms(m, fd, st_next.ux, st_next.uz)
    Nu_e = nu_energy(m, fd, pr, theta_n, st_next.ux, st_next.uz)
    Nu_avg = n_avg > 0 ? sum_Nu / n_avg : Nu_top
    Nub_avg = n_avg > 0 ? sum_Nub / n_avg : Nu_bot
    V_avg = n_avg > 0 ? sum_V / n_avg : Vr
    @printf("ADAPT r_max=%.2f r_min=%.2f umax_window=%d pic_fail=%d dt_cut=%d cfl_override=%d\n",
            r_max, r_min, umax_window, n_pic_fail, n_dt_cut, n_cfl_override)
    verbose && @printf("  done: steps=%d t=%.6e wall=%.1fs\n", step, t, time() - t0)
    verbose && n_avg > 0 && @printf("  time average over %d samples: Nu=%.6f/%.6f Vrms=%.6f\n",
                                    n_avg, Nu_avg, Nub_avg, V_avg)
    if pic_diag && !isempty(hot_du)
        hm(x) = sum(x) / length(x)
        @printf("PICDIAG n=%d hot |du| mean/max %.3e/%.3e hot |deta| mean/max %.3e/%.3e\n",
                length(hot_du), hm(hot_du), maximum(hot_du), hm(hot_de), maximum(hot_de))
        @printf("PICDIAG ref |du| mean/max %.3e/%.3e ref |deta| mean/max %.3e/%.3e\n",
                hm(ref_du), maximum(ref_du), hm(ref_de), maximum(ref_de))
        @printf("PICDIAG ratios hot/ref du(mean) %.2f deta(mean) %.2f\n",
                hm(hot_du) / max(hm(ref_du), 1.0e-30),
                hm(hot_de) / max(hm(ref_de), 1.0e-30))
    end
    return SimulationResult(theta_n, theta_nm1, theta_nm2, dt_prev, dt_prevprev,
                            st_next.ux, st_next.uz, st_next.p, t, step,
                            dt, Nu_top, Nu_bot, Vr, Nu_e,
                            Nu_avg, Nub_avg, V_avg, n_avg, hist)
end

"""
Steady-state solver: Picard on the nonlinear steady Stokes/energy system,

    K theta + C(u) theta + S(u) theta = F(u),

with the Stokes solve re-done after every theta update.  This is the fast path
for the steady benchmarks (1a, 2a); it is not used for time-dependent extreme
runs.  Convergence is measured on ``||dtheta||/||theta||``; the reported
``Nu_top`` and ``Nu_bot`` must agree at the fixed point.
"""
function run_steady(m::Mesh, fd::FEData, pr::Params;
                    theta0::Union{Nothing,Vector{Float64}}=nothing,
                    maxiter::Int=60, tol::Float64=1.0e-9,
                    relax::Float64=0.5, verbose::Bool=true,
                    project::Bool=false, advection::Symbol=:skew,
                    linear::Symbol=:direct,
                    anderson::Int=0,
                    velocity_space::Symbol=:taylor_hood,
                    res_bar::Float64=1.0e-8,
                    tosi::Union{Nothing,TosiParams}=nothing,
                    T_offset::Float64=0.0,
                    refstate=nothing, tala::Bool=true, mass_form::Symbol=:rho,
                    buoyancy_form::Symbol=:anomaly,
                    phases::Vector{PhaseTransition}=PhaseTransition[],
                    heating_adiabatic::Bool=false, heating_dissipation::Bool=false,
                    Di::Float64=0.0, ala_test::Bool=false)
    dm = make_dofmap(m, pr.bc)
    dmt = make_theta_map(m)
    theta = theta0 === nothing ? seed_theta(m, fd.gll_v) : copy(theta0)
    st = solve_stokes(m, fd, pr, theta, dm; res_bar=res_bar, linear=linear,
                      velocity_space=velocity_space, tosi=tosi, T_offset=T_offset,
                      refstate=refstate, tala=tala, mass_form=mass_form,
                      buoyancy_form=buoyancy_form, phases=phases, ala_test=ala_test)
    velq = st.velq !== nothing ? st.velq :
           (project ? potential_correction(m, fd, dm, st.ux, st.uz; verbose=verbose)[1] : nothing)
    hist = Vector{Vector{Float64}}()
    t0 = time()
    Nu_top = 0.0
    Nu_bot = 0.0
    Vr = 0.0
    Nu_e = 0.0
    change = Inf
    theta_new = copy(theta)
    res_e = 0.0
    Ghist = Vector{Vector{Float64}}()
    ghist = Vector{Vector{Float64}}()
    for it in 1:maxiter
        em = assemble_energy(m, fd, pr, st.ux, st.uz, dmt; advection=advection, velq=velq,
                             T_offset=T_offset, heating_adiabatic=heating_adiabatic,
                             heating_dissipation=heating_dissipation, Di=Di,
                             refstate=refstate, phases=phases)
        A = em.K + em.C + em.S
        theta_sol, res_e = solve_linear(A, em.F; res_bar=res_bar, name="steady energy")
        theta_sol = scatter_reduced(dmt.red, theta_sol, m.Nscalar)
        G = (1.0 - relax) * theta + relax * theta_sol
        g = G - theta
        theta_next = G
        if anderson > 0
            push!(Ghist, G)
            push!(ghist, g)
            while length(Ghist) > anderson
                popfirst!(Ghist)
                popfirst!(ghist)
            end
            if length(Ghist) >= 2
                mm = length(Ghist)
                Gmat = hcat(Ghist...)
                gmat = hcat(ghist...)
                GtG = gmat' * gmat
                KKT = [GtG + 1.0e-12 * I ones(mm); ones(mm)' 0.0]
                rhs = vcat(zeros(Float64, mm), 1.0)
                sol = try
                    KKT \ rhs
                catch
                    nothing
                end
                if sol !== nothing
                    alpha = sol[1:mm]
                    cand = Gmat * alpha
                    if all(isfinite, cand) && norm(cand) < 1.0e6 * max(norm(G), 1.0)
                        theta_next = cand
                    end
                end
            end
        end
        change = norm(theta_next - theta) / max(norm(theta_next), 1.0e-30)
        theta = theta_next
        st = solve_stokes(m, fd, pr, theta, dm; res_bar=res_bar, linear=linear,
                      velocity_space=velocity_space, tosi=tosi,
                      u_prev=(st.ux, st.uz), T_offset=T_offset,
                      refstate=refstate, tala=tala, mass_form=mass_form,
                      buoyancy_form=buoyancy_form, phases=phases, ala_test=ala_test,
                      p_prev=st.p)
        velq = st.velq !== nothing ? st.velq :
           (project ? potential_correction(m, fd, dm, st.ux, st.uz; verbose=verbose)[1] : nothing)
        Nu_top, Nu_bot = nu_walls(m, fd.gll_v, theta, pr)
        Vr = vrms(m, fd, st.ux, st.uz)
        Nu_e = nu_energy(m, fd, pr, theta, st.ux, st.uz)
        push!(hist, [Float64(it), 0.0, Nu_top, Nu_bot, Vr, Nu_e,
                     maximum(abs, theta), st.res, res_e, maximum(hypot.(st.ux, st.uz))])
        verbose && @printf("  steady it %3d  Nu=%.8f/%.8f Vrms=%.6f change=%.2e res=%.1e/%.1e div=%.1e\n",
                           it, Nu_top, Nu_bot, Vr, change, st.res, res_e,
                           max_divergence(m, fd, st.ux, st.uz))
        flush(stdout)
        if change < tol && it >= 3
            verbose && @printf("  steady converged: iter=%d wall=%.1fs\n", it, time() - t0)
            return SimulationResult(theta, theta, theta, 0.0, 0.0, st.ux, st.uz, st.p, 0.0,
                                    it, 0.0, Nu_top, Nu_bot, Vr, Nu_e, Nu_top, Nu_bot, Vr, 1, hist)
        end
    end
    verbose && @printf("  steady NOT converged after %d iterations (change=%.2e)\n", maxiter, change)
    return SimulationResult(theta, theta, theta, 0.0, 0.0, st.ux, st.uz, st.p, 0.0,
                            maxiter, 0.0, Nu_top, Nu_bot, Vr, Nu_e, Nu_top, Nu_bot, Vr, 1, hist)
end





































