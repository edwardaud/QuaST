# Energy assembly: the mapped leaf kernel, its block/threading assembler and
# the gate-facing block dispatcher.  The box body above is not routed through
# the mapped metric (GEOM_DESIGN section 4.1/4.3).

# ---------------------------------------------------------------------------
# Cut 3: the **mapped** energy leaf kernel.
#
# This is a second body, not a reroute of the box one.  The box body above is
# the historical arithmetic character for character (`0.25*hx*hy*gl_w*gl_w`,
# `dphi*phi*(2.0/hx)`, the physical `y` in every closure) and it is what keeps
# the nine state-dump hashes; `GEOM_DESIGN` §4.1/§4.3 forbid routing it through
# `J` (F1), re-associating the weight tree (F3) or changing the box EV (F4).
# The mapped body gets the same quantities from the geometry:
#
#   (x, y)   = quad_point(lm, sq, lf, fd, qx, qy)   # mapped quadrature point
#   w        = quad_measure(lm, fd, qx, qy)         # detJ * gl_w * gl_w
#   (gx, gz) = basis_grad(lm, fd, qx, qy, i, j)     # J^{-T} chain rule
#   h        = height(g, x, y)                      # physical height
#   grad h   = grad_height(g, x, y),  lap h = height_laplacian(g, x, y)
#
# and everything downstream — the apparent heat capacity, the latent source,
# the base-state advection, the mapped base Laplacian, the radiogenic and
# adiabatic/dissipation sources, the mass/stiffness/advection blocks and the
# COO scatter — repeats the box expression tree with `y -> h` and `uz -> u.grad h`
# (GEOM_DESIGN §3 cut 3, §6.4).  That is why the two kernels agree to round-off
# when the mapped metric is fed the box map (`test/t_g269_curved_energy.jl`
# part (c)).  The leaf metric and curvature are built once per leaf here (the
# §2.5 cache is cut 5).
function _semquad_assemble_energy_block_mapped!(buf::SemQuadEnergyBuf,
                                               cfg::SemQuadEnergyCfg,
                                               g::AbstractGeometry,
                                               sq::SemQuad,
                                               leafvec::Vector{NTuple{3,Int}},
                                               lo::Int, hi::Int, fd::FEData,
                                               leaves::Set{NTuple{3,Int}},
                                               fd_ev::Union{Nothing,FEData}=nothing,
                                               d2tab::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    hi < lo && return buf
    pr = cfg.pr
    ux = cfg.ux
    uz = cfg.uz
    theta = cfg.theta
    theta_lat = cfg.theta_lat
    rho_heating = cfg.rho_heating
    heating_scale = cfg.heating_scale
    rho_cp = cfg.rho_cp
    ev = cfg.ev
    ev_cmax = cfg.ev_cmax
    ev_alpha = cfg.ev_alpha
    ev_timestep = cfg.ev_timestep
    evg = cfg.evg
    theta_prev = cfg.theta_prev
    dt = cfg.dt
    ev_field = cfg.ev_field
    ev_out = cfg.ev_out
    heating_adiabatic = cfg.heating_adiabatic
    heating_dissipation = cfg.heating_dissipation
    eta_fun = cfg.eta_fun
    rho_ref = cfg.rho_ref
    phases = cfg.phases
    lcoef = semquad_latent_coefficients(pr, phases)
    latent_active = semquad_latent_active(pr, phases)
    n = sq.p + 1
    nv = n * n
    IM = buf.IM; JM = buf.JM; VM = buf.VM
    IK = buf.IK; JK = buf.JK; VK = buf.VK
    IC = buf.IC; JC = buf.JC; VC = buf.VC
    F = buf.F
    scr = buf.scr
    phi = scr.phi
    gx = scr.gx
    gz = scr.gz
    Me = scr.Me
    Ke = scr.Ke
    Ce = scr.Ce

    for ileaf in lo:hi
        lf = leafvec[ileaf]
        ids = sq.elem[lf]
        lm = leaf_metric(g, sq, lf, fd)
        nu_ev = ev_field !== nothing ? ev_field[ileaf] :
                (ev > 0.0 ?
                 semquad_ev_nu(g, sq, pr, theta, theta_prev, ux, uz, dt, lf, evg;
                               ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                               ev_timestep=ev_timestep, fd=fd_ev, d2tab=d2tab) : 0.0)
        ev_out !== nothing && (ev_out[ileaf] = nu_ev)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(lm, sq, lf, fd, qx, qy)
            w = quad_measure(lm, fd, qx, qy)
            h = height(g, x, y)
            uxq = 0.0; uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k], gz[k] = basis_grad(lm, fd, qx, qy, i, j)
                uxq += ux[ids[k]] * phi[k]
                uzq += uz[ids[k]] * phi[k]
            end
            rho = (rho_ref === nothing ? 1.0 : rho_ref(h))::Float64
            Tq = 0.0
            if rho_heating || rho_cp || (heating_adiabatic && pr.Di > 0.0) ||
               latent_active
                theta === nothing &&
                    throw(ArgumentError("theta is required when latent heat is on"))
                thq = 0.0
                # `theta` is a Union field of the config; the assertion keeps the
                # accumulation monomorphic (the box body's algebra does the same
                # through `_semquad_eval_at_leaf!`).
                for k in 1:nv
                    thq += (theta::Vector{Float64})[ids[k]] * phi[k]
                end
                # Stage 0.5b1: the base state is the geometry's conductive
                # profile (the annulus's radial log/H state), not the Cartesian
                # quadratic evaluated at `h`; on the box/validation geometries
                # the dispatch returns exactly the historical `Tbar(h,H)`.
                Tq = semquad_base_T(pr, g, x, y) + thq
                (rho_heating || rho_cp) && (rho = max(1.0 - Tq, 1.0e-6))
            end
            # `u.grad h`: the mapped generalisation of `u_z` in every
            # directional term (GEOM_DESIGN §2.7/§6.4).
            ugh = upward_velocity(g, x, y, uxq, uzq)
            lat_cp = 0.0
            lat_src = 0.0
            if latent_active
                Tl = Tq
                if theta_lat !== nothing && theta_lat !== theta
                    thl = 0.0
                    for k in 1:nv
                        thl += (theta_lat::Vector{Float64})[ids[k]] * phi[k]
                    end
                    Tl = semquad_base_T(pr, g, x, y) + thl
                end
                for i in eachindex(phases)
                    Li = lcoef[i]
                    pt = phases[i]
                    lat_cp += Li * phase_dgamma_dT(pt, h, Tl)
                    lat_src -= Li * phase_dgamma_dz(pt, h, Tl) * ugh
                end
                if 1.0 + lat_cp < 0.0
                    throw(ArgumentError("latent heat: the apparent heat capacity " *
                                        "1 + sum_i latent_i*dGamma_i/dT = " *
                                        "$(1.0 + lat_cp) is negative at h=$h; " *
                                        "reduce the latent coefficients or widen " *
                                        "the phase transitions"))
                end
            end
            src = -rho * ugh * semquad_base_gradient(pr, g, x, y) + lat_src
            # lap(Tbar(h)): the physical Laplacian of the base state in the
            # mapped metric, `Tbar''(h)|grad h|^2 + Tbar'(h) lap(h)`, plus the
            # absorbed `+H` for the Boussinesq base (`_mapped_base_curvature`;
            # the box body's `semquad_base_laplacian(pr,y)` term is the same
            # expression with `|grad h| = 1`, `lap y = 0`).  Cut 4 defect B1: the
            # first mapped body used the reference-state second derivative
            # alone, which is 0 for the Boussinesq base and silently dropped
            # `H*(1-|grad h|^2)` on a geometry with `|grad h| != 1`.
            src += _mapped_base_curvature(pr, g, x, y)
            if rho_heating
                src -= Tq * pr.H * heating_scale
            end
            if pr.Di > 0.0 && (heating_adiabatic || heating_dissipation)
                duxdx = 0.0; duxdz = 0.0; duzdx = 0.0; duzdz = 0.0
                for k in 1:nv
                    duxdx += ux[ids[k]] * gx[k]
                    duxdz += ux[ids[k]] * gz[k]
                    duzdx += uz[ids[k]] * gx[k]
                    duzdz += uz[ids[k]] * gz[k]
                end
                if heating_dissipation
                    exz = 0.5 * (duxdz + duzdx)
                    eed = duxdx^2 + duzdz^2 + 2.0 * exz^2
                    if pr.deviatoric
                        eed -= (1.0 / 3.0) * (duxdx + duzdz)^2
                    end
                    eta_q = (eta_fun === nothing ? 1.0 : eta_fun(x, y))::Float64
                    src += (pr.Di / pr.Ra) * 2.0 * eta_q * eed
                end
                if heating_adiabatic
                    Tfull = theta === nothing ? semquad_base_T(pr, g, x, y) : Tq
                    rho_ad = (pr.refstate === nothing ? 1.0 :
                              semquad_rho(pr.refstate, h))::Float64
                    src -= pr.Di * rho_ad * Tfull * ugh
                end
            end
            cM = (1.0 + lat_cp) * w * rho
            cK = w * (1.0 + nu_ev)
            ws = w * src
            for k in 1:nv
                F[ids[k]] += ws * phi[k]
            end
            for l in 1:nv
                pl = phi[l]
                xl = gx[l]
                zl = gz[l]
                cx = uxq * xl + uzq * zl
                @simd for k in 1:nv
                    pk = phi[k]
                    ck = cM * pk
                    Me[k, l] += ck * pl
                    Ke[k, l] += cK * (gx[k] * xl + gz[k] * zl)
                    Ce[k, l] += ck * cx
                end
            end
        end
        for l in 1:nv, k in 1:nv
            vm = Me[k, l]
            vm != 0.0 && (push!(IM, ids[k]); push!(JM, ids[l]); push!(VM, vm))
            vk = Ke[k, l]
            vk != 0.0 && (push!(IK, ids[k]); push!(JK, ids[l]); push!(VK, vk))
            vc = Ce[k, l]
            vc != 0.0 && (push!(IC, ids[k]); push!(JC, ids[l]); push!(VC, vc))
        end
        fill!(Me, 0.0)
        fill!(Ke, 0.0)
        fill!(Ce, 0.0)
    end
    return buf
end

# Mapped block assembler: the box blocks function above with the mapped leaf
# kernel.  Same buffer pool, same block merge order, same returned NamedTuple
# type (so the `isa` branch in `assemble_semquad_energy` is type stable).
function _semquad_assemble_energy_blocks_mapped(nt::Int,
                                                cfg::SemQuadEnergyCfg,
                                                g::AbstractGeometry,
                                                sq::SemQuad, fd::FEData,
                                                leaves::Set{NTuple{3,Int}})
    n = sq.p + 1
    nv = n * n
    nleaf = length(sq.leaves)
    _SEMQUAD_ENERGY_BUF_USED[] = 0
    fd_ev = (cfg.ev > 0.0 && cfg.ev_field === nothing) ?
            make_fedata(sq.p, sq.p) : nothing
    d2tab = fd_ev === nothing ? Vector{Vector{Float64}}() :
            [lagrange_d2(gll_rule_cached(sq.p), fd_ev.gl_x[k])[3]
             for k in 1:fd_ev.nq]
    if nt <= 1
        buf = _semquad_energy_buf(nleaf, nv, sq.N)
        _semquad_assemble_energy_block_mapped!(buf, cfg, g, sq, sq.leaves, 1,
                                               nleaf, fd, leaves, fd_ev, d2tab)
        return (IM=buf.IM, JM=buf.JM, VM=buf.VM,
                IK=buf.IK, JK=buf.JK, VK=buf.VK,
                IC=buf.IC, JC=buf.JC, VC=buf.VC, F=copy(buf.F))
    end
    bufs = Vector{SemQuadEnergyBuf}(undef, nt)
    los = Vector{Int}(undef, nt)
    his = Vector{Int}(undef, nt)
    base, rem = divrem(nleaf, nt)
    off = 1
    for t in 1:nt
        cnt = base + (t <= rem ? 1 : 0)
        los[t] = off
        his[t] = off + cnt - 1
        off += cnt
        bufs[t] = _semquad_energy_buf(cnt, nv, sq.N)
    end
    Threads.@threads :static for t in 1:nt
        _semquad_assemble_energy_block_mapped!(bufs[t], cfg, g, sq, sq.leaves,
                                               los[t], his[t], fd, leaves,
                                               fd_ev, d2tab)
    end
    F = zeros(Float64, sq.N)
    for t in 1:nt
        F .+= bufs[t].F
    end
    return (IM=_semquad_concat_blocks([b.IM for b in bufs]),
            JM=_semquad_concat_blocks([b.JM for b in bufs]),
            VM=_semquad_concat_blocks([b.VM for b in bufs]),
            IK=_semquad_concat_blocks([b.IK for b in bufs]),
            JK=_semquad_concat_blocks([b.JK for b in bufs]),
            VK=_semquad_concat_blocks([b.VK for b in bufs]),
            IC=_semquad_concat_blocks([b.IC for b in bufs]),
            JC=_semquad_concat_blocks([b.JC for b in bufs]),
            VC=_semquad_concat_blocks([b.VC for b in bufs]), F=F)
end

# Compatibility entry point for the gates that drive the block assembler
# directly (G181/G188, the same pattern as cut 2's `_semquad_assemble_blocks`
# shim): dispatches on the stored geometry.  The production path in
# `assemble_semquad_energy` calls the two concrete functions above directly, so
# this shim's dispatch frame is never in a hot path (it costs the measured
# cut-2 dispatch allocation in the gates only).
function _semquad_assemble_energy_blocks(nt::Int, cfg::SemQuadEnergyCfg,
                                         sq::SemQuad, fd::FEData,
                                         leaves::Set{NTuple{3,Int}})
    return sq.geom isa CartesianBox ?
           _semquad_assemble_energy_blocks_box(nt, cfg, sq, fd, leaves) :
           _semquad_assemble_energy_blocks_mapped(nt, cfg, sq.geom, sq, fd,
                                                  leaves)
end

