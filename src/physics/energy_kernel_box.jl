# Energy assembly: the historical box leaf kernel and its block/threading
# assembler.  Body kept character-for-character so the nine state hashes pin it.
# Assemble energy leaves `lo:hi` into `buf` (buffers are per block, so disjoint
# ranges can run concurrently).
function _semquad_assemble_energy_block!(buf::SemQuadEnergyBuf,
                                        cfg::SemQuadEnergyCfg, sq::SemQuad,
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
    # Hoisted out of the quadrature loop: `:global` gives one `latent_L` per
    # transition (bitwise the historical coefficient), `:per_phase` the stored
    # per-transition values.
    lcoef = semquad_latent_coefficients(pr, phases)
    # `semquad_latent_active` is a pure function of `(pr, phases)`; it was
    # re-evaluated at every quadrature point (160 000 times per assembly on the
    # 10 000-leaf state).  Same value, hoisted.
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
    evalw = SemQuadEvalWork(n)

    for ileaf in lo:hi
        lf = leafvec[ileaf]
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        nu_ev = ev_field !== nothing ? ev_field[ileaf] :
                (ev > 0.0 ?
                 semquad_ev_nu(sq, pr, theta, theta_prev, ux, uz, dt, lf, evg;
                               ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                               ev_timestep=ev_timestep, fd=fd_ev, d2tab=d2tab) : 0.0)
        ev_out !== nothing && (ev_out[ileaf] = nu_ev)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uxq = 0.0; uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                uxq += ux[ids[k]] * phi[k]
                uzq += uz[ids[k]] * phi[k]
            end
            # `rho_ref`/`eta_fun` are `Union{Nothing,Function}` fields of the
            # config struct.  A direct call joins the result type with `Any`
            # (the field is not specialised), which used to make `rho`/`src`
            # dynamically typed *inside* the nv x nv element loop: 328M boxed
            # allocations, 5 GiB and ~2.5 s per energy assembly on the
            # 1024-leaf blob tree.  The `::Float64` assertions keep the value
            # identical and the loop monomorphic.
            rho = (rho_ref === nothing ? 1.0 : rho_ref(y))::Float64
            Tq = 0.0
            if rho_heating || rho_cp || (heating_adiabatic && pr.Di > 0.0) ||
               latent_active
                theta === nothing &&
                    throw(ArgumentError("theta is required when latent heat is on"))
                # The `::Vector{Float64}` assertion is load-bearing: `theta` is a
                # `Union{Nothing,Vector{Float64}}` config field, and without it
                # inference makes `Tq` (and `lat_cp`, and `rho` below) `Any`,
                # which re-boxes the nv x nv element loop.
                # The quadrature point lies strictly inside `lf` (Gauss points
                # are interior), so the leaf-local evaluation is the same leaf
                # `_semquad_leaf_at` would find -- the point search is skipped,
                # the arithmetic is identical.
                Tq = semquad_base_T(pr, y) +
                     _semquad_eval_at_leaf!(evalw, sq, theta::Vector{Float64},
                                            x, y, lf)
                (rho_heating || rho_cp) && (rho = max(1.0 - Tq, 1.0e-6))
            end
            # Latent heat as an apparent heat capacity: with
            #   rho cp DT/Dt = div(k grad T) - rho L (Gamma_T DT/Dt + Gamma_z u_z)
            # the material derivative is scaled by (1 + L Gamma_T) and the
            # advective part of the phase change becomes a source -L Gamma_z u_z.
            lat_cp = 0.0
            lat_src = 0.0
            if latent_active
                # `Tl` is a *second* temperature: the latent coefficients can be
                # evaluated at the nonlinear iterate while the density/heating
                # terms above stay at the start-of-step `theta`.  With
                # `theta_lat === nothing` (every direct assembly call, G140
                # included) `Tl` is exactly `Tq` and the assembled matrices are
                # bit-identical to the pre-existing latent implementation.
                Tl = Tq
                if theta_lat !== nothing && theta_lat !== theta
                    Tl = semquad_base_T(pr, y) +
                         _semquad_eval_at_leaf!(evalw, sq,
                                                theta_lat::Vector{Float64},
                                                x, y, lf)
                end
                for i in eachindex(phases)
                    Li = lcoef[i]
                    pt = phases[i]
                    lat_cp += Li * phase_dgamma_dT(pt, y, Tl)
                    lat_src -= Li * phase_dgamma_dz(pt, y, Tl) * uzq
                end
                # ASPECT's guard (`advection.cc`): density*c_P + latent_heat_LHS
                # must stay non-negative.  Here cp = rho = 1, so the apparent
                # capacity is 1 + lat_cp; below zero the energy equation turns
                # backward in time and the run blows up (measured: latent_L = 10
                # with the G194 transition gives 1 + L*Gamma_T = -1.5 and
                # max|theta| = 14 after 120 steps).  In `:per_phase` mode
                # `lat_cp` is the sum over transitions of `latent_i*Gamma_T,i`.
                if 1.0 + lat_cp < 0.0
                    throw(ArgumentError("latent heat: the apparent heat capacity " *
                                        "1 + sum_i latent_i*dGamma_i/dT = " *
                                        "$(1.0 + lat_cp) is negative at y=$y; " *
                                        "reduce the latent coefficients or widen " *
                                        "the phase transitions"))
                end
            end
            src = -rho * uzq * semquad_base_gradient(pr, y) + lat_src
            # theta = T - Tbar, so lap(T) = lap(theta) + lap(Tbar); the base
            # curvature is a genuine source (zero for the linear Boussinesq
            # profile, (alpha g/cp)^2 Tad for the exponential adiabat).
            src += semquad_base_laplacian(pr, y)
            if rho_heating
                # `heating_scale` is 1.0 unless the driver is decaying `H`.
                src -= Tq * pr.H * heating_scale
            end
            if pr.Di > 0.0 && (heating_adiabatic || heating_dissipation)
                # Velocity gradients at this quadrature point (gx/gz are the
                # basis gradients, already scaled by 2/h).
                duxdx = 0.0; duxdz = 0.0; duzdx = 0.0; duzdz = 0.0
                for k in 1:nv
                    duxdx += ux[ids[k]] * gx[k]
                    duxdz += ux[ids[k]] * gz[k]
                    duzdx += uz[ids[k]] * gx[k]
                    duzdz += uz[ids[k]] * gz[k]
                end
                if heating_dissipation
                    exz = 0.5 * (duxdz + duzdx)
                    eed = duxdx^2 + duzdz^2 + 2.0 * exz^2      # edot:edot
                    if pr.deviatoric
                        # ASPECT embeds the 2-D flow in 3-D and uses the 1/3
                        # deviatoric projection (manual, "Two-dimensional
                        # models"), so eps_d:eps_d = eps:eps - (1/3) div(u)^2.
                        eed -= (1.0 / 3.0) * (duxdx + duzdz)^2
                    end
                    eta_q = (eta_fun === nothing ? 1.0 : eta_fun(x, y))::Float64
                    src += (pr.Di / pr.Ra) * 2.0 * eta_q * eed
                end
                if heating_adiabatic
                    # Tq already contains Tbar + theta (it is computed above when
                    # this term is on), so it must not be added again.
                    Tfull = theta === nothing ? semquad_base_T(pr, y) : Tq
                    # ASPECT's simplified adiabatic heating is alpha rho T (u.g):
                    # the density factor must stay (King has rho = exp(depth Di)).
                    rho_ad = (pr.refstate === nothing ? 1.0 :
                              semquad_rho(pr.refstate, y))::Float64
                    src -= pr.Di * rho_ad * Tfull * uzq
                end
            end
            # Local mass/stiffness/advection blocks.  The per-quadrature scalars
            # are hoisted once (their left-to-right product tree is preserved
            # exactly, e.g. `((1+lat_cp)*w)*rho*phi[k]` -> `cM*phi[k]`), and the
            # inner loop index is `k` so every `Me/Ke/Ce[k,l]` store is
            # unit-stride.  Each entry still accumulates over q in the same
            # order, so the assembled matrices are bitwise identical (G188).
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

# Split the energy leaves into `nt` contiguous blocks, assemble them (in
# parallel when Julia has more than one thread) and merge the block buffers in
# block order.  `nt == 1` is the historical serial path, returned without a
# copy or re-ordering.
#
# Cut 3: this is the historical **box** body, character for character (it is
# the arithmetic the nine state-dump hashes pin).  The mapped body is
# `_semquad_assemble_energy_blocks_mapped` below; `assemble_semquad_energy`
# selects it with a static `isa` branch on the stored geometry, so the box call
# is a plain static call (no dispatch frame, 0 B — the cut-2 allocation lesson).
function _semquad_assemble_energy_blocks_box(nt::Int, cfg::SemQuadEnergyCfg,
                                             sq::SemQuad, fd::FEData,
                                             leaves::Set{NTuple{3,Int}})
    n = sq.p + 1
    nv = n * n
    nleaf = length(sq.leaves)
    _SEMQUAD_ENERGY_BUF_USED[] = 0
    # EV quadrature data: `make_fedata(p, p)` ran once per *leaf* inside
    # `semquad_ev_cell_stats` (a GLL Newton iteration + a Gauss-Legendre
    # `eigen` + 6 matrices per leaf, 57.8 MB and 44 ms per assembly on the
    # 10 000-leaf state).  It depends only on `p`, so it is built once here and
    # threaded through; the values are identical, hence the EV field is bitwise
    # unchanged.
    fd_ev = (cfg.ev > 0.0 && cfg.ev_field === nothing) ?
            make_fedata(sq.p, sq.p) : nothing
    d2tab = fd_ev === nothing ? Vector{Vector{Float64}}() :
            [lagrange_d2(gll_rule_cached(sq.p), fd_ev.gl_x[k])[3]
             for k in 1:fd_ev.nq]
    if nt <= 1
        buf = _semquad_energy_buf(nleaf, nv, sq.N)
        _semquad_assemble_energy_block!(buf, cfg, sq, sq.leaves, 1, nleaf, fd,
                                        leaves, fd_ev, d2tab)
        # the COO arrays are consumed by `sparse` inside the caller; `F` is
        # returned to the caller (and `return_parts=true` keeps it across
        # further assemblies), so it must not alias the pooled buffer.
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
        _semquad_assemble_energy_block!(bufs[t], cfg, sq, sq.leaves, los[t],
                                        his[t], fd, leaves, fd_ev, d2tab)
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
