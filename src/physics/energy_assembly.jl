
# Per-block buffers for the energy assembly (S4 stage 1).  `assemble_threads=1`
# is the historical serial path; >1 uses one of these per contiguous leaf block
# and merges the blocks in order (deterministic run to run; the load vector is a
# sum of block sums, everything else is assembled in the same leaf order as the
# serial path).
struct SemQuadEnergyScratch
    phi::Vector{Float64}
    gx::Vector{Float64}
    gz::Vector{Float64}
    Me::Matrix{Float64}
    Ke::Matrix{Float64}
    Ce::Matrix{Float64}
end

struct SemQuadEnergyBuf
    IM::Vector{Int}
    JM::Vector{Int}
    VM::Vector{Float64}
    IK::Vector{Int}
    JK::Vector{Int}
    VK::Vector{Float64}
    IC::Vector{Int}
    JC::Vector{Int}
    VC::Vector{Float64}
    F::Vector{Float64}
    scr::SemQuadEnergyScratch
end

struct SemQuadEnergyCfg
    pr::Params
    ux::Vector{Float64}
    uz::Vector{Float64}
    theta::Union{Nothing,Vector{Float64}}
    # Temperature at which the *latent-heat coefficients* are evaluated.  `nothing`
    # means "the same `theta` as everything else" (the historical behaviour, and
    # what a direct `assemble_semquad_energy(...; theta=...)` call gets).  The
    # time-stepping driver passes the current nonlinear iterate here so that the
    # apparent heat capacity is implicit in `theta` while the `rho_heating` /
    # `rho_cp` / adiabatic-heating densities keep using the start-of-step
    # temperature (bitwise unchanged).
    theta_lat::Union{Nothing,Vector{Float64}}
    rho_heating::Bool
    rho_cp::Bool
    ev::Float64
    ev_cmax::Float64
    ev_alpha::Int
    ev_timestep::Int
    evg::Union{Nothing,SEMEVGlobals}
    theta_prev::Union{Nothing,Vector{Float64}}
    dt::Float64
    ev_field::Union{Nothing,Vector{Float64}}
    ev_out::Union{Nothing,Vector{Float64}}
    heating_adiabatic::Bool
    heating_dissipation::Bool
    eta_fun::Union{Nothing,Function}
    rho_ref::Union{Nothing,Function}
    phases::Vector{PhaseTransition}
    # Multiplier on the radiogenic source `H`, so a heating rate can decay in
    # time without rebuilding the operator: 1.0 is the constant-`H` path.
    heating_scale::Float64
end

# The 21-argument form predates `heating_scale`: callers that build the config by
# position (gates do, to drive the leaf kernels directly) keep working, and get
# the constant-`H` source.  Internal callers use the 22-argument form.
SemQuadEnergyCfg(pr, ux, uz, theta, theta_lat, rho_heating, rho_cp, ev,
                 ev_cmax, ev_alpha, ev_timestep, evg, theta_prev, dt,
                 ev_field, ev_out, heating_adiabatic, heating_dissipation,
                 eta_fun, rho_ref, phases) =
    SemQuadEnergyCfg(pr, ux, uz, theta, theta_lat, rho_heating, rho_cp, ev,
                     ev_cmax, ev_alpha, ev_timestep, evg, theta_prev, dt,
                     ev_field, ev_out, heating_adiabatic, heating_dissipation,
                     eta_fun, rho_ref, phases, 1.0)

function _semquad_energy_buf_alloc(nleaf::Int, nv::Int, N::Int)
    IM = Int[]; JM = Int[]; VM = Float64[]
    sizehint!(IM, nv * nv * nleaf)
    sizehint!(JM, nv * nv * nleaf)
    sizehint!(VM, nv * nv * nleaf)
    IK = Int[]; JK = Int[]; VK = Float64[]
    sizehint!(IK, nv * nv * nleaf)
    sizehint!(JK, nv * nv * nleaf)
    sizehint!(VK, nv * nv * nleaf)
    IC = Int[]; JC = Int[]; VC = Float64[]
    sizehint!(IC, nv * nv * nleaf)
    sizehint!(JC, nv * nv * nleaf)
    sizehint!(VC, nv * nv * nleaf)
    scr = SemQuadEnergyScratch(zeros(Float64, nv), zeros(Float64, nv),
                               zeros(Float64, nv), zeros(Float64, nv, nv),
                               zeros(Float64, nv, nv), zeros(Float64, nv, nv))
    return SemQuadEnergyBuf(IM, JM, VM, IK, JK, VK, IC, JC, VC,
                            zeros(Float64, N), scr)
end

# Pool of energy-assembly buffers, keyed by the local block size `nv` (S4/S15
# workspace rule: the owner creates the scratch once, the loop reuses it).  At
# p=2 and 10 000 leaves one acquisition `sizehint!`s nine arrays to 810 000
# entries = 58.3 MB; without the pool that capacity was allocated and dropped on
# *every* `assemble_semquad_energy` call (~58 MB of garbage per energy step).
# `_SEMQUAD_ENERGY_BUF_USED` is the per-assembly cursor: `assemble_threads > 1`
# hands one buffer to each contiguous block, so the blocks never share scratch.
# Buffers are `empty!`ed and re-`sizehint!`ed on acquisition, so a smaller tree
# reuses the larger capacity and the COO order is the pushing order, as before
# (bitwise identical assembly).  At most four distinct `nv` keys are kept.
const _SEMQUAD_ENERGY_BUF_CACHE = Dict{Int,Vector{SemQuadEnergyBuf}}()
const _SEMQUAD_ENERGY_BUF_USED = Ref(0)

function _semquad_energy_buf(nleaf::Int, nv::Int, N::Int)
    length(_SEMQUAD_ENERGY_BUF_CACHE) > 4 && empty!(_SEMQUAD_ENERGY_BUF_CACHE)
    pool = get!(() -> SemQuadEnergyBuf[], _SEMQUAD_ENERGY_BUF_CACHE, nv)
    idx = _SEMQUAD_ENERGY_BUF_USED[] + 1
    if idx > length(pool)
        push!(pool, _semquad_energy_buf_alloc(nleaf, nv, N))
    end
    _SEMQUAD_ENERGY_BUF_USED[] = idx
    buf = pool[idx]
    cap = nv * nv * nleaf
    empty!(buf.IM); sizehint!(buf.IM, cap)
    empty!(buf.JM); sizehint!(buf.JM, cap)
    empty!(buf.VM); sizehint!(buf.VM, cap)
    empty!(buf.IK); sizehint!(buf.IK, cap)
    empty!(buf.JK); sizehint!(buf.JK, cap)
    empty!(buf.VK); sizehint!(buf.VK, cap)
    empty!(buf.IC); sizehint!(buf.IC, cap)
    empty!(buf.JC); sizehint!(buf.JC, cap)
    empty!(buf.VC); sizehint!(buf.VC, cap)
    resize!(buf.F, N)
    fill!(buf.F, 0.0)
    return buf
end

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
    # per-transition values (docs/dev/PHYSICS.md §7.8).
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
                Tq = semquad_base_T(pr, h) + thq
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
                    Tl = semquad_base_T(pr, h) + thl
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
            src = -rho * ugh * semquad_base_gradient(pr, h) + lat_src
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
                    Tfull = theta === nothing ? semquad_base_T(pr, h) : Tq
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


"""
    assemble_semquad_energy(sq, qt, pr, ux, uz; advection=:skew)

Assemble `M` (mass), `K` (diffusion), `C` (advection of theta) and `F` (base
state `-u_z dTbar/dz`) on the quadtree.  `ux,uz` live on the same Q_p
quadtree space as theta; `advection` is `:galerkin`, `:skew` (`0.5(C-C')`) or
`:conservative` (`-C'`).  `rho_heating=true` adds the ASPECT-style
`rho H` correction `-T H` (since `rho=1-T`, `rho H - H = -T H`) and requires
`theta`.  No boundary conditions or hanging constraints are
applied here; the caller forms `C' A C` with `ThetaMap`.

`ev_field` reuses an elementwise EV field from an earlier assembly (one value per
leaf, in `sq.leaves` order) instead of recomputing it from `ev`/`evg`; this is
what lets the CBF residual use the exact operator that produced the state.
`ev_out` records the field that was actually used, so the driver can checkpoint
it.  Both are length-checked against `sq.leaves`.

`assemble_threads=1` (default) assembles the leaves serially, bit for bit as
before.  `assemble_threads=N>1` (S4 stage 1) cuts the leaves into `N` fixed
contiguous blocks, assembles each block into its own COO buffers and merges them
in block order; the result is deterministic run to run and the COO order equals
the serial one, but `F` is a sum of block sums (so it can differ at round-off).
With 1 Julia thread the request warns once and falls back to serial.

`phases` turns on the latent heat of the phase transitions as an apparent heat
capacity (docs/dev/PHYSICS.md §7): `M` and `C` are weighted by
`(1 + sum_i latent_i * Gamma_T,i(y,T))` and the source gains
`-sum_i latent_i * Gamma_z,i * u_z` (`Gamma_i` is the same `phase_gamma` the
phase buoyancy uses).  The coefficient `latent_i` is `pr.latent_L` for every
transition in `latent_mode = :global` (the historical behaviour) and
`pt.latent` in `latent_mode = :per_phase` (docs/dev/PHYSICS.md §7.8, where the
sign convention — the coefficient is ASPECT's `T dS_A` with the opposite sign —
is derived).  `theta_lat` evaluates the capacity/source coefficients at a
different temperature than `theta` (the current iterate of a time step, so the
capacity is implicit); it defaults to `theta` and only changes anything when
phases are active.

**Mapped geometry (stage 0 cut 3).**  When `sq` was built with a non-box
geometry (`make_semquad(qt, p; geom=CurvedBox(...))`), the leaf kernel is the
second body `_semquad_assemble_energy_block_mapped!`: the quadrature point,
volume measure and basis gradients come from the leaf metric, the physics
closures are evaluated at the physical **height** `h = height(g,x,y)` instead of
`y`, the directional terms use `u . grad h` instead of `u_z` (`upward_velocity`),
the base Laplacian is `Tbar''(h)|grad h|^2 + Tbar'(h) lap(h)`, and the
entropy-viscosity residual uses the map-curvature Laplacian
(`lap_phi`/`map_curvature`) with the mapped element length
`he = sqrt(2*|element|)` (`GEOM_DESIGN` §3 cut 3/§6.4).  The box body above is
unchanged; the two are selected once per assembly by the geometry type (a
static call on the box path, so the box allocation is unchanged).
"""
function assemble_semquad_energy(sq::SemQuad, qt::Quadtree, pr::Params,
                                 ux::Vector{Float64}, uz::Vector{Float64};
                                 advection::Symbol=:skew,
                                 theta::Union{Nothing,Vector{Float64}}=nothing,
                                 theta_lat::Union{Nothing,Vector{Float64}}=nothing,
                                 rho_heating::Bool=false,
                                 rho_cp::Bool=false,
                                 ev::Float64=0.0, ev_cmax::Float64=0.052,
                                 ev_alpha::Int=2, ev_timestep::Int=0,
                                 evg::Union{Nothing,SEMEVGlobals}=nothing,
                                 theta_prev::Union{Nothing,Vector{Float64}}=nothing,
                                 dt::Float64=0.0,
                                 ev_field::Union{Nothing,Vector{Float64}}=nothing,
                                 ev_out::Union{Nothing,Vector{Float64}}=nothing,
                                 heating_adiabatic::Bool=false,
                                 heating_dissipation::Bool=false,
                                 eta_fun::Union{Nothing,Function}=nothing,
                                 rho_ref::Union{Nothing,Function}=nothing,
                                 phases::Vector{PhaseTransition}=PhaseTransition[],
                                 heating_scale::Float64=1.0,
                                 assemble_threads::Int=1)
    advection in (:galerkin, :skew, :conservative) ||
        throw(ArgumentError("advection must be galerkin, skew or conservative"))
    assemble_threads >= 1 ||
        throw(ArgumentError("assemble_threads must be >= 1"))
    (rho_heating || rho_cp || (heating_adiabatic && pr.Di > 0.0)) &&
        theta === nothing &&
        throw(ArgumentError("theta is required when rho_heating/rho_cp/adiabatic heating is on"))
    # `ev_field` supplies the elementwise viscosity of a previous assembly (the
    # consistent-boundary-flux residual must use the operator that produced the
    # current theta).  When it is given, the EV globals are not needed and the
    # field is used verbatim; otherwise the field is recomputed from `ev`.
    if ev > 0.0 && ev_field === nothing && (evg === nothing || theta === nothing)
        throw(ArgumentError("evg and theta are required when ev>0"))
    end
    if ev_field !== nothing
        length(ev_field) == length(sq.leaves) ||
            throw(ArgumentError("ev_field has $(length(ev_field)) entries, " *
                                "mesh has $(length(sq.leaves)) leaves"))
    end
    if ev_out !== nothing
        length(ev_out) == length(sq.leaves) ||
            throw(ArgumentError("ev_out has $(length(ev_out)) entries, " *
                                "mesh has $(length(sq.leaves)) leaves"))
    end
    if theta_lat !== nothing
        length(theta_lat) == sq.N ||
            throw(ArgumentError("theta_lat has $(length(theta_lat)) entries, " *
                                "mesh has $(sq.N) dofs"))
    end
    N = sq.N
    p = sq.p
    fd = make_fedata(p, p - 1)
    leaves = Set(sq.leaves)
    cfg = SemQuadEnergyCfg(pr, ux, uz, theta, theta_lat, rho_heating, rho_cp,
                           ev, ev_cmax, ev_alpha, ev_timestep, evg, theta_prev,
                           dt, ev_field, ev_out, heating_adiabatic,
                           heating_dissipation, eta_fun, rho_ref, phases,
                           heating_scale)
    asm = if sq.geom isa CartesianBox
        # Box path: the historical body, reached by a *static* call (no
        # dispatch frame, 0 B extra; GEOM_DESIGN §5.8, the cut-2 lesson).
        _semquad_assemble_energy_blocks_box(
            _semquad_asm_thread_count(assemble_threads), cfg, sq, fd, leaves)
    else
        _semquad_assemble_energy_blocks_mapped(
            _semquad_asm_thread_count(assemble_threads), cfg, sq.geom, sq, fd,
            leaves)
    end
    IM = asm.IM; JM = asm.JM; VM = asm.VM
    IK = asm.IK; JK = asm.JK; VK = asm.VK
    IC = asm.IC; JC = asm.JC; VC = asm.VC
    F = asm.F

    M = sparse(IM, JM, VM, N, N)
    K = sparse(IK, JK, VK, N, N)
    C = sparse(IC, JC, VC, N, N)
    if advection === :skew
        C = 0.5 * (C - C')
    elseif advection === :conservative
        C = -C'
    end
    return M, K, C, F
end

"""
    assemble_semquad_energy_latent_jvp(sq, qt, pr, phases, dtheta, theta, ux, uz;
                                       advection=:galerkin, rho_ref=nothing,
                                       rho_heating=false, rho_cp=false)

Directional derivative of the **latent-heat** part of
[`assemble_semquad_energy`](@ref) in the temperature direction `dtheta`, at the
temperature `theta` (the apparent heat capacity is
`1 + sum_i latent_i * Gamma_T,i` and the source is
`-sum_i latent_i * Gamma_z,i u_z`; `latent_i` is `pr.latent_L` in `:global`
mode or `PhaseTransition.latent` in `:per_phase` mode, docs/dev/PHYSICS.md §7.8):

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
