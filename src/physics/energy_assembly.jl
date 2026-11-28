# Energy assembly: public entry point that selects the box or mapped block
# assembler with a static `isa` branch and builds the sparse operators.

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
capacity: `M` and `C` are weighted by
`(1 + sum_i latent_i * Gamma_T,i(y,T))` and the source gains
`-sum_i latent_i * Gamma_z,i * u_z` (`Gamma_i` is the same `phase_gamma` the
phase buoyancy uses).  The coefficient `latent_i` is `pr.latent_L` for every
transition in `latent_mode = :global` (the historical behaviour) and
`pt.latent` in `latent_mode = :per_phase`, where the
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
    # F5 guard: the temperature fields this assembly reads must be finite.  A
    # NaN in theta otherwise enters the assembled mass/stiffness/source through
    # `_semquad_eval_at_leaf!` and surfaces later as a NaN state or a bare
    # SingularException in the Stokes solve (AUDIT_VV F5).  The scans are
    # allocation-free; the throw path is cold.
    theta === nothing ||
        _require_finite!(theta, "the temperature field theta",
                         "the assemble_semquad_energy theta argument")
    theta_prev === nothing ||
        _require_finite!(theta_prev, "the previous temperature field theta_prev",
                         "the assemble_semquad_energy theta_prev argument")
    theta_lat === nothing ||
        _require_finite!(theta_lat, "the latent-evaluation temperature theta_lat",
                         "the assemble_semquad_energy theta_lat argument")
    ev_field === nothing ||
        _require_finite!(ev_field, "the entropy-viscosity field ev_field",
                         "the assemble_semquad_energy ev_field argument")
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
