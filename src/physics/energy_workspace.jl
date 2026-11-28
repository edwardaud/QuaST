# Energy-assembly workspace: per-block scratch/buffer/cfg types and the
# deterministic buffer pool.  This file has no kernel; it is included before
# the box/mapped kernels and the public entry.

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

