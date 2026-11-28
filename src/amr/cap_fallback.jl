# ---------------------------------------------------------------------------
# Opt-in max-level Dörfler fallback
# ---------------------------------------------------------------------------
#
# This is a pure marking transform: it never mutates the tree and never
# replaces the standard marking unless the standard candidate set is empty
# because the whole marked prefix sits at `max_level`.  It was extracted
# verbatim from `amr_cycle_balanced!` so the cycle policy and the fallback
# policy are separate files; every arithmetic expression and comparison order
# is intentionally unchanged (G295/G296 pin the contract).

"""
    _check_cap_aware(cap_aware, coarsen_gate)

Refuse the unsupported combination `cap_aware=true, coarsen_gate=false`: the
legacy gate-off coarsening branch has no hook for protecting the fallback
marks.  Kept as a separate guard so the cycle reads as policy, not validation.
"""
function _check_cap_aware(cap_aware::Bool, coarsen_gate::Bool)
    cap_aware && !coarsen_gate &&
        throw(ArgumentError("cap_aware requires coarsen_gate=true: the " *
                            "legacy gate-off coarsening branch has no " *
                            "mark-protection hook for the fallback marks"))
    return nothing
end

"""
    _cap_aware_marking(cand, want, mark_target, tau_r, tau_h, deadband,
                       cap_applied, refinable, refinable_mass,
                       cap_prefix, cap_mass, desc, eof, total,
                       refine_frac, tau_c, hysteresis, max_level, qt_maxlevel,
                       cap_aware, cool, per) -> NamedTuple

Apply the count-preserving max-level fallback to the standard Dörfler marking.
The input `desc` is the descending `(eta, leaf)` list, `eof` is the
non-negative indicator accessor and `cool` the optional cooldown.  The function
returns every effective marking value; when the trigger does not fire it
returns its inputs unchanged.  See [`amr_cycle_balanced!`](@ref) for the
policy description and G295/G296 for the pinned contract.
"""
function _cap_aware_marking(cand::Vector{NTuple{3,Int}}, want::Int,
                            mark_target::Float64, tau_r::Float64,
                            tau_h::Float64, deadband::Bool,
                            cap_applied::Bool, refinable::Int,
                            refinable_mass::Float64, cap_prefix::Int,
                            cap_mass::Float64,
                            desc::Vector{NTuple{3,Int}}, eof::F,
                            total::Float64, refine_frac::Float64,
                            tau_c::Float64, hysteresis::Float64,
                            max_level::Int, qt_maxlevel::Int,
                            cap_aware::Bool,
                            cool::Union{Nothing,AmrCooldown},
                            per::Int) where {F}
    # The standard marking above is authoritative whenever it finds any
    # candidate.  Only a genuinely cap-caused empty set (the whole marked
    # prefix is at the level cap, not merely in cooldown) triggers the
    # fallback.  It replaces the capped prefix one-for-one: take the same
    # number of leaves from the admissible (refinable, non-cooled) universe,
    # highest eta first.  Keeping the cardinality of the standard marked set
    # bounds the batch and avoids the mass-renormalised variant's flood at a
    # sustained cap (measured on the theta/0.2 ring: mass fallback went from
    # 24.5k to 30k leaves in ~35 steps, count fallback is per*cap_prefix leaves
    # per cycle).  The effective tau_r (lowest marked eta) and target (sum of
    # the marked eta) then drive the gate, budget exchange and refine loop.
    if cap_aware && refine_frac > 0.0 && isempty(cand) &&
       max_level <= qt_maxlevel
        prefix_refinable = 0
        cum_pref = 0.0
        for lf in desc
            eof(lf) >= tau_r || break
            if lf[1] >= max_level
                cap_prefix += 1
                cap_mass += eof(lf)
            else
                prefix_refinable += 1
            end
            cum_pref += eof(lf)
            cum_pref >= refine_frac * total && break
        end
        if cap_prefix > 0 && prefix_refinable == 0
            for lf in desc
                lf[1] < max_level || continue
                (cool !== nothing && haskey(cool.no_refine, lf)) && continue
                refinable += 1
                refinable_mass += eof(lf)
            end
            if refinable > 0
                kfb = min(cap_prefix, refinable)
                cand = NTuple{3,Int}[]
                want = 0
                for lf in desc
                    lf[1] < max_level || continue
                    (cool !== nothing && haskey(cool.no_refine, lf)) && continue
                    push!(cand, lf)
                    want += per
                    length(cand) >= kfb && break
                end
                if !isempty(cand)
                    mark_target = 0.0
                    for lf in cand
                        mark_target += eof(lf)
                    end
                    if mark_target > 0.0
                        tau_r = eof(cand[end])    # lowest marked leaf
                        tau_h = min(tau_c, hysteresis * tau_r)
                        deadband = hysteresis * tau_r < tau_c
                        cap_applied = true
                    else
                        empty!(cand)
                        want = 0
                        mark_target = refine_frac * total
                    end
                end
            end
        end
    end


    return (cand=cand, want=want, mark_target=mark_target, tau_r=tau_r,
            tau_h=tau_h, deadband=deadband, cap_applied=cap_applied,
            refinable=refinable, refinable_mass=refinable_mass,
            cap_prefix=cap_prefix, cap_mass=cap_mass)
end
