# Hysteresis state of the balanced AMR cycle: `AmrCooldown`.
"""
    AmrCooldown()

Hysteresis state of `amr_cycle_balanced!`: two cooldown dicts, `no_refine`
(leaves produced by a merge; not refined again for a few cycles) and
`no_coarsen` (leaves produced by a split; not merged again for a few cycles),
plus `cycle`, the number of AMR cycles this state has been aged through.

`cycle` is *not* read by the policy: it is an identity/diagnostic counter that a
checkpoint can round-trip, so a restored cooldown can be shown to be the same
object (the decisions themselves depend only on the dicts).  A key is removed
when its counter reaches zero or when the leaf stops being a leaf, so the state
never refers to a stale mesh.  The state is owned by the caller: pass the same
object to every cycle of a run.  `run_semquad_simulation` creates one for its run,
returns it in the result (`cooldown`) and writes it through `save_semquad_state`,
which persists it as optional HDF5 state (docs/dev/BUG_LEDGER.md §10.5); a
checkpoint without it loads an empty cooldown, which is the historical restart
behaviour.
"""
mutable struct AmrCooldown
    no_refine::Dict{NTuple{3,Int},Int}
    no_coarsen::Dict{NTuple{3,Int},Int}
    cycle::Int
end
AmrCooldown() =
    AmrCooldown(Dict{NTuple{3,Int},Int}(), Dict{NTuple{3,Int},Int}(), 0)
# Backwards-compatible two-dict constructor (tests and probes built the object
# before `cycle` existed); the counter starts at zero.
AmrCooldown(no_refine::Dict{NTuple{3,Int},Int},
            no_coarsen::Dict{NTuple{3,Int},Int}) =
    AmrCooldown(no_refine, no_coarsen, 0)

"""Age the two cooldown dicts by one cycle and drop dead/expired keys.

A value of `n` means "suppressed for the next `n` cycles": the decrement happens
at the *start* of a cycle and a key is dropped when it goes negative, so an
entry stored with `hysteresis_cycles=1` in cycle `k` still suppresses the
operation in cycle `k+1` and expires at the start of cycle `k+2`.

The age is counted in `c.cycle` (incremented here, once per cycle), so a
checkpoint taken after cycle `k` restores a state whose next cycle is `k+1`.
"""
function _age_cooldown!(c::AmrCooldown, leaves::Set{NTuple{3,Int}})
    c.cycle += 1
    for d in (c.no_refine, c.no_coarsen)
        for k in collect(keys(d))
            if !(k in leaves)
                delete!(d, k)                 # the leaf is gone
            else
                d[k] -= 1
                d[k] < 0 && delete!(d, k)     # the cooldown expired
            end
        end
    end
    return c
end

