# The validity band of the linearised rotation.
#
# `maxwell_stress_corotate` applies `1 + dt*ad_W`, the first-order rotation, not
# its exponential.  It is faithful while the per-step rotation
# `2|W|*min(dt, tau_M)` is small, and `MAXWELL_COROTATION_BOUND` quantifies
# "small": at most 0.1 rad (5.7 degrees) per resolved step, i.e. a leading
# linearisation error below 0.5% of the magnitude in the elastic regime and below
# 10% of spurious old-stress residue in the fast-relaxation regime.
# `maxwell_corotation_band` measures the three quantities the driver reports and
# warns about; it reads no state.

# --- the validity band of the linearised rotation (G254) --------------------
#
# `maxwell_stress_corotate` applies the rotation *linearised* over the step,
# `1 + dt ad_W`, not `exp(dt ad_W)`.  Its error is controlled by one of two
# dimensionless per-step rotations, depending on which timescale is shorter:
#
# * elastic regime, `dt < tau_M`: the step is the limiting time, so the
#   linearised rotation over-rotates the stress magnitude by
#   `sqrt(1 + (2 W dt)^2) ~ 1 + 2 (W dt)^2`, i.e. a relative error
#   `~ (2 W dt)^2 / 2` per step;
# * fast-relaxation regime, `tau_M < dt`: the old stress has relaxed by
#   `exp(-dt/tau_M)` before the step ends, so the spin only acts for `tau_M`.
#   The update instead keeps `~sqrt(1 + (2 W dt)^2)/(1 + dt/tau_M) ~ 2 |W| tau_M`
#   of the old stress (the exact solution keeps none), i.e. the per-step error
#   is `O(2 tau_M |W|)` of the steady stress.
#
# Both are `2 |W| min(dt, tau_M)`: over one step the update can resolve no finer
# time than `dt`, and the stress remembers the spin no longer than `tau_M`.  The
# documented `2 |W| dt << 1` / `2 tau_M |W| << 1` is quantified by the single
# constant below, which `run_semquad_simulation` reports every run and warns on
# when exceeded (G254).  It is a *validity/accuracy* bound, not a stability one:
# beyond it the update is still first order in `dt` and its fixed point is still
# exact, but the rotation it applies is no longer faithful.

"""
    MAXWELL_COROTATION_BOUND

The bound on the per-step rotation `2 |W| min(dt, tau_M)` at which the
linearised co-rotational term `1 + dt ad_W` is considered valid: `0.1`.

Choosing `0.1` makes `<< 1` concrete as one decade below `O(1)`: at the bound a
step rotates the stress by at most `0.1 rad` (5.7 degrees), and the leading
linearisation error is at most `0.1^2 / 2 = 0.5%` of the stress magnitude per
step (elastic regime) or at most `10%` of the steady stress retained spuriously
(fast-relaxation regime).  It is deliberately conservative rather than tuned to
keep the gates quiet: most verified cards sit far inside it (the analytic
rotation gate operates at `W dt = 3.9e-5`), but the stress-state card of G246
reaches `2|W|dt = 0.103` and G251's forced-varying-step composition card 1.2,
and the guard reports both in the suite log rather than the bound being moved
to hide them.  A card at `W dt ~ 1` -- the regime the PHASE2 record warns would
need the exact exponential -- is reported instead of being silently wrong.

`run_semquad_simulation` compares its run-maximum of
[`maxwell_corotation_band`](@ref)'s `resolved_rotation` against this value
(G254).  It reports the maxima once per run and warns once, at the first step
that exceeds the bound; it does not refuse, because the scheme remains
consistent outside the band (only the rotation accuracy degrades), so aborting
a long run would trade a quantified accuracy loss for a lost run.
"""
const MAXWELL_COROTATION_BOUND = 0.1

"""
    maxwell_corotation_band(wxy, dt, tau_M)

The three dimensionless rotations of the linearised co-rotational term at one
node (`wxy` is the spin `W_xy`, `tau_M = eta/G` the material Maxwell time), as a
named tuple:

    two_w_dt        = 2 |wxy| dt          explicit rotation over the step
    two_taum_w      = 2 tau_M |wxy|       rotation over the Maxwell time
    resolved_rotation = 2 |wxy| min(dt, tau_M)

`resolved_rotation` is the angle the update can actually resolve and is the
quantity compared against [`MAXWELL_COROTATION_BOUND`](@ref) (see the block
comment above).  It is `min(two_w_dt, two_taum_w)`; `wxy = 0` gives three zeros.
"""
@inline function maxwell_corotation_band(wxy::Float64, dt::Float64,
                                         tau_M::Float64)
    two_w_dt = 2.0 * abs(wxy) * dt
    two_taum_w = 2.0 * tau_M * abs(wxy)
    return (two_w_dt=two_w_dt, two_taum_w=two_taum_w,
            resolved_rotation=min(two_w_dt, two_taum_w))
end
