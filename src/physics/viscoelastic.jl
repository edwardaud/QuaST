# Maxwell stress history, step 1: the update kernel.
#
# This file holds the physics of the stress update and nothing else.  The state
# vector, the AMR transfer and the checkpoint live in the driver, and the elastic
# force term is bridged by `maxwell_force_stress`; keeping the update kernels here
# means they can be gated against the analytic curve independently of that
# machinery, so a failure here cannot be confused with a transport or wiring bug.
# (The composition of the kernels into the per-step update is
# `_advance_maxwell_stress!` in `src/driver/run.jl`, docs/dev/PHASE2.md 2b.)
#
# The update is *objective*: a Maxwell stress that does not rotate with the
# material is not frame-indifferent, and every mantle-convection run is a
# large-strain run.  `maxwell_stress_corotate` supplies the co-rotational
# (Jaumann) part and the driver composes it with the scalar backward-Euler
# kernel; the analytic reference for the composed update is
# `maxwell_stress_exact_corotational`.

"""
    maxwell_stress_step(tau_old, edot, dt, eta, G)

One backward-Euler step of the Maxwell relaxation

    dtau/dt = 2 G edot - tau / tau_M,        tau_M = eta / G,

from `tau_old` with the strain rate `edot` held over the step:

    tau_new = (tau_old + 2 G edot dt) / (1 + dt / tau_M).

`G <= 0` returns `tau_old` unchanged, matching the `shear_modulus = 0` convention of
`Params`: the viscoelastic terms are off, not elastic with zero modulus.

The steady state of the iteration is `2 eta edot`, the purely viscous stress, and the
first step from rest in the limit `dt << tau_M` is `2 G edot dt`, the elastic one. Both
are gated by G244.
"""
function maxwell_stress_step(tau_old::Float64, edot::Float64, dt::Float64,
                             eta::Float64, G::Float64)
    G <= 0.0 && return tau_old
    tau_M = eta / G
    return (tau_old + 2.0 * G * edot * dt) / (1.0 + dt / tau_M)
end

"""
    maxwell_stress_corotate(txx, tyy, txy, wxy, dt)

The **objective** (co-rotational, Jaumann) part of the Maxwell stress update:
the in-plane components of `tau + dt*(W tau - tau W)` for the deviatoric stress
`tau = [txx txy; txy tyy]` and the spin

    W = [0 wxy; -wxy 0],     wxy = (du_x/dy - du_z/dx) / 2,

with `(x, y)` the in-plane axes (`u_z` is the second in-plane velocity
component, the code's `uz`).  `wxy` is the off-diagonal component of the spin,
the antisymmetric part of the velocity gradient.

The in-plane component form follows row by row (`W_yx = -W_xy`):

    (W tau - tau W)_xx = 2 wxy txy
    (W tau - tau W)_yy = -2 wxy txy
    (W tau - tau W)_xy = wxy (tyy - txx)

Row `xx`: `(W tau)_xx = W_xy tau_yx = wxy txy` and
`(tau W)_xx = tau_xy W_yx = -wxy txy`, whose difference is twice the first.
Row `yy` is the negative of row `xx` because `W_yx = -W_xy`.  Row `xy`:
`(W tau)_xy = W_xy tau_yy = wxy tyy` and `(tau W)_xy = tau_xx W_xy = wxy txx`,
so their difference is `wxy (tyy - txx)`.  That last sign is easy to get
backwards -- the naive `wxy (txx - tyy)` rotates the material the wrong way;
`test/t_g250_corotation.jl` pins these three lines against an explicit matrix
product `W*T - T*W` before it runs a rotation.

The out-of-plane normal component is untouched, `(W tau - tau W)_zz = 0`, and
the in-plane trace is preserved (the two normal components are exact
negatives), so a deviatoric stress stays deviatoric: no projection is needed
on this term.  The spin is a pure rotation generator about the out-of-plane
axis, so the stored triple is closed under it.

This is the rotation *linearised* over the step, `1 + dt*ad_W`, not the exact
exponential of a finite rotation.  That is deliberate: it is the same
first-order linearisation as the backward-Euler relaxation it is composed with,
and it is the lagged numerator of the update in ASPECT's `viscoelastic` model,
`tau^{n+1} = (tau^n + 2 G dt ehat + dt (W tau^n - tau^n W)) / (1 + dt/tau_M)`.
The driver composes it as

    tau^{n+1} = maxwell_stress_step(maxwell_stress_corotate(tau^n, wxy, dt),
                                    ehat^n, dt, eta, G),

`dt = 0` or `wxy = 0` returns the input components exactly, which is why the
term is a no-op for a rotation-free flow and costs a handful of flops per node.
"""
function maxwell_stress_corotate(txx::Float64, tyy::Float64, txy::Float64,
                                 wxy::Float64, dt::Float64)
    return (txx + 2.0 * dt * wxy * txy,
            tyy - 2.0 * dt * wxy * txy,
            txy + dt * wxy * (tyy - txx))
end

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

# --- the viscosity regime of a viscoelastic run (G255) ----------------------
#
# The 2a closure replaces the material viscosity by
#
#     eta_eff = 1/(1/eta + 1/(G dt)),
#
# which *caps* eta from above at the elastic viscosity `G dt`.  That is correct
# physics, but it means a viscoelastic run does not solve "the same problem plus
# memory": every viscosity well above `G dt` is flattened to `G dt`, so a
# material contrast carried by high-viscosity material (a cold lid, say) can be
# removed from the problem the Stokes solve sees by several decades, and the
# solve becomes *softer*.  Measured on the developed production state: material
# contrast 3.3e6 against an effective ~5e2 at `G dt = 1.485` (docs/dev/PHASE2.md
# step 2 of the case).  `maxwell_regime` measures both ranges on one viscosity
# field and at one step so the driver can report the difference rather than
# leave it implicit.

"""
    maxwell_regime(eta, dt, G)

The material and effective viscosity ranges of one step, and the ratio
`G*dt/eta`, as a named tuple:

    material_min, material_max, material_contrast
    effective_min, effective_max, effective_contrast
    gdt_over_eta_min, gdt_over_eta_max

`eta` is the **material** (unsoftened) nodal viscosity field the Maxwell stress
update uses, `dt` the step actually taken, and `G = Params.shear_modulus`; the
effective viscosity is the 2a closure `1/(1/eta + 1/(G dt))` per node.  Each
contrast is `max/min` over the field.

`eta_eff` is the harmonic mean of `eta` and `G*dt`, so it never exceeds either:
the effective contrast never exceeds the material one, and it is smaller
whenever the field straddles the cap `G*dt`.  The run-time report that uses this
helper states the resulting collapse in decades (G255).

The function is read-only (it does not modify `eta`) and pure.  `G <= 0` and
`dt <= 0` are rejected: with the viscoelastic terms off there is no
effective-viscosity regime to report (the `shear_modulus = 0` convention), and
the driver only calls it for `G > 0`.
"""
function maxwell_regime(eta::Vector{Float64}, dt::Float64, G::Float64)
    G > 0.0 || throw(ArgumentError("maxwell_regime requires G > 0; with " *
                                   "shear_modulus = 0 there is no " *
                                   "effective-viscosity regime to report"))
    dt > 0.0 || throw(ArgumentError("maxwell_regime requires dt > 0"))
    isempty(eta) &&
        throw(ArgumentError("maxwell_regime requires a non-empty viscosity " *
                            "field"))
    Gdt = G * dt
    material_min = Inf
    material_max = -Inf
    effective_min = Inf
    effective_max = -Inf
    ratio_min = Inf
    ratio_max = -Inf
    @inbounds for i in eachindex(eta)
        e = eta[i]
        # The viscosity laws and their bounds are positive by construction; a
        # non-positive value here would be a closure bug, and the arithmetic
        # below would report it as an infinite contrast rather than hide it.
        eff = 1.0 / (1.0 / e + 1.0 / Gdt)
        material_min = min(material_min, e)
        material_max = max(material_max, e)
        effective_min = min(effective_min, eff)
        effective_max = max(effective_max, eff)
        ratio = Gdt / e
        ratio_min = min(ratio_min, ratio)
        ratio_max = max(ratio_max, ratio)
    end
    return (material_min=material_min, material_max=material_max,
            material_contrast=material_max / material_min,
            effective_min=effective_min, effective_max=effective_max,
            effective_contrast=effective_max / effective_min,
            gdt_over_eta_min=ratio_min, gdt_over_eta_max=ratio_max)
end

"""
    maxwell_stress_exact(t, edot, eta, G)

The continuous solution of the same equation from rest,

    tau(t) = 2 eta edot (1 - exp(-t / tau_M)),

used by the gate as the reference the discrete update converges to as `dt` shrinks.
"""
function maxwell_stress_exact(t::Float64, edot::Float64, eta::Float64, G::Float64)
    G <= 0.0 && return 0.0
    tau_M = eta / G
    return 2.0 * eta * edot * (1.0 - exp(-t / tau_M))
end

"""
    maxwell_stress_exact_corotational(t, edot_xx, edot_zz, edot_xz, wxy, eta, G)

The continuous solution from rest of the **objective** Maxwell equation

    dtau/dt = 2 G ehat + (W tau - tau W) - tau / tau_M,     tau_M = eta / G,

with a constant deviatoric strain rate `ehat` (the three in-plane components
`edot_xx = ehat_xx`, `edot_zz = ehat_zz`, `edot_xz = ehat_xz`; the out-of-plane
component is `-tr/3`, implied) and a constant spin `wxy = W_xy`.  It is the
reference for `maxwell_stress_exact` when the spin is on, the curve the discrete
update converges to as `dt` shrinks.

Decompose the in-plane tensor into

    m = (tau_xx + tau_yy)/2,   a = (tau_xx - tau_yy)/2,   b = tau_xy.

The spin term acts on `(a, b)` as a rotation at rate `-2 wxy` (`m` is invariant,
and so is the implied `tau_zz = -2m`), so with `E_m = (ehat_xx + ehat_zz)/2`,
`E_a = (ehat_xx - ehat_zz)/2`, `E_b = ehat_xz` the three scalar equations are

    dm/dt = 2 G E_m - m/tau_M
    d/dt [a; b] = 2 G [E_a; E_b] + (A2 - 1/tau_M) [a; b],
        A2 = [0 2 wxy; -2 wxy 0],

whose solution from `tau = 0` is

    m(t) = 2 eta E_m (1 - exp(-t/tau_M))
    [a; b](t) = M^{-1} (exp(M t) - I) 2 G [E_a; E_b],
        M = A2 - 1/tau_M,   exp(M t) = exp(-t/tau_M) [cos 2wxy t  sin 2wxy t;
                                                       -sin 2wxy t  cos 2wxy t],

and `tau_xx = m + a`, `tau_yy = m - a`, `tau_xy = b`.  The steady state is
`tau = 2 eta ehat + tau_M (W tau - tau W)`, i.e. the co-rotational (Jaumann)
steady stress: the rotation term does *not* vanish in a steady flow, which is
why the equilibrated `2 eta ehat` fixed point of the spin-free update moves by
`O(tau_M |wxy|)` once the term is on.

`G <= 0` returns `(0, 0, 0)`, matching `maxwell_stress_exact`: with the
viscoelastic terms off there is no stress history.
"""
function maxwell_stress_exact_corotational(t::Float64, edot_xx::Float64,
                                           edot_zz::Float64, edot_xz::Float64,
                                           wxy::Float64, eta::Float64,
                                           G::Float64)
    G <= 0.0 && return (0.0, 0.0, 0.0)
    tau_M = eta / G
    decay = exp(-t / tau_M)
    # the in-plane trace of ehat drives m; the third (out-of-plane) component is
    # implied by trace zero and never entered the two normal components directly.
    E_m = 0.5 * (edot_xx + edot_zz)
    E_a = 0.5 * (edot_xx - edot_zz)
    E_b = edot_xz
    m = 2.0 * eta * E_m * (1.0 - decay)
    # exp(M t) for M = A2 - 1/tau_M
    th = 2.0 * wxy * t
    E1 = decay * [cos(th) sin(th); -sin(th) cos(th)]
    # M^{-1} = 1/(1/tau_M^2 + 4 wxy^2) [-1/tau_M  -2 wxy;  2 wxy  -1/tau_M]
    det = 1.0 / tau_M^2 + 4.0 * wxy^2
    Mi = (1.0 / det) * [-1.0 / tau_M -2.0 * wxy; 2.0 * wxy -1.0 / tau_M]
    v = Mi * (E1 - [1.0 0.0; 0.0 1.0]) * (2.0 * G * [E_a, E_b])
    return (m + v[1], m - v[1], v[2])
end

"""
    maxwell_force_stress(txx, tyy, txy, eta, dt, G)

The previous step's deviatoric stress as the **momentum equation** sees it
(2b step 2), i.e. the elastic force term `div(tau)` of ASPECT's viscoelastic
formulation.

One backward-Euler step of the Maxwell update splits the new stress into

    tau^{n+1} = 2 eta_eff ehat^{n+1} + a tau^n,     a = eta/(eta + G dt),
    eta_eff   = 1/(1/eta + 1/(G dt)),

where `ehat` is the **deviatoric** strain rate (G249); the load assembled here
leaves out the co-rotational increment of the old stress, which the paragraph
further down states and justifies.  The first term is already carried by the
Stokes operator through the effective viscosity.  The second is an explicit
load: momentum gains `div(a tau^n)`, whose weak form against the velocity test
functions is `-int a tau^n : eps(v)`.  This function returns the three nodal
components of `a tau^n` (`eta` is the *material*, nodal viscosity; `G dt` is
the same product the 2a closure uses), which is exactly the tensor field the
Stokes assembly turns into that load.

The two states the scheme needs follow from `a`:

    tau = 0          -> no elastic force,
    tau = 2 eta ehat ->  eta_eff + a eta = eta,

so an equilibrated viscoelastic flow is the purely viscous one (material `eta`),
not the `eta_eff` one; the two agree as `G dt >> eta` (the coefficient is
`tau_M/dt`).  G247 pins that limit.

With the co-rotational term on, the exact split of the update is

    tau^{n+1} = 2 eta_eff ehat^{n+1} + a R(tau^n),
        R(tau) = tau + dt (W tau - tau W),     a = 1/(1 + dt/tau_M),

but the load assembled here is `a tau^n`, without the rotation increment.  That
is a deliberate scope choice: the split, its `a`, its fixed-point identity and
the gated force term stay exactly as 2b step 3 left them, the load is bit-identical
for a rotation-free flow, and the omitted piece `a dt (W tau - tau W)` is the
lagged part of the same first-order scheme.  The *stress state* -- what the
diagnostics report and what the next step's update reads -- does carry the
rotation in full (`_advance_maxwell_stress!`); only this one-step lagged load is
left out.  In the equilibrated limit the missing load is `O(tau_M |W|)` of the
momentum stress while its effect on the velocity fixed point is
`O((tau_M/dt) tau_M |W|)`, so for the fast-relaxation regime the gates run in
(`dt >> tau_M`) the material-viscosity fixed point of G247 is preserved, and for
`W = 0` nothing changes at all (G250).

`G <= 0` is rejected: with `shear_modulus = 0` the term must not be assembled at
all (the caller's convention), and the formula above would silently return
`tau` itself.
"""
function maxwell_force_stress(txx::Vector{Float64}, tyy::Vector{Float64},
                              txy::Vector{Float64}, eta::Vector{Float64},
                              dt::Float64, G::Float64)
    G > 0.0 || throw(ArgumentError("maxwell_force_stress requires G > 0; with " *
                                   "shear_modulus = 0 the elastic force term " *
                                   "must not be assembled"))
    dt > 0.0 || throw(ArgumentError("maxwell_force_stress requires dt > 0"))
    n = length(eta)
    (length(txx) == n && length(tyy) == n && length(txy) == n) ||
        throw(ArgumentError("the Maxwell stress fields and eta must have the " *
                            "same length; got ($(length(txx)), " *
                            "$(length(tyy)), $(length(txy))) against $n"))
    sx = Vector{Float64}(undef, n)
    sy = Vector{Float64}(undef, n)
    sz = Vector{Float64}(undef, n)
    Gdt = G * dt
    @inbounds for i in 1:n
        a = eta[i] / (eta[i] + Gdt)
        sx[i] = a * txx[i]
        sy[i] = a * tyy[i]
        sz[i] = a * txy[i]
    end
    return (sx, sy, sz)
end

# 2a effective-viscosity softening for the per-step Newton viscosity factory
# (`newton_make_eta` in driver/run.jl).  The Newton residual assembles
# `2 eta_eff eps(u)` with the softened viscosity, so every derivative closure it
# hands to the analytic tangent has to be softened with the chain rule
#
#     d eta_eff / dx = (eta_eff/eta)^2 d eta / dx.
#
# A `nothing` derivative entry (e.g. the strain-rate derivative of a
# temperature-only law) stays `nothing`: the tangent code uses `=== nothing` to
# decide whether a block exists at all.  G = 0 returns `mk` unchanged.
function _ve_soften_eta_factory(mk::Tuple, G::Float64, dt::Float64)
    G <= 0.0 && return mk
    eta_raw = mk[1]
    eta_eff = (x, y) -> begin
        e = eta_raw(x, y)
        1.0 / (1.0 / e + 1.0 / (G * dt))
    end
    r2 = (x, y) -> begin
        e = eta_raw(x, y)
        ee = eta_eff(x, y)
        (ee / e)^2
    end
    dT = mk[2] === nothing ? nothing : ((x, y) -> r2(x, y) * mk[2](x, y))
    de = length(mk) >= 3 && mk[3] !== nothing ?
         ((x, y, ed) -> r2(x, y) * mk[3](x, y, ed)) : nothing
    if length(mk) >= 5
        dp = mk[4] === nothing ? nothing :
             ((x, y, ed, pq) -> r2(x, y) * mk[4](x, y, ed, pq))
        dchi = mk[5] === nothing ? nothing :
               ((x, y, ed) -> r2(x, y) * mk[5](x, y, ed))
        return (eta_eff, dT, de, dp, dchi)
    end
    return (eta_eff, dT, de)
end
