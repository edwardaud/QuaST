# Public parameter card: the `Params` type and its field-level documentation.
# Defaults and validation are in params_validation.jl; the base-state functions
# are in params_base.jl.
# Physical parameters, base state, viscosity law and diagnostics.

"""
    Params(; Ra, H, eta_law, deta_T, eta_b, eta_c, ...)

Viscosity laws and their parameter semantics (**the two families do not share a
meaning**; mixing them up silently changes the viscosity by orders of magnitude):

- `eta_law = :expT` -- `eta = exp(-log(deta_T) * T)`, i.e. `deta_T` is a
  **viscosity ratio**: `eta(T=0) = 1` and `eta(T=1) = 1/deta_T`.  `deta_T = 1`
  is constant viscosity, `deta_T = 1e4` makes the hot end 1e4 times weaker.
  (`deta_T = 1e-8` therefore means the *hot* end is 1e8 times **stiffer**, which
  is the opposite of the usual mantle law.)
- `eta_law = :expTz` -- `eta = exp(-eta_b * T + eta_c * (1 - z))`, where `eta_b`
  and `eta_c` are **coefficients** (ASPECT's "Viscosity temperature/depth
  prefactor"); `eta_b = log(1e4)` gives a factor 1e4 across the box.
- `eta_law = :arrhenius` -- uses `eta_E` and `eta_T0`; `eta_law = :const` (and
  any other value) returns `eta_const`.
- `eta_law = :composite` -- diffusion + dislocation creep (ASPECT's
  `diffusion dislocation` material model), with two combination rules selected
  by `comp_rule`:

      eta_diff = comp_Ad^-1 exp((comp_Ed + comp_Vd*(1-z)) / T_nd)
      eta_disl = comp_As^-1 exp((comp_Es + comp_Vs*(1-z)) / (comp_n*T_nd))
                 * eps_II^((1-comp_n)/comp_n)

      comp_rule = :isostress (default, ASPECT's law):
          both mechanisms carry the same stress, the *partial* strain rates
          add:  tau/(2 eta_diff) + A_s exp(-(E_s+V_s(1-z))/(R T)) tau^n = eps_II
          solved for tau at every quadrature point; eta_eff = tau/(2 eps_II).
      comp_rule = :isostrain (legacy, pre-2026-11-08 results):
          eta_eff = 1 / (1/eta_diff + 1/eta_disl), both mechanisms evaluated at
          the *total* strain rate (the harmonic mean).

  with `T_nd = max(T + eta_T0, comp_Tcut)` (`comp_Tcut = 0` = off), `z` the
  height (0 bottom, 1 top, so `1-z` is depth) and `eps_II` the second invariant
  of the strain rate with the code's `sqrt(0.5 edot:edot)` convention.  The two
  rules agree exactly in the one-mechanism limits and differ by up to 41%
  (n = 3.5) where both mechanisms contribute; ASPECT 2.5.0-pre solves
  `:isostress`. The `eta_min/eta_max`
  bounds of `_clip_eta` apply to `eta_eff` (clip once, after the combination),
  and `comp_eps_min` floors the strain rate of the dislocation branch (so that
  `eps_II -> 0` is the diffusion limit with a finite viscosity instead of
  `eta_disl = Inf` and an infinite derivative).  `comp_n` is ASPECT's "Stress
  exponent for dislocation creep" (n = 3 in the mantle); `comp_Ad`/`comp_As`
  are ASPECT's prefactors in units of `1/eta0` of the code's
  nondimensionalisation, including the factor 2 that ASPECT's
  `eta = stress/edot/2` convention puts into the prefactors).

`eta_ref` (default `1.0`) is the **Rayleigh-number reference viscosity**, in the
units the material law returns.  It states the convention the nondimensional
equations have always used implicitly:

    Ra = rho g alpha dT d^3 / (kappa eta_ref),

and the time-stepping driver solves the momentum equation with
`eta_law/eta_ref` wherever the law's viscosity enters -- through the weak-zone
multiplier and the Maxwell effective-viscosity closure of the same wrapper chain.
The default 1.0 is the historical convention and leaves every existing run
bit-identical.  The point of the parameter is that a *rescaled law must be paired
with the same rescaling of `eta_ref`*: multiplying both `comp_Ad` and `comp_As`
by `1/c` multiplies every viscosity the law returns by `c` (both creep branches
and the isostress solution scale together), so `eta_law/eta_ref` -- and the whole
flow -- is unchanged only when `eta_ref` is also multiplied by `c`.  Rescaling
the law and leaving `eta_ref = 1` keeps `Ra` at its old value while moving every
physical viscosity by `c`: the same `Ra` then describes a different physical
Rayleigh number.  The `eta_min`/`eta_max` bounds are absolute bounds in the
law's units applied *before* this normalisation, so a pure law rescaling must
rescale them too.

`run_semquad_simulation` reports the convention (Ra, eta_ref, the normalised
field range, the interior Rayleigh number, and -- when the card supplies its
scale set -- the physical reference viscosity in Pa s) once at the first flow
solve and once at the end of the run, whenever `eta_ref != 1`.  The default
1.0 logs nothing.  The normalisation belongs to the Picard time-stepping path
only: `pic_mode = :newton` builds its viscosity from the
raw law and therefore **refuse** `eta_ref != 1` rather than solve a different
problem silently.
"""
struct Params
    Ra::Float64
    H::Float64
    # Half-life of the radiogenic heating, in the code's time unit.  `Inf` keeps
    # `H` constant, the historical path bit for bit.  The source decays as
    # `H * 2^(-t/half_life)`, the form ASPECT's `Radioactive decay` subsection
    # uses with per-isotope half-lives.  The conductive base state `Tbar` keeps
    # the initial `H`, so `T = Tbar + theta` stays exact and only the source
    # decays.
    H_half_life::Float64
    eta_law::Symbol       # :const, :expT, :expTz or :arrhenius
    deta_T::Float64
    eta_b::Float64        # :expTz temperature prefactor b (ASPECT "Viscosity temperature prefactor")
    eta_c::Float64        # :expTz depth prefactor c (ASPECT "Viscosity depth prefactor")
    eta_E::Float64        # :arrhenius nondimensional activation energy E/(R dT)
    eta_T0::Float64       # :arrhenius nondimensional surface temperature offset
    yield_c::Float64      # Drucker-Prager cohesion (nondimensional stress)
    yield_phi::Float64    # friction angle (rad); 0 = pressure independent
    yield_taumax::Float64 # cap on the yield stress (nondimensional)
    yield_eps_min::Float64# strain-rate floor in eta_yield = tau_y/(2 eps)
    yield_damper::Float64 # ASPECT plastic damper viscosity eta_d (0 = off)
    eta_const::Float64
    eta_min::Float64
    eta_max::Float64
    bc::Symbol            # :free or :rigid
    supg::Float64
    eta_soft::Float64     # smooth-floor width; 0 = hard clip
    div_penalty::Float64  # gamma * int (div u)(div v); 0 = off
    eta_custom::Union{Nothing,Function}   # eta(x,z), overrides the T law in Stokes
    # Multiplicative factor on the viscosity *after* the law: `eta_new =
    # eta * eta_multiplier(x, z)`.  `nothing` leaves every law untouched, bit for
    # bit.  This is the weak-zone hook: a line or rectangle of reduced viscosity
    # becomes `eta_multiplier = weak_zone_factor(...)`, without reimplementing the
    # law, which is how ASPECT's `weak zone` plugin is used next to a material
    # model rather than instead of it.
    eta_multiplier::Union{Nothing,Function}
    # Multiplicative factor on the *density* wherever the buoyancy and the mass
    # terms use it: `rho_new = rho * rho_multiplier(x, z)` (same signature as
    # `eta_multiplier`).  `nothing` leaves every density path untouched, bit for
    # bit.  This is the density half of the sticky-air hook: a light layer above
    # `z_a` is `rho_multiplier = (x, z) -> z > z_a ? eps : 1.0` combined with a
    # low-viscosity `eta_multiplier`, without reimplementing a material model
    # (ASPECT's sticky-air recipes). It scales a density
    # that is *already* in a term -- it does not add a compositional density
    # anomaly, so with a Boussinesq/refstate buoyancy `Ra rho(y) alpha(y) theta`
    # it acts on the thermal anomaly only (a temperature-neutral layer carries
    # no buoyancy force); the phase-transition buoyancy
    # (`semquad_phase_buoyancy`), which is its own density anomaly anchored to
    # `rho0`, is not scaled.
    rho_multiplier::Union{Nothing,Function}
    # Maxwell viscoelasticity (ASPECT's `viscoelastic`): the shear modulus `G`
    # scales the elastic response in both the
    # effective-viscosity closure eta_eff = 1/(1/eta + 1/(G*dt)) and the stress
    # history tau^{n+1} = (corotate(tau^n) + 2 G dt ehat)/(1 + dt/tau_M),
    # tau_M = eta/G.  Both use the driver's **actual** step `dt`, which is
    # adaptive (CFL + growth limits); that is consistent, because backward Euler
    # with the actual step converges to the analytic Maxwell curve at first order
    # for any variable step sequence summing to the same time (G251), so a fixed
    # elastic time step is not needed.  ASPECT's `Use fixed elastic time step`,
    # `Fixed elastic time step` and `Use stress averaging` evaluate the elastic
    # response on a chosen timescale instead of the numerical one; they are
    # modelling options for special cases such as seismic cycles, not a
    # consistency requirement, and are deliberately not implemented here (G251).
    # `G <= 0` disables every viscoelastic term and leaves every law bit-identical.
    shear_modulus::Float64
    # Per-wall *tangential* velocity, `(xminus, xplus, yminus, yplus)`; `nothing`
    # falls back to the global `uDx`/`uDz` closures, which is the historical
    # behaviour.  The normal component still comes from `uDx`/`uDz`, so a wall with a
    # prescribed tangential value must also be `:rigid` in `wall_bc`.  This is what
    # makes a lid-driven cavity expressible: the top wall's tangential value no
    # longer forces the x-walls' normal velocity (cross-check).
    wall_u::NTuple{4,Union{Nothing,Function}}
    # Per-wall *normal* velocity, in the same `(xminus, xplus, yminus, yplus)`
    # order.  `nothing` (the default) keeps the global `uDx`/`uDz` closure for
    # that wall, which is the historical behaviour and bit-identical; a function
    # `(x,y) -> Float64` prescribes the wall's normal *component* instead: `u_x`
    # on an x-wall and `u_z` on a y-wall, in the positive coordinate direction,
    # exactly the convention of the legacy closures (it is **not** the outward
    # flux `u·n`).  The normal component is Dirichlet on every `:free` and
    # `:rigid` wall either way; `wall_un` only refines its value, per wall
    # (G258).
    wall_un::NTuple{4,Union{Nothing,Function}}
    # Per-wall *tangential* traction (BC stage 2, G259/G260), same
    # `(xminus, xplus, yminus, yplus)` order.  `nothing` (the default) is zero
    # traction, so no boundary load is assembled and the default card is
    # bit-identical; a function `(x,y) -> Float64` prescribes the **total**
    # Cauchy traction's tangential component, as the global component in the
    # positive coordinate direction (`t_z` on an x-wall, `t_x` on a y-wall),
    # exactly the convention of `wall_u`/`wall_un` and *not* a wall-local
    # (outward) component.  Integration by parts puts `t = sigma.n` on the
    # right-hand side with a plus sign, and on an axis-aligned wall the
    # tangential component is pressure-free (`t_x = sigma_xy`, `t_z = sigma_zy`),
    # so this is the exact `2 eta eps(u).n` tangential traction; the normal
    # component, which carries `-p`, needs the `:open` wall and is stage 3.  A
    # value is only allowed where the tangential component is free, i.e. on a
    # `:free` or `:open` wall: on a `:rigid` wall the tangential velocity is
    # Dirichlet and the load would be silently dropped by the constraint matrix,
    # so it is an error.
    wall_t::NTuple{4,Union{Nothing,Function}}
    # Per-wall *normal* traction (BC stage 3, G261/G262/G263), same
    # `(xminus, xplus, yminus, yplus)` order.  `nothing` (the default) is zero
    # traction; a function `(x,y) -> Float64` prescribes the **normal component of
    # the total Cauchy traction** -- the global component `t_x` on an x-wall and
    # `t_z` on a y-wall, in the positive coordinate direction, exactly the
    # `wall_t`/`wall_un` convention and *not* the scalar `n·(sigma·n)`.  Because
    # the traction is `t = sigma.n` with the **outward** normal, a pressure `p0`
    # imposed on an open boundary is `wall_tn = -p0` on the top wall and
    # `wall_tn = +p0` on the bottom wall.  It carries the pressure
    # (`t_n = -p + 2 eta eps_nn`), so it is only allowed on an `:open` wall -- the
    # only wall with a free normal velocity; on a `:free`/`:rigid` wall the
    # normal component is Dirichlet and the load would be silently dropped.
    # It is the *dynamic* pressure traction in the solver's nondimensional
    # pressure unit, not a lithostatic value.
    wall_tn::NTuple{4,Union{Nothing,Function}}
    rho_heating::Bool     # add -T*H to the energy source (rho=(1-T) heating)
    rho_cp::Bool          # weight M and C by rho=1-T (ASPECT rho*cp LHS)
    tosi::Union{Nothing,TosiLaw}  # :tosi viscoplastic law (absolute units + eta0)
    Di::Float64           # dissipation number alpha*g*D/cp; 0 disables both terms
    heating_adiabatic::Bool     # -Di*T*u_z source in the energy equation
    heating_dissipation::Bool   # (Di/Ra)*2*eta*edot:edot source
    refstate::Union{Nothing,SemRefState}  # adiabatic base state; nothing = conductive
    rho_ref_weight::Bool  # weight energy M/C by rho(y)*cp of the refstate
    compress_p::Bool      # add -rho Di gamma p to the momentum equation
    compress_gamma::Float64  # gamma factor of that term
    Twall_top::Float64    # physical wall temperature (top); theta_wall = Twall - Tad
    Twall_bot::Float64    # physical wall temperature (bottom)
    # Side walls: `:insulating` is the historical natural boundary (zero normal
    # flux), and `:fixed` replaces it with the physical temperatures below.  The
    # two knobs are separate so a run cannot change its side-wall physics by
    # setting a temperature and forgetting the switch.
    side_thermal::Symbol  # :insulating (default) or :fixed
    Twall_left::Float64   # physical wall temperature (x = 0), used by :fixed
    Twall_right::Float64  # physical wall temperature (x = Lx), used by :fixed
    deviatoric::Bool      # use eps - (1/3) div u I in the stress and shear heating
    latent_L::Float64     # nondimensional latent heat L/(cp dT) of the phase transitions
    latent_mode::Symbol   # :global (latent_L for every transition) or :per_phase
    # :composite diffusion + dislocation creep (ASPECT `diffusion dislocation`)
    comp_Ad::Float64      # diffusion-creep prefactor A_d (units 1/eta0)
    comp_Ed::Float64      # diffusion activation energy E_d/(R dT), nondimensional
    comp_Vd::Float64      # diffusion activation volume V_d, nondimensional depth factor
    comp_As::Float64      # dislocation-creep prefactor A_s (units 1/eta0)
    comp_Es::Float64      # dislocation activation energy E_s/(R dT)
    comp_Vs::Float64      # dislocation activation volume (depth factor)
    comp_n::Float64       # dislocation stress exponent n (~3)
    comp_Tcut::Float64    # floor on T_nd in the Arrhenius exponent; 0 = off
    comp_eps_min::Float64 # strain-rate floor of the dislocation branch
    comp_rule::Symbol     # :isostress (ASPECT, default) or :isostrain (legacy)
    # Depth-dependent Drucker-Prager yield: with depth
    # d = 1-y the cohesion and the friction coefficient are linear,
    #   c(d)  = yield_c + yield_c_grad*d
    #   mu(d) = sin(yield_phi) + yield_mu_grad*d
    #   tau_y = min(c(d)*cos(yield_phi) + mu(d)*p, yield_taumax)
    # Both gradients default to 0, which is the historical constant yield
    # surface bit for bit.
    yield_c_grad::Float64   # d(cohesion)/d(depth)
    yield_mu_grad::Float64  # d(mu)/d(depth), mu = pressure coefficient
    # Strain weakening: a scalar field chi in [0,1] with
    #   f(chi) = 1 - (1-weak_f)*chi,   tau_y <- f(chi)*tau_y
    weak_f::Float64         # residual weakening factor f_weak in [0,1]; 1 = off
    weak_eps_ref::Float64   # reference (end) plastic strain; <= 0 = off
    weak_heal_time::Float64 # healing time tau_heal; Inf = no healing
    weak_mode::Symbol       # :rate or :linear (ASPECT accumulation)
    # Per-wall velocity condition, in the order (xminus, xplus, yminus, yplus).
    # `:same` means "follow `bc`", which is the default and therefore bit-identical
    # to the historical single-`bc` behaviour; `:free`/`:rigid` override one wall.
    # `:open` (BC stage 3) leaves **both** components free: the natural (traction)
    # condition applies, optionally with the `wall_t`/`wall_tn` data.  An `:open`
    # wall has no Dirichlet component, so with any `:open` wall the nodal pressure
    # pin is invalid and the solver derives a different gauge (no pin); the
    # well-posedness rules live in the constructor below
    # (G262).
    wall_bc::NTuple{4,Symbol}
    # Rayleigh-number reference viscosity, in the units the material law
    # returns: `Ra` is defined as rho g alpha dT d^3/(kappa eta_ref) and the flow
    # solve uses eta_law/eta_ref (the docstring above states the convention and
    # why a rescaled law must rescale this with it).  Default 1.0 is the
    # historical convention and skips the wrapper, bit for bit.
    eta_ref::Float64
end

