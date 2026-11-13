# PHYSICS.md: the physics this code implements, and what was measured

This is the physics volume. `SOLVER.md` states the system and the discretisation; this article writes each
physical ingredient the way the code assembles it, explains the symbols, and reports what the term was measured to
do. Every number is measured on this checkout (WSL2, Julia 1.12.7, `linear=:direct`, one thread) unless labelled
an inference or attributed to a document; composite measurements predating the 2026-11-08 switch to `:isostress`
are marked legacy. The production reference is the realistic card of `CASE_REALISTIC.md`: a 2.5 x 1 box, `Ra =
1e5`, `H = 10`, two phase transitions with latent heat, composite rheology, free slip, `p = 2`, `max_level = 7`.

---

## 1. The energy equation as assembled

### 1.1 The equation, symbol by symbol

The unknown is the anomaly `theta = T - Tbar(z)`. In the code's non-dimensional units (lengths by the box height
`D`, time by `D^2/kappa`, temperature by `dT`, and `rho = cp = k = 1`) `assemble_semquad_energy` builds

```
(1 + sum_i L_i Gamma_T,i) (dtheta/dt + u.grad theta)
    = lap(theta) - u_z dTbar/dz - sum_i L_i Gamma_z,i u_z
```

`u = (u_x, u_z)` is the velocity in units of `kappa/D`, with `u_z` positive upward, so a downwelling has `u_z <
0`. `dtheta/dt + u.grad theta` is the material derivative of the anomaly, and the leading parenthesis multiplies
both parts of it: that is the apparent-heat-capacity form. `lap(theta)` is diffusion with unit conductivity
(entropy viscosity adds a positive `nu_ev` inside the stiffness). `Tbar(z)` is the base state, conductive with
internal heating when there is no reference state (§1.4) and the adiabat when there is one (§3). `Gamma_i(y,T)` is
the phase function of transition `i`, a number in `[0,1]` (§4.1), with `Gamma_T,i = dGamma_i/dT` at fixed height
and `Gamma_z,i = dGamma_i/dz` at fixed temperature. `L_i` is the dimensionless latent heat of that transition, the
coefficient of `-DGamma_i/Dt` in this operator, whose sign is derived in §4.2.

The two right-hand terms are the base advection `-u_z dTbar/dz` and the latent source `-sum_i L_i Gamma_z,i u_z`.
The first keeps a purely conductive state steady. Since `Gamma_z < 0` for every transition in this code (the phase
fraction falls as height increases), the sign of the source is the sign of `L_i u_z`.

### 1.2 The apparent heat capacity

Start from `rho cp DT/Dt = div(k grad T) - rho L (Gamma_T DT/Dt + Gamma_z u_z)`. The bracket is `DGamma/Dt`, the
rate at which a parcel changes phase; moving the `Gamma_T` part left gives the capacity `1 + L Gamma_T` on `DT/Dt`
and leaves `-L Gamma_z u_z` as a source. `Gamma_T` is the sensitivity of the phase fraction to temperature at
fixed depth; near a transition it is of order `slope/(2 width)`, which for this card is `0.032/0.05 = 0.64`. So `L
Gamma_T` is the latent heat per unit sensible heat across the transition, O(0.1) here, and the latent heat is a
percent-level correction to the heat capacity rather than a term that changes the character of the equation.

The explicit-source form, `dtheta/dt + u.grad theta = lap(theta) - u_z dTbar/dz - L[Gamma_T (dtheta/dt + u.grad
theta) + Gamma_z u_z]`, is the same equation: the two assembled operators agree to `1.7e-16` relative (G194). It
is not used for the step, because the capacity form carries the latent term inside the mass matrix with no extra
lag, while the source form needs `DGamma/Dt` from the previous level and adds a negative source where the capacity
is smallest. The cheap version survives as `latent_scheme = :lagged`; at `maxpic = 1` it is bitwise the default,
because there is no iterate to evaluate.

ASPECT's guard `rho cp + lhs >= 0` is reproduced: the assembly throws when `1 + sum_i L_i Gamma_T,i < 0` at a
quadrature point. With `latent_L = 10` and the G194 transition the capacity reaches `-1.88` and the run is
rejected; without the guard that run reached `max|theta| = 14.4` in 120 steps. The guard is for hand-written
coefficients, because with coefficients derived from the Clapeyron slope it cannot fire when `drho > 0`: `dS_A =
gamma drho/rho^2` then has the sign of the slope and `L_i Gamma_T,i = (1/2) T_i dS_A,i slope_i /(cp dT width_i)(1
- tanh^2) >= 0`, so every transition *adds* to the capacity. Measured over a `(y,T)` grid with the card's two
transitions, the capacity stays in `[1.0, 1.1216]`.

The four phase derivatives are analytic and checked against central differences at five `(y,T)` states:
`dGamma/dT` to `2.6e-9`, `dGamma/dz` to `5.6e-10`, `d^2Gamma/dT^2` to `4.6e-9`, `d^2Gamma/dz dT` to `4.3e-9`
(relative, `h = 1e-6` and `1e-7`, G194).

### 1.3 The conservative form, and the discrete identity

Write `h = theta + sum_i L_i (Gamma_i(y,T) - Gamma_i(y,Tbar))`. The assembled equation is algebraically `dh/dt +
u.grad h = lap(theta) - u_z d(Tbar + sum_i L_i Gammabar_i)/dz`, and the last term integrates to zero over the box
because there is no normal flow and `div u = 0`. What remains is `d/dt int h = int_wall dtheta/dn`: the enthalpy
defined with the latent contribution is conserved by advection and diffusion, so the latent heat redistributes
heat along a streamline instead of creating it.

Two consequences look like omissions and are not. `Gamma_z` is the **full** derivative, not the anomaly
`Gamma_z(T) - Gamma_z(Tbar)`: the base part has been moved into the base advection, which is how the identity
closes. And the capacity multiplies `u.grad theta` but not `-u_z dTbar/dz`; applying it to the base term too would
leave a spurious `L Gamma_T u_z dTbar/dz` in the budget.

The discrete counterpart is the row-sum identity of the assembled operator. With `1` the vector of ones, `sum_i
phi_i = 1` and `sum_j grad phi_j = 0`:

```
1'(M v)     = int (1 + sum_i L_i Gamma_T,i) v dV     (the BDF2 enthalpy rate)
1'(K theta) = 0                                      (measured 5.2e-13 .. 8.8e-13)
1'(C theta) = 0.5 int (1 + sum_i L_i Gamma_T,i) u.grad theta dV
-1'F        = int u_z dTbar/dz dV + int sum_i L_i Gamma_z,i u_z dV
            = Ibase + latent_src
Qc + eps    = dH/dt + sum(C theta) + Ibase + latent_src                     (ID1)
```

`Qc` is the sum of the wall rows of the **step** operator and `eps` is the part of the residual the constrained
solve leaves outside the constraint range, measured at 0.05 to 0.28 percent of `|Qc|`, the linear solver's
backward error. `docs/VALIDATION.md` derives this identity and shows that the driver's historical `closure` column
is not it: it omits `dH/dt`, uses the PDE-level `latent_adv` where the skew assembly has `0.5 int (1 + L Gamma_T)
u.grad theta`, and adds `Ibase` un-normalised. Its residual decomposes exactly and `-dH/dt` supplies 87 to 137
percent of it.

### 1.4 The base state, and where the internal heating lives

Without a reference state the base profile is the conductive solution of the internally heated problem: `Tbar(z) =
(1-z) + (H/2) z(1-z)`, `dTbar/dz = -1 + (H/2)(1-2z)`, `lap(Tbar) = -H`. So the physical `dT/dt + u.grad T = lap(T)
+ H` becomes, for the anomaly, `dtheta/dt + u.grad theta = lap(theta) - u_z dTbar/dz` with no explicit source: the
`+H` is exactly `-lap(Tbar)`, `semquad_base_laplacian` returns zero, and the heating is already in the base
gradient. The Nusselt numbers are normalised with the same gradient, `Nu_top = -(dT/dy)|_top/(1+H/2)` and `Nu_bot
= -(dT/dy)|_0/(1-H/2)`, so a conductive state gives `Nu_top = Nu_bot = 1` and the steady balance is `Nu_top -
Nu_bot = H`.

With `H = 10` the conductive mid-depth temperature is `Tbar(0.5) = 1.75`, the interior sits far above the lid, and
the viscosity there is of order `1e-3` of its calibration-point value (the card's `2.484e-3` at `T = 1.5`), so
the interior Rayleigh number is `Ra/eta ~ 1e7` to `1e8` rather than the nominal `1e5` (`CASE_REALISTIC.md` §5.2).
With a reference state the base is the adiabat, `+lap(Tad) = (alpha g/cp)^2 Tad` becomes a source and `H` leaves
the energy equation; that incompatibility is §3.3.

### 1.5 The time step, and what the energy equation taught the hard way

The step is BDF2 with

```
A = (a/dt) M + K + C,   rhs = (b/dt) M theta_n - (c/dt) M theta_(n-1) + F
a = (2w+1)/(1+w),  b = 1+w,  c = w^2/(1+w),  w = dt/dt_prev
```

solved as `Cfree'(A theta - rhs) = 0`. The mass term carries the capacity, so the latent heat is implicit.

The defect this subsection records: for a while the Picard loop passed the *running iterate* as the `theta^n` of
the mass term. With `theta_n = theta_iter` the fixed point is not the time step. Since `a - b = -c` for BDF2, the
iteration converges to `(c/dt) M(theta - theta_(n-1)) + K theta + C theta = F`, a different scheme; at the first
step the mass terms cancel identically and the iteration converges to the **steady** energy equation. It is
invisible at `maxpic = 1`, the setting of every production run, because then `theta_iter == theta_n`.

Measured on the strong composite case (one step, `maxpic = 12`, `pic_tol = 1e-6`): before the fix the Picard
changes *grew* (`3.2e-3`, `5.0e-3`, ..., `7.8e-3`) and the Newton converged to `||theta|| = 3.4e-11`, that is to
`theta = 0`, the trivial steady solution of `K theta + C theta = 0`. After the fix the same iteration converges
(`3.2e-3 -> 1.9e-3 -> ... -> 4.3e-7` in ten iterations) and the state satisfies the backward-Euler residual to
better than `1e-8`; `t_g193` asserts that the steady equation does *not* hold there. The fix is one argument and
is bitwise identical at `maxpic = 1`. The lesson generalises: the fixed point of an iteration is the equation
actually assembled, and the mass term is the only part of the operator that distinguishes a step from a steady
solve. Every convergence claim about `maxpic > 1` made before the fix is void; `docs/BUG_LEDGER.md` inventories them
and `t_g142` had to be rewritten because it had encoded the defect as an expected divergence.

---

## 2. Rheology

### 2.1 The composite law and the two combination rules

Two creep mechanisms deform the same rock at the same stress, so their strain rates add. With `tau` the deviatoric
stress invariant and `eps_II = sqrt(0.5 edot:edot)`, the branches are

```
edot_diff(tau) = A_d exp(-(E_d + V_d(1-z))/T_nd) tau
edot_disl(tau) = A_s exp(-(E_s + V_s(1-z))/(n T_nd)) tau^n
eta = tau/(2 eps_II)   at the tau solving  edot_diff + edot_disl = eps_II
```

with `T_nd = max(T + eta_T0, comp_Tcut)` the non-dimensional absolute temperature (`T` runs 0 at the top wall to 1
at the bottom, so `eta_T0` is the surface temperature in units of `dT`), `z` the height so `1-z` is depth, `n` the
stress exponent and `comp_Tcut` a floor on `T_nd`. The code solves in the viscosity itself: with `x =
eta/eta_disl(eps_II)` and `k = eta_disl(eps_II)/eta_diff`, the partition is `k x + x^n = 1`, `eta = x
eta_disl(eps_II)`, by Newton with a bisection safeguard on `[1/(1+k), 1]`, tolerance `1e-15` on `g`, cap 100.

The one-mechanism limits are exact and skip the iteration: `comp_As = 0` is pure diffusion, `comp_Ad = 0` pure
dislocation, and `n = 1` makes both branches linear so the partition and the harmonic mean coincide.
`comp_eps_min` floors the strain rate in `eta_disl`; as `eps_II -> 0` the dislocation viscosity diverges and the
law is pure diffusion for any positive floor, so the floor matters only for the derivative.

The legacy rule `comp_rule = :isostrain` evaluates both branches at the **total** strain rate and takes the
harmonic mean, `1/eta = 1/eta_diff + 1/eta_disl(eps_II)`, which lets each mechanism see the whole strain rate. The
rules agree in the one-mechanism limits and at `n = 1`; with `r = eta_diff/eta_disl` and `x` the root of `x/r +
x^n = 1`, the isostress-to-isostrain ratio is `1 + x - x^n`, equal to `1.364656` (`n = 3`) and `1.410639` (`n =
3.5`) at `r = 1` and reaching `1.384900` / `1.432758` at `r = 0.714941` / `0.732692`. Two controlled ASPECT runs
measure `1.281706` at `r = 0.3` and `1.410639` at `r = 1`, so ASPECT 2.5 solves the isostress partition and this
code did not until 2026-11-08. The switch moves the 64-leaf case by `dVrms/Vrms = -35.28%`, `dNu/Nu = +1.7e-8` and
`|dtheta|/|theta| = 6.8e-3`, and the converged viscosity is larger by a median factor `1.3592` (max `1.3849`). The
implicit-differentiation tangents agree with central differences to `4.8e-9` (`deta/dT`) and `1.9e-8`
(`deta/deps`), and at `r = 1` they differ from the legacy ones by `27.2%` and `59.1%`, which `t_g201` asserts so
that a finite-difference test cannot pass with the wrong derivative.

### 2.2 The parameters, and the mapping to ASPECT

Eleven keywords: `comp_Ad`, `comp_Ed`, `comp_Vd` (diffusion), `comp_As`, `comp_Es`, `comp_Vs`, `comp_n`
(dislocation), `comp_eps_min`, `comp_Tcut`, `comp_rule`, and the shared `eta_T0`, `eta_min`, `eta_max`. All
default to off. The mapping to ASPECT's dimensional parameters, with `eta0` the reference viscosity, `P0 = rho0 g
D`, `S = kappa/D^2` and grain size `d`, is

```
comp_Ad = 2 eta0 A_d/d^m   comp_Ed = E_d/(R dT)   comp_Vd = V_d P0/(R dT)
comp_As = 2 eta0 A_s^(1/n) S^((n-1)/n)
comp_Es = E_s/(R dT)       comp_Vs = V_s P0/(R dT)   comp_n = n
```

The factor 2 is ASPECT's `stress = 2 eta edot` convention; the grain size is folded into `comp_Ad`, so it cannot
be varied independently and there is no grain-size field. Two limits of the mapping are measured: ASPECT
multiplies the activation volume by the solver's pressure while this code multiplies by the depth proxy `(1-z)`,
exact only for `P = P0(1-z)`; and ASPECT's strain-rate invariant uses deal.II's 2-D deviator, which subtracts
`tr/2`, so the two invariants differ by `tr^2/4`. For `div u = 0` they are identical, and this code's velocity is
pointwise divergence free. The first written version of the mapping was inverted by `4.0e51` (`comp_Ad`) and
`6.1e15` (`comp_As`); `t_g200` pins both.

### 2.3 The production card: calibration, and what the law reduces to

The card fixes `comp_Ed = 12`, `comp_Es = 17` (the ASPECT ratio 530/375 kJ/mol), `comp_n = 3`, `eta_T0 = 0.5`,
`comp_Vd = comp_Vs = 0`, `comp_eps_min = 1e-12`, `eta in [1e-4, 1e4]`, and solves the two prefactors from
`eta_eff(0.5, 0.5, edot_ref) = 1` and `eta_eff(0.1)/eta_eff(0.9) = 1e4`. Both conditions need a strain rate;
`edot_ref = 1.65e3` is measured from a constant-viscosity precursor of the same case (120 steps, `eta = 1`; median
`1.10e3`, max `7.20e3`). The contrast fixes only the branch ratio: with `s = 1/T_nd(0.1) - 1/T_nd(0.9) = 0.9524`
the achievable contrasts are `exp((E_s/n)s) = 2.2e2` and `exp(E_d s) = 9.2e4`, so `1e4` lies inside and the ratio
is unique at `r = eta_d/eta_s = 0.129720`, the dislocation branch being 7.7 times the diffusion viscosity at the
calibration point. The value condition then gives `comp_As = 2.6796419238e-1` and `comp_Ad = 1.6240183119e5`, and
the calibration closes to round-off (`1.000000000000000` and `9.999999999999991e3`, relative deviation `9.1e-16`).

That is a genuinely mixed branch ratio at the calibration point. Over the states the production run actually
occupies, the card is nevertheless effectively a pure diffusion-creep model. Measured on a state of the wall-refined
E run (step 7208, 9994 leaves, `case_ev_0.11/resume.h5`), evaluating the card's law at 202055
points (the velocity nodes plus the quadrature points of every leaf): the effective viscosity equals the
diffusion-only law to `0.09%` at the worst point (median relative difference `1e-7`), and the dislocation branch
carries at most `0.93%` of the strain rate (median `2.4e-6`, 90th percentile `3.1e-5`, 99th `2.2e-4`). The state
spans `T` from 0 to 1.652 (median 1.468) and `eps_II` from 0.016 to 4.95e4 (median 4567), so it covers the box.

The reason is structural. The calibration makes the dislocation branch the stiffer one at `T = 0.5`, and it stays
the stiffer one wherever the material is hot, which is where the strain rate is large. The only corner where
dislocation creep becomes the weaker branch is cold material moving fast: at `T = 0.1` and `eps_II = 1.65e3` it
would carry `89%` of the strain. A stagnant lid does not occupy that corner, because cold material in this run is
slow, so the card's strain-rate dependence is not a first-order ingredient (an inference; the numbers above are
the measurement). The run-level evidence is in `CASE_REALISTIC.md` §8.2: a restart that evaluates the lagged
viscosity once from a zero velocity field, putting the dislocation branch at the strain-rate floor, differs from
the uninterrupted run at the `1e-5` level. The rule switch is correspondingly small on this state, at most `9.6%`
in the viscosity (median `1.3%`), against the 35 percent `Vrms` shift of the synthetic case. The synthetic cards
are where shear-thinning belongs; this card calibrates a temperature contrast.

### 2.4 The other viscosity laws

`:expT` is `eta = exp(-log(deta_T) T)` with `deta_T` a ratio and `eta(T=0) = 1`; `:expTz` is ASPECT's `exp(-eta_b
T + eta_c(1-z))`, used for Blankenbach 2b with `eta_b = log(16384)`, `eta_c = log(64)`; `:arrhenius` is
`exp(E(1/(T+T0) - 1/(1+T0)))`. The diffusion-only limit of the composite law is exactly the Arrhenius law: with
`comp_Vd = comp_Es = comp_Vs = 0`, `comp_As = 0` and `comp_Ad = exp(comp_Ed/(1+eta_T0))` the expressions differ by
at most `3.7e-15` over `T` in `[0,1]`, `z` in `{0,0.5,1}`, and the two laws drive the same run to `max|dtheta| <=
1e-10` (G192).

`:tosi` is the Tosi et al. (2015) viscoplastic law in absolute units: `eta_lin = exp(-ln(etaT) T + ln(etaZ) z)`,
`eta_pl = eta_star + sigma_y/sqrt(edot:edot)`, `eta = 2/(1/eta_lin + 1/eta_pl)` clamped to `[eta_min, eta_max]`
and divided by `eta0`. The benchmark law is already non-dimensional and `eta0 = 1`: with `eta0 = 0.1` the
effective Rayleigh number drops tenfold and the case never convects, while `eta0 = 1` reproduces the reference
trajectory at `t = 0.136` to 5 to 8 percent. `z_is_depth` exists because the paper's equation (7) carries a depth
term with `etaZ = 10` and the ASPECT plugin drops it, so the choice must be explicit. The purely linear branch
(`eta_star = sigma_y = 0`, Case 1) returns `eta_lin` and must not use `eta_init`.

### 2.5 The plastic branch, and strain weakening

The Drucker-Prager branch applies when the viscous stress would exceed

```
tau_y = min( f(chi) ( c(d) cos(phi) + mu(d) p ), tau_max )
c(d)  = yield_c + yield_c_grad d          mu(d) = sin(yield_phi) + yield_mu_grad d
```

with `d = 1-y` the depth, `p` the *dynamic* pressure (no lithostatic part, no pore pressure) and `f(chi)` the
weakening factor. The profile is imposed on `mu = sin(phi)` because `mu` is the object in `tau_y = c + mu p` and
the quantity the literature profiles. Both gradients default to zero; the validator rejects a profile that goes
negative anywhere in `d` in `[0,1]`. The measured content, beyond correctness: the tangents `deta/dp`, `deta/deps`
and `deta/dT` match central differences at two depths to `1e-11` to `1e-10`; the Newton's pressure block `Kp` is
`1.957e-1` of the pressure-direction Jacobian norm and dropping it turns a `1.6e-9` tangent error into `2.041e-1`;
and the profile is a real effect, the constant-law run having `Vrms = 1.446056056e3` against `1.514343568e3`,
`4.51%` lower. ASPECT has no depth-dependent friction or cohesion, so the only cross-code statement is the
zero-gradient limit, bitwise identical to the pre-change code on 2500 parameter combinations (`t_g195`, 2791
assertions).

Strain weakening transports `chi` with the same BDF2 coefficients as the temperature:

```
dchi/dt + u.grad chi = (1-chi) eps_pl/eps_ref - chi/tau_heal   (:rate)
dchi/dt + u.grad chi = eps_pl/eps_ref                          (:linear)
f(chi) = 1 - (1-weak_f) clamp(chi, 0, 1)   in [weak_f, 1]
```

`eps_pl` is the second invariant of the total strain rate while the plastic branch is active and zero otherwise,
which is what ASPECT's `visco_plastic` accumulates. The 1-D analytic check is exact: uniform shear with `g = 25`,
`lambda = g + 1/tau_heal = 27` gives `chi_ss = 25/27 = 0.925925926`; after 600 steps the maximum error is
`8.502e-08` with a spatial spread of `6.22e-14`; the `:linear` ramp is exact to `1.166e-15`; and the `:rate`
transient is second order (error ratio `4.04` for a `dt` halving). The raw transported `chi` is not clamped inside
the solve, because clamping makes the Picard fixed point differ from the Newton root by a measured 14 percent; the
driver clamps the stored field after each step. `:galerkin` preserves a constant field exactly while the
production `:skew` operator does not (drift about `4e-4` over 40 steps at `Ra = 1e4`), so the 1-D test uses
`:galerkin` and the 2-D runs keep `:skew`.

### 2.6 The clamps

Every law goes through `_clip_eta`, which applies `eta_min` and `eta_max` to the *effective* viscosity once, after
the combination. The clamp is continuous and its first derivative is not: the crossing solves `eta_raw(eps*) =
eta_max` exactly, so both one-sided limits equal the bound and the analytic derivative returns zero on the clipped
side. Measured in the gate configuration, `eps* = 1.397061e-3` for `:isostress` and `1.270208e-3` for
`:isostrain`; at `eps*(1+1e-6)` the value is `0.99999935`, that is `6.5e-7` below the bound, while the slope just
outside is `-4.61e2` (finite difference `-4.607e2`, `0.08%`).

The clamp is not a corner case. On the strong composite test case `6.2%` of the viscosity evaluations sit on a
clamp under `:isostrain` and `7.6%` under `:isostress`; on the production state of §2.3, `0.92%` of the sampled
points are clipped (the driver's node census reports `0.94%`, all at `eta_max` in the lid). A census over `T` in
`[0,2.6]` at `edot_ref` clips `0/261` points for `eta in [1e-4,1e4]` and `75/261` for `[1e-3,1e5]`, which is why
the card uses the former. `comp_Tcut` floors `T_nd` in both exponents; below the floor the law is temperature
independent and `deta/dT = 0`.

---

## 3. Compressibility

### 3.1 ALA and TALA, as this code writes them

A reference state enters four places: the buoyancy `Ra rho(y) alpha(y) theta` (anomaly only), the base advection
`-u_z dTad/dz`, the base curvature `+lap(Tad)`, and the mass form. The hydrostatic reference pressure is never
discretised; the momentum equation is solved for the dynamic pressure with the hydrostatic gradient subtracted
analytically, so it is not in the load vector. Two mass forms are physical:

```
TALA  div u = 0                                (mass_form = :boussinesq)
ALA   div(rho u) = 0, i.e. int q (rho div u + u_z drho/dz) dV = 0   (:anelastic)
```

A third, `:rho_div`, is historically present, inconsistent and kept for regression only.

The anomaly form's payoff is measured: at `theta = 0` the load vector is exactly zero (`max|fv| = 0.0`), the solve
returns the zero state (`max|u| = 0.0`, `max|p| = 0.0`, residual `0.0`), and this is resolution independent
because no reference gradient is present to be projected. A *resting conductive* column is the hard version: `T =
1-z` has a smooth buoyancy a pressure could balance in the continuum, but the `Q1` pressure space cannot represent
its antiderivative, so the residual is `O(h^2)` (orders 1.94, 1.94, 1.96) and the spurious flow is quartic,
`max|u| = 8.690e1, 6.167, 0.4234, 2.781e-2` at 4, 16, 64, 256 leaves (orders 3.82, 3.87, 3.93); at 64 leaves it is
`0.42` against a buoyancy of order `Ra|theta_c| ~ 3.8e5`. The counterfactual is measured too: discretising the
*full* hydrostatic force balances a stratified density only at `O(h^2)` (orders 1.83, 1.91, 1.96) while a linear
`p_ref` is balanced to round-off, so the dynamic-pressure form avoids both problems at the price of the quartic
drift.

### 3.2 The EOS reference state

`SemRefState(...; eos=:adams_williamson)` and `:murnaghan` integrate from the surface down:

```
dT/dy = -Lambda(y) T,  Lambda = alpha(y) g/cp(y)
drho/dy = -rho g/phi(y),  phi = K_S/rho
dp/dy = -rho g
```

For Adams-Williamson the closure `K_S = rho cp/alpha` gives `phi = cp/alpha`, hence `dln rho/dy = -Lambda`:
density and adiabat obey the same exponential, the classical Adams-Williamson relation. With constant properties
this is `rho = rho0 exp(Lambda(1-y))`, exactly the `rho_fun = exp(1-y)` profile of the ALA/TALA comparison, so the
EOS is the consistent density of that set-up. For Murnaghan, `K_S = K0 + K0'(p - p_surf)` couples the density ODE
to the hydrostatic pressure and resists compaction: with `K0 = 1, K0' = 4`, `rho(0) = 1.587401051968` against `e =
2.718` for the self-compacting adiabat.

`T`, `rho` and `p` are tabulated on `nz` uniform nodes (RK4, 8 substeps per interval) and interpolated with cubic
Hermite whose node derivatives are the ODE right-hand sides. The derivatives the physics needs are not differences
of the table: `dTad/dz = -Lambda Tad`, `lap(Tad) = (Lambda^2 - dLambda/dy) Tad` and `drho/dz = -rho g/phi` are
evaluated from the EOS at the query point, so the first and second derivatives belong to the same adiabat. The
tabulation is fourth order (`2.485e-9` at `nz = 33` to `6.067e-13` at 257), and `rho(0) = 2.718281828459`, that is
`e`, not `rho0`.

The EOS matters once `alpha` depends on depth. With `alpha(y) = 1 + 0.5(1-y)`, `Tad(0)` moves from
`5.436563656918e-1` to `6.980685914924e-1` (28.4 percent) and `lap(Tad)` at mid-depth from `5.484545803001e-1` to
`7.239600460205e-1` (24.2 percent). The hydrostatic node identity `dp = -rho g` holds bitwise. The `u = 0` energy
budget closes at the same `0.9348` ratio as the closed-form state, so the EOS source reproduces the analytic
source integral to the same spatial quadrature error.

### 3.3 What the driver can and cannot combine

Internal heating and a reference state cannot be combined, and it is structural: `semquad_base_T` returns
`semquad_Tad`, so `Tbar(z,H)` and with it the only route by which `H` enters the energy equation is never used.
The energy equation then has no internal heating at all; `H` would reappear only through `rho_heating = true`,
which adds `-T H` *and* switches the operator to `rho = 1-T`, a different model. `pr.H` would still enter the
viscosity, because the composite `eta_fun` and the Newton evaluate `T` as `Tbar(y,H) + theta`, so with `H > 0` the
viscosity would see a temperature offset by `Tbar - Tad` while the buoyancy and energy equation see `Tad + theta`.
The production card therefore runs with `H = 10` and no reference state, and with `refstate = nothing` the
anelastic mass form reduces to `div u = 0` anyway. The price is the measured effect of §3.4; adding it back is
about six lines in `julia/src`, none on the default path. The EOS integration does not include `H` either, so `H =
0` throughout the EOS measurements.

### 3.4 The measured ALA-vs-TALA difference

Same case, same buoyancy, same reference density, only the mass form changed: `Ra = 1e6`, 64 leaves, the strong
composite law, `rho = exp(1-y)`, walls on the adiabat, `theta_0 = 0.05 sin(pi x) sin(pi y)`, 5 steps at `dt =
2e-5`. Under the current rule:

```
Vrms  TALA 4.297681547e1   ALA 4.002035085e1   dVrms/Vrms = -6.879e-2
Nu    TALA 0.301769659/0.446452505   ALA 0.301769612/0.445766111   dNu/Nu = -1.55e-7
max|div u| = 3.8e-14 (TALA)      max|div(rho u)| = 8.1e-11 (ALA)
```

Each field satisfies its own constraint and only its own: the wrong residuals are `3.87` and `1.68`. Under the
legacy rule the same comparison gave `-7.557e-2` at 64 **and** at 256 leaves to all printed digits, with `Nu`
differing by `-2.02e-7`. That resolution independence is the point: ALA and TALA are two continuum models of
compressibility, so their difference does not vanish under refinement, and the measured `-6.9%` (current) or
`-7.6%` (legacy) is a model difference, not a discretisation error. The expectation of an `h^2` interior effect is
not confirmed. The difference field is strongly non-uniform: in eight equal slabs from bottom to top it is `14.5,
12.3, 8.3, 7.5, 6.2, 3.9, 2.2, 1.3` percent of the local velocity, largest at the bottom wall where the reference
density is largest and most stratified. Fixing everything but `c` in `rho = exp(c(1-y))` gives `dVrms/Vrms =
-2.0e-15, -2.0045e-2, -3.8995e-2, -7.5568e-2` at `c = 0, 0.25, 0.5, 1` (legacy), so the difference is the
adiabatic mass term and vanishes at `c = 0` to round-off.

Two lessons from the same runs. `Nu` barely discriminates the mass forms in the composite case (`1e-7`), so a
Nusselt comparison alone would have missed a 7 percent velocity difference; at `eta = 1` the same comparison gives
`-5.607e-2` at 64 leaves and `-5.614e-2` at 256, with `dNu/Nu = -1.44e-1` at the top wall. And the 5-step `Ra =
1e6` state is a spin-up transient: `Nu_top` and `Nu_bot` differ by 50 percent, and at 20 steps the flow is still
growing (`Vrms` 43 at 5 steps to 260 at 20 with `maxpic = 20`; legacy `4.1502e2`/`3.1487e2` at `maxpic = 1` and
`2.0505e3`/`1.5406e3` at `maxpic = 20`). The ALA/TALA *difference* is converged at 5 steps, which is what the
comparison uses.

---

## 4. Phase transitions

### 4.1 The phase function and the card's two transitions

`phase.jl` implements ASPECT's phase function `Gamma(y,T) = 0.5(1 + tanh(deviation/width))` with `deviation = (z0
- y) - slope(T - Tref)`. `z0` is the transition height, `width` its half-width, `Tref` the reference temperature
and `slope = dz/dT` the Clapeyron slope as a height shift per unit temperature. Buoyancy enters as `-Ra B
(Gamma(y,T) - Gamma(y,Tad))` with `B = drho/(rho0 alpha0 dT)`; subtracting `Gamma(Tad)` keeps a state on the base
adiabat neutrally buoyant. The card's two transitions:

| | 410 | 660 |
|---|---|---|
| `z0` | 0.87 | 0.78 |
| `width` | 0.025 (72 km) | 0.025 |
| `dP/dT` | +3.0 MPa/K | -3.0 MPa/K |
| `slope = dz/dT` | +0.032 | -0.032 |
| `drho` | +180 kg/m^3 | +300 kg/m^3 |
| `B` | 1.818 | 3.030 |

The slope conversion is `slope = (dP/dT)/(rho0 g D) dT`; with `rho0 g D = 9.388e10 Pa` and `dT = 1000 K`, `3.0
MPa/K` gives `0.032` and `-3.0 MPa/K` gives `-0.032`. With this sign convention an upwelling gains buoyancy at the
410 (exothermic, positive slope) and loses it at the 660 (endothermic, negative slope), the textbook behaviour.

The width sets the resolution: at `max_level = 7` the leaf is `1/256 = 0.0039`, that is 6.4 leaves per width,
while level 6 would give 3.2, marginal for a `tanh` whose derivative `slope/(2 width) = 0.64` is the largest
gradient in the problem. That, not the boundary layer, is why the card uses `p = 2, max_level = 7`.

### 4.2 The latent coefficients

The latent heat is derived from the same slope, not fitted. The entropy change of a transition is `dS_A = gamma
drho/rho^2` with `gamma = dP/dT`, and the latent heat per unit mass is `T dS_A`. ASPECT writes `+T dS_A
DGamma/Dt`; this operator writes `-L DGamma/Dt`, so the two agree with

```
L_i = -T_i dS_A,i/(cp dT) = -T_i gamma_i drho_i/(rho_ref^2 cp)
    = -T_i slope_i drho_i g D /(rho_ref cp dT)     [code units]
    = -theta_abs,i slope_i B_i Di,   Di = alpha g D/cp
```

with `T_i` the *absolute* non-dimensional temperature at the phase depth, `Tbar(z0) + T_surface`, evaluated once
from the base state. For the card's scale set (`rho_ref = 3300`, `cp = 1000`, `dT = 1000`, `g = 9.81`, `D = 2.9e6
m`, `T_surface = 0.5`):

| | `T_abs` (code / K) | `dS_A` (J/kg/K) | ASPECT `L_A` | **our `L_i`** | old code |
|---|---|---|---|---|---|
| 410 | 1.1955 / 1195.5 | +49.66 | +0.060 | **-0.059364** | +0.11 |
| 660 | 1.5780 / 1578.0 | -82.76 | -0.131 | **+0.130596** | +0.11 |

The 1 percent gap to ASPECT is the rounded 1200 K / 1580 K of its table.

The sign is the physics. `slope > 0` gives `L < 0`, and for a downwelling (`u_z < 0`, hence `Gamma_z u_z > 0`
because `Gamma_z < 0`) the source `-L Gamma_z u_z` is positive: the 410 heats descending material, while `slope <
0` gives `L > 0` and the 660 cools it. A single positive `latent_L = 0.11` applied to both, as the code did before
2026-11-08, cools the 410 where the physics heats it and is right at the 660 by accident (§6.1). The apparent
capacity of the card measured over a `(y,T)` grid is `[1.0, 1.1216]`, bounded below by 1 for the reason in §1.2.
The old card's comment claimed `0.094` for the 410; that is not reproducible from the case's own values (`T =
1195.5 K`, `dS_A = 49.66 J/kg/K` give `0.0594`), and `0.11` was a compromise between two numbers of opposite sign.

Two approximations are measured and left in place: the coefficient is evaluated once at the phase depth from the
base state rather than at each quadrature-point temperature, an `O(theta/T_abs)` error of about 2 percent here
(`|theta| <= 0.03` against `T_abs = 1.2` to `1.6`); and `rho_ref^2` is the reference density where ASPECT uses the
local one, about 3 percent in this formulation.

### 4.3 The measured sign, and what the sign bug did

The 1-D column test makes the sign unambiguous: prescribed vertical velocity, one transition at a time, one step,
difference against the same step without latent heat, mean anomaly over the transition band.

| transition | descending (`u_z = -1`) | ascending (`u_z = +1`) |
|---|---|---|
| 410, `L = -0.059364` (exothermic) | **+1.141e-3 (heating)** | **-1.149e-3 (cooling)** |
| 660, `L = +0.130596` (endothermic) | **-1.229e-3 (cooling)** | **+1.219e-3 (heating)** |
| 410, `:global` `L = 0.11` (old code) | **-2.203e-3 (cooling, the bug)** | +2.20e-3 |

`t_g206` asserts those four signs plus `sign(mdown_global) != sign(mdown_per_phase)`, and the realistic
consequence is small but real: the 115-step A/B of the production card, identical except for `CASE_LATENT`, gives
`Nu_top` 5.99908 (global) against 5.99904 (per-phase), `Nu_bot` -4.8689 against -4.8684, `Vrms` 1.5980e4 against
1.5959e4 (`-0.13%`) and `Tmax` 1.8565 in both, while the ASPECT ratios are unchanged (`Nu_bot` 15.00 to 15.02
percent, `Vrms` 16.1 to 16.0 percent). The wrong sign was real physics at the percent level in `Vrms` and at the
`1e-5` level in `Nu`, and it was not the origin of the 5 to 16 percent dynamic difference between this code and
ASPECT.

### 4.4 The time accuracy of the latent coupling

The capacity is evaluated at the current iterate (`latent_scheme = :implicit`) or at the start-of-step temperature
(`:lagged`). At `maxpic = 1` there is no iterate, so the two are the same scheme and the returned `theta` is
bitwise identical for the default, `:lagged` and `:implicit` in 1-D and 2-D (`t_g199`).

On the 1-D column against a `dt/8` reference at `pic_tol = 1e-11`, the measured orders are: implicit `2.02`,
lagged-minus-implicit difference `1.01`, lagged total error `2.06` for the physical transition and about `1.1` for
a deliberately sharper one (`width = 0.05`, `L = 0.45`, `theta_0 = 0.3`) where the lag dominates. The crossover
where the latent lag becomes the accuracy bottleneck is `dt ~ 1.7e-4` for the first case and below `4e-3` for the
sharp one, so at a production `dt` on the physical transition the first-order lag is not the largest error term.
The one-step enthalpy budget closes at first order for both schemes (`-5.57e-7` at `dt = 4e-3`, halving with
`dt`): the capacity form advances `c(theta_lat) BDF2 theta` rather than `BDF2 h(theta)`, and those differ by an
`O(theta^2)` algebraic term wherever `c` is evaluated.

In 2-D the practical error is larger, and the isostress switch made it larger still. On the `Ra = 1e5`
composite-plus-phase case at `dt = 2.5e-4`, the error in `theta` against a `dt/8` reference is `4.603e-1`
(default), `4.125e-2` (iterated lagged) and `2.670e-2` (iterated implicit) under the current rule; legacy
`9.77e-2`, `3.80e-3`, `1.05e-3`. The latent-specific error is therefore `5.30e-2` at that `dt`, an order of
magnitude above a 0.5 percent tolerance, where the legacy rule had `3.72e-3`, just inside. The fine-range order of
the implicit scheme is `1.87` under both rules, so the change is in the error constant (a stiffer law, a more
vigorous late transient), not the convergence rate. The default's error is dominated by the one-sweep lag of the
flow, which exists with no phase transition at all (`9.4%` with no phases, `68.6%` with latent heat, current rule;
`23.4%` and `53.4%` legacy).

So a qualitative run can keep the default, while a benchmark comparison at the 0.5 percent level needs the
iterated step; with it the implicit latent evaluation comes for free, because the iteration costs 11 to 20 times
the one-sweep step while the lagged-versus-implicit split is free (266 against 270 iterations). No default was
changed: `:implicit` is already the default, it is inert at `maxpic = 1`, and making it effective would change
`maxpic` and the bitwise default path.

---

## 5. Benchmark validation

### 5.1 The constitutive card against ASPECT

ASPECT 2.5.0-pre (`gassmoeller/aspect:latest`, commit `340c0d018`, deal.II 9.4.0) was compared in two stages.
Stage 1 is a 495-point card (`T` in `{0, 0.1,..., 1}`, `eps_II` over nine decades, `z` in `{0, 0.25, 0.5, 0.75,
1}`) with each side's algorithm implemented independently, over four parameter sets (diffusion, mixed,
dislocation, ASPECT's defaults). Stage 2 runs the real binary on a state whose strain rate is analytic: a 1 x 1
box at uniform `T = 1500 K`, zero gravity, and `u = (2e-6 y, 0)` on all four boundaries, so `eps_II = 1e-6 1/s`
everywhere and the viscosity is uniform.

The branch laws agree to `6.166e-15` at worst over the card (the four sets give `6.166e-15`, `5.473e-15`,
`3.722e-15` and `2.872e-14`, the last being round-off at exponents of order 400). The full algorithm, against an
independent bisection implementation of ASPECT's Newton-plus-fallback, agrees to `6.6e-15` to `8.9e-15` on the
interior points of all four sets, while the legacy rule's offset there is 28 to 30 percent. Stage 2 pins the
absolute values: ASPECT returns `2.957782e20 Pa s` at `r = 0.3` and `7.053197e20 Pa s` at `r = 1`; the isostress
law reproduces both to `1.3e-7` and `2.6e-8`, and the legacy law returns `1/1.281706` and `1/1.410639` of them.

The card separates the error terms by perturbing one at a time; every perturbation gives a relative error above 1,
so none would have slipped through. ASPECT's own artefacts are separate: its `1e-22` strain-rate residual
tolerance is absolute, in the mixed regime its Newton can overshoot to a negative stress and the clamp-dominated
fallback cannot leave within the default iteration cap (both controlled runs needed `Maximum strain rate ratio
iterations = 400`), and its fallback clips each branch before combining, returning `eta_max/2 = 5.000000e27` where
the Newton path and this code return `1.000000e28`. The Stage-2 runs take the Newton path, so that row rests on
the source port.

### 5.2 The Tosi benchmark checks

The viscoplastic law is validated on the Tosi et al. (2015) community benchmark. Case 1 (temperature-dependent
viscosity, `Ra = 100`, free slip, `eta in [1e-6, 10]`, `eta_T = 1e5`) at 64 x 64 gives, against Table 2 of the
paper (eleven codes): mean temperature `0.7760` (spread 0.7758 to 0.7768), `Nu_top` `3.4246` (3.4091 to 3.4305),
`Nu_bot` `3.4255` by CBF (3.3130 to 3.4259), `u_RMS` `249.77` (248.93 to 252.09), surface RMS and maximum
`1.8696`/`2.6067` (1.8298 to 1.8999, 2.5516 to 2.6477), `⟨W⟩ = 2.4246` and `⟨U⟩/Ra = 2.4245` (2.369 to 2.4583,
2.3919 to 2.4333), with the energy balance `|RaW - U|/max = 3.6e-5`. All eight diagnostics are inside the spread
and within 0.6 percent of ASPECT's 64 x 64 column.

Two caveats are part of the result. The comparison uses the paper's convention of minimum and maximum over a full
period, not a steady value; and the wall-derivative `Nu_bot` at that resolution is `3.5410`, outside the spread
and 3.7 percent above ASPECT's `3.4142`, while the CBF and energy values are inside, which is the wall-functional
issue of §5.4. Case 5a (viscoplastic, periodic) is inside the spread on the period (`0.0790` against `0.0760` to
`0.0792`), on `⟨T⟩` (`0.6505/0.6685`), on `Nu_top` (`2.6975/7.3533`) and on `u_RMS` (`41.35/95.63`), with the
dissipation integral within 5 percent.

### 5.3 What each gate pins

The gates that pin this physics, with their assertion counts: `t_g192` (148) the legacy composite law; `t_g194`
(102) the latent capacity, its four derivatives, the two budget closures and the guard; `t_g195` (2791) the
depth-dependent Drucker-Prager profiles and tangents; `t_g196` (73) strain weakening, its analytic transient and
its Newton blocks; `t_g197` (43) TALA self-consistency and the ALA/TALA band; `t_g198` (81) the EOS reference
state and its tabulation order; `t_g199` (27) the latent time order; `t_g200` (39) the constitutive card and the
ASPECT mapping; `t_g201` (202) the isostress default, its tangents, its clamp and the 495-point card; `t_g206`
(68) the per-transition sign and the two latent modes.

### 5.4 The energy budget as a validation instrument

Three wall fluxes are related by exact identities of the assembled operator. **A**, the pointwise GLL derivative
of the discrete solution (`semquad_nu_walls`), is the physical conductive flux with truncation error `-(h^2/12)
theta'''`, measured at `3e-5` to `9e-5` relative. **B**, the steady consistent boundary flux (`semquad_nu_cbf`),
is a reaction whose wall functional is first order: it differs from A by exactly `-(2/(3h))(theta_0 - 2 theta_1) =
-(h/3) theta''_wall`, verified exactly on `theta = u^2` and `u^3` and to `1e-6` on measured states. **C**, the
time-step flux `Qc`, closes (ID1) to the solver floor, 0.05 to 0.43 percent of the budget scale.

The one-step form closes the latent case to `1.6e-7` relative, halving with `dt`, and the latent-free scheme to
round-off; the multi-step form against the physical flux closes to `1.4e-2` at `dt = 1e-3` with or without latent
heat (ratio 1.01 to 1.10 over five `dt`) and does not improve with `dt`, because the pointwise derivative carries
a wall-quadrature term that vanishes with the mesh (`5.14e-2` at 16 leaves, `1.43e-2` at 64, `4.86e-3` at 256).
`docs/VALIDATION.md` draws the consequence: the pointwise value is the truthful flux (error `O(h^2)`, invariant
under wall refinement at a fixed state), the reaction is the truthful boundary term of the discrete budget, and at
the late production states they differ by 0.3 to 1.2 percent of `Nu_top`. On a uniform mesh whose bottom layer is
under-resolved (`h = 1/16`) they differ by 433 percent with opposite signs while the pointwise value is 10 percent
off its converged value; at `h = 1/64` they agree to 0.0095 percent and converge with order 4.3 between the two
resolved meshes. The 17 percent gap reported at an earlier state was not reproducible at the late E states (spread
1 percent). Quote the pointwise `Nu_top`/`Nu_bot` and use the reaction as a cross-check on a resolved wall.

---

## 6. Known limitations and negative results

### 6.1 The latent sign bug

The latent heat was first implemented with one positive coefficient for every transition (`:global`, `latent_L >=
0` validated in `Params`). That is right for an endothermic transition and wrong for an exothermic one, and the
410 is exothermic. The bug survived because the code had no way to state the sign: a single positive number cannot
carry two transitions of opposite Clapeyron slope, and the old `0.11` was a compromise between `+0.059` and
`-0.131`. It was found by comparing against ASPECT, which heats a downwelling at the 410 where our code cooled it.
The fix is one coefficient per transition, `latent_mode = :per_phase`, with `L_i = -T_i dS_A,i/(cp dT)`. The
`:global` path is kept bitwise because checkpoints written before the change store it, and its measurements are
the ones quoted as legacy. The lesson is that a coefficient validated only as `>= 0` encodes no physics, and that
the cheapest possible physics check, the *direction* of the anomaly at a transition, would have caught it
immediately.

### 6.2 `T = Tad` is not a steady state

A static adiabat does not solve the diffusive energy equation, a statement that is measured rather than argued.
The exponential adiabat has `lap(Tad) = (alpha g/cp)^2 Tad > 0`, so with `u = 0` the implemented equation is
`dtheta/dt = lap(theta) + lap(Tad)` and `theta = 0` heats up. Over ten steps at `dt = 2e-5, 1e-5, 5e-6` the drift
is `1.1044e-4, 5.5582e-5, 2.7883e-5` at 16 leaves (ratios 1.986, 1.993, first order in `dt`), and its volume
integral is 93.4 to 98.1 percent of `10 dt int lap(Tad)`, the deficit shrinking with `h` (6.6 to 1.9 percent,
order about 0.9). The pointwise deviation is 5 to 19 percent and concentrated at wall-adjacent nodes. The
**conductive** profile `T = 1-z` is steady, and that is the control: `max|drift| = 4.2333e-8, 1.1184e-8,
1.9874e-9` at 16, 64, 256 leaves (orders 1.92, 2.49), so the discrete steady state has residual `O(h^2)` from the
mismatch between the analytic `lap(Tad)` quadrature and the stiffness applied to the interpolated `Tad`. The
reference state is a reference for buoyancy, mass and base advection, not a solution of the energy equation.

### 6.3 No resolved strain localization

The weakening machinery runs, converges and restarts bitwise, and produces no shear band at the parameters tested.
With a strong lid (`deta_T = 1e5`, `eta_max = 1e4`, `Ra = 1e4`) and an asymmetric seed, at fixed `dt = 2e-5` and
`t = 2e-3`, the plastic strain-rate concentration `max/mean` is `8.78` without weakening and `10.84`, `11.60`,
`8.46` for (`weak_f`, `eps_ref`) = (0.1, 2), (0.5, 2), (0.1, 5). That is a 23 to 32 percent increase for the two
`eps_ref = 2` cases, but the area fraction of cells above three times the mean is 3 to 6 percent with or without
weakening, so there is no narrow band. Over 150 to 200 steps the `chi` coefficient of variation is 0.10 to 0.25
and its correlation with the plastic strain rate is `-0.03` to `0.13`: the variation is grid-scale rather than
aligned with the deformation. The likely cause is numerical and measured: the `chi` equation has no diffusion and
is advected by unstabilised Galerkin at a cell Peclet number of order `1e5`, so a resolved shear-band experiment
needs upwind or entropy-viscosity-stabilised `chi` advection, which is not implemented. This is a negative result
about the experiment, not about the weakening law, which passes its analytic tests. A second result from the same
feature: the yielding indicator is a step function of `chi`, so the coupled system is only piecewise smooth and
has more than one root. At strong weakening (`weak_f = 0.2`, `eps_ref = 0.02`) the Picard and Newton fixed points
differ by `1e-1` in `chi` while both satisfy their criteria to `1e-12`; in the weak-coupling regime they agree to
`1.2e-3`. The Newton's Jacobian lags the indicator derivative, which is zero almost everywhere, so a better
Jacobian does not remove this.

### 6.4 The ASPECT 2.5 TALA absence

There is no cross-code TALA comparison, and that is a measured statement. Two configurations of the same 1 x 1 box
(`Ra = 1e6`, 64 x 64, free slip, fixed 1/0 temperatures, `simple compressible` with reference compressibility 0.1)
were attempted on ASPECT 2.5.0-pre. The ALA configuration runs (3 time steps in 17 s, RMS velocity `1.01e-5 m/s` =
202 in this code's units with `kappa = 5e-8`). The TALA-equivalent configuration (`Formulation = custom`, `Mass
conservation = incompressible`, `Temperature equation = reference density profile`) is rejected before the first
time step:

```
ASPECT detected an inconsistency in the provided input file. The mass
conservation equation was selected to be incompressible, but the provided
material model reports that it is compressible.
(helper_functions.cc:1997, material_model->is_compressible() == false)
```

`Formulation` accepts only `isentropic compression | custom | anelastic liquid approximation | Boussinesq
approximation`, and `anelastic liquid approximation` hard-wires `mass_conservation = reference_density_profile`;
the manual's TALA is that formulation with a pressure-independent material model. So the continuity equation and
the buoyancy are coupled through the material model's `is_compressible()`, and the combination "same reference
density, TALA continuity" is not expressible: a TALA run needs a pressure-independent density, which removes the
depth-dependent reference density the ALA run has, and the difference would confound the mass form with the
buoyancy. That is exactly the contrast `mass_form = :boussinesq | :anelastic` isolates here, and it has no ASPECT
counterpart at this version.

### 6.5 The rest

* **`:composite` cannot carry the plastic branch.** `Params` throws for `:composite` with any yield parameter; the
  plastic branch works with the temperature-dependent laws and with `:tosi`. Grain size is folded into `comp_Ad`
  with no grain-size evolution, and `comp_Vd(1-z)` is exact only for a linear lithostatic pressure. `tau_y` uses
  the dynamic `p` only, and the weakening source lags the velocity by one Picard sweep.

* **The EOS is not thermodynamically closed.** `alpha(y)` and `cp(y)` are prescribed and only the density relation
  is integrated; there is no `alpha(rho,T)`, no Grueneisen parameter and no phase-boundary thermodynamics.
  Adams-Williamson ties `rho` and `Tad` to one `Lambda`; Murnaghan decouples them through `K_S(p)` but is
  exercised only on its ODE identities and a finite smoke run, with `K0, K0'` not calibrated to a seismic model.
  There is no ASPECT comparison for the EOS either: its `ascii reference profile` reads a table produced
  elsewhere. `+lap(Tad)` is the ODE curvature, not the interpolant's: for constant `alpha` the two coincide to
  `1e-11` at `nz = 129`.

* **The convective energy budget does not close** with the pointwise flux: at step 5 of the `Ra = 1e6` composite
  case `lhs = 0.3208` against `flux - base_adv + lapint = 0.1462` (54 percent), and adding `int theta div u =
  0.4012` overshoots at -41 percent. That is the wall-flux diagnostic of §5.4 on a 5-step transient whose boundary
  layer is unresolved, not an EOS term. The `Ra = 1e6` composite-plus-phase case is itself a runaway: at `dt =
  2.5e-5` the one-sweep run reaches `umax = 1.7e12` and `max|theta| = 1e3` within 80 steps, and the iterated
  implicit run throws a singular energy solve in the same window, which is why the 2-D latent-order study uses `Ra
  = 1e5`. The old 20-step entry `Vrms 6.99e1 -> 6.00e3` is not recoverable: at 20 steps `maxpic = 1` gives ALA
  `3.1487e2` / TALA `4.1502e2` (legacy) and `1.0340e2` / `1.2412e2` (current), and `maxpic = 20` gives `1.5406e3`
  / `2.0505e3` and `1.8166e2` / `2.5976e2`, with no recorded `maxpic`/`dt`.

* **The latent heat costs 39 percent of the energy assembly** (0.391 ms to 0.543 ms on the 64-leaf tree), and its
  Newton tangent costs 13 to 16 percent of the Newton's wall time for a block that is `5.3e-6` of the Jacobian
  norm at moderate amplitude. It is kept because it is exact, not because the Newton fails without it. The `:tosi`
  guard's evidence is stale: the claim that `:tosi` with `maxpic > 1` diverges, and that the lagged `maxpic = 1`
  run differs by 5 to 8 percent, was measured on the defect of §1.5, so the guard stays but its quantitative basis
  is void.
