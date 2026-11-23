<p align="right"><a href="../en/guide.md">English</a> · <a href="../zh/guide.md">中文</a></p>

# Guide: the realistic 2-D mantle-convection case

This page documents the case that `examples/case_realistic_2d.jl` builds: the physics, the numerical method, every knob, every output file, restarting, and the thread and memory behavior. Every parameter value quoted here is either a literal in that script or a default of `run_semquad_simulation` in `src/driver/run.jl`; section 5 lists them all.

The problem is a 2.5 x 1 box of incompressible fluid with free-slip walls, temperature 0 at the top and 1 at the bottom, internal heating `H = 10`, a nominal Rayleigh number `Ra = 1e5`, composite diffusion–dislocation rheology, and two phase transitions with latent heat. Yielding and strain weakening are off.

<details>
<summary><b>Contents</b></summary>

- [1. The physical problem](#1-the-physical-problem)
- [2. The discretization](#2-the-discretization)
- [3. The adaptive policy](#3-the-adaptive-policy)
- [4. Linear solvers and the `:auto` rule](#4-linear-solvers-and-the-auto-rule)
- [5. Parameter reference](#5-parameter-reference)
- [6. Output formats](#6-output-formats)
- [7. Restarting and the configuration guard](#7-restarting-and-the-configuration-guard)
- [8. Threads and memory](#8-threads-and-memory)
- [9. Where to read next](#9-where-to-read-next)

</details>

## 1. The physical problem

### 1.1 Domain and boundary conditions

The domain is $[0,2.5]\times[0,1]$ , built from 5 x 2 base cells that the quadtree subdivides into square leaves. $x$ is horizontal and $y$ is height, with $y=0$ at the bottom and $y=1$ at the top; depth is $1-y$ .

All four walls are free slip: no normal flow, unrestricted tangential flow. The normal component is fixed,

$$
u_x = 0 \quad (x=0,\,2.5),
\qquad
u_y = 0 \quad (y=0,\,1),
$$

where $u\_x$ and $u\_y$ are the two velocity components. Temperature is $T=0$ at the top wall and $T=1$ at the bottom wall; the side walls are insulating.

The condition can be set per wall. `bc = :free` (the default) applies free slip to all four; `bc = :rigid` applies no slip to all four, which fixes the tangential component as well and takes it from the wall-velocity closures $u\_{Dx}(x,y)$ , $u\_{Dy}(x,y)$ . `wall_bc` overrides one wall at a time, in the order `(xminus, xplus, yminus, yplus)`, with entries `:same` (follow `bc`), `:free`, `:rigid` or `:open`. The card exposes the tuple as `CASE_WALL_BC` (section 5.1). An `:open` wall has **no** Dirichlet component on either component: the natural traction condition applies there, the optional data are the per-wall tractions of the two paragraphs below, and it changes the pressure gauge (see below).

The wall-velocity closures are functions of position, and the four walls sit at different $y$ (or $x$). A spatially varying wall velocity therefore needs no extra API: $u\_{Dx}=Uy$ with `bc = :rigid` drives a lid at $U$ over a resting floor and reproduces the Couette solution $u=(Uy,0)$ to $10^{-10}$ . The only assignment a single closure cannot express is one that disagrees at a corner, and continuity forbids a disagreement there anyway.

The *normal* component is set per wall by `Params.wall_un` (the solve takes it as the `wall_un` keyword), in the same `(xminus, xplus, yminus, yplus)` order. An entry `(x,y) -> value` prescribes that wall's normal velocity as the value of the wall's normal component in the positive coordinate direction -- $u\_x$ on a vertical wall and $u\_y$ on a horizontal one, exactly the convention of the closures above and not the outward flux $u\cdot n$ -- and `nothing` (the default) falls back to the historical closure, which the driver sets to zero, so the default card is unchanged. The normal component is Dirichlet on a `:free` and on a `:rigid` wall; an `:open` wall has no Dirichlet component, so a value there is an error. While every wall keeps a Dirichlet normal component, the prescribed data must satisfy $\int\_{\partial\Omega}u\cdot n\,\mathrm dS=0$ ; an unbalanced tuple (for example a through-flow prescribed on one vertical wall only) is refused with the measured flux instead of solving a discrete problem whose continuity equations are inconsistent. With an `:open` wall the free normal velocity absorbs the imbalance by construction and the net-flux check is skipped. The same Couette solution driven by `wall_un` on the side walls and `wall_u` on the lid is exact to $2\times10^{-14}$ (section 5.3; gate G258). A prescribed tangential value (`wall_u`) is only meaningful on a `:rigid` wall; on a `:free` wall the tangential component is traction-free and on an `:open` wall there is no Dirichlet component at all, so the value would be silently ignored and is an error. At a corner the walls are applied in the order above and the later wall owns the shared component: a rigid top or bottom wall's tangential value overrides the side walls' normal value at its two corners, which is what makes the lid-driven cavity's corners belong to the lid.

The tangential *traction* is set per wall by `Params.wall_t` (the solve takes it as the `wall_t` keyword), in the same `(xminus, xplus, yminus, yplus)` order. An entry `(x,y) -> value` prescribes the tangential component of the **total** Cauchy traction $t=\sigma\cdot n$ as a boundary load: integration by parts of the momentum equation puts $\int\_\Gamma t\cdot v\,\mathrm dS$ on the right-hand side with a plus sign, and the code adds it to the unreduced load vector, so the constraint matrix is what projects it onto the admissible test space -- it is data on the free side, never a Dirichlet value. The value is the **global component in the positive coordinate direction**: $t\_x$ on a horizontal wall and $t\_z$ on a vertical one, exactly the convention of `wall_un` above and not an outward-normal component. On an axis-aligned wall the tangential traction is pressure-free ($t\_x=\sigma\_{xy}$, $t\_z=\sigma\_{zy}$), so this is the exact $2\eta\,\varepsilon(u)\cdot n$ shear traction. Because it only exists where the tangential component is free, `wall_t` is accepted on a `:free` or `:open` wall and refused on a `:rigid` one (the tangential velocity is Dirichlet there and the load would be silently dropped by the constraint). `nothing` is zero traction, so the default card is unchanged; a traction-driven Couette flow $u=(t\_0y/\eta,0)$, $p=0$ is exact to $10^{-10}$ (gate G259).

The normal *traction* is set per wall by `Params.wall_tn` (the solve takes it as the `wall_tn` keyword, the card as `CASE_WALL_TN`), in the same order and with the same load convention -- the normal component of the full Cauchy traction $t=\sigma\cdot n$, as the global component in the positive coordinate direction ($t\_x$ on a vertical wall, $t\_z$ on a horizontal one). On an axis-aligned wall it is the pressure-bearing component, $t\_n=-p+2\eta\,\varepsilon\_{nn}$ in the wall's normal direction, and it is the data term that fixes the pressure level. Because $t=\sigma\cdot n$ uses the **outward** normal, a pressure $p\_0$ imposed on an open boundary is `wall_tn = -p0` on the **top** wall and `wall_tn = +p0` on the **bottom** wall; the prescribed value is the *dynamic* pressure traction in the solver's nondimensional pressure unit, not a lithostatic one. It is accepted only on an `:open` wall (the only wall with a free normal velocity); on a `:free` or `:rigid` wall the normal velocity is Dirichlet and the load would be silently dropped, so that is an error. Two consequences of an open wall:
- **The pressure gauge is derived.** The nodal pressure pin is exact only while the normal velocity is Dirichlet on the whole boundary (then the constant pressure is in the null space of the reduced gradient, and the dropped continuity equation is a linear combination of the others). With an `:open` wall the solver therefore pins one pressure dof **iff no wall is `:open`**; a pinned open configuration would replace a continuity equation by $p\_a=0$ -- a different problem. The resolved decision is reported as `LAST_SEMSTOKES_PRESSURE_GAUGE` (`:pin`/`:none`).
- **The open configurations must be well posed.** Every `:open` wall must leave at least one x-wall and one y-wall with a Dirichlet normal component; otherwise a rigid translation (or, with all four walls open, the whole rigid motion) is left admissible and the discrete saddle is singular. All four walls `:open` is the pure traction problem -- it needs a velocity gauge and net force/torque-compatible data -- and is refused, as is any configuration whose 3x3 rigid-mode Gram matrix over the Dirichlet dofs is singular (the message names the admissible motion). `linear=:direct` (with its shift ladder), `linear=:mg` and `linear=:auto` are supported without the pin; `:schur_cg`/`:lagged` assume the closed-boundary gauge and are refused on an open configuration.

The gate numbers for the open-wall surface: a rigid box with the top `:open` and `wall_tn = T` gives $u=0$, $p=-T$ (the outward-normal sign), while the same $T$ on the **bottom** wall gives $p=+T$; the body-force hydrostatic solution with an open top and zero normal traction is $u=0$, $p=1-y$ -- all exact to $10^{-10}$ (G261). A traction-free open lid gives $u=p=0$ with the unpinned saddle and the hydrostatic free-slip/open configuration gives $p=1-y$ (G262). The legacy closed-wall paths are bit-identical to the stage-2 tree and the three linear paths agree on an open case (G263). Section 1.10 combines the knobs of this section into a regional inflow/outflow channel.

### 1.2 The equations

Nondimensional Boussinesq momentum, continuity and energy:

$$
-\nabla\cdot\left(2\eta\,\boldsymbol\varepsilon(\mathbf u)\right) + \nabla p = \mathbf f,
\qquad
\nabla\cdot\mathbf u = 0,
$$

$$
\frac{\partial\theta}{\partial t} + \mathbf u\cdot\nabla\theta
= \nabla^2\theta - u_y\frac{\mathrm d\bar T}{\mathrm dy},
\qquad
f_y = \mathrm{Ra}\,\theta - \mathrm{Ra}\sum_i B_i\left[\Gamma_i(y,T)-\Gamma_i(y,\bar T)\right],
\qquad
f_x = 0.
$$

| Symbol | Meaning |
|---|---|
| $\mathbf u=(u\_x,u\_y)$ | Velocity. |
| $p$ | Dynamic pressure. |
| $\eta$ | Viscosity. |
| $\boldsymbol\varepsilon(\mathbf u)$ | Strain-rate tensor $\left(\nabla\mathbf u+\nabla\mathbf u^{\mathsf T}\right)/2$ . |
| $\theta=T-\bar T$ | Temperature anomaly. |
| $\bar T(y)$ | Conductive base state of section 1.3. |
| $\Gamma\_i$ | Phase fraction of transition $i$ , section 1.5. |
| $B\_i$ | Buoyancy number of that transition, positive for a dense phase, so excess phase fraction pushes material down. |
| $\mathrm{Ra}$ | Rayleigh number, `1e5` on this case. |
| $\mathbf f=(0,f\_y)$ | Body force, vertical only. |

`Ra = 1e5` multiplies the thermal buoyancy, with viscosity measured in units of the reference viscosity `eta_ref` (section 1.9). The prefactors of section 1.4 put $\eta=1$ at $T=0.5$ , a temperature that sits inside the cold lid, so the convecting interior runs near $T\approx1.3$ where $\eta$ is between $10^{-3}$ and $10^{-2}$ . In internal Rayleigh-number terms that is $10^7$ to $10^8$ , and one overturn takes about $10^{-3}$ in nondimensional time.

### 1.3 Base state, heating and Nusselt numbers

$H$ is the volumetric heating in $\partial T/\partial t + \mathbf u\cdot\nabla T = \nabla^2 T + H$ . The case evolves the anomaly about the conductive profile that carries that heating:

$$
\bar T(y) = (1-y) + \frac H2\,y(1-y),
\qquad
\nabla^2\bar T = -H,
\qquad
\frac{\mathrm d\bar T}{\mathrm dy} = -1 + \frac H2(1-2y).
$$

Substituting $T=\bar T+\theta$ cancels the source against $\nabla^2\bar T$ and leaves only $-u\_y\,\mathrm d\bar T/\mathrm dy$ , so no explicit heating term appears in the assembly. At $H=10$ , $\bar T(0.5)=1.75$ , the volume mean of the base state is $0.5+H/12=1.3333$ , and the conductive wall gradients are $-6$ at the top and $+4$ at the bottom. The two Nusselt numbers use those gradients as their reference:

$$
\mathrm{Nu}_{top} = 1 + \frac H2 - \left\langle\frac{\partial\theta}{\partial y}\right\rangle_{top},
\qquad
\mathrm{Nu}_{bot} = 1 - \frac H2 - \left\langle\frac{\partial\theta}{\partial y}\right\rangle_{bot}.
$$

The angle brackets average over $x$ , and the derivative is the one-sided GLL derivative of $\theta$ at the wall. In a purely conductive state the two values are 6 and -4. A steady convecting state satisfies

$$
\mathrm{Nu}_{top} - \mathrm{Nu}_{bot} = H;
$$

how far the run sits from that identity is the `closure` column, described in section 6.2.

### 1.4 Rheology

The case sets `eta_law = :composite` with `comp_rule = :isostress`, the combination ASPECT uses. Two creep mechanisms deform the material at once. At temperature $T$ , height $y$ and strain-rate invariant $\dot\varepsilon$ , their inverse viscosities are

$$
\eta_{\mathrm{diff}}^{-1}
= A_d\exp\left(-\frac{E_d + V_d(1-y)}{T_{nd}}\right),
\qquad
\eta_{\mathrm{disl}}^{-1}
= A_s\exp\left(-\frac{E_s + V_s(1-y)}{n\,T_{nd}}\right)\dot\varepsilon^{\frac{n-1}{n}}.
$$

| Symbol | Meaning |
|---|---|
| $\eta\_{\mathrm{diff}}$ , $\eta\_{\mathrm{disl}}$ | Diffusion-creep and dislocation-creep viscosities. |
| $T\_{nd}=\max(T+\eta\_{T0},\,T\_{\mathrm{cut}})$ | Absolute temperature in units of the temperature drop; `eta_T0 = 0.5`, `comp_Tcut = 0` (off). |
| $n$ | Dislocation stress exponent, `3` here. |
| $A\_d$ , $A\_s$ | Prefactors of the two mechanisms, in units of $1/\eta\_0$ . |
| $E\_d$ , $E\_s$ | Activation energies divided by $R\Delta T$ . |
| $V\_d$ , $V\_s$ | Activation volumes as depth factors; both are zero on this case. |
| $\dot\varepsilon$ | $\max(\varepsilon\_{II},\dot\varepsilon\_{\min})$ with $\varepsilon\_{II}=\sqrt{\tfrac12\,\boldsymbol\varepsilon:\boldsymbol\varepsilon}$ and `comp_eps_min = 1e-12`. |

The isostress rule gives both mechanisms the same stress, so the strain rate splits between them:

$$
\frac{\tau}{2\eta_{\mathrm{diff}}} + A_s\exp\left(-\frac{E_s+V_s(1-y)}{n\,T_{nd}}\right)\tau^n = \dot\varepsilon,
\qquad
\eta = \frac{\tau}{2\dot\varepsilon},
$$

with $\tau$ the deviatoric stress. The code solves the equivalent scalar equation for $x=\eta/\eta\_{\mathrm{disl}}$ :

$$
k\,x + x^n = 1,
\qquad
k = \frac{\eta_{\mathrm{disl}}}{\eta_{\mathrm{diff}}},
\qquad
\eta = x\,\eta_{\mathrm{disl}}.
$$

Here $k$ is the ratio of the two branch viscosities at the total strain rate, and $x\in(0,1]$ is the fraction of the effective viscosity carried by the dislocation branch. The root is unique for $k>0$ ; $k\to0$ is the dislocation-only limit and $k\to\infty$ the diffusion-only one. The solve is Newton with a bisection bracket on $[0,1]$ , a tolerance of `1e-15` on $kx+x^n-1$ and a cap of 100 iterations; the worst case measured over the 495-point constitutive card is 6. The case clips $\eta$ once, after the combination, to $[10^{-4},10^4]$ . That clip is a solver safeguard, and it sets the conditioning; the measured viscosity contrast is about $10^7$ .

The constants are $A\_d=1.6240183119\times10^5$ , $A\_s=2.6796419238\times10^{-1}$ , $E\_d=12$ , $E\_s=17$ , $n=3$ and $\eta\_{T0}=0.5$ . They satisfy two conditions at $y=0.5$ : $\eta(T=0.5,\dot\varepsilon\_{\mathrm{ref}})=1$ fixes the common scale, and $\eta(T=0.1)/\eta(T=0.9)=10^4$ at $\dot\varepsilon\_{\mathrm{ref}}$ fixes the branch ratio. The reference strain rate $\dot\varepsilon\_{\mathrm{ref}}=1.65\times10^3$ comes from a constant-viscosity precursor of the same case.

The library also carries simpler laws, none of which this case uses. Note that `Params` accepts `eta_law = :const` while `run_semquad_simulation` rejects it; use `:expT` with `deta_T = 1` for constant viscosity.

| `eta_law` | Viscosity | Parameters |
|---|---|---|
| `:expT` | $\eta = \exp[-\ln(\Delta\eta\_T)\,T]$ | `deta_T` is the viscosity ratio: $\eta(0)=1$ , $\eta(1)=1/\Delta\eta\_T$ . |
| `:expTz` | $\eta = \exp(-bT + c(1-y))$ | `eta_b` = b, `eta_c` = c. The Blankenbach 2b example sets both to the ASPECT values. |
| `:arrhenius` | $\eta = \exp\{E\,[1/(T+T\_0) - 1/(1+T\_0)]\}$ | `eta_E` = E, `eta_T0` = T_0. |
| `:composite` | The two equations above. | `comp_*`, `eta_T0`, `comp_rule`. |
| `:tosi` | The Tosi viscoplastic law in absolute units. | `tosi = TosiLaw(...)`, which carries its own plastic branch. |

### 1.5 The two phase transitions

With $y$ the height, the phase fraction of transition $i$ is

$$
\Gamma_i(y,T) = \frac12\left[
1 + \tanh\left(
\frac{(z_{0,i}-y) - s_i\,(T-T_{\mathrm{ref},i})}{w_i}
\right)
\right].
$$

| Symbol | Meaning |
|---|---|
| $z\_{0,i}$ | Reference height of the transition. |
| $w\_i$ | Transition width in height. |
| $s\_i = \mathrm dz/\mathrm dT$ | Clapeyron slope, as a height shift per unit temperature. |
| $T\_{\mathrm{ref},i}$ | Reference temperature. |

A hot upwelling pushes the transition down where the slope is positive and up where it is negative.

| | 410 | 660 |
|---|---|---|
| `z0` | 0.87 | 0.78 |
| `width` | 0.025 | 0.025 |
| `slope` | +0.032 | -0.032 |
| `Tref` | 0.5 | 0.5 |
| `drho` (kg/m³) | 180 | 300 |
| `B` | 1.8182 | 3.0303 |

The buoyancy number is

$$
B = \frac{\Delta\rho}{\rho_0\alpha_0\Delta T},
$$

with $\rho\_0=3300$ kg/m³ the reference density, $\alpha\_0=3\times10^{-5}$ 1/K the thermal expansivity, and $\Delta T=1000$ K the temperature drop. An upwelling gains buoyancy at the 410 and loses it at the 660, which is the behavior of an exothermic and an endothermic transition.

### 1.6 Latent heat

Each coefficient is derived from its own transition:

$$
\gamma_i = \frac{s_i\,\rho_0\,g\,d}{\Delta T},
\qquad
\Delta S_{A,i} = \frac{\gamma_i\,\Delta\rho_i}{\rho_0^2},
\qquad
\lambda_i = -\frac{T_{\mathrm{abs},i}\,\Delta S_{A,i}}{c_p}.
$$

| Symbol | Meaning |
|---|---|
| $\gamma\_i$ | Clapeyron slope in Pa/K. |
| $\Delta S\_{A,i}$ | Entropy change of the transition, J/(kg K). |
| $\lambda\_i$ | Latent coefficient in the energy equation, nondimensional. |
| $T\_{\mathrm{abs},i}$ | Base-state temperature at the transition, in units of $\Delta T$ , including the surface offset. |
| $\rho\_0$, $c\_p$, $g$, $d$, $\Delta T$ | Scale set: 3300 kg/m³, 1000 J/(kg K), 9.81 m/s², $2.9\times10^6$ m, 1000 K. |

The minus sign is this code's convention. It writes the latent term as $-\lambda\_i\,D\Gamma\_i/Dt$ , where ASPECT writes $+T\,\Delta S\_A\,D\Gamma\_i/Dt$ .

| Transition | `dS_A` | `latent` | Effect |
|---|---|---|---|
| 410 | +49.66 J/(kg K) | -0.059364 | Heats a downwelling. |
| 660 | -82.76 J/(kg K) | +0.130596 | Cools a downwelling. |

The energy equation carries them as an apparent heat capacity plus an advective source:

$$
\left(1+\sum_i\lambda_i\frac{\partial\Gamma_i}{\partial T}\right)
\left(\frac{\partial\theta}{\partial t} + \mathbf u\cdot\nabla\theta\right)
\;=\;
\nabla^2\theta \;-\; u_y\frac{\mathrm d\bar T}{\mathrm dy} \;-\; \sum_i\lambda_i\frac{\partial\Gamma_i}{\partial y}u_y.
$$

The capacity factor stays inside $[0.88,1.12]$ . The sum of the two coefficient magnitudes is 0.18996, and $\left|\partial\Gamma/\partial T\right|$ is bounded by $|s\_i|/(2w\_i)=0.64$ , which gives 0.122. The assembly throws if the factor would go negative, since that would reverse the energy equation in time. `CASE_LATENT=global` reproduces the historical runs that used a single `latent_L = 0.11` for both transitions; that value has the wrong sign at the 410, so it exists only to replay old results.

### 1.7 Initial condition

$$
\theta_0(x,y) = 0.05\sin(\pi y)\cos\left(\frac{2\pi x}{L_x}\right) + 0.02\sin(2\pi y)\cos\left(\frac{4\pi x}{L_x}\right),
$$

where $L\_x=2.5$ is the box width. It vanishes on the top and bottom walls, as an anomaly must, and is symmetric about $x=L\_x/2$ . Nothing is random: the card, the seed and a fixed step count make a run reproducible.

Yielding and strain weakening are off (`yield_c = yield_taumax = Inf`, `yield_phi = 0`, `weak_eps_ref = 0`), as are density-weighted heating (`rho_heating = rho_cp = false`), dissipation and adiabatic heating (`Di = 0`), the reference state (`refstate = nothing`), the deviatoric projection (`deviatoric = false`), and the anelastic mass form (`mass_form = :boussinesq`).

### 1.8 Viscoelasticity

Maxwell viscoelasticity is off by default. `Params.shear_modulus` is the nondimensional elastic shear modulus $G$ , and the case exposes it as `CASE_SHEAR_MODULUS` (unset or empty means 0, i.e. off). A positive value does two things: it carries a deviatoric stress history, and it replaces the viscosity the flow solve sees by the effective viscosity

$$
\eta_{\mathrm{eff}} = \frac{1}{1/\eta + 1/(G\,\Delta t)},
\qquad
\tau^{n+1} = \frac{\tau^{n} + 2G\,\Delta t\,\hat{\boldsymbol\varepsilon}^{\,n+1}}{1 + \Delta t/\tau_{\mathrm M}},
\qquad
\tau_{\mathrm M} = \eta/G .
$$

Here $\tau$ is the deviatoric stress, $\hat{\boldsymbol\varepsilon}$ its deviatoric strain rate, $\tau\_{\mathrm M}$ the Maxwell time, and $\Delta t$ the step the run actually took. The previous step's stress enters the momentum equation as the lagged load $\nabla\cdot(a\tau^n)$ with $a=\eta/(\eta+G\Delta t)$ , which is what makes an equilibrated viscoelastic flow the *material*-viscosity flow. With `shear_modulus = 0` every one of these terms is absent, nothing is sampled, and the run is bit-identical to the purely viscous one. The value is part of `run_config`, so a restart cannot switch it silently.

The two limits are the whole behavior:

- $G\Delta t \gg \eta$ : the fast-relaxation regime, where the step is longer than the Maxwell time. The stress relaxes within a step, $\eta\_{\mathrm{eff}} \approx \eta$ , and the flow is the purely viscous one; the memory is invisible.
- $G\Delta t \ll \eta$ : the elastic regime, where $\Delta t \ll \tau\_{\mathrm M}$ . Here $\eta\_{\mathrm{eff}} \approx G\Delta t$ , so the viscosity is capped at the elastic value and the first steps' stress grows like $2G\Delta t\,\hat{\varepsilon}$ .

**The regime change.** That cap is correct physics, but it means a viscoelastic card is not its `G = 0` twin plus memory: every viscosity well above $G\Delta t$ is flattened toward $G\Delta t$ , so the stiff end of the material contrast is removed and the contrast the solve sees collapses by about the factor by which the stiffest material exceeds $G\Delta t$ . Two measured states:

- At the developed production state (step 20 000, $\Delta t = 4.182\times10^{-7}$ , $G\Delta t = 1.485$ ) the material viscosity contrast was $3.3\times10^6$ and the realized effective contrast about $5\times10^2$ : three to four decades of contrast removed, and a measured arm ran 33% cheaper because of it.
- At a start-up segment with $\Delta t = 3.6\times10^{-8}$ and $G\Delta t = 0.129$ the collapse was from a material $1.0\times10^7$ to an effective $1.3\times10^2$ .

So a viscoelastic card is a *softer, different* problem, and its diagnostics are not those of its `G = 0` twin. Whenever `shear_modulus > 0` the driver reports the regime once at the first step that advances the stress and once at the end of the run: the material viscosity range and contrast, the effective range and contrast at that step's `dt`, the range of $G\Delta t/\eta$ , and one line stating the resulting contrast collapse. At `shear_modulus = 0` it prints nothing. Register the expected effective contrast before reading a $G > 0$ arm.

**The cost.** The viscoelastic machinery itself is cheap. Measured on the production card, it costs about 1% of a step (36.2 ms: 33.5 ms resampling the material viscosity at the nodes, 2.65 ms the stress update, 0.08 ms the elastic load), and a checkpoint grows by exactly $3\times8\times N$ bytes for the three stress fields, 1.730 MB to 2.679 MB, that is 1.55x. The total step time can still fall, as the 33% above shows: that is not the price of the feature, it is the different problem.

**The co-rotation band.** The Jaumann rotation is applied as the linearization $1+\Delta t\,ad\_W$ over the step, not as the exponential. It is valid while $2|W|\min(\Delta t,\tau\_{\mathrm M})$ is small, and `MAXWELL_COROTATION_BOUND = 0.1` quantifies "small". The driver reports the run maxima of $2|W|\Delta t$ , $2\tau\_{\mathrm M}|W|$ and the per-step rotation once per run, and warns once at the first step outside the band; this report is separate from the regime report above. The run continues: it is an accuracy band, not a stability limit, so reduce `dt` (`dtmax`/`cfl`) to get back inside.

Both reports are switchable: set `regime_report` or `corotation_guard` (both `true` by default) to `false` to silence one for a run whose output must stay quiet; the run is otherwise unchanged.

The stress update uses the step the run actually took, and it is first-order consistent on variable step sequences, so there is no fixed elastic step to set: `dt` keeps following the CFL rule, and no extra option is needed.

**What the stress history is not.** It is a tensor state carried on the mesh, not a free surface and not topography. This code has no moving top boundary: the sticky-air density hook (`rho_multiplier`) lets a low-density top layer flow, but its interface stays where the Eulerian mesh puts it, so no surface relief emerges. A viscoelastic card is not a topographic observable.

### 1.9 The reference viscosity and the Rayleigh number

The nondimensional equations of section 1.2 contain a scaling choice that is easy to miss: `Ra` is defined with one reference viscosity, `eta_ref`,

$$
\mathrm{Ra} = \frac{\rho g \alpha \Delta T d^3}{\kappa\,\eta_{\mathrm{ref}}},
$$

and the flow solve uses `eta_law/eta_ref` wherever the material law's viscosity enters, through `eta_multiplier` and the Maxwell effective-viscosity closure of section 1.8. `eta_ref` is `Params.eta_ref` (default `1.0`), in the units the material law returns. The case exposes it as `CASE_ETA_REF` (section 5.1); the header prints it and the checkpoint metadata records it.

The convention is consistent only when a rescaled law is paired with the same rescaling of `eta_ref`. Multiplying `comp_Ad` and `comp_As` by `1/c` multiplies every viscosity the law returns by `c`; with `eta_ref = c` the flow is unchanged, because the ratio `eta/eta_ref` is unchanged, while with `eta_ref = 1` the same `Ra` describes a different physical Rayleigh number. The bounds are absolute: `eta_min` and `eta_max` are in the law's units and are applied before the normalization, so a law rescaled by `c` needs its bounds rescaled by `c` as well. The default `1.0` leaves every existing run bit-identical.

When `eta_ref != 1` the driver reports the convention, once at the first flow solve and once at the end of the run. The report gives `Ra`, `eta_ref`, the range and contrast of the normalized field `eta/eta_ref`, `eta_interior` (the arithmetic volume mean of `eta/eta_ref` over the interior band `0.25 <= z <= 0.75`, both wall layers excluded), `Ra_interior = Ra/eta_interior`, and, when the card supplies its scale set, `eta_0` in Pa s. `pic_mode = :newton` builds its viscosity from the raw law, so it refuses `eta_ref != 1` rather than solve a different problem silently.

The production card's default is `eta_ref = 1`, but its material law varies by orders of magnitude across the box, so the interior Rayleigh number is far above the nominal one. The report prints only for `eta_ref != 1`; the convention's numbers at the card's developed state are `eta_interior = 4.280635e-3`, `Ra_interior = 2.336102e7` and `eta_0 = 2.368635e23 Pa s`, the physical viscosity that `Ra` is defined with, which puts the physical interior viscosity at `1.013926e21 Pa s`. At the calibration's hot interior point (`T = 1.75`, `edot_ref`) the same convention gives `Ra = 7.84e7` and a viscosity of `3.02e20 Pa s`. The label `Ra = 1e5` therefore describes the calibration point, not the interior: quote both.

### 1.10 A regional channel model

The per-wall knobs of section 1.1 compose into a regional model: fluid enters through one wall, crosses the domain and leaves through an open wall, while the other walls are no slip or free slip. The shipped demonstration is `examples/channel_flow_2d.jl`, a plane channel whose exact solution is plane Poiseuille; this subsection is the recipe, and the semantics of each knob stay in section 1.1.

The channel card has this shape, in the wall order `(xminus, xplus, yminus, yplus)`:

| wall | `wall_bc` | data |
|---|---|---|
| upstream x-wall | `:rigid` | `wall_un` = the normal velocity $u\_x(y)$ of the inflow profile; `wall_u` = the tangential velocity (zero for a normal inflow) |
| downstream x-wall | `:open` | `wall_tn` = $-p\_{\mathrm{out}}$ (the dynamic pressure of the outflow) and `wall_t = \eta\,\mathrm du\_x/\mathrm dy` (the shear traction) |
| top and bottom | `:rigid` or `:free` | nothing, or `wall_un` = 0 for a free-slip wall |

The inflow wall must be `:rigid`: a `:free` wall has zero tangential traction and cannot support the shear $\eta\,\mathrm du\_x/\mathrm dy$ that a shear-carrying profile has at its entrance. Prescribing the same normal velocity on a `:free` wall gives a genuinely different flow, measured at 5.6% of the centreline velocity on the plane channel; the design's channel therefore always uses `:rigid` on the inflow, with `wall_u` carrying any tangential part. The velocity of a flow entering through a normal-velocity boundary is prescribed completely by `wall_un` and `wall_u`, and the interior sees it as the same Dirichlet data whichever of the two mechanisms writes the value (gate G264 compares the new `wall_un` route against the legacy `uDx` closure and finds the solutions bit-identical).

Three compatibility conditions make the combination well posed:

- **Mass.** Continuity fixes the flux. The integral of the inflow data is the flux the outflow must carry, and the discrete continuity equations enforce the balance. The net-flux guard of section 1.1 cannot double as the check here: with an open wall an inlet whose integral does not match is not an error, its imbalance simply moves the outflow velocity. The open wall's free normal velocity is what absorbs it, and that is exactly the regional model's outflow.
- **Pressure level.** The open wall removes the nodal pressure pin (section 1.1), so the pressure is not fixed by a pin but by the normal traction data: `wall_tn` **is** the dynamic pressure of the outflow, $p\_{\mathrm{out}}$, with the sign of the outward normal. It is a dynamic pressure, not a lithostatic one: the solver's `p` excludes the base state and the buoyancy subtracts it analytically, so a user who passes a hydrostatic value imposes that whole column as a traction.
- **Well-posedness.** At least one x-wall and one y-wall keep a Dirichlet normal component, as section 1.1 requires; the channel satisfies this with the rigid inflow, the two channel walls and the open outflow.

The exact solution of the channel with height $H$, length $L$, pressure gradient $G$ and constant viscosity $\eta$ is plane Poiseuille:

$$
u_x=\frac{G\,y\,(H-y)}{2\eta},\qquad u_y=0,\qquad p=p\_{\mathrm{out}}+G\,(L-x),
$$

so the total traction on the outflow, whose outward normal is $+e\_x$, is

$$
\sigma\cdot n=\left(-p,\ \eta\frac{\mathrm du\_x}{\mathrm dy}\right)
=\left(-p\_{\mathrm{out}},\ G\left(\frac H2-y\right)\right).
$$

That is `wall_tn = -p_out` and `wall_t = G(H/2 - y)`; the tangential part is eta-independent at constant eta. The same profile is also the exact solution for a constant body force $f\_x=G$ with no pressure gradient, in which case the pressure is constant, $p=p\_0$, and the outflow normal traction is $-p\_0$. Both driving modes are gates (G264 (a) and (b)); the script uses the pressure-driven one because the driver's body force is the buoyancy and has no horizontal component.

$Q\_2$ velocity and $Q\_1$ pressure contain this solution exactly, and the face quadrature of the traction is exact for its polynomial data, so a channel run measures the linear solve, not the discretisation: the demonstration on a 128-leaf tree reports a profile error of $2\times10^{-14}$ of the centreline velocity and the same pressure field to $5\times10^{-13}$. The three mechanisms are needed together: without the inflow normal velocity nothing drives the region, without the tangential traction the outflow is a different (free-slip) flow, and without `:open`/`wall_tn` the outflow is not at a prescribed pressure.

If the outflow tangential traction is omitted (`wall_t = nothing`), the open wall is a free-slip outlet. The Poiseuille profile is then **not** the solution, because its outflow shear $G(H/2-y)$ violates the zero-tangential-traction condition; the free-slip solution keeps the same inflow data and mass flux, and differs from Poiseuille by exactly the response to the withheld shear traction. That difference is bounded and localised: on the plane channel it is 18% of the centreline velocity, and its strain energy is 2.4% of the derived scale $L\,\lVert\eta u'\rVert^2/(2\eta^2)$. G264 (d) checks the exact linearity in the withheld-traction scale and the response identities, and the script's page in [examples](examples.md) lists the run.

## 2. The discretization

### 2.1 Mesh and spaces

The mesh is a quadtree. A leaf is `(level, ix, iy)`, and the tree stays 2:1 balanced: face neighbors differ by at most one level. Refinement splits a quartet of siblings; coarsening merges one. The case caps depth at `max_level = 7`, which puts the finest leaf at $1/256$ of the box height, and caps the leaf count at `max_leaves = 10000`. The initial tree is uniform level 3 with 640 leaves.

Velocity and pressure form a Taylor–Hood pair: velocity in $Q\_p$ , pressure in $Q\_{p-1}$ , with `p = 2` on this case, so $Q\_2$ velocity and $Q\_1$ pressure. Nodes are the Gauss–Legendre–Lobatto points of the reference square, $p+1$ per direction for velocity and $p$ for pressure. Element matrices use Gauss–Legendre quadrature with `p+2` points per direction, the default `nq = p + 2` of `make_fedata`, which integrates the $Q\_p$ mass and stiffness matrices exactly.

### 2.2 Hanging nodes and constraints

A node on the face of a coarser leaf is a hanging node. Its value is whatever the coarse face's GLL interpolation gives, imposed through a sparse constraint matrix $C$ that is the identity on free rows. The system is assembled and solved as

$$
C^{\mathsf T} A\,C,
$$

where $A$ is the unconstrained assembly. This keeps symmetry and is exact for linear constraints. Boundary values enter the same affine map, and one pressure degree of freedom is pinned to remove the constant null space.

### 2.3 Time stepping

The flow is quasi-static, so each step solves Stokes once. The energy equation uses BDF2, with backward Euler on the first step. With the variable step ratio $w=\Delta t/\Delta t\_{\mathrm{prev}}$ the update reads

$$
\frac{a}{\Delta t}\theta^{n+1} + \mathbf u\cdot\nabla\theta^{n+1} - \nabla^2\theta^{n+1}
= \frac{b}{\Delta t}\theta^{n} - \frac{c}{\Delta t}\theta^{n-1},
$$

$$
a = \frac{2w+1}{1+w}, \qquad b = 1+w, \qquad c = \frac{w^2}{1+w}.
$$

| Symbol | Meaning |
|---|---|
| $\theta^n$ | Anomaly at the start of the step. |
| $\theta^{n-1}$ | Anomaly one step earlier. |
| $\Delta t$ | Current step; $\Delta t\_{\mathrm{prev}}$ the previous one. |
| $a, b, c$ | BDF2 mass coefficients; the first step uses $a=b=1/\Delta t$ and $c=0$ . |

With `maxpic = 1`, which is what the case uses, viscosity is lagged: the single flow solve of a step uses the start-of-step temperature. The step is therefore first order in the viscosity coupling, and deterministic.

The step size follows

$$
\Delta t = \mathrm{CFL}\,\frac{h_{\min}}{|\mathbf u|_{\max}},
$$

with `cfl = 0.5` and $h\_{\min}$ the smallest leaf height. It grows by at most 5% per step and shrinks by at most 5% per step unless the hard CFL cut fires, and it is clamped to $[\Delta t\_{\min},\Delta t\_{\max}]=[10^{-9},5\times10^{-5}]$ . The first step starts from $\Delta t\_0=10^{-6}$ and is immediately cut to the CFL value. The update reads the velocity and mesh of the step's first flow solve, so an AMR pass that zeroes the velocity cannot freeze `dt`.

### 2.4 Entropy viscosity

Oscillations are controlled by an elementwise entropy viscosity added to the energy operator, one value per leaf:

$$
\nu_e = c_{\mathrm{ev}}\frac{h_e^2 R_e}{\sigma},
\qquad
\nu_{\max} = c_{\max}|\mathbf u|\,h_e,
\qquad
\nu = \min(\nu_e, \nu_{\max}),
\qquad
h_e = \sqrt{2}\,h.
$$

| Symbol | Meaning |
|---|---|
| $c\_{\mathrm{ev}}$ | Entropy-viscosity coefficient, `ev = 0.11`. |
| $c\_{\max}$ | Cap coefficient, `ev_cmax = 0.052`. |
| $h$ | Leaf size; $h\_e$ is its diagonal scale. |
| $R\_e$ | Maximum over the leaf's quadrature points of the $\theta$ -equation residual, multiplied by $\lvert T-\langle T\rangle\rvert$ when `ev_alpha = 2`. |
| $\sigma$ | Global spread of the entropy residual, $\max(e\_{\max}-\bar e,\ \bar e-e\_{\min})$ of $(T-\langle T\rangle)^2$ over the mesh, with $\langle T\rangle=(T\_{\min}+T\_{\max})/2$ and $\bar e$ the area mean of $(T-\langle T\rangle)^2$ . |
| $\lvert\mathbf u\rvert$ | Maximum velocity magnitude in the leaf. |

The viscosity is zero on the wall rows and is recomputed every step. For the first two steps, and whenever a variation vanishes, the code falls back to $\nu=\nu\_{\max}$ . It is stored in the checkpoint, because the previous step's energy operator cannot be rebuilt without it.

## 3. The adaptive policy

### 3.1 Indicator

The indicator is the jump in the temperature gradient across leaf faces, plus a one-sided term on the Dirichlet walls:

$$
I_K = \frac{h}{2}\sum_{F\subset\partial K}\int_F [\![\nabla\theta\cdot\mathbf n]\!]^2\,\mathrm ds.
$$

| Symbol | Meaning |
|---|---|
| $h$ | Leaf height. |
| $F$ | A face of the leaf. |
| $[\![\cdot]\!]$ | Jump of the normal derivative between the two sides. |
| $\mathbf n$ | Face normal. |

The left and right walls are insulating and contribute nothing. The top and bottom walls contribute a one-sided normal derivative, and that term is what resolves a thermal boundary layer.

**Choosing the field.** `amr_indicator` selects the field the jump estimator is applied to, and the card sets it with `CASE_INDICATOR`. The default, `:theta`, is the temperature-anomaly field of the equation above. `:velocity` applies the same construction to the velocity components, and `:both` adds the two indicators after each is normalized by its own total:

$$
j_{\mathrm{both}} = j_\theta + \frac{\sum_K j_{\theta,K}}{\sum_K j_{v,K}}\,j_{v,K},
$$

where $j\_\theta$ and $j\_v$ are the per-leaf indicators of the two fields. Neither field dominates the marking, and at rest, where the velocity vanishes, `:both` reduces exactly to `:theta`.

**The measured comparison.** The measurement is one developed 2-D state at equal cost, 10 000 leaves and 43 steps at the same fixed `dt`, against a 40 000-leaf `:theta` solution. The relative error of `Nu_bot` is 1.10e-4 with `:theta` and 1.01e-5 with `:velocity`; of the volume-mean temperature, 1.75e-4 against 1.15e-4. `:theta` wins on the exact wall-flux residual `resid_A/Nu_top`, about 7e-4 percent against 5e-3 percent. The mechanism is where the leaves go: `:theta` puts 85% of its fine leaves in the wall strips, while `:velocity` puts 3.6% there and refines the interior instead. The choice therefore depends on the observable, not on a ranking. The measurement covers one state, one Rayleigh number, 43 steps (0.2% of the time axis) and 2-D, so keep the default unless you have a reason, and read the comparison as a property of that state rather than a guarantee.

### 3.2 Marking, coarsening and the leaf cap

Marking is Dörfler's: `refine_frac = 0.2` is the share of the total indicator to resolve, so the cell count follows from the indicator field. The balanced cycle coarsens from below with the relative threshold `coarsen_frac * total` on a complete quartet, pays for refinement by merging the lowest-error eligible quartet, and caps the leaves dropped per cycle at `round(coarsen_frac * n)`. The case uses `coarsen_frac = 0.05`.

Two hard limits apply after every cycle: the depth cap `max_level` and the leaf cap `max_leaves`. `amr_coarsen_gate` refuses a merge that the next cycle would have to undo, the 2:1 pre-check of the balanced policy. Fields move between meshes by nodal interpolation (`transfer_mode = :nodal`), and `amr_passes = 1` on this case.

### 3.3 Hysteresis and the wall rule

The hysteresis cooldown (`amr_hysteresis = 1`, `CASE_HYST`) keeps a quartet that was just merged from being split again for that many cycles; `0` disables it. The deadband multiplier of the balanced cycle is fixed at `0.5`, applied as `tau_h = min(tau_c, 0.5 tau_r)`.

The wall rule forces every leaf within `wall_strip = 0.1` of the top or bottom wall, and the wall rows themselves, back to at least `wall_min_level = 5` after every cycle, then enforces the leaf cap. `CASE_WALLMIN=0` turns it off. The rule exists because the indicator alone leaves the top wall at level 3 or 4 while the interior reaches level 5 or 6, and the pointwise `Nu_top` then differs from the wall-refined value by 1.2% and the consistent boundary flux by 9.6%.

## 4. Linear solvers and the `:auto` rule

### 4.1 The direct path

The direct path forms the reduced saddle

$$
K_{\mathrm{red}}\,\mathbf u + B^{\mathsf T} p = \mathbf f,
\qquad
B_s\,\mathbf u = 0,
$$

after constraint and boundary elimination, factorizes it with UMFPACK, and checks the true reduced residual. A wall-refined free-slip tree makes the unshifted factorization unstable, so the path walks a shift ladder: relative diagonal shifts of $10^{-12}$ , then $10^{-10}$ , then $10^{-8}$ . If no rung meets the acceptance tolerance, a few GMRES refinement iterations against the unshifted operator, preconditioned by the best rung, polish the iterate. Two smaller shifts, $10^{-14}$ and $10^{-13}$ , are tried first when the unshifted factorization has lost the solution outright, and again as a last resort before the step fails. With `stokes_scaling = :equilibrated` the path first tries the symmetric equilibration $S\to DSD$ , which lifts the near-null pressure modes above the round-off floor and engages only on a solve that would otherwise enter the ladder. The direct path is the only bitwise reproducible one, so a bitwise reference pins it explicitly.

### 4.2 The multigrid path

The multigrid path is a block-triangular preconditioner for flexible GMRES: a geometric V-cycle for the velocity block, with h-coarsening of at least 1.5 per level, damped-Jacobi smoothing, and every level built from the leaf-local matrix-free apply; and a three-factor w-BFBT approximation of the pressure Schur complement. FGMRES restarts every 100 iterations and is capped at 500, stopping on the true reduced-saddle residual against `mg_tol`. The case passes `mg_tol = 1e-8` , five orders below its `stokes_tol = 1e-3` ; the library default is `1e-10` .

### 4.3 The `:auto` rule

`linear = :auto` resolves the path before assembly, from two properties of the current saddle:

$$
N_{\mathrm{saddle}} = 2N_u + N_p,
$$

with $N\_u$ the velocity degrees of freedom and $N\_p$ the pressure ones. The rules are:

| Condition | Choice |
|---|---|
| $N\_{\mathrm{saddle}} \ge 20\,000$ | Multigrid, at any viscosity contrast. |
| Contrast at most 100 and $N\_{\mathrm{saddle}} \ge 2\,000$ | Multigrid. |
| Everything else | The direct path, which is also the bitwise path, so ties go there. |

The contrast is sampled on a fixed 33 x 33 grid of the viscosity field the solve actually uses, 1 089 evaluations at a fixed cost. Once `:auto` picks multigrid it also forces `mg_operator = :matrixfree` and `mg_threads = min(8, Threads.nthreads())` . If that solve misses `stokes_tol` or throws, the same solve is redone on the direct path and counted in `LAST_AUTO_FALLBACK` ; the run continues and the count is printed as `auto_fallbacks` at the end. An explicit `linear = :mg` has no such net and fails loudly. At this case's states the saddle is 78 720 degrees of freedom at 8 188 leaves and 95 308 at 10 000, both above the 20 000 rule, so a new run resolves to multigrid. The library also keeps two historical paths, `:schur_cg` (CG on the Schur complement, with `schur_maxiter` and `schur_precond`) and `:lagged` (a reused direct factorization); this case uses neither.

## 5. Parameter reference

### 5.1 Environment variables

Every knob is a `CASE_*` environment variable read when the script starts. The second column is the case's default; the third lists the accepted values.

| Variable | Default | Allowed | Effect |
|---|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | any path | Output root. Everything the run writes goes here. |
| `CASE_STEPS` | 0 | integer | Step limit, compared against the absolute step number. 0 selects the mode default: 10 smoke, 150 short, 10000 spinup, 20000 freeze, 2000 resume and wallrefine, none for symseed. |
| `CASE_FRAME` | 0 | integer | Frame stride in steps. 0 uses the case's 5; frames are written when `step % stride == 1`, one step past every AMR cycle. |
| `CASE_LX` | 2.5 | float | Box width. Keep `CASE_LX/CASE_CX == 1/CASE_CY`, or the leaves stop being square. |
| `CASE_CX`, `CASE_CY` | 5, 2 | integers | Base cells in x and y. |
| `CASE_MAXLEV` | 7 | integer | Maximum quadtree level, so the finest leaf is $1/(2\cdot2^{\ell})$ of the box height. |
| `CASE_MAXLEAVES` | 10000 | integer | Hard leaf cap, enforced after every AMR cycle. |
| `CASE_AMR_GATE` | 1 | 0, 1 | The 2:1 coarsening pre-check of the balanced policy. 0 turns it off. |
| `CASE_HYST` | 1 | integer | Hysteresis cooldown, in AMR cycles. 0 disables it. |
| `CASE_COARSEN` | 0.05 | float | Coarsening share: relative threshold and per-cycle cell cap of the balanced policy. |
| `CASE_WALLMIN` | 5 | integer | Minimum level inside the wall strip and on the wall rows. 0 turns the wall rule off. |
| `CASE_WALLSTRIP` | 0.1 | float in (0, 0.5) | Width of the wall strip, as a fraction of the box height from each wall. |
| `CASE_WALL_BC` | empty | `same`, `free`, `rigid` or `open`, four entries | Per-wall velocity condition, in the order (xminus, xplus, yminus, yplus); empty means all `same`, which follows `bc` (section 1.1). `open` leaves both components free and switches the pressure gauge to the derived one. |
| `CASE_WALL_UN` | empty (all `nothing`) | four entries, each empty or a float | Per-wall normal velocity, in the order (xminus, xplus, yminus, yplus); an entry is the wall's normal component in the positive coordinate direction, an empty entry keeps the historical zero, and while no wall is `open` the four values must have zero net outward flux (section 1.1). |
| `CASE_WALL_T` | empty (all `nothing`) | four entries, each empty or a float | Per-wall tangential traction, in the order (xminus, xplus, yminus, yplus); an entry is the total Cauchy traction's tangential global component (`t_x` on a horizontal wall, `t_z` on a vertical one), an empty entry is zero traction, and an entry is only accepted on a wall that resolves to `:free` or `:open` (section 1.1). |
| `CASE_WALL_TN` | empty (all `nothing`) | four entries, each empty or a float | Per-wall normal traction, in the order (xminus, xplus, yminus, yplus); an entry is the total Cauchy traction's normal global component (`t_x` on a vertical wall, `t_z` on a horizontal one), an empty entry is zero traction, and an entry is only accepted on an `:open` wall. Because the traction uses the outward normal, a pressure `p0` is `-p0` on the top wall and `+p0` on the bottom wall (section 1.1). |
| `CASE_SYMMETRY` | `none` | `none`, `x` | x-mirror closure of the AMR passes: `x` refines and merges mirror partners together, for symmetric benchmarks. |
| `CASE_INDICATOR` | `theta` | `theta`, `velocity`, `both` | Field the AMR jump indicator is applied to, as in section 3.1. |
| `CASE_LINEAR` | `auto` | `auto`, `direct`, `mg`, `schur_cg`, `lagged` | Linear solver, as in section 4. |
| `CASE_EV` | 0.11 | float | Entropy-viscosity coefficient. |
| `CASE_LATENT` | `per_phase` | `per_phase`, `global` | Latent coefficients: one per transition, or the single historical `latent_L = 0.11` for both. Any other value is rejected when `Params` is constructed. |
| `CASE_COMP_ED`, `CASE_COMP_ES` | 12.0, 17.0 | floats | Activation energies of the composite law's diffusion and dislocation branches, `E_d` and `E_s` (section 1.4). |
| `CASE_COMP_AD`, `CASE_COMP_AS` | 1.6240183119e5, 2.6796419238e-1 | floats | Prefactors of the two branches, in units of $1/\eta\_0$ . A run at another `Ra` rescales them by `Ra / Ra_new` to keep the dimensional material constants fixed. |
| `CASE_DT0` | 1.0e-6 | float | Initial step. The first step is cut to the CFL value. |
| `CASE_DTMAX`, `CASE_DTMIN` | 5.0e-5, 1.0e-9 | floats | Upper and lower bounds of the adaptive step. |
| `CASE_RA` | 1.0e5 | float | Rayleigh number of the card. |
| `CASE_ETA_MIN`, `CASE_ETA_MAX` | 1.0e-4, 1.0e4 | floats | Absolute viscosity bounds, in the law's units and applied before the `eta_ref` normalization (section 1.9). |
| `CASE_STOKES_SCALING` | `none` | `none`, `equilibrated` | Equilibration of the direct saddle. It engages only on a solve that would otherwise fail, so a healthy solve is unchanged. |
| `CASE_ETA_REF` | 1.0 | float > 0 | Rayleigh-number reference viscosity in the units the material law returns: `Ra` is defined with it and the flow solve uses `eta/eta_ref`. The header prints it and the checkpoint records it (section 1.9). |
| `CASE_SHEAR_MODULUS` | empty (off) | float >= 0 | Maxwell viscoelasticity: the nondimensional elastic shear modulus $G$ . Empty or 0 is off and leaves the run bit-identical; a positive value changes the effective viscosity and the physical regime, as section 1.8 describes. |
| `CASE_EXTDIAG` | 0 | 0, 1 (also `true`, `yes`, `on`) | Append the exact energy-identity residual `resid_A` to `history.csv`. It changes the column count, so it is accepted only in a new output directory. |

`main()` rejects an unknown `CASE_AMR_GATE` (not 0 or 1), `CASE_SYMMETRY` (not `none` or `x`) or `CASE_STOKES_SCALING` (not `none` or `equilibrated`) before doing any work. It also validates `CASE_SHEAR_MODULUS` (finite and at least 0), `CASE_ETA_REF` (finite and positive), the four entries of `CASE_WALL_BC`, `CASE_WALL_UN`, `CASE_WALL_T` and `CASE_WALL_TN` (each empty or a finite float), and the well-posedness of an `:open` wall combination (refused at `Params` construction). The following are literals rather than knobs: `H = 10`, `p = 2`, `stokes_tol = 1e-3`, `mg_tol = 1e-8`, `amr_every = 5`, `refine_frac = 0.2`, `cfl = 0.5`, `dt_growth = 1.05`, `maxpic = 1`, `pic_tol = 1e-6`, `ev_cmax = 0.052`, `ev_alpha = 2`, `checkpoint_every = 200`, and the 256 x 256 frame grid.

With no argument the script runs `short`. Otherwise the first argument selects one of seven modes:

| Mode | Default steps | What it does |
|---|---|---|
| `smoke` | 10 | Ten steps from the initial condition, AMR at steps 5 and 10, frames at 1, 6 and 10, no checkpoint. |
| `short` | 150 | Stability check; writes `short.h5` every 200 steps and at the end. |
| `spinup` | 10000 | Production spin-up with AMR; writes `spinup.h5` every 200 steps and at the end. |
| `freeze <ckpt>` | 20000 | Copies the checkpoint to `frozen.h5` with `amr_every = 0`, then continues on the frozen mesh. |
| `wallrefine <ckpt>` | 2000 | Writes a wall-refined seed under `seed/`, then continues it with AMR to `resume.h5`. |
| `resume <ckpt>` | 2000 | Continues any checkpoint with AMR to `resume.h5`. |
| `symseed <ckpt>` | none | Writes a copy of the checkpoint with `symmetry` rewritten under `seed/`, and runs no steps. |

The four restart modes read `$CASE_OUT/spinup.h5` when the checkpoint argument is omitted. The three seed modes copy a checkpoint and rewrite exactly the configuration strings the next stage changes, so the restart guard still checks everything else.

### 5.2 Keywords of `run_semquad_simulation`

The defaults are the library defaults; the case column is the value `case_realistic_2d.jl` passes.

| Keyword | Default | Case | Effect |
|---|---|---|---|
| `p` | 2 | 2 | Polynomial order: velocity $Q\_p$ , pressure $Q\_{p-1}$ . |
| `maxsteps` | 10000 | mode default, `CASE_STEPS` overrides | Absolute step number to stop at. |
| `tstop` | 1e-3 | `Inf` | Stop at this nondimensional time. |
| `dt0`, `dtmax`, `dtmin`, `dt_growth` | 1e-4, `Inf`, 1e-14, 1.05 | 1e-6, 5e-5, 1e-9, 1.05 | Step size and its bounds. |
| `cfl` | 0 | 0.5 | CFL number of the step update; 0 freezes `dt` at `dt0`. |
| `cfl_tol` | `Inf` | `Inf` | Abort the run when `umax*dt/hmin` exceeds this wall-CFL guard. |
| `tmin`, `steady_tol`, `steady_window` | 0, `Inf`, 5 | 0, `Inf`, 5 | Steady-state stop: after `t > tmin`, stop once the relative theta change and the relative `Nu` change stay below `steady_tol` for `steady_window` steps. |
| `tavg_start` | `Inf` | `Inf` | Start of the running time average returned as `avg`; `Inf` disables it. |
| `amr_every` | 0 | 5 | Steps between AMR cycles; 0 freezes the mesh. |
| `amr_indicator` | `:theta` | `CASE_INDICATOR` | Field of the AMR jump indicator: `:theta`, `:velocity` or `:both` (section 3.1). |
| `amr_passes` | 1 | 1 | Indicator, refine and transfer passes per cycle. |
| `refine_frac`, `coarsen_frac` | 0.2, 0.0 | 0.2, `CASE_COARSEN` | Cumulative-error shares of the balanced cycle, marked by Dörfler from above and from below. |
| `amr_hysteresis`, `amr_coarsen_gate` | 1, true | `CASE_HYST`, `CASE_AMR_GATE == 1` | Hysteresis cooldown and the 2:1 gate. |
| `max_level`, `max_leaves` | no cap | 7, 10000 | Depth cap and hard leaf cap. |
| `wall_min_level`, `wall_strip` | 0, 0.1 | 5, 0.1 | The wall rule; 0 turns it off. |
| `symmetry`, `transfer_mode` | `:none`, `:nodal` | `CASE_SYMMETRY`, `:nodal` | Mirror closure and the AMR transfer. |
| `initial_refine`, `theta_ic` | 0, `nothing` | 0, `nothing` | AMR cycles applied to the initial condition before the first step; `theta_ic` re-samples an analytic initial condition on each refined tree. |
| `hotspot_weight`, `hotspot_frac`, `hotspot_min_level`, `hotspot_mode`, `hotspot_zmin` | 0, 0.5, 0, `:theta`, 0 | off | Hotspot refinement rule. |
| `top_wall_level`, `top_wall_frac` | 0, 0.5 | off | Top-wall refinement rule. |
| `zero_side_eta`, `side_strip` | false, 0.05 | false, 0.05 | Side-strip viscosity zeroing. |
| `advection` | `:galerkin` | `:skew` | Advection form of the energy equation. |
| `ev`, `ev_cmax`, `ev_alpha` | 0, 0.052, 2 | 0.11, 0.052, 2 | Entropy viscosity, as in section 2.4; `ev = 0` switches it off. |
| `maxpic`, `pic_tol`, `pic_relax`, `pic_mode` | 1, 1e-6, 1.0, `:picard` | same | Nonlinear iteration inside a step. `pic_mode = :newton` is a research option: it requires `linear=:direct`, refuses the reference state, the compressible mass form and `nu_cbf`, and is slower than Picard with multigrid on the measured cases. |
| `pic_reject`, `pic_reject_clamp`, `pic_reject_res`, `pic_reject_max` | false, 0.25, 10, 4 | defaults | Reject a Picard trial when the clamp fraction or the residual grows, and redo it with a halved relaxation. |
| `pic_adapt`, `pic_relax_min`, `pic_eta_relax`, `pic_u_relax` | false, 0.05, 1.0, 1.0 | defaults | Relaxation adaptation, strain-rate averaging and velocity under-relaxation of the Picard loop. |
| `pic_div_window`, `pic_div_factor` | 0, 10 | defaults | Divergence detector of the Picard loop. |
| `newton_maxit`, `newton_krylov`, `newton_restart`, `newton_ls`, `newton_geo_safety` | 15, 80, 30, 8, 0 | defaults | Newton controls, inert while `pic_mode = :picard`. |
| `linear` | `:auto` | `CASE_LINEAR` | Linear solver, as in section 4. |
| `stokes_tol` | `Inf` | 1e-3 | Abort the step above this true residual; also the `:auto` fallback and the direct path's acceptance tolerance. |
| `schur_maxiter`, `schur_precond` | 200, `:mass` | defaults | Iteration cap and preconditioner of `linear = :schur_cg`. |
| `mg_tol`, `mg_threads` | 1e-10, 1 | 1e-8, 1 (`auto`: `min(8, nthreads)`) | Outer tolerance and thread count of the multigrid. |
| `mg_operator`, `mg_vel_solver`, `mg_smoothing`, `mg_omega` | `:matrixfree`, `:vcycle`, `:auto`, `nothing` | defaults | Multigrid geometry, velocity solver and V-cycle schedule. |
| `mg_pressure_eta_tol` | 1.0 | 1.0 | Rebuild threshold of the cached pressure block when the sampled viscosity drifts. |
| `stokes_scaling` | `:none` | `CASE_STOKES_SCALING` | `:none` or `:equilibrated`, direct path only. |
| `corotation_guard` | true | defaults | Report the validity band of the co-rotational term and warn once when the run leaves it; inert and free at `shear_modulus = 0` (section 1.8). |
| `regime_report` | true | defaults | Report the Maxwell regime once at the first stress step and once at the end of the run; inert and free at `shear_modulus = 0` (section 1.8). |
| `eta_ref_scale` | `nothing` | the card's `(rho, g, alpha, dT, d, kappa)` scale set | Dimensional scale set for the `eta_ref` report; with it the report also prints `eta_0` in Pa s (section 1.9). |
| `mass_form` | `:anelastic` | `:boussinesq` | Mass conservation form. |
| `phases`, `latent_scheme` | `[]`, `:implicit` | two transitions, `:implicit` | Phase transitions and the latent coupling; the two schemes agree when `maxpic = 1`. |
| `weak_chi0` | 0 | 0 | Initial strain-weakening field; active only when the `Params` weakening parameters are set. |
| `nu_cbf` | false | false | Append the consistent-boundary-flux Nusselt pair to the history. |
| `restart_from` | `""` | mode dependent | Checkpoint to restart from. |
| `checkpoint_path`, `checkpoint_every` | `""`, 0 | mode dependent (`spinup.h5`, `short.h5`, `frozen.h5`, `resume.h5`), 200 | Where and how often to checkpoint. |
| `checkpoint_meta`, `callback`, `verbose`, `profile_steps` | empty, `nothing`, false, false | case metadata, frame writer, true, false | Checkpoint metadata, step hook, per-step lines, per-step timing. |
| `pic_diag`, `extdiag` | false, false | false, `CASE_EXTDIAG` | Nonlinear diagnostics, and the `resid_A` history column. |

The function returns `theta`, `t`, `steps`, `history`, `history_extra`, `qt`, `ss`, `ux`, `uz`, `pv`, `avg`, `weak`, `chi`, `chi_prev`, `txx`, `tyy` and `txy`. The three stress components are `nothing` unless `shear_modulus > 0` (section 1.8). `history` holds one row per step actually run, with the columns of section 6.2, and `history_extra` names the columns appended after the fixed 18.

### 5.3 The `Params` card

`make_params()` in the case script sets these fields. Everything else in `Params` keeps its default, listed in `src/physics/params.jl`.

| Field | Value | Meaning |
|---|---|---|
| `Ra`, `H` | 1e5, 10 | Rayleigh number and internal heating. |
| `eta_law`, `comp_rule` | `:composite`, `:isostress` | Viscosity law and combination rule. |
| `comp_Ad`, `comp_As` | 1.6240183119e5, 2.6796419238e-1 | Diffusion and dislocation prefactors. |
| `comp_Ed`, `comp_Es` | 12, 17 | Activation energies. |
| `comp_Vd`, `comp_Vs` | 0, 0 | Activation volumes. |
| `comp_n`, `comp_Tcut`, `comp_eps_min` | 3, 0 (off), 1e-12 | Stress exponent, temperature floor, strain-rate floor. |
| `eta_T0`, `eta_min`, `eta_max` | 0.5, 1e-4, 1e4 | Surface offset and viscosity bounds. |
| `eta_ref` | 1.0 | Rayleigh-number reference viscosity: `Ra` is defined with it and the flow solve uses `eta/eta_ref` (section 1.9). |
| `bc`, `wall_bc`, `wall_un`, `wall_t`, `wall_tn` | `:free`, from `CASE_WALL_BC`, from `CASE_WALL_UN`, from `CASE_WALL_T`, from `CASE_WALL_TN` | Free-slip walls; `:rigid` is no-slip, `:open` leaves both components free, and `wall_bc` overrides one wall at a time (section 1.1). `wall_un` refines the Dirichlet *normal* value per wall; empty entries keep the historical zero, and while no wall is `open` the four values must have zero net flux. `wall_t` adds a per-wall *tangential* traction as a boundary-face load and `wall_tn` a per-wall *normal* traction on an `:open` wall; empty entries are zero traction, and a `wall_tn` entry also switches the pressure gauge to the derived (unpinned) one (section 1.1). |
| `latent_L`, `latent_mode` | 0.11, `:per_phase` | Historical single coefficient, unused here, and the per-transition mode. |
| `yield_c`, `yield_phi`, `yield_taumax` | `Inf`, 0, `Inf` | Yield surface off. |
| `yield_c_grad`, `yield_mu_grad` | 0, 0 | Depth dependence of the yield surface off. |
| `weak_f`, `weak_eps_ref` | 1, 0 | Strain weakening off. |
| `rho_heating`, `rho_cp` | false, false | Density-weighted heating off. |
| `side_thermal`, `Twall_left`, `Twall_right` | `:insulating` | The side walls are insulating (zero normal flux). `:fixed` instead holds them at the two physical temperatures, using the same `Twall - Tbar` convention as the top and bottom walls. |
| `H_half_life` | `Inf` | Radiogenic heating held constant. A finite value decays it as `H 2^(-t/H_half_life)`, in the code's time unit; it requires the Picard time loop (`pic_mode = :picard`). |
| `Di`, `heating_adiabatic`, `heating_dissipation` | 0, false, false | Dissipation and adiabatic heating off. |
| `refstate`, `deviatoric` | `nothing`, false | Conductive base state, full strain rate. |
| `shear_modulus` | 0 | Maxwell elastic shear modulus $G$ ; 0 disables every viscoelastic term and leaves the run bit-identical (section 1.8). |
| `Twall_top`, `Twall_bot` | 0, 1 | Wall temperatures. |

`Params` also validates the law. `:composite` refuses a Drucker–Prager yield surface, `:tosi` refuses `yield_*` because it carries its own plastic branch, and `:arrhenius` needs `eta_T0 > 0`.

## 6. Output formats

Everything is written under `CASE_OUT`. The program creates the directory; only a shell redirect into it needs the directory to exist first.

### 6.1 `run.log`

The program does not create this file. It is the shell redirect of stdout and stderr, and it holds the four header lines, two lines per step (the step record and the Picard update, with `verbose=true`), two lines every 5 steps (the `AMR:` record and the `balanced:` counters), a progress line every 100 steps from the case callback, and at the end

```text
[case] DONE mode=spinup steps=2000 t=<t> wall=<s> s (<s> s/step) linear=auto scaled=0 auto_fallbacks=0
[case] final: Nu_top=<v> Nu_bot=<v> Vrms=<v> meanT=<v> closure=<v> leaves=<n> N=<n>
```

stdout is block buffered, and the callback flushes it every 100 steps, so a redirected log lags by at most that much. Errors go to stderr, unbuffered.

### 6.2 `history.csv`

Written once, when the run returns, with one row per step actually run:

```text
t,dt,Nu_top,Nu_bot,Vrms,meanT,N,leaves,div_max,hmin,umax,cfl,closure,usrms_surf,umax_surf,W,U,balance
```

This case appends one more column because its latent terms are active.

| Column | Meaning |
|---|---|
| `t` | Nondimensional time at the end of the step. |
| `dt` | The step just taken. |
| `Nu_top`, `Nu_bot` | Wall Nusselt numbers, as in section 1.3. |
| `Vrms` | Volume RMS of the velocity magnitude. |
| `meanT` | Volume mean of the full temperature $\bar T+\theta$ . |
| `N` | Number of scalar-space nodes on the mesh. |
| `leaves` | Number of quadtree leaves. |
| `div_max` | Maximum pointwise divergence of the velocity at the GLL nodes. |
| `hmin`, `umax` | Smallest leaf height and largest velocity magnitude. |
| `cfl` | `umax * dt / hmin`, the CFL number of the step. |
| `closure` | $\mathrm{Nu}\_{top}-\mathrm{Nu}\_{bot}-(H-I\_{base})$ plus the latent term's time derivative over the area, with $I\_{base}=\int u\_y\,\mathrm d\bar T/\mathrm dy$ . It tends to zero at a steady state and is the enthalpy storage rate plus booking terms at a transient one. |
| `usrms_surf`, `umax_surf` | RMS and maximum velocity magnitude on the top wall. |
| `W`, `U` | Volume means of the buoyancy work $\langle\rho T u\_y\rangle$ and the viscous dissipation $\langle2\eta\,\boldsymbol\varepsilon:\boldsymbol\varepsilon\rangle$ . |
| `balance` | $(\mathrm{Ra}\,W-U)/\max(\lvert\mathrm{Ra}\,W\rvert,\lvert U\rvert)$ , the relative residual of the mechanical balance. |
| `closure_nolatent` | `closure` without the latent term. Present because this case has latent heat; a run without phases writes 18 columns. |

With `CASE_EXTDIAG=1` a `resid_A` column follows: the exact residual of the quoted pointwise Nusselt pair against the scheme's energy identity, per unit area, so it is zero for the time-step flux and $O(h)$ for the pointwise one. With `nu_cbf=true` the consistent-boundary-flux pair, `Nu_cbf_top` and `Nu_cbf_bot`, is appended. A run can carry both extras; on this case the appended columns are `closure_nolatent`, `Nu_cbf_top`, `Nu_cbf_bot` and `resid_A`, in that order.

### 6.3 Frames

One frame per `CASE_FRAME` steps, offset one step past every AMR cycle. The AMR pass zeroes the velocity, so a frame written on an AMR step would carry a zero velocity and a meaningless Nusselt number. A final frame is always written.

Each frame writes these files:

| File | Content | Format |
|---|---|---|
| `frames/frame_NNNNN_T.bin` | The full temperature $T=\bar T+\theta$ on a 256 x 256 grid. `x` varies fastest, then `y`. | Little-endian Float64, 512 KB. |
| `frames/frame_NNNNN_mesh.csv` | One line per leaf, in leaf order (level, then `ix`, then `iy`), with the columns `level,x0,y0,h` . `h` is the leaf height, `x0` and `y0` the lower-left corner. | 17 significant digits. |
| `frames/frame_NNNNN_eta.bin` | $\log\_{10}$ of the viscosity the flow solve uses ( $\eta/\eta\_{\mathrm{ref}}$ ), at the velocity nodes. | Float32, one value per node. |
| `frames/frame_NNNNN_speed.bin` | Velocity magnitude at the velocity nodes. | Float32, one value per node. |
| `frames/frame_NNNNN_nodes.bin` | The coordinates of the velocity nodes, `x` then `y` for each. | Float32. |

In numpy the temperature grid is `T = np.fromfile(path).reshape(256, 256)`, with `T[j, i]` the value at `(x_i, y_j)`, drawn with `origin="lower"`.

`frames/frame_meta.csv` is opened in append mode and gets one row per frame:

```text
frame,step,t,dt,N,leaves,nlev,Tmin,Tmax,Tmean,Nu_top,Nu_bot,Vrms,umax,eta_min,eta_max,eta_clip_frac,wall_s,ng
```

`nlev` is the number of distinct levels, `eta_clip_frac` the fraction of velocity nodes whose viscosity hit a bound, and `wall_s` the seconds since the frame writer was constructed.

### 6.4 Checkpoints

An HDF5 file, written every `checkpoint_every` steps and once at the end, always to the same path (`$CASE_OUT/spinup.h5`, `frozen.h5`, `resume.h5`, and so on). It holds the full state needed to continue the trajectory:

| Content | Shape |
|---|---|
| `leaves` | `3 x nleaves` Int32: level, ix, iy of every leaf, sorted. |
| `theta`, `theta_prev` | The two BDF2 temperature levels on the scalar nodes. |
| `ux`, `uz` | Velocity components on the scalar nodes. |
| `pv` | Pressure on the pressure space. |
| `txx`, `tyy`, `txy` | Optional, the Maxwell stress history on the scalar nodes. Written when `shear_modulus > 0` (section 1.8). |
| `ev` | Optional, one entropy-viscosity value per leaf, in `leaves` order. |
| `weak`, `weak_prev` | Optional, the strain-weakening field at its two levels. Absent on this case. |
| `cooldown_no_refine`, `cooldown_no_coarsen` | Optional, the balanced policy's hysteresis state, one column per leaf key. |
| Attributes | `t`, `dt`, `dt_prev`, `step`, `p`, `maxlevel`, `nx0`, `ny0`, `Lx`, `cooldown_cycle`, `velocity_stale`, the free-form metadata, and `config`: the trajectory-affecting settings as newline-separated `key=value` strings. |

`velocity_stale` marks a final checkpoint whose last step was an AMR step: the stored velocity is a diagnostic re-solve, and a restart zeroes it. A checkpoint is about 2 MB at the production state.

## 7. Restarting and the configuration guard

`restart_from` replaces the mesh, `p` and the initial state with the checkpoint contents before the first step, and `checkpoint_path` writes the continued run to its own file. `maxsteps` and `tstop` stay absolute. The quick start shows the command for this case (section "Stopping and resuming" there).

The guard compares the checkpoint's stored `config` against the settings of the live call, key by key, and refuses to start on any difference:

```text
ArgumentError: restart configuration mismatch in /path/resume.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

The compared keys are the ones that affect the trajectory. They cover the AMR policy: `amr_every`, `amr_passes`, `refine_frac`, `coarsen_frac`, `max_leaves`, `max_level`, `wall_min_level`, `wall_strip`, `symmetry`, `transfer_mode`, `amr_hysteresis`, `amr_coarsen_gate`, `amr_indicator`, `amr_policy`, the hotspot and top-wall keys, `zero_side_eta` and `side_strip`. They cover the energy step: `advection`, `ev`, `ev_cmax` and `ev_alpha`. They cover the nonlinear and time-step controls: `maxpic`, `pic_tol`, `pic_relax`, `pic_mode`, `pic_reject`, `cfl`, `dtmax`, `dtmin`, `mass_form` and `deviatoric`. They cover the physics: `H`, `H_half_life`, `rho_heating`, `rho_cp`, `Di`, `heating_adiabatic`, `heating_dissipation`, the active law's parameters (`comp_*`, `eta_T0`, `eta_min`, `eta_max` and `comp_rule` here; the `tosi_*` block for that law), the yield and weakening parameters, `latent_L`, `latent_mode`, `latent_scheme` and the phase list, the reference state and its `ref_*` values, the weak-zone and density hooks (`eta_multiplier` and `rho_multiplier`, stored as `none` or `custom`), the wall motion (`wall_bc`, and `wall_u`/`wall_un`/`wall_t`/`wall_tn` stored as one `none`/`custom` marker per wall: a function has no value to compare, so a restart that changes the value but not the presence of a wall function is not detected, and should be restarted from scratch), `side_thermal`, `Twall_left`, `Twall_right`, `shear_modulus`, `eta_ref`, `stokes_scaling`, and `physics_revision`.

`physics_revision` is a property of the physics code, not of a parameter value. The driver refuses a checkpoint written under another revision, and resumes a file that has no revision marker with a warning that states whether the file carries a Maxwell stress history. The comparison iterates over the checkpoint's keys, so a key added after a checkpoint was written is not compared. `stokes_scaling` has a defined default: a file without it reads as `none`, and switching the option on for such a chain is refused.

Some settings are state rather than configuration, so they are not compared: `p` and the mesh are restored from the file, and the balanced cooldown is restored with them. `dt0`, `dt_growth`, `initial_refine`, `linear`, `mg_tol`, `mg_threads` and the rest of the solver's settings are neither stored nor compared, so the same trajectory can be continued on a different linear path or with a different checkpoint cadence.

When a resume is refused, repeat the stored values, or use a seed mode. A seed mode copies the checkpoint and rewrites exactly the conflicting strings: `freeze` sets `amr_every = 0`, `wallrefine` sets `wall_min_level`, `wall_strip`, `max_leaves`, `ev` and `symmetry` and refines the wall, and `symseed` sets `symmetry`. The source file is never modified.

## 8. Threads and memory

Start Julia with `--threads=N`. The only part of this case that uses the Julia thread pool is the matrix-free V-cycle that `linear = :auto` selects, and it is capped at eight threads, `mg_threads = min(8, Threads.nthreads())`, so more than eight adds nothing on that path. With one thread the auto path warns once and runs serial. An explicit `linear = :mg` keeps `mg_threads = 1` unless the caller sets it, and the direct factorization is serial in every case.

At eight threads the warm production Stokes solve takes 1.36 s, against 3.37 s for the direct solve. The full solve gains 1.30 to 1.42x over a single thread, and the V-cycle gains 1.77 to 2.05x.

At the 10 000-leaf cap, the production state peaks at about 1.3 GB resident on the multigrid path with eight threads, and about 2.4 GB with the direct factorization, so the multigrid needs about half the memory of the direct solve at this size. Both figures are process peaks, so they include the Julia runtime and the loaded state. Plan on 2 GB of free memory for a default run and 3 GB for the direct path.

Disk is dominated by frames. Section 6.3 lists the files each frame writes; their size follows the tree, and the frame count follows `CASE_FRAME`. A checkpoint is about 2 MB (section 6.4).

## 9. Where to read next

**Next:** [Validation](validation.md) — the benchmark record behind these defaults, and the scope of what has been checked.
