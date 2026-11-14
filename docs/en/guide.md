<p align="right"><a href="../en/guide.md">English</a> · <a href="../zh/guide.md">中文</a></p>

# Guide: the realistic 2-D mantle-convection case

This page documents the case that `examples/case_realistic_2d.jl` builds: the physics, the numerical method, every knob, every output file, restarting, and the thread and memory behavior. Every value quoted here is either a literal in that script or a default of `run_semquad_simulation` in `src/driver/run.jl`. Section 9 points to the engineering record, where the measurements behind those defaults live.

The problem is a 2.5 x 1 box of incompressible fluid with free-slip walls, temperature 0 at the top and 1 at the bottom, internal heating `H = 10`, a nominal Rayleigh number `Ra = 1e5`, composite diffusion–dislocation rheology, and two phase transitions with latent heat. Yielding and strain weakening are off.

## 1. The physical problem

### 1.1 Domain and boundary conditions

The domain is $[0,2.5]\times[0,1]$ , built from 5 x 2 base cells that the quadtree subdivides into square leaves. $x$ is horizontal and $y$ is height, with $y=0$ at the bottom and $y=1$ at the top; depth is $1-y$ .

All four walls are free slip: no normal flow, unrestricted tangential flow. That is implemented by fixing the normal component,

$$
u_x = 0 \quad (x=0,\,2.5),
\qquad
u_y = 0 \quad (y=0,\,1),
$$

where $u\_x$ and $u\_y$ are the two velocity components. Temperature is $T=0$ at the top wall and $T=1$ at the bottom wall; the side walls are insulating.

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

`Ra = 1e5` multiplies the thermal buoyancy, with viscosity measured in units of the reference viscosity. The prefactors of section 1.4 put $\eta=1$ at $T=0.5$ , a temperature that sits inside the cold lid, so the convecting interior runs near $T\approx1.3$ where $\eta$ is between $10^{-3}$ and $10^{-2}$ . In internal Rayleigh-number terms that is $10^7$ to $10^8$ , and one overturn takes about $10^{-3}$ in nondimensional time.

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

Here $k$ is the ratio of the two branch viscosities at the total strain rate, and $x\in(0,1]$ is the fraction of the effective viscosity carried by the dislocation branch. The root is unique for $k>0$ ; $k\to0$ is the dislocation-only limit and $k\to\infty$ the diffusion-only one. The solve is Newton with a bisection bracket on $[0,1]$ , a tolerance of `1e-15` on $kx+x^n-1$ and a cap of 100 iterations; the worst case measured is 6. The case clips $\eta$ once, after the combination, to $[10^{-4},10^4]$ . That clip is a solver safeguard and it also sets the conditioning; the measured viscosity contrast is about $10^7$ .

The constants are $A\_d=1.6240183119\times10^5$ , $A\_s=2.6796419238\times10^{-1}$ , $E\_d=12$ , $E\_s=17$ , $n=3$ and $\eta\_{T0}=0.5$ . They satisfy two conditions at $y=0.5$ : $\eta(T=0.5,\dot\varepsilon\_{\mathrm{ref}})=1$ fixes the common scale, and $\eta(T=0.1)/\eta(T=0.9)=10^4$ at $\dot\varepsilon\_{\mathrm{ref}}$ fixes the branch ratio. The reference strain rate $\dot\varepsilon\_{\mathrm{ref}}=1.65\times10^3$ comes from a constant-viscosity precursor of the same case.

The library also carries simpler laws, none of which this case uses. Note that `Params` accepts `eta_law = :const` while `run_semquad_simulation` rejects it; use `:expT` with `deta_T = 1` for constant viscosity.

| `eta_law` | Viscosity | Parameters |
|---|---|---|
| `:expT` | $\eta = \exp[-\ln(\Delta\eta\_T)\,T]$ | `deta_T` is the viscosity ratio: $\eta(0)=1$ , $\eta(1)=1/\Delta\eta\_T$ . |
| `:expTz` | $\eta = \exp(-bT + c(1-y))$ | `eta_b` = b, `eta_c` = c; 2b uses $b=\ln 16384$ , $c=\ln 64$ . |
| `:arrhenius` | $\eta = \exp\{E\,[1/(T+T\_0) - 1/(1+T\_0)]\}$ | `eta_E` = E, `eta_T0` = T_0. |
| `:composite` | The two displays above. | `comp_*`, `eta_T0`, `comp_rule`. |
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

The capacity factor stays inside $[0.88,1.12]$ . The sum of the two coefficient magnitudes is 0.18996, and $\left|\partial\Gamma/\partial T\right|$ is bounded by $|s\_i|/(2w\_i)=0.64$ , which gives 0.122. The assembly throws if the factor would go negative, since that would reverse the energy equation in time. `CASE_LATENT=global` reproduces runs from before 2026-11-08, which used a single `latent_L = 0.11` for both transitions; that value even has the wrong sign at the 410, so it exists only to replay old results.

### 1.7 Initial condition

$$
\theta_0(x,y) = 0.05\sin(\pi y)\cos\left(\frac{2\pi x}{L_x}\right) + 0.02\sin(2\pi y)\cos\left(\frac{4\pi x}{L_x}\right),
$$

where $L\_x=2.5$ is the box width. It vanishes on the top and bottom walls, as an anomaly must, and is symmetric about $x=L\_x/2$ . Nothing is random: the card, the seed and a fixed step count make a run reproducible.

Yielding and strain weakening are off (`yield_c = yield_taumax = Inf`, `yield_phi = 0`, `weak_eps_ref = 0`), as are density-weighted heating (`rho_heating = rho_cp = false`), dissipation and adiabatic heating (`Di = 0`), the reference state (`refstate = nothing`), the deviatoric projection (`deviatoric = false`), and the anelastic mass form (`mass_form = :boussinesq`).

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

The multigrid path is a block-triangular preconditioner for flexible GMRES: a geometric V-cycle for the velocity block, with h-coarsening of at least 1.5 per level, damped-Jacobi smoothing, and every level built from the leaf-local matrix-free apply; and a three-factor w-BFBT approximation of the pressure Schur complement. FGMRES restarts every 100 iterations and is capped at 500, stopping on the true reduced-saddle residual against `mg_tol`. The case passes `mg_tol = 1e-8` , five orders below its `stokes_tol = 1e-3` ; the library default is `1e-10` . The iteration counts and timings behind these constants are in [the solver volume](../dev/SOLVER.md) section 4.

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

Every knob is a `CASE_*` environment variable read when the script starts. The second column is the case's default.

| Variable | Default | Allowed | Effect |
|---|---|---|---|
| `CASE_OUT` | `out_julia/case_realistic` | any path | Output root. Everything the run writes goes here. |
| `CASE_STEPS` | 0 | integer | Step limit, compared against the absolute step number. 0 selects the mode default: 10 smoke, 150 short, 10000 spinup, 20000 freeze, 2000 resume and wallrefine. |
| `CASE_FRAME` | 0 | integer | Frame stride in steps. 0 uses the case's 5; frames are written when `step % stride == 1`, one step past every AMR cycle. |
| `CASE_MAXLEV` | 7 | integer | Maximum quadtree level, so the finest leaf is $1/(2\cdot2^{\ell})$ of the box height. |
| `CASE_MAXLEAVES` | 10000 | integer | Hard leaf cap, enforced after every AMR cycle. |
| `CASE_AMR_GATE` | 1 | 0, 1 | The 2:1 coarsening pre-check of the balanced policy. 0 turns it off. |
| `CASE_HYST` | 1 | integer | Hysteresis cooldown, in AMR cycles. 0 disables it. |
| `CASE_COARSEN` | 0.05 | float | Coarsening share: relative threshold and per-cycle cell cap of the balanced policy. |
| `CASE_WALLMIN` | 5 | integer | Minimum level inside the wall strip and on the wall rows. 0 turns the wall rule off. |
| `CASE_WALLSTRIP` | 0.1 | float in (0, 0.5) | Width of the wall strip, as a fraction of the box height from each wall. |
| `CASE_SYMMETRY` | `none` | `none`, `x` | x-mirror closure of the AMR passes: `x` refines and merges mirror partners together, for symmetric benchmarks. |
| `CASE_LINEAR` | `auto` | `auto`, `direct`, `mg`, `schur_cg`, `lagged` | Linear solver, as in section 4. |
| `CASE_EV` | 0.11 | float | Entropy-viscosity coefficient. |
| `CASE_LATENT` | `per_phase` | `per_phase`, `global` | Latent coefficients: one per transition, or the single historical `latent_L = 0.11` for both. Any other value is rejected when `Params` is constructed. |
| `CASE_STOKES_SCALING` | `none` | `none`, `equilibrated` | Equilibration of the direct saddle. It engages only on a solve that would otherwise fail, so a healthy solve is unchanged. |
| `CASE_EXTDIAG` | 0 | 0, 1 (also `true`, `yes`, `on`) | Append the exact energy-identity residual `resid_A` to `history.csv`. It changes the column count, so it is accepted only in a new output directory. |

`main()` rejects an unknown `CASE_AMR_GATE`, `CASE_SYMMETRY` or `CASE_STOKES_SCALING` before doing any work. The following are literals rather than knobs: `Ra = 1e5`, `H = 10`, `p = 2`, `stokes_tol = 1e-3`, `mg_tol = 1e-8`, `amr_every = 5`, `refine_frac = 0.2`, `dt0 = 1e-6`, `dtmax = 5e-5`, `dtmin = 1e-9`, `cfl = 0.5`, `dt_growth = 1.05`, `maxpic = 1`, `pic_tol = 1e-6`, `ev_cmax = 0.052`, `ev_alpha = 2`, `checkpoint_every = 200`, and the 256 x 256 frame grid.

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
| `tmin`, `steady_tol`, `steady_window` | 0, `Inf`, 5 | 0, `Inf`, 5 | Steady-state stop: after `t > tmin`, stop once the relative theta change and the `Nu` change stay below `steady_tol` for `steady_window` steps. |
| `tavg_start` | `Inf` | `Inf` | Start of the running time average returned as `avg`; `Inf` disables it. |
| `amr_every` | 0 | 5 | Steps between AMR cycles; 0 freezes the mesh. |
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
| `maxpic`, `pic_tol`, `pic_relax`, `pic_mode` | 1, 1e-6, 1.0, `:picard` | same | Nonlinear iteration inside a step; Newton requires `linear=:direct`. |
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
| `mass_form` | `:anelastic` | `:boussinesq` | Mass conservation form. |
| `phases`, `latent_scheme` | `[]`, `:implicit` | two transitions, `:implicit` | Phase transitions and the latent coupling; the two schemes agree when `maxpic = 1`. |
| `weak_chi0` | 0 | 0 | Initial strain-weakening field; active only when the `Params` weakening parameters are set. |
| `nu_cbf` | false | false | Append the consistent-boundary-flux Nusselt pair to the history. |
| `restart_from` | `""` | mode dependent | Checkpoint to restart from. |
| `checkpoint_path`, `checkpoint_every` | `""`, 0 | mode dependent (`spinup.h5`, `short.h5`, `frozen.h5`, `resume.h5`), 200 | Where and how often to checkpoint. |
| `checkpoint_meta`, `callback`, `verbose`, `profile_steps` | empty, `nothing`, false, false | case metadata, frame writer, true, false | Checkpoint metadata, step hook, per-step lines, per-step timing. |
| `pic_diag`, `extdiag` | false, false | false, `CASE_EXTDIAG` | Nonlinear diagnostics, and the `resid_A` history column. |
| `steady_newton` | `:none` | `:none` | Optional steady Newton solve on the final state; `:auto`, `:map`, `:tosi` or `:dp`. |

The function returns `theta`, `t`, `steps`, `history`, `history_extra`, `qt`, `ss`, `ux`, `uz`, `pv`, `avg`, `steady`, `weak`, `chi` and `chi_prev`. `history` holds one row per step actually run, with the columns of section 6.2, and `history_extra` names the columns appended after the fixed 18.

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
| `bc` | `:free` | Free-slip walls; `:rigid` is no-slip. |
| `latent_L`, `latent_mode` | 0.11, `:per_phase` | Historical single coefficient, unused here, and the per-transition mode. |
| `yield_c`, `yield_phi`, `yield_taumax` | `Inf`, 0, `Inf` | Yield surface off. |
| `weak_f`, `weak_eps_ref` | 1, 0 | Strain weakening off. |
| `rho_heating`, `rho_cp` | false, false | Density-weighted heating off. |
| `Di`, `heating_adiabatic`, `heating_dissipation` | 0, false, false | Dissipation and adiabatic heating off. |
| `refstate`, `deviatoric` | `nothing`, false | Conductive base state, full strain rate. |
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

With `CASE_EXTDIAG=1` a `resid_A` column follows: the exact residual of the quoted pointwise Nusselt pair against the scheme's energy identity, per unit area, so it is zero for the time-step flux and $O(h)$ for the pointwise one.

### 6.3 Frames

One frame per `CASE_FRAME` steps, offset one step past every AMR cycle. The AMR pass zeroes the velocity, so a frame written on an AMR step would carry a zero velocity and a meaningless Nusselt number. A final frame is always written.

`frames/frame_NNNNN_T.bin`: the full temperature $T=\bar T+\theta$ on a 256 x 256 grid, little-endian Float64, 512 KB. `x` varies fastest, then `y`. In numpy that is `T = np.fromfile(path).reshape(256, 256)`, with `T[j, i]` the value at `(x_i, y_j)`, drawn with `origin="lower"`.

`frames/frame_NNNNN_mesh.csv`: one line per leaf, in leaf order (level, then `ix`, then `iy`), with the columns `level,x0,y0,h` at 17 significant digits. `h` is the leaf height, and `x0`, `y0` the lower-left corner.

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
| `ev` | Optional, one entropy-viscosity value per leaf, in `leaves` order. |
| `weak`, `weak_prev` | Optional, the strain-weakening field at its two levels. Absent on this case. |
| `cooldown_no_refine`, `cooldown_no_coarsen` | Optional, the balanced policy's hysteresis state, one column per leaf key. |
| Attributes | `t`, `dt`, `dt_prev`, `step`, `p`, `maxlevel`, `nx0`, `ny0`, `Lx`, `cooldown_cycle`, `velocity_stale`, the free-form metadata, and `config`: the trajectory-affecting settings as newline-separated `key=value` strings. |

`velocity_stale` marks a final checkpoint whose last step was an AMR step: the stored velocity is a diagnostic re-solve, and a restart zeroes it. A checkpoint is about 2 MB at the production state.

## 7. Restarting and the configuration guard

`restart_from` replaces the mesh, `p` and the initial state with the checkpoint contents before the first step, and `checkpoint_path` writes the continued run to its own file. `maxsteps` and `tstop` stay absolute. The resume command for this case is

```bash
CASE_OUT="$HOME/quast/run1" CASE_STEPS=4000 \
  julia --project=. --threads=8 examples/case_realistic_2d.jl resume \
  "$HOME/quast/run1/spinup.h5"
```

The guard compares the checkpoint's stored `config` against the settings of the live call, key by key, and refuses to start on any difference:

```text
ArgumentError: restart configuration mismatch in /path/resume.h5:
  symmetry: checkpoint=none, current=x
  wall_min_level: checkpoint=0.0, current=5.0
```

The compared keys are the ones that affect the trajectory: AMR cadence, passes, fractions, leaf cap, level cap, wall rule, hotspot and top-wall rules, symmetry, transfer mode, hysteresis and gate; advection and the entropy-viscosity coefficients; the Picard and time-step controls `maxpic`, `pic_tol`, `pic_relax`, `pic_mode`, `pic_reject`, `cfl`, `dtmax` and `dtmin`; `mass_form` and `deviatoric`; the heating flags and `H`; the rheology parameters of the active law (`comp_*` and `eta_*` here, the `tosi_*` block for that law); the yield and weakening parameters; `latent_L`, `latent_mode`, `latent_scheme` and the phase list; the reference state; and `stokes_scaling`. The comparison iterates over the checkpoint's keys, so a key added after a checkpoint was written is not compared. `stokes_scaling` has a defined default: a file without it reads as `none`, and switching the option on for such a chain is refused.

Some settings are state rather than configuration, so they are not compared: `p` and the mesh are restored from the file, and the balanced cooldown is restored with them. `dt0`, `dt_growth`, `initial_refine`, `linear`, `mg_tol`, `mg_threads` and the rest of the solver's settings are neither stored nor compared, so the same trajectory can be continued on a different linear path or with a different checkpoint cadence.

When a resume is refused, repeat the stored values, or use a seed mode. A seed mode copies the checkpoint and rewrites exactly the conflicting strings: `freeze` sets `amr_every = 0`, `wallrefine` sets `wall_min_level`, `wall_strip`, `max_leaves`, `ev` and `symmetry` and refines the wall, and `symseed` sets `symmetry`. The source file is never modified.

## 8. Threads and memory

Start Julia with `--threads=N`. The only part of this case that uses the Julia thread pool is the matrix-free V-cycle that `linear = :auto` selects, and it is capped at eight threads, `mg_threads = min(8, Threads.nthreads())`, so more than eight adds nothing on that path. With one thread the auto path warns once and runs serial. An explicit `linear = :mg` keeps `mg_threads = 1` unless the caller sets it, and the direct factorization is serial in every case.

At the 10 000-leaf cap, the production state peaks at about 1.3 GB resident on the multigrid path with eight threads, and about 2.4 GB with the direct factorization. Both figures are process peaks, so they include the Julia runtime and the loaded state. Plan on 2 GB of free memory for a default run and 3 GB for the direct path.

Disk is dominated by frames: about 0.8 MB each, a 512 KB `T.bin` plus a mesh CSV of a few hundred kB. A 20 000-step run at stride 5 costs about 3 GB, and a checkpoint is about 2 MB.

## 9. The engineering record

The measurements behind the defaults, the defects found along the way, the optimization work, and the validation against benchmarks and ASPECT are kept in [the engineering record](../dev/). Start with [the solver volume](../dev/SOLVER.md) for solver measurements, [the AMR volume](../dev/AMR.md) for the adaptive policy, and [the validation volume](../dev/VALIDATION.md) for the validation record.
