# Physics

The material and energy physics, evaluated at quadrature points or assembled
into the energy operator. `params.jl` is the parameter card every law reads; the
others hold the viscosity laws and their bounds, the Drucker–Prager plastic
branch, strain weakening, the phase transitions and their latent heat, the
reference state and EOS, and the BDF2 energy step. Stokes assembly is in
`stokes/`, the time loop in `driver/` and the mesh policy in `amr/`; none of
those belong here.

| File | What it is |
|---|---|
| `annulus_base.jl` | Stage 0.5b1: the geometry-dependent conductive base state. The annulus's radial log/H profile `Tbar(r) = (1 - H(R1^2-R0^2)/4) ln(R1/r)/ln(R1/R0) + (H/4)(R1^2-r^2)` and its derivative, the `semquad_base_T`/`semquad_base_gradient`/`semquad_base_wall_refs` geometry dispatch (box/validation geometries keep the historical quadratic-in-`h` values), and the annulus `_mapped_base_curvature` (identically zero: `lap(Tbar) = -H`).  Stage 0.5b2 adds the documented annular initial conditions: `_annulus_ic_total` (`:firedrake` = Davies et al. 2022 Eq. (41), `:aspect` = the ASPECT spherical hexagonal perturbation at angular mode 4) and `_annulus_ic_theta` = the total temperature minus `Tbar(r)`, the driver's `annulus_ic` knob. |
| `constitutive.jl` | Viscosity bounds and the `:expT`, `:expTz`, `:arrhenius` and `:composite` laws. |
| `energy.jl` | The BDF2 energy step on the quadtree mesh and its wall-flux condition. The mapped fixed-dof method uses the geometry's `nu_wall_sides` (the annulus's circles) and the geometry's base state at the wall node; the box body keeps its historical `y` rows. The mapped prescribed-flux branch (`flux_top`/`flux_bot`) likewise selects the geometry's wall side (stage-0.5 fix campaign, audit defect D4: it used to put the annulus flux on the logical `y` seam cuts while the circle kept its natural zero-flux condition; the load now lands on the circle with the analytic integral `q*2πR0`, G279). |
| `energy_assembly.jl` | The energy operator: mass, diffusion, advection and the entropy-viscosity/source terms. The assembly entry point refuses a non-finite temperature field (	heta, 	heta_prev, 	heta_lat) and a non-finite supplied EV field with a named ArgumentError (allocation-free scan, AUDIT_VV F5). The box leaf kernel is the historical body; the mapped leaf kernel (cut 3; stage 0.5b1 for the base state) routes the quadrature, the geometry's conductive base state (the annulus's radial log/H profile) and the second derivatives through the geometry. |
| `entropy_viscosity.jl` | Entropy viscosity on the quadtree (ASPECT port), including the recorded field. The mapped method takes gradients/Laplacians from the geometry (map curvature), the base state from the geometry's `semquad_base_T(pr,g,x,y)`/`semquad_base_gradient(pr,g,x,y)` (the annulus's radial log/H profile) and the element length from the mapped area. |
| `eta.jl` | The visco-plastic branch: yield stress, plastic viscosity and the strain-rate field. |
| `eta_laws.jl` | The `semquad_eta_at` viscosity closures and their analytic derivatives. Since stage 0.5b1 the mapped bodies evaluate the base temperature and the law's depth argument at the geometry's physical height (`height(g,x,y)`; the annulus gives `r - R0`), so depth-dependent laws work on the annulus with no parameter change. The **box viscosity row is frozen**: it keeps `Tbar(y,H)`/`y` verbatim even with a reference state (docs/dev/PHYSICS.md §3.3 — the buoyancy/energy use the adiabat, the viscosity does not; G197 pins it). |
| `eta_plastic.jl` | The Drucker-Prager branch and the strain-rate helpers it needs; the nodal plastic-strain-rate field evaluates the base state/depth through the geometry (stage 0.5b1). |
| `eta_weak_zone.jl` | The weak-zone multiplier and its activity predicates. |
| `latent.jl` | Per-transition latent-heat coefficients and the apparent heat capacity. |
| `maxwell_band.jl` | The validity band of the linearised co-rotational rotation. |
| `maxwell_force.jl` | The lagged Maxwell stress as the momentum equation sees it. |
| `maxwell_kernels.jl` | The Maxwell relaxation kernel and the co-rotational increment. |
| `maxwell_reference.jl` | The analytic Maxwell solutions the gates converge to. |
| `maxwell_regime.jl` | The material/effective viscosity regime of a viscoelastic run. |
| `maxwell_softening.jl` | The effective-viscosity softening for the Newton viscosity factory. |
| `params.jl` | `Params`: the physical/numerical card, base state and viscosity-law selection. |
| `phase.jl` | `PhaseTransition`, the phase function and its temperature/depth derivatives. |
| `phase_buoyancy.jl` | The phase-anomaly buoyancy force on the upward momentum. The mapped method (stage 0.5b1) evaluates the phase depth at the physical height (`z0` values are heights: on the annulus `r - R0`) and the comparison base state at the geometry's base profile. |
| `refstate.jl` | The closed-form/EOS reference state: adiabat, rho, alpha and cp, the scalar base state `Tbar(z,H)` and the mapped base-curvature helper. |
| `revision.jl` | `PHYSICS_REVISION`: the dated changelog of behaviour changes a checkpoint must not cross, plus the hashed list of the behaviour-defining sources that makes forgetting to bump it fail G252. |
| `tosi_law.jl` | The Tosi et al. (2015) viscoplastic law and its derivatives. |
| `weakening.jl` | Strain weakening: the advected chi field, its source and its JVP (including the mapped `d eta/d chi` momentum block). The source's temperature/depth now come from the geometry (stage 0.5b1). |

The energy path carries the geometry in two ways.  (1) Its **quadrature data**
come from the leaf metric (`quad_point`, `quad_measure`, `basis_grad`), so the
mass, diffusion, advection and load integrals are the physical integrals of the
mapped element.  (2) Its **physics closures are functions of the physical
height** `h = height(g,x,y)` instead of `y`, and every directional term uses
`u . grad h` instead of `u_z`: the base-state advection `-rho (u.grad h)
dTbar/dh`, the adiabatic heating `-Di rho T (u.grad h)`, the latent source
`-L dGamma/dh (u.grad h)`, and the mapped base-state Laplacian
`lap(Tbar(h)) + H = Tbar''(h)|grad h|^2 + Tbar'(h) lap(h) + H`, which is the
`_mapped_base_curvature(pr,g,x,y)` helper (the energy assembly adds it as a
source and the entropy-viscosity residual subtracts it, so the two cannot
drift; cut-4 defect B1 fixed the cut-3 body that used the reference-state
second derivative alone and so silently dropped the Boussinesq
`H(1-|grad h|^2)` term on a geometry with `|grad h| != 1`).  The
entropy-viscosity residual uses the map-curvature
Laplacian of the perturbation and the mapped element length
`he = sqrt(2|element|)`; the box path keeps its historical `(2/h)^2` Laplacian
and `sqrt(2)*sqrt(hx*hy)` length (GEOM_DESIGN E14/F4).  The box-dispatched
bodies are the historical statements character for character and the nine
state-dump hashes pin them; the mapped bodies are selected once per assembly
(`assemble_semquad_energy`, `solve_semquad_energy`, `semquad_ev_*`,
`assemble_semquad_energy_latent_jvp`, `assemble_semquad_deta_chi`,
`_semquad_jvp_ku`).  `test/t_g269_curved_energy.jl` is the gate: affine
exactness, CurvedBox convergence orders, the mapped-vs-box operator
cross-check and the map-curvature wrong-term catches.

**Stage 0.5b1: the base state is a function of the geometry.**  The energy
operator and every mapped physics body ask for the base temperature through
`semquad_base_T(pr, g, x, y)` and its height derivative through
`semquad_base_gradient(pr, g, x, y)` (`annulus_base.jl`), not through the
Cartesian `semquad_base_T(pr, h)`: the box and the box-in-height validation
geometries dispatch to exactly the historical quadratic-in-`h` state, while the
2-D polar annulus dispatches to the radial conduction solution
`Tbar(r) = (1 - H(R1^2-R0^2)/4) ln(R1/r)/ln(R1/R0) + (H/4)(R1^2-r^2)`.  That
profile satisfies `Tbar'' + Tbar'/r = -H` identically, so the `+H` volumetric
source is absorbed exactly as in the box and the annulus's
`_mapped_base_curvature` is *exactly zero*.  `semquad_base_wall_refs(pr, g)`
returns the wall reference gradients the Nusselt normalisation divides by; the
annulus evaluates them at the exact radii `R0`/`R1`, which is the ANNULUS_REFS
per-wall convention.  The depth-dependent viscosity/plastic/weakening and
phase closures evaluate their height argument at `height(g,x,y)`; the annulus's
is `r - R0` (the physics length unit is the gap).
