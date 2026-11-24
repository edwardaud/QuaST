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
| `constitutive.jl` | Viscosity bounds and the `:expT`, `:expTz`, `:arrhenius` and `:composite` laws. |
| `energy.jl` | The BDF2 energy step on the quadtree mesh and its wall-flux condition (box bodies verbatim; a mapped geometry has its own fixed-dof, side-temperature and wall-flux methods). |
| `energy_assembly.jl` | The energy operator: mass, diffusion, advection and the entropy-viscosity/source terms. The box leaf kernel is the historical body; the mapped leaf kernel (cut 3) routes the quadrature, the base state and the second derivatives through the geometry. |
| `entropy_viscosity.jl` | Entropy viscosity on the quadtree (ASPECT port), including the recorded field. The mapped method takes gradients/Laplacians from the geometry (map curvature), the base state from the physical height and the element length from the mapped area. |
| `eta.jl` | The visco-plastic branch: yield stress, plastic viscosity and the strain-rate field. |
| `eta_laws.jl` | The `semquad_eta_at` viscosity closures and their analytic derivatives. |
| `eta_plastic.jl` | The Drucker-Prager branch and the strain-rate helpers it needs. |
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
| `phase_buoyancy.jl` | The phase-anomaly buoyancy force on the vertical momentum. |
| `refstate.jl` | The closed-form/EOS reference state: adiabat, rho, alpha and cp. |
| `revision.jl` | `PHYSICS_REVISION`: the dated changelog of behaviour changes a checkpoint must not cross, plus the hashed list of the behaviour-defining sources that makes forgetting to bump it fail G252. |
| `tosi_law.jl` | The Tosi et al. (2015) viscoplastic law and its derivatives. |
| `weakening.jl` | Strain weakening: the advected chi field, its source and its JVP (including the mapped `d eta/d chi` momentum block). |

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
