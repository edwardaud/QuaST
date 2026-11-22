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
| `energy.jl` | The BDF2 energy step on the quadtree mesh and its wall-flux condition. |
| `energy_assembly.jl` | Per-block buffers for the threaded energy assembly. |
| `eta.jl` | The visco-plastic branch: yield stress, plastic viscosity and the strain-rate field. |
| `eta_laws.jl` | The `semquad_eta_at` viscosity closures and their analytic derivatives. |
| `eta_plastic.jl` | The Drucker-Prager branch and the strain-rate helpers it needs. |
| `eta_weak_zone.jl` | The weak-zone multiplier and its activity predicates. |
| `latent.jl` | Per-transition latent-heat coefficients and the apparent heat capacity. |
| `params.jl` | `Params`: the physical/numerical card, base state and viscosity-law selection. |
| `phase.jl` | `PhaseTransition`, the phase function and its temperature/depth derivatives. |
| `phase_buoyancy.jl` | The phase-anomaly buoyancy force on the vertical momentum. |
| `refstate.jl` | The closed-form/EOS reference state: adiabat, rho, alpha and cp. |
| `revision.jl` | `PHYSICS_REVISION`: the dated changelog of behaviour changes a checkpoint must not cross, plus the hashed list of the behaviour-defining sources that makes forgetting to bump it fail G252. |
| `tosi_law.jl` | The Tosi et al. (2015) viscoplastic law and its derivatives. |
| `weakening.jl` | Strain weakening: the advected chi field, its source and its JVP. |
