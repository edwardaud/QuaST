# Architecture

`src/` is a single Julia module. `src/QuaST.jl` is its body: it fixes the include order and the export list, and every other file is loaded by it. The eight folders hold 83 Julia files and 21 021 lines; `QuaST.jl` is the eighty-fourth file, with 224 lines. The tree is organized by the object a file acts on — the grid knows nothing about the rheology, and the rheology knows nothing about the Stokes solver. The five layers below are a second reading of the same tree: they order the code by what may call what.

## The five layers

| Layer | What it owns | Where it lives |
|---|---|---|
| Data and mathematical primitives | GLL nodes, weights and the Gauss–Legendre rule; the reference element; the 2:1-balanced quadtree; the scalar `Q_p` space, its mass matrix and its hanging-node constraints; point evaluation; inter-mesh transfer; the threading helpers | `core/` except `poisson.jl` |
| Discretization | The weak forms and the discrete objects built from them: the Taylor–Hood reference-element matrices and leaf assembly, the quadrature divergence and compressibility blocks, the matrix-free velocity stiffness, the `SemStokes` state, the BDF2 energy operator and its entropy-viscosity field, the gradient-jump indicator | `stokes/` (state, assembly, operators, `kop.jl` and `apply.jl`), `physics/` (`energy_assembly.jl`, `energy.jl`, `entropy_viscosity.jl`), `amr/indicator.jl`, `core/poisson.jl` |
| Linear solvers | The `:auto` selector, the sparse direct saddle with its shift ladder, the multigrid hierarchy with its V-cycle, cached levels and pressure block, and the Krylov solver around it | `stokes/` (`selector.jl`, `direct.jl`, `diag.jl`, the solve paths), `mg/` |
| Physics | The parameter card, the viscosity laws and their derivatives, the Drucker–Prager and weak-zone branches, the phase transitions and their latent heat, the reference state and equation of state, the Maxwell kernels and regime diagnostics, strain weakening | `physics/` except the three discretization files |
| Driver and services | The time loop and the per-step nonlinear solvers, the `eta_ref` and Maxwell reports, the balanced AMR policy, the HDF5 checkpoint and its configuration codec, the acceptance diagnostics | `driver/`, `amr/` except `indicator.jl`, `io/`, `diagnostics/` |

A file belongs to the discretization layer when it assembles, applies or evaluates a discrete operator, or holds the state of one. It belongs to the physics layer when it states a material or thermodynamic relation on its arguments. A folder groups files by the object they act on, so one folder can feed more than one layer: `stokes/` and `physics/` each feed two layers, `amr/` gives its indicator to the discretization layer and its policy to the driver layer, and `core/` holds the primitives with one verification operator beside them.

## The folders

| Folder | Files | Layer | What belongs there |
|---|---:|---|---|
| `core/` | 11 | 1, and `poisson.jl` in 2 | GLL and quadrature, `FEData`, the quadtree with its face table and balancer, `SemQuad` with its constraints, point evaluation, transfer, threading |
| `physics/` | 23 | 4, and three files in 2 | `Params` and the material laws, yielding, weak zones, phases and latent heat, reference state and EOS, Maxwell kernels, weakening, the energy assembly and step, entropy viscosity |
| `stokes/` | 13 | 2 and 3 | `SemStokes`, the reference-element and variable-viscosity assembly, the matrix-free operator, the direct solve, the Jacobi diagonal, the `:auto` selector, the solve entry points |
| `mg/` | 9 | 3 | The hierarchy, the level cache, the V-cycle, the pressure block, the BFBT diagonal, the inner solves and FGMRES |
| `amr/` | 9 | 5, and `indicator.jl` in 2 | The gradient-jump indicator, the leaf primitives and hard budget, the coarsening gate and plan, the balanced cycle, the cooldown, the hooks and wall refinement |
| `driver/` | 12 | 5 | `run_semquad_simulation`, the Picard and Newton paths, the strain-rate and Maxwell driver state, the Maxwell and `eta_ref` reports, the restart guard, the history row |
| `io/` | 4 | 5 | The HDF5 checkpoint, its configuration codec, the cooldown datasets and the tree rebuild |
| `diagnostics/` | 2 | 5 | The acceptance quantities — wall Nusselt numbers, Vrms, mean temperature, work and dissipation, the energy budget — and the consistent boundary flux |

Outside the module are the `examples/` cards, the `test/` gates and their runner, the `benchmarks/` timings, the `scripts/` campaign and documentation tools, and the `aspect_runs/` inputs and plugin. None of them is loaded by `QuaST`; the Julia ones call it, and `aspect_runs/` belongs to the other code in the comparison.

## Include order

`include` is textual insertion into the module body, so the order is constrained only by definition-time references: a struct field type, a supertype, a method signature, or a `const` initializer that names a type from another file. A function body may call anything defined later, which is why `driver/run.jl` can call `save_semquad_state` although `io/` is included after it.

```text
physics/     revision tosi_law refstate params constitutive phase
core/        gll fe quadtree quadtree_table quadtree_balance semquad
             parallel constraints poisson eval transfer
amr/         leaf_ops indicator
physics/     latent entropy_viscosity energy_assembly energy eta_weak_zone
             eta_plastic eta_laws eta maxwell_kernels maxwell_band
             maxwell_regime maxwell_reference maxwell_force
             maxwell_softening viscoelastic weakening phase_buoyancy
amr/         cooldown coarsen_gate coarsen_plan cycle
stokes/      state selector direct assembly kop apply diag solve_const
             assembly_var solve_var_core solve_var_fallback solve_var operators
mg/          hierarchy levelcache build vcycle pressure bfbt saddle s1solve gmres
amr/         wall_refine amr_pass hooks
driver/      strain_rate picard_control picard newton_tangent step_newton
             maxwell_state maxwell_reports eta_ref_report restart_guard
             diagnostics_row run mass
io/          checkpoint_cooldown checkpoint_config checkpoint_tree checkpoint
diagnostics/ global flux
```

The two `physics/` blocks are not a duplication. The first holds the type layer: `TosiLaw` and `SemRefState` are field types of `Params` in `physics/params.jl`, so `tosi_law.jl` and `refstate.jl` precede `params.jl`, and `constitutive.jl` and `phase.jl` follow it because their methods annotate `pr::Params`. The second block holds the rest of the physics: the energy assembly and step, the laws that take a `SemQuad`, and the Maxwell kernels. Two other seams are visible: `physics/entropy_viscosity.jl` precedes `physics/energy_assembly.jl`, whose `evg` argument is annotated `SEMEVGlobals`; and the `amr/` hooks block follows `stokes/` because `semquad_amr_passes!` names `SemStokes` in its signature. The blocks otherwise follow the layers: primitives, then the operators and laws, then the solvers, then the hooks and the driver.

## The dependency direction

Apart from the accepted cycles below, the calls between layers run down and not back up.

| Caller | Depends on | The boundary |
|---|---|---|
| `driver/` | every other layer | nothing calls the driver |
| `amr/` | `core/`; `stokes/` in `amr_pass.jl`, which rebuilds `SemStokes` after a cycle | assembles no operator |
| `mg/` | `stokes/` for the matrix-free operator, `core/` | no physics, and no direct factorization of the fine saddle |
| `stokes/` | `core/`; `mg/` for the preconditioner; the material closures handed in by the driver | does not build `Params` and does not own the time step |
| `physics/` | `core/` in the files that evaluate or assemble a field | assembles no Stokes block |
| `io/` | `core/` for the tree rebuild | writes no frames and no history; the case scripts do |
| `diagnostics/` | `core/`, the card it reads, and the solved fields | changes no state |

## The three deliberate cycles

### The matrix-free operator and the multigrid preconditioner

`stokes/kop.jl` and `stokes/apply.jl` call into the solver layer — `mg/levelcache.jl` for the cached level data and `mg/vcycle.jl` for the preconditioner hook — and `mg/` calls back into the operator to apply it on every level. The cycle is the structure of matrix-free multigrid: the preconditioner is a V-cycle over the operator, so it cannot be expressed through an interface that hides the operator's per-level kernels without giving up being matrix-free. deal.II's `MatrixFreeOperators`/`MGTransfer` pair and ASPECT's `StokesMatrixFree`/`StokesPreconditioner` split have the same shape. The cycle is accepted with two containment rules: the calls out of `stokes/` are confined to `kop.jl` and `apply.jl`, and `stokes/solve_var_core.jl` remains the only place that decides which linear path runs.

### The quadtree as one object in three files

`core/quadtree.jl`, `core/quadtree_table.jl` and `core/quadtree_balance.jl` are three views of one object: the refinement primitives maintain the flat face table, the table build asks the balancer whether a leaf set is legal, and the balancer splits leaves. The call graph between the three is cyclic by construction. The cycle is accepted because the three files share one leaf representation and one legality rule, so no interface can separate them; the split buys readability only, so a change to any of them is a change to the tree itself and the three are reviewed together.

### The coarsening gate and the plan as one policy

`amr/coarsen_gate.jl` and `amr/coarsen_plan.jl` are two halves of one policy and call each other: the plan asks the gate whether a pair may merge, and the gate reads the plan's ledger to decide. The cycle is accepted because the two files implement one policy; they are separate files only because the gate is the part that the card can switch off (`amr_coarsen_gate`). Any change to the merge rules belongs in both.

## Diagnostics are out-parameters

The module reports what a call did through module-level `Ref`s and counters: the `LAST_SEMSTOKES_*` family (for example in `stokes/state.jl`, `stokes/direct.jl` and `stokes/selector.jl`, with the pressure records in `mg/pressure.jl`); `LAST_AUTO_FALLBACK`; the `SEMSTOKES_*` assembly and pressure counters; the AMR records `LAST_AMR_STATS`, `LAST_AMR_BALANCED`, `LAST_BUDGET_PASS` and `LAST_MIRROR_CLOSE`; and the per-run records `SEMQUAD_STEP_PROFILE`, `SEMQUAD_PIC_DIAG` and `SEMQUAD_AMR_COOLDOWN`. These are out-parameters. The callee writes what it did and the caller reads it after the call, to log, to guard a restart, to choose what to write, or to abort with the numbers that identify the state. The solver's per-call records are write-once: a solve call sets each of them once, and no arithmetic inside a solve reads one back. The per-run records are updated across the run and read by the driver afterwards. A value that has to influence the next solve is passed in an argument, never published in this family. Settings are kept apart from records: `SEMQUAD_MG_LEVEL_CACHE`, `SEMQUAD_MG_SERIAL_DOFS` and the region switches `SEMQUAD_MG_REGION*` are inputs that the solver reads, while the `LAST_*` family records what it did.

## The flow of one time step

The flow is quasi-static: the driver solves the Stokes saddle, advances the energy equation with that velocity, and repeats. One step, from `run_semquad_simulation` down and back:

```text
run_semquad_simulation                              driver/run.jl
  └─ while step < maxsteps && t < tstop
       ├─ Picard or Newton iteration                driver/run.jl
       │    ├─ solve_flow!                          driver/run.jl
       │    │    ├─ viscosity closure               physics/ (laws, Maxwell)
       │    │    ├─ buoyancy and phase anomaly      driver/run.jl
       │    │    └─ solve_semquad_stokes_var        stokes/solve_var.jl
       │    │         ├─ semquad_choose_linear      stokes/selector.jl
       │    │         ├─ direct: LU and shifts      stokes/direct.jl
       │    │         └─ mg: FGMRES over V-cycle    mg/
       │    └─ solve_semquad_energy                 physics/energy.jl
       ├─ Maxwell stress (if G > 0), t and step advance driver/run.jl
       ├─ _step_diagnostics -> history row          driver/diagnostics_row.jl
       ├─ semquad_amr_passes!                       amr/amr_pass.jl
       ├─ callback -> frames and logs               examples/case_realistic_2d.jl
       └─ save_semquad_state                        io/checkpoint.jl
```

1. The step starts by copying the start-of-step temperature and weakening field, then enters the Picard loop (`pic_mode = :picard`, up to `maxpic` iterations; one on the production card). Each iteration calls `solve_flow!`, which builds the viscosity closure: temperature from the current iterate, the strain rate from the current velocity for the stress-dependent laws, the weak-zone factor, the `eta/eta_ref` normalization, and the Maxwell effective viscosity when the shear modulus is positive. It then assembles the vertical buoyancy `Ra theta` plus the phase anomaly and calls `solve_semquad_stokes_var`, which resolves `:auto` before assembly, runs the direct or multigrid path, and publishes the residual and the chosen path in the `LAST_SEMSTOKES_*` records. The driver compares `LAST_SEMSTOKES_RESIDUAL` with `stokes_tol` and, on a miss, writes the identifying numbers and aborts the step.
2. On the first iteration the CFL block updates `dt` from the maximum velocity and the smallest leaf height. The update sits inside the loop because the AMR pass zeroes the velocity, so an end-of-step update could only shrink `dt`.
3. `solve_semquad_energy` assembles the mass, stiffness, advection and source blocks for the temperature with the velocity just solved, adds the entropy-viscosity diffusion, the latent apparent capacity and the radiogenic decay, and applies the variable-step BDF2 update. With strain weakening on, one BDF2 transport for the weakening field follows the same pattern.
4. The Picard loop compares the new temperature with its iterate, under-relaxes, and stops below `pic_tol`; the optional trial rejection restores the previous flow state and halves the relaxation. `pic_mode = :newton` replaces the loop with a damped Newton on the coupled per-step residual (`driver/step_newton.jl`, `driver/newton_tangent.jl`) and falls back to Picard if the Newton fails.
5. After convergence, the Maxwell stress advances one backward-Euler step with the material viscosity of the step when the shear modulus is positive. The BDF2 temperature levels rotate, `t` and `step` advance, and `_step_diagnostics` builds the history row from the acceptance quantities in `diagnostics/global.jl` and the consistent-boundary-flux pair in `diagnostics/flux.jl`.
6. Every `amr_every` steps, `semquad_amr_passes!` (`amr/amr_pass.jl`) marks leaves with the indicator, runs the balanced cycle, enforces the depth and leaf caps, rebuilds `SemStokes`, and transfers the temperature levels, the weakening field and the stress history; the velocity is zeroed and the stale multigrid tree entries are dropped. The case callback writes frames and the log, the optional checkpoint writes the full state, and the steady-state test decides whether to stop.
7. The next step starts from what this one leaves: the rotated temperature levels, the transferred weakening and stress fields, and the current mesh. The cycle closes.

**Next:** read [src/README.md](../src/README.md) and `src/QuaST.jl` alongside this page, then the `README.md` of the folder you are about to change.
