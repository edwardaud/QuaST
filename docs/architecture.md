# Architecture

`src/` is a single Julia module. `src/QuaST.jl` is its body: it fixes the include order and the export list, and every other file is loaded by it. The eight folders hold 89 Julia files and 31 617 lines; `QuaST.jl` is the ninetieth file, with 265 lines. The discrete objects are Taylor–Hood `p`/`p - 1` pairs on Gauss–Legendre–Lobatto nodes, integrated with Gauss–Legendre quadrature at `p + 2` points per direction and no mass lumping; the default is `p = 2`. The tree is organized by the object a file acts on: the grid knows nothing about the rheology, and the rheology knows nothing about the Stokes solver. The six layers below are a second reading of the same tree; they order the code by what may call what.

The package needs one external dependency, `HDF5.jl` (0.17.4, pinned by `Project.toml` and the committed `Manifest.toml`), for the checkpoints. The other entries in `Project.toml` are the standard library: the module body imports `LinearAlgebra`, `SparseArrays` and `Printf`, the test runner imports `Test`, and the test and benchmark scripts use `Random`; `SHA` is declared as well.

## The six layers

| Layer | What it owns | Where it lives |
|---|---|---|
| Data and mathematical primitives | GLL nodes, weights and the Gauss–Legendre rule; the reference element; the 2:1-balanced quadtree; the scalar `Q_p` space, its mass matrix and its hanging-node constraints; point evaluation; inter-mesh transfer; the threading helpers; the named finite/positive input guards | `core/` except `geometry.jl` and `poisson.jl` |
| Geometry and mapping | The geometry interface `AbstractGeometry` and its four concrete types: the identity-mapped `CartesianBox`, the validation maps `AffineBox` and `CurvedBox`, and the 2-D polar `PolarAnnulus`; the global map, its analytic inverse and its derivatives; the leaf and face metrics; the logical boundary registry; the geometry identity a checkpoint records; the geometry-dependent conductive base state | `core/geometry.jl`, `physics/annulus_base.jl` |
| Discretization | The weak forms and the discrete objects built from them: the Taylor–Hood reference-element matrices and leaf assembly, the quadrature divergence and compressibility blocks, the boundary-traction face load, the matrix-free velocity stiffness, the `SemStokes` state, the BDF2 energy operator and its entropy-viscosity field, the gradient-jump indicator — all evaluated through the geometry's metrics | `stokes/` (state, assembly, operators, `kop.jl` and `apply.jl`), `physics/` (`energy_assembly.jl`, `energy.jl`, `entropy_viscosity.jl`), `amr/indicator.jl`, `core/poisson.jl` |
| Linear solvers | The `:auto` selector, the sparse direct saddle with its shift ladder, the multigrid hierarchy with its V-cycle, cached levels and pressure block, and the Krylov solver around it | `stokes/` (`selector.jl`, `direct.jl`, `diag.jl`, the solve paths), `mg/` |
| Physics | The parameter card, the viscosity laws and their derivatives, the Drucker–Prager and weak-zone branches, the phase transitions and their latent heat, the reference state and equation of state, the Maxwell kernels and regime diagnostics, strain weakening | `physics/` except the three discretization files and `annulus_base.jl` |
| Driver and services | The time loop and the per-step nonlinear solvers, the AMR step of the loop and the driver-owned diagnostics snapshot, the `eta_ref` and Maxwell reports, the balanced AMR policy, the HDF5 checkpoint and its configuration codec, the acceptance diagnostics | `driver/`, `amr/` except `indicator.jl`, `io/`, `diagnostics/` |

A file belongs to the discretization layer when it assembles, applies or evaluates a discrete operator, or holds the state of one. It belongs to the physics layer when it states a material or thermodynamic relation on its arguments. It belongs to the geometry layer when it defines the map between the logical reference square and physical space, or a quantity that depends on that map (a metric, a wall normal, the base state of a mapped domain); the layer is separate from the primitives because a leaf metric is a property of the map, not of the tree. A folder groups files by the object they act on, so one folder can feed more than one layer: `stokes/` and `physics/` each feed two layers, the geometry layer takes one file from `core/` and one from `physics/`, `amr/` gives its indicator to the discretization layer and its policy to the driver layer, and `core/` holds the primitives, the geometry layer and one verification operator.

## The folders

| Folder | Files | Lines | Layer | What belongs there |
|---|---:|---:|---|---|
| `core/` | 13 | 5 501 | 1, `geometry.jl` in 2, `poisson.jl` in 3 | GLL and quadrature, `FEData`, the finite/positive input guards, the quadtree with its face table and balancer, the geometry interface with its map, metrics and boundary registry, `SemQuad` with its constraints, point evaluation, transfer, threading |
| `physics/` | 24 | 5 674 | 5, `annulus_base.jl` in 2, three files in 3 | `Params` and the material laws, yielding, weak zones, phases and latent heat, reference state and EOS, the annulus radial base state, Maxwell kernels, weakening, the energy assembly and step, entropy viscosity |
| `stokes/` | 14 | 7 725 | 3 and 4 | `SemStokes`, the reference-element and variable-viscosity assembly, the boundary-traction load, the curved-wall constraint rows and rigid-mode analysis, the matrix-free operator, the direct solve, the Jacobi diagonal, the `:auto` selector, the solve entry points |
| `mg/` | 9 | 3 662 | 4 | The hierarchy, the level cache, the V-cycle, the pressure block, the BFBT diagonal, the inner solves and FGMRES |
| `amr/` | 9 | 2 134 | 6, and `indicator.jl` in 3 | The gradient-jump indicator, the leaf primitives and hard budget, the coarsening gate and plan, the balanced cycle, the cooldown, the hooks and wall refinement |
| `driver/` | 14 | 4 861 | 6 | `run_semquad_simulation`, the Picard and Newton paths, the strain-rate and Maxwell driver state, the Maxwell and `eta_ref` reports, the restart guard, the driver-owned diagnostics snapshot (`run_diagnostics.jl`), the AMR step of the loop (`amr_step.jl`), the history row |
| `io/` | 4 | 468 | 6 | The HDF5 checkpoint, its configuration codec (including the geometry identity), the cooldown datasets and the tree rebuild |
| `diagnostics/` | 2 | 1 592 | 6 | The acceptance quantities — wall Nusselt numbers, Vrms, mean temperature, work and dissipation, the energy budget — and the consistent boundary flux |

The counts are Julia files and their line counts (`wc -l`); each folder also carries a `README.md`.

Outside the module are the `examples/` cards, the `test/` gates and their runner, the `benchmarks/` timings and the archived benchmark-evidence records in `benchmarks/results/`, the `scripts/` campaign and documentation tools, and the `aspect_runs/` inputs and plugin. None of them is loaded by `QuaST`; the Julia ones call it, and `aspect_runs/` belongs to the other code in the comparison.

## Include order

`include` is textual insertion into the module body, so the order is constrained only by definition-time references: a struct field type, a supertype, a method signature, or a `const` initializer that names a type from another file. A function body may call anything defined later, which is why `driver/run.jl` can call `save_semquad_state` although `io/` is included after it.

```text
physics/     revision tosi_law refstate params constitutive phase
core/        gll guards fe quadtree quadtree_table quadtree_balance geometry
physics/     annulus_base
core/        semquad parallel constraints poisson eval transfer
amr/         leaf_ops indicator
physics/     latent entropy_viscosity energy_assembly energy eta_weak_zone
             eta_plastic eta_laws eta maxwell_kernels maxwell_band
             maxwell_regime maxwell_reference maxwell_force maxwell_softening
             viscoelastic weakening phase_buoyancy
amr/         cooldown coarsen_gate coarsen_plan cycle
stokes/      state selector direct assembly kop apply diag solve_const
             assembly_var wall_constraints solve_var_core solve_var_fallback
             solve_var operators
mg/          hierarchy levelcache build vcycle pressure bfbt saddle s1solve gmres
amr/         wall_refine amr_pass hooks
driver/      strain_rate picard_control picard newton_tangent step_newton
             maxwell_state maxwell_reports eta_ref_report restart_guard
             diagnostics_row run_diagnostics amr_step run mass
io/          checkpoint_cooldown checkpoint_config checkpoint_tree checkpoint
diagnostics/ global flux
```

The opening `physics/` block holds the type layer: `TosiLaw` and `SemRefState` are field types of `Params` in `physics/params.jl`, so `tosi_law.jl` and `refstate.jl` precede `params.jl`, and `constitutive.jl` and `phase.jl` follow it because their methods annotate `pr::Params`. `physics/revision.jl` opens the sequence: it is a bare constant with no dependencies. The second `physics/` block holds the rest: the energy assembly and step, the laws that take a `SemQuad`, and the Maxwell kernels.

The geometry layer interrupts the `core/` block for a reason. `core/geometry.jl` follows the quadtree because `CartesianBox(qt)` names `Quadtree`, and it must precede `core/semquad.jl`, whose constructors call the geometry accessors. `physics/annulus_base.jl` follows `core/geometry.jl` because its methods dispatch on the concrete geometry types and extend the base-state functions of `physics/refstate.jl`; nothing else constrains its position, because nothing earlier calls it at definition time. `core/guards.jl` has no dependency at all and is included as early as possible, right after `core/gll.jl`.

Three further seams are visible. `physics/entropy_viscosity.jl` precedes `physics/energy_assembly.jl`, whose `evg` argument is annotated `SEMEVGlobals`. The `amr/` hooks block follows `stokes/` because `semquad_amr_passes!` names `SemStokes` in its signature. `driver/run_diagnostics.jl` follows every producer because its field types are read from the producer Refs at include time (`typeof(LAST_SEMSTOKES_RESID_SPLIT[])` and three more), and `driver/amr_step.jl` and `driver/run.jl` follow it. The blocks otherwise follow the layers: primitives, then the geometry, then the operators and laws, then the solvers, then the hooks and the driver.

## The dependency direction

Apart from the accepted cycles below, the calls between layers run down and not back up.

| Caller | Depends on | The boundary |
|---|---|---|
| `driver/` | every other layer | nothing calls the driver |
| `amr/` | `core/`; `stokes/` in `amr_pass.jl`, which rebuilds `SemStokes` after a cycle | assembles no operator |
| `mg/` | `stokes/` for the matrix-free operator, `core/` | no physics, and no direct factorization of the fine saddle |
| `stokes/` | `core/`; the geometry layer for every leaf and face metric; `mg/` for the preconditioner; the material closures handed in by the driver | does not build `Params` and does not own the time step |
| `physics/` | `core/` in the files that evaluate or assemble a field; the geometry layer in `annulus_base.jl` | assembles no Stokes block |
| geometry layer (`core/geometry.jl`, `physics/annulus_base.jl`) | `core/` for the tree and the reference element; `physics/` for `Params` and the base-state functions in `annulus_base.jl` | defines a map and its metrics; never assembles, solves or integrates |
| `io/` | `core/` for the tree rebuild; the geometry layer for the identity it records | writes no frames and no history; the case scripts do |
| `diagnostics/` | `core/`, the geometry it evaluates the wall functionals on, the card it reads, and the solved fields | changes no state |

## The boundary-condition implementation map

The boundary conditions cross five files, and the geometry layer decides which form the constraint has. The card holds the data and checks it, the Stokes core turns it into constraints and a pressure gauge, the curved-wall kernel writes the mapped rows, the variable-viscosity assembly integrates the traction data, and the driver hands the card's fields to the solve.

| Part | File | What it does |
|---|---|---|
| Card API and validation | `physics/params.jl` | `Params` carries `bc` (`:free` or `:rigid`), the per-wall condition tuple `wall_bc`, and the per-wall data `wall_u`, `wall_un`, `wall_t` and `wall_tn`. The constructor resolves `:same` against `bc`. It refuses a value on a wall where the constraint would silently drop it, the all-open pure-traction configuration, and a configuration that leaves a rigid translation unconstrained. |
| Constraint construction, pressure gauge and refusals | `stokes/solve_var_core.jl` | Resolves the tuple, writes the Dirichlet values into `fixed` (the normal value from `wall_un`, the tangential one from `wall_u` on a `:rigid` wall only; an `:open` wall writes nothing), builds the constraint matrix, and pins one pressure dof iff no wall is `:open`, recording the choice in `LAST_SEMSTOKES_PRESSURE_GAUGE`. With an open wall it runs the 3x3 rigid-mode Gram test on the actual Dirichlet dof set; with every wall closed it runs the net-flux guard on the prescribed normal data. It refuses `:schur_cg` and `:lagged` on an open wall. |
| Curved-wall rows and the rigid-mode projection | `stokes/wall_constraints.jl` | On a mapped geometry a free-slip wall is the linear relation `n . u = g`, a moving rigid wall prescribes the wall-frame velocity `u = un n + ut tau`, and an open wall writes no velocity row; the wall-frame data are `u . n` and `u . tau` there, not global components. The file also assembles the Gram matrix of the three rigid modes, pins one velocity dof when a solid-body rotation is the only admissible nullspace (the free-slip ring), projects the returned velocity to zero net angular momentum, and refuses any other nullspace with the motion named. |
| Boundary-face load | `stokes/assembly_var.jl` | `_semquad_boundary_traction!` integrates the tangential (`wall_t`) and normal (`wall_tn`) traction data over the wall faces into the unreduced load vector, so the constraint matrix projects the load onto the admissible test space. `_semquad_validate_wall_traction` and `_semquad_validate_wall_tnormal` repeat the wall-type checks for direct library calls. |
| Driver hand-off | `driver/run.jl` | Passes `pr.bc`, `pr.wall_bc`, `pr.wall_u`, `pr.wall_un`, `pr.wall_t` and `pr.wall_tn` to `solve_semquad_stokes_var` in `solve_flow!`. The per-step Newton path refuses any non-default per-wall field, because it solves the single-`bc` problem. |

Every per-wall entry defaults to `nothing` and `wall_bc` to `:same`, so a card that sets none of them follows the closed-wall path unchanged; the geometry layer is what makes the box case a set of Cartesian dof rows and the annulus case the wall-frame relations above.

## The three deliberate cycles

### The matrix-free operator and the multigrid preconditioner

`stokes/solve_var_core.jl` calls into the solver layer — `mg/build.jl` for the hierarchy, `mg/pressure.jl` for the pressure block, `mg/saddle.jl` for the outer Krylov solve — and `mg/` calls back into the operator on every level: `mg/build.jl` calls `stokes/kop.jl`'s `semquad_kop` and `stokes/diag.jl`'s reduced diagonal and dense coarse-operator builders, `mg/vcycle.jl` calls `stokes/apply.jl`'s `_semquad_apply_Kred*` applies, and `mg/levelcache.jl` rebuilds a coarse level's constraint matrix through `solve_semquad_stokes_var(..., assemble_only=:mg)` in `stokes/solve_var.jl`. The cycle is the structure of matrix-free multigrid: the preconditioner is a V-cycle over the operator, so it cannot be expressed through an interface that hides the operator's per-level kernels without giving up being matrix-free. deal.II's `MatrixFreeOperators`/`MGTransfer` pair and ASPECT's `StokesMatrixFree`/`StokesPreconditioner` split have the same shape. The cycle is accepted with two containment rules: the calls from `stokes/` into `mg/` are confined to `stokes/solve_var_core.jl`, and that file remains the only place that decides which linear path runs.

### The quadtree as one object in three files

`core/quadtree.jl` holds the tree type and its split and merge primitives; `core/quadtree_table.jl` holds the flat face table those primitives maintain; `core/quadtree_balance.jl` holds the 2:1 balancer, which reads the table and splits or merges leaves through the primitives. The calls close a loop: the wall-graded builder in `quadtree.jl` calls `balance!`, and the balancer calls `refine!` and the table accessors. The cycle is accepted because the three files share one leaf representation and one legality rule, so no interface can separate them; the split buys readability only, so a change to any of them is a change to the tree itself and the three are reviewed together.

### The coarsening gate and the plan as one policy

`amr/coarsen_gate.jl` and `amr/coarsen_plan.jl` are two halves of one policy and call each other: the plan asks the gate whether a quartet may merge (`_merge_allowed`), and the gate reads the plan's ledger (`will_refine`, `granted`) and its quartet helper (`_kids4`) to decide. The cycle is accepted because the two files implement one policy; they are separate files only because the gate is the part that the card can switch off (`amr_coarsen_gate`). Any change to the merge rules belongs in both.

## Diagnostics are out-parameters

The module reports what a call did through module-level `Ref`s and counters: the `LAST_SEMSTOKES_*` family (for example in `stokes/state.jl`, `stokes/direct.jl`, `stokes/selector.jl` and `mg/pressure.jl`, including the boundary-condition gauge in `LAST_SEMSTOKES_PRESSURE_GAUGE`); `LAST_AUTO_FALLBACK`; the `SEMSTOKES_*` assembly and pressure counters; the AMR records `LAST_AMR_STATS` (`amr/amr_pass.jl`), `LAST_AMR_BALANCED` (`amr/cycle.jl`), `LAST_BUDGET_PASS` (`amr/leaf_ops.jl`) and `LAST_MIRROR_CLOSE` (`core/quadtree_balance.jl`); and the per-run records `SEMQUAD_STEP_PROFILE` and `SEMQUAD_PIC_DIAG` in `driver/run.jl`. These are out-parameters. The callee writes what it did and the caller reads it after the call, to log, to guard a restart, to choose what to write, or to abort with the numbers that identify the state.

`driver/run_diagnostics.jl` is the driver's owned snapshot of the family it reads. `RunDiagnostics` holds the solver and AMR values as fields, `snapshot!` is the only reader of the producer Refs in `src/driver`, and the driver re-takes the snapshot after each producer call (`solve_flow!`, the per-step Newton, an AMR pass, the final re-solve), because a snapshot is valid only until the next producer call. The producers still write the module Refs; the snapshot changes the reader side, not the contract.

The solver's per-call records describe one call: each solve overwrites them, and no arithmetic inside a solve reads one back. The per-run records are updated across the run and read by the driver afterwards. A value that has to influence the next solve is passed in an argument, never published in this family. Settings are kept apart from records: `SEMQUAD_MG_LEVEL_CACHE`, `SEMQUAD_MG_SERIAL_DOFS` and the region switches `SEMQUAD_MG_REGION*` are inputs that the solver reads, while the `LAST_*` family records what it did.

## The flow of one time step

The flow is quasi-static: the driver solves the Stokes saddle, advances the energy equation with that velocity, and repeats. One step, from `run_semquad_simulation` down and back:

```text
run_semquad_simulation                              driver/run.jl
  └─ while step < maxsteps && t < tstop
       ├─ Picard or Newton iteration                driver/run.jl
       │    ├─ solve_flow!, then snapshot!          driver/run.jl, run_diagnostics.jl
       │    │    ├─ viscosity closure               physics/ (laws, Maxwell)
       │    │    ├─ buoyancy and phase anomaly      driver/run.jl
       │    │    └─ solve_semquad_stokes_var        stokes/solve_var.jl
       │    │         ├─ semquad_choose_linear      stokes/selector.jl
       │    │         ├─ direct: LU and shifts      stokes/direct.jl
       │    │         └─ mg: FGMRES over V-cycle    mg/
       │    └─ solve_semquad_energy                 physics/energy.jl
       ├─ Maxwell stress (if G > 0), t and step     driver/run.jl
       ├─ _step_diagnostics -> history row          driver/diagnostics_row.jl
       ├─ _amr_step! -> AMR pass, snapshot!         driver/amr_step.jl
       ├─ callback -> frames and logs               examples/case_realistic_2d.jl
       └─ save_semquad_state                        io/checkpoint.jl
```

1. The step starts by copying the start-of-step temperature and weakening field, then enters the Picard loop (`pic_mode = :picard`, up to `maxpic` iterations; one on the production card). Each iteration calls `solve_flow!`, which builds the viscosity closure: temperature from the current iterate, the strain rate from the current velocity for the stress-dependent laws, the weak-zone factor, the `eta/eta_ref` normalization, and the Maxwell effective viscosity when the shear modulus is positive. Every metric it uses — the leaf height in the depth laws, the wall normals, the buoyancy direction — comes from the geometry. It then assembles the vertical buoyancy `Ra theta` plus the phase anomaly and calls `solve_semquad_stokes_var`, which resolves `:auto` before assembly, runs the direct or multigrid path, and publishes the residual and the chosen path in the `LAST_SEMSTOKES_*` records. The driver snapshots those records and compares the residual with `stokes_tol`; on a miss it writes the identifying numbers and aborts the step.
2. On the first iteration the CFL block updates `dt` from the maximum velocity and the smallest leaf height. The update sits inside the loop because the AMR pass zeroes the velocity, so an end-of-step update could only shrink `dt`.
3. `solve_semquad_energy` assembles the mass, stiffness, advection and source blocks for the temperature with the velocity just solved, adds the entropy-viscosity diffusion, the latent apparent capacity and the radiogenic decay, and applies the variable-step BDF2 update. With strain weakening on, one BDF2 transport for the weakening field follows the same pattern.
4. The Picard loop compares the new temperature with its iterate, under-relaxes, and stops below `pic_tol`; the optional trial rejection restores the previous flow state and halves the relaxation. `pic_mode = :newton` replaces the loop with a damped Newton on the coupled per-step residual (`driver/step_newton.jl`, `driver/newton_tangent.jl`), takes a snapshot after it, and falls back to Picard if the Newton fails.
5. After convergence, the Maxwell stress advances one backward-Euler step with the material viscosity of the step when the shear modulus is positive. The BDF2 temperature levels rotate, `t` and `step` advance, and `_step_diagnostics` builds the history row from the acceptance quantities in `diagnostics/global.jl` and the consistent-boundary-flux pair in `diagnostics/flux.jl`.
6. Every `amr_every` steps, `_amr_step!` (`driver/amr_step.jl`) marks leaves with the indicator, runs the balanced cycle, enforces the depth and leaf caps, rebuilds `SemStokes`, snapshots the AMR records, transfers the temperature levels, the weakening field and the stress history, zeroes the velocity and drops the stale multigrid tree entries. The case callback writes frames and the log, the optional checkpoint writes the full state, and the steady-state test decides whether to stop.
7. The next step starts from what this one leaves: the rotated temperature levels, the transferred weakening and stress fields, and the current mesh. The cycle closes.

**Next:** read [src/README.md](../src/README.md) and `src/QuaST.jl` alongside this page, then the `README.md` of the folder you are about to change.
