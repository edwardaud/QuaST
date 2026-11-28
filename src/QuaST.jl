"""
QuaST -- 2-D infinite-Prandtl Boussinesq mantle convection, spectral element,
primitive variables ``(u,p)``.

The authoritative design and gate list is ``JULIA_DESIGN.md`` at the repository
root.  Only the Julia standard library is used: ``LinearAlgebra`` (sparse LU via
UMFPACK), ``SparseArrays`` and ``Printf``.
"""
module QuaST

using LinearAlgebra
using SparseArrays
using HDF5
using Printf

# ---------------------------------------------------------------------------
# Include order.  `include` is textual insertion into the module body, so the
# order is constrained only by *definition-time* references: a struct field
# type, a supertype, a method/keyword signature, or a `const` initialiser that
# names a type defined in another file.  `physics/` therefore appears in two
# blocks: the parameter and material-law types that `core/semquad.jl` itself
# needs come first, and the laws that *take* a `SemQuad` come after it.
# ---------------------------------------------------------------------------

# The physics revision is a behaviour version, not a parameter: a bare constant
# with no dependencies, so it comes before every file that reads it.
include("physics/revision.jl")

# -- physics: parameter types and the material laws that define them ---------
include("physics/tosi_law.jl")
include("physics/eos_profile.jl")
include("physics/refstate.jl")
include("physics/params_types.jl")
include("physics/params_validation.jl")
include("physics/params_base.jl")
include("physics/constitutive.jl")
include("physics/phase.jl")

# -- core: quadrature, reference element, quadtree, the scalar SEM space -----
# `core/semquad.jl` is split along its top-level boundaries.  The residue (type,
# constructor, leaf metrics, mass matrix) keeps the first and last slices of the
# former file; the other slices follow in their original order (parallel
# utilities, constraints, the Nitsche Poisson demo, point evaluation, transfer).
# Every slice is a contiguous, unedited part of the original file.
include("core/gll.jl")
# Remediation batch P0 part 1: the named finite/positive input guards used by
# the assembly entry points and the driver (AUDIT_VV F5/F6).  No dependency on
# another file, so it is included as early as possible.
include("core/guards.jl")
include("core/fe.jl")
include("core/quadtree.jl")
include("core/quadtree_table.jl")
include("core/quadtree_balance.jl")
# The geometry/mapping layer (stage 0): the global map, the leaf/face metrics,
# the logical boundary registry and the checkpoint identity.  Included after
# the tree (it names `Quadtree` in `CartesianBox(qt)`) and before the SEM space
# (whose constructors call it).  No card selects a mapped geometry in cut 1.
include("core/geometry_types.jl")
include("core/geometry_map.jl")
include("core/geometry_metrics.jl")
include("core/geometry_measures.jl")
include("core/geometry_topology.jl")
include("core/geometry_polar_annulus.jl")
include("core/wall_strip.jl")
# Stage 0.5b1: the geometry-dependent conductive base state.  It adds methods to
# `semquad_base_T`/`semquad_base_gradient`/`_mapped_base_curvature` (defined in
# `physics/refstate.jl`, which is included before the geometry layer) that
# dispatch on the concrete geometry types, so it must come after
# `core/geometry.jl`.  Nothing earlier calls it at definition time.
include("physics/annulus_base.jl")
include("core/semquad.jl")
include("core/parallel.jl")
include("core/constraints.jl")
include("core/poisson.jl")
include("core/eval.jl")
include("core/transfer.jl")

# -- AMR leaf primitives, then the physics laws over the SEM field -----------
include("amr/leaf_ops.jl")
include("amr/budget.jl")
# The gradient-jump indicator is SEM data, but its only caller is the AMR pass
# (`amr/hooks.jl`); it depends on the quadtree and `core/semquad.jl` only.
include("amr/indicator.jl")
# A diagnostic product (the entropy-viscosity field), but the energy assembly
# annotates its `evg` argument with `SEMEVGlobals`, so it must come first.
include("physics/latent.jl")
include("physics/entropy_viscosity.jl")
include("physics/energy_workspace.jl")
include("physics/energy_kernel_box.jl")
include("physics/energy_kernel_mapped.jl")
include("physics/energy_assembly.jl")
include("physics/energy_latent_jvp.jl")
include("physics/energy.jl")
include("physics/energy_bc.jl")
include("physics/eta_weak_zone.jl")
include("physics/eta_plastic.jl")
include("physics/eta_laws.jl")
include("physics/eta.jl")
include("physics/maxwell_kernels.jl")
include("physics/maxwell_band.jl")
include("physics/maxwell_regime.jl")
include("physics/maxwell_reference.jl")
include("physics/maxwell_force.jl")
include("physics/maxwell_softening.jl")
include("physics/viscoelastic.jl")
include("physics/weakening.jl")
include("physics/phase_buoyancy.jl")

# -- AMR policy (needs the quadtree only) ------------------------------------
# `amr/balanced.jl` is split along its top-level boundaries into the cooldown
# state, the 2:1 coarsening gate, the plan search and the cycle.  Each slice is
# a contiguous, unedited part of the former file; the include order is the
# definition order (the `AmrCooldown` type first).
include("amr/cooldown.jl")
include("amr/cap_fallback.jl")
include("amr/coarsen_gate.jl")
include("amr/coarsen_plan.jl")
include("amr/cycle.jl")

# -- stokes: state, selection, assembly, matrix-free operators, solves --------
include("stokes/state.jl")
include("stokes/selector.jl")
include("stokes/direct.jl")
include("stokes/krylov.jl")
include("stokes/assembly.jl")
include("stokes/kop.jl")
include("stokes/apply.jl")
include("stokes/diag.jl")
include("stokes/solve_const.jl")
include("stokes/assembly_var.jl")
include("stokes/boundary_loads.jl")
include("stokes/velocity_bc.jl")
include("stokes/wall_constraints.jl")
include("stokes/solve_var_core.jl")
include("stokes/solve_var_fallback.jl")
include("stokes/solve_var.jl")
include("stokes/operators.jl")

# -- mg: hierarchy, caches, V-cycle, pressure blocks, saddle Krylov ----------
include("mg/hierarchy.jl")
include("mg/levelcache.jl")
include("mg/build.jl")
include("mg/vcycle.jl")
include("mg/pressure.jl")
include("mg/bfbt.jl")
include("mg/saddle.jl")
include("mg/s1solve.jl")

# -- AMR hooks and the driver (the time loop and the per-step Newton path) ---
include("amr/wall_refine.jl")
include("amr/amr_pass.jl")
include("amr/hooks.jl")
include("driver/strain_rate.jl")
include("driver/picard_control.jl")
include("driver/picard.jl")
include("driver/newton_tangent.jl")
include("driver/step_newton.jl")
include("driver/maxwell_state.jl")
include("driver/maxwell_reports.jl")
include("driver/eta_ref_report.jl")
# `run_semquad_simulation`'s run-config builder and restart guard name only
# types defined above (`Quadtree`, `Params`, `PhaseTransition`) in their
# signatures; the checkpoint functions they call are resolved at run time, like
# every other driver call, so they can be included just before the entry point.
include("driver/restart_guard.jl")
include("driver/diagnostics_row.jl")
# Stage 1 of the `RunState` refactor: the driver-owned snapshot of the
# solver/AMR diagnostic out-parameters.  It names only the value types of the
# Refs defined in `stokes/` and `amr/` above.
include("driver/run_diagnostics.jl")
# Stage 2 of the `RunState` refactor: the AMR block of the time loop as one
# function.  It calls `semquad_amr_passes!`, `transfer_semquad`, `snapshot!`
# and the MG tree-cache drop, so it comes after `run_diagnostics.jl`.
include("driver/amr_step.jl")
include("driver/run.jl")
include("driver/mass.jl")

# -- I/O and diagnostics -----------------------------------------------------
include("io/checkpoint_cooldown.jl")
include("io/checkpoint_config.jl")
include("io/checkpoint_tree.jl")
include("io/checkpoint.jl")
include("diagnostics/global.jl")
include("diagnostics/flux.jl")

export GLLRule, gll_rule, gauss_legendre, legendre_with_derivs, lagrange_eval, lagrange_d2,
       barycentric_weights, lagrange_eval!, lagrange_d2!, lagrange_values,
       lagrange_values!, FEData, make_fedata, Params, Tbar, dTbar, eta_of_T, eta_at,
       deta_dT_at, eta_composite_at, deta_composite_at, composite_inv_eta, composite_Tnd,
       CompositeIsoState, composite_isostress_state, COMP_ISO_MAXIT, COMP_ISO_TOL,
       PhaseTransition, phase_gamma, phase_dgamma_dT, phase_dgamma_dz, phase_d2gamma_dT2,
       phase_d2gamma_dzdT, phase_latent_coefficient, per_phase_latent,
       semquad_latent_coeff, semquad_latent_active, semquad_latent_coefficients,
       spd_factor, Quadtree, refine!, coarsen!, neighbors, balance!, rebalance_local!,
       LocalBalance, LOCAL_BALANCE, refine_where!, max_face_level_jump, leaf_box,
       total_area, face_pairs, face_pairs_logical, wall_graded_tree, mirror_defect, mirror_close!,
       LAST_MIRROR_CLOSE, semquad_enforce_leaf_budget!, LAST_BUDGET_PASS, SemQuad,
       AbstractGeometry, CartesianBox, AffineBox, CurvedBox, PolarAnnulus,
       LogicalBoundary,
       boundaries, geometry_name, geometry_identity, logical_map,
       element_map, element_inverse, element_jacobian, jacobian_inverse,
       global_logical, leaf_ref, locate_leaf, locate_leaf_logical, leaf_metric,
       quad_weight, quad_point, basis_grad, face_metric, face_point, face_weight,
       face_jacobian, face_normal, face_tangent, height, depth, grad_height,
       height_laplacian, gravity, gravity_magnitude, measure_weight, perimeter,
       leaf_measure, total_measure, boundary_measure, leaf_min_size,
       leaf_vertical_size, leaf_normal_size, minimum_leaf_size,
       minimum_leaf_height,
       nu_wall_sides, upward_velocity,
       BoxLeafMetric, MappedLeafMetric, BoxFaceMetric, MappedFaceMetric,
       geometry_stamp, empty_semquad_geometry_cache!, semquad_geometry_cache_stats,
       AbstractSemQuadKOp, SemQuadKOpMapped,
       make_semquad, semquad_constraints, assemble_semquad_poisson, transfer_semquad,
       semquad_mass, semquad_gradient_jump, semquad_velocity_jump,
       semquad_combined_jump,
       assemble_semquad_energy,
       assemble_semquad_energy_latent_jvp, SemStokes, make_semstokes, solve_semquad_stokes,
       solve_semquad_stokes_var, assemble_semquad_pressure_mass, SemQuadKOp, SemQuadKWork,
       semquad_kop, semquad_eta_field, semquad_eta_contrast, semquad_choose_linear,
       semquad_apply_K!, semquad_apply_Kred!, semstokes_eval, solve_semquad_energy,
       semquad_nu_walls, semquad_vrms, semquad_surface_velocity, semquad_work_dissipation,
       semquad_mean_T, semquad_nu_energy, semquad_heat_budget, semquad_exact_residual,
       semquad_max_divergence, semquad_base_advection_integral,
       semquad_theta2_divu_integral, semquad_eta_at, semquad_amr_passes!,
       refine_wall_to_min!, semquad_level_stats, run_semquad_simulation,
       save_semquad_state, load_semquad_state, TosiLaw, semquad_eta_tosi,
       semquad_deta_tosi, semquad_stokes_B, semquad_stokes_Pcm, compress_block_fun,
       mass_rho_fun, SemRefState, semquad_Tad, semquad_dTad_dz, semquad_base_gradient,
       semquad_base_T, semquad_base_laplacian, SemRefProfile, semquad_rho, semquad_alpha,
       semquad_cp, semquad_drho_dz, mass_drho_fun, theta_buoyancy_load,
       buoyancy_mass_matrix, semquad_cbf_theta_flux, semquad_nu_cbf,
       semquad_phase_buoyancy, semquad_ev_field, semquad_step_newton,
       semquad_dp_deta_T, yield_active, semquad_yield_stress,
       semquad_strain_rate_at, weak_active, semquad_weak_factor, semquad_yield_cohesion,
       weak_zone_factor,
       maxwell_stress_step, maxwell_stress_exact, maxwell_force_stress,
       maxwell_stress_corotate, maxwell_stress_exact_corotational,
       MAXWELL_COROTATION_BOUND,
       semquad_spin_component,
       semquad_yield_friction, semquad_plastic_strain_rate,
       semquad_plastic_strain_rate_field, semquad_weakening_parts,
       semquad_weakening_source_g, semquad_weakening_jvp, solve_semquad_weakening,
       semquad_weak_stats, semquad_visco_plastic_eta, semquad_strain_rate_field,
       PHYSICS_REVISION,
       blend_field, semquad_deta_dp, semquad_deta_dp_eps, semquad_deta_chi,
       picard_diverging, picard_adapt_relax, amr_cycle_balanced!, AmrCooldown,
       LAST_AMR_BALANCED, assemble_semquad_velocity_mass,
       solve_semquad_poisson, semquad_eval, LAST_SEMSTOKES_RESIDUAL,
       LAST_SEMSTOKES_SCHUR_ITERS, LAST_SEMSTOKES_VEL_CG_ITERS, LAST_SEMSTOKES_REGULARIZED,
       LAST_SEMSTOKES_PRESSURE_GAUGE,
       LAST_SEMSTOKES_ROTATION_GAUGE, LAST_SEMSTOKES_ROTATION_SHIFT,
       LAST_SEMSTOKES_RESID_SPLIT, LAST_SEMSTOKES_SHIFT, LAST_SEMSTOKES_REFINE_ITERS,
       LAST_SEMSTOKES_LADDER, LAST_SEMSTOKES_LADDER_PHASE, LAST_SEMSTOKES_SCALED,
       SEMSTOKES_K_ASSEMBLIES, LAST_SEMSTOKES_LAGGED_ITERS, LAST_SEMSTOKES_LAGGED_REFACTOR,
       SEMSTOKES_LAGGED_REFACTORS, SEMSTOKES_PRESSURE_BUILDS, SEMSTOKES_PRESSURE_REUSES,
       SEMQUAD_MG_LEVEL_CACHE, SEMQUAD_MG_LEVEL_STATS, semquad_mg_level_cache_stats,
       semquad_mg_level_cache_bytes, reset_semquad_mg_level_stats!,
       LAST_SEMSTOKES_PRESSURE_REUSED, LAST_SEMSTOKES_PRESSURE_JUMP,
       LAST_SEMSTOKES_PRESSURE_MODE, LAST_SEMSTOKES_MG_PROFILE, LAST_SEMSTOKES_AUTO_CHOICE,
       LAST_AUTO_FALLBACK, SEMQUAD_STEP_PROFILE, SEMQUAD_PIC_DIAG,
       SEMQUAD_PRESSURE_DEFAULT_ETA_TOL, SEMQUAD_PRESSURE_CACHE

end # module




















