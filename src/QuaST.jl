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

# -- physics: parameter types and the material laws that define them ---------
include("physics/tosi_law.jl")
include("physics/refstate.jl")
include("physics/params.jl")
include("physics/constitutive.jl")
include("physics/phase.jl")

# -- core: quadrature, reference element, quadtree, the scalar SEM space -----
# `core/semquad.jl` is split along its top-level boundaries.  The residue (type,
# constructor, leaf metrics, mass matrix) keeps the first and last slices of the
# former file; the other slices follow in their original order (parallel
# utilities, constraints, the Nitsche Poisson demo, point evaluation, transfer).
# Every slice is a contiguous, unedited part of the original file.
include("core/gll.jl")
include("core/fe.jl")
include("core/quadtree.jl")
include("core/semquad.jl")
include("core/parallel.jl")
include("core/constraints.jl")
include("core/poisson.jl")
include("core/eval.jl")
include("core/transfer.jl")

# -- AMR leaf primitives, then the physics laws over the SEM field -----------
include("amr/leaf_ops.jl")
# The gradient-jump indicator is SEM data, but its only caller is the AMR pass
# (`amr/hooks.jl`); it depends on the quadtree and `core/semquad.jl` only.
include("amr/indicator.jl")
# A diagnostic product (the entropy-viscosity field), but the energy assembly
# annotates its `evg` argument with `SEMEVGlobals`, so it must come first.
include("diagnostics/entropy_viscosity.jl")
include("physics/latent.jl")
include("physics/energy_assembly.jl")
include("physics/energy.jl")
include("physics/eta.jl")
include("physics/weakening.jl")
include("physics/phase_buoyancy.jl")

# -- AMR policy (needs the quadtree only) ------------------------------------
# `amr/balanced.jl` is split along its top-level boundaries into the cooldown
# state, the 2:1 coarsening gate, the plan search and the cycle.  Each slice is
# a contiguous, unedited part of the former file; the include order is the
# definition order (the `AmrCooldown` type first).
include("amr/cooldown.jl")
include("amr/coarsen_gate.jl")
include("amr/coarsen_plan.jl")
include("amr/cycle.jl")

# -- stokes: state, selection, assembly, matrix-free operators, solves --------
include("stokes/state.jl")
include("stokes/selector.jl")
include("stokes/direct.jl")
include("stokes/assembly.jl")
include("stokes/kop.jl")
include("stokes/apply.jl")
include("stokes/diag.jl")
include("stokes/solve_const.jl")
include("stokes/assembly_var.jl")
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
include("mg/gmres.jl")

# -- AMR hooks and the driver (the time loop and the steady/Newton paths) ----
include("amr/hooks.jl")
include("driver/picard.jl")
include("driver/newton.jl")
include("driver/step_newton.jl")
include("driver/tosi_steady.jl")
include("driver/dp_steady.jl")
include("driver/steady_entry.jl")
include("driver/run.jl")
include("driver/mass.jl")

# -- I/O and diagnostics -----------------------------------------------------
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
       total_area, face_pairs, wall_graded_tree, mirror_defect, mirror_close!,
       LAST_MIRROR_CLOSE, semquad_enforce_leaf_budget!, LAST_BUDGET_PASS, SemQuad,
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
       semquad_cp, semquad_drho_dz, mass_drho_fun, semquad_cbf_theta_flux, semquad_nu_cbf,
       semquad_phase_buoyancy, semquad_newton_steady, semquad_steady_step,
       semquad_steady_jv, semquad_steady_parts, semquad_ev_field, semquad_tosi_newton,
       semquad_tosi_inner, semquad_step_newton, semquad_tosi_steady_step,
       semquad_tosi_steady_jv, semquad_dp_inner, semquad_dp_steady_step,
       semquad_dp_steady_jv, semquad_dp_stokes_newton, semquad_dp_newton,
       semquad_dp_deta_T, semquad_steady_newton, yield_active, semquad_yield_stress,
       semquad_strain_rate_at, weak_active, semquad_weak_factor, semquad_yield_cohesion,
       weak_zone_factor,
       semquad_yield_friction, semquad_plastic_strain_rate,
       semquad_plastic_strain_rate_field, semquad_weakening_parts,
       semquad_weakening_source_g, semquad_weakening_jvp, solve_semquad_weakening,
       semquad_weak_stats, semquad_visco_plastic_eta, semquad_strain_rate_field,
       blend_field, semquad_deta_dp, semquad_deta_dp_eps, semquad_deta_chi,
       picard_diverging, picard_adapt_relax, amr_cycle_balanced!, AmrCooldown,
       SEMQUAD_AMR_COOLDOWN, LAST_AMR_BALANCED, assemble_semquad_velocity_mass,
       solve_semquad_poisson, semquad_eval, LAST_SEMSTOKES_RESIDUAL,
       LAST_SEMSTOKES_SCHUR_ITERS, LAST_SEMSTOKES_VEL_CG_ITERS, LAST_SEMSTOKES_REGULARIZED,
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




















