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

include("gll.jl")
include("mesh.jl")
include("fe.jl")
include("tosi_law.jl")
include("semquad_refstate.jl")
include("physics.jl")
include("refstate.jl")
include("phase.jl")
include("rheology.jl")
include("io.jl")
include("dg1d.jl")
include("quadtree.jl")
include("dgquad.jl")
# Before the `driver/` chunks: `semquad_amr_passes!`/`run_semquad_simulation`
# type their `amr_cooldown` keyword with `AmrCooldown`, which is defined in
# `amr_balanced.jl`.
include("amr_advanced.jl")
include("amr_balanced.jl")
include("semquad.jl")
# `semquadstokes.jl` is now the `src/stokes/` chunk tree.  Joined in the order
# below, the chunks are byte-identical to the former single file, and that
# order *is* the original statement order (the `SemStokes` state first, then
# the selectors and direct solver, then the operator kernels, then the two
# solves), so it must not be reordered.
include("stokes/state.jl")
include("stokes/selector.jl")
include("stokes/direct.jl")
include("stokes/assembly.jl")
include("stokes/kop.jl")
include("stokes/apply.jl")
include("stokes/solve.jl")
include("stokes/assembly_var.jl")
include("stokes/solve_var.jl")
include("stokes/operators.jl")
# Same contract for `semquad_mg.jl`, now `src/mg/`: byte-identical chunks in
# the original order (hierarchy cache -> build -> V-cycle -> pressure blocks ->
# BFBT -> saddle solver).
include("mg/hierarchy.jl")
include("mg/levelcache.jl")
include("mg/build.jl")
include("mg/vcycle.jl")
include("mg/pressure.jl")
include("mg/bfbt.jl")
include("mg/saddle.jl")
include("semquad_s1solve.jl")
include("semquadenergy.jl")
include("semquaddiagnostics.jl")
include("semquad_rheology.jl")
include("semquad_weaken.jl")
include("semquad_picard.jl")
include("semquad_newton.jl")
include("semquad_step_newton.jl")
include("semquad_tosi_steady.jl")
include("semquad_dp_steady.jl")
include("semquad_steady_entry.jl")
# `semquadconvection.jl` is now the `src/driver/` chunk tree; same byte-identity
# and original-order contract as the stokes and mg chunks above.
include("driver/amr_hooks.jl")
include("driver/run.jl")
include("driver/mass.jl")
include("semquad_checkpoint.jl")
include("semquad_flux.jl")
include("semquad_phase.jl")
include("amr_dwr.jl")
include("bcs.jl")
include("streamfunction.jl")
include("rt.jl")
include("rt1.jl")
include("assembly.jl")
include("projection.jl")
include("direct.jl")
include("timestep.jl")
include("newton_solver.jl")
include("gmres.jl")
include("mg.jl")
include("cbf.jl")
include("psi_stokes.jl")

export GLLRule, gll_rule, gauss_legendre, legendre_with_derivs,
       lagrange_eval, lagrange_d2, barycentric_weights,
       lagrange_eval!, lagrange_d2!, lagrange_values, lagrange_values!
export Mesh, make_mesh, vel_gid, pres_gid, element_vel_gids,
       element_pres_gids, node_coords, Nscalar, z_left, z_right, z_size,
       z_min_size
export FEData, make_fedata
export Params, Tbar, dTbar, eta_of_T, eta_at, deta_dT_at, diagnostics, nu_walls, vrms,
       nu_energy, max_divergence, theta_divu_integral,
       eta_composite_at, deta_composite_at, composite_inv_eta, composite_Tnd,
       CompositeIsoState, composite_isostress_state, COMP_ISO_MAXIT,
       COMP_ISO_TOL
export RefState, build_refstate
export PhaseTransition, phase_gamma, phase_dgamma_dT, phase_dgamma_dz,
       phase_d2gamma_dT2, phase_d2gamma_dzdT,
       phase_latent_coefficient, per_phase_latent,
       semquad_latent_coeff, semquad_latent_active, semquad_latent_coefficients
export RheologyModel, CreepParams, YieldParams, eta_dislocation, yield_stress, eta_plastic,
       eta_visco_plastic, eta_rheology_nondim,
       deta_deps_dislocation, eta_plastic_smooth, deta_deps_plastic_smooth,
       eta_visco_plastic_raw, smooth_cap_log, eta_visco_plastic_smooth, eta_vp_smooth_stable, assemble_velocity_mass, spd_factor, TosiParams, eta_tosi
export TemperatureGrid, read_temperature_ascii, interp_temperature,
       theta_from_ascii
export DG1D, make_dg1d, assemble_dg1d_poisson, solve_dg1d_poisson, dg1d_l2_error,
       
       Quadtree, refine!, coarsen!, neighbors, balance!, rebalance_local!, LocalBalance, LOCAL_BALANCE, refine_where!, max_face_level_jump, leaf_box, total_area, face_pairs, wall_graded_tree,
       mirror_defect, mirror_close!, LAST_MIRROR_CLOSE,
       DGQuad, make_dgquad, assemble_dgquad_poisson, dgquad_eval, is_uniform,
       dgquad_jump_indicator, amr_cycle!, semquad_enforce_leaf_budget!, LAST_BUDGET_PASS, assemble_dgquad_advection,
       run_dgquad_advection_subcycled,
       SemQuad, make_semquad, semquad_constraints, assemble_semquad_poisson,
       transfer_semquad, semquad_mass, semquad_gradient_jump, assemble_semquad_energy,
       assemble_semquad_energy_latent_jvp,
       SemStokes, make_semstokes, solve_semquad_stokes, solve_semquad_stokes_var,
       assemble_semquad_pressure_mass,
       SemQuadKOp, SemQuadKWork, semquad_kop, semquad_eta_field,
       semquad_eta_contrast, semquad_choose_linear,
       semquad_apply_K!, semquad_apply_Kred!,
       semstokes_eval, solve_semquad_energy,
       semquad_nu_walls, semquad_vrms, semquad_surface_velocity, semquad_work_dissipation, semquad_mean_T, semquad_nu_energy,
       semquad_heat_budget, semquad_exact_residual,
       semquad_max_divergence, semquad_base_advection_integral,
       semquad_theta2_divu_integral, semquad_eta_at, semquad_amr_passes!,
       refine_wall_to_min!,
       semquad_level_stats,
       run_semquad_simulation, save_semquad_state, load_semquad_state,
       TosiLaw, semquad_eta_tosi, semquad_deta_tosi,
       semquad_stokes_B, semquad_stokes_Pcm, compress_block_fun, mass_rho_fun, SemRefState, semquad_Tad, semquad_dTad_dz, semquad_base_gradient,
       semquad_base_T, semquad_base_laplacian,
       SemRefProfile, semquad_rho, semquad_alpha, semquad_cp, semquad_drho_dz,
       mass_drho_fun,
       
       semquad_cbf_theta_flux, semquad_nu_cbf, semquad_phase_buoyancy,
       
       semquad_newton_steady, semquad_steady_step, semquad_steady_jv,
       semquad_steady_parts, semquad_ev_field, semquad_tosi_newton, semquad_tosi_inner,
       semquad_step_newton,
       semquad_tosi_steady_step, semquad_tosi_steady_jv,
       semquad_dp_inner, semquad_dp_steady_step, semquad_dp_steady_jv, semquad_dp_stokes_newton,
       semquad_dp_newton, semquad_dp_deta_T,
       semquad_steady_newton,
       yield_active, semquad_yield_stress, semquad_strain_rate_at,
       weak_active, semquad_weak_factor, semquad_yield_cohesion,
       semquad_yield_friction, semquad_plastic_strain_rate,
       semquad_plastic_strain_rate_field,
       semquad_weakening_parts, semquad_weakening_source_g,
       semquad_weakening_jvp, solve_semquad_weakening, semquad_weak_stats,
       semquad_visco_plastic_eta, semquad_strain_rate_field, blend_field,
       semquad_deta_dp, semquad_deta_dp_eps, semquad_deta_chi,
       picard_diverging, picard_adapt_relax, amr_cycle_hysteresis!, amr_cycle_balanced!, AmrCooldown, SEMQUAD_AMR_COOLDOWN, LAST_AMR_BALANCED, goal_upper_half, dwr_leaf_indicator, solve_stokes_adjoint_system, stokes_residual, stokes_reduced_residual, assemble_semquad_velocity_mass, velocity_goal_rhs, hierarchical_stokes_residual, interpolate_semquad, dual_weights_semquad, poisson_dwr_cell_term, poisson_dwr_face_terms, poisson_dwr_estimate, restrict_indicator_to_tree, aggregate_indicator_to_level, stokes_dwr_indicator,
       solve_semquad_poisson, semquad_eval,
       DofMap, ThetaMap, make_dofmap, make_theta_map, gather_reduced, cfl_length, cfl_local_dt,
       scatter_reduced
export assemble_stokes, assemble_B, assemble_energy, assemble_body_force,
       bilinear_stokes, solve_stokes, solve_stokes_general, solve_energy,
       cbf_wall_flux,
       assemble_pressure_stiffness, assemble_pressure_mass, assemble_pressure_mass_visc, project_velocity,
       ScalarSpace, make_scalar_space, streamfunction_velocity,
       streamfunction_quads, quad_stats, base_advection_quad, QuadVelocity,
       potential_projection, LinearResidualError, solve_stokes_uzawa,
       solve_stokes_gmres, solve_stokes_schur, solve_stokes_schur_cg
export StepResult, StokesSolution, SimulationResult, seed_theta,
       run_simulation, run_steady, LAST_ENERGY_RESIDUAL, LAST_STOKES_RESIDUAL,
       LAST_SEMSTOKES_RESIDUAL, LAST_SEMSTOKES_SCHUR_ITERS,
       LAST_SEMSTOKES_VEL_CG_ITERS,
       LAST_SEMSTOKES_REGULARIZED, QUAD_DIAG,
       LAST_SEMSTOKES_RESID_SPLIT, LAST_SEMSTOKES_SHIFT,
       LAST_SEMSTOKES_REFINE_ITERS, LAST_SEMSTOKES_LADDER,
       LAST_SEMSTOKES_LADDER_PHASE, LAST_SEMSTOKES_SCALED,
       SEMSTOKES_K_ASSEMBLIES,
       LAST_SEMSTOKES_LAGGED_ITERS, LAST_SEMSTOKES_LAGGED_REFACTOR,
       SEMSTOKES_LAGGED_REFACTORS,
       SEMSTOKES_PRESSURE_BUILDS, SEMSTOKES_PRESSURE_REUSES,
       SEMQUAD_MG_LEVEL_CACHE, SEMQUAD_MG_LEVEL_STATS,
       semquad_mg_level_cache_stats, semquad_mg_level_cache_bytes,
       reset_semquad_mg_level_stats!,
       LAST_SEMSTOKES_PRESSURE_REUSED, LAST_SEMSTOKES_PRESSURE_JUMP,
       LAST_SEMSTOKES_PRESSURE_MODE,
       LAST_SEMSTOKES_MG_PROFILE,
       LAST_SEMSTOKES_AUTO_CHOICE,
       LAST_AUTO_FALLBACK,
       SEMQUAD_STEP_PROFILE,
       SEMQUAD_PIC_DIAG,
       SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
       SEMQUAD_PRESSURE_CACHE
export assemble_psi_stokes, solve_stokes_psi,
       assemble_psi_omega_stokes, solve_stokes_psi_omega,
       rt0_basis, rt0_divergence, rt0_edge_signs, rt0_edge_normal, rt0_edge_flux,
       RT0Map, make_rt0_map, assemble_rt0_stokes, solve_stokes_rt0,
       rt1_basis, rt1_edge_moment, RT1_REF,
       RT1Map, make_rt1_map, assemble_rt1_stokes, solve_stokes_rt1

end # module




















