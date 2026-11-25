# Quadtree Taylor-Hood Stokes (stage A-AMR, step 2).
#
# Velocity Q_p and pressure Q_{p-1} on the quadtree leaves, with the same
# hanging-node constraints as the scalar Poisson (semquad.jl).  The element
# matrices are the constant-viscosity (eta = 1) stress form
#   2 eps(u):eps(v) - p div(v) - q div(u) = f.v,
# assembled on the reference square (the h factors cancel in the stiffness and
# enter the divergence as h/2), with velocity Dirichlet values imposed through
# an affine constraint u = u0 + C u_free (u0 carries the boundary values and the
# fixed part of any hanging constraint).  One pressure dof is pinned to fix the
# constant null space.
#
# Gate: the polynomial field u = (2xy, -y^2), p = 0 is divergence free and
# satisfies -div(2 eps(u)) = (0, 2), so with f = (0, 2) and Q_2/Q_1 the discrete
# solution must reproduce it exactly on any (mixed) quadtree.

struct SemStokes
    p::Int
    q::Int
    su::SemQuad
    sp::SemQuad
end

# Diagnostics for the last quadtree Stokes solve: relative reduced-saddle
# residual and the number of Schur-CG iterations (0 for a direct solve).
const LAST_SEMSTOKES_RESIDUAL = Ref(0.0)
const LAST_SEMSTOKES_SCHUR_ITERS = Ref(0)
# Number of inner CG iterations spent in the velocity block of the `linear=:mg`
# solve (MF-4); 0 for every other path.
const LAST_SEMSTOKES_VEL_CG_ITERS = Ref(0)
# True when the direct path had to retry with the 1e-12 relative diagonal
# regularization (wall trees with free-slip; see the direct branch below).
const LAST_SEMSTOKES_REGULARIZED = Ref(false)
# Pressure gauge of the last variable-viscosity solve: `:pin` when one nodal
# pressure dof was pinned to zero (exact only while the normal velocity is
# Dirichlet on the whole boundary, i.e. when **no** wall is `:open`), `:none`
# when an `:open` wall forced the derived gauge.
# `nothing` before the first solve.  G262 pins "the pin is on exactly when no
# wall is open" on this record plus the column count of the returned `C_p`.
const LAST_SEMSTOKES_PRESSURE_GAUGE = Ref{Any}(nothing)
# Stage 0.5b2: velocity gauge of the last variable-viscosity solve.  `:none`
# when the boundary rows exclude all three rigid-body modes (the historical
# behaviour of every box path); `:rotation` when a free-slip curved wall left
# the solid-body rotation admissible, so one free velocity dof was pinned and
# the returned velocity was projected to zero net angular momentum.
# Read by the annulus gate G275.
const LAST_SEMSTOKES_ROTATION_GAUGE = Ref{Any}(:none)
# The rotation `omega` removed by the zero-net-angular-momentum projection of the
# last solve (0 when no gauge ran).  `LAST_SEMSTOKES_ROTATION_GAUGE` says
# whether the gauge was engaged; this records the measured size of the removed
# mode (`u_before = u_after + omega * (-y, x)`).
const LAST_SEMSTOKES_ROTATION_SHIFT = Ref(0.0)

# --- the direct path's shift ladder -------------
#
# The two blocks of the reduced saddle are measured separately: `vel` is
# `||Kred u + Bt p - fr|| / rden` (momentum rows) and `div` is
# `||Bs u - gr|| / rden` (continuity rows).  `LAST_SEMSTOKES_RESIDUAL` is their
# max, so this only says *which* block limits a marginal solve.
const LAST_SEMSTOKES_RESID_SPLIT = Ref((vel=0.0, div=0.0))
# Relative diagonal shift of the factorisation behind the returned solution
# (0.0 = the unshifted LU, 1e-12/1e-10/1e-8 = a ladder rung; when the returned
# iterate comes from the refinement, the shift of the factorisation it was
# preconditioned by).  Reset by every solve, so it describes the last solve only.
const LAST_SEMSTOKES_SHIFT = Ref(0.0)
# Iterative-refinement iterations (GMRES with the last factorisation as
# preconditioner) spent after the ladder missed; 0 when none were needed.
const LAST_SEMSTOKES_REFINE_ITERS = Ref(0)
# Every rung the last direct solve tried, in order, as `(shift, residual)`.
# `nothing` unless the ladder fired (so a healthy solve carries no allocation).
const LAST_SEMSTOKES_LADDER = Ref{Any}(nothing)
# Which phase of the ladder produced the last direct answer (docs/
# CASE_REALISTIC.md §17.4, G212): `:healthy` (the unshifted LU passed),
# `:tiny` (the priority tiny-shift phase of rule 2, for a *lost* unshifted
# solve), `:ladder` (the 1e-12/1e-10/1e-8 rungs, including the marginal but
# accepted unshifted answer), `:refine` (the iterative refinement improved the
# best), `:fallback` (the last-resort tiny shifts of rule 7 won) or `:fail`
# (every rung, the refinement and -- when it ran -- the last-resort phase missed
# `accept_tol`; the caller's error is due).  `nothing` unless the last solve took
# the direct path; reset with `LAST_SEMSTOKES_LADDER`.
const LAST_SEMSTOKES_LADDER_PHASE = Ref{Any}(nothing)
# Cumulative number of direct solves whose answer came from the equilibrated
# path (`stokes_scaling = :equilibrated` engaged on a solve that failed the
# unscaled health check and its scaled residual was accepted).  Unlike the
# per-solve records above this one is a counter, not reset by a solve: it
# counts the engages of a whole run (`LAST_SEMSTOKES_SCALED[] = 0` restarts it).
const LAST_SEMSTOKES_SCALED = Ref(0)
# Cumulative number of times the **assembled velocity stiffness** `K` (the
# `2N x 2N` sparse operator of the leaf assembly) has been formed since the
# process started.  It is a counter, not a per-solve record: a caller reads it
# before and after a solve to prove that a path did not assemble `K` at all.
# The matrix-free `linear=:mg, mg_operator=:matrixfree` path leaves it
# unchanged (S3 stage C: `K` and `Kred` are `nothing`); the direct path and the
# sparse MG geometry increment it.  `assemble_only=:mg` (the reduced assembly)
# increments it zero times.
const SEMSTOKES_K_ASSEMBLIES = Ref(0)

