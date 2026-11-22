# The *physics revision*: the version of the physics behaviour, not a parameter.
#
# Every other key in `run_config` records a *value* the run used, so the restart
# guard in `src/driver/run.jl` can refuse a restart that changes one of them.  A
# value comparison cannot see a change of what those values *mean*: two commits
# can leave every parameter number identical and still advance the Maxwell
# stress with a different update.  A checkpoint written before such a commit
# would then resume with a stress history produced by one update and continue it
# with another -- silently, because every recorded value matches.
#
# `PHYSICS_REVISION` is the single source of truth for that kind of change.
# `run_config["physics_revision"]` stores it in every checkpoint, the restart
# guard refuses a file whose stored revision differs, and a file without the key
# (every checkpoint written before the marker existed) is resumed with a warning
# that says whether it carries a stress history.  `test/t_g252_physics_revision.jl`
# pins all three behaviours plus the source hash below.
#
# Changelog.  Revisions count the states of the physics, oldest first.  A
# checkpoint written under a revision below `PHYSICS_REVISION` is not
# bit-compatible.  The state that changes is the viscoelastic one
# (`shear_modulus > 0`; the stress history `txx`/`tyy`/`txy`); `shear_modulus = 0`
# runs carry no stress history and are unaffected by every entry below.
#
# Revision 1 -- the state before 958af5e (2026-11-21).
#     The Maxwell stress update was driven by the *raw* strain-rate tensor
#     `eps_ij` and had no co-rotational (Jaumann) term.  A checkpoint with a
#     stress history written by this state stores that stress: its deviatoric
#     fixed point plus the spurious isotropic part of the discrete trace, in
#     the frame of the non-objective update.
#
# Revision 2 -- 958af5e (2026-11-21), "physics: project the stress update onto
#     the deviatoric strain rate", and 8c27a02 (2026-11-21), "physics: the
#     co-rotational (Jaumann) term in the Maxwell stress" -- the current
#     revision.
#     The update is driven by the deviatoric strain rate
#
#         ehat_ij = eps_ij - (1/3) tr(eps) delta_ij,    tr(eps) = eps_xx + eps_zz,
#
#     and the old stress is co-rotated by `dt (W tau - tau W)` before the
#     backward-Euler relaxation.  The stored stress is therefore the
#     deviatoric, objective one, and its fixed point is
#
#         tau = 2 eta ehat + tau_M (W tau - tau W),    tau_M = eta / G,
#
#     not `2 eta ehat`.  A stress history written under revision 1 is a tensor
#     in a different frame with a different (non-zero) trace, so resuming it
#     under this revision mixes two updates; the restart guard refuses it.
#
# Every checkpoint written before this marker existed has no
# `physics_revision` key.  If it carries a stress history at all, that history
# was written by revision 1; that is why the key-less path is a warning that
# names the stress history rather than a silent pass.  Bump this constant in the
# same commit as any change to the *behaviour* of an existing parameter, and add
# an entry above dated by that commit.
const PHYSICS_REVISION = 2

# ---------------------------------------------------------------------------
# Which sources the revision covers, and the hash that makes it enforceable.
# ---------------------------------------------------------------------------
#
# A commit that changes the Maxwell update does not necessarily remember to bump
# `PHYSICS_REVISION`; the marker alone is only as good as the human.  Every
# suite run therefore recomputes a hash of the behaviour-defining sources and
# fails when one differs from the values recorded below
# (`test/t_g252_physics_revision.jl`), so a physics change cannot land without
# the developer making the revision decision explicit.
#
# Exactly these sources are covered, and not more:
#
# * `src/physics/maxwell_kernels.jl` -- the relaxation kernel and the
#   co-rotational increment.  They are the equation of the update.
# * `src/physics/maxwell_force.jl` -- the lagged stress as the momentum equation
#   sees it (`div(a tau^n)` with `a = eta/(eta + G dt)`); it decides the fixed
#   point the flow relaxes to.
# * `src/physics/maxwell_softening.jl` -- the effective-viscosity softening the
#   per-step Newton residual is assembled with, so the two nonlinear paths solve
#   the same closure.
# * `src/driver/maxwell_state.jl` -- the whole file, which holds the driver-side
#   half of the update: `_nodal_eta` (the material viscosity whose `eta/G` is the
#   Maxwell time) and `_advance_maxwell_stress!` (where the kernels are composed
#   into the per-step update -- the deviatoric projection `ehat`, the spin
#   sampling, the advance order and the step `dt`).  Both revision-2 changes
#   (958af5e, 8c27a02) edited that function.  The file was carved out of `run.jl`
#   so the lock covers the part that defines the stored stress without firing on
#   the time loop around it.
#
# Not covered on purpose: `src/physics/revision.jl` (it holds the hashes), the
# I/O and AMR-transfer code (generic state plumbing, pinned by their own gates),
# the diagnostics and references of the subject (`maxwell_band.jl`,
# `maxwell_regime.jl`, `maxwell_reference.jl`, `src/driver/maxwell_reports.jl` --
# they read the state or define the curves it is checked against, they never
# write it), and the accessors `semquad_strain_rate_components` /
# `semquad_spin_component` (their sign and scaling are pinned against the analytic
# rigid rotation by G250, which fails loudly on its own if they change).  Adding a
# file requires editing the expected list in G252 as well, so extending coverage
# is a deliberate act rather than a one-line change here.
#
# The hash is the first 16 hex digits of the SHA-256 of the source bytes with
# the line endings normalised (every `\r\n` and lone `\r` becomes `\n`, so the
# checkout's line-ending setting cannot change the result).  A spec may also name
# a function region as `path#function_name!`, in which case the bytes are the
# lines from `function <name>(` through its matching top-level `end`, joined with
# `\n` and terminated by one `\n` (the docstring above it is not hashed: it
# documents behaviour, it does not define it).  Both shipped entries hash whole
# files, because each file now has exactly one subject.  The gate prints the
# current values when it fails, so recovery is copy-and-paste.
#
# To update: decide, explicitly, whether a stored stress history means something
# different now.  If it does -- bump `PHYSICS_REVISION`, replace the hash, and
# add a dated changelog entry for the new revision.  If it does not (a comment,
# docstring or formatting edit), replace the hash and leave the revision alone.
# The gate's failure message says the same thing.
const PHYSICS_SOURCE_HASHES = (
    "src/physics/maxwell_kernels.jl" => "592d30ad098fae22",
    "src/physics/maxwell_force.jl" => "d9780355291738da",
    "src/physics/maxwell_softening.jl" => "6f12a39bbea5e34e",
    "src/driver/maxwell_state.jl" => "1e1c4501db863a0c",
)
