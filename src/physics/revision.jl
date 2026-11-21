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
# pins all three behaviours.
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
