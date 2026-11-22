# Maxwell stress history, step 1: the update kernel.
#
# This file holds the physics of the stress update and nothing else.  The state
# vector, the AMR transfer and the checkpoint live in the driver, and the elastic
# force term is bridged by `maxwell_force_stress`; keeping the update kernels here
# means they can be gated against the analytic curve independently of that
# machinery, so a failure here cannot be confused with a transport or wiring bug.
# (The composition of the kernels into the per-step update is
# `_advance_maxwell_stress!` in `src/driver/run.jl`, docs/dev/PHASE2.md 2b.)
#
# The update is *objective*: a Maxwell stress that does not rotate with the
# material is not frame-indifferent, and every mantle-convection run is a
# large-strain run.  `maxwell_stress_corotate` supplies the co-rotational
# (Jaumann) part and the driver composes it with the scalar backward-Euler
# kernel; the analytic reference for the composed update is
# `maxwell_stress_exact_corotational`.
# This subject is split by concern, one file per part:
#
#   `maxwell_kernels.jl`    the scalar relaxation and the co-rotational increment;
#   `maxwell_band.jl`       the validity band of the linearised rotation;
#   `maxwell_regime.jl`     the material/effective viscosity regime measurement;
#   `maxwell_reference.jl`  the analytic solutions the gates converge to;
#   `maxwell_force.jl`      the lagged stress as the momentum equation sees it;
#   `maxwell_softening.jl`  the effective-viscosity softening for the Newton factory.
#
# The include order in `src/QuaST.jl` follows that list.
