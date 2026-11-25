# Visco-plastic rheology for the quadtree SEM path.
#
# The viscosity law itself (`eta_at`) is shared with the earlier structured-grid layer; this file adds the
# strain-rate dependent plastic branch on top of it:
#
#     tau_y   = min(c cos(phi) + p sin(phi), tau_max)      Drucker-Prager
#     eta_pl  = tau_y / (2 |edot|)
#     eta     = min(eta_visc, eta_pl)                      then eta_min/max clip
#
# Nondimensionalisation (this solver, Boussinesq):
#   viscosity  [eta0],  length [H],  time [H^2/kappa],  pressure and stress
#   [eta0 kappa / H^2],  strain rate [kappa / H^2].
# `yield_c` and `yield_taumax` are therefore in units of `eta0 kappa / H^2`, and
# `p` is the *dynamic* pressure of the Stokes solve (top-mean zero).  Using the
# dynamic rather than the total pressure is the standard Boussinesq
# approximation; the hydrostatic part is a function of depth only and is
# deliberately not included here (this solver has no reference state yet, see P3).
#
# Depth dependence: with the depth d = 1-y the cohesion
# and the friction coefficient are linear profiles,
#
#     c(d)  = yield_c + yield_c_grad*d
#     mu(d) = sin(yield_phi) + yield_mu_grad*d
#     tau_y = min(c(d)*cos(yield_phi) + mu(d)*p, yield_taumax)
#
# and the weakening factor of §9 multiplies the whole uncapped stress:
#
#     f(chi) = 1 - (1-weak_f)*clamp(chi,0,1),    tau_y = min(f*tau_raw, taumax).
#
# With zero gradients and f = 1 every expression below is the historical one
# bit for bit (G150/G162/G163/G168 unchanged).

# This subject is split by concern across four files:
#
#   `eta_weak_zone.jl`  the weak-zone multiplier and its activity predicates;
#   `eta_plastic.jl`    the Drucker-Prager branch and the strain-rate helpers;
#   `eta_laws.jl`       the `semquad_eta_at` closures and their derivatives.
#
# The include order in `src/QuaST.jl` follows that list.
