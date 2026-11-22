# The Maxwell diagnostics the driver reports.
#
# Two independent records, both diagnostic only (no arithmetic result of the
# solve reads them), and both built by pure string functions so every `@sprintf`
# format is a literal and the gates can pin the wording:
#
#   * the co-rotation validity band -- how far the linearised Jaumann rotation is
#     from the resolved one on this run (`MAXWELL_COROTATION_BOUND` in
#     `src/physics/viscoelastic.jl`);
#   * the viscosity regime -- the material against the effective contrast, i.e.
#     what `G*dt` caps and what the solve therefore sees.
#
# The statistics they render are produced by `_advance_maxwell_stress!`
# (`maxwell_state.jl`) and `maxwell_regime` (`src/physics/viscoelastic.jl`).

# ---------------------------------------------------------------------------
# G254 message builders.  Kept as functions so every `@sprintf` format in them
# is a literal (the macro needs one), and so the gate can pin their wording.
# They build a `String` only: no state is read or written.
# ---------------------------------------------------------------------------

function _corotation_band_warning(step::Int, t::Real, two_w_dt::Real,
                                  two_taum_w::Real, resolved::Real)
    return string("Maxwell co-rotation validity band exceeded at step ", step,
                  " (t=", @sprintf("%.6e", t), "): the update applies the ",
                  "linearised Jaumann rotation 1 + dt*ad_W, valid while the ",
                  "per-step rotation 2|W|*min(dt, tau_M) is small.  Measured ",
                  "maxima over the domain on this step: 2|W|dt = ",
                  @sprintf("%.6e", two_w_dt), ", 2*tau_M*|W| = ",
                  @sprintf("%.6e", two_taum_w), ", per-step rotation = ",
                  @sprintf("%.6e", resolved), "; bound = ",
                  @sprintf("%.3g", MAXWELL_COROTATION_BOUND),
                  ".  Above the bound the rotation is no longer faithful: the ",
                  "per-step magnitude error is about (2|W|dt)^2/2 and, for a ",
                  "step longer than tau_M = eta/G, the update keeps an ",
                  "old-stress residue of order 2*tau_M*|W|, so the stress ",
                  "frame (and every diagnostic reading it) drifts by more ",
                  "than the scheme's stated accuracy allows.  The run ",
                  "continues -- this is an accuracy band, not a stability ",
                  "limit -- but treat its stress history as outside the ",
                  "stated band; reduce dt (dtmax/cfl) to get back inside, or ",
                  "use the exact exponential of the rotation if this regime ",
                  "is genuinely needed.")
end

function _corotation_band_report(steps::Int, two_w_dt::Real,
                                 two_taum_w::Real, resolved::Real,
                                 warned::Bool)
    return string("Maxwell co-rotation validity (G254): over ", steps,
                  " stress step(s), max 2|W|dt = ",
                  @sprintf("%.6e", two_w_dt),
                  ", max 2*tau_M*|W| = ", @sprintf("%.6e", two_taum_w),
                  ", max per-step rotation 2|W|*min(dt, tau_M) = ",
                  @sprintf("%.6e", resolved), "; bound = ",
                  @sprintf("%.3g", MAXWELL_COROTATION_BOUND), " -- ",
                  warned ? "OUTSIDE the band, see the warning above" :
                           "inside the band", ".")
end

# G255: the Maxwell regime message builder.  Like the G254 pair above it is a
# pure function returning a String, so every `@sprintf` format is a literal and
# the gate can pin the wording.  It is deliberately a *separate* record from the
# co-rotation report: one says what viscosity field the solve sees, the other
# measures the accuracy of the rotation.
function _maxwell_regime_message(r::NamedTuple, step::Int, t::Real,
                                 dt::Real, G::Real, final::Bool)
    collapse = r.material_contrast / r.effective_contrast
    return string("Maxwell regime (G=", @sprintf("%.6e", G), ", ",
                  final ? "end of run" : "first stress step",
                  ", step ", step, ", t=", @sprintf("%.6e", t),
                  ", dt=", @sprintf("%.6e", dt), "): material eta in [",
                  @sprintf("%.6e", r.material_min), ", ",
                  @sprintf("%.6e", r.material_max), "], contrast ",
                  @sprintf("%.6e", r.material_contrast),
                  "; effective eta_eff = 1/(1/eta + 1/(G*dt)) in [",
                  @sprintf("%.6e", r.effective_min), ", ",
                  @sprintf("%.6e", r.effective_max), "], contrast ",
                  @sprintf("%.6e", r.effective_contrast),
                  "; G*dt/eta in [", @sprintf("%.6e", r.gdt_over_eta_min),
                  ", ", @sprintf("%.6e", r.gdt_over_eta_max),
                  "] -- G*dt caps eta_eff from above, so the contrast the ",
                  "solve sees is ", @sprintf("%.3g", collapse), "x smaller (",
                  @sprintf("%.2f", log10(collapse)),
                  " decades removed): a viscoelastic card solves a softer, ",
                  "different problem than its shear_modulus=0 twin, not the ",
                  "same problem plus memory.")
end
