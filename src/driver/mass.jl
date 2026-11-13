"""Density function for the TALA mass constraint, or nothing for div(u) = 0."""
mass_rho_fun(pr) = pr.refstate === nothing ? nothing :
    (pr.refstate.prof !== nothing ? (y -> semquad_rho(pr.refstate, y)) :
     (pr.refstate.rho_fun === nothing ? nothing :
      (y -> semquad_rho(pr.refstate, y))))
# EOS reference state (docs/dev/PHYSICS.md §11): the anelastic mass form uses the
# *integrated* drho/dz instead of the finite difference of a user `rho_fun`.
# `nothing` for every other path, which then stays bit for bit the old code.
mass_drho_fun(pr) = (pr.refstate === nothing ||
                     pr.refstate.prof === nothing) ? nothing :
    (y -> semquad_drho_dz(pr.refstate, y))
"""Compressibility block rho_ad Di gamma p (entered into the upward momentum
with a plus sign; see `semquad_stokes_Pcm`), or nothing when the switch is off."""
compress_block_fun(ss, qt, pr) = pr.compress_p ?
    semquad_stokes_Pcm(ss, qt; rho_fun=mass_rho_fun(pr), Di=pr.Di,
                       gamma=pr.compress_gamma) : nothing

"""
    semquad_eta_range(sq, eta_fun, pr) -> (min, max, nclip, n)

Range of the viscosity field a Stokes solve used, sampled on the velocity nodes
one per node), plus the number of samples sitting on a clamp bound
(`eta_min`/`eta_max`, 1e-12 relative tolerance) and the sample count.  This is
the diagnostic of the Stokes-failure line only: it runs once, on the path that
is about to raise, and never touches the solver state.
"""
function semquad_eta_range(sq::SemQuad, eta_fun::Function, pr::Params)
    lo = Inf; hi = -Inf; nclip = 0
    lo_tol = pr.eta_min * (1.0 + 1.0e-12)
    hi_tol = pr.eta_max * (1.0 - 1.0e-12)
    for g in 1:sq.N
        (x, y) = sq.coords[g]
        e = eta_fun(x, y)
        e < lo && (lo = e)
        e > hi && (hi = e)
        (e <= lo_tol || e >= hi_tol) && (nclip += 1)
    end
    return (min=lo, max=hi, nclip=nclip, n=sq.N)
end
