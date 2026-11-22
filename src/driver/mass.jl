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
                       gamma=pr.compress_gamma,
                       rho_mult=pr.rho_multiplier) : nothing

"""
    theta_buoyancy_load(pr, th_at) -> fz

Thermal-buoyancy load of the Boussinesq momentum equation for a temperature
perturbation whose evaluator is `th_at(x, y)`, with the density hook
`Params.rho_multiplier` applied:

    f_z(x, y) = Ra * rho_multiplier(x, y) * theta(x, y).

`rho_multiplier === nothing` returns exactly the historical closure
`(x, y) -> Ra * theta(x, y)` and never evaluates a multiplier, so every path
that used it is bit-identical to the pre-hook code.  A uniform multiplier `c`
therefore gives exactly the same load as `Ra -> Ra*c`, which is the identity the
gate asserts.  This is the *Boussinesq* form used by the steady/Newton paths,
which carry no reference state (`Ra * theta`; see `run.jl` for the TALA/ALA
`rho(y) alpha(y) theta` form).
"""
function theta_buoyancy_load(pr::Params, th_at::Function)
    m = pr.rho_multiplier
    ra = pr.Ra
    return m === nothing ? ((x, y) -> ra * th_at(x, y)) :
                           ((x, y) -> ra * m(x, y) * th_at(x, y))
end

"""
    buoyancy_mass_matrix(sq, pr) -> SparseMatrixCSC

Pressure mass matrix of the theta-derivative of the buoyancy load
`theta_buoyancy_load`, i.e. `int c(x, y) phi_i phi_j` with
`c = rho_multiplier(x, y)`.  Without a density multiplier this is exactly
`assemble_semquad_pressure_mass(sq)` (the historical matrix, bit for bit); the
multiplier enters the Newton tangent so the Jacobian sees the same density the
residual does.  The factor `Ra` stays outside the matrix (the Newton codes
multiply it), exactly as before.
"""
function buoyancy_mass_matrix(sq::SemQuad, pr::Params)
    m = pr.rho_multiplier
    m === nothing && return assemble_semquad_pressure_mass(sq)
    return assemble_semquad_pressure_mass(sq; weight=m)
end

"""
    semquad_eta_range(sq, eta_fun, pr) -> (min, max, nclip, n)

Range of the viscosity field a Stokes solve used, sampled on the velocity nodes
one per node), plus the number of samples sitting on a clamp bound
(`eta_min`/`eta_max`, 1e-12 relative tolerance) and the sample count.  This is
the diagnostic of the Stokes-failure line only: it runs once, on the path that
is about to raise, and never touches the solver state.

`eta_fun` is the closure the flow solve used, i.e. the material law divided by
`Params.eta_ref` (G257); the clamp bounds are absolute bounds in the law's
units, so they are divided by `eta_ref` too before the census.  `eta_ref = 1`
(the default) is the historical call, bit for bit.
"""
function semquad_eta_range(sq::SemQuad, eta_fun::Function, pr::Params)
    lo = Inf; hi = -Inf; nclip = 0
    lo_tol = (pr.eta_min / pr.eta_ref) * (1.0 + 1.0e-12)
    hi_tol = (pr.eta_max / pr.eta_ref) * (1.0 - 1.0e-12)
    for g in 1:sq.N
        (x, y) = sq.coords[g]
        e = eta_fun(x, y)
        e < lo && (lo = e)
        e > hi && (hi = e)
        (e <= lo_tol || e >= hi_tol) && (nclip += 1)
    end
    return (min=lo, max=hi, nclip=nclip, n=sq.N)
end
