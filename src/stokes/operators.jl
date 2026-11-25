"""
    semquad_stokes_B(ss, qt; rho_fun=nothing)

Quadrature-assembled divergence matrix on the quadtree,

    B[pi, uj] = int q_pi * rho(y) * dphi_j/dx  dV,     B[pi, vzj] likewise,

with the velocity/pressure bases of `ss`.  `rho_fun=nothing` gives the constant
density form used by the reference-element assembly, which lets the two be
compared (G129).  Passing `y -> rho(y)` gives the density-weighted divergence
`int q rho div(u)`, which alone enforces `div u = 0`, **not** `div(rho u) = 0`.
The anelastic constraint additionally needs the `int q u_z drho/dy` term, which
`drho_fun` adds (G130):

    int q (rho div u + u_z drho/dy) dV = 0    <=>    div(rho u) = 0.

`rho_mult` is the density hook of `Params.rho_multiplier` (G248): the assembled
density becomes `rho(y) * rho_mult(x, y)`, and its gradient is differentiated
consistently (central differences, `h = 1e-6`) so the matrix is the Galerkin form
of `div(rho_tot u) = 0`:

    int q (rho_tot div u + u_x d rho_tot/dx + u_z d rho_tot/dy) dV = 0,

with `d rho_tot/dx = rho d rho_mult/dx` and
`d rho_tot/dy = rho_mult d rho/dy + rho d rho_mult/dy`.  This is the anelastic
mass form of a *smooth* density anomaly.  A multiplier with a jump (a sharp
sticky-air interface) has no bounded pointwise gradient, so the strong-form
`int q u.grad(rho)` term cannot represent it; such a configuration must use the
default `mass_form=:boussinesq` (`div u = 0`, no density) or a smooth interface
profile.  `rho_mult === nothing` (the default) never evaluates the multiplier
and reproduces the historical assembly bit for bit.
"""
function semquad_stokes_B(ss, qt::Quadtree; rho_fun::Union{Nothing,Function}=nothing,
                          drho_fun::Union{Nothing,Function}=nothing,
                          rho_mult::Union{Nothing,Function}=nothing)
    return _semquad_stokes_B(ss.su.geom, ss, qt; rho_fun=rho_fun,
                             drho_fun=drho_fun, rho_mult=rho_mult)
end

# Historical box body, verbatim.  `ss.su.geom` is a CartesianBox on every
# production path; the geometry argument is not read here (cut-2 protocol:
# the box path is the current statements, GEOM_DESIGN §4.1).
function _semquad_stokes_B(::CartesianBox, ss, qt::Quadtree;
                           rho_fun::Union{Nothing,Function}=nothing,
                           drho_fun::Union{Nothing,Function}=nothing,
                           rho_mult::Union{Nothing,Function}=nothing)
    su = ss.su
    sp = ss.sp
    p = su.p
    n = p + 1
    q = sp.p
    nq1 = q + 1
    fd = make_fedata(p, q)
    nu = su.N
    np = sp.N
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, np * 4 * n * n)
    sizehint!(J, np * 4 * n * n)
    sizehint!(V, np * 4 * n * n)
    hd = 1.0e-6
    # derivative of the y-only density: exactly the supplied `drho_fun` (the
    # caller owns the finite-difference fallback, as before).  It is kept as the
    # literal argument, not rebuilt from `rho_fun`, so a call without
    # `drho_fun` remains the historical `int q rho div u` matrix bit for bit.
    drho0 = drho_fun
    if rho_mult !== nothing && drho0 === nothing && rho_fun !== nothing
        # Only the hook path needs the derivative even if the caller gave none
        # (the driver always supplies one); without this the `rho dM/dy` part
        # would silently lose the `M drho/dy` part of d(rho*M)/dy.
        drho0 = y -> begin
            yp = min(y + hd, 1.0); ym = max(y - hd, 0.0)
            (rho_fun(yp) - rho_fun(ym)) / (yp - ym)
        end
    end
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            rho = rho_fun === nothing ? 1.0 : rho_fun(y)
            drho_dx = 0.0
            drho_dy = drho0 === nothing ? 0.0 : drho0(y)
            if rho_mult !== nothing
                # density hook: rho_tot = rho(y) * rho_mult(x, y), gradient by
                # central differences of the multiplier (the y density keeps its
                # analytic/finite-difference derivative above)
                x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
                rho0 = rho
                xp = min(x + hd, 1.0); xm = max(x - hd, 0.0)
                yp = min(y + hd, 1.0); ym = max(y - hd, 0.0)
                dmx = (rho_mult(xp, y) - rho_mult(xm, y)) / (xp - xm)
                dmy = (rho_mult(x, yp) - rho_mult(x, ym)) / (yp - ym)
                rho = rho0 * rho_mult(x, y)
                drho_dx = rho0 * dmx
                drho_dy = drho_dy * rho_mult(x, y) + rho0 * dmy
            end
            for b in 1:nq1, a in 1:nq1
                pq = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                pq == 0.0 && continue
                piq = pids[(b - 1) * nq1 + a]
                for j in 1:n, i in 1:n
                    uj = uids[(j - 1) * n + i]
                    dpx = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                    dpy = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                    ph = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                    c = w * rho * pq
                    push!(I, piq); push!(J, uj); push!(V, c * dpx)
                    push!(I, piq); push!(J, nu + uj); push!(V, c * dpy)
                    if drho0 !== nothing || rho_mult !== nothing
                        # div(rho u) = rho div u + u.grad rho: the second term
                        # turns the rho-weighted incompressibility into the
                        # anelastic continuity equation.
                        push!(I, piq); push!(J, nu + uj)
                        push!(V, w * pq * drho_dy * ph)
                    end
                    if drho_dx != 0.0
                        push!(I, piq); push!(J, uj)
                        push!(V, w * pq * drho_dx * ph)
                    end
                end
            end
        end
    end
    return sparse(I, J, V, np, 2 * nu)
end

# Mapped body: the same `int q (rho div u + u.grad rho)` Galerkin form, with the
# physical gradients of the mapped basis (J^{-T} chain rule) and the density
# evaluated as a function of the physical **height** `h = height(g,x,y)` (the
# annulus generalisation of "a function of y", GEOM_DESIGN §2.7/§6.4):
# `grad rho = rho'(h) * grad h` with the geometry's `grad_height`.
#
# Cut-2 scope: `rho_mult` is refused.  It is a Cartesian `(x,y)` multiplier
# whose gradient is differentiated by central differences clamped to the unit
# box (`min(x+hd,1.0)`, `max(x-hd,0.0)`, see the box body below); that clamp has
# no mapped-geometry meaning, and the density-hook gradient is annulus-stage
# physics wiring.  Refusing it is better than evaluating a box-only gradient at
# mapped points.
function _semquad_stokes_B(g::AbstractGeometry, ss, qt::Quadtree;
                           rho_fun::Union{Nothing,Function}=nothing,
                           drho_fun::Union{Nothing,Function}=nothing,
                           rho_mult::Union{Nothing,Function}=nothing)
    rho_mult === nothing ||
        throw(ArgumentError("semquad_stokes_B: rho_mult (the Cartesian " *
            "density multiplier with its finite-difference gradient) is " *
            "box-only; a mapped geometry has no clamps to [0,1]^2.  Pass " *
            "rho_fun/drho_fun (functions of the physical height) instead."))
    su = ss.su
    sp = ss.sp
    p = su.p
    n = p + 1
    q = sp.p
    nq1 = q + 1
    fd = make_fedata(p, q)
    nu = su.N
    np = sp.N
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, np * 4 * n * n)
    sizehint!(J, np * 4 * n * n)
    sizehint!(V, np * 4 * n * n)
    for lf in su.leaves
        lm = leaf_metric(g, su, lf, fd)
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(lm, fd, qx, qy)
            x, y = quad_point(lm, su, lf, fd, qx, qy)
            h = height(g, x, y)
            rho = rho_fun === nothing ? 1.0 : rho_fun(h)
            drho_dx = 0.0
            drho_dy = 0.0
            if drho_fun !== nothing
                dr = drho_fun(h)
                ghx, ghy = grad_height(g, x, y)
                drho_dx = dr * ghx
                drho_dy = dr * ghy
            end
            for b in 1:nq1, a in 1:nq1
                pq = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                pq == 0.0 && continue
                piq = pids[(b - 1) * nq1 + a]
                for j in 1:n, i in 1:n
                    uj = uids[(j - 1) * n + i]
                    dpx, dpy = basis_grad(lm, fd, qx, qy, i, j)
                    ph = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                    c = w * rho * pq
                    push!(I, piq); push!(J, uj); push!(V, c * dpx)
                    push!(I, piq); push!(J, nu + uj); push!(V, c * dpy)
                    if drho_fun !== nothing
                        # div(rho u) = rho div u + u.grad rho
                        push!(I, piq); push!(J, uj)
                        push!(V, w * pq * drho_dx * ph)
                        push!(I, piq); push!(J, nu + uj)
                        push!(V, w * pq * drho_dy * ph)
                    end
                end
            end
        end
    end
    return sparse(I, J, V, np, 2 * nu)
end

"""
    semquad_stokes_Pcm(ss, qt; rho_fun=nothing, gamma=1.0, Di=1.0)

Pressure-velocity block of the TALA/ALA compressibility term.  In ASPECT the
pressure dependence enters the *density* (code.cc:93-94),

    rho = rho_ad(y) * (1 - alpha (T - Tad) + Di gamma (p - p_ad)),

so its contribution to the momentum equation is a buoyancy force along gravity,

    f_z = -rho_ad(y) Di gamma (p - p_ad),

and nothing in the horizontal direction.  This block is therefore

    Pcm[i, pj] = int phi_i(y) * rho(y) * Di * gamma * phi_pj dV

in the **vertical** rows only; with rho = 1 it is the pressure mass matrix (up to
the Di*gamma factor).  `rho_mult` is the density hook: the coefficient becomes
`rho(y) * rho_mult(x, y)`, the density `Params.rho_multiplier` scales everywhere
else; `nothing` leaves the block bit-identical.  `components=:both` reproduces
the historical behaviour that applied the same column to both velocity
components: that is a spurious horizontal force proportional to pressure (it
over-drove King 2010 by ~25x in Vrms) and is kept only so the old wiring can be
regression-tested.

The `p_ad` part of `(p - p_ad)` is a function of y alone, so it is a gradient
force that can be absorbed into the pressure; dropping it does not change u.

Sign convention: the block itself is assembled positive and the saddle assembly
enters it into the upward momentum with a **plus** sign
(`Bmm = B' - Pcm` in `solve_semquad_stokes_var`).  That is the sign that
reproduces ASPECT's operator: prescribing ASPECT's own p field with the force
`-rho_ad Di gamma p` gives ASPECT's velocity to six digits (Vrms 24.7681 vs
24.7657, max|u| 46.861 vs 46.858), while the opposite sign gives 27.88.
See the operator-agreement test.
"""
function semquad_stokes_Pcm(ss, qt::Quadtree; rho_fun::Union{Nothing,Function}=nothing,
                            gamma::Float64=1.0, Di::Float64=1.0,
                            components::Symbol=:z,
                            rho_mult::Union{Nothing,Function}=nothing)
    return _semquad_stokes_Pcm(ss.su.geom, ss, qt; rho_fun=rho_fun,
                               gamma=gamma, Di=Di, components=components,
                               rho_mult=rho_mult)
end

# Historical box body, verbatim; the geometry argument is not read here.
function _semquad_stokes_Pcm(::CartesianBox, ss, qt::Quadtree;
                             rho_fun::Union{Nothing,Function}=nothing,
                             gamma::Float64=1.0, Di::Float64=1.0,
                             components::Symbol=:z,
                             rho_mult::Union{Nothing,Function}=nothing)
    components in (:z, :both) ||
        throw(ArgumentError("components must be :z or :both"))
    su = ss.su
    sp = ss.sp
    p = su.p
    n = p + 1
    q = sp.p
    nq1 = q + 1
    fd = make_fedata(p, q)
    nu = su.N
    np = sp.N
    I = Int[]; J = Int[]; V = Float64[]
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx
        y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            c = w * Di * gamma * (rho_fun === nothing ? 1.0 : rho_fun(y))
            if rho_mult !== nothing
                x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
                c *= rho_mult(x, y)
            end
            for b in 1:nq1, a in 1:nq1
                pq = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                pq == 0.0 && continue
                piq = pids[(b - 1) * nq1 + a]
                for j in 1:n, i in 1:n
                    uj = uids[(j - 1) * n + i]
                    ph = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                    ph == 0.0 && continue
                    if components === :z
                        push!(I, nu + uj); push!(J, piq); push!(V, c * ph * pq)
                    else
                        push!(I, uj); push!(J, piq); push!(V, c * ph * pq)
                        push!(I, nu + uj); push!(J, piq); push!(V, c * ph * pq)
                    end
                end
            end
        end
    end
    return sparse(I, J, V, 2 * nu, np)
end

# Mapped body of the compressibility block.  The momentum force of the ALA
# density perturbation is along the geometry's **gravity direction**
# (GEOM_DESIGN §6.4):
#
#     f = -rho_ad Di gamma (p - p_ad) * ghat,        |ghat| = 1,
#
# so the block has a row for **both** components and the sign of each is the
# mapped gravity vector's.  On the box `ghat = (0,-1)`, the `u_x` row is exactly
# zero and `-(-1)*c = c` is the historical positive `u_z` entry, which is how
# G268 compares it against the unchanged box body.  The `components` keyword is
# a legacy selector of the box's two conventions (`:z` historical vertical,
# `:both` the spurious symmetric form); a mapped geometry has no "vertical
# coordinate", so the keyword is validated but the geometry decides the rows.
# The density is evaluated at the physical height `h = height(g,x,y)`.
#
# Cut-2 scope: `rho_mult` is refused for the same reason as in
# `semquad_stokes_B` (box-only finite-difference/clamp hook).
function _semquad_stokes_Pcm(g::AbstractGeometry, ss, qt::Quadtree;
                             rho_fun::Union{Nothing,Function}=nothing,
                             gamma::Float64=1.0, Di::Float64=1.0,
                             components::Symbol=:z,
                             rho_mult::Union{Nothing,Function}=nothing)
    components in (:z, :both) ||
        throw(ArgumentError("components must be :z or :both"))
    rho_mult === nothing ||
        throw(ArgumentError("semquad_stokes_Pcm: rho_mult (the Cartesian " *
            "density multiplier with its finite-difference gradient) is " *
            "box-only; a mapped geometry has no clamps to [0,1]^2.  Pass " *
            "rho_fun (a function of the physical height) instead."))
    su = ss.su
    sp = ss.sp
    p = su.p
    n = p + 1
    q = sp.p
    nq1 = q + 1
    fd = make_fedata(p, q)
    nu = su.N
    np = sp.N
    I = Int[]; J = Int[]; V = Float64[]
    for lf in su.leaves
        lm = leaf_metric(g, su, lf, fd)
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(lm, fd, qx, qy)
            x, y = quad_point(lm, su, lf, fd, qx, qy)
            h = height(g, x, y)
            c = w * Di * gamma * (rho_fun === nothing ? 1.0 : rho_fun(h))
            gvx, gvy = gravity(g, x, y)
            for b in 1:nq1, a in 1:nq1
                pq = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                pq == 0.0 && continue
                piq = pids[(b - 1) * nq1 + a]
                for j in 1:n, i in 1:n
                    uj = uids[(j - 1) * n + i]
                    ph = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                    ph == 0.0 && continue
                    cph = c * ph * pq
                    push!(I, uj); push!(J, piq); push!(V, -(cph * gvx))
                    push!(I, nu + uj); push!(J, piq); push!(V, -(cph * gvy))
                end
            end
        end
    end
    return sparse(I, J, V, 2 * nu, np)
end
