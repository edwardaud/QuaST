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
"""
function semquad_stokes_B(ss, qt::Quadtree; rho_fun::Union{Nothing,Function}=nothing,
                          drho_fun::Union{Nothing,Function}=nothing)
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
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            rho = rho_fun === nothing ? 1.0 : rho_fun(y)
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
                    if drho_fun !== nothing
                        # div(rho u) = rho div u + u_z drho/dy: the second term
                        # turns the rho-weighted incompressibility into the
                        # anelastic continuity equation.
                        push!(I, piq); push!(J, nu + uj)
                        push!(V, w * pq * drho_fun(y) * ph)
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
the Di*gamma factor).  `components=:both` reproduces the historical behaviour
that applied the same column to both velocity components: that is a spurious
horizontal force proportional to pressure (it over-drove King 2010 by ~25x in
Vrms) and is kept only so the old wiring can be regression-tested.

The `p_ad` part of `(p - p_ad)` is a function of y alone, so it is a gradient
force that can be absorbed into the pressure; dropping it does not change u.

Sign convention: the block itself is assembled positive and the saddle assembly
enters it into the upward momentum with a **plus** sign
(`Bmm = B' - Pcm` in `solve_semquad_stokes_var`).  That is the sign that
reproduces ASPECT's operator: prescribing ASPECT's own p field with the force
`-rho_ad Di gamma p` gives ASPECT's velocity to six digits (Vrms 24.7681 vs
24.7657, max|u| 46.861 vs 46.858), while the opposite sign gives 27.88.
See docs/dev/SOLVER.md for the operator-agreement test.
"""
function semquad_stokes_Pcm(ss, qt::Quadtree; rho_fun::Union{Nothing,Function}=nothing,
                            gamma::Float64=1.0, Di::Float64=1.0,
                            components::Symbol=:z)
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
        y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            c = w * Di * gamma * (rho_fun === nothing ? 1.0 : rho_fun(y))
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
