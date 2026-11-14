# Reference Stokes element matrices, make_semstokes and the sparse pressure mass/stiffness.
function make_semstokes(qt::Quadtree, p::Int)
    return SemStokes(p, p - 1, make_semquad(qt, p), make_semquad(qt, p - 1))
end

"""Reference element matrices for Q_p velocity / Q_q pressure (constant eta)."""
function _sem_stokes_elem(p::Int, q::Int)
    ru = gll_rule(p)
    n = p + 1
    D = zeros(Float64, n, n)
    for i in 1:n
        _, dL = lagrange_eval(ru, ru.nodes[i]); D[i, :] .= dL
    end
    W = Diagonal(ru.weights)
    W2 = kron(W, W)
    Dx = kron(Matrix{Float64}(I, n, n), D)
    Dz = kron(D, Matrix{Float64}(I, n, n))
    Axx = 2.0 .* (Dx' * W2 * Dx) .+ Dz' * W2 * Dz
    Azz = 2.0 .* (Dz' * W2 * Dz) .+ Dx' * W2 * Dx
    Axz = Dz' * W2 * Dx
    # pressure basis evaluated at the velocity GLL nodes
    rq = gll_rule(q)
    nq = q + 1
    E = zeros(Float64, n, nq)
    for i in 1:n
        val, _ = lagrange_eval(rq, ru.nodes[i])
        E[i, :] .= val
    end
    Bx = kron(E' * W, E' * W * D)
    Bz = kron(E' * W * D, E' * W)
    return Axx, Azz, Axz, Bx, Bz
end

"""
    assemble_semquad_pressure_mass(sp; weight=nothing, nq=0)

Sparse pressure mass matrix `M_ij = int weight(x,y) q_i q_j` on the quadtree
pressure space.  This is the standard spectrally-equivalent preconditioner for
the Taylor–Hood Schur complement (Elman, Silvester & Wathen 2014, §4.2): use
`Cp' * M * Cp` as the preconditioner for `S = Bs K^{-1} Bt`.  The full matrix
integrates to the domain area with `weight=nothing`.

`nq = 0` keeps the historical `p+2` Gauss points; the strain-weakening assembly
(docs/dev/PHYSICS.md §9) passes the energy quadrature explicitly so that its
weighted mass matrix uses the same rule as the energy mass matrix.
"""
function assemble_semquad_pressure_mass(sp::SemQuad;
                                        weight::Union{Nothing,Function}=nothing,
                                        nq::Int=0)
    p = sp.p
    n = p + 1
    fd = make_fedata(2, p; nq=nq)   # Gauss points; fd.phi_p is the Q_p basis
    I = Int[]; J = Int[]; V = Float64[]
    for lf in sp.leaves
        hx = leaf_hx(sp, lf[1]); hy = leaf_hy(sp, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            weight === nothing || (w *= weight(x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx,
                                               y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy))
            for b in 1:n, a in 1:n
                pa = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                ga = pids[(b - 1) * n + a]
                for b2 in 1:n, a2 in 1:n
                    pb = fd.phi_p[a2, qx] * fd.phi_p[b2, qy]
                    gb = pids[(b2 - 1) * n + a2]
                    v = w * pa * pb
                    v != 0.0 && (push!(I, ga); push!(J, gb); push!(V, v))
                end
            end
        end
    end
    return sparse(I, J, V, sp.N, sp.N)
end

"""
    assemble_semquad_pressure_stiffness(sp; weight=nothing)

Sparse pressure stiffness `int weight(x,y) grad q_i . grad q_j` on the quadtree
pressure space.  With one pressure dof pinned (the caller's `Cp`) it is SPD and
can serve as a Schur-complement preconditioner for strongly variable viscosity,
where a (weighted) pressure mass matrix is a poorer approximation.
"""
function assemble_semquad_pressure_stiffness(sp::SemQuad;
                                              weight::Union{Nothing,Function}=nothing)
    p = sp.p
    n = p + 1
    fd = make_fedata(2, p)
    I = Int[]; J = Int[]; V = Float64[]
    for lf in sp.leaves
        hx = leaf_hx(sp, lf[1]); hy = leaf_hy(sp, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            weight === nothing || (w *= weight(x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx,
                                               y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy))
            for b in 1:n, a in 1:n
                gxa = fd.dphi_p[a, qx] * fd.phi_p[b, qy] * (2.0 / hx)
                gza = fd.phi_p[a, qx] * fd.dphi_p[b, qy] * (2.0 / hy)
                ga = pids[(b - 1) * n + a]
                for b2 in 1:n, a2 in 1:n
                    gxb = fd.dphi_p[a2, qx] * fd.phi_p[b2, qy] * (2.0 / hx)
                    gzb = fd.phi_p[a2, qx] * fd.dphi_p[b2, qy] * (2.0 / hy)
                    gb = pids[(b2 - 1) * n + a2]
                    v = w * (gxa * gxb + gza * gzb)
                    v != 0.0 && (push!(I, ga); push!(J, gb); push!(V, v))
                end
            end
        end
    end
    return sparse(I, J, V, sp.N, sp.N)
end

