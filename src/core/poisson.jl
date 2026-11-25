# SIPG/Nitsche Poisson operator: a verification-only assembler and its constrained solve.
"""Assemble the SIPG/Nitsche Poisson operator on the quadtree SEM mesh."""
function assemble_semquad_poisson(sq::SemQuad, qt::Quadtree, f::Function, uD::Function;
                                  kappa::Float64=1.0, sigma_C::Float64=50.0)
    p = sq.p
    n = p + 1
    N = sq.N
    nv = n * n
    I = Int[]; J = Int[]; V = Float64[]
    b = zeros(Float64, N)
    W = Diagonal(sq.w)
    DW = sq.D' * W * sq.D
    # Reference-space stiffness terms; the physical metric enters as the aspect
    # ratios hy/hx and hx/hy (both 1 for the unit square).
    K1 = kron(W, DW)        # d/dxi  d/dxi
    K2 = kron(DW, W)        # d/deta d/deta
    for lf in sq.leaves
        ids = sq.elem[lf]
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        Ke = kappa .* ((hy / hx) .* K1 .+ (hx / hy) .* K2)
        for kk in 1:nv, ll in 1:nv
            v = Ke[kk, ll]
            v != 0.0 && (push!(I, ids[kk]); push!(J, ids[ll]); push!(V, v))
        end
        x0 = lf[2] * hx; y0 = lf[3] * hy
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (sq.xi[i] + 1.0) * hx
            y = y0 + 0.5 * (sq.xi[j] + 1.0) * hy
            fv = f(x, y)
            # F5 guard: a non-finite load must be named here, not silently
            # turned into a NaN solution field.  (No interpolated string: the
            # success path must not allocate.)
            _require_load!(fv, "the Poisson load f(x, y)", x, y)
            b[ids[(j - 1) * n + i]] += 0.25 * hx * hy * sq.w[i] * sq.w[j] * fv
        end
    end
    # Nitsche boundaries (same formulas as the DG prototype).  The penalty uses
    # the face normal's inverse length: 1/hx on x-faces, 1/hy on y-faces.
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                sig = sigma_C * (p + 1)^2 / hx
                c = (2.0 / hx) .* sq.D[iF, :]
                s = dir === :xplus ? -1.0 : 1.0
                for j in 1:n
                    wf = 0.5 * hy * sq.w[j]
                    xf = dir === :xplus ? x0 + hx : x0
                    Td = uD(xf, y0 + 0.5 * (sq.xi[j] + 1.0) * hy)
                    gF = ids[(j - 1) * n + iF]
                    for i in 1:n
                        gi = ids[(j - 1) * n + i]
                        push!(I, gF); push!(J, gi); push!(V, s * wf * c[i])
                        push!(I, gi); push!(J, gF); push!(V, s * wf * c[i])
                        b[gi] += s * wf * c[i] * Td
                    end
                    push!(I, gF); push!(J, gF); push!(V, sig * wf)
                    b[gF] += sig * wf * Td
                end
            else
                jF = dir === :yminus ? 1 : n
                sig = sigma_C * (p + 1)^2 / hy
                c = (2.0 / hy) .* sq.D[jF, :]
                s = dir === :yplus ? -1.0 : 1.0
                for i in 1:n
                    wf = 0.5 * hx * sq.w[i]
                    yf = dir === :yplus ? y0 + hy : y0
                    Td = uD(x0 + 0.5 * (sq.xi[i] + 1.0) * hx, yf)
                    gF = ids[(jF - 1) * n + i]
                    for j in 1:n
                        gj = ids[(j - 1) * n + i]
                        push!(I, gF); push!(J, gj); push!(V, s * wf * c[j])
                        push!(I, gj); push!(J, gF); push!(V, s * wf * c[j])
                        b[gj] += s * wf * c[j] * Td
                    end
                    push!(I, gF); push!(J, gF); push!(V, sig * wf)
                    b[gF] += sig * wf * Td
                end
            end
        end
    end
    # F5 guard: also cover the Nitsche boundary data uD(x, y), whose values enter
    # the load through the boundary terms above.
    _require_finite!(b, "the assembled Poisson load vector b",
                     "the load / boundary-data closures of this solve")
    return sparse(I, J, V, N, N), b
end

"""Solve the constrained system (C' A C u = C' b) and return `(u, cons)`."""
function solve_semquad_poisson(sq::SemQuad, qt::Quadtree, f::Function, uD::Function;
                               kappa::Float64=1.0, sigma_C::Float64=50.0)
    A, b = assemble_semquad_poisson(sq, qt, f, uD; kappa=kappa, sigma_C=sigma_C)
    cons = semquad_constraints(sq, qt)
    free = [i for i in 1:sq.N if !haskey(cons, i)]
    fmap = Dict(v => k for (k, v) in enumerate(free))
    I = Int[]; J = Int[]; V = Float64[]
    for (k, i) in enumerate(free)
        push!(I, i); push!(J, k); push!(V, 1.0)
    end
    for (c, (ms, ws)) in cons, (m, w) in zip(ms, ws)
        push!(I, c); push!(J, fmap[m]); push!(V, w)
    end
    C = sparse(I, J, V, sq.N, length(free))
    ur = (C' * A * C) \ (C' * b)
    return C * ur, cons
end

