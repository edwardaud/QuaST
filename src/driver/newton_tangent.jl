# The matrix-free tangent-vector product of the per-step Newton.
#
# `_semquad_jvp_ku` evaluates the action of the perturbed velocity stiffness on
# the base velocity at each element's own quadrature points, without the global
# sparse assembly.  It was written for the steady Newton and is the only piece of
# that machinery the time-stepping Newton still needs, so it lives on its own.

"""
    _semquad_jvp_ku(sq, q, pr, theta, v, u)

Matrix-free `K(deta) * u` for the Newton JVP, with
`deta = (deta/dT) * delta-theta`.

The JVP only needs the action of the perturbed velocity stiffness on the base
velocity, but the historical path built the whole global K/B/constraint system
(`assemble_only=true`) just to use it.  This routine evaluates the temperature
and its perturbation at each element's own quadrature points from the local
nodal values (no tree search), builds the same element stiffness as the
production assembler (`deviatoric=false`, same quadrature), and applies it to
the element-local base velocity.  The cost is O(nleaves * (2*nv)^2) instead of
a global sparse assembly, and it matches the assembled `K * u` to round-off.
"""
function _semquad_jvp_ku(sq::SemQuad, q::Int, pr::Params,
                         theta::Vector{Float64}, v::Vector{Float64},
                         u::Vector{Float64};
                         deta_T_fun::Union{Nothing,Function}=nothing)
    N = sq.N
    p = sq.p
    n = p + 1
    nv = n * n
    fd = make_fedata(p, q)
    phi = zeros(Float64, nv)
    gx = zeros(Float64, nv)
    gz = zeros(Float64, nv)
    Kloc = zeros(Float64, 2nv, 2nv)
    uloc = zeros(Float64, 2nv)
    outl = zeros(Float64, 2nv)
    out = zeros(Float64, 2 * N)
    a2 = 2.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        fill!(Kloc, 0.0)
        for qy in 1:fd.nq, qx in 1:fd.nq
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            Tq = 0.0; dTq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                Tq += theta[ids[k]] * phi[k]
                dTq += v[ids[k]] * phi[k]
            end
            T = Tbar(y, pr.H) + Tq
            # `deta_T_fun` overrides the built-in d eta/d T of the temperature
            # laws (the composite/strain-rate-dependent laws need their own
            # derivative, which also has to see the strain rate).
            deta = deta_T_fun === nothing ? deta_dT_at(T, y, pr) :
                   deta_T_fun(x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx, y)
            etaq = deta * dTq
            for k in 1:nv, l in 1:nv
                Kloc[k, l] += etaq * (a2 * gx[k] * gx[l] + gz[k] * gz[l]) * w
                Kloc[nv + k, nv + l] += etaq * (a2 * gz[k] * gz[l] +
                                                gx[k] * gx[l]) * w
                Kloc[k, nv + l] += etaq * (gz[k] * gx[l]) * w
                Kloc[nv + k, l] += etaq * (gx[k] * gz[l]) * w
            end
        end
        for k in 1:nv
            uloc[k] = u[ids[k]]
            uloc[nv + k] = u[N + ids[k]]
        end
        mul!(outl, Kloc, uloc)
        for k in 1:nv
            out[ids[k]] += outl[k]
            out[N + ids[k]] += outl[nv + k]
        end
    end
    return out
end
