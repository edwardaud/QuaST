# Quadtree SEM with hanging-node constraints (stage A-AMR, step 1: Poisson).
#
# The continuous Q_p space lives on the quadtree leaves; nodes shared by
# conforming faces get one global id (geometric hash), and a node that lies on
# the face of a coarser leaf is constrained by the coarse face's GLL
# interpolation.  The constraints are applied as a sparse matrix C (identity on
# free dofs, interpolation weights on constrained rows) and the system is solved
# as C' A C u = C' b, which is exact for linear constraints and keeps symmetry.
# Boundary conditions are Nitsche, as in the DG prototype (same formulas).

struct SemQuad
    p::Int
    Lx::Float64                 # box width; the height is 1
    nx0::Int                    # base cells in x (see Quadtree)
    ny0::Int                    # base cells in y
    leaves::Vector{NTuple{3,Int}}
    coords::Vector{NTuple{2,Float64}}
    elem::Dict{NTuple{3,Int},Vector{Int}}
    N::Int
    xi::Vector{Float64}
    w::Vector{Float64}
    D::Matrix{Float64}
end

# Physical leaf extents; `nx0 == ny0 == 1, Lx == 1` is the historical unit square.
@inline leaf_hx(sq::SemQuad, l::Int) = sq.Lx / (sq.nx0 * (1 << l))
@inline leaf_hy(sq::SemQuad, l::Int) = 1.0 / (sq.ny0 * (1 << l))
@inline leaf_h(sq::SemQuad, l::Int) = leaf_hy(sq, l)

function make_semquad(qt::Quadtree, p::Int)
    p >= 1 || throw(ArgumentError("p must be >= 1"))
    rule = gll_rule_cached(p)
    n = p + 1
    D = zeros(Float64, n, n)
    for i in 1:n
        _, dL = lagrange_eval(rule, rule.nodes[i])
        for j in 1:n
            D[i, j] = dL[j]
        end
    end
    tol = 1.0e-12
    # One sorted leaf list for both the element build and the `leaves` field
    # (it used to sort the tree twice per mesh rebuild: `sort(collect(...))`
    # is 2.3 MB and ~0.3 ms per 10 000-leaf call, so the second copy was pure
    # waste; the sorted order is the same, hence bitwise identical).
    leaves = sort(collect(qt.leaves))
    coords = NTuple{2,Float64}[]
    idmap = Dict{NTuple{2,Int},Int}()
    elem = Dict{NTuple{3,Int},Vector{Int}}()
    for lf in leaves
        hx = leaf_hx(qt, lf[1]); hy = leaf_hy(qt, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = Vector{Int}(undef, n * n)
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (rule.nodes[i] + 1.0) * hx
            y = y0 + 0.5 * (rule.nodes[j] + 1.0) * hy
            key = (round(Int, x / tol), round(Int, y / tol))
            gid = get(idmap, key, 0)
            if gid == 0
                push!(coords, (x, y))
                gid = length(coords)
                idmap[key] = gid
            end
            ids[(j - 1) * n + i] = gid
        end
        elem[lf] = ids
    end
    SemQuad(p, qt.Lx, qt.nx0, qt.ny0, leaves, coords, elem, length(coords),
            copy(rule.nodes), copy(rule.weights), D)
end

"""
    semquad_mass(sq)

Global SEM mass matrix on `sq`, assembled with a `p+2` quadrature rule.
"""
function semquad_mass(sq::SemQuad)
    p = sq.p
    n = p + 1
    nv = n * n
    fd = make_fedata(p, p + 2)
    I = Int[]; J = Int[]; V = Float64[]
    # Every quadrature point contributes one nv x nv block (the duplicates are
    # summed by `sparse`), i.e. nq^2 * nv^2 entries per leaf.  Without the hint
    # the three arrays grew by repeated doubling: 1.37 GB per call on the
    # 10 000-leaf state, versus ~311 MB of exact capacity.
    sizehint!(I, fd.nq^2 * nv * nv * length(sq.leaves))
    sizehint!(J, fd.nq^2 * nv * nv * length(sq.leaves))
    sizehint!(V, fd.nq^2 * nv * nv * length(sq.leaves))
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi_k = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                for l in 1:n, m in 1:n
                    q = (l - 1) * n + m
                    phi_q = fd.phi_v[m, qx] * fd.phi_v[l, qy]
                    v = w * phi_k * phi_q
                    v != 0.0 && (push!(I, ids[k]); push!(J, ids[q]); push!(V, v))
                end
            end
        end
    end
    return sparse(I, J, V, sq.N, sq.N)
end

# The reduced mass matrix of the element, shared with the variable-viscosity
# assembly.
function assemble_semquad_velocity_mass(sq::SemQuad)
    M = semquad_mass(sq)
    return [M spzeros(sq.N, sq.N); spzeros(sq.N, sq.N) M]
end

# ---------------------------------------------------------------------------
# Energy equation on the quadtree SEM mesh.
# The operator is for theta = T - Tbar, with Tbar = 1-z+(H/2)z(1-z).  Since
# Tbar is the steady conductive profile (lap(Tbar) + H = 0), the theta equation
# is d(theta)/dt + u.grad(theta) = lap(theta) - u_z dTbar/dz; the +H source is
# already absorbed by Tbar and must NOT be added again.  The same convention is
# used by the uniform-mesh assembler (assembly.jl).
# ---------------------------------------------------------------------------

