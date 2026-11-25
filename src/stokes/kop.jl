# ============================================================================
# S3 / MF-1: matrix-free velocity stiffness `K * v`
# ============================================================================
#
# The assembled `K` of `solve_semquad_stokes_var` is a sum of leaf-local
# operators, and each leaf-local operator is a *tensor product* of 1-D GLL
# operators in the two grid directions.  Nothing about the apply needs the
# matrix: gather the `(p+1)^2` nodal values of each leaf, contract them to the
# four velocity-gradient components at the Gauss points, multiply by the local
# viscosity, and contract back.  That is `O(nq^2 n^2)` work per leaf
# (`n = p+1`, `nq = p+2`) instead of streaming a dense `2n^2 x 2n^2` block.
#
# The quadrature is the *same* Gauss-Legendre rule the assembler uses
# (`make_fedata(p, ss.q)`), and the viscosity is sampled at the same points, so
# `semquad_apply_K!` reproduces `K * v` to round-off for an arbitrary viscosity
# field -- not only for constant eta.  That exactness is what lets the MG
# hierarchy replace the sparse operator outright (MF-3) without changing the
# preconditioner's spectrum.
#
# Hanging constraints are *not* baked in: the operator is the unconstrained
# one, and the caller restricts/prolongs with the existing `Cu`/`Cp`
# (`semquad_apply_Kred!` does `Cu' * (K * (Cu * v))`), so `C'AC` is never
# formed.
#
# Performance: the innermost loop is always one of the Gauss-point directions
# (`nq` long), `phiT`/`dphiT` hold the transposed 1-D tables that the
# accumulation stage needs, and the leaf operator is specialised on the *static*
# polynomial degree (`Val(n)`), which lets LLVM unroll the length-`nq` loops
# (measured: 1.7x over the runtime-sized version, 0.31 ms vs 0.54 ms per apply
# at p=4 on 256 leaves).  The leaf-blocked kernel over `SEMQUAD_KBLK` lanes is
# `_semquad_kn_blk_core!` (stage D); its width is set so the lane axis survives
# to the vectoriser -- docs/dev/MG.md  The remaining cost is the
# gather/scatter, which stays scalar because a scatter has loop-carried
# dependences through shared nodes.

"""
    AbstractSemQuadKOp

Supertype of the leaf-local matrix-free velocity operators:

* [`SemQuadKOp`](@ref) — the historical box operator, whose per-leaf data are
  the seven box scalars (`2/hx`, `2/hy`, `hx*hy/4`) and the historical point
  map.  Its kernel bodies are the current expressions, character for character
  (GEOM_DESIGN §4: the box path must stay bitwise identical);
* [`SemQuadKOpMapped`](@ref) — the mapped operator of cut 5, whose per-leaf data
  are the cached `det J` and `J^{-T}` arrays plus the physical quadrature points.

Keeping them as two concrete types is what lets every kernel method be a
separate body (the box body is never routed through the general Jacobian,
GEOM_DESIGN §4.3 F1/F5), while the gather/scatter, the eta sampling/lanes, the
threaded block partition and the MG plumbing are shared: they only read the
common fields that both structs declare with the same names.
"""
abstract type AbstractSemQuadKOp end

"""
    SemQuadKOp

Leaf-local geometry and 1-D basis data for the matrix-free velocity stiffness.
`ids[:, lf]` are the velocity-node ids of leaf `lf` in the assembler's
`(j-1)*n + i` order, `ihx`/`ihy` the `2/h` reference-to-physical scalings, and
`xq`/`yq` the quadrature-point coordinates (used only when the viscosity is
supplied as a function).
"""
struct SemQuadKOp <: AbstractSemQuadKOp
    p::Int
    n::Int
    q::Int
    nq::Int
    deviatoric::Bool
    nleaf::Int
    N::Int
    gl_w::Vector{Float64}
    phi::Matrix{Float64}                 # n x nq, basis values at Gauss points
    dphi::Matrix{Float64}                # n x nq, basis derivatives
    phiT::Matrix{Float64}                # nq x n, transposed (accumulation)
    dphiT::Matrix{Float64}
    ids::Matrix{Int}                     # n*n x nleaf
    ihx::Vector{Float64}                 # 2/hx per leaf
    ihy::Vector{Float64}                 # 2/hy per leaf
    wfac::Vector{Float64}                # hx*hy/4 per leaf
    xq::Vector{Float64}                  # nleaf*nq*nq quadrature coordinates
    yq::Vector{Float64}
    blk::Int                             # leaves per blocked kernel (0 = scalar)
end

"""
    SemQuadKOpMapped

Leaf-local data of the **mapped** matrix-free velocity stiffness (cut 5).  It
carries the same gather/scatter and 1-D basis tables as [`SemQuadKOp`](@ref), but
instead of the box scalars it holds the per-leaf/per-quadrature-point metric
from the geometry cache (GEOM_DESIGN §2.5/§2.6):

* `detJ[k]` — the volume factor `det J` at quadrature point `k`;
* `j11..j22[k]` — the four entries of `J^{-T}` in the cut-1 order
  `(xi_x, eta_x, xi_y, eta_y)`;
* `xq/yq[k]` — the physical quadrature point (only for sampling a viscosity
  function, exactly like the box operator's `xq`/`yq`).

The arrays are flat and leaf-major (`(li-1)*nq^2 + k`), so the mapped kernels
index precomputed `Vector{Float64}`s and never evaluate the map inside a loop.
"""
struct SemQuadKOpMapped <: AbstractSemQuadKOp
    p::Int
    n::Int
    q::Int
    nq::Int
    deviatoric::Bool
    nleaf::Int
    N::Int
    gl_w::Vector{Float64}
    phi::Matrix{Float64}
    dphi::Matrix{Float64}
    phiT::Matrix{Float64}
    dphiT::Matrix{Float64}
    ids::Matrix{Int}
    detJ::Vector{Float64}                # nleaf*nq*nq
    j11::Vector{Float64}
    j12::Vector{Float64}
    j21::Vector{Float64}
    j22::Vector{Float64}
    xq::Vector{Float64}
    yq::Vector{Float64}
    blk::Int
end

# The two concrete operator types as a small union: the caches and the MG
# hierarchy store operators in heterogeneous containers, and a `Union` of two
# concrete types lets the compiler union-split the kernel call (a *type-stable*
# dispatch, measured 0 B per apply) where an `::AbstractSemQuadKOp` annotation
# would force a dynamic dispatch (measured 96 B per MG apply, i.e. the box path
# would have regressed against G222's zero-allocation assertion).
const AnySemQuadKOp = Union{SemQuadKOp,SemQuadKOpMapped}

"""
    SemQuadKScratch

The `SEMQUAD_KBLK`-lane scratch of the leaf-blocked kernel: the gathered nodal
values of one block, its velocity gradients at the Gauss points, the
tensor-product intermediate, the weighted gradients and the leaf-local output.
`SemQuadKWork` owns one instance for the serial path; the threaded path
(`SemQuadKPar`, S5-2) owns one per leaf block, so two threads never share a
lane buffer.
"""
struct SemQuadKScratch
    bxv::Matrix{Float64}                 # (BLK, n*n) gathered nodal x values
    bzv::Matrix{Float64}                 # (BLK, n*n) gathered nodal z values
    buxx::Array{Float64,3}               # (BLK, nq, nq) du_x/dx at Gauss points
    buxy::Array{Float64,3}               # (BLK, nq, nq) du_x/dy
    buzx::Array{Float64,3}               # (BLK, nq, nq) du_z/dx
    buzy::Array{Float64,3}               # (BLK, nq, nq) du_z/dy
    bt1::Array{Float64,3}                # (BLK, n, nq) tensor-product work
    bF::Array{Float64,3}                 # (BLK, nq, nq) weighted gradients
    # Cut-5 mapped kernels only; `nothing` on the box path so the box solver
    # allocates exactly the historical scratch (the mapped bodies assert them
    # non-nothing at their first use).
    bF2::Union{Nothing,Array{Float64,3}} # (BLK, nq, nq)
    bF3::Union{Nothing,Array{Float64,3}}
    bF4::Union{Nothing,Array{Float64,3}}
    box::Array{Float64,3}                # (BLK, n, n) leaf-local x output
    boz::Array{Float64,3}                # (BLK, n, n) leaf-local z output
    bshx::Vector{Float64}                # (BLK) 2/hx per block lane
    bshy::Vector{Float64}                # (BLK) 2/hy per block lane
    bswf::Vector{Float64}                # (BLK) hx hy / 4 per block lane
    bone::Vector{Float64}                # (BLK) 1.0, the unit contraction scale
    bkofs::Vector{Int}                   # (BLK) per-lane offset into etav
end

"""
    SemQuadKScratch(n, nq; mapped=false)

The SIMD-block scratch buffers.  `mapped=true` additionally allocates the three
extra weighted-gradient planes the cut-5 mapped blocked kernel needs; the box
kernel keeps the historical set (`mapped=false`), so a box hierarchy build or
apply allocates no mapped-only scratch (measured: exactly the pre-cut-5 bytes).
"""
function SemQuadKScratch(n::Int, nq::Int; mapped::Bool=false)
    B = SEMQUAD_KBLK
    xs = mapped ? (zeros(B, nq, nq), zeros(B, nq, nq), zeros(B, nq, nq)) :
                  (nothing, nothing, nothing)
    SemQuadKScratch(zeros(B, n * n), zeros(B, n * n),
                    zeros(B, nq, nq), zeros(B, nq, nq), zeros(B, nq, nq),
                    zeros(B, nq, nq), zeros(B, n, nq), zeros(B, nq, nq),
                    xs[1], xs[2], xs[3],
                    zeros(B, n, n), zeros(B, n, n),
                    zeros(Float64, B), zeros(Float64, B), zeros(Float64, B),
                    ones(Float64, B), zeros(Int, B))
end

"""
    SemQuadKWork

Preallocated scratch for `semquad_apply_K!`; one per operator, reused by every
smoother sweep so the apply does not allocate.

`bs` is the leaf-blocked scratch of the SIMD kernel (stage D) and is used by the
serial path only; the threaded path holds one `SemQuadKScratch` per block in
`SemQuadKPar` instead.
"""
struct SemQuadKWork
    xv::Matrix{Float64}
    zv::Matrix{Float64}
    uxx::Matrix{Float64}
    uxy::Matrix{Float64}
    uzx::Matrix{Float64}
    uzy::Matrix{Float64}
    t1::Matrix{Float64}
    t2::Matrix{Float64}
    F::Matrix{Float64}
    # Cut-5 mapped kernels only (`nothing` on the box path: the box assembler,
    # hierarchy build and apply allocate exactly the historical scratch).
    F2::Union{Nothing,Matrix{Float64}}
    F3::Union{Nothing,Matrix{Float64}}
    F4::Union{Nothing,Matrix{Float64}}
    ox::Matrix{Float64}
    oz::Matrix{Float64}
    full::Vector{Float64}
    op_out::Vector{Float64}
    bs::SemQuadKScratch
end

function SemQuadKWork(n::Int, nq::Int, N2::Int; mapped::Bool=false)
    Fs = mapped ? (zeros(nq, nq), zeros(nq, nq), zeros(nq, nq)) :
                  (nothing, nothing, nothing)
    SemQuadKWork(zeros(n, n), zeros(n, n), zeros(nq, nq), zeros(nq, nq),
                 zeros(nq, nq), zeros(nq, nq), zeros(n, nq), zeros(n, nq),
                 zeros(nq, nq), Fs[1], Fs[2], Fs[3],
                 zeros(n, n), zeros(n, n), zeros(N2), zeros(N2),
                 SemQuadKScratch(n, nq; mapped=mapped))
end

"""
    SemQuadKWork(op::AbstractSemQuadKOp)

Scratch sized for `op`, with the cut-5 mapped-only buffers allocated iff the
operator is mapped.  This is the constructor the solve/hierarchy builders use, so
a box run allocates exactly the pre-cut-5 scratch and a mapped run has everything
its kernel needs.
"""
function SemQuadKWork(op::AbstractSemQuadKOp)
    return SemQuadKWork(op.n, op.nq, 2 * op.N;
                        mapped = op isa SemQuadKOpMapped)
end

function SemQuadKOp(p::Int, n::Int, q::Int, nq::Int, deviatoric::Bool, nl::Int, N::Int,
                    gl_w::Vector{Float64}, phi::Matrix{Float64},
                    dphi::Matrix{Float64}, phiT::Matrix{Float64},
                    dphiT::Matrix{Float64}, ids::Matrix{Int},
                    ihx::Vector{Float64}, ihy::Vector{Float64},
                    wfac::Vector{Float64}, xq::Vector{Float64}, yq::Vector{Float64})
    SemQuadKOp(p, n, q, nq, deviatoric, nl, N, gl_w, phi, dphi, phiT, dphiT,
               ids, ihx, ihy, wfac, xq, yq, SEMQUAD_KBLK)
end

const SEMQUAD_KOP_CACHE = Dict{Any,AnySemQuadKOp}()

"""
    _semquad_kop_key(ss, qt, deviatoric) -> Tuple

The cache key of [`semquad_kop`](@ref): the historical layout/degree fields, the
leaf-set fingerprint and (cut 5) the geometry stamp.  Exposed so the gate can
assert that two geometries on one tree give two distinct keys (and that the
pre-cut key, which lacks the stamp, did not).
"""
function _semquad_kop_key(ss::SemStokes, qt::Quadtree, deviatoric::Bool)
    su = ss.su
    return (ss.p, ss.q, qt.nx0, qt.ny0, qt.Lx, deviatoric,
            length(qt.leaves), hash(qt.leaves), geometry_stamp(su.geom))
end

"""
    semquad_kop(ss, qt; deviatoric=false) -> AbstractSemQuadKOp

Build (or fetch from cache) the leaf-local data of the matrix-free `K`.

The cache key is the historical `(p, q, nx0, ny0, Lx, deviatoric, nleaf,
leafhash)` **plus the geometry stamp** (GEOM_DESIGN §2.5; cut 5).  Without the
geometry the same tree under two geometries (`CartesianBox` and an annulus, or
two validation geometries) would share one operator: a mapped space would get a
box operator or vice versa — the collision class §8 risk 5 warns about and G271
(d) measures.  The box and the mapped geometry also produce different *types*
(`SemQuadKOp` vs `SemQuadKOpMapped`), so a wrong hit is a wrong operator, not a
wrong buffer.

Invalidation rule: the operator is a pure function of the key (tree leaf set,
layout, degree, viscosity input is passed to the apply, not stored here) and the
entry is never invalidated in place; a tree change or a geometry change is a new
key.  `empty!(SEMQUAD_KOP_CACHE)` (or the process exit) is the only clearing
operation, as before.
"""
function semquad_kop(ss::SemStokes, qt::Quadtree; deviatoric::Bool=false)
    key = _semquad_kop_key(ss, qt, deviatoric)
    if haskey(SEMQUAD_KOP_CACHE, key)
        return SEMQUAD_KOP_CACHE[key]::AnySemQuadKOp
    end
    op = ss.su.geom isa CartesianBox ?
         _semquad_kop_box(ss, qt, deviatoric) :
         _semquad_kop_mapped(ss, qt, deviatoric)
    SEMQUAD_KOP_CACHE[key] = op
    return op
end

# Historical box body: the seven box scalars per leaf and the historical point
# map, character for character (GEOM_DESIGN E6).  Never reachable through the
# mapped branch below.
function _semquad_kop_box(ss::SemStokes, qt::Quadtree, deviatoric::Bool)
    su = ss.su
    p = ss.p
    q = ss.q
    fd = make_fedata(p, q)
    n = p + 1
    nq = fd.nq
    leaves = sort(collect(qt.leaves))
    nl = length(leaves)
    ids = Matrix{Int}(undef, n * n, nl)
    ihx = Vector{Float64}(undef, nl)
    ihy = Vector{Float64}(undef, nl)
    wfac = Vector{Float64}(undef, nl)
    xq = Vector{Float64}(undef, nl * nq * nq)
    yq = Vector{Float64}(undef, nl * nq * nq)
    for (li, lf) in enumerate(leaves)
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        for idx in 1:(n * n)
            ids[idx, li] = uids[idx]
        end
        ihx[li] = 2.0 / hx; ihy[li] = 2.0 / hy; wfac[li] = 0.25 * hx * hy
        k = (li - 1) * nq * nq
        for qy in 1:nq, qx in 1:nq
            k += 1
            xq[k] = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            yq[k] = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
        end
    end
    return SemQuadKOp(p, n, q, nq, deviatoric, nl, su.N, fd.gl_w,
                      copy(fd.phi_v), copy(fd.dphi_v),
                      Matrix(transpose(fd.phi_v)), Matrix(transpose(fd.dphi_v)),
                      ids, ihx, ihy, wfac, xq, yq, SEMQUAD_KBLK)
end

# Mapped body: the per-leaf/per-quadrature-point metric from the geometry cache
# (GEOM_DESIGN §2.5/§2.6).  The leaf order is the cache's canonical sorted order,
# which is also the box operator's `sort(collect(qt.leaves))` order.
function _semquad_kop_mapped(ss::SemStokes, qt::Quadtree, deviatoric::Bool)
    su = ss.su
    g = su.geom
    p = ss.p
    q = ss.q
    fd = make_fedata(p, q)
    n = p + 1
    nq = fd.nq
    n2 = nq * nq
    table = _geom_cache(g, su, fd)
    lm = table.lm
    lm === nothing && (lm = _ensure_leaf_metrics!(table))
    leaves = table.leaves
    nl = length(leaves)
    nl == length(qt.leaves) ||
        throw(ArgumentError("the mapped operator's space has $(nl) leaves but " *
                            "the tree has $(length(qt.leaves))"))
    ids = Matrix{Int}(undef, n * n, nl)
    detJ = Vector{Float64}(undef, nl * n2)
    j11 = Vector{Float64}(undef, nl * n2)
    j12 = Vector{Float64}(undef, nl * n2)
    j21 = Vector{Float64}(undef, nl * n2)
    j22 = Vector{Float64}(undef, nl * n2)
    xq = Vector{Float64}(undef, nl * n2)
    yq = Vector{Float64}(undef, nl * n2)
    @inbounds for li in 1:nl
        lf = leaves[li]
        uids = su.elem[lf]
        for idx in 1:(n * n)
            ids[idx, li] = uids[idx]
        end
        m = (lm::Vector{MappedLeafMetric})[table.index[lf]]
        k0 = (li - 1) * n2
        for k in 1:n2
            detJ[k0 + k] = m.detJ[k]
            j11[k0 + k] = m.jinv[1][k]
            j12[k0 + k] = m.jinv[2][k]
            j21[k0 + k] = m.jinv[3][k]
            j22[k0 + k] = m.jinv[4][k]
            xq[k0 + k] = m.xq[k]
            yq[k0 + k] = m.yq[k]
        end
    end
    return SemQuadKOpMapped(p, n, q, nq, deviatoric, nl, su.N, fd.gl_w,
                            copy(fd.phi_v), copy(fd.dphi_v),
                            Matrix(transpose(fd.phi_v)),
                            Matrix(transpose(fd.dphi_v)), ids,
                            detJ, j11, j12, j21, j22, xq, yq, SEMQUAD_KBLK)
end

"""Viscosity sampled once at every leaf/quadrature point of `op`."""
function semquad_eta_field(op::AbstractSemQuadKOp, eta::Function)
    vals = Vector{Float64}(undef, length(op.xq))
    @inbounds for i in eachindex(vals)
        x = op.xq[i]
        y = op.yq[i]
        e = eta(x, y)
        # F5/F6 guard (cold throw path): the matrix-free/MG paths sample the raw
        # closure here, so a non-finite or non-positive viscosity must be
        # refused at the sample rather than become part of the operator.
        _require_eta!(e, x, y)
        vals[i] = e
    end
    return vals
end

@inline function _semquad_resolve_eta(op::AbstractSemQuadKOp, eta)
    eta isa Function && return semquad_eta_field(op, eta)
    length(eta) == length(op.xq) ||
        throw(ArgumentError("eta has $(length(eta)) values, expected one per " *
                            "leaf/quadrature point ($(length(op.xq)))"))
    return eta
end

"""
    SemQuadEtaLane

The viscosity field of one operator, sampled once per leaf/quadrature point and
held in **two layouts**:

* `scalar` -- the leaf-major field `nleaf*nq^2`, i.e. the layout
  [`semquad_eta_field`](@ref) returns and the scalar kernel, the Jacobi diagonal
  and the leaf stiffness read (`etav[(li-1)*nq^2 + kq]`);
* `lane` -- the **lane-major table** the leaf-blocked kernel reads:

      lane[(bi-1)*nq2*BLK + (kq-1)*BLK + b] = scalar[((bi-1)*BLK + b - 1)*nq2 + kq]

  one contiguous `SEMQUAD_KBLK`-lane vector per block and quadrature point.  The
  last block is padded with its **final leaf**, which is exactly the leaf the
  kernel's padded lanes already read (their weight is `0.0`), so the padded
  lanes contribute the same `0.0` as before.

The values are the same `Float64`s in the same order; only the addressing moves
-- a unit-stride lane read instead of the per-lane gather `etav[kofs[b] + kq]`
-- so every apply stays **bitwise** (G225).  This is the
`Table<2,VectorizedArray>` pre-gather of deal.II's step-37
(docs/dev/MG_LITERATURE.md item 1) at `SEMQUAD_KBLK = 32`.

Subtyping `AbstractVector{Float64}` (indexing leaf-major, i.e. through `scalar`)
lets every existing eta consumer keep working unchanged.
"""
struct SemQuadEtaLane <: AbstractVector{Float64}
    scalar::Vector{Float64}
    lane::Vector{Float64}
    nleaf::Int
    nq2::Int
end

Base.size(e::SemQuadEtaLane) = (length(e.scalar),)
Base.length(e::SemQuadEtaLane) = length(e.scalar)
Base.IndexStyle(::Type{SemQuadEtaLane}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(e::SemQuadEtaLane, i::Int) = e.scalar[i]
Base.parent(e::SemQuadEtaLane) = e.scalar

"""
    semquad_eta_lanes(op, etav) -> SemQuadEtaLane

Build the lane-major table of `etav` (a leaf-major field of `op`, or an existing
`SemQuadEtaLane`, which is returned unchanged).  Called **once per operator
construction** -- per level in the MG hierarchy -- never per apply: it is a pure
re-layout, so it changes no value and no arithmetic (G225).

Cost: one pass over `nleaf*nq^2` doubles and `nblk*nq^2*BLK` of storage, i.e.
the size of the field rounded up to a whole number of `SEMQUAD_KBLK` blocks
(`F` level 6: 131 008 -> 131 072 doubles, `P1` level 7: 160 000 -> 160 256).
"""
function semquad_eta_lanes(op::AbstractSemQuadKOp, etav::AbstractVector{Float64})
    nq2 = op.nq * op.nq
    nl = op.nleaf
    length(etav) == nl * nq2 ||
        throw(ArgumentError("eta has $(length(etav)) values, expected one per " *
                            "leaf/quadrature point ($(nl * nq2))"))
    BLK = SEMQUAD_KBLK
    nblk = cld(nl, BLK)
    lane = Vector{Float64}(undef, nblk * nq2 * BLK)
    @inbounds for bi in 1:nblk
        lo = (bi - 1) * BLK + 1
        hi = min(nl, bi * BLK)
        tb = (bi - 1) * nq2 * BLK
        for kq in 1:nq2
            kb = tb + (kq - 1) * BLK
            for b in 1:BLK
                # the last block repeats its final leaf: the kernel's padded
                # lane reads the same leaf and is weighted by 0.0
                li = lo + min(b - 1, hi - lo)
                lane[kb + b] = etav[(li - 1) * nq2 + kq]
            end
        end
    end
    scalar = etav isa Vector{Float64} ? etav : collect(etav)
    return SemQuadEtaLane(scalar, lane, nl, nq2)
end

function semquad_eta_lanes(op::AbstractSemQuadKOp, e::SemQuadEtaLane)
    (e.nleaf == op.nleaf && e.nq2 == op.nq * op.nq) ||
        throw(ArgumentError("SemQuadEtaLane was built for $(e.nleaf) leaves x " *
                            "$(e.nq2) quadrature points, but this operator has " *
                            "$(op.nleaf) x $(op.nq * op.nq)"))
    return e
end

# dest[qx,qy] = sb*sc * sum_{i,j} A[i,j] * B[i,qx] * C[j,qy]
# (one tensor-product gradient contraction; qy is the innermost index)
@inline function _semquad_contract!(dest::Matrix{Float64}, A::Matrix{Float64},
                                    B::Matrix{Float64}, C::Matrix{Float64},
                                    sb::Float64, sc::Float64,
                                    tmp::Matrix{Float64}, ::Val{N}) where {N}
    nq = N + 1
    @inbounds begin
        for qy in 1:nq, i in 1:N
            tmp[i, qy] = 0.0
        end
        for j in 1:N, i in 1:N
            a = A[i, j]
            for qy in 1:nq
                tmp[i, qy] += a * C[j, qy]
            end
        end
        for qy in 1:nq, i in 1:N
            tmp[i, qy] *= sc
        end
        for qy in 1:nq, qx in 1:nq
            dest[qx, qy] = 0.0
        end
        for qx in 1:nq, i in 1:N
            b = B[i, qx] * sb
            for qy in 1:nq
                dest[qx, qy] += tmp[i, qy] * b
            end
        end
    end
    return nothing
end

# dest[i,j] += sum_{qx,qy} B[i,qx] * F[qx,qy] * CT[qy,j]
# (one tensor-product accumulation; j is the innermost index)
@inline function _semquad_accum!(dest::Matrix{Float64}, B::Matrix{Float64},
                                 F::Matrix{Float64}, CT::Matrix{Float64},
                                 tmp::Matrix{Float64}, ::Val{N}) where {N}
    nq = N + 1
    @inbounds begin
        for qy in 1:nq, i in 1:N
            tmp[i, qy] = 0.0
        end
        for qx in 1:nq, i in 1:N
            b = B[i, qx]
            for qy in 1:nq
                tmp[i, qy] += b * F[qx, qy]
            end
        end
        for qy in 1:nq, i in 1:N
            t = tmp[i, qy]
            for j in 1:N
                dest[i, j] += t * CT[qy, j]
            end
        end
    end
    return nothing
end

"""
    SEMQUAD_MG_ETA_LANES

`true` (the default) has the leaf-blocked kernel read the viscosity through the
block-major lane table of [`SemQuadEtaLane`](@ref) -- one unit-stride
`SEMQUAD_KBLK`-lane read per quadrature point instead of a per-lane gather.
`false` restores the gather form `etav[kofs[b] + kq]` for the A/B; the two read
the same values, so the switch changes the instruction mix only, never a bit
(docs/dev/MG.md, G225).  The switch is consulted only when the eta field
carries a lane table, i.e. never for a plain `Vector{Float64}` field.
"""
const SEMQUAD_MG_ETA_LANES = Ref(true)

"""
    SEMQUAD_KBLK

Leaves per blocked kernel pass (stage D).  The block index is the innermost
loop index, so the `(p+1)^2` and `nq^2` inner loops become unit-stride vector
operations across `SEMQUAD_KBLK` independent leaves; the leaf-specific scalars
(`2/hx`, `2/hy`, `hx hy/4`, `eta`) are gathered into length-`BLK` vectors first.

**32, not 8** (docs/dev/MG.md): at `BLK = 8` LLVM fully unrolls the
8-lane loop into scalar `vaddsd`/`vmulsd` and the kernel is 2.0-2.4x slower
than at 16-32 lanes, where the loop survives to the vectoriser and becomes
`vaddpd`/`vmulpd`/`vsubpd` (4-wide AVX2).  32 is the measured best (1 thread,
F: raw apply 2 150 us at 8, 835 at 16, 789 at 32, 891 at 64).  A short final
block is padded with a zero-weight copy of its last leaf, so the padded lanes
contribute exactly `0.0` and the result stays bitwise (G180).
"""
const SEMQUAD_KBLK = 32

# dest[b,qx,qy] = sb[b]*sc[b] * sum_{i,j} A[b,(j-1)N+i] * B[i,qx] * C[j,qy]
@inline function _semquad_contract_blk!(dest::Array{Float64,3},
                                        A::Matrix{Float64},
                                        B::Matrix{Float64}, C::Matrix{Float64},
                                        sb::Vector{Float64}, sc::Vector{Float64},
                                        tmp::Array{Float64,3},
                                        ::Val{N}) where {N}
    nq = N + 1
    @inbounds begin
        for qy in 1:nq, i in 1:N
            @simd ivdep for b in 1:SEMQUAD_KBLK
                tmp[b, i, qy] = 0.0
            end
            for j in 1:N
                c = C[j, qy]
                @simd ivdep for b in 1:SEMQUAD_KBLK
                    tmp[b, i, qy] += A[b, (j - 1) * N + i] * c
                end
            end
            @simd ivdep for b in 1:SEMQUAD_KBLK
                tmp[b, i, qy] *= sc[b]
            end
        end
        for qy in 1:nq, qx in 1:nq
            @simd ivdep for b in 1:SEMQUAD_KBLK
                dest[b, qx, qy] = 0.0
            end
            for i in 1:N
                bi = B[i, qx]
                @simd ivdep for b in 1:SEMQUAD_KBLK
                    dest[b, qx, qy] += tmp[b, i, qy] * (bi * sb[b])
                end
            end
        end
    end
    return nothing
end

# dest[b,(j-1)N+i] += sum_{qx,qy} B[i,qx] * F[b,qx,qy] * CT[qy,j]
@inline function _semquad_accum_blk!(dest::Array{Float64,3}, B::Matrix{Float64},
                                     F::Array{Float64,3}, CT::Matrix{Float64},
                                     tmp::Array{Float64,3},
                                     ::Val{N}) where {N}
    nq = N + 1
    @inbounds begin
        for qy in 1:nq, i in 1:N
            @simd ivdep for b in 1:SEMQUAD_KBLK
                tmp[b, i, qy] = 0.0
            end
            for qx in 1:nq
                bi = B[i, qx]
                @simd ivdep for b in 1:SEMQUAD_KBLK
                    tmp[b, i, qy] += bi * F[b, qx, qy]
                end
            end
        end
        for qy in 1:nq, i in 1:N
            for j in 1:N
                ctj = CT[qy, j]
                @simd ivdep for b in 1:SEMQUAD_KBLK
                    dest[b, (j - 1) * N + i] += tmp[b, i, qy] * ctj
                end
            end
        end
    end
    return nothing
end

# The SEMQUAD_KBLK leaves of one block; `nb < BLK` at the end of the leaf list.
# A short final block is padded with a zero-weight copy of its last leaf.
function _semquad_apply_K_blk!(out::Vector{Float64}, op::SemQuadKOp,
                               etav::AbstractVector{Float64},
                               v::AbstractVector{Float64}, work::SemQuadKWork,
                               nb::Int)
    nb = clamp(nb, 1, SEMQUAD_KBLK)
    n = op.n
    n == 2 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(2))
    n == 3 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(3))
    n == 4 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(4))
    n == 5 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(5))
    n == 6 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(6))
    n == 7 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(7))
    n == 8 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(8))
    return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(n))
end

"""
    _semquad_kn_blk!(out, op, etav, v, work, nb, ::Val{N})

Leaf-blocked matrix-free velocity stiffness: the same operator as
`_semquad_kn!`, `SEMQUAD_KBLK` leaves per pass with the block index innermost.
Every inner loop has the *constant* trip count `SEMQUAD_KBLK` (the final block
is padded with zero-weight leaves), which is what lets LLVM vectorise across the
block and unroll the `(p+1)^2` / `nq^2` loops.

The per-leaf arithmetic (contraction order, accumulation order) is unchanged, so
the result is **bitwise** identical to the scalar kernel -- G180 checks it.
"""
function _semquad_kn_blk!(out::Vector{Float64}, op::SemQuadKOp,
                          etav::AbstractVector{Float64},
                          v::AbstractVector{Float64}, work::SemQuadKWork,
                          nbmax::Int, ::Val{N}) where {N}
    return _semquad_kn_blk_core!(out, nothing, nothing, op, etav, v, work.bs,
                                 1, op.nleaf, nbmax, Val(N), Val(false))
end

"""
    _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, ::Val{N})

The element-buffer form of `_semquad_kn_blk!`: process the leaves `lo:hi` of
the block `scr` and write each leaf's element vector into the columns
`lo:hi` of `elx`/`elz` (each `(n*n) x nleaf`, leaf-major) **instead of**
scattering into the node vector.  The arithmetic is the same code
(`_semquad_kn_blk_core!` with `Val(true)`), so the element vectors are bitwise
the ones the serial kernel would have added.  The caller scatters them serially
in leaf order (`_semquad_scatter_elts!`); see `SemQuadKPar`.
"""
function _semquad_kn_blk_store!(op::SemQuadKOp, etav::AbstractVector{Float64},
                                v::AbstractVector{Float64},
                                scr::SemQuadKScratch, lo::Int, hi::Int,
                                elx::Matrix{Float64}, elz::Matrix{Float64},
                                ::Val{N}) where {N}
    return _semquad_kn_blk_core!(nothing, elx, elz, op, etav, v, scr, lo, hi,
                                 SEMQUAD_KBLK, Val(N), Val(true))
end

# The shared body of the serial blocked kernel and of the threaded
# element-buffer kernel.  `STORE=false` scatters the leaf-local vectors straight
# into `out` (the serial path, unchanged and bitwise); `STORE=true` writes them
# into the `li`-th column of `elx`/`elz`, and `out`/`elx`/`elz` that are not used
# are `nothing` (the branch is on a `Val`, so it disappears at compile time).
function _semquad_kn_blk_core!(out::Union{Vector{Float64},Nothing},
                               elx::Union{Matrix{Float64},Nothing},
                               elz::Union{Matrix{Float64},Nothing},
                               op::SemQuadKOp, etav::AbstractVector{Float64},
                               v::AbstractVector{Float64},
                               scr::SemQuadKScratch, lo::Int, hi::Int,
                               nbmax::Int, ::Val{N},
                               ::Val{STORE}) where {N,STORE}
    # The eta read is the only difference between the two kernel forms and it is
    # a pure addressing choice (docs/dev/MG.md): with a lane table the
    # lane axis is unit stride, so LLVM emits a vector load instead of the
    # per-lane `vgather` over `etav[kofs[b] + kq]`.  Both read the same
    # `Float64`s, so the result is bitwise (G225).  `SEMQUAD_MG_ETA_LANES =
    # false` picks the gather form for the A/B; a plain `Vector` field has no
    # table and always takes it.
    if etav isa SemQuadEtaLane && SEMQUAD_MG_ETA_LANES[]
        return _semquad_kn_blk_body!(out, elx, elz, op, etav.lane, v, scr,
                                     lo, hi, nbmax, Val(N), Val(STORE),
                                     Val(true))
    end
    return _semquad_kn_blk_body!(out, elx, elz, op, etav, v, scr, lo, hi,
                                 nbmax, Val(N), Val(STORE), Val(false))
end

# `LANE=true`: eta of lane `b` of block `bi` comes from the block-major lane
# table `etab[(bi-1)*nq2*BLK + (kq-1)*BLK + b]` (unit stride in `b`);
# `LANE=false`: from the leaf-major field `etab[kofs[b] + kq]` (per-lane gather).
function _semquad_kn_blk_body!(out::Union{Vector{Float64},Nothing},
                               elx::Union{Matrix{Float64},Nothing},
                               elz::Union{Matrix{Float64},Nothing},
                               op::SemQuadKOp, etab::AbstractVector{Float64},
                               v::AbstractVector{Float64},
                               scr::SemQuadKScratch, lo::Int, hi::Int,
                               nbmax::Int, ::Val{N}, ::Val{STORE},
                               ::Val{LANE}) where {N,STORE,LANE}
    BLK = SEMQUAD_KBLK
    nq = N + 1
    N_ = op.N
    n2 = N * N
    nq2 = nq * nq
    phi = op.phi; dphi = op.dphi; phiT = op.phiT; dphiT = op.dphiT
    gl_w = op.gl_w
    ids = op.ids
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    c2 = op.deviatoric ? 2.0 / 3.0 : 0.0
    xv = scr.bxv; zv = scr.bzv
    uxx = scr.buxx; uxy = scr.buxy; uzx = scr.buzx; uzy = scr.buzy
    t1 = scr.bt1
    F = scr.bF; ox = scr.box; oz = scr.boz
    ihx = op.ihx; ihy = op.ihy; wfac = op.wfac
    # per-leaf scalars and per-leaf offsets into `etav`, gathered once.  These
    # live in the scratch (not `zeros(BLK)` here) so the apply is allocation-free.
    shx = scr.bshx; shy = scr.bshy
    swf = scr.bswf
    one = scr.bone
    kofs = scr.bkofs
    STORE || fill!(out, 0.0)
    @inbounds for lb in lo:BLK:hi
        nb = min(BLK, hi - lb + 1)
        lplast = lb + nb - 1
        # lane-table base of this block.  `lb` is on the `SEMQUAD_KBLK` block
        # grid: the serial call starts at 1 and the threaded chunk bounds are
        # block-aligned (`SemQuadKPar`), so `(lb-1) ÷ BLK` is the block index.
        tbase = LANE ? (((lb - 1) ÷ BLK) * (nq2 * BLK)) : 0
        for b in 1:BLK
            # Pad a short final block with a zero-weight copy of its last leaf:
            # the gather stays in bounds and every contribution is exactly 0.0,
            # so the padded lanes do not change the result.
            li = b <= nb ? (lb + b - 1) : lplast
            shx[b] = ihx[li]; shy[b] = ihy[li]
            swf[b] = b <= nb ? wfac[li] : 0.0
            LANE || (kofs[b] = (li - 1) * nq2)
            for idx in 1:n2
                g = ids[idx, li]
                xv[b, idx] = v[g]
                zv[b, idx] = v[N_ + g]
            end
        end
        _semquad_contract_blk!(uxx, xv, dphi, phi, shx, one, t1, Val(N))
        _semquad_contract_blk!(uxy, xv, phi, dphi, one, shy, t1, Val(N))
        _semquad_contract_blk!(uzx, zv, dphi, phi, shx, one, t1, Val(N))
        _semquad_contract_blk!(uzy, zv, phi, dphi, one, shy, t1, Val(N))
        for idx in 1:n2
            # idx outer / b inner: the lane stores are unit-stride
            @simd ivdep for b in 1:BLK
                ox[b, idx] = 0.0
                oz[b, idx] = 0.0
            end
        end
        # same four separable pieces as the scalar kernel
        for qy in 1:nq, qx in 1:nq
            gw = gl_w[qx] * gl_w[qy]
            kq = (qy - 1) * nq + qx
            kb = LANE ? (tbase + (kq - 1) * BLK) : 0
            @simd ivdep for b in 1:BLK
                etaq = LANE ? etab[kb + b] : etab[kofs[b] + kq]
                cw1 = swf[b] * gw * etaq * shx[b]
                F[b, qx, qy] = cw1 * (a2 * uxx[b, qx, qy] - c2 * uzy[b, qx, qy])
            end
        end
        _semquad_accum_blk!(ox, dphi, F, phiT, t1, Val(N))
        for qy in 1:nq, qx in 1:nq
            gw = gl_w[qx] * gl_w[qy]
            kq = (qy - 1) * nq + qx
            kb = LANE ? (tbase + (kq - 1) * BLK) : 0
            @simd ivdep for b in 1:BLK
                etaq = LANE ? etab[kb + b] : etab[kofs[b] + kq]
                cw2 = swf[b] * gw * etaq * shy[b]
                F[b, qx, qy] = cw2 * (uxy[b, qx, qy] + uzx[b, qx, qy])
            end
        end
        _semquad_accum_blk!(ox, phi, F, dphiT, t1, Val(N))
        for qy in 1:nq, qx in 1:nq
            gw = gl_w[qx] * gl_w[qy]
            kq = (qy - 1) * nq + qx
            kb = LANE ? (tbase + (kq - 1) * BLK) : 0
            @simd ivdep for b in 1:BLK
                etaq = LANE ? etab[kb + b] : etab[kofs[b] + kq]
                cw2 = swf[b] * gw * etaq * shy[b]
                F[b, qx, qy] = cw2 * (a2 * uzy[b, qx, qy] - c2 * uxx[b, qx, qy])
            end
        end
        _semquad_accum_blk!(oz, phi, F, dphiT, t1, Val(N))
        for qy in 1:nq, qx in 1:nq
            gw = gl_w[qx] * gl_w[qy]
            kq = (qy - 1) * nq + qx
            kb = LANE ? (tbase + (kq - 1) * BLK) : 0
            @simd ivdep for b in 1:BLK
                etaq = LANE ? etab[kb + b] : etab[kofs[b] + kq]
                cw1 = swf[b] * gw * etaq * shx[b]
                F[b, qx, qy] = cw1 * (uzx[b, qx, qy] + uxy[b, qx, qy])
            end
        end
        _semquad_accum_blk!(oz, dphi, F, phiT, t1, Val(N))
        if STORE
            for b in 1:BLK
                b <= nb || continue          # padded lanes write nothing
                li = lb + b - 1
                for idx in 1:n2
                    elx[idx, li] = ox[b, idx]
                    elz[idx, li] = oz[b, idx]
                end
            end
        else
            for b in 1:BLK
                b <= nb || continue          # padded lanes write nothing
                li = lb + b - 1
                for idx in 1:n2
                    g = ids[idx, li]
                    out[g] += ox[b, idx]
                    out[N_ + g] += oz[b, idx]
                end
            end
        end
    end
    if STORE
        return nothing
    end
    return out
end

"""
    _semquad_kn_blk_mapped!(out, op, etav, v, work, nb, ::Val{N})

Leaf-blocked matrix-free velocity stiffness for the **mapped** operator (cut 5,
GEOM_DESIGN §2.6).  Same block/lane structure, scratch reuse and scatter order as
the box kernel [`_semquad_kn_blk!`](@ref) — only the element arithmetic differs:

* the four tensor-product contractions produce the **raw reference** derivatives
  `u_{x,xi}, u_{x,eta}, u_{z,xi}, u_{z,eta}` (the box kernel folds `2/hx`/`2/hy`
  into the contraction here instead);
* at every quadrature point the physical gradient is the chain rule
  `u_{x,x} = J11 u_{x,xi} + J12 u_{x,eta}`, `u_{x,y} = J21 u_{x,xi} +
  J22 u_{x,eta}` (and likewise for `u_z`) with the cached `J^{-T}`, and the
  weighted reference-space right-hand sides are the design's four pieces

      Fx_xi  = W[(a2 ux_x - c2 uz_y) J11 + (ux_y + uz_x) J21]
      Fx_eta = W[(a2 ux_x - c2 uz_y) J12 + (ux_y + uz_x) J22]
      Fz_xi  = W[(ux_y + uz_x) J11 + (a2 uz_y - c2 ux_x) J21]
      Fz_eta = W[(ux_y + uz_x) J12 + (a2 uz_y - c2 ux_x) J22]

  with `W = detJ * gl_w[qx] * gl_w[qy] * eta`;
* the four accumulations are the box kernel's two separable pieces, so the cost
  is two extra multiply-adds and seven array reads per quadrature point.

The kernel is bitwise-identical to its scalar twin
[`_semquad_kn_mapped!`](@ref) in the sense the box blocked/scalar pair is
(identical per-leaf order, added in leaf order), and it matches the mapped
sparse `K` to round-off (G271 (a)).  The box operator is never routed through
this body (GEOM_DESIGN §4.3 F5).
"""
function _semquad_kn_blk_mapped!(out::Vector{Float64}, op::SemQuadKOpMapped,
                                 etav::AbstractVector{Float64},
                                 v::AbstractVector{Float64},
                                 work::SemQuadKWork, nbmax::Int,
                                 ::Val{N}) where {N}
    return _semquad_kn_blk_mapped_core!(out, nothing, nothing, op, etav, v,
                                        work.bs, 1, op.nleaf, nbmax, Val(N),
                                        Val(false))
end

"""
    _semquad_kn_blk_mapped_store!(op, etav, v, scr, lo, hi, elx, elz, ::Val{N})

Element-buffer form of [`_semquad_kn_blk_mapped!`](@ref) for the threaded
V-cycle (the S5-2 pattern): the leaves `lo:hi` of the block are computed with
the mapped kernel and their element vectors stored in the columns of
`elx`/`elz`; the caller scatters them serially in leaf order, which is exactly
the serial accumulation order.
"""
function _semquad_kn_blk_mapped_store!(op::SemQuadKOpMapped,
                                       etav::AbstractVector{Float64},
                                       v::AbstractVector{Float64},
                                       scr::SemQuadKScratch, lo::Int, hi::Int,
                                       elx::Matrix{Float64},
                                       elz::Matrix{Float64},
                                       ::Val{N}) where {N}
    return _semquad_kn_blk_mapped_core!(nothing, elx, elz, op, etav, v, scr,
                                        lo, hi, SEMQUAD_KBLK, Val(N), Val(true))
end

function _semquad_kn_blk_mapped_core!(out::Union{Vector{Float64},Nothing},
                                      elx::Union{Matrix{Float64},Nothing},
                                      elz::Union{Matrix{Float64},Nothing},
                                      op::SemQuadKOpMapped,
                                      etav::AbstractVector{Float64},
                                      v::AbstractVector{Float64},
                                      scr::SemQuadKScratch, lo::Int, hi::Int,
                                      nbmax::Int, ::Val{N},
                                      ::Val{STORE}) where {N,STORE}
    if etav isa SemQuadEtaLane && SEMQUAD_MG_ETA_LANES[]
        return _semquad_kn_blk_mapped_body!(out, elx, elz, op, etav.lane, v,
                                            scr, lo, hi, nbmax, Val(N),
                                            Val(STORE), Val(true))
    end
    return _semquad_kn_blk_mapped_body!(out, elx, elz, op, etav, v, scr, lo, hi,
                                        nbmax, Val(N), Val(STORE), Val(false))
end

function _semquad_kn_blk_mapped_body!(out::Union{Vector{Float64},Nothing},
                                      elx::Union{Matrix{Float64},Nothing},
                                      elz::Union{Matrix{Float64},Nothing},
                                      op::SemQuadKOpMapped,
                                      etab::AbstractVector{Float64},
                                      v::AbstractVector{Float64},
                                      scr::SemQuadKScratch, lo::Int, hi::Int,
                                      nbmax::Int, ::Val{N}, ::Val{STORE},
                                      ::Val{LANE}) where {N,STORE,LANE}
    BLK = SEMQUAD_KBLK
    nq = N + 1
    N_ = op.N
    n2 = N * N
    nq2 = nq * nq
    phi = op.phi; dphi = op.dphi; phiT = op.phiT; dphiT = op.dphiT
    gl_w = op.gl_w
    ids = op.ids
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    c2 = op.deviatoric ? 2.0 / 3.0 : 0.0
    xv = scr.bxv; zv = scr.bzv
    uxi_x = scr.buxx; ueta_x = scr.buxy
    uxi_z = scr.buzx; ueta_z = scr.buzy
    t1 = scr.bt1
    Fx1 = scr.bF
    Fx2 = scr.bF2::Array{Float64,3}
    Fz1 = scr.bF3::Array{Float64,3}
    Fz2 = scr.bF4::Array{Float64,3}
    ox = scr.box; oz = scr.boz
    one = scr.bone
    kofs = scr.bkofs
    detJ = op.detJ
    j11v = op.j11; j12v = op.j12; j21v = op.j21; j22v = op.j22
    STORE || fill!(out, 0.0)
    @inbounds for lb in lo:BLK:hi
        nb = min(BLK, hi - lb + 1)
        lplast = lb + nb - 1
        tbase = LANE ? (((lb - 1) ÷ BLK) * (nq2 * BLK)) : 0
        for b in 1:BLK
            li = b <= nb ? (lb + b - 1) : lplast
            kofs[b] = (li - 1) * nq2
            for idx in 1:n2
                g = ids[idx, li]
                xv[b, idx] = v[g]
                zv[b, idx] = v[N_ + g]
            end
        end
        # raw reference derivatives: no 2/h scaling (the box kernel's ihx/ihy
        # are replaced by the chain rule below), so the scale is 1.0 on both
        # contraction slots.
        _semquad_contract_blk!(uxi_x, xv, dphi, phi, one, one, t1, Val(N))
        _semquad_contract_blk!(ueta_x, xv, phi, dphi, one, one, t1, Val(N))
        _semquad_contract_blk!(uxi_z, zv, dphi, phi, one, one, t1, Val(N))
        _semquad_contract_blk!(ueta_z, zv, phi, dphi, one, one, t1, Val(N))
        for idx in 1:n2
            @simd ivdep for b in 1:BLK
                ox[b, idx] = 0.0
                oz[b, idx] = 0.0
            end
        end
        for qy in 1:nq, qx in 1:nq
            gw = gl_w[qx] * gl_w[qy]
            kq = (qy - 1) * nq + qx
            kb = LANE ? (tbase + (kq - 1) * BLK) : 0
            @simd ivdep for b in 1:BLK
                etaq = LANE ? etab[kb + b] : etab[kofs[b] + kq]
                k = kofs[b] + kq
                W = detJ[k] * gw * etaq
                j11 = j11v[k]; j12 = j12v[k]
                j21 = j21v[k]; j22 = j22v[k]
                ax = uxi_x[b, qx, qy]; bx = ueta_x[b, qx, qy]
                az = uxi_z[b, qx, qy]; bz = ueta_z[b, qx, qy]
                uxx = j11 * ax + j12 * bx
                uxy = j21 * ax + j22 * bx
                uzx = j11 * az + j12 * bz
                uzy = j21 * az + j22 * bz
                A = W * (a2 * uxx - c2 * uzy)
                Bq = W * (uxy + uzx)
                Az = W * (a2 * uzy - c2 * uxx)
                Fx1[b, qx, qy] = A * j11 + Bq * j21
                Fx2[b, qx, qy] = A * j12 + Bq * j22
                Fz1[b, qx, qy] = Bq * j11 + Az * j21
                Fz2[b, qx, qy] = Bq * j12 + Az * j22
            end
        end
        _semquad_accum_blk!(ox, dphi, Fx1, phiT, t1, Val(N))
        _semquad_accum_blk!(ox, phi, Fx2, dphiT, t1, Val(N))
        _semquad_accum_blk!(oz, dphi, Fz1, phiT, t1, Val(N))
        _semquad_accum_blk!(oz, phi, Fz2, dphiT, t1, Val(N))
        if STORE
            for b in 1:BLK
                b <= nb || continue          # padded lanes write nothing
                li = lb + b - 1
                for idx in 1:n2
                    elx[idx, li] = ox[b, idx]
                    elz[idx, li] = oz[b, idx]
                end
            end
        else
            for b in 1:BLK
                b <= nb || continue          # padded lanes write nothing
                li = lb + b - 1
                for idx in 1:n2
                    g = ids[idx, li]
                    out[g] += ox[b, idx]
                    out[N_ + g] += oz[b, idx]
                end
            end
        end
    end
    if STORE
        return nothing
    end
    return out
end

"""
    SEMQUAD_MG_FORKS, SEMQUAD_MG_FORK_TASKS

Diagnostics for the threaded matrix-free path: `SEMQUAD_MG_FORKS` counts
fork/join regions (each one is a `@sync` + `@spawn` set, i.e. one task-creation
event) and `SEMQUAD_MG_FORK_TASKS` the worker tasks those regions created.  Both
stay 0 in a serial run.  A `Ref{Int}` increment is a few ns and does not
allocate, so the counters do not move the timings they are used to explain
(G223 pins the fork count of the consolidated path against the per-call one).
"""
const SEMQUAD_MG_FORKS = Ref(0)
const SEMQUAD_MG_FORK_TASKS = Ref(0)

"""
    _semquad_td_run!(f, nt, nchunks)

Run `f(c)` for `c in 1:nchunks` on **at most `nt`** worker tasks that pull the
chunk index from a shared counter, so the work is dynamically balanced over the
`nt` workers but never spread over the rest of the Julia pool.  `nt <= 1` runs
the chunks in order on the calling task (same values, no scheduling).
"""
function _semquad_td_run!(f::F, nt::Int, nchunks::Int) where {F}
    if nt <= 1 || nchunks <= 1
        for c in 1:nchunks
            f(c)
        end
        return nothing
    end
    SEMQUAD_MG_FORKS[] += 1
    SEMQUAD_MG_FORK_TASKS[] += nt
    counter = Threads.Atomic{Int}(0)
    Threads.@sync for _ in 1:nt
        Threads.@spawn begin
            while true
                c = Threads.atomic_add!(counter, 1) + 1
                c > nchunks && break
                f(c)
            end
        end
    end
    return nothing
end

