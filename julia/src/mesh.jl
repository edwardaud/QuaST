# Structured quadrilateral mesh and global degree-of-freedom numbering.
#
# Scalar nodes (temperature / velocity component) are Q_p on a conforming
# tensor-product GLL grid: global 1-D index ix = ex*p + i, so adjacent elements
# share their boundary node exactly.  Pressure is Q_q with q = p-1 by default
# (Taylor-Hood; G1 measured 5.9/4.0 orders for Q4-Q3 versus 3.5/2.8 for Q4-Q2).
#
# The z mesh may be graded.  ``zedges`` holds the physical element boundaries;
# within each element the map is linear, so the Jacobian is the element height
# ``zedges[ez+2]-zedges[ez+1]`` and all integrals still use the same reference
# Gauss points.  ``zgrading=:cosine`` clusters elements at both walls, which is
# needed for the low-viscosity bottom layer of Blankenbach 2a.

struct Mesh
    Lx::Float64
    nxe::Int
    nze::Int
    hx::Float64
    hz::Float64          # average height, for reporting only
    p::Int
    q::Int
    nvx::Int             # nxe*p + 1 scalar nodes in x
    nvz::Int             # nze*p + 1 scalar nodes in z
    npx::Int             # nxe*q + 1 pressure nodes in x
    npz::Int
    Nscalar::Int         # nvx*nvz  (theta and one velocity component)
    Nvel::Int            # 2*Nscalar
    Npres::Int           # npx*npz
    zedges::Vector{Float64}   # length nze+1, 0 = zedges[1] < ... < zedges[end] = 1
end

function _z_edges(nze::Int, zgrading::Symbol, zbeta::Float64)
    s = collect(range(0.0, 1.0, length=nze+1))
    if zgrading === :uniform
        z = copy(s)
    elseif zgrading === :cosine
        z = 0.5 .* (1.0 .- cos.(pi .* s))
    elseif zgrading === :tanh
        zbeta > 0.0 || throw(ArgumentError("zbeta must be > 0 for :tanh grading"))
        z = 0.5 .* (1.0 .+ tanh.(zbeta .* (2.0 .* s .- 1.0)) ./ tanh(zbeta))
    else
        throw(ArgumentError("zgrading must be :uniform, :cosine or :tanh"))
    end
    z[1] = 0.0
    z[end] = 1.0
    return z
end

function make_mesh(Lx::Real, nxe::Int, nze::Int, p::Int; q::Int=-1,
                   zgrading::Symbol=:uniform, zbeta::Float64=2.0,
                   zedges::Union{Nothing,Vector{Float64}}=nothing)
    nxe >= 1 && nze >= 1 || throw(ArgumentError("nxe,nze must be >= 1"))
    p >= 2 || throw(ArgumentError("p must be >= 2, got $p"))
    qq = q < 0 ? max(1, p - 1) : q
    qq >= 1 || throw(ArgumentError("pressure degree q must be >= 1"))
    Lxf = Float64(Lx)
    hx = Lxf / nxe
    hz = 1.0 / nze
    nvx = nxe * p + 1
    nvz = nze * p + 1
    npx = nxe * qq + 1
    npz = nze * qq + 1
    Ns = nvx * nvz
    ze = zedges === nothing ? _z_edges(nze, zgrading, zbeta) : copy(zedges)
    length(ze) == nze + 1 || throw(ArgumentError("zedges must have nze+1 entries"))
    ze[1] == 0.0 && ze[end] == 1.0 || throw(ArgumentError("zedges must span [0,1]"))
    Mesh(Lxf, nxe, nze, hx, hz, p, qq, nvx, nvz, npx, npz,
         Ns, 2 * Ns, npx * npz, ze)
end

Nscalar(m::Mesh) = m.Nscalar

@inline z_left(m::Mesh, ez::Int) = m.zedges[ez+1]
@inline z_right(m::Mesh, ez::Int) = m.zedges[ez+2]
@inline z_size(m::Mesh, ez::Int) = m.zedges[ez+2] - m.zedges[ez+1]
@inline z_min_size(m::Mesh) = minimum(diff(m.zedges))

@inline function vel_gid(m::Mesh, ex::Int, ez::Int, i::Int, j::Int)
    # ex, ez are 0-based element indices; i, j are 1-based local GLL indices.
    ix = ex * m.p + i
    iz = ez * m.p + j
    return (iz - 1) * m.nvx + ix
end

@inline function pres_gid(m::Mesh, ex::Int, ez::Int, a::Int, b::Int)
    ix = ex * m.q + a
    iz = ez * m.q + b
    return (iz - 1) * m.npx + ix
end

"""Scalar (velocity-component / theta) global ids of one element, tensor order."""
function element_vel_gids(m::Mesh, ex::Int, ez::Int)
    n = (m.p + 1)^2
    g = Vector{Int}(undef, n)
    k = 0
    @inbounds for j in 1:m.p+1
        iz = ez * m.p + j
        for i in 1:m.p+1
            k += 1
            g[k] = (iz - 1) * m.nvx + (ex * m.p + i)
        end
    end
    return g
end

"""Pressure global ids of one element, tensor order."""
function element_pres_gids(m::Mesh, ex::Int, ez::Int)
    n = (m.q + 1)^2
    g = Vector{Int}(undef, n)
    k = 0
    @inbounds for b in 1:m.q+1
        iz = ez * m.q + b
        for a in 1:m.q+1
            k += 1
            g[k] = (iz - 1) * m.npx + (ex * m.q + a)
        end
    end
    return g
end

"""Physical coordinates of the global scalar nodes."""
function node_coords(m::Mesh, rule::GLLRule)
    xs = zeros(Float64, m.nvx)
    zs = zeros(Float64, m.nvz)
    @inbounds for ix in 1:m.nvx
        if ix == m.nvx
            xs[ix] = m.Lx
        else
            ex = (ix - 1) ÷ m.p
            i = mod(ix - 1, m.p) + 1
            xs[ix] = ex * m.hx + 0.5 * (rule.nodes[i] + 1.0) * m.hx
        end
    end
    @inbounds for iz in 1:m.nvz
        if iz == m.nvz
            zs[iz] = 1.0
        else
            ez = (iz - 1) ÷ m.p
            j = mod(iz - 1, m.p) + 1
            hze = z_size(m, ez)
            zs[iz] = z_left(m, ez) + 0.5 * (rule.nodes[j] + 1.0) * hze
        end
    end
    return xs, zs
end
