# Boundary conditions: reduced (free) degree-of-freedom maps.
#
# Free slip: u.n = 0 on each wall -> u_x = 0 on x = 0,L and u_z = 0 on z = 0,1.
# The tangential component is free and the tangential traction is zero, which is
# the natural boundary condition of the weak form (no extra term is assembled).
# Rigid: both components zero on every wall.
#
# The pressure has a constant null space in a closed box; exactly one pressure
# degree of freedom is pinned to zero.

struct DofMap
    red_ux::Vector{Int}   # length Nscalar, 0 = fixed
    red_uz::Vector{Int}   # length Nscalar, 0 = fixed
    red_p::Vector{Int}    # length Npres,   0 = pinned
    nux::Int
    nuz::Int
    npres::Int
    ntotal::Int
    pin::Int
end

function make_dofmap(m::Mesh, bc::Symbol)
    bc in (:free, :rigid) || throw(ArgumentError("bc must be :free or :rigid"))
    red_ux = zeros(Int, m.Nscalar)
    red_uz = zeros(Int, m.Nscalar)
    red_p = zeros(Int, m.Npres)
    nux = 0
    nuz = 0
    @inbounds for iz in 1:m.nvz
        for ix in 1:m.nvx
            g = (iz - 1) * m.nvx + ix
            on_x = (ix == 1 || ix == m.nvx)
            on_z = (iz == 1 || iz == m.nvz)
            if bc === :rigid
                fixed_x = on_x || on_z
                fixed_z = on_x || on_z
            else
                fixed_x = on_x
                fixed_z = on_z
            end
            if !fixed_x
                nux += 1
                red_ux[g] = nux
            end
            if !fixed_z
                nuz += 1
                red_uz[g] = nuz
            end
        end
    end
    pin = pres_gid(m, 0, 0, 1, 1)
    npres = 0
    @inbounds for g in 1:m.Npres
        g == pin && continue
        npres += 1
        red_p[g] = npres
    end
    return DofMap(red_ux, red_uz, red_p, nux, nuz, npres,
                  nux + nuz + npres, pin)
end

"""Reduced map for theta: homogeneous Dirichlet at z = 0,1, natural at x = 0,L."""
struct ThetaMap
    red::Vector{Int}
    n::Int
end

function make_theta_map(m::Mesh)
    red = zeros(Int, m.Nscalar)
    n = 0
    @inbounds for iz in 2:m.nvz-1
        for ix in 1:m.nvx
            g = (iz - 1) * m.nvx + ix
            n += 1
            red[g] = n
        end
    end
    return ThetaMap(red, n)
end

"""Gather the free entries of a full vector into reduced ordering."""
function gather_reduced(red::Vector{Int}, full::Vector{Float64}, n::Int)
    out = zeros(Float64, n)
    @inbounds for g in eachindex(red)
        r = red[g]
        r > 0 && (out[r] = full[g])
    end
    return out
end

"""Scatter reduced values into a full vector of length nfull (zeros elsewhere)."""
function scatter_reduced(red::Vector{Int}, redvec::Vector{Float64}, nfull::Int)
    full = zeros(Float64, nfull)
    @inbounds for g in eachindex(red)
        r = red[g]
        r > 0 && (full[g] = redvec[r])
    end
    return full
end


