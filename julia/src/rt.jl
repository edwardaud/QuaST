# Lowest-order Raviart-Thomas (RT0) on a reference quad [-1,1]^2.
#
# Edge numbering / outward normals:
#   1 left   x=-1, n=(-1,0)
#   2 right  x=+1, n=(+1,0)
#   3 bottom z=-1, n=(0,-1)
#   4 top    z=+1, n=(0,+1)
#
# The four basis functions have unit outward normal flux on their own edge and
# zero on the other three edges:
#   u1 = (-(1+x)/2, 0)
#   u2 = ( (1+x)/2, 0)
#   u3 = (0, -(1-z)/2)
#   u4 = (0,  (1+z)/2)
# Their divergence is piecewise constant, div = (-1/2, +1/2, +1/2, +1/2).
#
# For a global H(div) DOF the normal is fixed once per edge.  With the global
# normals (+x on vertical edges, +z on horizontal edges), the local outward
# normals have signs (-1, +1, -1, +1), so the local-to-global map multiplies
# the left/bottom basis by -1.

"""
RT0 basis vectors at reference point (xi, eta).
Returns `(ux, uz)` with four entries each.
"""
@inline function rt0_basis(xi::Float64, eta::Float64)
    ux = ((xi - 1.0) / 2.0, (1.0 + xi) / 2.0, 0.0, 0.0)
    uz = (0.0, 0.0, -(1.0 - eta) / 2.0, (1.0 + eta) / 2.0)
    return ux, uz
end

"""RT0 divergence per basis function (constant on the reference element)."""
@inline rt0_divergence() = (0.5, 0.5, 0.5, 0.5)

"""Signs mapping local outward normals to global normals (+x vertical, +z horizontal)."""
@inline rt0_edge_signs() = (-1.0, 1.0, -1.0, 1.0)

"""Outward unit normal of reference edge `e`."""
@inline function rt0_edge_normal(e::Int)
    e == 1 && return (-1.0, 0.0)
    e == 2 && return (1.0, 0.0)
    e == 3 && return (0.0, -1.0)
    e == 4 && return (0.0, 1.0)
    throw(ArgumentError("edge must be 1..4"))
end

"""Outward normal flux of RT0 basis `j` on edge `e` at a point on that edge."""
@inline function rt0_edge_flux(xi::Float64, eta::Float64, e::Int, j::Int)
    ux, uz = rt0_basis(xi, eta)
    nx, nz = rt0_edge_normal(e)
    return ux[j] * nx + uz[j] * nz
end

# ---------------------------------------------------------------------------
# Global RT0 edge map (one normal-flux DOF per edge).
# ---------------------------------------------------------------------------

struct RT0Map
    nedge::Int
    red::Vector{Int}       # 1..nedge -> free index or 0 (fixed)
    idv::Matrix{Int}       # (nxe+1, nze), vertical edges, global normal +x
    idh::Matrix{Int}       # (nxe, nze+1), horizontal edges, global normal +z
    nfree::Int
end

function make_rt0_map(m::Mesh, bc::Symbol=:free)
    nxe, nze = m.nxe, m.nze
    idv = zeros(Int, nxe + 1, nze)
    idh = zeros(Int, nxe, nze + 1)
    n = 0
    for ez in 0:nze-1, ex in 0:nxe
        n += 1
        idv[ex+1, ez+1] = n
    end
    for ez in 0:nze, ex in 0:nxe-1
        n += 1
        idh[ex+1, ez+1] = n
    end
    red = collect(1:n)
    for ez in 0:nze-1
        red[idv[1, ez+1]] = 0
        red[idv[nxe+1, ez+1]] = 0
    end
    for ex in 0:nxe-1
        red[idh[ex+1, 1]] = 0
        red[idh[ex+1, nze+1]] = 0
    end
    # compact the free indices
    nf = 0
    for i in 1:n
        if red[i] > 0
            nf += 1
            red[i] = nf
        end
    end
    return RT0Map(n, red, idv, idh, nf)
end

"""
RT0-DG0 Stokes assembly.  Velocity: one normal-flux DOF per edge (free-slip
boundary edges fixed).  Pressure: one constant per element, first element
pinned.  The viscous kernel for RT0 on axis-aligned rectangles has
eps_xx = (U_right-U_left)/hx and eps_zz = (U_top-U_bottom)/hz, eps_xz=0.
"""
function assemble_rt0_stokes(m::Mesh, fd::FEData, pr::Params,
                             theta::Vector{Float64};
                             bc::Symbol=:free, verbose::Bool=false,
                             return_system::Bool=false)
    rmap = make_rt0_map(m, bc)
    nv = rmap.nfree
    ne = m.nxe * m.nze
    pred = collect(0:(ne-1))
    np = ne - 1
    ntot = nv + np
    II = Int[]; JJ = Int[]; VV = Float64[]
    f = zeros(Float64, ntot)
    eta_min = Inf; eta_max = -Inf; nclip = 0
    hx = m.hx
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez); z1 = z0 + hze
        for ex in 0:m.nxe-1
            x0 = ex * hx; x1 = x0 + hx
            gids = (rmap.idv[ex+1, ez+1], rmap.idv[ex+2, ez+1],
                    rmap.idh[ex+1, ez+1], rmap.idh[ex+1, ez+2])
            cols = ntuple(i -> rmap.red[gids[i]], 4)
            signs = rt0_edge_signs()
            area = hx * hze
            epsxx = (1.0 / hx, 1.0 / hx, 0.0, 0.0)
            epszz = (0.0, 0.0, 1.0 / hze, 1.0 / hze)
            divv = (1.0 / hx, 1.0 / hx, 1.0 / hze, 1.0 / hze)
            # quadrature for eta(T) and body force
            for qy in 1:fd.nq
                z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                for qx in 1:fd.nq
                    x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
                    Tloc = Tbar(z, pr.H)
                    thloc = 0.0
                    kv = 0
                    for j in 1:m.p+1
                        for i in 1:m.p+1
                            kv += 1
                            thv = theta[(ez*m.p+j-1)*m.nvx + (ex*m.p+i)] *
                                  fd.phi_v[i, qx] * fd.phi_v[j, qy]
                            Tloc += thv
                            thloc += thv
                        end
                    end
                    eta, clipped = eta_of_T(Tloc, pr)
                    eta_min = min(eta_min, eta); eta_max = max(eta_max, eta)
                    clipped && (nclip += 1)
                    w = fd.gl_w[qx] * fd.gl_w[qy] * area / 4.0
                    for i in 1:4
                        cols[i] == 0 && continue
                        for j in 1:4
                            cols[j] == 0 && continue
                            val = 2.0 * eta * (epsxx[i] * epsxx[j] + epszz[i] * epszz[j]) * w
                            push!(II, cols[i]); push!(JJ, cols[j])
                            push!(VV, signs[i] * signs[j] * val)
                        end
                        vz = i == 3 ? (z - z1) / hze : (i == 4 ? (z - z0) / hze : 0.0)
                        f[cols[i]] += signs[i] * pr.Ra * thloc * vz * w
                    end
                end
            end
            pe = (ez * m.nxe + ex) + 1
            if pred[pe] > 0
                row = nv + pred[pe]
                for i in 1:4
                    cols[i] == 0 && continue
                    push!(II, row); push!(JJ, cols[i])
                    push!(VV, -area * divv[i] * signs[i])
                    push!(II, cols[i]); push!(JJ, row)
                    push!(VV, -area * divv[i] * signs[i])
                end
            end
        end
    end
    A = sparse(II, JJ, VV, ntot, ntot)
    A = A + sparse(1:ntot, 1:ntot, fill(1.0e-12, ntot), ntot, ntot)
    rhs = vcat(f[1:nv], zeros(Float64, np))
    if return_system
        return A, rhs, rmap
    end
    sol = A \ rhs
    res = norm(A * sol - rhs) / max(norm(rhs), 1.0e-30)
    U = sol[1:nv]

    # quadrature velocity
    uxq = Array{Float64,4}(undef, m.nxe, m.nze, fd.nq, fd.nq)
    uzq = Array{Float64,4}(undef, m.nxe, m.nze, fd.nq, fd.nq)
    for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez); z1 = z0 + hze
        for ex in 0:m.nxe-1
            x0 = ex * hx; x1 = x0 + hx
            gids = (rmap.idv[ex+1, ez+1], rmap.idv[ex+2, ez+1],
                    rmap.idh[ex+1, ez+1], rmap.idh[ex+1, ez+2])
            cols = ntuple(i -> rmap.red[gids[i]], 4)
            signs = rt0_edge_signs()
            Ul = ntuple(i -> cols[i] == 0 ? 0.0 : signs[i] * U[cols[i]], 4)
            for qy in 1:fd.nq
                z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                for qx in 1:fd.nq
                    x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
                    uxq[ex+1, ez+1, qx, qy] = Ul[1] * (x - x1) / hx + Ul[2] * (x - x0) / hx
                    uzq[ex+1, ez+1, qx, qy] = Ul[3] * (z - z1) / hze + Ul[4] * (z - z0) / hze
                end
            end
        end
    end
    velq = QuadVelocity(uxq, uzq)
    ux = zeros(Float64, m.Nscalar); uz = zeros(Float64, m.Nscalar)
    for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez); z1 = z0 + hze
        for ex in 0:m.nxe-1
            x0 = ex * hx; x1 = x0 + hx
            gids = (rmap.idv[ex+1, ez+1], rmap.idv[ex+2, ez+1],
                    rmap.idh[ex+1, ez+1], rmap.idh[ex+1, ez+2])
            cols = ntuple(i -> rmap.red[gids[i]], 4)
            signs = rt0_edge_signs()
            Ul = ntuple(i -> cols[i] == 0 ? 0.0 : signs[i] * U[cols[i]], 4)
            for j in 1:m.p+1
                z = z0 + 0.5 * (fd.gll_v.nodes[j] + 1.0) * hze
                for i in 1:m.p+1
                    x = x0 + 0.5 * (fd.gll_v.nodes[i] + 1.0) * hx
                    g = (ez*m.p + j - 1) * m.nvx + (ex*m.p + i)
                    ux[g] = Ul[1] * (x - x1) / hx + Ul[2] * (x - x0) / hx
                    uz[g] = Ul[3] * (z - z1) / hze + Ul[4] * (z - z0) / hze
                end
            end
        end
    end
    verbose && @printf("  rt0-Stokes: nv=%d np=%d res=%.2e eta=[%.3e,%.3e]\n",
                       nv, np, res, eta_min, eta_max)
    return StokesSolution(ux, uz, zeros(Float64, m.Npres), res,
                          eta_min, eta_max, nclip, velq)
end

function solve_stokes_rt0(m::Mesh, fd::FEData, pr::Params,
                          theta::Vector{Float64}, dm::DofMap;
                          res_bar::Float64=1.0e-8, linear::Symbol=:direct)
    pr.bc === :free ||
        throw(ArgumentError("RT0 velocity space currently supports free-slip only"))
    return assemble_rt0_stokes(m, fd, pr, theta; bc=pr.bc)
end


