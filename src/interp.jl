# Evaluation of a nodal spectral-element field on another mesh.
#
# Used to restart a run on a finer mesh (e.g. King n=16 -> n=32 -> n=64)
# without re-integrating the whole transient.  The transfer is an evaluation of
# the source Q_p expansion at the destination nodes, not a re-projection, so it
# is exact for the source discretisation.

"""
    interpolate_theta(ms, fds, thetas, md, ruled)

Evaluate the nodal field `thetas` (defined on mesh `ms` with data `fds`) at the
nodes of mesh `md`.  Both meshes must cover the same rectangular domain.
"""
function interpolate_theta(ms::Mesh, fds::FEData, thetas::Vector{Float64},
                           md::Mesh, ruled::GLLRule)
    xd, zd = node_coords(md, ruled)
    out = zeros(Float64, md.Nscalar)
    p = ms.p
    @inbounds for iz in eachindex(zd), ix in eachindex(xd)
        x = xd[ix]; z = zd[iz]
        ex = clamp(floor(Int, x / ms.hx), 0, ms.nxe - 1)
        ez = clamp(searchsortedlast(ms.zedges, z) - 1, 0, ms.nze - 1)
        ze0 = ms.zedges[ez+1]
        hz = ms.zedges[ez+2] - ze0
        xi = clamp(2.0 * (x - ex * ms.hx) / ms.hx - 1.0, -1.0, 1.0)
        eta = clamp(2.0 * (z - ze0) / hz - 1.0, -1.0, 1.0)
        Lx, _ = lagrange_eval(fds.gll_v, xi)
        Ly, _ = lagrange_eval(fds.gll_v, eta)
        g = element_vel_gids(ms, ex, ez)
        val = 0.0
        for j in 1:p+1, i in 1:p+1
            val += thetas[g[(j-1)*(p+1)+i]] * Lx[i] * Ly[j]
        end
        out[(iz - 1) * md.nvx + ix] = val
    end
    return out
end
