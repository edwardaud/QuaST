# ASCII initial-temperature input (ASPECT Blankenbach 2a format).
#
# File format:
#   # POINTS: nx ny
#   # x y T
#   x y T        (one line per point, cell-centred 16x16 grid)
#
# The data are interpolated bilinearly onto the GLL nodes; at x=0,1 the
# interpolation is clamped (the side walls are insulating and the data do not
# include them), and the top/bottom theta values are reset to zero because the
# solver imposes homogeneous Dirichlet data there.

struct TemperatureGrid
    x::Vector{Float64}
    y::Vector{Float64}
    T::Matrix{Float64}   # T[ix, iy]
end

function read_temperature_ascii(path::String)
    xs = Float64[]
    ys = Float64[]
    Ts = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        parts = split(s)
        length(parts) >= 3 || continue
        push!(xs, parse(Float64, parts[1]))
        push!(ys, parse(Float64, parts[2]))
        push!(Ts, parse(Float64, parts[3]))
    end
    isempty(xs) && error("no data in $path")
    ux = sort(unique(xs))
    uy = sort(unique(ys))
    T = fill(NaN, length(ux), length(uy))
    for k in eachindex(xs)
        i = searchsortedfirst(ux, xs[k])
        j = searchsortedfirst(uy, ys[k])
        T[i, j] = Ts[k]
    end
    any(isnan, T) && error("non-rectangular grid in $path")
    return TemperatureGrid(ux, uy, T)
end

"""Bilinear interpolation with clamping at the data boundary."""
function interp_temperature(g::TemperatureGrid, x::Float64, y::Float64)
    nx = length(g.x)
    ny = length(g.y)
    i = clamp(searchsortedlast(g.x, x), 1, nx - 1)
    j = clamp(searchsortedlast(g.y, y), 1, ny - 1)
    tx = (x - g.x[i]) / (g.x[i+1] - g.x[i])
    ty = (y - g.y[j]) / (g.y[j+1] - g.y[j])
    tx = clamp(tx, 0.0, 1.0)
    ty = clamp(ty, 0.0, 1.0)
    return (1 - tx) * (1 - ty) * g.T[i, j] +
           tx * (1 - ty) * g.T[i+1, j] +
           (1 - tx) * ty * g.T[i, j+1] +
           tx * ty * g.T[i+1, j+1]
end

"""Build theta = T_ascii - Tbar(z) on the GLL nodes (zero at z=0,1).

``blend`` smooths theta to zero over a wall layer (smoothstep), which removes
the artificial sub-grid top/bottom spike of the ASCII snapshot.  This changes
only the initial guess, never the steady state.
"""
function theta_from_ascii(m::Mesh, rule::GLLRule, path::String;
                          H::Float64=0.0, blend::Float64=0.05)
    g = read_temperature_ascii(path)
    xs, zs = node_coords(m, rule)
    theta = zeros(Float64, m.Nscalar)
    for iz in 1:m.nvz
        z = zs[iz]
        w = 1.0
        if blend > 0.0
            t = clamp(min(z, 1.0 - z) / blend, 0.0, 1.0)
            w = t * t * (3.0 - 2.0 * t)
        end
        for ix in 1:m.nvx
            x = xs[ix]
            T = interp_temperature(g, x, z)
            theta[(iz - 1) * m.nvx + ix] = w * (T - Tbar(z, H))
        end
    end
    # homogeneous Dirichlet at top and bottom
    for ix in 1:m.nvx
        theta[ix] = 0.0
        theta[(m.nvz - 1) * m.nvx + ix] = 0.0
    end
    return theta
end
