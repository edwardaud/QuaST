# Visualization output and depth averages for the quadtree SEM path (route C).
#
# Route C keeps its state in (theta, ux, uz, pv) on the Q_p / Q_{p-1} spaces;
# until now the only way to look at it was the HDF5 checkpoint.  This module
# writes a legacy ASCII VTK unstructured grid that ParaView/VisIt can open, and
# a horizontally averaged depth profile.
#
# The VTK convention follows deal.II's VTU output for high-order elements: each
# Q_p leaf is subdivided into p x p Q1 quads using the leaf's GLL nodes, so the
# viewer sees the spectral field without needing biquadratic cells (legacy VTK
# only has those for p = 2).  Points are duplicated per leaf, which is simple and
# robust across hanging nodes; the files stay small for the meshes used here
# (L6: 4096 leaves, 36k points).

"""
    write_semquad_vtk(path, ss, qt, theta, ux, uz, pv; eta_fun=nothing,
                      extra_scalars=Dict{String,Vector{Float64}}(), title=...)

Write the quadtree SEM state to a legacy ASCII VTK `UNSTRUCTURED_GRID`.

- each leaf becomes `p*p` Q1 quads from its GLL nodes (deal.II convention);
- point data: `theta`, `velocity` (3 components, `w = 0`), `pressure` (the
  Q_{p-1} pressure evaluated at the Q_p nodes), and `viscosity` when
  `eta_fun(x, y)` is given;
- cell data: `level` and `h`;
- `extra_scalars` adds named nodal fields (same length as `theta`).

Returns the number of points and cells written.
"""
function write_semquad_vtk(path::AbstractString, ss::SemStokes, qt::Quadtree,
                           theta::Vector{Float64}, ux::Vector{Float64},
                           uz::Vector{Float64}, pv::Vector{Float64};
                           eta_fun::Union{Nothing,Function}=nothing,
                           extra_scalars::AbstractDict{String,Vector{Float64}}=Dict{String,Vector{Float64}}(),
                           title::AbstractString="mantle-spectral quadtree SEM")
    sq = ss.su
    sp = ss.sp
    p = sq.p
    n = p + 1
    q = sp.p
    nq = q + 1
    rq = gll_rule(q)
    leaves = sort(collect(qt.leaves))
    nleaf = length(leaves)
    npts = nleaf * n * n
    ncells = nleaf * p * p

    X = Vector{Float64}(undef, npts)
    Y = Vector{Float64}(undef, npts)
    TH = Vector{Float64}(undef, npts)
    VX = Vector{Float64}(undef, npts)
    VZ = Vector{Float64}(undef, npts)
    PR = Vector{Float64}(undef, npts)
    ET = eta_fun === nothing ? Float64[] : Vector{Float64}(undef, npts)
    extra_vals = Dict{String,Vector{Float64}}(k => Vector{Float64}(undef, npts)
                                              for k in keys(extra_scalars))
    cells = Vector{NTuple{4,Int}}(undef, ncells)
    levels = Vector{Int}(undef, ncells)
    hs = Vector{Float64}(undef, ncells)
    for (c, lf) in enumerate(leaves)
        hx = leaf_hx(sq, lf[1])
        hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx
        y0 = lf[3] * hy
        ids = sq.elem[lf]
        pids = sp.elem[lf]
        base = (c - 1) * n * n
        for j in 1:n, i in 1:n
            k = base + (j - 1) * n + i
            x = x0 + 0.5 * (sq.xi[i] + 1.0) * hx
            y = y0 + 0.5 * (sq.xi[j] + 1.0) * hy
            X[k] = x; Y[k] = y
            g = ids[(j - 1) * n + i]
            TH[k] = theta[g]
            VX[k] = ux[g]
            VZ[k] = uz[g]
            # Q_{p-1} pressure interpolated to the Q_p node
            La, _ = lagrange_eval(rq, sq.xi[i])
            Lb, _ = lagrange_eval(rq, sq.xi[j])
            pval = 0.0
            for b in 1:nq, a in 1:nq
                pval += pv[pids[(b - 1) * nq + a]] * La[a] * Lb[b]
            end
            PR[k] = pval
            eta_fun === nothing || (ET[k] = eta_fun(x, y))
            for (name, vals) in extra_scalars
                extra_vals[name][k] = vals[g]
            end
        end
        for j in 1:p, i in 1:p
            cid = (c - 1) * p * p + (j - 1) * p + i
            a = base + (j - 1) * n + i
            cells[cid] = (a, a + 1, a + n + 1, a + n)
            levels[cid] = lf[1]
            hs[cid] = hy
        end
    end

    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, title)
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")
        println(io, "POINTS ", npts, " double")
        for k in 1:npts
            println(io, X[k], " ", Y[k], " 0.0")
        end
        println(io, "CELLS ", ncells, " ", 5 * ncells)
        for cell in cells
            println(io, "4 ", cell[1], " ", cell[2], " ", cell[3], " ", cell[4])
        end
        println(io, "CELL_TYPES ", ncells)
        for _ in 1:ncells
            println(io, 9)          # VTK_QUAD
        end
        println(io, "POINT_DATA ", npts)
        println(io, "SCALARS theta double 1")
        println(io, "LOOKUP_TABLE default")
        for k in 1:npts
            println(io, TH[k])
        end
        println(io, "VECTORS velocity double")
        for k in 1:npts
            println(io, VX[k], " 0.0 ", VZ[k])
        end
        println(io, "SCALARS pressure double 1")
        println(io, "LOOKUP_TABLE default")
        for k in 1:npts
            println(io, PR[k])
        end
        if eta_fun !== nothing
            println(io, "SCALARS viscosity double 1")
            println(io, "LOOKUP_TABLE default")
            for k in 1:npts
                println(io, ET[k])
            end
        end
        for name in sort(collect(keys(extra_scalars)))
            println(io, "SCALARS ", name, " double 1")
            println(io, "LOOKUP_TABLE default")
            for k in 1:npts
                println(io, extra_vals[name][k])
            end
        end
        println(io, "CELL_DATA ", ncells)
        println(io, "SCALARS level int 1")
        println(io, "LOOKUP_TABLE default")
        for v in levels
            println(io, v)
        end
        println(io, "SCALARS h double 1")
        println(io, "LOOKUP_TABLE default")
        for v in hs
            println(io, v)
        end
    end
    return (points=npts, cells=ncells)
end

"""
    semquad_depth_average(ss, qt, theta, ux, uz, pv; nbins=100)

Horizontally averaged profiles on `nbins` equal bins of `y in [0,1]`, integrated
with the leaf Gauss rules (not a node average, so graded/refined meshes are
weighted by area).  Returns a named tuple of vectors `(y, theta, ux, uz, p)`.

The temperature returned is the perturbation `theta`; add the base state
(`semquad_base_T(pr, y)`) for the physical temperature.
"""
function semquad_depth_average(ss::SemStokes, qt::Quadtree,
                               theta::Vector{Float64}, ux::Vector{Float64},
                               uz::Vector{Float64}, pv::Vector{Float64};
                               nbins::Int=100)
    sq = ss.su
    sp = ss.sp
    nbins >= 1 || throw(ArgumentError("nbins must be >= 1"))
    p = sq.p
    n = p + 1
    q = sp.p
    nq = q + 1
    fd = make_fedata(p, q)
    accT = zeros(Float64, nbins); accX = zeros(Float64, nbins)
    accZ = zeros(Float64, nbins); accP = zeros(Float64, nbins)
    accW = zeros(Float64, nbins)
    for lf in qt.leaves
        hx = leaf_hx(sq, lf[1])
        hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx
        y0 = lf[3] * hy
        ids = sq.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            thq = 0.0; uxq = 0.0; uzq = 0.0; pq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                thq += theta[ids[k]] * b
                uxq += ux[ids[k]] * b
                uzq += uz[ids[k]] * b
            end
            for b in 1:nq, a in 1:nq
                pq += pv[pids[(b - 1) * nq + a]] *
                      fd.phi_p[a, qx] * fd.phi_p[b, qy]
            end
            bin = clamp(floor(Int, nbins * y) + 1, 1, nbins)
            accT[bin] += w * thq
            accX[bin] += w * uxq
            accZ[bin] += w * uzq
            accP[bin] += w * pq
            accW[bin] += w
        end
    end
    ys = [(b - 0.5) / nbins for b in 1:nbins]
    # A bin can be empty when it is finer than the leaf quadrature spacing; return
    # NaN there so a profile plot shows the gap instead of a fake zero.
    th = fill(NaN, nbins); uxp = fill(NaN, nbins)
    uzp = fill(NaN, nbins); pp = fill(NaN, nbins)
    for b in 1:nbins
        accW[b] > 0.0 || continue
        th[b] = accT[b] / accW[b]
        uxp[b] = accX[b] / accW[b]
        uzp[b] = accZ[b] / accW[b]
        pp[b] = accP[b] / accW[b]
    end
    return (y=ys, theta=th, ux=uxp, uz=uzp, p=pp)
end
