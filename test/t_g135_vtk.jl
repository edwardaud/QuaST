using Test, QuaST

# G135: VTK output and depth averages.
#
# The state is chosen so every field is a low-order polynomial on the leaves:
#   theta = 2y,  ux = 1,  uz = 0,  p = x + 2y.
# Then the VTK point data can be checked against semquad_eval, and the
# horizontal average has the closed form  theta -> 2y,  ux -> 1,  p -> 0.5 + 2y.

function _parse_scalars(lines, name, npts)
    i0 = findfirst(l -> l == "SCALARS $name double 1", lines)
    @test i0 !== nothing
    @test lines[i0 + 1] == "LOOKUP_TABLE default"
    return [parse(Float64, lines[i0 + 1 + k]) for k in 1:npts]
end

@testset "G135 VTK output and depth average" begin
    qt = Quadtree(5)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    sp = ss.sp
    # velocity / temperature fields at the Q_p nodes
    theta = [2.0 * y for (_, y) in sq.coords]
    ux = ones(sq.N)
    uz = zeros(sq.N)
    # pressure = x + 2y is bilinear, so it is represented exactly by Q_1:
    # set its nodal values from the pressure nodes' physical coordinates
    pv = zeros(sp.N)
    nq1 = sp.p + 1
    for lf in qt.leaves
        h = 2.0^(-lf[1])
        x0 = lf[2] * h; y0 = lf[3] * h
        ids = sp.elem[lf]
        for j in 1:nq1, i in 1:nq1
            x = x0 + 0.5 * (sp.xi[i] + 1.0) * h
            y = y0 + 0.5 * (sp.xi[j] + 1.0) * h
            pv[ids[(j - 1) * nq1 + i]] = x + 2.0 * y
        end
    end
    @test maximum(abs, [semstokes_eval(ss, vcat(ux, uz), pv, x, y)[3] -
                        (x + 2.0 * y) for (x, y) in sq.coords]) < 1.0e-12

    tmp = tempname() * ".vtk"
    nleaf = length(qt.leaves)
    p = sq.p
    n = p + 1
    stats = write_semquad_vtk(tmp, ss, qt, theta, ux, uz, pv;
                              eta_fun=(x, y) -> 1.0 + x)
    @test stats.points == nleaf * n * n
    @test stats.cells == nleaf * p * p

    lines = readlines(tmp)
    @test lines[1] == "# vtk DataFile Version 3.0"
    @test lines[3] == "ASCII"
    @test lines[4] == "DATASET UNSTRUCTURED_GRID"
    ip = findfirst(l -> startswith(l, "POINTS "), lines)
    @test parse(Int, split(lines[ip])[2]) == stats.points
    ic = findfirst(l -> startswith(l, "CELLS "), lines)
    @test parse(Int, split(lines[ic])[2]) == stats.cells
    @test parse(Int, split(lines[ic])[3]) == 5 * stats.cells
    # all cells are VTK_QUAD and their node indices are in range
    it = findfirst(l -> startswith(l, "CELL_TYPES "), lines)
    @test all(parse(Int, lines[it + k]) == 9 for k in 1:stats.cells)
    for k in 1:stats.cells
        v = split(lines[ic + k])
        @test v[1] == "4"
        @test all(1 <= parse(Int, v[j]) <= stats.points for j in 2:5)
    end

    # point coordinates and theta agree with the interpolation
    pts = zeros(stats.points, 2)
    for k in 1:stats.points
        v = split(lines[ip + k])
        pts[k, 1] = parse(Float64, v[1])
        pts[k, 2] = parse(Float64, v[2])
        @test parse(Float64, v[3]) == 0.0
    end
    th_vtk = _parse_scalars(lines, "theta", stats.points)
    p_vtk = _parse_scalars(lines, "pressure", stats.points)
    eta_vtk = _parse_scalars(lines, "viscosity", stats.points)
    thr = [semquad_eval(sq, theta, pts[k, 1], pts[k, 2]) for k in 1:stats.points]
    @test maximum(abs, th_vtk .- thr) < 1.0e-12
    @test maximum(abs, p_vtk .- [pts[k, 1] + 2.0 * pts[k, 2] for k in 1:stats.points]) < 1.0e-12
    @test maximum(abs, eta_vtk .- [1.0 + pts[k, 1] for k in 1:stats.points]) < 1.0e-12
    @test maximum(abs, th_vtk .- 2.0 .* pts[:, 2]) < 1.0e-12

    # velocity vectors: (1, 0, 0)
    iv = findfirst(l -> l == "VECTORS velocity double", lines)
    @test iv !== nothing
    for k in 1:stats.points
        v = split(lines[iv + k])
        @test parse(Float64, v[1]) == 1.0
        @test parse(Float64, v[2]) == 0.0
        @test parse(Float64, v[3]) == 0.0
    end

    # cell data: level and h
    il = findfirst(l -> l == "SCALARS level int 1", lines)
    @test il !== nothing
    lev = [parse(Int, lines[il + 1 + k]) for k in 1:stats.cells]
    @test all(lev .== 2)                      # two uniform refinement rounds
    ih = findfirst(l -> l == "SCALARS h double 1", lines)
    @test ih !== nothing
    hs = [parse(Float64, lines[ih + 1 + k]) for k in 1:stats.cells]
    @test maximum(abs, hs .- 2.0 .^ (-lev)) < 1.0e-15

    # ---- depth average -----------------------------------------------------
    # bins aligned with the 4 leaf rows (h = 0.25): the Gauss points in each bin
    # are symmetric, so a linear field is averaged exactly
    prof = semquad_depth_average(ss, qt, theta, ux, uz, pv; nbins=4)
    @test length(prof.y) == 4
    @test maximum(abs, prof.theta .- 2.0 .* prof.y) < 1.0e-12
    @test maximum(abs, prof.ux .- 1.0) < 1.0e-12
    @test maximum(abs, prof.uz) < 1.0e-12
    @test maximum(abs, prof.p .- (0.5 .+ 2.0 .* prof.y)) < 1.0e-12
    # a finer binning is only O(h^2) accurate (points are assigned to bins, not
    # integrated over them); the error must stay at that level
    prof8 = semquad_depth_average(ss, qt, theta, ux, uz, pv; nbins=8)
    @test length(prof8.y) == 8
    @test all(isfinite, prof8.theta)
    @test maximum(abs, prof8.theta .- 2.0 .* prof8.y) < 2.0e-2
    @test maximum(abs, prof8.ux .- 1.0) < 1.0e-12
    # bins finer than the quadrature spacing are empty and reported as NaN
    # (a gap in the profile, not a fake zero)
    prof25 = semquad_depth_average(ss, qt, theta, ux, uz, pv; nbins=25)
    @test any(isnan, prof25.theta)
    fin = .!isnan.(prof25.theta)
    @test any(fin)
    @test maximum(abs, prof25.theta[fin] .- 2.0 .* prof25.y[fin]) < 2.0 / 25 + 1.0e-12

    rm(tmp; force=true)

    # ---- stretched box -----------------------------------------------------
    qt3 = Quadtree(2; Lx=3.0)
    for _ in 1:2, lf in collect(qt3.leaves)
        refine!(qt3, lf)
    end
    balance!(qt3)
    ss3 = make_semstokes(qt3, 2)
    th3 = ones(ss3.su.N)
    ux3 = ones(ss3.su.N)
    uz3 = zeros(ss3.su.N)
    pv3 = zeros(ss3.sp.N)
    tmp3 = tempname() * ".vtk"
    st3 = write_semquad_vtk(tmp3, ss3, qt3, th3, ux3, uz3, pv3)
    lines3 = readlines(tmp3)
    ip3 = findfirst(l -> startswith(l, "POINTS "), lines3)
    xs3 = [parse(Float64, split(lines3[ip3 + k])[1]) for k in 1:st3.points]
    ys3 = [parse(Float64, split(lines3[ip3 + k])[2]) for k in 1:st3.points]
    @test maximum(xs3) ≈ 3.0 atol = 1.0e-14
    @test maximum(ys3) ≈ 1.0 atol = 1.0e-14
    pr3 = semquad_depth_average(ss3, qt3, th3, ux3, uz3, pv3; nbins=4)
    @test all(abs.(pr3.theta .- 1.0) .< 1.0e-14)
    @test all(abs.(pr3.ux .- 1.0) .< 1.0e-14)
    rm(tmp3; force=true)
end
