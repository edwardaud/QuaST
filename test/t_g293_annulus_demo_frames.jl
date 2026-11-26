# G293 閳?the annulus AMR demonstration preset: frame stream, per-frame mesh dump
# and the tree cap that makes the AMR cycle refine at all.
#
# What this gate pins (cheap; no multi-minute run):
#
#   (a) the `demo` preset's physics card (H and deta_T; eta = 1000^(-T)) and
#       that it does not leak into the benchmark card: without the `demo` token
#       (or an ANN_* knob) `annulus_benchmark_params()` is the historical
#       isoviscous card and the default run prints the pre-existing header
#       lines;
#   (b) the tree cap bug: a tree built at level 2 has maxlevel = 2 and `refine!`
#       refuses there, so a run that asked for AMR would silently keep a frozen
#       mesh.  The demo tree is built with `maxlevel = ANN_MAX_LEVEL`; the
#       benchmark tree keeps its historical cap (maxlevel == its level);
#   (c) a tiny end-to-end demo run (tstop = 0.002, frames on, mesh dump on)
#       whose frame stream is consistent: for every frame the binary has the
#       declared node count and finite fields, the mesh CSV exists with exactly
#       `leaves` rows, the meta level range and leaf count equal the dump's,
#       the leaves tile the logical unit square, the tree is 2:1 balanced and
#       the ring seam is periodic.
#
# The full demonstration (t = 0 -> 0.2, 41 frames, 12 000-leaf cap) is not in
# this gate; the figure and its JSON sidecar are its record.
using Test
using QuaST
using Printf
const MS293 = QuaST

# The shipped example, included in an isolated module (its `main` is guarded by
# `abspath(PROGRAM_FILE) == @__FILE__` and does not execute on include).
module G293Harness
using QuaST, Printf
include(joinpath(@__DIR__, "..", "examples", "annulus_benchmark.jl"))
end
const HB = G293Harness

"""Run `f` with stdout captured; Julia 1.12 redirects only Pipe/DevNull."""
function g293_capture(f)
    p = Pipe()
    local result
    redirect_stdout(p) do
        result = f()
    end
    close(p.in)
    return result, read(p.out, String)
end


"""Read a v2 frame_meta.csv into (header lines, column names, row dicts)."""
function g293_meta(path)
    header, cols, rows = String[], String[], Vector{String}[]
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && (push!(header, s); continue)
        push!(rows, split(s, ','))
        isempty(cols) && (cols = split(s, ','))
    end
    data = [Dict(zip(cols, v)) for v in rows[2:end]]
    return header, cols, data
end

@testset "G293 annulus AMR demo frames" begin

    # ==================================================================
    # (a) the demo card and the untouched benchmark card
    # ==================================================================
    @testset "(a) demo preset card" begin
        pr = HB.annulus_benchmark_params(; H=0.0, deta_T=1000.0)
        @test pr.H == 0.0
        @test pr.deta_T == 1000.0
        @test pr.eta_law === :expT
        # the law is eta = 1000^(-T) = exp(-log(1000) T)
        @test MS293.eta_at(0.0, 0.0, pr)[1] == 1.0
        @test isapprox(MS293.eta_at(1.0, 1.0, pr)[1], 1.0e-3; rtol=1e-12)
        @test HB.annulus_eta_label(pr.deta_T) == "eta=1000^(-T)"
        # the benchmark default is untouched: no keyword -> historical card
        pr0 = HB.annulus_benchmark_params()
        @test pr0.Ra == 1.0e5
        @test pr0.H == 0.0
        @test pr0.deta_T == 1.0
        @test MS293.eta_at(0.37, 0.42, pr0)[1] == 1.0
    end

    # ==================================================================
    # (b) the demo tree carries the refinement cap; the benchmark tree does not
    # ==================================================================
    @testset "(b) tree refinement cap" begin
        qt = HB.annulus_benchmark_tree(4, 48; maxlevel=7)
        @test qt.maxlevel == 7
        @test qt.seam === :periodic
        @test length(qt.leaves) == 192
        @test all(lf -> lf[1] == 2, qt.leaves)
        # the cap makes refinement possible (without it refine! refuses at the
        # construction level, which is exactly the silent-freeze defect)
        @test refine!(qt, first(qt.leaves))
        @test length(qt.leaves) == 195
        # the historical benchmark tree keeps maxlevel == its own level
        q8 = HB.annulus_benchmark_tree(8, 96)
        @test q8.maxlevel == 3
        @test all(lf -> lf[1] == 3, q8.leaves)
        @test !refine!(q8, first(q8.leaves))   # refused at the historical cap
    end

    # ==================================================================
    # (c) the tiny end-to-end demo run and its frame/mesh stream
    # ==================================================================
    @testset "(c) frame stream consistency" begin
        frames = mktempdir()
        st, text = g293_capture() do
            withenv("ANN_FRAMES" => frames, "ANN_MESH_DUMP" => "1",
                    "ANN_TSTOP" => "0.002", "ANN_FRAME_DT" => "5.0e-4",
                    "ANN_REPORT" => "1000000") do
                HB.main(["demo"])
            end
        end
        @test st !== nothing
        @test st.steps >= 1
        @test st.t >= 0.002 - 1.0e-12
        # the demo header lines are the demo's, and the benchmark header lines
        # are not printed by this path
        @test occursin("# demo: purpose-configured AMR demonstration", text)
        @test occursin("mesh_dump=on", text)
        @test occursin("ANNULUS_AMR preset=demo", text)
        @test !occursin("# card: Ra=1e5 isoviscous", text)

        meta = joinpath(frames, "frame_meta.csv")
        @test isfile(meta)
        header, cols, rows = g293_meta(meta)
        @test any(h -> occursin("annulus_frames_v2", h), header)
        @test any(h -> occursin("mesh_dump", h), header)
        for c in ("frame", "step", "t", "Nu_T", "Nu_B", "Vrms", "meanT",
                  "Tmin", "Tmax", "nnodes", "leaves", "level_min",
                  "level_max", "dt")
            @test c in cols
        end
        @test length(rows) >= 2            # several frames over the mesh changes
        # the run must actually have refined: initial_refine = 2 and
        # wall_min_level = 3 cannot leave every frame at the level-2 tree
        @test maximum(parse.(Int, [r["level_max"] for r in rows])) >= 3
        @test maximum(parse.(Int, [r["leaves"] for r in rows])) > 192

        for r in rows
            f = parse(Int, r["frame"])
            n = parse(Int, r["nnodes"])
            binp = joinpath(frames, @sprintf("frame_%05d.bin", f))
            @test isfile(binp)
            @test filesize(binp) == 8 + 5 * 8 * n
            raw = open(binp, "r") do io
                nn = Int(read(io, Int64))
                data = Vector{Float64}(undef, 5 * nn)
                read!(io, data)
                (nn, data)
            end
            @test raw[1] == n
            @test all(isfinite, raw[2])
            x = raw[2][1:n]
            y = raw[2][(n + 1):(2 * n)]
            T = raw[2][(3 * n + 1):(4 * n)]
            rr = hypot.(x, y)
            @test abs(minimum(rr) - 1.22) < 1e-9       # the card's ring
            @test abs(maximum(rr) - 2.22) < 1e-9
            @test minimum(T) >= -1.0e-9
            @test maximum(T) <= 1.0 + 1.0e-9

            meshp = joinpath(frames, @sprintf("frame_%05d_mesh.csv", f))
            @test isfile(meshp)
            mkheader, mrows = String[], Vector{NTuple{5,Float64}}()
            for line in eachline(meshp)
                s = strip(line)
                isempty(s) && continue
                startswith(s, "#") && (push!(mkheader, s); continue)
                v = split(s, ',')
                @test length(v) == 5
                push!(mrows, (parse(Int, v[1]), parse(Float64, v[2]),
                              parse(Float64, v[3]), parse(Float64, v[4]),
                              parse(Float64, v[5])))
            end
            @test any(h -> occursin("logical", h), mkheader)
            @test length(mrows) == parse(Int, r["leaves"])
            lv = [m[1] for m in mrows]
            @test minimum(lv) == parse(Int, r["level_min"])
            @test maximum(lv) == parse(Int, r["level_max"])
            # the leaves tile the logical unit square: the ring tree has
            # nx0*ny0 = 1*12 base cells, so sum(hx*hy) = sum(4^-level)/(12) = 1,
            # i.e. sum(4^-level) == 12
            @test isapprox(sum(4.0 .^ (-float.(lv))), 12.0; rtol=1e-12)
            @test all(m -> m[4] > 0.0 && m[5] > 0.0, mrows)
            # logical coordinates lie inside the unit square
            @test all(m -> m[2] >= -1e-12 && m[2] + m[4] <= 1.0 + 1e-12 &&
                           m[3] >= -1e-12 && m[3] + m[5] <= 1.0 + 1e-12, mrows)
        end

        # the tree the driver returns is a balanced periodic ring on the demo's
        # refinement cap.  (Its leaves are not the last *frame's* leaves: a
        # frame is written only when the cadence is due, and is deferred by one
        # step after an AMR step, so the last frame predates the last step.)
        @test st.qt.seam === :periodic
        @test MS293._geom_periodic(st.ss.su.geom)
        @test MS293.max_face_level_jump(st.qt) <= 1
        @test st.qt.maxlevel == 7
        @test length(st.qt.leaves) > 192
    end

    # ==================================================================
    # (d) the demo preset does not change the benchmark default output
    # ==================================================================
    @testset "(d) benchmark default path unchanged" begin
        st0, text = g293_capture() do
            withenv("ANN_TSTOP" => nothing, "ANN_MAXSTEPS" => nothing,
                    "ANN_RA" => nothing, "ANN_H" => nothing, "ANN_DETA_T" => nothing,
                    "ANN_IC" => nothing, "ANN_FRAMES" => nothing,
                    "ANN_MESH_DUMP" => nothing, "ANN_AMR_EVERY" => nothing,
                    "ANN_INITIAL_REFINE" => nothing, "ANN_MAX_LEVEL" => nothing,
                    "ANN_LINEAR" => nothing, "ANN_REPORT" => nothing) do
                HB.main(["4x48", "1.0e-12"])
            end
        end
        # the pre-existing header lines, byte for byte
        @test occursin("# annulus benchmark nx=4 ny=48 level=2 cells_y=12 " *
                       "leaves=192 dofs=1968 (u=1728 p=240 T=864)", text)
        @test occursin("# card: Ra=1e5 isoviscous H=0 free-slip ring R0=1.22 " *
                       "R1=2.22 phi=2pi; advection=galerkin ev=0.0 " *
                       "ic=firedrake", text)
        @test occursin("# criterion: t > 0.02, change_t/|dNu_T|/|dV_RMS| < " *
                       "1.0e-6 for 20 consecutive steps (steady_vrms=true)",
                       text)
        @test !occursin("# demo:", text)
        @test !occursin("mesh_dump", text)
        @test st0 !== nothing
        @test st0.steps >= 1
    end
end
