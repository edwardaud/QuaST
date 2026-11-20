#!/usr/bin/env julia
# Where the multigrid hierarchy's memory goes, per level, as the leaf count grows.
#
#   julia --project=. scripts/measure_mg_memory.jl
#
# The 25 GB SIGBUS at 160 000 leaves (docs/dev/VALIDATION.md 9.x) says *that* it
# runs out; this says *where*, so the fix can be aimed: a lazier level cache if the
# per-level matrices dominate, a smaller leaf budget if the tree itself does.
#
# Reports, per target size: leaves, dofs, the hierarchy's total `summarysize`, the
# same split by level (tree, Cu, Cut, P, A) and the process RSS from /proc.

using QuaST, Printf, SparseArrays

const P = 2

rss_kb() = parse(Int, split(read("/proc/self/status", String), '\n')[findfirst(
    l -> startswith(l, "VmRSS:"),
    split(read("/proc/self/status", String), '\n'))] |> l -> split(l)[2])

function build_tree(nleaf_target::Int, maxlev::Int)
    qt = Quadtree(maxlev; Lx=2.5, cells_x=5, cells_y=2)
    # refine uniformly until close to the target, then stop
    while length(qt.leaves) < nleaf_target
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        length(qt.leaves) > 8 * nleaf_target && break
    end
    return qt
end

function measure(nleaf_target::Int, maxlev::Int)
    qt = build_tree(nleaf_target, maxlev)
    ss = QuaST.make_semstokes(qt, P)
    mg = QuaST.make_semquad_mg(ss, qt, P, :free, false, (x, y) -> 1.0, nothing;
                         cycles=1, operator=:matrixfree,
                         geom_assembly=:minimal)
    geo = mg.geo
    gbytes = Base.summarysize(geo.qts) + Base.summarysize(geo.sss) +
             Base.summarysize(geo.Cus) + Base.summarysize(geo.Cuts) +
             Base.summarysize(geo.Ps)
    abytes = Base.summarysize(mg.A)
    total = Base.summarysize(mg)
    @printf("%8d %10d %8d %9.1f %9.1f %9.1f %9.1f %9.1f %8.0f\n",
            length(qt.leaves), 2 * ss.su.N, length(geo.Cus),
            total / 2.0^20, gbytes / 2.0^20, Base.summarysize(geo.Cus) / 2.0^20,
            Base.summarysize(geo.Ps) / 2.0^20, abytes / 2.0^20, rss_kb() / 1024.0)
end

println(" leaves       dofs   levels    mg MB    geo MB     Cu MB     P MB    A MB   RSS MB")
for (n, lev) in ((10_000, 8), (20_000, 9), (40_000, 9), (80_000, 9))
    measure(n, lev)
end
