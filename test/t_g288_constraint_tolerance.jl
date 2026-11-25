# G288 (remediation batch P0 part 1): the hanging-constraint tolerance is
# discriminated by a near-coincident node (AUDIT_GATES M11 (iii), mutation m17).
#
# The hole: `semquad_constraints` skips a coarse GLL node that coincides with a
# fine node (`minimum(abs.(sq.xi .- xiL)) < 1.0e-10`).  Every existing fixture
# had shared nodes at distance 0 or distinct nodes at O(0.1), so mutating the
# tolerance to `1e-3` (m17/m17b) was invisible to G24/G46/G268/G273/G271.
#
# The fixture: a two-level-gap tree, coarse leaf (level 2) covering
# [0, 1/4] x [1/4, 1/2], with four level-22 leaves directly below it.  The fine
# leaves' top-face nodes sit at coarse-reference abscissae -1 + (xi+1) 2^-20,
# i.e. the first non-shared node is 2^-21 in *relative face position* from the
# coarse corner -- genuinely a different physical node (different global id, a
# different x), but within the (1e-10, 1e-3) band the tolerance must decide.
# With the production `1e-10` all eight non-corner fine nodes are constrained;
# with the mutated `1e-3` all eight are skipped as "shared", which is a wrong
# constraint set (the fine side becomes discontinuous against the coarse face).
#
# Why the assertions are not tautological:
#   * the gate asserts the *count* of constraint rows on the coarse face (8) and
#     that the shared corner itself is not constrained, so both a too-loose
#     tolerance (0 rows) and a too-tight one (9 rows) are red;
#   * it measures the near-coincidence in physical coordinates (the smallest
#     node-to-corner separation as a fraction of the coarse face) and asserts it
#     lies strictly inside (1e-10, 1e-3): this is the band the constant
#     resolves, printed in the gate output;
#   * the weights are checked against the endpoint behavior of the coarse Q_2
#     Lagrange basis: for a point 2^-20 from the endpoint the corner weight is
#     1 - O(2^-20) (> 0.99999), so a wrong master set is red;
#   * the same fixture is run through the **mapped** constraint body as well
#     (an `AffineBox` geometry), because the 1e-10 appears in both bodies; a
#     mutation of either body is caught.
using Test
using LinearAlgebra
using QuaST
const MS288 = QuaST

const G288_LCOARSE = 2
const G288_LFINE = 22

function g288_tree()
    coarse = (G288_LCOARSE, 0, 1)          # [0,1/4] x [1/4,1/2]
    leaves = Set{NTuple{3,Int}}([coarse])
    toprow = (1 << 20) - 1                 # level-22 cells y in [1/4 - h, 1/4]
    for ix in 0:3
        push!(leaves, (G288_LFINE, ix, toprow))
    end
    return MS288.Quadtree(G288_LFINE, 1.0, 1, 1, leaves)
end

# every constraint row anchored on the coarse face y = 1/4, x in (0, 1/4)
function g288_face_rows(sq, cons)
    rows = Tuple{Int,Vector{Int},Vector{Float64}}[]
    for (c, (ms, ws)) in cons
        x, y = sq.coords[c]
        (abs(y - 0.25) < 1.0e-12 && x > 0.0 && x < 0.25) || continue
        push!(rows, (c, ms, ws))
    end
    return rows
end

@testset "G288 hanging-constraint tolerance is discriminated" begin
    qt = g288_tree()
    @test length(qt.leaves) == 5
    sq = make_semquad(qt, 2)
    cons = semquad_constraints(sq, qt)
    rows = g288_face_rows(sq, cons)
    println("G288: constraints on the coarse face = ", length(rows),
            "  (production tolerance 1.0e-10; mutation 1.0e-3 would skip all)")
    # ---- the production tolerance keeps every non-corner fine node ----------
    @test length(rows) == 8
    # the shared corner (x = 0) must NOT be a constraint (exact coincidence skip)
    @test all(!(abs(sq.coords[c][1]) < 1.0e-14) for (c, _, _) in rows)
    @test minimum(abs(sq.coords[c][1]) for (c, _, _) in rows) < 1.0e-6

    # ---- the near-coincidence is inside the band (1e-10, 1e-3) --------------
    # coarse face length 0.25; the smallest separation is 2^-23 (half a level-22
    # cell) = 1.19e-7 in x, i.e. 4.77e-7 of the face
    sep = minimum(abs(sq.coords[c][1]) for (c, _, _) in rows) / 0.25
    println("G288: smallest near-coincident separation / face length = ", sep,
            "   (band: 1.0e-10 < sep < 1.0e-3)")
    @test sep > 1.0e-10
    @test sep < 1.0e-3
    @test sep == 2.0^-21

    # ---- the weights are the endpoint-dominated coarse Q_2 interpolation ----
    for (c, ms, ws) in rows
        @test ms[1] == sq.elem[(G288_LCOARSE, 0, 1)][1]   # the corner master
        @test sum(ws) ≈ 1.0 atol=1.0e-12                  # partition of unity
        @test all(isfinite, ws)
        # corner dominated: the farthest near-coincident node is 8*2^-20
        # (7.6e-6) from the corner, so 1 - ws[1] <= ~1.2e-5 measured; a wrong
        # master set would deviate by O(1).
        @test ws[1] > 0.99998
        @test abs(ws[1] - 1.0) < 2.0e-5
    end

    # ---- the x-face pair exercises the *other* box-body tolerance site ----
    # (the 1e-10 appears once per face orientation; the audit mutation m17 is
    # the x-face site, so the gate must cover both orientations)
    coarse_x = (G288_LCOARSE, 1, 0)          # [1/4,1/2] x [0,1/4]
    leaves_x = Set{NTuple{3,Int}}([coarse_x])
    lastcol = (1 << 20) - 1                  # level-22 cells x in [1/4-h, 1/4]
    for iy in 0:3
        push!(leaves_x, (G288_LFINE, lastcol, iy))
    end
    qtx = MS288.Quadtree(G288_LFINE, 1.0, 1, 1, leaves_x)
    sqx = make_semquad(qtx, 2)
    consx = semquad_constraints(sqx, qtx)
    rowsx = Tuple{Int,Vector{Int},Vector{Float64}}[]
    for (c, (ms, ws)) in consx
        x, y = sqx.coords[c]
        (abs(x - 0.25) < 1.0e-12 && y > 0.0 && y < 0.25) || continue
        push!(rowsx, (c, ms, ws))
    end
    println("G288: x-face constraints on the coarse face = ", length(rowsx))
    @test length(rowsx) == 8
    sepx = minimum(abs(sqx.coords[c][2]) for (c, _, _) in rowsx) / 0.25
    @test sepx == 2.0^-21
    @test all(abs(sum(ws) - 1.0) < 1.0e-12 for (_, _, ws) in rowsx)

    # ---- the mapped constraint body (same constant, logical arithmetic) -----
    aff = AffineBox(A=(1.0, 0.0, 0.0, 1.0), b=(0.0, 0.0), Lx=1.0)
    sqm = make_semquad(qt, 2; geom=aff)
    consm = semquad_constraints(sqm, qt)          # dispatches on sqm.geom
    rows = g288_face_rows(sqm, consm)
    @test length(rows) == 8
    sep = minimum(abs(sqm.coords[c][1]) for (c, _, _) in rows) / 0.25
    @test sep == 2.0^-21
    for (c, ms, ws) in rows
        @test sum(ws) ≈ 1.0 atol=1.0e-12
        @test ws[1] > 0.99998
    end
end
