# Gates

One file per gate: `t_g<N>_<slug>.jl`. Numbers are historical, not contiguous:
retired gates left gaps and split gates kept their number. Each is a standalone
test script that starts with `using`; most wrap one `@testset`. The 185 files
hold the suite's 134 086 assertions (134 084 passed, 2 broken, 0 failed) — the
count of the stage-0 cut-1 tree, recorded with its evidence in
`docs/dev/GEOM_PROGRESS.md`. The top-level README still quotes the pre-cut-1
total (133 716); the top-level markdown was outside that cut's write scope.
`docs/dev/FILES.md` lists the one-line header of every gate, so they are not
repeated here.

G266 is the geometry/mapping-layer gate of stage 0 cut 1: it rebuilds the box
coordinates, node ids, leaf/face metrics, nodal transfer, AMR indicator and the
checkpoint geometry record from the geometry layer and compares them, in
process, with the historical implementations carried in the gate (no recorded
cross-machine SHA literal), and it checks the map identities of the two
validation geometries `AffineBox`/`CurvedBox` (roundtrip, analytic `det J`,
outward normals, face/volume measures, patch test) plus the checkpoint
geometry record/rejection. G265 is the gate of `examples/subduction_2d.jl` (the
regional subduction-style case). It pins only the checks the design classifies
as exact identities or as the mass-flux conservation identity, with their
stated tolerances. The example's open-box heat budget, plate-cooling profile
and resolution sequence are `rheology` / `phases` / `full` / `converge` mode
outputs, not gate pins: they are asymptotic or discretisation-level comparisons
and cost minutes each, which is stated in the gate header as well.

| File | What it is |
|---|---|
| `runtests.jl` | Includes every `t_g*.jl` in order inside `@testset "QuaST gates"`; this is the suite CI runs. |
| `run_gates.jl` | Includes the gate files named on the command line, in one process. |

From the repository root, one gate:

    julia --project=. test/run_gates.jl t_g229_default_balanced.jl

and the whole suite:

    julia --project=. --threads=8 test/runtests.jl

A kernel gate such as `t_g0.jl` takes about 2 s warm; the heavy
checkpoint/regression gates run to minutes. The full suite is about 22 min on
eight threads (CI allows 90 at four).
