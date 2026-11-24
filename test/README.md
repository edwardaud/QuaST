# Gates

One file per gate: `t_g<N>_<slug>.jl`. Numbers are historical, not contiguous:
retired gates left gaps and split gates kept their number. Each is a standalone
test script that starts with `using`; most wrap one `@testset`. The 184 files
hold the suite's 133 716 assertions (133 714 passed, 2 broken)
quoted in the top-level README, and
`docs/dev/FILES.md` lists their one-line headers, so they are not repeated here.

G265 is the gate of `examples/subduction_2d.jl` (the regional subduction-style
case). It pins only the checks the design classifies as exact identities or as
the mass-flux conservation identity, with their stated tolerances. The example's
open-box heat budget, plate-cooling profile and resolution sequence are
`rheology` / `phases` / `full` / `converge` mode outputs, not gate pins: they
are asymptotic or discretisation-level comparisons and cost minutes each, which
is stated in the gate header as well.

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
