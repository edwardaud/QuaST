# Blankenbach template cards

Five ASPECT cards that reproduce the Blankenbach et al. (1989) benchmark cases,
used to check that ASPECT and QuaST agree on a case with a published answer.

| Card | Case |
|---|---|
| `tpl_case1a.prm` | 1a: steady, `Ra = 1e4`, `H = 0` |
| `tpl_case1b.prm` | 1b: steady, `Ra = 1e4`, `H = 10` |
| `tpl_case1c.prm` | 1c: steady, `Ra = 1e4`, `H = 0`, `T`-dependent viscosity |
| `tpl_case2a.prm` | 2a: steady, `Ra = 1e5`, `H = 10` |
| `tpl_case2b.prm` | 2b: steady, `Ra = 1e4`, joint `T`- and depth-dependent viscosity |

They are templates rather than finished cards: the mesh refinement and the boundary
blocks are meant to be varied by whoever runs them. QuaST's side of the same cases
is `examples/blankenbach_2b.jl` and the gates under `test/`.
