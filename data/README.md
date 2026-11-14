# Data

Input data for the cards and gates. Nothing is tracked here at HEAD: the three
ASCII initial-temperature fields this directory held were read only by a gate
that was deleted, so they went with it at `8a73c9f`. The one initial-temperature
field still in the tree is `examples/data/initial_temperature_case2b.txt`, which
the Blankenbach 2b examples and gates read. Keep a new field here only when a
tracked script reads it; run outputs belong under `out_julia/`, which git
ignores.

| File | What it is |
|---|---|
| `examples/data/initial_temperature_case2b.txt` | The 2b initial temperature read by `examples/blankenbach_2b_cbf_ev.jl` and four gates. |
