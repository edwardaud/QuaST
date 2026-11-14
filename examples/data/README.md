# Initial-condition data

`initial_temperature_case2b.txt` is the Blankenbach 2b initial temperature as an
ASPECT `ascii data` file: a header block, then a grid of values that
`examples/blankenbach_2b_cbf_ev.jl` reads to start from the same state ASPECT
starts from. The gate `t_g150_regression_snapshot` reads the same field.

The format is ASPECT's: `# POINTS: nx nz`, the two coordinate blocks, then the
data block. Do not reformat it; the reader expects that layout.

The other Blankenbach initial conditions (1b, 1c, 2a) were removed with the
uniform-mesh solver, which was their only reader.

Keep a new field in this directory only when a tracked script reads it. Run
outputs belong under `out_julia/`, which git ignores.
