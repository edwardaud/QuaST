# Initial-condition data

`initial_temperature_case2b.txt` is the Blankenbach 2b initial temperature as an
ASPECT `ascii data` file: a header block, then a grid of values that
`examples/benchmark_cases.jl` reads to start from the same state ASPECT
starts from. The gate `t_g150_regression_snapshot` reads the same field.

The format is ASPECT's: `# POINTS: nx nz`, the two coordinate blocks, then the
data block. Do not reformat it; the reader expects that layout.

The other Blankenbach initial conditions (1b, 1c, 2a) were removed with the
uniform-mesh solver, which was their only reader.

Keep a new field in this directory only when a tracked script reads it. Run
outputs belong under `out_julia/`, which git ignores.

## Licence

The five `initial_temperature_case*.txt` fields are published under
**CC0 1.0 Universal** (SPDX: `CC0-1.0`); the full text is in `LICENSE` in this
directory, fetched from
<https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt>.

The reason is that they are small machine-generated numeric fields with no
creative expression, so CC0 gives a reader the least friction: no attribution
obligation, no licence-compatibility question beside the MIT code, and the
scientific credit stays with the citation of Blankenbach et al. (1989) that
`examples/benchmark_cases.jl` prints and the validation page repeats. The rest
of the repository is MIT; the ASPECT plugin is GPL-2.0-or-later. See the
top-level [`NOTICE`](../../NOTICE) for the whole split.
