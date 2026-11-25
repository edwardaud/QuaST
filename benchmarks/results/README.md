# Benchmark provenance records

One record per number that a paper, `docs/en/validation.md` or the README
quotes. The directory is the repository-side answer to the Royal Society /
GMD / AGU requirement that every reported result has a traceable run: not a
summary of the campaign, but the command, the conditions and the raw log of the
run that produced *that* number.

The records are released with the code under CC0-1.0 (see `NOTICE` at the
repository root); they are measured data, not software. Nothing here is
fabricated: a record may say `UNVERIFIED` or `stale` and give the reason, but it
may not carry a number that no run produced.

## What a record holds

Every `*.md` file under this directory is one number, or one tight group of
numbers from the same run, and answers all of:

| Field | What it must say |
|---|---|
| Quantity | the number and what it is (`Nu_top`, `Nu_T`, `Tmax` at frame 25, a relative difference against a published value, ...) |
| Command | the exact command line, with `QUAST_ROOT`, `QUAST_JULIA` and `QUAST_OUT` shown when they were set |
| Input card / fixture | the tracked file the run started from, plus its SHA-256; for a generated card, the generator and the seed |
| Resolution | level, leaf count and the reported DOF count (`N`) |
| Solver path | `linear=:direct` or `linear=:mg`, and any `CASE_*` knob that differs from the card default |
| Wall clock | the measured wall time of the run, and the number of Julia threads |
| Environment | Julia version, `Manifest.toml` hash, OS, CPU model, RAM |
| Log | the log file name, its SHA-256 and its size; the log text itself if it is under about 40 kB, otherwise the Zenodo record that holds it |
| Status | `verified`, `unverified`, `stale` or `superseded`, and for anything but `verified` the reason |
| Recorded | the date the record was written and by which commit the run was made |

A record is a small text file. **Keep it under 100 kB**; a run log larger than
that does not belong in the repository: put the log in the Zenodo deposit of
the corresponding release and reference it with its DOI and SHA-256. No
checkpoints, frames, HDF5 files, videos or other binaries belong here; they are
covered by `.gitignore` and must be archived instead.

## File names

`<case>_<quantity>_<resolution>[_<variant>].md`, lower case, ASCII, hyphen-free
segments:

- case: `blankenbach1a`, `blankenbach1b`, `blankenbach1c`, `blankenbach2a`,
  `blankenbach2b`, `tosi1`, `tosi5a`, `annulus`, `blob`, `realistic`,
  `aspc-realistic` (the ASPECT arm of the cross-code comparison), `channel`,
  `subduction`, `gates` (suite totals), `dump` (the nine state hashes);
- quantity: `nu-top`, `nu-bot`, `nu`, `vrms`, `tmax`, `nusselt-convention`,
  `suite-total`, `state-hashes`;
- resolution: `L3`/`L4`/`L5` for a quadtree level, `walls5` for a wall-refined
  card, `128x1536` for a fixed grid, `rung3` for an annulus ladder rung,
  `frame25` for a transient frame;
- variant (optional): `steady`, `window1`, `direct`, `mg`, `rep2`, `parent`.

Examples: `blankenbach1a_nu-L5_steady.md`,
`tosi5a_nu-top-max-L3_limit-cycle.md`,
`annulus_nu-t_128x1536_rung3.md`, `realistic_nu-bot_window1.md`,
`gates_suite-total_main.md`, `dump_state-hashes_main.md`.

## Template

```text
# blankenbach1a: Nu at level 5, steady state

- Quantity: Nu = 4.888x (published 4.884409, relative difference +0.0x %)
- Command: `QUAST_JULIA=... QUAST_OUT=... bash scripts/run_one.sh 1a 5 0.2 8`
- Input: analytic roll of examples/benchmark_cases.jl case 1a (no file), SHA-256 n/a
- Resolution: level 5, 4096 leaves, N = ...
- Solver path: linear = auto -> direct; CASE_CFL = 5.0, CASE_WALLMIN = 0
- Wall clock: ... s on 8 threads
- Environment: Julia 1.12.7, Manifest.toml sha256 ..., Ubuntu 24.04, <CPU>, 25 GB RAM
- Log: out_julia/benchmarks/1a_L5.log, sha256 ..., ... bytes (archived in the release)
- Status: verified
- Recorded: 2026-11-2x by commit ...
```

## Index

The index is rebuilt when records are added: one row per record, newest first.
It is empty until the P2 batch fills it.

| Date | Record | Quantity | Status |
|---|---|---|---|
| — | — | — | — |
