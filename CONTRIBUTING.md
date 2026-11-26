# Contributing

QuaST is a research solver for two-dimensional mantle convection. Patches that make it
faster, more accurate or easier to check are welcome. The habits below are the ones the
project follows; the private maintainer record kept what each one cost to learn.

## Set up a development environment

Julia 1.12 is what the realistic card is run with. `Project.toml` carries
`julia = "1.10"`, so a later 1.10, 1.11 or 1.12 release resolves.

```bash
git clone git@github.com:edwardaud/QuaST.git
cd QuaST
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using QuaST'
```

`Pkg.instantiate()` installs the versions pinned in `Manifest.toml` and downloads the HDF5
artifacts. On a clean depot it takes five to ten minutes and under 2 GB. The second command
precompiles the package. Run every command below from the repository root.

## Run the gates

One gate, in its own Julia process:

```bash
julia --project=. test/run_gates.jl t_g229_default_balanced.jl
```

`run_gates.jl` accepts several gate files and includes them in one process, and it exits
non-zero when a gate fails. A gate is usually seconds to a couple of minutes of work plus
Julia startup; G229 measured 1 min 40 s wall on this checkout.

CI does not run on every push. `.github/workflows/gates.yml` is manual dispatch only: the
suite costs about 22 minutes of runner time, so it is started when a change warrants it,
from the Actions tab or with `gh workflow run gates`.

The full suite:

```bash
julia --project=. test/runtests.jl
```

It takes 30 m 54 s on eight threads; the engineering record prices the same run at
40 to 75 minutes on a loaded machine. The suite has 212 gate files, and the last
recorded full-suite run reported 146 099 passed, 0 failed and 2 broken
(2026-11-26, the recorded totals live in `test/README.md`).

The suite currently reports no failures; the two `broken` entries are explicit,
documented `@test_skip` environment gates in G209 and G214 that wait for
data-dependent realistic-state checkpoints the repository does not ship (the
decision and the alternatives are recorded in the commit that made the change).
Any failure is a regression.

## Check the documentation counts

```bash
julia --project=. scripts/check_docs.jl
julia --project=. scripts/check_counts.jl
```

`check_docs.jl` validates the Markdown layout (display-math blocks, code fences,
language switchers) and prints `docs: ok`. `check_counts.jl` (run by hand, not
part of the gate suite) checks the gate-file count quoted by `test/README.md`
and this file against the tree and `test/runtests.jl`, the structural sanity of
the committed full-suite totals in `test/README.md`, and every other documented
surface against that record; the top-level `README.md` is a warning-only surface
because the documentation batch owns it. Its limits (the assertion total is a
runtime count and cannot be derived from the tree, so only agreement with the
committed record is checked) are stated in the script header.

## House rules

1. **Timing claims are same-session A/B.** Absolute seconds move 5 to 12 percent between
   runs, so a speed claim needs both arms in one session on one machine, and the number to
   report is the ratio. Build the baseline arm with `git archive HEAD`, copy
   `Manifest.toml` into it, and run the same script and checkpoint against both projects.
2. **An inert change is verified bitwise.** A change that leaves the arithmetic alone must
   reproduce the pins bit for bit: FNV-1a `d22fdab099ae84e0` for the 20-step 2b state and
   `45ac8acd099adadc` for the blob frame-19 state, over `theta`, `ux`, `uz` and `pv`.
   Compare raw IEEE bits (`reinterpret(UInt64, x)`), because `==` accepts `-0.0` for `0.0`.
3. **A preconditioner change is gated on accuracy against the direct solve.** Compare
   `|du|/|u|` and the gauge-removed `|dp|/|p|` against
   `linear=:direct, stokes_scaling=:equilibrated` in the same process. A converged residual
   does not certify the change: a deflated `S1` reached `relres` 7.2e-7 with a pressure
   error 20 to 90 times the baseline.
4. **A default flip is staged and re-baselined.** Pin the dependents, flip the default, run
   the full suite twice, re-baseline the canonical baselines, then document. The AMR flip
   pinned 28 callers in 20 gates and 8 scripts, added G229 with the flip, and re-baselined
   the 25-frame blob; the linear flip pinned the direct-path numbers in Stage 0 (`42ac0f8`)
   and added G230.
5. **A commit names its pathspec.** Use `git add -- path` and `git commit -- path`, one
   file or one directory per commit. Shared checkouts carry untracked data, and `git add -A`
   sweeps whatever another process had staged.

## Scripts and paths

The launchers in `scripts/`, and `examples/verify100_case.sh`, take the
repository root, the Julia executable and the run-output root from the
environment instead of a hard-coded machine path. All three are optional:

- `QUAST_ROOT` — the checkout to run from; default: the repository the script
  lives in;
- `QUAST_JULIA` — the Julia executable; default: `julia` from `PATH`;
- `QUAST_OUT` — where run outputs land; default: `$QUAST_ROOT/out_julia`.

`scripts/watchdog_case.sh` also reads `QUAST_PY` (Python with `h5py`; default
`python3`) and `scripts/three_row_video.py` reads `QUAST_FFMPEG` (default
`ffmpeg` from `PATH`). `scripts/README.md` has the table. Nothing else in a
tracked script depends on where the checkout is:

```bash
QUAST_JULIA=$HOME/opt/julia-1.12.7/bin/julia bash scripts/run_one.sh 1a 3 0.2 4
```

## Versioning and deprecation

The project follows semantic versioning and is in the 0.x series: a minor
release may change the public surface — the exported names of `QuaST` and the
`CASE_*` knobs of the driver — and a change that moves a bitwise-pinned
trajectory is a minor release even when it is a fix. `CHANGELOG.md` is the
record: every user-visible change of the next release is written there before
the release, under `Added`, `Changed`, `Deprecated`, `Removed` or `Fixed`, with
the replacement named for anything removed.

From v0.1.0 onward the deprecation pattern is: keep the old name working and
note it under `Deprecated` with the replacement for at least one minor release
before removing it; a change that cannot be kept working (a changed default
that a checkpoint guard compares, for example) is recorded under `Changed` and
is called out in the release notes. The gate suite pins the current behaviour,
so a planned break comes with the gate re-baselines in the same commit.

## Report a bug

Include:

- the Julia version (`julia --version`) and the thread count (`--threads=N`);
- the gate file, or the example or benchmark script, with its mode or arguments;
- the exact command you ran;
- the last lines of the log, or the testset summary and stack trace.

State any `CASE_*` knob that differs from its default, because the restart guard and most
bitwise pins depend on them. The issue form
`.github/ISSUE_TEMPLATE/bug_report.yml` asks for the same fields.

The archived benchmark logs under `benchmarks/results/logs/` are covered by a manifest of sizes and SHA-256 values; after touching them (or a checkout with different line endings), run `julia --project=. scripts/check_manifest.jl` and refresh the manifest entry rather than leaving a stale hash.
