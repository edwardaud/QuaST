# Contributing

QuaST is a research solver for two-dimensional mantle convection. Patches that make it
faster, more accurate or easier to check are welcome. The habits below are the ones the
project follows; `docs/dev/HANDOFF.md` section 7 records what each one cost to learn.

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

It takes about 25 minutes on eight threads; the engineering record prices the same run at
40 to 75 minutes on a loaded machine. The suite has 184 gate files, and the last recorded
full-suite run reported 133 714 passed, 0 failed and 2 broken.

The suite currently reports no failures; the two `broken` entries are `@test_skip`
calls in G209 and G214 that wait for checkpoints the repository does not ship.
Any failure is a regression.

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
   error 20 to 90 times the baseline (`docs/dev/HANDOFF.md` section 7, rule 7).
4. **A default flip is staged and re-baselined.** Pin the dependents, flip the default, run
   the full suite twice, re-baseline the canonical baselines, then document. The AMR flip
   pinned 28 callers in 20 gates and 8 scripts, added G229 with the flip, and re-baselined
   the 25-frame blob; the linear flip pinned the direct-path numbers in Stage 0 (`42ac0f8`)
   and added G230 (`docs/dev/AMR.md` section 4.4, `docs/dev/HANDOFF.md` section 7).
5. **A commit names its pathspec.** Use `git add -- path` and `git commit -- path`, one
   file or one directory per commit. Shared checkouts carry untracked data, and `git add -A`
   sweeps whatever another process had staged.

## Report a bug

Include:

- the Julia version (`julia --version`) and the thread count (`--threads=N`);
- the gate file, or the example or benchmark script, with its mode or arguments;
- the exact command you ran;
- the last lines of the log, or the testset summary and stack trace.

State any `CASE_*` knob that differs from its default, because the restart guard and most
bitwise pins depend on them. The issue form
`.github/ISSUE_TEMPLATE/bug_report.yml` asks for the same fields.
