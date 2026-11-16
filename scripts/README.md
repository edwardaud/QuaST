# Scripts

Repository tooling: documentation checks, the benchmark campaign, and the frame
renderers. None of it is solver code, and none of it is a case input.

| File | What it is |
|---|---|
| `check_docs.jl` | Documentation checks, run in CI: display-math layout, formulas against Chinese punctuation, escaped underscores inside inline math, language switchers, code-fence languages. |
| `list_files.jl` | Regenerate `docs/dev/FILES.md`: every file with its one-line header. Do not hand-edit that table. |
| `benchmark_table.jl` | Turn the benchmark logs into the rows of the validation table. It checks that a run is steady before quoting it, and labels aborted logs. |
| `run_benchmarks.sh` | Reproduce the validation campaign: every benchmark row, with the settings the published cards use. |
| `run_one.sh` | Run one benchmark case in its own process. The primitive the other launchers call. |
| `run_parallel.sh` | Launch the five Blankenbach cases at one level, one tmux session each. |
| `queue_level.sh` | Start a finer-level run once a coarser one has written its last line. |
| `queue_after_session.sh` | Start a case once an unrelated tmux session has finished, for cases that share a memory budget. |
| `live_render.py` | Render new frames to PNG, a panel with Nu/Vrms, and a self-refreshing page. |
| `three_row_video.py` | Stack runs of one case into a labelled comparison video. |
| `watchdog_case.sh` | Restart the realistic-case spin-up after a crash, with a log. |

## Reproducing the validation campaign

```bash
bash scripts/run_benchmarks.sh 4        # four threads per case
julia --project=. scripts/benchmark_table.jl
```

The first command starts one tmux session per case with the settings the
published benchmark cards use, so the numbers land on the published values
rather than on whatever the driver defaults happen to be. The second reads
`out_julia/benchmarks/*.log` and prints the table rows.

Two settings matter more than the rest, and both come from the cards rather than
from this repository's defaults:

- **`CASE_CFL=5`** for the Blankenbach cases. `base_case2b.prm` sets `CFL number
  = 5.0`; the driver defaults to 0.5, which costs ten times the steps for a
  steady answer that does not depend on the step size at all.
- **`CASE_RA=1000`** for the Tosi cases. What the equations see is
  `Ra/(eta/eta0)`, so `Ra` and `eta_0` are one choice; the pairing that
  reproduces Table 2 is `eta0 = 0.1` with `Ra = 1000`.

The Tosi runs also pass `CASE_STEADY=Inf`, because a Tosi flow evolves on a
timescale far longer than one step and the per-step criterion stops the run long
before the state is steady.

## Memory

A wall-refined case at 4 000 leaves peaks above 15 GB once the direct
factorization fills in, so run those one at a time. `ps -o rss= -p <pid>` is a
better guide than the leaf count.
