# Scripts

Repository tooling and run post-processing, not solver code and not case
inputs. `list_files.jl` regenerates `docs/dev/FILES.md` from the first header
line of every tracked file, so do not hand-edit that table; the Python renderers
read the `frames/` a case writes and never write into the run directory; the
shell watchdog restarts a long production case that died.

| File | What it is |
|---|---|
| `list_files.jl` | Regenerate `docs/dev/FILES.md`: every file with its one-line header. |
| `live_render.py` | Render new frames to PNG, a panel with Nu/Vrms and a self-refreshing page. |
| `three_row_video.py` | Stack runs of one case into a labelled comparison video. |
| `watchdog_case.sh` | Restart the realistic-case spin-up after a crash, with a log. |
