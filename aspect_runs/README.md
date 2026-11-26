# ASPECT runs

The ASPECT 2.5 side of the cross-code comparison. `realistic/` holds the
production cards, the `realistic composite` material-model plugin and the Python
analysis that turns ASPECT's statistics and VTU output into the numbers in
`docs/en/validation.md`. `blankenbach_templates/` holds the hand-written ASPECT
templates for the Blankenbach 1989 cases. ASPECT and its image are not in this
repository; build the plugin inside the image with `CMakeLists.txt`. Inputs are
maintained by hand; every card marked generated below is written by the script
beside it and must not be edited.

## The pinned image

The ASPECT side is run from the image

```text
gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac
```

which the tag `gassmoeller/aspect:latest` currently resolves to. The image was
created 2023-04-19 and reports **ASPECT 2.5.0-pre (main, 340c0d018), deal.II
9.4.0, Trilinos 13.2.0, p4est 2.2.0**. It ships two builds of that commit,
`aspect` (DEBUG, the default on `PATH`) and `aspect-release` (OPTIMIZED), the
complete source tree at `/home/dealii/aspect`, and prebuilt plugin libraries for
the `blankenbach` and `davies_et_al` benchmark directories. The image digest,
both version banners and the run commands are archived in
[`../benchmarks/results/logs/aspect_environment.txt`](../benchmarks/results/logs/aspect_environment.txt);
it is the digest, not the mutable tag, that identifies the environment.

Docker Desktop's WSL integration is off in this environment, so `docker` is run
from **Windows PowerShell**, not from WSL; inputs and outputs must live on an
ASCII path (the campaign used `C:\temp\quast_aspect`), because the mount path
cannot carry the CJK repository name.

The 2026-11-26 cross-code campaign ran the shipped Blankenbach 2b and Davies et
al. cards inside this image; the provenance records are
[`../benchmarks/results/aspc-blankenbach2b_nu-vrms_ref3.md`](../benchmarks/results/aspc-blankenbach2b_nu-vrms_ref3.md),
[`aspc-davies-case11_nu-vrms_ref5.md`](../benchmarks/results/aspc-davies-case11_nu-vrms_ref5.md)
and (for the free-slip cases) `../benchmarks/results/aspc-davies-*.md`. The
measured cost is the limiting factor: the default `aspect` binary is a DEBUG
build and is about 17x slower per time step than `aspect-release` from the same
image (measured on Blankenbach 2b refinement 3: 1.56 s/step DEBUG against
0.0905 s/step OPTIMIZED, i.e. about 1.8 h against 6 min for the full card). The
completed comparisons therefore use `aspect-release`; every record states which
binary was used and the DEBUG cost measured beside it.

| Card or script | Role |
|---|---|
| `realistic.prm` | Base production card (2.5 x 1, H=10, composite rheology, two phases); the controls include it. |
| `realistic_E.prm` | From-scratch long control for the Julia `case_realistic_E` run. |
| `realistic_E6.prm` | E6: `realistic_E.prm` with the bottom band (y <= 0.05) one level finer. |
| `verify100.prm`, `verify7000.prm` | From-scratch arms to t_nd = 3.3e-5 and 3.2e-4. |
| `test.prm`, `test2.prm`, `prodtest.prm` | Smoke test, fixed-level check and one-step production/MPI test. |
| `control_nolatent.prm`, `control_nophase.prm` | Controls A and B, isolating latent heat and phase buoyancy. |
| `blankenbach_templates/tpl_case*.prm` | Hand-written templates for Blankenbach 1a/1b/1c/2a/2b. |
| `make_ens.py` -> `realistic_ens1..3.prm` | Generated IC-perturbation ensemble members. |
| `make_cont.py`, `make_long.py`, `make_e6_long.py` | Generated continuations: `realistic_E_cont.prm`, `realistic_E_long.prm`, `realistic_E6_long.prm`. |
| `make_noav.py` -> `realistic_noav.prm` | Generated card with artificial viscosity switched off. |
| `realistic_composite.cc`, `CMakeLists.txt` | The ASPECT plugin and its build. |
| `*.py` (not `make_*`) | Analysis: cross-code Nu/Vrms profiles, VTU extraction and cost projections. |

## Licence

The ASPECT plugin is **GPL-2.0-or-later**, not MIT, because it is linked against
ASPECT: `realistic/realistic_composite.cc` and `realistic/CMakeLists.txt` are
derived from ASPECT, which is GPL-2.0-or-later, and are distributed under that
licence with the text in `realistic/LICENSE`. Each of the two files carries the
SPDX identifier `GPL-2.0-or-later` in its header. The `*.prm` input cards, the
`blankenbach_templates/` and the Python analysis scripts are original work and
stay under the repository's MIT licence (see `../LICENSE`). The per-directory
split is also recorded in the top-level `../NOTICE`.
