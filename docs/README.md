<p align="right"><a href="en/quickstart.md">English</a> · <a href="zh/quickstart.md">中文</a></p>

# Documentation

The user guides live in `en/` and `zh/`, the same five pages in English and
Chinese: install and run, the method and its parameters, the examples, the
validation evidence, the FAQ. [architecture.md](architecture.md), beside this
page, maps the source tree. The engineering record lives in `dev/`: derivations,
measured numbers and the defect ledger, written for maintainers. A user page
answers how to do something; a `dev/` volume answers why a default has the value
it has and which measurement settled it.

## User guides

| Page | Read it when |
|---|---|
| [Quick start](en/quickstart.md) | You have not run the code yet: install, the ten-step smoke, stopping and resuming, the failures a new user hits |
| [Guide](en/guide.md) | You need the physics, the method, every case variable and keyword, the output formats |
| [Examples](en/examples.md) | You want a case to copy: four runnable scripts with runtimes and expected accuracy |
| [Validation](en/validation.md) | You are about to trust a number: benchmarks, the ASPECT comparison, the two Nusselt conventions, the limits |
| [FAQ](en/faq.md) | Your task is "change X": the Rayleigh number, a phase, the rheology, threads, restart, citation |

The five pages have the same names in `zh/`.

## The engineering record

| Volume | What it records |
|---|---|
| [SOLVER.md](dev/SOLVER.md) | The equations, elements, constraints, time step and linear paths the code implements, and where its limits are |
| [PHYSICS.md](dev/PHYSICS.md) | Each physical ingredient as the code assembles it, and what the term was measured to do |
| [AMR.md](dev/AMR.md) | How leaves are marked, balanced and moved, and what an adaptation pass costs |
| [MG.md](dev/MG.md) | How the matrix-free multigrid works, which optimisation stages paid off, where the defaults come from |
| [MG_LITERATURE.md](dev/MG_LITERATURE.md) | What mature codes do for each multigrid lever, with sources, and what each claim cost to verify |
| [PERFORMANCE.md](dev/PERFORMANCE.md) | Where a step's time goes, why a solve is direct or multigrid at a given size, the crossover table |
| [VALIDATION.md](dev/VALIDATION.md) | Benchmarks, wall-flux conventions and cross-code comparisons, and the conditions they hold under |
| [CASE_REALISTIC.md](dev/CASE_REALISTIC.md) | The realistic 2.5 x 1 card, how the production runs were launched, what they produced |
| [VIDEOS.md](dev/VIDEOS.md) | How the three-row comparison clips are rendered, and what they are meant to show |
| [BUG_LEDGER.md](dev/BUG_LEDGER.md) | Which defects were found, how each appeared, what caused it, which gate now pins it |
| [HANDOFF.md](dev/HANDOFF.md) | Where the project stands, what is open, the runbook, the rules the project learned the hard way |
| [FILES.md](dev/FILES.md) | Every file in the repository with the one line that says what it is |

[AUDIT.md](dev/AUDIT.md) is a frozen snapshot of the tree at 1747345, before the
route-A layer was removed; it records what was there and why it went.

## Where to start

- Run a case: [Quick start](en/quickstart.md), then [Examples](en/examples.md).
- Understand the method: [Guide](en/guide.md), sections 1 and 2, then `dev/SOLVER.md` and `dev/PHYSICS.md`.
- Change the code: [Architecture](architecture.md) for the map, `dev/HANDOFF.md` for the runbook, `dev/BUG_LEDGER.md` before debugging.
