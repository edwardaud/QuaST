<p align="right"><a href="en/README.md">English</a> · <a href="zh/README.md">中文</a></p>

# Documentation

<p align="center">
  <img alt="Julia" src="https://img.shields.io/badge/Julia-1.12-9558B2?logo=julia&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-131%20851%20assertions-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

The user guides live in `en/` and `zh/`, five pages in each language: install and run, the method and its parameters, the examples, the validation evidence, the FAQ. [architecture.md](architecture.md), beside this page, maps the source tree. The engineering record lives in `dev/`: derivations, measured numbers and the defect ledger, written for maintainers. A user page tells you how to do something; a `dev/` volume tells you why a default has the value it has, and which measurement settled it.

## User guides

| Page | Read it when |
|---|---|
| [Quick start](en/quickstart.md) | You have not run the code yet: install, the ten-step smoke test, stopping and resuming, the failures a new user hits |
| [Guide](en/guide.md) | You need the equations, the method, every knob and keyword, the output formats |
| [Examples](en/examples.md) | You want a case to copy: four runnable scripts with measured runtimes and numbers to check |
| [Validation](en/validation.md) | You are about to trust a number: benchmark convergence, the ASPECT comparison, the two Nusselt conventions, the limits |
| [FAQ](en/faq.md) | Your task is "change X": the Rayleigh number, a phase, the rheology, threads, restarting, citation |

The same five pages exist in `zh/`.

## The engineering record

| Volume | What it records |
|---|---|
| [SOLVER.md](dev/SOLVER.md) | The equations, elements, constraints, time step and linear paths the code implements, and where their limits are |
| [PHYSICS.md](dev/PHYSICS.md) | Each physical ingredient as the code assembles it, and what the term was measured to do |
| [AMR.md](dev/AMR.md) | How leaves are marked, balanced and moved, and what one adaptation pass costs |
| [MG.md](dev/MG.md) | How the matrix-free multigrid works, which optimization stages paid off, where the defaults come from |
| [MG_LITERATURE.md](dev/MG_LITERATURE.md) | What mature codes do for each multigrid lever, with sources, and what each claim cost to verify |
| [PERFORMANCE.md](dev/PERFORMANCE.md) | Where a step's time goes, why a solve is direct or multigrid at a given size, the crossover table |
| [VALIDATION.md](dev/VALIDATION.md) | Benchmarks, wall-flux conventions, cross-code comparisons, and the conditions they hold under |
| [CASE_REALISTIC.md](dev/CASE_REALISTIC.md) | The realistic 2.5 x 1 card, how the production runs were launched, and what they produced |
| [VIDEOS.md](dev/VIDEOS.md) | How the three-row comparison clips are rendered, and what they are meant to show |
| [BUG_LEDGER.md](dev/BUG_LEDGER.md) | Which defects were found, how each one surfaced, what caused it, and which gate now pins it |
| [HANDOFF.md](dev/HANDOFF.md) | Where the project stands, what is open, the runbook, and the rules the project learned the hard way |
| [FILES.md](dev/FILES.md) | Every file in the repository, one line each |

[AUDIT.md](dev/AUDIT.md) is a frozen snapshot of the tree at commit 1747345, taken before the earlier structured-grid layer was removed. It records what was there and why it went.

## Where to start

- Run a case: [Quick start](en/quickstart.md), then [Examples](en/examples.md).
- Understand the method: [Guide](en/guide.md) sections 1 and 2, then `dev/SOLVER.md` and `dev/PHYSICS.md`.
- Change the code: [Architecture](architecture.md) for the map, `dev/HANDOFF.md` for the runbook, and `dev/BUG_LEDGER.md` before you start debugging.
