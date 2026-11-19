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

The engineering record is a **local** volume set. It records how each part of the
code came to be, which defects were found and how, what each measurement cost, and
where the project stands. It is not part of the published documentation: it lives
in `docs/dev/` on the development machine and is not tracked by this repository.

If you are working on the code and have that directory, the volumes are the ones
the source comments cite by name: `SOLVER.md`, `PHYSICS.md`, `AMR.md`, `MG.md`,
`VALIDATION.md`, `CASE_REALISTIC.md`, `VIDEOS.md`, `BUG_LEDGER.md`, `HANDOFF.md`,
`PERFORMANCE.md`, `FILES.md`. Without it, the user guides above stand on their own:
every number a user needs is in them or in the tests.

## Where to start

- Run a case: [Quick start](en/quickstart.md), then [Examples](en/examples.md).
- Understand the method: [Guide](en/guide.md) sections 1 and 2, then `dev/SOLVER.md` and `dev/PHYSICS.md`.
- Change the code: [Architecture](architecture.md) for the map, `dev/HANDOFF.md` for the runbook, and `dev/BUG_LEDGER.md` before you start debugging.
