# Documentation

This page is the map of the documentation. It lists every document with the question it answers, then gives two reading orders: one for a new user and one for a user who wants to verify a number. The five user pages exist in English in `en/` and in Chinese in `zh/`, and each of them opens with a language switch, so each row of the table below names both versions. The two top-level README files are separate documents, not translations of each other.

## Documents

| Document | The question it answers |
|---|---|
| [README](../README.md) · [README.zh.md](../README.zh.md) | What is QuaST, what does it not do, and what are the three commands to a first result? |
| This page | Which documents exist, and in what order should I read them? |
| [Quick start](en/quickstart.md) · [快速开始](zh/quickstart.md) | How do I install it, run it for the first time, read the files it produces, and recover when something fails? |
| [Guide](en/guide.md) · [用户指南](zh/guide.md) | What are the equations, the discretization, the adaptive policy and the linear solvers, and what does every parameter, output format and restart rule mean? |
| [Examples](en/examples.md) · [算例](zh/examples.md) | Which examples ship, what command runs each one, and what numbers should it print? |
| [Validation](en/validation.md) · [验证](zh/validation.md) | What has been verified, against what, with which measured numbers, and what does the evidence not cover? |
| [FAQ](en/faq.md) · [常见问题](zh/faq.md) | How do I make a specific change once the quick start is behind me? |
| [Architecture](architecture.md) | How is the code organized into five layers, and where does one time step go? |
| [English pages](en/README.md) · [中文页面](zh/README.md) | Which user pages exist in this language, and what is each one for? |
| Engineering record, `docs/dev/` | Why does a default have the value it has, and which measurement settled it? This is maintainer material and is not tracked by this repository. |

The repository records — `CHANGELOG.md`, `CONTRIBUTING.md`, and the `README.md` files under `src/`, `test/`, `examples/`, `benchmarks/`, `scripts/` and `aspect_runs/` — are records and reference pages of the repository, not pages of this documentation set.

## Reading order

### If you are new

1. [README](../README.md) — what the solver is, what it does not do, and a first result in three commands.
2. [Quick start](en/quickstart.md) — install, the first run, the files it produces, and what to do when it fails.
3. [Examples](en/examples.md) — take one case at a time; each page gives the command and a number to check.
4. [Guide](en/guide.md) — read the sections you need; the parameter reference is a lookup table, not a chapter.
5. [FAQ](en/faq.md) — when your task is "change X".

### If you want to verify a number

1. [Validation](en/validation.md) — which benchmark, which quantity, which agreement, and how far the evidence reaches.
2. [Examples](en/examples.md) — rerun the shipped case behind the number you care about; each page names the value to check.
3. [Architecture](architecture.md) — if you also want to follow the code path that produced it.
4. The automated suite: `julia --project=. test/runtests.jl`; the recorded full-suite result is in [CONTRIBUTING.md](../CONTRIBUTING.md).

**Next:** [Quick start](en/quickstart.md) is the shortest path from a clone to a running case.
