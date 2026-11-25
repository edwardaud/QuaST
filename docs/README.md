# Documentation

This page is the map of the documentation. It lists every document with the question it answers, then gives two reading orders: one for a new user and one for a user who wants to verify a number. The five user pages exist in English in `en/` and in Chinese in `zh/`, and each opens with a language switch, so a row names both versions where both exist. The two top-level README files are separate documents with the same facts and link set, not translations of each other.

## Documents

| Document | The question it answers |
|---|---|
| [README](../README.md) · [README.zh.md](../README.zh.md) | What is QuaST, what does it not do, and what are the three commands to a first result? |
| This page | Which documents exist, and in what order should I read them? |
| [Quick start](en/quickstart.md) · [快速开始](zh/quickstart.md) | How do I install it, run it for the first time, read the files it produces, and recover when something fails? |
| [Guide](en/guide.md) · [用户指南](zh/guide.md) | What are the equations, the discretization, the adaptive policy, the linear solvers and the boundary conditions, and what does every parameter, output format and restart rule mean? |
| [Examples](en/examples.md) · [算例](zh/examples.md) | Which examples ship — the realistic card, the Blankenbach benchmarks, the blob regression, the two regional cases and a programmatic run — with which command and which numbers to check? |
| [Validation](en/validation.md) · [验证](zh/validation.md) | What has been verified, against what, with which measured numbers, and what does the evidence not cover? |
| [Code and data availability](paper/availability.md) | Where are the code, the archived release and its DOI, the data and the per-number benchmark records, and which licence covers each part? |
| [FAQ](en/faq.md) · [常见问题](zh/faq.md) | How do I make a specific change once the quick start is behind me? |
| [Architecture](architecture.md) | How is the code organized into six layers, where do the boundary conditions enter, and where does one time step go? |
| [English pages](en/README.md) · [中文页面](zh/README.md) | Which user pages exist in this language, and what is each one for? |
| Engineering record, `docs/dev/` | Why does a default have the value it has, and which measurement settled it? This is maintainer material and is not tracked by this repository. |

The repository records — `CHANGELOG.md`, `CONTRIBUTING.md`, and the `README.md` files under `src/`, `test/`, `examples/`, `benchmarks/` (including the per-number provenance records under `benchmarks/results/`), `scripts/` and `aspect_runs/` — are reference pages of the repository, not pages of this documentation set.

## Reading order

### If you are new

1. [README](../README.md) — what the solver is, what it does not do, and a first result in three commands.
2. [Quick start](en/quickstart.md) — install, the first run, the files it produces, and what to do when it fails.
3. [Examples](en/examples.md) — take one case at a time; each case gives the command and a number to check.
4. [Guide](en/guide.md) — read the sections you need; the parameter reference is a lookup table, not a chapter.
5. [FAQ](en/faq.md) — when your task is "change X".

### If you want to verify a number

1. [Validation](en/validation.md) — which benchmark, which quantity, which agreement, and how far the evidence reaches.
2. [Examples](en/examples.md) — rerun the shipped case behind the number you care about; each case names the value to check.
3. [Architecture](architecture.md) — if you also want to follow the code path that produced it.
4. The automated suite, `julia --project=. --threads=8 test/runtests.jl`; the recorded result is in [test/README.md](../test/README.md).

**Next:** the [quick start](en/quickstart.md) is the shortest path from a clone to a running case.
