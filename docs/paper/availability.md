# Code and data availability

This page is the repository-side **code and data availability statement** for a
manuscript that uses QuaST. It carries one ready-to-paste statement per journal
policy — Nature Portfolio, Royal Society, GMD (Copernicus) and AGU/JGR — because
the four word their requirements differently. The facts are shared; only the
heading and the level of detail change.

## Status: the DOI is a placeholder

**No archived release exists yet and no DOI has been minted.** The Zenodo DOI is
written as `<ZENODO-DOI>` below and it must be replaced with a real, resolvable
DOI before a manuscript is submitted. The same replacement is needed in
`CITATION.cff` (the `identifiers` block is present but commented out) and in the
README badge if one is added. The exact steps that mint the DOI and close the
placeholder are in [Minting the release DOI](#minting-the-release-doi) at the end
of this page.

| Placeholder | Replace with | Where |
|---|---|---|
| `<ZENODO-DOI>` | version DOI of the `v0.1.0` release, e.g. `10.5281/zenodo.12345678` | every statement below; `CITATION.cff` |
| `<CONCEPT-DOI>` | Zenodo concept DOI, the one that resolves to the newest release | the "all versions" sentence |
| `<RELEASE-COMMIT>` | the commit the annotated tag `v0.1.0` points at | statements that quote a commit |

Repository state at the time of writing: <https://github.com/edwardaud/QuaST>,
commit `f37312698bac800e444d46674b48608858058436` (2026-11-25), package version
`0.1.0`, Julia `1.12.7` (the `julia_version` line of the committed
`Manifest.toml`), 8 threads, 25 GB of shared memory.

## Shared facts behind the statements

- **Where the code lives.** Development and all history:
  <https://github.com/edwardaud/QuaST>. The archived, DOI-bearing version is the
  release described in [Release and archive](#release-and-archive).
- **Archived release.** `v0.1.0` is tagged `v0.1.0` and archived on Zenodo under
  `<ZENODO-DOI>`; `<CONCEPT-DOI>` always resolves to the newest release. GitHub
  alone is a development site and is not the archival record (GMD policy).
- **Licences.** The main package (Julia source, tests, documentation, scripts,
  benchmark instruments) is MIT. The ASPECT material-model plugin
  (`aspect_runs/realistic/realistic_composite.cc` and its `CMakeLists.txt`) is
  **GPL-2.0-or-later**, because it is derived from and linked against ASPECT,
  which is GPL-2.0-or-later. The example initial-condition data in
  `examples/data/` and the benchmark provenance records in `benchmarks/results/`
  are **CC0-1.0**. The full split is in [`NOTICE`](../../NOTICE), the MIT text in
  [`LICENSE`](../../LICENSE), the GPL text in
  [`aspect_runs/realistic/LICENSE`](../../aspect_runs/realistic/LICENSE) and the
  CC0 text in [`examples/data/LICENSE`](../../examples/data/LICENSE).
- **Environment.** Ubuntu 24.04 (WSL2), Julia 1.12.7, 8 Julia threads, 25 GB of
  shared memory, HDF5.jl 0.17.4; the committed `Manifest.toml` pins the Julia
  package set. The production 10 000-leaf state peaks at about 1.3 GB resident on
  the matrix-free multigrid path and about 2.4 GB on the sparse-direct path. No
  container image is shipped: the environment is the Julia version, the manifest
  and the thread count.
- **Where the benchmark records are.** `benchmarks/results/` holds the
  provenance record for each number quoted in a paper or in
  `docs/en/validation.md`: the command, the input card, the resolution, the wall
  clock, the environment and the log file, under the size and naming convention
  in [`benchmarks/results/README.md`](../../benchmarks/results/README.md). The
  same directory is part of the archived release; the campaign entry points are
  `scripts/run_benchmarks.sh`, `examples/benchmark_cases.jl` and
  `examples/annulus_benchmark.jl`. The scripts take the repository root, the
  Julia executable and the output root from `QUAST_ROOT`, `QUAST_JULIA` and
  `QUAST_OUT`, so a clone at any path reproduces them without edits.

## Nature Portfolio

Used as the separate **Code availability** statement that follows the data
availability statement and precedes the references. Nature Portfolio also
requires the code to be deposited in a DOI-minting repository and cited in the
reference list; both are done by the statement.

```text
Code availability. QuaST v0.1.0 (commit <RELEASE-COMMIT>), an open-source
spectral-element solver for two-dimensional mantle convection, is available at
https://github.com/edwardaud/QuaST and the exact version used for all results
in this paper is archived in Zenodo under DOI <ZENODO-DOI>. The main package is
released under the MIT licence; the ASPECT material-model plugin in
aspect_runs/realistic/ is released under GPL-2.0-or-later because it is derived
from ASPECT; the example initial-condition data and the benchmark provenance
records are released under CC0-1.0. No registration, request or restriction is
involved. The code was run with Julia 1.12.7, 8 threads and 25 GB of shared
memory, with the environment pinned by the committed Manifest.toml. Every number
reported in this paper has a provenance record (the command, the input card, the
resolution, the wall clock, the environment and the raw log) under
benchmarks/results/ in the repository and in the archived release. The archive is
cited in the reference list as ref. <N>.
```

## Royal Society

Used as the **Data accessibility** statement. The Society requires the
repository and the DOI in the reference list, an open licence, and a read-me
that describes the files and the data columns; the statement points at the pages
that carry the read-me.

```text
Data accessibility. The code and the data supporting this article are publicly
available at https://github.com/edwardaud/QuaST. The exact version used for all
results is QuaST v0.1.0 (commit <RELEASE-COMMIT>), archived under DOI <ZENODO-DOI>
(<CONCEPT-DOI> resolves to the newest release), and this DOI is given in the
reference list. The code is released under the MIT licence, except the ASPECT
material-model plugin in aspect_runs/realistic/, which is GPL-2.0-or-later
because it is derived from ASPECT; the example initial-condition data in
examples/data/ and the benchmark provenance records are released under CC0-1.0.
The read-me material is the repository README files and docs/en/guide.md, which
describe every output file and every column of history.csv; all reported
quantities are nondimensional and the nondimensionalisation is stated in the
guide. The calculations used Julia 1.12.7 with 8 threads and 25 GB of shared
memory under Ubuntu 24.04, with the environment pinned by the committed
Manifest.toml. For each number reported in the article, benchmarks/results/
records the command, the input card, the resolution, the wall clock, the
environment and the log file. The code and data are available without
restriction from the date of submission; nothing is available only on request.
```

## GMD (Geoscientific Model Development)

Used as the **Code and data availability** section that GMD requires at the end
of the paper, before the Acknowledgements. Copernicus requires a persistent
archive of the precise version and says GitHub is not an archive; the statement
therefore names Zenodo as the archive and GitHub as the development version, and
it covers source, manual, configuration, input data and every run-control script.

```text
Code and data availability. The exact version of QuaST used to produce the
results in this paper is v0.1.0 (commit <RELEASE-COMMIT>), archived on Zenodo under DOI
<ZENODO-DOI>; <CONCEPT-DOI> resolves to the newest release. The model is
developed at https://github.com/edwardaud/QuaST. The licence of the code is the
MIT licence, except the ASPECT material-model plugin in aspect_runs/realistic/,
which is GPL-2.0-or-later as a derivative work of ASPECT; the input data in
examples/data/ and the benchmark provenance records are released under CC0-1.0.
The archive contains the complete model source, the documentation (docs/en and
docs/zh), the configuration and boundary-condition cards, the input data, and
the run-control and post-processing scripts for every result in the paper:
scripts/run_benchmarks.sh launches the benchmark campaign,
examples/benchmark_cases.jl is the per-case entry point,
examples/annulus_benchmark.jl runs the annulus benchmark, and
benchmarks/results/ gives, for each reported number, the command, the input
card, the resolution, the wall clock, the environment and the log file. The
scripts take the repository root, the Julia executable and the output root from
the environment variables QUAST_ROOT, QUAST_JULIA and QUAST_OUT, so the campaign
runs from a clone at any path; see scripts/README.md. The calculations used
Julia 1.12.7 with 8 threads and 25 GB of shared memory under Ubuntu 24.04, with
the environment pinned by the committed Manifest.toml. The archive is cited in
the reference list as ref. <N>.
```

## AGU / JGR

Used in the **Open Research** section, which AGU splits into a Data
Availability Statement and a Software Availability Statement, each backed by a
reference-list citation with a DOI.

```text
Data Availability Statement. The initial-condition data and the benchmark
provenance records behind the figures of this paper are available under
CC0-1.0 in the archived release of QuaST v0.1.0, Zenodo DOI <ZENODO-DOI>
(<CONCEPT-DOI> resolves to the newest release), and in the repository
https://github.com/edwardaud/QuaST; the archived DOI is given in the reference
list. The tracked input data are in examples/data/ and the per-number records
(command, input card, resolution, wall clock, environment and log file) are in
benchmarks/results/. No data are available only on request.

Software Availability Statement. QuaST v0.1.0 (commit <RELEASE-COMMIT>), the
spectral-element solver used for all simulations, is available at
https://github.com/edwardaud/QuaST and is archived in Zenodo under DOI
<ZENODO-DOI>, cited in the reference list. The main package is MIT-licensed; the
ASPECT material-model plugin in aspect_runs/realistic/ is GPL-2.0-or-later
because it is derived from ASPECT. The simulations were run with Julia 1.12.7,
8 threads and 25 GB of shared memory, with the environment pinned by the
committed Manifest.toml. Per-number provenance records are in benchmarks/results/
of the repository and of the archived release. The software is available without
restriction from the date of submission.
```

## Release and archive

The archived release is the annotated tag `v0.1.0` on the release commit. Its
contents are the frozen public tree; the release notes are curated from
`CHANGELOG.md`. The internal engineering record is deliberately not part of the
public release, so nothing in the statements above depends on it.

`CITATION.cff` carries the same facts in machine-readable form: version
`0.1.0`, the release date, the author alias and, once minted, the DOI in the
`identifiers` block. The DOI is not minted by this repository; it is minted by
Zenodo when the GitHub release is published (see below).

## Minting the release DOI

The DOI is minted once, by Zenodo, from the published GitHub release. Order
matters: the tag must exist before the release, and the release must exist
before the DOI. The owner keeps a private release checklist; the short form is:

1. Tag the release commit: `git tag -a v0.1.0 -m "QuaST v0.1.0"`, then
   `git push origin main` and `git push origin v0.1.0`.
2. In Zenodo, sign in with GitHub and enable the `edwardaud/QuaST` repository
   under Settings, GitHub, so new releases are archived automatically.
3. On GitHub, create the release `QuaST v0.1.0` from the tag, with the curated
   release notes, and publish it.
4. Zenodo ingests the release within a few minutes and mints a version DOI
   `10.5281/zenodo.<N>` plus a concept DOI `10.5281/zenodo.<M>`.
5. Replace `<ZENODO-DOI>`, `<CONCEPT-DOI>` and `<RELEASE-COMMIT>` here, uncomment
   the `identifiers` block in `CITATION.cff` and put the DOI in it, then commit
   that metadata change as a follow-up and push it. Optionally add the Zenodo
   badge to the README.
6. Archive the benchmark logs under `benchmarks/results/` (or as a second Zenodo
   deposit with its own DOI if they exceed the size limit) and cite that DOI here
   as the data DOI.

## How to cite

Software:

```text
edwardaud (2026). QuaST: a quadtree spectral-element solver for two-dimensional
mantle convection (v0.1.0). Zenodo. https://doi.org/<ZENODO-DOI>
```

Data (initial conditions and benchmark records):

```text
edwardaud (2026). QuaST v0.1.0 example data and benchmark provenance records
(CC0-1.0). Zenodo. https://doi.org/<ZENODO-DOI>
```

The scientific benchmarks keep their own citations: Blankenbach et al. (1989)
for the steady cases, Tosi et al. (2015) for the temperature-dependent-viscosity
cases, Davies et al. (2022) for the 2-D annulus, and the ASPECT documentation
for the cross-code comparison. The repository's `CHANGELOG.md` records the
release history and `CONTRIBUTING.md` the versioning policy.
