# Security policy

QuaST is a research solver for two-dimensional mantle convection. The realistic
attack surface is a malformed input file, not a network service: the package
opens no socket, fetches no remote data and runs with the permissions of the user
who starts it. The inputs it parses are an HDF5 checkpoint file (the restart
mode), an ASCII `x y T` initial-condition field (the `examples/data/` format) and
the `.prm` cards under `aspect_runs/` (read by ASPECT, not by this package); the
outputs go to the directory the caller names in `CASE_OUT`.

## Supported versions

The `0.1.x` series is the supported line. The package is a pre-1.0 research
code, so fixes land on `main` and are recorded in `CHANGELOG.md`; there are no
back-ported maintenance branches yet.

## Reporting a vulnerability

There is no separate security mailbox. Report a suspected vulnerability through
the repository:

- open an issue at <https://github.com/edwardaud/QuaST/issues> with `security` in
  the title and a minimal reproduction, or
- use GitHub's private vulnerability-reporting form on the repository's
  **Security** tab if it is enabled, which keeps the report out of the public
  tracker while it is being handled.

Please include the Julia version, the thread count, the exact command, the input
file (or a minimal file that reproduces the problem) and the last lines of the
log. This is a research project with no security team and no response-time
commitment; reports are handled on a best-effort basis, and a fix is credited in
`CHANGELOG.md` unless the reporter asks not to be named.

## Out of scope

Numerical accuracy, convergence, performance and out-of-memory failures from a
deliberately oversized run are not security issues; report them as ordinary bugs
(see `CONTRIBUTING.md`). ASPECT and the `aspect_runs/` plugin are built and run
outside this repository; report ASPECT-side vulnerabilities to the ASPECT
project, and plugin issues through this repository's tracker.
