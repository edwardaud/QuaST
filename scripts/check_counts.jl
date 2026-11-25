#!/usr/bin/env julia
# Documentation-count consistency check.  Run from the repository root:
#
#     julia --project=. scripts/check_counts.jl
#
# It is run by hand like `scripts/check_docs.jl` (the task of remediation batch
# P0 part 1 explicitly keeps it out of the gate suite: it reads documentation,
# it does not exercise the solver).
#
# What it checks
# --------------
# 1. **The tree.**  The number of `test/t_g*.jl` files, and that
#    `test/runtests.jl` includes each of them exactly once (the suite's gate
#    count is a property of the tree, so this part is derived, not trusted).
# 2. **The recorded full-suite totals.**  `test/README.md` is the one committed
#    record of a full-suite run.  The script parses its
#    `Recorded full-suite totals: <T> total / <P> passed / <F> failed /
#    <B> broken` line and validates it structurally:
#      * `T == P + F + B` (the Julia summary counts the broken entries inside the
#        total),
#      * T equals the gate-file count times the documented per-file rate?  No:
#        assertion counts are runtime counts (loops), so the total cannot be
#        derived from the tree.  See "limits" below.
# 3. **The other documented surfaces.**  Every public surface that quotes the
#    counts must agree with the record:
#      * `test/README.md`'s gate-file count,
#      * `CONTRIBUTING.md`'s gate-file count and passed/failed/broken numbers.
#    `README.md` (repo root) is checked as a **warning-only** surface: this
#    batch's write scope excludes the top-level markdown (the documentation
#    batch P0 item 1 owns it), so a stale root README is reported but does not
#    fail the script; the warning names the exact numbers it read.
#
# Limits (stated, not hidden)
# ---------------------------
# * The assertion **total** cannot be recomputed from the tree: runtime
#   assertions expand from `@test` sites inside loops (the audit's static parse
#   counts 5 259 macro sites for 137 619 runtime assertions).  The only honest
#   check is "every surface agrees with the committed record", which is what
#   this script does; it cannot detect a record that was itself written from a
#   different tree.  The structural check `T == P + F + B` and the gate-overlap
#   check catch the two failure modes seen in this repository (a stale total
#   from another commit, and an edited count that no longer adds up).
# * `README.md` and any `docs/en` / `docs/zh` surface are outside this batch's
#   write scope; the former is warning-only, the latter are not scanned at all.
# * The record's date/time string is informational; the script does not compare
#   it to the file's mtime.
#
# Exit status: 0 when every required surface agrees (warnings do not fail),
# 1 with a `counts: N problem(s)` summary otherwise.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const TESTDIR = joinpath(ROOT, "test")
const README = joinpath(ROOT, "test", "README.md")
const CONTRIB = joinpath(ROOT, "CONTRIBUTING.md")
const ROOTREADME = joinpath(ROOT, "README.md")

const problems = String[]
const warnings = String[]

bad(msg) = push!(problems, msg)
warn(msg) = push!(warnings, msg)

# ---------------------------------------------------------------------------
# 1. the tree: gate files and the suite's include list
# ---------------------------------------------------------------------------
gate_files = sort(filter(f -> occursin(r"^t_g\d+.*\.jl$", f),
                         readdir(TESTDIR)))
include_text = read(joinpath(TESTDIR, "runtests.jl"), String)
includes = String[]
for m in eachmatch(r"include\(\"(t_g[^\"]+\.jl)\"\)", include_text)
    push!(includes, m.captures[1])
end
for f in gate_files
    n = count(==(f), includes)
    n == 1 || bad("runtests.jl includes $f $n times (expected exactly once)")
end
for f in includes
    f in gate_files ||
        bad("runtests.jl includes $f, which is not a test/t_g*.jl file")
end
length(includes) == length(gate_files) ||
    bad("runtests.jl includes $(length(includes)) gate files, the tree has " *
        "$(length(gate_files))")

# ---------------------------------------------------------------------------
# 2. the recorded full-suite totals in test/README.md
# ---------------------------------------------------------------------------
readme_text = read(README, String)
# "Recorded full-suite totals: **137 619 total / 137 617 passed / 0 failed /
#  2 broken**, ..." -- spaces may be non-breaking; accept either.
record_re = r"Recorded full-suite totals:\s*\**\s*(\d[\d ]*\d|\d)\s+total\s*/\s*" *
            r"(\d[\d ]*\d|\d)\s+passed\s*/\s*(\d[\d ]*\d|\d)\s+failed\s*/\s*" *
            r"(\d[\d ]*\d|\d)\s+broken"
m = match(record_re, readme_text)
record = nothing
if m === nothing
    bad("test/README.md has no machine-readable 'Recorded full-suite totals: " *
        "<T> total / <P> passed / <F> failed / <B> broken' line")
else
    num(s) = parse(Int, replace(s, " " => ""))
    record = (total=num(m.captures[1]), passed=num(m.captures[2]),
              failed=num(m.captures[3]), broken=num(m.captures[4]))
    if record.total != record.passed + record.failed + record.broken
        bad("test/README.md record does not add up: $(record.total) total != " *
            "$(record.passed) passed + $(record.failed) failed + " *
            "$(record.broken) broken")
    end
end

# the record's own gate count: "The tree has <N> gate files and `runtests.jl`"
file_count_re = r"The tree has\s+(\d+)\s+gate files"
mrc = match(file_count_re, readme_text)
if mrc === nothing
    bad("test/README.md states no gate-file count")
else
    n = parse(Int, mrc.captures[1])
    n == length(gate_files) ||
        bad("test/README.md says $n gate files, the tree has $(length(gate_files))")
end

# ---------------------------------------------------------------------------
# 3. the other documented surfaces
# ---------------------------------------------------------------------------
if isfile(CONTRIB)
    ct = read(CONTRIB, String)
    mc = match(r"suite has\s+(\d+)\s+gate files", ct)
    if mc === nothing
        bad("CONTRIBUTING.md states no gate-file count")
    else
        n = parse(Int, mc.captures[1])
        n == length(gate_files) ||
            bad("CONTRIBUTING.md says $n gate files, the tree has " *
                "$(length(gate_files))")
    end
    #
    mt = match(r"reported\s+(\d[\d ]*\d|\d)\s+passed\s*,\s*" *
               r"(\d[\d ]*\d|\d)\s+failed\s+and\s+(\d[\d ]*\d|\d)\s+broken", ct)
    if mt === nothing
        bad("CONTRIBUTING.md states no passed/failed/broken totals")
    elseif record !== nothing
        num(s) = parse(Int, replace(s, " " => ""))
        got = (parse(Int, replace(mt.captures[1], " " => "")),
               parse(Int, replace(mt.captures[2], " " => "")),
               parse(Int, replace(mt.captures[3], " " => "")))
        (got[1], got[2], got[3]) ==
            (record.passed, record.failed, record.broken) ||
            bad("CONTRIBUTING.md totals $(got) disagree with the record " *
                "($(record.passed) passed / $(record.failed) failed / " *
                "$(record.broken) broken)")
    end
else
    bad("CONTRIBUTING.md is missing")
end

# README.md: warning-only (out of this batch's write scope)
if isfile(ROOTREADME)
    rt = read(ROOTREADME, String)
    mr = match(r"gate suite,\s*(\d+)\s*files\s*\|\s*\**\s*(\d[\d ]*\d|\d)\s*" *
               r"assertions:\s*(\d[\d ]*\d|\d)\s*passed\s*,\s*" *
               r"(\d[\d ]*\d|\d)\s*failed\s*,\s*(\d[\d ]*\d|\d)\s*broken", rt)
    if mr === nothing
        warn("README.md quotes no suite counts in the scanned form " *
             "(warning only: the top-level markdown is owned by the " *
             "documentation batch)")
    elseif record !== nothing
        num(s) = parse(Int, replace(s, " " => ""))
        got = (parse(Int, mr.captures[1]), num(mr.captures[2]),
               num(mr.captures[3]), num(mr.captures[4]))
        (got[1], got[2], got[3], got[4]) ==
            (length(gate_files), record.total, record.passed,
             record.failed) ||
            warn("README.md is stale: it says $(got[1]) files / " *
                 "$(got[2]) assertions / $(got[3]) passed / $(got[4]) failed, " *
                 "the record says $(length(gate_files)) / $(record.total) / " *
                 "$(record.passed) / $(record.failed) (warning only: the " *
                 "top-level markdown is corrected by the documentation batch)")
    end
end

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
for w in warnings
    println("counts: warning: ", w)
end
if isempty(problems)
    println("counts: ok (", length(gate_files), " gate files, ",
            record === nothing ? "no recorded totals" :
            "$(record.total) recorded assertions)")
else
    for p in problems
        println("counts: ", p)
    end
    println("counts: ", length(problems), " problem(s)")
    exit(1)
end
