#!/usr/bin/env julia
# Regenerate docs/dev/FILES.md: every file in the repository with the one line
# that says what it is. Run from the repository root:
#
#     julia --project=. scripts/list_files.jl

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(ROOT, "docs", "dev", "FILES.md")

const DIRS = ["src", "test", "examples", "benchmarks", "docs", "scripts",
              "aspect_runs", "data", ".github"]
const ROOTFILES = ["README.md", "LICENSE", "CITATION.cff", "CONTRIBUTING.md",
                   "CHANGELOG.md", "Project.toml", "Manifest.toml", ".gitignore"]

first_line(path) = try
    for l in Iterators.take(eachline(path), 14)
        t = strip(l)
        if startswith(t, "#") && !startswith(t, "#!")
            d = strip(t, '#')
            length(d) > 6 && return first(d, 150)
        end
    end
    m = match(r"\"\"\"\s*\n(.+)", read(path, String))
    m === nothing ? "source file" : first(strip(m.captures[1]), 150)
catch
    "unreadable"
end

files = String[]
for d in DIRS
    p = joinpath(ROOT, d)
    isdir(p) || continue
    for (root, _, fs) in walkdir(p)
        for f in fs
            rel = relpath(joinpath(root, f), ROOT)
            (occursin("out_julia", rel) || endswith(rel, ".cov")) && continue
            push!(files, replace(rel, '\\' => '/'))
        end
    end
end
for f in ROOTFILES
    isfile(joinpath(ROOT, f)) && push!(files, f)
end
sort!(files)

isdatapath(rel) = any(e -> endswith(rel, e), (".h5", ".txt", ".prm"))

open(OUT, "w") do io
    println(io, "# Files in this repository\n")
    println(io, length(files), " files at this revision, each with the one line that says what it is.")
    println(io, "Regenerate with `julia --project=. scripts/list_files.jl`.\n")
    println(io, "| File | What it is |")
    println(io, "|---|---|")
    for rel in files
        p = joinpath(ROOT, rel)
        desc = isdatapath(rel) ? string(filesize(p), " bytes of data") :
               replace(first_line(p), '|' => '/')
        println(io, "| `", rel, "` | ", desc, " |")
    end
end
println("wrote ", OUT, " with ", length(files), " files")
