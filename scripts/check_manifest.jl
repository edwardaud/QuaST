# Verify the archived benchmark logs against `benchmarks/results/logs/MANIFEST.txt`.
#
# The provenance records quote a byte size and a SHA-256 for every archived log,
# so a record can be checked years later without trusting the working tree.  This
# script is the mechanical check: it reads the manifest, normalises the line
# endings exactly as the manifest convention does (`\r\n` and lone `\r` to `\n`,
# so a checkout's line-ending setting cannot change a hash), and reports any
# entry whose file is missing, whose size differs, or whose hash differs.
#
# Run it by hand like `scripts/check_docs.jl`:
#
#     julia --project=. scripts/check_manifest.jl
#
# Exit code 0 when every entry matches; 1 otherwise.  It is not part of the gate
# suite (it validates data, not code).
using SHA
import Pkg

const ROOT = normpath(joinpath(@__DIR__, ".."))
const MANIFEST = joinpath(ROOT, "benchmarks", "results", "logs", "MANIFEST.txt")

isdir(joinpath(ROOT, "benchmarks")) ||
    error("run from a QuaST checkout; manifest not found at $MANIFEST")
isfile(MANIFEST) || error("missing $MANIFEST")

function normalised_bytes(path::AbstractString)
    raw = read(path)
    return replace(String(raw), "\r\n" => "\n", '\r' => '\n')
end

function check_manifest(manifest::AbstractString, root::AbstractString)
bad = 0
checked = 0
for (i, line) in enumerate(eachline(manifest))
    isempty(strip(line)) && continue
    parts = split(strip(line))
    if length(parts) != 3 || !startswith(parts[1], "benchmarks/results/")
        println("manifest line $i is not a `path size sha256` entry: $(strip(line))")
        bad += 1
        continue
    end
    rel, size_s, hash_s = parts
    path = joinpath(root, rel)
    checked += 1
    if !isfile(path)
        println("missing: $(rel)")
        bad += 1
        continue
    end
    bytes = normalised_bytes(path)
    size = sizeof(bytes)
    hash = bytes2hex(sha256(bytes))
    if string(size) != size_s || hash != hash_s
        println("mismatch: $(rel) (manifest $(size_s) B $(hash_s), file $(size) B $(hash))")
        bad += 1
    end
end

if bad == 0
    println("manifest: ok ($(checked) entries)")
else
    println("manifest: $(bad) of $(checked) entries do not match")
    exit(1)
end
end

check_manifest(MANIFEST, ROOT)
