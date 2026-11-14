#!/usr/bin/env julia
# Documentation checks that catch what a reader would otherwise see as literal
# dollar signs or a broken table.  Run from the repository root:
#
#     julia --project=. scripts/check_docs.jl
#
# It checks the layout rules GitHub applies before it hands a formula to KaTeX:
# a display block needs `$$` alone on its line with a blank line before it, no
# blank line inside, no LaTeX environment, and no `$$` inside a text line.  It
# also checks that an inline formula never touches a Chinese character, that code
# fences are balanced and carry a language, and that the language switcher is the
# first line of every user page.
#
# KaTeX syntax itself is not checked here; run the pages through KaTeX if you
# change a formula (see docs/README.md).

const ROOT = normpath(joinpath(@__DIR__, ".."))
const CJK = r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]"

problems = String[]
function bad(rel, line, msg)
    push!(problems, string(rel, ":", line, "  ", msg))
end

function check(path)
    rel = relpath(path, ROOT)
    lines = readlines(path)
    in_fence = false
    in_block = false
    block_start = 0
    prev_blank = true
    for (i, line) in enumerate(lines)
        if startswith(strip(line), "```")
            if !in_fence
                isempty(strip(line)[4:end]) && bad(rel, i, "code fence without a language")
            end
            in_fence = !in_fence
            prev_blank = false
            continue
        end
        if in_fence
            prev_blank = false
            continue
        end
        if !in_block && strip(line) == "\$\$"
            !prev_blank && bad(rel, i, "display block after a non-blank line")
            in_block = true
            block_start = i
            prev_blank = false
            continue
        end
        if in_block
            if strip(line) == "\$\$"
                in_block = false
            elseif isempty(strip(line))
                bad(rel, block_start, "blank line inside a display block")
            elseif occursin("\\begin{", line)
                bad(rel, block_start, "LaTeX environment inside a display block")
            elseif occursin(r"^\s*([-+*>#|]|\d+[.)])\s"m, line) ||
                   occursin(r"^\s*[=-]{1,}\s*$"m, line)
                bad(rel, block_start, "line inside a display block is Markdown block syntax")
            end
            prev_blank = false
            continue
        end
        occursin("\$\$", line) && bad(rel, i, "\$\$ inside a text line")

        # inline math must not touch a Chinese character; inline code is not math
        bare = replace(line, r"`[^`]*`" => "")
        for m in eachmatch(r"(?<!\$)\$([^$\n]+)\$(?!\$)", bare)
            s = m.offset
            e = s + length(m.match)
            before = s > 1 ? bare[prevind(bare, s):prevind(bare, s)] : ""
            after = e <= lastindex(bare) ? string(bare[e]) : ""
            occursin(CJK, before) && bad(rel, i, "inline math touches a Chinese character on the left")
            occursin(CJK, after) && bad(rel, i, "inline math touches a Chinese character on the right")
        end
        prev_blank = isempty(strip(line))
    end
    in_block && bad(rel, block_start, "display block never closed")
    in_fence && bad(rel, length(lines), "code fence never closed")
end

for (root, dirs, files) in walkdir(ROOT)
    ("out_julia" in dirs) && filter!(d -> d != "out_julia", dirs)
    (".git" in dirs) && filter!(d -> d != ".git", dirs)
    for f in files
        endswith(f, ".md") && check(joinpath(root, f))
    end
end

for sub in ("docs/en", "docs/zh")
    for f in sort(readdir(joinpath(ROOT, sub)))
        endswith(f, ".md") || continue
        first = readline(joinpath(ROOT, sub, f))
        (occursin("English", first) && occursin("中文", first)) ||
            bad(joinpath(sub, f), 1, "user page does not open with the language switch")
    end
end

if isempty(problems)
    println("docs: ok")
else
    foreach(println, problems)
    println("\n", length(problems), " problem(s)")
    exit(1)
end
