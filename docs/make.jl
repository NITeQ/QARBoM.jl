using Documenter
using DocumenterDiagrams
using QARBoM

# Set up to run docstrings with jldoctest
DocMeta.setdocmeta!(QARBoM, :DocTestSetup, :(using QARBoM); recursive = true)

makedocs(;
    modules  = [QARBoM],
    doctest  = true,
    clean    = true,
    sitename = "QARBoM.jl",
    authors  = "Pedro Ripper",
    workdir  = @__DIR__,
    warnonly = [:missing_docs],
    pages    = [
        "Home" => "index.md",
        "Manual" => [
            "Introduction" => "manual/1-intro.md",
        ],
        "API Reference" => "api.md",
    ],
    format   = Documenter.HTML(
        assets           = ["assets/extra_styles.css", "assets/favicon.ico"],
        mathengine       = Documenter.KaTeX(),
        sidebar_sitename = false,
    ),
)

if "--skip-deploy" ∈ ARGS
    @warn "Skipping deployment"
else
    deploydocs(repo = raw"github.com/NITeQ/QARBoM.jl.git", push_preview = true)
end
