using Documenter
using DocumenterDiagrams
using QARBoM

pages = [
    "Home" => "index.md",
    "Manual" => [
        "Getting Started" => "manual/getting_started.md",
    ],
    "API Reference" => "api.md",
]

# Set up to run docstrings with jldoctest
DocMeta.setdocmeta!(QARBoM, :DocTestSetup, :(using QARBoM); recursive = true)

makedocs(;
    modules = [QARBoM],
    doctest = false,
    clean = true,
    format = Documenter.HTML(;
        sidebar_sitename = false,
        assets = ["assets/favicon.ico"],
        mathengine = Documenter.MathJax2(),
        prettyurls = false,
        edit_link = nothing,
        footer = nothing,
        disable_git = true,
        # Disabling the size thresholds is not a good practice but 
        # it is necessary in the current state of the documentation

        # Setting it to nothing will write every example block
        example_size_threshold = nothing,
        # Setting it to nothing will ignore the size threshold
        size_threshold = nothing,
    ),
    sitename = "QARBoM.jl",
    authors = "Pedro Ripper, João Neto and Pedro Xavier",
    warnonly = false,
    pages = pages,
    remotes = nothing,
)

if "--skip-deploy" ∈ ARGS
    @warn "Skipping deployment"
else
    deploydocs(; repo = raw"github.com/NITeQ/QARBoM.jl.git", push_preview = true)
end
