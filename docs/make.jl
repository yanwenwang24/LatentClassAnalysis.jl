using Documenter, DocumenterCitations, LatentClassAnalysis

# The changelog page is generated from the repository CHANGELOG.md so that it is
# maintained in one place. docs/src/changelog.md is git-ignored.
let src = joinpath(@__DIR__, "..", "CHANGELOG.md"),
    dst = joinpath(@__DIR__, "src", "changelog.md")

    if isfile(src)
        cp(src, dst; force=true)
    else
        write(dst, "# Changelog\n\nSee the repository CHANGELOG.md.\n")
    end
end

DocMeta.setdocmeta!(
    LatentClassAnalysis,
    :DocTestSetup,
    :(using LatentClassAnalysis, DataFrames, CategoricalArrays);
    recursive=true,
)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

makedocs(;
    sitename="LatentClassAnalysis.jl",
    modules=[LatentClassAnalysis],
    authors="Yanwen Wang",
    repo=Remotes.GitHub("yanwenwang24", "LatentClassAnalysis.jl"),
    format=Documenter.HTML(;
        canonical="https://yanwenwang24.github.io/LatentClassAnalysis.jl",
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "tutorial.md",
        "Guide" => [
            "Model selection" => "guide/model_selection.md",
            "Missing data" => "guide/missing_data.md",
            "Covariates" => "guide/covariates.md",
            "Standard errors and the bootstrap" => "guide/inference.md",
        ],
        "Methodology" => "methodology.md",
        "Example: childlessness in Singapore" => "example_childless.md",
        "Migrating from 0.2 to 0.3" => "migration.md",
        "API reference" => ["api/core.md", "api/inference.md"],
        "Changelog" => "changelog.md",
    ],
    plugins=[bib],
    doctest=true,
    checkdocs=:exports,
    warnonly=false,
)

deploydocs(;
    repo="github.com/yanwenwang24/LatentClassAnalysis.jl.git",
    devbranch="main",
    push_preview=true,
)
