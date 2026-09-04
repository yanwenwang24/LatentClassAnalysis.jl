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

# Stopgap for this build: docstrings in src/ cross-reference `coef`, `aic` and `bic` with
# `@ref`, but the package has no docstring for these StatsAPI generics yet (`aic`/`bic`
# use the StatsAPI defaults, `coef` has no method until standard errors land). Documenter
# can only resolve an `@ref` to a documented object, so document the three bindings in the
# package module here. Delete this block once src/ carries their docstrings.
Core.eval(LatentClassAnalysis, quote
    @doc """
        aic(m::LCAModel) -> Float64

    Akaike information criterion, `-2·loglikelihood(m) + 2·dof(m)` (the StatsAPI default).
    """ aic
    @doc """
        bic(m::LCAModel) -> Float64

    Bayesian information criterion, `-2·loglikelihood(m) + dof(m)·log(nobs(m))` (the
    StatsAPI default).
    """ bic
    @doc """
        coef(m::LCAModel) -> Vector{Float64}

    Free parameters of the model on the logit scale, `dof(m)` in total: the
    class-membership block first, then the item-response logits. Coming in the 0.3.0
    release; there is no method in this build.
    """ coef
end)

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
        "Methodology" => "methodology.md",
        "Example: childlessness in Singapore" => "example_childless.md",
        "Upgrading from 0.2" => "migration.md",
        "API reference" => "api.md",
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
