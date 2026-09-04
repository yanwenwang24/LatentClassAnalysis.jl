# LatentClassAnalysis.jl

*Latent class analysis for categorical data in Julia.*

Latent class analysis (LCA) is a statistical method for finding unobserved subgroups
in a population from the pattern of answers people give to a set of categorical
questions [lazarsfeld1968](@cite). A survey may ask, for instance, whether someone
married, when, whether they completed university, and what kind of job they held;
LCA looks for a small number of *latent classes* such that, within each class, the
answers to those questions follow a characteristic profile. Each respondent is then
assigned a probability of belonging to each class. Typical applications include
market segmentation, typologies of life-course trajectories, symptom profiles, and
attitude or behaviour patterns [collins2010](@cite).

Statistically, LCA is a finite mixture model for categorical indicators: the
population is modelled as a mixture of ``K`` groups, and within each group the
indicators are assumed to be independent of one another (*local independence*). The
parameters are the size of each class and, for every item, the probability of each
response category within each class. They are estimated by maximum likelihood using
the EM algorithm, and the number of classes is chosen by comparing information
criteria across models. See [Methodology](@ref) for the details.

## Installation

```julia
using Pkg
Pkg.add("LatentClassAnalysis")
```

The package supports Julia 1.10 and later.

## Quick example

The example below simulates 500 respondents from two hidden groups, each answering
five yes/no items, and recovers the groups. Everything the package exports is used
once: [`prepare_data`](@ref) recodes a `DataFrame`, [`LCAModel`](@ref) sets up a model
with random starting values, [`fit!`](@ref) runs the EM algorithm, [`diagnostics!`](@ref)
computes fit statistics, [`show_profiles`](@ref) prints the class profiles, and
[`predict`](@ref) returns class assignments and posterior probabilities.

```@example quick
using LatentClassAnalysis, DataFrames, Random

Random.seed!(1)
n = 500
cls = rand(1:2, n)                                   # two hidden groups
item(p1, p2) = [rand() < (c == 1 ? p1 : p2) ? 1 : 0 for c in cls]
df = DataFrame(item1 = item(0.9, 0.2), item2 = item(0.8, 0.3), item3 = item(0.85, 0.25),
               item4 = item(0.7, 0.2), item5 = item(0.9, 0.3))

data, n_categories = prepare_data(df, :item1, :item2, :item3, :item4, :item5)
model = LCAModel(2, size(data, 2), n_categories)
ll = fit!(model, data)
diag = diagnostics!(model, data, ll)
show_profiles(model, df, [:item1, :item2, :item3, :item4, :item5])
```

The profile shows that one class answers "1" on every item with high probability
and the other rarely does, which is how the data were generated. Finally,
[`predict`](@ref) returns each respondent's most likely class and the posterior
probability of each class:

```@example quick
assignments, probabilities = predict(model, data)
```

Starting values are drawn from Julia's global random number generator, so call
`Random.seed!` before constructing a model if you need a reproducible fit.

## Where to go next

- [Getting started](@ref): a complete walk-through on simulated data, from a
  `DataFrame` with mixed column types to choosing the number of classes and attaching
  class assignments to the data.
- [Methodology](@ref): the model, the EM algorithm, the fit statistics, and
  identifiability, written for social scientists.
- [Example: childlessness in Singapore](@ref): a replication of a published
  latent class analysis using the data bundled with the package.
- [API reference](@ref): docstrings of every exported function and type.
- [Changelog](@ref): what changed in each release.

## Citing

If you use LatentClassAnalysis.jl in published work, please cite the software
(LatentClassAnalysis.jl, Yanwen Wang, <https://github.com/yanwenwang24/LatentClassAnalysis.jl>)
and, if it is relevant to your application, the article for which the package was
originally developed [wang2024](@cite):

```bibtex
@article{wang2024,
  author  = {Wang, Yanwen and Teerawichitchainan, Bussarawan and Ho, Christine},
  title   = {Diverse pathways to permanent childlessness in {Singapore}: A latent class analysis},
  journal = {Advances in Life Course Research},
  volume  = {61},
  pages   = {100628},
  year    = {2024},
  doi     = {10.1016/j.alcr.2024.100628}
}
```

## Related software

Comparable tools in other environments include the R package poLCA
[linzer2011](@cite), Mplus, and Latent GOLD. LatentClassAnalysis.jl currently
implements the basic latent class model without covariates; see
[Limitations of the current version and roadmap](@ref limitations) for what is planned.
