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
response category within each class. LatentClassAnalysis.jl estimates them by maximum
likelihood with the EM algorithm run from many random starting values, accepts data from
any Tables.jl source with missing responses in the indicators, lets covariates predict
class membership, reports standard errors from the observed information matrix or the
bootstrap, and offers the information criteria, entropy and bootstrap likelihood-ratio
test used to choose the number of classes. See [Methodology](@ref) for the details.

## Installation

```julia
using Pkg
Pkg.add("LatentClassAnalysis")
```

The package supports Julia 1.10 and later.

## Quick example

The example below simulates 500 respondents from two hidden groups, each answering
five yes/no items, and recovers the groups. [`prepare_data`](@ref) recodes the
indicator columns of a table into an [`LCAData`](@ref); [`fit`](@ref) estimates one
model per requested number of classes, each from 20 random starts;
[`diagnostics`](@ref) collects the fit statistics of every model into a table that
`DataFrame` can display; [`show_profiles`](@ref) prints the class profiles of the
selected model; and [`classify`](@ref) and [`predict`](@ref) return the class
assignments and the posterior membership probabilities.

```@example quick
using LatentClassAnalysis, DataFrames, StableRNGs

# Simulate 500 respondents from two hidden groups with different response tendencies
rng = StableRNG(1)
n = 500
cls = rand(rng, 1:2, n)
item(p1, p2) = [rand(rng) < (c == 1 ? p1 : p2) ? 1 : 0 for c in cls]
df = DataFrame(item1 = item(0.9, 0.2), item2 = item(0.8, 0.3), item3 = item(0.85, 0.25),
               item4 = item(0.7, 0.2), item5 = item(0.9, 0.3))

# 1. Recode the indicators (any Tables.jl table works, not only a DataFrame)
d = prepare_data(df, [:item1, :item2, :item3, :item4, :item5])

# 2. Fit models with one to three classes; each fit uses 20 random starts
models = fit(LCAModel, d, 1:3; rng = StableRNG(1))

# 3. Compare them and pick the number of classes by BIC
DataFrame(diagnostics(models))
```

BIC is smallest for the two-class model, which is how the data were generated (the
three-class fit also prints a warning, not shown here, that some response probabilities
were estimated at exactly 0 or 1, a symptom of asking for more classes than the data
support). The profile of the selected model shows one class that answers `1` on every
item with high probability and one that rarely does:

```@example quick
best = models[argmin(bic.(models))]     # the two-class model
show_profiles(best)                     # class sizes and response probabilities per class
```

Finally, [`classify`](@ref) returns each respondent's most likely class and
[`predict`](@ref) the posterior probability of each class:

```@example quick
df.class = classify(best)               # most likely class of every row
posterior = predict(best)               # 500 × 2 matrix of posterior probabilities
first(df, 5)
```

Fits are reproducible for a given `rng`: the random starts are seeded from it, so the
same generator and the same data always give the same model.

## Where to go next

- [Getting started](@ref): a complete walk-through on simulated data, from a
  `DataFrame` with mixed column types to choosing the number of classes, reading the
  profiles, and attaching class assignments to the data.
- The guide pages, one per topic, each a self-contained tutorial:
  [Model selection](@ref guide-model-selection) (information criteria, random restarts,
  the bootstrap likelihood-ratio test), [Missing data](@ref guide-missing-data),
  [Covariates](@ref guide-covariates) (latent class regression), and
  [Standard errors and the bootstrap](@ref guide-inference).
- [Methodology](@ref): the model, the EM algorithm with random restarts, missing data,
  covariates, the fit statistics and the bootstrap likelihood-ratio test, standard
  errors, and identifiability, written for social scientists.
- [Example: childlessness in Singapore](@ref): a replication of a published
  latent class analysis using the data bundled with the package, with a covariate model
  and a bootstrap likelihood-ratio test.
- [Migrating from 0.2 to 0.3](@ref): every 0.2 call next to its 0.3 replacement, and
  what changed in the results.
- API reference, in two pages: [data, fitting, prediction](@ref api-core) and
  [inference, bootstrap, deprecated](@ref api-inference): docstrings of every exported
  function and type.
- [Changelog](@ref): what changed in each release.

## Citing

If you use LatentClassAnalysis.jl in published work, please cite the software; the
`CITATION.cff` file in the repository carries the metadata, and GitHub's "Cite this
repository" button formats it. Add the version you used:

```bibtex
@software{wang_latentclassanalysis_jl,
  author  = {Wang, Yanwen},
  title   = {{LatentClassAnalysis.jl}: Latent class analysis in {Julia}},
  url     = {https://github.com/yanwenwang24/LatentClassAnalysis.jl},
  version = {0.3.0},
  year    = {2026}
}
```

The package was originally developed for the analysis in [wang2024](@cite), which the
[childlessness example](@ref "Example: childlessness in Singapore") replicates; cite the
article when it is relevant to your application:

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
[linzer2011](@cite), Mplus, and Latent GOLD. LatentClassAnalysis.jl implements the
latent class model with random restarts, missing-data support, covariates on class
membership, standard errors from the observed information matrix or the bootstrap, and
the bootstrap likelihood-ratio test for the number of classes (see
[Bootstrap likelihood-ratio test](@ref blrt)).
