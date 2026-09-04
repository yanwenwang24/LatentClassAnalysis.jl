# LatentClassAnalysis.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yanwenwang24.github.io/LatentClassAnalysis.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/)
[![CI](https://github.com/yanwenwang24/LatentClassAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/yanwenwang24/LatentClassAnalysis.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/yanwenwang24/LatentClassAnalysis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/yanwenwang24/LatentClassAnalysis.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Latent class analysis (LCA) in Julia.

LCA identifies unobserved subgroups in a population from patterns of categorical
responses. Typical uses are behavioral profiles, market segments, and life-course
pathways. LatentClassAnalysis.jl fits latent class models by the EM algorithm and reports
the fit indices needed to choose the number of classes. It was written for, and
replicates, the analysis in Wang, Teerawichitchainan and Ho (2024), *Diverse Pathways to
Permanent Childlessness in Singapore: A Latent Class Analysis*, Advances in Life Course
Research 61:100628, [doi:10.1016/j.alcr.2024.100628](https://doi.org/10.1016/j.alcr.2024.100628).

## Features

- Input from any [Tables.jl](https://github.com/JuliaData/Tables.jl) source (a
  `DataFrame`, a `NamedTuple` of vectors, an Arrow table, ...); binary, integer-coded,
  string, and categorical indicators are recoded automatically, in the level order of a
  `CategoricalArray` or an order you supply
- Maximum likelihood estimation by EM with random restarts (20 short runs, the best 4
  continued to convergence), a numerically stable E-step, and an `rng` keyword for
  reproducible fits
- Missing responses in the indicators, handled in the E-step under the missing-at-random
  assumption
- The StatsAPI verbs you expect from a fitted model: `fit`, `nobs`, `dof`,
  `loglikelihood`, `aic`, `bic`, `aicc`, `predict`, plus `sbic`, `entropy`, and
  `classify` for modal class assignments
- A model-selection table: `fit(LCAModel, d, 1:5)` fits several class counts and
  `DataFrame(diagnostics(models))` tabulates log-likelihood, AIC, BIC, sBIC, and entropy
- Class profiles as a printed report (`show_profiles`) or as a table (`profiles`)

- Covariates for class membership (latent class regression) and standard errors and
  confidence intervals from the observed information matrix (`coeftable`, and the
  `se`/`lower`/`upper` columns of `profiles`)

- Simulation from a fitted model (`simulate`), bootstrap standard errors and percentile
  intervals (`bootstrap`), and the bootstrap likelihood-ratio test for the number of
  classes (`bootstrap_lrt`)

## Installation

```julia
using Pkg
Pkg.add("LatentClassAnalysis")
```

## Quick example

```julia
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
DataFrame(diagnostics(models))          # n_classes, nobs, dof, ll, aic, bic, sbic, entropy, converged
best = models[argmin(bic.(models))]     # the two-class model

# 4. Class profiles and class membership
show_profiles(best)                     # class sizes and response probabilities per class
df.class = classify(best)               # most likely class of every row
posterior = predict(best)               # 500 × 2 matrix of posterior probabilities
```

The three-class fit prints a warning that some response probabilities were estimated at
exactly 0 or 1, which is one of the symptoms of asking for more classes than the data
support. Fits are reproducible for a given `rng`; the
[tutorial](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/tutorial/) walks
through the full workflow, including missing data and how to read the profiles.

## Upgrading from 0.2

Version 0.3 replaced `LCAModel(k, n_items, n_categories)` + `fit!` with
`fit(LCAModel, data, k)`, `diagnostics!` with `diagnostics`, and the tuple returned by
`predict` with `predict` (posterior) and `classify` (assignments). The
[migration guide](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/migration/)
lists every old call next to its replacement and explains what changed in the results.

## Documentation

- [Getting started](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/tutorial/)
- [Methodology](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/methodology/):
  the model, EM with restarts, missing data, fit indices, identifiability
- [Example: childlessness in Singapore](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/example_childless/):
  replication of the 2024 paper with the bundled data
- [Upgrading from 0.2](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/migration/)
- API reference: [data, fitting, prediction](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/api/core/)
  and [inference, bootstrap, deprecated](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/api/inference/)

The scripts in [`examples/`](examples/) run the same workflows outside the docs:

```
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/example.jl
julia --project=examples examples/example_childless.jl
```

## Citing

If you use this package, please cite the paper it was developed for and the software
(see [CITATION.cff](CITATION.cff)):

```bibtex
@article{wang2024childlessness,
  title   = {Diverse Pathways to Permanent Childlessness in Singapore: A Latent Class Analysis},
  author  = {Wang, Yanwen and Teerawichitchainan, Bussarawan and Ho, Christine},
  journal = {Advances in Life Course Research},
  volume  = {61},
  pages   = {100628},
  year    = {2024},
  doi     = {10.1016/j.alcr.2024.100628}
}
```

## Related software

R users may know [poLCA](https://cran.r-project.org/package=poLCA); commercial
alternatives are Mplus and Latent GOLD. This package covers the core LCA workflow of those
tools in pure Julia.

## License

MIT, see [LICENSE](LICENSE).
