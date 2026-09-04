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

- Binary, integer-coded, string, and categorical indicators; any coding (`0/1`, `1/2`,
  `"yes"/"no"`, ...) is recoded automatically
- Maximum likelihood estimation via the EM algorithm
- AIC, BIC, sample-size adjusted BIC, and relative entropy for choosing the number of classes
- Class profiles and posterior class membership probabilities

Planned for version 0.3 (see [CHANGELOG.md](CHANGELOG.md)): random restarts, missing
indicator values, covariates for class membership, standard errors and confidence
intervals, a bootstrap likelihood-ratio test, and input from any Tables.jl source.

## Installation

```julia
using Pkg
Pkg.add("LatentClassAnalysis")
```

## Quick example

```julia
using LatentClassAnalysis, DataFrames, Random

# Simulate 500 respondents from two hidden groups with different response tendencies
Random.seed!(1)
n = 500
cls = rand(1:2, n)
item(p1, p2) = [rand() < (c == 1 ? p1 : p2) ? 1 : 0 for c in cls]
df = DataFrame(item1 = item(0.9, 0.2), item2 = item(0.8, 0.3), item3 = item(0.85, 0.25),
               item4 = item(0.7, 0.2), item5 = item(0.9, 0.3))

# 1. Recode the indicators to 1-based integer codes
data, n_categories = prepare_data(df, :item1, :item2, :item3, :item4, :item5)

# 2. Fit a two-class model (LCAModel draws its starting values from the global RNG)
model = LCAModel(2, size(data, 2), n_categories)
ll = fit!(model, data)

# 3. Fit indices, class profiles, and class membership
diag = diagnostics!(model, data, ll)          # diag.aic, diag.bic, diag.sbic, diag.entropy
show_profiles(model, df, [:item1, :item2, :item3, :item4, :item5])
assignments, probabilities = predict(model, data)
```

To choose the number of classes, fit models with `k = 2, 3, ...` and compare `diag.bic`;
the [tutorial](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/tutorial/) shows
the full workflow. Because EM can stop at a local maximum, fit each model from several
seeds and keep the one with the highest log-likelihood.

## Documentation

- [Getting started](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/tutorial/)
- [Methodology](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/methodology/):
  the model, EM, fit indices, identifiability
- [Example: childlessness in Singapore](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/example_childless/):
  replication of the 2024 paper with the bundled data
- [API reference](https://yanwenwang24.github.io/LatentClassAnalysis.jl/dev/api/)

The scripts in [`examples/`](examples/) run the same workflows outside the docs:

```
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
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
