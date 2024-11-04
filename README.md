# LatentClassAnalysis.jl

[![Build status (Github Actions)](https://github.com/yanwenwang24/LatentClassAnalysis.jl/workflows/CI/badge.svg)](https://github.com/yanwenwang24/LatentClassAnalysis.jl/actions)
[![Coverage](https://codecov.io/gh/yanwenwang24/LatentClassAnalysis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/yanwenwang24/LatentClassAnalysis.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Julia package for Latent Class Analysis (LCA)

Latent Class Analysis (LCA) is a statistical method for identifying unobserved subgroups within a population based on categorical response patterns. Common applications include market segmentation, behavioral pattern identification, and psychological profiling.

This package brings LCA to the Julia ecosystem, with key features such as:

- Supports binary, string, and categorical variables
- Fit LCA models via EM algorithm
- Model diagnostics (e.g., AIC, BIC, SBIC, entropy)
- Show latent class profiles and class assignment probabilities

## Installation

```julia
using Pkg
Pkg.add("LatentClassAnalysis")
```

## Quick Example

Let's create a dataset with three response variables.
`LatentClassAnalysis` is designed to handle dummy and categorical variables.
String variables will be automatically converted to categorical ones.

```julia
using LatentClassAnalysis
using DataFrames

# Create sample data (100 observations)
df = DataFrame(
    item1 = repeat([0, 1, 0, 1, 1], 20),  # Binary responses with 0/1 coding
    item2 = categorical(repeat(["A", "B", "A", "B", "A"], 20)),  # Categorical
    item3 = repeat([1, 1, 2, 2, 1], 20)   # Binary responses with 1/2 coding
)
```

Step 1: prepare data.

```julia
# Prepare data
data, n_categories = prepare_data(df, :item1, :item2, :item3)
```

Step 2: fit models with different number of classes.

```julia
# Fit models with different numbers of classes
results = []
for k in 2:3
    model = LCAModel(k, size(data, 2), n_categories)
    ll = fit!(model, data)
    diag = diagnostics!(model, data, ll)
    push!(results, (n_classes=k, model=model, diagnostics=diag))
    
    println("Model with $k classes:")
    println("  BIC: $(round(diag.bic, digits=2))")
    println("  Entropy: $(round(diag.entropy, digits=3))")
end
```

Step 3: select best model and show profile.

```julia
# Select best model by BIC
best_result = argmin(r -> r.diagnostics.bic, results)
best_model = best_result.model

# Show profile
show_profiles(best_model, df, [:item1, :item2, :item3])

# Get class predictions
assignments, probabilities = predict(best_model, data)
```

## Extended Example

See [example.jl](examples/example.jl) for an example with simulated data.
See [example_childless.jl](examples/example_childless_jl) for an example with real-world data on childlessness, 
replicating this research:
- Wang, Yanwen, Bussarawan Teerawichitchainan, and Christine Ho. 2024. “Diverse Pathways to Permanent Childlessness in Singapore: A Latent Class Analysis.” Advances in Life Course Research 61:100628. doi: 10.1016/j.alcr.2024.100628.

