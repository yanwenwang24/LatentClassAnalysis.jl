# LatentClassAnalysis.jl

[![Build status (Github Actions)](https://github.com/yanwenwang24/LatentClassAnalysis.jl/workflows/CI/badge.svg)](https://github.com/yanwenwang24/LatentClassAnalysis.jl/actions)
[![Coverage](https://codecov.io/gh/yanwenwang24/LatentClassAnalysis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/yanwenwang24/LatentClassAnalysis.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Julia package for Latent Class Analysis (LCA)

## Installation

```julia
using Pkg
Pkg.add("LatentClassAnalysis")
```

## Quick Example

```julia
using LatentClassAnalysis
using DataFrames

# Create sample data (100 observations)
df = DataFrame(
    item1 = repeat([1, 2, 1, 2, 1], 20),  # Binary responses
    item2 = categorical(repeat(["A", "B", "A", "B", "A"], 20)),  # Categorical
    item3 = repeat([1, 1, 2, 2, 1], 20)   # Binary responses
)

# Prepare data
data, n_categories = prepare_data(df, :item1, :item2, :item3)

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

# Select best model by BIC
best_result = argmin(r -> r.diagnostics.bic, results)
best_model = best_result.model

# Get class predictions
assignments, probabilities = predict(best_model, data)
```

## Extended Example

See [example.jl](examples/example.jl) for a comprehensive example.
