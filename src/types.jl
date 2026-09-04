"""
    LCAModel(n_classes, n_items, n_categories)

A latent class model for categorical manifest variables.

# Fields
- `n_classes::Int`: Number of latent classes
- `n_items::Int`: Number of manifest variables (items)
- `n_categories::Vector{Int}`: Number of categories of each manifest variable
- `class_probs::Vector{Float64}`: Class membership probabilities (length `n_classes`)
- `item_probs::Vector{Matrix{Float64}}`: Item response probabilities. `item_probs[j]` is an
  `n_classes × n_categories[j]` matrix whose row `k` holds the probability of each response
  category of item `j` given membership in class `k`; every row sums to one.

# Constructor
`LCAModel(n_classes, n_items, n_categories)` validates its arguments (at least two classes,
at least one item, at least two categories per item, `length(n_categories) == n_items`),
warns when the model may not be identifiable, and initializes `class_probs` uniformly and
`item_probs` with random draws from the global random number generator. Call
`Random.seed!` before constructing a model for reproducible fits.

# Example
```jldoctest
julia> using LatentClassAnalysis

julia> model = LCAModel(2, 3, [2, 2, 3]);

julia> model.n_classes, model.n_items, model.n_categories
(2, 3, [2, 2, 3])

julia> model.class_probs
2-element Vector{Float64}:
 0.5
 0.5

julia> size(model.item_probs[3])
(2, 3)
```
"""
mutable struct LCAModel
    n_classes::Int
    n_items::Int
    n_categories::Vector{Int}
    class_probs::Vector{Float64}
    item_probs::Vector{Matrix{Float64}}

    function LCAModel(n_classes::Integer, n_items::Integer, n_categories::AbstractVector{<:Integer})
        # Validate number of classes, items, and categories
        if n_classes < 2
            throw(ArgumentError("Number of classes must be ≥ 2, got $n_classes"))
        end
        if n_items < 1
            throw(ArgumentError("Number of items must be ≥ 1, got $n_items"))
        end
        if length(n_categories) != n_items
            throw(ArgumentError("Length of n_categories ($(length(n_categories))) must match n_items ($n_items)"))
        end
        for (i, cats) in enumerate(n_categories)
            if cats < 2
                throw(ArgumentError("Each item must have ≥ 2 categories, item $i has $cats"))
            end
        end
        # Check identifiability
        check_identifiability(n_items, n_classes, n_categories)

        class_probs = fill(1 / n_classes, n_classes)
        item_probs = [rand(n_classes, cats) for cats in n_categories]
        # Normalize probabilities
        for probs in item_probs
            probs ./= sum(probs, dims=2)
        end

        new(n_classes, n_items, n_categories, class_probs, item_probs)
    end
end

"""
    ModelDiagnostics

Model fit statistics of a fitted [`LCAModel`](@ref), as returned by [`diagnostics!`](@ref).

# Fields
- `ll::Float64`: Log-likelihood
- `aic::Float64`: Akaike Information Criterion, `-2ll + 2p`
- `bic::Float64`: Bayesian Information Criterion, `-2ll + p·log(n)`
- `sbic::Float64`: Sample-size adjusted BIC, `-2ll + p·log((n + 2)/24)`
- `entropy::Float64`: Relative entropy of the class assignments, between 0 and 1, where 1
  means every observation is assigned to a class with certainty

Here `p` is the number of free parameters and `n` the number of observations.
"""
struct ModelDiagnostics
    ll::Float64
    aic::Float64
    bic::Float64
    sbic::Float64
    entropy::Float64
end
