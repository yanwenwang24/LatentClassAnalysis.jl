"""
    LCAModel

Represents a Latent Class Analysis model.

# Fields
- `n_classes::Int`: Number of latent classes
- `n_items::Int`: Number of manifest variables
- `n_categories::Vector{Int}`: Number of categories for each manifest variable
- `class_probs::Vector{Float64}`: Class membership probabilities
- `item_probs::Vector{Matrix{Float64}}`: Item response probabilities for each class
"""

mutable struct LCAModel
    n_classes::Int
    n_items::Int
    n_categories::Vector{Int}
    class_probs::Vector{Float64}
    item_probs::Vector{Matrix{Float64}}
    
    function LCAModel(n_classes::Int, n_items::Int, n_categories::Vector{Int})
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

        class_probs = fill(1/n_classes, n_classes)
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

Stores model fit statistics for LCA model.

# Fields
- `ll::Float64`: Log-likelihood
- `aic::Float64`: Akaike Information Criterion
- `bic::Float64`: Bayesian Information Criterion
- `sbic::Float64`: Sample-size adjusted BIC
- `entropy::Float64`: Entropy of class assignments
"""

struct ModelDiagnostics
    ll::Float64
    aic::Float64
    bic::Float64
    sbic::Float64
    entropy::Float64
end
