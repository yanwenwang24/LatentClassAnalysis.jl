"""
    prepare_data(df::DataFrame, cols::Symbol...)

Prepare DataFrame for LCA by converting categorical columns to dummy variables if needed.

# Arguments
- `df::DataFrame`: Input DataFrame
- `cols::Symbol...`: Column names to use for analysis

# Returns
- `Matrix{Int}`: Prepared data matrix
- `Vector{Int}`: Number of categories for each variable
"""

function prepare_data(df::DataFrame, cols::Symbol...)
    data = Matrix{Int}(undef, nrow(df), length(cols))
    n_categories = Int[]

    for (i, col) in enumerate(cols)
        if eltype(df[!, col]) <: Number
            data[:, i] = df[!, col]
            push!(n_categories, length(unique(df[!, col])))
        else
            # Convert categorical to integers
            categories = sort(unique(df[!, col]))
            data[:, i] = indexin(df[!, col], categories)
            push!(n_categories, length(categories))
        end
    end
    
    return data, n_categories
end

"""
    diagnostics!(model::LCAModel, data::Matrix{Int}, ll::Float64)

Calculate model fit statistics including AIC, BIC, SBIC, and entropy.

# Arguments
- `model::LCAModel`: Fitted model
- `data::Matrix{Int}`: Data matrix
- `ll::Float64`: Log-likelihood from model fitting

# Returns
- `ModelDiagnostics`: Structure containing fit statistics
"""

function diagnostics!(model::LCAModel, data::Matrix{Int}, ll::Float64)
    n_obs = size(data, 1)
    
    # Calculate number of parameters
    # Class probabilities (K-1) + Item probabilities for each class and item
    n_params = (model.n_classes - 1) +
               sum(cats -> (model.n_classes * (cats - 1)), model.n_categories)
    
    # Calculate AIC and BIC
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + log(n_obs) * n_params
    
    # Calculate sample-size adjusted BIC
    n_star = (n_obs + 2) / 24
    sbic = -2 * ll + log(n_star) * n_params
    
    # Calculate entropy
    posterior = zeros(n_obs, model.n_classes)
    for i in 1:n_obs
        for k in 1:model.n_classes
            prob = log(model.class_probs[k])
            for j in 1:model.n_items
                prob += log(model.item_probs[j][k, data[i, j]])
            end
            posterior[i, k] = exp(prob)
        end
        posterior[i, :] ./= sum(posterior[i, :])
    end
    
    entropy = 0.0
    for i in 1:n_obs
        for k in 1:model.n_classes
            p = posterior[i, k]
            entropy -= p * log(p + eps()) # eps() to avoid log(0)
        end
    end
    entropy = 1 - (entropy / (n_obs * log(model.n_classes)))
    
    return ModelDiagnostics(ll, aic, bic, sbic, entropy)
end
