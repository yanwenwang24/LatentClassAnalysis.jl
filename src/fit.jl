"""
    fit!(model::LCAModel, data::Matrix{Int}; 
         max_iter::Int=1000, tol::Float64=1e-6, verbose::Bool=false)

Fit the LCA model using EM algorithm.

# Arguments
- `model::LCAModel`: Model to fit
- `data::Matrix{Int}`: Prepared data matrix
- `max_iter::Int=1000`: Maximum number of iterations
- `tol::Float64=1e-6`: Convergence tolerance
- `verbose::Bool=false`: Whether to print progress

# Returns
- `Float64`: Final log-likelihood
"""

function fit!(
    model::LCAModel, data::Matrix{Int};
    max_iter::Int=10000, tol::Float64=1e-6, verbose::Bool=false
)
    # Validate data dimensions
    n_obs, n_items = size(data)
    if n_items != model.n_items
        throw(ArgumentError("Number of items in data ($n_items) doesn't match model ($(model.n_items))"))
    end

    # Validate data values
    for j in 1:n_items
        valid_range = 1:model.n_categories[j]
        if !all(x -> x in valid_range, view(data, :, j))
            min_val, max_val = extrema(view(data, :, j))
            throw(ArgumentError(
                "Invalid category in column $j. Expected values in $valid_range, " *
                "but got values in $min_val:$max_val. Data should be 1-based."
            ))
        end
    end

    n_obs = size(data, 1)
    old_ll = -Inf

    for iter in 1:max_iter
        # E-step: Calculate posterior probabilities
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

        # M-step: Update parameters
        # Update class probabilities
        model.class_probs .= vec(mean(posterior, dims=1))

        # Update item probabilities
        for j in 1:model.n_items
            for k in 1:model.n_classes
                for c in 1:model.n_categories[j]
                    numerator = sum(posterior[data[:, j].==c, k])
                    denominator = sum(posterior[:, k])
                    model.item_probs[j][k, c] = numerator / denominator
                end
            end
        end

        # Calculate log-likelihood
        ll = 0.0
        for i in 1:n_obs
            probs = zeros(model.n_classes)
            for k in 1:model.n_classes
                prob = log(model.class_probs[k])
                for j in 1:model.n_items
                    prob += log(model.item_probs[j][k, data[i, j]])
                end
                probs[k] = exp(prob)
            end
            ll += log(sum(probs))
        end

        # Check convergence
        if abs(ll - old_ll) < tol
            verbose && println("Converged after $iter iterations")
            return ll
        end

        old_ll = ll
        verbose && println("Iteration $iter: log-likelihood = $ll")
    end

    verbose && println("Maximum iterations reached")
    return old_ll

    diagnostics = diagnostics!(model, data, old_ll)
    return diagnostics
end