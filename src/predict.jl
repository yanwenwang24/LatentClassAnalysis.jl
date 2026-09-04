"""
    predict(model::LCAModel, data::AbstractMatrix{<:Integer})

Predict class memberships for (new) data.

# Arguments
- `model::LCAModel`: Fitted model
- `data::AbstractMatrix{<:Integer}`: Data matrix with 1-based codes, prepared in the same
  way as the data used to fit the model

# Returns
- `Vector{Int}`: Most likely class of each observation
- `Matrix{Float64}`: Posterior class membership probabilities, one row per observation
"""
function predict(
    model::LCAModel, data::AbstractMatrix{<:Integer}
)
    n_obs = size(data, 1)
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

    assignments = [argmax(posterior[i, :]) for i in 1:n_obs]
    return assignments, posterior
end
