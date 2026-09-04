# Posterior class membership (predict) and hard assignment (classify).

const _MATRIX_PREDICT_REMOVED =
    "predict/classify on a matrix were removed in v0.3: wrap the codes in " *
    "LCAData(y; n_categories=model.n_categories) and call predict(model, d) for posterior " *
    "probabilities or classify(model, d) for class assignments"

"""
    predict(m::LCAModel) -> Matrix{Float64}
    predict(m::LCAModel, d::LCAData) -> Matrix{Float64}
    predict(m::LCAModel, table) -> Matrix{Float64}

Posterior class-membership probabilities: one row per observation, one column per class,
rows summing to one. Without data, the posterior of the training data is returned (a copy
of `m.posterior`).

New data are given as an [`LCAData`](@ref) with the same items (an item may show fewer
categories than the model, never more) or as any Tables.jl table, which is prepared with
[`prepare_data`](@ref) using the training levels of every item (`drop_unused_levels=false`),
so the coding matches the model even when a level is absent from the new table. Missing
responses are skipped; a row with all indicators missing gets the class prior.

A model fitted with covariates needs the same covariates in the new data (an `LCAData`
with the same `covariate_names`, or a table with those columns; a missing column is an
`ArgumentError`): the prior of every row is its covariate-specific membership probability
`softmax(x_i'β)`. A model fitted without covariates ignores any covariates of the new data.

Matrices are deliberately not accepted (the 0.2 method returned a tuple); wrap codes in
`LCAData(y; n_categories=m.n_categories)`.

# Returns
- `Matrix{Float64}` of size `n × n_classes`

See also [`classify`](@ref).
"""
StatsAPI.predict(m::LCAModel) = copy(m.posterior)

function StatsAPI.predict(m::LCAModel, d::LCAData)
    post, _ = _posterior_and_ll(m, d)
    return post
end

StatsAPI.predict(m::LCAModel, table) = predict(m, _prepare_like(m, table))

StatsAPI.predict(::LCAModel, ::AbstractMatrix) = throw(ArgumentError(_MATRIX_PREDICT_REMOVED))

"""
    classify(m::LCAModel) -> Vector{Int}
    classify(m::LCAModel, d::LCAData) -> Vector{Int}
    classify(m::LCAModel, table) -> Vector{Int}

Modal class assignment: the class with the largest posterior probability of every
observation (`argmax` of each row of [`predict`](@ref); ties go to the lower class). The
data arguments are the same as for `predict`.

# Returns
- `Vector{Int}` of length `n` with values in `1:n_classes`
"""
classify(m::LCAModel) = _argmax_rows(m.posterior)
classify(m::LCAModel, d::LCAData) = _argmax_rows(predict(m, d))
classify(m::LCAModel, table) = _argmax_rows(predict(m, table))
classify(::LCAModel, ::AbstractMatrix) = throw(ArgumentError(_MATRIX_PREDICT_REMOVED))

function _argmax_rows(post::AbstractMatrix{<:Real})
    n, K = size(post)
    out = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        best, pbest = 1, post[i, 1]
        for k in 2:K
            if post[i, k] > pbest
                best, pbest = k, post[i, k]
            end
        end
        out[i] = best
    end
    return out
end

# Prepare a table with the coding of the training data (and its covariate columns).
function _prepare_like(m::LCAModel, table)
    lev = Dict{Symbol,Vector{String}}(zip(m.data.item_names, m.data.item_levels))
    covs = hascovariates(m) ? m.data.covariate_names[2:end] : Symbol[]
    if !isempty(covs) && Tables.istable(table)
        available = collect(Tables.columnnames(Tables.columns(table)))
        absent = filter(c -> !(c in available), covs)
        isempty(absent) || throw(ArgumentError(
            "the model was fitted with the covariate(s) $(join(string.(covs), ", ")); " *
            "column(s) $(join(string.(absent), ", ")) are absent from the table"))
    end
    return prepare_data(table, m.data.item_names; covariates=covs, levels=lev,
                        drop_unused_levels=false)
end

function _check_compatible(m::LCAModel, d::LCAData)
    J = size(d.y, 2)
    J == m.n_items || throw(ArgumentError("the data has $J items but the model has $(m.n_items)"))
    for j in 1:J
        d.n_categories[j] <= m.n_categories[j] || throw(ArgumentError(
            "item $(d.item_names[j]) has $(d.n_categories[j]) categories in the data but only $(m.n_categories[j]) in the model"))
    end
    if hascovariates(m)
        d.covariate_names == m.data.covariate_names || throw(ArgumentError(
            "the model was fitted with the covariate(s) $(join(string.(m.data.covariate_names[2:end]), ", ")) " *
            "but the data has " * (hascovariates(d) ? join(string.(d.covariate_names[2:end]), ", ") : "none")))
    end
    return nothing
end

# Posterior (n × K) and log-likelihood of the model on data d. With covariates the E-step
# runs on the raw design of d with the raw-scale coefficients [0 beta].
function _posterior_and_ll(m::LCAModel, d::LCAData)
    _check_compatible(m, d)
    withcov = hascovariates(m)
    ws = LCAWorkspace(d, m.n_classes; aggregate=!withcov, covariates=withcov,
                      n_categories=m.n_categories, standardize=false)
    coefs = withcov ? _raw_coefs(m) : nothing
    θ = LCAParams(m.class_probs, m.item_probs, coefs)
    ll = estep!(ws, θ)
    return _expand_posterior(ws), ll
end
