# StatsAPI accessors, information criteria, entropy, and the diagnostics table.

"""
    nobs(m::LCAModel) -> Int

Number of observations the model was fitted to.
"""
StatsAPI.nobs(m::LCAModel) = size(m.data.y, 1)

"""
    dof(m::LCAModel) -> Int

Number of free parameters: `(K - 1)·P + K·Σ_j (C_j - 1)`, where `P` is the number of
columns of the class-membership design (`1` without covariates).
"""
StatsAPI.dof(m::LCAModel) =
    (m.n_classes - 1) * size(m.beta, 1) + m.n_classes * sum(c - 1 for c in m.n_categories; init=0)

"""
    loglikelihood(m::LCAModel) -> Float64
    loglikelihood(m::LCAModel, d::LCAData) -> Float64

Observed-data log-likelihood of the fitted model on its training data (the stored value)
or on other data `d` with the same items (missing responses are skipped).
"""
StatsAPI.loglikelihood(m::LCAModel) = m.loglik
StatsAPI.loglikelihood(m::LCAModel, d::LCAData) = _posterior_and_ll(m, d)[2]

"""
    isfitted(m::LCAModel) -> Bool

Always `true`: an `LCAModel` is only ever created by [`fit`](@ref).
"""
StatsAPI.isfitted(::LCAModel) = true

"""
    sbic(m::LCAModel) -> Float64

Sample-size adjusted BIC, `-2·loglikelihood + dof·log((nobs + 2) / 24)` (Sclove, 1987).
"""
sbic(m::LCAModel) = -2 * loglikelihood(m) + log((nobs(m) + 2) / 24) * dof(m)

"""
    entropy(m::LCAModel; relative=true) -> Float64

Entropy of the posterior classification. With `relative=true` the relative entropy
`1 - Σ_i Σ_k p_ik log p_ik / (n log K)`, which lies in `[0, 1]` and equals 1 when every
observation is assigned to a class with certainty (defined as 1 for a single class).
With `relative=false` the raw posterior entropy `-Σ_i Σ_k p_ik log p_ik` in nats.
"""
function StatsBase.entropy(m::LCAModel; relative::Bool=true)
    post = m.posterior
    n, K = size(post)
    h = 0.0
    @inbounds for p in post
        p > 0 && (h -= p * log(p))
    end
    relative || return h
    K == 1 && return 1.0
    return 1 - h / (n * log(K))
end

"""
    diagnostics(m::LCAModel) -> ModelDiagnostics
    diagnostics(models::AbstractVector{<:LCAModel}) -> Vector{ModelDiagnostics}

Fit statistics of one or several fitted models: number of classes, observations and free
parameters, log-likelihood, AIC, BIC, sample-size adjusted BIC, relative entropy, and
convergence. A `Vector{ModelDiagnostics}` is a Tables.jl row table, so
`DataFrame(diagnostics(models))` is a model-selection table.

# Returns
- [`ModelDiagnostics`](@ref), or a vector with one entry per model (in the given order)
"""
diagnostics(m::LCAModel) = ModelDiagnostics(m.n_classes, nobs(m), dof(m), loglikelihood(m),
                                            aic(m), bic(m), sbic(m), entropy(m), m.converged)
diagnostics(ms::AbstractVector{<:LCAModel}) = ModelDiagnostics[diagnostics(m) for m in ms]

# Tables.jl row interface for vectors of diagnostics (the element type is ours).
Tables.isrowtable(::Type{<:AbstractVector{ModelDiagnostics}}) = true
Tables.schema(::AbstractVector{ModelDiagnostics}) =
    Tables.Schema(fieldnames(ModelDiagnostics), fieldtypes(ModelDiagnostics))
