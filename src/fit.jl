# StatsAPI.fit methods, keyword → LCAOptions, identifiability warning, post-fit flags,
# assembly of the immutable LCAModel, and the erroring fit!.

"""
    fit(LCAModel, d::LCAData, k::Integer; kwargs...) -> LCAModel
    fit(LCAModel, d::LCAData, ks::AbstractVector{<:Integer}; kwargs...) -> Vector{LCAModel}
    fit(LCAModel, table, items, k; covariates=Symbol[], levels=nothing,
        drop_unused_levels=true, kwargs...)

Fit a latent class model with `k` classes to `d` by maximum likelihood: EM with random
restarts (`n_starts` short runs of `short_iters` iterations, the `n_final` best of which
are continued to convergence), followed by ordering the classes by decreasing size. The
third form calls [`prepare_data`](@ref) on any Tables.jl table first; the second fits one
model per element of `ks` (for model selection with [`diagnostics`](@ref)).

`k == 1` is allowed and solved in closed form. Missing responses (code `0`) are handled by
the E-step under the missing-at-random assumption: the class sizes use every row and the
response probabilities of an item use the rows where it is observed.

# Arguments
- `d::LCAData`: prepared data, see [`LCAData`](@ref) and [`prepare_data`](@ref)
- `k::Integer`: number of latent classes (`≥ 1`)

# Keyword arguments
- `rng::AbstractRNG=Random.default_rng()`: source of the seeds of the random starts; pass
  a seeded generator for a reproducible fit
- `covariates::Bool=hascovariates(d)`: fit the class-membership model on the covariates of
  `d` (latent class regression). `covariates=false` fits the unconditional model on the
  same data, for nested comparisons. Not available in this version.
- `init=nothing`: starting values used as the first start(s): an [`LCAModel`](@ref) with
  the same `k` and categories, an internal `LCAParams`, a `NamedTuple` with fields
  `class_probs` and `item_probs`, or a vector of those. If more starts are supplied than
  `n_starts`, all of them are run.
- `n_starts::Integer=20`: number of starts
- `n_final::Integer=4`: number of best short runs continued to convergence (capped at
  `n_starts`)
- `short_iters::Integer=50`: EM iterations of every short run
- `max_iter::Integer=10_000`: maximum EM iterations of every final run
- `tol::Real=1e-10`: relative convergence tolerance, `|ll - ll_old| ≤ tol·(1 + |ll|)`
- `se::Symbol=:hessian`: standard errors, `:hessian` or `:none`; stored in `options`. No
  covariance matrix is computed in this version (`vcov` is `nothing`).
- `aggregate::Bool=true`: collapse identical response patterns before running EM (exact;
  disabled automatically with covariates)
- `multithreaded::Bool=false`: run the starts on all Julia threads (results are identical
  to the serial run)
- `verbose::Bool=false`: print a line per start

# Returns
- [`LCAModel`](@ref). A single aggregated warning reports any raised [`FitFlags`](@ref)
  (non-convergence, boundary probabilities, empty classes, a best log-likelihood found by
  a single start). A warning is also issued when the model is not identified by the
  necessary condition `(k - 1) + k·Σ_j (C_j - 1) ≤ ∏_j C_j - 1`.

# Example
```julia
using LatentClassAnalysis, StableRNGs
d = prepare_data(table, [:x1, :x2, :x3, :x4])
m = fit(LCAModel, d, 3; rng=StableRNG(1))
models = fit(LCAModel, d, 1:4; rng=StableRNG(1))
diagnostics(models)
```
"""
function StatsAPI.fit(::Type{LCAModel}, d::LCAData, k::Integer;
                      rng::AbstractRNG=Random.default_rng(),
                      covariates::Bool=hascovariates(d),
                      init=nothing,
                      n_starts::Integer=20, n_final::Integer=4, short_iters::Integer=50,
                      max_iter::Integer=10_000, tol::Real=1e-10, se::Symbol=:hessian,
                      aggregate::Bool=true, multithreaded::Bool=false, verbose::Bool=false)
    K = Int(k)
    K >= 1 || throw(ArgumentError("the number of classes must be at least 1, got $K"))
    nobs(d) >= 1 || throw(ArgumentError("the data has no observations"))
    if covariates
        hascovariates(d) || throw(ArgumentError(
            "covariates=true requires data with covariates; pass them to prepare_data or LCAData"))
        throw(ErrorException(_COVARIATES_NOT_IMPLEMENTED))
    end
    opts = LCAOptions(; n_starts=Int(n_starts), n_final=min(Int(n_final), Int(n_starts)),
                      short_iters=Int(short_iters), max_iter=Int(max_iter), tol=Float64(tol),
                      se=se, aggregate=aggregate, verbose=verbose)
    check_identifiability(K, d.n_categories)

    ws = LCAWorkspace(d, K; aggregate=opts.aggregate)
    if K == 1
        θ, ll = _fit_single_class(ws)
        start_loglik = [ll]
        iterations, converged, replicated = 0, true, true
    else
        res = _multistart(ws, opts, rng, init, multithreaded)
        θ = res.θ
        start_loglik = [r.final_ll for r in res.records]
        iterations, converged, replicated = res.iterations, res.converged, res.replicated
    end
    _sort_by_size!(θ)
    ll = estep!(ws, θ)
    posterior = _expand_posterior(ws)
    beta = _beta_from_probs(θ.class_probs)

    flags = FitFlags(converged, _count_boundary(θ.item_probs),
                     findall(<(1e-6), θ.class_probs), replicated, false)
    msgs = _flag_messages(flags, opts)
    isempty(msgs) || @warn "$K-class fit: " * join(msgs, "; ")

    return LCAModel(K, ws.J, copy(d.n_categories), θ.class_probs, θ.item_probs, beta, d,
                    posterior, ll, converged, iterations, start_loglik, opts, nothing, flags)
end

function StatsAPI.fit(::Type{LCAModel}, d::LCAData, ks::AbstractVector{<:Integer}; kwargs...)
    return LCAModel[fit(LCAModel, d, k; kwargs...) for k in ks]
end

function StatsAPI.fit(::Type{LCAModel}, table, items::AbstractVector,
                      k::Union{Integer,AbstractVector{<:Integer}};
                      covariates::AbstractVector=Symbol[], levels=nothing,
                      drop_unused_levels::Bool=true, kwargs...)
    d = prepare_data(table, items; covariates=covariates, levels=levels,
                     drop_unused_levels=drop_unused_levels)
    return fit(LCAModel, d, k; kwargs...)
end

"""
    fit!(model::LCAModel, args...; kwargs...)

Not supported: the 0.2 workflow `model = LCAModel(k, n_items, n_categories); fit!(model,
data)` was replaced by `fit(LCAModel, data, k)` in v0.3, which returns the fitted model.
Always throws an `ArgumentError`.
"""
function StatsAPI.fit!(::LCAModel, args...; kwargs...)
    throw(ArgumentError("fit!(model, data) was replaced by fit(LCAModel, data, k) in v0.3; see CHANGELOG.md"))
end

# Multinomial-logit intercepts implied by the class sizes, class 1 as reference (1 × (K-1)).
function _beta_from_probs(class_probs::AbstractVector{<:Real})
    K = length(class_probs)
    return reshape([log(class_probs[k] / class_probs[1]) for k in 2:K], 1, K - 1)
end

"""
    check_identifiability(n_classes, n_categories) -> Bool

Check the necessary condition for identification of a latent class model: the number of
free parameters `(K - 1) + K·Σ_j (C_j - 1)` must not exceed the number of independent
cells of the response-pattern table, `∏_j C_j - 1`. Warns and returns `false` when the
condition fails; equality is generically identified and does not warn. The product is
computed in floating point so it does not overflow for many items. Internal.
"""
function check_identifiability(n_classes::Integer, n_categories::AbstractVector{<:Integer})
    K = Int(n_classes)
    n_params = (K - 1) + K * sum(Int(c) - 1 for c in n_categories; init=0)
    n_cells = prod(Float64.(n_categories)) - 1
    if n_params > n_cells
        cells = n_cells < 1e15 ? string(round(Int, n_cells)) : string(n_cells)
        @warn "Model may not be identified: $n_params free parameters exceed the $cells " *
              "degrees of freedom of the response-pattern table (prod(n_categories) - 1); " *
              "use fewer classes or more items"
        return false
    end
    return true
end

# Number of item-response probabilities within 1e-6 of 0 or 1.
function _count_boundary(item_probs::AbstractVector{<:AbstractMatrix{<:Real}})
    n = 0
    for P in item_probs, p in P
        (p <= 1e-6 || p >= 1 - 1e-6) && (n += 1)
    end
    return n
end

# Human-readable list of raised flags (empty when the fit is clean).
function _flag_messages(f::FitFlags, opts::Union{Nothing,LCAOptions}=nothing)
    msgs = String[]
    if !f.converged
        push!(msgs, opts === nothing ? "EM did not converge" :
                    "EM did not converge within $(opts.max_iter) iterations")
    end
    f.n_boundary > 0 && push!(msgs, "$(f.n_boundary) item-response probabilities are on the boundary (0 or 1)")
    isempty(f.empty_classes) || push!(msgs, "empty class(es) $(f.empty_classes) (size < 1e-6)")
    if !f.best_ll_replicated
        push!(msgs, "the best log-likelihood was reached by only one of the continued starts; increase n_starts/n_final")
    end
    f.coef_divergence && push!(msgs, "covariate coefficients diverged (quasi-complete separation)")
    return msgs
end
