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

With covariates (latent class regression) the class membership of observation `i` follows
the multinomial-logit model `log(π_k(x_i) / π_1(x_i)) = x_i'β_k` for `k = 2, …, K`, with
`x_i` the row of `d.X` (intercept first) and class 1 (the largest class) as reference.
The M-step for `β` is one damped Newton step (generalized EM), so the log-likelihood
still increases at every iteration. The covariates are standardized internally; the
returned `beta` is on the raw scale and `class_probs` are the membership probabilities
averaged over the sample. Response-pattern aggregation is disabled with covariates.

# Arguments
- `d::LCAData`: prepared data, see [`LCAData`](@ref) and [`prepare_data`](@ref)
- `k::Integer`: number of latent classes (`≥ 1`)

# Keyword arguments
- `rng::AbstractRNG=Random.default_rng()`: source of the seeds of the random starts; pass
  a seeded generator for a reproducible fit
- `covariates::Bool=hascovariates(d)`: fit the class-membership model on the covariates of
  `d` (latent class regression). `covariates=false` fits the unconditional model on the
  same data, for nested comparisons (its log-likelihood is never larger). A constant or
  collinear covariate is an `ArgumentError`.
- `init=nothing`: starting values used as the first start(s): an [`LCAModel`](@ref) with
  the same `k` and categories, an internal `LCAParams`, a `NamedTuple` with fields
  `class_probs` and `item_probs`, or a vector of those. If more starts are supplied than
  `n_starts`, all of them are run. With covariates, the coefficients of an `LCAModel`
  fitted with the same covariates seed the start; otherwise the slopes start at zero and
  the intercepts at the log-odds of the supplied class probabilities. Every random start
  begins with zero coefficients (uniform class sizes).
- `n_starts::Integer=20`: number of starts
- `n_final::Integer=4`: number of best short runs continued to convergence (capped at
  `n_starts`)
- `short_iters::Integer=50`: EM iterations of every short run
- `max_iter::Integer=10_000`: maximum EM iterations of every final run
- `tol::Real=1e-10`: relative convergence tolerance, `|ll - ll_old| ≤ tol·(1 + |ll|)`
- `se::Symbol=:hessian`: standard errors. `:hessian` computes the covariance matrix of
  the free parameters from the observed information matrix (analytic score, central
  finite-difference Hessian; two E-step passes per free parameter) and stores it in `vcov`, which
  [`vcov`](@ref), [`stderror`](@ref), [`confint`](@ref), [`coeftable`](@ref) and the
  `se`/`lower`/`upper` columns of [`profiles`](@ref) read. Parameters on the boundary
  (a probability within `1e-6` of 0 or 1), the response parameters of an empty class and
  parameters with zero observed information are held fixed and get `NaN` standard
  errors with a warning; the remaining standard errors are conditional on them (for a
  row with a boundary cell, on that cell being fixed). The whole matrix is `NaN` when
  the observed information of the remaining parameters is not positive definite or the
  covariate coefficients diverged. `:none` skips the computation (`vcov` is `nothing`);
  use it for bootstrap replicates or very large models.
- `aggregate::Bool=true`: collapse identical response patterns before running EM (exact;
  disabled automatically with covariates)
- `multithreaded::Bool=false`: run the starts on all Julia threads (results are identical
  to the serial run)
- `verbose::Bool=false`: print a line per start

# Returns
- [`LCAModel`](@ref). A single aggregated warning reports any raised [`FitFlags`](@ref)
  (non-convergence, boundary probabilities, empty classes, a best log-likelihood found by
  a single start, diverging covariate coefficients under quasi-complete separation) and
  any standard errors that could not be computed. A warning is also issued when the
  model is not identified by the necessary condition
  `(k - 1) + k·Σ_j (C_j - 1) ≤ ∏_j C_j - 1`.

# Example
```julia
using LatentClassAnalysis, StableRNGs
d = prepare_data(table, [:x1, :x2, :x3, :x4])
m = fit(LCAModel, d, 3; rng=StableRNG(1))
models = fit(LCAModel, d, 1:4; rng=StableRNG(1))
diagnostics(models)
dr = prepare_data(table, [:x1, :x2, :x3, :x4]; covariates=[:age, :female])
mr = fit(LCAModel, dr, 3; rng=StableRNG(1))     # latent class regression; mr.beta
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
    covariates && !hascovariates(d) && throw(ArgumentError(
        "covariates=true requires data with covariates; pass them to prepare_data or LCAData"))
    opts = LCAOptions(; n_starts=Int(n_starts), n_final=min(Int(n_final), Int(n_starts)),
                      short_iters=Int(short_iters), max_iter=Int(max_iter), tol=Float64(tol),
                      se=se, aggregate=aggregate && !covariates, verbose=verbose)
    check_identifiability(K, d.n_categories)
    _check_init_covariates(init, d, covariates)

    ws = LCAWorkspace(d, K; aggregate=opts.aggregate, covariates=covariates)
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
    if covariates
        # K == 1 carries no coefficients; beta is then P × 0
        coefs = θ.coefs === nothing ? zeros(size(d.X, 2), K) : θ.coefs
        beta = (ws.A * coefs)[:, 2:K]                       # raw scale
        class_probs = vec(mean(_class_prior(beta, d.X), dims=1))
        diverged = maximum(abs, coefs; init=0.0) > COEF_DIVERGENCE_THRESHOLD
    else
        beta = _beta_from_probs(θ.class_probs)
        class_probs = θ.class_probs
        diverged = false
    end
    # The covariance matrix reuses the workspace (its posterior was expanded above)
    vc, se_msgs = _fit_vcov(θ, ws, opts, diverged)

    flags = FitFlags(converged, _count_boundary(θ.item_probs),
                     findall(<=(BOUNDARY_TOL), class_probs), replicated, diverged)
    msgs = vcat(_unobserved_messages(d), _flag_messages(flags, opts), se_msgs)
    isempty(msgs) || @warn "$K-class fit: " * join(msgs, "; ")

    return LCAModel(K, ws.J, copy(d.n_categories), class_probs, θ.item_probs, beta, d,
                    posterior, ll, converged, iterations, start_loglik, opts, vc, flags)
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

# Number of item-response probabilities on the boundary (within BOUNDARY_TOL of 0 or 1).
function _count_boundary(item_probs::AbstractVector{<:AbstractMatrix{<:Real}})
    n = 0
    for P in item_probs, p in P
        _interior(p) || (n += 1)
    end
    return n
end

# Items without a single observed response: their response probabilities stay uniform
# and cannot be estimated.
function _unobserved_messages(d::LCAData)
    unobserved = [d.item_names[j] for j in 1:size(d.y, 2) if all(iszero, view(d.y, :, j))]
    isempty(unobserved) && return String[]
    return ["item(s) $unobserved have no observed responses; their response probabilities are not estimable"]
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
