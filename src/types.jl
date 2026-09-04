# ---------------------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------------------

"""
    LCAData(y; n_categories=nothing, item_names=nothing, item_levels=nothing,
            covariates=nothing, covariate_names=nothing)

Prepared data for latent class analysis: an `n × J` matrix of integer response codes plus
optional covariates for the class-membership model. [`prepare_data`](@ref) builds one from
any Tables.jl source (a `DataFrame`, a `NamedTuple` of vectors, ...); this constructor
accepts an already coded matrix.

# Arguments
- `y::AbstractMatrix{<:Union{Missing,Integer}}`: response codes, one row per observation
  and one column per item. The codes of item `j` are `1, 2, …, n_categories[j]`; a
  `missing` entry (or the code `0`) marks a missing response.
- `n_categories`: number of response categories of each item. Defaults to the largest code
  observed in each column, which must then be at least 2 (a column whose largest code is
  1 has a single observed category).
- `item_names`: item names as `Symbol`s (default `:item1, :item2, …`).
- `item_levels`: `item_levels[j][c]` is the label of code `c` of item `j`
  (default `"1", "2", …`).
- `covariates`: an `n × p` matrix (or a vector, for `p = 1`) of real-valued covariates for
  the class-membership model, *without* an intercept; an intercept column is prepended.
  Missing values are not allowed: drop those rows first.
- `covariate_names`: names of the `p` covariate columns (default `:x1, :x2, …`).

# Fields
- `y::Matrix{Int}`: codes; `0` marks a missing response
- `n_categories::Vector{Int}`: number of categories `C_j` of every item
- `item_names::Vector{Symbol}`
- `item_levels::Vector{Vector{String}}`: `item_levels[j][c]` is the label of code `c`
- `X::Matrix{Float64}`: `n × P` covariate matrix whose first column is the intercept;
  `P == 1` when there are no covariates
- `covariate_names::Vector{Symbol}`: first entry is `:intercept`

Every field is validated: codes lie in `0:n_categories[j]`, label vectors have
`n_categories[j]` entries, `X` has `n` rows, a leading column of ones and no `NaN`/`Inf`.

Accessors: [`nobs`](@ref), `size`, [`hasmissing`](@ref), [`nmissing`](@ref),
[`hascovariates`](@ref).

# Example
```jldoctest
julia> using LatentClassAnalysis

julia> d = LCAData([1 2 1; 2 2 missing; 1 1 1; 2 1 2]);

julia> size(d), d.n_categories
((4, 3), [2, 2, 2])

julia> d.y[2, 3], hasmissing(d), nmissing(d)
(0, true, [0, 0, 1])

julia> d.item_names, d.item_levels[1], hascovariates(d)
([:item1, :item2, :item3], ["1", "2"], false)
```
"""
struct LCAData
    y::Matrix{Int}
    n_categories::Vector{Int}
    item_names::Vector{Symbol}
    item_levels::Vector{Vector{String}}
    X::Matrix{Float64}
    covariate_names::Vector{Symbol}

    function LCAData(y::Matrix{Int}, n_categories::Vector{Int}, item_names::Vector{Symbol},
                     item_levels::Vector{Vector{String}}, X::Matrix{Float64},
                     covariate_names::Vector{Symbol})
        n, J = size(y)
        J >= 1 || throw(ArgumentError("the data must contain at least one item (column)"))
        length(n_categories) == J ||
            throw(ArgumentError("length of n_categories ($(length(n_categories))) must equal the number of items ($J)"))
        length(item_names) == J ||
            throw(ArgumentError("length of item_names ($(length(item_names))) must equal the number of items ($J)"))
        allunique(item_names) || throw(ArgumentError("item_names must be unique"))
        length(item_levels) == J ||
            throw(ArgumentError("length of item_levels ($(length(item_levels))) must equal the number of items ($J)"))
        for j in 1:J
            C = n_categories[j]
            C >= 2 || throw(ArgumentError("item $(item_names[j]) must have at least two categories, got $C"))
            length(item_levels[j]) == C ||
                throw(ArgumentError("item_levels for $(item_names[j]) has $(length(item_levels[j])) labels but the item has $C categories"))
            for i in 1:n
                0 <= y[i, j] <= C ||
                    throw(ArgumentError("invalid code $(y[i, j]) in row $i of item $(item_names[j]); codes must be in 0:$C (0 = missing)"))
            end
        end
        size(X, 1) == n ||
            throw(ArgumentError("the covariate matrix has $(size(X, 1)) rows but the data has $n observations"))
        size(X, 2) >= 1 || throw(ArgumentError("the covariate matrix must contain the intercept column"))
        all(isone, view(X, :, 1)) ||
            throw(ArgumentError("the first column of the covariate matrix must be the intercept (all ones)"))
        all(isfinite, X) || throw(ArgumentError("the covariate matrix contains NaN or Inf"))
        length(covariate_names) == size(X, 2) ||
            throw(ArgumentError("length of covariate_names ($(length(covariate_names))) must equal the number of columns of X ($(size(X, 2)))"))
        covariate_names[1] === :intercept ||
            throw(ArgumentError("the first covariate name must be :intercept, got $(repr(covariate_names[1]))"))
        allunique(covariate_names) || throw(ArgumentError("covariate_names must be unique"))
        return new(y, n_categories, item_names, item_levels, X, covariate_names)
    end
end

function LCAData(y::AbstractMatrix{<:Union{Missing,Integer}};
                 n_categories::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                 item_names::Union{Nothing,AbstractVector}=nothing,
                 item_levels::Union{Nothing,AbstractVector}=nothing,
                 covariates::Union{Nothing,AbstractVecOrMat{<:Real}}=nothing,
                 covariate_names::Union{Nothing,AbstractVector}=nothing)
    n, J = size(y)
    J >= 1 || throw(ArgumentError("the data must contain at least one item (column)"))
    codes = Matrix{Int}(undef, n, J)
    for j in 1:J, i in 1:n
        v = y[i, j]
        if ismissing(v)
            codes[i, j] = 0
        else
            v >= 0 || throw(ArgumentError("invalid code $v in row $i of column $j; codes must be non-negative (0 = missing)"))
            codes[i, j] = Int(v)
        end
    end
    names = item_names === nothing ? _default_item_names(J) : Symbol.(item_names)
    if n_categories === nothing
        C = Vector{Int}(undef, J)
        for j in 1:J
            C[j] = n == 0 ? 0 : maximum(view(codes, :, j))
            C[j] >= 2 || throw(ArgumentError(
                "column $j has no code larger than 1, so it shows a single observed category. " *
                "Codes are 1-based and 0 marks a missing response: 0/1-coded data must be " *
                "shifted by one (or prepared with prepare_data); otherwise pass n_categories explicitly"))
        end
    else
        C = Int.(n_categories)
        for j in 1:J
            col = view(codes, :, j)
            if n > 0 && any(==(0), col) && maximum(col) <= 1
                @warn "column $j contains only the codes 0 and 1; 0 is the missing code, so if these " *
                      "data are 0/1-coded shift them by one or use prepare_data"
            end
        end
    end
    levels = item_levels === nothing ? [string.(1:c) for c in C] :
             [String.(collect(l)) for l in item_levels]
    if covariates === nothing
        X = ones(n, 1)
        cnames = [:intercept]
    else
        Xc = covariates isa AbstractVector ? reshape(covariates, :, 1) : covariates
        size(Xc, 1) == n ||
            throw(ArgumentError("covariates has $(size(Xc, 1)) rows but y has $n rows"))
        X = hcat(ones(n), Float64.(Xc))
        p = size(Xc, 2)
        cnames = covariate_names === nothing ? [Symbol("x", i) for i in 1:p] : Symbol.(covariate_names)
        length(cnames) == p ||
            throw(ArgumentError("covariate_names has $(length(cnames)) entries but covariates has $p columns"))
        cnames = [:intercept; cnames]
    end
    return LCAData(codes, C, names, levels, X, cnames)
end

_default_item_names(J::Integer) = [Symbol("item", j) for j in 1:J]

"""
    nobs(d::LCAData)

Number of observations (rows) in the data.
"""
StatsAPI.nobs(d::LCAData) = size(d.y, 1)

Base.size(d::LCAData) = size(d.y)
Base.size(d::LCAData, i::Integer) = size(d.y, i)

"""
    hasmissing(d::LCAData) -> Bool
    hasmissing(m::LCAModel) -> Bool

Whether any indicator response is missing (coded `0`).

# Example
```jldoctest
julia> using LatentClassAnalysis

julia> hasmissing(LCAData([1 2; 2 1; missing 2]))
true

julia> hasmissing(LCAData([1 2; 2 1]))
false
```
"""
hasmissing(d::LCAData) = any(iszero, d.y)

"""
    nmissing(d::LCAData) -> Vector{Int}
    nmissing(m::LCAModel) -> Vector{Int}

Number of missing responses of every item.

# Example
```jldoctest
julia> using LatentClassAnalysis

julia> nmissing(LCAData([1 2; 2 missing; missing 2]))
2-element Vector{Int64}:
 1
 1
```
"""
nmissing(d::LCAData) = [count(iszero, view(d.y, :, j)) for j in 1:size(d.y, 2)]

"""
    hascovariates(d::LCAData) -> Bool
    hascovariates(m::LCAModel) -> Bool

Whether the data carry covariates for the class-membership model (columns of `X` beyond
the intercept); for a model, whether it was fitted with covariates.

# Example
```jldoctest
julia> using LatentClassAnalysis

julia> hascovariates(LCAData([1 2; 2 1]; covariates=[0.5, 1.5]))
true

julia> hascovariates(LCAData([1 2; 2 1]))
false
```
"""
hascovariates(d::LCAData) = size(d.X, 2) > 1

# ---------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------

"""
    LCAOptions(; n_starts=20, n_final=4, short_iters=50, max_iter=10_000, tol=1e-10,
                 se=:hessian, aggregate=true, verbose=false)

Estimation settings of [`fit`](@ref), stored in the `options` field of a fitted
[`LCAModel`](@ref). Every keyword of `fit` with the same name maps to a field.

# Fields
- `n_starts::Int`: number of random starting values (short EM runs)
- `n_final::Int`: number of best short runs continued to convergence (`fit` caps it at
  `n_starts`)
- `short_iters::Int`: EM iterations of each short run
- `max_iter::Int`: maximum EM iterations of each final run
- `tol::Float64`: relative convergence tolerance, `|ll - ll_old| ≤ tol·(1 + |ll|)`
- `se::Symbol`: standard errors, `:hessian` or `:none`
- `aggregate::Bool`: collapse identical response patterns before running EM (exact; turned
  off automatically when covariates are present)
- `verbose::Bool`: print a summary of every start

# Example
```jldoctest
julia> using LatentClassAnalysis

julia> opts = LCAOptions(n_starts=5, n_final=2);

julia> opts.n_starts, opts.n_final, opts.max_iter, opts.tol, opts.se
(5, 2, 10000, 1.0e-10, :hessian)
```
"""
Base.@kwdef struct LCAOptions
    n_starts::Int = 20
    n_final::Int = 4
    short_iters::Int = 50
    max_iter::Int = 10_000
    tol::Float64 = 1e-10
    se::Symbol = :hessian
    aggregate::Bool = true
    verbose::Bool = false

    function LCAOptions(n_starts, n_final, short_iters, max_iter, tol, se, aggregate, verbose)
        n_starts >= 1 || throw(ArgumentError("n_starts must be at least 1, got $n_starts"))
        n_final >= 1 || throw(ArgumentError("n_final must be at least 1, got $n_final"))
        short_iters >= 0 || throw(ArgumentError("short_iters must be non-negative, got $short_iters"))
        max_iter >= 1 || throw(ArgumentError("max_iter must be at least 1, got $max_iter"))
        tol >= 0 || throw(ArgumentError("tol must be non-negative, got $tol"))
        se in (:none, :hessian) || throw(ArgumentError("se must be :none or :hessian, got $(repr(se))"))
        return new(n_starts, n_final, short_iters, max_iter, tol, se, aggregate, verbose)
    end
end

# ---------------------------------------------------------------------------------------
# Fitted model
# ---------------------------------------------------------------------------------------

"""
    FitFlags

Post-fit diagnostics collected by [`fit`](@ref) and stored in the `flags` field of an
[`LCAModel`](@ref). `fit` emits one aggregated warning when any flag is raised and
`show(model)` prints them.

# Fields
- `converged::Bool`: EM reached the convergence tolerance within `max_iter` iterations
- `n_boundary::Int`: number of item-response probabilities within `1e-6` of 0 or 1
- `empty_classes::Vector{Int}`: classes with a size below `1e-6`
- `best_ll_replicated::Bool`: the best log-likelihood was reached by at least two of the
  continued starts (trivially `true` when only one start was continued)
- `coef_divergence::Bool`: a covariate coefficient exceeds 20 in absolute value on the
  standardized covariate scale, the signature of quasi-complete separation (a covariate
  that determines class membership almost perfectly); the estimates are then unstable
"""
struct FitFlags
    converged::Bool
    n_boundary::Int
    empty_classes::Vector{Int}
    best_ll_replicated::Bool
    coef_divergence::Bool
end

_clean(f::FitFlags) = f.converged && f.n_boundary == 0 && isempty(f.empty_classes) &&
                      f.best_ll_replicated && !f.coef_divergence

"""
    LCAModel <: StatsAPI.StatisticalModel

A fitted latent class model, returned by [`fit`](@ref). The struct is immutable and the
classes are ordered by decreasing size, so class 1 is the largest class and the reference
class of `beta`.

# Fields
- `n_classes::Int`, `n_items::Int`, `n_categories::Vector{Int}`
- `class_probs::Vector{Float64}`: class sizes (marginal membership probabilities; with
  covariates, the covariate-specific membership probabilities averaged over the sample)
- `item_probs::Vector{Matrix{Float64}}`: `item_probs[j]` is `n_classes × n_categories[j]`;
  row `k` holds the probability of each response category of item `j` in class `k`
- `beta::Matrix{Float64}`: `P × (n_classes - 1)` multinomial-logit coefficients of the
  class-membership model `log(π_k(x) / π_1(x)) = x'β_k` with class 1 as reference, on the
  raw covariate scale; row `p` belongs to `data.covariate_names[p]` (row 1 is the
  intercept) and column `k - 1` to class `k`. Without covariates `P == 1` and
  `beta[1, k-1] == log(class_probs[k] / class_probs[1])`
- `data::LCAData`: the data the model was fitted to
- `posterior::Matrix{Float64}`: `nobs × n_classes` posterior membership probabilities
- `loglik::Float64`, `converged::Bool`, `iterations::Int`: final log-likelihood and EM
  status of the selected start
- `start_loglik::Vector{Float64}`: log-likelihood reached by every start (short runs keep
  their short-run value); its maximum is `loglik`
- `options::LCAOptions`: the settings used
- `vcov::Union{Nothing,Matrix{Float64}}`: `dof × dof` covariance matrix of [`coef`](@ref)
  from the observed information matrix (`se=:hessian`, the default), with `NaN` rows and
  columns for parameters on the boundary (held fixed, so the remaining entries are
  conditional on them); `nothing` when fitted with `se=:none`. Read it through
  [`vcov`](@ref), [`stderror`](@ref), [`confint`](@ref) and [`coeftable`](@ref)
- `flags::FitFlags`: post-fit warnings, see [`FitFlags`](@ref)

Use [`predict`](@ref)/[`classify`](@ref) for memberships, [`profiles`](@ref) and
[`show_profiles`](@ref) for the item-response profiles (with delta-method standard
errors), [`coeftable`](@ref) for the parameters on the logit scale, and
[`diagnostics`](@ref), [`loglikelihood`](@ref), [`aic`](@ref), [`bic`](@ref),
[`sbic`](@ref), [`entropy`](@ref) for fit statistics.

The 0.2 constructor `LCAModel(n_classes, n_items, n_categories)` throws an
`ArgumentError`; fit models with `fit(LCAModel, data, k)` instead.
"""
struct LCAModel <: StatsAPI.StatisticalModel
    n_classes::Int
    n_items::Int
    n_categories::Vector{Int}
    class_probs::Vector{Float64}
    item_probs::Vector{Matrix{Float64}}
    beta::Matrix{Float64}
    data::LCAData
    posterior::Matrix{Float64}
    loglik::Float64
    converged::Bool
    iterations::Int
    start_loglik::Vector{Float64}
    options::LCAOptions
    vcov::Union{Nothing,Matrix{Float64}}
    flags::FitFlags
end

hasmissing(m::LCAModel) = hasmissing(m.data)
nmissing(m::LCAModel) = nmissing(m.data)
hascovariates(m::LCAModel) = size(m.beta, 1) > 1

# ---------------------------------------------------------------------------------------
# Diagnostics and bootstrap results
# ---------------------------------------------------------------------------------------

"""
    ModelDiagnostics

Fit statistics of a fitted [`LCAModel`](@ref), returned by [`diagnostics`](@ref).
A `Vector{ModelDiagnostics}` implements the Tables.jl row interface, so
`DataFrame(diagnostics(models))` gives a model-selection table.

# Fields
- `n_classes::Int`, `nobs::Int`, `dof::Int`: number of classes, observations and free
  parameters
- `ll::Float64`: log-likelihood
- `aic::Float64`: `-2ll + 2·dof`
- `bic::Float64`: `-2ll + dof·log(nobs)`
- `sbic::Float64`: sample-size adjusted BIC, `-2ll + dof·log((nobs + 2) / 24)`
- `entropy::Float64`: relative entropy of the classification in `[0, 1]`, 1 meaning every
  observation is assigned with certainty
- `converged::Bool`
"""
struct ModelDiagnostics
    n_classes::Int
    nobs::Int
    dof::Int
    ll::Float64
    aic::Float64
    bic::Float64
    sbic::Float64
    entropy::Float64
    converged::Bool
end

"""
    LCABootstrap

Result of [`bootstrap`](@ref): bootstrap replicates of the free parameters of an
[`LCAModel`](@ref) on the [`coef`](@ref) scale, with labels aligned to the fitted model.

# Fields
- `model::LCAModel`: the model that was bootstrapped
- `n_boot::Int`: number of replicates
- `coefs::Matrix{Float64}`: `n_boot × dof(model)` aligned coefficient replicates, one row
  per replicate in the order of [`coef`](@ref) (`NaN` rows for replicates whose fit
  failed)
- `converged::Vector{Bool}`: convergence status of every replicate fit

Read it with [`vcov`](@ref vcov(::LCABootstrap)), [`stderror`](@ref stderror(::LCABootstrap)),
[`confint`](@ref confint(::LCABootstrap)), [`coeftable`](@ref coeftable(::LCABootstrap))
and [`profiles`](@ref profiles(::LCABootstrap)).
"""
struct LCABootstrap
    model::LCAModel
    n_boot::Int
    coefs::Matrix{Float64}
    converged::Vector{Bool}
end

"""
    BootstrapLRT

Result of [`bootstrap_lrt`](@ref), the parametric bootstrap likelihood-ratio test of a
`K`-class model against a `K + 1`-class model.

# Fields
- `null::LCAModel`, `alternative::LCAModel`: the two fitted models
- `statistic::Float64`: observed `2(ll_alternative - ll_null)`
- `replicates::Vector{Float64}`: bootstrap statistics `2(ll_{K+1} - ll_K)` of the data
  sets simulated from the null model
- `pvalue::Float64`: `(1 + #{replicates ≥ statistic}) / (n_boot + 1)`, read with
  [`pvalue`](@ref)
- `n_boot::Int`: number of replicates
- `n_negative::Int`: number of replicates whose statistic is below `-1e-6` (the
  `K + 1`-class fit of that replicate ended at a lower log-likelihood than its `K`-class
  fit, a sign of insufficient random starts); zero in a clean run
- `converged::Vector{Bool}`: whether both fits of every replicate converged
"""
struct BootstrapLRT
    null::LCAModel
    alternative::LCAModel
    statistic::Float64
    replicates::Vector{Float64}
    pvalue::Float64
    n_boot::Int
    n_negative::Int
    converged::Vector{Bool}
end
