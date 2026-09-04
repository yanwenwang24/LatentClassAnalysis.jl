# Inference on the free-parameter (logit) scale: coef/coefnames, the observed information
# matrix, vcov/stderror/confint/coeftable and the delta-method standard errors of the
# item-response profiles land in a later phase. This file holds only what is needed now:
# `vcov` on the stored field and `profiles` (probability scale, standard errors NaN when no
# covariance matrix is available).

"""
    vcov(m::LCAModel) -> Matrix{Float64}

Covariance matrix of the free parameters on the [`coef`](@ref) scale. Throws an
`ErrorException` when the model carries no covariance matrix (fitted with `se=:none`, or
standard errors not available in this version).
"""
function StatsAPI.vcov(m::LCAModel)
    m.vcov === nothing && throw(ErrorException(
        "no covariance matrix is available for this model (fitted with se=:none, or standard errors are not implemented in this version)"))
    return m.vcov
end

const _ProfileRow = NamedTuple{(:item, :level, :class, :prob, :se, :lower, :upper),
                               Tuple{Symbol,String,Int,Float64,Float64,Float64,Float64}}

"""
    profiles(m::LCAModel; level=0.95) -> Vector{NamedTuple}

Item-response profiles of a fitted model as a row table (a `Vector` of `NamedTuple`s,
usable with any Tables.jl sink such as `DataFrame`). Each row holds one item-response
probability: `item::Symbol`, `level::String` (the response label), `class::Int`,
`prob::Float64`, its delta-method standard error `se` and the bounds `lower`/`upper` of a
`level` confidence interval computed on the logit scale (so they stay within `[0, 1]`).
Rows are ordered by item, then level, then class; there are `Σ_j C_j · K` rows.

`se`, `lower` and `upper` are `NaN` when the model carries no covariance matrix (fitted
with `se=:none`, or standard errors not available in this version).

# Arguments
- `m::LCAModel`: fitted model
- `level::Real=0.95`: confidence level, in `(0, 1)`

# Returns
- `Vector{NamedTuple}` with fields `(:item, :level, :class, :prob, :se, :lower, :upper)`

See also [`show_profiles`](@ref) for a printed version.
"""
function profiles(m::LCAModel; level::Real=0.95)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    z = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    rows = _ProfileRow[]
    for j in 1:m.n_items, c in 1:m.n_categories[j], k in 1:m.n_classes
        p = m.item_probs[j][k, c]
        se = _profile_se(m, j, k, c)
        lower, upper = _logit_ci(p, se, z)
        push!(rows, (item=m.data.item_names[j], level=m.data.item_levels[j][c], class=k,
                     prob=p, se=se, lower=lower, upper=upper))
    end
    return rows
end

# Delta-method standard error of item_probs[j][k, c]; NaN without a covariance matrix.
_profile_se(m::LCAModel, j::Integer, k::Integer, c::Integer) = NaN

# Confidence interval of a probability from its standard error, on the logit scale.
function _logit_ci(p::Real, se::Real, z::Real)
    (isnan(se) || isnan(p)) && return (NaN, NaN)
    if p <= 0 || p >= 1
        return (p, p)
    end
    se_logit = se / (p * (1 - p))
    lo = log(p / (1 - p)) - z * se_logit
    hi = log(p / (1 - p)) + z * se_logit
    return (1 / (1 + exp(-lo)), 1 / (1 + exp(-hi)))
end
