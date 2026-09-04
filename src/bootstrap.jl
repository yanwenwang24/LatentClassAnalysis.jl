# Simulation from a fitted model, alignment of the class labels of replicate fits,
# bootstrap standard errors (LCABootstrap) and the parametric bootstrap likelihood-ratio
# test for the number of classes (BootstrapLRT). Every replicate is driven by its own
# `Xoshiro(seed)` with the seeds drawn up front, so serial and threaded runs agree bitwise.

# ---------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------

# Draw an index from the probability vector `p` by inverting its cumulative distribution
# at one uniform draw; round-off in the cumulative sum falls back to the last category.
@inline function _rand_category(rng::AbstractRNG, p::AbstractVector{<:Real})
    u = rand(rng)
    s = 0.0
    n = length(p)
    @inbounds for c in 1:n-1
        s += p[c]
        u < s && return c
    end
    return n
end

"""
    _simulate(rng, m::LCAModel, n, X, missing_mask) -> (d::LCAData, z::Vector{Int})

Simulation kernel of [`simulate`](@ref): draws the class `z[i]` of every observation from
`m.class_probs` (or from `softmax(x_i'β)` with `x_i` the rows of the full `n × P` design
`X`, intercept first, for a model with covariates), then every response from the class's
row of `m.item_probs[j]`, both by inverse-CDF at a single `rand(rng)`. Cells where
`missing_mask` is `true` are set to `0`. Returns the data with the model's item names,
levels and categories (and the covariates of `X`) together with the class vector.
Internal.
"""
function _simulate(rng::AbstractRNG, m::LCAModel, n::Integer,
                   X::Union{Nothing,AbstractMatrix{<:Real}},
                   missing_mask::Union{Nothing,AbstractMatrix{Bool}})
    n >= 1 || throw(ArgumentError("the number of observations must be at least 1, got $n"))
    J = m.n_items
    withcov = hascovariates(m)
    if withcov
        X === nothing && throw(ArgumentError("the model has covariates: a design matrix is required"))
        P = size(m.beta, 1)
        size(X) == (n, P) ||
            throw(DimensionMismatch("the design has size $(size(X)), expected ($n, $P)"))
        prior = _class_prior(m.beta, X)          # n × K membership probabilities
        Xfull = Matrix{Float64}(X)
        cnames = m.data.covariate_names
    else
        Xfull = ones(n, 1)
        cnames = [:intercept]
    end
    if missing_mask !== nothing
        size(missing_mask) == (n, J) ||
            throw(DimensionMismatch("missing_mask has size $(size(missing_mask)), expected ($n, $J)"))
    end
    y = Matrix{Int}(undef, n, J)
    z = Vector{Int}(undef, n)
    π = m.class_probs
    @inbounds for i in 1:n
        k = withcov ? _rand_category(rng, view(prior, i, :)) : _rand_category(rng, π)
        z[i] = k
        for j in 1:J
            y[i, j] = _rand_category(rng, view(m.item_probs[j], k, :))
        end
    end
    if missing_mask !== nothing
        @inbounds for j in 1:J, i in 1:n
            missing_mask[i, j] && (y[i, j] = 0)
        end
    end
    d = LCAData(y, m.n_categories, m.data.item_names, m.data.item_levels, Xfull, cnames)
    return d, z
end

# Full design (n × P, intercept first) of a simulation from the user's `X`: the model's
# own design when none is given and n == nobs(m); `nothing` for a model without
# covariates, which accepts no `X`.
function _simulation_design(m::LCAModel, n::Integer, X)
    if !hascovariates(m)
        X === nothing || throw(ArgumentError(
            "the model has no covariates, so X must not be given; attach covariates to the " *
            "simulated data with LCAData(d.y; covariates=...) if needed"))
        return nothing
    end
    P = size(m.beta, 1)
    covs = join(string.(m.data.covariate_names[2:end]), ", ")
    if X === nothing
        n == nobs(m) || throw(ArgumentError(
            "the model has covariates ($covs): pass X, an n × $(P - 1) matrix of covariate " *
            "values without the intercept, to simulate $n observations, or use n = $(nobs(m)) " *
            "to reuse the model's covariates"))
        return m.data.X
    end
    Xc = X isa AbstractVector ? reshape(X, :, 1) : X
    size(Xc, 1) == n || throw(DimensionMismatch("X has $(size(Xc, 1)) rows but n = $n"))
    size(Xc, 2) == P - 1 || throw(DimensionMismatch(
        "X has $(size(Xc, 2)) columns but the model has $(P - 1) covariate(s) ($covs); " *
        "pass the covariates without the intercept column"))
    Xfull = hcat(ones(n), Float64.(Xc))
    all(isfinite, Xfull) || throw(ArgumentError("X contains NaN or Inf"))
    return Xfull
end

"""
    simulate(m::LCAModel, n=nobs(m); rng=Random.default_rng(), X=nothing,
             missing_mask=nothing) -> LCAData

Draw `n` observations from the fitted model `m`. The class of every observation is drawn
from the class sizes `m.class_probs` or, for a model with covariates, from its
covariate-specific membership probabilities `softmax(x_i'β)`; the response to every item is
then drawn from the class's response probabilities `m.item_probs[j]`. Both draws invert the
cumulative distribution at a single `rand(rng)`, so a `StableRNG` gives data that are
identical across Julia versions.

The result carries the item names, level labels and number of categories of `m` (an item
keeps all its categories even when a level is not drawn) and, for a model with covariates,
the covariates it was simulated with, so it can be refitted with [`fit`](@ref) or used with
[`predict`](@ref) directly.

# Arguments
- `m::LCAModel`: fitted model
- `n::Integer=nobs(m)`: number of observations to draw

# Keyword arguments
- `rng::AbstractRNG=Random.default_rng()`: random number generator
- `X=nothing`: covariate values for a model with covariates, as an `n × (P - 1)` matrix or
  a vector for a single covariate, *without* the intercept column (the convention of the
  `covariates` keyword of [`LCAData`](@ref)). Defaults to the model's own covariates
  `m.data.X` when `n == nobs(m)` and must be given otherwise. A model without covariates
  accepts no `X`; attach covariates to the result with `LCAData(d.y; covariates=...)` if
  needed.
- `missing_mask=nothing`: an `n × J` `Bool` matrix; responses where it is `true` are set
  to missing (code `0`). Pass `m.data.y .== 0` to reproduce the missingness pattern of the
  training data, as the parametric [`bootstrap`](@ref) and [`bootstrap_lrt`](@ref) do.

# Returns
- [`LCAData`](@ref) with `n` rows

# Example
```julia
d_sim = simulate(m, 5000; rng=StableRNG(1))
m_sim = fit(LCAModel, d_sim, m.n_classes; rng=StableRNG(1))     # parameter recovery
```
"""
function simulate(m::LCAModel, n::Integer=nobs(m); rng::AbstractRNG=Random.default_rng(),
                  X::Union{Nothing,AbstractVecOrMat{<:Real}}=nothing,
                  missing_mask::Union{Nothing,AbstractMatrix{Bool}}=nothing)
    d, _ = _simulate(rng, m, n, _simulation_design(m, n, X), missing_mask)
    return d
end

# ---------------------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------------------

# Largest number of classes for which the alignment is solved exactly (K! permutations).
const ALIGN_EXACT_MAX_K = 7

_align_probs(θ::LCAParams) = (θ.class_probs, θ.item_probs)
_align_probs(m::LCAModel) = (m.class_probs, m.item_probs)

"""
    _align_labels(rep, ref) -> perm

Permutation of the classes of the replicate `rep` (an `LCAParams` or an `LCAModel`) that
best matches the reference `ref`: `perm[k]` is the replicate class matched to reference
class `k`, so that `rep.item_probs[j][perm, :] ≈ ref.item_probs[j]` and
`rep.class_probs[perm] ≈ ref.class_probs`. The cost of a permutation is
`Σ_j ‖B_j[perm, :] - B_j^ref‖² + ‖π[perm] - π^ref‖²`, minimized exactly over all `K!`
permutations (depth-first with branch and bound) for `K ≤ 7` and by a greedy assignment
(smallest remaining distance first, with a one-time warning) for larger `K`. Internal.
"""
function _align_labels(rep, ref)
    π_rep, B_rep = _align_probs(rep)
    π_ref, B_ref = _align_probs(ref)
    K = length(π_ref)
    length(π_rep) == K ||
        throw(DimensionMismatch("the replicate has $(length(π_rep)) classes, the reference $K"))
    length(B_rep) == length(B_ref) ||
        throw(DimensionMismatch("the replicate has $(length(B_rep)) items, the reference $(length(B_ref))"))
    # D[k, l]: squared distance of replicate class l from reference class k
    D = Matrix{Float64}(undef, K, K)
    for l in 1:K, k in 1:K
        c = (π_rep[l] - π_ref[k])^2
        for (Br, Bf) in zip(B_rep, B_ref)
            size(Br, 2) == size(Bf, 2) ||
                throw(DimensionMismatch("item matrices with $(size(Br, 2)) and $(size(Bf, 2)) categories cannot be aligned"))
            @inbounds for cc in 1:size(Bf, 2)
                c += (Br[l, cc] - Bf[k, cc])^2
            end
        end
        D[k, l] = c
    end
    return _assignment(D)
end

# Minimum-cost assignment for a K × K cost matrix: exact for K ≤ ALIGN_EXACT_MAX_K.
function _assignment(D::AbstractMatrix{Float64})
    K = size(D, 1)
    K == 1 && return [1]
    if K <= ALIGN_EXACT_MAX_K
        best = collect(1:K)
        _assign_dfs!(best, Ref(Inf), zeros(Int, K), falses(K), D, 1, 0.0)
        return best
    end
    @warn "aligning the labels of $K classes uses a greedy assignment (exact matching is " *
          "limited to $ALIGN_EXACT_MAX_K classes); the alignment may not be optimal" maxlog = 1
    perm = zeros(Int, K)
    rowdone = falses(K)
    coldone = falses(K)
    for _ in 1:K
        bk, bl, bc = 0, 0, Inf
        for l in 1:K, k in 1:K
            (rowdone[k] || coldone[l]) && continue
            if D[k, l] < bc
                bk, bl, bc = k, l, D[k, l]
            end
        end
        perm[bk] = bl
        rowdone[bk] = true
        coldone[bl] = true
    end
    return perm
end

# Depth-first enumeration of the permutations with branch and bound on the partial cost.
# Equal costs keep the first permutation found, so an exact match returns the identity.
function _assign_dfs!(best::Vector{Int}, best_cost::Ref{Float64}, perm::Vector{Int},
                      used::BitVector, D::AbstractMatrix{Float64}, k::Int, cost::Float64)
    K = length(perm)
    if k > K
        if cost < best_cost[]
            best_cost[] = cost
            copyto!(best, perm)
        end
        return nothing
    end
    for l in 1:K
        used[l] && continue
        c = cost + D[k, l]
        c >= best_cost[] && continue
        used[l] = true
        perm[k] = l
        _assign_dfs!(best, best_cost, perm, used, D, k + 1, c)
        used[l] = false
    end
    return nothing
end

"""
    _align!(θ::LCAParams, ref) -> perm

Reorder the classes of `θ` in place to match the reference `ref` (see
[`_align_labels`](@ref)), re-basing the coefficients so that class 1 stays the reference
class. Returns the permutation applied. Internal.
"""
function _align!(θ::LCAParams, ref)
    perm = _align_labels(θ, ref)
    _permute_classes!(θ, perm)
    return perm
end

# Parameters of a replicate fit with the coefficients on the raw covariate scale
# (`[0 beta]`, P × K), aligned to the reference model.
function _aligned_params(m_rep::LCAModel, ref::LCAModel)
    K = m_rep.n_classes
    coefs = (hascovariates(m_rep) && K > 1) ? hcat(zeros(size(m_rep.beta, 1)), m_rep.beta) : nothing
    θ = LCAParams(copy(m_rep.class_probs), [copy(B) for B in m_rep.item_probs], coefs)
    _align!(θ, ref)
    return θ
end

# ---------------------------------------------------------------------------------------
# Replicate machinery shared by bootstrap and bootstrap_lrt
# ---------------------------------------------------------------------------------------

# Run f(b) for b in 1:n, on all threads when requested. Every replicate seeds its own
# generator, so the results do not depend on the schedule.
function _foreach_replicate(f, n::Integer, multithreaded::Bool)
    if multithreaded && Threads.nthreads() > 1
        Threads.@threads for b in 1:n
            f(b)
        end
    else
        for b in 1:n
            f(b)
        end
    end
    return nothing
end

# Fit K classes to d without the per-fit warnings (the callers aggregate them), with the
# tolerance, iteration budget and aggregation setting of `o` and no covariance matrix.
function _fit_quietly(d::LCAData, K::Integer, rng::AbstractRNG, covariates::Bool, init,
                      n_starts::Integer, n_final::Integer, o::LCAOptions)
    return Logging.with_logger(Logging.NullLogger()) do
        fit(LCAModel, d, K; rng=rng, covariates=covariates, init=init, n_starts=n_starts,
            n_final=n_final, short_iters=o.short_iters, max_iter=o.max_iter, tol=o.tol,
            se=:none, aggregate=o.aggregate)
    end
end

# Resample the rows of d with replacement (with their covariates and missing pattern).
function _resample(rng::AbstractRNG, d::LCAData)
    n = nobs(d)
    idx = rand(rng, 1:n, n)
    return LCAData(d.y[idx, :], d.n_categories, d.item_names, d.item_levels, d.X[idx, :],
                   d.covariate_names)
end

# ---------------------------------------------------------------------------------------
# Bootstrap standard errors
# ---------------------------------------------------------------------------------------

"""
    bootstrap(m::LCAModel; n_boot=200, rng=Random.default_rng(), parametric=false,
              n_starts=1, multithreaded=false) -> LCABootstrap

Bootstrap the free parameters of a fitted model. Every replicate draws a data set of
`nobs(m)` rows, refits the model with the same number of classes, aligns the class labels
of the refit to `m` and records the refitted parameters on the [`coef`](@ref) scale
(logits; covariate coefficients on the raw scale). The replicates are read through
[`vcov`](@ref vcov(::LCABootstrap)), [`stderror`](@ref stderror(::LCABootstrap)),
[`confint`](@ref confint(::LCABootstrap)), [`coeftable`](@ref coeftable(::LCABootstrap))
and, on the probability scale, [`profiles`](@ref profiles(::LCABootstrap)).

- `parametric=false` (the non-parametric bootstrap) resamples the rows of the training
  data with replacement, each with its covariates and its missing responses.
- `parametric=true` simulates complete data from `m` with [`simulate`](@ref) (with the
  model's covariates) and re-applies the missingness pattern of the training data.

Each refit is warm-started from `m` and continued to convergence; `n_starts > 1` adds
`n_starts - 1` random starts, all continued (`n_final = n_starts`), which guards against
label switching into a different local maximum at a higher cost. The refits use the
`tol`, `max_iter` and `aggregate` settings of `m.options` and no standard errors. The
class labels of every replicate are matched to `m` by the permutation that minimizes the
squared distance between the response probabilities and class sizes (exactly for up to
7 classes). The seeds of the replicates are drawn from `rng` up front and every replicate
runs on its own generator, so the result is reproducible for a given `rng` and identical
with `multithreaded=true`, which runs the replicates on all Julia threads.

The per-replicate warnings of `fit` are silenced; one aggregated warning reports
replicates that did not converge, that hit the boundary or an empty class, or whose fit
failed (a resampled covariate that became constant or collinear; their rows are `NaN`
and excluded from the summaries).

# Arguments
- `m::LCAModel`: fitted model
- `n_boot::Integer=200`: number of replicates (at least 2)
- `rng::AbstractRNG=Random.default_rng()`
- `parametric::Bool=false`: simulate from the model instead of resampling rows
- `n_starts::Integer=1`: starts per replicate fit (the warm start plus random starts)
- `multithreaded::Bool=false`: run the replicates on all threads

# Returns
- [`LCABootstrap`](@ref)

# Example
```julia
b = bootstrap(m; n_boot=500, rng=StableRNG(1))
stderror(b)                      # bootstrap standard errors of coef(m)
confint(b)                       # percentile intervals
coeftable(b; which=:class)       # estimates with bootstrap standard errors
DataFrame(profiles(b))           # response probabilities with percentile intervals
```
"""
function bootstrap(m::LCAModel; n_boot::Integer=200, rng::AbstractRNG=Random.default_rng(),
                   parametric::Bool=false, n_starts::Integer=1, multithreaded::Bool=false)
    n_boot >= 2 || throw(ArgumentError("n_boot must be at least 2, got $n_boot"))
    n_starts >= 1 || throw(ArgumentError("n_starts must be at least 1, got $n_starts"))
    layout = ParamLayout(m)
    n = nobs(m)
    withcov = hascovariates(m)
    X = withcov ? m.data.X : nothing
    mask = (parametric && hasmissing(m)) ? (m.data.y .== 0) : nothing
    seeds = rand(rng, UInt64, n_boot)
    coefs = Matrix{Float64}(undef, n_boot, layout.n_total)
    converged = fill(false, n_boot)
    failed = fill(false, n_boot)
    flagged = fill(false, n_boot)
    _foreach_replicate(n_boot, multithreaded) do b
        rng_b = Xoshiro(seeds[b])
        d_b = parametric ? _simulate(rng_b, m, n, X, mask)[1] : _resample(rng_b, m.data)
        m_b = try
            _fit_quietly(d_b, m.n_classes, rng_b, withcov, m, n_starts, n_starts, m.options)
        catch err
            err isa ArgumentError || rethrow()
            nothing
        end
        if m_b === nothing
            coefs[b, :] .= NaN
            failed[b] = true
        else
            coefs[b, :] = _pack(_aligned_params(m_b, m), layout)
            converged[b] = m_b.converged
            flagged[b] = m_b.flags.n_boundary > 0 || !isempty(m_b.flags.empty_classes)
        end
    end
    msgs = String[]
    n_fail = count(failed)
    n_fail > 0 && push!(msgs, "$n_fail replicate fit(s) failed (a resampled covariate was " *
                              "constant or collinear); their coefficients are NaN and excluded")
    n_nc = count(b -> !failed[b] && !converged[b], 1:n_boot)
    n_nc > 0 && push!(msgs, "$n_nc replicate fit(s) did not converge within $(m.options.max_iter) iterations")
    n_flag = count(flagged)
    n_flag > 0 && push!(msgs, "$n_flag replicate fit(s) have response probabilities on the boundary or an empty class")
    isempty(msgs) || @warn "bootstrap of the $(m.n_classes)-class model: " * join(msgs, "; ")
    return LCABootstrap(m, Int(n_boot), coefs, converged)
end

# Indices of the replicates whose coefficients are all finite.
_finite_rows(R::AbstractMatrix) = [i for i in 1:size(R, 1) if all(isfinite, view(R, i, :))]

"""
    vcov(b::LCABootstrap) -> Matrix{Float64}

Bootstrap covariance matrix of [`coef`](@ref)`(b.model)`: the sample covariance of the
aligned replicates (`dof × dof`). Replicates with a non-finite coefficient are left out;
the matrix is all `NaN` when fewer than two replicates remain.
"""
function StatsAPI.vcov(b::LCABootstrap)
    rows = _finite_rows(b.coefs)
    p = size(b.coefs, 2)
    length(rows) >= 2 || return fill(NaN, p, p)
    return cov(b.coefs[rows, :]; dims=1)
end

"""
    stderror(b::LCABootstrap) -> Vector{Float64}

Bootstrap standard errors of [`coef`](@ref)`(b.model)`, the standard deviations of the
aligned replicates (`sqrt.(diag(vcov(b)))`).
"""
StatsAPI.stderror(b::LCABootstrap) = sqrt.(diag(vcov(b)))

"""
    confint(b::LCABootstrap; level=0.95, method=:percentile) -> Matrix{Float64}

Bootstrap confidence intervals of [`coef`](@ref)`(b.model)` on the logit scale, a
`dof × 2` matrix of lower and upper bounds. `method=:percentile` (the default) takes the
`(1 - level)/2` and `1 - (1 - level)/2` quantiles of the replicates of every parameter
(the finite ones); `method=:normal` is the Wald interval `coef ± z·stderror(b)` with the
bootstrap standard error. For intervals of the response probabilities use
[`profiles`](@ref profiles(::LCABootstrap)).
"""
function StatsAPI.confint(b::LCABootstrap; level::Real=0.95, method::Symbol=:percentile)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    method in (:percentile, :normal) ||
        throw(ArgumentError("method must be :percentile or :normal, got $(repr(method))"))
    p = size(b.coefs, 2)
    if method === :normal
        z = _zquantile(level)
        c = coef(b.model)
        se = stderror(b)
        return hcat(c .- z .* se, c .+ z .* se)
    end
    α = (1 - level) / 2
    out = fill(NaN, p, 2)
    for i in 1:p
        col = filter(isfinite, view(b.coefs, :, i))
        length(col) >= 2 || continue
        out[i, 1] = quantile(col, α)
        out[i, 2] = quantile(col, 1 - α)
    end
    return out
end

"""
    coeftable(b::LCABootstrap; level=0.95, which=:all) -> StatsBase.CoefTable

Coefficient table of [`coef`](@ref)`(b.model)` with the columns `Estimate` (the model's
estimate), `Std. Error` (bootstrap), `z` and `Pr(>|z|)` (the Wald test against zero with
the bootstrap standard error) and the bounds of the `level` percentile interval. `which`
selects the rows as in [`coeftable`](@ref coeftable(::LCAModel)): `:all`, `:class` or
`:items`.
"""
function StatsAPI.coeftable(b::LCABootstrap; level::Real=0.95, which::Symbol=:all)
    which in (:all, :class, :items) ||
        throw(ArgumentError("which must be :all, :class or :items, got $(repr(which))"))
    m = b.model
    c = coef(m)
    se = stderror(b)
    ci = confint(b; level=level)
    names = coefnames(m)
    n_class = (m.n_classes - 1) * size(m.beta, 1)
    idx = which === :all ? (1:length(c)) : which === :class ? (1:n_class) : (n_class + 1:length(c))
    z = c[idx] ./ se[idx]
    pv = 2 .* Distributions.ccdf.(Distributions.Normal(), abs.(z))
    pctstr = _level_string(level)
    return StatsBase.CoefTable(
        [c[idx], se[idx], z, pv, ci[idx, 1], ci[idx, 2]],
        ["Estimate", "Std. Error", "z", "Pr(>|z|)", "Lower $pctstr%", "Upper $pctstr%"],
        names[idx], 4, 3)
end

"""
    profiles(b::LCABootstrap; level=0.95, classes=false) -> Vector{NamedTuple}

Item-response profiles of `b.model` with bootstrap uncertainty, in the row-table format
of [`profiles`](@ref profiles(::LCAModel)) (`item`, `level`, `class`, `prob`, `se`,
`lower`, `upper`). `prob` is the model's estimate; the replicate coefficients are mapped
back to probabilities, so that `se` is the standard deviation of the replicate
probabilities and `lower`/`upper` are their `(1 - level)/2` and `1 - (1 - level)/2`
percentiles (which lie within `[0, 1]` without any transformation). Replicates with a
non-finite coefficient are left out.

With `classes=true` the table starts with one row per class holding its size; for a
model with covariates the replicate class sizes are the sample averages of the
covariate-specific membership probabilities over the training covariates, so they get a
bootstrap standard error (unlike the delta method of `profiles(m)`).
"""
function profiles(b::LCABootstrap; level::Real=0.95, classes::Bool=false)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    m = b.model
    layout = ParamLayout(m)
    K, J = m.n_classes, m.n_items
    rows_ok = _finite_rows(b.coefs)
    R = length(rows_ok)
    class_reps = Matrix{Float64}(undef, R, K)
    item_reps = [Array{Float64}(undef, R, K, C) for C in layout.C]
    θ = _params_buffer(layout)
    for (r, row) in enumerate(rows_ok)
        _unpack!(θ, view(b.coefs, row, :), layout)
        if layout.covariates
            class_reps[r, :] = mean(_class_prior(θ.coefs[:, 2:K], m.data.X), dims=1)
        else
            class_reps[r, :] = θ.class_probs
        end
        for j in 1:J
            item_reps[j][r, :, :] = θ.item_probs[j]
        end
    end
    α = (1 - level) / 2
    summary(v) = R >= 2 ? (std(v), quantile(v, α), quantile(v, 1 - α)) : (NaN, NaN, NaN)
    rows = _ProfileRow[]
    if classes
        for k in 1:K
            se, lo, hi = K == 1 ? (0.0, 1.0, 1.0) : summary(view(class_reps, :, k))
            push!(rows, (item=:class, level=string(k), class=k, prob=m.class_probs[k],
                         se=se, lower=lo, upper=hi))
        end
    end
    for j in 1:J, c in 1:m.n_categories[j], k in 1:K
        se, lo, hi = summary(view(item_reps[j], :, k, c))
        push!(rows, (item=m.data.item_names[j], level=m.data.item_levels[j][c], class=k,
                     prob=m.item_probs[j][k, c], se=se, lower=lo, upper=hi))
    end
    return rows
end

# ---------------------------------------------------------------------------------------
# Bootstrap likelihood-ratio test
# ---------------------------------------------------------------------------------------

_same_data(a::LCAData, b::LCAData) =
    a === b || (a.y == b.y && a.n_categories == b.n_categories && a.X == b.X)

# Parameters of a fitted model with the coefficients on the standardized scale of the
# data `d` (the scale the EM workspace of a fit to `d` uses), for `_init_split`.
function _params_on(m::LCAModel, d::LCAData)
    K = m.n_classes
    coefs = nothing
    if hascovariates(m) && K > 1
        _, A = _standardize(d.X; names=d.covariate_names)
        coefs = A \ hcat(zeros(size(m.beta, 1)), m.beta)
    end
    return LCAParams(copy(m.class_probs), [copy(B) for B in m.item_probs], coefs)
end

"""
    bootstrap_lrt(null::LCAModel, alternative::LCAModel; n_boot=100,
                  rng=Random.default_rng(), n_starts_boot=10, n_final_boot=2,
                  multithreaded=false) -> BootstrapLRT
    bootstrap_lrt(d::LCAData, k::Integer; rng=Random.default_rng(), n_boot=100,
                  n_starts_boot=10, n_final_boot=2, multithreaded=false, kwargs...)

Parametric bootstrap likelihood-ratio test (McLachlan, 1987) of a `K`-class model against
the `K + 1`-class model fitted to the same data, the test recommended by Nylund, Asparouhov
and Muthén (2007) for choosing the number of classes. The observed statistic is
`T = 2(ll_{K+1} - ll_K)`. Because the `K`-class model lies on the boundary of the
`K + 1`-class parameter space, `T` does not follow a chi-squared distribution under the
null hypothesis; its distribution is obtained by simulation instead: `n_boot` data sets
are drawn from the fitted `null` model with [`simulate`](@ref) (with the model's covariates
and the missingness pattern of the training data), each is fitted with `K` and with
`K + 1` classes, and the bootstrap p-value is

    p = (1 + #{T_b ≥ T}) / (n_boot + 1),

the fraction of replicates at least as extreme as the observed statistic (with the
observed data counted as one replicate, so that `p > 0` and the test is exact). A small
`p` favours `K + 1` classes. The resolution of the p-value is `1/(n_boot + 1)`, so choose
`n_boot` with the significance level in mind: `n_boot = 99` resolves 0.01, and several
hundred replicates are advisable when `p` is near the threshold.

For every replicate the `K`-class fit is warm-started from the null model plus two random
starts (all continued to convergence), and the `K + 1`-class fit from the replicate's
`K`-class solution with its largest class split in two (see `_init_split`) plus
`n_starts_boot` random starts, of which the `n_final_boot` best are continued. The split
start makes the `K + 1`-class log-likelihood of the replicate at least that of its
`K`-class fit up to the split perturbation, so the replicate statistics are non-negative
in a clean run; should the `K + 1`-class fit nevertheless end below the `K`-class fit, the
split start alone is continued to convergence as a fallback, and any statistic that is
still below `-1e-6` is counted in `n_negative` with a warning. A warning is also issued
when the alternative model's best log-likelihood was not replicated across its starts (its
statistic may then be understated; refit with more starts first). The replicate fits use
the `tol`, `max_iter` and `aggregate` settings of the two models and no standard errors;
their warnings are silenced. The seeds of the replicates are drawn from `rng` up front, so
the result is reproducible for a given `rng` and identical with `multithreaded=true`.

The second form fits the `k`- and `(k + 1)`-class models with [`fit`](@ref) (`kwargs...`
are passed on, for example `n_starts`), consuming `rng` for the two fits before the
replicates, and calls the first form; `k == 1` tests one class against two.

# Arguments
- `null::LCAModel`, `alternative::LCAModel`: models with `K` and `K + 1` classes fitted
  to the same data (an `ArgumentError` otherwise), both with or both without covariates
- `n_boot::Integer=100`: number of bootstrap replicates
- `rng::AbstractRNG=Random.default_rng()`
- `n_starts_boot::Integer=10`: random starts of every `K + 1`-class replicate fit, in
  addition to the split start
- `n_final_boot::Integer=2`: starts of every `K + 1`-class replicate fit continued to
  convergence
- `multithreaded::Bool=false`: run the replicates on all threads

# Returns
- [`BootstrapLRT`](@ref); read the p-value with [`pvalue`](@ref)

# Example
```julia
models = fit(LCAModel, d, 1:4; rng=StableRNG(1))
t = bootstrap_lrt(models[2], models[3]; n_boot=199, rng=StableRNG(2))   # 2 vs 3 classes
pvalue(t)
bootstrap_lrt(d, 1; n_boot=99, rng=StableRNG(3))                        # 1 vs 2 classes
```
"""
function bootstrap_lrt(null::LCAModel, alt::LCAModel; n_boot::Integer=100,
                       rng::AbstractRNG=Random.default_rng(), n_starts_boot::Integer=10,
                       n_final_boot::Integer=2, multithreaded::Bool=false)
    K = null.n_classes
    alt.n_classes == K + 1 || throw(ArgumentError(
        "the alternative model must have one class more than the null model; got $(alt.n_classes) and $K classes"))
    _same_data(null.data, alt.data) ||
        throw(ArgumentError("the two models must be fitted to the same data"))
    hascovariates(null) == hascovariates(alt) || throw(ArgumentError(
        "the two models must both be fitted with covariates or both without"))
    n_boot >= 1 || throw(ArgumentError("n_boot must be at least 1, got $n_boot"))
    n_starts_boot >= 1 || throw(ArgumentError("n_starts_boot must be at least 1, got $n_starts_boot"))
    n_final_boot >= 1 || throw(ArgumentError("n_final_boot must be at least 1, got $n_final_boot"))

    T_obs = 2 * (loglikelihood(alt) - loglikelihood(null))
    alt.flags.best_ll_replicated || @warn "the best log-likelihood of the $(K + 1)-class model " *
        "was reached by only one of its continued starts; the observed statistic may be " *
        "understated. Refit the alternative model with more starts (n_starts, n_final) before testing"
    T_obs < -1e-6 && @warn "the $(K + 1)-class model has a lower log-likelihood than the " *
        "$K-class model (statistic $T_obs); its fit is a poor local maximum"

    withcov = hascovariates(null)
    n = nobs(null)
    X = withcov ? null.data.X : nothing
    mask = hasmissing(null) ? (null.data.y .== 0) : nothing
    seeds = rand(rng, UInt64, n_boot)
    T = Vector{Float64}(undef, n_boot)
    converged = Vector{Bool}(undef, n_boot)
    _foreach_replicate(n_boot, multithreaded) do b
        rng_b = Xoshiro(seeds[b])
        d_b, _ = _simulate(rng_b, null, n, X, mask)
        m0 = _fit_quietly(d_b, K, rng_b, withcov, null, 3, 3, null.options)
        θ0 = _params_on(m0, d_b)
        θsplit = _init_split(θ0, argmax(θ0.class_probs), rng_b)
        m1 = _fit_quietly(d_b, K + 1, rng_b, withcov, θsplit, n_starts_boot + 1, n_final_boot,
                          alt.options)
        ll1, conv1 = m1.loglik, m1.converged
        if ll1 < m0.loglik
            # Fallback: the split start alone, continued to convergence
            m1b = _fit_quietly(d_b, K + 1, rng_b, withcov, θsplit, 1, 1, alt.options)
            if m1b.loglik > ll1
                ll1, conv1 = m1b.loglik, m1b.converged
            end
        end
        T[b] = 2 * (ll1 - m0.loglik)
        converged[b] = m0.converged && conv1
    end
    n_ge = count(t -> t >= T_obs, T)
    p = (1 + n_ge) / (n_boot + 1)
    n_negative = count(t -> t < -1e-6, T)
    n_negative > 0 && @warn "$n_negative of $n_boot bootstrap statistics are negative: the " *
        "$(K + 1)-class fit of those replicates ended below their $K-class fit; increase " *
        "n_starts_boot and n_final_boot"
    n_nc = count(!, converged)
    n_nc > 0 && @warn "$n_nc of $n_boot replicate fits did not converge within max_iter " *
        "iterations (EM converges slowly when the extra class is superfluous); their " *
        "statistics are kept"
    return BootstrapLRT(null, alt, T_obs, T, p, Int(n_boot), n_negative, converged)
end

function bootstrap_lrt(d::LCAData, k::Integer; rng::AbstractRNG=Random.default_rng(),
                       n_boot::Integer=100, n_starts_boot::Integer=10,
                       n_final_boot::Integer=2, multithreaded::Bool=false, kwargs...)
    null = fit(LCAModel, d, k; rng=rng, multithreaded=multithreaded, kwargs...)
    alt = fit(LCAModel, d, k + 1; rng=rng, multithreaded=multithreaded, kwargs...)
    return bootstrap_lrt(null, alt; n_boot=n_boot, rng=rng, n_starts_boot=n_starts_boot,
                         n_final_boot=n_final_boot, multithreaded=multithreaded)
end

"""
    pvalue(t::BootstrapLRT) -> Float64

Bootstrap p-value of a [`BootstrapLRT`](@ref), `(1 + #{replicates ≥ statistic}) / (n_boot + 1)`.
"""
StatsAPI.pvalue(t::BootstrapLRT) = t.pvalue
