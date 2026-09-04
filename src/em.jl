# EM core: parameter bundle, preallocated workspace with pattern aggregation, missing-aware
# E-step (log-sum-exp), sufficient statistics, M-step, and the single-run EM loop.

const PROB_FLOOR = 1e-10

"""
    LCAParams(class_probs, item_probs, coefs)

Mutable parameter bundle used as the EM state. `class_probs` has length `K`,
`item_probs[j]` is `K × C_j`, and `coefs` is either `nothing` (no covariates) or the
`P × K` multinomial-logit coefficient matrix on the standardized covariate scale
(column 1 is identically zero). Internal.
"""
mutable struct LCAParams
    class_probs::Vector{Float64}
    item_probs::Vector{Matrix{Float64}}
    coefs::Union{Nothing,Matrix{Float64}}
end

LCAParams(class_probs, item_probs) = LCAParams(class_probs, item_probs, nothing)

Base.copy(θ::LCAParams) = LCAParams(copy(θ.class_probs), [copy(P) for P in θ.item_probs],
                                    θ.coefs === nothing ? nothing : copy(θ.coefs))

"""
    LCAWorkspace(d::LCAData, K; aggregate=true, covariates=false, n_categories=d.n_categories)
    LCAWorkspace(ws::LCAWorkspace)

Preallocated buffers for EM on `d` with `K` classes. The data are stored transposed
(`yt` is `J × U`) so inner loops walk contiguous memory. With `aggregate=true` and
`covariates=false` identical response patterns are collapsed into `U` unique rows with
weights `freq` and a `row_index` mapping every observation to its pattern; with
`covariates=true` every row is kept and `Xt` holds the transposed covariate matrix.
`n_categories` sizes the per-item buffers (a model may have more categories than a
held-out data set shows). The second form shares the (immutable) data buffers of `ws` and
allocates fresh scratch buffers, for use by another thread. Internal.
"""
struct LCAWorkspace
    K::Int
    J::Int
    U::Int              # number of (unique) rows
    n::Int              # number of observations
    C::Vector{Int}
    yt::Matrix{Int}     # J × U codes, 0 = missing
    freq::Vector{Float64}
    row_index::Vector{Int}
    Xt::Matrix{Float64} # P × U covariates (intercept first); P == 1 when aggregated
    aggregated::Bool
    # scratch
    post::Matrix{Float64}          # K × U posterior
    logpi::Vector{Float64}         # K
    logB::Vector{Matrix{Float64}}  # K × C_j
    Nk::Vector{Float64}            # K
    Njkc::Vector{Matrix{Float64}}  # K × C_j
    w::Vector{Float64}             # K
end

function LCAWorkspace(d::LCAData, K::Integer; aggregate::Bool=true, covariates::Bool=false,
                      n_categories::AbstractVector{<:Integer}=d.n_categories)
    n, J = size(d.y)
    length(n_categories) == J ||
        throw(ArgumentError("n_categories has $(length(n_categories)) entries, expected $J"))
    C = Int.(n_categories)
    covariates && !hascovariates(d) &&
        throw(ArgumentError("covariates=true requires data with covariates"))
    aggregated = aggregate && !covariates
    if aggregated
        index = Dict{Vector{Int},Int}()
        patterns = Vector{Vector{Int}}()
        row_index = Vector{Int}(undef, n)
        for i in 1:n
            key = d.y[i, :]
            u = get(index, key, 0)
            if u == 0
                push!(patterns, key)
                u = length(patterns)
                index[key] = u
            end
            row_index[i] = u
        end
        U = length(patterns)
        yt = Matrix{Int}(undef, J, U)
        for u in 1:U
            yt[:, u] = patterns[u]
        end
        freq = zeros(U)
        for i in 1:n
            freq[row_index[i]] += 1
        end
        Xt = ones(1, U)
    else
        U = n
        yt = permutedims(d.y)
        freq = ones(U)
        row_index = collect(1:n)
        Xt = permutedims(d.X)
    end
    return LCAWorkspace(K, J, U, n, C, yt, freq, row_index, Xt, aggregated,
                        Matrix{Float64}(undef, K, U), Vector{Float64}(undef, K),
                        [Matrix{Float64}(undef, K, c) for c in C],
                        Vector{Float64}(undef, K), [zeros(K, c) for c in C],
                        Vector{Float64}(undef, K))
end

function LCAWorkspace(ws::LCAWorkspace)
    K, U = ws.K, ws.U
    return LCAWorkspace(K, ws.J, U, ws.n, ws.C, ws.yt, ws.freq, ws.row_index, ws.Xt,
                        ws.aggregated,
                        Matrix{Float64}(undef, K, U), Vector{Float64}(undef, K),
                        [Matrix{Float64}(undef, K, c) for c in ws.C],
                        Vector{Float64}(undef, K), [zeros(K, c) for c in ws.C],
                        Vector{Float64}(undef, K))
end

"""
    estep!(ws, θ) -> ll

E-step: fill `ws.post` with the posterior class probabilities of every (unique) row under
`θ` and return the observed-data log-likelihood. Missing responses (code 0) are skipped;
the per-row normalization uses log-sum-exp. Internal.
"""
function estep!(ws::LCAWorkspace, θ::LCAParams)
    K, J, U = ws.K, ws.J, ws.U
    logpi, logB, w, post, yt, freq = ws.logpi, ws.logB, ws.w, ws.post, ws.yt, ws.freq
    hascoefs = θ.coefs !== nothing
    if !hascoefs
        @inbounds for k in 1:K
            logpi[k] = log(θ.class_probs[k])
        end
    end
    @inbounds for j in 1:J
        B = logB[j]
        P = θ.item_probs[j]
        for idx in eachindex(B, P)
            B[idx] = log(P[idx])
        end
    end
    ll = 0.0
    @inbounds for u in 1:U
        if hascoefs
            _logprior!(w, θ, ws, u)
        else
            for k in 1:K
                w[k] = logpi[k]
            end
        end
        for j in 1:J
            y = yt[j, u]
            y == 0 && continue
            B = logB[j]
            for k in 1:K
                w[k] += B[k, y]
            end
        end
        m = w[1]
        for k in 2:K
            m = max(m, w[k])
        end
        s = 0.0
        for k in 1:K
            s += exp(w[k] - m)
        end
        lse = m + log(s)
        for k in 1:K
            post[k, u] = exp(w[k] - lse)
        end
        ll += freq[u] * lse
    end
    return ll
end

"""
    _accumulate!(ws)

Weighted sufficient statistics from the current posterior: `Nk[k] = Σ_u f_u post[k,u]` and
`Njkc[j][k,c] = Σ_{u: y_ju == c} f_u post[k,u]` (missing cells add to nothing). Internal.
"""
function _accumulate!(ws::LCAWorkspace)
    K, J, U = ws.K, ws.J, ws.U
    Nk, Njkc, post, yt, freq = ws.Nk, ws.Njkc, ws.post, ws.yt, ws.freq
    fill!(Nk, 0.0)
    for N in Njkc
        fill!(N, 0.0)
    end
    @inbounds for u in 1:U
        f = freq[u]
        for k in 1:K
            Nk[k] += f * post[k, u]
        end
        for j in 1:J
            y = yt[j, u]
            y == 0 && continue
            N = Njkc[j]
            for k in 1:K
                N[k, y] += f * post[k, u]
            end
        end
    end
    return ws
end

"""
    _update!(θ, ws)

M-step from the accumulated statistics: class probabilities `Nk / ΣNk` (or one damped
Newton step on the covariate coefficients), item probabilities row-normalized per item
with a `1e-10` floor; an empty class row (no posterior mass among the observed responses
of that item) becomes uniform. Internal.
"""
function _update!(θ::LCAParams, ws::LCAWorkspace)
    K = ws.K
    if θ.coefs === nothing
        total = sum(ws.Nk)
        @inbounds for k in 1:K
            θ.class_probs[k] = ws.Nk[k] / total
        end
    else
        _update_coefs!(θ, ws)
    end
    @inbounds for j in 1:ws.J
        N = ws.Njkc[j]
        P = θ.item_probs[j]
        C = ws.C[j]
        for k in 1:K
            rowsum = 0.0
            for c in 1:C
                rowsum += N[k, c]
            end
            if rowsum <= 0
                for c in 1:C
                    P[k, c] = 1 / C
                end
            else
                s = 0.0
                for c in 1:C
                    p = N[k, c] / rowsum
                    p = p < PROB_FLOOR ? PROB_FLOOR : (p > 1.0 ? 1.0 : p)
                    P[k, c] = p
                    s += p
                end
                for c in 1:C
                    P[k, c] /= s
                end
            end
        end
    end
    return θ
end

"""
    _em!(θ, ws; max_iter, tol, ll_trace=nothing) -> (ll, iterations, converged)

Run EM in place on `θ`. Every iteration performs an E-step (which yields the
log-likelihood of the current parameters) and checks convergence *before* the M-step, so
the returned log-likelihood and `ws.post` correspond to the returned parameters.
Convergence is `|ll - ll_old| ≤ tol·(1 + |ll|)`. `iterations` counts M-steps; at most
`max_iter` are taken. Pass a vector as `ll_trace` to record the log-likelihood of every
E-step. Internal.
"""
function _em!(θ::LCAParams, ws::LCAWorkspace; max_iter::Integer, tol::Real,
              ll_trace::Union{Nothing,Vector{Float64}}=nothing)
    ll_old = -Inf
    ll = -Inf
    iterations = 0
    converged = false
    while true
        ll = estep!(ws, θ)
        ll_trace === nothing || push!(ll_trace, ll)
        if ll < ll_old
            @debug "log-likelihood decreased from $ll_old to $ll at iteration $iterations"
        end
        if abs(ll - ll_old) <= tol * (1 + abs(ll))
            converged = true
            break
        end
        iterations >= max_iter && break
        _accumulate!(ws)
        _update!(θ, ws)
        iterations += 1
        ll_old = ll
    end
    return ll, iterations, converged
end

# Closed-form single-class solution: posterior ≡ 1, item probabilities are the observed
# marginals of every item. Returns (θ, ll) with ws.post filled.
function _fit_single_class(ws::LCAWorkspace)
    ws.K == 1 || throw(ArgumentError("workspace has $(ws.K) classes"))
    θ = LCAParams([1.0], [fill(1 / c, 1, c) for c in ws.C], nothing)
    fill!(ws.post, 1.0)
    _accumulate!(ws)
    _update!(θ, ws)
    ll = estep!(ws, θ)
    return θ, ll
end

# Expand the K × U pattern posterior to the n × K observation posterior.
function _expand_posterior(ws::LCAWorkspace)
    post = Matrix{Float64}(undef, ws.n, ws.K)
    @inbounds for i in 1:ws.n
        u = ws.row_index[i]
        for k in 1:ws.K
            post[i, k] = ws.post[k, u]
        end
    end
    return post
end
