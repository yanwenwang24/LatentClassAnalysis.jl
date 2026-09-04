# Latent class regression: the multinomial-logit class-membership model
# log(π_k(x) / π_1(x)) = x'β_k. Covariate standardization and its back-transform, the
# log-prior hook of the E-step, the damped Newton M-step on the coefficients (generalized
# EM), and the raw-scale membership probabilities used by fit, predict and simulate.

# |β| on the standardized scale beyond which quasi-complete separation is flagged.
const COEF_DIVERGENCE_THRESHOLD = 20.0
# Largest change of a single coefficient (standardized scale) in one M-step.
const NEWTON_STEP_CAP = 10.0
const NEWTON_MAX_HALVINGS = 20

"""
    _standardize(X; names=nothing) -> (Xst, A)

Standardize the columns of the `n × P` design matrix `X`, whose first column must be the
intercept and whose other columns must not be constant: `xs_j = (x_j - m_j) / s_j` with the
sample mean `m_j` and standard deviation `s_j`. Returns the standardized design transposed
(`Xst` is `P × n`) and the `P × P` back-transform `A` with `A[1,1] = 1`,
`A[1,j] = -m_j / s_j` and `A[j,j] = 1 / s_j`, so that a coefficient column on the
standardized scale maps to the raw scale as `β_raw = A * β_std` (and
`x'β_raw == xs'β_std` for every row). A constant covariate or a rank-deficient design
(checked with a pivoted QR) throws an `ArgumentError`; `names` labels the columns in the
message. Internal.
"""
function _standardize(X::AbstractMatrix{<:Real}; names=nothing)
    n, P = size(X)
    P >= 1 || throw(ArgumentError("the design matrix must contain the intercept column"))
    all(isone, view(X, :, 1)) ||
        throw(ArgumentError("the first column of the design matrix must be the intercept (all ones)"))
    label(j) = names === nothing ? "column $j of the design matrix" : "covariate $(names[j])"
    A = zeros(P, P)
    A[1, 1] = 1.0
    Xst = Matrix{Float64}(undef, P, n)
    Xst[1, :] .= 1.0
    for j in 2:P
        col = view(X, :, j)
        m = mean(col)
        ss = 0.0
        for x in col
            ss += (x - m)^2
        end
        s = n > 1 ? sqrt(ss / (n - 1)) : 0.0
        (isfinite(s) && s > 0) || throw(ArgumentError(
            "$(label(j)) is constant; drop it (the intercept is added automatically)"))
        A[1, j] = -m / s
        A[j, j] = 1 / s
        for i in 1:n
            Xst[j, i] = (col[i] - m) / s
        end
    end
    if P > 1
        R = qr(permutedims(Xst), ColumnNorm()).R
        tol = max(n, P) * eps(Float64) * abs(R[1, 1])
        rank = count(i -> abs(R[i, i]) > tol, 1:min(n, P))
        rank == P || throw(ArgumentError(
            "the covariates are collinear: the design matrix has $P columns but rank $rank; drop redundant covariates"))
    end
    return Xst, A
end

# Linear predictors η = coefs' * X (K × U) into `eta`.
function _eta!(eta::AbstractMatrix{Float64}, coefs::AbstractMatrix{Float64}, X::AbstractMatrix{Float64})
    (size(eta) == (size(coefs, 2), size(X, 2)) && size(coefs, 1) == size(X, 1)) ||
        throw(DimensionMismatch("coefficients of size $(size(coefs)) do not match a $(size(X, 1))-column design with $(size(X, 2)) rows"))
    return mul!(eta, transpose(coefs), X)
end

# Softmax of column u of eta into π; returns the log-sum-exp of the column.
@inline function _softmax_col!(π::AbstractVector{Float64}, eta::AbstractMatrix{Float64},
                               u::Integer, K::Integer)
    @inbounds begin
        m = eta[1, u]
        for k in 2:K
            m = max(m, eta[k, u])
        end
        s = 0.0
        for k in 1:K
            e = exp(eta[k, u] - m)
            π[k] = e
            s += e
        end
        for k in 1:K
            π[k] /= s
        end
        return m + log(s)
    end
end

# Log prior of every class for (unique) row `u` from the cached linear predictors
# `ws.eta` (filled by `estep!` from `θ.coefs`), written into `w`.
function _logprior!(w::AbstractVector{Float64}, θ::LCAParams, ws::LCAWorkspace, u::Integer)
    eta = ws.eta
    K = ws.K
    @inbounds begin
        m = eta[1, u]
        for k in 2:K
            m = max(m, eta[k, u])
        end
        s = 0.0
        for k in 1:K
            s += exp(eta[k, u] - m)
        end
        lse = m + log(s)
        for k in 1:K
            w[k] = eta[k, u] - lse
        end
    end
    return w
end

"""
    _coef_objective(ws, coefs) -> Q

Expected complete-data log-likelihood of the class-membership model at `coefs` (`P × K`
on the standardized scale, column 1 zero) given the posterior in `ws.post`:
`Q(β) = Σ_u f_u Σ_k post[k,u] log π_k(x_u; β)`. Overwrites the scratch buffers `ws.eta2`
and `ws.w`. Internal.
"""
function _coef_objective(ws::LCAWorkspace, coefs::AbstractMatrix{Float64})
    K, U = ws.K, ws.U
    eta, post, freq, π = ws.eta2, ws.post, ws.freq, ws.w
    _eta!(eta, coefs, ws.Xst)
    q = 0.0
    @inbounds for u in 1:U
        lse = _softmax_col!(π, eta, u, K)
        s = 0.0
        for k in 1:K
            s += post[k, u] * (eta[k, u] - lse)
        end
        q += freq[u] * s
    end
    return q
end

"""
    _coef_derivatives!(g, H, ws, coefs) -> Q

Gradient `g` (length `(K - 1)·P`) and Hessian `H` of [`_coef_objective`](@ref) with
respect to the free coefficients `vec(coefs[:, 2:K])` (column-major: the `P` coefficients
of class 2 first), evaluated at `coefs`; returns the objective itself. With
`π[k,u] = softmax(coefs' x_u)_k`,
`g_k = Σ_u f_u (post[k,u] - π[k,u]) x_u` and
`H_kl = -Σ_u f_u π[k,u] (δ_kl - π[l,u]) x_u x_u'`. Internal.
"""
function _coef_derivatives!(g::AbstractVector{Float64}, H::AbstractMatrix{Float64},
                            ws::LCAWorkspace, coefs::AbstractMatrix{Float64})
    K, U = ws.K, ws.U
    P = size(coefs, 1)
    dim = (K - 1) * P
    (length(g) == dim && size(H) == (dim, dim)) ||
        throw(DimensionMismatch("g and H must have $dim entries and be $dim × $dim"))
    eta, post, freq, X, π = ws.eta2, ws.post, ws.freq, ws.Xst, ws.w
    _eta!(eta, coefs, X)
    fill!(g, 0.0)
    fill!(H, 0.0)
    q = 0.0
    @inbounds for u in 1:U
        f = freq[u]
        lse = _softmax_col!(π, eta, u, K)
        s = 0.0
        for k in 1:K
            s += post[k, u] * (eta[k, u] - lse)
        end
        q += f * s
        for k in 2:K
            r = f * (post[k, u] - π[k])
            off = (k - 2) * P
            for p in 1:P
                g[off + p] += r * X[p, u]
            end
        end
        for l in 2:K
            offl = (l - 2) * P
            for k in 2:K
                offk = (k - 2) * P
                c = -f * π[k] * ((k == l ? 1.0 : 0.0) - π[l])
                for r in 1:P
                    xr = c * X[r, u]
                    for p in 1:P
                        H[offk + p, offl + r] += xr * X[p, u]
                    end
                end
            end
        end
    end
    return q
end

# Average class-membership probabilities over the (weighted) rows at `coefs`.
function _mean_prior!(class_probs::AbstractVector{Float64}, ws::LCAWorkspace,
                      coefs::AbstractMatrix{Float64})
    K, U = ws.K, ws.U
    eta, freq, π = ws.eta2, ws.freq, ws.w
    _eta!(eta, coefs, ws.Xst)
    fill!(class_probs, 0.0)
    total = 0.0
    @inbounds for u in 1:U
        _softmax_col!(π, eta, u, K)
        f = freq[u]
        total += f
        for k in 1:K
            class_probs[k] += f * π[k]
        end
    end
    class_probs ./= total
    return class_probs
end

"""
    _update_coefs!(θ, ws)

M-step of the class-membership model: one damped Newton step on the coefficients
(generalized EM), so the observed log-likelihood never decreases. Solves
`(-H + λI) Δ = g` with the ridge `λ = 1e-6·max(1, tr(-H)/dim)` by a Cholesky
factorization, caps `max|Δ|` at `NEWTON_STEP_CAP` on the standardized scale, and halves
the step until the objective [`_coef_objective`](@ref) does not decrease (at most
`NEWTON_MAX_HALVINGS` halvings; otherwise the coefficients are kept). The ridge and the
cap only affect the path: the fixed point is the exact maximizer. `θ.class_probs` is then
set to the average membership probabilities over the rows. Internal.
"""
function _update_coefs!(θ::LCAParams, ws::LCAWorkspace)
    coefs = θ.coefs
    P, K = size(coefs)
    dim = (K - 1) * P
    if dim > 0
        g = Vector{Float64}(undef, dim)
        H = Matrix{Float64}(undef, dim, dim)
        q0 = _coef_derivatives!(g, H, ws, coefs)
        tr = 0.0
        @inbounds for i in 1:dim
            tr -= H[i, i]
        end
        λ = 1e-6 * max(1.0, tr / dim)
        M = -H
        @inbounds for i in 1:dim
            M[i, i] += λ
        end
        F = cholesky(Symmetric(M); check=false)
        if issuccess(F) && all(isfinite, g) && isfinite(q0)
            Δ = F \ g
            mx = maximum(abs, Δ)
            mx > NEWTON_STEP_CAP && (Δ .*= NEWTON_STEP_CAP / mx)
            trial = similar(coefs)
            t = 1.0
            for _ in 0:NEWTON_MAX_HALVINGS
                copyto!(trial, coefs)
                @inbounds for k in 2:K, p in 1:P
                    trial[p, k] += t * Δ[(k - 2) * P + p]
                end
                q = _coef_objective(ws, trial)
                if q >= q0
                    copyto!(coefs, trial)
                    break
                end
                t /= 2
            end
        end
    end
    _mean_prior!(θ.class_probs, ws, coefs)
    return θ
end

"""
    _class_prior(beta, X) -> Matrix{Float64}

Class-membership probabilities `softmax(x_i' [0 beta])` of every row of the raw design `X`
(`n × P`, intercept first) for raw-scale coefficients `beta` (`P × (K - 1)`, class 1 as
reference): an `n × K` matrix whose rows sum to one. Used for the class sizes of a fitted
model, for prediction on new data and for simulation. Internal.
"""
function _class_prior(beta::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real})
    n, P = size(X)
    size(beta, 1) == P ||
        throw(DimensionMismatch("beta has $(size(beta, 1)) rows but the design has $P columns"))
    K = size(beta, 2) + 1
    eta = X * beta                      # n × (K - 1)
    prior = Matrix{Float64}(undef, n, K)
    @inbounds for i in 1:n
        m = 0.0
        for k in 1:K-1
            m = max(m, eta[i, k])
        end
        s = exp(-m)
        prior[i, 1] = s
        for k in 1:K-1
            e = exp(eta[i, k] - m)
            prior[i, k + 1] = e
            s += e
        end
        for k in 1:K
            prior[i, k] /= s
        end
    end
    return prior
end
