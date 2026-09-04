# Inference on the free-parameter (logit) scale: the parameter layout every numeric routine
# shares, packing/unpacking, the analytic score, the finite-difference observed information,
# the covariance matrix computed by `fit`, the StatsAPI verbs coef/coefnames/vcov/stderror/
# confint/coeftable/informationmatrix, and the delta-method standard errors of the class
# sizes and item-response profiles.

# A probability within BOUNDARY_TOL of 0 or 1 is on the boundary: its logit is held fixed
# in the information matrix and its standard error is NaN.
const BOUNDARY_TOL = 1e-6
# Relative step of the central finite differences of the score.
const FD_STEP = 6e-6

_interior(p::Real) = BOUNDARY_TOL < p < 1 - BOUNDARY_TOL

"""
    ParamLayout(class_probs, item_probs, coefs)
    ParamLayout(m::LCAModel)

Layout of the free-parameter vector `v` of a latent class model, the contract shared by
[`coef`](@ref), [`vcov`](@ref), the score and the information matrix. `v` holds the class
block first, then the items:

- class block, length `(K - 1)·P`: without covariates (`coefs === nothing`, `P == 1`) the
  log-odds `α_k = log(π_k / π_1)` for `k = 2, …, K`; with covariates `vec(coefs[:, 2:K])`
  (column-major, the `P` coefficients of class 2 first);
- items: for `j in 1:J`, `k in 1:K` and every category `c ≠ ref_cat[j][k]`, the logit
  `log(B_jkc / B_jk,ref)` against the reference category `ref_cat[j][k] = argmax` of row
  `k` of `item_probs[j]`. `row_start[j][k]` is the index of the first parameter of that row
  (the row has `C_j - 1` parameters, in increasing category order).

`free[i]` is `false` for a parameter whose probability (or whose reference probability)
lies within `1e-6` of 0 or 1, and for every item-response logit of an empty class
(`class_probs[k] ≤ 1e-6`, whose response probabilities are not estimable); such
parameters are held fixed in the information matrix. Coefficients on covariates are
always free. `n_total == dof` of the model. Internal.
"""
struct ParamLayout
    K::Int
    J::Int
    P::Int
    C::Vector{Int}
    covariates::Bool
    ref_cat::Vector{Vector{Int}}
    row_start::Vector{Vector{Int}}
    n_class::Int
    n_total::Int
    free::BitVector
end

function ParamLayout(class_probs::AbstractVector{<:Real},
                     item_probs::AbstractVector{<:AbstractMatrix{<:Real}},
                     coefs::Union{Nothing,AbstractMatrix{<:Real}})
    K = length(class_probs)
    J = length(item_probs)
    K >= 1 || throw(ArgumentError("at least one class is required"))
    C = [size(B, 2) for B in item_probs]
    for (j, B) in enumerate(item_probs)
        size(B, 1) == K || throw(DimensionMismatch("item_probs[$j] has $(size(B, 1)) rows, expected $K"))
    end
    covariates = coefs !== nothing
    P = covariates ? size(coefs, 1) : 1
    if covariates
        size(coefs, 2) == K ||
            throw(DimensionMismatch("coefs has $(size(coefs, 2)) columns, expected $K"))
    end
    n_class = (K - 1) * P
    ref_cat = [[argmax(view(B, k, :)) for k in 1:K] for B in item_probs]
    row_start = Vector{Vector{Int}}(undef, J)
    idx = n_class + 1
    for j in 1:J
        row_start[j] = Vector{Int}(undef, K)
        for k in 1:K
            row_start[j][k] = idx
            idx += C[j] - 1
        end
    end
    n_total = idx - 1
    free = trues(n_total)
    if !covariates
        for k in 2:K
            free[k - 1] = _interior(class_probs[k]) && _interior(class_probs[1])
        end
    end
    for j in 1:J
        B = item_probs[j]
        for k in 1:K
            empty = class_probs[k] <= BOUNDARY_TOL
            r = ref_cat[j][k]
            i = row_start[j][k]
            for c in 1:C[j]
                c == r && continue
                free[i] = !empty && _interior(B[k, c]) && _interior(B[k, r])
                i += 1
            end
        end
    end
    return ParamLayout(K, J, P, C, covariates, ref_cat, row_start, n_class, n_total, free)
end

# Numbers of parameters held fixed by the layout: (on the boundary, in an empty class).
function _fixed_counts(class_probs::AbstractVector{<:Real}, layout::ParamLayout)
    n_boundary = count(!, view(layout.free, 1:layout.n_class))
    n_empty = 0
    for k in 1:layout.K
        empty = class_probs[k] <= BOUNDARY_TOL
        for j in 1:layout.J
            idx = _row_indices(layout, j, k)
            if empty
                n_empty += length(idx)
            else
                n_boundary += count(!, view(layout.free, idx))
            end
        end
    end
    return n_boundary, n_empty
end

function ParamLayout(m::LCAModel)
    coefs = (hascovariates(m) && m.n_classes > 1) ? _raw_coefs(m) : nothing
    return ParamLayout(m.class_probs, m.item_probs, coefs)
end

# Indices of the parameters of row k of item j (its C_j - 1 logits).
_row_indices(layout::ParamLayout, j::Integer, k::Integer) =
    layout.row_start[j][k]:(layout.row_start[j][k] + layout.C[j] - 2)

"""
    _pack(θ::LCAParams, layout) -> Vector{Float64}

Free-parameter vector of `θ` in the order of [`ParamLayout`](@ref). Internal.
"""
function _pack(θ::LCAParams, layout::ParamLayout)
    K, P = layout.K, layout.P
    v = Vector{Float64}(undef, layout.n_total)
    if layout.covariates
        coefs = θ.coefs
        coefs === nothing && throw(ArgumentError(
            "the layout has covariate coefficients but the parameters carry none"))
        size(coefs) == (P, K) ||
            throw(DimensionMismatch("coefs has size $(size(coefs)), expected ($P, $K)"))
        idx = 1
        @inbounds for k in 2:K, p in 1:P
            v[idx] = coefs[p, k]
            idx += 1
        end
    else
        length(θ.class_probs) == K ||
            throw(DimensionMismatch("class_probs has length $(length(θ.class_probs)), expected $K"))
        @inbounds for k in 2:K
            v[k - 1] = log(θ.class_probs[k] / θ.class_probs[1])
        end
    end
    _pack_items!(v, θ.item_probs, layout)
    return v
end

# Item logits of `item_probs` into the item block of `v`.
function _pack_items!(v::AbstractVector{Float64}, item_probs::AbstractVector{<:AbstractMatrix{<:Real}},
                      layout::ParamLayout)
    length(item_probs) == layout.J ||
        throw(DimensionMismatch("item_probs has $(length(item_probs)) items, expected $(layout.J)"))
    @inbounds for j in 1:layout.J
        B = item_probs[j]
        size(B) == (layout.K, layout.C[j]) ||
            throw(DimensionMismatch("item_probs[$j] has size $(size(B)), expected ($(layout.K), $(layout.C[j]))"))
        for k in 1:layout.K
            r = layout.ref_cat[j][k]
            lr = log(B[k, r])
            i = layout.row_start[j][k]
            for c in 1:layout.C[j]
                c == r && continue
                v[i] = log(B[k, c]) - lr
                i += 1
            end
        end
    end
    return v
end

"""
    _unpack!(θ::LCAParams, v, layout) -> θ

Inverse of [`_pack`](@ref): write the class probabilities (or the coefficients, with
column 1 zero) and the row-normalized item probabilities encoded by `v` into `θ`. With
covariates `θ.class_probs` is left untouched. Internal.
"""
function _unpack!(θ::LCAParams, v::AbstractVector{<:Real}, layout::ParamLayout)
    K, P = layout.K, layout.P
    length(v) == layout.n_total ||
        throw(DimensionMismatch("v has length $(length(v)), expected $(layout.n_total)"))
    if layout.covariates
        coefs = θ.coefs
        coefs === nothing && throw(ArgumentError(
            "the layout has covariate coefficients but the parameters carry none"))
        size(coefs) == (P, K) ||
            throw(DimensionMismatch("coefs has size $(size(coefs)), expected ($P, $K)"))
        coefs[:, 1] .= 0.0
        idx = 1
        @inbounds for k in 2:K, p in 1:P
            coefs[p, k] = v[idx]
            idx += 1
        end
    else
        π = θ.class_probs
        m = 0.0
        @inbounds for k in 2:K
            m = max(m, v[k - 1])
        end
        s = exp(-m)
        π[1] = s
        @inbounds for k in 2:K
            e = exp(v[k - 1] - m)
            π[k] = e
            s += e
        end
        π ./= s
    end
    @inbounds for j in 1:layout.J
        B = θ.item_probs[j]
        C = layout.C[j]
        for k in 1:K
            r = layout.ref_cat[j][k]
            i0 = layout.row_start[j][k]
            m = 0.0
            i = i0
            for c in 1:C
                c == r && continue
                m = max(m, v[i])
                i += 1
            end
            s = 0.0
            i = i0
            for c in 1:C
                if c == r
                    e = exp(-m)
                else
                    e = exp(v[i] - m)
                    i += 1
                end
                B[k, c] = e
                s += e
            end
            for c in 1:C
                B[k, c] /= s
            end
        end
    end
    return θ
end

# Parameter buffer with the shapes of `layout`, for `_unpack!`.
_params_buffer(layout::ParamLayout) =
    LCAParams(zeros(layout.K), [zeros(layout.K, c) for c in layout.C],
              layout.covariates ? zeros(layout.P, layout.K) : nothing)

"""
    _score!(g, v, layout, θ, ws) -> ll

Analytic score of the observed-data log-likelihood at the free-parameter vector `v`,
written into `g` (length `layout.n_total`); returns the log-likelihood. Unpacks `v` into
the buffer `θ`, runs [`estep!`](@ref) and [`_accumulate!`](@ref), then
`∂ℓ/∂α_m = Nk[m] - n̄·π_m` (no covariates, `n̄ = Σ freq`), `∂ℓ/∂β` from
[`_coef_derivatives!`](@ref) (Fisher's identity: the gradient of `Q(β)` at the current
posterior), and `∂ℓ/∂γ_jkc = Njkc[j][k,c] - N_jk·B_jkc` for `c ≠ ref` with
`N_jk = Σ_c Njkc[j][k,c]`. Internal.
"""
function _score!(g::AbstractVector{Float64}, v::AbstractVector{<:Real}, layout::ParamLayout,
                 θ::LCAParams, ws::LCAWorkspace)
    length(g) == layout.n_total ||
        throw(DimensionMismatch("g has length $(length(g)), expected $(layout.n_total)"))
    (ws.K == layout.K && ws.J == layout.J && ws.C == layout.C) ||
        throw(DimensionMismatch("the workspace does not match the parameter layout"))
    _unpack!(θ, v, layout)
    ll = estep!(ws, θ)
    _accumulate!(ws)
    K = layout.K
    if layout.covariates
        dim = layout.n_class
        dim > 0 && _coef_gradient!(view(g, 1:dim), ws, θ.coefs)
    else
        nbar = sum(ws.freq)
        @inbounds for k in 2:K
            g[k - 1] = ws.Nk[k] - nbar * θ.class_probs[k]
        end
    end
    @inbounds for j in 1:layout.J
        N = ws.Njkc[j]
        B = θ.item_probs[j]
        C = layout.C[j]
        for k in 1:K
            Njk = 0.0
            for c in 1:C
                Njk += N[k, c]
            end
            r = layout.ref_cat[j][k]
            i = layout.row_start[j][k]
            for c in 1:C
                c == r && continue
                g[i] = N[k, c] - Njk * B[k, c]
                i += 1
            end
        end
    end
    return ll
end

"""
    _observed_information(v, layout, ws) -> Matrix{Float64}

Observed information matrix `-∂²ℓ/∂v∂v'` at `v`, restricted to the free parameters of
`layout` (`n_free × n_free`, symmetric): column `p` of the Hessian is the central finite
difference of the analytic score with step `h = 6e-6·max(1, |v_p|)`, and the result is
symmetrized as `-(H + H') / 2`. Costs `2·n_free` score evaluations. Internal.
"""
function _observed_information(v::AbstractVector{<:Real}, layout::ParamLayout, ws::LCAWorkspace)
    free = findall(layout.free)
    nf = length(free)
    θ = _params_buffer(layout)
    gp = Vector{Float64}(undef, layout.n_total)
    gm = Vector{Float64}(undef, layout.n_total)
    vp = Vector{Float64}(v)
    H = Matrix{Float64}(undef, nf, nf)
    for (a, p) in enumerate(free)
        h = FD_STEP * max(1.0, abs(v[p]))
        vp[p] = v[p] + h
        _score!(gp, vp, layout, θ, ws)
        vp[p] = v[p] - h
        _score!(gm, vp, layout, θ, ws)
        vp[p] = v[p]
        @inbounds for (b, q) in enumerate(free)
            H[b, a] = (gp[q] - gm[q]) / (2h)
        end
    end
    info = Matrix{Float64}(undef, nf, nf)
    @inbounds for a in 1:nf, b in 1:nf
        info[b, a] = -(H[b, a] + H[a, b]) / 2
    end
    return info
end

# Covariance of the free parameters at v (n_total × n_total, internal scale; NaN rows and
# columns for fixed parameters). Free parameters with numerically zero information
# (`_informative`, e.g. an item never observed in a class) are cleared from `layout.free`
# and masked too, so that the others keep finite standard errors. Returns
# (V, posdef, n_uninformative); V is all NaN when the information is not positive definite.
function _covariance(v::AbstractVector{<:Real}, layout::ParamLayout, ws::LCAWorkspace)
    n = layout.n_total
    V = fill(NaN, n, n)
    info = _observed_information(v, layout, ws)
    free = findall(layout.free)
    keep = _informative(info)
    n_uninformative = count(!, keep)
    if n_uninformative > 0
        for (a, p) in enumerate(free)
            keep[a] || (layout.free[p] = false)
        end
        info = info[keep, keep]
        free = free[keep]
    end
    posdef = all(isfinite, info)
    if posdef
        F = cholesky(Symmetric(info); check=false)
        posdef = issuccess(F)
        if posdef
            Vf = inv(F)
            @inbounds for (b, q) in enumerate(free), (a, p) in enumerate(free)
                V[p, q] = (Vf[a, b] + Vf[b, a]) / 2
            end
        end
    end
    return V, posdef, n_uninformative
end

# Diagonal entries of the observed information below this fraction of the largest one
# (and of 1) count as zero.
const INFO_ZERO_TOL = 1e-12

# `false` for parameters whose diagonal information is numerically zero. NaN or infinite
# entries count as informative so that they surface as a non-positive-definite matrix.
function _informative(info::AbstractMatrix{<:Real})
    nf = size(info, 1)
    scale = 1.0
    @inbounds for a in 1:nf
        d = info[a, a]
        isfinite(d) && (scale = max(scale, abs(d)))
    end
    return BitVector(!(isfinite(info[a, a]) && abs(info[a, a]) <= INFO_ZERO_TOL * scale) for a in 1:nf)
end

# Linear map from the internal parameter vector to the public coef scale: block diagonal
# with `A` (β_raw = A β_std) repeated for every class column of the class block and the
# identity for the items. The identity without covariates.
function _coef_transform(layout::ParamLayout, A::AbstractMatrix{<:Real})
    T = Matrix{Float64}(I, layout.n_total, layout.n_total)
    if layout.covariates && layout.n_class > 0
        P = layout.P
        size(A) == (P, P) || throw(DimensionMismatch("A has size $(size(A)), expected ($P, $P)"))
        for k in 2:layout.K
            off = (k - 2) * P
            T[off + 1:off + P, off + 1:off + P] = A
        end
    end
    return T
end

# Congruence transform `T M T'` of a matrix with NaN rows/columns for the fixed parameters
# (the NaN entries are excluded from the products and restored afterwards; T never couples
# a fixed parameter with a free one).
function _transform_masked(M::AbstractMatrix{<:Real}, T::AbstractMatrix{<:Real}, free::BitVector)
    n = length(free)
    idx = findall(free)
    out = fill(NaN, n, n)
    all(isfinite, view(M, idx, idx)) || return out
    Mz = zeros(n, n)
    Mz[idx, idx] = M[idx, idx]
    R = T * Mz * transpose(T)
    @inbounds for q in idx, p in idx
        out[p, q] = (R[p, q] + R[q, p]) / 2
    end
    return out
end

# Covariance on the public coef scale, `T V T'`.
_to_public_vcov(V::AbstractMatrix{<:Real}, layout::ParamLayout, A::AbstractMatrix{<:Real}) =
    _transform_masked(V, _coef_transform(layout, A), layout.free)

# Covariance matrix (public scale) and warning messages for `fit`.
function _fit_vcov(θ::LCAParams, ws::LCAWorkspace, opts::LCAOptions, diverged::Bool)
    opts.se == :hessian || return nothing, String[]
    layout = ParamLayout(θ.class_probs, θ.item_probs, θ.coefs)
    n = layout.n_total
    msgs = String[]
    if diverged
        push!(msgs, "standard errors are not computed for diverged coefficients and are NaN")
        return fill(NaN, n, n), msgs
    end
    v = _pack(θ, layout)
    nfix, n_empty = _fixed_counts(θ.class_probs, layout)
    V, posdef, n_zero = _covariance(v, layout, ws)
    posdef || push!(msgs, "observed information is not positive definite (weakly identified " *
                          "or non-converged model); standard errors are NaN")
    conditional = "the remaining standard errors are conditional on them being held fixed"
    if nfix > 0
        push!(msgs, nfix == 1 ?
              "1 parameter is on the boundary (0 or 1); its standard error is undefined and reported as NaN, " *
              "and the remaining standard errors are conditional on it being held fixed" :
              "$nfix parameters are on the boundary (0 or 1); their standard errors are undefined and reported as NaN, " *
              "and $conditional")
    end
    n_empty > 0 && push!(msgs, "the $n_empty response parameters of the empty class(es) are not estimable " *
                               "and have NaN standard errors; $conditional")
    n_zero > 0 && push!(msgs, "$n_zero parameter(s) have zero observed information (a class never observed " *
                              "with an item) and NaN standard errors; $conditional")
    return _to_public_vcov(V, layout, ws.A), msgs
end

# Workspace, internal parameters (coefficients on the standardized scale) and layout of a
# fitted model, for recomputing the score or the information matrix.
function _model_params(m::LCAModel)
    K = m.n_classes
    withcov = hascovariates(m) && K > 1
    ws = LCAWorkspace(m.data, K; aggregate=!withcov, covariates=withcov)
    coefs = withcov ? ws.A \ _raw_coefs(m) : nothing
    θ = LCAParams(copy(m.class_probs), [copy(B) for B in m.item_probs], coefs)
    return ws, θ, ParamLayout(θ.class_probs, θ.item_probs, θ.coefs)
end

# ---------------------------------------------------------------------------------------
# StatsAPI verbs
# ---------------------------------------------------------------------------------------

"""
    coef(m::LCAModel) -> Vector{Float64}

Free parameters of the model on the logit scale, `dof(m)` in total. The class-membership
block comes first: `vec(m.beta)`, the multinomial-logit coefficients of classes `2, …, K`
against class 1 (column-major, so the `P` coefficients of class 2 first; without
covariates these are the log-odds `log(π_k / π_1)`). Then, for every item `j`, class `k`
and response category `c` other than the class's modal category `r`, the log-odds
`log(item_probs[j][k, c] / item_probs[j][k, r])`. [`coefnames`](@ref) labels the entries,
[`vcov`](@ref) is their covariance matrix and [`coeftable`](@ref) tabulates them.
Parameters whose probability is on the boundary (within `1e-6` of 0 or 1) are included
but have no standard error.

# Example
```julia
c = coef(m)
Dict(zip(coefnames(m), c))
```
"""
function StatsAPI.coef(m::LCAModel)
    layout = ParamLayout(m)
    v = Vector{Float64}(undef, layout.n_total)
    v[1:layout.n_class] = vec(m.beta)
    _pack_items!(v, m.item_probs, layout)
    return v
end

"""
    coefnames(m::LCAModel) -> Vector{String}

Names of the entries of [`coef`](@ref), in the same order. Class-membership coefficients
are named `"class2: (Intercept)"`, `"class2: age"`, … (the class against the reference
class 1, and the covariate); item logits are named `"edu[middle/low]|class1"`, the
log-odds of level `middle` against the reference level `low` of item `edu` in class 1.
The reference level of an item row is its modal category in that class.
"""
function StatsAPI.coefnames(m::LCAModel)
    layout = ParamLayout(m)
    names = Vector{String}(undef, layout.n_total)
    K, P = layout.K, layout.P
    cn = hascovariates(m) ? m.data.covariate_names : [:intercept]
    idx = 1
    for k in 2:K, p in 1:P
        names[idx] = "class$k: " * (p == 1 ? "(Intercept)" : string(cn[p]))
        idx += 1
    end
    for j in 1:layout.J
        item = m.data.item_names[j]
        levels = m.data.item_levels[j]
        for k in 1:K
            r = layout.ref_cat[j][k]
            for c in 1:layout.C[j]
                c == r && continue
                names[idx] = "$item[$(levels[c])/$(levels[r])]|class$k"
                idx += 1
            end
        end
    end
    return names
end

"""
    vcov(m::LCAModel) -> Matrix{Float64}

Covariance matrix of [`coef`](@ref) (`dof × dof`), the inverse of the observed information
matrix computed by [`fit`](@ref) with `se=:hessian` (the default). Rows and columns of
parameters that are not estimable are `NaN`: parameters on the boundary (a probability
within `1e-6` of 0 or 1), the response parameters of an empty class (size `≤ 1e-6`), and
parameters with zero observed information. Those parameters are held fixed in the
information matrix, so the remaining entries are conditional on them. The whole matrix
is `NaN` when the observed information of the remaining parameters is not positive
definite or the covariate coefficients diverged. Throws an `ErrorException` for a model
fitted with `se=:none`.
"""
function StatsAPI.vcov(m::LCAModel)
    m.vcov === nothing && throw(ErrorException(
        "no covariance matrix is available for this model (fitted with se=:none); refit with se=:hessian"))
    return copy(m.vcov)
end

"""
    stderror(m::LCAModel) -> Vector{Float64}

Standard errors of [`coef`](@ref), `sqrt.(diag(vcov(m)))`; `NaN` for parameters on the
boundary. Throws for a model fitted with `se=:none`.
"""
StatsAPI.stderror(m::LCAModel) = sqrt.(diag(vcov(m)))

"""
    confint(m::LCAModel; level=0.95) -> Matrix{Float64}

Wald confidence intervals of [`coef`](@ref) on the logit scale: a `dof × 2` matrix of
lower and upper bounds, `coef ± z·stderror` with `z` the `1 - (1 - level)/2` quantile of
the standard normal distribution. For intervals of the response probabilities themselves
use [`profiles`](@ref). Throws for a model fitted with `se=:none`.
"""
function StatsAPI.confint(m::LCAModel; level::Real=0.95)
    z = _zquantile(level)
    c = coef(m)
    se = stderror(m)
    return hcat(c .- z .* se, c .+ z .* se)
end

function _zquantile(level::Real)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    return Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
end

# "95" for level 0.95, "99.9" for 0.999: the confidence level as a percentage string.
function _level_string(level::Real)
    pct = round(100 * level; digits=10)     # 0.57 * 100 is 56.99999999999999
    return isinteger(pct) ? string(Int(pct)) : string(pct)
end

"""
    coeftable(m::LCAModel; level=0.95, which=:all) -> StatsBase.CoefTable

Coefficient table of the free parameters ([`coef`](@ref)) with the columns `Estimate`,
`Std. Error`, `z`, `Pr(>|z|)` (two-sided Wald test against zero) and the bounds of the
`level` confidence interval, one row per entry of [`coefnames`](@ref). `which` selects
the rows: `:all`, `:class` (the `(K - 1)·P` class-membership coefficients) or `:items`
(the `Σ_j K·(C_j - 1)` item logits). Throws for a model fitted with `se=:none`.

# Example
```julia
coeftable(m; which=:class)      # covariate effects on class membership
```
"""
function StatsAPI.coeftable(m::LCAModel; level::Real=0.95, which::Symbol=:all)
    _check_which(which)
    return _coeftable(m, stderror(m), confint(m; level=level), level, which)
end

_check_which(which::Symbol) = which in (:all, :class, :items) ||
    throw(ArgumentError("which must be :all, :class or :items, got $(repr(which))"))

# Coefficient table of the free parameters of `m` from standard errors and `level`
# confidence intervals of either source (observed information or bootstrap), restricted
# to the rows selected by `which`.
function _coeftable(m::LCAModel, se::AbstractVector, ci::AbstractMatrix, level::Real, which::Symbol)
    c = coef(m)
    names = coefnames(m)
    n_class = (m.n_classes - 1) * size(m.beta, 1)
    idx = which === :all ? (1:length(c)) : which === :class ? (1:n_class) : (n_class + 1:length(c))
    z = c[idx] ./ se[idx]
    p = 2 .* Distributions.ccdf.(Distributions.Normal(), abs.(z))
    pctstr = _level_string(level)
    return StatsBase.CoefTable(
        [c[idx], se[idx], z, p, ci[idx, 1], ci[idx, 2]],
        ["Estimate", "Std. Error", "z", "Pr(>|z|)", "Lower $pctstr%", "Upper $pctstr%"],
        names[idx], 4, 3)
end

"""
    informationmatrix(m::LCAModel; expected=false) -> Matrix{Float64}

Observed information matrix of the free parameters on the [`coef`](@ref) scale
(`dof × dof`): the negative Hessian of the log-likelihood, recomputed from the analytic
score by central finite differences (see [`vcov`](@ref) for its inverse). Rows and columns
of parameters on the boundary are `NaN`. Only the observed information is available, so
the default is `expected=false` (StatsAPI's generic default is `true`); `expected=true`
throws an `ArgumentError`.
"""
function StatsAPI.informationmatrix(m::LCAModel; expected::Bool=false)
    expected && throw(ArgumentError("only the observed information is available"))
    ws, θ, layout = _model_params(m)
    v = _pack(θ, layout)
    info = _observed_information(v, layout, ws)
    n = layout.n_total
    full = fill(NaN, n, n)
    free = findall(layout.free)
    full[free, free] = info
    T = _coef_transform(layout, ws.A)
    # v_public = T v, so I_public = T⁻ᵀ I T⁻¹
    return _transform_masked(full, transpose(inv(T)), layout.free)
end

# ---------------------------------------------------------------------------------------
# Delta method: class sizes and item-response probabilities
# ---------------------------------------------------------------------------------------

# Delta-method covariance of one softmax block `p` (a row of item_probs or the class
# sizes) from the covariance `V` of its logits at indices `idx` (category order, reference
# `r` omitted). Only the free logits enter, so the result is conditional on the boundary
# cells being fixed (the Mplus / Latent GOLD convention): `S = J V_free J'` with
# `∂p_c/∂γ_d = p_c(δ_cd - p_d)`, NaN rows and columns for the boundary cells, and all NaN
# when no logit is free or the free block of `V` is not finite.
function _softmax_covariance(p::AbstractVector{<:Real}, r::Integer, idx::AbstractUnitRange,
                             free::AbstractVector{Bool}, V::AbstractMatrix)
    C = length(p)
    out = fill(NaN, C, C)
    free_idx = Int[]
    free_cats = Int[]
    i = first(idx)
    for c in 1:C
        c == r && continue
        if free[i]
            push!(free_idx, i)
            push!(free_cats, c)
        end
        i += 1
    end
    isempty(free_idx) && return out
    Vf = V[free_idx, free_idx]
    all(isfinite, Vf) || return out
    Jm = Matrix{Float64}(undef, C, length(free_cats))
    for (d, dc) in enumerate(free_cats), c in 1:C
        Jm[c, d] = p[c] * ((c == dc ? 1.0 : 0.0) - p[dc])
    end
    S = Jm * Vf * transpose(Jm)
    for c in 1:C
        if !_interior(p[c])
            S[c, :] .= NaN
            S[:, c] .= NaN
        end
    end
    return S
end

# Delta-method covariance of the class sizes (K × K): NaN without a covariance matrix or
# with covariates (the sizes are then sample averages), zero for a single class; an empty
# class gets NaN and the others conditional standard errors.
function _class_covariance(class_probs::AbstractVector{<:Real}, layout::ParamLayout,
                           V::Union{Nothing,AbstractMatrix})
    K = layout.K
    V === nothing && return fill(NaN, K, K)
    K == 1 && return zeros(1, 1)
    layout.covariates && return fill(NaN, K, K)
    return _softmax_covariance(class_probs, 1, 1:(K - 1), layout.free, V)
end
_class_covariance(m::LCAModel, layout::ParamLayout, V::Union{Nothing,AbstractMatrix}) =
    _class_covariance(m.class_probs, layout, V)

# Delta-method covariance of row k of item_probs[j] (C_j × C_j); NaN without a covariance
# matrix, and see `_softmax_covariance` for boundary cells.
function _profile_covariance(m::LCAModel, layout::ParamLayout, V::Union{Nothing,AbstractMatrix},
                             j::Integer, k::Integer)
    V === nothing && return fill(NaN, layout.C[j], layout.C[j])
    return _softmax_covariance(view(m.item_probs[j], k, :), layout.ref_cat[j][k],
                               _row_indices(layout, j, k), layout.free, V)
end

_sqrt_diag(S::AbstractMatrix) = [isnan(S[i, i]) ? NaN : sqrt(max(S[i, i], 0.0)) for i in 1:size(S, 1)]

# Standard errors of the class sizes (length K) and of every item-response probability
# (matrices shaped like item_probs); NaN where undefined.
function _profile_se(m::LCAModel)
    layout = ParamLayout(m)
    V = m.vcov
    se_class = _sqrt_diag(_class_covariance(m, layout, V))
    se_items = Vector{Matrix{Float64}}(undef, layout.J)
    for j in 1:layout.J
        S = Matrix{Float64}(undef, layout.K, layout.C[j])
        for k in 1:layout.K
            S[k, :] = _sqrt_diag(_profile_covariance(m, layout, V, j, k))
        end
        se_items[j] = S
    end
    return se_class, se_items
end

const _ProfileRow = NamedTuple{(:item, :level, :class, :prob, :se, :lower, :upper),
                               Tuple{Symbol,String,Int,Float64,Float64,Float64,Float64}}

"""
    profiles(m::LCAModel; level=0.95, classes=false) -> Vector{NamedTuple}

Item-response profiles of a fitted model as a row table (a `Vector` of `NamedTuple`s,
usable with any Tables.jl sink such as `DataFrame`). Each row holds one item-response
probability: `item::Symbol`, `level::String` (the response label), `class::Int`,
`prob::Float64`, its delta-method standard error `se` and the bounds `lower`/`upper` of a
`level` confidence interval computed on the logit scale (so they stay within `[0, 1]`).
Rows are ordered by item, then level, then class; there are `Σ_j C_j · K` rows.

The standard errors come from [`vcov`](@ref) by the delta method: for a row of
probabilities `B = softmax(γ)` with the model's logits `γ`, `Var(B) = J vcov(γ) J'` with
`∂B_c/∂γ_d = B_c(δ_cd - B_d)`. A probability on the boundary (within `1e-6` of 0 or 1)
has no standard error: its logit is held fixed, `se` is `NaN` and `lower = upper = prob`.
The standard errors of the remaining cells in a row with a boundary cell are conditional
on the boundary cell being fixed (only the free logits of the row enter the delta
method, as in Mplus and Latent GOLD). When a row has no free logit (its modal
probability is 1, or the row belongs to an empty class) `se`, `lower` and `upper` are
`NaN` for the whole row, as they are for every row of a model fitted with `se=:none`.

With `classes=true` the table starts with one row per class holding its size:
`item = :class`, `level = "k"`, `class = k`, `prob = class_probs[k]`, with the
delta-method standard error from the class-membership block (softmax Jacobian; an
empty class is treated like a boundary cell). For a model with covariates the class
sizes are sample averages of the covariate-specific membership probabilities and their
standard errors are reported as `NaN`.

# Arguments
- `m::LCAModel`: fitted model
- `level::Real=0.95`: confidence level, in `(0, 1)`
- `classes::Bool=false`: prepend the class-size rows

# Returns
- `Vector{NamedTuple}` with fields `(:item, :level, :class, :prob, :se, :lower, :upper)`

See also [`show_profiles`](@ref) for a printed version and [`coeftable`](@ref) for the
logit-scale parameters.
"""
function profiles(m::LCAModel; level::Real=0.95, classes::Bool=false)
    z = _zquantile(level)
    se_class, se_items = _profile_se(m)
    rows = _ProfileRow[]
    if classes
        conditional = any(isfinite, se_class)
        for k in 1:m.n_classes
            p = m.class_probs[k]
            se = se_class[k]
            lower, upper = _profile_ci(p, se, z, conditional)
            push!(rows, (item=:class, level=string(k), class=k, prob=p, se=se,
                         lower=lower, upper=upper))
        end
    end
    for j in 1:m.n_items, c in 1:m.n_categories[j], k in 1:m.n_classes
        p = m.item_probs[j][k, c]
        se = se_items[j][k, c]
        conditional = any(isfinite, view(se_items[j], k, :))
        lower, upper = _profile_ci(p, se, z, conditional)
        push!(rows, (item=m.data.item_names[j], level=m.data.item_levels[j][c], class=k,
                     prob=p, se=se, lower=lower, upper=upper))
    end
    return rows
end

# Confidence interval of one probability: on the logit scale when its standard error is
# defined, (p, p) for a boundary cell whose block has conditional standard errors, NaN
# otherwise.
function _profile_ci(p::Real, se::Real, z::Real, conditional::Bool)
    isnan(se) && conditional && !_interior(p) && return (Float64(p), Float64(p))
    return _logit_ci(p, se, z)
end

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
