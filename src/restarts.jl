# Random restarts: initial values, the two-stage (emEM) multi-start driver, class
# reordering by size, and the class-splitting initializer used by the bootstrap LRT.

"""
    StartRecord

Outcome of one start of the multi-start EM driver: the `seed` of its random initial
values, the log-likelihood after the short run (`short_ll`) and after the final run
(`final_ll`, equal to `short_ll` when the start was not continued), the total number of
EM `iterations`, whether the final run `converged`, and whether it was `continued` to
convergence. Internal.
"""
struct StartRecord
    seed::UInt64
    short_ll::Float64
    final_ll::Float64
    iterations::Int
    converged::Bool
    continued::Bool
end

# Clamp row k of P to [floor, 1] and renormalize it.
function _floor_row!(P::AbstractMatrix{Float64}, k::Integer, floor::Real=PROB_FLOOR)
    s = 0.0
    @inbounds for c in 1:size(P, 2)
        p = P[k, c]
        p = p < floor ? floor : (p > 1.0 ? 1.0 : p)
        P[k, c] = p
        s += p
    end
    @inbounds for c in 1:size(P, 2)
        P[k, c] /= s
    end
    return P
end

"""
    _init_random(rng, K, C) -> LCAParams

Random starting values: uniform class probabilities and, for every item and class, an
exact Dirichlet(1, …, 1) draw for the response probabilities. Internal.
"""
function _init_random(rng::AbstractRNG, K::Integer, C::AbstractVector{<:Integer})
    class_probs = fill(1.0 / K, K)
    item_probs = Vector{Matrix{Float64}}(undef, length(C))
    for (j, c) in enumerate(C)
        P = Matrix{Float64}(undef, K, c)
        for k in 1:K
            s = 0.0
            for cc in 1:c
                e = -log(1 - rand(rng))
                P[k, cc] = e
                s += e
            end
            for cc in 1:c
                P[k, cc] /= s
            end
            _floor_row!(P, k)
        end
        item_probs[j] = P
    end
    return LCAParams(class_probs, item_probs, nothing)
end

# ---- user-supplied starting values -----------------------------------------------------

function _check_init_dims(class_probs, item_probs, K, C)
    length(class_probs) == K ||
        throw(ArgumentError("init has $(length(class_probs)) class probabilities, expected $K"))
    length(item_probs) == length(C) ||
        throw(ArgumentError("init has $(length(item_probs)) item matrices, expected $(length(C))"))
    for (j, P) in enumerate(item_probs)
        size(P) == (K, C[j]) ||
            throw(ArgumentError("init item_probs[$j] has size $(size(P)), expected ($K, $(C[j]))"))
    end
    return nothing
end

function _normalize_init!(θ::LCAParams)
    cp = θ.class_probs
    (all(isfinite, cp) && all(>=(0), cp) && sum(cp) > 0) ||
        throw(ArgumentError("init class probabilities must be finite, non-negative and not all zero"))
    cp ./= sum(cp)
    # Floor the class sizes so that a class supplied with probability zero (e.g. an init
    # model that lost a class) neither stays dead forever nor yields log(0) coefficients.
    for k in eachindex(cp)
        cp[k] = max(cp[k], PROB_FLOOR)
    end
    cp ./= sum(cp)
    for P in θ.item_probs
        (all(isfinite, P) && all(>=(0), P)) ||
            throw(ArgumentError("init item probabilities must be finite and non-negative"))
        for k in 1:size(P, 1)
            row = view(P, k, :)
            s = sum(row)
            s > 0 || throw(ArgumentError("init item probabilities must not have an all-zero row"))
            row ./= s          # normalize before flooring, or [3, 1] would clamp to [1, 1]
            _floor_row!(P, k)
        end
    end
    return θ
end

# Give θ the coefficient matrix the workspace expects: none without covariates; with
# covariates, a missing `coefs` starts with zero slopes and the log-odds of the class
# probabilities as intercepts.
function _seed_coefs!(θ::LCAParams, ws::LCAWorkspace)
    if !ws.covariates
        θ.coefs = nothing
    elseif θ.coefs === nothing
        P, K = size(ws.Xst, 1), ws.K
        coefs = zeros(P, K)
        for k in 2:K
            coefs[1, k] = log(max(θ.class_probs[k], PROB_FLOOR) / max(θ.class_probs[1], PROB_FLOOR))
        end
        θ.coefs = coefs
    end
    return θ
end

function _as_params(m::LCAModel, ws::LCAWorkspace)
    K, C = ws.K, ws.C
    _check_init_dims(m.class_probs, m.item_probs, K, C)
    θ = _normalize_init!(LCAParams(copy(m.class_probs), [copy(P) for P in m.item_probs], nothing))
    if ws.covariates && hascovariates(m)
        P = size(ws.Xst, 1)
        size(m.beta, 1) == P || throw(ArgumentError(
            "init model has $(size(m.beta, 1) - 1) covariates but the data has $(P - 1)"))
        all(isfinite, m.beta) || throw(ArgumentError("init model has non-finite covariate coefficients"))
        # Raw-scale coefficients (class 1 as reference) to the standardized scale
        θ.coefs = ws.A \ _raw_coefs(m)
    end
    return _seed_coefs!(θ, ws)
end

# An init model fitted on other covariates would seed the slopes of the wrong columns.
function _check_init_covariates(init, d::LCAData, covariates::Bool)
    covariates || return nothing
    for m in (init isa AbstractVector ? init : (init,))
        m isa LCAModel && hascovariates(m) && m.data.covariate_names != d.covariate_names &&
            throw(ArgumentError("the init model was fitted with the covariates " *
                                "$(m.data.covariate_names[2:end]) but the data has $(d.covariate_names[2:end])"))
    end
    return nothing
end

function _as_params(θ::LCAParams, ws::LCAWorkspace)
    K, C = ws.K, ws.C
    _check_init_dims(θ.class_probs, θ.item_probs, K, C)
    θc = _normalize_init!(copy(θ))
    if θc.coefs !== nothing && ws.covariates
        P = size(ws.Xst, 1)
        size(θc.coefs) == (P, K) ||
            throw(ArgumentError("init coefs has size $(size(θc.coefs)), expected ($P, $K)"))
        all(isfinite, θc.coefs) || throw(ArgumentError("init coefs must be finite"))
        θc.coefs .-= θc.coefs[:, 1]          # class 1 is the reference
    end
    return _seed_coefs!(θc, ws)
end

function _as_params(nt::NamedTuple, ws::LCAWorkspace)
    K, C = ws.K, ws.C
    (haskey(nt, :class_probs) && haskey(nt, :item_probs)) ||
        throw(ArgumentError("an init NamedTuple must have the fields class_probs and item_probs"))
    cp = Vector{Float64}(nt.class_probs)
    ip = [Matrix{Float64}(P) for P in nt.item_probs]
    _check_init_dims(cp, ip, K, C)
    return _seed_coefs!(_normalize_init!(LCAParams(cp, ip, nothing)), ws)
end

_as_params(x, ::LCAWorkspace) = throw(ArgumentError(
    "init must be nothing, an LCAModel, an LCAParams, a NamedTuple with class_probs and " *
    "item_probs, or a vector of those; got $(typeof(x))"))

_init_list(::Nothing, ::LCAWorkspace) = LCAParams[]
_init_list(xs::AbstractVector, ws::LCAWorkspace) = LCAParams[_as_params(x, ws) for x in xs]
_init_list(x, ws::LCAWorkspace) = LCAParams[_as_params(x, ws)]

# ---- multi-start driver -----------------------------------------------------------------

# Run `_em!` for the starts in `idxs`, storing (ll, iterations, converged) in `results`.
# With `multithreaded` the starts are split into at most `Threads.nthreads()` chunks, each
# with its own scratch workspace, so results are bitwise identical to the serial run.
function _run_starts!(results::Vector{Tuple{Float64,Int,Bool}}, θs::Vector{LCAParams},
                      idxs, ws::LCAWorkspace, max_iter::Integer, tol::Real,
                      multithreaded::Bool)
    if multithreaded && Threads.nthreads() > 1 && length(idxs) > 1
        nchunks = min(Threads.nthreads(), length(idxs))
        chunks = collect(Iterators.partition(idxs, cld(length(idxs), nchunks)))
        Threads.@threads for c in eachindex(chunks)
            wsc = LCAWorkspace(ws)
            for s in chunks[c]
                results[s] = _em!(θs[s], wsc; max_iter=max_iter, tol=tol)
            end
        end
    else
        for s in idxs
            results[s] = _em!(θs[s], ws; max_iter=max_iter, tol=tol)
        end
    end
    return results
end

"""
    _multistart(ws, opts, rng, init, multithreaded) -> NamedTuple

Two-stage multi-start EM (emEM). Seeds for all starts are drawn from `rng` up front and
every random start uses its own `Xoshiro(seed)`, so serial and threaded runs agree
bitwise. Stage 1 runs `opts.short_iters` iterations from every start; stage 2 continues
the `opts.n_final` best to convergence. User-supplied `init` values occupy the first
start(s). With covariates every start carries a coefficient matrix (zero slopes and
intercepts from its class probabilities unless supplied). Returns the best parameters,
the `StartRecord` of every start, and the index, log-likelihood, iteration count,
convergence status and replication flag of the winner. Internal.
"""
function _multistart(ws::LCAWorkspace, opts::LCAOptions, rng::AbstractRNG, init,
                     multithreaded::Bool)
    K, C = ws.K, ws.C
    inits = _init_list(init, ws)
    n_starts = max(opts.n_starts, length(inits))
    n_final = min(opts.n_final, n_starts)
    seeds = rand(rng, UInt64, n_starts)
    θs = Vector{LCAParams}(undef, n_starts)
    for s in 1:n_starts
        θs[s] = s <= length(inits) ? inits[s] :
                _seed_coefs!(_init_random(Xoshiro(seeds[s]), K, C), ws)
    end

    # Stage 1: short runs from every start
    short = Vector{Tuple{Float64,Int,Bool}}(undef, n_starts)
    _run_starts!(short, θs, 1:n_starts, ws, opts.short_iters, opts.tol, multithreaded)
    short_ll = [r[1] for r in short]
    order = sortperm(short_ll; rev=true)
    continued = order[1:n_final]
    if opts.verbose
        for s in 1:n_starts
            @printf("start %3d (seed %#018x): short-run log-likelihood %.6f%s\n", s, seeds[s],
                    short_ll[s], s in continued ? "  [continued]" : "")
        end
    end

    # Stage 2: continue the best starts to convergence
    final = copy(short)
    _run_starts!(final, θs, continued, ws, opts.max_iter, opts.tol, multithreaded)
    final_ll = [final[s][1] for s in 1:n_starts]
    best = continued[argmax(final_ll[continued])]
    ll_best = final_ll[best]
    n_rep = count(s -> abs(final_ll[s] - ll_best) <= 1e-6 * (1 + abs(ll_best)), continued)
    replicated = n_final == 1 || n_rep >= 2
    if opts.verbose
        for s in continued
            @printf("start %3d: final log-likelihood %.6f after %d iterations%s%s\n", s, final_ll[s],
                    short[s][2] + final[s][2], final[s][3] ? "" : " (not converged)",
                    s == best ? "  [best]" : "")
        end
        @printf("best log-likelihood %.6f replicated by %d of %d continued starts\n", ll_best, n_rep, n_final)
    end

    records = Vector{StartRecord}(undef, n_starts)
    for s in 1:n_starts
        cont = s in continued
        records[s] = StartRecord(seeds[s], short_ll[s], final_ll[s],
                                 short[s][2] + (cont ? final[s][2] : 0), final[s][3], cont)
    end
    return (θ=θs[best], records=records, best=best, loglik=ll_best,
            iterations=records[best].iterations, converged=final[best][3],
            replicated=replicated, n_final=n_final)
end

# ---- class ordering ---------------------------------------------------------------------

# Reorder the classes of θ by `perm` (new class k is old class perm[k]); coefficients are
# re-based so that the new class 1 is the reference (column identically zero).
function _permute_classes!(θ::LCAParams, perm::AbstractVector{<:Integer})
    θ.class_probs .= θ.class_probs[perm]
    for P in θ.item_probs
        P .= P[perm, :]
    end
    if θ.coefs !== nothing
        B = θ.coefs[:, perm]
        B .-= B[:, 1]
        θ.coefs .= B
    end
    return θ
end

# Sort classes by decreasing size (stable). Returns the permutation applied.
function _sort_by_size!(θ::LCAParams)
    perm = sortperm(θ.class_probs; rev=true)
    _permute_classes!(θ, perm)
    return perm
end

"""
    _init_split(θ, k, rng; delta=0.05) -> LCAParams

Starting values for a `K + 1`-class fit from a `K`-class solution: class `k` is split in
two halves of equal size whose item-response probabilities are the original ones perturbed
by `±U(0, delta)` in opposite directions (floored at `1e-3` and renormalized). Used to
warm-start the alternative model in the bootstrap likelihood-ratio test. Internal.
"""
function _init_split(θ::LCAParams, k::Integer, rng::AbstractRNG; delta::Real=0.05)
    K = length(θ.class_probs)
    1 <= k <= K || throw(ArgumentError("class $k does not exist (K = $K)"))
    class_probs = [θ.class_probs; θ.class_probs[k] / 2]
    class_probs[k] /= 2
    item_probs = Vector{Matrix{Float64}}(undef, length(θ.item_probs))
    for (j, P) in enumerate(θ.item_probs)
        C = size(P, 2)
        Q = Matrix{Float64}(undef, K + 1, C)
        Q[1:K, :] = P
        for c in 1:C
            d = delta * (2 * rand(rng) - 1)
            Q[K + 1, c] = P[k, c] + d
            Q[k, c] = P[k, c] - d
        end
        _floor_row!(Q, k, 1e-3)
        _floor_row!(Q, K + 1, 1e-3)
        item_probs[j] = Q
    end
    coefs = nothing
    if θ.coefs !== nothing
        B = hcat(θ.coefs, θ.coefs[:, k])
        B[1, k] -= log(2)
        B[1, K + 1] -= log(2)
        B .-= B[:, 1]
        coefs = B
    end
    return LCAParams(class_probs, item_probs, coefs)
end
