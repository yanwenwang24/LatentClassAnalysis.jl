# Shared helpers for the test suite. Included once from runtests.jl; every test file also
# includes it (guarded) so that each file can be run on its own.

using Random

"""
    simulate_lca(rng, n, class_probs, item_probs) -> (data::Matrix{Int}, classes::Vector{Int})

Draw `n` observations from a latent class model with known parameters.

- `class_probs`: vector of `K` class membership probabilities (must sum to one)
- `item_probs`: vector of `J` matrices; `item_probs[j]` is `K × C_j` and row `k` holds the
  probability of each of the `C_j` response categories of item `j` given class `k`

All draws come from `rng`, so a `StableRNG` seed makes the data reproducible across Julia
versions. `data` holds 1-based codes as expected by `fit!`, and `classes[i]` is the true
class of observation `i`.
"""
function simulate_lca(rng::AbstractRNG, n::Integer,
                      class_probs::AbstractVector{<:Real},
                      item_probs::AbstractVector{<:AbstractMatrix{<:Real}})
    K = length(class_probs)
    J = length(item_probs)
    isapprox(sum(class_probs), 1.0; atol=1e-8) ||
        throw(ArgumentError("class_probs must sum to one"))
    for (j, P) in enumerate(item_probs)
        size(P, 1) == K ||
            throw(ArgumentError("item_probs[$j] must have $K rows, got $(size(P, 1))"))
        all(isapprox.(sum(P, dims=2), 1.0; atol=1e-8)) ||
            throw(ArgumentError("rows of item_probs[$j] must sum to one"))
    end

    data = Matrix{Int}(undef, n, J)
    classes = Vector{Int}(undef, n)
    for i in 1:n
        k = _draw_category(rng, class_probs)
        classes[i] = k
        for j in 1:J
            data[i, j] = _draw_category(rng, view(item_probs[j], k, :))
        end
    end
    return data, classes
end

# Draw an index from a discrete distribution given by a probability vector.
function _draw_category(rng::AbstractRNG, probs::AbstractVector{<:Real})
    u = rand(rng)
    cum = 0.0
    for (c, p) in enumerate(probs)
        cum += p
        u <= cum && return c
    end
    return length(probs)  # guard against round-off in the cumulative sum
end

"""
    align_classes(est_item_probs, true_item_probs) -> Vector{Int}

Find the permutation `perm` of the estimated classes that best matches the true classes,
by brute force over all permutations (intended for `K ≤ 4`). The match is measured by the
total absolute difference of the item response probabilities.

`perm[k]` is the estimated class corresponding to true class `k`, so that
`est_item_probs[j][perm, :] ≈ true_item_probs[j]` and, for a fitted `model`,
`model.class_probs[perm] ≈ true_class_probs`.
"""
function align_classes(est_item_probs::AbstractVector{<:AbstractMatrix{<:Real}},
                       true_item_probs::AbstractVector{<:AbstractMatrix{<:Real}})
    K = size(true_item_probs[1], 1)
    K <= 4 || throw(ArgumentError("align_classes is meant for K ≤ 4 classes, got $K"))
    length(est_item_probs) == length(true_item_probs) ||
        throw(ArgumentError("est_item_probs and true_item_probs must have the same length"))

    best_perm = collect(1:K)
    best_cost = Inf
    for perm in _permutations(collect(1:K))
        cost = 0.0
        for (E, T) in zip(est_item_probs, true_item_probs)
            cost += sum(abs, E[perm, :] .- T)
        end
        if cost < best_cost
            best_cost = cost
            best_perm = perm
        end
    end
    return best_perm
end

# All permutations of a vector (small inputs only).
function _permutations(v::Vector{Int})
    length(v) <= 1 && return [copy(v)]
    out = Vector{Vector{Int}}()
    for (i, x) in enumerate(v)
        rest = v[[1:i-1; i+1:end]]
        for p in _permutations(rest)
            push!(out, vcat(x, p))
        end
    end
    return out
end

"""
    capture_stdout(f) -> String

Run `f()` with `stdout` redirected and return everything it printed. `redirect_stdout`
does not accept an `IOBuffer`, so this uses a pipe drained by an asynchronous reader.
"""
function capture_stdout(f)
    original_stdout = stdout
    rd, wr = redirect_stdout()
    reader = @async read(rd, String)
    try
        f()
    finally
        redirect_stdout(original_stdout)
        close(wr)
    end
    return fetch(reader)
end
