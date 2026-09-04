# Shared helpers for the test suite. Included once from runtests.jl; every test file also
# includes it (guarded) so that each file can be run on its own.

using Random
using LatentClassAnalysis

"""
    simulate_lca(rng, n, class_probs, item_probs; missing_rate=0.0)
        -> (y::Matrix{Int}, classes::Vector{Int})

Draw `n` observations from a latent class model with known parameters.

- `class_probs`: vector of `K` class membership probabilities (must sum to one)
- `item_probs`: vector of `J` matrices; `item_probs[j]` is `K × C_j` and row `k` holds the
  probability of each of the `C_j` response categories of item `j` given class `k`
- `missing_rate`: probability that any single response is set to `0` (missing completely
  at random), drawn after the responses so the same `rng` seed gives the same complete
  data regardless of the rate

All draws come from `rng`, so a `StableRNG` seed makes the data reproducible across Julia
versions. `y` holds 1-based codes (0 = missing) as expected by `LCAData`, and `classes[i]`
is the true class of observation `i`.
"""
function simulate_lca(rng::AbstractRNG, n::Integer,
                      class_probs::AbstractVector{<:Real},
                      item_probs::AbstractVector{<:AbstractMatrix{<:Real}};
                      missing_rate::Real=0.0)
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

    y = Matrix{Int}(undef, n, J)
    classes = Vector{Int}(undef, n)
    for i in 1:n
        k = _draw_category(rng, class_probs)
        classes[i] = k
        for j in 1:J
            y[i, j] = _draw_category(rng, view(item_probs[j], k, :))
        end
    end
    missing_rate > 0 && mcar!(rng, y, missing_rate)
    return y, classes
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
    mcar!(rng, y, rate) -> y

Set every entry of the code matrix `y` to `0` (missing) independently with probability
`rate`.
"""
function mcar!(rng::AbstractRNG, y::AbstractMatrix{Int}, rate::Real)
    for i in eachindex(y)
        rand(rng) < rate && (y[i] = 0)
    end
    return y
end

"""
    align_classes(est_item_probs, true_item_probs) -> Vector{Int}

Find the permutation `perm` of the estimated classes that best matches the true classes,
by brute force over all permutations (intended for `K ≤ 5`). The match is measured by the
total absolute difference of the item response probabilities.

`perm[k]` is the estimated class corresponding to true class `k`, so that
`est_item_probs[j][perm, :] ≈ true_item_probs[j]` and, for a fitted `model`,
`model.class_probs[perm] ≈ true_class_probs`.
"""
function align_classes(est_item_probs::AbstractVector{<:AbstractMatrix{<:Real}},
                       true_item_probs::AbstractVector{<:AbstractMatrix{<:Real}})
    K = size(true_item_probs[1], 1)
    K <= 5 || throw(ArgumentError("align_classes is meant for K ≤ 5 classes, got $K"))
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
    max_abs_error(model, perm, true_class_probs, true_item_probs) -> (Float64, Float64)

Largest absolute error of the class sizes and of the item-response probabilities of a
fitted model after aligning its classes with `perm` (see [`align_classes`](@ref)).
"""
function max_abs_error(model::LCAModel, perm::AbstractVector{<:Integer},
                       true_class_probs::AbstractVector{<:Real},
                       true_item_probs::AbstractVector{<:AbstractMatrix{<:Real}})
    e_class = maximum(abs.(model.class_probs[perm] .- true_class_probs))
    e_item = maximum(maximum(abs.(model.item_probs[j][perm, :] .- true_item_probs[j]))
                     for j in eachindex(true_item_probs))
    return e_class, e_item
end

"""
    same_fit(a::LCAModel, b::LCAModel) -> Bool

Whether two fitted models are bitwise identical in every estimated quantity.
"""
function same_fit(a::LCAModel, b::LCAModel)
    return a.n_classes == b.n_classes &&
           a.class_probs == b.class_probs &&
           a.item_probs == b.item_probs &&
           a.beta == b.beta &&
           a.posterior == b.posterior &&
           a.loglik == b.loglik &&
           a.converged == b.converged &&
           a.iterations == b.iterations &&
           a.start_loglik == b.start_loglik
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

# A well-separated two-class design shared by several test files: 6 binary items with
# item-specific separation between 0.75 and 0.9 and class sizes (0.6, 0.4).
const TWO_CLASS_PROBS = [0.6, 0.4]
const TWO_CLASS_ITEMS = [[s 1-s; 1-s s] for s in (0.75, 0.8, 0.85, 0.9, 0.78, 0.88)]

# A well-separated three-class design with items of 2, 3 and 4 categories.
const THREE_CLASS_PROBS = [0.45, 0.35, 0.2]
const THREE_CLASS_ITEMS = [
    [0.9 0.1; 0.2 0.8; 0.6 0.4],
    [0.7 0.2 0.1; 0.1 0.2 0.7; 0.15 0.7 0.15],
    [0.7 0.1 0.1 0.1; 0.1 0.1 0.1 0.7; 0.1 0.7 0.1 0.1],
    [0.85 0.15; 0.15 0.85; 0.85 0.15],
    [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8],
    [0.1 0.1 0.7 0.1; 0.7 0.1 0.1 0.1; 0.1 0.1 0.1 0.7],
]
