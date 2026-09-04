"""
    _used_levels(col)

Sorted distinct values of a column. Code `i` of a prepared column corresponds to
`_used_levels(col)[i]`. For a `CategoricalArray` the order is the level order and
unused levels are not included.
"""
_used_levels(col) = sort(unique(col))

"""
    prepare_data(df::DataFrame, cols::Symbol...)

Prepare a `DataFrame` for LCA by recoding each selected column to consecutive integer
codes `1, 2, …, K`, where `K` is the number of distinct values in that column.

Numeric columns are recoded by the rank of their sorted distinct values, so `0/1`, `1/2`,
and even `1/3` codings all become `1/2`. String and categorical columns are coded by their
sorted distinct values (level order for a `CategoricalArray`). Code `1` always corresponds
to the smallest value or the first level.

# Arguments
- `df::DataFrame`: Input DataFrame
- `cols::Symbol...`: Names of the columns to use as manifest variables
- `zero_based`: Deprecated and ignored; codes are always inferred from the data. It is
  kept for backward compatibility and, if given, its length must still match `cols`.

# Returns
- `Matrix{Int}`: Prepared data matrix (one row per observation, one column per item)
- `Vector{Int}`: Number of categories of each item

# Example
```jldoctest
julia> using LatentClassAnalysis, DataFrames

julia> df = DataFrame(a = [0, 1, 1, 0], b = ["no", "yes", "yes", "no"], c = [1, 3, 3, 1]);

julia> data, n_categories = prepare_data(df, :a, :b, :c);

julia> data
4×3 Matrix{Int64}:
 1  1  1
 2  2  2
 2  2  2
 1  1  1

julia> n_categories
3-element Vector{Int64}:
 2
 2
 2
```
"""
function prepare_data(
    df::DataFrame, cols::Symbol...;
    zero_based::Union{Nothing,Vector{Bool}}=nothing
)
    if !isnothing(zero_based) && length(zero_based) != length(cols)
        throw(ArgumentError("Length of zero_based must match number of columns"))
    end

    data = Matrix{Int}(undef, nrow(df), length(cols))
    n_categories = Vector{Int}(undef, length(cols))

    for (i, col) in enumerate(cols)
        levels = _used_levels(df[!, col])
        data[:, i] = indexin(df[!, col], levels)
        n_categories[i] = length(levels)
    end

    return data, n_categories
end

"""
    check_identifiability(n_items, n_classes, n_categories)

Warn when the model may not be identifiable. This is a rule of thumb: with `K` classes
and items with at least `C` categories each, at least `2⌈log_C(K)⌉ + 1` items are
recommended. Returns `true`.
"""
function check_identifiability(n_items::Integer, n_classes::Integer, n_categories::AbstractVector{<:Integer})
    # Use minimum categories as worst case bound
    min_cat = minimum(n_categories)
    required_items = 2 * ceil(Int, log(min_cat, n_classes)) + 1

    if n_items < required_items
        @warn(
            "Model may not be identifiable. " *
            "With $n_classes classes and minimum of $min_cat categories, " *
            "need ideally $required_items items (got $n_items)."
        )
    end
    return true
end

"""
    diagnostics!(model::LCAModel, data::AbstractMatrix{<:Integer}, ll::Real)

Calculate model fit statistics (AIC, BIC, sample-size adjusted BIC, and relative entropy)
of a fitted model. Despite the `!`, neither `model` nor `data` is modified.

# Arguments
- `model::LCAModel`: Fitted model
- `data::AbstractMatrix{<:Integer}`: Data matrix used to fit the model
- `ll::Real`: Log-likelihood returned by [`fit!`](@ref)

# Returns
- [`ModelDiagnostics`](@ref): Structure containing the fit statistics
"""
function diagnostics!(model::LCAModel, data::AbstractMatrix{<:Integer}, ll::Real)
    n_obs = size(data, 1)

    # Calculate number of parameters
    # Class probabilities (K-1) + Item probabilities for each class and item
    n_params = (model.n_classes - 1) +
               sum(cats -> (model.n_classes * (cats - 1)), model.n_categories)

    # Calculate AIC and BIC
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + log(n_obs) * n_params

    # Calculate sample-size adjusted BIC
    n_star = (n_obs + 2) / 24
    sbic = -2 * ll + log(n_star) * n_params

    # Calculate entropy
    posterior = zeros(n_obs, model.n_classes)
    for i in 1:n_obs
        for k in 1:model.n_classes
            prob = log(model.class_probs[k])
            for j in 1:model.n_items
                prob += log(model.item_probs[j][k, data[i, j]])
            end
            posterior[i, k] = exp(prob)
        end
        posterior[i, :] ./= sum(posterior[i, :])
    end

    entropy = 0.0
    for i in 1:n_obs
        for k in 1:model.n_classes
            p = posterior[i, k]
            entropy -= p * log(p + eps()) # eps() to avoid log(0)
        end
    end
    entropy = 1 - (entropy / (n_obs * log(model.n_classes)))

    return ModelDiagnostics(ll, aic, bic, sbic, entropy)
end

"""
    show_profiles(model::LCAModel, data::DataFrame, cols::Vector{Symbol};
                  var_names=nothing, var_labels=nothing, digits=3)

Print the latent class profiles: the size of each class and, for every item, the
probability of each response category within each class.

# Arguments
- `model::LCAModel`: Fitted model
- `data::DataFrame`: The DataFrame passed to [`prepare_data`](@ref); used to recover
  category labels
- `cols::Vector{Symbol}`: The columns passed to [`prepare_data`](@ref), in the same order
- `var_names::Union{Nothing,Vector{String}}`: Display names of the items (default: the
  column names)
- `var_labels::Union{Nothing,Vector{Vector{String}}}`: Display labels of the categories of
  each item (default: the sorted distinct values of each column)
- `digits::Int`: Number of decimal places of the printed percentages
"""
function show_profiles(model::LCAModel, data::DataFrame, cols::Vector{Symbol};
    var_names::Union{Nothing,Vector{String}}=nothing,
    var_labels::Union{Nothing,Vector{Vector{String}}}=nothing,
    digits::Int=3)

    # Use DataFrame column names if var_names not provided
    display_names = isnothing(var_names) ? String[string(col) for col in cols] : var_names

    # Use the same sorted distinct values as prepare_data, so labels line up with codes
    if isnothing(var_labels)
        var_labels = [string.(_used_levels(data[!, col])) for col in cols]
    end

    fmt = Printf.Format("%.$(digits)f%%")

    # Print header
    println("\nLatent Class Profiles")
    println("="^80)

    # Print class sizes with better alignment
    println("Class Sizes:")
    for k in 1:model.n_classes
        pct = model.class_probs[k] * 100
        println("  Class $k: $(rpad(@sprintf("%.1f", pct), 6))%")
    end
    println("-"^80)

    # Calculate maximum label length for alignment
    max_label_length = maximum(maximum(length.(labels)) for labels in var_labels)

    # Print item probabilities for each class
    for (i, var) in enumerate(display_names)
        println("\n$var:")

        # Print header row with class numbers
        print(" "^(max_label_length + 2))  # Space for labels
        for k in 1:model.n_classes
            print("Class $k" * " "^7)
        end
        println()

        # Print probabilities for each category
        for (j, label) in enumerate(var_labels[i])
            # Print label with padding
            print(rpad("$label:", max_label_length + 2))

            # Print probabilities
            for k in 1:model.n_classes
                pct = model.item_probs[i][k, j] * 100
                print(rpad(Printf.format(fmt, pct), 12))
            end
            println()
        end
    end
    println("\n" * "-"^80)
end
