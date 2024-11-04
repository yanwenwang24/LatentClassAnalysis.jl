"""
    prepare_data(df::DataFrame, cols::Symbol...)

Prepare DataFrame for LCA by converting categorical columns to dummy variables if needed.

# Arguments
- `df::DataFrame`: Input DataFrame
- `cols::Symbol...`: Column names to use for analysis
- `zero_based::Union{Nothing, Vector{Bool}}=nothing`: Specify which columns are 0/1 coded.
    If nothing, automatically detect based on data.

# Returns
- `Matrix{Int}`: Prepared data matrix
- `Vector{Int}`: Number of categories for each variable
- `Vector{Bool}`: Whether each column was treated as zero-based
"""

function prepare_data(
    df::DataFrame, cols::Symbol...;
    zero_based::Union{Nothing,Vector{Bool}}=nothing
)
    # Initialize zero_based vector if not provided
    if isnothing(zero_based)
        zero_based = Vector{Bool}(undef, length(cols))
        # Detect coding scheme for each column
        for (i, col) in enumerate(cols)
            if eltype(df[!, col]) <: Number
                zero_based[i] = minimum(df[!, col]) == 0
            else
                zero_based[i] = false  # Categorical columns are 1-based by default
            end
        end
    else
        if length(zero_based) != length(cols)
            throw(ArgumentError("Length of zero_based must match number of columns"))
        end
    end

    data = Matrix{Int}(undef, nrow(df), length(cols))
    n_categories = Int[]

    for (i, col) in enumerate(cols)
        if eltype(df[!, col]) <: Number
            # For zero-based columns, shift values to 1-based
            if zero_based[i]
                data[:, i] = df[!, col] .+ 1
                push!(n_categories, length(unique(df[!, col])))
            else
                data[:, i] = df[!, col]
                push!(n_categories, length(unique(df[!, col])))
            end
        else
            # Categorical columns are always 1-based
            categories = sort(unique(df[!, col]))
            data[:, i] = indexin(df[!, col], categories)
            push!(n_categories, length(categories))
        end

        # Validate data
        if any(x -> x < 1, view(data, :, i))
            throw(ArgumentError("Column $(cols[i]): Invalid values after processing. " *
                                "Check if zero_based specification is correct."))
        end
    end

    return data, n_categories
end

"""
    diagnostics!(model::LCAModel, data::Matrix{Int}, ll::Float64)

Calculate model fit statistics including AIC, BIC, SBIC, and entropy.

# Arguments
- `model::LCAModel`: Fitted model
- `data::Matrix{Int}`: Data matrix
- `ll::Float64`: Log-likelihood from model fitting

# Returns
- `ModelDiagnostics`: Structure containing fit statistics
"""

function diagnostics!(model::LCAModel, data::Matrix{Int}, ll::Float64)
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
                 var_names::Union{Nothing, Vector{String}}=nothing,
                 var_labels::Union{Nothing, Vector{Vector{String}}}=nothing,
                 digits::Int=3)

Display the latent class profiles with aligned columns.
"""
function show_profiles(model::LCAModel, data::DataFrame, cols::Vector{Symbol};
    var_names::Union{Nothing,Vector{String}}=nothing,
    var_labels::Union{Nothing,Vector{Vector{String}}}=nothing,
    digits::Int=3)

    # Use DataFrame column names if var_names not provided
    display_names = isnothing(var_names) ? String[string(col) for col in cols] : var_names

    # Extract or use provided category labels
    if isnothing(var_labels)
        var_labels = Vector{Vector{String}}()
        for (i, col) in enumerate(cols)
            if eltype(data[!, col]) <: CategoricalValue
                push!(var_labels, string.(levels(data[!, col])))
            else
                unique_vals = sort(unique(data[!, col]))
                push!(var_labels, string.(unique_vals))
            end
        end
    end

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
                print(rpad(@sprintf("%.3f%%", pct), 12))
            end
            println()
        end
    end
    println("\n" * "-"^80)
end