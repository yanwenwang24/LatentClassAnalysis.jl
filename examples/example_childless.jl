using Arrow
using CategoricalArrays
using DataFrames, DataFramesMeta
using LatentClassAnalysis
using Random

Random.seed!(123)

# Load dataset
df = DataFrame(Arrow.Table("examples/childless_df.arrow"))

# Step 1: Data Preparation
data, n_categories = prepare_data(
    df,
    :age_fmarry, :marry_end, :infertility,
    :edu,
    :ocp20s, :ocp30s, :flexible, :familyleave
)

# Step 2: Model Selection - Try different numbers of classes
results = []

for n_classes in 2:6
    println("\nFitting model with $n_classes classes...")

    # Initialize model
    model = LCAModel(n_classes, size(data, 2), n_categories)

    # Fit model and get log-likelihood
    ll = fit!(model, data, verbose=true)

    # Calculate diagnostics
    diag = diagnostics!(model, data, ll)

    # Store results
    push!(results, (
        n_classes = n_classes,
        model = model,
        diagnostics = diag
    ))

    println("Log-likelihood: $(diag.ll)")
    println("AIC: $(diag.aic)")
    println("BIC: $(diag.bic)")
    println("SBIC: $(diag.sbic)")
    println("Entropy: $(diag.entropy)")
end

# Find best model based on BIC
best_n_classes = argmin(k -> results[k].diagnostics.bic, keys(results)) + 1
best_model = results[best_n_classes - 1].model
println("\nBest model has $best_n_classes classes based on BIC")

# Step 3: Analyze best model
# Show profiles
show_profiles(best_model, df, [:age_fmarry, :marry_end, :infertility, :edu, :ocp20s, :ocp30s, :flexible, :familyleave])

# Get predictions
assignments, probabilities = predict(best_model, data)

# Add predicted classes to original DataFrame
df[!, :predicted_class] = assignments