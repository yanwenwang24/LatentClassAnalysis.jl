using Arrow
using CategoricalArrays
using DataFrames, DataFramesMeta
using LatentClassAnalysis
using Random

Random.seed!(1024)

# Load dataset
df = DataFrame(Arrow.Table("examples/childless_df.arrow"))

# Step 1: Data Preparation
data, n_categories = prepare_data(
    df,
    # Indicators for the partnership domain during respondents' 20s and 30s
    :age_fmarry, # marriage timing ("no", "early", "norm", "late")
    :marry_end, # whether marriage dissolved (0/1)
    :infertility, # whether infertility is reported (0/1)
    # Indicators for the education domain
    :edu, # education level ("low", "medium", "high")
    # Indicators for the occupational domain during respondents' 20s and 30s
    :ocp20s, # occupation in 20s ("Unemployed", "Blue-collared", "Semi-professional", "Professional")
    :ocp30s, # ... in 30s
    :flexible, # whether flexible work arrangements are available (0/1)
    :familyleave # whether generous family leave is available (0/1)
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