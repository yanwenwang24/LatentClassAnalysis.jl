# Latent class analysis of simulated data.
#
# Run from the repository root after setting up the examples environment once:
#   julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=examples examples/example.jl

using CategoricalArrays
using DataFrames
using LatentClassAnalysis
using Random

# Generate synthetic data
Random.seed!(123)
n_samples = 1000

# True class assignments (2 latent classes)
true_classes = rand(1:2, n_samples)

# Binary responses with class-specific probabilities of answering 1:
# class 1 answers 1 with probability 0.8, class 2 with probability 0.3
function generate_response(class)
    p = class == 1 ? 0.8 : 0.3
    return rand() < p ? 1 : 2
end

# Create DataFrame with three informative items and two uninformative categorical items
df = DataFrame(
    item1 = [generate_response(c) for c in true_classes],
    item2 = [generate_response(c) for c in true_classes],
    item3 = [generate_response(c) for c in true_classes],
    item4 = categorical([rand(["Yes", "No"]) for _ in 1:n_samples]),
    item5 = categorical([rand(["Yes", "No"]) for _ in 1:n_samples])
)

# Step 1: Data Preparation
data, n_categories = prepare_data(df, :item1, :item2, :item3, :item4, :item5)

# Step 2: Model Selection - Try different numbers of classes
results = []
for n_classes in 2:4
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
best = argmin(r -> r.diagnostics.bic, results)
best_model = best.model
println("\nBest model has $(best.n_classes) classes based on BIC")

# Step 3: Analyze best model
# Get predictions
assignments, probabilities = predict(best_model, data)

# Add predicted classes to original DataFrame
df[!, :predicted_class] = assignments

# Calculate class sizes
class_sizes = [sum(assignments .== k) / length(assignments) for k in 1:best.n_classes]
println("\nClass sizes:")
for (k, size) in enumerate(class_sizes)
    println("Class $k: $(round(size * 100, digits=1))%")
end

# Show item response probabilities for each class
show_profiles(best_model, df, [:item1, :item2, :item3, :item4, :item5])

# Example output for first few cases
println("\nSample of individual predictions:")
first_few = 5
println("Row\tMost Likely Class\tClass Probabilities")
for i in 1:first_few
    probs = round.(probabilities[i, :], digits=3)
    println("$i\t$(assignments[i])\t\t$probs")
end
