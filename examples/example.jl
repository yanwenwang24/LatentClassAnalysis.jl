using LatentClassAnalysis
using DataFrames
using Random

# Generate synthetic data
Random.seed!(123)
n_samples = 1000

# True class assignments (2 latent classes)
true_classes = rand(1:2, n_samples)

# Generate responses for 4 binary items
# Different response patterns for each class
function generate_response(class)
    if class == 1
        return rand() < 0.8 ? 2 : 3  # High probability of 1
    else
        return rand() < 0.3 ? 1 : 2  # Low probability of 1
    end
end

# Create DataFrame with responses
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
best_n_classes = argmin(k -> results[k].diagnostics.bic, keys(results)) + 1
best_model = results[best_n_classes - 1].model
println("\nBest model has $best_n_classes classes based on BIC")

# Step 3: Analyze best model
# Get predictions
assignments, probabilities = predict(best_model, data)

# Add predicted classes to original DataFrame
df[!, :predicted_class] = assignments

# Calculate class sizes
class_sizes = [sum(assignments .== k) / length(assignments) for k in 1:best_n_classes]
println("\nClass sizes:")
for (k, size) in enumerate(class_sizes)
    println("Class $k: $(round(size * 100, digits=1))%")
end

# Show item response probabilities for each class
println("\nItem response probabilities:")
for j in 1:best_model.n_items
    println("\nItem $j:")
    for k in 1:best_model.n_classes
        probs = best_model.item_probs[j][k, :]
        println("Class $k: $probs")
    end
end

# Example output for first few cases
println("\nSample of individual predictions:")
first_few = 5
println("Row\tMost Likely Class\tClass Probabilities")
for i in 1:first_few
    probs = round.(probabilities[i, :], digits=3)
    println("$i\t$(assignments[i])\t\t$probs")
end