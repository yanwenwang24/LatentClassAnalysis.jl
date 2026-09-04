# Latent class analysis of simulated data.
#
# Run from the repository root after setting up the examples environment once:
#   julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=examples examples/example.jl

using CategoricalArrays
using DataFrames
using LatentClassAnalysis
using StableRNGs

# ---------------------------------------------------------------------------------------
# Simulate data: 1000 respondents from two hidden classes answering five items
# ---------------------------------------------------------------------------------------
rng = StableRNG(123)
n = 1000
true_class = rand(rng, 1:2, n)

# Items 1-3 are informative: class 1 answers "yes" with probability 0.8, class 2 with 0.3.
# Items 4-5 are noise: both classes answer "Yes"/"No" at random. Item 4 is a string
# column and item 5 a CategoricalArray, to show that any column type is accepted.
yes(c) = rand(rng) < (c == 1 ? 0.8 : 0.3) ? 1 : 0
df = DataFrame(
    item1 = [yes(c) for c in true_class],
    item2 = [yes(c) for c in true_class],
    item3 = [yes(c) for c in true_class],
    item4 = [rand(rng, ("Yes", "No")) for _ in 1:n],
    item5 = categorical([rand(rng, ("Yes", "No")) for _ in 1:n]; levels = ["Yes", "No"]),
)
items = [:item1, :item2, :item3, :item4, :item5]

# ---------------------------------------------------------------------------------------
# Step 1: prepare the data (any Tables.jl table; missing values would be allowed)
# ---------------------------------------------------------------------------------------
d = prepare_data(df, items)
println(d)

# ---------------------------------------------------------------------------------------
# Step 2: fit models with one to four classes and compare them
# ---------------------------------------------------------------------------------------
# Every fit runs 20 random starts and continues the 4 best to convergence; pass an rng
# for a reproducible result. Fits with too many classes typically print a warning about
# response probabilities on the boundary (0 or 1).
models = fit(LCAModel, d, 1:4; rng = StableRNG(1))

selection = DataFrame(diagnostics(models))
println("\nModel selection:")
println(selection)

best = models[argmin(selection.bic)]
println("\nBest model by BIC has $(best.n_classes) classes")
println(best)

# ---------------------------------------------------------------------------------------
# Step 3: inspect the selected model
# ---------------------------------------------------------------------------------------
# Class profiles: class sizes and, for every item, the response probabilities per class.
# Classes are ordered by size, so class 1 is the largest.
show_profiles(best)

# The same numbers as a table, with delta-method standard errors and confidence intervals
println(first(DataFrame(profiles(best; classes = true)), 6))

# The free parameters on the logit scale, with standard errors, Wald tests and intervals
# (which = :class restricts the table to the class-membership block)
println("\nResponse logits with standard errors:")
display(coeftable(best; which = :items))

# Is a third class more than chance? Bootstrap likelihood-ratio test of 2 against 3
# classes (models[k] has k classes; 19 replicates resolve the 5% level)
println("\nBootstrap likelihood-ratio test:")
println(bootstrap_lrt(models[2], models[3]; n_boot = 19, rng = StableRNG(2)))

# Posterior membership probabilities and modal class assignments
posterior = predict(best)
df.class = classify(best)
df.max_posterior = vec(maximum(posterior; dims = 2))

println("\nAssigned class sizes:")
println(combine(groupby(df, :class), nrow => :n))

println("\nFirst rows with their assignment:")
println(first(df, 5))

# Because the data are simulated we can cross-tabulate the assignments against the truth
# (class labels are arbitrary: each true class should map onto one estimated class)
println("\nTrue class (rows) by assigned class (columns):")
tab = combine(groupby(DataFrame(truth = true_class, class = df.class), [:truth, :class]), nrow => :n)
println(unstack(tab, :truth, :class, :n))
