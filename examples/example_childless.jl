# Latent class analysis of pathways to permanent childlessness in Singapore.
#
# This example replicates the article:
#   Wang, Yanwen, Bussarawan Teerawichitchainan, and Christine Ho. 2024.
#   "Diverse Pathways to Permanent Childlessness in Singapore: A Latent Class Analysis."
#   Advances in Life Course Research 61:100628. doi: 10.1016/j.alcr.2024.100628.
#
# Run from any directory after setting up the examples environment once:
#   julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=examples examples/example_childless.jl

using Arrow
using DataFrames
using LatentClassAnalysis
using StableRNGs
using Statistics

# Load the dataset (bundled next to this script)
df = DataFrame(Arrow.Table(joinpath(@__DIR__, "childless_df.arrow")))

# ---------------------------------------------------------------------------------------
# Step 1: indicators and their level order
# ---------------------------------------------------------------------------------------
indicators = [
    # Partnership domain during respondents' 20s and 30s
    :age_fmarry,  # marriage timing ("no", "early", "norm", "late")
    :marry_end,   # whether the marriage dissolved (0/1)
    :infertility, # whether infertility is reported (0/1)
    # Education domain
    :edu,         # education level ("low", "middle", "high")
    # Occupational domain during respondents' 20s and 30s
    :ocp20s,      # occupation in 20s ("Unemployed", "Blue-collared", "Semi-professional", "Professional")
    :ocp30s,      # ... in 30s
    :flexible,    # whether flexible work arrangements are available (0/1)
    :familyleave, # whether generous family leave is available (0/1)
]

# String columns are coded alphabetically unless the level order is given explicitly.
# The order only affects the display of the profiles, not the fit.
occupations = ["Unemployed", "Blue-collared", "Semi-professional", "Professional"]
levels = Dict(:age_fmarry => ["no", "early", "norm", "late"],
              :edu => ["low", "middle", "high"],
              :ocp20s => occupations,
              :ocp30s => occupations)

# ---------------------------------------------------------------------------------------
# Step 2: fit models with two to six classes and choose by BIC
# ---------------------------------------------------------------------------------------
# fit(LCAModel, table, items, ks) prepares the table and fits one model per class count,
# each from 20 random starts. Fits with many classes print a warning about response
# probabilities on the boundary (0 or 1); that is expected with 493 observations.
models = fit(LCAModel, df, indicators, 2:6; levels = levels, rng = StableRNG(1024))

selection = DataFrame(diagnostics(models))
println("Model selection:")
println(selection)

best = models[argmin(selection.bic)]
println("\nBest model by BIC has $(best.n_classes) classes")
println(best)

# The best log-likelihood of every start; the maximum should be reached more than once
println("\nLog-likelihood of the continued starts (best first):")
println(sort(best.start_loglik; rev = true)[1:best.options.n_final])

# ---------------------------------------------------------------------------------------
# Step 3: profiles and class membership
# ---------------------------------------------------------------------------------------
show_profiles(best;
              var_names = ["Marriage timing", "Marriage dissolved", "Infertility", "Education",
                           "Occupation in 20s", "Occupation in 30s", "Flexible work", "Family leave"])

df.class = classify(best)
df.max_posterior = vec(maximum(predict(best); dims = 2))

println("Class sizes and composition:")
println(combine(groupby(df, :class), nrow => :n,
                :female => mean => :share_female, :age => mean => :mean_age,
                :max_posterior => mean => :mean_max_posterior))
