# Getting started

This tutorial walks through a complete latent class analysis on simulated data:
building a `DataFrame` with the kinds of columns the package accepts, recoding it,
fitting models with different numbers of classes, choosing one, reading its
profile, and attaching class assignments to the data. All code blocks on this page
share one session, so later blocks can use variables defined in earlier ones.

## Simulating data with three known classes

We generate 1000 respondents from three hidden classes answering six items. Items
1–4 are 0/1 integers, item 5 is a string column with the values `"no"` and `"yes"`,
and item 6 is a `CategoricalArray` with the ordered levels `"low"`, `"middle"`,
`"high"`. The data are drawn from a `StableRNG` so that they are identical across
Julia versions.

```@example tutorial
using LatentClassAnalysis, DataFrames, CategoricalArrays, Random, StableRNGs

rng = StableRNG(2024)
n = 1000
truth = rand(rng, 1:3, n)                # the hidden class of each respondent

# Probability of answering 1 on items 1–4, by class (rows) and item (columns)
p_binary = [0.85 0.80 0.90 0.80;         # class 1: high on all four items
            0.15 0.20 0.10 0.20;         # class 2: low on all four items
            0.85 0.20 0.85 0.20]         # class 3: high on items 1 and 3 only
# Probability of "yes" on item 5, by class
p_yes = [0.80, 0.20, 0.50]
# Probability of "low", "middle", "high" on item 6, by class
p_level = [0.10 0.20 0.70;
           0.70 0.20 0.10;
           0.20 0.60 0.20]

draw(rng, p) = searchsortedfirst(cumsum(p), rand(rng))   # sample a category index

df = DataFrame(
    item1 = [Int(rand(rng) < p_binary[c, 1]) for c in truth],
    item2 = [Int(rand(rng) < p_binary[c, 2]) for c in truth],
    item3 = [Int(rand(rng) < p_binary[c, 3]) for c in truth],
    item4 = [Int(rand(rng) < p_binary[c, 4]) for c in truth],
    item5 = [rand(rng) < p_yes[c] ? "yes" : "no" for c in truth],
    item6 = categorical([("low", "middle", "high")[draw(rng, p_level[c, :])] for c in truth];
                        levels = ["low", "middle", "high"]),
)
first(df, 5)
```

## Preparing the data

[`prepare_data`](@ref) takes the `DataFrame` and the names of the indicator
columns and returns an integer matrix with codes `1, 2, …` plus the number of
categories of every item. Integer columns are coded by the rank of their sorted
values (so `0/1` becomes `1/2`), string columns by their sorted distinct values
(`"no"` is 1 and `"yes"` is 2), and categorical columns by their level order (`"low"`
is 1, `"middle"` is 2, `"high"` is 3). If you want a string column coded in a
particular order, convert it with `categorical(col; levels = [...])` first.

```@example tutorial
cols = [:item1, :item2, :item3, :item4, :item5, :item6]
data, n_categories = prepare_data(df, cols...)
n_categories
```

```@example tutorial
data[1:5, :]
```

## Fitting models with two to five classes

The number of classes is not estimated but chosen by comparing models. We fit
models with two to five classes and collect the log-likelihood, the number of free
parameters, the information criteria, and the relative entropy of each. Starting
values are drawn from Julia's global random number generator, so we call
`Random.seed!` before constructing each model to make the run reproducible.

```@example tutorial
results = DataFrame(k = Int[], loglik = Float64[], npar = Int[],
                    AIC = Float64[], BIC = Float64[], sBIC = Float64[], entropy = Float64[])
models = Dict{Int,LCAModel}()

for k in 2:5
    Random.seed!(2024)
    model = LCAModel(k, size(data, 2), n_categories)
    ll = fit!(model, data)
    d = diagnostics!(model, data, ll)
    npar = (k - 1) + k * sum(n_categories .- 1)
    push!(results, (k, ll, npar, d.aic, d.bic, d.sbic, d.entropy))
    models[k] = model
end
results
```

If you run this loop yourself, constructing the five-class model prints a warning
(log messages are not captured on this page): with six items of which the smallest
has two categories, the package's rule of thumb recommends seven items for five
classes (see [Identifiability](@ref identifiability)). The model is still fitted, but
its solution should be treated with caution.

## Choosing the number of classes

The log-likelihood always increases with more classes, so the information criteria
penalise the number of parameters; lower is better. We follow the usual
recommendation and choose by BIC (see [Choosing the number of classes](@ref choosing-k) for why),
which here recovers the three classes we simulated.

```@example tutorial
best_k = results.k[argmin(results.BIC)]
```

```@example tutorial
best = models[best_k]
show_profiles(best, df, cols)
```

### How to read the profile

- **Class sizes** are the estimated share of the population in each class
  (`best.class_probs`).
- Each item then gets one table whose **columns are the classes and rows are the
  response categories**; the entries are the probability, within the class, of each
  response, so every column of every table sums to 100%. In the output above, the
  class that answers 0 on items 1–4 with high probability and is mostly `"low"` on
  item 6 is the "low on everything" class we simulated.
- Items whose rows differ sharply across the columns are the ones that
  distinguish the classes; items with similar rows in every column contribute
  little to the classification.
- **Class numbers are arbitrary.** They depend on the random starting values, and
  a different seed may return the same classes in a different order. Name the
  classes by their profiles, not by their numbers.

## Attaching class assignments to the data

[`predict`](@ref) returns each respondent's most likely class and the full matrix of
posterior membership probabilities. Because we simulated the data we can also check
the assignments against the true classes; the labels differ, but each simulated
class maps onto one estimated class.

```@example tutorial
assignments, posterior = predict(best, data)
df.class = assignments
df.max_posterior = vec(maximum(posterior, dims = 2))
first(df, 5)
```

```@example tutorial
tab = combine(groupby(DataFrame(truth = truth, class = assignments), [:truth, :class]), nrow => :n)
unstack(tab, :truth, :class, :n)
```

## Caveats

- **One random start.** Version 0.2 fits each model once, from a single random
  starting point. EM can stop at a local maximum, so for any model you intend to
  report, fit it from several seeds and keep the fit with the highest
  log-likelihood. The loop below does exactly that for the three-class model:

```@example tutorial
fits = map(1:10) do seed
    Random.seed!(seed)
    m = LCAModel(3, size(data, 2), n_categories)
    (loglik = fit!(m, data), model = m)
end
best_fit = argmax(f -> f.loglik, fits)
best_fit.loglik
```

  If the log-likelihoods differ across seeds, the highest one is the one to use;
  if they agree, as here, the solution is well determined.

- **Sample size.** [`fit!`](@ref) warns when there are fewer than 300 observations.
  Small samples produce unstable profiles, especially with many classes or rare
  response categories.
- **Missing values** are not supported: drop incomplete rows before calling
  [`prepare_data`](@ref).
- **Entropy** describes how cleanly respondents are classified, not how well the
  model fits. With low entropy, carry the posterior probabilities forward instead
  of the hard assignments.

See [Limitations of the current version and roadmap](@ref limitations) for what is planned
for version 0.3.0.
