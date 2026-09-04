# Getting started

This tutorial walks through a complete latent class analysis on simulated data:
building a `DataFrame` with the kinds of columns the package accepts, preparing it,
fitting models with different numbers of classes, choosing one, reading its
profile, attaching class assignments to the data, and handling missing responses. All
code blocks on this page share one session, so later blocks can use variables defined
in earlier ones.

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

[`prepare_data`](@ref) takes a table and the names of the indicator columns and
returns an [`LCAData`](@ref): a matrix of integer codes `1, 2, …` per item together
with the item names and the label of every code. Integer columns are coded by the rank
of their sorted values (so `0/1` becomes `1/2`), string columns by their sorted
distinct values (`"no"` is 1 and `"yes"` is 2), and categorical columns by their level
order (`"low"` is 1, `"middle"` is 2, `"high"` is 3). To code a column in a particular
order without converting it, pass `levels = Dict(:item5 => ["yes", "no"])`.

```@example tutorial
items = [:item1, :item2, :item3, :item4, :item5, :item6]
d = prepare_data(df, items)
```

The codes are in `d.y` and the number of categories per item in `d.n_categories`:

```@example tutorial
d.y[1:5, :]
```

```@example tutorial
d.n_categories
```

## Fitting models with one to five classes

The number of classes is not estimated but chosen by comparing models. Passing a
range to [`fit`](@ref) fits one model per class count and returns a vector of
[`LCAModel`](@ref)s. Every model is estimated by EM from 20 random starting values
(50 iterations each), of which the 4 best are continued to convergence, and the best
of those is kept. The starting values are seeded from the `rng` keyword, so the fits
are reproducible.

[`diagnostics`](@ref) collects the fit statistics of every model — number of classes,
observations and free parameters, log-likelihood, AIC, BIC, sample-size adjusted BIC,
relative entropy, and convergence — into a table:

```@example tutorial
models = fit(LCAModel, d, 1:5; rng = StableRNG(1))
selection = DataFrame(diagnostics(models))
```

If you run this yourself, the four- and five-class fits print a warning (log messages
are not captured on this page) that several response probabilities were estimated at
exactly 0 or 1. Such *boundary* estimates are a typical symptom of asking for more
classes than the data support; the fit is still returned, and the warning is stored in
`model.flags`.

## Choosing the number of classes

The log-likelihood always increases with more classes, so the information criteria
penalise the number of parameters; lower is better. We follow the usual
recommendation and choose by BIC (see [Choosing the number of classes](@ref choosing-k)
for why), which here recovers the three classes we simulated.

```@example tutorial
best = models[argmin(selection.bic)]
```

Printing the model shows its log-likelihood, degrees of freedom and BIC, how many EM
iterations the winning start took, the class sizes, and any fit flags. The classes are
ordered by size, so class 1 is always the largest.

```@example tutorial
show_profiles(best)
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
- **Class numbers carry no meaning.** They are assigned by decreasing class size, so
  the numbering is stable for a given fit but says nothing about what a class is; two
  classes of nearly equal size can swap numbers between data sets or seeds. Name the
  classes by their profiles, not by their numbers.

Each percentage is followed by `±` its standard error (in percentage points), and
the class sizes by theirs. The same numbers are available as a table from
[`profiles`](@ref), with one row per item, response level and class: `se` is the
delta-method standard error of the probability and `lower`/`upper` are the bounds of a
95% confidence interval computed on the logit scale, so they stay within `[0, 1]`
(`level = 0.9` changes the level, `classes = true` prepends the class sizes).

```@example tutorial
prof = DataFrame(profiles(best; classes = true))
first(prof, 6)
```

The standard errors come from the observed information matrix, which `fit` computes
by default (`se = :none` skips it). The free parameters on the logit scale, their
standard errors, Wald tests and confidence intervals are tabulated by
[`coeftable`](@ref); `which = :class` restricts it to the class-membership block
(here the log-odds of each class against class 1) and `which = :items` to the
response logits. A probability estimated at exactly 0 or 1 is held fixed and has no
standard error (`NaN`), which `fit` reports in its warning; the other cells of its row
keep theirs, conditional on the boundary cell being fixed.

```@example tutorial
coeftable(best; which = :class)
```

Because it is a table, it can be reshaped with the usual tools; for instance, item 6
with one column per class:

```@example tutorial
unstack(prof[prof.item .== :item6, :], [:item, :level], :class, :prob)
```

## Attaching class assignments to the data

[`classify`](@ref) returns each respondent's most likely class and [`predict`](@ref)
the full matrix of posterior membership probabilities (one row per respondent, one
column per class). Both also accept new data — an `LCAData` or a table with the same
columns — which is coded with the levels of the training data.

```@example tutorial
df.class = classify(best)
posterior = predict(best)
df.max_posterior = vec(maximum(posterior, dims = 2))
first(df, 5)
```

Because we simulated the data we can check the assignments against the true classes.
The labels differ from the simulation's (class 1 of the fit is the largest class), but
each simulated class maps onto one estimated class. Class 3 of the simulation, which
gives the same answers as class 1 on items 1 and 3, is the least cleanly separated one.

```@example tutorial
tab = combine(groupby(DataFrame(truth = truth, class = df.class), [:truth, :class]), nrow => :n)
unstack(tab, :truth, :class, :n)
```

## Random restarts and local maxima

EM converges to a local maximum of the likelihood that depends on where it starts.
Rather than fitting each model once, [`fit`](@ref) runs `n_starts` short EM runs from
random starting values, continues the `n_final` best of them to convergence, and
keeps the best. The log-likelihood reached by every start is stored in
`start_loglik`; starts that were not continued keep the value they reached after
`short_iters` iterations. Sorting it shows how many of the continued starts ended at
the same maximum:

```@example tutorial
sort(best.start_loglik; rev = true)[1:6]
```

The four continued starts all reached the same log-likelihood, so the solution is
well determined. When the best log-likelihood is found by only one of the continued
starts, `fit` warns, sets `model.flags.best_ll_replicated` to `false`, and the
remedy is to increase `n_starts` and `n_final`:

```@example tutorial
best.flags.best_ll_replicated
```

Fewer starts are cheaper but riskier. The four-class model fitted from five starts
with only the best one continued stops at a lower log-likelihood than the same model
from the default 20 starts:

```@example tutorial
cheap = fit(LCAModel, d, 4; rng = StableRNG(1), n_starts = 5, n_final = 1)
loglikelihood(cheap), loglikelihood(models[4])
```

## Testing the number of classes

The information criteria rank the models but do not test them. The bootstrap
likelihood-ratio test ([`bootstrap_lrt`](@ref)) compares ``K`` against ``K + 1``
classes: it simulates data sets from the fitted ``K``-class model, refits both models to
each of them, and reports the share of simulated data sets whose statistic
``2(\ell_{K+1} - \ell_K)`` is at least as large as the observed one (see
[Bootstrap likelihood-ratio test](@ref blrt)). The convenience form takes the data and
``K``, fits both models and runs the test. `n_boot` is the number of simulated data sets;
the p-value cannot go below `1 / (n_boot + 1)`, so 19 replicates resolve the 5% level,
and a few hundred are advisable when the p-value is near the threshold.

```@example tutorial
test23 = bootstrap_lrt(d, 2; n_boot = 19, rng = StableRNG(3))
```

None of the 19 data sets simulated from two classes comes close to the observed gain of
the third class, so the p-value is at its floor and two classes are rejected. The
two-model form tests models that are already fitted; three against four classes tells
the opposite story:

```@example tutorial
test34 = bootstrap_lrt(models[3], models[4]; n_boot = 19, rng = StableRNG(4))
pvalue(test34)
```

The gain of a fourth class is typical of what three classes produce by chance, so the
sequence of tests stops at three classes, in agreement with BIC.

## Bootstrap standard errors

[`bootstrap`](@ref) resamples the respondents with replacement, refits the model to
each resample (warm-started from the fitted model), matches the class labels of every
refit to the model, and collects the refitted parameters. Its standard errors and
percentile confidence intervals do not rely on the asymptotic approximation behind the
observed information matrix (see [Bootstrap standard errors](@ref bootstrap-se)). Fifty
replicates keep this page quick; a few hundred are the usual choice.

```@example tutorial
boot = bootstrap(best; n_boot = 50, rng = StableRNG(5))
```

[`coeftable`](@ref) of the result lists the estimates with their bootstrap standard
errors and percentile intervals on the logit scale, and [`profiles`](@ref) the response
probabilities with the percentile intervals of the replicate probabilities:

```@example tutorial
coeftable(boot; which = :class)
```

```@example tutorial
first(DataFrame(profiles(boot; classes = true)), 6)
```

Most bootstrap standard errors are close to those from the observed information
matrix, as they should be in a sample of this size with well separated classes. The
largest ratios belong to response probabilities near 0 or 1, whose logits are poorly
determined: there the bootstrap reflects the skewed sampling distribution that the
observed-information approximation misses.

```@example tutorial
round.(stderror(boot) ./ stderror(best); digits = 2)
```

## Missing responses

Missing values in the indicators are allowed: `missing` becomes the code `0`, which
the E-step skips, so an incomplete row still contributes the items it does have. The
class sizes are estimated from every row and the response probabilities of an item
from the rows where it is observed, which is the maximum-likelihood treatment under
the missing-at-random assumption (see [Missing data](@ref missing-data)). Here we
blank out 10% of item 2 and refit the three-class model:

```@example tutorial
df_m = copy(df[:, items])
allowmissing!(df_m, :item2)
df_m.item2[randsubseq(StableRNG(7), 1:n, 0.10)] .= missing
d_m = prepare_data(df_m, items)
```

[`hasmissing`](@ref) and [`nmissing`](@ref) report the missing responses per item:

```@example tutorial
hasmissing(d_m), nmissing(d_m)
```

```@example tutorial
m_m = fit(LCAModel, d_m, 3; rng = StableRNG(1))
round.(m_m.class_probs; digits = 3), round.(best.class_probs; digits = 3)
```

The class sizes and profiles are close to those from the complete data. Rows with all
indicators missing are kept (with a warning) and receive the class sizes as their
posterior; covariates do not accept missing values.

## Input from any table

[`prepare_data`](@ref) accepts any Tables.jl source, not only a `DataFrame`: a
`NamedTuple` of vectors, a vector of `NamedTuple`s, an Arrow or CSV table, and so on.
`fit` can also take the table directly and prepare it for you:

```@example tutorial
table = (a = df.item1, b = df.item5, c = df.item6)
d_nt = prepare_data(table, [:a, :b, :c])
```

```@example tutorial
m_nt = fit(LCAModel, table, [:a, :b, :c], 2; rng = StableRNG(1))
```

## Reproducibility

The fitted model depends on the random starting values, which are drawn from the
`rng` keyword of [`fit`](@ref). Passing the same seeded generator to the same data
reproduces a fit exactly; different seeds usually reach the same maximum (as the
[restarts section](@ref Random-restarts-and-local-maxima) shows) but can differ in
the last digits of the log-likelihood or, for poorly determined models, in the
solution. Use a `StableRNG` for results that are stable across Julia versions;
`Random.Xoshiro(seed)` is reproducible within one Julia version.

## Caveats

- **Entropy** describes how cleanly respondents are classified, not how well the
  model fits. With low entropy, carry the posterior probabilities forward instead
  of the hard assignments.
- **Boundary estimates.** Response probabilities of exactly 0 or 1 are flagged in
  `model.flags.n_boundary`. A few are common in small samples; many, or an empty
  class, suggest the model has too many classes for the data.
- **Sample size.** Small samples produce unstable profiles, especially with many
  classes or rare response categories; the fit flags and the replication of the best
  log-likelihood across starts are the symptoms to watch.
- **Standard errors** are asymptotic (observed information) and describe uncertainty
  given the number of classes; they say nothing about whether that number is right.
  Standard errors of boundary estimates are undefined and reported as `NaN`; those of
  the other cells in the same row are conditional on the boundary cell being fixed.
- **The bootstrap likelihood-ratio test** resolves p-values only to `1 / (n_boot + 1)`
  and, like the information criteria, assumes that the ``K``-class model is correctly
  specified; use several hundred replicates when the p-value is near the threshold, and
  make sure the alternative model is a well-replicated maximum before testing.
