# Getting started

This tutorial walks through the basic workflow of a latent class analysis on simulated
data: building a `DataFrame` with the kinds of columns the package accepts, preparing it,
fitting models with different numbers of classes, choosing one, reading its profile, and
attaching class assignments to the data. All code blocks on this page share one session,
so later blocks can use variables defined in earlier ones. The guide pages take each step
further: [Model selection](@ref guide-model-selection), [Missing data](@ref guide-missing-data),
[Covariates](@ref guide-covariates) and [Standard errors and the bootstrap](@ref guide-inference).

## Simulating data with three known classes

We generate 1000 respondents from three hidden classes answering six items. Items
1–4 are 0/1 integers, item 5 is a string column with the values `"no"` and `"yes"`,
and item 6 is a `CategoricalArray` with the ordered levels `"low"`, `"middle"`,
`"high"`. The data are drawn from a `StableRNG` so that they are identical across
Julia versions.

```@example tutorial
using LatentClassAnalysis, DataFrames, CategoricalArrays, StableRNGs

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
order without converting it, pass `levels = Dict(:item5 => ["yes", "no"])`. A `missing`
entry becomes the code `0` and is handled by the fit (see
[Missing data](@ref guide-missing-data)).

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
recommendation and choose by BIC, which here recovers the three classes we simulated.
The [Model selection](@ref guide-model-selection) page explains the criteria, shows how to
check that a fit found its maximum, and adds the bootstrap likelihood-ratio test.

```@example tutorial
best = models[argmin(selection.bic)]
```

Printing the model shows its log-likelihood, degrees of freedom and BIC, how many EM
iterations the winning start took, the class sizes, and any fit flags. The classes are
ordered by size, so class 1 is always the largest.

## Reading the profile

[`show_profiles`](@ref) prints the class sizes and, for every item, the response
probabilities within each class:

```@example tutorial
show_profiles(best)
```

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
- Each percentage is followed by `±` its **standard error** in percentage points, from
  the observed information matrix that `fit` computes by default. The
  [Standard errors and the bootstrap](@ref guide-inference) page explains them, the
  `NaN` that a probability estimated at exactly 0 or 1 receives, and the bootstrap
  alternative.

The same numbers are available as a table from [`profiles`](@ref), with one row per
item, response level and class, the standard error `se` and the bounds `lower`/`upper` of
a 95% confidence interval (`classes = true` prepends the class sizes):

```@example tutorial
prof = DataFrame(profiles(best; classes = true))
first(prof, 6)
```

Because it is a table, it can be reshaped with the usual tools; for instance, item 6
with one column per class:

```@example tutorial
unstack(prof[prof.item .== :item6, :], [:item, :level], :class, :prob)
```

The free parameters on the logit scale, with standard errors, Wald tests and confidence
intervals, are tabulated by [`coeftable`](@ref)`(best)`; see the
[inference guide](@ref guide-inference).

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
reproduces a fit exactly; different seeds usually reach the same maximum but can differ
in the last digits of the log-likelihood or, for poorly determined models, in the
solution. The log-likelihood reached by every start is stored in `model.start_loglik`,
and `model.flags.best_ll_replicated` records whether the best value was found more than
once (see [Model selection](@ref guide-model-selection)). Use a `StableRNG` for results
that are stable across Julia versions; `Random.Xoshiro(seed)` is reproducible within one
Julia version.

## Going further

- [Model selection](@ref guide-model-selection): the criteria in the table above, random
  restarts and local maxima, the bootstrap likelihood-ratio test, and a decision
  procedure.
- [Missing data](@ref guide-missing-data): `missing` in the indicators, what the fit
  assumes, and the posterior of incomplete respondents.
- [Covariates](@ref guide-covariates): predicting class membership from covariates
  (latent class regression) and reading the coefficients.
- [Standard errors and the bootstrap](@ref guide-inference): the standard errors on the
  logit and probability scales, boundary estimates, `se = :none`, and bootstrap standard
  errors and intervals.
- [Example: childlessness in Singapore](@ref): the same workflow on the real data bundled
  with the package.

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
- **Standard errors** describe uncertainty given the number of classes; they say nothing
  about whether that number is right.
