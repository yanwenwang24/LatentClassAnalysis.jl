# [Model selection](@id guide-model-selection)

The number of classes is not estimated by a latent class model; the analyst fits models
with different numbers of classes and chooses between them. This page walks through the
tools the package offers for that choice on simulated data with three known classes:
the model-selection table of [`diagnostics`](@ref), the fit flags and the log-likelihoods
of the random starts (which tell you whether a fit can be trusted at all), and the
bootstrap likelihood-ratio test [`bootstrap_lrt`](@ref). It ends with a recommended
procedure. The definitions of the criteria and the algorithm of the test are in
[Choosing the number of classes](@ref choosing-k).

## Data with three known classes

Six yes/no items, 600 respondents, three classes of equal size: one class answers 1 on
every item with high probability, one rarely does, and one answers 1 on items 1, 3 and 5
only. The data are drawn from a `StableRNG` so that the page is reproducible.

```@example selection
using LatentClassAnalysis, DataFrames, StableRNGs

rng = StableRNG(11)
n = 600
truth = rand(rng, 1:3, n)
p = [0.85 0.80 0.90 0.80 0.85 0.75;      # class 1: high on every item
     0.15 0.20 0.10 0.20 0.15 0.25;      # class 2: low on every item
     0.85 0.20 0.85 0.20 0.80 0.30]      # class 3: high on items 1, 3 and 5
items = [Symbol("item", j) for j in 1:6]
df = DataFrame([items[j] => [Int(rand(rng) < p[c, j]) for c in truth] for j in 1:6]...)
d = prepare_data(df, items)
```

## The model-selection table

Passing a range to [`fit`](@ref) fits one model per number of classes. Every model is
estimated from 20 random starts (see below), so a comparison between the models is a
comparison between well-searched maxima rather than between lucky and unlucky starting
values. [`diagnostics`](@ref) collects the fit statistics of every model, and because a
vector of diagnostics is a Tables.jl table, `DataFrame` turns it into the usual
model-selection table:

```@example selection
models = fit(LCAModel, d, 1:5; rng = StableRNG(1))
selection = DataFrame(diagnostics(models))
```

How to read the columns:

- `ll` is the maximised log-likelihood. It always increases with the number of classes,
  so it cannot choose between the models on its own.
- `dof` is the number of free parameters, ``(K - 1) + K \sum_j (C_j - 1)``: 7 more for
  every additional class with six binary items.
- `aic`, `bic` and `sbic` are ``-2\,\ell`` plus a penalty per parameter: 2 for AIC,
  ``\log n \approx 6.4`` for BIC at ``n = 600``, and ``\log((n + 2)/24) \approx 3.2`` for
  the sample-size adjusted BIC. **Lower is better.** BIC penalises most heavily and is
  the criterion that recovers the true number of classes most reliably in simulation
  studies [nylund2007](@cite); AIC tends to choose too many classes.
- `entropy` is the relative entropy of the classification, in ``[0, 1]``: 1 when every
  respondent is assigned to one class with certainty, 0 when the posterior probabilities
  are uniform. It describes how cleanly the classes separate respondents, **not** how
  well the model fits, and it is not monotone in the number of classes (here it drops
  from the two- to the three-class model and rises again). Do not use it to choose
  ``K``; use it to decide whether the modal assignments of [`classify`](@ref) are
  trustworthy (values above roughly 0.8 are conventionally regarded as good). The
  one-class model has entropy 1 by convention.
- `converged` says whether EM met its tolerance within `max_iter` iterations.

Here the three criteria agree on three classes, which is how the data were generated:

```@example selection
argmin(selection.bic), argmin(selection.aic), argmin(selection.sbic)
```

The AIC margin between three and four classes is small (about one unit), which is
typical: AIC is nearly indifferent to an extra class that improves the log-likelihood by a
few units, while BIC is not.

## Was the maximum found?

Every criterion above is computed from the *best* log-likelihood the fit found, and EM
converges to a local maximum that depends on its starting values. [`fit`](@ref) therefore
runs `n_starts` (default 20) short EM runs of `short_iters` (default 50) iterations from
random starting values, continues the `n_final` (default 4) best of them to convergence,
and keeps the best (see [Random restarts and local maxima](@ref restarts)). The
log-likelihood reached by every start is stored in the model, so you can see whether the
best value was found more than once; starts that were not continued keep their short-run
value:

```@example selection
best = models[3]
sort(best.start_loglik; rev = true)[1:6]
```

The four continued starts all ended at the same log-likelihood: the maximum was
*replicated*, and `fit` records this in the fit flags. A maximum that only one of the
continued starts reached is suspect — a better one may exist, and the estimates at a
poorly determined maximum are unstable — so `fit` warns and sets the flag to `false`.
Even with the default 20 starts, the five-class model's best value was found only once:

```@example selection
[m.flags.best_ll_replicated for m in models]
```

(The warning itself is printed when the model is fitted; log messages are not shown on
this page.) A non-replicated maximum is one of the symptoms of asking for more classes
than the data support, together with response probabilities estimated at exactly 0 or 1
(`m.flags.n_boundary`) and empty classes.

Fewer starts are cheaper and riskier. With only two starts, both continued, one of the two
runs on this data stops at a local maximum 42 log-likelihood units below the best one, and
the fit warns:

```@example selection
cheap = fit(LCAModel, d, 3; rng = StableRNG(5), n_starts = 2, n_final = 2)
cheap.flags.best_ll_replicated, round.(cheap.start_loglik; digits = 3)
```

Had this been the only fit, the model would still have been returned — with the right
log-likelihood for *its* maximum but the wrong one for the data. The remedy for a
non-replicated maximum is always the same: increase `n_starts` and `n_final` and refit.
The defaults are enough for most applications with a handful of items; models with many
classes, many items or weakly separated classes may need 50 or 100 starts. `short_iters`
controls how long each short run gets to reveal where it is heading; raising it helps when
the short-run log-likelihoods are all similar. `multithreaded = true` runs the starts on
all Julia threads with identical results.

## The bootstrap likelihood-ratio test

The information criteria rank the models but do not test them. The bootstrap
likelihood-ratio test compares ``K`` against ``K + 1`` classes: it simulates data sets from
the fitted ``K``-class model, refits both models to each of them, and reports the share of
simulated data sets whose statistic ``2(\ell_{K+1} - \ell_K)`` is at least as large as the
observed one (see [Bootstrap likelihood-ratio test](@ref blrt)). A small p-value says that
the gain from the extra class is larger than ``K`` classes would produce by chance.

The convenience form `bootstrap_lrt(d, k; ...)` fits the two models and runs the test.
The usual procedure tests 1 against 2 classes, 2 against 3, and so on, and stops at the
first ``K`` that is not rejected:

```@example selection
tests = [bootstrap_lrt(d, k; n_boot = 19, rng = StableRNG(100 + k)) for k in 1:3]
DataFrame(K = 1:3, statistic = [t.statistic for t in tests], pvalue = pvalue.(tests))
```

One and two classes are rejected; three are not, in agreement with BIC. Printing a test
shows the observed statistic against the distribution of the replicates:

```@example selection
tests[3]
```

Two things to keep in mind:

- **Resolution.** With `n_boot` replicates the p-value cannot be smaller than
  `1 / (n_boot + 1)`, so 19 replicates resolve the 5% level and no more (the p-values of
  0.05 above are at that floor) and 99 resolve 1%. A small `n_boot` is fine for a first
  look; use a few hundred replicates when the p-value comes out close to the threshold,
  and remember that every replicate costs one fit of each model.
- **The alternative must be a good maximum.** If the ``(K + 1)``-class fit stopped at a
  local maximum, the observed statistic is too small and the test is conservative;
  `bootstrap_lrt` warns when the alternative model's best log-likelihood was not
  replicated across its starts. Fit with more starts before testing in that case. The
  replicate fits use fewer starts than a careful analysis (`n_starts_boot`, default 10),
  which makes the test slightly liberal if the replicate ``(K + 1)``-class fits miss their
  maxima; the `n_negative` field of the result counts replicates where that visibly
  happened.

The two-model form `bootstrap_lrt(models[3], models[4]; ...)` tests models that are
already fitted, which avoids refitting them, and is the form to use after a careful fit
with many starts.

## A recommended procedure

1. Fit models from one class up to a few more than you expect, with the default restarts,
   and look at the model-selection table.
2. Check the flags of every candidate model: a best log-likelihood that was not
   replicated, many boundary probabilities, or an empty class mark a model that is not
   well determined. Refit with more starts if the maximum was not replicated; if the
   other flags persist, the model has more classes than the data support.
3. Choose by BIC, and look at sBIC and AIC to see how sensitive the choice is. When the
   criteria disagree, the bootstrap likelihood-ratio test of the smaller model against the
   next one settles whether the extra class is more than chance.
4. Ask whether the additional class is substantively interpretable and large enough to
   matter. A class of a few per cent that differs from another class on one item is rarely
   worth keeping, whatever the criteria say; and with large samples the criteria and the
   test will eventually reject every ``K`` because no model is exactly true.
5. Use the entropy and the posterior probabilities of the selected model to decide whether
   to work with the modal assignments of [`classify`](@ref) or to carry the posterior
   probabilities of [`predict`](@ref) forward.
