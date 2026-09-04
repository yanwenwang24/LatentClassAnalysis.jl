# [Standard errors and the bootstrap](@id guide-inference)

A fitted model reports every class size and response probability with a standard error,
and every free parameter with a standard error, a Wald test and a confidence interval.
This page shows where those numbers come from, on which scale they live, what happens
when an estimate sits on the boundary of the parameter space, how to switch the
computation off, and when to replace the asymptotic standard errors by the bootstrap.
The formulas are in [Standard errors and confidence intervals](@ref standard-errors) on
the methodology page.

## A model with a boundary estimate

500 respondents from two classes answering five yes/no items and one three-level item on
which class 2 never answers `"high"`. The maximum-likelihood estimate of that response
probability is therefore 0, on the boundary of the parameter space, which is a common
situation in real data with rare responses and something the standard errors have to deal
with.

```@example inference
using LatentClassAnalysis, DataFrames, StableRNGs, LinearAlgebra

rng = StableRNG(41)
n = 500
truth = rand(rng, 1:2, n)
p = [0.85 0.80 0.90 0.75 0.80;           # class 1: mostly 1
     0.15 0.20 0.10 0.20 0.25]           # class 2: mostly 0
items = [Symbol("item", j) for j in 1:5]
df = DataFrame([items[j] => [Int(rand(rng) < p[c, j]) for c in truth] for j in 1:5]...)
q = [0.30 0.30 0.40;                     # class 1: "low", "mid", "high"
     0.50 0.50 0.00]                     # class 2: never "high"
draw(rng, pr) = searchsortedfirst(cumsum(pr), rand(rng))
df.item6 = [("low", "mid", "high")[draw(rng, q[c, :])] for c in truth]
push!(items, :item6)
d = prepare_data(df, items; levels = Dict(:item6 => ["low", "mid", "high"]))
m = fit(LCAModel, d, 2; rng = StableRNG(1))
```

[`fit`](@ref) computes the covariance matrix of the free parameters from the observed
information matrix by default (`se = :hessian`): the gradient of the log-likelihood is
evaluated analytically at the maximum and its Hessian by finite differences. It warns (log
messages are not shown on this page) that one probability is on the boundary and that its
standard error is undefined, and the printed model says the same: the standard errors are
available for 14 of the 15 parameters.

## The parameters and their scale

The free parameters of the model live on the *logit* scale, which is where the asymptotic
normal approximation behind the standard errors works best. [`coef`](@ref) returns them,
[`coefnames`](@ref) labels them, and [`coeftable`](@ref) tabulates them with standard
errors, Wald ``z`` statistics and p-values against zero, and confidence intervals:

```@example inference
coeftable(m)
```

The first block is the class-membership model: without covariates it is the log-odds of
each class against class 1, here a single row (see [Covariates](@ref guide-covariates) for
the general case). The remaining rows are the item logits: `item1[0/1]|class1` is the
log-odds of answering `0` rather than `1` on item 1 in class 1, where `1` is the
*reference level* of that row — the most probable response of that class on that item,
chosen so that a rare response never serves as the reference. A binary item has one logit
per class, the three-level item has two. The `NaN` row belongs to the boundary estimate,
discussed below.

[`stderror`](@ref), [`confint`](@ref)`(m; level)` and [`vcov`](@ref) return the same
numbers as vectors and matrices, in the order of `coef(m)`:

```@example inference
round.(stderror(m); digits = 3)
```

The Wald test against zero is only interesting for the class-membership coefficients: a
response logit of zero means that the class is indifferent between two levels, which is
rarely a hypothesis of interest.

## Probabilities: `profiles` and `show_profiles`

Most readers want the response probabilities, not their logits. [`show_profiles`](@ref)
prints them as percentages followed by `±` their standard error in percentage points,
computed from the logit-scale covariance matrix by the delta method:

```@example inference
show_profiles(m)
```

[`profiles`](@ref) gives the same numbers as a table, one row per item, level and class,
with the standard error and the bounds of a confidence interval (`level = 0.95` by
default; `classes = true` prepends the class sizes). The interval is computed on the
logit scale and mapped back, so it stays inside ``[0, 1]`` and is asymmetric around the
estimate — the right shape for a probability of 0.05 or 0.95. The rows of the three-level
item show how the boundary is handled:

```@example inference
prof = DataFrame(profiles(m; classes = true))
prof[prof.item .== :item6, :]
```

The probability of `"high"` in class 2 was estimated at (numerically) zero. A parameter on
the boundary has no standard error in the Wald sense, so its `se` is `NaN` and the
interval collapses to the point. The other two cells of that row, `"low"` and `"mid"`,
keep a standard error, computed *conditionally* on the boundary cell being fixed at zero:
with the third cell fixed, the row is in effect a binary choice between `"low"` and
`"mid"`, which is why the two standard errors are equal. This is the convention of Mplus
and Latent GOLD, and it is the right reading of those standard errors: they describe the
uncertainty of the split between the observed levels, not of whether the boundary cell
is really zero. For a binary item a boundary cell fixes the whole row, and every entry of
that row is `NaN`. The same rules apply to the class sizes: an empty class has no standard
error and the other class sizes are conditional on it.

The class sizes of a model *with covariates* are sample averages of the covariate-specific
membership probabilities, and their standard errors are `NaN` for a different reason: the
delta method has nothing to offer for them, but the bootstrap does (see below).

## Skipping the standard errors

`se = :none` skips the information matrix altogether. The fit is identical; the model
prints "standard errors: none", every accessor of the covariance matrix throws, and the
`se`, `lower` and `upper` columns of `profiles` are `NaN`:

```@example inference
m_none = fit(LCAModel, d, 2; rng = StableRNG(1), se = :none)
```

```@repl inference
stderror(m_none)
```

The information matrix costs two E-step passes per free parameter, which is negligible
for a handful of items but not for a model with hundreds of parameters; `se = :none` is
also the right setting for fits whose only purpose is a log-likelihood, such as
exploratory sweeps over many class counts.

## Bootstrap standard errors

The standard errors above rest on the asymptotic normality of the maximum-likelihood
estimator. [`bootstrap`](@ref) offers the resampling alternative: it draws `n_boot` data
sets by resampling the respondents with replacement, refits the model to each
(warm-started from the fitted model), matches the class labels of every refit to the model,
and collects the refitted parameters. Fifty replicates keep this page quick; a few hundred
are the usual choice for standard errors and a thousand or more for 95% percentile
intervals.

```@example inference
b = bootstrap(m; n_boot = 50, rng = StableRNG(2))
```

The result has the same accessors as the model — [`stderror`](@ref stderror(::LCABootstrap)),
[`confint`](@ref confint(::LCABootstrap)), [`vcov`](@ref vcov(::LCABootstrap)),
[`coeftable`](@ref coeftable(::LCABootstrap)) and [`profiles`](@ref profiles(::LCABootstrap))
— now computed from the replicates. Side by side, the two kinds of standard error agree
to within the noise of 50 replicates for every interior parameter:

```@example inference
DataFrame(parameter = coefnames(m), hessian = round.(stderror(m); digits = 3),
          bootstrap = round.(stderror(b); digits = 3))
```

The boundary parameter is the exception: its logit is about ``-22`` (the log of the floor
at which the package holds probabilities) and stays there in every replicate, so a
bootstrap standard error for it would be a meaningless near-zero number, and in other data
sets a cell that leaves the boundary in some replicates would produce a huge one instead.
[`stderror`](@ref stderror(::LCABootstrap)), [`vcov`](@ref vcov(::LCABootstrap)),
[`confint`](@ref confint(::LCABootstrap)) and [`coeftable`](@ref coeftable(::LCABootstrap))
therefore report `NaN` for every parameter that is on the boundary in the reference model,
exactly as the observed-information version does. Look at the probability scale instead:
[`profiles`](@ref profiles(::LCABootstrap)) summarises the replicate probabilities directly
and is not masked.

[`confint`](@ref confint(::LCABootstrap)) returns *percentile* intervals by default — the
2.5th and 97.5th percentiles of the replicates — and Wald intervals with the bootstrap
standard error with `method = :normal`:

```@example inference
hcat(confint(b)[1:3, :], confint(b; method = :normal)[1:3, :])
```

On the probability scale, [`profiles`](@ref profiles(::LCABootstrap)) maps every replicate
back to probabilities before summarising, so its `se` is the standard deviation of the
replicate probabilities and `lower`/`upper` their percentiles. Here nothing is held fixed:
the boundary cell gets the spread of its replicates like every other cell (zero in this
case, because every replicate put it on the boundary).

```@example inference
pb = DataFrame(profiles(b; classes = true))
pb[pb.item .== :item6, :]
```

When should you prefer the bootstrap? The observed-information standard errors are
accurate and cheap for models with a few hundred respondents and well-separated classes,
as the comparison above shows. The bootstrap is the better choice

- in small samples, where the normal approximation on the logit scale is poor;
- for probabilities near 0 or 1, whose sampling distributions are skewed and whose
  percentile intervals do not rely on any approximation;
- for the class sizes of a model with covariates, which have no delta-method standard
  error;
- when `fit` warns that the observed information is not positive definite (a weakly
  identified model), in which case every asymptotic standard error is `NaN`.

Its cost is `n_boot` refits. `parametric = true` simulates the replicates from the fitted
model instead of resampling rows (and re-applies the pattern of missing responses), and
`n_starts > 1` adds random starts to every refit as a guard against replicates that
converge to a different local maximum. Keep in mind that all of these standard errors
describe uncertainty *given the number of classes*; choosing that number is the subject of
[Model selection](@ref guide-model-selection).

## The information matrix

[`informationmatrix`](@ref) returns the observed information itself — the negative Hessian
of the log-likelihood at the estimate, on the `coef` scale — recomputed from the model.
Its inverse on the free parameters is [`vcov`](@ref), which is a useful check:

```@example inference
V = vcov(m)
free = findall(i -> !isnan(V[i, i]), 1:dof(m))
maximum(abs, informationmatrix(m)[free, free] * V[free, free] - I)
```

Only the observed information is available; `expected = true` throws.
