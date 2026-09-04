# Migrating from 0.2 to 0.3

Version 0.3.0 redesigned the package around a single entry point, `fit(LCAModel,
data, k)`, which returns a fitted model, and around the StatsAPI verbs (`nobs`,
`dof`, `loglikelihood`, `aic`, `bic`, `predict`, ...). Data can come from any
Tables.jl source, fits use random restarts, and missing responses are allowed. This
page lists every 0.2 call next to its replacement, states the deprecation policy, and
explains the changes in behaviour that affect results. The features that have no 0.2
counterpart — covariates on class membership, standard errors and confidence intervals,
bootstrap standard errors, and the bootstrap likelihood-ratio test — are introduced in
the guide pages: [Model selection](@ref guide-model-selection),
[Missing data](@ref guide-missing-data), [Covariates](@ref guide-covariates) and
[Standard errors and the bootstrap](@ref guide-inference).

## Before and after

| 0.2 | 0.3 |
|:----|:----|
| `using LatentClassAnalysis, DataFrames` | `using LatentClassAnalysis` — DataFrames is no longer loaded by the package; add `using DataFrames` yourself if you use it |
| `data, n_categories = prepare_data(df, :a, :b, :c)` | `d = prepare_data(df, [:a, :b, :c])` returns an [`LCAData`](@ref); the codes are `d.y`, the category counts `d.n_categories` |
| `prepare_data(df, cols...; zero_based = [...])` | keyword removed; codes are always `1, 2, …, C_j` |
| a `DataFrame` argument | any Tables.jl table (`DataFrame`, `NamedTuple` of vectors, Arrow table, ...) |
| a `Matrix{Int}` of codes | `LCAData(y)` or `LCAData(y; n_categories = [...])` |
| `Random.seed!(1)` before constructing the model | `rng = StableRNG(1)` (or `Random.Xoshiro(1)`) passed to `fit` |
| `model = LCAModel(k, size(data, 2), n_categories)` | not needed; the constructor now throws |
| `ll = fit!(model, data; max_iter, tol, verbose)` | `model = fit(LCAModel, d, k; max_iter, tol, verbose, rng)`; `loglikelihood(model)` or `model.loglik` is `ll` |
| a loop over `k` | `models = fit(LCAModel, d, 2:5; rng)` |
| `diag = diagnostics!(model, data, ll)` | `diag = diagnostics(model)` |
| `diag.ll`, `diag.aic`, `diag.bic`, `diag.sbic`, `diag.entropy` | unchanged, plus `diag.n_classes`, `diag.nobs`, `diag.dof`, `diag.converged`; or call `loglikelihood(model)`, `aic(model)`, `bic(model)`, `sbic(model)`, `entropy(model)` directly |
| a `DataFrame` of statistics built by hand | `DataFrame(diagnostics(models))` |
| `(k - 1) + k * sum(n_categories .- 1)` | `dof(model)` |
| `assignments, probs = predict(model, data)` | `probs = predict(model)` and `assignments = classify(model)` |
| `predict(model, new_matrix)` | `predict(model, new_table)` or `predict(model, LCAData(new_matrix; n_categories = model.n_categories))`; a bare matrix throws |
| `show_profiles(model, df, cols; var_names, var_labels, digits)` | `show_profiles(model; var_names, var_labels, digits)` — the item names and level labels are stored in the model |
| `model.class_probs`, `model.item_probs[j][k, c]` | unchanged (classes are now ordered by decreasing size) |
| `model.n_classes`, `model.n_items`, `model.n_categories` | unchanged |
| rows with `missing` had to be dropped | keep them; `missing` is handled in the E-step |

A 0.2 script in the new API:

```@example migration
using LatentClassAnalysis, DataFrames, StableRNGs

rng = StableRNG(1)
cls = rand(rng, 1:2, 400)
item(p1, p2) = [rand(rng) < (c == 1 ? p1 : p2) ? 1 : 0 for c in cls]
df = DataFrame(a = item(0.9, 0.2), b = item(0.8, 0.3), c = item(0.85, 0.25), e = item(0.7, 0.2))

d = prepare_data(df, [:a, :b, :c, :e])          # was: data, n_categories = prepare_data(df, :a, :b, :c, :e)
model = fit(LCAModel, d, 2; rng = StableRNG(1))  # was: model = LCAModel(2, 4, n_categories); ll = fit!(model, data)
diag = diagnostics(model)                        # was: diagnostics!(model, data, ll)
df.class = classify(model)                       # was: assignments, probs = predict(model, data)
probs = predict(model)
diag.bic, dof(model), size(probs)
```

The removed calls fail loudly, with a message that names the replacement:

```@repl migration
LCAModel(2, 4, [2, 2, 2, 2])
fit!(model, d.y)
predict(model, d.y)
```

## Deprecation policy

Three 0.2 calls keep working in 0.3 with a deprecation warning and will be removed
in 0.4.0:

- `prepare_data(df, cols::Symbol...)` returns the 0.2 tuple `(codes, n_categories)`;
  the `zero_based` keyword is accepted and ignored.
- `diagnostics!(model, data, ll)` ignores `data` and `ll` and returns
  `diagnostics(model)`.
- `show_profiles(model, df, cols; kwargs...)` ignores `df` and `cols` and calls
  `show_profiles(model; kwargs...)`.

Julia prints deprecation warnings only when started with `--depwarn=yes` (the
default is `no`), so a script that uses only these three calls may run silently on
0.3; start Julia with `--depwarn=yes` to locate them. The remaining 0.2 calls —
`LCAModel(k, n_items, n_categories)`, `fit!(model, data)`, and `predict`/`classify`
on a matrix — cannot be shimmed faithfully and throw an `ArgumentError` that points
to the replacement, so an old script fails on its first model rather than
returning something subtly different.

## What changed in the results, and why

**Fits are no longer reproduced by `Random.seed!`, and a seed gives a different
solution than in 0.2.** In 0.2 the constructor drew one set of starting values
from the global random number generator and `fit!` ran EM once from there. In 0.3
`fit` draws 20 starting values from the `rng` keyword, runs each briefly, continues
the four best to convergence, and keeps the best (see [Random restarts and local
maxima](@ref restarts)). The result is reproducible for a given `rng` object, not
for a global seed, and it is typically a higher log-likelihood than the 0.2
single-start fit reached from the same seed. If your 0.2 solution differs from the
0.3 one, the 0.2 run had most likely stopped at a local maximum.

**Missing values no longer become a category.** `prepare_data` in 0.2 treated
`missing` as one more level of the item. In 0.3 `missing` is coded `0`, skipped in
the E-step, and excluded from `n_categories`, so a column with `missing` values has
one category fewer than before, the model has fewer parameters, and the fit
statistics are not comparable with 0.2 values. To keep the old behaviour, replace
`missing` with an explicit label (`coalesce.(col, "not answered")`) before calling
`prepare_data`.

**`predict` returns only the posterior.** The 0.2 tuple `(assignments,
probabilities)` made the return type of `predict` unlike that of every other
StatsAPI model, and a bare matrix argument would silently have been destructured
into two variables. `predict` now returns the ``n \times K`` posterior matrix, and
[`classify`](@ref) returns the modal assignments. Both accept the model alone (its
training data), an `LCAData`, or a table, which is coded with the training levels.

**Classes are ordered by size.** After fitting, classes are sorted by decreasing
size, so class 1 is the largest. In 0.2 the order was whatever the starting values
produced. Class numbers still carry no substantive meaning, and two classes of
nearly equal size can swap numbers between data sets.

**The `n_obs < 300` warning is gone.** Sample size alone does not tell whether a
particular model is well determined. It is replaced by fit flags that report actual
symptoms: EM not converging, response probabilities on the boundary, an empty
class, and a best log-likelihood found by only one start. They are aggregated into
one warning per fit and stored in `model.flags`.

**The identifiability check changed.** The 0.2 rule of thumb on the number of items
(``2\lceil \log_C K \rceil + 1``) is replaced by the necessary condition that the
number of free parameters does not exceed the number of independent response
patterns, ``p \le \prod_j C_j - 1`` (see [Identifiability](@ref identifiability)). The
warning is issued only when the condition fails, so some models that warned in 0.2
no longer do, and a few that did not now do.

**Convergence is relative and tighter.** EM stops when ``|\ell_t - \ell_{t-1}| \le
10^{-10}(1 + |\ell_t|)`` rather than when the absolute change drops below
``10^{-6}``, and the check happens before the M-step so that the reported
log-likelihood, posterior and parameters belong to the same iteration. Fits take
more iterations and agree with 0.2 estimates only to roughly the old tolerance.

**`ModelDiagnostics` gained fields.** `n_classes`, `nobs`, `dof` and `converged` were
added; the 0.2 fields `ll`, `aic`, `bic`, `sbic`, `entropy` are unchanged. A vector of
diagnostics is a Tables.jl table, which is what makes `DataFrame(diagnostics(models))`
work.

**Levels of categorical columns.** Unused levels of a `CategoricalArray` are now
dropped (`drop_unused_levels = false` keeps them), and the `levels` keyword fixes the
order of any column's levels without converting it.

**`show_profiles` prints standard errors.** Every percentage is followed by `±` its
standard error, because `fit` computes the observed information matrix by default. Pass
`se = :none` to `fit` to skip the computation and get the 0.2-style output; see
[Standard errors and the bootstrap](@ref guide-inference).
