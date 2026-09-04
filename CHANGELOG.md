# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Standard errors: the item-response parameters of an empty class (size at most `1e-6`)
  are held fixed and reported as `NaN`, like boundary parameters, and so are parameters
  with zero observed information; the remaining standard errors stay finite (conditional
  on the fixed parameters) instead of the whole covariance matrix becoming `NaN` because
  the information matrix is singular. The fit warning names both cases.
- The analytic score no longer computes the covariate Hessian it does not use, which
  makes the observed information matrix, and therefore `fit` with `se=:hessian`, faster
  for models with many covariates or classes.
- `CITATION.cff` now cites the package itself (`preferred-citation` removed, so GitHub's
  "Cite this repository" produces a software reference), lists the 2024 article under
  `references`, and carries the author's ORCID. The README and the documentation give a
  `@software` BibTeX entry for the package next to the article.
- Stricter input checks: `fit` rejects an `init` model that was fitted with different
  covariates, `prepare_data` rejects `levels` entries for names that are not items, and
  `LCAData` rejects `covariate_names` without `covariates`. `simulate` throws an
  `ArgumentError` (previously `DimensionMismatch`) for a design or missing mask of the
  wrong size.
- Items without a single observed response are named in the fit warning and get `NaN`
  standard errors instead of making the whole covariance matrix `NaN`.
- The Newton step of the covariate M-step works in preallocated workspace buffers, so
  EM with covariates no longer allocates per iteration.
- CI runs the tests with two Julia threads so that `multithreaded=true` is exercised.

### Fixed
- Starting values (`init`) whose item-response rows were not normalized were clamped to
  `[1e-10, 1]` before being normalized, so a row such as `[3, 1]` became `[0.5, 0.5]`;
  rows are now normalized first.
- `aicc` returns `NaN` instead of a meaningless value when `nobs ≤ dof + 1`.
- `show(::LCABootstrap)` reported "fewer than two usable replicates" whenever the
  reference model had a parameter on the boundary; it now summarizes the finite standard
  errors and counts the boundary ones.
- `show_profiles` columns no longer run into each other when a standard error is 100
  percentage points or more.
- The confidence-level label of `coeftable` (for example `"Lower 57%"`) no longer shows
  floating-point noise.
- `LCAData` no longer warns about 0/1 coding for a column that is entirely missing.

### Documentation
- The childlessness example describes the survey the data come from (the Childless Aging
  in Singapore Study, 2022) and how the bundled extract relates to the article's sample.

## [0.3.0] - 2026-09-04

Version 0.3.0 redesigns the package around a single entry point, `fit(LCAModel, data, k)`,
which returns the fitted model, and the StatsAPI verbs. Every 0.2 call has a replacement;
the "Migrating from 0.2 to 0.3" page of the documentation lists them side by side. Items
marked **Breaking** change the signature, the return value, or the results of a 0.2 call.

### Added
- `fit(LCAModel, data, k)`: maximum likelihood by EM with random restarts (`n_starts`
  short runs of `short_iters` iterations, the `n_final` best continued to convergence,
  the emEM scheme), a numerically stable log-sum-exp E-step, exact response-pattern
  aggregation (`aggregate`), an `rng` keyword for reproducible fits, `multithreaded=true`
  for parallel starts with results identical to the serial run, `init` for user-supplied
  starting values, and `verbose`. `fit(LCAModel, data, 1:4)` fits several class counts and
  `fit(LCAModel, table, items, k)` prepares a table first. `n_classes == 1` is supported
  (closed form).
- `LCAData`: the prepared-data container (integer codes, item names and level labels,
  optional covariate matrix), built by `prepare_data(table, items; covariates, levels,
  drop_unused_levels)` from any Tables.jl source — a `DataFrame`, a `NamedTuple` of
  vectors, an Arrow table — or directly from a code matrix with `LCAData(y; ...)`. Levels
  come from `DataAPI.levels` (the level order of a `CategoricalArray`, sorted values
  otherwise), `levels` fixes the level order per item, and unused levels are dropped unless
  `drop_unused_levels=false`. Accessors `nobs`, `size`, `hasmissing`, `nmissing`,
  `hascovariates`.
- Missing responses in the indicators: `missing` (code `0`) is skipped in the E-step under
  the missing-at-random assumption; the class sizes use every row and the response
  probabilities of an item the rows where it is observed; rows with all indicators missing
  are kept (with a warning) and receive the class sizes as their posterior.
- Covariates on class membership (latent class regression): `prepare_data(table, items;
  covariates=[:age, :female])` or `LCAData(y; covariates=X)` and `fit(LCAModel, d, k)` fit
  the multinomial-logit membership model `log(π_k(x)/π_1(x)) = x'β_k` (class 1, the largest
  class, as reference) by EM with one damped Newton step per M-step; `model.beta` holds the
  coefficients on the raw scale, `model.class_probs` the sample-averaged membership
  probabilities, `dof` counts `(k - 1)·P` membership parameters, `predict`, `classify` and
  `loglikelihood` on new data use its covariates, `coeftable(m; which=:class)` tabulates
  the coefficients, and `show` prints them. `covariates=false` fits the unconditional model
  on the same data for a nested comparison. A constant or collinear covariate is an error;
  quasi-complete separation raises the `coef_divergence` fit flag.
- Standard errors and confidence intervals: `fit` computes the covariance matrix of the
  free parameters from the observed information matrix (analytic score, central
  finite-difference Hessian) unless `se=:none`. `coef` and `coefnames` give the parameters
  on the logit scale (class-membership block first, then the item logits against each
  row's modal category); `vcov`, `stderror`, `confint(m; level)`, `coeftable(m; level,
  which)` and `informationmatrix` read them. `profiles(m; level, classes)` returns the
  class sizes and item-response probabilities with delta-method standard errors and
  logit-scale confidence intervals, and `show_profiles` prints every percentage with `±`
  its standard error. Parameters on the boundary (a probability within 1e-6 of 0 or 1) are
  held fixed and get `NaN` standard errors with a warning; the other cells of a row with a
  boundary cell keep standard errors conditional on it being fixed (the Mplus and Latent
  GOLD convention). The whole matrix is `NaN` when the observed information is not positive
  definite or the coefficients diverged.
- Simulation and bootstrap: `simulate(m, n; rng, X, missing_mask)` draws data sets from a
  fitted model; `bootstrap(m; n_boot, rng, parametric, n_starts, multithreaded)` refits the
  model to resampled rows or simulated data, aligns the class labels of every replicate to
  the model, and returns an `LCABootstrap` with `vcov`, `stderror`, `confint(; level,
  method)` (percentile or normal), `coeftable` and `profiles` (standard deviations and
  percentiles of the replicate probabilities, including the class sizes of covariate
  models).
- The bootstrap likelihood-ratio test for the number of classes: `bootstrap_lrt(null, alt;
  n_boot, rng, n_starts_boot, n_final_boot, multithreaded)` (McLachlan 1987; Nylund et al.
  2007) returns a `BootstrapLRT` with `pvalue`, and `bootstrap_lrt(d, k; ...)` fits both
  models first. Bootstrap replicates are seeded up front, so serial and `multithreaded=true`
  runs agree bitwise.
- StatsAPI verbs on the fitted model: `nobs`, `dof`, `loglikelihood(m)`,
  `loglikelihood(m, data)`, `aic`, `bic`, `aicc`, `isfitted`, plus `sbic`, `entropy(m;
  relative)` and `diagnostics(m)`/`diagnostics(models)`; a `Vector{ModelDiagnostics}` is a
  Tables.jl row table, so `DataFrame(diagnostics(models))` is a model-selection table.
- `classify(m[, data])` for modal class assignments; `predict(m[, data])` accepts the
  model alone (its training data), an `LCAData`, or a table coded with the training levels.
- `LCAOptions` (the estimation settings, stored in `model.options`) and `FitFlags`
  (`model.flags`): non-convergence, boundary probabilities, empty classes, a best
  log-likelihood reached by only one of the continued starts, and diverging covariate
  coefficients are collected into one warning per fit and printed by `show`;
  `model.start_loglik` records the log-likelihood of every start.
- The identifiability check uses the necessary condition
  `(K - 1) + K·Σ(C_j - 1) ≤ ∏C_j - 1` and warns only when it fails.
- Documentation: four guide pages (model selection, missing data, covariates, standard
  errors and the bootstrap), methodology sections on covariates, standard errors and the
  bootstrap likelihood-ratio test, a migration page, and a covariate model and bootstrap
  likelihood-ratio test on the childlessness example. Tests cover every feature; the slow
  bootstrap tests are skipped with `LCA_SLOW_TESTS=false`.

### Changed
- **Breaking:** `LCAModel` is immutable and returned by `fit`; the 0.2 workflow
  `LCAModel(k, n_items, n_categories)` + `fit!` no longer exists (see Removed). Classes are
  ordered by decreasing size, so class 1 is always the largest.
- **Breaking:** `prepare_data(table, items)` takes a vector of column names and returns an
  `LCAData` instead of a `(codes, n_categories)` tuple; the varargs form is deprecated.
- **Breaking:** `missing` in an indicator is no longer coded as an additional category but
  as a missing response (code `0`), so a column with `missing` values has one category
  fewer than in 0.2 and the fit statistics are not comparable with 0.2 values. Replace
  `missing` by an explicit label before `prepare_data` to keep the old behaviour.
- **Breaking:** `predict` returns only the posterior probability matrix; the 0.2 tuple
  `(assignments, probabilities)` is split into `classify` and `predict`.
- **Breaking:** fits are reproduced by the `rng` keyword of `fit`, not by `Random.seed!`,
  and a given seed no longer yields the 0.2 single-start solution: `fit` searches 20
  starting values and typically reaches a higher log-likelihood.
- **Breaking:** `ModelDiagnostics` gained the fields `n_classes`, `nobs`, `dof` and
  `converged` (the 0.2 fields `ll`, `aic`, `bic`, `sbic`, `entropy` are unchanged).
- Convergence uses the relative tolerance `|ll - ll_old| ≤ tol·(1 + |ll|)` with
  `tol=1e-10` instead of an absolute `1e-6`, and is checked before the M-step, so the
  reported log-likelihood, posterior and parameters belong to the same iteration. Fits
  take more iterations and agree with 0.2 estimates only to roughly the old tolerance.
- `show_profiles(model; var_names, var_labels, digits, io)` reads the item names and level
  labels from the model and prints standard errors; `show(model)` prints the fit
  statistics, class sizes, covariate coefficients and fit flags.
- Unused levels of a `CategoricalArray` are dropped by default (`drop_unused_levels=false`
  keeps them).
- Dependencies: DataFrames and CategoricalArrays are no longer dependencies (Tables.jl and
  DataAPI.jl are used instead); StatsAPI, StatsBase, Tables, DataAPI and the Logging
  standard library were added.

### Deprecated
- `prepare_data(df, cols::Symbol...; zero_based)` returns the 0.2 tuple with a deprecation
  warning (`zero_based` is accepted and ignored); `diagnostics!(m, data, ll)` forwards to
  `diagnostics(m)`; `show_profiles(m, df, cols; kwargs...)` forwards to
  `show_profiles(m; kwargs...)`. All three will be removed in 0.4.0.

### Removed
- **Breaking:** `LCAModel(n_classes, n_items, n_categories)` and `fit!(model, data)` throw
  an `ArgumentError` pointing at `fit(LCAModel, data, k)`.
- **Breaking:** `predict(model, ::Matrix)` and `classify(model, ::Matrix)` throw an
  `ArgumentError`; wrap codes in `LCAData(y; n_categories=model.n_categories)`.
- The `n_obs < 300` warning and the item-count identifiability heuristic, replaced by the
  necessary condition above and by the fit flags.

### Fixed
- The E-step no longer underflows or takes `log(0)`: posteriors are computed on the log
  scale with log-sum-exp and response probabilities are floored at 1e-10, so models with
  hundreds of items or with empty cells return finite log-likelihoods.
- The log-likelihood reported by a fit belongs to the returned parameters; 0.2 reported
  the value of the previous iteration when `max_iter` was reached.

## [0.2.2] - 2026-09-04

### Added
- Documentation site built with Documenter.jl: getting-started tutorial, methodology page,
  childlessness replication example, and API reference.
- `CHANGELOG.md`, `CITATION.cff`, and `CLAUDE.md`.
- `examples/Project.toml` so that the example scripts run in their own environment; CI now
  runs both scripts.
- Test suite split by concern, Aqua.jl quality checks, StableRNGs for reproducible test
  data, a parameter-recovery test, and hand-computed checks of the fit indices.
- CompatHelper workflow.

### Changed
- Documenter is no longer a package dependency (it lives in `docs/`).
- Compat: `CategoricalArrays = "0.10, 1"` (CategoricalArrays 1.x can now be installed
  alongside the package); `julia = "1.10"` (Julia 1.10 through any later 1.x, including
  1.12).
- CI: Julia version aliases (`min`, `lts`, `1`, `pre`), updated actions, a documentation
  job, and no Rosetta builds on macOS.
- `prepare_data` recodes every column to consecutive codes `1..K` by the rank of its
  sorted distinct values. Inputs that already worked produce identical codes; codes with
  gaps or offsets (for example values `1` and `3`, or `2` and `3`), which previously made
  `fit!` throw, now work. The `zero_based` keyword is ignored and deprecated.
- `show_profiles` honors its `digits` keyword.

### Fixed
- The docstrings of `LCAModel`, `ModelDiagnostics`, `fit!`, `predict`, `prepare_data`, and
  `diagnostics!` were separated from their definitions by a blank line, so `?fit!` showed
  no documentation.
- `check_identifiability` and `diagnostics!` rejected the wider integer types that the
  `LCAModel` constructor and `fit!` accept since 0.2.1 (`Int32`, `Int8` vectors, ranges,
  matrix views).
- `show_profiles` threw a `BoundsError` for a categorical column with an unused level.
- The README quick example and `examples/example.jl` failed with `UndefVarError:
  categorical`.
- `examples/example_childless.jl`: the dataset path is now relative to the script, a stray
  triple-backtick block (parsed as a command literal) is now a comment, and an unused
  `DataFramesMeta` import is gone.
- Unreachable code after `return` in `fit!`; the `fit!` docstring stated the wrong
  default for `max_iter`.
- Removed the stray `LatentClassAnalysis/Project.toml` directory.

## [0.2.1] - 2025-01-09

### Changed
- `fit!` and `predict` accept `AbstractMatrix{<:Integer}`; `LCAModel` accepts any `Integer`
  sizes (#4).
- The identifiability and sample-size checks warn instead of erroring.

### Added
- Identifiability check in the `LCAModel` constructor; warning for fewer than 300
  observations in `fit!`.

## [0.2.0] - 2024-11-04

### Added
- `show_profiles`; the childlessness example with real data; support for 0/1-coded binary
  variables.

### Removed
- Julia 1.6 support (the minimum is 1.10).

## [0.1.0] - 2024-10-27

### Added
- Initial release: `prepare_data`, `LCAModel`, `fit!`, `diagnostics!`, `predict`.

[Unreleased]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/releases/tag/v0.1.0
