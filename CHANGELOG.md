# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 0.3.0 (breaking; in development on branch `v0.3.0`)

Still to come in 0.3.0: covariates on class membership (latent class regression), standard
errors and confidence intervals (`coef`, `vcov`, `stderror`, `confint`, `coeftable`, the
`se`/`lower`/`upper` columns of `profiles`), and `simulate`, `bootstrap`,
`bootstrap_lrt`.

#### Added
- `fit(LCAModel, data, k)` is the single entry point: EM with random restarts (`n_starts`
  short runs, the `n_final` best continued to convergence), a numerically stable
  log-sum-exp E-step, exact response-pattern aggregation, an `rng` keyword for
  reproducible fits, `multithreaded=true` for parallel starts (results identical to the
  serial run), and `init` for user-supplied starting values. `fit(LCAModel, data, 1:4)`
  fits several class counts; `fit(LCAModel, table, items, k)` prepares a table first.
- `LCAData`: the prepared-data container (codes, item names and level labels, optional
  covariate matrix), built by `prepare_data` from any Tables.jl source or directly from a
  code matrix. Accessors `nobs`, `size`, `hasmissing`, `nmissing`, `hascovariates`.
- Missing responses in indicators: `missing` (code `0`) is skipped in the E-step under the
  missing-at-random assumption; rows with all indicators missing get the class sizes as
  posterior.
- StatsAPI verbs on the fitted model: `nobs`, `dof`, `loglikelihood(m)`,
  `loglikelihood(m, data)`, `aic`, `bic`, `aicc`, `isfitted`, plus `sbic`, `entropy(m;
  relative)`, `diagnostics(m)`/`diagnostics(models)` (a `Vector{ModelDiagnostics}` is a
  Tables.jl row table: `DataFrame(diagnostics(models))`).
- `classify(m[, data])` for modal class assignments; `predict(m[, data])` for posterior
  probabilities on the training data, an `LCAData`, or a table coded with the training
  levels.
- `profiles(m)`: item-response profiles as a row table; `show_profiles(m; io)`.
- `LCAOptions` (estimation settings, stored in `model.options`) and `FitFlags`
  (`model.flags`): non-convergence, boundary probabilities, empty classes and a
  non-replicated best log-likelihood are collected into one warning and printed by `show`.
- `n_classes == 1` is supported (closed form).
- The identifiability check now uses the necessary condition
  `(K - 1) + K·Σ(C_j - 1) ≤ ∏C_j - 1`.

#### Changed
- `LCAModel` is immutable and is returned by `fit`; its classes are ordered by decreasing
  size. `beta` holds the multinomial-logit intercepts of the class sizes.
- `prepare_data(table, items)` returns an `LCAData`; levels come from `DataAPI.levels`
  (level order of a `CategoricalArray`, sorted values otherwise); unused levels are dropped
  unless `drop_unused_levels=false`; `levels` fixes the level order per item.
- `ModelDiagnostics` gained `n_classes`, `nobs`, `dof` and `converged` fields.
- Convergence uses a relative tolerance `|ll - ll_old| ≤ tol·(1 + |ll|)` with `tol=1e-10`
  and is checked before the M-step, so the reported log-likelihood, posterior and
  parameters are consistent.
- Dependencies: DataFrames and CategoricalArrays are no longer dependencies (Tables.jl and
  DataAPI.jl are used instead); StatsAPI, StatsBase, Tables and DataAPI were added.

#### Deprecated
- `prepare_data(df, cols::Symbol...)` (returns the 0.2 tuple), `diagnostics!(m, data, ll)`
  and `show_profiles(m, df, cols)` keep working with a deprecation warning.

#### Removed
- `LCAModel(n_classes, n_items, n_categories)` and `fit!(model, data)` throw an
  `ArgumentError` pointing at `fit(LCAModel, data, k)`.
- `predict(model, ::Matrix)` (which returned a tuple) throws an `ArgumentError`; wrap codes
  in `LCAData(y)`.
- The `n_obs < 300` warning and the item-count identifiability heuristic.

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

- Initial release: `prepare_data`, `LCAModel`, `fit!`, `diagnostics!`, `predict`.

[Unreleased]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yanwenwang24/LatentClassAnalysis.jl/releases/tag/v0.1.0
