# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The next breaking release, 0.3.0, is planned to add: a `fit(LCAModel, data, k)` entry point that accepts any
Tables.jl source and returns a fitted model object; StatsAPI verbs (`loglikelihood`,
`nobs`, `dof`, `aic`, `bic`, `coef`, `stderror`, ...); random restarts; missing values in
indicators; covariates for class membership; standard errors and confidence intervals; a
bootstrap likelihood-ratio test. The 0.2 functions `prepare_data`, `diagnostics!`, and
`show_profiles` will keep working with deprecation warnings; `LCAModel(k, n_items,
n_categories)` and `fit!` will be replaced.

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
