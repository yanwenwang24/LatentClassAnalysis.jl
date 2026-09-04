# LatentClassAnalysis.jl

Julia package for latent class analysis (LCA): finite mixtures of independent multinomials
fitted by EM to categorical indicators. The audience is applied social scientists. The
bundled example replicates Wang, Teerawichitchainan & Ho (2024, Advances in Life Course
Research, doi 10.1016/j.alcr.2024.100628).

## Layout

- `src/LatentClassAnalysis.jl` module root: imports (StatsAPI/StatsBase verbs are
  imported and re-exported), exports, includes in dependency order.
- `src/types.jl` `LCAData`, `LCAOptions`, `LCAModel` (immutable, returned by `fit`),
  `FitFlags`, `ModelDiagnostics`, `LCABootstrap`, `BootstrapLRT`.
- `src/data.jl` `prepare_data` (Tables.jl + DataAPI, no DataFrames dependency).
- `src/em.jl` `LCAParams` (EM state), `LCAWorkspace` (transposed data, pattern
  aggregation), `estep!`, `_accumulate!`, `_update!`, `_em!`.
- `src/covariates.jl` latent class regression: `_standardize` (covariates standardized
  internally, `A` maps coefficients back to the raw scale), the E-step log-prior hook, the
  damped Newton M-step `_update_coefs!` on `Q(β)`, and `_class_prior` (raw-scale
  membership probabilities used by `fit`, `predict` and `simulate`).
- `src/restarts.jl` random starts, two-stage multi-start driver, `_sort_by_size!`,
  `_init_split`.
- `src/fit.jl` `StatsAPI.fit` methods, `check_identifiability`, fit flags, erroring `fit!`.
- `src/predict.jl` `predict`, `classify`. `src/inference.jl` `ParamLayout` (the
  free-parameter vector: class block, then item logits against each row's modal
  category), `_pack`/`_unpack!`, the analytic `_score!`, the finite-difference
  `_observed_information`, `coef`/`coefnames`/`vcov`/`stderror`/`confint`/`coeftable`/
  `informationmatrix`, and the delta-method `profiles`. `src/bootstrap.jl` `simulate`
  (`_simulate` also returns the classes), label alignment of replicate fits
  (`_align_labels`, `_align!`), `bootstrap` with the `LCABootstrap` methods of
  `vcov`/`stderror`/`confint`/`coeftable`/`profiles`, `bootstrap_lrt`, `pvalue`.
  `src/diagnostics.jl` StatsAPI accessors, criteria, `entropy`, `diagnostics`, Tables
  interface. `src/show.jl` `show` (all types), `show_profiles`. `src/deprecated.jl` 0.2
  shims, included last.
- `test/` one file per concern, included from `runtests.jl`; `testutils.jl` has the
  simulation, class-alignment and `same_fit` helpers and the shared two-/three-class designs.
- `docs/` Documenter site (`make.jl`, `src/*.md`, `src/refs.bib`). `docs/src/changelog.md`
  is generated from `CHANGELOG.md` at build time and is gitignored.
- `examples/` runnable scripts with their own `Project.toml`; `childless_df.arrow` is the
  real dataset used by the docs and the childlessness example.

## Commands

```
julia --project=. -e 'using Pkg; Pkg.test()'                                   # tests (incl. Aqua)
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'  # once
julia --project=docs docs/make.jl                                              # build docs to docs/build
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'  # once
julia --project=examples examples/example_childless.jl
```

## Conventions

- A docstring must be immediately followed by its definition. A blank line in between
  silently detaches it (this happened to six functions before 0.2.2).
- Every exported symbol has a docstring and an `@docs` entry in `docs/src/api/` (core.md or inference.md); the docs
  build runs with `warnonly=false` and `checkdocs=:exports`, so a missing entry fails CI.
- Tests generate data with `StableRNG` seeds and pass `rng=StableRNG(seed)` to every
  `fit`; a fit is bitwise reproducible for a given `rng`, so tests may compare fits with
  `same_fit`. Never use the global RNG in tests.
- `prepare_data` always yields dense codes `1..C_j` per item (`0` = missing), ordered by
  `DataAPI.levels` (level order for categorical columns, sorted values otherwise); the
  labels are stored in `LCAData.item_levels` and `show_profiles`/`profiles` read them.
- Scales: `LCAParams.coefs` lives on the standardized covariate scale (column 1 zero),
  `LCAModel.beta` on the raw scale (`P × (K - 1)`); `LCAModel.vcov` and
  `LCABootstrap.coefs` are on the public `coef` scale (raw covariates), with `NaN`
  rows/columns for boundary parameters in `vcov`.
- Bootstrap conventions: seeds are drawn from `rng` up front and every replicate runs on
  its own `Xoshiro(seed)` (serial and threaded runs agree bitwise); replicate fits run
  under a `NullLogger` with one aggregated warning afterwards; a replicate is aligned to
  the reference model with `_align_labels` (`_permute_classes!` re-bases β) and packed
  with the reference model's `ParamLayout`, never its own argmax; bootstrap/BLRT
  simulate complete data and re-apply the observed missingness mask.
- Every user-visible change gets a line under `[Unreleased]` in `CHANGELOG.md`.
- Compat entries use caret ranges (`"1.10"` means `>= 1.10, < 2`). Do not add a compat
  entry for `LatentClassAnalysis` in `docs/Project.toml` or `examples/Project.toml`.
- 0.2.x is the non-breaking line. API changes (result object, StatsAPI verbs, Tables.jl
  input, restarts, missing data, covariates, inference) belong to 0.3.0.

## Release checklist

1. Land all changes, including workflow edits, in PRs before the version bump.
2. Bump `version` in `Project.toml`, `version`/`date-released` in `CITATION.cff`, and turn
   `[Unreleased]` into a dated section in `CHANGELOG.md`, in one commit that touches no
   file under `.github/workflows/` (otherwise TagBot cannot create the release).
3. Comment `@JuliaRegistrator register` on that commit, with the changelog section pasted
   below `Release notes:` (TagBot does not read `CHANGELOG.md`).
4. After AutoMerge, TagBot tags the release over the `DOCUMENTER_KEY` deploy key, the tag
   push runs CI, and the docs job publishes the versioned docs and moves `stable`.
