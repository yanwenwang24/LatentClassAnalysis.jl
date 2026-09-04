# LatentClassAnalysis.jl

Julia package for latent class analysis (LCA): finite mixtures of independent multinomials
fitted by EM to categorical indicators, with random restarts, missing data, covariates on
class membership, standard errors, the bootstrap and the bootstrap likelihood-ratio test.
The audience is applied social scientists. The bundled example replicates Wang,
Teerawichitchainan & Ho (2024, Advances in Life Course Research, doi
10.1016/j.alcr.2024.100628).

## Layout

- `src/LatentClassAnalysis.jl` module root: imports (StatsAPI/StatsBase verbs are
  imported and re-exported), exports, includes in dependency order.
- `src/types.jl` `LCAData`, `LCAOptions`, `FitFlags`, `LCAModel` (immutable, returned by
  `fit`), `ModelDiagnostics`, `LCABootstrap`, `BootstrapLRT`, and the accessors
  `hasmissing`, `nmissing`, `hascovariates`.
- `src/data.jl` `prepare_data` (Tables.jl + DataAPI, no DataFrames dependency), level
  coding, the covariate matrix builder.
- `src/em.jl` `LCAParams` (EM state), `LCAWorkspace` (transposed data, pattern
  aggregation), `estep!`, `_accumulate!`, `_update!`, `_em!`, the closed-form one-class
  fit, `_expand_posterior`.
- `src/covariates.jl` latent class regression: `_standardize` (covariates standardized
  internally, `A` maps coefficients back to the raw scale), the E-step log-prior hook, the
  damped Newton M-step `_update_coefs!` on `Q(β)` with `_coef_objective` and
  `_coef_derivatives!`, and `_class_prior` (raw-scale membership probabilities used by
  `fit`, `predict` and `simulate`).
- `src/restarts.jl` `_init_random`, user-supplied `init` handling, the two-stage
  multi-start driver `_multistart` (`StartRecord` per start), `_sort_by_size!`,
  `_permute_classes!`, `_init_split`.
- `src/fit.jl` `StatsAPI.fit` methods, keyword → `LCAOptions`, `check_identifiability`,
  fit flags and the aggregated warning, the erroring `fit!`.
- `src/predict.jl` `predict`, `classify`, `_prepare_like` (tables coded with the training
  levels and covariates), `_posterior_and_ll`.
- `src/inference.jl` `ParamLayout` (the free-parameter vector: class block, then item
  logits against each row's modal category), `_pack`/`_unpack!`, the analytic `_score!`,
  the finite-difference `_observed_information`, `_fit_vcov`, `coef`/`coefnames`/`vcov`/
  `stderror`/`confint`/`coeftable`/`informationmatrix`, and the delta-method `profiles`
  (`_softmax_covariance` implements the conditional standard errors at the boundary).
- `src/bootstrap.jl` `simulate` (`_simulate` also returns the classes), label alignment of
  replicate fits (`_align_labels`, `_align!`), `bootstrap` with the `LCABootstrap` methods
  of `vcov`/`stderror`/`confint`/`coeftable`/`profiles`, `bootstrap_lrt`, `pvalue`.
- `src/diagnostics.jl` StatsAPI accessors (`nobs`, `dof`, `loglikelihood`, `aic`, `bic`,
  `aicc`, `isfitted`), `sbic`, `entropy`, `diagnostics`, the Tables.jl interface of
  `Vector{ModelDiagnostics}`.
- `src/show.jl` `show` for every type, `show_profiles`.
- `src/deprecated.jl` 0.2 shims (`prepare_data` varargs, `diagnostics!`, the
  `show_profiles(m, df, cols)` form, the throwing `LCAModel` constructor), included last;
  scheduled for removal in 0.4.0.
- `test/` one file per concern, included from `runtests.jl`; `testutils.jl` has the
  simulation, class-alignment and `same_fit` helpers and the shared two-/three-class
  designs; `test_docs.jl` checks that every export has a docstring; the slow bootstrap
  tests in `test_bootstrap.jl` are gated by `LCA_SLOW_TESTS`.
- `docs/` Documenter site (`make.jl`, `src/*.md`, `src/guide/*.md`, `src/api/*.md`,
  `src/refs.bib`). `docs/src/changelog.md` is generated from `CHANGELOG.md` at build time
  and is gitignored.
- `examples/` runnable scripts with their own `Project.toml`; `childless_df.arrow` is the
  real dataset used by the docs and the childlessness example.

## Commands

```
julia --project=. -e 'using Pkg; Pkg.test()'                                   # tests (incl. Aqua, ~75 s)
LCA_SLOW_TESTS=false julia --project=. -e 'using Pkg; Pkg.test()'              # skip the slow bootstrap tests
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'  # once
julia --project=docs docs/make.jl                                              # build docs to docs/build (strict, ~1 min)
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'  # once
julia --project=examples examples/example.jl
julia --project=examples examples/example_childless.jl
```

The README quick example must run as pasted; check it with the docs environment
(`julia --project=docs`), which has DataFrames and StableRNGs.

## Conventions

- A docstring must be immediately followed by its definition. A blank line in between
  silently detaches it (this happened to six functions before 0.2.2).
- Every exported symbol has a docstring and an `@docs` entry in `docs/src/api/core.md` or
  `docs/src/api/inference.md`; the docs build runs with `warnonly=false`,
  `checkdocs=:exports` and doctests on, so a missing entry or a stale doctest fails CI.
- Randomness always flows through an explicit `rng` keyword (`fit`, `simulate`,
  `bootstrap`, `bootstrap_lrt`); nothing in `src/` touches the global RNG. Seeds are drawn
  from `rng` up front and every start or replicate runs on its own `Xoshiro(seed)`, so
  serial and multithreaded runs agree bitwise and a fit is reproducible for a given `rng`.
- Tests and docs generate data with `StableRNG` seeds and pass `rng=StableRNG(seed)` to
  every `fit`; tests may compare fits with `same_fit`. Never use the global RNG in tests
  or docs. Docs `@example` blocks use small `n` so the whole build stays under about
  three minutes; errors that should be shown go in `@repl` blocks.
- `prepare_data` always yields dense codes `1..C_j` per item (`0` = missing), ordered by
  `DataAPI.levels` (level order for categorical columns, sorted values otherwise); the
  labels are stored in `LCAData.item_levels` and `show_profiles`/`profiles` read them.
  Covariates must be numeric and complete; categorical covariates are dummy-coded by the
  user in the table.
- Parameter layout: `ParamLayout` in `src/inference.jl` is the single contract for the
  free-parameter vector used by `coef`, `vcov`, the score, the information matrix and the
  bootstrap: class block first (`vec(beta)`, classes 2..K against class 1, column-major),
  then for every item and class the logits of the non-modal categories against the modal
  category. `length(coef(m)) == dof(m)`.
- Scales: `LCAParams.coefs` lives on the standardized covariate scale (column 1 zero),
  `LCAModel.beta` on the raw scale (`P × (K - 1)`); `LCAModel.vcov` and
  `LCABootstrap.coefs` are on the public `coef` scale (raw covariates), with `NaN`
  rows/columns for boundary parameters in `vcov`.
- Boundary convention: a probability within `1e-6` of 0 or 1 is held fixed in the
  information matrix and gets a `NaN` standard error; the other cells of its row (or the
  other class sizes) get standard errors conditional on the fixed cell, computed by
  `_softmax_covariance` over the free logits only (the Mplus / Latent GOLD convention).
  The item-response logits of an empty class (size `≤ 1e-6`) are held fixed too, and
  `_covariance` additionally masks any free parameter whose observed information is
  numerically zero (`_informative`, an item never observed in a class), so a singular
  block never turns the whole matrix `NaN`. `profiles` gives a fixed cell the degenerate
  interval `(p, p)`. The bootstrap holds nothing fixed.
- Bootstrap conventions: replicate fits run under a `NullLogger` with one aggregated
  warning afterwards; a replicate is aligned to the reference model with `_align_labels`
  (`_permute_classes!` re-bases β) and packed with the reference model's `ParamLayout`,
  never its own argmax; bootstrap/BLRT simulate complete data and re-apply the observed
  missingness mask; the BLRT p-value is `(1 + #{T_b ≥ T_obs}) / (n_boot + 1)`.
- Classes are sorted by decreasing size at the end of `fit`; class 1 is the reference of
  `beta`. Class numbers carry no substantive meaning and the docs say so.
- Every user-visible change gets a line under `[Unreleased]` in `CHANGELOG.md`.
- Compat entries use caret ranges (`"1.10"` means `>= 1.10, < 2`). Do not add a compat
  entry for `LatentClassAnalysis` in `docs/Project.toml` or `examples/Project.toml`.
- 0.3.x is the current line. The 0.2 shims in `src/deprecated.jl` are removed in 0.4.0.

## Release checklist

1. Land all changes, including workflow edits, in PRs before the version bump.
2. Bump `version` in `Project.toml`, `version`/`date-released` in `CITATION.cff`, and turn
   `[Unreleased]` into a dated section in `CHANGELOG.md`, in one commit that touches no
   file under `.github/workflows/` (otherwise TagBot cannot create the release).
3. Comment `@JuliaRegistrator register` on that commit, with the changelog section pasted
   below `Release notes:` (TagBot does not read `CHANGELOG.md`).
4. After AutoMerge, TagBot tags the release over the `DOCUMENTER_KEY` deploy key, the tag
   push runs CI, and the docs job publishes the versioned docs and moves `stable`.
