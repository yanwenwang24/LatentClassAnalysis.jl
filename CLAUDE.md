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
- `src/covariates.jl` hooks for latent class regression (later phase).
- `src/restarts.jl` random starts, two-stage multi-start driver, `_sort_by_size!`,
  `_init_split`.
- `src/fit.jl` `StatsAPI.fit` methods, `check_identifiability`, fit flags, erroring `fit!`.
- `src/predict.jl` `predict`, `classify`. `src/inference.jl` `vcov`, `profiles` (SEs
  later). `src/bootstrap.jl` stubs (later). `src/diagnostics.jl` StatsAPI accessors,
  criteria, `entropy`, `diagnostics`, Tables interface. `src/show.jl` `show`,
  `show_profiles`. `src/deprecated.jl` 0.2 shims, included last.
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
- Every exported symbol has a docstring and an `@docs` entry in `docs/src/api.md`; the docs
  build runs with `warnonly=false` and `checkdocs=:exports`, so a missing entry fails CI.
- Tests generate data with `StableRNG` seeds and pass `rng=StableRNG(seed)` to every
  `fit`; a fit is bitwise reproducible for a given `rng`, so tests may compare fits with
  `same_fit`. Never use the global RNG in tests.
- `prepare_data` always yields dense codes `1..C_j` per item (`0` = missing), ordered by
  `DataAPI.levels` (level order for categorical columns, sorted values otherwise); the
  labels are stored in `LCAData.item_levels` and `show_profiles`/`profiles` read them.
- Covariates, standard errors, and the bootstrap are unfinished in 0.3.0: their exported
  functions exist (docstrings say "not available in this version") and throw.
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
