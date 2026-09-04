# LatentClassAnalysis.jl

Julia package for latent class analysis (LCA): finite mixtures of independent multinomials
fitted by EM to categorical indicators. The audience is applied social scientists. The
bundled example replicates Wang, Teerawichitchainan & Ho (2024, Advances in Life Course
Research, doi 10.1016/j.alcr.2024.100628).

## Layout

- `src/LatentClassAnalysis.jl` module root: imports, exports, includes.
- `src/types.jl` `LCAModel` (mutable; parameters live here) and `ModelDiagnostics`.
- `src/utils.jl` `prepare_data`, `check_identifiability`, `diagnostics!`, `show_profiles`.
- `src/fit.jl` EM loop (`fit!`). `src/predict.jl` posterior class membership (`predict`).
- `test/` one file per concern, included from `runtests.jl`; `testutils.jl` has the
  simulation and class-alignment helpers.
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
- Tests generate data with `StableRNG` seeds. In 0.2.x fits use the global RNG, so tests
  assert properties and tolerances, never exact fitted values.
- `prepare_data` always yields dense codes `1..K` per column, ordered by sorted distinct
  value (level order for categorical columns). `show_profiles` labels rely on that order.
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
