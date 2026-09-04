# API reference

The sections below follow the workflow: prepare the data, fit, inspect the fit
statistics, predict, and read the profiles. Every function and type listed under a
section heading is exported by `LatentClassAnalysis`, except where noted. The
[Internals](@ref) section documents unexported helpers that are useful to know about
but may change between minor versions.

Covariates, standard errors, and the bootstrap are coming in the 0.3.0 release: their
functions are exported and documented but throw an error in this build.

## Data

```@docs
prepare_data(::Any, ::AbstractVector{<:Union{Symbol,AbstractString}})
LCAData
hasmissing
nmissing
hascovariates
```

## Fitting

```@docs
fit
LCAOptions
LatentClassAnalysis.FitFlags
```

## Model

```@docs
LCAModel
```

## Fit statistics

`aic`, `bic` and `aicc` are the StatsAPI defaults computed from
[`loglikelihood`](@ref), [`dof`](@ref) and [`nobs`](@ref).

```@docs
nobs
dof
loglikelihood
isfitted
aic
bic
sbic
entropy
diagnostics
ModelDiagnostics
```

## Prediction

```@docs
predict
classify
```

## Profiles

```@docs
profiles
show_profiles
```

## Standard errors

Coming in the 0.3.0 release: `coef`, `coefnames`, `stderror`, `confint`, `coeftable`
and `informationmatrix` are exported for that purpose but have no methods yet;
`vcov` throws until a covariance matrix is computed.

```@docs
coef
vcov
```

## Simulation and bootstrap

Coming in the 0.3.0 release; the functions below throw in this build.

```@docs
simulate
bootstrap
LCABootstrap
bootstrap_lrt
BootstrapLRT
pvalue
```

## Deprecated

The three shims below work with a deprecation warning and will be removed in 0.4.0;
the two constructors of the 0.2 workflow throw. See [Upgrading from 0.2](@ref).

```@docs
prepare_data(::Any, ::Symbol...)
diagnostics!
fit!
LCAModel(::Integer, ::Integer, ::AbstractVector{<:Integer})
```

`show_profiles(model, df, cols; kwargs...)` forwards to `show_profiles(model; kwargs...)`.

## Internals

```@autodocs
Modules = [LatentClassAnalysis]
Public = false
Filter = t -> t !== LatentClassAnalysis.FitFlags
```
