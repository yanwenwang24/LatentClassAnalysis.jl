# API reference

The sections below follow the workflow: prepare the data, fit, inspect the fit
statistics, predict, and read the profiles. Every function and type listed under a
section heading is exported by `LatentClassAnalysis`, except where noted. The
[Internals](@ref) section documents unexported helpers that are useful to know about
but may change between minor versions.

The bootstrap is coming in the 0.3.0 release: its functions are exported and documented
but throw an error in this build.

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

```@docs
nobs
dof
loglikelihood
isfitted
aic
bic
aicc
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

The free parameters live on the logit scale (see [`coef`](@ref) for the layout); their
covariance matrix is the inverse of the observed information, computed by `fit` unless
`se = :none`. [`profiles`](@ref) carries the delta-method standard errors of the
response probabilities.

```@docs
coef
coefnames
vcov
stderror
confint
coeftable
informationmatrix
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
