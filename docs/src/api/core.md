# [API: data, fitting, prediction](@id api-core)

The two API pages follow the workflow. This page covers preparing the data, fitting,
inspecting the fit statistics, predicting, and reading the profiles; the second,
[API: inference, bootstrap, deprecated](@ref api-inference), covers standard errors,
simulation and the bootstrap, the deprecated 0.2 names, and the internals. Every
function and type listed under a section heading is exported by
`LatentClassAnalysis`, except where noted.

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
profiles(::LCAModel)
show_profiles
```
