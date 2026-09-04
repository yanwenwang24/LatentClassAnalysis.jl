# [API: inference, bootstrap, deprecated](@id api-inference)

Standard errors and confidence intervals, simulation and the bootstrap, the deprecated
0.2 names, and the unexported internals. The first page,
[API: data, fitting, prediction](@ref api-core), covers preparing the data, fitting,
the fit statistics, prediction and the profiles. Every function and type listed under
a section heading is exported by `LatentClassAnalysis`, except where noted; the
[Internals](@ref) section documents unexported helpers that are useful to know about
but may change between minor versions.

## Standard errors

The free parameters live on the logit scale (see [`coef`](@ref) for the layout); their
covariance matrix is the inverse of the observed information, computed by `fit` unless
`se = :none`. [`profiles`](@ref) carries the delta-method standard errors of the
response probabilities.

```@docs
coef
coefnames
vcov(::LCAModel)
stderror(::LCAModel)
confint(::LCAModel)
coeftable(::LCAModel)
informationmatrix
```

## Simulation and bootstrap

[`simulate`](@ref) draws data sets from a fitted model. [`bootstrap`](@ref) refits the
model to resampled (or simulated) data sets, aligns the class labels of every replicate
to the model, and returns an [`LCABootstrap`](@ref) whose `vcov`, `stderror`, `confint`,
`coeftable` and `profiles` methods give bootstrap standard errors and percentile
intervals. [`bootstrap_lrt`](@ref) is the parametric bootstrap likelihood-ratio test of
``K`` against ``K + 1`` classes; see the [methodology](@ref blrt) for the algorithm.

```@docs
simulate
bootstrap
LCABootstrap
vcov(::LCABootstrap)
stderror(::LCABootstrap)
confint(::LCABootstrap)
coeftable(::LCABootstrap)
profiles(::LCABootstrap)
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
