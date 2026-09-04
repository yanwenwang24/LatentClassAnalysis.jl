module LatentClassAnalysis

using LinearAlgebra   # Newton step and information matrix (later phases)
using Printf
using Random
using Tables
import DataAPI
import Distributions
import Statistics: mean   # covariate-averaged class sizes (later phase)
import StatsAPI
import StatsAPI: fit, fit!, predict, loglikelihood, nobs, dof, aic, bic, aicc, coef,
    coefnames, vcov, stderror, confint, coeftable, informationmatrix, isfitted, pvalue
import StatsBase
import StatsBase: entropy

# StatsAPI / StatsBase verbs extended for LCAModel
export fit, fit!, predict, loglikelihood, nobs, dof, aic, bic, aicc, coef, coefnames, vcov,
    stderror, confint, coeftable, informationmatrix, isfitted, pvalue, entropy

# Package types and verbs
export LCAData, LCAOptions, LCAModel, ModelDiagnostics,
    prepare_data, diagnostics, sbic, classify, profiles, show_profiles,
    simulate, bootstrap, bootstrap_lrt, LCABootstrap, BootstrapLRT,
    hasmissing, nmissing, hascovariates

include("types.jl")
include("data.jl")
include("em.jl")
include("covariates.jl")
include("restarts.jl")
include("fit.jl")
include("predict.jl")
include("inference.jl")
include("bootstrap.jl")
include("diagnostics.jl")
include("show.jl")
include("deprecated.jl")

end # module LatentClassAnalysis
