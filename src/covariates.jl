# Latent class regression (multinomial-logit class-membership model with covariates) lands
# in a later phase. This file defines only the hooks that the EM core dispatches to when a
# parameter bundle carries coefficients (`θ.coefs !== nothing`).

const _COVARIATES_NOT_IMPLEMENTED =
    "latent class regression with covariates is not implemented in this version; " *
    "fit the unconditional model with covariates=false"

# Log prior of every class for (unique) row `u` given the coefficients, written into `w`.
_logprior!(w, θ::LCAParams, ws::LCAWorkspace, u::Integer) =
    throw(ErrorException(_COVARIATES_NOT_IMPLEMENTED))

# One damped Newton step on the coefficients (generalized EM M-step).
_update_coefs!(θ::LCAParams, ws::LCAWorkspace) = throw(ErrorException(_COVARIATES_NOT_IMPLEMENTED))
