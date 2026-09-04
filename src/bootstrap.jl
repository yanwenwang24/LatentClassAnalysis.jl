# Simulation, bootstrap standard errors and the bootstrap likelihood-ratio test land in a
# later phase. The exported functions are defined here so the API resolves; they throw
# until implemented. `pvalue(::BootstrapLRT)` is complete.

const _BOOTSTRAP_NOT_IMPLEMENTED = "is not implemented in this version"

"""
    simulate(m::LCAModel, n=nobs(m); rng=Random.default_rng()) -> LCAData

Draw `n` observations from the fitted model: a class for every observation from the class
sizes (or the covariate-specific membership probabilities), then a response to every item
from the class's response probabilities. The result carries the item names and levels of
`m`. Not available in this version.
"""
function simulate(m::LCAModel, n::Integer=nobs(m); rng::AbstractRNG=Random.default_rng())
    throw(ErrorException("simulate $_BOOTSTRAP_NOT_IMPLEMENTED"))
end

"""
    bootstrap(m::LCAModel; n_boot=200, rng=Random.default_rng(), parametric=false) -> LCABootstrap

Bootstrap standard errors of the free parameters of `m`: resample rows (or simulate from
the model with `parametric=true`), refit, align the class labels to `m`, and collect the
coefficients on the [`coef`](@ref) scale. Not available in this version.
"""
function bootstrap(m::LCAModel; n_boot::Integer=200, rng::AbstractRNG=Random.default_rng(),
                   parametric::Bool=false)
    throw(ErrorException("bootstrap $_BOOTSTRAP_NOT_IMPLEMENTED"))
end

"""
    bootstrap_lrt(null::LCAModel, alternative::LCAModel; n_boot=100, rng=Random.default_rng()) -> BootstrapLRT

Parametric bootstrap likelihood-ratio test of a `K`-class model against the `K + 1`-class
model fitted to the same data. Not available in this version.
"""
function bootstrap_lrt(null::LCAModel, alternative::LCAModel; n_boot::Integer=100,
                       rng::AbstractRNG=Random.default_rng())
    throw(ErrorException("bootstrap_lrt $_BOOTSTRAP_NOT_IMPLEMENTED"))
end

"""
    pvalue(t::BootstrapLRT) -> Float64

Bootstrap p-value of a [`BootstrapLRT`](@ref), `(1 + #{replicates ≥ statistic}) / (n_boot + 1)`.
"""
StatsAPI.pvalue(t::BootstrapLRT) = t.pvalue
