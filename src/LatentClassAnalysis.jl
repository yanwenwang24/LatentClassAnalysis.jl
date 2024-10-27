module LatentClassAnalysis

using CategoricalArrays
using DataFrames
using Distributions
using LinearAlgebra
using Random
using Statistics

export LCAModel, 
       ModelDiagnostics,
       fit!, 
       predict, 
       prepare_data,
       diagnostics!

include("types.jl")
include("utils.jl")
include("fit.jl")
include("predict.jl")

end # module LatentClassAnalysis
