using Test
using LatentClassAnalysis

include("testutils.jl")

@testset "LatentClassAnalysis.jl" begin
    include("test_aqua.jl")
    include("test_data.jl")
    include("test_em.jl")
    include("test_fit.jl")
    include("test_missing.jl")
    include("test_covariates.jl")
    include("test_statsapi.jl")
    include("test_inference.jl")
    include("test_predict.jl")
    include("test_show.jl")
    include("test_deprecated.jl")
    include("test_docs.jl")
end
