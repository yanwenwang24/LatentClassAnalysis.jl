using Test
using LatentClassAnalysis

include("testutils.jl")

@testset "LatentClassAnalysis.jl" begin
    include("test_aqua.jl")
    include("test_data.jl")
    include("test_model.jl")
    include("test_fit.jl")
    include("test_diagnostics.jl")
    include("test_predict.jl")
    include("test_show.jl")
    include("test_docs.jl")
    include("test_recovery.jl")
end
