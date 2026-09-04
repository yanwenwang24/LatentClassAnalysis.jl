using Test
using LatentClassAnalysis
using DataFrames
using StableRNGs

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

@testset "Deprecations" begin
    y, _ = simulate_lca(StableRNG(41), 300, [0.6, 0.4], [[0.8 0.2; 0.2 0.8] for _ in 1:4])
    df = DataFrame([Symbol("x$j") => y[:, j] for j in 1:4]...)
    cols = [:x1, :x2, :x3, :x4]
    d = prepare_data(df, cols)
    m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=2, n_final=1)

    @testset "prepare_data(df, cols...) returns the 0.2 tuple" begin
        r = @test_deprecated prepare_data(df, :x1, :x2, :x3, :x4)
        @test r isa Tuple{Matrix{Int},Vector{Int}}
        @test r[1] == d.y
        @test r[2] == d.n_categories
        r1 = @test_deprecated prepare_data(df, :x1)
        @test r1[1] == d.y[:, 1:1] && r1[2] == [2]
        # zero_based is ignored but its length is still validated
        rz = @test_deprecated prepare_data(df, :x1, :x2; zero_based=[true, false])
        @test rz[1] == d.y[:, 1:2]
        @test_throws ArgumentError prepare_data(df, :x1, :x2; zero_based=[true])
    end

    @testset "diagnostics! forwards to diagnostics" begin
        diag = @test_deprecated diagnostics!(m, y, m.loglik)
        @test diag isa ModelDiagnostics
        @test diag == diagnostics(m)
    end

    @testset "show_profiles(m, df, cols) forwards to show_profiles(m)" begin
        buf = IOBuffer()
        r = @test_deprecated show_profiles(m, df, cols; io=buf)
        @test r === nothing
        out = String(take!(buf))
        @test out == sprint(io -> show_profiles(m; io=io))
        @test occursin("Latent Class Profiles", out)
    end

    @testset "Removed 0.2 entry points" begin
        @test_throws ArgumentError LCAModel(2, 5, fill(2, 5))
        @test_throws "replaced by fit(LCAModel, data, k) in v0.3" LCAModel(2, 5, fill(2, 5))
        @test_throws ArgumentError LCAModel(Int32(2), Int8(4), Int32[2, 2, 2, 2])
        @test_throws ArgumentError fit!(m, y)
        @test_throws "replaced by fit(LCAModel, data, k) in v0.3" fit!(m, y)
        @test_throws ArgumentError fit!(m, y; max_iter=10, tol=1e-6, verbose=true)
        @test_throws ArgumentError fit!(m)
        # The 0.2 idiom `assignments, probs = predict(model, data)` fails loudly
        @test_throws ArgumentError predict(m, y)
    end
end
