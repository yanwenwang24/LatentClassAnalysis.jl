using Test
using LatentClassAnalysis
using Random
using StableRNGs

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

@testset "Model Fitting" begin
    # 2 classes, 5 well-separated binary items, 500 observations
    rng = StableRNG(2024)
    n_items = 5
    class_probs = [0.55, 0.45]
    item_probs = [[0.8 0.2; 0.2 0.8] for _ in 1:n_items]
    data, _ = simulate_lca(rng, 500, class_probs, item_probs)

    @testset "Basic fit" begin
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))
        ll = fit!(model, data)

        @test ll isa Float64
        @test !isnan(ll)
        @test !isinf(ll)
        @test ll < 0
        @test all(0 .<= model.class_probs .<= 1)
        @test isapprox(sum(model.class_probs), 1.0, atol=1e-10)
        for item_prob in model.item_probs
            @test all(0 .<= item_prob .<= 1)
            @test all(isapprox.(sum(item_prob, dims=2), 1.0, atol=1e-10))
        end

        # Refitting an already converged model leaves the log-likelihood essentially unchanged
        ll2 = fit!(model, data)
        @test isapprox(ll2, ll; atol=1e-4)
    end

    @testset "Low number of observations warning" begin
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))

        # Exact message for 100 rows
        @test_logs (:warn, "Low number of observations (100) may affect model fitting. Consider using more data for better results.") begin
            fit!(model, data[1:100, :])
        end

        # The threshold is 300 rows
        @test_logs (:warn, r"^Low number of observations \(299\) may affect model fitting") begin
            fit!(model, data[1:299, :])
        end
        @test_logs fit!(model, data[1:300, :])
        @test_logs fit!(model, data)
    end

    @testset "Dimension mismatch" begin
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))
        @test_throws ArgumentError fit!(model, data[:, 1:4])
        @test_throws "Number of items in data (4) doesn't match model (5)" fit!(model, data[:, 1:4])
        @test_throws ArgumentError fit!(model, hcat(data, data[:, 1]))
        @test_throws "Number of items in data (6) doesn't match model (5)" fit!(model, hcat(data, data[:, 1]))
        # The model is untouched by a failed call
        @test model.class_probs == [0.5, 0.5]
    end

    @testset "Invalid category" begin
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))

        bad = copy(data)
        bad[7, 3] = 3  # a 3 in a 2-category column
        @test_throws ArgumentError fit!(model, bad)
        @test_throws "Invalid category in column 3. Expected values in 1:2, but got values in 1:3" fit!(model, bad)

        bad0 = copy(data)
        bad0[1, 1] = 0  # zero-based codes are rejected
        @test_throws ArgumentError fit!(model, bad0)
        @test_throws "Data should be 1-based" fit!(model, bad0)

        # A model with more categories accepts the same data
        Random.seed!(1)
        wide = LCAModel(2, n_items, [2, 2, 3, 2, 2])
        @test isfinite(fit!(wide, bad))
    end

    @testset "verbose output" begin
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))
        out = capture_stdout(() -> fit!(model, data; verbose=true))
        @test occursin("Iteration 1: log-likelihood = ", out)
        @test occursin(r"Converged after \d+ iterations", out)
        @test !occursin("Maximum iterations reached", out)

        # Nothing is printed by default
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))
        @test capture_stdout(() -> fit!(model, data)) == ""
    end

    @testset "max_iter" begin
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))
        ll1 = fit!(model, data; max_iter=1)
        @test ll1 isa Float64
        @test isfinite(ll1)

        # Hitting the iteration limit is reported when verbose
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))
        out = capture_stdout(() -> fit!(model, data; max_iter=1, verbose=true))
        @test occursin("Maximum iterations reached", out)
        @test !occursin("Converged", out)

        # EM never decreases the log-likelihood: a full fit from the same start is at least as good
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))
        ll_full = fit!(model, data)
        @test ll_full >= ll1 - 1e-8

        # ... and it is monotone along the way (each call continues from the current parameters)
        Random.seed!(1)
        model = LCAModel(2, n_items, fill(2, n_items))
        lls = [fit!(model, data; max_iter=1) for _ in 1:5]
        @test all(diff(lls) .>= -1e-8)
    end

    @testset "tol" begin
        Random.seed!(1)
        m_loose = LCAModel(2, n_items, fill(2, n_items))
        Random.seed!(1)
        m_tight = LCAModel(2, n_items, fill(2, n_items))
        out_loose = capture_stdout(() -> fit!(m_loose, data; tol=1e-2, verbose=true))
        out_tight = capture_stdout(() -> fit!(m_tight, data; tol=1e-8, verbose=true))
        iters(s) = parse(Int, match(r"Converged after (\d+) iterations", s).captures[1])
        @test iters(out_loose) <= iters(out_tight)
        # Float32 tolerance is accepted
        Random.seed!(1)
        m32 = LCAModel(2, n_items, fill(2, n_items))
        @test isfinite(fit!(m32, data; tol=1f-4))
    end

    @testset "Abstract matrix inputs" begin
        Random.seed!(1)
        m_ref = LCAModel(2, n_items, fill(2, n_items))
        ll_ref = fit!(m_ref, data)

        Random.seed!(1)
        m_view = LCAModel(2, n_items, fill(2, n_items))
        ll_view = fit!(m_view, view(data, :, :))

        Random.seed!(1)
        m_32 = LCAModel(2, n_items, fill(2, n_items))
        ll_32 = fit!(m_32, Int32.(data))

        @test isapprox(ll_view, ll_ref; atol=1e-8)
        @test isapprox(ll_32, ll_ref; atol=1e-8)
        @test all(isapprox.(m_view.class_probs, m_ref.class_probs; atol=1e-8))
        @test all(isapprox.(m_32.class_probs, m_ref.class_probs; atol=1e-8))
        for j in 1:n_items
            @test all(isapprox.(m_view.item_probs[j], m_ref.item_probs[j]; atol=1e-8))
            @test all(isapprox.(m_32.item_probs[j], m_ref.item_probs[j]; atol=1e-8))
        end
    end

    @testset "Polytomous items" begin
        rng_poly = StableRNG(7)
        cp = [0.5, 0.5]
        ip = [
            [0.7 0.2 0.1; 0.1 0.2 0.7],
            [0.6 0.4; 0.3 0.7],
            [0.1 0.3 0.6; 0.6 0.3 0.1],
            [0.8 0.2; 0.2 0.8],
        ]
        poly, _ = simulate_lca(rng_poly, 400, cp, ip)
        @test [maximum(poly[:, j]) for j in 1:4] == [3, 2, 3, 2]

        Random.seed!(3)
        model = LCAModel(2, 4, [3, 2, 3, 2])
        ll = fit!(model, poly)
        @test isfinite(ll)
        @test [size(P) for P in model.item_probs] == [(2, 3), (2, 2), (2, 3), (2, 2)]
        for P in model.item_probs
            @test all(0 .<= P .<= 1)
            @test all(isapprox.(sum(P, dims=2), 1.0, atol=1e-10))
        end
    end
end
