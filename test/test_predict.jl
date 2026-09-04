using Test
using LatentClassAnalysis
using Random
using StableRNGs
using Statistics

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

@testset "Prediction" begin
    rng = StableRNG(21)
    n_items = 6
    n_train = 600
    class_probs = [0.5, 0.5]
    item_probs = [[0.9 0.1; 0.1 0.9] for _ in 1:n_items]
    data, classes = simulate_lca(rng, n_train, class_probs, item_probs)
    Random.seed!(8)
    model = LCAModel(2, n_items, fill(2, n_items))
    fit!(model, data)

    @testset "Shapes and consistency" begin
        assignments, probs = predict(model, data)
        @test assignments isa Vector{Int}
        @test probs isa Matrix{Float64}
        @test length(assignments) == n_train
        @test size(probs) == (n_train, 2)
        @test all(1 .<= assignments .<= 2)
        @test all(0 .<= probs .<= 1)
        @test all(isapprox.(sum(probs, dims=2), 1.0, atol=1e-10))
        @test all(assignments[i] == argmax(probs[i, :]) for i in 1:n_train)
        @test length(unique(assignments)) == 2  # both classes are used
    end

    @testset "Held-out rows" begin
        n_test = 37
        new_data, _ = simulate_lca(StableRNG(22), n_test, class_probs, item_probs)
        assignments, probs = predict(model, new_data)
        @test length(assignments) == n_test
        @test size(probs) == (n_test, 2)
        @test all(1 .<= assignments .<= 2)
        @test all(isapprox.(sum(probs, dims=2), 1.0, atol=1e-10))

        # A single observation
        a1, p1 = predict(model, new_data[1:1, :])
        @test length(a1) == 1
        @test size(p1) == (1, 2)
        @test a1[1] == assignments[1]
        @test p1[1, :] ≈ probs[1, :]

        # Predictions are row-wise: a row gives the same posterior wherever it appears
        a5, p5 = predict(model, new_data[5:5, :])
        @test a5[1] == assignments[5]
        @test p5[1, :] ≈ probs[5, :]

        # Zero rows
        a0, p0 = predict(model, new_data[1:0, :])
        @test isempty(a0)
        @test size(p0) == (0, 2)
    end

    @testset "Recovers simulated classes" begin
        perm = align_classes(model.item_probs, item_probs)
        est_to_true = invperm(perm)  # estimated class label -> true class label
        assignments, _ = predict(model, data)
        accuracy = mean(est_to_true[assignments] .== classes)
        @test accuracy > 0.9
    end

    @testset "Abstract matrix inputs" begin
        a_ref, p_ref = predict(model, data)
        a_view, p_view = predict(model, view(data, 1:50, :))
        @test a_view == a_ref[1:50]
        @test p_view ≈ p_ref[1:50, :]
        a_32, p_32 = predict(model, Int32.(data))
        @test a_32 == a_ref
        @test p_32 ≈ p_ref
    end

    @testset "Posterior follows Bayes' rule" begin
        # posterior ∝ class prior × product of item response probabilities
        for i in (1, 17, n_train)
            x = data[i, :]
            w = [model.class_probs[k] * prod(model.item_probs[j][k, x[j]] for j in 1:n_items) for k in 1:2]
            _, p = predict(model, data[i:i, :])
            @test p[1, :] ≈ w ./ sum(w)
        end
    end

    @testset "predict does not modify its inputs" begin
        cp = copy(model.class_probs)
        ip = deepcopy(model.item_probs)
        d0 = copy(data)
        predict(model, data)
        @test model.class_probs == cp
        @test model.item_probs == ip
        @test data == d0
    end
end
