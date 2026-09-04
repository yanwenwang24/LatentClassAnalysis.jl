using Test
using LatentClassAnalysis
using Random
using StableRNGs
using Statistics

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

@testset "Parameter Recovery" begin
    rng = StableRNG(1)
    n = 2000
    K, J = 2, 6
    true_class_probs = [0.6, 0.4]
    # Item-specific separation between 0.75 and 0.9
    separations = [0.75, 0.8, 0.85, 0.9, 0.78, 0.88]
    true_item_probs = [[s 1-s; 1-s s] for s in separations]
    data, classes = simulate_lca(rng, n, true_class_probs, true_item_probs)

    @test size(data) == (n, J)
    @test all(x -> x in (1, 2), data)
    @test length(classes) == n
    @test isapprox(mean(classes .== 1), true_class_probs[1]; atol=0.05)

    # fit! has no rng argument: the only randomness is the starting point drawn from the
    # global RNG in the LCAModel constructor, and there is a single start per fit. A single
    # seed can therefore land in a poor local optimum, so try a few seeds and require that
    # the fit with the best log-likelihood recovers the truth.
    best_ll = -Inf
    best_model = nothing
    for s in 1:5
        Random.seed!(s)
        model = LCAModel(K, J, fill(2, J))
        ll = fit!(model, data; tol=1e-8)
        if ll > best_ll
            best_ll = ll
            best_model = model
        end
    end
    @test best_model !== nothing
    @test isfinite(best_ll)

    perm = align_classes(best_model.item_probs, true_item_probs)
    @test sort(perm) == 1:K

    @testset "Class probabilities" begin
        est = best_model.class_probs[perm]
        @test all(abs.(est .- true_class_probs) .< 0.05)
    end

    @testset "Item probabilities" begin
        for j in 1:J
            est = best_model.item_probs[j][perm, :]
            @test all(abs.(est .- true_item_probs[j]) .< 0.05)
        end
    end

    @testset "Likelihood and classification" begin
        # The maximum-likelihood fit is at least as good as the true parameters
        ll_true = 0.0
        for i in 1:n
            ll_true += log(sum(true_class_probs[k] * prod(true_item_probs[j][k, data[i, j]] for j in 1:J) for k in 1:K))
        end
        @test best_ll >= ll_true - 1e-6

        assignments, _ = predict(best_model, data)
        accuracy = mean(invperm(perm)[assignments] .== classes)
        @test accuracy > 0.8
    end
end
