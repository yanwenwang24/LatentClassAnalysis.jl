using Test
using LatentClassAnalysis
using DataFrames
using Random
using StableRNGs

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

@testset "Model Diagnostics" begin
    # Tiny model: 2 classes, 3 items with 2, 3 and 2 categories
    rng = StableRNG(11)
    n = 400
    class_probs = [0.6, 0.4]
    item_probs = [
        [0.85 0.15; 0.2 0.8],
        [0.7 0.2 0.1; 0.1 0.2 0.7],
        [0.9 0.1; 0.3 0.7],
    ]
    data, _ = simulate_lca(rng, n, class_probs, item_probs)
    Random.seed!(5)
    model = LCAModel(2, 3, [2, 3, 2])
    ll = fit!(model, data)
    diag = diagnostics!(model, data, ll)

    @testset "Hand-computed criteria" begin
        K = model.n_classes
        p = (K - 1) + K * sum(model.n_categories .- 1)
        @test p == 1 + 2 * (1 + 2 + 1)  # 9 free parameters

        @test diag isa ModelDiagnostics
        @test diag.ll == ll
        @test isapprox(diag.aic, -2ll + 2p)
        @test isapprox(diag.bic, -2ll + p * log(n))
        @test isapprox(diag.sbic, -2ll + p * log((n + 2) / 24))
        @test all(isfinite, (diag.ll, diag.aic, diag.bic, diag.sbic, diag.entropy))
        @test diag.bic > diag.aic   # log(400) > 2, so BIC penalizes more
        @test diag.sbic < diag.bic  # log((n + 2) / 24) < log(n)
        @test diag.aic > -2ll       # penalties are positive
    end

    @testset "Entropy" begin
        @test 0 <= diag.entropy <= 1

        # The relative entropy is 1 - (mean posterior entropy) / log(K), recomputed from predict
        _, post = predict(model, data)
        h = -sum(post .* log.(post .+ eps())) / (n * log(model.n_classes))
        @test isapprox(diag.entropy, 1 - h; atol=1e-10)

        # Identical classes: every posterior equals the class prior, so the entropy is 0
        toy = rand(StableRNG(3), 1:2, 100, 3)
        m_same = LCAModel(2, 3, [2, 2, 2])
        m_same.class_probs .= [0.5, 0.5]
        for P in m_same.item_probs
            P .= [0.7 0.3; 0.7 0.3]
        end
        @test isapprox(diagnostics!(m_same, toy, -1.0).entropy, 0.0; atol=1e-8)

        # Nearly deterministic items: every observation is classified with certainty, entropy ≈ 1
        m_sharp = LCAModel(2, 3, [2, 2, 2])
        m_sharp.class_probs .= [0.5, 0.5]
        for P in m_sharp.item_probs
            P .= [0.999 0.001; 0.001 0.999]
        end
        @test diagnostics!(m_sharp, toy, -1.0).entropy > 0.99

        # A well-separated fitted model has a high entropy
        rng2 = StableRNG(12)
        sharp_probs = [[0.95 0.05; 0.05 0.95] for _ in 1:5]
        d_sharp, _ = simulate_lca(rng2, 600, [0.5, 0.5], sharp_probs)
        Random.seed!(6)
        m_fit = LCAModel(2, 5, fill(2, 5))
        ll_fit = fit!(m_fit, d_sharp)
        @test diagnostics!(m_fit, d_sharp, ll_fit).entropy > 0.8
    end

    @testset "Abstract matrix and Real inputs" begin
        # A view of the data gives identical results
        d_view = diagnostics!(model, view(data, :, :), ll)
        @test d_view.ll == diag.ll
        @test d_view.aic == diag.aic
        @test d_view.bic == diag.bic
        @test d_view.sbic == diag.sbic
        @test isapprox(d_view.entropy, diag.entropy; atol=1e-12)

        # Int32 data and a Float32 log-likelihood
        d_32 = diagnostics!(model, Int32.(data), Float32(ll))
        @test d_32 isa ModelDiagnostics
        @test d_32.ll isa Float64
        @test isapprox(d_32.ll, ll; rtol=1e-6)        # Float32 precision
        @test isapprox(d_32.aic, diag.aic; rtol=1e-5)
        @test isapprox(d_32.bic, diag.bic; rtol=1e-5)
        @test isapprox(d_32.sbic, diag.sbic; rtol=1e-5)
        @test isapprox(d_32.entropy, diag.entropy; atol=1e-10)

        # Int32 data with an integer log-likelihood
        d_int = diagnostics!(model, Int32.(data), -500)
        @test d_int.ll == -500.0
        @test d_int.aic == 1000.0 + 2 * 9

        # Only n enters BIC and sBIC, so a subset of rows changes those but not AIC
        half = view(data, 1:200, :)
        d_half = diagnostics!(model, half, ll)
        @test d_half.aic == diag.aic
        @test isapprox(d_half.bic, -2ll + 9 * log(200))
        @test isapprox(d_half.sbic, -2ll + 9 * log(202 / 24))
    end

    @testset "diagnostics! does not modify its inputs" begin
        cp = copy(model.class_probs)
        ip = deepcopy(model.item_probs)
        d0 = copy(data)
        diagnostics!(model, data, ll)
        @test model.class_probs == cp
        @test model.item_probs == ip
        @test data == d0
    end

    @testset "Prepared DataFrame round trip" begin
        # The existing workflow: prepare_data -> LCAModel -> fit! -> diagnostics!
        df = DataFrame([Symbol("x$j") => data[:, j] for j in 1:3]...)
        pdata, n_cats = prepare_data(df, :x1, :x2, :x3)
        @test pdata == data
        @test n_cats == [2, 3, 2]
        Random.seed!(5)
        m2 = LCAModel(2, 3, n_cats)
        ll2 = fit!(m2, pdata)
        d2 = diagnostics!(m2, pdata, ll2)
        @test isapprox(d2.aic, diag.aic; atol=1e-6)
        @test isapprox(d2.bic, diag.bic; atol=1e-6)
    end
end
