using Test
using LatentClassAnalysis
using CategoricalArrays
using DataFrames
using StableRNGs
using Statistics

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

@testset "Prediction" begin
    n_items = 6
    n_train = 600
    class_probs = [0.5, 0.5]
    item_probs = [[0.9 0.1; 0.1 0.9] for _ in 1:n_items]
    y, classes = simulate_lca(StableRNG(21), n_train, class_probs, item_probs)
    d = LCAData(y)
    m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=4, n_final=2)

    @testset "Training data" begin
        post = predict(m)
        @test post == m.posterior
        @test post !== m.posterior                       # a copy
        @test post isa Matrix{Float64}
        @test size(post) == (n_train, 2)
        @test all(0 .<= post .<= 1)
        @test all(abs.(sum(post, dims=2) .- 1) .< 1e-12)
        @test predict(m, d) == m.posterior               # recomputed from the data
        c = classify(m)
        @test c isa Vector{Int}
        @test length(c) == n_train
        @test all(c[i] == argmax(post[i, :]) for i in 1:n_train)
        @test classify(m, d) == c
        @test length(unique(c)) == 2
    end

    @testset "Held-out LCAData" begin
        n_test = 37
        new_y, _ = simulate_lca(StableRNG(22), n_test, class_probs, item_probs)
        dn = LCAData(new_y)
        post = predict(m, dn)
        @test size(post) == (n_test, 2)
        @test all(abs.(sum(post, dims=2) .- 1) .< 1e-12)
        @test classify(m, dn) == [argmax(post[i, :]) for i in 1:n_test]

        # Row-wise: a row gives the same posterior wherever and however often it appears
        p1 = predict(m, LCAData(new_y[1:1, :]; n_categories=fill(2, n_items)))
        @test size(p1) == (1, 2)
        @test p1[1, :] == post[1, :]
        p5 = predict(m, LCAData(new_y[[5, 5, 1], :]; n_categories=fill(2, n_items)))
        @test p5[1, :] == post[5, :] && p5[2, :] == post[5, :] && p5[3, :] == post[1, :]

        # Zero rows
        d0 = LCAData(new_y[1:0, :]; n_categories=fill(2, n_items))
        @test size(predict(m, d0)) == (0, 2)
        @test isempty(classify(m, d0))

        # An item may show fewer categories than the model, never more
        @test_throws ArgumentError predict(m, LCAData(new_y; n_categories=[3; fill(2, 5)]))
        items3 = [[0.7 0.2 0.1; 0.1 0.2 0.7], item_probs[2:end]...]
        y3, _ = simulate_lca(StableRNG(23), 400, class_probs, items3)
        m3 = fit(LCAModel, LCAData(y3), 2; rng=StableRNG(1), n_starts=2, n_final=1)
        @test m3.n_categories == [3; fill(2, 5)]
        two = LCAData(new_y)                        # first item shows two of the three categories
        p3 = predict(m3, two)
        @test size(p3) == (n_test, 2)
        @test p3 == predict(m3, LCAData(new_y; n_categories=[3; fill(2, 5)]))
        @test_throws ArgumentError predict(m, LCAData(new_y[:, 1:5]))
        @test_throws ArgumentError classify(m, LCAData(new_y[:, 1:5]))
    end

    @testset "Posterior follows Bayes' rule" begin
        for i in (1, 17, n_train)
            x = y[i, :]
            w = [m.class_probs[k] * prod(m.item_probs[j][k, x[j]] for j in 1:n_items) for k in 1:2]
            @test m.posterior[i, :] ≈ w ./ sum(w)
        end
    end

    @testset "Recovers simulated classes" begin
        perm = align_classes(m.item_probs, item_probs)
        est_to_true = invperm(perm)
        @test mean(est_to_true[classify(m)] .== classes) > 0.9
    end

    @testset "Tables" begin
        names = [Symbol("x$j") for j in 1:n_items]
        df = DataFrame([names[j] => y[:, j] for j in 1:n_items]...)
        dt = prepare_data(df, names)
        mt = fit(LCAModel, dt, 2; rng=StableRNG(1), n_starts=4, n_final=2)
        @test predict(mt, df) == mt.posterior
        @test classify(mt, df) == classify(mt)

        # A table whose column shows a single level still uses the training coding
        df1 = DataFrame([names[j] => fill(2, 3) for j in 1:n_items]...)
        p_absent = predict(mt, df1)
        @test size(p_absent) == (3, 2)
        expected = predict(mt, LCAData(fill(2, 3, n_items); n_categories=fill(2, n_items)))
        @test p_absent == expected

        # Level labels are matched by their string form, whatever the column type
        df_str = DataFrame([names[j] => string.(y[1:10, j]) for j in 1:n_items]...)
        @test predict(mt, df_str) == mt.posterior[1:10, :]
        df_cat = DataFrame([names[j] => categorical(y[1:10, j]; levels=[2, 1]) for j in 1:n_items]...)
        @test predict(mt, df_cat) == mt.posterior[1:10, :]
        nt = NamedTuple{Tuple(names)}(Tuple(y[1:10, j] for j in 1:n_items))
        @test predict(mt, nt) == mt.posterior[1:10, :]

        # Missing responses and unknown values
        dfm = DataFrame([names[j] => Vector{Union{Missing,Int}}(y[1:10, j]) for j in 1:n_items]...)
        dfm[1, :x1] = missing
        pm = predict(mt, dfm)
        @test pm[2:end, :] == mt.posterior[2:10, :]
        @test pm[1, :] != mt.posterior[1, :]
        df_bad = DataFrame([names[j] => y[1:10, j] for j in 1:n_items]...)
        df_bad[1, :x1] = 3
        @test_throws ArgumentError predict(mt, df_bad)
        @test_throws ArgumentError predict(mt, df[:, 1:5])         # missing item column
    end

    @testset "Matrices are rejected" begin
        @test_throws ArgumentError predict(m, y)
        @test_throws ArgumentError predict(m, view(y, 1:10, :))
        @test_throws ArgumentError classify(m, y)
        @test_throws "wrap the codes in LCAData" predict(m, y)
    end

    @testset "predict does not modify its inputs" begin
        cp = copy(m.class_probs)
        ip = deepcopy(m.item_probs)
        d0 = copy(y)
        predict(m, d)
        classify(m, d)
        @test m.class_probs == cp
        @test m.item_probs == ip
        @test d.y == d0
    end
end
