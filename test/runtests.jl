using Test
using LatentClassAnalysis
using CategoricalArrays
using DataFrames
using Distributions
using LinearAlgebra
using Random
using Statistics

@testset "LatentClassAnalysis.jl" begin
    Random.seed!(123)  # For reproducibility

    @testset "Data Preparation" begin
        # Create test data with sufficient variables
        df = DataFrame(
            x1=repeat([1, 2], 50),         # binary 1/2
            x2=repeat([0, 1], 50),         # binary 0/1
            x3=categorical(repeat(["A", "B", "C"], 34)[1:100]),  # 3 categories
            x4=repeat([1, 2], 50),         # binary 1/2
            x5=repeat([1, 2], 50)          # binary 1/2
        )

        # Test basic data preparation
        data, n_cats = prepare_data(df, :x1, :x2, :x3, :x4, :x5)
        @test size(data) == (100, 5)
        @test n_cats == [2, 2, 3, 2, 2]
        @test all(x -> x ≥ 1, data)  # All values should be 1-based

        # Test with single column
        data_single, n_cats_single = prepare_data(df, :x1)
        @test size(data_single) == (100, 1)
        @test n_cats_single == [2]
    end

    @testset "Model Initialization" begin
        # Valid model (5 binary items, 2 classes)
        n_items = 5
        n_classes = 2
        n_categories = fill(2, n_items)
        model = LCAModel(n_classes, n_items, n_categories)

        # Test model structure
        @test model.n_classes == 2
        @test model.n_items == 5
        @test length(model.class_probs) == 2
        @test length(model.item_probs) == 5
        @test all(isapprox(sum(model.class_probs), 1.0))

        # Test invalid inputs
        @test_throws ArgumentError LCAModel(1, n_items, n_categories)  # < 2 classes
        @test_throws ArgumentError LCAModel(2, n_items, [1, 2, 2, 2, 2])  # < 2 categories

        # Test identifiability conditions
        @test_logs (:warn, "Model may not be identifiable. With 3 classes and minimum of 2 categories, need ideally 5 items (got 2).") begin
            LCAModel(3, 2, [2, 2])
        end
    end

    @testset "Model Fitting" begin
        # Generate synthetic data with known structure
        n_samples = 100
        n_items = 5  # Sufficient for 2 classes with binary items

        # Create data matrix
        data = zeros(Int, n_samples, n_items)
        true_classes = rand(1:2, n_samples)

        for i in 1:n_samples
            for j in 1:n_items
                p = true_classes[i] == 1 ? 0.8 : 0.2
                data[i, j] = rand() < p ? 1 : 2
            end
        end

        # Fit model
        model = LCAModel(2, n_items, fill(2, n_items))
        ll = fit!(model, data)

        # Test warning for number of observations
        @test_logs (:warn, "Low number of observations (100) may affect model fitting. Consider using more data for better results.") begin
            fit!(model, data)
        end

        # Test results
        @test !isnan(ll)
        @test !isinf(ll)
        @test all(0 .<= model.class_probs .<= 1)
        @test isapprox(sum(model.class_probs), 1.0, atol=1e-10)
        for item_prob in model.item_probs
            @test all(0 .<= item_prob .<= 1)
            @test all(isapprox.(sum(item_prob, dims=2), 1.0, atol=1e-10))
        end
    end

    @testset "Model Diagnostics" begin
        # Prepare data
        n_items = 5  # Sufficient for 2 classes
        df = DataFrame(
            [Symbol("x$i") => repeat([1, 2], 50) for i in 1:n_items]...
        )
        data, n_cats = prepare_data(df, [Symbol("x$i") for i in 1:n_items]...)

        # Fit model
        model = LCAModel(2, n_items, n_cats)
        ll = fit!(model, data)

        # Calculate diagnostics
        diag = diagnostics!(model, data, ll)

        # Test diagnostic values
        @test !isnan(diag.ll)
        @test !isnan(diag.aic)
        @test !isnan(diag.bic)
        @test !isnan(diag.sbic)
        @test diag.bic > diag.aic  # BIC should be more conservative
    end

    @testset "Prediction" begin
        # Prepare data
        n_items = 5  # Sufficient for 2 classes
        n_samples = 100
        data = ones(Int, n_samples, n_items)
        for i in 1:n_samples
            data[i, :] .= rand(1:2, n_items)
        end

        # Fit model and predict
        model = LCAModel(2, n_items, fill(2, n_items))
        fit!(model, data)
        assignments, probs = predict(model, data)

        # Test predictions
        @test length(assignments) == n_samples
        @test size(probs) == (n_samples, 2)
        @test all(1 .<= assignments .<= 2)
        @test all(0 .<= probs .<= 1)
        @test all(isapprox.(sum(probs, dims=2), 1.0, atol=1e-10))

        # Test consistency
        for i in 1:n_samples
            @test assignments[i] == argmax(probs[i, :])
        end
    end

    @testset "Show Profiles" begin
        # Prepare data with sufficient items
        n_items = 5
        df = DataFrame(
            [Symbol("x$i") => repeat([1, 2], 50) for i in 1:n_items]...
        )
        data, n_cats = prepare_data(df, [Symbol("x$i") for i in 1:n_items]...)

        # Fit model
        model = LCAModel(2, n_items, n_cats)
        fit!(model, data)

        # Test basic display
        @test_nowarn show_profiles(model, df, [Symbol("x$i") for i in 1:n_items])

        # Test with custom names
        @test_nowarn show_profiles(model, df, [Symbol("x$i") for i in 1:n_items],
            var_names=["Var$i" for i in 1:n_items])

        # Test with custom labels
        custom_labels = [["No", "Yes"] for _ in 1:n_items]
        @test_nowarn show_profiles(model, df, [Symbol("x$i") for i in 1:n_items],
            var_labels=custom_labels)
    end
end