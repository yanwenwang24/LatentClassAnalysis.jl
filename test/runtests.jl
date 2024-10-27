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
        # Test with numeric data
        df_numeric = DataFrame(
            x1 = repeat([1, 2], 25),  # 50 observations
            x2 = repeat([1, 2], 25),
            x3 = repeat([2, 1], 25)
        )
        data_num, cats_num = prepare_data(df_numeric, :x1, :x2, :x3)
        @test size(data_num) == (50, 3)
        @test cats_num == [2, 2, 2]
        
        # Test with categorical data
        df_cat = DataFrame(
            x1 = categorical(repeat(["A", "B"], 25)),
            x2 = categorical(repeat(["X", "Y"], 25))
        )
        data_cat, cats_cat = prepare_data(df_cat, :x1, :x2)
        @test size(data_cat) == (50, 2)
        @test cats_cat == [2, 2]
        
        # Test with mixed data
        df_mixed = DataFrame(
            x1 = repeat([1, 2], 25),
            x2 = categorical(repeat(["A", "B", "C"], 17)[1:50])
        )
        data_mixed, cats_mixed = prepare_data(df_mixed, :x1, :x2)
        @test size(data_mixed) == (50, 2)
        @test cats_mixed == [2, 3]
        
        # Test error handling
        @test_throws ArgumentError prepare_data(DataFrame(), :nonexistent)
    end

    @testset "Model Initialization" begin
        # Test basic initialization
        model = LCAModel(2, 3, [2, 2, 2])
        @test model.n_classes == 2
        @test model.n_items == 3
        @test length(model.class_probs) == 2
        @test length(model.item_probs) == 3
        @test all(sum(model.class_probs) ≈ 1.0)
        
        # Test probability normalization
        for probs in model.item_probs
            @test all(sum(probs, dims=2) .≈ 1.0)
        end
        
        # Test input validation
        @test_throws ArgumentError LCAModel(0, 3, [2, 2, 2])  # Invalid number of classes
        @test_throws ArgumentError LCAModel(2, 3, [1, 2, 2])  # Invalid category count
    end

    @testset "Model Fitting" begin
        # Generate synthetic data
        n_samples = 100  # Increased sample size
        n_items = 2      # Reduced number of items for simpler model
        true_classes = rand(1:2, n_samples)
        data = zeros(Int, n_samples, n_items)
        
        # Generate responses based on true classes
        for i in 1:n_samples
            for j in 1:n_items
                data[i, j] = true_classes[i] == 1 ? rand() < 0.8 ? 1 : 2 : rand() < 0.3 ? 1 : 2
            end
        end
        
        # Test model fitting
        model = LCAModel(2, n_items, fill(2, n_items))
        ll = fit!(model, data)
        @test !isnan(ll)
        @test !isinf(ll)
        @test all(0 .<= model.class_probs .<= 1)
        @test sum(model.class_probs) ≈ 1.0
        
        # Test convergence with different parameters
        ll_verbose = fit!(model, data, verbose=true)
        ll_high_tol = fit!(model, data, tol=1e-3)
        ll_low_iter = fit!(model, data, max_iter=10)
        @test !isnan(ll_verbose)
        @test !isnan(ll_high_tol)
        @test !isnan(ll_low_iter)
    end

    @testset "Model Diagnostics" begin
        # Generate test data with sufficient sample size
        n_samples = 100
        data = rand(1:2, n_samples, 2)  # Simple 2-class, 2-item model
        model = LCAModel(2, 2, [2, 2])
        ll = fit!(model, data)
        
        # Calculate diagnostics
        diag = diagnostics!(model, data, ll)
        
        # Test diagnostic calculations
        @test diag.ll ≈ ll
        @test !isnan(diag.aic)
        @test !isnan(diag.bic)
        @test !isnan(diag.sbic)
        @test 0 <= diag.entropy <= 1
        
        # Test relative magnitude
        @test diag.bic > diag.aic  # BIC penalizes complexity more than AIC
    end

    @testset "Prediction" begin
        # Generate test data
        n_test = 100
        n_items = 2  # Simplified model
        data = rand(1:2, n_test, n_items)
        
        # Fit model and make predictions
        model = LCAModel(2, n_items, fill(2, n_items))
        fit!(model, data)
        assignments, probs = predict(model, data)
        
        # Test predictions
        @test length(assignments) == n_test
        @test size(probs) == (n_test, 2)
        @test all(1 .<= assignments .<= 2)
        @test all(0 .<= probs .<= 1)
        @test all(sum(probs, dims=2) .≈ 1.0)
        
        # Test consistency between assignments and probabilities
        for i in 1:n_test
            @test assignments[i] == argmax(probs[i, :])
        end
        
        # Test prediction with new data
        new_data = rand(1:2, 10, n_items)
        new_assignments, new_probs = predict(model, new_data)
        @test length(new_assignments) == 10
        @test size(new_probs) == (10, 2)
    end

    @testset "Edge Cases" begin
        # Test with minimum size dataset
        min_data = [1 1]
        min_model = LCAModel(2, 2, [2, 2])
        @test_throws ArgumentError fit!(min_model, min_data)
        
        # Test with perfect separation
        # Need sufficient sample size
        perfect_data = repeat([1 1; 1 1; 2 2; 2 2], 10)  # 40 samples
        perfect_model = LCAModel(2, 2, [2, 2])
        ll_perfect = fit!(perfect_model, perfect_data)
        @test !isnan(ll_perfect)
        
        # Test with all same responses
        same_data = fill(1, 40, 2)  # 40 samples
        same_model = LCAModel(2, 2, [2, 2])
        ll_same = fit!(same_model, same_data)
        @test !isnan(ll_same)
    end
end