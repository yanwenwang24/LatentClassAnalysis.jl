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
            x1=repeat([1, 2], 25),  # 50 observations
            x2=repeat([1, 2], 25),
            x3=repeat([2, 1], 25)
        )
        data_num, cats_num = prepare_data(df_numeric, :x1, :x2, :x3)
        @test size(data_num) == (50, 3)
        @test cats_num == [2, 2, 2]

        # Test with categorical data
        df_cat = DataFrame(
            x1=categorical(repeat(["A", "B"], 25)),
            x2=categorical(repeat(["X", "Y"], 25))
        )
        data_cat, cats_cat = prepare_data(df_cat, :x1, :x2)
        @test size(data_cat) == (50, 2)
        @test cats_cat == [2, 2]

        # Test with mixed data
        df_mixed = DataFrame(
            x1=repeat([1, 2], 25),
            x2=categorical(repeat(["A", "B", "C"], 17)[1:50])
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
    @testset "Show Profiles" begin
        # Create test data with sufficient observations
        df = DataFrame(
            x=repeat([1, 2], 50),             # 100 observations, binary
            y=categorical(repeat(["A", "B"], 50)), # 100 observations, categorical
            z=repeat([0, 1], 50)              # 100 observations, binary
        )

        # Prepare data and model
        data, n_categories = prepare_data(df, :x, :y, :z)
        model = LCAModel(2, size(data, 2), n_categories)
        ll = fit!(model, data)

        # Basic functionality test
        @test_nowarn show_profiles(model, df, [:x, :y, :z])

        # Test custom variable names
        @test_nowarn show_profiles(model, df, [:x, :y, :z],
            var_names=["Variable 1", "Variable 2", "Variable 3"])

        # Test custom labels
        @test_nowarn show_profiles(model, df, [:x, :y, :z],
            var_labels=[["Low", "High"],
                ["Group A", "Group B"],
                ["No", "Yes"]])
    end

    @testset "Binary Variable Coding" begin
        # Create test data with different binary codings
        df_binary = DataFrame(
            var01=repeat([0, 1], 50),         # 0/1 coding
            var12=repeat([1, 2], 50),         # 1/2 coding
            var_cat=categorical(repeat(["No", "Yes"], 50)),  # categorical binary
            var_mix01=repeat([0, 1], 50),     # another 0/1 for testing multiple
        )

        @testset "Prepare Data Detection" begin
            # Test automatic detection
            data, n_categories = prepare_data(df_binary, :var01)
            @test minimum(data) == 1  # Should be shifted to 1-based
            @test maximum(data) == 2
            @test n_categories == [2]

            # Test with 1/2 coded variable
            data, n_categories = prepare_data(df_binary, :var12)
            @test minimum(data) == 1  # Should remain 1-based
            @test maximum(data) == 2
            @test n_categories == [2]

            # Test with categorical
            data, n_categories = prepare_data(df_binary, :var_cat)
            @test minimum(data) == 1
            @test maximum(data) == 2
            @test n_categories == [2]

            # Test multiple columns
            data, n_categories = prepare_data(df_binary, :var01, :var12, :var_cat, :var_mix01)
            @test size(data, 2) == 4
            @test all(n_categories .== 2)
            @test all(x -> x in (1, 2), data)  # All values should be 1 or 2
        end

        @testset "Model Fitting with Binary Data" begin
            # Prepare mixed binary data
            data, n_categories = prepare_data(df_binary, :var01, :var12, :var_cat, :var_mix01)

            # Fit model
            model = LCAModel(2, size(data, 2), n_categories)
            ll = fit!(model, data)

            # Check model parameters
            @test !isnan(ll)
            @test size(model.item_probs[1]) == (2, 2)  # 2 classes, 2 categories
            @test all(0 .<= model.item_probs[1] .<= 1)  # Valid probabilities
        end

        @testset "Profile Display with Binary Data" begin
            # Prepare data
            data, n_categories = prepare_data(df_binary, :var01, :var12)
            model = LCAModel(2, size(data, 2), n_categories)
            ll = fit!(model, data)

            # Test default display
            @test_nowarn show_profiles(model, df_binary, [:var01, :var12])

            # Test with custom labels
            @test_nowarn show_profiles(model, df_binary, [:var01, :var12],
                var_labels=[["No", "Yes"], ["Low", "High"]])

            # Test with custom names
            @test_nowarn show_profiles(model, df_binary, [:var01, :var12],
                var_names=["Binary 0/1", "Binary 1/2"])
        end

        @testset "Mixed Data Types" begin
            # Create data with binary and multi-category variables
            df_mixed = DataFrame(
                bin01=repeat([0, 1], 50),     # binary 0/1
                bin12=repeat([1, 2], 50),     # binary 1/2
                cat3=repeat(1:3, 34)[1:100],  # 3 categories
                cat_bin=categorical(repeat(["No", "Yes"], 50))  # categorical binary
            )

            # Test data preparation
            data, n_categories = prepare_data(df_mixed, :bin01, :bin12, :cat3, :cat_bin)
            @test length(n_categories) == 4
            @test n_categories == [2, 2, 3, 2]
            @test size(data, 2) == 4
            @test all(1 .<= data[:, 1] .<= 2)  # Binary 0/1 converted to 1/2
            @test all(1 .<= data[:, 2] .<= 2)  # Binary 1/2 unchanged
            @test all(1 .<= data[:, 3] .<= 3)  # Three categories
            @test all(1 .<= data[:, 4] .<= 2)  # Categorical binary

            # Fit model and test profiles
            model = LCAModel(2, size(data, 2), n_categories)
            ll = fit!(model, data)

            # Test profile display with mixed data
            @test_nowarn show_profiles(model, df_mixed,
                [:bin01, :bin12, :cat3, :cat_bin],
                var_names=["Binary 0/1", "Binary 1/2",
                    "Three Cats", "Cat Binary"],
                var_labels=[
                    ["No", "Yes"],
                    ["Low", "High"],
                    ["Low", "Medium", "High"],
                    ["No", "Yes"]
                ])
        end

        @testset "Edge Cases" begin
            # Single binary column
            df_single = DataFrame(bin=repeat([0, 1], 50))
            data, n_cats = prepare_data(df_single, :bin)
            @test n_cats == [2]
            @test all(x -> x in (1, 2), data)

            # All zeros
            df_zeros = DataFrame(zeros=zeros(Int, 100))
            data, n_cats = prepare_data(df_zeros, :zeros)
            @test n_cats == [1]  # Should detect as single category

            # All ones
            df_ones = DataFrame(ones=ones(Int, 100))
            data, n_cats = prepare_data(df_ones, :ones)
            @test n_cats == [1]  # Should detect as single category
        end
    end
end