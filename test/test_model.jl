using Test
using LatentClassAnalysis
using Random

@testset "Model Initialization" begin
    @testset "Valid model" begin
        n_items = 5
        n_classes = 2
        n_categories = fill(2, n_items)
        model = LCAModel(n_classes, n_items, n_categories)

        @test model.n_classes == 2
        @test model.n_items == 5
        @test model.n_categories == n_categories
        @test length(model.class_probs) == 2
        @test length(model.item_probs) == 5
        @test all(model.class_probs .== 1 / n_classes)  # uniform start
        @test isapprox(sum(model.class_probs), 1.0)
        for (j, P) in enumerate(model.item_probs)
            @test size(P) == (n_classes, n_categories[j])
            @test all(0 .<= P .<= 1)
            @test all(isapprox.(sum(P, dims=2), 1.0; atol=1e-12))
        end
    end

    @testset "Mixed category counts" begin
        model = LCAModel(2, 3, [2, 3, 4])
        @test model.n_categories == [2, 3, 4]
        @test [size(P) for P in model.item_probs] == [(2, 2), (2, 3), (2, 4)]
        for P in model.item_probs
            @test all(isapprox.(sum(P, dims=2), 1.0; atol=1e-12))
        end
    end

    @testset "Invalid inputs" begin
        @test_throws ArgumentError LCAModel(1, 5, fill(2, 5))        # < 2 classes
        @test_throws ArgumentError LCAModel(0, 5, fill(2, 5))
        @test_throws ArgumentError LCAModel(-2, 5, fill(2, 5))
        @test_throws ArgumentError LCAModel(2, 0, Int[])             # < 1 item
        @test_throws ArgumentError LCAModel(2, 5, [1, 2, 2, 2, 2])   # < 2 categories
        @test_throws ArgumentError LCAModel(2, 5, [2, 2, 0, 2, 2])
        @test_throws ArgumentError LCAModel(2, 5, [2, 2, 2])         # length mismatch
        @test_throws ArgumentError LCAModel(2, 2, [2, 2, 2])
        @test_throws ArgumentError LCAModel(2, 3, 1:3)               # range containing a 1

        # Error messages name the offending argument
        @test_throws "Number of classes must be ≥ 2, got 1" LCAModel(1, 5, fill(2, 5))
        @test_throws "Number of items must be ≥ 1, got 0" LCAModel(2, 0, Int[])
        @test_throws "Length of n_categories (3) must match n_items (5)" LCAModel(2, 5, [2, 2, 2])
        @test_throws "item 3 has 0" LCAModel(2, 5, [2, 2, 0, 2, 2])
    end

    @testset "Integer-typed and range inputs" begin
        model = LCAModel(Int32(2), Int8(5), Int32[2, 2, 2, 2, 2])
        @test model.n_classes isa Int
        @test model.n_items isa Int
        @test model.n_classes == 2
        @test model.n_items == 5
        @test model.n_categories isa Vector{Int}
        @test model.n_categories == [2, 2, 2, 2, 2]
        @test model.class_probs isa Vector{Float64}
        @test model.item_probs isa Vector{Matrix{Float64}}
        @test all(size(P) == (2, 2) for P in model.item_probs)

        model_u = LCAModel(UInt8(2), UInt16(3), UInt8[2, 3, 2])
        @test model_u.n_classes == 2
        @test model_u.n_items == 3
        @test model_u.n_categories == [2, 3, 2]

        # A range of category counts (every element must be ≥ 2)
        model_range = LCAModel(2, 5, 2:2:10)
        @test model_range.n_categories isa Vector{Int}
        @test model_range.n_categories == [2, 4, 6, 8, 10]
        @test [size(P) for P in model_range.item_probs] == [(2, c) for c in 2:2:10]
        for P in model_range.item_probs
            @test all(isapprox.(sum(P, dims=2), 1.0; atol=1e-12))
        end
    end

    @testset "Identifiability warning" begin
        @test_logs (:warn, "Model may not be identifiable. With 3 classes and minimum of 2 categories, need ideally 5 items (got 2).") begin
            LCAModel(3, 2, [2, 2])
        end
        @test_logs (:warn, "Model may not be identifiable. With 2 classes and minimum of 2 categories, need ideally 3 items (got 2).") begin
            LCAModel(2, 2, [2, 2])
        end
        # The item with the fewest categories determines the requirement
        @test_logs (:warn, "Model may not be identifiable. With 3 classes and minimum of 2 categories, need ideally 5 items (got 3).") begin
            LCAModel(3, 3, [2, 5, 5])
        end
        # The warning does not prevent construction
        model = @test_logs (:warn, r"^Model may not be identifiable") LCAModel(3, 2, [2, 2])
        @test model isa LCAModel
        @test model.n_classes == 3

        # No warning when there are enough items
        @test_logs LCAModel(2, 5, [2, 2, 2, 2, 2])
        @test_logs LCAModel(2, 3, [2, 2, 2])
        @test_logs LCAModel(3, 5, fill(2, 5))
        @test_logs LCAModel(3, 3, [3, 3, 3])
        @test_logs LCAModel(2, 5, 2:2:10)
        @test_logs LCAModel(Int32(2), Int8(5), Int32[2, 2, 2, 2, 2])
    end

    @testset "check_identifiability" begin
        ci = LatentClassAnalysis.check_identifiability
        @test (@test_logs ci(5, 2, fill(2, 5))) === true
        @test (@test_logs ci(Int32(5), Int8(2), Int32[2, 2, 2, 2, 2])) === true
        @test (@test_logs ci(5, 2, 2:2:10)) === true
        @test (@test_logs (:warn, r"need ideally 5 items \(got 2\)") ci(2, 3, [2, 2])) === true
    end

    @testset "Reproducible initialization" begin
        Random.seed!(42)
        m1 = LCAModel(2, 5, fill(2, 5))
        Random.seed!(42)
        m2 = LCAModel(2, 5, fill(2, 5))
        @test m1.item_probs == m2.item_probs
        @test m1.class_probs == m2.class_probs

        Random.seed!(43)
        m3 = LCAModel(2, 5, fill(2, 5))
        @test m1.item_probs != m3.item_probs
    end

    @testset "Fields are mutable" begin
        model = LCAModel(2, 3, [2, 2, 2])
        model.class_probs .= [0.3, 0.7]
        @test model.class_probs == [0.3, 0.7]
        model.item_probs[1] .= [0.9 0.1; 0.2 0.8]
        @test model.item_probs[1] == [0.9 0.1; 0.2 0.8]
    end
end
