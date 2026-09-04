using Test
using LatentClassAnalysis
using CategoricalArrays
using DataFrames

@testset "Data Preparation" begin
    @testset "Basic preparation" begin
        df = DataFrame(
            x1=repeat([1, 2], 50),                               # binary 1/2
            x2=repeat([0, 1], 50),                               # binary 0/1
            x3=categorical(repeat(["A", "B", "C"], 34)[1:100]),  # 3 categories
            x4=repeat([1, 2], 50),                               # binary 1/2
            x5=repeat([1, 2], 50)                                # binary 1/2
        )

        data, n_cats = prepare_data(df, :x1, :x2, :x3, :x4, :x5)
        @test data isa Matrix{Int}
        @test n_cats isa Vector{Int}
        @test size(data) == (100, 5)
        @test n_cats == [2, 2, 3, 2, 2]
        @test all(x -> x ≥ 1, data)  # All values are 1-based
        for j in 1:5
            # Every code between 1 and n_cats[j] is used
            @test sort(unique(data[:, j])) == 1:n_cats[j]
        end
        @test data[:, 1] == data[:, 2] == data[:, 4] == data[:, 5]
        @test data[1:6, 3] == [1, 2, 3, 1, 2, 3]

        # Single column
        data_single, n_cats_single = prepare_data(df, :x1)
        @test size(data_single) == (100, 1)
        @test n_cats_single == [2]
        @test data_single[:, 1] == data[:, 1]
    end

    @testset "Dense recoding of integer columns" begin
        recode(v) = vec(prepare_data(DataFrame(x=v), :x)[1])
        n_cats(v) = prepare_data(DataFrame(x=v), :x)[2][1]

        @test recode([1, 2]) == [1, 2]      # unchanged
        @test recode([1, 3]) == [1, 2]
        @test recode([2, 3]) == [1, 2]
        @test recode([-1, 1]) == [1, 2]
        @test recode([0, 2]) == [1, 2]
        @test recode([0, 1]) == [1, 2]
        # Codes follow the sorted values, not the order of appearance
        @test recode([2, 1]) == [2, 1]
        @test recode([30, 10, 20, 10]) == [3, 1, 2, 1]
        @test recode([5, 5, 5, 7]) == [1, 1, 1, 2]
        @test recode([-3, 0, 3, 0, -3]) == [1, 2, 3, 2, 1]

        # The number of categories is the number of distinct values, not the largest code
        @test n_cats([1, 3, 1, 3]) == 2
        @test n_cats([0, 5, 10]) == 3
        @test n_cats([7, 7, 7, 7]) == 1

        # Non-Int integer columns are recoded too
        @test recode(Int8[0, 1, 1]) == [1, 2, 2]
        @test recode(UInt8[3, 9, 3]) == [1, 2, 1]
    end

    @testset "Bool column" begin
        df = DataFrame(b=[true, false, true, true])
        data, n_cats = prepare_data(df, :b)
        @test data isa Matrix{Int}
        @test vec(data) == [2, 1, 2, 2]  # false -> 1, true -> 2
        @test n_cats == [2]
    end

    @testset "String column" begin
        df = DataFrame(s=["yes", "no", "yes", "maybe"])
        data, n_cats = prepare_data(df, :s)
        # Sorted distinct values: maybe < no < yes
        @test vec(data) == [3, 2, 3, 1]
        @test n_cats == [3]
    end

    @testset "Categorical column" begin
        # An unused level must not count as a category
        df = DataFrame(c=categorical(["a", "b", "a"]; levels=["a", "b", "c"]))
        data, n_cats = prepare_data(df, :c)
        @test vec(data) == [1, 2, 1]
        @test n_cats == [2]

        # Codes follow the level order, not the lexical order
        df2 = DataFrame(c=categorical(["a", "b", "a"]; levels=["b", "a"]))
        data2, n_cats2 = prepare_data(df2, :c)
        @test vec(data2) == [2, 1, 2]
        @test n_cats2 == [2]

        # Ordered categorical
        df3 = DataFrame(c=categorical(["low", "high", "mid", "low"];
            levels=["low", "mid", "high"], ordered=true))
        data3, n_cats3 = prepare_data(df3, :c)
        @test vec(data3) == [1, 3, 2, 1]
        @test n_cats3 == [3]

        # Categorical made from integers
        df4 = DataFrame(c=categorical([10, 30, 30, 10]))
        data4, n_cats4 = prepare_data(df4, :c)
        @test vec(data4) == [1, 2, 2, 1]
        @test n_cats4 == [2]
    end

    @testset "Mixed column types encode the same pattern identically" begin
        df = DataFrame(
            i=[1, 3, 3, 1],
            b=[false, true, true, false],
            s=["no", "yes", "yes", "no"],
            c=categorical(["x", "y", "y", "x"]; levels=["x", "y", "z"])
        )
        data, n_cats = prepare_data(df, :i, :b, :s, :c)
        @test n_cats == [2, 2, 2, 2]
        @test data[:, 1] == [1, 2, 2, 1]
        @test all(data[:, j] == data[:, 1] for j in 2:4)
    end

    @testset "zero_based keyword" begin
        df = DataFrame(x=[0, 1, 0, 1], y=[1, 2, 1, 2])

        # Wrong length still throws
        @test_throws ArgumentError prepare_data(df, :x, :y; zero_based=[true])
        @test_throws ArgumentError prepare_data(df, :x; zero_based=[true, false])
        @test_throws ArgumentError prepare_data(df, :x, :y; zero_based=Bool[])

        # Right length is accepted and ignored: codes are always inferred from the data
        ref_data, ref_cats = prepare_data(df, :x, :y)
        @test ref_data == [1 1; 2 2; 1 1; 2 2]
        @test ref_cats == [2, 2]
        for zb in ([true, false], [false, true], [true, true], [false, false])
            data, n_cats = prepare_data(df, :x, :y; zero_based=zb)
            @test data == ref_data
            @test n_cats == ref_cats
        end
        data_nothing, cats_nothing = prepare_data(df, :x, :y; zero_based=nothing)
        @test data_nothing == ref_data
        @test cats_nothing == ref_cats
    end

    @testset "Column selection" begin
        df = DataFrame(a=[1, 2, 2], b=[10, 10, 20], unused=["p", "q", "r"])
        data_ab, cats_ab = prepare_data(df, :a, :b)
        data_ba, cats_ba = prepare_data(df, :b, :a)
        @test data_ab == data_ba[:, [2, 1]]
        @test cats_ab == cats_ba[[2, 1]]
        @test size(data_ab) == (3, 2)
        @test_throws ArgumentError prepare_data(df, :missing_column)
    end
end
