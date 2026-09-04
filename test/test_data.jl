using Test
using LatentClassAnalysis
using CategoricalArrays
using DataFrames

@testset "Data preparation" begin
    @testset "Tables.jl sources" begin
        df = DataFrame(a=[1, 2, 2, 1], b=["no", "yes", "yes", "no"], c=[1, 3, 3, 1])
        d_df = prepare_data(df, [:a, :b, :c])
        @test d_df isa LCAData
        @test d_df.y == [1 1 1; 2 2 2; 2 2 2; 1 1 1]
        @test d_df.n_categories == [2, 2, 2]
        @test d_df.item_names == [:a, :b, :c]
        @test d_df.item_levels == [["1", "2"], ["no", "yes"], ["1", "3"]]
        @test nobs(d_df) == 4
        @test size(d_df) == (4, 3)
        @test size(d_df, 2) == 3
        @test !hasmissing(d_df)
        @test nmissing(d_df) == [0, 0, 0]
        @test !hascovariates(d_df)
        @test d_df.X == ones(4, 1)
        @test d_df.covariate_names == [:intercept]

        # NamedTuple of vectors
        nt = (a=[1, 2, 2, 1], b=["no", "yes", "yes", "no"], c=[1, 3, 3, 1])
        d_nt = prepare_data(nt, [:a, :b, :c])
        @test d_nt.y == d_df.y
        @test d_nt.item_levels == d_df.item_levels

        # Vector of NamedTuples (row table)
        rows = [(a=1, b="no", c=1), (a=2, b="yes", c=3), (a=2, b="yes", c=3), (a=1, b="no", c=1)]
        d_rows = prepare_data(rows, [:a, :b, :c])
        @test d_rows.y == d_df.y

        # String column names
        d_str = prepare_data(df, ["a", "b"])
        @test d_str.item_names == [:a, :b]
        @test d_str.y == d_df.y[:, 1:2]

        # Column subset and order follow `items`
        d_rev = prepare_data(df, [:c, :a])
        @test d_rev.y == d_df.y[:, [3, 1]]
        @test d_rev.item_names == [:c, :a]
    end

    @testset "Errors" begin
        df = DataFrame(a=[1, 2, 2, 1], b=["no", "yes", "yes", "no"])
        @test_throws ArgumentError prepare_data([1 2; 2 1], [:a])          # not a table
        @test_throws ArgumentError prepare_data(df, [:a, :zzz])              # unknown column
        @test_throws "available columns: a, b" prepare_data(df, [:zzz])
        @test_throws ArgumentError prepare_data(df, Symbol[])                # no items
        @test_throws ArgumentError prepare_data(df, [:a, :a])                # duplicate
        @test_throws ArgumentError prepare_data(DataFrame(a=[1, 1, 1]), [:a])  # one level
        @test_throws ArgumentError prepare_data(DataFrame(a=[1, 2], b=[1, 2]), [:a]; covariates=[:zzz])
    end

    @testset "Dense recoding of integer columns" begin
        recode(v) = vec(prepare_data((x=v,), [:x]).y)
        n_cats(v) = prepare_data((x=v,), [:x]).n_categories[1]
        labels(v) = prepare_data((x=v,), [:x]).item_levels[1]

        @test recode([1, 2]) == [1, 2]
        @test recode([1, 3]) == [1, 2]
        @test recode([2, 3]) == [1, 2]
        @test recode([-1, 1]) == [1, 2]
        @test recode([0, 2]) == [1, 2]
        @test recode([0, 1]) == [1, 2]
        @test recode([2, 1]) == [2, 1]                     # sorted values, not order of appearance
        @test recode([30, 10, 20, 10]) == [3, 1, 2, 1]
        @test recode([5, 5, 5, 7]) == [1, 1, 1, 2]
        @test n_cats([1, 3, 1, 3]) == 2
        @test n_cats([0, 5, 10]) == 3
        @test labels([0, 5, 10]) == ["0", "5", "10"]
        @test recode(Int8[0, 1, 1]) == [1, 2, 2]
        @test recode(UInt8[3, 9, 3]) == [1, 2, 1]
    end

    @testset "Bool, String and integer-valued Float64 columns" begin
        d_bool = prepare_data((b=[true, false, true, true],), [:b])
        @test vec(d_bool.y) == [2, 1, 2, 2]
        @test d_bool.item_levels[1] == ["false", "true"]

        d_str = prepare_data((s=["yes", "no", "yes", "maybe"],), [:s])
        @test vec(d_str.y) == [3, 2, 3, 1]                  # maybe < no < yes
        @test d_str.item_levels[1] == ["maybe", "no", "yes"]

        d_flt = prepare_data((f=[1.0, 2.0, 2.0, 3.0],), [:f])
        @test vec(d_flt.y) == [1, 2, 2, 3]
        @test d_flt.item_levels[1] == ["1", "2", "3"]

        d_frac = prepare_data((f=[0.5, 1.5, 0.5],), [:f])
        @test d_frac.item_levels[1] == ["0.5", "1.5"]
    end

    @testset "Categorical columns" begin
        # Level order is kept; unused levels are dropped by default
        c = categorical(["a", "b", "a"]; levels=["b", "a", "c"])
        d = prepare_data((c=c,), [:c])
        @test vec(d.y) == [2, 1, 2]
        @test d.n_categories == [2]
        @test d.item_levels[1] == ["b", "a"]

        # ... or kept on request
        d_keep = prepare_data((c=c,), [:c]; drop_unused_levels=false)
        @test vec(d_keep.y) == [2, 1, 2]
        @test d_keep.n_categories == [3]
        @test d_keep.item_levels[1] == ["b", "a", "c"]

        # Ordered categorical and categorical of integers
        o = categorical(["low", "high", "mid", "low"]; levels=["low", "mid", "high"], ordered=true)
        @test vec(prepare_data((o=o,), [:o]).y) == [1, 3, 2, 1]
        ci = categorical([10, 30, 30, 10])
        dci = prepare_data((c=ci,), [:c])
        @test vec(dci.y) == [1, 2, 2, 1]
        @test dci.item_levels[1] == ["10", "30"]

        # Categorical with missing
        cm = categorical(["a", missing, "b"])
        dcm = @test_logs (:warn, r"1 row\(s\) have all 1 indicators missing") prepare_data((c=cm,), [:c])
        @test vec(dcm.y) == [1, 0, 2]
        @test hasmissing(dcm)
    end

    @testset "levels override" begin
        x = ["no", "yes", "yes", "no"]
        d = prepare_data((x=x,), [:x]; levels=Dict(:x => ["yes", "no"]))
        @test vec(d.y) == [2, 1, 1, 2]
        @test d.item_levels[1] == ["yes", "no"]

        # String keys work too; an unused supplied level is dropped by default ...
        d2 = prepare_data((x=x,), [:x]; levels=Dict("x" => ["yes", "maybe", "no"]))
        @test d2.item_levels[1] == ["yes", "no"]
        @test vec(d2.y) == [2, 1, 1, 2]
        # ... and kept with drop_unused_levels=false
        d3 = prepare_data((x=x,), [:x]; levels=Dict(:x => ["yes", "maybe", "no"]), drop_unused_levels=false)
        @test d3.item_levels[1] == ["yes", "maybe", "no"]
        @test vec(d3.y) == [3, 1, 1, 3]
        @test d3.n_categories == [3]

        # Values are matched to the supplied levels by their string form
        d4 = prepare_data((x=[0, 1, 1],), [:x]; levels=Dict(:x => ["1", "0"]))
        @test vec(d4.y) == [2, 1, 1]
        d5 = prepare_data((x=[true, false],), [:x]; levels=Dict(:x => [true, false]))
        @test vec(d5.y) == [1, 2]

        # A value outside the supplied levels is an error; items without an entry use the data
        @test_throws ArgumentError prepare_data((x=x,), [:x]; levels=Dict(:x => ["yes"]))
        @test_throws ArgumentError prepare_data((x=x,), [:x]; levels=Dict(:x => ["yes", "yes"]))
        d6 = prepare_data((x=x, z=[1, 2, 1, 2]), [:x, :z]; levels=Dict(:x => ["yes", "no"]))
        @test d6.item_levels == [["yes", "no"], ["1", "2"]]
    end

    @testset "Missing responses" begin
        tbl = (a=[1, missing, 2, 2], b=["x", "y", missing, "y"], c=[1, 2, 1, 2])
        d = prepare_data(tbl, [:a, :b, :c])
        @test d.y == [1 1 1; 0 2 2; 2 0 1; 2 2 2]
        @test hasmissing(d)
        @test nmissing(d) == [1, 1, 0]
        @test d.n_categories == [2, 2, 2]

        # Rows with every indicator missing are kept and counted in a warning
        tbl2 = (a=[1, missing, 2, missing], b=[1, missing, 2, missing])
        d2 = @test_logs (:warn, r"^2 row\(s\) have all 2 indicators missing") prepare_data(tbl2, [:a, :b])
        @test nobs(d2) == 4
        @test d2.y[2, :] == [0, 0]
        @test_logs prepare_data(tbl, [:a, :b, :c])   # no warning otherwise
    end

    @testset "Covariates" begin
        tbl = (a=[1, 2, 2, 1], b=[1, 1, 2, 2], age=[30.0, 41.5, 25.0, 60.0],
               female=[true, false, false, true], name=["p", "q", "r", "s"],
               agemiss=[30.0, missing, 25.0, 60.0])
        d = prepare_data(tbl, [:a, :b]; covariates=[:age, :female])
        @test hascovariates(d)
        @test d.covariate_names == [:intercept, :age, :female]
        @test d.X == [1.0 30.0 1.0; 1.0 41.5 0.0; 1.0 25.0 0.0; 1.0 60.0 1.0]
        @test d.X isa Matrix{Float64}
        @test d.y == [1 1; 2 1; 2 2; 1 2]

        # String covariate names
        d2 = prepare_data(tbl, [:a, :b]; covariates=["age"])
        @test d2.covariate_names == [:intercept, :age]

        @test_throws ArgumentError prepare_data(tbl, [:a, :b]; covariates=[:agemiss])   # missing
        @test_throws "drop rows with missing covariates" prepare_data(tbl, [:a, :b]; covariates=[:agemiss])
        @test_throws ArgumentError prepare_data(tbl, [:a, :b]; covariates=[:name])      # string
        @test_throws ArgumentError prepare_data(tbl, [:a, :b]; covariates=[:age, :age]) # duplicate
    end

    @testset "LCAData from a matrix" begin
        y = [1 2 1; 2 2 missing; 1 1 1; 2 1 2]
        d = LCAData(y)
        @test d.y == [1 2 1; 2 2 0; 1 1 1; 2 1 2]
        @test d.y isa Matrix{Int}
        @test d.n_categories == [2, 2, 2]
        @test d.item_names == [:item1, :item2, :item3]
        @test d.item_levels == [["1", "2"], ["1", "2"], ["1", "2"]]
        @test hasmissing(d)
        @test nmissing(d) == [0, 0, 1]
        @test !hascovariates(d)

        # Explicit metadata; codes may be given as 0 directly; other integer types
        d2 = LCAData(Int8[1 2; 2 0; 1 3]; n_categories=[2, 3], item_names=["u", "v"],
                     item_levels=[["a", "b"], ["x", "y", "z"]])
        @test d2.y == [1 2; 2 0; 1 3]
        @test d2.item_names == [:u, :v]
        @test d2.item_levels == [["a", "b"], ["x", "y", "z"]]

        # n_categories larger than what the data shows
        d3 = LCAData([1 1; 2 1]; n_categories=[3, 2])
        @test d3.n_categories == [3, 2]
        @test d3.item_levels[1] == ["1", "2", "3"]

        # Covariates: matrix or vector, with or without names
        d4 = LCAData([1 2; 2 1; 1 1]; covariates=[1.0 0.0; 2.0 1.0; 3.0 0.0], covariate_names=[:age, :female])
        @test d4.X == [1.0 1.0 0.0; 1.0 2.0 1.0; 1.0 3.0 0.0]
        @test d4.covariate_names == [:intercept, :age, :female]
        @test hascovariates(d4)
        d5 = LCAData([1 2; 2 1; 1 1]; covariates=[true, false, true])
        @test d5.X == [1.0 1.0; 1.0 0.0; 1.0 1.0]
        @test d5.covariate_names == [:intercept, :x1]

        # Validation errors
        @test LCAData([1 2; 2 3]).n_categories == [2, 3]                                # inferred per column
        @test_throws ArgumentError LCAData([1 2; 2 3]; n_categories=[2, 2])              # code above C_j
        @test_throws ArgumentError LCAData([1 -1; 2 2])                                  # negative code
        @test_throws ArgumentError LCAData([1 1; 1 1])                                   # one category
        @test_throws ArgumentError LCAData([1 missing; 2 missing])                       # all-missing column
        @test_throws ArgumentError LCAData(Matrix{Int}(undef, 3, 0))                     # no items
        @test_throws ArgumentError LCAData([1 2; 2 1]; n_categories=[2])                 # length mismatch
        @test_throws ArgumentError LCAData([1 2; 2 1]; n_categories=[1, 2])              # C < 2
        @test_throws ArgumentError LCAData([1 2; 2 1]; item_names=[:a])                  # names length
        @test_throws ArgumentError LCAData([1 2; 2 1]; item_names=[:a, :a])              # duplicate names
        @test_throws ArgumentError LCAData([1 2; 2 1]; item_levels=[["a", "b"]])         # levels length
        @test_throws ArgumentError LCAData([1 2; 2 1]; item_levels=[["a"], ["a", "b"]])  # label count
        @test_throws ArgumentError LCAData([1 2; 2 1]; covariates=[1.0, 2.0, 3.0])       # X rows
        @test_throws ArgumentError LCAData([1 2; 2 1]; covariates=[1.0, NaN])            # NaN
        @test_throws ArgumentError LCAData([1 2; 2 1]; covariates=[1.0, Inf])            # Inf
        @test_throws ArgumentError LCAData([1 2; 2 1]; covariates=[1.0, 2.0], covariate_names=[:a, :b])
        @test_throws ArgumentError LCAData([1 2; 2 1]; covariates=[1.0 1.0; 2.0 2.0], covariate_names=[:a, :a])

        # Direct (positional) constructor validation
        @test_throws ArgumentError LCAData([1 2; 2 1], [2, 2], [:a, :b], [["1", "2"], ["1", "2"]],
                                           [2.0 0.0; 1.0 0.0], [:intercept, :x])           # no intercept
        @test_throws ArgumentError LCAData([1 2; 2 1], [2, 2], [:a, :b], [["1", "2"], ["1", "2"]],
                                           ones(2, 2), [:x, :y])                          # first name
        @test_throws ArgumentError LCAData([1 2; 2 1], [2, 2], [:a, :b], [["1", "2"], ["1", "2"]],
                                           ones(2, 2), [:intercept])                      # names length
    end
end

@testset "LCAData: 0/1-coded matrices are caught" begin
    y01 = [0 1; 1 0; 1 1; 0 1; 1 1]
    @test_throws ArgumentError LCAData(y01)
    @test_throws "0/1-coded" LCAData(y01)
    logs, d = Test.collect_test_logs(() -> LCAData(y01; n_categories=[2, 2]))
    @test count(l -> occursin("contains only the codes 0 and 1", l.message), logs) == 2
    @test nmissing(d) == [2, 1]
    # a genuine 1-based column with missing values does not warn
    logs2, _ = Test.collect_test_logs(() -> LCAData([0 1; 2 2; 1 2]; n_categories=[2, 2]))
    @test !any(l -> occursin("only the codes 0 and 1", l.message), logs2)
end

@testset "levels for unknown items and names without covariates are rejected" begin
    tbl = (a=[1, 2, 2, 1], b=[1, 1, 2, 2])
    @test_throws ArgumentError prepare_data(tbl, [:a, :b]; levels=Dict(:zzz => [1, 2]))
    @test_throws "not among the items" prepare_data(tbl, [:a, :b]; levels=Dict("zzz" => [1, 2]))
    @test prepare_data(tbl, [:a]; levels=Dict(:a => [2, 1])).item_levels == [["2", "1"]]
    @test_throws ArgumentError LCAData([1 2; 2 1]; covariate_names=[:x])
    # An all-missing column is not mistaken for 0/1 coding
    @test_logs LCAData([1 0; 2 0; 1 0]; n_categories=[2, 2])
end
