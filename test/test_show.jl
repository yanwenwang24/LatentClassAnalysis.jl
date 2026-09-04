using Test
using LatentClassAnalysis
using CategoricalArrays
using DataFrames
using Random
using StableRNGs

@isdefined(capture_stdout) || include(joinpath(@__DIR__, "testutils.jl"))

@testset "Show Profiles" begin
    # Simulate 5 binary items and store them with five different column types
    n_items = 5
    rng = StableRNG(31)
    raw, _ = simulate_lca(rng, 300, [0.6, 0.4], [[0.8 0.2; 0.2 0.8] for _ in 1:n_items])
    df = DataFrame(
        x1=raw[:, 1],                                                                # 1/2 coded
        x2=raw[:, 2] .- 1,                                                           # 0/1 coded
        x3=[c == 1 ? "no" : "yes" for c in raw[:, 3]],                               # strings
        x4=categorical([c == 1 ? "a" : "b" for c in raw[:, 4]]; levels=["a", "b", "c"]),  # unused level
        x5=raw[:, 5] .== 2,                                                          # Bool
    )
    cols = [Symbol("x$i") for i in 1:n_items]
    data, n_cats = prepare_data(df, cols...)
    @test data == raw
    @test n_cats == fill(2, n_items)

    Random.seed!(9)
    model = LCAModel(2, n_items, n_cats)
    fit!(model, data)

    @testset "Basic display" begin
        out = @test_logs capture_stdout(() -> show_profiles(model, df, cols))
        @test occursin("Latent Class Profiles", out)
        @test occursin("Class Sizes:", out)
        @test occursin("Class 1", out)
        @test occursin("Class 2", out)
        for c in cols
            @test occursin("\n$c:\n", out)
        end

        # Default category labels are the sorted distinct values of each column
        @test occursin(r"^1:"m, out) && occursin(r"^2:"m, out)          # x1
        @test occursin(r"^0:"m, out)                                     # x2
        @test occursin(r"^no:"m, out) && occursin(r"^yes:"m, out)        # x3
        @test occursin(r"^a:"m, out) && occursin(r"^b:"m, out)           # x4
        @test !occursin(r"^c:"m, out)                                    # unused level is not shown
        @test occursin(r"^false:"m, out) && occursin(r"^true:"m, out)    # x5

        # Three decimals by default; one percentage per class size and per class × category
        @test occursin(r"\d+\.\d{3}%", out)
        @test count("%", out) == model.n_classes + model.n_classes * sum(n_cats)

        # Output is deterministic for a fixed model
        @test capture_stdout(() -> show_profiles(model, df, cols)) == out
    end

    @testset "Printed values match the model" begin
        out = capture_stdout(() -> show_profiles(model, df, cols))

        # Class sizes (printed with one decimal) sum to 100% and match class_probs
        sizes = [parse(Float64, m.captures[2]) for m in eachmatch(r"Class (\d+): (\d+\.\d)\s*%", out)]
        @test length(sizes) == model.n_classes
        @test isapprox(sum(sizes), 100.0; atol=0.11)
        @test all(isapprox.(sizes, model.class_probs .* 100; atol=0.051))

        # First category of x1, one column per class
        m = match(r"^1:\s+(\d+\.\d{3})%\s+(\d+\.\d{3})%"m, out)
        @test m !== nothing
        vals = parse.(Float64, m.captures)
        @test all(isapprox.(vals, model.item_probs[1][:, 1] .* 100; atol=0.00051))
    end

    @testset "digits keyword" begin
        out1 = capture_stdout(() -> show_profiles(model, df, cols; digits=1))
        @test occursin("%", out1)
        @test occursin(r"\d+\.\d%", out1)
        @test !occursin(r"\d+\.\d{2,}%", out1)
        m = match(r"^1:\s+(\d+\.\d)%\s+(\d+\.\d)%"m, out1)
        @test m !== nothing
        vals = parse.(Float64, m.captures)
        @test all(isapprox.(vals, model.item_probs[1][:, 1] .* 100; atol=0.051))

        out0 = capture_stdout(() -> show_profiles(model, df, cols; digits=0))
        @test occursin(r"^1:\s+\d+%\s+\d+%"m, out0)

        out4 = capture_stdout(() -> show_profiles(model, df, cols; digits=4))
        @test occursin(r"^1:\s+\d+\.\d{4}%\s+\d+\.\d{4}%"m, out4)
    end

    @testset "var_names and var_labels" begin
        names = ["Item $i" for i in 1:n_items]
        labels = [["No", "Yes"] for _ in 1:n_items]

        out = capture_stdout(() -> show_profiles(model, df, cols; var_names=names, var_labels=labels))
        for nm in names
            @test occursin("\n$nm:\n", out)
        end
        @test !occursin("\nx1:\n", out)
        @test occursin(r"^No:"m, out)
        @test occursin(r"^Yes:"m, out)
        @test !occursin(r"^no:"m, out)   # default labels are replaced
        @test !occursin(r"^a:"m, out)
        @test count(r"^No:"m, out) == n_items
        @test count(r"^Yes:"m, out) == n_items

        # Only var_names
        out_n = capture_stdout(() -> show_profiles(model, df, cols; var_names=names))
        @test occursin("\nItem 1:\n", out_n)
        @test occursin(r"^no:"m, out_n)

        # Only var_labels
        out_l = capture_stdout(() -> show_profiles(model, df, cols; var_labels=labels))
        @test occursin("\nx1:\n", out_l)
        @test occursin(r"^Yes:"m, out_l)

        # Long labels widen the label column but the values still follow on the same line
        long = [["a very long category label", "b"] for _ in 1:n_items]
        out_long = capture_stdout(() -> show_profiles(model, df, cols; var_labels=long))
        @test occursin(r"^a very long category label:\s+\d+\.\d{3}%"m, out_long)
        @test occursin(r"^b:\s+\d+\.\d{3}%"m, out_long)

        # Existing behaviour: none of these warn
        @test_nowarn capture_stdout(() -> show_profiles(model, df, cols))
        @test_nowarn capture_stdout(() -> show_profiles(model, df, cols; var_names=names))
        @test_nowarn capture_stdout(() -> show_profiles(model, df, cols; var_labels=labels))
    end

    @testset "Categorical columns with unused levels" begin
        # Previously threw because the unused level had no matching probability column
        dfc = DataFrame([Symbol("c$j") => categorical([c == 1 ? "a" : "b" for c in raw[:, j]];
            levels=["a", "b", "c"]) for j in 1:3]...)
        colsc = [:c1, :c2, :c3]
        datac, catsc = prepare_data(dfc, colsc...)
        @test catsc == [2, 2, 2]
        Random.seed!(10)
        mc = LCAModel(2, 3, catsc)
        fit!(mc, datac)

        out = capture_stdout(() -> show_profiles(mc, dfc, colsc))
        @test occursin(r"^a:"m, out)
        @test occursin(r"^b:"m, out)
        @test !occursin(r"^c:"m, out)
        @test count("%", out) == mc.n_classes + mc.n_classes * sum(catsc)

        # Level order (not lexical order) determines the label order
        dfo = DataFrame([Symbol("o$j") => categorical([c == 1 ? "z" : "y" for c in raw[:, j]];
            levels=["z", "y"]) for j in 1:3]...)
        datao, catso = prepare_data(dfo, [:o1, :o2, :o3]...)
        @test datao == datac  # same codes: "z" is level 1, as "a" was
        out_o = capture_stdout(() -> show_profiles(mc, dfo, [:o1, :o2, :o3]))
        @test findfirst(r"^z:"m, out_o)[1] < findfirst(r"^y:"m, out_o)[1]
    end

    @testset "Polytomous item" begin
        rng3 = StableRNG(33)
        raw3, _ = simulate_lca(rng3, 300, [0.5, 0.5],
            [[0.7 0.2 0.1; 0.1 0.2 0.7], [0.8 0.2; 0.2 0.8], [0.8 0.2; 0.2 0.8]])
        df3 = DataFrame(
            t=[("low", "mid", "high")[c] for c in raw3[:, 1]],
            u=raw3[:, 2],
            v=raw3[:, 3],
        )
        data3, cats3 = prepare_data(df3, :t, :u, :v)
        @test cats3 == [3, 2, 2]
        Random.seed!(11)
        m3 = LCAModel(2, 3, cats3)
        fit!(m3, data3)
        out = capture_stdout(() -> show_profiles(m3, df3, [:t, :u, :v]))
        # Sorted distinct strings: high < low < mid
        @test occursin(r"^high:"m, out) && occursin(r"^low:"m, out) && occursin(r"^mid:"m, out)
        @test findfirst(r"^high:"m, out)[1] < findfirst(r"^low:"m, out)[1] < findfirst(r"^mid:"m, out)[1]
        @test count("%", out) == 2 + 2 * (3 + 2 + 2)
    end
end
