using Test
using LatentClassAnalysis
using CategoricalArrays
using DataFrames
using StableRNGs

@isdefined(capture_stdout) || include(joinpath(@__DIR__, "testutils.jl"))

@testset "Display" begin
    n_items = 5
    raw, _ = simulate_lca(StableRNG(31), 300, [0.6, 0.4], [[0.8 0.2; 0.2 0.8] for _ in 1:n_items])
    df = DataFrame(
        x1=raw[:, 1],                                                                # 1/2 coded
        x2=raw[:, 2] .- 1,                                                           # 0/1 coded
        x3=[c == 1 ? "no" : "yes" for c in raw[:, 3]],                               # strings
        x4=categorical([c == 1 ? "a" : "b" for c in raw[:, 4]]; levels=["a", "b", "c"]),  # unused level
        x5=raw[:, 5] .== 2,                                                          # Bool
    )
    cols = [Symbol("x$i") for i in 1:n_items]
    d = prepare_data(df, cols)
    @test d.y == raw
    model = fit(LCAModel, d, 2; rng=StableRNG(9), n_starts=4, n_final=2)

    @testset "show(LCAData)" begin
        s = sprint(show, d)
        @test occursin("LCAData with 300 observations and 5 items", s)
        @test occursin("x1: 2 levels (1, 2), 300 observed", s)
        @test occursin("x2: 2 levels (0, 1)", s)
        @test occursin("x3: 2 levels (no, yes)", s)
        @test occursin("x4: 2 levels (a, b)", s)
        @test occursin("x5: 2 levels (false, true)", s)
        @test occursin("covariates: none", s)
        @test !occursin("missing", s)
        @test sprint(show, d; context=:compact => true) == "LCAData(300 × 5)"

        dm = prepare_data((a=[1, missing, 2], b=[1, 2, 2]), [:a, :b]; covariates=[:b])
        sm = sprint(show, dm)
        @test occursin("(1 missing responses)", sm)
        @test occursin("a: 2 levels (1, 2), 2 observed", sm)
        @test occursin("covariates: b", sm)

        # Many items are abbreviated
        big = LCAData(rand(StableRNG(1), 1:2, 40, 30))
        sb = sprint(show, big)
        @test occursin("… and 5 more items", sb)
        @test occursin("item25", sb) && !occursin("item26", sb)
    end

    @testset "show(LCAModel)" begin
        s = sprint(show, model)
        @test occursin("LCAModel with 2 classes, 5 items and 300 observations", s)
        @test occursin("log-likelihood: ", s)
        @test occursin("dof: $(dof(model))", s)
        @test occursin("BIC: ", s)
        @test occursin("converged after $(model.iterations) iterations", s)
        @test occursin("best of 4 start(s)", s)
        @test occursin("class sizes: ", s)
        @test occursin("fit flags: none", s)
        @test occursin("standard errors: observed information", s)
        @test !occursin("covariates", s)
        @test sprint(show, model; context=:compact => true) == "LCAModel(2 classes, 5 items, n = 300)"
        @test occursin("LCAModel with 2 classes", sprint(show, MIME("text/plain"), model))

        # Flags are printed
        bad = @test_logs (:warn, r"did not converge") fit(LCAModel, d, 2; rng=StableRNG(9), n_starts=2, short_iters=1, max_iter=1)
        sb = sprint(show, bad)
        @test occursin("not converged after", sb)
        @test occursin("fit flags: EM did not converge within 1 iterations", sb)

        m1 = fit(LCAModel, d, 1)
        @test occursin("single class: closed-form solution", sprint(show, m1))

        dmiss = LCAData(mcar!(StableRNG(2), copy(raw), 0.1))
        mm = fit(LCAModel, dmiss, 2; rng=StableRNG(9), n_starts=2, n_final=1)
        @test occursin("missing responses: $(sum(nmissing(dmiss)))", sprint(show, mm))
    end

    @testset "show(ModelDiagnostics)" begin
        diag = diagnostics(model)
        s = sprint(show, diag)
        @test startswith(s, "ModelDiagnostics(n_classes = 2, nobs = 300, dof = $(dof(model))")
        @test occursin("bic = ", s) && occursin("entropy = ", s) && occursin("converged = true", s)

        models = fit(LCAModel, d, 1:2; rng=StableRNG(9), n_starts=2, n_final=1)
        v = diagnostics(models)
        t = sprint(show, MIME("text/plain"), v)
        @test occursin("2-element Vector{ModelDiagnostics}:", t)
        @test occursin("classes", t) && occursin("BIC", t) && occursin("sBIC", t) && occursin("entropy", t)
        @test count("\n", t) == 3     # header + one line per model
        @test sprint(show, MIME("text/plain"), ModelDiagnostics[]) == "ModelDiagnostics[]"
    end

    @testset "show_profiles: basic display" begin
        out = capture_stdout(() -> show_profiles(model))
        @test occursin("Latent Class Profiles", out)
        @test occursin("Class Sizes:", out)
        @test occursin("Class 1", out)
        @test occursin("Class 2", out)
        for c in cols
            @test occursin("\n$c:\n", out)
        end

        # Default labels are the level labels stored with the data
        @test occursin(r"^1:"m, out) && occursin(r"^2:"m, out)          # x1
        @test occursin(r"^0:"m, out)                                     # x2
        @test occursin(r"^no:"m, out) && occursin(r"^yes:"m, out)        # x3
        @test occursin(r"^a:"m, out) && occursin(r"^b:"m, out)           # x4
        @test !occursin(r"^c:"m, out)                                    # unused level not shown
        @test occursin(r"^false:"m, out) && occursin(r"^true:"m, out)    # x5

        # Three decimals by default; one percentage per class size and per class × level
        @test occursin(r"\d+\.\d{3}%", out)
        @test count("%", out) == model.n_classes + model.n_classes * sum(d.n_categories)

        # Deterministic; the io keyword writes elsewhere; nothing is returned
        @test capture_stdout(() -> show_profiles(model)) == out
        buf = IOBuffer()
        @test show_profiles(model; io=buf) === nothing
        @test String(take!(buf)) == out
        @test capture_stdout(() -> show_profiles(model; io=IOBuffer())) == ""
    end

    @testset "show_profiles: printed values match the model" begin
        out = sprint(io -> show_profiles(model; io=io))
        sizes = [parse(Float64, mt.captures[2]) for mt in eachmatch(r"Class (\d+): (\d+\.\d)\s*%", out)]
        @test length(sizes) == model.n_classes
        @test isapprox(sum(sizes), 100.0; atol=0.11)
        @test all(isapprox.(sizes, model.class_probs .* 100; atol=0.051))

        mt = match(r"^1:\s+(\d+\.\d{3})% ±\S+\s+(\d+\.\d{3})% ±\S+"m, out)
        @test mt !== nothing
        vals = parse.(Float64, mt.captures)
        @test all(isapprox.(vals, model.item_probs[1][:, 1] .* 100; atol=0.00051))
    end

    @testset "show_profiles: digits" begin
        out1 = sprint(io -> show_profiles(model; digits=1, io=io))
        @test occursin(r"\d+\.\d%", out1)
        @test !occursin(r"\d+\.\d{2,}%", out1)
        out0 = sprint(io -> show_profiles(model; digits=0, io=io))
        @test occursin(r"^1:\s+\d+% ±\d+\s+\d+% ±\d+"m, out0)
        out4 = sprint(io -> show_profiles(model; digits=4, io=io))
        @test occursin(r"^1:\s+\d+\.\d{4}% ±\d+\.\d{4}\s+\d+\.\d{4}% ±\d+\.\d{4}"m, out4)
        @test_throws ArgumentError show_profiles(model; digits=-1, io=IOBuffer())
    end

    @testset "show_profiles: var_names and var_labels" begin
        vnames = ["Item $i" for i in 1:n_items]
        labels = [["No", "Yes"] for _ in 1:n_items]
        out = sprint(io -> show_profiles(model; var_names=vnames, var_labels=labels, io=io))
        for nm in vnames
            @test occursin("\n$nm:\n", out)
        end
        @test !occursin("\nx1:\n", out)
        @test count(r"^No:"m, out) == n_items
        @test count(r"^Yes:"m, out) == n_items
        @test !occursin(r"^no:"m, out)

        out_n = sprint(io -> show_profiles(model; var_names=vnames, io=io))
        @test occursin("\nItem 1:\n", out_n) && occursin(r"^no:"m, out_n)
        out_l = sprint(io -> show_profiles(model; var_labels=labels, io=io))
        @test occursin("\nx1:\n", out_l) && occursin(r"^Yes:"m, out_l)

        # Long labels widen the label column but the values still follow on the same line
        long = [["a very long category label", "b"] for _ in 1:n_items]
        out_long = sprint(io -> show_profiles(model; var_labels=long, io=io))
        @test occursin(r"^a very long category label:\s+\d+\.\d{3}%"m, out_long)

        @test_throws ArgumentError show_profiles(model; var_names=["only one"], io=IOBuffer())
        @test_throws ArgumentError show_profiles(model; var_labels=[["a", "b"]], io=IOBuffer())
        @test_throws ArgumentError show_profiles(model; var_labels=[["a", "b", "c"] for _ in 1:n_items], io=IOBuffer())
    end

    @testset "Polytomous item" begin
        raw3, _ = simulate_lca(StableRNG(33), 300, [0.5, 0.5],
            [[0.7 0.2 0.1; 0.1 0.2 0.7], [0.8 0.2; 0.2 0.8], [0.8 0.2; 0.2 0.8]])
        tbl = (t=[("low", "mid", "high")[c] for c in raw3[:, 1]], u=raw3[:, 2], v=raw3[:, 3])
        d3 = prepare_data(tbl, [:t, :u, :v])
        @test d3.n_categories == [3, 2, 2]
        m3 = fit(LCAModel, d3, 2; rng=StableRNG(1), n_starts=2, n_final=1)
        out = sprint(io -> show_profiles(m3; io=io))
        # Sorted distinct strings: high < low < mid
        @test findfirst(r"^high:"m, out)[1] < findfirst(r"^low:"m, out)[1] < findfirst(r"^mid:"m, out)[1]
        @test count("%", out) == 2 + 2 * (3 + 2 + 2)
    end

    @testset "profiles table" begin
        prof = profiles(model)
        @test prof isa Vector{<:NamedTuple}
        @test length(prof) == sum(d.n_categories) * model.n_classes
        @test propertynames(prof[1]) == (:item, :level, :class, :prob, :se, :lower, :upper)
        @test prof[1].item == :x1 && prof[1].level == "1" && prof[1].class == 1
        @test prof[1].prob == model.item_probs[1][1, 1]
        @test prof[2].class == 2 && prof[2].prob == model.item_probs[1][2, 1]
        @test prof[3].level == "2"
        @test all(isfinite, (prof[1].se, prof[1].lower, prof[1].upper))
        @test prof[1].lower <= prof[1].prob <= prof[1].upper
        @test [r.level for r in prof if r.item == :x3 && r.class == 1] == ["no", "yes"]
        @test [r.level for r in prof if r.item == :x5 && r.class == 2] == ["false", "true"]
        pdf = DataFrame(prof)
        @test size(pdf) == (length(prof), 7)
        @test names(pdf) == ["item", "level", "class", "prob", "se", "lower", "upper"]
        @test all(isapprox.(combine(groupby(pdf, [:item, :class]), :prob => sum).prob_sum, 1.0; atol=1e-12))
        @test length(profiles(model; level=0.9)) == length(prof)
    end
end
