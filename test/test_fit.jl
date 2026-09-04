using Test
using LatentClassAnalysis
using DataFrames
using Random
using StableRNGs
using Statistics

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

const LCA = LatentClassAnalysis

@testset "Model fitting" begin
    n = 2000
    y, classes = simulate_lca(StableRNG(1), n, TWO_CLASS_PROBS, TWO_CLASS_ITEMS)
    d = LCAData(y)

    @testset "Two-class parameter recovery" begin
        m = @test_logs fit(LCAModel, d, 2; rng=StableRNG(1))
        @test m isa LCAModel
        @test m.n_classes == 2 && m.n_items == 6 && m.n_categories == fill(2, 6)
        @test m.converged
        @test m.iterations > 0
        @test isfinite(m.loglik) && m.loglik < 0
        @test m.flags.converged && m.flags.best_ll_replicated
        @test m.flags.n_boundary == 0 && isempty(m.flags.empty_classes)
        @test m.vcov isa Matrix{Float64} && size(m.vcov) == (dof(m), dof(m))
        @test m.data === d

        perm = align_classes(m.item_probs, TWO_CLASS_ITEMS)
        e_class, e_item = max_abs_error(m, perm, TWO_CLASS_PROBS, TWO_CLASS_ITEMS)
        @test e_class < 0.05
        @test e_item < 0.05

        # Classes are sorted by decreasing size, so class 1 is the 0.6 class
        @test issorted(m.class_probs; rev=true)
        @test perm == [1, 2]
        @test sum(m.class_probs) ≈ 1
        @test all(all(abs.(sum(P, dims=2) .- 1) .< 1e-12) for P in m.item_probs)

        # beta holds the logit of the class sizes against class 1
        @test size(m.beta) == (1, 1)
        @test m.beta[1, 1] ≈ log(m.class_probs[2] / m.class_probs[1])

        # The MLE is at least as good as the truth
        ll_true = sum(log(sum(TWO_CLASS_PROBS[k] * prod(TWO_CLASS_ITEMS[j][k, y[i, j]] for j in 1:6)
                              for k in 1:2)) for i in 1:n)
        @test m.loglik >= ll_true - 1e-6

        # Classification accuracy
        @test mean(invperm(perm)[classify(m)] .== classes) > 0.9

        # Options and start bookkeeping
        @test m.options == LCAOptions()
        @test length(m.start_loglik) == 20
        @test maximum(m.start_loglik) == m.loglik
    end

    @testset "Three classes with 2, 3 and 4 categories" begin
        y3, _ = simulate_lca(StableRNG(3), 3000, THREE_CLASS_PROBS, THREE_CLASS_ITEMS)
        d3 = LCAData(y3)
        @test d3.n_categories == [2, 3, 4, 2, 3, 4]
        m3 = fit(LCAModel, d3, 3; rng=StableRNG(1))
        @test m3.converged
        perm = align_classes(m3.item_probs, THREE_CLASS_ITEMS)
        e_class, e_item = max_abs_error(m3, perm, THREE_CLASS_PROBS, THREE_CLASS_ITEMS)
        @test e_class < 0.06
        @test e_item < 0.06
        @test issorted(m3.class_probs; rev=true)
        @test [size(P) for P in m3.item_probs] == [(3, 2), (3, 3), (3, 4), (3, 2), (3, 3), (3, 4)]
        @test size(m3.beta) == (1, 2)
        @test dof(m3) == 2 + 3 * (1 + 2 + 3 + 1 + 2 + 3)
    end

    @testset "Reproducibility" begin
        m1 = fit(LCAModel, d, 2; rng=StableRNG(1))
        m2 = fit(LCAModel, d, 2; rng=StableRNG(1))
        @test same_fit(m1, m2)
        m3 = fit(LCAModel, d, 2; rng=StableRNG(2))
        @test m3.start_loglik != m1.start_loglik   # different starts ...
        @test isapprox(m3.loglik, m1.loglik; rtol=1e-6)  # ... same optimum

        # Threaded and serial runs agree bitwise (trivially so with one thread)
        mt = fit(LCAModel, d, 2; rng=StableRNG(1), multithreaded=true)
        @test same_fit(mt, m1)

        # Aggregated and row-wise EM agree to numerical precision
        ma = fit(LCAModel, d, 2; rng=StableRNG(1), aggregate=false)
        @test !ma.options.aggregate
        @test isapprox(ma.loglik, m1.loglik; rtol=1e-10)
        @test maximum(abs.(ma.class_probs .- m1.class_probs)) < 1e-10
        @test all(maximum(abs.(ma.item_probs[j] .- m1.item_probs[j])) < 1e-10 for j in 1:6)
        @test maximum(abs.(ma.posterior .- m1.posterior)) < 1e-10
        @test maximum(abs.(ma.start_loglik .- m1.start_loglik)) < 1e-6
    end

    @testset "Start selection" begin
        # With every start continued, the winner is the best final log-likelihood
        m = fit(LCAModel, d, 2; rng=StableRNG(5), n_starts=6, n_final=6)
        @test m.options.n_starts == 6 && m.options.n_final == 6
        @test length(m.start_loglik) == 6
        @test m.loglik == maximum(m.start_loglik)

        # n_final is capped at n_starts
        m1 = fit(LCAModel, d, 2; rng=StableRNG(5), n_starts=1)
        @test m1.options.n_final == 1
        @test length(m1.start_loglik) == 1
        @test m1.flags.best_ll_replicated   # trivially, with a single start
        @test m1.converged

        # Short runs only: starts not continued keep their short-run log-likelihood
        ms = fit(LCAModel, d, 2; rng=StableRNG(5), n_starts=5, n_final=1, short_iters=2)
        @test length(ms.start_loglik) == 5
        @test ms.loglik == maximum(ms.start_loglik)
        @test count(==(ms.loglik), ms.start_loglik) == 1

        # verbose prints one line per start and a summary
        out = capture_stdout(() -> fit(LCAModel, d, 2; rng=StableRNG(5), n_starts=3, n_final=2, verbose=true))
        @test count("short-run log-likelihood", out) == 3
        @test count("[continued]", out) == 2
        @test count("final log-likelihood", out) == 2
        @test occursin("[best]", out)
        @test occursin("replicated by", out)
        @test capture_stdout(() -> fit(LCAModel, d, 2; rng=StableRNG(5), n_starts=3)) == ""
    end

    @testset "Convergence control" begin
        # A tiny iteration budget does not converge and is flagged
        m = @test_logs (:warn, r"did not converge within 2 iterations") fit(LCAModel, d, 2; rng=StableRNG(1), max_iter=2, n_starts=2, short_iters=1)
        @test !m.converged
        @test !m.flags.converged
        @test m.iterations == 1 + 2
        @test m.options.max_iter == 2
        @test isfinite(m.loglik)
        @test m.loglik == maximum(m.start_loglik)

        # A loose tolerance stops earlier than a tight one
        m_loose = fit(LCAModel, d, 2; rng=StableRNG(1), tol=1e-3, n_starts=1)
        m_tight = fit(LCAModel, d, 2; rng=StableRNG(1), tol=1e-12, n_starts=1)
        @test m_loose.iterations < m_tight.iterations
        @test m_loose.options.tol == 1e-3
        @test isapprox(m_loose.loglik, m_tight.loglik; rtol=1e-4)
        @test fit(LCAModel, d, 2; rng=StableRNG(1), tol=1f-6, n_starts=1).options.tol == Float64(1f-6)
    end

    @testset "Several class counts and the table method" begin
        # (the 3- and 4-class fits to two-class data warn about non-replicated maxima,
        # boundary probabilities or non-convergence)
        models = @test_logs (:warn, r"3-class fit") (:warn, r"4-class fit") match_mode = :any fit(
            LCAModel, d, 2:4; rng=StableRNG(1), n_starts=4, n_final=2)
        @test models isa Vector{LCAModel}
        @test [m.n_classes for m in models] == [2, 3, 4]
        @test all(issorted(m.class_probs; rev=true) for m in models)
        @test models[1].loglik <= models[2].loglik + 1e-6   # nested models
        @test all(isfinite(m.loglik) for m in models)

        # An rng shared across fits is consumed in sequence, so the vector method equals
        # calling fit repeatedly with the same generator
        rng = StableRNG(1)
        m2 = fit(LCAModel, d, 2; rng=rng, n_starts=4, n_final=2)
        m3 = @test_logs (:warn, r"3-class fit") match_mode = :any fit(LCAModel, d, 3; rng=rng, n_starts=4, n_final=2)
        @test same_fit(m2, models[1]) && same_fit(m3, models[2])

        # Table convenience method equals prepare_data + fit
        df = DataFrame([Symbol("x$j") => y[:, j] for j in 1:6]...)
        items = [Symbol("x$j") for j in 1:6]
        mt = fit(LCAModel, df, items, 2; rng=StableRNG(1))
        mp = fit(LCAModel, prepare_data(df, items), 2; rng=StableRNG(1))
        @test same_fit(mt, mp)
        @test mt.data.item_names == items
        mts = @test_logs (:warn, r"3-class fit") match_mode = :any fit(
            LCAModel, df, items, 2:3; rng=StableRNG(1), n_starts=2, n_final=1)
        @test mts isa Vector{LCAModel} && length(mts) == 2
        mtn = fit(LCAModel, (x1=y[:, 1], x2=y[:, 2], x3=y[:, 3], x4=y[:, 4], x5=y[:, 5], x6=y[:, 6]),
                  items, 2; rng=StableRNG(1))
        @test same_fit(mtn, mp)
        # levels and drop_unused_levels are passed through to prepare_data
        lv = Dict(:x1 => [2, 1])
        @test same_fit(fit(LCAModel, df, items, 2; levels=lv, rng=StableRNG(1)),
                       fit(LCAModel, prepare_data(df, items; levels=lv), 2; rng=StableRNG(1)))
        @test_throws ArgumentError fit(LCAModel, df, [:zzz], 2)
        @test_throws ArgumentError fit(LCAModel, y, items, 2)   # a matrix is not a table
        @test_throws MethodError fit(LCAModel, y, 2)            # no matrix entry point
    end

    @testset "Single class" begin
        m1 = fit(LCAModel, d, 1; rng=StableRNG(1))
        @test m1.n_classes == 1
        @test m1.class_probs == [1.0]
        @test m1.converged && m1.iterations == 0
        @test m1.start_loglik == [m1.loglik]
        @test size(m1.beta) == (1, 0)
        @test size(m1.posterior) == (n, 1) && all(==(1.0), m1.posterior)
        @test dof(m1) == 6
        @test entropy(m1) == 1.0
        for j in 1:6
            @test m1.item_probs[j][1, :] ≈ [mean(y[:, j] .== c) for c in 1:2]
        end
        @test m1.loglik ≈ sum(log(m1.item_probs[j][1, y[i, j]]) for i in 1:n, j in 1:6)
        @test m1.flags.best_ll_replicated
        @test isempty(m1.flags.empty_classes)
    end

    @testset "Starting values" begin
        m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=4, n_final=2)

        # A fitted model as the first start reproduces its own solution
        mw = fit(LCAModel, d, 2; rng=StableRNG(1), init=m, n_starts=1)
        @test isapprox(mw.loglik, m.loglik; rtol=1e-10)
        @test mw.iterations <= 2
        @test maximum(abs.(mw.class_probs .- m.class_probs)) < 1e-5   # EM converges linearly

        # LCAParams and NamedTuple starts; a vector of starts fills the first starts
        θ = LCA.LCAParams(copy(m.class_probs), [copy(P) for P in m.item_probs], nothing)
        mp = fit(LCAModel, d, 2; rng=StableRNG(1), init=θ, n_starts=1)
        @test same_fit(mp, mw)
        mn = fit(LCAModel, d, 2; rng=StableRNG(1), init=(class_probs=m.class_probs, item_probs=m.item_probs), n_starts=1)
        @test same_fit(mn, mw)
        mv = fit(LCAModel, d, 2; rng=StableRNG(1), init=[m, θ], n_starts=1)
        @test length(mv.start_loglik) == 2                  # more starts than n_starts
        @test isapprox(mv.loglik, m.loglik; rtol=1e-10)
        mv2 = fit(LCAModel, d, 2; rng=StableRNG(1), init=[m], n_starts=3, n_final=3)
        @test length(mv2.start_loglik) == 3
        @test mv2.start_loglik[1] ≈ m.loglik rtol = 1e-10

        # Unnormalized starts are normalized (row by row, before the floor is applied) and
        # reach the same optimum; invalid starts are rejected
        θu = LCA._normalize_init!(LCA.LCAParams([2.0, 2.0], [[3.0 1.0; 1.0 3.0]], nothing))
        @test θu.class_probs ≈ [0.5, 0.5] && θu.item_probs[1] ≈ [0.75 0.25; 0.25 0.75]
        mu = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=1,
                 init=(class_probs=[2.0, 2.0], item_probs=[[3.0 1.0; 1.0 3.0] for _ in 1:6]))
        @test mu.converged
        @test mu.loglik ≈ m.loglik rtol = 1e-8
        @test_throws ArgumentError LCA._normalize_init!(LCA.LCAParams([1.0, 1.0], [[0.0 0.0; 0.5 0.5]], nothing))
        @test_throws ArgumentError LCA._normalize_init!(LCA.LCAParams([1.0, 1.0], [[-1.0 2.0; 0.5 0.5]], nothing))
        @test_throws ArgumentError LCA._normalize_init!(LCA.LCAParams([0.0, 0.0], [[0.5 0.5; 0.5 0.5]], nothing))
        @test_throws ArgumentError fit(LCAModel, d, 3; init=m)                      # wrong K
        @test_throws ArgumentError fit(LCAModel, d, 2; init=(class_probs=[0.5, 0.5],))
        @test_throws ArgumentError fit(LCAModel, d, 2; init=(class_probs=[0.5, 0.5], item_probs=[[0.5 0.5]]))
        @test_throws ArgumentError fit(LCAModel, d, 2; init=(class_probs=[-1.0, 2.0], item_probs=m.item_probs))
        @test_throws ArgumentError fit(LCAModel, d, 2; init=1.0)
    end

    @testset "Argument validation" begin
        @test_throws ArgumentError fit(LCAModel, d, 0)
        @test_throws ArgumentError fit(LCAModel, d, 2; n_starts=0)
        @test_throws ArgumentError fit(LCAModel, d, 2; n_final=0)
        @test_throws ArgumentError fit(LCAModel, d, 2; short_iters=-1)
        @test_throws ArgumentError fit(LCAModel, d, 2; max_iter=0)
        @test_throws ArgumentError fit(LCAModel, d, 2; tol=-1.0)
        @test_throws ArgumentError fit(LCAModel, d, 2; se=:bootstrap)
        @test_throws ArgumentError fit(LCAModel, LCAData(y[1:0, :]; n_categories=fill(2, 6)), 2)
        @test_throws ArgumentError fit(LCAModel, d, 2; covariates=true)   # data without covariates
        dc = LCAData(y; covariates=randn(StableRNG(1), n))
        @test hascovariates(fit(LCAModel, dc, 2; rng=StableRNG(1), n_starts=2, n_final=1))  # regression fit
        mc = fit(LCAModel, dc, 2; rng=StableRNG(1), covariates=false)               # unconditional fit
        @test same_fit(mc, fit(LCAModel, d, 2; rng=StableRNG(1)))
        @test !hascovariates(mc)
        @test LCAOptions(se=:none).se == :none
        @test_throws ArgumentError LCAOptions(n_starts=0)
        @test_throws ArgumentError LCAOptions(se=:foo)
        m_none = fit(LCAModel, d, 2; rng=StableRNG(1), se=:none, n_starts=2)
        @test m_none.options.se == :none && m_none.vcov === nothing
    end

    @testset "Identifiability check" begin
        ci = LCA.check_identifiability
        # 2 classes × 3 binary items: 7 parameters, 7 cells (equality is fine)
        @test (@test_logs ci(2, [2, 2, 2])) === true
        @test (@test_logs ci(2, fill(2, 6))) === true
        @test (@test_logs ci(3, [3, 3, 3])) === true
        @test (@test_logs ci(2, 2:2:10)) === true
        @test (@test_logs ci(Int32(2), Int8[2, 2, 2])) === true
        # Many items: no overflow
        @test (@test_logs ci(2, fill(2, 70))) === true
        @test (@test_logs ci(5, fill(2, 200))) === true
        # Too many parameters
        @test (@test_logs (:warn, r"^Model may not be identified: 11 free parameters exceed the 7 degrees of freedom") ci(3, [2, 2, 2])) === false
        @test (@test_logs (:warn, r"^Model may not be identified: 5 free parameters exceed the 3") ci(2, [2, 2])) === false
        @test (@test_logs (:warn, r"Model may not be identified") ci(4, [2, 2, 2, 2])) === false

        # fit warns but still fits
        d3 = LCAData(y[:, 1:3])
        m = @test_logs (:warn, r"Model may not be identified") match_mode = :any fit(LCAModel, d3, 3; rng=StableRNG(1), n_starts=2, n_final=1)
        @test m isa LCAModel
        @test_logs fit(LCAModel, LCAData(y[:, 1:4]), 2; rng=StableRNG(1), n_starts=2, n_final=1)
    end

    @testset "Fit flags" begin
        # An item that reveals the class exactly drives its probabilities to the boundary
        yb = copy(y)
        yb[:, 1] = classes
        mb = @test_logs (:warn, r"item-response probabilities are on the boundary") fit(LCAModel, LCAData(yb), 2; rng=StableRNG(1), n_starts=4, n_final=2)
        @test mb.flags.n_boundary >= 2
        @test !clean_flags(mb.flags)
        @test all(x -> x <= 1e-6 || x >= 1 - 1e-6, mb.item_probs[1])
        @test all(all(P .>= 1e-10) for P in mb.item_probs)   # floored, never exactly zero
        @test isfinite(mb.loglik)

        # Several raised flags are joined into one warning: one EM step from the boundary
        # solution with perturbed class sizes and a zero tolerance
        m2 = @test_logs (:warn, r"did not converge within 1 iterations; .*boundary") fit(
            LCAModel, LCAData(yb), 2; rng=StableRNG(1), n_starts=1, short_iters=0, max_iter=1, tol=0.0,
            init=(class_probs=[0.55, 0.45], item_probs=mb.item_probs))
        @test !m2.flags.converged && m2.flags.n_boundary > 0
        @test m2.iterations == 1

        # Messages
        f = LCA.FitFlags(false, 3, [2], false, true)
        msgs = LCA._flag_messages(f, LCAOptions(max_iter=7))
        @test length(msgs) == 5
        @test occursin("7 iterations", msgs[1])
        @test occursin("3 item-response probabilities", msgs[2])
        @test occursin("[2]", msgs[3])
        @test occursin("only one of the continued starts", msgs[4])
        @test occursin("separation", msgs[5])
        @test isempty(LCA._flag_messages(LCA.FitFlags(true, 0, Int[], true, false)))
        @test clean_flags(LCA.FitFlags(true, 0, Int[], true, false))
    end
end

@testset "nothing touches the global RNG" begin
    y, _ = simulate_lca(StableRNG(70), 200, TWO_CLASS_PROBS, TWO_CLASS_ITEMS)
    d = LCAData(y)
    r0 = copy(Random.default_rng())
    m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=2, n_final=1, se=:none)
    simulate(m, 20; rng=StableRNG(2))
    Test.collect_test_logs(() -> bootstrap(m; n_boot=2, rng=StableRNG(3)))
    @test copy(Random.default_rng()) == r0
end

@testset "flag messages and unobserved items" begin
    @test LCA._flag_messages(LCA.FitFlags(false, 0, Int[], true, false)) == ["EM did not converge"]
    @test isempty(LCA._flag_messages(LCA.FitFlags(true, 0, Int[], true, false)))
    # An item without a single observed response keeps uniform probabilities, is named in
    # the warning, and gets NaN standard errors while the other item keeps its own
    y0 = [1 0; 2 0; 1 0; 2 0; 1 0; 1 0]
    d0 = LCAData(y0; n_categories=[2, 2])
    m0 = @test_logs (:warn, r"item\(s\) \[:item2\] have no observed responses.*zero observed information") fit(LCAModel, d0, 1)
    @test m0.item_probs[2] == [0.5 0.5]
    @test isfinite(vcov(m0)[1, 1]) && isnan(vcov(m0)[2, 2])
end
