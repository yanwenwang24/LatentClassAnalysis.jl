using Test
using LatentClassAnalysis
using DataFrames
using LinearAlgebra
using StableRNGs
using Tables

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

@testset "StatsAPI interface" begin
    n = 600
    items = [[0.85 0.15; 0.2 0.8], [0.7 0.2 0.1; 0.1 0.2 0.7], [0.9 0.1; 0.3 0.7]]
    y, _ = simulate_lca(StableRNG(11), n, [0.6, 0.4], items)
    d = LCAData(y)
    m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=4, n_final=2)

    @testset "Accessors" begin
        @test nobs(m) == n
        @test nobs(d) == n
        @test dof(m) == 1 + 2 * (1 + 2 + 1)      # 9 free parameters
        @test loglikelihood(m) == m.loglik
        @test loglikelihood(m, d) == m.loglik
        @test loglikelihood(m, LCAData(y[1:100, :])) > m.loglik      # fewer rows
        @test isfitted(m)
        @test hascovariates(m) == false
        @test !hasmissing(m)
        @test nmissing(m) == [0, 0, 0]
    end

    @testset "Information criteria" begin
        ll = m.loglik
        p = dof(m)
        @test aic(m) ≈ -2ll + 2p
        @test bic(m) ≈ -2ll + p * log(n)
        @test aicc(m) ≈ -2ll + 2p + 2p * (p + 1) / (n - p - 1)
        @test sbic(m) ≈ -2ll + p * log((n + 2) / 24)
        @test bic(m) > aic(m)
        @test sbic(m) < bic(m)
    end

    @testset "Entropy" begin
        @test 0 <= entropy(m) <= 1
        @test entropy(m; relative=false) >= 0
        post = m.posterior
        h = -sum(p > 0 ? p * log(p) : 0.0 for p in post)
        @test entropy(m; relative=false) ≈ h
        @test entropy(m) ≈ 1 - h / (n * log(2))
        @test entropy(m) > 0.5                       # well separated
        m1 = fit(LCAModel, d, 1)
        @test entropy(m1) == 1.0
        @test entropy(m1; relative=false) == 0.0
    end

    @testset "Diagnostics" begin
        diag = diagnostics(m)
        @test diag isa ModelDiagnostics
        @test diag.n_classes == 2 && diag.nobs == n && diag.dof == dof(m)
        @test diag.ll == m.loglik
        @test diag.aic == aic(m) && diag.bic == bic(m) && diag.sbic == sbic(m)
        @test diag.entropy == entropy(m)
        @test diag.converged == m.converged

        models = fit(LCAModel, d, 1:3; rng=StableRNG(1), n_starts=4, n_final=2)
        diags = diagnostics(models)
        @test diags isa Vector{ModelDiagnostics}
        @test [x.n_classes for x in diags] == [1, 2, 3]
        @test [x.dof for x in diags] == [4, 9, 14]

        # Tables.jl row interface
        @test Tables.istable(diags)
        @test Tables.rowaccess(diags)
        @test Tables.rows(diags) === diags
        sch = Tables.schema(diags)
        @test sch.names == (:n_classes, :nobs, :dof, :ll, :aic, :bic, :sbic, :entropy, :converged)
        df = DataFrame(diags)
        @test names(df) == ["n_classes", "nobs", "dof", "ll", "aic", "bic", "sbic", "entropy", "converged"]
        @test size(df) == (3, 9)
        @test df.bic == [x.bic for x in diags]
        @test df.n_classes == [1, 2, 3]
        @test argmin(df.bic) == 2
        cols = Tables.columntable(diags)
        @test cols.ll == [x.ll for x in diags]
        @test isempty(DataFrame(ModelDiagnostics[]))
    end

    @testset "Inference verbs" begin
        @test m.vcov isa Matrix{Float64}
        @test size(vcov(m)) == (dof(m), dof(m))
        @test stderror(m) == sqrt.(diag(vcov(m)))
        @test length(coef(m)) == length(coefnames(m)) == dof(m)
        @test size(confint(m)) == (dof(m), 2)
        @test length(coeftable(m)) == dof(m)
        @test size(informationmatrix(m)) == (dof(m), dof(m))
        prof = profiles(m)
        @test length(prof) == sum(d.n_categories) * 2
        @test all(isfinite(r.se) && r.lower <= r.prob <= r.upper for r in prof)
        @test_throws ArgumentError profiles(m; level=1.5)
        mn = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=2, n_final=1, se=:none)
        @test mn.vcov === nothing
        @test_throws ErrorException vcov(mn)
        @test_throws ErrorException stderror(mn)
        @test all(isnan(r.se) && isnan(r.lower) && isnan(r.upper) for r in profiles(mn))
    end

    @testset "Bootstrap placeholders" begin
        @test_throws ErrorException simulate(m)
        @test_throws ErrorException simulate(m, 10; rng=StableRNG(1))
        @test_throws ErrorException bootstrap(m)
        @test_throws ErrorException bootstrap_lrt(m, m)
        t = BootstrapLRT(m, m, 3.5, [1.0, 2.0, 4.0], 0.5, 3)
        @test pvalue(t) == 0.5
        b = LCABootstrap(m, 2, zeros(2, dof(m)), [true, true])
        @test b.n_boot == 2
    end
end
