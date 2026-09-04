using Test
using LatentClassAnalysis
using LinearAlgebra
using StableRNGs
using Statistics
using StatsBase: CoefTable

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

const LCA = LatentClassAnalysis

@testset "Simulation and bootstrap" begin
    # Reference two-class model (n = 1000, six binary items), the same model fitted to data
    # with 10% missing responses, and a latent class regression with one covariate
    n = 1000
    y, _ = simulate_lca(StableRNG(601), n, TWO_CLASS_PROBS, TWO_CLASS_ITEMS)
    d = LCAData(y)
    m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=4, n_final=2)
    m1 = fit(LCAModel, d, 1)

    ymiss = mcar!(StableRNG(604), copy(y), 0.1)
    dmiss = LCAData(ymiss)
    mm = fit(LCAModel, dmiss, 2; rng=StableRNG(1), n_starts=2, n_final=1)

    x = randn(StableRNG(602), n)
    yc, _ = simulate_lca_reg(StableRNG(603), hcat(ones(n), x), reshape([0.3, 1.0], 2, 1),
                             [[0.8 0.2; 0.2 0.8] for _ in 1:6])
    dc = LCAData(yc; covariates=x, covariate_names=[:x])
    mc = fit(LCAModel, dc, 2; rng=StableRNG(1), n_starts=4, n_final=2)

    @testset "simulate" begin
        N = 100_000
        d_sim = simulate(m, N; rng=StableRNG(11))
        @test d_sim isa LCAData
        @test size(d_sim) == (N, 6)
        @test d_sim.n_categories == m.n_categories
        @test d_sim.item_names == d.item_names && d_sim.item_levels == d.item_levels
        @test !hascovariates(d_sim) && !hasmissing(d_sim)
        @test all(in(1:2), d_sim.y)

        # The internal form also returns the classes; the public form draws the same data
        d2, z = LCA._simulate(StableRNG(11), m, N, nothing, nothing)
        @test d2.y == d_sim.y
        @test length(z) == N && all(in(1:2), z)
        for k in 1:2
            @test abs(mean(z .== k) - m.class_probs[k]) < 0.01
            for j in 1:6, c in 1:2
                @test abs(mean(d2.y[z .== k, j] .== c) - m.item_probs[j][k, c]) < 0.01
            end
        end
        # A refit to the simulated data recovers the model within 0.01
        ms = fit(LCAModel, d_sim, 2; rng=StableRNG(1), n_starts=2, n_final=1, se=:none)
        perm = LCA._align_labels(ms, m)
        @test maximum(abs.(ms.class_probs[perm] .- m.class_probs)) < 0.01
        @test maximum(maximum(abs.(ms.item_probs[j][perm, :] .- m.item_probs[j])) for j in 1:6) < 0.01

        # Default n, reproducibility
        @test nobs(simulate(m; rng=StableRNG(1))) == n
        @test simulate(m, 50; rng=StableRNG(3)).y == simulate(m, 50; rng=StableRNG(3)).y
        @test simulate(m, 50; rng=StableRNG(3)).y != simulate(m, 50; rng=StableRNG(4)).y

        # The missing mask blanks cells of otherwise identical draws
        mask = falses(N, 6)
        mask[1:2:N, 2] .= true
        mask[1:10, 5] .= true
        dm = simulate(m, N; rng=StableRNG(11), missing_mask=mask)
        @test hasmissing(dm) && nmissing(dm) == [0, N ÷ 2, 0, 0, 10, 0]
        @test dm.y[.!mask] == d_sim.y[.!mask]
        @test all(iszero, dm.y[mask])
        @test_throws ArgumentError simulate(m, 10; missing_mask=falses(5, 6))
        @test_throws ArgumentError simulate(m, 0)
        @test_throws ArgumentError simulate(m, 10; X=randn(StableRNG(1), 10))    # no covariates in the model
        # The training missingness pattern
        dmm = simulate(mm; rng=StableRNG(1), missing_mask=mm.data.y .== 0)
        @test (dmm.y .== 0) == (ymiss .== 0)
        @test nmissing(dmm) == nmissing(dmiss)

        # Covariate model: class shares within covariate bins follow the membership model
        xs = randn(StableRNG(13), N)
        dcs, zc = LCA._simulate(StableRNG(14), mc, N, hcat(ones(N), xs), nothing)
        @test hascovariates(dcs) && dcs.covariate_names == [:intercept, :x]
        @test dcs.X[:, 2] == xs && all(isone, dcs.X[:, 1])
        prior = LCA._class_prior(mc.beta, dcs.X)
        edges = quantile(xs, 0:0.2:1)
        for b in 1:5
            sel = (xs .>= edges[b]) .& (b == 5 ? (xs .<= edges[6]) : (xs .< edges[b + 1]))
            @test count(sel) > 0.15 * N
            @test abs(mean(zc[sel] .== 2) - mean(prior[sel, 2])) < 0.02
        end
        # The class shares differ across bins, so the covariate matters
        lo = xs .< edges[2]
        hi = xs .>= edges[5]
        @test abs(mean(zc[lo] .== 2) - mean(zc[hi] .== 2)) > 0.2
        @test simulate(mc, N; rng=StableRNG(14), X=xs).y == dcs.y
        @test simulate(mc, N; rng=StableRNG(14), X=reshape(xs, :, 1)).y == dcs.y
        @test simulate(mc; rng=StableRNG(1)).X == dc.X                 # n == nobs reuses the design
        @test_throws ArgumentError simulate(mc, 10)                    # n ≠ nobs needs X
        @test_throws ArgumentError simulate(mc, 10; X=randn(StableRNG(1), 5))
        @test_throws ArgumentError simulate(mc, 10; X=randn(StableRNG(1), 10, 2))
        @test_throws ArgumentError simulate(mc, 10; X=[NaN; randn(StableRNG(1), 9)])
        @test_throws ArgumentError LCA._simulate(StableRNG(1), mc, 10, randn(StableRNG(1), 10, 3), nothing)
        @test_throws ArgumentError LCA._simulate(StableRNG(1), mc, 10, nothing, nothing)
        # Predictions on the simulated data use its covariates
        @test size(predict(mc, dcs)) == (N, 2)

        # Single class, polytomous items, and the categories of the model are kept
        d1 = simulate(m1, 500; rng=StableRNG(1))
        @test size(d1) == (500, 6) && d1.n_categories == fill(2, 6)
        y3, _ = simulate_lca(StableRNG(605), 500, THREE_CLASS_PROBS, THREE_CLASS_ITEMS)
        m3 = fit(LCAModel, LCAData(y3), 3; rng=StableRNG(1), n_starts=2, n_final=1, se=:none)
        d3 = simulate(m3, 20; rng=StableRNG(1))
        @test d3.n_categories == [2, 3, 4, 2, 3, 4]
        @test all(all(1 .<= d3.y[:, j] .<= d3.n_categories[j]) for j in 1:6)

        # Inverse-CDF draws
        @test LCA._rand_category(StableRNG(1), [1.0, 0.0, 0.0]) == 1
        @test LCA._rand_category(StableRNG(1), [0.0, 0.0, 1.0]) == 3
        @test LCA._rand_category(StableRNG(1), [0.0, 1.0, 0.0]) == 2
        draws = [LCA._rand_category(StableRNG(i), [0.2, 0.3, 0.5]) for i in 1:5000]
        @test abs(mean(draws .== 1) - 0.2) < 0.02 && abs(mean(draws .== 3) - 0.5) < 0.02
    end

    @testset "Label alignment" begin
        θ = LCA.LCAParams([0.5, 0.3, 0.2], [copy(P) for P in THREE_CLASS_ITEMS], nothing)
        for perm in ([2, 3, 1], [3, 1, 2], [1, 3, 2], [2, 1, 3], [1, 2, 3])
            θp = copy(θ)
            LCA._permute_classes!(θp, perm)             # new class k is old class perm[k]
            q = LCA._align_labels(θp, θ)
            @test q == invperm(perm)
            @test θp.class_probs[q] == θ.class_probs
            θa = copy(θp)
            @test LCA._align!(θa, θ) == q
            @test θa.class_probs == θ.class_probs
            @test all(θa.item_probs[j] == θ.item_probs[j] for j in 1:6)
        end
        # Perturbed replicate: still recovered
        θn = copy(θ)
        rngn = StableRNG(21)
        for P in θn.item_probs
            P .+= 0.03 .* (rand(rngn, size(P)...) .- 0.5)
            P ./= sum(P, dims=2)
        end
        θn.class_probs .= [0.28, 0.52, 0.2]              # sizes alone would mislead
        LCA._permute_classes!(θn, [2, 1, 3])
        @test LCA._align_labels(θn, θ) == [2, 1, 3]

        # With covariates the coefficients are re-based and the priors are unchanged
        X = hcat(ones(50), randn(StableRNG(22), 50))
        θc = LCA.LCAParams([0.5, 0.3, 0.2], [copy(P) for P in THREE_CLASS_ITEMS],
                           [0.0 0.4 -0.3; 0.0 1.0 -0.5])
        prior = LCA._class_prior(θc.coefs[:, 2:3], X)
        θcp = copy(θc)
        LCA._permute_classes!(θcp, [3, 1, 2])
        @test all(iszero, θcp.coefs[:, 1])
        @test LCA._class_prior(θcp.coefs[:, 2:3], X) ≈ prior[:, [3, 1, 2]]
        q = LCA._align!(θcp, θc)
        @test q == invperm([3, 1, 2])
        @test θcp.coefs ≈ θc.coefs
        @test LCA._class_prior(θcp.coefs[:, 2:3], X) ≈ prior
        @test θcp.class_probs == θc.class_probs

        # Models as arguments, a single class, and the greedy path for many classes
        @test LCA._align_labels(m, m) == [1, 2]
        @test LCA._align_labels(mc, mc) == [1, 2]
        @test LCA._align_labels(m1, m1) == [1]
        K8 = 8
        θ8 = LCA._init_random(StableRNG(23), K8, fill(3, 4))
        θ8.class_probs .= (1:K8) ./ sum(1:K8)
        perm8 = [3, 8, 1, 5, 2, 7, 4, 6]
        θ8p = copy(θ8)
        LCA._permute_classes!(θ8p, perm8)
        q8 = @test_logs (:warn, r"greedy assignment") LCA._align_labels(θ8p, θ8)
        @test q8 == invperm(perm8)
        @test LCA._assignment([0.0 1.0; 1.0 0.0]) == [1, 2]
        @test LCA._assignment([1.0 0.0; 0.0 1.0]) == [2, 1]
        @test LCA._assignment([1.0 1.0; 1.0 1.0]) == [1, 2]     # ties keep the identity
        @test_throws DimensionMismatch LCA._align_labels(θ, m)
        @test_throws DimensionMismatch LCA._align_labels(θ, LCA.LCAParams([0.5, 0.3, 0.2], THREE_CLASS_ITEMS[1:5], nothing))
    end

    @testset "bootstrap" begin
        b = @test_logs bootstrap(m; n_boot=20, rng=StableRNG(31))
        @test b isa LCABootstrap
        @test b.model === m && b.n_boot == 20
        @test size(b.coefs) == (20, dof(m)) && length(b.converged) == 20
        @test all(b.converged) && all(isfinite, b.coefs)
        # The replicates scatter around the estimate
        @test all(abs.(vec(mean(b.coefs, dims=1)) .- coef(m)) .< 3 .* stderror(m))

        # Bitwise reproducible; a different seed differs; threads agree with the serial run
        b2 = bootstrap(m; n_boot=20, rng=StableRNG(31))
        @test b2.coefs == b.coefs && b2.converged == b.converged
        @test bootstrap(m; n_boot=20, rng=StableRNG(32)).coefs != b.coefs
        bt = bootstrap(m; n_boot=20, rng=StableRNG(31), multithreaded=true)
        @test bt.coefs == b.coefs && bt.converged == b.converged

        # vcov, stderror, confint
        V = vcov(b)
        @test size(V) == (dof(m), dof(m)) && issymmetric(V)
        @test V ≈ cov(b.coefs; dims=1)
        se = stderror(b)
        @test all(isfinite, se) && all(se .> 0)
        @test se == sqrt.(diag(V))
        ci = confint(b)
        @test size(ci) == (dof(m), 2)
        @test all(ci[:, 1] .<= ci[:, 2])
        @test ci[:, 1] ≈ [quantile(b.coefs[:, i], 0.025) for i in 1:dof(m)]
        @test ci[:, 2] ≈ [quantile(b.coefs[:, i], 0.975) for i in 1:dof(m)]
        @test confint(b; method=:percentile) == ci
        cin = confint(b; method=:normal)
        @test cin ≈ hcat(coef(m) .- 1.959963984540054 .* se, coef(m) .+ 1.959963984540054 .* se)
        ci80 = confint(b; level=0.8)
        @test all((ci80[:, 2] .- ci80[:, 1]) .<= (ci[:, 2] .- ci[:, 1]))
        @test_throws ArgumentError confint(b; method=:bca)
        @test_throws ArgumentError confint(b; level=1.5)
        @test_throws ArgumentError confint(b; level=0.0)

        # coeftable
        ct = coeftable(b)
        @test ct isa CoefTable && length(ct) == dof(m)
        @test ct.cols[1] == coef(m) && ct.cols[2] == se
        @test ct.cols[3] ≈ coef(m) ./ se
        @test all(0 .<= ct.cols[4] .<= 1)
        @test ct.cols[5] == ci[:, 1] && ct.cols[6] == ci[:, 2]
        @test ct.rownms == coefnames(m)
        @test ct.colnms == ["Estimate", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
        @test ct.pvalcol == 4 && ct.teststatcol == 3
        @test length(coeftable(b; which=:class)) == 1
        @test length(coeftable(b; which=:items)) == 12
        @test coeftable(b; level=0.9).colnms[6] == "Upper 90%"
        @test_throws ArgumentError coeftable(b; which=:everything)
        @test occursin("class2: (Intercept)", sprint(show, MIME("text/plain"), ct))

        # profiles on the probability scale: percentile intervals of the replicates
        prof = profiles(b)
        @test length(prof) == 2 * 12
        @test all(0 <= r.lower <= r.upper <= 1 for r in prof)
        @test all(isfinite(r.se) && r.se > 0 for r in prof)
        @test [r.prob for r in prof] == [r.prob for r in profiles(m)]
        @test [(r.item, r.level, r.class) for r in prof] == [(r.item, r.level, r.class) for r in profiles(m)]
        # A binary row: the two levels mirror each other
        for j in 1:6, k in 1:2
            rows = [r for r in prof if r.item == Symbol("item", j) && r.class == k]
            @test rows[1].se ≈ rows[2].se
            @test rows[1].lower ≈ 1 - rows[2].upper
        end
        # ... and agree with the delta-method standard errors within a factor of two
        prof_h = profiles(m)
        @test all(0.5 < prof[i].se / prof_h[i].se < 2 for i in eachindex(prof))
        @test count(prof[i].lower <= prof_h[i].prob <= prof[i].upper for i in eachindex(prof)) == length(prof)
        prof90 = profiles(b; level=0.9)
        @test all(prof90[i].upper - prof90[i].lower <= prof[i].upper - prof[i].lower for i in eachindex(prof))
        @test_throws ArgumentError profiles(b; level=1.2)
        pc = profiles(b; classes=true)
        @test length(pc) == 2 + 24
        @test pc[1].item == :class && pc[1].level == "1" && pc[1].class == 1
        @test pc[1].prob == m.class_probs[1] && pc[2].prob == m.class_probs[2]
        @test 0 < pc[1].se < 0.05 && pc[1].lower <= pc[1].prob <= pc[1].upper
        @test pc[1].se ≈ pc[2].se
        @test pc[3:end] == prof
        pcm = profiles(m; classes=true)
        @test 0.5 < pc[1].se / pcm[1].se < 2

        # NaN safety: replicates with a non-finite coefficient are dropped
        bad = LCABootstrap(m, 20, copy(b.coefs), copy(b.converged))
        bad.coefs[3, 2] = NaN
        bad.coefs[7, 1] = -Inf
        Vb = vcov(bad)
        @test all(isfinite, Vb)
        @test Vb ≈ cov(b.coefs[setdiff(1:20, [3, 7]), :]; dims=1)
        @test all(isfinite, stderror(bad))
        @test all(isfinite, confint(bad))
        # ... column by column for the percentile bounds, row-wise for the covariance
        @test confint(bad)[2, :] ≈ [quantile(b.coefs[setdiff(1:20, 3), 2], 0.025), quantile(b.coefs[setdiff(1:20, 3), 2], 0.975)]
        @test confint(bad)[1, :] ≈ [quantile(b.coefs[setdiff(1:20, 7), 1], 0.025), quantile(b.coefs[setdiff(1:20, 7), 1], 0.975)]
        @test confint(bad)[3, :] == ci[3, :]
        pb = profiles(bad; classes=true)
        @test length(pb) == 26 && all(isfinite(r.se) for r in pb)
        empty = LCABootstrap(m, 2, fill(NaN, 2, dof(m)), [true, true])
        @test all(isnan, vcov(empty)) && all(isnan, stderror(empty)) && all(isnan, confint(empty))
        @test all(isnan(r.se) && isnan(r.lower) for r in profiles(empty))
        @test occursin("undefined", sprint(show, empty))
        @test occursin("excluded", sprint(show, bad))

        # Parametric bootstrap, missing data, extra starts
        bp = @test_logs bootstrap(m; n_boot=10, rng=StableRNG(33), parametric=true)
        @test size(bp.coefs) == (10, dof(m)) && all(isfinite, bp.coefs)
        @test bp.coefs != bootstrap(m; n_boot=10, rng=StableRNG(33)).coefs
        @test all(abs.(vec(mean(bp.coefs, dims=1)) .- coef(m)) .< 3 .* stderror(m))
        bm = bootstrap(mm; n_boot=5, rng=StableRNG(35), parametric=true)
        @test all(isfinite, bm.coefs) && size(bm.coefs) == (5, dof(mm))
        bm2 = bootstrap(mm; n_boot=5, rng=StableRNG(35))
        @test all(isfinite, bm2.coefs) && bm2.coefs != bm.coefs
        b3 = bootstrap(m; n_boot=5, rng=StableRNG(36), n_starts=3)
        @test size(b3.coefs) == (5, dof(m)) && all(b3.converged)

        # Covariate model: the class block is on the raw scale
        bc = @test_logs bootstrap(mc; n_boot=10, rng=StableRNG(37))
        @test size(bc.coefs) == (10, dof(mc)) && all(isfinite, bc.coefs)
        @test all(isfinite, stderror(bc)) && all(stderror(bc) .> 0)
        @test all(abs.(vec(mean(bc.coefs[:, 1:2], dims=1)) .- coef(mc)[1:2]) .< 4 .* stderror(mc)[1:2])
        @test all(0.4 .< stderror(bc)[1:2] ./ stderror(mc)[1:2] .< 2.5)
        pcc = profiles(bc; classes=true)
        @test all(isfinite(r.se) && r.se > 0 for r in pcc[1:2])   # class sizes get a bootstrap SE
        @test pcc[1].prob == mc.class_probs[1]
        @test pcc[1].lower <= pcc[1].prob <= pcc[1].upper
        @test length(coeftable(bc; which=:class)) == 2
        bcp = bootstrap(mc; n_boot=5, rng=StableRNG(38), parametric=true)
        @test all(isfinite, bcp.coefs)
        bct = bootstrap(mc; n_boot=5, rng=StableRNG(38), parametric=true, multithreaded=true)
        @test bct.coefs == bcp.coefs

        # Single class
        b1 = bootstrap(m1; n_boot=5, rng=StableRNG(39))
        @test size(b1.coefs) == (5, 6) && all(isfinite, stderror(b1))
        p1 = profiles(b1; classes=true)
        @test p1[1].se == 0.0 && p1[1].lower == 1.0 && p1[1].upper == 1.0

        # A rare binary covariate is constant in some resamples: those replicates fail
        yr, _ = simulate_lca(StableRNG(40), 60, TWO_CLASS_PROBS, TWO_CLASS_ITEMS)
        xr = zeros(60)
        xr[1:2] .= 1
        mr = @test_logs (:warn, r"on the boundary") match_mode = :any fit(
            LCAModel, LCAData(yr; covariates=xr), 2; rng=StableRNG(1), n_starts=2, n_final=1, se=:none)
        br = @test_logs (:warn, r"replicate fit\(s\) failed") match_mode = :any bootstrap(mr; n_boot=20, rng=StableRNG(41))
        @test any(isnan, br.coefs) && !all(br.converged)
        @test all(i -> all(isnan, br.coefs[i, :]) || all(isfinite, br.coefs[i, :]), 1:20)
        free_r = LatentClassAnalysis.ParamLayout(mr).free   # boundary parameters are NaN by design
        @test all(isfinite, vcov(br)[free_r, free_r])
        @test occursin("excluded", sprint(show, br))

        # Argument validation and display
        @test_throws ArgumentError bootstrap(m; n_boot=1)
        @test_throws ArgumentError bootstrap(m; n_starts=0)
        s = sprint(show, b)
        @test occursin("LCABootstrap with 20 replicates of a 2-class model (6 items, n = $n)", s)
        @test occursin("converged replicate fits: 20 of 20", s)
        @test occursin("median", s)
        @test sprint(show, b; context=:compact => true) == "LCABootstrap(20 replicates, 2 classes)"
    end

    @testset "bootstrap_lrt" begin
        # (a two-class fit to data from one class converges sublinearly, so replicate fits
        # may hit max_iter; that is warned about and does not affect the statistics)
        nonconv = (:warn, r"replicate fits did not converge")
        t = @test_logs nonconv match_mode = :any bootstrap_lrt(m1, m; n_boot=5, rng=StableRNG(41))
        @test t isa BootstrapLRT
        @test t.null === m1 && t.alternative === m
        @test t.statistic == 2 * (loglikelihood(m) - loglikelihood(m1))
        @test length(t.replicates) == 5 && t.n_boot == 5
        @test all(t.replicates .>= -1e-6) && t.n_negative == 0
        @test t.pvalue == (1 + count(t.replicates .>= t.statistic)) / 6
        @test pvalue(t) == t.pvalue == 1 / 6       # no replicate from one class reaches T_obs
        @test length(t.converged) == 5 && t.converged isa Vector{Bool}
        @test all(t.replicates .< t.statistic)
        @test all(0 .<= t.replicates .< 30)

        # Reproducible; threads agree with the serial run
        t2 = @test_logs nonconv match_mode = :any bootstrap_lrt(m1, m; n_boot=5, rng=StableRNG(41))
        @test t2.replicates == t.replicates && t2.pvalue == t.pvalue
        tt = @test_logs nonconv match_mode = :any bootstrap_lrt(m1, m; n_boot=5, rng=StableRNG(41), multithreaded=true)
        @test tt.replicates == t.replicates
        t42 = @test_logs nonconv match_mode = :any bootstrap_lrt(m1, m; n_boot=5, rng=StableRNG(42))
        @test t42.replicates != t.replicates

        # 2 vs 3 on two-class data
        m3 = @test_logs (:warn, r"on the boundary") match_mode = :any fit(
            LCAModel, d, 3; rng=StableRNG(1), n_starts=8, n_final=4, se=:none)
        t23 = bootstrap_lrt(m, m3; n_boot=5, rng=StableRNG(43))
        @test all(t23.replicates .>= -1e-6) && t23.n_negative == 0
        @test 0 < t23.pvalue <= 1
        @test t23.statistic == 2 * (loglikelihood(m3) - loglikelihood(m))
        @test all(isfinite, t23.replicates)

        # Validation
        @test_throws ArgumentError bootstrap_lrt(m1, m3)                 # K + 2
        @test_throws ArgumentError bootstrap_lrt(m, m1)                  # reversed
        m3other = fit(LCAModel, LCAData(y[1:500, :]), 3; rng=StableRNG(1), n_starts=2, n_final=1, se=:none)
        @test_throws ArgumentError bootstrap_lrt(m, m3other)             # other data
        mc_plain = fit(LCAModel, dc, 2; rng=StableRNG(1), covariates=false, n_starts=2, n_final=1, se=:none)
        @test_throws ArgumentError bootstrap_lrt(fit(LCAModel, dc, 1), mc_plain)   # covariate mismatch
        @test_throws ArgumentError bootstrap_lrt(m1, m; n_boot=0)
        @test_throws ArgumentError bootstrap_lrt(m1, m; n_starts_boot=0)
        @test_throws ArgumentError bootstrap_lrt(m1, m; n_final_boot=0)
        # The same data by value is accepted
        m1c = fit(LCAModel, LCAData(copy(y)), 1)
        @test bootstrap_lrt(m1c, m; n_boot=2, rng=StableRNG(41)).replicates == bootstrap_lrt(m1, m; n_boot=2, rng=StableRNG(41)).replicates

        # A non-replicated best log-likelihood of the alternative warns
        malt = LCAModel(m.n_classes, m.n_items, m.n_categories, m.class_probs, m.item_probs, m.beta,
                        m.data, m.posterior, m.loglik, m.converged, m.iterations, m.start_loglik,
                        m.options, m.vcov, LCA.FitFlags(true, 0, Int[], false, false))
        tw = @test_logs (:warn, r"reached by only one of its continued starts") bootstrap_lrt(m1, malt; n_boot=2, rng=StableRNG(41))
        @test tw.statistic == t.statistic
        # A worse alternative warns too
        mbad = LCAModel(m.n_classes, m.n_items, m.n_categories, m.class_probs, m.item_probs, m.beta,
                        m.data, m.posterior, m1.loglik - 1, m.converged, m.iterations, m.start_loglik,
                        m.options, m.vcov, m.flags)
        tb = @test_logs (:warn, r"lower log-likelihood than the 1-class model") bootstrap_lrt(m1, mbad; n_boot=2, rng=StableRNG(41))
        @test tb.statistic == -2.0 && tb.pvalue == 1.0

        # show
        s = sprint(show, t)
        @test occursin("Bootstrap likelihood-ratio test of 1 against 2 classes (n = $n)", s)
        @test occursin("statistic 2(ll_2 - ll_1)", s)
        @test occursin("bootstrap p-value: 0.1667  (5 replicates; resolution 1/6 = 0.1667)", s)
        @test occursin("median", s) && !occursin("negative", s)
        @test sprint(show, t; context=:compact => true) == "BootstrapLRT(1 vs 2 classes, p = 0.1667)"
        tneg = BootstrapLRT(m1, m, 3.0, [-1.0, 5.0], 2 / 3, 2, 1, [true, false])
        sn = sprint(show, tneg)
        @test occursin("negative replicate statistics: 1", sn) && occursin("not converged: 1", sn)

        # The convenience form equals the two-model form with the same generator
        tc = bootstrap_lrt(d, 1; rng=StableRNG(44), n_boot=3, n_starts=4, n_final=2)
        rng = StableRNG(44)
        null = fit(LCAModel, d, 1; rng=rng, n_starts=4, n_final=2)
        alt = fit(LCAModel, d, 2; rng=rng, n_starts=4, n_final=2)
        tm = bootstrap_lrt(null, alt; rng=rng, n_boot=3)
        @test tc.replicates == tm.replicates && tc.statistic == tm.statistic
        @test tc.null.n_classes == 1 && tc.alternative.n_classes == 2
        @test same_fit(tc.alternative, alt)
        @test tc.alternative.vcov !== nothing          # the fitted models are full models

        # Covariates and missing data
        tcv = bootstrap_lrt(fit(LCAModel, dc, 1), mc; n_boot=3, rng=StableRNG(45))
        @test all(tcv.replicates .>= -1e-6) && hascovariates(tcv.null)
        @test tcv.pvalue == 1 / 4
        mc3 = fit(LCAModel, dc, 3; rng=StableRNG(1), n_starts=4, n_final=2, se=:none)
        tcv23 = bootstrap_lrt(mc, mc3; n_boot=2, rng=StableRNG(46))
        @test all(isfinite, tcv23.replicates)
        tmiss = bootstrap_lrt(fit(LCAModel, dmiss, 1), mm; n_boot=3, rng=StableRNG(47))
        @test all(isfinite, tmiss.replicates) && all(tmiss.replicates .>= -1e-6)
        @test tmiss.pvalue == 1 / 4
    end

    if lowercase(get(ENV, "LCA_SLOW_TESTS", "true")) in ("true", "1", "yes")
        @testset "slow: bootstrap versus observed-information standard errors" begin
            b = bootstrap(m; n_boot=100, rng=StableRNG(51))
            @test all(b.converged)
            se_b = stderror(b)
            se_h = stderror(m)
            @test all(isfinite, se_h)
            ratio = se_b ./ se_h
            @test all(0.75 .< ratio .< 1.3)
            ci = confint(b)
            c = coef(m)
            @test count(ci[:, 1] .<= c .<= ci[:, 2]) >= 0.8 * dof(m)
            @info "bootstrap/Hessian standard-error ratio" extrema(ratio) count(ci[:, 1] .<= c .<= ci[:, 2]) dof(m)
        end

        @testset "slow: bootstrap likelihood-ratio test" begin
            y8, _ = simulate_lca(StableRNG(52), 1000, [0.6, 0.4], [[0.8 0.2; 0.2 0.8] for _ in 1:6])
            d8 = LCAData(y8)
            models = fit(LCAModel, d8, 1:3; rng=StableRNG(1), se=:none)
            t12 = bootstrap_lrt(models[1], models[2]; n_boot=39, rng=StableRNG(53))
            @test t12.pvalue < 0.05
            @test t12.pvalue == 1 / 40
            @test all(t12.replicates .>= -1e-6) && t12.n_negative == 0
            @test t12.statistic == 2 * (loglikelihood(models[2]) - loglikelihood(models[1]))
            @test length(t12.replicates) == 39 && length(t12.converged) == 39
            t23 = bootstrap_lrt(models[2], models[3]; n_boot=39, rng=StableRNG(54))
            @test t23.pvalue > 0.05
            @test all(t23.replicates .>= -1e-6) && t23.n_negative == 0
            @test t23.statistic == 2 * (loglikelihood(models[3]) - loglikelihood(models[2]))
            @test length(t23.replicates) == 39
            @info "bootstrap likelihood-ratio test p-values" t12.pvalue t23.pvalue
        end
    end
end

@testset "bootstrap masks parameters on the boundary of the reference model" begin
    rng = StableRNG(62)
    n = 400
    z = rand(rng, 1:2, n)
    # item 1 never takes category 2 in class 1, so that cell ends on the boundary
    y = Matrix{Int}(undef, n, 5)
    for i in 1:n, j in 1:5
        p = z[i] == 1 ? (j == 1 ? 1.0 : 0.85) : 0.15
        y[i, j] = rand(rng) < p ? 1 : 2
    end
    d = LCAData(y; n_categories=fill(2, 5))
    _, m = Test.collect_test_logs(() -> fit(LCAModel, d, 2; rng=StableRNG(1)))
    free = LatentClassAnalysis.ParamLayout(m).free
    @test !all(free)
    b = @test_logs (:warn, r"on the boundary or an empty class") match_mode = :any bootstrap(m; n_boot=10, rng=StableRNG(2))
    se = stderror(b)
    @test all(isnan, se[.!free])
    @test all(isfinite, se[free])
    # The printed summary covers the finite standard errors and counts the boundary ones
    sb = sprint(show, b)
    @test occursin("bootstrap standard errors of $(count(free)) parameters", sb)
    @test occursin("(NaN for $(count(.!free)) parameter$(count(.!free) == 1 ? "" : "s") on the boundary)", sb)
    ci = confint(b)
    @test all(isnan, ci[.!free, :])
    @test all(isfinite, ci[free, :])
    @test all(isnan, confint(b; method=:normal)[.!free, :])
    ct = coeftable(b)
    @test length(ct.rownms) == length(se)
    # the probability-scale summary is not masked
    pr = profiles(b)
    @test all(r -> isfinite(r.prob), pr)
end

@testset "exact label assignment for 4 to 7 classes" begin
    # Branch-and-bound assignment against brute force over all permutations
    for K in 4:7
        D = rand(StableRNG(100 + K), K, K)
        perms = _permutations(collect(1:K))
        best = perms[argmin([sum(D[k, p[k]] for k in 1:K) for p in perms])]
        @test LCA._assignment(D) == best
    end
end

@testset "parametric bootstrap and BLRT with covariates and missing data" begin
    rng = StableRNG(64)
    n = 300
    x = randn(rng, n)
    y, _ = simulate_lca_reg(rng, hcat(ones(n), x), reshape([0.3, 0.8], 2, 1), TWO_CLASS_ITEMS)
    mcar!(rng, y, 0.15)
    d = LCAData(y; n_categories=fill(2, 6), covariates=x, covariate_names=[:x])
    _, m = Test.collect_test_logs(() -> fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=2, n_final=1, se=:none))
    ds = simulate(m, n; rng=StableRNG(2), X=x, missing_mask=iszero.(d.y))
    @test nmissing(ds) == nmissing(d) && hascovariates(ds) && ds.X == d.X
    _, b = Test.collect_test_logs(() -> bootstrap(m; n_boot=3, rng=StableRNG(3), parametric=true))
    @test size(b.coefs) == (3, dof(m)) && all(isfinite, b.coefs)
    _, m1 = Test.collect_test_logs(() -> fit(LCAModel, d, 1))
    _, t = Test.collect_test_logs(() -> bootstrap_lrt(m1, m; n_boot=2, rng=StableRNG(4)))
    @test length(t.replicates) == 2 && all(isfinite, t.replicates)
end
