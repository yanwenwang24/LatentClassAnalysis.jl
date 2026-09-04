using Test
using LatentClassAnalysis
using LinearAlgebra
using StableRNGs
using Statistics

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

const LCA = LatentClassAnalysis

@testset "Covariates" begin
    # Regression design shared by most tests: n = 3000, one N(0, 1) covariate,
    # log(π₂/π₁) = 0.5 + 1.0·x, six binary items with 0.8/0.2 response probabilities
    n = 3000
    rng = StableRNG(401)
    x = randn(rng, n)
    X = hcat(ones(n), x)
    beta_true = reshape([0.5, 1.0], 2, 1)
    items = [[0.8 0.2; 0.2 0.8] for _ in 1:6]
    y, classes = simulate_lca_reg(rng, X, beta_true, items)
    d = LCAData(y; covariates=x, covariate_names=[:x])
    @test hascovariates(d) && d.covariate_names == [:intercept, :x]

    # Three-class design with two covariates for the derivative checks
    n3 = 300
    rng3 = StableRNG(402)
    X3 = hcat(ones(n3), randn(rng3, n3), randn(rng3, n3))
    beta3 = [0.2 -0.3; 0.8 -0.5; -0.6 0.4]
    y3, _ = simulate_lca_reg(rng3, X3, beta3, THREE_CLASS_ITEMS)
    d3 = LCAData(y3; covariates=X3[:, 2:3])
    @test d3.covariate_names == [:intercept, :x1, :x2]

    @testset "Standardization" begin
        Xs, A = LCA._standardize(X)
        @test size(Xs) == (2, n) && size(A) == (2, 2)
        @test all(==(1.0), Xs[1, :])
        @test mean(Xs[2, :]) ≈ 0 atol = 1e-12
        @test std(Xs[2, :]) ≈ 1
        m, s = mean(x), std(x)
        @test A ≈ [1 -m / s; 0 1 / s]
        # The raw and standardized linear predictors agree for β_raw = A β_std
        bstd = [0.3, -0.7]
        @test X * (A * bstd) ≈ permutedims(Xs) * bstd
        Xs3, A3 = LCA._standardize(X3; names=d3.covariate_names)
        @test X3 * (A3 * [0.1, 0.2, 0.3]) ≈ permutedims(Xs3) * [0.1, 0.2, 0.3]
        @test LCA._standardize(ones(5, 1)) == (ones(1, 5), ones(1, 1))

        @test_throws ArgumentError LCA._standardize(hcat(ones(5), zeros(5)))            # constant covariate
        @test_throws "is constant" LCA._standardize(hcat(ones(5), fill(3.0, 5)))
        @test_throws ArgumentError LCA._standardize(hcat(ones(5), 1:5, 2 .* (1:5)))     # collinear
        @test_throws "collinear" LCA._standardize(hcat(ones(5), 1:5, 2 .* (1:5)))
        @test_throws ArgumentError LCA._standardize(hcat(ones(5), 1:5, 5:-1:1))          # x2 = 6 - x1
        @test_throws ArgumentError LCA._standardize(hcat(zeros(5), 1:5))                 # no intercept
        @test_throws ArgumentError LCA._standardize(hcat(ones(2), 1:2, [1.0, 2.0]))      # n < P
        # Through fit: the covariate name is in the message
        dconst = LCAData(y[1:20, :]; covariates=zeros(20), covariate_names=[:z])
        @test_throws ArgumentError fit(LCAModel, dconst, 2; rng=StableRNG(1))
        @test_throws "covariate z is constant" fit(LCAModel, dconst, 2; rng=StableRNG(1))
        @test_throws ArgumentError fit(LCAModel, LCAData(y; covariates=hcat(x, 2x)), 2; rng=StableRNG(1))
    end

    @testset "Workspace" begin
        ws = LCA.LCAWorkspace(d3, 3; covariates=true)
        @test ws.covariates && !ws.aggregated && ws.U == n3
        @test size(ws.Xt) == (3, n3) && ws.Xt == permutedims(X3)
        @test size(ws.Xst) == (3, n3) && size(ws.A) == (3, 3)
        @test all(abs.(mean(ws.Xst[2:3, :], dims=2)) .< 1e-12)
        @test size(ws.eta) == (3, n3) && size(ws.eta2) == (3, n3)
        ws2 = LCA.LCAWorkspace(ws)
        @test ws2.Xst === ws.Xst && ws2.A === ws.A && ws2.Xt === ws.Xt
        @test ws2.eta !== ws.eta && ws2.eta2 !== ws.eta2 && size(ws2.eta) == size(ws.eta)
        # Raw-scale workspace (used by predict)
        wr = LCA.LCAWorkspace(d3, 3; covariates=true, standardize=false)
        @test wr.Xst === wr.Xt && wr.A == I
        # Without covariates no linear predictors are stored
        w0 = LCA.LCAWorkspace(d3, 3)
        @test !w0.covariates && w0.aggregated && size(w0.eta) == (3, 0) && size(w0.Xst) == (1, w0.U)
        @test_throws ArgumentError LCA.LCAWorkspace(LCAData(y3), 3; covariates=true)
        # Parameters with coefficients need a covariate workspace
        θ = LCA._init_random(StableRNG(1), 3, d3.n_categories)
        θ.coefs = zeros(3, 3)
        @test_throws ArgumentError LCA.estep!(w0, θ)
        @test isfinite(LCA.estep!(ws, θ))
        @test all(abs.(sum(ws.post, dims=1) .- 1) .< 1e-12)
        # Zero coefficients give a uniform prior, so the posterior equals the plain one
        θu = LCA.LCAParams(fill(1 / 3, 3), θ.item_probs, nothing)
        wf = LCA.LCAWorkspace(d3, 3; aggregate=false)
        llu = LCA.estep!(wf, θu)
        @test LCA.estep!(ws, θ) ≈ llu
        @test maximum(abs.(ws.post .- wf.post)) < 1e-12
    end

    @testset "Gradient and Hessian of Q versus finite differences" begin
        ws = LCA.LCAWorkspace(d3, 3; covariates=true)
        θ = LCA._init_random(StableRNG(1), 3, d3.n_categories)
        θ.coefs = [0.0 0.3 -0.2; 0.0 0.5 0.1; 0.0 -0.4 0.6]
        LCA.estep!(ws, θ)
        post = copy(ws.post)
        β = [0.0 -0.2 0.4; 0.0 0.7 -0.3; 0.0 0.1 0.5]     # a point away from the current one
        P, K = 3, 3
        dim = (K - 1) * P
        g = zeros(dim)
        H = zeros(dim, dim)
        q0 = LCA._coef_derivatives!(g, H, ws, β)
        @test q0 == LCA._coef_objective(ws, β)
        @test ws.post == post                                  # scratch only
        # Q equals Σ_u Σ_k post log π with the prior from _class_prior on the standardized design
        prior = LCA._class_prior(β[:, 2:3], permutedims(ws.Xst))
        @test q0 ≈ sum(post[k, u] * log(prior[u, k]) for u in 1:n3, k in 1:K)

        perturb(B, i, h) = (C = copy(B); C[1 + (i - 1) % P, 2 + (i - 1) ÷ P] += h; C)
        h = 1e-5
        g_fd = [(LCA._coef_objective(ws, perturb(β, i, h)) -
                 LCA._coef_objective(ws, perturb(β, i, -h))) / 2h for i in 1:dim]
        @test isapprox(g, g_fd; rtol=1e-6)
        H_fd = zeros(dim, dim)
        for i in 1:dim
            gp, gm = zeros(dim), zeros(dim)
            LCA._coef_derivatives!(gp, zeros(dim, dim), ws, perturb(β, i, h))
            LCA._coef_derivatives!(gm, zeros(dim, dim), ws, perturb(β, i, -h))
            H_fd[:, i] = (gp - gm) / 2h
        end
        @test isapprox(H, H_fd; rtol=1e-6)
        @test isapprox(H, H'; rtol=1e-12)
        @test isposdef(Symmetric(-H))
        @test_throws DimensionMismatch LCA._coef_derivatives!(zeros(2), zeros(2, 2), ws, β)

        # One M-step increases Q and keeps class 1 as reference
        θn = LCA.LCAParams(fill(1 / 3, 3), θ.item_probs, copy(β))
        LCA._accumulate!(ws)
        LCA._update_coefs!(θn, ws)
        @test LCA._coef_objective(ws, θn.coefs) > q0
        @test all(iszero, θn.coefs[:, 1])
        @test θn.coefs != β
        @test sum(θn.class_probs) ≈ 1
        @test θn.class_probs ≈ vec(mean(LCA._class_prior(θn.coefs[:, 2:3], permutedims(ws.Xst)), dims=1))
        # At the maximizer of Q the step is (numerically) zero
        for _ in 1:30
            LCA._update_coefs!(θn, ws)
        end
        gopt = zeros(dim)
        LCA._coef_derivatives!(gopt, zeros(dim, dim), ws, θn.coefs)
        @test maximum(abs, gopt) < 1e-6 * n3
        before = copy(θn.coefs)
        LCA._update_coefs!(θn, ws)
        @test maximum(abs.(θn.coefs .- before)) < 1e-6
    end

    @testset "Monotone log-likelihood with covariates" begin
        ws = LCA.LCAWorkspace(d, 2; covariates=true)
        for s in 1:3
            θ = LCA._init_random(StableRNG(410 + s), 2, d.n_categories)
            θ.coefs = zeros(2, 2)
            trace = Float64[]
            ll, iters, conv = LCA._em!(θ, ws; max_iter=500, tol=1e-10, ll_trace=trace)
            @test all(diff(trace) .>= -1e-8 .* (1 .+ abs.(trace[2:end])))
            @test conv && trace[end] == ll && length(trace) == iters + 1
            @test all(iszero, θ.coefs[:, 1])
            @test sum(θ.class_probs) ≈ 1 atol = 1e-12
            @test LCA.estep!(ws, θ) == ll
            @test all(abs.(sum(ws.post, dims=1) .- 1) .< 1e-12)
        end
        # Three classes, two covariates, from a supplied start with non-zero slopes
        ws3 = LCA.LCAWorkspace(d3, 3; covariates=true)
        θ3 = LCA._init_random(StableRNG(5), 3, d3.n_categories)
        θ3.coefs = [0.0 0.3 -0.2; 0.0 0.5 0.1; 0.0 -0.4 0.6]
        trace3 = Float64[]
        ll3, _, conv3 = LCA._em!(θ3, ws3; max_iter=1000, tol=1e-10, ll_trace=trace3)
        @test conv3 && isfinite(ll3)
        @test all(diff(trace3) .>= -1e-8 .* (1 .+ abs.(trace3[2:end])))
    end

    @testset "Parameter recovery" begin
        m = @test_logs fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=8, n_final=2)
        @test m isa LCAModel && hascovariates(m)
        @test size(m.beta) == (2, 1)
        @test m.converged && LCA._clean(m.flags)
        @test !m.options.aggregate                             # disabled with covariates
        @test issorted(m.class_probs; rev=true)
        @test sum(m.class_probs) ≈ 1
        @test m.data === d
        @test size(m.posterior) == (n, 2)
        @test all(abs.(sum(m.posterior, dims=2) .- 1) .< 1e-12)

        perm = align_classes(m.item_probs, items)
        b = aligned_beta(m, perm)
        @test size(b) == (2, 1)
        @test maximum(abs.(b .- beta_true)) < 0.2
        e_item = maximum(maximum(abs.(m.item_probs[j][perm, :] .- items[j])) for j in 1:6)
        @test e_item < 0.05
        # The larger class (true class 2, since E[logistic(0.5 + x)] > 0.5) is class 1
        @test perm == [2, 1]
        @test m.beta[2, 1] < 0

        # Class sizes are the sample average of the covariate-specific prior
        prior = LCA._class_prior(m.beta, d.X)
        @test m.class_probs ≈ vec(mean(prior, dims=1))
        prior_true = LCA._class_prior(beta_true, X)
        @test maximum(abs.(m.class_probs[perm] .- vec(mean(prior_true, dims=1)))) < 0.05

        # Log-likelihood bookkeeping and consistency of the stored quantities
        @test m.loglik ≈ maximum(m.start_loglik) rtol = 1e-12
        @test length(m.start_loglik) == 8
        @test loglikelihood(m, d) ≈ m.loglik rtol = 1e-10
        @test maximum(abs.(predict(m, d) .- m.posterior)) < 1e-10
        ll_true = sum(log(sum(prior_true[i, k] * prod(items[j][k, y[i, j]] for j in 1:6) for k in 1:2))
                      for i in 1:n)
        @test m.loglik >= ll_true - 1e-6
        @test mean(invperm(perm)[classify(m)] .== classes) > 0.85

        # Posterior follows Bayes' rule with the row-specific prior
        for i in (1, 100, n)
            w = [prior[i, k] * prod(m.item_probs[j][k, y[i, j]] for j in 1:6) for k in 1:2]
            @test m.posterior[i, :] ≈ w ./ sum(w)
        end

        # Reproducible, threaded run identical, table entry point identical
        @test same_fit(m, fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=8, n_final=2))
        @test same_fit(m, fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=8, n_final=2, multithreaded=true))
        tbl = (item1=y[:, 1], item2=y[:, 2], item3=y[:, 3], item4=y[:, 4], item5=y[:, 5],
               item6=y[:, 6], x=x)
        mt = fit(LCAModel, tbl, [:item1, :item2, :item3, :item4, :item5, :item6], 2;
                 covariates=[:x], rng=StableRNG(1), n_starts=8, n_final=2)
        @test same_fit(mt, m) && mt.data.covariate_names == [:intercept, :x]
        ms = fit(LCAModel, d, 2:3; rng=StableRNG(1), n_starts=2, n_final=1)
        @test [size(q.beta) for q in ms] == [(2, 1), (2, 2)]
    end

    @testset "Nested unconditional model" begin
        m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=8, n_final=2)
        m0 = fit(LCAModel, d, 2; rng=StableRNG(1), covariates=false, n_starts=8, n_final=2)
        @test !hascovariates(m0) && size(m0.beta) == (1, 1)
        @test m0.options.aggregate
        @test m.loglik >= m0.loglik - 1e-6
        @test m.loglik - m0.loglik > 50                       # the covariate is strongly informative
        @test dof(m) == dof(m0) + 1
        # covariates=false on data with covariates is bitwise the fit on the stripped data
        ds = LCAData(d.y; n_categories=d.n_categories, item_names=d.item_names, item_levels=d.item_levels)
        @test !hascovariates(ds)
        @test same_fit(m0, fit(LCAModel, ds, 2; rng=StableRNG(1), n_starts=8, n_final=2))
        @test predict(m0, d) == predict(m0, ds)               # covariates of new data are ignored

        # A pure-noise covariate changes nothing of substance
        yn, _ = simulate_lca(StableRNG(420), 1500, TWO_CLASS_PROBS, TWO_CLASS_ITEMS)
        z = randn(StableRNG(421), 1500)
        dn = LCAData(yn; covariates=z, covariate_names=[:z])
        mn = fit(LCAModel, dn, 2; rng=StableRNG(2), n_starts=6, n_final=2)
        mu = fit(LCAModel, dn, 2; rng=StableRNG(2), covariates=false, n_starts=6, n_final=2)
        @test mn.loglik >= mu.loglik - 1e-6
        @test mn.loglik - mu.loglik < 3                       # LRT statistic 2Δll ≈ χ²₁
        permn = align_classes(mn.item_probs, mu.item_probs)
        @test maximum(maximum(abs.(mn.item_probs[j][permn, :] .- mu.item_probs[j])) for j in 1:6) < 0.02
        @test abs(mn.beta[2, 1]) < 0.3
        @test aligned_beta(mn, permn)[1, 1] ≈ log(mu.class_probs[2] / mu.class_probs[1]) atol = 0.05
        @test maximum(abs.(mn.class_probs[permn] .- mu.class_probs)) < 0.02
    end

    @testset "Scaling invariance" begin
        m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=8, n_final=2)
        d1000 = LCAData(y; covariates=1000 .* x, covariate_names=[:x])
        m1000 = fit(LCAModel, d1000, 2; rng=StableRNG(1), n_starts=8, n_final=2)
        @test m1000.beta[1, 1] ≈ m.beta[1, 1] rtol = 1e-6
        @test m1000.beta[2, 1] ≈ m.beta[2, 1] / 1000 rtol = 1e-6
        @test m1000.loglik ≈ m.loglik atol = 1e-6
        @test maximum(abs.(m1000.posterior .- m.posterior)) < 1e-8
        @test maximum(abs.(m1000.class_probs .- m.class_probs)) < 1e-8
        @test m1000.iterations == m.iterations
        # Shifting the covariate moves only the intercept
        dshift = LCAData(y; covariates=x .+ 10, covariate_names=[:x])
        mshift = fit(LCAModel, dshift, 2; rng=StableRNG(1), n_starts=8, n_final=2)
        @test mshift.beta[2, 1] ≈ m.beta[2, 1] rtol = 1e-6
        @test mshift.beta[1, 1] ≈ m.beta[1, 1] - 10 * m.beta[2, 1] rtol = 1e-6
        @test mshift.loglik ≈ m.loglik atol = 1e-6
    end

    @testset "Degrees of freedom and criteria" begin
        m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=4, n_final=1)
        @test dof(m) == 1 * 2 + 2 * 6
        @test aic(m) ≈ -2 * m.loglik + 2 * dof(m)
        @test bic(m) ≈ -2 * m.loglik + dof(m) * log(n)
        @test sbic(m) ≈ -2 * m.loglik + dof(m) * log((n + 2) / 24)
        m3 = fit(LCAModel, d3, 3; rng=StableRNG(1), n_starts=4, n_final=2)
        @test hascovariates(m3) && size(m3.beta) == (3, 2)
        @test dof(m3) == 2 * 3 + 3 * sum(d3.n_categories .- 1)
        @test issorted(m3.class_probs; rev=true)
        @test m3.class_probs ≈ vec(mean(LCA._class_prior(m3.beta, d3.X), dims=1))
        diag = diagnostics(m3)
        @test diag.dof == dof(m3) && diag.bic == bic(m3) && diag.n_classes == 3
        @test 0 <= entropy(m3) <= 1
        @test m3.loglik ≈ maximum(m3.start_loglik) rtol = 1e-12
        @test loglikelihood(m3, d3) ≈ m3.loglik rtol = 1e-10

        # A single class carries no coefficients but remembers the covariates
        m1 = fit(LCAModel, d, 1)
        @test size(m1.beta) == (2, 0) && hascovariates(m1)
        @test m1.class_probs == [1.0]
        @test dof(m1) == 6
        @test loglikelihood(m1, d) == m1.loglik
        @test predict(m1, d) == ones(n, 1)
        @test sprint(show, m1) isa String
    end

    @testset "Prediction on new data" begin
        m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=8, n_final=2)
        rng_h = StableRNG(430)
        n_h = 200
        xh = randn(rng_h, n_h)
        yh, _ = simulate_lca_reg(rng_h, hcat(ones(n_h), xh), beta_true, items)
        dh = LCAData(yh; covariates=xh, covariate_names=[:x])
        ph = predict(m, dh)
        @test size(ph) == (n_h, 2)
        @test all(abs.(sum(ph, dims=2) .- 1) .< 1e-12)
        @test all(0 .<= ph .<= 1)
        @test classify(m, dh) == [argmax(ph[i, :]) for i in 1:n_h]
        prior_h = LCA._class_prior(m.beta, dh.X)
        for i in (1, 50, n_h)
            w = [prior_h[i, k] * prod(m.item_probs[j][k, yh[i, j]] for j in 1:6) for k in 1:2]
            @test ph[i, :] ≈ w ./ sum(w)
        end
        @test loglikelihood(m, dh) ≈ sum(log(sum(prior_h[i, k] * prod(m.item_probs[j][k, yh[i, j]] for j in 1:6)
                                                for k in 1:2)) for i in 1:n_h)
        # Same responses, different covariate value: different posterior
        d_same = LCAData(yh[[1, 1], :]; n_categories=fill(2, 6), covariates=[-2.0, 2.0], covariate_names=[:x])
        p_same = predict(m, d_same)
        @test p_same[1, :] != p_same[2, :]
        # A single row (whose covariate column is constant) is fine
        @test size(predict(m, LCAData(yh[1:1, :]; n_categories=fill(2, 6), covariates=xh[1:1], covariate_names=[:x]))) == (1, 2)
        @test predict(m, LCAData(yh[1:1, :]; n_categories=fill(2, 6), covariates=xh[1:1], covariate_names=[:x]))[1, :] == ph[1, :]
        # A row with all responses missing gets its covariate-specific prior
        dmiss = LCAData(fill(0, 2, 6); n_categories=fill(2, 6), covariates=[-1.0, 1.0], covariate_names=[:x])
        @test predict(m, dmiss) ≈ LCA._class_prior(m.beta, dmiss.X)

        # Tables carry the covariate column; a missing column is a clear error
        cols = [Symbol("item", j) for j in 1:6]
        tbl = merge(NamedTuple{Tuple(cols)}(Tuple(yh[:, j] for j in 1:6)), (x=xh,))
        @test predict(m, tbl) == ph
        @test classify(m, tbl) == classify(m, dh)
        tbl_nox = NamedTuple{Tuple(cols)}(Tuple(yh[:, j] for j in 1:6))
        @test_throws ArgumentError predict(m, tbl_nox)
        @test_throws "absent from the table" predict(m, tbl_nox)
        @test_throws ArgumentError classify(m, tbl_nox)
        @test_throws ArgumentError predict(m, merge(tbl_nox, (x=["a" for _ in 1:n_h],)))   # not numeric

        # LCAData without the covariates, or with other names, is rejected
        @test_throws ArgumentError predict(m, LCAData(yh))
        @test_throws "fitted with the covariate(s) x" predict(m, LCAData(yh))
        @test_throws ArgumentError predict(m, LCAData(yh; covariates=xh, covariate_names=[:z]))
        @test_throws ArgumentError predict(m, LCAData(yh; covariates=hcat(xh, xh .^ 2), covariate_names=[:x, :x2]))
        @test_throws ArgumentError loglikelihood(m, LCAData(yh))
        @test_throws ArgumentError predict(m, yh)                # matrices are still rejected
    end

    @testset "Starting values" begin
        m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=8, n_final=2)
        m0 = fit(LCAModel, d, 2; rng=StableRNG(1), covariates=false, n_starts=8, n_final=2)
        # The fitted regression model reproduces its own solution
        mw = fit(LCAModel, d, 2; rng=StableRNG(1), init=m, n_starts=1)
        @test mw.loglik ≈ m.loglik rtol = 1e-10
        @test mw.iterations <= 2
        @test maximum(abs.(mw.beta .- m.beta)) < 1e-4
        # The unconditional model seeds the intercept only and converges to the same optimum
        mw0 = fit(LCAModel, d, 2; rng=StableRNG(1), init=m0, n_starts=1)
        @test mw0.loglik ≈ m.loglik rtol = 1e-8
        @test maximum(abs.(mw0.beta .- m.beta)) < 1e-2
        # LCAParams with coefficients on the standardized scale, and a NamedTuple
        ws = LCA.LCAWorkspace(d, 2; covariates=true)
        θ = LCA.LCAParams(copy(m.class_probs), [copy(P) for P in m.item_probs], ws.A \ hcat(zeros(2), m.beta))
        mp = fit(LCAModel, d, 2; rng=StableRNG(1), init=θ, n_starts=1)
        @test same_fit(mp, mw)
        mn = fit(LCAModel, d, 2; rng=StableRNG(1), init=(class_probs=m.class_probs, item_probs=m.item_probs), n_starts=1)
        @test mn.loglik ≈ m.loglik rtol = 1e-8
        # A regression model as the start of an unconditional fit contributes its class sizes
        mu = fit(LCAModel, d, 2; rng=StableRNG(1), init=m, covariates=false, n_starts=1)
        @test mu.loglik ≈ m0.loglik rtol = 1e-8
        @test !hascovariates(mu)
        # Wrong number of covariates or coefficient shape
        d2c = LCAData(y; covariates=hcat(x, x .^ 2))
        @test_throws ArgumentError fit(LCAModel, d2c, 2; init=m, n_starts=1)
        θbad = LCA.LCAParams(copy(m.class_probs), [copy(P) for P in m.item_probs], zeros(3, 2))
        @test_throws ArgumentError fit(LCAModel, d, 2; init=θbad, n_starts=1)
        @test_throws ArgumentError fit(LCAModel, d, 3; init=m, n_starts=1)
    end

    @testset "Class reordering keeps class 1 as reference" begin
        m3 = fit(LCAModel, d3, 3; rng=StableRNG(1), n_starts=4, n_final=2)
        @test issorted(m3.class_probs; rev=true)
        # Permuting the classes of the parameters and re-basing gives the same priors
        θ = LCA.LCAParams(copy(m3.class_probs), [copy(P) for P in m3.item_probs], hcat(zeros(3), m3.beta))
        prior = LCA._class_prior(θ.coefs[:, 2:3], d3.X)
        perm = [3, 1, 2]
        LCA._permute_classes!(θ, perm)
        @test all(iszero, θ.coefs[:, 1])
        @test θ.class_probs == m3.class_probs[perm]
        @test LCA._class_prior(θ.coefs[:, 2:3], d3.X) ≈ prior[:, perm]
        # Sorting by size is idempotent on a fitted model
        θs = LCA.LCAParams(copy(m3.class_probs), [copy(P) for P in m3.item_probs], hcat(zeros(3), m3.beta))
        @test LCA._sort_by_size!(θs) == [1, 2, 3]
        @test θs.coefs == hcat(zeros(3), m3.beta)
    end

    @testset "Quasi-complete separation" begin
        rng_s = StableRNG(440)
        ns = 600
        xs = randn(rng_s, ns)
        cls = 1 .+ (xs .> 0)                                  # the covariate determines the class
        sep_items = [[0.9 0.1; 0.1 0.9] for _ in 1:8]
        ys = [rand(rng_s) < sep_items[j][cls[i], 1] ? 1 : 2 for i in 1:ns, j in 1:8]
        dsep = LCAData(ys; covariates=xs, covariate_names=[:x])
        msep = @test_logs (:warn, r"covariate coefficients diverged \(quasi-complete separation\)") match_mode = :any fit(
            LCAModel, dsep, 2; rng=StableRNG(1), n_starts=4, n_final=2, max_iter=200)
        @test msep.flags.coef_divergence
        @test !LCA._clean(msep.flags)
        @test occursin("separation", sprint(show, msep))
        @test !any(isnan, msep.beta) && all(isfinite, msep.beta)
        @test isfinite(msep.loglik)
        @test !any(isnan, msep.posterior)
        @test all(abs.(sum(msep.posterior, dims=2) .- 1) .< 1e-12)
        @test all(isfinite, msep.class_probs) && sum(msep.class_probs) ≈ 1
        @test abs(msep.beta[2, 1]) > 20                        # the slope ran away
        @test !any(isnan, predict(msep, dsep))
        # The same data without covariates is a clean two-class fit
        mplain = fit(LCAModel, dsep, 2; rng=StableRNG(1), covariates=false, n_starts=4, n_final=2)
        @test !mplain.flags.coef_divergence
        @test msep.loglik > mplain.loglik
    end

    @testset "_class_prior" begin
        m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=4, n_final=1)
        pr = LCA._class_prior(m.beta, d.X)
        @test size(pr) == (n, 2)
        @test all(abs.(sum(pr, dims=2) .- 1) .< 1e-12)
        @test all(0 .<= pr .<= 1)
        @test pr[:, 2] ≈ 1 ./ (1 .+ exp.(-(d.X * m.beta[:, 1])))
        @test LCA._class_prior(zeros(2, 2), d.X) == fill(1 / 3, n, 3)
        @test LCA._class_prior(zeros(2, 0), d.X) == ones(n, 1)
        Xt = [1.0 -1.0; 1.0 0.0; 1.0 1.0]
        @test LCA._class_prior(reshape([0.0, 1.0], 2, 1), Xt)[:, 2] ≈ [1 / (1 + exp(1)), 0.5, 1 / (1 + exp(-1))]
        # Numerically stable for large linear predictors
        big = LCA._class_prior(reshape([0.0, 1000.0], 2, 1), Xt)
        @test big ≈ [1.0 0.0; 0.5 0.5; 0.0 1.0]
        @test !any(isnan, big)
        @test_throws DimensionMismatch LCA._class_prior(zeros(3, 1), d.X)
    end

    @testset "Display" begin
        m = fit(LCAModel, d, 2; rng=StableRNG(1), n_starts=4, n_final=1)
        s = sprint(show, m)
        @test occursin("covariates: x", s)
        @test occursin("class-membership coefficients (log-odds against class 1):", s)
        @test occursin(r"class 2", s)
        @test occursin(r"^\s+\(Intercept\)\s+-?\d+\.\d{4}$"m, s)     # the label of coefnames
        @test !occursin(r"^\s+intercept\b"m, s)
        @test occursin(r"^\s+x\s+-?\d+\.\d{4}$"m, s)
        @test occursin("fit flags: none", s)
        @test sprint(show, m; context=:compact => true) == "LCAModel(2 classes, 6 items, n = $n)"
        m3 = fit(LCAModel, d3, 3; rng=StableRNG(1), n_starts=4, n_final=2)
        s3 = sprint(show, m3)
        @test occursin("covariates: x1, x2", s3)
        @test occursin(r"class 2\s+class 3", s3)
        @test count(r"^\s+(\(Intercept\)|x1|x2)\s+-?\d+\.\d{4}\s+-?\d+\.\d{4}$"m, s3) == 3
        # show_profiles is unchanged
        out = sprint(io -> show_profiles(m; io=io))
        @test occursin("Latent Class Profiles", out) && !occursin("coefficients", out)
    end
end

@testset "init with a zero class probability under covariates stays finite" begin
    rng = StableRNG(11)
    n = 300
    y = rand(rng, 1:2, n, 4)
    x = randn(rng, n)
    d = LCAData(y; n_categories=fill(2, 4), covariates=reshape(x, :, 1), covariate_names=[:x])
    init = (class_probs=[1.0, 0.0], item_probs=[fill(0.5, 2, 2) for _ in 1:4])
    _, m = Test.collect_test_logs(() -> fit(LCAModel, d, 2; init=init, n_starts=1, rng=StableRNG(1)))
    @test all(isfinite, m.beta)
    @test all(isfinite, m.class_probs)
    @test all(isfinite, coef(m))
    # a plain model that lost a class can still seed a covariate fit
    _, mp = Test.collect_test_logs(() -> fit(LCAModel, d, 2; covariates=false, init=init, n_starts=1, rng=StableRNG(2)))
    _, m2 = Test.collect_test_logs(() -> fit(LCAModel, d, 2; init=mp, n_starts=1, rng=StableRNG(3)))
    @test all(isfinite, m2.beta) && all(isfinite, m2.class_probs)
end
