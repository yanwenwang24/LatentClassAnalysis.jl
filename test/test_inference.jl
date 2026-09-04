using Test
using LatentClassAnalysis
using LinearAlgebra
using StableRNGs
using Statistics
using StatsBase: CoefTable

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

const LCA = LatentClassAnalysis

# Analytic score at a parameter vector v versus central finite differences of the
# log-likelihood returned by estep!. Returns (g, g_fd).
function score_vs_fd(ws, θ, layout; h=1e-5)
    v = LCA._pack(θ, layout)
    θb = LCA._params_buffer(layout)
    g = zeros(layout.n_total)
    LCA._score!(g, v, layout, θb, ws)
    g_fd = similar(g)
    for p in 1:layout.n_total
        vp = copy(v)
        vp[p] += h
        LCA._unpack!(θb, vp, layout)
        lp = LCA.estep!(ws, θb)
        vp[p] -= 2h
        LCA._unpack!(θb, vp, layout)
        lm = LCA.estep!(ws, θb)
        g_fd[p] = (lp - lm) / 2h
    end
    return g, g_fd
end

# Score of a fitted model at its own parameters (internal scale).
function fitted_score(m::LCAModel)
    ws, θ, layout = LCA._model_params(m)
    g = zeros(layout.n_total)
    LCA._score!(g, LCA._pack(θ, layout), layout, LCA._params_buffer(layout), ws)
    return g
end

@testset "Inference" begin
    # Three-class design (items with 2, 3 and 4 categories): complete, 20% MCAR, covariates
    n = 500
    y, _ = simulate_lca(StableRNG(501), n, THREE_CLASS_PROBS, THREE_CLASS_ITEMS)
    d = LCAData(y)
    dm = LCAData(mcar!(StableRNG(502), copy(y), 0.2); n_categories=d.n_categories)
    X = randn(StableRNG(503), n, 2)
    dc = LCAData(y; covariates=X, covariate_names=[:age, :female])

    # Well-separated two-class design, n = 4000, six binary items with 0.85/0.15
    n2 = 4000
    p2 = 0.85
    y2, _ = simulate_lca(StableRNG(504), n2, [0.6, 0.4], [[p2 1-p2; 1-p2 p2] for _ in 1:6])
    d2 = LCAData(y2)
    m2 = @test_logs fit(LCAModel, d2, 2; rng=StableRNG(1), n_starts=4, n_final=2)
    mc = fit(LCAModel, dc, 3; rng=StableRNG(1), n_starts=4, n_final=2)

    @testset "ParamLayout, pack and unpack" begin
        layout = LCA.ParamLayout(m2)
        @test layout.n_total == dof(m2) == 13
        @test layout.n_class == 1 && layout.P == 1 && !layout.covariates
        @test layout.ref_cat == [[argmax(m2.item_probs[j][k, :]) for k in 1:2] for j in 1:6]
        @test all(layout.free)
        @test LCA._row_indices(layout, 1, 1) == 2:2 && LCA._row_indices(layout, 6, 2) == 13:13

        # Random parameters, no covariates: exact round trip
        θ = LCA._init_random(StableRNG(11), 3, d.n_categories)
        θ.class_probs .= [0.5, 0.3, 0.2]
        lay = LCA.ParamLayout(θ.class_probs, θ.item_probs, nothing)
        @test lay.n_total == 2 + 3 * sum(d.n_categories .- 1)
        @test lay.n_class == 2
        v = LCA._pack(θ, lay)
        @test v[1] ≈ log(0.3 / 0.5) && v[2] ≈ log(0.2 / 0.5)
        r = lay.ref_cat[2][1]
        @test v[LCA._row_indices(lay, 2, 1)] ≈ [log(θ.item_probs[2][1, c] / θ.item_probs[2][1, r]) for c in 1:3 if c != r]
        θc = copy(θ)
        θc.class_probs .= 0
        foreach(P -> fill!(P, 0.0), θc.item_probs)
        LCA._unpack!(θc, v, lay)
        @test maximum(abs.(θc.class_probs .- θ.class_probs)) < 1e-12
        @test maximum(maximum(abs.(θc.item_probs[j] .- θ.item_probs[j])) for j in 1:6) < 1e-12
        @test all(all(abs.(sum(P, dims=2) .- 1) .< 1e-12) for P in θc.item_probs)
        @test LCA._pack(θc, lay) ≈ v atol = 1e-12

        # With covariates: coefficients on the standardized scale, column 1 zero
        ws = LCA.LCAWorkspace(dc, 3; covariates=true)
        θr = copy(θ)
        θr.coefs = hcat(zeros(3), 0.3 .* randn(StableRNG(12), 3, 2))
        layr = LCA.ParamLayout(θr.class_probs, θr.item_probs, θr.coefs)
        @test layr.covariates && layr.P == 3 && layr.n_class == 6
        @test layr.n_total == 6 + 3 * sum(d.n_categories .- 1)
        vr = LCA._pack(θr, layr)
        @test vr[1:6] == vec(θr.coefs[:, 2:3])
        θrc = copy(θr)
        fill!(θrc.coefs, 1.0)
        LCA._unpack!(θrc, vr, layr)
        @test maximum(abs.(θrc.coefs .- θr.coefs)) < 1e-12
        @test all(iszero, θrc.coefs[:, 1])
        @test maximum(maximum(abs.(θrc.item_probs[j] .- θr.item_probs[j])) for j in 1:6) < 1e-12

        # Boundary parameters are not free; the reference is the modal category
        θb = copy(θ)
        θb.item_probs[1][2, :] .= [1e-10, 1 - 1e-10]
        θb.class_probs .= [0.6, 0.4 - 1e-8, 1e-8]
        layb = LCA.ParamLayout(θb.class_probs, θb.item_probs, nothing)
        @test layb.ref_cat[1][2] == 2
        @test !layb.free[LCA._row_indices(layb, 1, 2)[1]]
        @test !layb.free[2] && layb.free[1]
        @test count(!, layb.free) == 2
        vb = LCA._pack(θb, layb)
        θbc = copy(θb)
        LCA._unpack!(θbc, vb, layb)
        @test θbc.item_probs[1][2, 1] ≈ 1e-10 rtol = 1e-8
        @test θbc.class_probs ≈ θb.class_probs rtol = 1e-8

        # Dimension checks
        @test_throws DimensionMismatch LCA._unpack!(copy(θ), v[1:end-1], lay)
        @test_throws ArgumentError LCA._pack(θ, layr)                       # no coefficients
        @test_throws ArgumentError LCA._unpack!(copy(θ), vr, layr)
        @test_throws DimensionMismatch LCA._pack(θr, LCA.ParamLayout(θr.class_probs, θr.item_probs, zeros(2, 3)))
        @test_throws DimensionMismatch LCA._pack(LCA._init_random(StableRNG(1), 2, d.n_categories), lay)
        @test_throws DimensionMismatch LCA.ParamLayout(θ.class_probs, θ.item_probs, zeros(3, 2))
    end

    @testset "Analytic score versus finite differences" begin
        θ = LCA._init_random(StableRNG(21), 3, d.n_categories)
        θ.class_probs .= [0.5, 0.3, 0.2]
        lay = LCA.ParamLayout(θ.class_probs, θ.item_probs, nothing)
        for (data, label) in ((d, "complete"), (dm, "missing"))
            ws = LCA.LCAWorkspace(data, 3)
            g, g_fd = score_vs_fd(ws, θ, lay)
            @test isapprox(g, g_fd; rtol=1e-6)
            @test length(g) == lay.n_total
            @test maximum(abs.(g .- g_fd)) < 1e-6 * maximum(abs, g_fd)
        end
        # Aggregated and plain workspaces give the same score
        g_agg, _ = score_vs_fd(LCA.LCAWorkspace(d, 3), θ, lay)
        g_full, _ = score_vs_fd(LCA.LCAWorkspace(d, 3; aggregate=false), θ, lay)
        @test g_agg ≈ g_full rtol = 1e-10

        # Covariates: the coefficient block is the gradient of Q at the current posterior
        ws = LCA.LCAWorkspace(dc, 3; covariates=true)
        θr = copy(θ)
        θr.coefs = hcat(zeros(3), [0.2 -0.4; 0.5 0.1; -0.3 0.6])
        layr = LCA.ParamLayout(θr.class_probs, θr.item_probs, θr.coefs)
        g, g_fd = score_vs_fd(ws, θr, layr)
        @test isapprox(g, g_fd; rtol=1e-6)
        @test maximum(abs.(g .- g_fd)) < 1e-6 * maximum(abs, g_fd)
        # Missing data with covariates
        dmc = LCAData(dm.y; n_categories=d.n_categories, covariates=X)
        gm, gm_fd = score_vs_fd(LCA.LCAWorkspace(dmc, 3; covariates=true), θr, layr)
        @test isapprox(gm, gm_fd; rtol=1e-6)

        # The score vanishes at a converged fit
        mt = fit(LCAModel, d2, 2; rng=StableRNG(1), n_starts=2, n_final=1, tol=1e-12)
        @test maximum(abs, fitted_score(mt)) < 1e-6 * nobs(mt)
        mct = fit(LCAModel, dc, 3; rng=StableRNG(1), init=mc, n_starts=1, tol=1e-12)
        @test maximum(abs, fitted_score(mct)) < 1e-6 * nobs(mct)
        m1 = fit(LCAModel, d, 1)
        @test maximum(abs, fitted_score(m1)) < 1e-8 * nobs(m1)

        # Wrong workspace or buffer sizes
        @test_throws DimensionMismatch LCA._score!(zeros(3), LCA._pack(θ, lay), lay, LCA._params_buffer(lay), LCA.LCAWorkspace(d, 3))
        @test_throws DimensionMismatch LCA._score!(zeros(lay.n_total), LCA._pack(θ, lay), lay, LCA._params_buffer(lay), LCA.LCAWorkspace(d, 2))
    end

    @testset "coef and coefnames" begin
        for m in (m2, mc, fit(LCAModel, d, 1), fit(LCAModel, dc, 1))
            @test length(coef(m)) == dof(m) == length(coefnames(m))
            @test allunique(coefnames(m))
        end
        c = coef(m2)
        @test c[1] ≈ log(m2.class_probs[2] / m2.class_probs[1])
        @test c[1] == m2.beta[1, 1]
        names = coefnames(m2)
        @test names[1] == "class2: (Intercept)"
        @test names[2] == "item1[2/1]|class1" || names[2] == "item1[1/2]|class1"
        @test count(endswith("|class1"), names) == 6 && count(endswith("|class2"), names) == 6
        for j in 1:6, k in 1:2
            r = argmax(m2.item_probs[j][k, :])
            o = 3 - r
            @test c[1 + (j - 1) * 2 + k] ≈ log(m2.item_probs[j][k, o] / m2.item_probs[j][k, r])
            @test names[1 + (j - 1) * 2 + k] == "item$j[$o/$r]|class$k"
        end
        # Covariate model: the class block is vec(beta) on the raw scale, class 2 first
        cc = coef(mc)
        @test cc[1:6] == vec(mc.beta)
        cn = coefnames(mc)
        @test cn[1:6] == ["class2: (Intercept)", "class2: age", "class2: female",
                          "class3: (Intercept)", "class3: age", "class3: female"]
        @test occursin("class2: age", join(cn, " "))
        @test any(endswith("|class1"), cn) && any(endswith("|class3"), cn)
        # Level labels appear in the item names
        tbl = (edu=[("low", "middle", "high")[c] for c in y[:, 2]], a=y[:, 1], b=y[:, 4])
        dt = prepare_data(tbl, [:edu, :a, :b]; levels=Dict(:edu => ["low", "middle", "high"]))
        mt = fit(LCAModel, dt, 2; rng=StableRNG(1), n_starts=2, n_final=1)
        nt = coefnames(mt)
        @test count(startswith("edu["), nt) == 4
        @test all(occursin(r"^edu\[(low|middle|high)/(low|middle|high)\]\|class[12]$", s) for s in nt if startswith(s, "edu["))
        @test nt[1] == "class2: (Intercept)"
        # Single class: no class block
        m1 = fit(LCAModel, d, 1)
        @test length(coef(m1)) == sum(d.n_categories .- 1)
        @test all(endswith("|class1"), coefnames(m1))
        @test !any(startswith("class"), coefnames(m1))
    end

    @testset "vcov, stderror, confint" begin
        V = vcov(m2)
        @test size(V) == (dof(m2), dof(m2))
        @test issymmetric(V)
        @test isposdef(V)
        @test V == m2.vcov && V !== m2.vcov
        @test all(isfinite, V)
        se = stderror(m2)
        @test se == sqrt.(diag(V))
        @test all(0 .< se .< 0.2)
        ci95 = confint(m2)
        ci90 = confint(m2; level=0.9)
        ci99 = confint(m2; level=0.99)
        @test size(ci95) == (dof(m2), 2)
        @test ci95 ≈ hcat(coef(m2) .- 1.959963984540054 .* se, coef(m2) .+ 1.959963984540054 .* se)
        @test all(ci95[:, 1] .< coef(m2) .< ci95[:, 2])
        w(ci) = ci[:, 2] .- ci[:, 1]
        @test all(w(ci90) .< w(ci95) .< w(ci99))
        @test_throws ArgumentError confint(m2; level=1.0)
        @test_throws ArgumentError confint(m2; level=0.0)
        # Covariate model
        Vc = vcov(mc)
        @test issymmetric(Vc) && isposdef(Vc) && size(Vc) == (dof(mc), dof(mc))
        @test stderror(mc) == sqrt.(diag(Vc))
        # Rescaling a covariate rescales its coefficient and standard error alike
        n_s = 1000
        x_s = randn(StableRNG(41), n_s)
        y_s, _ = simulate_lca_reg(StableRNG(42), hcat(ones(n_s), x_s), reshape([0.4, 0.8], 2, 1),
                                  [[0.8 0.2; 0.2 0.8] for _ in 1:6])
        ms = fit(LCAModel, LCAData(y_s; covariates=x_s, covariate_names=[:x]), 2;
                 rng=StableRNG(1), n_starts=4, n_final=2)
        ms1000 = fit(LCAModel, LCAData(y_s; covariates=1000 .* x_s, covariate_names=[:x]), 2;
                     rng=StableRNG(1), n_starts=4, n_final=2)
        @test stderror(ms1000)[2] ≈ stderror(ms)[2] / 1000 rtol = 1e-6
        @test stderror(ms1000)[1] ≈ stderror(ms)[1] rtol = 1e-6
        @test stderror(ms1000)[3:end] ≈ stderror(ms)[3:end] rtol = 1e-6
        @test isfinite(stderror(ms)[2]) && stderror(ms)[2] < 0.5
        # Class 1 is the larger class (the one favoured by x), so the slope is about -0.8
        @test coef(ms)[2] < 0
        @test abs(abs(coef(ms)[2]) - 0.8) < 3 * stderror(ms)[2]
    end

    @testset "coeftable" begin
        ct = coeftable(m2)
        @test ct isa CoefTable
        @test length(ct) == dof(m2)
        @test ct.colnms == ["Estimate", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
        @test ct.rownms == coefnames(m2)
        @test ct.cols[1] == coef(m2) && ct.cols[2] == stderror(m2)
        @test ct.cols[3] ≈ coef(m2) ./ stderror(m2)
        @test all(0 .<= ct.cols[4] .<= 1)
        @test all(ct.cols[4] .< 1e-10)                       # everything is far from zero
        @test ct.cols[5] == confint(m2)[:, 1] && ct.cols[6] == confint(m2)[:, 2]
        @test ct.pvalcol == 4 && ct.teststatcol == 3
        s = sprint(show, MIME("text/plain"), ct)
        @test occursin("Estimate", s) && occursin("class2: (Intercept)", s) && occursin("|class2", s)
        @test coeftable(m2; level=0.9).colnms[5] == "Lower 90%"
        @test coeftable(m2; level=0.999).colnms[6] == "Upper 99.9%"
        # Row selection
        @test length(coeftable(m2; which=:class)) == 1 * (2 - 1)
        @test length(coeftable(m2; which=:items)) == 2 * 6
        @test length(coeftable(mc; which=:class)) == 3 * (3 - 1)
        @test length(coeftable(mc; which=:items)) == 3 * sum(d.n_categories .- 1)
        @test coeftable(mc; which=:class).rownms == coefnames(mc)[1:6]
        @test coeftable(mc; which=:items).rownms == coefnames(mc)[7:end]
        @test length(coeftable(mc)) == dof(mc)
        @test_throws ArgumentError coeftable(m2; which=:everything)
        # A single class has an empty class block
        m1 = fit(LCAModel, d, 1)
        @test length(coeftable(m1; which=:class)) == 0
        @test length(coeftable(m1; which=:items)) == dof(m1)
    end

    @testset "Profiles: delta method" begin
        prof = profiles(m2)
        @test length(prof) == 2 * 12
        @test all(isfinite(r.se) && r.se > 0 for r in prof)
        @test all(0 <= r.lower <= r.prob <= r.upper <= 1 for r in prof)
        # A binary row has one free logit: both levels share the standard error
        for j in 1:6, k in 1:2
            rows = [r for r in prof if r.item == Symbol("item", j) && r.class == k]
            @test rows[1].se ≈ rows[2].se
            @test rows[1].lower ≈ 1 - rows[2].upper
        end
        # Delta-method covariance of a row sums to zero along rows and columns
        layout = LCA.ParamLayout(m2)
        for j in (1, 4), k in 1:2
            S = LCA._profile_covariance(m2, layout, m2.vcov, j, k)
            @test size(S) == (2, 2)
            @test maximum(abs, sum(S, dims=1)) < 1e-10
            @test maximum(abs, sum(S, dims=2)) < 1e-10
            @test S[1, 1] ≈ [r.se for r in prof if r.item == Symbol("item", j) && r.class == k][1]^2
        end
        # Polytomous rows
        m3 = fit(LCAModel, d, 3; rng=StableRNG(1), n_starts=4, n_final=2)
        lay3 = LCA.ParamLayout(m3)
        for j in (2, 3), k in 1:3
            S = LCA._profile_covariance(m3, lay3, m3.vcov, j, k)
            @test size(S) == (d.n_categories[j], d.n_categories[j])
            @test maximum(abs, sum(S, dims=1)) < 1e-10
            @test issymmetric(round.(S; digits=14))
        end
        prof3 = profiles(m3)
        @test all(0 <= r.lower <= r.prob <= r.upper <= 1 for r in prof3 if isfinite(r.se))
        # Narrower interval at a lower level
        prof90 = profiles(m2; level=0.9)
        @test all(prof90[i].upper - prof90[i].lower < prof[i].upper - prof[i].lower for i in eachindex(prof))
        @test all(prof90[i].se == prof[i].se for i in eachindex(prof))
        @test_throws ArgumentError profiles(m2; level=1.2)

        # Class-size rows
        pc = profiles(m2; classes=true)
        @test length(pc) == 2 + length(prof)
        @test pc[1].item == :class && pc[1].level == "1" && pc[1].class == 1
        @test pc[2].item == :class && pc[2].level == "2" && pc[2].class == 2
        @test [r.prob for r in pc[1:2]] == m2.class_probs
        @test pc[1].se ≈ pc[2].se
        @test 0 < pc[1].se < 0.02
        @test pc[1].lower < pc[1].prob < pc[1].upper
        @test pc[3:end] == prof
        Sc = LCA._class_covariance(m2, layout, m2.vcov)
        @test maximum(abs, sum(Sc, dims=1)) < 1e-10
        @test Sc[1, 1] ≈ pc[1].se^2
        # Three classes: the class block has two parameters
        pc3 = profiles(m3; classes=true)
        @test all(isfinite(r.se) for r in pc3[1:3])
        @test maximum(abs, sum(LCA._class_covariance(m3, lay3, m3.vcov), dims=1)) < 1e-10
        # Covariate model: class sizes are averaged priors, no standard error
        pcc = profiles(mc; classes=true)
        @test all(isnan(r.se) && isnan(r.lower) && isnan(r.upper) for r in pcc[1:3])
        @test [r.prob for r in pcc[1:3]] == mc.class_probs
        @test all(isfinite(r.se) for r in pcc[4:end])
        # Single class: size 1 with zero standard error
        p1 = profiles(fit(LCAModel, d, 1); classes=true)
        @test p1[1].prob == 1.0 && p1[1].se == 0.0 && p1[1].lower == 1.0 && p1[1].upper == 1.0
    end

    @testset "Exact single-class standard errors" begin
        # With one class the delta-method standard error of a response probability is the
        # multinomial standard error, and the logit standard error is 1/sqrt(n p (1 - p))
        for data in (d2, d)
            m1 = fit(LCAModel, data, 1)
            nn = nobs(m1)
            for r in profiles(m1)
                @test r.se ≈ sqrt(r.prob * (1 - r.prob) / nn) rtol = 1e-6
            end
            se = stderror(m1)
            layout = LCA.ParamLayout(m1)
            for j in 1:m1.n_items
                if data.n_categories[j] == 2
                    p = m1.item_probs[j][1, 1]
                    @test se[LCA._row_indices(layout, j, 1)[1]] ≈ 1 / sqrt(nn * p * (1 - p)) rtol = 1e-6
                end
            end
        end
        # With missing responses the per-item sample size is the number observed
        m1m = fit(LCAModel, dm, 1)
        nobs_j = nobs(m1m) .- nmissing(m1m)
        for r in profiles(m1m)
            j = findfirst(==(r.item), m1m.data.item_names)
            @test r.se ≈ sqrt(r.prob * (1 - r.prob) / nobs_j[j]) rtol = 1e-6
        end
    end

    @testset "Approximate standard errors" begin
        # Well separated classes: the profile standard errors are close to those of the
        # within-class binomial proportions, and the class-size standard error to that of a
        # binomial proportion
        prof = profiles(m2; classes=true)
        for r in prof
            if r.item == :class
                @test 0.75 < r.se / sqrt(r.prob * (1 - r.prob) / n2) < 1.25
            else
                nk = n2 * m2.class_probs[r.class]
                @test 0.75 < r.se / sqrt(r.prob * (1 - r.prob) / nk) < 1.25
            end
        end
        # The true values lie within the intervals for most parameters
        truth = [r.item == :class ? (r.class == 1 ? 0.6 : 0.4) :
                 (m2.item_probs[findfirst(==(r.item), m2.data.item_names)][r.class, 1] > 0.5) == (r.level == "1") ? p2 : 1 - p2
                 for r in prof]
        # (the two levels of a binary row share one interval, so misses come in pairs)
        covered = count(i -> prof[i].lower <= truth[i] <= prof[i].upper, eachindex(prof))
        @test covered >= length(prof) - 6
    end

    @testset "Boundary parameters" begin
        # One cell with true probability 0.999 in a sample of 300: the estimate hits the floor
        items_b = [[0.999 0.001; 0.2 0.8], [0.8 0.2; 0.2 0.8], [0.85 0.15; 0.15 0.85],
                   [0.8 0.2; 0.2 0.8], [0.8 0.2; 0.25 0.75]]
        yb, _ = simulate_lca(StableRNG(61), 300, [0.6, 0.4], items_b)
        db = LCAData(yb)
        mb = @test_logs (:warn, r"on the boundary \(0 or 1\); its standard error is undefined and reported as NaN") fit(
            LCAModel, db, 2; rng=StableRNG(1), n_starts=4, n_final=2)
        @test mb.flags.n_boundary == 2
        kb = argmax(mb.item_probs[1][:, 1])
        @test mb.item_probs[1][kb, 1] >= 1 - 1e-6
        layout = LCA.ParamLayout(mb)
        @test count(!, layout.free) == 1
        @test !layout.free[LCA._row_indices(layout, 1, kb)[1]]
        V = vcov(mb)
        se = stderror(mb)
        bad = LCA._row_indices(layout, 1, kb)[1]
        @test isnan(se[bad])
        @test count(isnan, se) == 1
        @test all(isnan, V[bad, :]) && all(isnan, V[:, bad])
        good = setdiff(1:dof(mb), bad)
        @test all(isfinite, V[good, good])
        @test isposdef(V[good, good])
        prof = profiles(mb; classes=true)
        for r in prof
            if r.item == :item1 && r.class == kb
                @test isnan(r.se) && isnan(r.lower) && isnan(r.upper)
            else
                @test isfinite(r.se) && isfinite(r.lower) && isfinite(r.upper)
            end
        end
        @test occursin("NaN for 1 of $(dof(mb)) parameters", sprint(show, mb))
        out = sprint(io -> show_profiles(mb; io=io))
        @test occursin("±NaN", out)
        @test count("±NaN", out) == 2
        ct = coeftable(mb)
        @test isnan(ct.cols[2][bad]) && isnan(ct.cols[4][bad])
        @test occursin("NaN", sprint(show, MIME("text/plain"), ct))
        ci = confint(mb)
        @test all(isnan, ci[bad, :]) && all(isfinite, ci[good, :])
        Ib = informationmatrix(mb)
        @test all(isnan, Ib[bad, :]) && all(isfinite, Ib[good, good])
        @test Ib[good, good] * V[good, good] ≈ I atol = 1e-8
    end

    @testset "Diverged coefficients" begin
        rng_s = StableRNG(71)
        ns = 600
        xs = randn(rng_s, ns)
        cls = 1 .+ (xs .> 0)                                  # the covariate determines the class
        sep_items = [[0.9 0.1; 0.1 0.9] for _ in 1:8]
        ys = [rand(rng_s) < sep_items[j][cls[i], 1] ? 1 : 2 for i in 1:ns, j in 1:8]
        dsep = LCAData(ys; covariates=xs, covariate_names=[:x])
        msep = @test_logs (:warn, r"quasi-complete separation.*standard errors are not computed") match_mode = :any fit(
            LCAModel, dsep, 2; rng=StableRNG(1), n_starts=4, n_final=2, max_iter=200)
        @test msep.flags.coef_divergence
        @test msep.vcov !== nothing && size(msep.vcov) == (dof(msep), dof(msep))
        @test all(isnan, vcov(msep))
        @test all(isnan, stderror(msep))
        @test all(isnan, confint(msep))
        @test all(isnan(r.se) for r in profiles(msep))
        @test length(coeftable(msep)) == dof(msep)
        @test occursin("NaN for $(dof(msep)) of $(dof(msep))", sprint(show, msep))
        @test sprint(io -> show_profiles(msep; io=io)) isa String
    end

    @testset "Not positive definite" begin
        # Far from the optimum (one EM iteration) the observed information is indefinite
        # for this design; the fit still returns, with an all-NaN covariance matrix
        θ0 = (class_probs=[0.5, 0.5], item_probs=[[0.5 0.5; 0.5 0.5] for _ in 1:6])
        mnp = @test_logs (:warn, r"did not converge.*observed information is not positive definite.*standard errors are NaN") fit(
            LCAModel, d2, 2; rng=StableRNG(1), init=θ0, n_starts=1, short_iters=0, max_iter=1, tol=0.0)
        @test !mnp.converged
        @test mnp.vcov !== nothing && size(mnp.vcov) == (dof(mnp), dof(mnp))
        @test all(isnan, vcov(mnp))
        @test all(isnan, stderror(mnp))
        @test !isposdef(Symmetric(informationmatrix(mnp)))
        @test all(isnan(r.se) for r in profiles(mnp))
        @test occursin("NaN for $(dof(mnp)) of $(dof(mnp))", sprint(show, mnp))
        # The warning message of a failed factorization
        ws, θ, layout = LCA._model_params(m2)
        V, posdef = LCA._covariance(LCA._pack(θ, layout), layout, ws)
        @test posdef && all(isfinite, V)
        @test V ≈ m2.vcov
    end

    @testset "se = :none" begin
        mn = fit(LCAModel, d2, 2; rng=StableRNG(1), n_starts=4, n_final=2, se=:none)
        @test mn.vcov === nothing
        @test mn.options.se == :none
        @test_throws ErrorException vcov(mn)
        @test_throws ErrorException stderror(mn)
        @test_throws ErrorException confint(mn)
        @test_throws ErrorException coeftable(mn)
        @test coef(mn) == coef(m2)                            # the estimates are unaffected
        @test coefnames(mn) == coefnames(m2)
        prof = profiles(mn; classes=true)
        @test all(isnan(r.se) && isnan(r.lower) && isnan(r.upper) for r in prof)
        @test [r.prob for r in prof[3:end]] == [r.prob for r in profiles(m2)]
        out = sprint(io -> show_profiles(mn; io=io))
        @test occursin("Latent Class Profiles", out) && !occursin("±", out)
        @test occursin("standard errors: none", sprint(show, mn))
        # informationmatrix does not need a stored covariance matrix
        @test informationmatrix(mn) ≈ informationmatrix(m2)
        @test_throws ArgumentError fit(LCAModel, d2, 2; se=:bootstrap)
    end

    @testset "informationmatrix" begin
        Im = informationmatrix(m2)
        @test size(Im) == (dof(m2), dof(m2))
        @test issymmetric(Im)
        @test isposdef(Im)
        @test Im * vcov(m2) ≈ I atol = 1e-8
        @test Im ≈ inv(vcov(m2)) rtol = 1e-6
        @test_throws ArgumentError informationmatrix(m2; expected=true)
        # Covariate model: the public scale is the raw covariate scale
        Ic = informationmatrix(mc)
        @test issymmetric(Ic) && isposdef(Ic)
        @test Ic * vcov(mc) ≈ I atol = 1e-6
        # Single class: the information of every binary item is n p (1 - p)
        m1 = fit(LCAModel, d2, 1)
        I1 = informationmatrix(m1)
        @test isdiag(round.(I1; digits=6))
        for j in 1:6
            p = m1.item_probs[j][1, 1]
            @test I1[j, j] ≈ n2 * p * (1 - p) rtol = 1e-6
        end
    end

    @testset "show" begin
        s = sprint(show, m2)
        @test occursin("standard errors: observed information", s)
        @test !occursin("NaN", s)
        @test occursin("LCAModel with 1 class, 6 items", sprint(show, fit(LCAModel, d2, 1)))
        @test sprint(show, fit(LCAModel, d2, 1); context=:compact => true) == "LCAModel(1 class, 6 items, n = $n2)"
        @test occursin("LCAModel with 2 classes, 6 items and $n2 observations", s)
        out = sprint(io -> show_profiles(m2; io=io))
        @test occursin(r"Class 1: \d+\.\d\s*% ±\d\.\d", out)
        @test occursin(r"^1:\s+\d+\.\d{3}% ±\d+\.\d{3}\s+\d+\.\d{3}% ±\d+\.\d{3}"m, out)
        @test count("±", out) == 2 + 2 * 12
        out1 = sprint(io -> show_profiles(m2; digits=1, io=io))
        @test occursin(r"^1:\s+\d+\.\d% ±\d+\.\d\s+\d+\.\d% ±\d+\.\d"m, out1)
        # Covariate model: class sizes have no standard error, items do
        outc = sprint(io -> show_profiles(mc; io=io))
        @test count("±NaN", outc) == 3
        @test occursin(r"^1:\s+\d+\.\d{3}% ±\d+\.\d{3}"m, outc)
    end
end
