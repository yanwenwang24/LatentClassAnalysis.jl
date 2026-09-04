using Test
using LatentClassAnalysis
using StableRNGs
using Statistics

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

const LCA = LatentClassAnalysis

@testset "Missing data" begin
    n = 2000
    y, classes = simulate_lca(StableRNG(1), n, TWO_CLASS_PROBS, TWO_CLASS_ITEMS)
    d = LCAData(y)
    m = fit(LCAModel, d, 2; rng=StableRNG(1))

    @testset "Complete data through the Union{Missing,Int} path" begin
        ym = Matrix{Union{Missing,Int}}(y)
        dm = LCAData(ym)
        @test dm.y == y
        @test !hasmissing(dm)
        mm = fit(LCAModel, dm, 2; rng=StableRNG(1))
        @test same_fit(mm, m)
    end

    @testset "20% MCAR recovery" begin
        ymiss = mcar!(StableRNG(11), copy(y), 0.2)
        dmiss = LCAData(ymiss)
        @test hasmissing(dmiss)
        rate = sum(nmissing(dmiss)) / length(ymiss)
        @test 0.17 < rate < 0.23
        @test dmiss.n_categories == fill(2, 6)

        mmiss = fit(LCAModel, dmiss, 2; rng=StableRNG(1))
        @test mmiss.converged
        @test isfinite(mmiss.loglik)
        @test dof(mmiss) == dof(m)          # missingness does not change the parameter count
        @test nobs(mmiss) == n
        perm = align_classes(mmiss.item_probs, TWO_CLASS_ITEMS)
        e_class, e_item = max_abs_error(mmiss, perm, TWO_CLASS_PROBS, TWO_CLASS_ITEMS)
        @test e_class < 0.06
        @test e_item < 0.06
        @test size(mmiss.posterior) == (n, 2)
        @test all(abs.(sum(mmiss.posterior, dims=2) .- 1) .< 1e-12)
        @test mean(invperm(perm)[classify(mmiss)] .== classes) > 0.85

        # The log-likelihood of missing data is larger (fewer observed responses) and
        # equals the stored value when recomputed
        @test mmiss.loglik > m.loglik
        @test loglikelihood(mmiss, dmiss) == mmiss.loglik
        @test hasmissing(mmiss) && nmissing(mmiss) == nmissing(dmiss)
    end

    @testset "Fully missing rows get the prior" begin
        yr = copy(y)
        yr[1, :] .= 0
        yr[n, :] .= 0
        dr = LCAData(yr)
        mr = fit(LCAModel, dr, 2; rng=StableRNG(1))
        @test maximum(abs.(mr.posterior[1, :] .- mr.class_probs)) < 1e-12
        @test maximum(abs.(mr.posterior[n, :] .- mr.class_probs)) < 1e-12
        # ... and contribute log Σ π_k = 0 to the log-likelihood
        d_rest = LCAData(y[2:n-1, :])
        @test loglikelihood(mr, dr) ≈ loglikelihood(mr, d_rest)
        # The same holds for prediction on new data
        pnew = predict(mr, LCAData([0 0 0 0 0 0; 1 2 1 2 1 2]; n_categories=fill(2, 6)))
        @test maximum(abs.(pnew[1, :] .- mr.class_probs)) < 1e-12
        @test classify(mr, LCAData([0 0 0 0 0 0]; n_categories=fill(2, 6))) == [1]  # largest class
    end

    @testset "Per-item denominators use observed rows only" begin
        # With item 1 missing in half of the rows, its M-step denominator is the posterior
        # mass of the rows where it is observed, while the class sizes use every row.
        yh = copy(y)
        obs = 1:2:n
        yh[2:2:n, 1] .= 0
        dh = LCAData(yh)
        ws = LCA.LCAWorkspace(dh, 2; aggregate=false)
        θ = LCA.LCAParams(copy(m.class_probs), [copy(P) for P in m.item_probs], nothing)
        LCA.estep!(ws, θ)
        post = permutedims(ws.post)
        LCA._accumulate!(ws)
        @test ws.Nk ≈ vec(sum(post, dims=1))
        for k in 1:2, c in 1:2
            @test ws.Njkc[1][k, c] ≈ sum(post[i, k] for i in obs if yh[i, 1] == c)
            @test ws.Njkc[2][k, c] ≈ sum(post[i, k] for i in 1:n if yh[i, 2] == c)
        end
        @test sum(ws.Njkc[1]) ≈ length(obs)
        @test sum(ws.Njkc[2]) ≈ n
        LCA._update!(θ, ws)
        for k in 1:2
            denom = sum(post[i, k] for i in obs)
            @test θ.item_probs[1][k, :] ≈ [sum(post[i, k] for i in obs if yh[i, 1] == c) / denom for c in 1:2]
        end
        @test θ.class_probs ≈ vec(sum(post, dims=1)) ./ n

        # Aggregation with missing codes treats 0 as part of the pattern
        wsa = LCA.LCAWorkspace(dh, 2)
        @test wsa.U == length(unique(eachrow(yh)))
        @test any(==(0), wsa.yt)
        ll_a = LCA.estep!(wsa, LCA.LCAParams(copy(m.class_probs), [copy(P) for P in m.item_probs], nothing))
        ll_f = LCA.estep!(LCA.LCAWorkspace(dh, 2; aggregate=false),
                          LCA.LCAParams(copy(m.class_probs), [copy(P) for P in m.item_probs], nothing))
        @test isapprox(ll_a, ll_f; rtol=1e-12)
    end

    @testset "Missing data through prepare_data" begin
        tbl = (a=[1, missing, 2, 2, 1, 2], b=["x", "y", missing, "y", "x", "y"], c=[1, 2, 1, 2, 1, 2])
        d = @test_logs prepare_data(tbl, [:a, :b, :c])
        @test d.y == [1 1 1; 0 2 2; 2 0 1; 2 2 2; 1 1 1; 2 2 2]
        mm = fit(LCAModel, d, 1)
        @test mm.item_probs[1][1, :] ≈ [2 / 5, 3 / 5]     # a: observed 1,2,2,1,2
        @test mm.item_probs[2][1, :] ≈ [2 / 5, 3 / 5]     # b: observed x,y,y,x,y
        @test mm.item_probs[3][1, :] ≈ [3 / 6, 3 / 6]
        @test size(predict(mm, tbl)) == (6, 1)
    end
end
