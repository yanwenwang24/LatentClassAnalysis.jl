using Test
using LatentClassAnalysis
using StableRNGs

@isdefined(simulate_lca) || include(joinpath(@__DIR__, "testutils.jl"))

const LCA = LatentClassAnalysis

@testset "EM core" begin
    y, _ = simulate_lca(StableRNG(101), 800, TWO_CLASS_PROBS, TWO_CLASS_ITEMS)
    d = LCAData(y)
    ymiss, _ = simulate_lca(StableRNG(101), 800, TWO_CLASS_PROBS, TWO_CLASS_ITEMS; missing_rate=0.2)
    dmiss = LCAData(ymiss)
    @test count(iszero, ymiss) > 0

    @testset "Workspace and pattern aggregation" begin
        ws = LCA.LCAWorkspace(d, 2)
        @test ws.aggregated
        @test ws.K == 2 && ws.J == 6 && ws.n == 800
        @test ws.U == length(unique(eachrow(y)))
        @test ws.U < ws.n
        @test sum(ws.freq) == 800
        @test size(ws.yt) == (6, ws.U)
        @test size(ws.post) == (2, ws.U)
        # Every observation maps to its own pattern
        @test all(ws.yt[:, ws.row_index[i]] == y[i, :] for i in 1:800)
        # Frequencies count the observations of each pattern
        @test all(ws.freq[u] == count(==(u), ws.row_index) for u in 1:ws.U)

        ws_full = LCA.LCAWorkspace(d, 2; aggregate=false)
        @test !ws_full.aggregated
        @test ws_full.U == 800
        @test ws_full.yt == permutedims(y)
        @test ws_full.row_index == 1:800
        @test all(==(1.0), ws_full.freq)

        # A copy shares the data buffers but not the scratch buffers
        ws2 = LCA.LCAWorkspace(ws)
        @test ws2.yt === ws.yt && ws2.freq === ws.freq && ws2.row_index === ws.row_index
        @test ws2.post !== ws.post && ws2.Nk !== ws.Nk
    end

    @testset "Monotone log-likelihood from random starts" begin
        for (data, label) in ((d, "complete"), (dmiss, "20% missing"))
            ws = LCA.LCAWorkspace(data, 2)
            for s in 1:5
                θ = LCA._init_random(StableRNG(200 + s), 2, data.n_categories)
                trace = Float64[]
                ll, iters, conv = LCA._em!(θ, ws; max_iter=500, tol=1e-10, ll_trace=trace)
                @test all(diff(trace) .>= -1e-8)
                @test trace[end] == ll
                @test length(trace) == iters + 1
                @test conv
                @test isfinite(ll)
                # The returned ll is the ll of the returned parameters and posterior
                post = copy(ws.post)
                @test LCA.estep!(ws, θ) == ll
                @test ws.post == post
                # Posterior columns (one per pattern) sum to one
                @test all(abs.(sum(ws.post, dims=1) .- 1) .< 1e-12)
                # Parameters are proper probabilities
                @test sum(θ.class_probs) ≈ 1 atol = 1e-12
                @test all(all(abs.(sum(P, dims=2) .- 1) .< 1e-12) for P in θ.item_probs)
                @test all(all(P .>= 1e-10) for P in θ.item_probs)
            end
        end
    end

    @testset "Loop order and iteration counts" begin
        ws = LCA.LCAWorkspace(d, 2)
        θ0 = LCA._init_random(StableRNG(7), 2, d.n_categories)

        # max_iter = 0: one E-step, no M-step, parameters untouched, not converged
        θ = copy(θ0)
        ll0, iters, conv = LCA._em!(θ, ws; max_iter=0, tol=1e-10)
        @test iters == 0 && !conv
        @test θ.class_probs == θ0.class_probs && θ.item_probs == θ0.item_probs
        @test ll0 == LCA.estep!(ws, θ0)

        # max_iter = 1 takes exactly one M-step and reports the ll after it
        θ = copy(θ0)
        trace = Float64[]
        ll1, iters1, conv1 = LCA._em!(θ, ws; max_iter=1, tol=1e-10, ll_trace=trace)
        @test iters1 == 1 && !conv1
        @test trace == [ll0, ll1]
        @test ll1 > ll0
        @test ll1 == LCA.estep!(ws, θ)

        # Continuation is exact: two calls equal one longer call
        θa = copy(θ0)
        LCA._em!(θa, ws; max_iter=3, tol=0.0)
        lla, _, _ = LCA._em!(θa, ws; max_iter=2, tol=0.0)
        θb = copy(θ0)
        llb, _, _ = LCA._em!(θb, ws; max_iter=5, tol=0.0)
        @test lla == llb
        @test θa.item_probs == θb.item_probs

        # A huge tolerance converges immediately after the first M-step
        θ = copy(θ0)
        _, iters_big, conv_big = LCA._em!(θ, ws; max_iter=100, tol=1e3)
        @test conv_big && iters_big == 1
    end

    @testset "Aggregation is exact" begin
        θ0 = LCA._init_random(StableRNG(11), 2, d.n_categories)
        ws_a = LCA.LCAWorkspace(d, 2; aggregate=true)
        ws_f = LCA.LCAWorkspace(d, 2; aggregate=false)
        θa, θf = copy(θ0), copy(θ0)
        lla, ia, _ = LCA._em!(θa, ws_a; max_iter=200, tol=1e-12)
        llf, if_, _ = LCA._em!(θf, ws_f; max_iter=200, tol=1e-12)
        @test ia == if_
        @test isapprox(lla, llf; rtol=1e-10)
        @test maximum(abs.(θa.class_probs .- θf.class_probs)) < 1e-10
        @test all(maximum(abs.(θa.item_probs[j] .- θf.item_probs[j])) < 1e-10 for j in 1:6)
        # The expanded posterior respects the row order of the data
        pa = LCA._expand_posterior(ws_a)
        pf = LCA._expand_posterior(ws_f)
        @test size(pa) == (800, 2)
        @test maximum(abs.(pa .- pf)) < 1e-10
        @test all(abs.(sum(pa, dims=2) .- 1) .< 1e-12)
        # Same row, same posterior
        i1 = 1
        i2 = findfirst(i -> i != i1 && y[i, :] == y[i1, :], 1:800)
        @test i2 !== nothing
        @test pa[i1, :] == pa[i2, :]
    end

    @testset "Sufficient statistics use observed responses only" begin
        # Tiny hand-computed case: 3 rows, 2 items, item 2 missing in row 2
        yt = [1 2 1; 2 0 1]
        dt = LCAData(permutedims(yt))
        ws = LCA.LCAWorkspace(dt, 2; aggregate=false)
        ws.post .= [0.9 0.2 0.5; 0.1 0.8 0.5]
        LCA._accumulate!(ws)
        @test ws.Nk ≈ [0.9 + 0.2 + 0.5, 0.1 + 0.8 + 0.5]
        # Item 1: row 1 -> cat 1, row 2 -> cat 2, row 3 -> cat 1
        @test ws.Njkc[1] ≈ [0.9 + 0.5 0.2; 0.1 + 0.5 0.8]
        # Item 2: row 2 is missing, so only rows 1 and 3 contribute (both cat 2 and 1)
        @test ws.Njkc[2] ≈ [0.5 0.9; 0.5 0.1]
        @test sum(ws.Njkc[2]) ≈ 2.0
        @test sum(ws.Njkc[1]) ≈ 3.0

        # The M-step divides each item by its own denominator
        θ = LCA.LCAParams([0.5, 0.5], [fill(0.5, 2, 2), fill(0.5, 2, 2)], nothing)
        LCA._update!(θ, ws)
        @test θ.class_probs ≈ [1.6, 1.4] ./ 3
        @test θ.item_probs[1] ≈ [1.4 0.2; 0.6 0.8] ./ [1.6, 1.4]
        @test θ.item_probs[2] ≈ [0.5 0.9; 0.5 0.1] ./ [1.4, 0.6]

        # An empty class row becomes uniform; probabilities are floored at 1e-10
        ws.post .= [1.0 1.0 1.0; 0.0 0.0 0.0]
        LCA._accumulate!(ws)
        LCA._update!(θ, ws)
        @test θ.item_probs[1][2, :] == [0.5, 0.5]
        @test θ.item_probs[2][2, :] == [0.5, 0.5]
        @test all(θ.item_probs[1][1, :] .>= 1e-10)
        @test sum(θ.item_probs[1][1, :]) ≈ 1
    end

    @testset "Single class closed form" begin
        ws = LCA.LCAWorkspace(dmiss, 1)
        θ, ll = LCA._fit_single_class(ws)
        @test θ.class_probs == [1.0]
        for j in 1:6
            obs = ymiss[ymiss[:, j] .> 0, j]
            @test θ.item_probs[j][1, :] ≈ [count(==(c), obs) / length(obs) for c in 1:2]
        end
        @test all(==(1.0), ws.post)
        @test isfinite(ll)
        # The closed form is a fixed point of EM
        ll2, iters, conv = LCA._em!(θ, ws; max_iter=10, tol=1e-12)
        @test conv && ll2 ≈ ll
    end

    @testset "200 binary items, 3 classes" begin
        items = [[0.8 0.2; 0.2 0.8; 0.5 0.5] for _ in 1:200]
        yb, _ = simulate_lca(StableRNG(202), 200, [0.4, 0.35, 0.25], items)
        db = LCAData(yb)
        ws = LCA.LCAWorkspace(db, 3)
        @test ws.U == 200   # every pattern is unique
        θ = LCA._init_random(StableRNG(1), 3, db.n_categories)
        ll, _, conv = LCA._em!(θ, ws; max_iter=500, tol=1e-10)
        @test isfinite(ll)
        @test !any(isnan, ws.post)
        @test all(abs.(sum(ws.post, dims=1) .- 1) .< 1e-12)
        @test all(!any(isnan, P) for P in θ.item_probs)
    end

    @testset "Random starts" begin
        θ = LCA._init_random(StableRNG(3), 3, [2, 3, 4])
        @test θ.class_probs == fill(1 / 3, 3)
        @test [size(P) for P in θ.item_probs] == [(3, 2), (3, 3), (3, 4)]
        @test all(all(abs.(sum(P, dims=2) .- 1) .< 1e-12) for P in θ.item_probs)
        @test all(all(P .> 0) for P in θ.item_probs)
        @test θ.coefs === nothing
        @test LCA._init_random(StableRNG(3), 3, [2, 3, 4]).item_probs == θ.item_probs
        @test LCA._init_random(StableRNG(4), 3, [2, 3, 4]).item_probs != θ.item_probs
        θc = copy(θ)
        @test θc.item_probs == θ.item_probs && θc.item_probs[1] !== θ.item_probs[1]
    end

    @testset "Class reordering and splitting" begin
        θ = LCA.LCAParams([0.2, 0.5, 0.3], [[0.1 0.9; 0.5 0.5; 0.7 0.3]], nothing)
        perm = LCA._sort_by_size!(θ)
        @test perm == [2, 3, 1]
        @test θ.class_probs == [0.5, 0.3, 0.2]
        @test θ.item_probs[1] == [0.5 0.5; 0.7 0.3; 0.1 0.9]

        # Coefficients are re-based so that the new class 1 is the reference
        θc = LCA.LCAParams([0.2, 0.5, 0.3], [[0.1 0.9; 0.5 0.5; 0.7 0.3]], [0.0 1.0 -1.0; 0.0 2.0 0.5])
        LCA._permute_classes!(θc, [2, 3, 1])
        @test θc.coefs ≈ [0.0 -2.0 -1.0; 0.0 -1.5 -2.0]
        @test all(iszero, θc.coefs[:, 1])

        # Splitting class 2 of a 2-class solution
        θ2 = LCA.LCAParams([0.7, 0.3], [[0.8 0.2; 0.3 0.7], [0.1 0.6 0.3; 0.5 0.25 0.25]], nothing)
        θ3 = LCA._init_split(θ2, 2, StableRNG(9))
        @test length(θ3.class_probs) == 3
        @test θ3.class_probs ≈ [0.7, 0.15, 0.15]
        @test [size(P) for P in θ3.item_probs] == [(3, 2), (3, 3)]
        @test θ3.item_probs[1][1, :] == [0.8, 0.2]                 # untouched class
        @test all(all(abs.(sum(P, dims=2) .- 1) .< 1e-12) for P in θ3.item_probs)
        @test maximum(abs.(θ3.item_probs[1][2, :] .- [0.3, 0.7])) <= 0.05 + 1e-12
        @test maximum(abs.(θ3.item_probs[1][3, :] .- [0.3, 0.7])) <= 0.05 + 1e-12
        @test θ3.item_probs[1][2, :] != θ3.item_probs[1][3, :]
        @test_throws ArgumentError LCA._init_split(θ2, 3, StableRNG(9))
    end
end
