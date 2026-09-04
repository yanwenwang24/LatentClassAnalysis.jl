using Test
using LatentClassAnalysis

@testset "Docstrings" begin
    hasdoc(obj) = !occursin("No documentation found", string(Base.Docs.doc(obj)))

    @testset "Every exported name is documented" begin
        exported = filter(n -> n !== :LatentClassAnalysis, names(LatentClassAnalysis))
        @test Set(exported) == Set([
            # StatsAPI / StatsBase verbs
            :fit, :fit!, :predict, :loglikelihood, :nobs, :dof, :aic, :bic, :aicc, :coef,
            :coefnames, :vcov, :stderror, :confint, :coeftable, :informationmatrix, :isfitted,
            :pvalue, :entropy,
            # package names
            :LCAData, :LCAOptions, :LCAModel, :ModelDiagnostics, :prepare_data, :diagnostics,
            :sbic, :classify, :profiles, :show_profiles, :simulate, :bootstrap, :bootstrap_lrt,
            :LCABootstrap, :BootstrapLRT, :hasmissing, :nmissing, :hascovariates,
            # deprecated (exported by Base.@deprecate)
            :diagnostics!])
        for name in exported
            obj = getfield(LatentClassAnalysis, name)
            @test hasdoc(obj)
        end
    end

    @testset "Package docstrings mention their keywords" begin
        docstr(obj) = string(Base.Docs.doc(obj))
        for kw in ("rng", "init", "n_starts", "n_final", "short_iters", "max_iter", "tol", "se",
                   "aggregate", "multithreaded", "verbose", "covariates")
            @test occursin(kw, docstr(fit))
        end
        for kw in ("covariates", "levels", "drop_unused_levels", "missing")
            @test occursin(kw, docstr(prepare_data))
        end
        for kw in ("n_categories", "item_names", "item_levels", "covariates", "covariate_names")
            @test occursin(kw, docstr(LCAData))
        end
        for fld in ("class_probs", "item_probs", "beta", "posterior", "start_loglik", "flags", "vcov")
            @test occursin(fld, docstr(LCAModel))
        end
        for fld in ("n_starts", "n_final", "short_iters", "max_iter", "tol", "se", "aggregate")
            @test occursin(fld, docstr(LCAOptions))
        end
        @test occursin("entropy", docstr(ModelDiagnostics))
        @test occursin("Tables.jl", docstr(ModelDiagnostics))
        @test occursin("relative", docstr(entropy))
        @test occursin("level", docstr(profiles))
        for kw in ("var_names", "var_labels", "digits", "io")
            @test occursin(kw, docstr(show_profiles))
        end
        @test occursin("argmax", docstr(classify))
        @test occursin("LCAData", docstr(predict))
        @test occursin("DataFrame", docstr(diagnostics))
        @test occursin("(nobs + 2) / 24", docstr(sbic))
        @test occursin("fit(LCAModel, data, k)", docstr(fit!))
        @test occursin("replaced", docstr(LCAModel))
        @test occursin("Deprecated", docstr(diagnostics!))
        @test occursin("Deprecated", docstr(prepare_data))
        @test occursin("deprecated", docstr(show_profiles))
    end

    @testset "Internal docstrings" begin
        LCA = LatentClassAnalysis
        for obj in (LCA.FitFlags, LCA.LCAParams, LCA.LCAWorkspace, LCA.StartRecord, LCA.estep!,
                    LCA._accumulate!, LCA._update!, LCA._em!, LCA._multistart, LCA._init_random,
                    LCA._init_split, LCA.check_identifiability)
            @test hasdoc(obj)
        end
    end
end
