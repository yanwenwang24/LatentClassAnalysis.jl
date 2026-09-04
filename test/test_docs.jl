using Test
using LatentClassAnalysis

@testset "Docstrings" begin
    documented = (LCAModel, ModelDiagnostics, fit!, predict, prepare_data, diagnostics!, show_profiles)

    @testset "Attached to $(nameof(f))" for f in documented
        doc = string(Base.Docs.doc(f))
        @test !occursin("No documentation found", doc)
        @test !isempty(strip(doc))
        @test occursin(string(nameof(f)), doc)
    end

    @testset "Every exported name is documented" begin
        for name in names(LatentClassAnalysis)
            name === :LatentClassAnalysis && continue
            obj = getfield(LatentClassAnalysis, name)
            @test !occursin("No documentation found", string(Base.Docs.doc(obj)))
        end
    end

    @testset "Docstrings describe the current API" begin
        @test occursin("zero_based", string(Base.Docs.doc(prepare_data)))
        @test occursin("digits", string(Base.Docs.doc(show_profiles)))
        @test occursin("var_names", string(Base.Docs.doc(show_profiles)))
        @test occursin("entropy", string(Base.Docs.doc(ModelDiagnostics)))
        @test occursin("max_iter", string(Base.Docs.doc(fit!)))
        @test occursin("tol", string(Base.Docs.doc(fit!)))
        @test occursin("verbose", string(Base.Docs.doc(fit!)))
        @test occursin("n_categories", string(Base.Docs.doc(LCAModel)))
        @test occursin("ModelDiagnostics", string(Base.Docs.doc(diagnostics!)))
    end
end
