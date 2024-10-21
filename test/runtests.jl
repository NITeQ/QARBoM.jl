using Test
using QARBoM

@testset "QARBoM" begin
    @testset "Generative" begin
        include("generative.jl")
    end
    @testset "Classification" begin
        include("classification.jl")
    end
end
