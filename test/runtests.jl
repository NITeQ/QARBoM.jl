using Test
using QARBoM

@testset "QARBoM" begin
    @testset "MNIST" begin
        include("mnist.jl")
    end
end
