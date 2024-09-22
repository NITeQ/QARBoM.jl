using Test
using QARBoM
# Include the dbn.jl file to access the defined structures and functions


# Test for initialize_dbn function
@testset "initialize_dbn tests" begin
    layers_size = [3, 2, 2]
    dbn = QARBoM.initialize_dbn(layers_size)
    
    @test length(dbn.layers) == 3
    @test dbn.layers[1] isa QARBoM.VisibleLayer
    @test dbn.layers[2] isa QARBoM.HiddenLayer
    @test dbn.layers[3] isa QARBoM.TopLayer
    @test size(dbn.layers[1].W) == (3, 2)
    @test size(dbn.layers[2].W) == (2, 2)
    @test length(dbn.layers[1].bias) == 3
    @test length(dbn.layers[2].bias) == 2
    @test length(dbn.layers[3].bias) == 2
end

# Test for propagate_up function
@testset "propagate_up tests" begin
    layers_size = [2, 2, 2]
    weights = [zeros(2, 2), zeros(2, 2)]
    biases = [[1.0,1.0], [2.0,2.0], [3.0,3.0]]
    dbn = QARBoM.initialize_dbn(layers_size, weights=weights, biases=biases)
    x = rand(2)
    
    @test QARBoM.propagate_up(dbn, x, 1, 3) == QARBoM._sigmoid.(biases[3])
    @test QARBoM.propagate_up(dbn, x, 1, 2) == QARBoM._sigmoid.(biases[2])
    @test QARBoM.propagate_up(dbn, x, 2, 3) == QARBoM._sigmoid.(biases[3])
end

# Test for propagate_down function
@testset "propagate_down tests" begin
    layers_size = [2, 2, 2]
    weights = [zeros(2, 2), zeros(2, 2)]
    biases = [[1.0,1.0], [2.0,2.0], [3.0,3.0]]

    dbn = QARBoM.initialize_dbn(layers_size, weights=weights, biases=biases)
    x = rand(2)
    
    @test QARBoM.propagate_down(dbn, x, 3, 1) == QARBoM._sigmoid.(biases[1])
    @test QARBoM.propagate_down(dbn, x, 2, 1) == QARBoM._sigmoid.(biases[1])
    @test QARBoM.propagate_down(dbn, x, 3, 2) == QARBoM._sigmoid.(biases[2])
end

@testset "reconstruct" begin
    layers_size = [2, 2, 2]
    weights = [zeros(2, 2), zeros(2, 2)]
    biases = [[1.0,1.0], [2.0,2.0], [3.0,3.0]]
    dbn = QARBoM.initialize_dbn(layers_size, weights=weights, biases=biases)
    x = rand(2)
    
    @test QARBoM.reconstruct(dbn, x, 3) == QARBoM._sigmoid.(biases[1])
    @test QARBoM.reconstruct(dbn, x, 2) == QARBoM._sigmoid.(biases[1])
end