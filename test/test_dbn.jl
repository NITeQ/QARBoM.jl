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
    biases = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    dbn = QARBoM.initialize_dbn(layers_size; weights = weights, biases = biases)
    x = rand(2)

    @test QARBoM.propagate_up(dbn, x, 1, 3) == QARBoM._sigmoid.(biases[3])
    @test QARBoM.propagate_up(dbn, x, 1, 2) == QARBoM._sigmoid.(biases[2])
    @test QARBoM.propagate_up(dbn, x, 2, 3) == QARBoM._sigmoid.(biases[3])
end

# Test for propagate_down function
@testset "propagate_down tests" begin
    layers_size = [2, 2, 2]
    weights = [zeros(2, 2), zeros(2, 2)]
    biases = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    dbn = QARBoM.initialize_dbn(layers_size; weights = weights, biases = biases)
    x = rand(2)

    @test QARBoM.propagate_down(dbn, x, 3, 1) == QARBoM._sigmoid.(biases[1])
    @test QARBoM.propagate_down(dbn, x, 2, 1) == QARBoM._sigmoid.(biases[1])
    @test QARBoM.propagate_down(dbn, x, 3, 2) == QARBoM._sigmoid.(biases[2])
end

@testset "reconstruct" begin
    layers_size = [2, 2, 2]
    weights = [zeros(2, 2), zeros(2, 2)]
    biases = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    dbn = QARBoM.initialize_dbn(layers_size; weights = weights, biases = biases)
    x = rand(2)

    @test QARBoM.reconstruct(dbn, x, 3) == QARBoM._sigmoid.(biases[1])
    @test QARBoM.reconstruct(dbn, x, 2) == QARBoM._sigmoid.(biases[1])

    layers_size = [3, 2, 2]
    weights = [randn(3, 2), randn(2, 2)]
    biases = [rand(3), rand(2), rand(2)]

    dbn = QARBoM.initialize_dbn(layers_size; weights = weights, biases = biases)

    x = rand(3)

    pass_1 = QARBoM.propagate_up(dbn, x, 1, 2)
    @test pass_1 == QARBoM._sigmoid.(weights[1]' * x .+ biases[2])

    pass_2 = QARBoM.propagate_up(dbn, pass_1, 2, 3)
    @test pass_2 == QARBoM._sigmoid.(weights[2]' * pass_1 .+ biases[3])

    pass_3 = QARBoM.propagate_down(dbn, pass_2, 3, 2)
    @test pass_3 == QARBoM._sigmoid.(weights[2] * pass_2 .+ biases[2])

    pass_4 = QARBoM.propagate_down(dbn, pass_3, 2, 1)
    @test pass_4 == QARBoM._sigmoid.(weights[1] * pass_3 .+ biases[1])

    @test QARBoM.reconstruct(dbn, x, 3) == pass_4
end
