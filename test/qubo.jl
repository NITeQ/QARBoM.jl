using MLDatasets

function test_qubo()
    # Initialize RBM
    visible_units = 100
    hidden_units = 20

    rbm = QARBoM.RBM(visible_units, hidden_units, DWave.Neal.Optimizer)

    # Load MNIST dataset
    x_train = [
        [round(Int, rand()) for _ in 1:visible_units] for _ in 1:100
    ]

    x_bin = [vec(round.(Int, x_test[:, :, i])) for i in 1:100]

    # Train RBM
    return QARBoM.train(rbm, x_bin, QARBoM.CD(); n_epochs = 10, gibbs_steps = 1, learning_rate = 0.1)
end

test_qubo()
