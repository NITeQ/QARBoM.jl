using MLDatasets

function test_cd()
    # Initialize RBM
    visible_units = 784
    hidden_units = 500
    rbm = QARBoM.BernoulliRBM(visible_units, hidden_units)

    # Load MNIST dataset
    trainset = MNIST(:train)
    x_test, y_test = trainset[:]

    x_bin = [vec(round.(Int, x_test[:, :, i])) for i = 1:100]

    # Train RBM
    QARBoM.train(rbm, x_bin, QARBoM.CD(); n_epochs = 10, cd_steps = 1, learning_rate = 0.1)
end

test_cd()
