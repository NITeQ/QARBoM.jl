using MLDatasets

function test_pcd()
    # Initialize RBM
    visible_units = 784
    hidden_units = 500
    rbm = QARBoM.RBM(visible_units, hidden_units)

    # Load MNIST dataset
    trainset = MNIST(:train)
    x_test, y_test = trainset[:]

    x_bin = [vec(round.(Int, x_test[:, :, i])) for i in 1:50000]

    # Train RBM
    return QARBoM.train_pcd(rbm, x_bin; batch_size = 50, n_epochs = 10, learning_rate = 0.1)
end

test_pcd()
