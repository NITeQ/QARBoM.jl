using Test
using QARBoM

function test_cd()
    x_train = [[1, 0, 0] for _ in 1:1000]
    y_train = [[1] for _ in 1:1000]

    rbm = RBMClassifier(3, 2, 1)

    QARBoM.train!(
        rbm,
        x_train,
        y_train,
        CD;
        n_epochs = 50,
        gibbs_steps = 3, # number of gibbs sampling steps
        learning_rate = [0.01 / (j^0.8) for j in 1:1000],
        label_learning_rate = [0.01 / (j^0.8) for j in 1:1000],
        early_stopping = true,
    )

    return rec = QARBoM.classify(rbm, [1, 0, 0])
end

function test_pcd()
    x_train = [[1, 0, 0] for _ in 1:1000]
    y_train = [[1] for _ in 1:1000]

    rbm = RBMClassifier(3, 2, 1)

    QARBoM.train!(
        rbm,
        x_train,
        y_train,
        PCD;
        n_epochs = 50,
        batch_size = 2,
        learning_rate = [0.01 / (j^0.8) for j in 1:1000],
        label_learning_rate = [0.01 / (j^0.8) for j in 1:1000],
        early_stopping = true,
    )

    return rec = QARBoM.classify(rbm, [1, 0, 0])
end

function fast_pcd()
    x_train = [[1, 0, 0] for _ in 1:1000]
    y_train = [[1] for _ in 1:1000]

    rbm = RBMClassifier(3, 2, 1)

    QARBoM.train!(
        rbm,
        x_train,
        y_train,
        FastPCD;
        n_epochs = 50,
        batch_size = 2,
        learning_rate = [0.01 / (j^0.8) for j in 1:1000],
        label_learning_rate = [0.01 / (j^0.8) for j in 1:1000],
        fast_learning_rate = 0.1,
        fast_label_learning_rate = 0.1,
        early_stopping = true,
    )

    return rec = QARBoM.classify(rbm, [1, 0, 0])
end

@testset "CD" begin
    test_cd()
end
@testset "PCD" begin
    test_pcd()
end
@testset "FastPCD" begin
    fast_pcd()
end
