using Test
using QARBoM

function test_cd()
    x_train = [[1, 0, 0] for _ in 1:1000]

    rbm_cd = RBM(3, 2)

    QARBoM.train!(
        rbm_cd,
        x_train,
        CD;
        n_epochs = 50,
        gibbs_steps = 3, # number of gibbs sampling steps
        learning_rate = [0.01 / (j^0.8) for j in 1:1000],
        early_stopping = true,
    )

    return rec = QARBoM.reconstruct(rbm_cd, [1, 0, 0])
end

function test_pcd()
    x_train = [[1, 0, 0] for _ in 1:1000]

    rbm_pcd = RBM(3, 2)

    QARBoM.train!(
        rbm_pcd,
        x_train,
        PCD;
        n_epochs = 50,
        batch_size = 2,
        learning_rate = [0.01 / (j^0.8) for j in 1:1000],
        early_stopping = true,
    )

    return rec = QARBoM.reconstruct(rbm_pcd, [1, 0, 0])
end

function fast_pcd()
    x_train = [[1, 0, 0] for _ in 1:1000]

    rbm_pcd = RBM(3, 2)

    QARBoM.train!(
        rbm_pcd,
        x_train,
        FastPCD;
        n_epochs = 50,
        batch_size = 2,
        learning_rate = [0.01 / (j^0.8) for j in 1:1000],
        fast_learning_rate = 0.1,
        early_stopping = true,
    )

    return rec = QARBoM.reconstruct(rbm_pcd, [1, 0, 0])
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
