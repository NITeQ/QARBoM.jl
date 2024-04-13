function train(rbm::BernoulliRBM, x_train::Vector{Vector{Int}}; n_epochs::Int, lr::Float64, cd_steps::Int = 3)
    for epoch = 1:n_epochs
        println("Epoch: $epoch")
        avg_loss = contrastive_divergence(rbm, x_train, steps = cd_steps, learning_rate = lr)
        println("Average loss: $avg_loss")
    end
end