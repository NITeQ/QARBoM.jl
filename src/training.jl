function train(rbm::BernoulliRBM, x_train::Vector{Vector{Int}}, n_epochs::Int, lr::Float64)
    for epoch = 1:n_epochs
        println("Epoch: $epoch")
        loss = contrastive_divergence(rbm, x_train, steps = 3, learning_rate = lr)
        println("Loss: $loss")
    end
end
