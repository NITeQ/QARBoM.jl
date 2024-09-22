# CD-K algorithm
function contrastive_divergence!(rbm::AbstractRBM, x; steps::Int, learning_rate::Float64 = 0.1)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    loss = 0.0
    for sample in x
        t_sample = time()
        v_data = sample # training visible
        h_data = conditional_prob_h(rbm, v_data) # hidden from training visible
        total_t_sample += time() - t_sample

        t_gibbs = time()
        v_model = _get_v_model(rbm, v_data, steps) # v~
        h_model = conditional_prob_h(rbm, v_model) # h~
        total_t_gibbs += time() - t_gibbs

        # Update hyperparameter
        t_update = time()
        update_rbm!(rbm, v_data, h_data, v_model, h_model, learning_rate)
        total_t_update += time() - t_update

        # Update loss
        reconstructed = reconstruct(rbm, sample)
        loss += sum((sample .- reconstructed) .^ 2)
    end
    return loss / length(x), total_t_sample, total_t_gibbs, total_t_update
end

function train_cd!(
    rbm::AbstractRBM,
    x_train::Vector{Vector{Int}};
    n_epochs::Int,
    cd_steps::Int = 3,
    learning_rate::Float64,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    avg_loss_vector = Vector{Float64}(undef, n_epochs)
    for epoch in 1:n_epochs
        avg_loss, t_sample, t_gibbs, t_update = contrastive_divergence!(
            rbm,
            x_train;
            steps = cd_steps,
            learning_rate = learning_rate,
        )
        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update
        println(
            "|------------------------------------------------------------------------------|",
        )
        println(
            "| Epoch | Avg. Loss | Time (Sample) | Time (Gibbs) | Time (Update) | Total     |",
        )
        println(
            "|------------------------------------------------------------------------------|",
        )
        println(
            @sprintf(
                "| %5d | %9.4f | %13.4f | %12.4f | %13.4f | %9.4f |",
                epoch,
                avg_loss,
                t_sample,
                t_gibbs,
                t_update,
                total_t_sample + total_t_gibbs + total_t_update,
            )
        )
        println(
            "|------------------------------------------------------------------------------|",
        )

        avg_loss_vector[epoch] = avg_loss
    end
    println("Finished training after $n_epochs epochs.")

    println("Total time spent sampling: $total_t_sample")
    println("Total time spent in Gibbs sampling: $total_t_gibbs")
    println("Total time spent updating parameters: $total_t_update")
    println("Total time spent training: $(total_t_sample + total_t_gibbs + total_t_update)")
    return avg_loss_vector
end
