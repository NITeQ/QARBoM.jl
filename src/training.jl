function train(
    rbm::BernoulliRBM,
    x_train::Vector{Vector{Int}},
    method::AbstractMethod;
    n_epochs::Int,
    cd_steps::Int = 3,
    learning_rate::Float64,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    for epoch = 1:n_epochs
        avg_loss, t_sample, t_gibbs, t_update = contrastive_divergence(
            rbm,
            x_train,
            method;
            steps = cd_steps,
            learning_rate = learning_rate,
        )
        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update
        println(
            "|--------------------------------------------------------------------------|",
        )
        println(
            "| Epoch | Avg. Loss | Time (Sample) | Time (Gibbs) | Time (Update) | Total |",
        )
        println(
            "|--------------------------------------------------------------------------|",
        )
        println(
            @sprintf(
                "| %5d | %9.4f | %13.4f | %12.4f | %13.4f | %5.2f |",
                epoch,
                avg_loss,
                t_sample,
                t_gibbs,
                t_update,
                total_t_sample + total_t_gibbs + total_t_update,
            )
        )
        println(
            "|--------------------------------------------------------------------------|",
        )
    end
    println("Finished training after $n_epochs epochs.")

    println("Total time spent sampling: $total_t_sample")
    println("Total time spent in Gibbs sampling: $total_t_gibbs")
    println("Total time spent updating parameters: $total_t_update")
    println("Total time spent training: $(total_t_sample + total_t_gibbs + total_t_update)")
end
