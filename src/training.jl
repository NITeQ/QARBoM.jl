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

function train_pcd!(
    rbm::AbstractRBM,
    x_train;
    n_epochs::Int,
    batch_size::Int,
    learning_rate::Vector{Float64},
    evaluation_function::Function,
    metrics::Any,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    println("Setting mini-batches")
    mini_batches = _set_mini_batches(length(x_train), batch_size)
    fantasy_data = _init_fantasy_data(rbm, batch_size)
    println("Starting training")

    for epoch in 1:n_epochs
        t_sample, t_gibbs, t_update = persistent_contrastive_divergence!(
            rbm,
            x_train,
            epoch,
            mini_batches,
            fantasy_data;
            evaluation_function = evaluation_function,
            learning_rate = learning_rate[epoch],
            metrics = metrics,
        )

        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update
        println(
            "|------------------------------------------------------------------------------|",
        )
        println(
            "| Epoch | Time (Sample) | Time (Gibbs) | Time (Update) | Total     |",
        )
        println(
            "|------------------------------------------------------------------------------|",
        )
        println(
            @sprintf(
                "| %5d | %13.4f | %12.4f | %13.4f | %9.4f |",
                epoch,
                t_sample,
                t_gibbs,
                t_update,
                total_t_sample + total_t_gibbs + total_t_update,
            )
        )
        println(
            "|------------------------------------------------------------------------------|",
        )
        for metric_name in keys(metrics)
            println("$metric_name: $(metrics[metric_name][epoch])")
        end
    end
    println("Finished training after $n_epochs epochs.")

    println("Total time spent sampling: $total_t_sample")
    println("Total time spent in Gibbs sampling: $total_t_gibbs")
    println("Total time spent updating parameters: $total_t_update")
    return println("Total time spent training: $(total_t_sample + total_t_gibbs + total_t_update)")
end

function train_fast_pcd!(
    rbm::AbstractRBM,
    x_train::Vector{Vector{Int}};
    n_epochs::Int,
    batch_size::Int,
    learning_rate::Vector{Float64},
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    println("Setting mini-batches")
    mini_batches = _set_mini_batches(length(x_train), batch_size)
    fantasy_data = _init_fantasy_data(rbm, batch_size)
    println("Starting training")
    avg_loss_vector = Vector{Float64}(undef, n_epochs)

    for epoch in 1:n_epochs
        avg_loss, t_sample, t_gibbs, t_update = fast_persistent_contrastive_divergence!(
            rbm,
            x_train,
            mini_batches,
            fantasy_data;
            learning_rate = learning_rate[epoch],
            fast_learning_rate = learning_rate[1],
        )

        avg_loss_vector[epoch] = avg_loss

        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update
        println(
            "|------------------------------------------------------------------------------|",
        )
        println(
            "| Epoch |    MSE    | Time (Sample) | Time (Gibbs) | Time (Update) | Total     |",
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
    end
    println("Finished training after $n_epochs epochs.")

    println("Total time spent sampling: $total_t_sample")
    println("Total time spent in Gibbs sampling: $total_t_gibbs")
    println("Total time spent updating parameters: $total_t_update")
    println("Total time spent training: $(total_t_sample + total_t_gibbs + total_t_update)")
    return avg_loss_vector
end

function train_persistent_qubo!(
    rbm::AbstractRBM,
    x_train;
    n_epochs::Int,
    batch_size::Int,
    learning_rate::Vector{Float64},
    model_setup::Function,
    sampler,
    evaluation_function::Function,
    metrics::Any,
)
    println("Setting up QUBO model")
    qubo_model = _create_qubo_model(rbm, sampler, model_setup)
    total_t_sample, total_t_qs, total_t_update = 0.0, 0.0, 0.0
    println("Setting mini-batches")
    mini_batches = _set_mini_batches(length(x_train), batch_size)
    println("Starting training")
    avg_loss_vector = Vector{Float64}(undef, n_epochs)

    for epoch in 1:n_epochs
        t_sample, t_qs, t_update =
            persistent_qubo!(
                rbm,
                qubo_model,
                x_train,
                epoch,
                mini_batches;
                learning_rate = learning_rate[epoch],
                evaluation_function = evaluation_function,
                metrics = metrics,
            )

        total_t_sample += t_sample
        total_t_qs += t_qs
        total_t_update += t_update
        println(
            "|------------------------------------------------------------------------------|",
        )
        println(
            "| Epoch | Time (Sample) | Time (Qsamp) | Time (Update) | Total     |",
        )
        println(
            "|------------------------------------------------------------------------------|",
        )
        println(
            @sprintf(
                "| %5d | %13.4f | %12.4f | %13.4f | %9.4f |",
                epoch,
                t_sample,
                t_qs,
                t_update,
                total_t_sample + total_t_qs + total_t_update,
            )
        )
        println(
            "|------------------------------------------------------------------------------|",
        )

        for metric_name in keys(metrics)
            println("$metric_name: $(metrics[metric_name][epoch])")
        end
    end
    println("Finished training after $n_epochs epochs.")

    println("Total time spent sampling: $total_t_sample")
    println("Total time spent in Quantum sampling: $total_t_qs")
    println("Total time spent updating parameters: $total_t_update")
    return println("Total time spent training: $(total_t_sample + total_t_qs + total_t_update)")
end
