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

        _log_epoch_quantum(epoch, t_sample, t_qs, t_update, total_t_sample + total_t_qs + total_t_update)
        _log_metrics(metrics, epoch)
    end
    _log_finish_quantum(n_epochs, total_t_sample, total_t_qs, total_t_update)

    return
end
