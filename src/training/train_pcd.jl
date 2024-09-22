# PCD-K mini-batch algorithm
function persistent_contrastive_divergence!(
    rbm::AbstractRBM,
    x,
    epoch::Int,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyData};
    learning_rate::Float64 = 0.1,
    evaluation_function::Function,
    metrics::Any,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        batch_index = 1
        for sample in x[mini_batch]
            t_sample = time()
            v_data = sample # training visible
            h_data = conditional_prob_h(rbm, v_data) # hidden from training visible
            total_t_sample += time() - t_sample

            # Update hyperparameter
            t_update = time()
            update_rbm!(
                rbm,
                v_data,
                h_data,
                fantasy_data[batch_index].v,
                fantasy_data[batch_index].h,
                (learning_rate / length(mini_batch)),
            )
            total_t_update += time() - t_update

            evaluation_function(rbm, sample, metrics, epoch)
            batch_index += 1
        end

        # Update fantasy data
        t_gibbs = time()
        _update_fantasy_data!(rbm, fantasy_data)
        total_t_gibbs += time() - t_gibbs
    end
    return total_t_sample, total_t_gibbs, total_t_update
end

# PCD-K mini-batch algorithm
function persistent_contrastive_divergence!(
    rbm::RBMClassifier,
    x,
    epoch::Int,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyDataClassifier};
    learning_rate::Float64 = 0.1,
    evaluation_function::Function,
    metrics::Any,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        batch_index = 1
        for sample in x[mini_batch]
            t_sample = time()
            v_data, y_data = sample # training visible
            h_data = conditional_prob_h(rbm, v_data, y_data) # hidden from training visible
            total_t_sample += time() - t_sample

            # Update hyperparameter
            t_update = time()
            update_rbm!(
                rbm,
                v_data,
                h_data,
                y_data,
                fantasy_data[batch_index].v,
                fantasy_data[batch_index].h,
                fantasy_data[batch_index].y,
                (learning_rate / length(mini_batch)),
            )
            total_t_update += time() - t_update

            evaluation_function(rbm, sample, metrics, epoch)
            batch_index += 1
        end

        # Update fantasy data
        t_gibbs = time()
        _update_fantasy_data!(rbm, fantasy_data)
        total_t_gibbs += time() - t_gibbs
    end
    return total_t_sample, total_t_gibbs, total_t_update
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

        _log_epoch(epoch, t_sample, t_gibbs, t_update, total_t_sample + total_t_gibbs + total_t_update)
        _log_metrics(metrics, epoch)
    end

    _log_finish(n_epochs, total_t_sample, total_t_gibbs, total_t_update)

    return
end
