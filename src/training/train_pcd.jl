# PCD-K mini-batch algorithm
function persistent_contrastive_divergence!(
    rbm::AbstractRBM,
    x,
    epoch::Int,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyData};
    learning_rate::Float64 = 0.1,
    evaluation_function::Union{Function, Nothing} = nothing,
    metrics = nothing,
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
                (learning_rate / length(mini_batch));
                update_visible_bias = update_visible_bias,
            )
            total_t_update += time() - t_update

            if !isnothing(evaluation_function)
                evaluation_function(rbm, sample, metrics, epoch)
            end
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
    label,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyDataClassifier};
    learning_rate::Float64 = 0.1,
    label_learning_rate::Float64 = 0.1,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        batch_index = 1
        for sample_i in eachindex(mini_batch)
            t_sample = time()
            v_data = x[sample_i]
            y_data = label[sample_i]
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
                (label_learning_rate / length(mini_batch)),
            )
            total_t_update += time() - t_update

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
    evaluation_function::Union{Function, Nothing} = nothing,
    metrics = nothing,
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
        println("Finished epoch $epoch")

        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update

        if !isnothing(evaluation_function)
            _log_epoch(epoch, t_sample, t_gibbs, t_update, total_t_sample + total_t_gibbs + total_t_update)
            _log_metrics(metrics, epoch)
        end
    end

    _log_finish(n_epochs, total_t_sample, total_t_gibbs, total_t_update)

    return
end

function train_pcd!(
    rbm::RBMClassifier,
    x_train,
    label_train,
    ::Type{PCD};
    n_epochs::Int,
    batch_size::Int,
    learning_rate::Vector{Float64},
    label_learning_rate::Vector{Float64},
    evaluation_function::Union{Function, Nothing} = nothing,
    metrics = nothing,
)
    best_rbm = copy_rbm(rbm)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    println("Setting mini-batches")
    mini_batches = _set_mini_batches(length(x_train) + length(label_train), batch_size)
    fantasy_data = _init_fantasy_data(rbm, batch_size)
    println("Starting training")

    for epoch in 1:n_epochs
        t_sample, t_gibbs, t_update = persistent_contrastive_divergence!(
            rbm,
            x_train,
            label_train,
            mini_batches,
            fantasy_data;
            learning_rate = learning_rate[epoch],
            label_learning_rate = label_learning_rate[epoch],
        )
        println("Finished epoch $epoch")

        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update

        if !isnothing(evaluation_function)
            evaluation_function(rbm, metrics, epoch)
        end

        if !isnothing(evaluation_function)
            _log_epoch(epoch, t_sample, t_gibbs, t_update, total_t_sample + total_t_gibbs + total_t_update)
            _log_metrics(metrics, epoch)
        end
    end

    _log_finish(n_epochs, total_t_sample, total_t_gibbs, total_t_update)

    return
end
