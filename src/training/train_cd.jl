# CD-K algorithm
function contrastive_divergence!(rbm::AbstractRBM, x; steps::Int, learning_rate::Float64 = 0.1)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
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
    end
    return total_t_sample, total_t_gibbs, total_t_update
end

function contrastive_divergence!(rbm::RBMClassifiers, x, y; steps::Int, learning_rate::Float64 = 0.1, label_learning_rate::Float64 = 0.1)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    for sample_i in eachindex(x)
        v_data = x[sample_i]
        y_data = y[sample_i]
        t_sample = time()
        h_data = conditional_prob_h(rbm, v_data, y_data)
        total_t_sample += time() - t_sample

        t_gibbs = time()
        v_model, y_model = _get_v_y_model(rbm, v_data, y_data, steps)
        h_model = conditional_prob_h(rbm, v_model)
        total_t_gibbs += time() - t_gibbs

        # Update hyperparameter
        t_update = time()
        update_rbm!(rbm, v_data, h_data, y_data, v_model, h_model, y_model, learning_rate, label_learning_rate)
        total_t_update += time() - t_update
    end
    return total_t_sample, total_t_gibbs, total_t_update
end

function train!(
    rbm::AbstractRBM,
    x_train,
    ::Type{CD};
    n_epochs::Int,
    gibbs_steps::Int = 3,
    learning_rate::Vector{Float64},
    metrics::Vector{<:DataType} = [MeanSquaredError],
    early_stopping::Bool = false,
    store_best_rbm::Bool = true,
    patience::Int = 10,
    stopping_metric::Type{<:EvaluationMethod} = MeanSquaredError,
    x_test_dataset = nothing,
    file_path = "cd_metrics.csv",
)
    best_rbm = copy_rbm(rbm)
    metrics_dict = _initialize_metrics(metrics)
    initial_patience = patience

    initial_metrics = if !isnothing(x_test_dataset) && !isnothing(y_test_dataset)
        initial_evaluation(rbm, metrics, x_test_dataset, y_test_dataset)
    else
        initial_evaluation(rbm, metrics, x_train)
    end

    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0

    for epoch in 1:n_epochs
        t_sample, t_gibbs, t_update = contrastive_divergence!(
            rbm,
            x_train;
            steps = gibbs_steps,
            learning_rate = learning_rate[epoch],
        )
        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update

        if !isnothing(x_test_dataset)
            evaluate(rbm, metrics, x_test_dataset, metrics_dict, epoch)
        else
            evaluate(rbm, metrics, x_train, metrics_dict, epoch)
        end

        if _diverged(metrics_dict, epoch, stopping_metric)
            if early_stopping
                if patience == 0
                    println("Early stopping at epoch $epoch")
                    break
                end
                patience -= 1
            end
        else
            patience = initial_patience
            if store_best_rbm
                copy_rbm!(rbm, best_rbm)
            end
        end

        _log_epoch(epoch, t_sample, t_gibbs, t_update, total_t_sample + total_t_gibbs + total_t_update)
        _log_metrics(metrics_dict, epoch)
    end

    if store_best_rbm
        copy_rbm!(best_rbm, rbm)
    end

    metrics_dict = merge_metrics(initial_metrics, metrics_dict)

    CSV.write(file_path, DataFrame(metrics_dict))

    _log_finish(n_epochs, total_t_sample, total_t_gibbs, total_t_update)
    return
end

function train!(
    rbm::RBMClassifiers,
    x_train,
    label_train,
    ::Type{CD};
    n_epochs::Int,
    gibbs_steps::Int = 3,
    learning_rate::Vector{Float64},
    label_learning_rate::Vector{Float64},
    metrics::Vector{<:DataType} = [Accuracy],
    early_stopping::Bool = false,
    store_best_rbm::Bool = true,
    patience::Int = 10,
    stopping_metric::Type{<:EvaluationMethod} = Accuracy,
    x_test_dataset = nothing,
    y_test_dataset = nothing,
    file_path = "cd_classifier_metrics.csv",
)
    best_rbm = copy_rbm(rbm)
    metrics_dict = _initialize_metrics(metrics)
    initial_patience = patience

    initial_metrics = if !isnothing(x_test_dataset) && !isnothing(y_test_dataset)
        initial_evaluation(rbm, metrics, x_test_dataset, y_test_dataset)
    else
        initial_evaluation(rbm, metrics, x_train, label_train)
    end

    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0

    for epoch in 1:n_epochs
        t_sample, t_gibbs, t_update = contrastive_divergence!(
            rbm,
            x_train,
            label_train;
            steps = gibbs_steps,
            learning_rate = learning_rate[epoch],
            label_learning_rate = label_learning_rate[epoch],
        )
        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update

        if !isnothing(x_test_dataset) && !isnothing(y_test_dataset)
            evaluate(rbm, metrics, x_test_dataset, y_test_dataset, metrics_dict, epoch)
        else
            evaluate(rbm, metrics, x_train, label_train, metrics_dict, epoch)
        end

        if _diverged(metrics_dict, epoch, stopping_metric)
            if early_stopping
                if patience == 0
                    println("Early stopping at epoch $epoch")
                    break
                end
                patience -= 1
            end
        else
            patience = initial_patience
            if store_best_rbm
                copy_rbm!(rbm, best_rbm)
            end
        end

        _log_epoch(epoch, t_sample, t_gibbs, t_update, total_t_sample + total_t_gibbs + total_t_update)
        _log_metrics(metrics_dict, epoch)
    end

    if store_best_rbm
        copy_rbm!(best_rbm, rbm)
    end

    metrics_dict = merge_metrics(initial_metrics, metrics_dict)

    CSV.write(file_path, DataFrame(metrics_dict))

    _log_finish(n_epochs, total_t_sample, total_t_gibbs, total_t_update)
    return
end
