function persistent_qubo!(
    rbm::AbstractRBM,
    model,
    x,
    mini_batches::Vector{UnitRange{Int}};
    learning_rate::Float64 = 0.1,
)
    total_t_sample, total_t_qs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        t_qs = time()
        v_model, h_model = _qubo_sample(rbm, model) # v~, h~
        total_t_qs += time() - t_qs
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
                v_model,
                h_model,
                (learning_rate / length(mini_batch)),
            )
            total_t_update += time() - t_update
        end
        t_update = time()
        _update_qubo_model!(model, rbm)
        total_t_update += time() - t_update
    end
    return total_t_sample, total_t_qs, total_t_update
end

function persistent_qubo!(
    rbm::RBMClassifier,
    model,
    x,
    label,
    mini_batches::Vector{UnitRange{Int}};
    learning_rate::Float64 = 0.1,
    label_learning_rate::Float64 = 0.1,
    weight::Int64
)
    total_t_sample, total_t_qs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        t_qs = time()
        v_model, h_model, label_model = _qubo_sample(rbm, model) # v~, h~
        total_t_qs += time() - t_qs
        for sample_i in mini_batch
            t_sample = time()
            v_data = x[sample_i]
            label_data = label[sample_i]
            h_data = conditional_prob_h(rbm, v_data, label_data) # hidden from training visible
            total_t_sample += time() - t_sample

            # Update hyperparameter
            t_update = time()
            update_rbm!(
                rbm,
                v_data,
                h_data,
                label_data,
                v_model,
                h_model,
                label_model,
                (learning_rate / length(mini_batch)),
                (label_learning_rate / length(mini_batch)),
                weight
            )
            total_t_update += time() - t_update
        end
        t_update = time()
        _update_qubo_model!(model, rbm)
        total_t_update += time() - t_update
    end
    return total_t_sample, total_t_qs, total_t_update
end

function train!(
    rbm::AbstractRBM,
    x_train,
    ::Type{QSampling};
    n_epochs::Int,
    batch_size::Int,
    learning_rate::Vector{Float64},
    metrics::Vector{<:DataType} = [MeanSquaredError],
    early_stopping::Bool = false,
    store_best_rbm::Bool = true,
    patience::Int = 10,
    stopping_metric::Type{<:EvaluationMethod} = MeanSquaredError,
    x_test_dataset = nothing,
    file_path = "qsamp_classifier_metrics.csv",
    model_setup::Function,
    sampler,
    kwargs...,
)
    best_rbm = copy_rbm(rbm)
    metrics_dict = _initialize_metrics(metrics)
    initial_patience = patience

    println("Setting up QUBO model")
    qubo_model = _create_qubo_model(rbm, sampler, model_setup; kwargs...)
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
                mini_batches;
                learning_rate = learning_rate[epoch],
            )

        total_t_sample += t_sample
        total_t_qs += t_qs
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

        _log_epoch_quantum(epoch, t_sample, t_qs, t_update, total_t_sample + total_t_qs + total_t_update)
        _log_metrics(metrics, epoch)
    end

    if store_best_rbm
        copy_rbm!(best_rbm, rbm)
    end

    CSV.write(file_path, DataFrame(metrics_dict))

    _log_finish_quantum(n_epochs, total_t_sample, total_t_qs, total_t_update)

    return
end

"""
    train!(
        rbm::RBMClassifier,
        x_train,
        label_train,
        ::Type{QSampling};
        n_epochs::Int,
        batch_size::Int,
        learning_rate::Vector{Float64},
        label_learning_rate::Vector{Float64},
        metrics::Vector{<:DataType} = [Accuracy],
        early_stopping::Bool = false,
        store_best_rbm::Bool = true,
        patience::Int = 10,
        stopping_metric::Type{<:EvaluationMethod} = Accuracy,
        x_test_dataset = nothing,
        y_test_dataset = nothing,
        file_path = "qsamp_classifier_metrics.csv",
        model_setup::Function,
        sampler,
        kwargs...,
    )

Train an RBMClassifier using Quantum sampling.

### Arguments

  - `rbm::RBMClassifier`: The RBM classifier to train.

  - `x_train`: The training data.
  - `label_train`: The training labels.
  - `n_epochs::Int`: The number of epochs to train the RBM.
  - `batch_size::Int`: The size of the mini-batches.
  - `learning_rate::Vector{Float64}`: The learning rate for each epoch.
  - `label_learning_rate::Vector{Float64}`: The learning rate for the labels for each epoch.
  - `metrics::Vector{<:EvaluationMethod}`: The evaluation metrics to use.
  - `early_stopping::Bool`: Whether to use early stopping.
  - `stopping_metric::Type{<:EvaluationMethod}`: The metric to use for early stopping.
  - `store_best_rbm::Bool`: Whether to store the rbm with the best `stopping_metric`.
  - `patience::Int`: The number of epochs to wait before stopping.
  - `x_test_dataset`: The test data to evaluate the model. If not set the training data will be used.
  - `y_test_dataset`: The test labels to evaluate the model. If not set the training labels will be used.
  - `file_path`: The file path to store the metrics.
  - `model_setup::Function`: The function to setup the QUBO sampler.
  - `sampler`: The QUBO sampler to use.
  - `kwargs...`: Additional arguments for the QUBO sampler

      + `max_visible::Vector{Float64}`: The maximum value for the visible nodes.
      + `min_visible::Vector{Float64}`: The minimum value for the visible nodes.
"""
function train!(
    rbm::RBMClassifier,
    x_train,
    label_train,
    ::Type{QSampling};
    n_epochs::Int,
    batch_size::Int,
    learning_rate::Vector{Float64},
    label_learning_rate::Vector{Float64},
    metrics::Vector{<:DataType} = [Accuracy],
    early_stopping::Bool = false,
    store_best_rbm::Bool = true,
    patience::Int = 10,
    stopping_metric::Type{<:EvaluationMethod} = Accuracy,
    x_test_dataset = nothing,
    y_test_dataset = nothing,
    file_path = "qsamp_classifier_metrics.csv",
    model_setup::Function,
    sampler,
    weight::Int64 = 0,
    show_stats::Bool = false,
    kwargs...
)
    best_rbm = copy_rbm(rbm)
    stats = [FalsePositive,FalseNegative,TruePositive,TrueNegative]
    metrics = append!(stats,metrics)
    metrics_dict = _initialize_metrics(metrics)
    initial_patience = patience

    println("Setting up QUBO model")
    qubo_model = _create_qubo_model(rbm, sampler, model_setup; kwargs...)
    total_t_sample, total_t_qs, total_t_update = 0.0, 0.0, 0.0
    println("Setting mini-batches")
    mini_batches = _set_mini_batches(length(x_train), batch_size)
    println("Starting training")

    for epoch in 1:n_epochs
        for key in keys(metrics_dict)
            push!(metrics_dict[key], 0.0)
        end

        t_sample, t_qs, t_update =
            persistent_qubo!(
                rbm,
                qubo_model,
                x_train,
                label_train,
                mini_batches;
                learning_rate = learning_rate[epoch],
                label_learning_rate = label_learning_rate[epoch],
                weight
            )

        total_t_sample += t_sample
        total_t_qs += t_qs
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

        _log_epoch_quantum(epoch, t_sample, t_qs, t_update, total_t_sample + total_t_qs + total_t_update)
        _log_metrics(metrics_dict, epoch,show_stats)
    end

    if store_best_rbm
        copy_rbm!(best_rbm, rbm)
    end

    CSV.write(file_path, DataFrame(metrics_dict))

    _log_finish_quantum(n_epochs, total_t_sample, total_t_qs, total_t_update)

    return
end
