# PCD-K mini-batch algorithm
function persistent_contrastive_divergence!(
    rbm::AbstractRBM,
    x,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyData};
    steps::Int = 1,
    learning_rate::Float64 = 0.1,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        index = 1

        t_gibbs = time()
        _update_fantasy_data!(rbm, fantasy_data, steps)
        total_t_gibbs += time() - t_gibbs

        δ_W = zeros(size(rbm.W))
        δ_a = zeros(size(rbm.a))
        δ_b = zeros(size(rbm.b))

        for sample in x[mini_batch]
            t_sample = time()
            v_data = sample # training visible
            h_data = conditional_prob_h(rbm, v_data) # hidden from training visible
            total_t_sample += time() - t_sample

            # Update hyperparameter
            t_update = time()
            δ_W += (v_data * h_data' .- fantasy_data[index].v * fantasy_data[index].h')
            δ_a += (v_data .- fantasy_data[index].v)
            δ_b += (h_data .- fantasy_data[index].h)
            total_t_update += time() - t_update

            index += 1
        end

        t_update = time()
        update_rbm!(rbm, δ_W, δ_a, δ_b, learning_rate / length(mini_batch))
        total_t_update += time() - t_update
    end
    return total_t_sample, total_t_gibbs, total_t_update
end

# PCD-K mini-batch algorithm
function persistent_contrastive_divergence!(
    rbm::Union{RBMClassifier, GRBMClassifier},
    x,
    label,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyDataClassifier};
    steps::Int = 1,
    learning_rate::Float64 = 0.1,
    label_learning_rate::Float64 = 0.1,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        index = 1

        δ_W = zeros(size(rbm.W))
        δ_U = zeros(size(rbm.U))
        δ_a = zeros(size(rbm.a))
        δ_b = zeros(size(rbm.b))
        δ_c = zeros(size(rbm.c))

        for sample_i in mini_batch
            t_sample = time()
            v_data = x[sample_i]
            y_data = label[sample_i]
            h_data = conditional_prob_h(rbm, v_data, y_data)
            total_t_sample += time() - t_sample

            # Update hyperparameter
            t_update = time()
            δ_W += (v_data * h_data' .- fantasy_data[index].v * fantasy_data[index].h')
            δ_U += (y_data * h_data' .- fantasy_data[index].y * fantasy_data[index].h')
            δ_a += (v_data .- fantasy_data[index].v)
            δ_b += (h_data .- fantasy_data[index].h)
            δ_c += (y_data .- fantasy_data[index].y)
            total_t_update += time() - t_update

            index += 1
        end

        t_update = time()
        update_rbm!(rbm, δ_W, δ_U, δ_a, δ_b, δ_c, learning_rate / length(mini_batch), label_learning_rate / length(mini_batch))
        total_t_update += time() - t_update

        # Update fantasy data
        t_gibbs = time()
        _update_fantasy_data!(rbm, fantasy_data, steps)
        total_t_gibbs += time() - t_gibbs
    end
    return total_t_sample, total_t_gibbs, total_t_update
end

"""
    train!(
        rbm::AbstractRBM,
        x_train,
        ::Type{PCD};
        n_epochs::Int,
        batch_size::Int,
        learning_rate::Vector{Float64},
        metrics::Vector{<:EvaluationMethod} = [MeanSquaredError],
        early_stopping::Bool = false,
        store_best_rbm::Bool = true,
        patience::Int = 10,
        stopping_metric::Type{<:EvaluationMethod} = MeanSquaredError,
        x_test_dataset = nothing,
        y_test_dataset = nothing,
        file_path = "pcd_metrics.csv",
    )

Train an RBM using the Persistent Contrastive Divergence (PCD) algorithm.

### Arguments

  - `rbm::AbstractRBM`: The RBM to train.
  - `x_train`: The training data.
  - `n_epochs::Int`: The number of epochs to train the RBM.
  - `batch_size::Int`: The size of the mini-batches.
  - `learning_rate::Vector{Float64}`: The learning rate for each epoch.
  - `metrics::Vector{<:EvaluationMethod}`: The evaluation metrics to use.
  - `early_stopping::Bool`: Whether to use early stopping.
  - `stopping_metric::Type{<:EvaluationMethod}`: The metric to use for early stopping.
  - `store_best_rbm::Bool`: Whether to store the rbm with the best `stopping_metric`.
  - `patience::Int`: The number of epochs to wait before stopping.
  - `patience::Int`: The number of epochs to wait before stopping.
  - `x_test_dataset`: The test data to evaluate the model. If not set the training data will be used.
  - `file_path`: The file path to store the metrics.
"""
function train!(
    rbm::AbstractRBM,
    x_train,
    ::Type{PCD};
    n_epochs::Int,
    gibbs_steps::Int = 1,
    batch_size::Int,
    learning_rate::Vector{Float64},
    metrics::Vector{<:DataType} = [MeanSquaredError],
    early_stopping::Bool = false,
    store_best_rbm::Bool = true,
    patience::Int = 10,
    stopping_metric::Type{<:EvaluationMethod} = MeanSquaredError,
    x_test_dataset = nothing,
    file_path = "pcd_metrics.csv",
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
    println("Setting mini-batches")
    mini_batches = _set_mini_batches(length(x_train), batch_size)
    fantasy_data = _init_fantasy_data(rbm, batch_size)
    println("Starting training")

    for epoch in 1:n_epochs
        t_sample, t_gibbs, t_update = persistent_contrastive_divergence!(
            rbm,
            x_train,
            mini_batches,
            fantasy_data;
            steps = gibbs_steps,
            learning_rate = learning_rate[epoch],
        )

        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update

        if !isnothing(x_test_dataset)
            evaluate(rbm, metrics, x_test_dataset, metrics_dict)
        else
            evaluate(rbm, metrics, x_train, metrics_dict)
        end

        if _diverged(metrics_dict, stopping_metric)
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

"""
    train!(
        rbm::Union{RBMClassifier, GRBMClassifier},
        x_train,
        label_train,
        ::Type{PCD};
        n_epochs::Int,
        batch_size::Int,
        learning_rate::Vector{Float64},
        label_learning_rate::Vector{Float64},
        metrics::Vector{<:EvaluationMethod} = [Accuracy],
        early_stopping::Bool = false,
        store_best_rbm::Bool = true,
        patience::Int = 10,
    stopping_metric::Type{<:EvaluationMethod} = Accuracy,
        x_test_dataset = nothing,
        y_test_dataset = nothing,
        file_path = "pcd_classifier_metrics.csv",
    )    

Train an RBM classifier using the Persistent Contrastive Divergence (PCD) algorithm.

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
  - `patience::Int`: The number of epochs to wait before stopping.
  - `x_test_dataset`: The test data to evaluate the model. If not set the training data will be used.
  - `y_test_dataset`: The test labels to evaluate the model. If not set the training labels will be used.
  - `file_path`: The file path to store the metrics.
"""
function train!(
    rbm::Union{RBMClassifier, GRBMClassifier},
    x_train,
    label_train,
    ::Type{PCD};
    n_epochs::Int,
    gibbs_steps::Int = 1,
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
    file_path = "pcd_classifier_metrics.csv",
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
    println("Setting mini-batches")
    mini_batches = _set_mini_batches(length(x_train), batch_size)
    fantasy_data = _init_fantasy_data(rbm, batch_size)
    println("Starting training")

    for epoch in 1:n_epochs
        t_sample, t_gibbs, t_update = persistent_contrastive_divergence!(
            rbm,
            x_train,
            label_train,
            mini_batches,
            fantasy_data;
            steps = gibbs_steps,
            learning_rate = learning_rate[epoch],
            label_learning_rate = label_learning_rate[epoch],
        )

        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update

        if !isnothing(x_test_dataset) && !isnothing(y_test_dataset)
            evaluate(rbm, metrics, x_test_dataset, y_test_dataset, metrics_dict)
        else
            evaluate(rbm, metrics, x_train, label_train, metrics_dict)
        end

        if _diverged(metrics_dict, stopping_metric)
            if early_stopping
                if patience == 0
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
