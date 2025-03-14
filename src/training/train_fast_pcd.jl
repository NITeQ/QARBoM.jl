function fast_persistent_contrastive_divergence!(
    rbm::AbstractRBM,
    x,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyData};
    steps::Int = 1,
    learning_rate::Float64 = 0.1,
    fast_learning_rate::Float64 = 0.1,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0

    W_fast = zeros(num_visible_nodes(rbm), num_hidden_nodes(rbm))
    a_fast = zeros(num_visible_nodes(rbm))
    b_fast = zeros(num_hidden_nodes(rbm))
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

            W_fast =
                W_fast .* (19 / 20) +
                (fast_learning_rate / length(mini_batch)) .*
                (v_data * h_data' .- fantasy_data[batch_index].v * fantasy_data[batch_index].h')
            a_fast =
                a_fast .* (19 / 20) +
                (fast_learning_rate / length(mini_batch)) .* (v_data .- fantasy_data[batch_index].v)
            b_fast =
                b_fast .* (19 / 20) +
                (fast_learning_rate / length(mini_batch)) .* (h_data .- fantasy_data[batch_index].h)

            total_t_update += time() - t_update

            batch_index += 1
        end

        # Update fantasy data
        t_gibbs = time()
        _update_fantasy_data!(rbm, fantasy_data, W_fast, a_fast, b_fast, steps)
        total_t_gibbs += time() - t_gibbs
    end
    return total_t_sample, total_t_gibbs, total_t_update
end

function fast_persistent_contrastive_divergence!(
    rbm::RBMClassifiers,
    x,
    label,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyDataClassifier};
    steps::Int = 1,
    learning_rate::Float64 = 0.1,
    label_learning_rate::Float64 = 0.1,
    fast_learning_rate::Float64 = 0.1,
    fast_label_learning_rate::Float64 = 0.1,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0

    W_fast = zeros(num_visible_nodes(rbm), num_hidden_nodes(rbm))
    U_fast = zeros(num_label_nodes(rbm), num_hidden_nodes(rbm))
    a_fast = zeros(num_visible_nodes(rbm))
    b_fast = zeros(num_hidden_nodes(rbm))
    c_fast = zeros(num_label_nodes(rbm))
    for mini_batch in mini_batches
        batch_index = 1
        for sample_i in mini_batch
            t_sample = time()
            v_data = x[sample_i]
            y_data = label[sample_i]
            h_data = conditional_prob_h(rbm, v_data, y_data)
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

            W_fast =
                W_fast .* (19 / 20) +
                (fast_learning_rate / length(mini_batch)) .*
                (v_data * h_data' .- fantasy_data[batch_index].v * fantasy_data[batch_index].h')
            U_fast =
                U_fast .* (19 / 20) +
                (fast_label_learning_rate / length(mini_batch)) .*
                (y_data * h_data' .- fantasy_data[batch_index].y * fantasy_data[batch_index].h')
            a_fast =
                a_fast .* (19 / 20) +
                (fast_learning_rate / length(mini_batch)) .* (v_data .- fantasy_data[batch_index].v)
            b_fast =
                b_fast .* (19 / 20) +
                (fast_learning_rate / length(mini_batch)) .* (h_data .- fantasy_data[batch_index].h)
            c_fast =
                c_fast .* (19 / 20) +
                (fast_label_learning_rate / length(mini_batch)) .* (y_data .- fantasy_data[batch_index].y)

            total_t_update += time() - t_update
        end

        # Update fantasy data
        t_gibbs = time()
        _update_fantasy_data!(rbm, fantasy_data, W_fast, U_fast, a_fast, b_fast, c_fast, steps)
        total_t_gibbs += time() - t_gibbs
    end
    return total_t_sample, total_t_gibbs, total_t_update
end

"""
    train!(
        rbm::AbstractRBM,
        x_train,
        ::Type{FastPCD};
        n_epochs::Int,
        gibbs_steps::Int = 1,
        batch_size::Int,
        learning_rate::Vector{Float64},
        fast_learning_rate::Float64,
        metrics::Vector{<:DataType} = [MeanSquaredError],
        early_stopping::Bool = false,
        store_best_rbm::Bool = true,
        patience::Int = 10,
        stopping_metric::Type{<:EvaluationMethod} = MeanSquaredError,
        x_test_dataset = nothing,
        file_path = "fast_pcd_metrics.csv",
    )

Train an RBM using Fast Persistent Contrastive Divergence (FastPCD) algorithm.

Tieleman and Hinton (2009) "Using fast weights to improve persistent contrastive divergence"

### Arguments

  - `rbm::AbstractRBM`: The RBM to train.
  - `x_train`: The training data.
  - `n_epochs::Int`: The number of epochs to train the RBM.
  - `gibbs_steps::Int`: The number of Gibbs Sampling steps to use.
  - `batch_size::Int`: The size of the mini-batches.
  - `learning_rate::Vector{Float64}`: The learning rate for each epoch.
  - `fast_learning_rate::Float64`: The fast learning rate.
  - `metrics::Vector{<:EvaluationMethod}`: The evaluation metrics to use.
  - `early_stopping::Bool`: Whether to use early stopping.
  - `stopping_metric::Type{<:EvaluationMethod}`: The metric to use for early stopping.
  - `store_best_rbm::Bool`: Whether to store the rbm with the best `stopping_metric`.
  - `patience::Int`: The number of epochs to wait before stopping.
  - `x_test_dataset`: The test data.
  - `file_path`: The file path to save the metrics.
"""
function train!(
    rbm::AbstractRBM,
    x_train,
    ::Type{FastPCD};
    n_epochs::Int,
    gibbs_steps::Int = 1,
    batch_size::Int,
    learning_rate::Vector{Float64},
    fast_learning_rate::Float64,
    metrics::Vector{<:DataType} = [MeanSquaredError],
    early_stopping::Bool = false,
    store_best_rbm::Bool = true,
    patience::Int = 10,
    stopping_metric::Type{<:EvaluationMethod} = MeanSquaredError,
    x_test_dataset = nothing,
    file_path = "fast_pcd_metrics.csv",
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
        t_sample, t_gibbs, t_update = fast_persistent_contrastive_divergence!(
            rbm,
            x_train,
            mini_batches,
            fantasy_data;
            steps = gibbs_steps,
            learning_rate = learning_rate[epoch],
            fast_learning_rate = fast_learning_rate,
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

"""
    train!(
        rbm::Union{RBMClassifier, GRBMClassifier},
        x_train,
        label_train,
        ::Type{FastPCD};
        n_epochs::Int,
        gibbs_steps::Int = 1,
        batch_size::Int,
        learning_rate::Vector{Float64},
        fast_learning_rate::Float64,
        label_learning_rate::Vector{Float64},
        metrics::Vector{<:DataType} = [Accuracy],
        early_stopping::Bool = false,
        store_best_rbm::Bool = true,
        patience::Int = 10,
        stopping_metric::Type{<:EvaluationMethod} = Accuracy,
        x_test_dataset = nothing,
        y_test_dataset = nothing,
        file_path = "fast_pcd_classifier_metrics.csv",
    )

Train an RBM using Fast Persistent Contrastive Divergence (FastPCD) algorithm.

Tieleman and Hinton (2009) "Using fast weights to improve persistent contrastive divergence"

### Arguments

  - `rbm::RBMClassifier`: The RBM to train.
  - `x_train`: The training data.
  - `label_train`: The training labels.
  - `n_epochs::Int`: The number of epochs to train the RBM.
  - `gibbs_steps::Int`: The number of Gibbs Sampling steps to use.
  - `batch_size::Int`: The size of the mini-batches.
  - `learning_rate::Vector{Float64}`: The learning rate for each epoch.
  - `fast_learning_rate::Float64`: The fast learning rate.
  - `label_learning_rate::Vector{Float64}`: The label learning rate for each epoch.
  - `fast_label_learning_rate::Float64`: The fast label learning rate.
  - `metrics::Vector{<:EvaluationMethod}`: The evaluation metrics to use.
  - `early_stopping::Bool`: Whether to use early stopping.
  - `stopping_metric::Type{<:EvaluationMethod}`: The metric to use for early stopping.
  - `store_best_rbm::Bool`: Whether to store the rbm with the best `stopping_metric`.
  - `patience::Int`: The number of epochs to wait before stopping.
  - `x_test_dataset`: The test data.
  - `y_test_dataset`: The test labels.
  - `file_path`: The file path to save the metrics.
"""
function train!(
    rbm::RBMClassifiers,
    x_train,
    label_train,
    ::Type{FastPCD};
    n_epochs::Int,
    gibbs_steps::Int = 1,
    batch_size::Int,
    learning_rate::Vector{Float64},
    fast_learning_rate::Float64 = 0.1,
    label_learning_rate::Vector{Float64},
    fast_label_learning_rate::Float64 = 0.1,
    metrics::Vector{<:DataType} = [Accuracy],
    early_stopping::Bool = false,
    store_best_rbm::Bool = true,
    patience::Int = 10,
    stopping_metric::Type{<:EvaluationMethod} = Accuracy,
    x_test_dataset = nothing,
    y_test_dataset = nothing,
    file_path = "fast_pcd_classifier_metrics.csv",
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
        t_sample, t_gibbs, t_update = fast_persistent_contrastive_divergence!(
            rbm,
            x_train,
            label_train,
            mini_batches,
            fantasy_data;
            steps = gibbs_steps,
            learning_rate = learning_rate[epoch],
            label_learning_rate = label_learning_rate[epoch],
            fast_learning_rate = fast_learning_rate,
            fast_label_learning_rate = fast_label_learning_rate,
        )

        total_t_sample += t_sample
        total_t_gibbs += t_gibbs
        total_t_update += t_update

        if !isnothing(x_test_dataset)
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
