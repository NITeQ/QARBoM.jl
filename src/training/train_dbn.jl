# PCD-K mini-batch algorithm
function persistent_contrastive_divergence!(
    top_layer::DBNLayer,
    bottom_layer::DBNLayer,
    x,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyData};
    learning_rate::Float64 = 0.1,
    update_visible_bias::Bool = true,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        batch_index = 1
        for sample in x[mini_batch]
            t_sample = time()
            v_data = sample # training visible
            h_data = propagate_up(top_layer, bottom_layer, v_data)# hidden from training visible
            total_t_sample += time() - t_sample

            # Update hyperparameter
            t_update = time()
            update_layer!(
                top_layer,
                bottom_layer,
                v_data,
                h_data,
                fantasy_data[batch_index],
                (learning_rate / length(mini_batch));
                update_bottom_bias = update_visible_bias,
            )
            total_t_update += time() - t_update

            batch_index += 1
        end

        # Update fantasy data
        t_gibbs = time()
        _update_fantasy_data!(top_layer, bottom_layer, fantasy_data)
        total_t_gibbs += time() - t_gibbs
    end
    return total_t_sample, total_t_gibbs, total_t_update
end

function train_layer!(
    dbn::DBN,
    layer_index::Int,
    x_train;
    n_epochs::Int,
    batch_size::Int,
    learning_rate::Vector{Float64},
    update_bottom_layer::Bool = false,
    evaluation_function::Function,
    metrics::Any,
)
    top_layer = dbn.layers[layer_index+1]
    bottom_layer = dbn.layers[layer_index]
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    println("Setting mini-batches")
    mini_batches = _set_mini_batches(length(x_train), batch_size)
    fantasy_data = _init_fantasy_data(top_layer, bottom_layer, batch_size)
    println("Starting training")

    for epoch in 1:n_epochs
        t_sample, t_gibbs, t_update = persistent_contrastive_divergence!(
            top_layer,
            bottom_layer,
            x_train,
            mini_batches,
            fantasy_data;
            learning_rate = learning_rate[epoch],
            update_visible_bias = update_bottom_layer,
        )
        _log_epoch(epoch, t_sample, t_gibbs, t_update, t_sample + t_gibbs + t_update)

        evaluation_function(dbn, layer_index, epoch, metrics)
        for key in keys(metrics)
            println("Metric $key: $(metrics[key][layer_index][epoch])")
        end
    end

    return
end

function pretrain_dbn!(
    dbn::DBN,
    x_train;
    n_epochs::Vector{Int},
    batch_size::Int,
    learning_rate::Vector{Vector{Float64}},
    evaluation_function::Function,
    metrics::Any,
)
    new_x_train = copy(x_train)
    for l_i in 1:(length(dbn.layers)-1)
        update_visible_bias = l_i == 1

        if l_i > 1
            # warm Starting
            if length(dbn.layers[l_i-1].bias) == length(dbn.layers[l_i+1].bias)
                println("Warm starting the weights between layers $l_i and $(l_i+1)")
                dbn.layers[l_i].W = Matrix(copy(dbn.layers[l_i-1].W'))
                dbn.layers[l_i+1].bias = copy(dbn.layers[l_i-1].bias)
            end
        end

        println("Training layer $l_i")
        train_layer!(
            dbn,
            l_i,
            new_x_train;
            n_epochs = n_epochs[l_i],
            batch_size = batch_size,
            learning_rate = learning_rate[l_i],
            update_bottom_layer = update_visible_bias,
            evaluation_function = evaluation_function,
            metrics = metrics,
        )

        if l_i < length(dbn.layers) - 1
            new_x_train = [propagate_up(dbn.layers[l_i+1], dbn.layers[l_i], new_x_train[i]) for i in eachindex(new_x_train)]
        end
    end
end

function fine_tune_dbn!(
    dbn::DBN,
    x_train::Vector{Vector{Float64}},
    y_train::Vector{Vector{Float64}};  # Supervised labels
    learning_rate::Vector{Float64},
    n_epochs::Int = 10,
    batch_size::Int = 32,
    evaluation_function::Function,
    metrics::Any,
)
    if isnothing(dbn.label)
        dbn.label = LabelLayer(
            randn(length(y_train[1]), length(dbn.layers[end].bias)),  # Initialize label layer weights
            zeros(length(y_train[1])),  # Initialize label layer biases
        )
    end

    mini_batches = _set_mini_batches(length(x_train), batch_size)

    for epoch in 1:n_epochs
        for mini_batch in mini_batches
            # Mini-batch training
            for idx in mini_batch
                x = x_train[idx]
                y_true = y_train[idx]

                # Forward pass
                top_layer = propagate_up(dbn, x, 1, length(dbn.layers))
                y_pred = _softmax(dbn.label.W * top_layer .+ dbn.label.bias)

                # Compute loss 
                loss = cross_entropy_loss(y_true, y_pred)

                # Label layer gradient
                δ_label = y_pred .- y_true
                δ_W_label = δ_label * top_layer'
                δ_bias_label = δ_label

                # Update label layer
                dbn.label.W .-= learning_rate[epoch] .* δ_W_label
                dbn.label.bias .-= learning_rate[epoch] .* δ_bias_label

                # Backpropagate the error through the hidden layers
                δ_hidden = (dbn.label.W' * δ_label) .* _sigmoid_derivative.(top_layer)

                # Update the weights and biases of the hidden layers
                for i in length(dbn.layers):-1:2
                    δ_W = propagate_down(dbn, top_layer, i, i - 1) * δ_hidden'
                    δ_bias = δ_hidden
                    i > 2 ? dbn.layers[i-1].W .-= learning_rate[epoch] .* δ_W : nothing
                    dbn.layers[i].bias .-= learning_rate[epoch] .* δ_bias
                    i > 2 ? δ_hidden = (dbn.layers[i-1].W * δ_hidden) .* _sigmoid_derivative.(propagate_down(dbn, top_layer, i, i - 1)) : nothing
                    top_layer = propagate_down(dbn, top_layer, i, i - 1)
                end

                # Update the weights and biases of the visible layer
                δ_W = x * δ_hidden'
                δ_bias = (dbn.layers[1].W * δ_hidden) .* _sigmoid_derivative.(x)
                dbn.layers[1].W .-= learning_rate[epoch] .* δ_W
                dbn.layers[1].bias .-= learning_rate[epoch] .* δ_bias
            end
        end
        evaluation_function(dbn, epoch, metrics)
        _log_metrics(metrics, epoch)
    end
end
