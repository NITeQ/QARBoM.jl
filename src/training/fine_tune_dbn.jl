
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
