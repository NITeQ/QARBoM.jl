function persistent_qubo!(
    top_layer::DBNLayer,
    bottom_layer::DBNLayer,
    model,
    x,
    mini_batches::Vector{UnitRange{Int}};
    learning_rate::Float64 = 0.1,
    update_bottom_bias::Bool = false,
)
    total_t_sample, total_t_qs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        t_qs = time()
        v_model, h_model = _qubo_sample(model) # v~, h~
        total_t_qs += time() - t_qs
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
                v_model,
                h_model,
                (learning_rate / length(mini_batch));
                update_bottom_bias = update_bottom_bias,
            )
            total_t_update += time() - t_update
        end
        t_update = time()
        _update_qubo_model!(model, bottom_layer, top_layer)
        total_t_update += time() - t_update
    end
    return total_t_sample, total_t_qs, total_t_update
end

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
    x_train,
    ::Type{PCD};
    n_epochs::Int,
    batch_size::Int,
    learning_rate::Vector{Float64},
    update_bottom_layer::Bool = false,
    evaluation_function::Function,
    metrics::Any,
    sampler = nothing,
    model_setup = nothing,
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

function train_layer!(
    dbn::DBN,
    layer_index::Int,
    x_train,
    ::Type{QSampling};
    n_epochs::Int,
    batch_size::Int,
    learning_rate::Vector{Float64},
    update_bottom_layer::Bool = false,
    evaluation_function::Function,
    metrics::Any,
    sampler = nothing,
    model_setup = nothing,
)
    if isnothing(sampler) || isnothing(model_setup)
        throw(ArgumentError("Sampler and model_setup must be provided for Quantum Sampling training"))
    end

    top_layer = dbn.layers[layer_index+1]
    bottom_layer = dbn.layers[layer_index]
    total_t_sample, total_t_qs, total_t_update = 0.0, 0.0, 0.0
    println("Setting up QUBO model")
    qubo_model = _create_qubo_model(bottom_layer, top_layer, sampler, model_setup)
    println("Setting mini-batches")
    mini_batches = _set_mini_batches(length(x_train), batch_size)
    println("Starting training")

    for epoch in 1:n_epochs
        t_sample, t_qs, t_update = persistent_qubo!(
            top_layer,
            bottom_layer,
            qubo_model,
            x_train,
            mini_batches;
            learning_rate = learning_rate[epoch],
            update_bottom_bias = update_bottom_layer,
        )
        _log_epoch_quantum(epoch, t_sample, t_qs, t_update, t_sample + t_qs + t_update)

        evaluation_function(dbn, layer_index, epoch, metrics)
        for key in keys(metrics)
            println("Metric $key: $(metrics[key][layer_index][epoch])")
        end
    end

    return
end

function pretrain_dbn!(
    dbn::DBN,
    x_train,
    methods::Array{DataType, 1};
    n_epochs::Vector{Int},
    batch_size::Vector{Int},
    learning_rate::Vector{Vector{Float64}},
    evaluation_function::Function,
    metrics::Any,
    sampler = nothing,
    model_setup = nothing,
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
            new_x_train,
            methods[l_i],;
            n_epochs = n_epochs[l_i],
            batch_size = batch_size[l_i],
            learning_rate = learning_rate[l_i],
            update_bottom_layer = update_visible_bias,
            evaluation_function = evaluation_function,
            metrics = metrics,
            sampler = sampler,
            model_setup = model_setup,
        )

        if l_i < length(dbn.layers) - 1
            println("Updating the input data for layer $(l_i+1)")
            new_x_train = [propagate_up(dbn.layers[l_i+1], dbn.layers[l_i], new_x_train[i]) for i in eachindex(new_x_train)]
        end
    end
end
