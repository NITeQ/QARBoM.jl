function _create_qubo_model(bottom_layer::DBNLayer, top_layer::DBNLayer, sampler, model_setup; label_size::Int = 0)
    model = Model(() -> ToQUBO.Optimizer(sampler))
    model_setup(model, sampler)

    if label_size > 0
        x_size = length(bottom_layer.bias) - label_size
        if bottom_layer isa GaussianVisibleLayer
            @variable(model, bottom_layer.min_visible[i] <= vis[i = 1:length(bottom_layer.bias[1:x_size])] <= bottom_layer.max_visible[i])
        else
            @variable(model, vis[1:length(bottom_layer.bias[1:x_size])], Bin)
        end
        @variable(model, label[1:label_size], Bin)
        @variable(model, hid[1:length(top_layer.bias)], Bin)
        @objective(model, Min, -vcat(vis, label)' * bottom_layer.W * hid)
    else
        if bottom_layer isa GaussianVisibleLayer
            @variable(model, bottom_layer.min_visible[i] <= vis[i = 1:length(bottom_layer.bias)] <= bottom_layer.max_visible[i])
        else
            @variable(model, vis[1:length(bottom_layer.bias)], Bin)
        end
        @variable(model, hid[1:length(top_layer.bias)], Bin)
        @objective(model, Min, -vis' * bottom_layer.W * hid)
    end

    return model
end

function _create_qubo_model(rbm::RBM, sampler, model_setup)
    model = Model(sampler)
    model_setup(model, sampler)
    @variable(model, vis[1:rbm.n_visible], Bin)
    @variable(model, hid[1:rbm.n_hidden], Bin)
    @objective(model, Min, -vis' * rbm.W * hid)
    return model
end

function _create_qubo_model(rbm::GRBM, sampler, model_setup)
    model = Model(() -> ToQUBO.Optimizer(sampler))
    model_setup(model, sampler)
    @variable(model, rbm.min_visible[i] <= vis[i = 1:rbm.n_visible] <= rbm.max_visible[i])
    @variable(model, hid[1:rbm.n_hidden], Bin)
    @objective(model, Min, -vis' * rbm.W * hid)
    return model
end

function _create_qubo_model(rbm::RBMClassifier, sampler, model_setup)
    model = Model(() -> ToQUBO.Optimizer(sampler))
    model_setup(model, sampler)
    @variable(model, rbm.min_visible[i] <= vis[i = 1:rbm.n_visible] <= rbm.max_visible[i])
    @variable(model, label[1:rbm.n_classifiers], Bin)
    @variable(model, hid[1:rbm.n_hidden], Bin)
    @objective(model, Min, -vis' * rbm.W * hid - label' * rbm.U * hid)
    return model
end

function _update_qubo_model!(model, bottom_layer::DBNLayer, top_layer::DBNLayer; label_size::Int = 0)
    if label_size > 0
        @objective(
            model,
            Min,
            -vcat(model[:vis], model[:label])' * bottom_layer.W * model[:hid] - bottom_layer.bias'vcat(model[:vis], model[:label]) -
            top_layer.bias'model[:hid]
        )
    else
        @objective(
            model,
            Min,
            -model[:vis]' * bottom_layer.W * model[:hid] - bottom_layer.bias'model[:vis] - top_layer.bias'model[:hid]
        )
    end
end

function _update_qubo_model!(model, rbm::AbstractRBM)
    @objective(
        model,
        Min,
        -model[:vis]' * rbm.W * model[:hid] - rbm.a'model[:vis] - rbm.b'model[:hid]
    )
end

function _update_qubo_model!(model, rbm::RBMClassifier)
    @objective(
        model,
        Min,
        -model[:vis]' * rbm.W * model[:hid] - rbm.a'model[:vis] - rbm.b'model[:hid] - model[:label]' * rbm.U * model[:hid] - rbm.c'model[:label]
    )
end

function _qubo_sample(model; has_label::Bool = false)
    optimize!(model)
    v_samples = [value.(model[:vis], result = i) for i in 1:result_count(model)]
    h_samples = [value.(model[:hid], result = i) for i in 1:result_count(model)]
    has_label ? label_samples = [value.(model[:label], result = i) for i in 1:result_count(model)] : nothing
    v_sampled = sum(v_samples) / result_count(model)
    h_sampled = sum(h_samples) / result_count(model)
    has_label ? label_sampled = sum(label_samples) / result_count(model) : nothing
    if has_label
        return vcat(v_sampled, label_sampled), h_sampled
    else
        return v_sampled, h_sampled
    end
end

function _qubo_sample(rbm::AbstractRBM, model)
    optimize!(model)
    v_sampled = zeros(Float64, num_visible_nodes(rbm))
    h_sampled = zeros(Float64, num_hidden_nodes(rbm))
    total_samples = result_count(model)
    for i in 1:total_samples
        v_sampled .+= value.(model[:vis], result = i)
        h_sampled .+= value.(model[:hid], result = i)
    end
    return v_sampled ./ total_samples, h_sampled ./ total_samples
end

function _qubo_sample(rbm::RBMClassifier, model)
    optimize!(model)
    v_sampled = zeros(Float64, num_visible_nodes(rbm))
    label_sampled = zeros(Float64, num_label_nodes(rbm))
    h_sampled = zeros(Float64, num_hidden_nodes(rbm))
    total_samples = result_count(model)
    for i in 1:total_samples
        v_sampled .+= vcat(value.(model[:vis], result = i))
        h_sampled .+= value.(model[:hid], result = i)
        label_sampled .+= value.(model[:label], result = i)
    end
    return v_sampled ./ total_samples, h_sampled ./ total_samples, label_sampled ./ total_samples
end

function persistent_qubo!(
    rbm::AbstractRBM,
    model,
    x,
    epoch::Int,
    mini_batches::Vector{UnitRange{Int}};
    learning_rate::Float64 = 0.1,
    evaluation_function::Function,
    metrics::Any,
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

            evaluation_function(rbm, sample, metrics, epoch)
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
    epoch::Int,
    mini_batches::Vector{UnitRange{Int}};
    learning_rate::Float64 = 0.1,
    evaluation_function::Function,
    metrics::Any,
)
    total_t_sample, total_t_qs, total_t_update = 0.0, 0.0, 0.0
    for mini_batch in mini_batches
        t_qs = time()
        v_model, h_model, label_model = _qubo_sample(rbm, model) # v~, h~
        total_t_qs += time() - t_qs
        for sample in x[mini_batch]
            t_sample = time()
            v_data, label_data = sample # training visible
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
            )
            total_t_update += time() - t_update

            evaluation_function(rbm, sample, metrics, epoch)
        end
        t_update = time()
        _update_qubo_model!(model, rbm)
        total_t_update += time() - t_update
    end
    return total_t_sample, total_t_qs, total_t_update
end
