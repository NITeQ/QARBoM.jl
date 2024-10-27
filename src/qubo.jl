function _create_qubo_model(bottom_layer::DBNLayer, top_layer::DBNLayer, sampler, model_setup; label_size::Int = 0)
    model = Model(() -> ToQUBO.Optimizer(sampler))

    if label_size > 0
        x_size = length(bottom_layer.bias) - label_size
        if bottom_layer isa GaussianVisibleLayer
            @variable(model, bottom_layer.min_visible[i] <= vis[i = 1:length(bottom_layer.bias[1:x_size])] <= bottom_layer.max_visible[i])
        else
            @variable(model, vis[1:length(bottom_layer.bias[1:x_size])], Bin)
        end
        @variable(model, label[1:label_size], Bin)
        @variable(model, hid[1:length(top_layer.bias)], Bin)
        @objective(model, Min, -vcat(vis, label)' * bottom_layer.W * hid - bottom_layer.bias'vcat(vis, label) - top_layer.bias'hid)
    else
        if bottom_layer isa GaussianVisibleLayer
            @variable(model, bottom_layer.min_visible[i] <= vis[i = 1:length(bottom_layer.bias)] <= bottom_layer.max_visible[i])
        else
            @variable(model, vis[1:length(bottom_layer.bias)], Bin)
        end
        @variable(model, hid[1:length(top_layer.bias)], Bin)
        @objective(model, Min, -vis' * bottom_layer.W * hid - bottom_layer.bias'vis - top_layer.bias'hid)
    end

    model_setup(model, sampler)
    return model
end

function _create_qubo_model(rbm::RBM, sampler, model_setup; kwargs...)
    max_visible = get(kwargs, :max_visible, nothing)
    min_visible = get(kwargs, :min_visible, nothing)

    model = Model(sampler)
    model_setup(model, sampler)
    if !isnothing(max_visible) && !isnothing(min_visible)
        @variable(model, min_visible[i] <= vis[i = 1:rbm.n_visible] <= max_visible[i])
    else
        @variable(model, vis[1:rbm.n_visible], Bin)
    end
    @variable(model, hid[1:rbm.n_hidden], Bin)
    @objective(model, Min, -vis' * rbm.W * hid)
    return model
end

function _create_qubo_model(rbm::RBMClassifier, sampler, model_setup; kwargs...)
    max_visible = get(kwargs, :max_visible, nothing)
    min_visible = get(kwargs, :min_visible, nothing)

    model = Model(() -> ToQUBO.Optimizer(sampler))
    model_setup(model, sampler)
    if !isnothing(max_visible) && !isnothing(min_visible)
        @variable(model, min_visible[i] <= vis[i = 1:rbm.n_visible] <= max_visible[i])
    else
        @variable(model, vis[1:rbm.n_visible], Bin)
    end
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
    h_sampled .+= value.(model[:hid], result = 1)
    v_sampled .+= value.(model[:vis], result = 1)
    return v_sampled, h_sampled
end

function _qubo_sample(rbm::RBMClassifier, model; kwargs...)
    optimize!(model)
    v_sampled = zeros(Float64, num_visible_nodes(rbm))
    label_sampled = zeros(Float64, num_label_nodes(rbm))
    h_sampled = zeros(Float64, num_hidden_nodes(rbm))
    v_sampled = value.(model[:vis], result = 1)
    h_sampled = value.(model[:hid], result = 1)
    label_sampled = value.(model[:label], result = 1)
    return v_sampled, h_sampled, label_sampled
end
