function _create_qubo_model(rbm::RBM, sampler, model_setup)
    model = Model(sampler)
    model_setup(model, sampler)
    @variable(model, vis[1:rbm.n_visible], Bin)
    @variable(model, hid[1:rbm.n_hidden], Bin)
    @objective(model, Min, -vis' * rbm.W * hid)
    return model
end

function _update_qubo_model!(model, rbm::RBM)
    @objective(
        model,
        Min,
        -model[:vis]' * rbm.W * model[:hid] - rbm.a'model[:vis] - rbm.b'model[:hid]
    )
end

function _qubo_sample(rbm::RBM, model)
    optimize!(model)
    v_sampled = zeros(Int, num_visible_nodes(rbm))
    h_sampled = zeros(Int, num_hidden_nodes(rbm))
    total_samples = result_count(model)
    for i in 1:total_samples
        v_sampled .+= value.(model[:vis], result = i)
        h_sampled .+= value.(model[:hid], result = i)
    end
    return v_sampled ./ total_samples, h_sampled ./ total_samples
end

function persistent_qubo!(
    rbm::RBM,
    model,
    x::Vector{Vector{Int}},
    mini_batches::Vector{UnitRange{Int}},
    learning_rate::Float64 = 0.1,
)
    total_t_sample, total_t_qs, total_t_update = 0.0, 0.0, 0.0
    loss = 0.0
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

            # loss by Mean Squared Error
            reconstructed = reconstruct(rbm, sample)
            loss += sum((sample .- reconstructed) .^ 2)
        end
        t_update = time()
        _update_qubo_model!(model, rbm)
        total_t_update += time() - t_update
    end
    return loss / length(x), total_t_sample, total_t_qs, total_t_update
end
