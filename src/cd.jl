# Estimates v~ from the RBM model using the Contrastive Divergence algorithm
function _get_v_model(rbm::RBM, v::Vector{Int}, n_gibbs::Int)
    for _ in 1:n_gibbs
        h = gibbs_sample_hidden(rbm, v)
        v = gibbs_sample_visible(rbm, h)
    end
    return v
end

# CD-K algorithm
function contrastive_divergence!(rbm::RBM, x; steps::Int, learning_rate::Float64 = 0.1)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    loss = 0.0
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

        # Update loss
        reconstructed = reconstruct(rbm, sample)
        loss += sum((sample .- reconstructed) .^ 2)
    end
    return loss / length(x), total_t_sample, total_t_gibbs, total_t_update
end
