# Estimates v~ from the RBM model using the Contrastive Divergence algorithm
function get_v_estimate(rbm::BernoulliRBM, v::Vector{Int}, n_gibbs::Int)
    for _ = 1:n_gibbs
        h = gibbs_sample_hidden(rbm, v)
        v = gibbs_sample_visible(rbm, h)
    end
    return v
end

# CD-K algorithm
function contrastive_divergence(
    rbm::BernoulliRBM,
    x,
    ::CD;
    steps::Int,
    learning_rate::Float64 = 0.1,
)
    loss = 0.0
    total_t_sample = 0.0
    total_t_gibbs = 0.0
    total_t_update = 0.0
    for sample in x
        t_sample = time()
        v_test = sample # training visible
        h_test = conditional_prob_h(rbm, v_test) # hidden from training visible
        total_t_sample += time() - t_sample

        t_gibbs = time()
        v_estimate = get_v_estimate(rbm, v_test, steps) # v~
        h_estimate = conditional_prob_h(rbm, v_estimate) # h~
        total_t_gibbs += time() - t_gibbs

        # Update hyperparameter
        t_update = time()
        rbm.W .+= learning_rate * (v_test * h_test' .- v_estimate * h_estimate')
        rbm.a .+= learning_rate * (v_test .- v_estimate)
        rbm.b .+= learning_rate * (h_test .- h_estimate)
        total_t_update += time() - t_update

        # Update loss
        loss += sum(abs.((v_test - v_estimate)))
    end
    return loss / length(x), total_t_sample, total_t_gibbs, total_t_update
end

# PCD-K algorithm
# Tieleman (2008) "Training restricted Boltzmann machines using approximations to the likelihood gradient"
function contrastive_divergence(
    rbm::BernoulliRBM,
    x,
    ::PCD;
    steps::Int,
    learning_rate::Float64 = 0.1,
)
    loss = 0.0
    total_t_sample = 0.0
    total_t_gibbs = 0.0
    total_t_update = 0.0
    v_persistent = nothing
    for sample in x
        t_sample = time()
        v_test = sample # training visible
        h_test = conditional_prob_h(rbm, v_test) # hidden from training visible
        total_t_sample += time() - t_sample

        if isnothing(v_persistent)
            v_persistent = v_test
        end

        t_gibbs = time()
        v_persistent = get_v_estimate(rbm, v_persistent, steps) # v~
        h_estimate = conditional_prob_h(rbm, v_persistent) # h~
        total_t_gibbs += time() - t_gibbs


        # Update hyperparameters
        t_update = time()
        rbm.W += learning_rate * (v_test * h_test' - v_persistent * h_estimate')
        rbm.a += learning_rate * (v_test - v_persistent)
        rbm.b += learning_rate * (h_test - h_estimate)

        # Update loss
        loss += sum(abs.((v_test - v_persistent)))
        total_t_update = time() - t_update
    end
    return loss / length(x), total_t_sample, total_t_gibbs, total_t_update
end
