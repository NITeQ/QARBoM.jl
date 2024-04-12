# Estimates v~ from the QARBoM model using the Contrastive Divergence algorithm
function get_v_estimate(rbm::BernoulliRBM, v::Vector{Int}, n_gibbs::Int)
    for _ = 1:n_gibbs
        h = sample_hidden(rbm, v)
        v = sample_visible(rbm, h)
    end
    return v
end

# CD-K algorithm
function contrastive_divergence(
    rbm::BernoulliRBM,
    x;
    steps::Int,
    learning_rate::Float64 = 0.1,
)
    loss = 0.0
    for sample in x
        v_test = sample # training visible
        h_test = sample_hidden(rbm, v_test) # hidden from training visible
        v_estimate = get_v_estimate(rbm, v_test, steps) # v~
        h_estimate = sample_hidden(rbm, v_estimate) # h~

        # Update hyperparameters
        rbm.W += learning_rate * (v_test * h_test' - v_estimate * h_estimate')
        rbm.a += learning_rate * (v_test - v_estimate)
        rbm.b += learning_rate * (h_test - h_estimate)

        # Update loss
        loss += sum((v_test - v_estimate) .^ 2)
    end
    return loss
end
