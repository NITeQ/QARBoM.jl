
# Gibbs sampling
function gibbs_sample_hidden(rbm::AbstractRBM, v::Vector{<:Number})
    probs = conditional_prob_h(rbm, v)
    return [rand() < p ? 1 : 0 for p in probs]
end

function gibbs_sample_hidden(rbm::AbstractRBM, v::Vector{<:Number}, W_fast::Matrix{Float64}, b_fast::Vector{Float64})
    probs = conditional_prob_h(rbm, v, W_fast, b_fast)
    return [rand() < p ? 1 : 0 for p in probs]
end

function gibbs_sample_visible(rbm::AbstractRBM, h::Vector{<:Number})
    probs = conditional_prob_v(rbm, h)
    return [rand() < p ? 1 : 0 for p in probs]
end

function gibbs_sample_visible(rbm::AbstractRBM, h::Vector{<:Number}, W_fast::Matrix{Float64}, a_fast::Vector{Float64})
    probs = conditional_prob_v(rbm, h, W_fast, a_fast)
    return [rand() < p ? 1 : 0 for p in probs]
end

# for classification

function gibbs_sample_hidden(rbm::RBMClassifier, v::Vector{<:Number}, y::Vector{<:Number})
    probs = conditional_prob_h(rbm, v, y)
    return [rand() < p ? 1 : 0 for p in probs]
end

function gibbs_sample_hidden(
    rbm::RBMClassifier,
    v::Vector{<:Number},
    y::Vector{<:Number},
    W_fast::Matrix{Float64},
    U_fast::Matrix{Float64},
    b_fast::Vector{Float64},
)
    probs = conditional_prob_h(rbm, v, y, W_fast, U_fast, b_fast)
    return [rand() < p ? 1 : 0 for p in probs]
end

function gibbs_sample_label(rbm::RBMClassifier, h::Vector{<:Number})
    probs = conditional_prob_y_given_h(rbm, h)
    return [rand() < p ? 1 : 0 for p in probs]
end

function gibbs_sample_label(rbm::RBMClassifier, h::Vector{<:Number}, W_fast::Matrix{Float64}, c_fast::Vector{Float64})
    probs = conditional_prob_y_given_h(rbm, h, W_fast, c_fast)
    return [rand() < p ? 1 : 0 for p in probs]
end

# Estimates v~ from the RBM model using the Contrastive Divergence algorithm
function _get_v_model(rbm::AbstractRBM, v::Vector{<:Number}, n_gibbs::Int)
    for _ in 1:n_gibbs
        h = gibbs_sample_hidden(rbm, v)
        v = gibbs_sample_visible(rbm, h)
    end
    return v
end

function _get_v_y_model(rbm::AbstractRBM, v::Vector{<:Number}, y::Vector{<:Number}, n_gibbs::Int)
    for _ in 1:n_gibbs
        h = gibbs_sample_hidden(rbm, v, y)
        v = gibbs_sample_visible(rbm, h)
        y = gibbs_sample_label(rbm, h)
    end
    return v, y
end
