mutable struct RBM <: AbstractRBM
    W::Matrix{Float64} # weight matrix
    a::Vector{Float64} # visible bias
    b::Vector{Float64} # hidden bias
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
end

function RBM(n_visible::Int, n_hidden::Int)
    W = randn(n_visible, n_hidden)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return RBM(W, a, b, n_visible, n_hidden)
end

function RBM(n_visible::Int, n_hidden::Int, W::Matrix{Float64})
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return RBM(W, a, b, n_visible, n_hidden)
end

function update_rbm!(
    rbm::RBM,
    v_data::Vector{Int},
    h_data::Vector{Float64},
    v_model::Vector{Float64},
    h_model::Vector{Float64},
    learning_rate::Float64,
)
    rbm.W .+= learning_rate .* (v_data * h_data' .- v_model * h_model')
    rbm.a .+= learning_rate .* (v_data .- v_model)
    return rbm.b .+= learning_rate .* (h_data .- h_model)
end

# P(vᵢ = 1 | h) = sigmoid(aᵢ + Σⱼ Wᵢⱼ hⱼ)
function _prob_v_given_h(rbm::RBM, v_i::Int, h::Vector{T}) where {T <: Union{Int, Float64}}
    return _sigmoid(rbm.a[v_i] + rbm.W[v_i, :]' * h)
end

function _prob_v_given_h(
    rbm::RBM,
    W_fast::Matrix{Float64},
    a_fast::Vector{Float64},
    v_i::Int,
    h::Vector{T},
) where {T <: Union{Int, Float64}}
    return _sigmoid((rbm.a[v_i] + a_fast[v_i]) + (rbm.W[v_i, :] + W_fast[v_i, :])' * h)
end

# P(hⱼ = 1 | v) = sigmoid(bⱼ + Σᵢ Wᵢⱼ vᵢ)
function _prob_h_given_v(rbm::RBM, h_i::Int, v::Vector{T}) where {T <: Union{Int, Float64}}
    return _sigmoid(rbm.b[h_i] + rbm.W[:, h_i]' * v)
end

function _prob_h_given_v(
    rbm::RBM,
    W_fast::Matrix{Float64},
    b_fast::Vector{Float64},
    h_i::Int,
    v::Vector{T},
) where {T <: Union{Int, Float64}}
    return _sigmoid((rbm.b[h_i] + b_fast[h_i]) + (rbm.W[:, h_i] + W_fast[:, h_i])' * v)
end

conditional_prob_h(rbm::RBM, v::Vector{T}) where {T <: Union{Int, Float64}} =
    [_prob_h_given_v(rbm, h_i, v) for h_i in 1:num_hidden_nodes(rbm)]
conditional_prob_v(rbm::RBM, h::Vector{T}) where {T <: Union{Int, Float64}} =
    [_prob_v_given_h(rbm, v_i, h) for v_i in 1:num_visible_nodes(rbm)]

function reconstruct(rbm::RBM, v::Vector{Int})
    h = conditional_prob_h(rbm, v)
    v_reconstructed = conditional_prob_v(rbm, h)
    return v_reconstructed
end
