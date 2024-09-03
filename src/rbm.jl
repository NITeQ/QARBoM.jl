# Bernoulli-Bernoulli RBM
mutable struct RBM <: AbstractRBM
    W::Matrix{Float64} # weight matrix
    a::Vector{Float64} # visible bias
    b::Vector{Float64} # hidden bias
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
end

# Gaussian-Bernoulli RBM
mutable struct GRBM <: AbstractRBM
    W::Matrix{Float64} # weight matrix
    a::Vector{Float64} # visible bias
    b::Vector{Float64} # hidden bias
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
    max_visible::Vector{Float64}
    min_visible::Vector{Float64}
end

# GRBM for classification
mutable struct GRBMClassifier <: AbstractRBM
    W::Matrix{Float64} # weight matrix
    a::Vector{Float64} # visible bias
    b::Vector{Float64} # hidden bias
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
    n_classifiers::Int # number of classifier bits
    max_visible::Vector{Float64}
    min_visible::Vector{Float64}
end

mutable struct RBMClassifier <: AbstractRBM
    W::Matrix{Float64} # visble-hidden weight matrix
    U::Matrix{Float64} # classifier-hidden weight matrix
    a::Vector{Float64} # visible bias
    b::Vector{Float64} # hidden bias
    c::Vector{Float64} # classifier bias
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
    n_classifiers::Int # number of classifier bits
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
    return RBM(copy(W), a, b, n_visible, n_hidden)
end

function GRBM(n_visible::Int, n_hidden::Int, max_visible::Vector{Float64}, min_visible::Vector{Float64})
    W = randn(n_visible, n_hidden)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return GRBM(W, a, b, n_visible, n_hidden, max_visible, min_visible)
end

function GRBM(n_visible::Int, n_hidden::Int, W::Matrix{Float64}, max_visible::Vector{Float64}, min_visible::Vector{Float64})
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return GRBM(copy(W), a, b, n_visible, n_hidden, max_visible, min_visible)
end

function GRBMClassifier(n_visible::Int, n_hidden::Int, n_classifiers::Int, max_visible::Vector{Float64}, min_visible::Vector{Float64})
    W = randn(n_visible, n_hidden)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return GRBMClassifier(W, a, b, n_visible, n_hidden, n_classifiers, max_visible, min_visible)
end

function GRBMClassifier(n_visible::Int, n_hidden::Int, n_classifiers::Int)
    W = randn(n_visible, n_hidden)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return GRBMClassifier(W, a, b, n_visible, n_hidden, n_classifiers, [], [])
end

function GRBMClassifier(
    n_visible::Int,
    n_hidden::Int,
    n_classifiers::Int,
    W::Matrix{Float64},
    max_visible::Vector{Float64},
    min_visible::Vector{Float64},
)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return GRBMClassifier(copy(W), a, b, n_visible, n_hidden, n_classifiers, max_visible, min_visible)
end

function GRBMClassifier(n_visible::Int, n_hidden::Int, n_classifiers::Int, W::Matrix{Float64})
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return GRBMClassifier(copy(W), a, b, n_visible, n_hidden, n_classifiers, [], [])
end

function RBMClassifier(n_visible::Int, n_hidden::Int, n_classifiers::Int)
    W = randn(n_visible, n_hidden)
    U = randn(n_classifiers, n_hidden)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    c = zeros(n_classifiers)
    return RBMClassifier(W, U, a, b, c, n_visible, n_hidden, n_classifiers)
end

function RBMClassifier(n_visible::Int, n_hidden::Int, n_classifiers::Int, W::Matrix{Float64}, U::Matrix{Float64})
    a = zeros(n_visible)
    b = zeros(n_hidden)
    c = zeros(n_classifiers)
    return RBMClassifier(copy(W), copy(U), a, b, c, n_visible, n_hidden, n_classifiers)
end

function update_rbm!(
    rbm::RBMClassifier,
    v_data::T,
    h_data::Vector{Float64},
    y_data::Vector{Float64},
    v_model::Vector{Float64},
    h_model::Vector{Float64},
    y_model::Vector{Float64},
    learning_rate::Float64,
) where {T <: Union{Vector{Int}, Vector{Float64}}}
    rbm.W .+= learning_rate .* (v_data * h_data' .- v_model * h_model')
    rbm.a .+= learning_rate .* (v_data .- v_model)
    rbm.b .+= learning_rate .* (h_data .- h_model)
    rbm.U .+= learning_rate .* (y_data * h_data' .- y_model * h_model')
    rbm.c .+= learning_rate .* (y_data .- y_model)
    return
end

function update_rbm!(
    rbm::AbstractRBM,
    v_data::T,
    h_data::Vector{Float64},
    v_model::Vector{Float64},
    h_model::Vector{Float64},
    learning_rate::Float64,
) where {T <: Union{Vector{Int}, Vector{Float64}}}
    rbm.W .+= learning_rate .* (v_data * h_data' .- v_model * h_model')
    rbm.a .+= learning_rate .* (v_data .- v_model)
    return rbm.b .+= learning_rate .* (h_data .- h_model)
end

# P(vᵢ = 1 | h) = sigmoid(aᵢ + Σⱼ Wᵢⱼ hⱼ)
function _prob_v_given_h(rbm::AbstractRBM, v_i::Int, h::Vector{T}) where {T <: Union{Int, Float64}}
    return _sigmoid(rbm.a[v_i] + rbm.W[v_i, :]' * h)
end

function _prob_v_given_h(
    rbm::AbstractRBM,
    W_fast::Matrix{Float64},
    a_fast::Vector{Float64},
    v_i::Int,
    h::Vector{T},
) where {T <: Union{Int, Float64}}
    return _sigmoid((rbm.a[v_i] + a_fast[v_i]) + (rbm.W[v_i, :] + W_fast[v_i, :])' * h)
end

# P(hⱼ = 1 | v) = sigmoid(bⱼ + Σᵢ Wᵢⱼ vᵢ)
function _prob_h_given_v(rbm::AbstractRBM, h_i::Int, v::Vector{T}) where {T <: Union{Int, Float64}}
    return _sigmoid(rbm.b[h_i] + rbm.W[:, h_i]' * v)
end

function _prob_h_given_v(
    rbm::AbstractRBM,
    W_fast::Matrix{Float64},
    b_fast::Vector{Float64},
    h_i::Int,
    v::Vector{T},
) where {T <: Union{Int, Float64}}
    return _sigmoid((rbm.b[h_i] + b_fast[h_i]) + (rbm.W[:, h_i] + W_fast[:, h_i])' * v)
end

function _prob_h_given_vy(rbm::RBMClassifier, h_i::Int, v::Vector{T}, y::Vector{T}) where {T <: Union{Int, Float64}}
    return _sigmoid(rbm.b[h_i] + rbm.W[:, h_i]' * v + rbm.U[:, h_i]' * y)
end

function _prob_y_given_h(rbm::RBMClassifier, y_i::Int, h::Vector{T}) where {T <: Union{Int, Float64}}
    return exp(rbm.c[y_i] + rbm.U[y_i, :]' * h) / sum(exp.(rbm.c .+ rbm.U * h))
end

conditional_prob_h(rbm::AbstractRBM, v::Vector{T}) where {T <: Union{Int, Float64}} =
    [_prob_h_given_v(rbm, h_i, v) for h_i in 1:num_hidden_nodes(rbm)]
conditional_prob_h(rbm::RBMClassifier, v::Vector{T}, y::Vector{T}) where {T <: Union{Int, Float64}} =
    [_prob_h_given_vy(rbm, h_i, v, y) for h_i in 1:num_hidden_nodes(rbm)]

conditional_prob_v(rbm::AbstractRBM, h::Vector{T}) where {T <: Union{Int, Float64}} =
    [_prob_v_given_h(rbm, v_i, h) for v_i in 1:num_visible_nodes(rbm)]
conditional_prob_y(rbm::RBMClassifier, h::Vector{T}) where {T <: Union{Int, Float64}} =
    [_prob_y_given_h(rbm, y_i, h) for y_i in 1:rbm.n_classifiers]


function reconstruct(rbm::AbstractRBM, v::Vector{T}) where {T <: Union{Int, Float64}}
    h = conditional_prob_h(rbm, v)
    v_reconstructed = conditional_prob_v(rbm, h)
    return v_reconstructed
end
