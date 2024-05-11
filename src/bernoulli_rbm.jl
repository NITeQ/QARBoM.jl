mutable struct BernoulliRBM <: AbstractRBM
    W::Matrix{Float64} # weight matrix
    a::Vector{Float64} # visible bias
    b::Vector{Float64} # hidden bias
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
end

function BernoulliRBM(n_visible::Int, n_hidden::Int)
    W = randn(n_visible, n_hidden)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return BernoulliRBM(W, a, b, n_visible, n_hidden)
end

# Energy function: -aᵀv - bᵀh - vᵀWh
function energy(rbm::BernoulliRBM, v::Vector{Int}, h::Vector{Int})
    return -rbm.a' * v - rbm.b' * h - v' * rbm.W * h
end

# Partition function: Σₕ Σᵥ exp(-E(v, h))
function _partition_function(rbm::BernoulliRBM)
    Z = 0.0
    for v in _get_permutations(num_visible_nodes(rbm))
        for h in _get_permutations(num_hidden_nodes(rbm))
            Z += exp(-energy(rbm, v, h))
        end
    end
    return Z
end

# P(v, h) = exp(-E(v, h)) / Z
function _prob(rbm::BernoulliRBM, v::Vector{Int}, h::Vector{Int})
    Z = partition_function(rbm)
    return exp(-energy(rbm, v, h)) / Z
end

# P(v) = Σₕ P(v, h)
function _prob_v(rbm::BernoulliRBM, v::Vector{Int})
    Z = partition_function(rbm)
    return sum(exp(-energy(rbm, v, h)) for h in _get_permutations(num_hidden_nodes(rbm))) /
           Z
end

# P(h) = Σᵥ P(v, h)
function _prob_h(rbm::BernoulliRBM, h::Vector{Int})
    Z = partition_function(rbm)
    return sum(exp(-energy(rbm, v, h)) for v in _get_permutations(num_visible_nodes(rbm))) /
           Z
end

# P(vᵢ = 1 | h) = sigmoid(aᵢ + Σⱼ Wᵢⱼ hⱼ)
function _prob_v_given_h(
    rbm::BernoulliRBM,
    v_i::Int,
    h::Vector{T},
) where {T<:Union{Int,Float64}}
    return _sigmoid(rbm.a[v_i] + rbm.W[v_i, :]' * h)
end

# P(hⱼ = 1 | v) = sigmoid(bⱼ + Σᵢ Wᵢⱼ vᵢ)
function _prob_h_given_v(
    rbm::BernoulliRBM,
    h_i::Int,
    v::Vector{T},
) where {T<:Union{Int,Float64}}
    return _sigmoid(rbm.b[h_i] + rbm.W[:, h_i]' * v)
end

# Free energy: -ln(Σₕ exp(-E(v, h)))
function free_energy(rbm::BernoulliRBM, v::Vector{Int})
    return -log(
        sum(
            exp(-energy(rbm, v, h)) for h in _get_permutations(num_hidden_nodes(rbm));
            init = 0.0,
        ),
    ) - rbm.a' * v
end

# Gibbs sampling
gibbs_sample_hidden(rbm::BernoulliRBM, v::Vector{Float64}) =
    [rand() < _prob_h_given_v(rbm, h_i, v) ? 1 : 0 for h_i = 1:num_hidden_nodes(rbm)]
gibbs_sample_visible(rbm::BernoulliRBM, h::Vector{Float64}) =
    [rand() < _prob_v_given_h(rbm, v_i, h) ? 1 : 0 for v_i = 1:num_visible_nodes(rbm)]

conditional_prob_h(rbm::BernoulliRBM, v::Vector{T}) where {T<:Union{Int,Float64}} =
    [_prob_h_given_v(rbm, h_i, v) for h_i = 1:num_hidden_nodes(rbm)]
conditional_prob_v(rbm::BernoulliRBM, h::Vector{T}) where {T<:Union{Int,Float64}} =
    [_prob_v_given_h(rbm, v_i, h) for v_i = 1:num_visible_nodes(rbm)]

function reconstruct(rbm::BernoulliRBM, v::Vector{Int})
    h = conditional_prob_h(rbm, v)
    v_reconstructed = conditional_prob_v(rbm, h)
    return v_reconstructed
end
