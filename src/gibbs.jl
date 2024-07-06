
# Gibbs sampling
gibbs_sample_hidden(rbm::RBM, v::Vector{T}) where {T <: Union{Int, Float64}} =
    [rand() < _prob_h_given_v(rbm, h_i, v) ? 1 : 0 for h_i in 1:num_hidden_nodes(rbm)]
gibbs_sample_hidden(
    rbm::RBM,
    v::Vector{T},
    W_fast::Matrix{Float64},
    b_fast::Vector{Float64},
) where {T <: Union{Int, Float64}} = [
    rand() < _prob_h_given_v(rbm, W_fast, b_fast, h_i, v) ? 1 : 0 for
    h_i in 1:num_hidden_nodes(rbm)
]
gibbs_sample_visible(rbm::RBM, h::Vector{T}) where {T <: Union{Int, Float64}} =
    [rand() < _prob_v_given_h(rbm, v_i, h) ? 1 : 0 for v_i in 1:num_visible_nodes(rbm)]
gibbs_sample_visible(
    rbm::RBM,
    h::Vector{T},
    W_fast::Matrix{Float64},
    a_fast::Vector{Float64},
) where {T <: Union{Int, Float64}} = [
    rand() < _prob_v_given_h(rbm, W_fast, a_fast, v_i, h) ? 1 : 0 for
    v_i in 1:num_visible_nodes(rbm)
]
