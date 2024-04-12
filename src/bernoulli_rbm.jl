struct BernoulliRBM <: AbstractRBM
    W::Matrix{Float64} # weight matrix
    a::Vector{Float64} # visible bias
    b::Vector{Float64} # hidden bias
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
end

function BernoulliRBM(n_visible::Int, n_hidden::Int)
    W = randn(n_visible, n_hidden) * 0.01
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return BernoulliRBM(W, a, b, n_visible, n_hidden)
end

function energy(rbm::BernoulliRBM, v::Vector{Float64}, h::Vector{Float64})
    return -dot(rbm.a, v) - dot(rbm.b, h) - dot(v, rbm.W * h)
end