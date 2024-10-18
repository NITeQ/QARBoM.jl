# Bernoulli-Bernoulli RBM
mutable struct RBM <: AbstractRBM
    W::Matrix{Float64} # weight matrix
    a::Vector{Float64} # visible bias
    b::Vector{Float64} # hidden bias
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
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

function RBMClassifier(n_visible::Int, n_hidden::Int, n_classifiers::Int, max_visible::Vector{Float64}, min_visible::Vector{Float64})
    W = randn(n_visible, n_hidden)
    U = randn(n_classifiers, n_hidden)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    c = zeros(n_classifiers)
    return RBMClassifier(W, U, a, b, c, n_visible, n_hidden, n_classifiers)
end

function RBMClassifier(
    n_visible::Int,
    n_hidden::Int,
    n_classifiers::Int,
    max_visible::Vector{Float64},
    min_visible::Vector{Float64},
    W::Matrix{Float64},
    U::Matrix{Float64},
)
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
    label_learning_rate::Float64,
) where {T <: Union{Vector{Int}, Vector{Float64}}}
    rbm.W .+= learning_rate .* (v_data * h_data' .- v_model * h_model')
    rbm.a .+= learning_rate .* (v_data .- v_model)
    rbm.b .+= learning_rate .* (h_data .- h_model)
    rbm.U .+= label_learning_rate .* (y_data * h_data' .- y_model * h_model')
    rbm.c .+= label_learning_rate .* (y_data .- y_model)
    return
end

function update_rbm!(
    rbm::AbstractRBM,
    v_data::T,
    h_data::Vector{Float64},
    v_model::Vector{Float64},
    h_model::Vector{Float64},
    learning_rate::Float64;
    update_visible_bias::Bool = true,
) where {T <: Union{Vector{Int}, Vector{Float64}}}
    rbm.W .+= learning_rate .* (v_data * h_data' .- v_model * h_model')
    update_visible_bias ? rbm.a .+= learning_rate .* (v_data .- v_model) : nothing
    rbm.b .+= learning_rate .* (h_data .- h_model)
    return
end

conditional_prob_h(rbm::AbstractRBM, v::Vector{T}) where {T <: Union{Int, Float64}} = _sigmoid.(rbm.b .+ rbm.W' * v)

conditional_prob_h(rbm::AbstractRBM, v::Vector{T}, W_fast::Matrix{Float64}, b_fast::Vector{Float64}) where {T <: Union{Int, Float64}} =
    _sigmoid.(rbm.b .+ b_fast .+ (rbm.W .+ W_fast)' * v)

conditional_prob_h(rbm::RBMClassifier, v::Vector{T}, y::Vector{T}) where {T <: Union{Int, Float64}} = _sigmoid.(rbm.b .+ rbm.W' * v .+ rbm.U' * y)

conditional_prob_h(
    rbm::RBMClassifier,
    v::Vector{T},
    y::Vector{T},
    W_fast::Matrix{Float64},
    U_fast::Matrix{Float64},
    b_fast::Vector{Float64},
) where {T <: Union{Int, Float64}} = _sigmoid.(rbm.b .+ b_fast .+ (rbm.W .+ W_fast)' * v .+ (rbm.U .+ U_fast)' * y)

conditional_prob_v(rbm::AbstractRBM, h::Vector{T}) where {T <: Union{Int, Float64}} = _sigmoid.(rbm.a .+ rbm.W * h)

conditional_prob_v(rbm::AbstractRBM, h::Vector{T}, W_fast::Matrix{Float64}, a_fast::Vector{Float64}) where {T <: Union{Int, Float64}} =
    _sigmoid.(rbm.a .+ a_fast .+ (rbm.W .+ W_fast) * h)

function conditional_prob_y_given_v(rbm::RBMClassifier, v::Vector{T}) where {T <: Union{Int, Float64}}
    class_probabilities = zeros(rbm.n_classifiers)

    for y_i in 1:rbm.n_classifiers
        class_probabilities[y_i] = rbm.c[y_i] + sum(log1pexp(rbm.U[y_i, h_j] + rbm.W[:, h_j]' * v + rbm.b[h_j]) for h_j in 1:rbm.n_hidden)
    end

    log_denominator = logsumexp(class_probabilities)

    return exp.(class_probabilities .- log_denominator)
end

function conditional_prob_y_given_h(rbm::RBMClassifier, h::Vector{T}) where {T <: Union{Int, Float64}}
    # Compute the log of the numerator
    log_numerator = rbm.c .+ rbm.U * h

    # Compute the log of the denominator using logsumexp for numerical stability
    log_denominator = logsumexp(rbm.c .+ rbm.U * h)

    # Return the probability p(y|h) in a numerically stable way
    return exp.(log_numerator .- log_denominator)
end

function conditional_prob_y_given_h(
    rbm::RBMClassifier,
    h::Vector{T},
    W_fast::Matrix{Float64},
    c_fast::Vector{Float64},
) where {T <: Union{Int, Float64}}
    log_numerator = (rbm.c .+ c_fast) + (rbm.U .+ W_fast) * h

    log_denominator = logsumexp((rbm.c .+ c_fast) .+ (rbm.U .+ W_fast) * h)

    return exp.(log_numerator .- log_denominator)
end

function reconstruct(rbm::AbstractRBM, v::Vector{T}) where {T <: Union{Int, Float64}}
    h = conditional_prob_h(rbm, v)
    v_reconstructed = conditional_prob_v(rbm, h)
    return v_reconstructed
end

function classify(rbm::RBMClassifier, v::Vector{T}) where {T <: Union{Int, Float64}}
    y = conditional_prob_y_given_v(rbm, v)
    return y
end

function copy_rbm(rbm::AbstractRBM)
    return RBM(copy(rbm.W), copy(rbm.a), copy(rbm.b), rbm.n_visible, rbm.n_hidden)
end
