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

mutable struct GRBMClassifier <: AbstractRBM
    W::Matrix{Float64} # visble-hidden weight matrix
    U::Matrix{Float64} # classifier-hidden weight matrix
    a::Vector{Float64} # visible bias
    b::Vector{Float64} # hidden bias
    c::Vector{Float64} # classifier bias
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
    n_classifiers::Int # number of classifier bits
end

const RBMClassifiers = Union{RBMClassifier, GRBMClassifier}

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

function GRBM(n_visible::Int, n_hidden::Int)
    W = randn(n_visible, n_hidden)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return GRBM(W, a, b, n_visible, n_hidden)
end

function GRBM(n_visible::Int, n_hidden::Int, W::Matrix{Float64})
    a = zeros(n_visible)
    b = zeros(n_hidden)
    return GRBM(copy(W), a, b, n_visible, n_hidden)
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

function GRBMClassifier(n_visible::Int, n_hidden::Int, n_classifiers::Int)
    W = randn(n_visible, n_hidden)
    U = randn(n_classifiers, n_hidden)
    a = zeros(n_visible)
    b = zeros(n_hidden)
    c = zeros(n_classifiers)
    return GRBMClassifier(W, U, a, b, c, n_visible, n_hidden, n_classifiers)
end

function GRBMClassifier(n_visible::Int, n_hidden::Int, n_classifiers::Int, W::Matrix{Float64}, U::Matrix{Float64})
    a = ones(n_visible) * 1e-5
    b = ones(n_hidden) * 1e-5
    c = ones(n_classifiers) * 1e-5
    return GRBMClassifier(copy(W), copy(U), a, b, c, n_visible, n_hidden, n_classifiers)
end

function update_rbm!(
    rbm::Union{RBMClassifier, GRBMClassifier},
    v_data::Vector{<:Number},
    h_data::Vector{<:Number},
    y_data::Vector{<:Number},
    v_model::Vector{<:Number},
    h_model::Vector{<:Number},
    y_model::Vector{<:Number},
    learning_rate::Float64,
    label_learning_rate::Float64,
)
    rbm.W .+= learning_rate .* (v_data * h_data' .- v_model * h_model')
    rbm.a .+= learning_rate .* (v_data .- v_model)
    rbm.b .+= learning_rate .* (h_data .- h_model)
    rbm.U .+= label_learning_rate .* (y_data * h_data' .- y_model * h_model')
    rbm.c .+= label_learning_rate .* (y_data .- y_model)
    return
end

function update_rbm!(
    rbm::Union{RBMClassifier, GRBMClassifier},
    δ_W::Matrix{Float64},
    δ_U::Matrix{Float64},
    δ_a::Vector{Float64},
    δ_b::Vector{Float64},
    δ_c::Vector{Float64},
    learning_rate::Float64,
    label_learning_rate::Float64,
)
    rbm.W .+= δ_W .* learning_rate
    rbm.U .+= δ_U .* label_learning_rate
    rbm.a .+= δ_a .* learning_rate
    rbm.b .+= δ_b .* learning_rate
    rbm.c .+= δ_c .* label_learning_rate
    return
end

function update_rbm!(
    rbm::AbstractRBM,
    v_data::Vector{<:Number},
    h_data::Vector{<:Number},
    v_model::Vector{<:Number},
    h_model::Vector{<:Number},
    learning_rate::Float64;
)
    rbm.W .+= learning_rate .* (v_data * h_data' .- v_model * h_model')
    rbm.a .+= learning_rate .* (v_data .- v_model)
    rbm.b .+= learning_rate .* (h_data .- h_model)
    return
end

function update_rbm!(
    rbm::AbstractRBM,
    δ_W::Matrix{Float64},
    δ_a::Vector{Float64},
    δ_b::Vector{Float64},
    learning_rate::Float64,
)
    rbm.W .+= δ_W .* learning_rate
    rbm.a .+= δ_a .* learning_rate
    rbm.b .+= δ_b .* learning_rate
    return
end

conditional_prob_h(rbm::AbstractRBM, v::Vector{<:Number}) = logistic.(rbm.b .+ rbm.W' * v)

conditional_prob_h(rbm::AbstractRBM, v::Vector{<:Number}, W_fast::Matrix{Float64}, b_fast::Vector{Float64}) =
    logistic.(rbm.b .+ b_fast .+ (rbm.W .+ W_fast)' * v)

conditional_prob_h(rbm::Union{RBMClassifier, GRBMClassifier}, v::Vector{<:Number}, y::Vector{<:Number}) =
    logistic.(rbm.b .+ rbm.W' * v .+ rbm.U' * y)

conditional_prob_h(
    rbm::Union{RBMClassifier, GRBMClassifier},
    v::Vector{<:Number},
    y::Vector{<:Number},
    W_fast::Matrix{Float64},
    U_fast::Matrix{Float64},
    b_fast::Vector{Float64},
) = logistic.(rbm.b .+ b_fast .+ (rbm.W .+ W_fast)' * v .+ (rbm.U .+ U_fast)' * y)

conditional_prob_v(rbm::AbstractRBM, h::Vector{<:Number}) = logistic.(rbm.a .+ rbm.W * h)

conditional_prob_v(rbm::AbstractRBM, h::Vector{<:Number}, W_fast::Matrix{Float64}, a_fast::Vector{Float64}) =
    logistic.(rbm.a .+ a_fast .+ (rbm.W .+ W_fast) * h)

conditional_prob_v(rbm::Union{GRBMClassifier, GRBM}, h::Vector{<:Number}) = rand.(Normal.(rbm.a .+ rbm.W * h, 1.0))
conditional_prob_v(rbm::Union{GRBMClassifier, GRBM}, h::Vector{<:Number}, W_fast::Matrix{Float64}, a_fast::Vector{Float64}) =
    rand.(Normal.(rbm.a .+ a_fast .+ (rbm.W .+ W_fast) * h, 1.0))

function conditional_prob_y_given_v(rbm::Union{RBMClassifier, GRBMClassifier}, v::Vector{<:Number})
    h = conditional_prob_h(rbm, v)
    return conditional_prob_y_given_h(rbm, h)
end

function conditional_prob_y_given_h(rbm::Union{RBMClassifier, GRBMClassifier}, h::Vector{<:Number})
    # Compute the log of the numerator
    log_numerator = rbm.c .+ rbm.U * h

    # Compute the log of the denominator using logsumexp for numerical stability
    log_denominator = logsumexp(rbm.c .+ rbm.U * h)

    # Return the probability p(y|h) in a numerically stable way
    return exp.(log_numerator .- log_denominator)
end

function conditional_prob_y_given_h(
    rbm::Union{RBMClassifier, GRBMClassifier},
    h::Vector{<:Number},
    W_fast::Matrix{Float64},
    c_fast::Vector{Float64},
)
    log_numerator = (rbm.c .+ c_fast) + (rbm.U .+ W_fast) * h

    log_denominator = logsumexp((rbm.c .+ c_fast) .+ (rbm.U .+ W_fast) * h)

    return exp.(log_numerator .- log_denominator)
end

function reconstruct(rbm::AbstractRBM, v::Vector{<:Number})
    h = conditional_prob_h(rbm, v)
    v_reconstructed = conditional_prob_v(rbm, h)
    return v_reconstructed
end

function classify(rbm::RBMClassifiers, v::Vector{<:Number})
    y = conditional_prob_y_given_v(rbm, v)
    return [l == maximum(y) ? 1 : 0 for l in y]
end

function copy_rbm(rbm::AbstractRBM)
    return RBM(copy(rbm.W), copy(rbm.a), copy(rbm.b), rbm.n_visible, rbm.n_hidden)
end

function copy_rbm!(rbm_src::AbstractRBM, rbm_target::AbstractRBM)
    rbm_target.W .= rbm_src.W
    rbm_target.a .= rbm_src.a
    rbm_target.b .= rbm_src.b
    rbm_target.n_visible = rbm_src.n_visible
    rbm_target.n_hidden = rbm_src.n_hidden
    return
end

function copy_rbm(rbm::RBMClassifier)
    return RBMClassifier(copy(rbm.W), copy(rbm.U), copy(rbm.a), copy(rbm.b), copy(rbm.c), rbm.n_visible, rbm.n_hidden, rbm.n_classifiers)
end

function copy_rbm!(rbm_src::RBMClassifiers, rbm_target::RBMClassifier)
    rbm_target.W .= rbm_src.W
    rbm_target.U .= rbm_src.U
    rbm_target.a .= rbm_src.a
    rbm_target.b .= rbm_src.b
    rbm_target.c .= rbm_src.c
    return
end
