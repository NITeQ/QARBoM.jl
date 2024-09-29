abstract type AdamOpt end

export AdamOpt

mutable struct Adam
    lr::Float64
    β_1::Float64
    β_2::Float64
    ϵ::Float64
    mini_batch::Int
    iteration::Int
    m_v::Array{Float64} # running mean for the visible layer
    m_h::Array{Float64} # running mean for the hidden layer
    m_w::Array{Float64} # running mean for the weights
    v_v::Array{Float64} # running variance for the visible layer
    v_h::Array{Float64} # running variance for the hidden layer
    v_w::Array{Float64} # running variance for the weights
end

mutable struct Adagrad
    lr::Float64          # Learning rate
    ϵ::Float64           # Small constant to avoid division by zero
    mini_batch::Int      # Mini-batch size
    G_v::Vector{Float64} # Accumulated squared gradients for bottom bias
    G_h::Vector{Float64} # Accumulated squared gradients for top bias
    G_w::Matrix{Float64} # Accumulated squared gradients for weights
end

function Adam(;
    num_visible::Int,
    num_hidden::Int,
    lr = 0.0001,
    β_1 = 0.9,
    β_2 = 0.999,
    ϵ = 1e-8,
)
    return Adam(
        lr,
        β_1,
        β_2,
        ϵ,
        1, # size of mini-batch
        1, # iteration
        zeros(num_visible),
        zeros(num_hidden),
        zeros(num_visible, num_hidden),
        zeros(num_visible),
        zeros(num_hidden),
        zeros(num_visible, num_hidden),
    )
end

function Adagrad(;
    num_visible::Int,
    num_hidden::Int,
    lr = 0.01,
    ϵ = 1e-8,
    mini_batch = 1,
)
    return Adagrad(
        lr,
        ϵ,
        mini_batch,
        zeros(num_visible),
        zeros(num_hidden),
        zeros(num_visible, num_hidden),
    )
end

_set_mini_batch_length!(adam::Adam, mini_batch::Int) = (adam.mini_batch = mini_batch)
_set_mini_batch_length!(adagrad::Adagrad, mini_batch::Int) = (adagrad.mini_batch = mini_batch)
