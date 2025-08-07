mutable struct FantasyData
    v::Vector{Float64}
    h::Vector{Float64}
end

mutable struct FantasyDataClassifier
    v::Vector{Float64}
    h::Vector{Float64}
    y::Vector{Float64}
end

function _update_fantasy_data!(rbm::AbstractRBM, fantasy_data::Vector{FantasyData}, steps::Int)
    for _ in 1:steps
        for i in 1:length(fantasy_data)
            fantasy_data[i].h = gibbs_sample_hidden(rbm, fantasy_data[i].v)
            fantasy_data[i].v = gibbs_sample_visible(rbm, fantasy_data[i].h)
        end
    end
end

function _update_fantasy_data!(rbm::Union{RBMClassifier, GRBMClassifier}, fantasy_data::Vector{FantasyDataClassifier}, steps::Int)
    for _ in 1:steps
        for i in 1:length(fantasy_data)
            fantasy_data[i].h = gibbs_sample_hidden(rbm, fantasy_data[i].v, fantasy_data[i].y)
            fantasy_data[i].v = gibbs_sample_visible(rbm, fantasy_data[i].h)
            fantasy_data[i].y = gibbs_sample_label(rbm, fantasy_data[i].h)
        end
    end
end

function _update_fantasy_data!(
    rbm::AbstractRBM,
    fantasy_data::Vector{FantasyData},
    W_fast::Matrix{Float64},
    a_fast::Vector{Float64},
    b_fast::Vector{Float64},
    steps::Int,
)
    for _ in 1:steps
        for i in 1:length(fantasy_data)
            fantasy_data[i].h = gibbs_sample_hidden(rbm, fantasy_data[i].v, W_fast, b_fast)
            fantasy_data[i].v = gibbs_sample_visible(rbm, fantasy_data[i].h, W_fast, a_fast)
        end
    end
end

function _update_fantasy_data!(
    rbm::Union{RBMClassifier, GRBMClassifier},
    fantasy_data::Vector{FantasyDataClassifier},
    W_fast::Matrix{Float64},
    U_fast::Matrix{Float64},
    a_fast::Vector{Float64},
    b_fast::Vector{Float64},
    c_fast::Vector{Float64},
    steps::Int,
)
    for _ in 1:steps
        for i in 1:length(fantasy_data)
            fantasy_data[i].h = gibbs_sample_hidden(rbm, fantasy_data[i].v, fantasy_data[i].y, W_fast, U_fast, b_fast)
            fantasy_data[i].v = gibbs_sample_visible(rbm, fantasy_data[i].h, W_fast, a_fast)
            fantasy_data[i].y = gibbs_sample_label(rbm, fantasy_data[i].h, U_fast, c_fast)
        end
    end
end

function _init_fantasy_data(rbm::AbstractRBM, batch_size::Int)
    fantasy_data = Vector{FantasyData}(undef, batch_size)
    for i in 1:batch_size
        fantasy_data[i] =
            FantasyData(rand(num_visible_nodes(rbm)), rand(num_hidden_nodes(rbm)))
    end
    return fantasy_data
end

function _init_fantasy_data(rbm::Union{RBMClassifier, GRBMClassifier}, batch_size::Int)
    fantasy_data = Vector{FantasyDataClassifier}(undef, batch_size)
    for i in 1:batch_size
        fantasy_data[i] =
            FantasyDataClassifier(
                rand(num_visible_nodes(rbm)),
                rand(num_hidden_nodes(rbm)),
                rand(num_label_nodes(rbm)),
            )
    end
    return fantasy_data
end

