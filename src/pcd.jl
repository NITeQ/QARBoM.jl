mutable struct FantasyData
    v::Vector{Float64}
    h::Vector{Float64}
end

function _update_fantasy_data!(rbm::RBM, fantasy_data::Vector{FantasyData})
    for i in 1:length(fantasy_data)
        fantasy_data[i].h = gibbs_sample_hidden(rbm, fantasy_data[i].v)
        fantasy_data[i].v = gibbs_sample_visible(rbm, fantasy_data[i].h)
    end
end

function _update_fantasy_data!(
    rbm::RBM,
    fantasy_data::Vector{FantasyData},
    W_fast::Matrix{Float64},
    a_fast::Vector{Float64},
    b_fast::Vector{Float64},
)
    for i in 1:length(fantasy_data)
        fantasy_data[i].h = gibbs_sample_hidden(rbm, fantasy_data[i].v, W_fast, b_fast)
        fantasy_data[i].v = gibbs_sample_visible(rbm, fantasy_data[i].h, W_fast, a_fast)
    end
end

function _init_fantasy_data(rbm::RBM, batch_size::Int)
    fantasy_data = Vector{FantasyData}(undef, batch_size)
    for i in 1:batch_size
        fantasy_data[i] =
            FantasyData(rand(num_visible_nodes(rbm)), rand(num_hidden_nodes(rbm)))
    end
    return fantasy_data
end

# PCD-K mini-batch algorithm
# Tieleman (2008) "Training restricted Boltzmann machines using approximations to the likelihood gradient"
function persistent_contrastive_divergence!(
    rbm::RBM,
    x,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyData};
    learning_rate::Float64 = 0.1,
)
    total_t_sample, total_t_gibbs, total_t_update = 0.0, 0.0, 0.0
    loss = 0.0
    for mini_batch in mini_batches
        i = 1
        for sample in x[mini_batch]
            t_sample = time()
            v_data = sample # training visible
            h_data = conditional_prob_h(rbm, v_data) # hidden from training visible
            total_t_sample += time() - t_sample

            # Update hyperparameter
            t_update = time()
            update_rbm!(
                rbm,
                v_data,
                h_data,
                fantasy_data[i].v,
                fantasy_data[i].h,
                (learning_rate / length(mini_batch)),
            )
            total_t_update += time() - t_update

            # loss by Mean Squared Error
            reconstructed = reconstruct(rbm, sample)
            loss += sum((sample .- reconstructed) .^ 2)
            i += 1
        end

        # Update fantasy data
        t_gibbs = time()
        _update_fantasy_data!(rbm, fantasy_data)
        total_t_gibbs += time() - t_gibbs
    end
    return loss / length(x), total_t_sample, total_t_gibbs, total_t_update
end

# Fast PCD-K mini-batch algorithm
# Tieleman and Hinton (2009) "Using fast weights to improve persistent contrastive divergence"
function fast_persistent_contrastive_divergence!(
    rbm::RBM,
    x,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyData};
    learning_rate::Float64 = 0.1,
    fast_learning_rate::Float64 = 0.1,
)
    total_t_sample = 0.0
    total_t_gibbs = 0.0
    total_t_update = 0.0
    loss = 0.0
    W_fast = zeros(num_visible_nodes(rbm), num_hidden_nodes(rbm))
    a_fast = zeros(num_visible_nodes(rbm))
    b_fast = zeros(num_hidden_nodes(rbm))
    for mini_batch in mini_batches

        # update fantasy FantasyData
        t_gibbs = time()
        total_t_gibbs += time() - t_gibbs

        i = 1
        for sample in x[mini_batch]
            t_sample = time()
            v_data = sample # training visible
            h_data = conditional_prob_h(rbm, v_data) # hidden from training visible
            total_t_sample += time() - t_sample

            # Update hyperparameter
            t_update = time()
            update_rbm!(
                rbm,
                v_data,
                h_data,
                fantasy_data[i].v,
                fantasy_data[i].h,
                (learning_rate / length(mini_batch)),
            )
            total_t_update += time() - t_update

            W_fast =
                W_fast .* (19 / 20) +
                (fast_learning_rate / length(mini_batch)) .*
                (v_data * h_data' .- fantasy_data[i].v * fantasy_data[i].h')
            a_fast =
                a_fast .* (19 / 20) +
                (fast_learning_rate / length(mini_batch)) .* (v_data .- fantasy_data[i].v)
            b_fast =
                b_fast .* (19 / 20) +
                (fast_learning_rate / length(mini_batch)) .* (h_data .- fantasy_data[i].h)

            # loss by Mean Squared Error
            reconstructed = reconstruct(rbm, sample)
            loss += sum((sample .- reconstructed) .^ 2)
            i += 1
        end

        # Update fantasy data
        t_gibbs = time()
        _update_fantasy_data!(rbm, fantasy_data, W_fast, a_fast, b_fast)
        total_t_gibbs += time() - t_gibbs
    end
    return loss / length(x), total_t_sample, total_t_gibbs, total_t_update
end
