mutable struct FantasyData
    v::Vector{Float64}
    h::Vector{Float64}
end

function update_fantasy_data!(rbm::BernoulliRBM, fantasy_data::Vector{FantasyData})
    for i = 1:length(fantasy_data)
        fantasy_data[i].h = gibbs_sample_hidden(rbm, fantasy_data[i].v)
        fantasy_data[i].v = gibbs_sample_visible(rbm, fantasy_data[i].h)
    end
end

function init_fantasy_data(rbm::BernoulliRBM, batch_size::Int)
    fantasy_data = Vector{FantasyData}(undef, batch_size)
    for i = 1:batch_size
        fantasy_data[i] =
            FantasyData(rand(num_visible_nodes(rbm)), rand(num_hidden_nodes(rbm)))
    end
    return fantasy_data
end

function get_avg_fantasy_outer(fantasy_data::Vector{FantasyData})
    return mean([fantasy.v * fantasy.h' for fantasy in fantasy_data])
end

function get_avg_fantasy_data(fantasy_data::Vector{FantasyData})
    return mean([fantasy.v for fantasy in fantasy_data]),
    mean([fantasy.h for fantasy in fantasy_data])
end

# PCD-K mini-batch algorithm
# Tieleman (2008) "Training restricted Boltzmann machines using approximations to the likelihood gradient"
function persistent_contrastive_divergence(
    rbm::BernoulliRBM,
    x,
    mini_batches::Vector{UnitRange{Int}},
    fantasy_data::Vector{FantasyData};
    learning_rate::Float64 = 0.1,
)
    loss = Vector{Float64}(undef, num_visible_nodes(rbm))
    total_t_sample = 0.0
    total_t_gibbs = 0.0
    total_t_update = 0.0
    for mini_batch in mini_batches

        # update fantasy FantasyData
        t_gibbs = time()
        # v_estimate, h_estimate = get_avg_fantasy_data(fantasy_data) # v~, h~
        total_t_gibbs += time() - t_gibbs

        i = 1
        for sample in x[mini_batch]

            t_sample = time()
            v_test = sample # training visible
            h_test = conditional_prob_h(rbm, v_test) # hidden from training visible
            total_t_sample += time() - t_sample



            # Update hyperparameter
            t_update = time()
            rbm.W .+=
                (learning_rate / length(mini_batch)) .*
                (v_test * h_test' .- fantasy_data[i].v * fantasy_data[i].h')
            rbm.a .+= (learning_rate / length(mini_batch)) .* (v_test .- fantasy_data[i].v)
            rbm.b .+= (learning_rate / length(mini_batch)) .* (h_test .- fantasy_data[i].h)
            total_t_update += time() - t_update

            loss += abs.(v_test - fantasy_data[i].v)
            i += 1
        end

        # Update fantasy data
        t_gibbs = time()
        update_fantasy_data!(rbm, fantasy_data)
        total_t_gibbs += time() - t_gibbs
    end
    return sum(loss) / length(x), total_t_sample, total_t_gibbs, total_t_update
end
