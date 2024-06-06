@enum RBMStatus UNTRAINED TRAINED 

mutable struct QUBORBM <: AbstractRBM
    model
    n_visible::Int # number of visible units
    n_hidden::Int # number of hidden units
    status::RBMStatus
end

function QUBORBM(n_visible::Int, n_hidden::Int, sampler)
    Q = _rbm_qubo(n_visible, n_hidden)

    model = Model(sampler)
    @variable(model, vis[1:n_visible], Bin)
    @variable(model, hid[1:n_hidden], Bin)
    @objective(model, Min, vcat(vis, hid)' * Q * vcat(vis, hid))

    return QUBORBM(model, n_visible, n_hidden, UNTRAINED)
end

function _hyper_parameters(rbm::QUBORBM)
    if rbm.status == UNTRAINED
        n, L, Q, a, b = QUBOTools.qubo(QUBOTools.Model(JuMP.backend(rbm.model)), :dense)
        return Q[1:num_visible_nodes(rbm), num_visible_nodes(rbm)+1:end], L
    end
    n, L, Q, α, β = QUBOTools.qubo(unsafe_backend(rbm.model), :dense)
    return Q[1:num_visible_nodes(rbm), num_visible_nodes(rbm)+1:end], L
end

function energy(rbm::QUBORBM, v::Vector{Int}, h::Vector{Int})
    W, L = _hyper_parameters(rbm)
    return -L[1:num_visible_nodes(rbm)]' * v - L[num_visible_nodes(rbm)+1:end]' * h - v' * W * h
end

function conditional_prob_h(rbm::QUBORBM, v::Vector{T}) where {T<:Union{Int,Float64}} 
    W, L = _hyper_parameters(rbm)
    b = L[num_visible_nodes(rbm)+1:end]
    return [_sigmoid(b[j] + W[:, j]' * v) for j = 1:num_hidden_nodes(rbm)]
end

function conditional_prob_v(rbm::QUBORBM, h::Vector{T}) where {T<:Union{Int,Float64}}
    W, L = _hyper_parameters(rbm)
    a = L[1:num_visible_nodes(rbm)]
    return [_sigmoid(a[i] + W[i, :]' * h) for i = 1:num_visible_nodes(rbm)]
end

function reconstruct(rbm::QUBORBM, v::Vector{Int})
    W, L = _hyper_parameters(rbm)
    b = L[num_visible_nodes(rbm)+1:end]
    a = L[1:num_visible_nodes(rbm)]
    h = [_sigmoid(b[j] + W[:, j]' * v) for j = 1:num_hidden_nodes(rbm)]
    v_reconstructed = [_sigmoid(a[i] + W[i, :]' * h) for i = 1:num_visible_nodes(rbm)]
    return v_reconstructed
end

function qubo_sample(rbm::QUBORBM, n_samples::Int)
    optimize!(rbm.model)
    rbm.status = TRAINED
    v_sampled = zeros(Int, num_visible_nodes(rbm))
    h_sampled = zeros(Int, num_hidden_nodes(rbm))
    total_samples = result_count(rbm.model)
    for i in 1:total_samples
        v_sampled .+= value.(rbm.model[:vis], result = i)
        h_sampled .+= value.(rbm.model[:hid], result = i)
    end
    return v_sampled./total_samples, h_sampled./total_samples
end

function quantum_sampling(
    rbm::QUBORBM,
    x::Vector{Vector{Int}},
    n_samples::Int;
    learning_rate::Float64 = 0.1,
)
    total_t_sample = 0.0
    total_t_qs = 0.0
    total_t_update = 0.0
    loss = 0.0
    for sample in x

        t_sample = time()
        v_test = sample # training visible
        h_test = conditional_prob_h(rbm, v_test) # hidden from training visible
        total_t_sample += time() - t_sample

        t_qs = time()
        v_estimate, h_estimate = qubo_sample(rbm, n_samples) # v~, h~
        total_t_qs += time() - t_qs

        t_update = time()
        update_qubo!(rbm, v_test, h_test, v_estimate, h_estimate, learning_rate)
        total_t_update += time() - t_update

        # loss by Mean Squared Error
        reconstructed = reconstruct(rbm, sample)
        loss += sum((sample .- reconstructed) .^ 2)

    end
    return loss / length(x), total_t_sample, total_t_qs, total_t_update
end

function persistent_qubo_sampling(
    rbm::QUBORBM,
    x::Vector{Vector{Int}},
    mini_batches::Vector{UnitRange{Int}},
    n_samples::Int;
    learning_rate::Float64 = 0.1,
)
    total_t_sample = 0.0
    total_t_qs = 0.0
    total_t_update = 0.0
    loss = 0.0
    for mini_batch in mini_batches
        t_qs = time()
        v_estimate, h_estimate = qubo_sample(rbm, n_samples) # v~, h~
        total_t_qs += time() - t_qs
        for sample in x[mini_batch]
            t_sample = time()
            v_test = sample # training visible
            h_test = conditional_prob_h(rbm, v_test) # hidden from training visible
            total_t_sample += time() - t_sample

            # Update hyperparameter
            t_update = time()
            update_qubo!(rbm, v_test, h_test, v_estimate, h_estimate, (learning_rate / length(mini_batch)))
            total_t_update += time() - t_update

            # loss by Mean Squared Error
            reconstructed = reconstruct(rbm, sample)
            # @show sample
            # @show reconstructed
            loss += sum((sample .- reconstructed) .^ 2)
        end
        # @show _hyper_parameters(rbm)
    end
    return loss / length(x), total_t_sample, total_t_qs, total_t_update
end


