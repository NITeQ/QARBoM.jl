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

    return QUBORBM(model, n_visible, n_hidden)
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
    v_sampled = zeros(Int, num_visible_nodes(rbm))
    h_sampled = zeros(Int, num_hidden_nodes(rbm))
    for i in 1:result_count(model)
        v_sampled .+= value.(model[:vis], result = i)
        h_sampled .+= value.(model[:hid], result = i)
    end
    return v_sampled./result_count(model), h_sampled./result_count(model)
end

