_sigmoid(x::Float64) = 1 / (1 + exp(-x))

_get_permutations(n::Int) = [_int_to_bin_array(i, n) for i = 0:2^n-1]

function _int_to_bin_array(x::Int, n::Int)
    return [parse(Int, i) for i in string(x, base = 2, pad = n)]
end

num_visible_nodes(rbm::AbstractRBM) = rbm.n_visible
num_hidden_nodes(rbm::AbstractRBM) = rbm.n_hidden

function _set_mini_batches(training_set_length::Int, batch_size::Int)
    n_batches = round(Int, training_set_length / batch_size)
    last_batch_size = training_set_length % batch_size
    if last_batch_size > 0
        @warn "The last batch size is not equal to the batch size. Will dismiss $(last_batch_size) samples."
    end
    mini_batches = Vector{UnitRange{Int64}}(undef, n_batches)
    for i = 1:n_batches
        mini_batches[i] = (i-1)*batch_size+1:i*batch_size
    end
    return mini_batches
end

function _rbm_qubo(num_visible::Int, num_hidden::Int)
    Q = zeros(num_visible + num_hidden, num_visible + num_hidden)
    Q[1:num_visible, num_visible+1:num_visible+num_hidden] = randn(num_visible, num_hidden)
    Q[num_visible+1:num_visible+num_hidden, 1:num_visible] = Q[1:num_visible, num_visible+1:num_visible+num_hidden]'
    return Q
end
