_sigmoid(x::Float64) = 1 / (1 + exp(-x))

_get_permutations(n::Int) = [_int_to_bin_array(i, n) for i = 0:2^n-1]

function _int_to_bin_array(x::Int, n::Int)
    return [parse(Int, i) for i in string(x, base = 2, pad = n)]
end

num_visible_nodes(rbm::AbstractRBM) = rbm.n_visible
num_hidden_nodes(rbm::AbstractRBM) = rbm.n_hidden
