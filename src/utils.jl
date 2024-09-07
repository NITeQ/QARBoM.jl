_sigmoid(x::Float64) = 1 / (1 + exp(-x))

_relu(x::Float64) = max(0, x)

num_visible_nodes(rbm::AbstractRBM) = rbm.n_visible
num_hidden_nodes(rbm::AbstractRBM) = rbm.n_hidden
num_label_nodes(rbm::AbstractRBM) = rbm.n_classifiers

function _set_mini_batches(training_set_length::Int, batch_size::Int)
    n_batches = round(Int, training_set_length / batch_size)
    last_batch_size = training_set_length % batch_size
    if last_batch_size > 0
        @warn "The last batch size is not equal to the batch size. Will dismiss $(last_batch_size) samples."
    end
    mini_batches = Vector{UnitRange{Int64}}(undef, n_batches)
    for i in 1:n_batches
        mini_batches[i] = (i-1)*batch_size+1:i*batch_size
    end
    return mini_batches
end