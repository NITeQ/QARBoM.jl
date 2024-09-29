_sigmoid(x) = 1 / (1 + exp(-x))
_sigmoid_derivative(x) = x .* (1.0 .- x)
_relu(x::Float64) = max(0, x)

function _softmax(x::Vector{T}) where {T <: Union{Int, Float64}}
    exp_x = exp.(x)
    return exp_x ./ sum(exp_x)
end

num_visible_nodes(rbm::AbstractRBM) = rbm.n_visible
num_hidden_nodes(rbm::AbstractRBM) = rbm.n_hidden
num_label_nodes(rbm::AbstractRBM) = rbm.n_classifiers

function _set_mini_batches(training_set_length::Int, batch_size::Int)
    n_batches = round(Int, training_set_length / batch_size)
    last_batch_size = training_set_length % batch_size
    if last_batch_size > 0
        @warn "The last batch size is not equal to the other batches. Will dismiss $(last_batch_size) samples."
        training_set_length -= last_batch_size
    end
    mini_batches = Vector{UnitRange{Int64}}(undef, n_batches)
    for i in 1:n_batches
        mini_batches[i] = (i-1)*batch_size+1:i*batch_size
    end
    return mini_batches
end

function _log_epoch(epoch::Int, t_sample::Float64, t_gibbs::Float64, t_update::Float64, total_t::Float64)
    println("|------------------------------------------------------------------------------|")
    println("| Epoch | Time (Sample) | Time (Gibbs) | Time (Update) | Total     |")
    println("|------------------------------------------------------------------------------|")
    println(
        @sprintf(
            "| %5d | %13.4f | %12.4f | %13.4f | %9.4f |",
            epoch,
            t_sample,
            t_gibbs,
            t_update,
            total_t,
        )
    )
    return println("|------------------------------------------------------------------------------|")
end

function _log_epoch_quantum(epoch::Int, t_sample::Float64, t_qs::Float64, t_update::Float64, total_t::Float64)
    println("|------------------------------------------------------------------------------|")
    println("| Epoch | Time (Sample) | Time (Qsamp) | Time (Update) | Total     |")
    println("|------------------------------------------------------------------------------|")
    println(
        @sprintf(
            "| %5d | %13.4f | %12.4f | %13.4f | %9.4f |",
            epoch,
            t_sample,
            t_qs,
            t_update,
            total_t,
        )
    )
    return println("|------------------------------------------------------------------------------|")
end

function _log_metrics(metrics::Dict{String, Vector{Float64}}, epoch::Int)
    for metric_name in keys(metrics)
        println("$metric_name: $(metrics[metric_name][epoch])")
    end
end

function _log_finish(n_epochs::Int, total_t_sample::Float64, total_t_gibbs::Float64, total_t_update::Float64)
    println("QARBoM.jl finished training after $n_epochs epochs.")
    println("Total time spent sampling: $total_t_sample")
    println("Total time spent in Gibbs sampling: $total_t_gibbs")
    println("Total time spent updating parameters: $total_t_update")
    return println("Total time spent training: $(total_t_sample + total_t_gibbs + total_t_update)")
end

function _log_finish_quantum(n_epochs::Int, total_t_sample::Float64, total_t_qs::Float64, total_t_update::Float64)
    println("QARBoM.jl finished training after $n_epochs epochs.")
    println("Total time spent sampling: $total_t_sample")
    println("Total time spent in Quantum sampling: $total_t_qs")
    println("Total time spent updating parameters: $total_t_update")
    return println("Total time spent training: $(total_t_sample + total_t_qs + total_t_update)")
end
