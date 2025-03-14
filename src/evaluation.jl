abstract type EvaluationMethod end

abstract type Accuracy <: EvaluationMethod end
abstract type CrossEntropy <: EvaluationMethod end
abstract type MeanSquaredError <: EvaluationMethod end

export Accuracy, MeanSquaredError, CrossEntropy

function _evaluate(::Type{Accuracy}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    sample = kwargs[:y_sample]
    predicted = kwargs[:y_pred]
    tp = all(i -> i == 1, round.(Int, sample) .== round.(Int, predicted)) ? 1 : 0
    return metrics_dict["accuracy"][epoch] += tp / dataset_size
end

function _evaluate(::Type{MeanSquaredError}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    sample = kwargs[:x_sample]
    predicted = kwargs[:x_pred]
    return metrics_dict["mse"][epoch] += sum((sample .- predicted) .^ 2) / dataset_size
end

function _evaluate(::Type{CrossEntropy}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    sample = kwargs[:y_sample]
    predicted = kwargs[:y_pred]
    return metrics_dict["cross_entropy"][epoch] += sum(sample .* log.(predicted) .+ (1 .- sample) .* log.(1 .- predicted)) / dataset_size
end

function evaluate(
    rbm::Union{RBMClassifier, GRBMClassifier},
    metrics::Vector{<:DataType},
    x_dataset::Vector{Vector{T}},
    y_dataset::Vector{Vector{T}},
    metrics_dict::Dict{String, Vector{Float64}},
    epoch::Int,
) where {T <: Union{Float64, Int}}
    for key in keys(metrics_dict)
        push!(metrics_dict[key], 0.0)
    end
    dataset_size = length(x_dataset)
    for sample_i in eachindex(x_dataset)
        vis = x_dataset[sample_i]
        label = y_dataset[sample_i]
        y_pred = QARBoM.classify(rbm, vis)
        max_val = maximum(y_pred)
        rounded_y_pred = [(y_pred[i] == max_val) ? 1 : 0 for i in eachindex(y_pred)]

        for metric in metrics
            _evaluate(metric, metrics_dict, epoch, dataset_size; y_sample = label, y_pred = rounded_y_pred)
        end
    end
end

function evaluate(
    rbm::AbstractRBM,
    metrics::Vector{<:DataType},
    x_dataset::Vector{Vector{T}},
    metrics_dict::Dict{String, Vector{Float64}},
    epoch::Int,
) where {T <: Union{Float64, Int}}
    for key in keys(metrics_dict)
        push!(metrics_dict[key], 0.0)
    end
    dataset_size = length(x_dataset)
    for sample_i in eachindex(x_dataset)
        vis = x_dataset[sample_i]
        vis_pred = QARBoM.reconstruct(rbm, vis)

        for metric in metrics
            _evaluate(metric, metrics_dict, epoch, dataset_size; x_sample = vis, x_pred = vis_pred)
        end
    end
end

function initial_evaluation(
    rbm::AbstractRBM,
    metrics::Vector{<:DataType},
    x_dataset::Vector{Vector{T}},
) where {T <: Union{Float64, Int}}
    metrics_dict = _initialize_metrics(metrics)
    for key in keys(metrics_dict)
        push!(metrics_dict[key], 0.0)
    end
    dataset_size = length(x_dataset)
    for sample_i in eachindex(x_dataset)
        vis = x_dataset[sample_i]
        vis_pred = QARBoM.reconstruct(rbm, vis)

        for metric in metrics
            _evaluate(metric, metrics_dict, 1, dataset_size; x_sample = vis, x_pred = vis_pred)
        end
    end
    return metrics_dict
end

function initial_evaluation(
    rbm::Union{RBMClassifier, GRBMClassifier},
    metrics::Vector{<:DataType},
    x_dataset::Vector{Vector{T}},
    y_dataset::Vector{Vector{T}},
) where {T <: Union{Float64, Int}}
    metrics_dict = _initialize_metrics(metrics)
    for key in keys(metrics_dict)
        push!(metrics_dict[key], 0.0)
    end
    dataset_size = length(x_dataset)
    for sample_i in eachindex(x_dataset)
        vis = x_dataset[sample_i]
        label = y_dataset[sample_i]
        y_pred = QARBoM.classify(rbm, vis)
        max_val = maximum(y_pred)
        rounded_y_pred = [(y_pred[i] == max_val) ? 1 : 0 for i in eachindex(y_pred)]

        for metric in metrics
            _evaluate(metric, metrics_dict, 1, dataset_size; y_sample = label, y_pred = rounded_y_pred)
        end
    end
    return metrics_dict
end


function _initialize_metrics(metrics::Vector{<:DataType})
    metrics_dict = Dict{String, Vector{Float64}}()
    for metric in metrics
        if metric == Accuracy
            metrics_dict["accuracy"] = Float64[]
        elseif metric == MeanSquaredError
            metrics_dict["mse"] = Float64[]
        elseif metric == CrossEntropy
            metrics_dict["cross_entropy"] = Float64[]
        end
    end
    return metrics_dict
end

function _diverged(metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, ::Type{MeanSquaredError})
    if epoch == 1
        return false
    end
    if metrics_dict["mse"][epoch] > metrics_dict["mse"][epoch-1]
        return true
    elseif metrics_dict["mse"][epoch] > minimum(metrics_dict["mse"])
        return true
    end
    return false
end

function _diverged(metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, ::Type{CrossEntropy})
    if epoch == 1
        return false
    end
    if metrics_dict["cross_entropy"][epoch] > metrics_dict["cross_entropy"][epoch-1]
        return true
    elseif metrics_dict["cross_entropy"][epoch] > minimum(metrics_dict["cross_entropy"])
        return true
    end
    return false
end

function _diverged(metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, ::Type{Accuracy})
    if epoch == 1
        return false
    end
    if metrics_dict["accuracy"][epoch] < metrics_dict["accuracy"][epoch-1]
        println("Accuracy decreased from $(metrics_dict["accuracy"][epoch-1]) to $(metrics_dict["accuracy"][epoch])")
        return true
    elseif metrics_dict["accuracy"][epoch] < maximum(metrics_dict["accuracy"])
        println("Accuracy $(metrics_dict["accuracy"][epoch]) is not the maximum value $(maximum(metrics_dict["accuracy"]))")
        return true
    end
    return false
end

function merge_metrics(metrics_dict_1::Dict{String, Vector{Float64}}, metrics_dict_2::Dict{String, Vector{Float64}})
    for key in keys(metrics_dict_1)
        append!(metrics_dict_1[key], metrics_dict_2[key])
    end
    return metrics_dict_1
end