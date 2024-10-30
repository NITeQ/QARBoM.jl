abstract type EvaluationMethod end

abstract type Accuracy <: EvaluationMethod end
abstract type CrossEntropy <: EvaluationMethod end
abstract type MeanSquaredError <: EvaluationMethod end
abstract type FalsePositive <: EvaluationMethod end
abstract type TruePositive <: EvaluationMethod end
abstract type FalseNegative <: EvaluationMethod end
abstract type TrueNegative <: EvaluationMethod end
abstract type Precision <: EvaluationMethod end
abstract type Recall <: EvaluationMethod end

export Accuracy, MeanSquaredError, CrossEntropy, FalsePositive, TruePositive, FalseNegative, TrueNegative, Precision, Recall

function _evaluate(::Type{Accuracy}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    sample = kwargs[:y_sample]
    predicted = kwargs[:y_pred]
    tp = all(i -> i == 1, round.(Int, sample) .== round.(Int, predicted)) ? 1 : 0
    return metrics_dict["accuracy"][epoch] += tp / dataset_size
end

function _evaluate(::Type{FalsePositive}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    sample = kwargs[:y_sample]
    predicted = kwargs[:y_pred]
    tp = 0
    if sample[1] == 0 && predicted[1] == 1
        tp = 1
    end
    return metrics_dict["false_positive"][epoch] += tp 
end

function _evaluate(::Type{TruePositive}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    sample = kwargs[:y_sample]
    predicted = kwargs[:y_pred]
    tp = 0
    if sample[1] == 1 && predicted[1] == 1
        tp = 1
    end
    return metrics_dict["true_positive"][epoch] += tp
end

function _evaluate(::Type{FalseNegative}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    sample = kwargs[:y_sample]
    predicted = kwargs[:y_pred]
    tp = 0
    if sample[1] == 1 && predicted[1] == 0
        tp = 1
    end
    return metrics_dict["false_negative"][epoch] += tp
end

function _evaluate(::Type{Precision}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    return metrics_dict["precision"][epoch]  = metrics_dict["true_positive"][epoch]/(metrics_dict["true_positive"][epoch] + metrics_dict["false_positive"][epoch])
end


function _evaluate(::Type{Recall}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    pre = 
    return metrics_dict["recall"][epoch]  = metrics_dict["true_positive"][epoch]/(metrics_dict["true_positive"][epoch] + metrics_dict["false_negative"][epoch])
end

function _evaluate(::Type{TrueNegative}, metrics_dict::Dict{String, Vector{Float64}}, epoch::Int, dataset_size::Int; kwargs...)
    sample = kwargs[:y_sample]
    predicted = kwargs[:y_pred]
    tp = 0
    if sample[1] == 0 && predicted[1] == 0
        tp = 1
    end
    return metrics_dict["true_negative"][epoch] += tp
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
    rbm::RBMClassifier,
    metrics::Vector{<:DataType},
    x_dataset::Vector{Vector{T}},
    y_dataset::Vector{Vector{T}},
    metrics_dict::Dict{String, Vector{Float64}},
    epoch::Int,
) where {T <: Union{Float64, Int}}
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
    dataset_size = length(x_dataset)
    for sample_i in eachindex(x_dataset)
        vis = x_dataset[sample_i]
        vis_pred = QARBoM.reconstruct(rbm, vis)

        for metric in metrics
            _evaluate(metric, metrics_dict, epoch, dataset_size; x_sample = vis, x_pred = vis_pred)
        end
    end
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
        elseif metric == FalseNegative
            metrics_dict["false_negative"] = Float64[]
        elseif metric == FalsePositive
            metrics_dict["false_positive"] = Float64[]
        elseif metric == TrueNegative
            metrics_dict["true_negative"] = Float64[]
        elseif metric == TruePositive
            metrics_dict["true_positive"] = Float64[]
        elseif metric == Precision
            metrics_dict["precision"] = Float64[]
        elseif metric == Recall
            metrics_dict["recall"] = Float64[]
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
        return true
    elseif metrics_dict["accuracy"][epoch] < maximum(metrics_dict["accuracy"])
        return true
    end
    return false
end
