mutable struct VisibleLayer <: DBNLayer
    W::Matrix{Float64}
    bias::Vector{Float64}
end

mutable struct HiddenLayer <: DBNLayer
    W::Matrix{Float64}
    bias::Vector{Float64}
end

mutable struct TopLayer <: DBNLayer
    bias::Vector{Float64}
end

mutable struct LabelLayer <: DBNLayer
    W::Matrix{Float64}
    bias::Vector{Float64}
end

mutable struct DBN
    layers::Vector{DBNLayer}
    label::Union{LabelLayer, Nothing}
end

function copy_dbn(dbn::DBN)
    layers = Vector{DBNLayer}(undef, length(dbn.layers))
    layers[1] = VisibleLayer(copy(dbn.layers[1].W), copy(dbn.layers[1].bias))
    for i in 2:length(dbn.layers)-1
        layers[i] = HiddenLayer(copy(dbn.layers[i].W), copy(dbn.layers[i].bias))
    end
    layers[end] = TopLayer(copy(dbn.layers[end].bias))
    return DBN(layers, nothing)
end

function initialize_dbn(
    layers_size::Vector{Int};
    weights::Union{Vector{Matrix{Float64}}, Nothing} = nothing,
    biases::Union{Vector{Vector{Float64}}, Nothing} = nothing,
)
    layers = Vector{DBNLayer}()
    for i in 1:length(layers_size)-1
        W = isnothing(weights) ? randn(layers_size[i], layers_size[i+1]) : weights[i]
        bias = isnothing(biases) ? zeros(layers_size[i]) : biases[i]
        i == 1 ? push!(layers, VisibleLayer(W, bias)) : push!(layers, HiddenLayer(W, bias))
    end

    bias = isnothing(biases) ? zeros(layers_size[end]) : biases[end]
    push!(layers, TopLayer(bias))

    return DBN(layers, nothing)
end

function propagate_up(
    top_layer::DBNLayer,
    bottom_layer::DBNLayer,
    x,
)
    return _sigmoid.(bottom_layer.W' * x .+ top_layer.bias)
end

function propagate_up(
    dbn::DBN,
    x,
    from_layer::Int,
    to_layer::Int,
)
    for i in from_layer:(to_layer-1)
        x = _sigmoid.(dbn.layers[i].W' * x .+ dbn.layers[i+1].bias)
    end
    return x
end

function propagate_down(
    bottom_layer::DBNLayer,
    x,
)
    return _sigmoid.(bottom_layer.W * x .+ bottom_layer.bias)
end

function propagate_down(
    dbn::DBN,
    x,
    from_layer::Int,
    to_layer::Int,
)
    for i in from_layer:-1:(to_layer+1)
        x = _sigmoid.(dbn.layers[i-1].W * x .+ dbn.layers[i-1].bias)
    end
    return x
end

function reconstruct(
    dbn::DBN,
    x,
    final_layer::Union{Int, Nothing} = nothing,
)
    final_layer = isnothing(final_layer) ? length(dbn.layers) : final_layer
    rec = propagate_up(dbn, x, 1, final_layer)
    rec = propagate_down(dbn, rec, final_layer, 1)
    return rec
end

function update_layer!(
    top_layer::DBNLayer,
    bottom_layer::DBNLayer,
    v_data::Vector{Float64},
    h_data::Vector{Float64},
    fantasy_data::FantasyData,
    learning_rate::Float64;
    update_bottom_bias::Bool = false,
)
    bottom_layer.W .+= learning_rate .* (v_data * h_data' .- fantasy_data.v * fantasy_data.h')
    update_bottom_bias ? bottom_layer.bias .+= learning_rate .* (v_data .- fantasy_data.v) : nothing
    return top_layer.bias .+= learning_rate .* (h_data .- fantasy_data.h)
end

function classify(
    dbn::DBN,
    x,
)
    x = propagate_up(dbn, x, 1, length(dbn.layers))
    return softmax(dbn.label.W * x .+ dbn.label.bias)
end
