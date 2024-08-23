function cross_entropy_loss(v::Vector{Float64}, v_reconstructed::Vector{Float64})
    return -sum(v .* log.(v_reconstructed) .+ (1 .- v) .* log.(1 .- v_reconstructed))/length(v)
end

function evaluate_classifier(y::Vector{Float64}, y_pred::Vector{Float64}, threshold::Float64=0.75)
    # println("Expected y: ", y)
    # println("Predicted y: ", y_pred)
    rounded_y_pred = [y_pred[i] > threshold ? 1 : 0 for i in 1:length(y_pred)]
    # println("Evaluated correctly: ", all(i -> i == 1, y .== rounded_y_pred) ? 1 : 0)
    return all(i -> i == 1, y .== rounded_y_pred) ? 1 : 0
end