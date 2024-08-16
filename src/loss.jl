function cross_entropy_loss(v::Vector{Float64}, v_reconstructed::Vector{Float64})
    return -sum(v .* log.(v_reconstructed) .+ (1 .- v) .* log.(1 .- v_reconstructed))/length(v)
end