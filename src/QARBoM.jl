module QARBoM

using Printf

abstract type AbstractRBM end
abstract type AbstractMethod end
struct CD <: AbstractMethod end # Contrastive Divergence
struct PCD <: AbstractMethod end # Persistent Contrastive Divergence

include("utils.jl")
include("bernoulli_rbm.jl")
include("contrastive_divergence.jl")
include("training.jl")

end # module QARBoM
