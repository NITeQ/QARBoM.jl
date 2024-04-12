module QARBoM

abstract type AbstractRBM end

include("utils.jl")
include("abstract_rbm.jl")
include("bernoulli_rbm.jl")
include("contrastive_divergence.jl")
include("training.jl")

end # module QARBoM
