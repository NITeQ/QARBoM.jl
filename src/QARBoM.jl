module QARBoM

using Printf
using Statistics
using JuMP
using ToQUBO
using LogExpFunctions

abstract type TrainingMethod end
abstract type QSampling <: TrainingMethod end
abstract type PCD <: TrainingMethod end
abstract type CD <: TrainingMethod end

abstract type AbstractDBN end
abstract type AbstractRBM end
abstract type DBNLayer end

export RBM, GRBM, RBMClassifier
export QSampling, PCD, CD

include("utils.jl")
include("adam.jl")
include("rbm.jl")
include("gibbs.jl")
include("qubo.jl")
include("fantasy_data.jl")
include("dbn.jl")
include("training/train_cd.jl")
include("training/train_pcd.jl")
include("training/train_fast_pcd.jl")
include("training/train_quantum_pcd.jl")
include("training/pretrain_dbn.jl")
include("training/fine_tune_dbn.jl")

end # module QARBoM
