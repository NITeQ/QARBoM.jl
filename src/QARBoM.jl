module QARBoM

using QUBO
using Printf
using Statistics
using JuMP
using LogExpFunctions
using CSV
using DataFrames
using Distributions

abstract type TrainingMethod end
abstract type QSampling <: TrainingMethod end
abstract type PCD <: TrainingMethod end
abstract type CD <: TrainingMethod end
abstract type FastPCD <: TrainingMethod end

abstract type AbstractDBN end
abstract type AbstractRBM end
abstract type DBNLayer end

export RBM, RBMClassifier, GRBM, GRBMClassifier
export QSampling, PCD, CD, FastPCD

include("utils.jl")
include("rbm.jl")
include("evaluation.jl")
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
