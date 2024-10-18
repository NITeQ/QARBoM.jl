
<picture>
  <source media="(prefers-color-scheme: light)" srcset="./assets/logo-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-dark.svg">
  <img alt="QARBoM.jl logo">
</picture>

---

Quantum-Assisted Restricted Boltzmann Machine Training Framework

This package provides a framework for training Restricted Boltzmann Machines (RBMs) via classical algorithms (Contrastive Divergence and Persistent Contrastive Divergence) and quantum-assisted methods, which involve the use of Quantum Samplimg.

Using the [QUBO.jl](https://github.com/JuliaQUBO/QUBO.jl) package, this package allows training RBMs using different quantum computers and simulators and converting continuous visible nodes to binary visible nodes seamlessly. 

## Installation

```julia
] add www.github.com/NITeQ/QARBoM.jl
```

## Getting started

```julia
using QARBoM, DWave

# Get a dataset. Should be a Vector{Vector{Float64}} where each inner vector is a sample.
train_data = MY_DATA_TRAIN
test_data = MY_DATA_TEST


# Create a new RBM with:
# - 10 visible nodes (the size of each sample)
# - 5 hidden nodes (the number of hidden nodes that you can choose)

rbm = RBM(10, 5)

# Define a metric to evaluate the RBM after each epoch

function evaluate_classifier(rbm, metrics, epoch)
    for sample_i in eachindex(test_data)
        v = test_data[sample_i]
        h = sample_hidden(rbm, v)
        v_reconstructed = sample_visible(rbm, h)
        metrics["reconstruction_error"] += sum((v - v_reconstructed).^2) / length(test_data)
    end
end


# Train the RBM using Persistent Contrastive Divergence
N_EPCHOS = 100
BATCH_SIZE = 10

train_metrics = Dict(
    "reconstruction_error" => zeros(N_EPOCHS)
)

QARBoM.train!(
    rbm, 
    x_train, 
    PCD; # method
    n_epochs = N_EPOCHS, 
    batch_size = BATCH_SIZE, 
    learning_rate = [0.005/(j^0.5) for j in 1:N_EPOCHS], # has to be a vector of the same length as n_epochs
    evaluation_function = evaluate_classifier,
    metrics = train_metrics
)

# Train RBM using Quantum Sampling

# Define a setup for you quantum sampler

MOI = QARBoM.ToQUBO.MOI
MOI.supports(::DWave.Neal.Optimizer, ::MOI.ObjectiveSense) = true
function setup_dwave(model, sampler)
  MOI.set(model, MOI.RawOptimizerAttribute("num_reads"), 25)
  MOI.set(model, MOI.RawOptimizerAttribute("num_sweeps"), 100)
end

train_metrics_quantum = Dict(
    "reconstruction_error" => zeros(N_EPOCHS)
)

QARBoM.train!(
    rbm, 
    x_train, 
    QSampling; # method
    n_epochs = N_EPOCHS, 
    batch_size = BATCH_SIZE, 
    learning_rate = [0.005/(j^0.5) for j in 1:N_EPOCHS], # has to be a vector of the same length as n_epochs
    evaluation_function = evaluate_classifier,
    metrics = train_metrics_quantum,
    model_setup=setup_dwave,
    sampler=DWave.Neal.Optimizer,
)
