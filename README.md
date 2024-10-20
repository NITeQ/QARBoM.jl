<picture style="display: grid; place-items: center;">
  <source media="(prefers-color-scheme: light)" srcset="./assets/logo-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-dark.svg">

  <img height="200" alt="QARBoM.jl logo">
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

### Defining your dataset and RBM

```julia
using QARBoM

# Get a dataset. Should be a Vector{Vector{Float64}} where each inner vector is a sample.
train_data = MY_DATA_TRAIN
test_data = MY_DATA_TEST


# Create a new RBM with:
# - 10 visible nodes (the size of each sample)
# - 5 hidden nodes (the number of hidden nodes that you can choose)

rbm = RBM(10, 5)
````

### Training RBM using Contrastive Divergence
```julia
# Train the RBM using Persistent Contrastive Divergence
N_EPOCHS = 100
BATCH_SIZE = 10

QARBoM.train!(
    rbm, 
    x_train,
    CD; 
    n_epochs = N_EPOCHS,  
    cd_steps = 3, # number of gibbs sampling steps
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    metrics = [MeanSquaredError], # the metrics you want to track
    x_test_dataset = x_test,
    early_stopping = true,
    file_path = "my_cd_metrics.csv",
)

```


### Training RBM using Persistent Contrastive Divergence
```julia
# Train the RBM using Persistent Contrastive Divergence
N_EPOCHS = 100
BATCH_SIZE = 10

QARBoM.train!(
    rbm, 
    x_train,
    PCD; 
    n_epochs = N_EPOCHS, 
    batch_size = BATCH_SIZE, 
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    metrics = [MeanSquaredError], # the metrics you want to track
    x_test_dataset = x_test,
    early_stopping = true,
    file_path = "my_pcd_metrics.csv",
)

```


### Train RBM using Quantum Sampling

```julia
# Define a setup for you quantum sampler
MOI = QARBoM.ToQUBO.MOI
MOI.supports(::DWave.Neal.Optimizer, ::MOI.ObjectiveSense) = true

function setup_dwave(model, sampler)
  MOI.set(model, MOI.RawOptimizerAttribute("num_reads"), 25)
  MOI.set(model, MOI.RawOptimizerAttribute("num_sweeps"), 100)
end


QARBoM.train!(
    rbm, 
    x_train,
    QSampling; 
    n_epochs = N_EPOCHS, 
    batch_size = 5, 
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    label_learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    metrics = [Accuracy],
    x_test_dataset = x_test,
    early_stopping = true,
    file_path = "qubo_train.csv",
    model_setup=setup_dwave,
    sampler=DWave.Neal.Optimizer,
)
```
