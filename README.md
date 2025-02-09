<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./assets/logo-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-dark.svg">

  <img height="200" alt="QARBoM.jl logo">
</picture>

[build-img]: https://github.com/NITeQ/QARBoM.jl/actions/workflows/ci.yml/badge.svg?branch=main
![Build Status][build-img]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14841099.svg)](https://doi.org/10.5281/zenodo.14841099)


</div>



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

# Get a dataset. Should be a Vector{Vector{<:Number}} where each inner vector is a sample.
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
    train_data,
    CD; 
    n_epochs = N_EPOCHS,  
    cd_steps = 3, # number of gibbs sampling steps
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    metrics = [MeanSquaredError], # the metrics you want to track
    x_test_dataset = test_data,
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
    train_data,
    PCD; 
    n_epochs = N_EPOCHS, 
    batch_size = BATCH_SIZE, 
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    metrics = [MeanSquaredError], # the metrics you want to track
    x_test_dataset = test_data,
    early_stopping = true,
    file_path = "my_pcd_metrics.csv",
)

```


### Train RBM using Quantum Sampling

```julia
using DWave # or any other quantum sampler


# Define a setup for you quantum sampler
MOI = QARBoM.ToQUBO.MOI
MOI.supports(::DWave.Neal.Optimizer, ::MOI.ObjectiveSense) = true

function setup_dwave(model, sampler)
  MOI.set(model, MOI.RawOptimizerAttribute("num_reads"), 25)
  MOI.set(model, MOI.RawOptimizerAttribute("num_sweeps"), 100)
end


QARBoM.train!(
    rbm, 
    train_data,
    QSampling; 
    n_epochs = N_EPOCHS, 
    batch_size = 5, 
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    x_test_dataset = test_data,
    early_stopping = true,
    file_path = "qubo_train.csv",
    model_setup=setup_dwave,
    sampler=DWave.Neal.Optimizer,
)
```
## RBM for classification

You can use the `RBMClassifier` to train an RBM for classification. It's architecture was based on a [paper](https://dl.acm.org/doi/10.1145/1390156.1390224) by Larochelle et al. (see Figure 1 from this paper).

```julia
using QARBoM

rbm = RBMClassifier(
    10, # number of visible nodes
    5, # number of hidden nodes
    2, # number of nodes for the label
)

QARBoM.train!(
    rbm, 
    train_data,
    y_train,
    PCD; 
    n_epochs = N_EPOCHS, 
    batch_size = BATCH_SIZE, 
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    label_learning_rate = [0.001/(j^0.6) for j in 1:N_EPOCHS], 
    metrics = [Accuracy],
    x_test_dataset = test_data,
    y_test_dataset = y_test,
    early_stopping = true,
    file_path = "my_pcd_metrics_classification.csv",
)
```


## Non-binary visible nodes

You can work with continuous visible nodes with `QARBoM`. 

According to [Hinton's article](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf), you need to normalize your dataset to have zero mean and unit variance. 

In order to use continuous visible nodes for quantum sampling, there is an extra step.
We leverage [QUBO.jl](https://github.com/JuliaQUBO/QUBO.jl) to convert continuous visible nodes to binary visible nodes. 
To do so, you need to define the maximum and minimum values for each visible node. 
See the example below:


```julia
using QARBoM, DWave

train_data = MY_DATA_TRAIN
test_data = MY_DATA_TEST

label_train_data = MY_DATA_TRAIN
label_test_data = MY_DATA_TEST

rbm = RBMClassifier(
    10, # number of visible nodes
    5, # number of hidden nodes
    2, # number of nodes for the label
)

QARBoM.train!(
    rbm, 
    train_data,
    label_train_data,
    QSampling; 
    n_epochs = N_EPOCHS, 
    batch_size = 5, 
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    label_learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    metrics = [Accuracy],
    x_test_dataset = test_data,
    y_test_dataset = label_test_data,
    early_stopping = true,
    file_path = "qubo_train.csv",
    model_setup=setup_dwave,
    sampler=DWave.Neal.Optimizer,
    max_visible = max_visible, # Vector{Float64}
    min_visible = min_visible # Vector{Float64}
)
```
