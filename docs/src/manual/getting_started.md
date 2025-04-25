## Getting Started

This guide will help you get started with `QARBoM.jl`, a Julia package for training Restricted Boltzmann Machines (RBMs) using various methods, including classical and quantum approaches.

### Defining Your Dataset and RBM

To begin, you need a dataset and an RBM model. The dataset should be a `Vector{Vector{<:Number}}`, where each inner vector represents a sample.

```julia
using QARBoM

# Define your dataset
train_data = MY_DATA_TRAIN
test_data = MY_DATA_TEST

# Create an RBM with 10 visible nodes and 5 hidden nodes
rbm = RBM(10, 5)
```

### Training RBM Using Contrastive Divergence

Contrastive Divergence (CD) is a common method for training RBMs. Below is an example of how to train your RBM using CD:

```julia
N_EPOCHS = 100
BATCH_SIZE = 10

QARBoM.train!(
    rbm, 
    train_data,
    CD; 
    n_epochs = N_EPOCHS,  
    gibbs_steps = 3, 
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    metrics = [MeanSquaredError], 
    x_test_dataset = test_data,
    early_stopping = true,
    file_path = "my_cd_metrics.csv",
)
```

### Training RBM Using Persistent Contrastive Divergence

Persistent Contrastive Divergence (PCD) is an alternative training method that maintains a persistent chain of Gibbs samples.

```julia
QARBoM.train!(
    rbm, 
    train_data,
    PCD; 
    n_epochs = N_EPOCHS, 
    batch_size = BATCH_SIZE, 
    learning_rate = [0.0001/(j^0.8) for j in 1:N_EPOCHS], 
    metrics = [MeanSquaredError], 
    x_test_dataset = test_data,
    early_stopping = true,
    file_path = "my_pcd_metrics.csv",
)
```

### Training RBM Using Quantum Sampling

Quantum Sampling leverages quantum annealers for training RBMs. Below is an example setup using the `DWave` package:

```julia
using DWave

# Define a setup for your quantum sampler
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
    model_setup = setup_dwave,
    sampler = DWave.Neal.Optimizer,
)
```

### RBM for Classification

For classification tasks, you can use the `RBMClassifier`, which includes label nodes in its architecture.

```julia
rbm = RBMClassifier(
    10, # number of visible nodes
    5,  # number of hidden nodes
    2,  # number of label nodes
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

### Non-Binary Visible Nodes

`QARBoM.jl` supports continuous visible nodes. Ensure your dataset is normalized to have zero mean and unit variance, as recommended by [Hinton's guide](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf).

For quantum sampling with continuous visible nodes, you need to define the maximum and minimum values for each visible node:

```julia
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
    model_setup = setup_dwave,
    sampler = DWave.Neal.Optimizer,
    max_visible = max_visible, 
    min_visible = min_visible
)
```

