
<picture>
  <source media="(prefers-color-scheme: light)" srcset="./assets/logo-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-dark.svg">
  <img alt="QARBoM.jl logo">
</picture>

---

Quantum-Assisted Restricted Boltzmann Machine Training Framework

This package provides a framework for training Restricted Boltzmann Machines (RBMs) via classical algorithms (Contrastive Divergence and Persistent Contrastive Divergence) and quantum-assisted methods, which involve the use of Quantum Samplimg.

Using the [QUBO.jl](www.github.com/JuliaQUBO/QUBO.jl) package, this package allows training RBMs using different quantum computers and simulators and converting continuous visible nodes to binary visible nodes seamlessly. 

## Installation

```julia
] add www.github.com/NITeQ/QARBoM.jl
```
