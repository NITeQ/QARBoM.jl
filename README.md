<div align="center">

<picture>

  <source media="(prefers-color-scheme: light)" srcset="./docs/src/assets/logo-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./docs/src/assets/logo-dark.svg">

  <img height="200" alt="QARBoM.jl logo">
  
</picture>

[build-img]: https://github.com/NITeQ/QARBoM.jl/actions/workflows/ci.yml/badge.svg?branch=main
![Build Status][build-img]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14841099.svg)](https://doi.org/10.5281/zenodo.14841099)
[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://niteq.github.io/QARBoM.jl/dev/)


</div>



---

Quantum-Assisted Restricted Boltzmann Machine Training Framework

This package provides a framework for training Restricted Boltzmann Machines (RBMs) via classical algorithms (Contrastive Divergence and Persistent Contrastive Divergence) and quantum-assisted methods, which involve the use of Quantum Samplimg.

Using the [QUBO.jl](https://github.com/JuliaQUBO/QUBO.jl) package, this package allows training RBMs using different quantum computers and simulators and converting continuous visible nodes to binary visible nodes seamlessly. 

## Installation

```julia
] add www.github.com/NITeQ/QARBoM.jl
```

For more information on how to install and use the package, please refer to the [documentation](https://niteq.github.io/QARBoM.jl/dev/).

