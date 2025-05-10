# QARBoM.jl

## Introduction

`QARBoM.jl`, a platform for benchmarking quantum-assisted against classical training of Restricted Boltzmann Machines (RBMs).
Recent works have been testing the training of RBMs using quantum sampling techniques, such as Quantum Annealing, and comparing their results against classical methods.
However, these projects are mainly limited to a specific dataset and only one RBM classical training procedure to compare.
With that said, `QARBoM.jl` establishes an agnostic benchmarking framework where, with minor code adjustments, one can select over different training algorithms~(classical or quantum-assisted) and parameters to model integer or real-valued datasets, expediting the research endeavor on the applications of Quantum Computing for RBMs. 

## Installation

QARBoM is avaible through Julia's General Registry:

```julia-repl
julia> import Pkg

julia> Pkg.add("https://github.com/NITeQ/QARBoM.jl")

julia> using QARBoM
```
