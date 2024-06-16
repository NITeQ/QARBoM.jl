using JuMP
using LinearAlgebra
using QUBO
using DWave

Q = [
    0  0  2  2
    0  0  2  2
    2  2  0  0
    2  2  0  0
]

model = Model(DWave.Neal.Optimizer)
@variable(model, x[1:2], Bin)
@variable(model, y[1:2], Bin)

@objective(model, Min, x' * Q * x)

optimize!(model)

n, L, Q_opt, a, b = QUBOTools.qubo(unsafe_backend(model), :dense)

@assert (Q_opt+Q_opt')/2 == Q - Diagonal(L)

new_Q = [
    0  0  1  1
    0  0  1  1
    1  1  0  0 
    1  1  0  0
]

coefficients = Vector{Float64}()
variables_1 = Vector{JuMP.VariableRef}()    
variables_2 = Vector{JuMP.VariableRef}()
for i in 1:2, j in 1:2
    push!(variables_1, model[:x][i])
    push!(variables_2, model[:x][j])
    push!(coefficients, 2*new_Q[i, j])
end
JuMP.set_objective_coefficient(model, variables_1, variables_2, coefficients)

optimize!(model)

n, L, Q_opt_new, a, b = QUBOTools.qubo(unsafe_backend(model), :dense)

@assert (Q_opt_new+Q_opt_new')/2 == new_Q - Diagonal(L)