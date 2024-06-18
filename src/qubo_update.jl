function update_qubo!(
	rbm::QUBORBM,
	W::Matrix{Float64},
    a::Vector{Float64},
    b::Vector{Float64},
)

	quadratic_coefficients = Vector{Float64}()
	quadratic_terms_1 = Vector{JuMP.VariableRef}()
	quadratic_terms_2 = Vector{JuMP.VariableRef}()

	for i in 1:num_visible_nodes(rbm)
		for j in 1:num_hidden_nodes(rbm)
			push!(quadratic_terms_1, rbm.model[:vis][i])
			push!(quadratic_terms_2, rbm.model[:hid][j])
			push!(quadratic_coefficients, - W[i,j])
		end
		# JuMP.set_objective_coefficient(rbm.model, quadratic_terms_1, quadratic_terms_2, quadratic_coefficients)
		# empty!(quadratic_coefficients)
		# empty!(quadratic_terms_1)
		# empty!(quadratic_terms_2)
	end

	JuMP.set_objective_coefficient(rbm.model, quadratic_terms_1, quadratic_terms_2, quadratic_coefficients)

	linear_coefficients = Vector{Float64}()
	linear_terms = Vector{JuMP.VariableRef}()

	for i in 1:num_visible_nodes(rbm)
		push!(linear_terms, rbm.model[:vis][i])
		push!(linear_coefficients, - a[i])
	end
	for j in 1:num_hidden_nodes(rbm)
		push!(linear_terms, rbm.model[:hid][j])
		push!(linear_coefficients, - b[j])
	end
        
	JuMP.set_objective_coefficient(rbm.model, linear_terms, linear_coefficients)

	return
end
