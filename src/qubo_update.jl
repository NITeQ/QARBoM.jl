function update_qubo!(
	rbm::QUBORBM,
	W::Matrix{Float64},
    a::Vector{Float64},
    b::Vector{Float64},
)

	quadratic_coefficient_changes = Vector{MOI.ScalarQuadraticCoefficientChange{Float64}}()
	# quadratic_terms_1 = Vector{JuMP.VariableRef}()
	# quadratic_terms_2 = Vector{JuMP.VariableRef}()

	for i in 1:num_visible_nodes(rbm)
		for j in 1:num_hidden_nodes(rbm)
			# push!(quadratic_terms_1, rbm.model[:vis][i])
			# push!(quadratic_terms_2, rbm.model[:hid][j])
			# push!(quadratic_coefficients, - W[i,j])
			push!(quadratic_coefficient_changes, MOI.ScalarQuadraticCoefficientChange(MOI.VariableIndex(rbm.model[:vis][i]) , MOI.VariableIndex(rbm.model[:hid][j]), - W[i,j]))
		end

		# JuMP.set_objective_coefficient(rbm.model, quadratic_terms_1, quadratic_terms_2, quadratic_coefficients)
		# empty!(quadratic_coefficients)
		# empty!(quadratic_terms_1)
		# empty!(quadratic_terms_2)
	end
	MOI.modify(rbm.model.moi_backend, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), quadratic_coefficient_changes)

	# JuMP.set_objective_coefficient(rbm.model, quadratic_terms_1, quadratic_terms_2, quadratic_coefficients)

	# linear_coefficients = Vector{Float64}()
	# linear_terms = Vector{JuMP.VariableRef}()

	linear_coefficient_changes = Vector{MOI.ScalarAffineCoefficientChange{Float64}}()

	for i in 1:num_visible_nodes(rbm)
		# push!(linear_terms, rbm.model[:vis][i])
		# push!(linear_coefficients, - a[i])
		push!(linear_coefficient_changes, MOI.ScalarAffineCoefficientChange(MOI.VariableIndex(rbm.model[:vis][i]), - a[i]))
	end
	for j in 1:num_hidden_nodes(rbm)
		# push!(linear_terms, rbm.model[:hid][j])
		# push!(linear_coefficients, - b[j])
		push!(linear_coefficient_changes, MOI.ScalarAffineCoefficientChange(MOI.VariableIndex(rbm.model[:hid][j]), - b[j]))
	end
        
	# JuMP.set_objective_coefficient(rbm.model, linear_terms, linear_coefficients)
	MOI.modify(rbm.model.moi_backend, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), linear_coefficient_changes)

	return
end
