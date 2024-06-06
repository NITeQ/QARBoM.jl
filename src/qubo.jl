function update_qubo!(
    rbm::QUBORBM, 
    v_data::Vector{Int}, 
    h_data::Vector{Float64},
    v_model::Vector{Float64},
    h_model::Vector{Float64},
    learning_rate::Float64
    )

    n, L, Q, α, β =  QUBOTools.qubo(QUBOTools.Model(JuMP.backend(rbm.model)), :dense)
 
    W_update = learning_rate .* (v_data*h_data' - v_model*h_model')
    L_update_vis = learning_rate .* (v_data - v_model)
    L_update_hid = learning_rate .* (h_data - h_model)

    # make W upper triangular
    # for i in 1:num_visible_nodes(rbm), j in 1:num_hidden_nodes(rbm)
    #     j_mapped = j + num_visible_nodes(rbm)
    #     Q[i, j_mapped] += W_update[i, j]
    # end
    Q[1:num_visible_nodes(rbm), num_visible_nodes(rbm)+1:end] .+= W_update
    L .+= vcat(L_update_vis, L_update_hid)
    # for i in 1:num_visible_nodes(rbm)
    #     L[i] += L_update_vis[i]
    # end
    # for j in 1:num_hidden_nodes(rbm)
    #     j_mapped = j + num_visible_nodes(rbm)
    #     L[j_mapped] += L_update_hid[j]
    # end

    quadratic_coefficients = Vector{Float64}()
    quadratic_terms_1 = Vector{JuMP.VariableRef}()
    quadratic_terms_2 = Vector{JuMP.VariableRef}()

    for i in 1:num_visible_nodes(rbm), j in 1:num_hidden_nodes(rbm)

        j_mapped = j + num_visible_nodes(rbm)

        push!(quadratic_terms_1, rbm.model[:vis][i])
        push!(quadratic_terms_2, rbm.model[:hid][j])
        push!(quadratic_coefficients, Q[i, j_mapped])
    end

    JuMP.set_objective_coefficient(rbm.model, quadratic_terms_1, quadratic_terms_2, quadratic_coefficients)

    linear_coefficients = Vector{Float64}()
    linear_terms = Vector{JuMP.VariableRef}()

    for i in 1:num_visible_nodes(rbm)
        push!(linear_terms, rbm.model[:vis][i])
        push!(linear_coefficients, L[i])
    end
    for j in 1:num_hidden_nodes(rbm)
        j_mapped = j + num_visible_nodes(rbm)
        push!(linear_terms, rbm.model[:hid][j])
        push!(linear_coefficients, L[j_mapped])
    end

    
    JuMP.set_objective_coefficient(rbm.model, linear_terms, linear_coefficients)

    return
end
    