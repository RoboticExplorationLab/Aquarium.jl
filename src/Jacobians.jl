function constraint_jacobians(model::FSIModel, boundary::ImmersedBoundary, Uk, x_bk)

    E = generate_E(model, boundary, x_bk)

    n_b = Int(length(x_bk)/4)

    ∂vecE_∂x_bk = boundary_coupling_jacobian(model, boundary, x_bk)
    ∂c_∂x_bk = kron(sparse(Uk'), sparse(I, n_b*2, n_b*2))*∂vecE_∂x_bk - kron(sparse([0, 1]'), sparse(I, n_b*2, n_b*2))

    ∂c_∂Uk = E

    return ∂c_∂x_bk, ∂c_∂Uk

end

function NS_jacobians(model::FSIModel, boundary::ImmersedBoundary, Uk, Ukm1, Fk, x_bk)

    ds = vcat(boundary.ds, boundary.ds)
    F = Fk ./ ((model.h_x * model.h_y) ./ ds)

    E_transpose = generate_E(model, boundary, x_bk)'
    G = model.FVM_ops.G
    L = model.FVM_ops.L[1]

    dt = model.dt
    Re = model.Re

    nf_U = size(L, 1)

    # define kkt system submatrices
    A = (1/dt).*(sparse(I, nf_U, nf_U) - (dt/(2*Re)).*L)

    ∂vecE_t∂x_b = boundary_coupling_transpose_jacobian(model, boundary, x_bk)

    ∂NS_∂Uk = A + 0.5.*dNdU(model, Uk)
    ∂NS_∂Ukm1 = -(1/dt).*(sparse(I, nf_U, nf_U) + (dt/(2*Re)).*L) + 0.5.*dNdU(model, Ukm1)
    ∂NS_∂pk = G
    ∂NS_∂Fk = sparse(E_transpose) * spdiagm( 1 ./ ((model.h_x * model.h_y) ./ ds))
    ∂NS_∂x_bk = kron(sparse(F'), sparse(I, nf_U, nf_U))*∂vecE_t∂x_b

    return ∂NS_∂Uk, ∂NS_∂Ukm1, ∂NS_∂pk, ∂NS_∂Fk, ∂NS_∂x_bk

end

function boundary_coupling_jacobian(model::FSIModel, boundary::ImmersedBoundary, X_b)

    x_b = X_b[1:boundary.nodes]
    y_b = X_b[boundary.nodes+1:2*boundary.nodes]

    x_u = model.x_u
    y_u = model.y_u

    x_v = model.x_v
    y_v = model.y_v

    # extract spatial interval steps
    h_x = model.h_x
    h_y = model.h_y

    # determine number of eulerian and lagrange coordinates
    nf_u = length(x_u)
    nf_v = length(x_v)
    n_b = length(x_b)

    # create coordinate matrices
    u_x = (ones(1, n_b) .* x_u)'
    u_y = (ones(1, n_b) .* y_u)'
    
    b_u_x = ones(1, nf_u) .* x_b
    b_u_y = ones(1, nf_u) .* y_b
    
    v_x = (ones(1, n_b) .* x_v)'
    v_y = (ones(1, n_b) .* y_v)'
    
    b_v_x = ones(1, nf_v) .* x_b
    b_v_y = ones(1, nf_v) .* y_b
        
    # calculate delta function values for x and y

    Du_x = discrete_delta((u_x - b_u_x) ./ h_x)
    Du_y = discrete_delta((u_y - b_u_y) ./ h_y)

    Dv_x = discrete_delta((v_x - b_v_x) ./ h_x)
    Dv_y = discrete_delta((v_y - b_v_y) ./ h_y)

    dDu_dxb = -derivative_delta((u_x - b_u_x) ./ h_x) ./ h_x
    dDu_dyb = -derivative_delta((u_y - b_u_y) ./ h_y) ./ h_y

    dDv_dxb = -derivative_delta((v_x - b_v_x) ./ h_x) ./ h_x
    dDv_dyb = -derivative_delta((v_y - b_v_y) ./ h_y) ./ h_y

    grad_Eu = spzeros(nf_u*n_b*2, n_b*4)
    grad_Ev = spzeros(nf_v*n_b*2, n_b*4)
    
    for i in 1:nf_u

        Du_x_i = Du_x[:, i]
        Du_y_i = Du_y[:, i]
    
        Dv_x_i = Dv_x[:, i]
        Dv_y_i = Dv_y[:, i]
    
        dDu_dxb_i = dDu_dxb[:, i]
        dDu_dyb_i = dDu_dyb[:, i]
    
        dDv_dxb_i = dDv_dxb[:, i]
        dDv_dyb_i = dDv_dyb[:, i]
    
        dEu_dxb_i = dDu_dxb_i .* Du_y_i
        dEu_dyb_i = Du_x_i .* dDu_dyb_i
    
        dEv_dxb_i = dDv_dxb_i .* Dv_y_i
        dEv_dyb_i = Dv_x_i .* dDv_dyb_i
    
        grad_Eu[1 + n_b*2*(i-1):n_b*2*(i-1) + n_b, 1:end÷2] = hcat(spdiagm(dEu_dxb_i), spdiagm(dEu_dyb_i))
        grad_Ev[1 + n_b + n_b*2*(i-1):2*n_b + n_b*2*(i-1), 1:end÷2] = hcat(spdiagm(dEv_dxb_i), spdiagm(dEv_dyb_i))
    
    end
    
    grad_E = vcat(grad_Eu, grad_Ev)

    return grad_E
        
end

function boundary_coupling_transpose_jacobian(model::FSIModel, boundary::ImmersedBoundary, X_b)

    x_b = X_b[1:boundary.nodes]
    y_b = X_b[boundary.nodes+1:2*boundary.nodes]

    x_u = model.x_u
    y_u = model.y_u

    x_v = model.x_v
    y_v = model.y_v

    # extract spatial interval steps
    h_x = model.h_x
    h_y = model.h_y

    # determine number of eulerian and lagrange coordinates
    nf_u = length(x_u)
    nf_v = length(x_v)
    n_b = length(x_b)

    # create coordinate matrices
    u_x = ones(1, n_b) .* x_u
    u_y = ones(1, n_b) .* y_u

    b_u_x = (ones(1, nf_u) .* x_b)'
    b_u_y = (ones(1, nf_u) .* y_b)'

    v_x = ones(1, n_b) .* x_v
    v_y = ones(1, n_b) .* y_v

    b_v_x = (ones(1, nf_v) .* x_b)'
    b_v_y = (ones(1, nf_v) .* y_b)'
        
    # calculate delta function values for x and y

    Du_x = discrete_delta((u_x - b_u_x) ./ h_x)
    Du_y = discrete_delta((u_y - b_u_y) ./ h_y)

    Dv_x = discrete_delta((v_x - b_v_x) ./ h_x)
    Dv_y = discrete_delta((v_y - b_v_y) ./ h_y)

    dDu_dxb = -derivative_delta((u_x - b_u_x) ./ h_x) ./ h_x
    dDu_dyb = -derivative_delta((u_y - b_u_y) ./ h_y) ./ h_y

    dDv_dxb = -derivative_delta((v_x - b_v_x) ./ h_x) ./ h_x
    dDv_dyb = -derivative_delta((v_y - b_v_y) ./ h_y) ./ h_y

    grad_Eu_t = spzeros(nf_u*n_b*2, n_b*4)
    grad_Ev_t = spzeros(nf_v*n_b*2, n_b*4)

    for i in 1:boundary.nodes

        Du_x_i = Du_x[:, i]
        Du_y_i = Du_y[:, i]

        Dv_x_i = Dv_x[:, i]
        Dv_y_i = Dv_y[:, i]

        dDu_dxb_i = dDu_dxb[:, i]
        dDu_dyb_i = dDu_dyb[:, i]

        dDv_dxb_i = dDv_dxb[:, i]
        dDv_dyb_i = dDv_dyb[:, i]

        dEu_dxb_i = dDu_dxb_i .* Du_y_i
        dEu_dyb_i = Du_x_i .* dDu_dyb_i

        dEv_dxb_i = dDv_dxb_i .* Dv_y_i
        dEv_dyb_i = Dv_x_i .* dDv_dyb_i

        grad_Eu_t[1 + nf_u*2*(i-1):nf_u*(2*(i-1)+1), i] = dEu_dxb_i
        grad_Eu_t[1 + nf_u*2*(i-1):nf_u*(2*(i-1)+1), i+n_b] = dEu_dyb_i

        grad_Ev_t[1 + nf_v*(2*(i-1)+1):nf_v*2*(i), i] = dEv_dxb_i
        grad_Ev_t[1 + nf_v*(2*(i-1)+1):nf_v*2*(i), i+n_b] = dEv_dyb_i

    end

    grad_E = vcat(grad_Eu_t, grad_Ev_t)

    return grad_E
        
end