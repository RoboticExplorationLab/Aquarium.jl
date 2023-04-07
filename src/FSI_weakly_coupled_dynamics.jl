"""
Functions for simulating FSI dynamics with predefined boundary motion
"""

## WITHOUT jacobians
function simulate(model::FSIModel, boundary::ImmersedBoundary,
    u0::AbstractVector, p0::AbstractVector, f0_b::AbstractVector,
    x_rollout::VecOrMat{<:AbstractVector}; t0=0.0, tf=5.0, max_iter=10,
    λ1=1e-6, λ2=0.0, tol=1e-6, iter_refine=false, alg=:pardiso,
    verbose=false)
    
    # build time history vectors
    if model.normalize
        dt = model.dt * (model.ref_L / model.ref_u) 
    else
        dt = model.dt 
        model = normalize(model)
    end

    N = Int((tf-t0)÷dt + 1)
    x_hist = copy(x_rollout)
    u_hist = [copy(u0) for _ in 1:N]
    p_hist = [copy(p0) for _ in 1:N]
    f_b_hist = [copy(f0_b) for _ in 1:N]
    t_hist = t0:dt:tf

    En = boundary_coupling(model, boundary, boundary_state(boundary, x_hist[1]))

    if alg == :pardiso
        solver = Pardiso.MKLPardisoSolver()
        Pardiso.set_nprocs!(solver, Base.Threads.nthreads())
        Pardiso.set_matrixtype!(solver, Pardiso.REAL_NONSYM)
        Pardiso.pardisoinit(solver)
        Pardiso.fix_iparm!(solver, :N)
        Pardiso.set_iparm!(solver, 5, 0)
        Pardiso.set_iparm!(solver, 8, 50)
        Pardiso.set_iparm!(solver, 10, 13)
        Pardiso.set_iparm!(solver, 11, 0)
        Pardiso.set_iparm!(solver, 13, 0)
    else
        solver = nothing
    end

    @showprogress "Simulating..." for ind in 2:N

        t = t_hist[ind]

        if verbose
            println("\n")
            @show t
        end

        if ind > length(x_rollout)
            
            xn = x_hist[end]
            xk = x_hist[end]
            push!(x_hist, x_hist[end])
        
        else
            
            xn = x_hist[ind]
            xk = x_hist[ind-1]
            
            if !(xn == xk)
                En = boundary_coupling(model, boundary, boundary_state(boundary, xn)
                )
            end
        end

        # integrate over N-S
        discrete_dynamics!(model, boundary, En, xn, u_hist[ind], p_hist[ind], f_b_hist[ind],
            u_hist[ind-1], p_hist[ind-1], f_b_hist[ind-1]; max_iter=max_iter, λ1=λ1, λ2=λ2, 
            tol=tol, alg=alg, iter_refine=iter_refine, verbose=verbose, solver=solver
        )

    end

    ind = minimum([length(t_hist), length(f_b_hist)])

    return t_hist[1:ind], x_hist[1:ind], u_hist[1:ind], p_hist[1:ind], f_b_hist[1:ind]
end
function simulate!(model::FSIModel, boundary::ImmersedBoundary,
    un::AbstractVector, pn::AbstractVector, fn_b::AbstractVector,
    u0::AbstractVector, p0::AbstractVector, f0_b::AbstractVector,
    x_rollout::VecOrMat{<:AbstractVector}; t0=0.0, tf=5.0, max_iter=10,
    λ1=1e-6, λ2=0.0, tol=1e-6, iter_refine=false, alg=:pardiso, verbose=false)

    uk = deepcopy(u0)
    pk = deepcopy(p0)
    fk_b = deepcopy(f0_b)
    
    if model.normalize
        dt = model.dt * (model.ref_L / model.ref_u) 
    else
        dt = model.dt 
    end

    N = Int((tf-t0)÷dt + 1)
    t_hist = t0:dt:tf

    if alg == :pardiso
        solver = Pardiso.MKLPardisoSolver()
        Pardiso.set_nprocs!(solver, Threads.nthreads())
        Pardiso.set_matrixtype!(solver, Pardiso.REAL_NONSYM)
        Pardiso.pardisoinit(solver)
        Pardiso.fix_iparm!(solver, :N)
        Pardiso.set_iparm!(solver, 5, 0)
        Pardiso.set_iparm!(solver, 8, 50)
        Pardiso.set_iparm!(solver, 11, 0)
        Pardiso.set_iparm!(solver, 13, 0)
    else
        solver = nothing
    end
    
    @showprogress "Simulating..." for ind in 2:N

        t = t_hist[ind]

        if verbose
            println("\n")
            @show t
        end

        if ind > length(x_rollout)
            
            xn = x_rollout[end]
            xk = x_rollout[end]
        
        elseif ind >= 2
            
            xn = x_rollout[ind]
            xk = x_rollout[ind-1]
            
            if !(xn == xk)
                En = boundary_coupling(model, boundary, boundary_state(boundary, xn)
                )
            end
        end

        # integrate over N-S
        discrete_dynamics!(model, boundary, En, xn, uk, pk, fk_b, uk, pk, fk_b;
            max_iter=max_iter, λ1=λ1, λ2=λ2, tol=tol, alg=alg, iter_refine=iter_refine,
            verbose=verbose, solver=solver
        )

    end

    un .= uk
    pn .= pk
    fn_b .= fk_b

end

function discrete_dynamics(model::FSIModel, boundary::ImmersedBoundary, En::AbstractMatrix,
    xn::AbstractVector, uk::AbstractVector, pk::AbstractVector, fk_b::AbstractVector;
    max_iter=10, λ1=1e-6, λ2=0.0, tol=1e-6, iter_refine=false, alg=:pardiso,
    verbose=false, solver=nothing)

    un = deepcopy(uk)
    pn = deepcopy(pk)
    f_bn = deepcopy(fk_b)

    discrete_dynamics!(model, boundary, En, xn, un, pn, f_bn, uk, pk, fk_b;
        max_iter=max_iter, λ1=λ1, λ2=λ2, tol=tol, alg=alg, iter_refine=iter_refine,
        verbose=verbose, solver=solver)

    return un, pn, f_bn

end
function discrete_dynamics!(model::FSIModel, boundary::ImmersedBoundary, En::AbstractMatrix, xn::AbstractVector,
    un::AbstractVector, pn::AbstractVector, fn_b::AbstractVector, uk::AbstractVector, pk::AbstractVector,
    fk_b::AbstractVector; max_iter=10, λ1=1e-6, λ2=0.0, tol=1e-6, iter_refine=false, alg=:pardiso, verbose=false,
    solver=nothing)

    xn_b = boundary_state(boundary, xn)
    un_b = xn_b[end÷2+1:end]
    
    # deepcopy
    un .= deepcopy(uk)
    pn .= deepcopy(pk)
    fn_b_tilda = deepcopy(fk_b) ./ (model.ρ * model.h_x * model.h_y)

    # extract operators
    G = model.FVM_ops.G
    L = model.FVM_ops.L[1]
    L_bc = model.FVM_ops.L[3]
    D_bc = model.FVM_ops.D[2]

    dt = model.dt
    Re = model.Re

    nf_u = length(uk)
    m_D = length(pk)
    m_E = length(fk_b)

    # define kkt system submatrices
    A = (1/dt).*(sparse(I, nf_u, nf_u) - (dt/(2*Re)).*L)
    r = (1/dt).*(sparse(I, nf_u, nf_u) + (dt/(2*Re)).*L)*un - 0.5.*N(model, un)
    bc1 = (1/Re).*L_bc
    bc2 = -D_bc
    bc3 = sparse(-un_b)

    ∂c1∂un = G'
    ∂c2∂un = En

    # define residuals
    R1(u, p, f_b_tilda) = A*u + 0.5.*N(model, u) - (r + bc1) + G*p + En'*f_b_tilda
    c1(u) = G'*u + bc2
    c2(u) = En*u + bc3
    
    # solve using Newton's Method
    num_iter = 0

    if verbose
        @show maximum(abs.(R1(un, pn, fn_b_tilda)))
    end

    if alg == :qr

        while num_iter == 0 || maximum(abs.(R1(un, pn, fn_b_tilda))) > tol && num_iter <= max_iter
            
            num_iter = num_iter + 1
            
            # form KKT matrix
            ∂R1∂un = A + 0.5.*N_jacobian(model, un)

            KKT_matrix = vcat(hcat(∂R1∂un, ∂c1∂un', ∂c2∂un'),
                hcat(∂c1∂un, -λ1.*sparse(I, m_D, m_D), spzeros(m_D, m_E)),
                hcat(∂c2∂un, spzeros(m_E, m_D), -λ2.*sparse(I, m_E, m_E)))
            
            b = vcat(-R1(un, pn, fn_b_tilda), -c1(un), -c2(un))
        
            # solve using QR
            KKT_sol = qr(KKT_matrix)\Vector(b)

            # save updated fluid states to CFDModel
            un .+= KKT_sol[1:nf_u]
            pn .+= KKT_sol[nf_u+1:nf_u+m_D]
            fn_b_tilda .+= KKT_sol[end-m_E+1:end]
        
            if verbose
                @show maximum(abs.(R1(un, pn, fn_b_tilda)))
            end
        
        end

    
    elseif alg == :lu

        KKT_prev = 1.0 .* sparse(I, 5, 5)
        F = lu(KKT_prev)

        while num_iter == 0 || maximum(abs.(R1(un, pn, fn_b_tilda))) > tol && num_iter <= max_iter
            
            num_iter = num_iter + 1
            
            # form KKT matrix
            ∂R1∂un = A + 0.5.*N_jacobian(model, un)
        
            KKT_matrix_reg = vcat(hcat(∂R1∂un, ∂c1∂un', ∂c2∂un'),
                hcat(∂c1∂un, -λ1.*sparse(I, m_D, m_D), spzeros(m_D, m_E)),
                hcat(∂c2∂un, spzeros(m_E, m_D), -λ2.*sparse(I, m_E, m_E)))

            KKT_matrix = vcat(hcat(∂R1∂un, ∂c1∂un', ∂c2∂un'),
                hcat(∂c1∂un, spzeros(m_D, m_D), spzeros(m_D, m_E)),
                hcat(∂c2∂un, spzeros(m_E, m_D), spzeros(m_E, m_E)))
            
            b = vcat(-R1(un, pn, fn_b_tilda), -c1(un), -c2(un))

            # solve using Lu
            if KKT_prev.colptr == KKT_matrix_reg.colptr &&
                KKT_prev.rowval == KKT_matrix_reg.rowval
                lu!(F, KKT_matrix_reg)
            else
                F = lu(KKT_matrix_reg)
            end

            KKT_sol = F\Vector(b)

            if iter_refine

                # calculate residual
                ϵ = b - KKT_matrix*KKT_sol

                if verbose
                    @show maximum(abs.(ϵ))
                end

                # iterative refinement
                while maximum(ϵ) > tol && num_iter <= max_iter

                    u = F\Vector(ϵ)
                    KKT_sol .+= u

                    ϵ .= b - KKT_matrix*KKT_sol

                    if verbose
                        @show maximum(abs.(ϵ))
                    end
                    
                end
            end

            # save updated fluid states to CFDModel
            un .+= KKT_sol[1:nf_u]
            pn .+= KKT_sol[nf_u+1:nf_u+m_D]
            fn_b_tilda .+= KKT_sol[end-m_E+1:end]

            # define previous iter KKT
            KKT_prev = deepcopy(KKT_matrix_reg)
        
            if verbose
                @show maximum(abs.(R1(un, pn, fn_b_tilda)))
            end
        
        end

    elseif alg == :pardiso

        KKT_sol = zeros(nf_u + m_D + m_E)

        while num_iter == 0 || maximum(abs.(R1(un, pn, fn_b_tilda))) > tol && num_iter <= max_iter
            
            num_iter = num_iter + 1
            
            # form KKT matrix

            ∂R1∂un = A + 0.5.*N_jacobian(model, un)

            KKT_matrix = [∂R1∂un ∂c1∂un' ∂c2∂un';
                ∂c1∂un -λ1.*sparse(I, m_D, m_D) spzeros(m_D, m_E);
                ∂c2∂un spzeros(m_E, m_D) -λ2.*sparse(I, m_E, m_E);
            ]

            b = vcat(-R1(un, pn, fn_b_tilda), -c1(un), -c2(un))
            KKT_matrix_pardiso = get_matrix(solver, KKT_matrix, :N)
        
            # Analyze the matrix and compute a symbolic factorization.
            set_phase!(solver, Pardiso.ANALYSIS)
            pardiso(solver, KKT_matrix_pardiso, Vector(b))
        
            # Compute the solutions x_rollout using the symbolic factorization.
            set_phase!(solver, Pardiso.NUM_FACT_SOLVE_REFINE)
            pardiso(solver, KKT_sol, KKT_matrix_pardiso, Vector(b))
        
            # Free the PARDISO data structures.
            set_phase!(solver, Pardiso.RELEASE_ALL)
            pardiso(solver)
        
            # save updated fluid states to CFDModel
            un .+= KKT_sol[1:nf_u]
            pn .+= KKT_sol[nf_u+1:nf_u+m_D]
            fn_b_tilda .+= KKT_sol[end-m_E+1:end]
        
            if verbose
                @show maximum(abs.(R1(un, pn, fn_b_tilda)))
            end
        
        end

    end

    fn_b .= fn_b_tilda .* (model.ρ * model.h_x * model.h_y)

end

## WITH jacobians
function simulate_diff(model::FSIModel, boundary::ImmersedBoundary,
    u0::AbstractVector, p0::AbstractVector, f0_b::AbstractVector,
    x_rollout::VecOrMat{<:AbstractVector}, θs::AbstractVector, θg::AbstractVector;
    df0_bdθs::AbstractArray=zeros(length(f0_b), length(θs)),
    df0_bdθg::AbstractArray=zeros(length(f0_b), length(θg)),
    t0=0.0, tf=5.0, max_iter=10, λ1=1e-6, λ2=0.0, tol=1e-6,
    iter_refine=false, alg=:pardiso, verbose=false, kwargs...)
    
    # build time history vectors
    if model.normalize
        dt = model.dt * (model.ref_L / model.ref_u) 
    else
        dt = model.dt 
        model = normalize(model)
    end

    N = Int((tf-t0)÷dt + 1)
    x_hist = copy(x_rollout)
    u_hist = [copy(u0) for _ in 1:N]
    p_hist = [copy(p0) for _ in 1:N]
    f_b_hist = [copy(f0_b) for _ in 1:N]
    t_hist = t0:dt:tf
    df0_bdθ = [df0_bdθs df0_bdθg]

    dudθ_hist = [zeros(length(u0), length(θs)+length(θg)) for _ in 1:N]
    dpdθ_hist = [zeros(length(p0), length(θs)+length(θg)) for _ in 1:N]
    df_bdθ_hist = [copy(df0_bdθ) for _ in 1:N]

    En = boundary_coupling(model, boundary, boundary_state(boundary, x_hist[1]))

    if alg == :pardiso
        solver = Pardiso.MKLPardisoSolver()
        Pardiso.set_nprocs!(solver, Base.Threads.nthreads())
        Pardiso.set_matrixtype!(solver, Pardiso.REAL_NONSYM)
        Pardiso.pardisoinit(solver)
        Pardiso.fix_iparm!(solver, :N)
        Pardiso.set_iparm!(solver, 5, 0)
        Pardiso.set_iparm!(solver, 8, 50)
        Pardiso.set_iparm!(solver, 10, 13)
        Pardiso.set_iparm!(solver, 11, 0)
        Pardiso.set_iparm!(solver, 13, 0)
    else
        solver = nothing
    end

    @showprogress "Simulating..." for ind in 2:N

        t = t_hist[ind]

        if verbose
            println("\n")
            @show t
        end

        if ind > length(x_hist)
            
            xn = x_hist[end]
            xk = x_hist[end]
            push!(x_hist, x_hist[end])
        
        else
            
            xn = x_hist[ind]
            xk = x_hist[ind-1]
            
            if !(xn == xk)
                En = boundary_coupling(model, boundary, boundary_state(boundary, xn))
            end
        end

        # integrate over N-S

        model.normalize ? tn=t/(model.ref_L / model.ref_u) : tn=t
        ∂xn∂θg = x_θg_jacobian(boundary, θg, tn; kwargs...)
        ∂xn_b∂θs = x_b_θs_jacobian(boundary, θs, xn; kwargs...)

        discrete_dynamics_diff!(model, boundary, En, xn, u_hist[ind], p_hist[ind], f_b_hist[ind],
            dudθ_hist[ind], dpdθ_hist[ind], df_bdθ_hist[ind], u_hist[ind-1], p_hist[ind-1],
            f_b_hist[ind-1], dudθ_hist[ind-1], ∂xn_b∂θs, ∂xn∂θg; max_iter=max_iter, λ1=λ1, λ2=λ2, 
            tol=tol, alg=alg, iter_refine=iter_refine, verbose=verbose, solver=solver
        )

    end

    ind = minimum([length(t_hist), length(f_b_hist)])

    return t_hist[1:ind], x_hist[1:ind], u_hist[1:ind], p_hist[1:ind], f_b_hist[1:ind],
        dudθ_hist[1:ind], dpdθ_hist[1:ind], df_bdθ_hist[1:ind]
    
end

function discrete_dynamics_diff(model::FSIModel, boundary::ImmersedBoundary, En::AbstractMatrix,
    xn::AbstractVector, uk::AbstractVector, pk::AbstractVector, fk_b::AbstractVector,
    dukdθ::AbstractMatrix, ∂xn_b∂θs::AbstractMatrix, ∂xn∂θg::AbstractMatrix; max_iter=10, λ1=1e-6, λ2=0.0,
    tol=1e-6, iter_refine=false, alg=:pardiso, verbose=false, solver=nothing)

    un = deepcopy(uk)
    pn = deepcopy(pk)
    f_bn = deepcopy(fk_b)
    dundθ = zeros(length(uk), size(∂xn_b∂θs, 2)+size(∂xn∂θg, 2))
    dpndθ = zeros(length(pk), size(∂xn_b∂θs, 2)+size(∂xn∂θg, 2))
    dfn_bdθ = zeros(length(fk_b), size(∂xn_b∂θs, 2)+size(∂xn∂θg, 2))

    discrete_dynamics_diff!(model, boundary, En, xn, un, pn, f_bn,
        dundθ, dpndθ, dfn_bdθ, uk, pk, fk_b, dukdθ, ∂xn_b∂θs, ∂xn∂θg;
        max_iter=max_iter, λ1=λ1, λ2=λ2, tol=tol, alg=alg, iter_refine=iter_refine,
        verbose=verbose, solver=solver
    )

    return un, pn, f_bn, dundθ, dpndθ, dfn_bdθ

end

function discrete_dynamics_diff!(model::FSIModel, boundary::ImmersedBoundary, En::AbstractMatrix,
    xn::AbstractVector, un::AbstractVector, pn::AbstractVector, fn_b::AbstractVector, 
    dundθ::AbstractVecOrMat, dpndθ::AbstractVecOrMat, dfn_bdθ::AbstractVecOrMat,
    uk::AbstractVector, pk::AbstractVector, fk_b::AbstractVector, dukdθ::AbstractVecOrMat,
    ∂xn_b∂θs::AbstractVecOrMat, ∂xn∂θg::AbstractVecOrMat; max_iter=10, λ1=1e-6, λ2=0.0, tol=1e-6,
    iter_refine=false, alg=:pardiso, verbose=false, solver=nothing)

    xn_b = boundary_state(boundary, xn)
    un_b = xn_b[end÷2+1:end]
    
    # deepcopy
    un .= deepcopy(uk)
    pn .= deepcopy(pk)
    fn_b_tilda = deepcopy(fk_b) ./ (model.ρ * model.h_x * model.h_y)

    # extract operators
    G = model.FVM_ops.G
    L = model.FVM_ops.L[1]
    L_bc = model.FVM_ops.L[3]
    D_bc = model.FVM_ops.D[2]

    dt = model.dt
    Re = model.Re

    nf_u = length(uk)
    m_D = length(pk)
    m_E = length(fk_b)

    # define kkt system submatrices
    A = (1/dt).*(sparse(I, nf_u, nf_u) - (dt/(2*Re)).*L)
    r = (1/dt).*(sparse(I, nf_u, nf_u) + (dt/(2*Re)).*L)*un - 0.5.*N(model, un)
    bc1 = (1/Re).*L_bc
    bc2 = -D_bc
    bc3 = sparse(-un_b)

    ∂c1∂un = G'
    ∂c2∂un = En

    # define residuals
    R1(u, p, f_b_tilda) = A*u + 0.5.*N(model, u) - (r + bc1) + G*p + En'*f_b_tilda
    c1(u) = G'*u + bc2
    c2(u) = En*u + bc3
    
    # solve using Newton's Method
    num_iter = 0

    if verbose
        @show maximum(abs.(R1(un, pn, fn_b_tilda)))
    end

    if alg == :qr

        while num_iter == 0 || maximum(abs.(R1(un, pn, fn_b_tilda))) > tol && num_iter <= max_iter
            
            num_iter = num_iter + 1
            
            # form KKT matrix
            ∂R1∂un = A + 0.5.*N_jacobian(model, un)

            KKT_matrix = vcat(hcat(∂R1∂un, ∂c1∂un', ∂c2∂un'),
                hcat(∂c1∂un, -λ1.*sparse(I, m_D, m_D), spzeros(m_D, m_E)),
                hcat(∂c2∂un, spzeros(m_E, m_D), -λ2.*sparse(I, m_E, m_E)))
            
            b = vcat(-R1(un, pn, fn_b_tilda), -c1(un), -c2(un))
        
            # solve using QR
            KKT_sol = qr(KKT_matrix)\Vector(b)

            # save updated fluid states to CFDModel
            un .+= KKT_sol[1:nf_u]
            pn .+= KKT_sol[nf_u+1:nf_u+m_D]
            fn_b_tilda .+= KKT_sol[end-m_E+1:end]
        
            if verbose
                @show maximum(abs.(R1(un, pn, fn_b_tilda)))
            end
        
        end

    
    elseif alg == :lu

        KKT_prev = 1.0 .* sparse(I, 5, 5)
        F = lu(KKT_prev)

        while num_iter == 0 || maximum(abs.(R1(un, pn, fn_b_tilda))) > tol && num_iter <= max_iter
            
            num_iter = num_iter + 1
            
            # form KKT matrix
            ∂R1∂un = A + 0.5.*N_jacobian(model, un)
        
            KKT_matrix_reg = vcat(hcat(∂R1∂un, ∂c1∂un', ∂c2∂un'),
                hcat(∂c1∂un, -λ1.*sparse(I, m_D, m_D), spzeros(m_D, m_E)),
                hcat(∂c2∂un, spzeros(m_E, m_D), -λ2.*sparse(I, m_E, m_E)))

            KKT_matrix = vcat(hcat(∂R1∂un, ∂c1∂un', ∂c2∂un'),
                hcat(∂c1∂un, spzeros(m_D, m_D), spzeros(m_D, m_E)),
                hcat(∂c2∂un, spzeros(m_E, m_D), spzeros(m_E, m_E)))
            
            b = vcat(-R1(un, pn, fn_b_tilda), -c1(un), -c2(un))

            # solve using Lu
            if KKT_prev.colptr == KKT_matrix_reg.colptr &&
                KKT_prev.rowval == KKT_matrix_reg.rowval
                lu!(F, KKT_matrix_reg)
            else
                F = lu(KKT_matrix_reg)
            end

            KKT_sol = F\Vector(b)

            if iter_refine

                # calculate residual
                ϵ = b - KKT_matrix*KKT_sol

                if verbose
                    @show maximum(abs.(ϵ))
                end

                # iterative refinement
                while maximum(ϵ) > tol && num_iter <= max_iter

                    u = F\Vector(ϵ)
                    KKT_sol .+= u

                    ϵ .= b - KKT_matrix*KKT_sol

                    if verbose
                        @show maximum(abs.(ϵ))
                    end
                    
                end
            end

            # save updated fluid states to CFDModel
            un .+= KKT_sol[1:nf_u]
            pn .+= KKT_sol[nf_u+1:nf_u+m_D]
            fn_b_tilda .+= KKT_sol[end-m_E+1:end]

            # define previous iter KKT
            KKT_prev = deepcopy(KKT_matrix_reg)
        
            if verbose
                @show maximum(abs.(R1(un, pn, fn_b_tilda)))
            end
        
        end

    elseif alg == :pardiso

        KKT_sol = zeros(nf_u + m_D + m_E)
        KKT_matrix_pardiso = spzeros(nf_u + m_D + m_E, nf_u + m_D + m_E)

        while num_iter == 0 || maximum(abs.(R1(un, pn, fn_b_tilda))) > tol && num_iter <= max_iter
            
            num_iter = num_iter + 1
            
            # form KKT matrix

            ∂R1∂un = A + 0.5.*N_jacobian(model, un)

            KKT_matrix = [∂R1∂un ∂c1∂un' ∂c2∂un';
                ∂c1∂un -λ1.*sparse(I, m_D, m_D) spzeros(m_D, m_E);
                ∂c2∂un spzeros(m_E, m_D) -λ2.*sparse(I, m_E, m_E);
            ]

            b = vcat(-R1(un, pn, fn_b_tilda), -c1(un), -c2(un))
            KKT_matrix_pardiso .= get_matrix(solver, KKT_matrix, :N)
        
            # Analyze the matrix and compute a symbolic factorization.
            set_phase!(solver, Pardiso.ANALYSIS)
            pardiso(solver, KKT_matrix_pardiso, Vector(b))

            # Compute the numeric factorization.
            set_phase!(solver, Pardiso.NUM_FACT)
            pardiso(solver, KKT_matrix_pardiso, Vector(b))
        
            # Compute the solution
            set_phase!(solver, Pardiso.SOLVE_ITERATIVE_REFINE)
            pardiso(solver, KKT_sol, KKT_matrix_pardiso, Vector(b))
        
            # save updated fluid states to CFDModel
            un .+= KKT_sol[1:nf_u]
            pn .+= KKT_sol[nf_u+1:nf_u+m_D]
            fn_b_tilda .+= KKT_sol[end-m_E+1:end]
        
            if verbose
                @show maximum(abs.(R1(un, pn, fn_b_tilda)))
            end
        
        end

        ∂R1∂uk, ∂R1∂xn, ∂c2∂xn, ∂R1∂xn_b, ∂c2∂xn_b = discrete_dynamics_jacobian_wc(
            model, boundary, un, fn_b_tilda, xn, uk
        )

        ∂g∂θ = -vcat(∂R1∂uk * dukdθ + [∂R1∂xn_b*∂xn_b∂θs ∂R1∂xn*∂xn∂θg],
            zeros(length(pn), size(∂xn_b∂θs, 2)+size(∂xn∂θg, 2)),
            [∂c2∂xn_b*∂xn_b∂θs ∂c2∂xn*∂xn∂θg]
        )

        for i in 1:size(∂g∂θ, 2)
            
            # Compute the solutions using the numerical factorization.
            pardiso(solver, KKT_sol, KKT_matrix_pardiso, Vector(∂g∂θ[:, i]))

            dundθ[:, i] .= KKT_sol[1:nf_u]
            dpndθ[:, i] .= KKT_sol[nf_u+1:nf_u+m_D]
            dfn_bdθ[:, i] .= KKT_sol[end-m_E+1:end] .* (model.ρ * model.h_x * model.h_y)

        end

    end

    fn_b .= fn_b_tilda .* (model.ρ * model.h_x * model.h_y)

    # Free the PARDISO data structures.
    set_phase!(solver, Pardiso.RELEASE_ALL)
    pardiso(solver)

end

function discrete_dynamics_jacobian_wc(model::FSIModel, boundary::ImmersedBoundary,
    uk::AbstractVector, fk_b_tilda::AbstractVector, xk::AbstractVector, ukm1::AbstractVector)

    xk_b = boundary_state(boundary, xk)
    _, ∂vecE_∂xk_b, ∂vecE_T∂xk_b = boundary_coupling_with_jacobian(model, boundary, xk_b)
    L = model.FVM_ops.L[1]

    dt = model.dt
    Re = model.Re

    nf_u = size(L, 1)
    n_b = Int(length(xk_b)/4)

    # define kkt system submatrices

    ∂xk_b∂xk = boundary_state_jacobian(boundary, xk)

    ∂R1∂ukm1 = -(1/dt).*(sparse(I, nf_u, nf_u) + (dt/(2*Re)).*L) + 0.5.*N_jacobian(model, ukm1)
    ∂R1∂xk_b = kron(sparse(fk_b_tilda'), sparse(I, nf_u, nf_u))*∂vecE_T∂xk_b
    ∂R1∂xk = ∂R1∂xk_b*∂xk_b∂xk
        
    ∂c2∂xk_b = kron(sparse(uk'), sparse(I, n_b*2, n_b*2))*∂vecE_∂xk_b - kron(sparse([0, 1]'), sparse(I, n_b*2, n_b*2))
    ∂c2∂xk = ∂c2∂xk_b*∂xk_b∂xk

    return ∂R1∂ukm1, ∂R1∂xk, ∂c2∂xk, ∂R1∂xk_b, ∂c2∂xk_b

end