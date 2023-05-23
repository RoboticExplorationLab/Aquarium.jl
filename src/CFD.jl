"""
Basic CFD with no FSI
"""

struct CFDModel
    dt::Float64 # time step
    ρ::Float64 # fluid density (kg/m^3)
    μ::Float64 # fluid dynamic viscosity (Pa*s)
    L_x::Float64 # fluid environment length (m)
    L_y::Float64 # fluid environment height (m)
    ref_L::Float64 # reference length for normalization
    ref_u::Float64 # reference length for normalization
    ne_x::Int64 # number of elements in x 
    ne_y::Int64 # number of elements in y
    h_x::Float64 # spatial step size in x 
    h_y::Float64 # spatial step size in y
    x::Vector{Float64} # x coordinates of pressure cv
    y::Vector{Float64} # y coordinates of pressure cv
    u_west_bc::SVector{2, <:AbstractFloat} # boundary conditions west (m/s)
    u_east_bc::SVector{2, <:AbstractFloat} # boundary conditions east (m/s)
    u_north_bc::SVector{2, <:AbstractFloat} # boundary conditions north (m/s)
    u_south_bc::SVector{2, <:AbstractFloat} # boundary conditions south (m/s)
    outflow::SVector{4, Bool} # outflow boundary conditions
    Re::Float64 # Reynolds number
    FVM_ops::FVM_CDS_2D # FVM discretized ops
    normalize::Bool # are parameters normalized
end

function CFDModel(dt::AbstractFloat=0.01, ρ::AbstractFloat=997.0, μ::AbstractFloat=8.9e-4,
    L_x::AbstractFloat=1.0, L_y::AbstractFloat=1.0, ref_L::AbstractFloat=1.0,
    ref_u::AbstractFloat=0.0, ne_x::Int=20, ne_y::Int=20,
    u_west_bc::SVector{2, <:AbstractFloat}=SA[0.0, 0.0],
    u_east_bc::SVector{2, <:AbstractFloat}=SA[0.0, 0.0],
    u_north_bc::SVector{2, <:AbstractFloat}=SA[0.0, 0.0],
    u_south_bc::SVector{2, <:AbstractFloat}=SA[0.0, 0.0],
    outflow::SVector{4, Bool}=SA[false, false, false, false];
    normalize=true)
    
    # combine boundary conditions
    bc_vel = [u_west_bc, u_east_bc, u_north_bc, u_south_bc]

    # calculate spatial stepsizes
    h_x = L_x/ne_x
    h_y = L_y/ne_y
    
    # calculate Reynolds #
    Re = ref_u*ref_L/(μ/ρ)

    # FVM discretization
    FVM_ops = FVM_CDS_2D(SA[ne_x, ne_y], SA[h_x, h_y], bc_vel, outflow)

    # define positions of pressure cv
    x = LinRange(h_x/2, L_x-h_x/2, ne_x)
    y = LinRange(h_y/2, L_y-h_y/2, ne_y)

    # create CFDModel
    model = CFDModel(dt, ρ, μ, L_x, L_y, ref_L, ref_u, ne_x, ne_y, h_x, h_y,
        x, y, u_west_bc, u_east_bc, u_north_bc, u_south_bc, outflow, Re, 
        FVM_ops, normalize)
    
    # normalize
    if normalize
        model = normalize(model)
    end

    return model
end

function normalize(model::CFDModel)

    # extract reference length and velocity value
    ref_L = model.ref_L
    ref_u = model.ref_u

    # define normalized cavity dimensions
    L_x = model.L_x / ref_L
    L_y = model.L_y / ref_L

    h_x = L_x / model.ne_x
    h_y = L_y / model.ne_y

    # define normalized boundary conditions
    u_west_bc = model.u_west_bc ./ ref_u
    u_east_bc = model.u_east_bc ./ ref_u
    u_north_bc = model.u_north_bc ./ ref_u
    u_south_bc = model.u_south_bc ./ ref_u

    # combine boundary conditions
    bc_vel = [u_west_bc, u_east_bc, u_north_bc, u_south_bc]

    # FVM discretization
    FVM_ops = FVM_CDS_2D(SA[model.ne_x, model.ne_y], SA[h_x, h_y],
        bc_vel, model.outflow)

    # define normalized positions of pressure cv
    x = LinRange(h_x/2, L_x-h_x/2, model.ne_x)
    y = LinRange(h_y/2, L_y-h_y/2, model.ne_y)

    # define normalized time properties
    dt = model.dt / (ref_L/ref_u)

    # set normalize to true
    normalize = true

    # make normalized CFDModel
    nmodel = CFDModel(dt, model.ρ, model.μ, L_x, L_y, ref_L, ref_u,
        model.ne_x, model.ne_y, h_x, h_y, x, y, u_west_bc, u_east_bc,
        u_north_bc, u_south_bc, model.outflow, model.Re, FVM_ops, normalize)
    
    return nmodel
end

function unnormalize(nmodel::CFDModel)

    # extract reference length and velocity value
    ref_L = nmodel.ref_L
    ref_u = nmodel.ref_u

    # define normalized cavity dimensions
    L_x = nmodel.L_x * ref_L
    L_y = nmodel.L_y * ref_L

    h_x = L_x / nmodel.ne_x
    h_y = L_y / nmodel.ne_y

    # define normalized boundary conditions
    u_west_bc = nmodel.u_west_bc .* ref_u
    u_east_bc = nmodel.u_east_bc .* ref_u
    u_north_bc = nmodel.u_north_bc .* ref_u
    u_south_bc = nmodel.u_south_bc .* ref_u

    # combine boundary conditions
    bc_vel = [u_west_bc, u_east_bc, u_north_bc, u_south_bc]

    # FVM discretization
    FVM_ops = FVM_CDS_2D(SA[nmodel.ne_x, nmodel.ne_y], SA[h_x, h_y],
        bc_vel, nmodel.outflow)

    # define normalized positions of pressure cv
    x = LinRange(h_x/2, L_x-h_x/2, nmodel.ne_x)
    y = LinRange(h_y/2, L_y-h_y/2, nmodel.ne_y)

    # define normalized time properties
    dt = nmodel.dt * (ref_L/ref_u)

    # set normalize to true
    normalize = false

    # make normalized CFDModel
    nmodel = CFDModel(dt, nmodel.ρ, nmodel.μ, L_x, L_y, ref_L, ref_u,
        nmodel.ne_x, nmodel.ne_y, h_x, h_y, x, y, u_west_bc, u_east_bc,
        u_north_bc, u_south_bc, nmodel.outflow, nmodel.Re, FVM_ops, normalize)
    
    return nmodel
end

function initialize(model::CFDModel)
    # extract bc
    u_west_bc = model.u_west_bc
    u_east_bc = model.u_east_bc
    u_north_bc = model.u_north_bc
    u_south_bc = model.u_south_bc

    # extract outflow booleans
    outflow_w, outflow_e, outflow_n, outflow_s = model.outflow

    # extract element #
    ne_x = model.ne_x
    ne_y = model.ne_y

    # number of cv
    nf_u = (ne_x-1)*ne_y
    nf_v = ne_x*(ne_y-1)
    nf_p = ne_x*ne_y

    # initialize velocity and pressure states
    Uk = zeros(nf_u + nf_v)
    pk = zeros(nf_p)

    if u_west_bc == u_east_bc
        Uk[1:nf_u] .= u_west_bc[1]
        Uk[nf_u+1:end] .= u_west_bc[2]
    elseif u_north_bc == u_south_bc
        Uk[1:nf_u] .= u_north_bc[1]
        Uk[nf_u+1:end] .= u_north_bc[2]
    end

    if outflow_e
        Uk[1:nf_u] .= u_west_bc[1]
        Uk[nf_u+1:end] .= u_west_bc[2]
    elseif outflow_w
        Uk[1:nf_u] .= u_east_bc[1]
        Uk[nf_u+1:end] .= u_east_bc[2]
    elseif outflow_n
        Uk[1:nf_u] .= u_south_bc[1]
        Uk[nf_u+1:end] .= u_south_bc[2]
    elseif outflow_s
        Uk[1:nf_u] .= u_north_bc[1]
        Uk[nf_u+1:end] .= u_north_bc[2]
    end

    return Uk, pk

end

function simulate(model::CFDModel, U::AbstractVector, p::AbstractVector;
    t=0.0, tf=5.0, max_iter=10, λ=1e-6, tol=1e-6, iter_refine=false,
    alg=:pardiso, verbose=false)

    Uk = deepcopy(U)
    pk = deepcopy(p)
    
    # build time history vectors
    if model.normalize
        dt = model.dt * (model.ref_L / model.ref_u) 
    else
        dt = model.dt 
    end

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

    N = Int((tf-t)/dt + 1)
    u_hist = [U for _ in 1:N]
    p_hist = [p for _ in 1:N]
    T_hist = t:dt:tf

    @showprogress "Simulating..." for ind in 2:N

        t = T_hist[ind]

        if verbose
            println("\n")
            @show t
        end

        # integrate over N-S
        discrete_dynamics!(model, Uk, pk, Uk, pk;
            max_iter=max_iter, λ=λ, tol=tol,
            iter_refine=iter_refine, alg=alg,
            verbose=verbose, solver=solver)

        # populate time histories
        u_hist[ind] = deepcopy(Uk)
        p_hist[ind] = deepcopy(pk)

    end

    return T_hist, u_hist, p_hist
end

function simulate!(model::CFDModel, Un::AbstractVector, pn::AbstractVector,
    U::AbstractVector, p::AbstractVector; t=0.0, tf=5.0, max_iter=10, λ=1e-6,
    tol=1e-6, iter_refine=false, alg=:pardiso, verbose=false)

    Uk = deepcopy(U)
    pk = deepcopy(p)
    
    if model.normalize
        dt = model.dt * (model.ref_L / model.ref_u) 
    else
        dt = model.dt
    end

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

    N = Int((tf-t)/dt + 1)
    T_hist = t:dt:tf

    @showprogress "Simulating..." for ind in 2:N

        t = T_hist[ind]

        if verbose
            println("\n")
            @show t
        end

        # integrate over N-S
        discrete_dynamics!(model, Uk, pk, Uk, pk;
            max_iter=max_iter, λ=λ, tol=tol, 
            iter_refine=iter_refine, alg=alg,
            verbose=verbose, solver=solver)

    end

    Un .= Uk
    pn .= pk

end

function discrete_dynamics(model::CFDModel, U::AbstractVector,
    p::AbstractVector; max_iter=10, λ=1e-6, tol=1e-6,
    iter_refine=false, alg=:pardiso, verbose=false,
    solver=nothing)

    Un = deepcopy(U)
    pn = deepcopy(p)

    discrete_dynamics!(model, Un, pn, U, p; max_iter=max_iter, λ=λ,
        tol=tol, iter_refine=iter_refine, alg=alg, verbose=verbose,
        solver=solver
    )

    return Un, pn

end

function discrete_dynamics!(model::CFDModel, Un::AbstractVector, pn::AbstractVector,
    Uk::AbstractVector, pk::AbstractVector; max_iter=10, λ=1e-6, tol=1e-6,
    iter_refine=false, alg=:pardiso, verbose=false, solver=nothing)

    Ukp1 = deepcopy(Uk)
    pkp1 = deepcopy(pk)
    
    # extract operators
    G = model.FVM_ops.G
    L = model.FVM_ops.L[1]
    L_bc = model.FVM_ops.L[3]
    D_bc = model.FVM_ops.D[2]
    
    dt = model.dt
    Re = model.Re
    
    nf_u = size(L, 1)
    m_D = size(G, 2)
    
    # define kkt system submatrices
    A = (1/dt).*(sparse(I, nf_u, nf_u) - (dt/(2*Re)).*L)
    r = (1/dt).*(sparse(I, nf_u, nf_u) + (dt/(2*Re)).*L)*Uk - 0.5.*N(model, Uk)
    bc1 = (1/Re).*L_bc
    bc2 = -D_bc
    
    c1 = G'

    # solve using Newton's Method
    num_iter = 0
    dLdU = A*Ukp1 + 0.5.*N(model, Ukp1) - (r + bc1) + G*pkp1

    if verbose
        @show maximum(abs.(dLdU))
    end

    if alg == :qr

        while maximum(abs.(dLdU)) > tol && num_iter <= max_iter
        
            num_iter = num_iter + 1
            
            # form KKT matrix
            ∇2f = A + 0.5.*dNdU(model, Ukp1)
            c1 = G'
        
            KKT_matrix = hvcat((2, 2), ∇2f, c1', c1, spzeros(m_D, m_D))
            b = vcat(-dLdU, (-G'*Ukp1 - bc2))
        
            # solve using QR
            F = qr(KKT_matrix)
            KKT_sol = F\Vector(b)
        
            # save updated fluid states to CFDModel
            Ukp1 += KKT_sol[1:length(Ukp1)]
            pkp1 += KKT_sol[length(Ukp1)+1:end]
    
            # calculate gradient of Lagrangian
            dLdU = A*Ukp1 + 0.5.*N(model, Ukp1) - (r + bc1) + G*pkp1

            if verbose
                @show maximum(abs.(dLdU))
            end
    
        end
    
    elseif alg == :lu

        KKT_prev = 1.0 .* sparse(I, 5, 5)
        F = lu(KKT_prev)

        while maximum(abs.(dLdU)) > tol && num_iter <= max_iter
            
            num_iter = num_iter + 1
            
            # form KKT matrix
            ∇2f = A + 0.5.*dNdU(model, Ukp1)
            c1 = G'
        
            KKT_matrix_reg = hvcat((2, 2), ∇2f, c1', c1, -λ.*sparse(I, m_D, m_D))
            KKT_matrix = hvcat((2, 2), ∇2f, c1', c1, spzeros(m_D, m_D))
            b = vcat(-dLdU, (-G'*Ukp1 - bc2))

            # solve using LU
            if KKT_prev.colptr == KKT_matrix_reg.colptr &&
                KKT_prev.rowval == KKT_matrix_reg.rowval
                lu!(F, KKT_matrix_reg)
            else
                F = lu(KKT_matrix_reg)
            end

            KKT_sol = F\Vector(b)
            
            # iterative refinement
            if iter_refine

                # calculate residual
                ϵ = b - KKT_matrix*KKT_sol

                if verbose
                    @show maximum(abs.(ϵ))
                end

                while maximum(ϵ) >= 1e-6

                    u = F\Vector(ϵ)
                    KKT_sol += u

                    ϵ = b - KKT_matrix*KKT_sol

                    if verbose
                        @show maximum(abs.(ϵ))
                    end

                end

            end
        
            # save updated fluid states to CFDModel
            Ukp1 += KKT_sol[1:length(Ukp1)]
            pkp1 += KKT_sol[length(Ukp1)+1:end]

            # calculate gradient of Lagrangian
            dLdU = A*Ukp1 + 0.5.*N(model, Ukp1) - (r + bc1) + G*pkp1

            # define previous iter KKT
            KKT_prev = deepcopy(KKT_matrix_reg)
            
            if verbose
                @show maximum(abs.(dLdU))
            end

        end
    
    elseif alg == :pardiso

        KKT_sol = zeros(size(A, 2) + size(c1, 1))

        while num_iter == 0 || maximum(abs.(dLdU)) > tol && max_iter <= max_iter
            
            num_iter = num_iter + 1
            
            # form KKT matrix
            ∇2f = A + 0.5.*dNdU(model, Ukp1)

            KKT_matrix = hvcat((2, 2), ∇2f, c1', c1, -λ.*sparse(I, m_D, m_D))
            b = vcat(-dLdU, (-G'*Ukp1 - bc2))

            KKT_matrix_pardiso = get_matrix(solver, KKT_matrix, :N)
        
            # Analyze the matrix and compute a symbolic factorization.
            set_phase!(solver, Pardiso.ANALYSIS)
            pardiso(solver, KKT_matrix_pardiso, Vector(b))
        
            # Compute the solutions X using the symbolic factorization.
            set_phase!(solver, Pardiso.NUM_FACT_SOLVE_REFINE)
            pardiso(solver, KKT_sol, KKT_matrix_pardiso, Vector(b))
        
            # Free the PARDISO data structures.
            set_phase!(solver, Pardiso.RELEASE_ALL)
            pardiso(solver)
                
            # save updated fluid states to CFDModel
            Ukp1 += KKT_sol[1:length(Ukp1)]
            pkp1 += KKT_sol[length(Ukp1)+1:end]
        
            # calculate gradient of Lagrangian
            dLdU = A*Ukp1 + 0.5.*N(model, Ukp1) - (r + bc1) + G*pkp1
        
            if verbose
                @show maximum(abs.(dLdU))
            end
        
        end

    end

    Un .= Ukp1
    pn .= pkp1

end

function dNdU(model::CFDModel, U::AbstractVector)

    m1, m2, m3, m4, m5, m6 = model.FVM_ops.N[1]
    m1_bc, m2_bc, m3_bc, m4_bc, m5_bc, m6_bc = model.FVM_ops.N[2]
    
    N1 = m1*U + m1_bc
    N2 = m2*U + m2_bc
    N3 = m3*U + m3_bc
    N4 = m4*U + m4_bc
    N5 = m5*U + m5_bc
    N6 = m6*U + m6_bc
        
    dNdU = (spdiagm(N1)*m1.*2 - spdiagm(N2)*m2.*2) + 
        (spdiagm(N3)*m4 + spdiagm(N4)*m3) -
        (spdiagm(N5)*m6 + spdiagm(N6)*m5)
    return dNdU

end

function N(model::CFDModel, U::AbstractVector)

    m1, m2, m3, m4, m5, m6 = model.FVM_ops.N[1]
    m1_bc, m2_bc, m3_bc, m4_bc, m5_bc, m6_bc = model.FVM_ops.N[2]
    
    N1 = m1*U + m1_bc
    N2 = m2*U + m2_bc
    N3 = m3*U + m3_bc
    N4 = m4*U + m4_bc
    N5 = m5*U + m5_bc
    N6 = m6*U + m6_bc
    
    convective = (N1.*N1 - N2.*N2) + (N3.*N4 - N5.*N6)
    return convective
    
end