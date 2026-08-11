"""
The linear_solver module provides functions for creating and solving linear systems.
It follows the naming convention:

- A: The linear-system matrix
- b: The right-hand side vector
- x: The solution vector to update in place
"""

#############################################################################################
## Solver creation
#############################################################################################

function create_solver(A, x, solver_type;
    n_pardiso_threads::Int=Sys.CPU_THREADS,
    gmres_memory::Int=20
)

    if solver_type == :pardiso
        if !PARDISO_LOADED[]
            error("Pardiso is not available. Please use a different solver_type or install Pardiso on Linux/Windows and load it with 'using Pardiso'.")
        end
        solver = create_pardiso_solver()
        pardiso_set_nprocs!(solver, n_pardiso_threads)
        # Use symmetric indefinite matrix type for saddle point systems
        pardiso_set_matrixtype!(solver, Val(:REAL_NONSYM))
        pardiso_init!(solver)
        pardiso_fix_iparm!(solver)

        pardiso_set_iparm!(solver, 5, 0)
        pardiso_set_iparm!(solver, 8, 5)
        pardiso_set_iparm!(solver, 10, 13)
        pardiso_set_iparm!(solver, 11, 0)
        pardiso_set_iparm!(solver, 13, 0)

        return solver
    elseif solver_type == :mumps
        # Create and initialize MUMPS solver for unsymmetric matrices
        MPI.Init()
        solver = MUMPS.Mumps{Float64}(MUMPS.mumps_unsymmetric, MUMPS.default_icntl, MUMPS.default_cntl64)
        return solver
    elseif solver_type == :gmres
        # Create GMRES workspace
        return Krylov.GmresWorkspace(A, x; memory=gmres_memory)
    else
        return nothing
    end
end

#############################################################################################
## Helper functions for creating AMG solvers
#############################################################################################

function get_amg_smoother(smoother_type::Symbol)
    if smoother_type == :forward_gs
        return AlgebraicMultigrid.GaussSeidel(AlgebraicMultigrid.ForwardSweep())
    elseif smoother_type == :backward_gs
        return AlgebraicMultigrid.GaussSeidel(AlgebraicMultigrid.BackwardSweep())
    elseif smoother_type == :symmetric_gs
        return AlgebraicMultigrid.GaussSeidel(AlgebraicMultigrid.SymmetricSweep())
    else
        error("Unknown smoother_type: $smoother_type. Valid options: :forward_gs, :backward_gs, :symmetric_gs")
    end
end

#############################################################################################
## Preconditioner calculation
#############################################################################################

function calculate_preconditioner(A, x, preconditioner_type;
    preconditioner_solver=nothing,
    ilu_drop_tolerance::Float64=0.0,
    amg_smoother_type::Symbol=:forward_gs,
    verbose=false,
    schur_dimension::Int=0,
    ilu_fallback_regularization::Float64=1e-4
)

    if verbose
        print("Computing $preconditioner_type preconditioner...")
    end

    preconditioner = I

    if preconditioner_type == :ilu0
        # Compute ILU0 factorization (no dropping)
        ilu_fact = ilu0(A)

        # Wrap in ILU0Preconditioner for Krylov.jl compatibility
        preconditioner = ILU0Preconditioner(ilu_fact)

    elseif preconditioner_type == :ilu
        # Compute ILU factorization with drop tolerance
        ilu_fact = ilu(A, τ=ilu_drop_tolerance)

        u_diag_min = minimum(abs, diag(ilu_fact.U))
        if u_diag_min < sqrt(eps(Float64))
            @warn "ILU factorization near-singular, retrying with diagonal regularization" u_diag_min ilu_drop_tolerance
            n = size(A, 1)
            @inbounds for i in 1:n
                A[i, i] += ilu_fallback_regularization
            end
            ilu_fact = ilu(A, τ=ilu_drop_tolerance)
            @inbounds for i in 1:n
                A[i, i] -= ilu_fallback_regularization
            end
        end
        preconditioner = ILUPreconditioner(ilu_fact)

    elseif preconditioner_type == :amg
        # Create AMG hierarchy for full KKT matrix (not block-decomposed)
        smoother = get_amg_smoother(amg_smoother_type)
        kkt_ml = AlgebraicMultigrid.smoothed_aggregation(A,
            max_levels=8,
            max_coarse=150,
            strength=AlgebraicMultigrid.SymmetricStrength(0.25),
            postsmoother=smoother,
            presmoother=smoother)
        kkt_precond = AlgebraicMultigrid.aspreconditioner(kkt_ml)
        preconditioner = AMGPreconditioner(kkt_precond)

    elseif preconditioner_type == :approx_schur_ilu ||
        preconditioner_type == :approx_schur_ilu0 ||
        preconditioner_type == :approx_schur_full_amg ||
        preconditioner_type == :approx_schur_partial_amg
        # Create approximate Schur complement preconditioner
        # For saddle point system [A B; C D][u; λ] = [f; g]
        # Preconditioner approximates inv([A 0; 0 S]) where S ≈ D - C*A^-1*B

        n_total = size(A, 1)
        n_primal = n_total - schur_dimension
        n_dual = schur_dimension

        # Extract blocks from KKT matrix (save original matrix before overwriting A)
        A_full = A
        A_block = A_full[1:n_primal, 1:n_primal]
        B = A_full[1:n_primal, n_primal+1:n_total]
        C = A_full[n_primal+1:n_total, 1:n_primal]
        D = A_full[n_primal+1:n_total, n_primal+1:n_total]
        
        # Create preconditioner for A block (primal/momentum equations)
        if preconditioner_type == :approx_schur_ilu0
            A_ilu = ilu0(A_block)
            A_inner_preconditioner = ILU0Preconditioner(A_ilu)
        elseif preconditioner_type == :approx_schur_ilu ||
               preconditioner_type == :approx_schur_partial_amg
            A_ilu = ilu(A_block, τ=ilu_drop_tolerance)
            u_diag_min = minimum(abs, diag(A_ilu.U))
            if u_diag_min < sqrt(eps(Float64))
                @warn "A-block ILU near-singular, retrying with diagonal regularization" u_diag_min
                @inbounds for i in 1:n_primal
                    A_block[i, i] += ilu_fallback_regularization
                end
                A_ilu = ilu(A_block, τ=ilu_drop_tolerance)
                @inbounds for i in 1:n_primal
                    A_block[i, i] -= ilu_fallback_regularization
                end
            end
            A_inner_preconditioner = ILUPreconditioner(A_ilu)
        elseif preconditioner_type == :approx_schur_full_amg
            # Create AMG hierarchy for A block using Smoothed Aggregation method
            smoother = get_amg_smoother(amg_smoother_type)
            A_ml = AlgebraicMultigrid.smoothed_aggregation(A_block, max_levels=8,
                max_coarse=150,
                strength=AlgebraicMultigrid.SymmetricStrength(0.25),
                postsmoother=smoother,
                presmoother=smoother)
            A_precond = AlgebraicMultigrid.aspreconditioner(A_ml)
            A_inner_preconditioner = AMGPreconditioner(A_precond)
        end

        # Approximate Schur complement: S ≈ D - C * diag(A)^-1 * B
        # Using diagonal approximation is much cheaper than D - C * A^-1 * B
        A_diag_inv = 1.0 ./ diag(A_block)
        S_approx = D - C * (A_diag_inv .* B)

        # Create preconditioner for approximate Schur complement
        if preconditioner_type == :approx_schur_ilu0
            S_ilu = ilu0(S_approx)
            S_inner_preconditioner = ILU0Preconditioner(S_ilu)
        elseif preconditioner_type == :approx_schur_ilu
            S_ilu = ilu(S_approx, τ=ilu_drop_tolerance)
            u_diag_min = minimum(abs, diag(S_ilu.U))
            if u_diag_min < sqrt(eps(Float64))
                @warn "S-block ILU near-singular, retrying with diagonal regularization" u_diag_min
                @inbounds for i in 1:n_dual
                    S_approx[i, i] += ilu_fallback_regularization
                end
                S_ilu = ilu(S_approx, τ=ilu_drop_tolerance)
                @inbounds for i in 1:n_dual
                    S_approx[i, i] -= ilu_fallback_regularization
                end
            end
            S_inner_preconditioner = ILUPreconditioner(S_ilu)
        elseif preconditioner_type in (:approx_schur_partial_amg, :approx_schur_full_amg)
            # Create AMG hierarchy for S block (approximate Schur complement)
            smoother = get_amg_smoother(amg_smoother_type)
            S_ml = AlgebraicMultigrid.smoothed_aggregation(S_approx, max_levels=8,
                max_coarse=150,
                strength=AlgebraicMultigrid.SymmetricStrength(0.25),
                postsmoother=smoother,
                presmoother=smoother)
            S_precond = AlgebraicMultigrid.aspreconditioner(S_ml)
            S_inner_preconditioner = AMGPreconditioner(S_precond)
        end

        # Create block lower triangular preconditioner
        preconditioner = BlockTriangularPreconditioner(
            A_inner_preconditioner,
            S_inner_preconditioner,
            C,
            n_primal,
            n_dual
        )

    elseif preconditioner_type == :pardiso
        
        # GMRES with Pardiso preconditioner
        if !PARDISO_LOADED[]
            error("Pardiso is not available. Please use a different preconditioner_type or install Pardiso on Linux/Windows.")
        end

        # Perform Pardiso factorization once
        pardiso_set_phase!(preconditioner_solver, Val(:ANALYSIS_NUM_FACT))
        pardiso_solve!(preconditioner_solver, A, x)

        # Create preconditioner struct using the extension's type
        PardisoExt = Base.get_extension(AquariumClosed, :PardisoExt)
        preconditioner = PardisoExt.PardisoPreconditioner(preconditioner_solver, A)

    end

    if verbose
        println(" Finished!")
    end

    return preconditioner
end

#############################################################################################
## Matrix scaling
#############################################################################################

function calculate_ruiz_scale!(A; max_iterations::Int=10, tolerance::Float64=1e-3)

    n = size(A, 1)
    m = size(A, 2)

    # Initialize cumulative scaling factors to 1
    left_scale_cumulative = ones(n)
    right_scale_cumulative = ones(n)

    # Get sparse structure for efficient traversal
    rows = rowvals(A)
    vals = nonzeros(A)
    colptr = A.colptr  # Column pointers for direct access

    for _ in 1:max_iterations
        # Compute row infinity norms using chunked task-based parallelism
        (row_norms,) = taccumulate_max(1:n) do chunk
            local_row_norms = zeros(n)
            @inbounds for col in chunk
                for j in colptr[col]:(colptr[col+1]-1)
                    row = rows[j]
                    val = abs(vals[j])
                    local_row_norms[row] = max(local_row_norms[row], val)
                end
            end
            (local_row_norms,)
        end

        left_scale = 1.0 ./ sqrt.(row_norms)

        # Apply left scaling to matrix in-place
        tforeach(1:n) do col
            @inbounds for j in colptr[col]:(colptr[col+1]-1)
                row = rows[j]
                vals[j] *= left_scale[row]
            end
        end
        
        # Update cumulative left scaling
        left_scale_cumulative .*= left_scale

        # Compute column infinity norms (matrix is already left-scaled)
        col_norms = zeros(n)
        tforeach(1:n) do col
            local_max = 0.0
            @inbounds for j in colptr[col]:(colptr[col+1]-1)
                local_max = max(local_max, abs(vals[j]))
            end
            col_norms[col] = local_max
        end
        right_scale = 1.0 ./ sqrt.(col_norms)

        # Apply right scaling to matrix in-place
        tforeach(1:n) do col
            col_scale = right_scale[col]
            @inbounds for j in colptr[col]:(colptr[col+1]-1)
                vals[j] *= col_scale
            end
        end
        
        # Update cumulative right scaling
        right_scale_cumulative .*= right_scale

        # Check convergence
        left_change = maximum(abs.(left_scale .- 1.0) ./ (1.0 .+ eps()))
        right_change = maximum(abs.(right_scale .- 1.0) ./ (1.0 .+ eps()))

        if left_change < tolerance && right_change < tolerance
            break
        end
    end

    return left_scale_cumulative, right_scale_cumulative

end

function scale_linear_system!(A, b; scaling_type::Symbol=:ruiz, verbose::Bool=false)

    if verbose
        print("Scaling matrix and right-hand side using $scaling_type...")
    end

    if scaling_type == :ruiz

        # Calculate scaling factors and apply to matrix in-place
        left_scale, right_scale = calculate_ruiz_scale!(A)

        # Scale the right-hand side in-place: b = Diagonal(left_scale) * b
        b .*= left_scale

    else

        left_scale = ones(length(b))
        right_scale = ones(length(b))

    end

    if verbose
        println(" Finished!")
    end

    return left_scale, right_scale

end

function scale_linear_system!(A, b, left_scale, right_scale; verbose::Bool=false)

    if verbose
        print("Scaling matrix and right-hand side using provided factors...")
    end

    # Apply provided scaling factors to matrix and right-hand side in-place
    n = size(A, 1)

    # Scale matrix: kkt_matrix = Diagonal(left_scale) * kkt_matrix * Diagonal(right_scale)
    rows = rowvals(A)
    vals = nonzeros(A)
    colptr = A.colptr  # Column pointers for direct access

    tforeach(1:n) do col
        col_scale = right_scale[col]
        @inbounds for j in colptr[col]:(colptr[col+1]-1)
            row = rows[j]
            vals[j] *= left_scale[row] * col_scale
        end
    end

    # Scale right-hand side: b = Diagonal(left_scale) * b
    b .*= left_scale

    if verbose
        println(" Finished!")
    end

    return nothing
end

function scale_linear_system_matrix!(A::AbstractSparseArray, left_scale, right_scale; verbose::Bool=false)

    if verbose
        print("Scaling matrix and right-hand side using provided factors...")
    end

    # Apply provided scaling factors to matrix and right-hand side in-place
    n = size(A, 1)

    # Scale matrix: kkt_matrix = Diagonal(left_scale) * kkt_matrix * Diagonal(right_scale)
    rows = rowvals(A)
    vals = nonzeros(A)
    colptr = A.colptr  # Column pointers for direct access

    tforeach(1:n) do col
        col_scale = right_scale[col]
        @inbounds for j in colptr[col]:(colptr[col+1]-1)
            row = rows[j]
            vals[j] *= left_scale[row] * col_scale
        end
    end

    if verbose
        println(" Finished!")
    end

    return nothing
end

function scale_rhs_matrix!(B::AbstractMatrix, left_scale; verbose::Bool=false)

    if verbose
        print("Scaling matrix using provided factors...")
    end

    for j in axes(B, 2)
        B[:, j] .*= left_scale
    end

    if verbose
        println(" Finished!")
    end

end

function scale_vector!(b::AbstractVector, left_scale; verbose::Bool=false)

    if verbose
        print("Scaling vector using provided factors...")
    end

    # Apply provided scaling factors to vector in-place
    b .*= left_scale

    if verbose
        println(" Finished!")
    end

    return nothing
end

#############################################################################################
## Matrix pivoting
#############################################################################################

function pivot_linear_system!(A, b; pivot_type::Symbol=:rcm, verbose::Bool=false)

    if verbose
        print("Pivoting matrix and right-hand side using $pivot_type...")
    end

    if pivot_type == :rcm
        # Compute RCM permutation
        perm = symrcm(A)

        # Compute inverse permutation for unpermuting solution later
        inv_perm = invperm(perm)

        # Permute matrix: P * A * P'
        A .= A[perm, perm]

        # Permute right-hand side: P * b
        b .= b[perm]

    elseif pivot_type == :amd

        # Compute AMD permutation
        perm = amd(A)

        # Compute inverse permutation for unpermuting solution later
        inv_perm = invperm(perm)

        # Permute matrix: P * A * P'
        A .= A[perm, perm]

        # Permute right-hand side: P * b
        b .= b[perm]

    elseif pivot_type == :colamd

        # Compute COLAMD permutation
        perm = colamd(A)

        # Compute inverse permutation for unpermuting solution later
        inv_perm = invperm(perm)

        # Permute matrix: P * A * P'
        A .= A[perm, perm]

        # Permute right-hand side: P * b
        b .= b[perm]

    elseif pivot_type == :metis

        # Compute Metis nested dissection permutation
        # Metis requires a symmetric sparsity pattern, so we use the pattern of A + A'
        # This ensures all edges are represented symmetrically
        symmetric_pattern = A + A'
        perm, inv_perm = Metis.permutation(symmetric_pattern)

        # Permute matrix: P * A * P'
        A .= A[perm, perm]

        # Permute right-hand side: P * b
        b .= b[perm]

    else
        # No pivoting; return identity permutation
        n = length(b)
        perm = collect(1:n)
        inv_perm = collect(1:n)
    end

    if verbose
        println(" Finished!")
    end

    return perm, inv_perm
end

function pivot_linear_system!(A, b, perm; verbose::Bool=false)
    # Apply provided permutation to matrix and residual in-place

    if verbose
        print("Permuting matrix and right-hand side using provided permutation...")
    end

    # Permute matrix: P * A * P'
    A .= A[perm, perm]
    
    # Permute right-hand side: P * b
    b .= b[perm]

    if verbose
        println(" Finished!")
    end
    
    return nothing
end

function pivot_linear_system_matrix!(A::AbstractSparseArray, perm; verbose::Bool=false)
    # Apply provided permutation to matrix and residual in-place

    if verbose
        print("Permuting matrix and right-hand side using provided permutation...")
    end

    # Permute matrix: P * A * P'
    A .= A[perm, perm]

    if verbose
        println(" Finished!")
    end
    
    return nothing
end

function pivot_rhs_matrix!(B::AbstractMatrix, perm; verbose::Bool=false)

    # Apply provided permutation to matrix in-place
    if verbose
        print("Permuting matrix using provided permutation...")
    end

    for j in axes(B, 2)
        B[:, j] .= B[perm, j]
    end

    if verbose
        println(" Finished!")
    end

    return nothing
end

function pivot_vector!(b::AbstractVector, perm; verbose::Bool=false)
    # Apply provided permutation to vector in-place

    if verbose
        print("Permuting vector using provided permutation...")
    end
    
    # Permute vector: P * b
    b .= b[perm]

    if verbose
        println(" Finished!")
    end

    return nothing
end

function apply_regularization!(A; regularization_indices::Vector{Int}=Int[],
                                   regularization_value::Float64=0.0,
                                   verbose::Bool=false)

    if verbose
        print("Applying regularization...")
    end

    if !isempty(regularization_indices) && regularization_value != 0.0
        # Simple loop - fast and handles diagonal elements that are guaranteed to exist
        @inbounds for idx in regularization_indices
            A[idx, idx] -= regularization_value
        end
    end

    if verbose
        println(" Finished!")
    end

    return nothing
end

#############################################################################################
## Linear solve
#############################################################################################

function linear_solve!(x, A, b, solver, solver_type;
    preconditioner=I,
    gmres_tolerance::Float64=1e-6,
    gmres_max_iterations::Int=1000,
    right_scale=ones(length(x)),
    inverse_permutation=1:length(x),
    verbose::Bool=false,
    reuse_factorization::Bool=false
)

    if verbose
        print("Solving linear system using $solver_type...")
    end

    if solver_type == :pardiso

        if !reuse_factorization
            # Solve using factorization (precomputed by calculate_preconditioner)
            pardiso_set_phase!(solver, Val(:ANALYSIS_NUM_FACT))
            pardiso_solve!(solver, A, b)

        end

        pardiso_set_phase!(solver, Val(:SOLVE_ITERATIVE_REFINE))
        pardiso_solve!(solver, x, A, b)

    elseif solver_type == :mumps

        # Update matrix and solve using MUMPS (already initialized)
        associate_matrix!(solver, A)
        if !reuse_factorization
            factorize!(solver)
        end
        associate_rhs!(solver, b)
        MUMPS.solve!(solver)
        x .= get_solution(solver)

    elseif solver_type == :gmres

        # Use GMRES with preconditioner
        try
            Krylov.gmres!(solver, A, b;
                M=preconditioner, atol=gmres_tolerance, itmax=gmres_max_iterations)
            x .= solver.x
        catch e
            if e isa LinearAlgebra.LAPACKException || e isa LinearAlgebra.SingularException
                @warn "GMRES failed with preconditioner, retrying unpreconditioned" exception=e
                Krylov.gmres!(solver, A, b;
                    M=I, atol=gmres_tolerance, itmax=gmres_max_iterations)
                x .= solver.x
            else
                rethrow(e)
            end
        end

        # Check convergence and warn if not achieved
        if !solver.stats.solved
            @warn "GMRES did not converge to tolerance" status=solver.stats.status iterations=solver.stats.niter tolerance=gmres_tolerance
        end

    elseif solver_type == :backslash

        x .= A \ b

    end

    # Unpermute and unscale the solution
    x .= (x[inverse_permutation]) .* right_scale

    if verbose
        println(" Finished!")

        if solver_type == :gmres
            println("GMRES iterations: $(solver.stats.niter)")
            if solver.stats.solved
                println("GMRES converged successfully")
            else
                println("GMRES convergence status: $(solver.stats.status)")
            end
        end

    end

    return x

end

#############################################################################################
## Block linear solve (multiple RHS)
#############################################################################################

"""
    block_linear_solve!(X, A, B, solver_type; kwargs...)

Solve AX = B where B is a matrix with multiple right-hand sides.
Uses block GMRES for iterative solvers, which is more efficient than solving each RHS separately.

# Arguments
- `X`: Solution matrix (n × m), modified in-place
- `A`: System matrix (n × n)
- `B`: Right-hand side matrix (n × m)
- `solver_type`: Type of solver (:gmres, :pardiso, :mumps, :backslash)

# Keyword Arguments
- `preconditioner`: Preconditioner for iterative solvers (default: I)
- `gmres_tolerance`: Convergence tolerance for GMRES (default: 1e-6)
- `gmres_max_iterations`: Maximum GMRES iterations (default: 1000)
- `gmres_memory`: Memory parameter for block GMRES (default: 20)
- `right_scale`: Right scaling vector (default: ones)
- `inverse_permutation`: Inverse permutation for solution (default: 1:n)
- `verbose`: Print progress information (default: false)
- `reuse_factorization`: Reuse previous factorization for direct solvers (default: false)
"""
function block_linear_solve!(X, A, B, solver, solver_type;
    preconditioner=I,
    gmres_tolerance::Float64=1e-6,
    gmres_max_iterations::Int=1000,
    gmres_memory::Int=20,
    right_scale=ones(size(X, 1)),
    inverse_permutation=1:size(X, 1),
    verbose::Bool=false,
    reuse_factorization::Bool=false
)
    n_rhs = size(B, 2)

    if n_rhs == 0
        return X
    end

    if verbose
        print("Solving block linear system ($n_rhs RHS) using $solver_type...")
    end

    if solver_type == :gmres
        # Use block GMRES for multiple RHS - much more efficient than individual solves
        local X_temp, stats
        try
            X_temp, stats = Krylov.block_gmres(A, B;
                M=preconditioner, atol=gmres_tolerance, itmax=gmres_max_iterations, memory=gmres_memory)
        catch e
            if e isa LinearAlgebra.LAPACKException || e isa LinearAlgebra.SingularException
                @warn "Block GMRES failed with preconditioner, retrying unpreconditioned" exception=e
                X_temp, stats = Krylov.block_gmres(A, B;
                    M=I, atol=gmres_tolerance, itmax=gmres_max_iterations, memory=gmres_memory)
            else
                rethrow(e)
            end
        end

        # Apply inverse permutation and right scaling to each column
        # Unpermute first, then scale: x_unpermuted = x[inverse_permutation], then x_final = x_unpermuted .* right_scale
        @inbounds for j in 1:n_rhs
            for i in 1:size(X, 1)
                X[i, j] = X_temp[inverse_permutation[i], j] * right_scale[i]
            end
        end

        # Check convergence and warn if not achieved
        if !stats.solved
            @warn "Block GMRES did not converge to tolerance" status=stats.status iterations=stats.niter tolerance=gmres_tolerance n_rhs=n_rhs
        end

        if verbose
            println(" Finished! ($(stats.niter) iterations)")
            if stats.solved
                println("Block GMRES converged successfully")
            else
                println("Block GMRES convergence status: $(stats.status)")
            end
        end

    elseif solver_type == :pardiso
        # For Pardiso, solve each RHS (Pardiso handles multiple RHS internally)
        if !reuse_factorization
            pardiso_set_phase!(solver, Val(:ANALYSIS_NUM_FACT))
            pardiso_solve!(solver, A, B[:, 1])
        end

        pardiso_set_phase!(solver, Val(:SOLVE_ITERATIVE_REFINE))
        for j in 1:n_rhs
            x_col = @view X[:, j]
            pardiso_solve!(solver, x_col, A, B[:, j])
            x_col .= (x_col[inverse_permutation]) .* right_scale
        end

        if verbose
            println(" Finished!")
        end

    elseif solver_type == :mumps
        associate_matrix!(solver, A)
        if !reuse_factorization
            factorize!(solver)
        end

        for j in 1:n_rhs
            associate_rhs!(solver, B[:, j])
            MUMPS.solve!(solver)
            X[:, j] .= (get_solution(solver)[inverse_permutation]) .* right_scale
        end

        if verbose
            println(" Finished!")
        end

    elseif solver_type == :backslash
        X_temp = A \ B
        # Unpermute first, then scale: x_unpermuted = x[inverse_permutation], then x_final = x_unpermuted .* right_scale
        @inbounds for j in 1:n_rhs
            for i in 1:size(X, 1)
                X[i, j] = X_temp[inverse_permutation[i], j] * right_scale[i]
            end
        end

        if verbose
            println(" Finished!")
        end
    end

    return X
end
