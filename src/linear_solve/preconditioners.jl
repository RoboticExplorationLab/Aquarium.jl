# Preconditioner wrapper for ILU0 to support Krylov.jl interface
struct ILU0Preconditioner{T}
    ilu::T
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::ILU0Preconditioner, x::AbstractVector)
    ldiv!(y, P.ilu, x)
    return y
end

function LinearAlgebra.ldiv!(P::ILU0Preconditioner, x::AbstractVector)
    ldiv!(P.ilu, x)
    return x
end

function LinearAlgebra.mul!(y::AbstractVector, P::ILU0Preconditioner, x::AbstractVector)
    ldiv!(y, P.ilu, x)
    return y
end

# Matrix-matrix mul! for block Krylov methods (e.g., block GMRES)
function LinearAlgebra.mul!(Y::AbstractMatrix, P::ILU0Preconditioner, X::AbstractMatrix, α::Number, β::Number)
    if β == 0
        for j in axes(X, 2)
            ldiv!(view(Y, :, j), P.ilu, view(X, :, j))
        end
        if α != 1
            Y .*= α
        end
    else
        temp = similar(view(Y, :, 1))
        for j in axes(X, 2)
            ldiv!(temp, P.ilu, view(X, :, j))
            Y[:, j] .= α .* temp .+ β .* view(Y, :, j)
        end
    end
    return Y
end

# Preconditioner wrapper for ILU to support Krylov.jl interface
struct ILUPreconditioner{T}
    ilu::T
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::ILUPreconditioner, x::AbstractVector)
    ldiv!(y, P.ilu, x)
    return y
end

function LinearAlgebra.ldiv!(P::ILUPreconditioner, x::AbstractVector)
    ldiv!(P.ilu, x)
    return x
end

function LinearAlgebra.mul!(y::AbstractVector, P::ILUPreconditioner, x::AbstractVector)
    ldiv!(y, P.ilu, x)
    return y
end

# Matrix-matrix mul! for block Krylov methods (e.g., block GMRES)
function LinearAlgebra.mul!(Y::AbstractMatrix, P::ILUPreconditioner, X::AbstractMatrix, α::Number, β::Number)
    if β == 0
        for j in axes(X, 2)
            ldiv!(view(Y, :, j), P.ilu, view(X, :, j))
        end
        if α != 1
            Y .*= α
        end
    else
        temp = similar(view(Y, :, 1))
        for j in axes(X, 2)
            ldiv!(temp, P.ilu, view(X, :, j))
            Y[:, j] .= α .* temp .+ β .* view(Y, :, j)
        end
    end
    return Y
end

# AMG Preconditioner wrapper for AlgebraicMultigrid.jl
struct AMGPreconditioner{T}
    precond::T  # Preconditioner from AlgebraicMultigrid.aspreconditioner
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::AMGPreconditioner, x::AbstractVector)
    # Use the aspreconditioner's ldiv! directly
    LinearAlgebra.ldiv!(y, P.precond, x)
    return y
end

function LinearAlgebra.ldiv!(P::AMGPreconditioner, x::AbstractVector)
    LinearAlgebra.ldiv!(P.precond, x)
    return x
end

function LinearAlgebra.mul!(y::AbstractVector, P::AMGPreconditioner, x::AbstractVector)
    ldiv!(y, P, x)
    return y
end

# Matrix-matrix mul! for block Krylov methods (e.g., block GMRES)
function LinearAlgebra.mul!(Y::AbstractMatrix, P::AMGPreconditioner, X::AbstractMatrix, α::Number, β::Number)
    if β == 0
        for j in axes(X, 2)
            ldiv!(view(Y, :, j), P.precond, view(X, :, j))
        end
        if α != 1
            Y .*= α
        end
    else
        temp = similar(view(Y, :, 1))
        for j in axes(X, 2)
            ldiv!(temp, P.precond, view(X, :, j))
            Y[:, j] .= α .* temp .+ β .* view(Y, :, j)
        end
    end
    return Y
end

# Block Triangular Preconditioner for Saddle Point Systems

"""
    BlockTriangularPreconditioner

Block lower triangular preconditioner for saddle point systems.
For system [A B; C D][u; λ] = [f; g], applies approximate inverse of:
P = [A 0; C S] where S ≈ D - C*A^-1*B (generalized Schur complement)

When D = 0, this reduces to S ≈ -C*A^-1*B (classical Schur complement).
Uses ILU factorizations to approximate A^-1 and S^-1.
The lower triangular structure captures the coupling between primal and dual variables.
Includes preallocated workspace to avoid allocations during application.
"""
struct BlockTriangularPreconditioner{TA, TC, TM, TV}
    A_precond::TA  # ILU preconditioner for primal block
    S_precond::TC  # ILU preconditioner for Schur complement approximation
    C::TM          # C block matrix (coupling term)
    n_primal::Int  # Number of primal variables
    n_dual::Int    # Number of dual variables
    temp::TV       # Preallocated workspace vector
end

"""
    BlockTriangularPreconditioner(A_precond, S_precond, C, n_primal, n_dual)

Constructor with automatic workspace allocation.
"""
function BlockTriangularPreconditioner(A_precond, S_precond, C, n_primal, n_dual)
    temp = zeros(n_dual)
    return BlockTriangularPreconditioner(A_precond, S_precond, C, n_primal, n_dual, temp)
end

# Implement LinearAlgebra.mul! for use with Krylov.jl
function LinearAlgebra.mul!(y, P::BlockTriangularPreconditioner, x)
    n_primal = P.n_primal
    n_dual = P.n_dual

    # Extract primal and dual parts
    x_primal = @view x[1:n_primal]
    x_dual = @view x[n_primal+1:n_primal+n_dual]
    y_primal = @view y[1:n_primal]
    y_dual = @view y[n_primal+1:n_primal+n_dual]

    # Apply block lower triangular preconditioner
    # Step 1: Solve A * y_primal = x_primal
    ldiv!(y_primal, P.A_precond, x_primal)

    # Step 2: Compute temp = x_dual - C * y_primal (using preallocated workspace)
    mul!(P.temp, P.C, y_primal)
    P.temp .= x_dual .- P.temp

    # Step 3: Solve S * y_dual = temp
    ldiv!(y_dual, P.S_precond, P.temp)

    return y
end

# Matrix-matrix mul! for block Krylov methods (e.g., block GMRES)
function LinearAlgebra.mul!(Y::AbstractMatrix, P::BlockTriangularPreconditioner, X::AbstractMatrix, α::Number, β::Number)
    if β == 0
        for j in axes(X, 2)
            mul!(view(Y, :, j), P, view(X, :, j))
        end
        if α != 1
            Y .*= α
        end
    else
        temp = similar(view(Y, :, 1))
        for j in axes(X, 2)
            mul!(temp, P, view(X, :, j))
            Y[:, j] .= α .* temp .+ β .* view(Y, :, j)
        end
    end
    return Y
end

# Required for Krylov.jl
Base.eltype(::BlockTriangularPreconditioner{TA, TC, TM, TV}) where {TA, TC, TM, TV} = Float64
Base.size(P::BlockTriangularPreconditioner) = (P.n_primal + P.n_dual, P.n_primal + P.n_dual)
Base.size(P::BlockTriangularPreconditioner, i::Int) = size(P)[i]