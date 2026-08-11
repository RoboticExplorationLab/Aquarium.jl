module PardisoExt

using AquariumClosed
using Pardiso
using LinearAlgebra
using SparseArrays

# Preconditioner wrapper for Pardiso
struct PardisoPreconditioner{T}
    solver::Pardiso.MKLPardisoSolver
    matrix::T
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::PardisoPreconditioner, x::AbstractVector)
    Pardiso.set_phase!(P.solver, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(P.solver, y, P.matrix, x)
    return y
end

function LinearAlgebra.ldiv!(P::PardisoPreconditioner, x::AbstractVector)
    y = similar(x)
    ldiv!(y, P, x)
    x .= y
    return x
end

function LinearAlgebra.mul!(y::AbstractVector, P::PardisoPreconditioner, x::AbstractVector)
    ldiv!(y, P, x)
    return y
end

# Matrix-matrix mul! for block Krylov methods (e.g., block GMRES)
function LinearAlgebra.mul!(Y::AbstractMatrix, P::PardisoPreconditioner, X::AbstractMatrix, α::Number, β::Number)
    if β == 0
        for j in axes(X, 2)
            ldiv!(view(Y, :, j), P, view(X, :, j))
        end
        if α != 1
            Y .*= α
        end
    else
        temp = similar(view(Y, :, 1))
        for j in axes(X, 2)
            ldiv!(temp, P, view(X, :, j))
            Y[:, j] .= α .* temp .+ β .* view(Y, :, j)
        end
    end
    return Y
end

# Wrapper functions for Pardiso operations
# These allow the main module to use Pardiso without directly importing it

function AquariumClosed.create_pardiso_solver()
    return Pardiso.MKLPardisoSolver()
end

function AquariumClosed.pardiso_set_nprocs!(solver::Pardiso.MKLPardisoSolver, n::Int)
    Pardiso.set_nprocs!(solver, n)
end

function AquariumClosed.pardiso_set_matrixtype!(solver::Pardiso.MKLPardisoSolver, ::Val{:REAL_NONSYM})
    Pardiso.set_matrixtype!(solver, Pardiso.REAL_NONSYM)
end

function AquariumClosed.pardiso_init!(solver::Pardiso.MKLPardisoSolver)
    Pardiso.pardisoinit(solver)
end

function AquariumClosed.pardiso_fix_iparm!(solver::Pardiso.MKLPardisoSolver)
    Pardiso.fix_iparm!(solver, :N)
end

function AquariumClosed.pardiso_set_iparm!(solver::Pardiso.MKLPardisoSolver, parm::Int, value::Int)
    Pardiso.set_iparm!(solver, parm, value)
end

function AquariumClosed.pardiso_set_phase!(solver::Pardiso.MKLPardisoSolver, ::Val{:ANALYSIS})
    Pardiso.set_phase!(solver, Pardiso.ANALYSIS)
end

function AquariumClosed.pardiso_set_phase!(solver::Pardiso.MKLPardisoSolver, ::Val{:ANALYSIS_NUM_FACT})
    Pardiso.set_phase!(solver, Pardiso.ANALYSIS_NUM_FACT)
end

function AquariumClosed.pardiso_set_phase!(solver::Pardiso.MKLPardisoSolver, ::Val{:SOLVE_ITERATIVE_REFINE})
    Pardiso.set_phase!(solver, Pardiso.SOLVE_ITERATIVE_REFINE)
end

function AquariumClosed.pardiso_set_phase!(solver::Pardiso.MKLPardisoSolver, ::Val{:RELEASE_ALL})
    Pardiso.set_phase!(solver, Pardiso.RELEASE_ALL)
end

# Pardiso solve with different signatures
function AquariumClosed.pardiso_solve!(solver::Pardiso.MKLPardisoSolver, matrix, vector)
    Pardiso.pardiso(solver, matrix, vector)
end

function AquariumClosed.pardiso_solve!(solver::Pardiso.MKLPardisoSolver, output, matrix, input)
    Pardiso.pardiso(solver, output, matrix, input)
end

function AquariumClosed.pardiso_solve!(solver::Pardiso.MKLPardisoSolver)
    Pardiso.pardiso(solver)
end

function AquariumClosed.pardiso_factorize!(solver::Pardiso.MKLPardisoSolver, matrix, vector)
    Pardiso.set_phase!(solver, Pardiso.ANALYSIS_NUM_FACT)
    Pardiso.pardiso(solver, matrix, vector)
end

# Indicate that Pardiso is loaded
function __init__()
    AquariumClosed.PARDISO_LOADED[] = true
end

end # module