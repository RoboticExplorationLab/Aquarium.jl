mutable struct FVMGrid
    n_cell_x::Int # number of cells in x direction
    n_cell_y::Int # number of cells in y direction
    length_x::Float64 # length in x direction
    length_y::Float64 # length in y direction

    n_vx_x::Int # number of x-velocity/fluxes in x direction
    n_vx_y::Int # number of y-velocity/fluxes in y direction
    n_vy_x::Int # number of y-velocity/fluxes in x direction
    n_vy_y::Int # number of y-velocity/fluxes in y direction

    n_vx::Int # number of x-velocity/fluxes
    n_vy::Int # number of y-velocity/fluxes
    n_v::Int # total number of velocity/fluxes
    n_v_boundary::Int # number of boundary velocity/fluxes
    n_cell::Int # number of cells

    h_x::Float64 # grid spacing in x direction
    h_y::Float64 # grid spacing in y direction

    vx_left_indices::Vector{Int} # indices of left boundary x-velocities/fluxes
    vx_right_indices::Vector{Int} # indices of right boundary x-velocities/fluxes
    vx_bottom_indices::Vector{Int} # indices of bottom boundary x-velocities/fluxes
    vx_top_indices::Vector{Int} # indices of top boundary x-velocities/fluxes
    vy_left_indices::Vector{Int} # indices of left boundary y-velocities/fluxes
    vy_right_indices::Vector{Int} # indices of right boundary y-velocities/fluxes
    vy_bottom_indices::Vector{Int} # indices of bottom boundary y-velocities/fluxes
    vy_top_indices::Vector{Int} # indices of top boundary y-velocities/fluxes

    v_boundary_indices::Vector{Int} # indices of boundary velocity/fluxes
    v_interior_indices::Vector{Int} # indices of interior velocity/fluxes

    cell_left_indices::Vector{Int} # indices of left boundary cells
    cell_right_indices::Vector{Int} # indices of right boundary cells
    cell_bottom_indices::Vector{Int} # indices of bottom boundary cells
    cell_top_indices::Vector{Int} # indices of top boundary cells
    cell_boundary_indices::Vector{Int} # indices of boundary cells

    x_coord_cell_flat::Vector{Float64} # vectorized x-coordinates of cell centers
    y_coord_cell_flat::Vector{Float64} # vectorized y-coordinates of cell centers
    x_coord_vx_flat::Vector{Float64} # vectorized x-coordinates of x-velocity/flux locations
    y_coord_vx_flat::Vector{Float64} # vectorized y-coordinates of x-velocity/flux locations
    x_coord_vy_flat::Vector{Float64} # vectorized x-coordinates of y-velocity/flux locations
    y_coord_vy_flat::Vector{Float64} # vectorized y-coordinates of y-velocity/flux locations

    x_coord_cell_grid::Matrix{Float64} # grid of x-coordinates of cell centers
    y_coord_cell_grid::Matrix{Float64} # grid of y-coordinates of cell centers
    x_coord_vx_grid::Matrix{Float64} # grid of x-coordinates of x-velocity/flux locations
    y_coord_vx_grid::Matrix{Float64} # grid of y-coordinates of x-velocity/flux locations
    x_coord_vy_grid::Matrix{Float64} # grid of x-coordinates of y-velocity/flux locations
    y_coord_vy_grid::Matrix{Float64} # grid of y-coordinates of y-velocity/flux locations

    x_coord_cell_values::AbstractRange # vectorized x-coordinates of cell centers
    y_coord_cell_values::AbstractRange # vectorized y-coordinates of cell centers
    x_coord_vx_values::AbstractRange # vectorized x-coordinates of x-velocity/flux locations
    y_coord_vx_values::AbstractRange # vectorized y-coordinates of x-velocity/flux locations
    x_coord_vy_values::AbstractRange # vectorized x-coordinates of y-velocity/flux locations
    y_coord_vy_values::AbstractRange # vectorized y-coordinates of y-velocity/flux locations

end

function FVMGrid(grid_size::Tuple{Int,Int},
    grid_dimensions::Tuple{AbstractFloat,AbstractFloat}
)
    
    # extract grid size and dimensions
    n_cell_x, n_cell_y = grid_size
    length_x, length_y = grid_dimensions

    # calculate grid spacing
    h_x = length_x / n_cell_x
    h_y = length_y / n_cell_y

    # account for the fact that boundary cells lie beyond the physical domain
    # i.e., add two extra cells in each direction
    n_cell_x += 2
    n_cell_y += 2

    # calculate number of variables
    n_vx_x, n_vx_y = n_cell_x+1, n_cell_y # number of x velocity/fluxes (u) in x and y direction
    n_vy_x, n_vy_y = n_cell_x, n_cell_y+1 # number of y velocity/fluxes (v) in x and y direction

    n_vx = n_vx_x*n_vx_y
    n_vy = n_vy_x*n_vy_y
    n_v = n_vx + n_vy
    n_cell = n_cell_x*n_cell_y

    # calculate coordinates of vx, vy, and cell centers
    x_coord_cell_values = h_x .* ((0:n_cell_x-1) .- 0.5)
    y_coord_cell_values = h_y .* ((0:n_cell_y-1) .- 0.5)

    x_coord_vx_values = h_x .* ((0:n_vx_x-1) .- 1.0)
    y_coord_vx_values = h_y .* ((0:n_vx_y-1) .- 0.5)

    x_coord_vy_values = h_x .* ((0:n_vy_x-1) .- 0.5)
    y_coord_vy_values = h_y .* ((0:n_vy_y-1) .- 1.0)

    # create grid versions (2D matrices)
    x_coord_cell_grid = x_coord_cell_values' .* ones(n_cell_y)
    y_coord_cell_grid = ones(n_cell_x)' .* y_coord_cell_values

    x_coord_vx_grid = x_coord_vx_values' .* ones(n_vx_y)
    y_coord_vx_grid = ones(n_vx_x)' .* y_coord_vx_values

    x_coord_vy_grid = x_coord_vy_values' .* ones(n_vy_y)
    y_coord_vy_grid = ones(n_vy_x)' .* y_coord_vy_values

    # create flattened versions (1D vectors)
    x_coord_cell_flat = x_coord_cell_grid[:]
    y_coord_cell_flat = y_coord_cell_grid[:]

    x_coord_vx_flat = x_coord_vx_grid[:]
    y_coord_vx_flat = y_coord_vx_grid[:]

    x_coord_vy_flat = x_coord_vy_grid[:]
    y_coord_vy_flat = y_coord_vy_grid[:]

    # determine boundary indices
    vx_left_indices = collect(2 : n_vx_y-1)
    vx_right_indices = collect(n_vx-n_vx_y+2 : n_vx-1)
    # vx_bottom_indices = collect(n_vx_y+1 : n_vx_y : n_vx-2*n_vx_y+1)
    # vx_top_indices = collect(2*n_vx_y : n_vx_y : n_vx-n_vx_y)
    vx_bottom_indices = collect(1 : n_vx_y : n_vx-n_vx_y+1)
    vx_top_indices = collect(n_vx_y : n_vx_y : n_vx)

    vy_left_indices = collect(2 : n_vy_y-1) .+ n_vx
    vy_right_indices = collect(n_vy-n_vy_y+2 : n_vy-1) .+ n_vx
    # vy_bottom_indices = collect(n_vy_y+1 : n_vy_y : n_vy-2*n_vy_y+1) .+ n_vx
    # vy_top_indices = collect(2*n_vy_y : n_vy_y : n_vy-n_vy_y) .+ n_vx
    vy_bottom_indices = collect(1 : n_vy_y : n_vy-n_vy_y+1) .+ n_vx
    vy_top_indices = collect(n_vy_y : n_vy_y : n_vy) .+ n_vx

    v_boundary_indices = vcat(
        vx_left_indices,
        vx_right_indices,
        vx_bottom_indices,
        vx_top_indices,
        vy_left_indices,
        vy_right_indices,
        vy_bottom_indices,
        vy_top_indices
    )
    v_interior_indices = setdiff(1:n_v, v_boundary_indices)
    n_v_boundary = length(v_boundary_indices)

    # determine boundary cells
    cell_left_indices = collect(2 : n_cell_y-1)
    cell_right_indices = collect(n_cell-n_cell_y+2 : n_cell-1)
    cell_bottom_indices = collect(1 : n_cell_y : n_cell-n_cell_y+1)
    cell_top_indices = collect(n_cell_y : n_cell_y : n_cell)

    cell_boundary_indices = vcat(
        cell_left_indices,
        cell_right_indices,
        cell_bottom_indices,
        cell_top_indices
    )

    # construct and return FVMGrid struct
    return FVMGrid(
        n_cell_x,
        n_cell_y,
        Float64(length_x),
        Float64(length_y),
        n_vx_x,
        n_vx_y,
        n_vy_x,
        n_vy_y,
        n_vx,
        n_vy,
        n_v,
        n_v_boundary,
        n_cell,
        Float64(h_x),
        Float64(h_y),
        vx_left_indices,
        vx_right_indices,
        vx_bottom_indices,
        vx_top_indices,
        vy_left_indices,
        vy_right_indices,
        vy_bottom_indices,
        vy_top_indices,
        v_boundary_indices,
        v_interior_indices,
        cell_left_indices,
        cell_right_indices,
        cell_bottom_indices,
        cell_top_indices,
        cell_boundary_indices,
        x_coord_cell_flat,
        y_coord_cell_flat,
        x_coord_vx_flat,
        y_coord_vx_flat,
        x_coord_vy_flat,
        y_coord_vy_flat,
        x_coord_cell_grid,
        y_coord_cell_grid,
        x_coord_vx_grid,
        y_coord_vx_grid,
        x_coord_vy_grid,
        y_coord_vy_grid,
        x_coord_cell_values,
        y_coord_cell_values,
        x_coord_vx_values,
        y_coord_vx_values,
        x_coord_vy_values,
        y_coord_vy_values
    )

end

function calculate_constant_boundary_condition_operator(grid::FVMGrid,
    boundary_condition_type::Symbol
)

    n_v = grid.n_v

    vx_left_indices = grid.vx_left_indices
    vx_right_indices = grid.vx_right_indices

    vy_left_indices = grid.vy_left_indices
    vy_right_indices = grid.vy_right_indices

    boundary_indices = grid.v_boundary_indices
    bc_constant_matrix = sparse(boundary_indices, boundary_indices, 1.0, n_v, n_v)

    if boundary_condition_type in (:wall, :lid_cavity, :channel_flow_theoretical, :channel_flow_uniform)

        # All entries are zero (already initialized)

    elseif boundary_condition_type == :freestream

        bc_constant_matrix[vx_right_indices, :] .= 0.0
        bc_constant_matrix[vy_right_indices, :] .= 0.0

    else

        error("Unsupported boundary condition type: $boundary_condition_type")
    
    end

    dropzeros!(bc_constant_matrix)

    return bc_constant_matrix
end

function calculate_constant_boundary_condition_vector(grid::FVMGrid,
    boundary_velocity::AbstractVector,
    boundary_condition_type::Symbol
)

    vx_left_indices = grid.vx_left_indices
    vx_right_indices = grid.vx_right_indices
    vx_bottom_indices = grid.vx_bottom_indices
    vx_top_indices = grid.vx_top_indices

    vy_left_indices = grid.vy_left_indices
    vy_right_indices = grid.vy_right_indices
    vy_bottom_indices = grid.vy_bottom_indices
    vy_top_indices = grid.vy_top_indices

    # Wall boundary conditions are all zero
    bc_vector = zeros(eltype(boundary_velocity), grid.n_v)

    if boundary_condition_type == :wall
        
        # All entries are zero (already initialized)

    elseif boundary_condition_type == :lid_cavity

        bc_vector[vx_top_indices] .= boundary_velocity[1]

    elseif boundary_condition_type == :channel_flow_theoretical

        # Left boundary (inflow) - parabolic profile
        y_coords_left = grid.y_coord_vx_flat[vx_left_indices]
        for (i, idx) in enumerate(vx_left_indices)
            y = y_coords_left[i]
            bc_vector[idx] = 4.0 * boundary_velocity[1] * (y * (grid.length_y - y)) / (grid.length_y^2)
        end

        # Right boundary (outflow) - zero velocity
        # Already initialized to zero

    elseif boundary_condition_type == :channel_flow_uniform

        # Left boundary (inflow) - uniform velocity
        bc_vector[vx_left_indices] .= boundary_velocity[1]

        # Right boundary (outflow) - zero velocity
        # Already initialized to zero

    elseif boundary_condition_type == :freestream

        bc_vector[vx_left_indices] .= boundary_velocity[1]
        bc_vector[vx_bottom_indices] .= boundary_velocity[1]
        bc_vector[vx_top_indices] .= boundary_velocity[1]
        bc_vector[vy_left_indices] .= boundary_velocity[2]
        bc_vector[vy_bottom_indices] .= boundary_velocity[2]
        bc_vector[vy_top_indices] .= boundary_velocity[2]

        # vx_right and vy_right are zero (already initialized)

    else

        error("Unsupported boundary condition type: $boundary_condition_type")
    
    end

    return bc_vector

end

function calculate_constant_boundary_condition_vector_jacobian(grid::FVMGrid,
    boundary_condition_type::Symbol
)
    """
    Calculate the Jacobian of the boundary condition vector with respect to boundary_velocity.

    Returns a sparse matrix ∂bc_vector/∂boundary_velocity of size (n_v, 2)
    where boundary_velocity is a 2-element vector [vx, vy].
    """

    n_v = grid.n_v

    vx_left_indices = grid.vx_left_indices
    vx_right_indices = grid.vx_right_indices
    vx_bottom_indices = grid.vx_bottom_indices
    vx_top_indices = grid.vx_top_indices

    vy_left_indices = grid.vy_left_indices
    vy_right_indices = grid.vy_right_indices
    vy_bottom_indices = grid.vy_bottom_indices
    vy_top_indices = grid.vy_top_indices

    if boundary_condition_type == :wall

        # Wall boundary conditions are all zero, so Jacobian is zero
        jacobian = spzeros(n_v, 2)

    elseif boundary_condition_type == :lid_cavity

        # Only vx_top depends on boundary_velocity[1]
        # All other entries are zero
        n_vx_top = length(vx_top_indices)

        # Build sparse matrix using COO format
        rows = vx_top_indices
        cols = ones(Int, n_vx_top)
        vals = ones(n_vx_top)

        jacobian = sparse(rows, cols, vals, n_v, 2)

    elseif boundary_condition_type == :channel_flow_theoretical

        # Parabolic velocity profile at left boundary depends on boundary_velocity[1]
        # bc_vector[idx] = 4.0 * boundary_velocity[1] * (y * (L - y)) / L²
        # ∂bc_vector/∂boundary_velocity[1] = 4.0 * (y * (L - y)) / L²
        n_vx_left = length(vx_left_indices)
        L = grid.length_y

        # Get y-coordinates for left boundary
        y_coords_left = grid.y_coord_vx_flat[vx_left_indices]

        # Calculate derivative values: 4 * y * (L - y) / L²
        derivative_vals = [4.0 * y * (L - y) / (L^2) for y in y_coords_left]

        # Build sparse matrix using COO format
        rows = vx_left_indices
        cols = ones(Int, n_vx_left)
        vals = derivative_vals

        jacobian = sparse(rows, cols, vals, n_v, 2)

    elseif boundary_condition_type == :channel_flow_uniform

        # Uniform velocity profile at left boundary depends on boundary_velocity[1]
        # bc_vector[idx] = boundary_velocity[1]
        # ∂bc_vector/∂boundary_velocity[1] = 1.0
        n_vx_left = length(vx_left_indices)

        # Build sparse matrix using COO format
        rows = vx_left_indices
        cols = ones(Int, n_vx_left)
        vals = ones(n_vx_left)

        jacobian = sparse(rows, cols, vals, n_v, 2)

    elseif boundary_condition_type == :freestream

        # Dirichlet BCs at left, bottom, top boundaries
        # vx components depend on boundary_velocity[1]
        # vy components depend on boundary_velocity[2]
        n_vx_left = length(vx_left_indices)
        n_vx_bottom = length(vx_bottom_indices)
        n_vx_top = length(vx_top_indices)
        n_vy_left = length(vy_left_indices)
        n_vy_bottom = length(vy_bottom_indices)
        n_vy_top = length(vy_top_indices)

        rows = vcat(
            vx_left_indices,
            vx_bottom_indices,
            vx_top_indices,
            vy_left_indices,
            vy_bottom_indices,
            vy_top_indices
        )

        cols = vcat(
            ones(Int, n_vx_left),
            ones(Int, n_vx_bottom),
            ones(Int, n_vx_top),
            fill(2, n_vy_left),
            fill(2, n_vy_bottom),
            fill(2, n_vy_top)
        )

        vals = ones(n_vx_left + n_vx_bottom + n_vx_top + n_vy_left + n_vy_bottom + n_vy_top)

        jacobian = sparse(rows, cols, vals, n_v, 2)

    else

        error("Unsupported boundary condition type: $boundary_condition_type")

    end

    return jacobian

end

@testitem "3×3 Grid Boundary Indices" begin
    using AquariumClosed
    grid = FVMGrid((1, 1), (1.0, 1.0))

    @test (typeof(grid.vx_left_indices) <: Vector{Int}) &
        (typeof(grid.vx_right_indices) <: Vector{Int}) &
        (typeof(grid.vx_bottom_indices) <: Vector{Int}) &
        (typeof(grid.vx_top_indices) <: Vector{Int}) &
        (typeof(grid.vy_left_indices) <: Vector{Int}) &
        (typeof(grid.vy_right_indices) <: Vector{Int}) &
        (typeof(grid.vy_bottom_indices) <: Vector{Int}) &
        (typeof(grid.vy_top_indices) <: Vector{Int})

    @test (grid.vx_left_indices == [2]) &
        (grid.vx_right_indices == [11]) &
        (grid.vx_bottom_indices == [1, 4, 7, 10]) &
        (grid.vx_top_indices == [3, 6, 9, 12])

    @test (grid.vy_left_indices .- grid.n_vx == [2, 3]) &
        (grid.vy_right_indices .- grid.n_vx == [10, 11]) &
        (grid.vy_bottom_indices .- grid.n_vx == [1, 5, 9]) &
        (grid.vy_top_indices .- grid.n_vx == [4, 8, 12])

    @test (typeof(grid.cell_left_indices) <: Vector{Int}) &
        (typeof(grid.cell_right_indices) <: Vector{Int}) &
        (typeof(grid.cell_bottom_indices) <: Vector{Int}) &
        (typeof(grid.cell_top_indices) <: Vector{Int})

    @test (grid.cell_left_indices == [2]) &
        (grid.cell_right_indices == [8]) &
        (grid.cell_bottom_indices == [1, 4, 7]) &
        (grid.cell_top_indices == [3, 6, 9])
end

@testitem "4×3 Grid Boundary Indices" begin
    using AquariumClosed
    grid = FVMGrid((2, 1), (1.0, 1.0))

    @test (grid.vx_left_indices == [2]) &
        (grid.vx_right_indices == [14]) &
        (grid.vx_bottom_indices == [1, 4, 7, 10, 13]) &
        (grid.vx_top_indices == [3, 6, 9, 12, 15])

    @test (grid.vy_left_indices .- grid.n_vx == [2, 3]) &
        (grid.vy_right_indices .- grid.n_vx == [14, 15]) &
        (grid.vy_bottom_indices .- grid.n_vx == [1, 5, 9, 13]) &
        (grid.vy_top_indices .- grid.n_vx == [4, 8, 12, 16])

    @test (grid.cell_left_indices == [2]) &
        (grid.cell_right_indices == [11]) &
        (grid.cell_bottom_indices == [1, 4, 7, 10]) &
        (grid.cell_top_indices == [3, 6, 9, 12])
end

@testitem "3×4 Grid Boundary Indices" begin
    using AquariumClosed
    grid = FVMGrid((1, 2), (1.0, 1.0))

    @test (grid.vx_left_indices == [2, 3]) &
        (grid.vx_right_indices == [14, 15]) &
        (grid.vx_bottom_indices == [1, 5, 9, 13]) &
        (grid.vx_top_indices == [4, 8, 12, 16])

    @test (grid.vy_left_indices .- grid.n_vx == [2, 3, 4]) &
        (grid.vy_right_indices .- grid.n_vx == [12, 13, 14]) &
        (grid.vy_bottom_indices .- grid.n_vx == [1, 6, 11]) &
        (grid.vy_top_indices .- grid.n_vx == [5, 10, 15])

    @test (grid.cell_left_indices == [2, 3]) &
        (grid.cell_right_indices == [10, 11]) &
        (grid.cell_bottom_indices == [1, 5, 9]) &
        (grid.cell_top_indices == [4, 8, 12])
end

@testitem "Cell Boundary Index Properties" begin
    using AquariumClosed
    grid_size = (5, 5)
    grid = FVMGrid(grid_size, (1.0, 1.0))

    n_cell_x = grid_size[1] + 2
    n_cell_y = grid_size[2] + 2
    total_cells = n_cell_x * n_cell_y

    @test grid.n_cell == total_cells

    expected_left = collect(2:n_cell_y-1)
    @test grid.cell_left_indices == expected_left
    @test length(grid.cell_left_indices) == n_cell_y - 2

    expected_right = collect(total_cells - n_cell_y + 2 : total_cells - 1)
    @test grid.cell_right_indices == expected_right
    @test length(grid.cell_right_indices) == n_cell_y - 2

    expected_bottom = collect(1:n_cell_y:total_cells-n_cell_y+1)
    @test grid.cell_bottom_indices == expected_bottom
    @test length(grid.cell_bottom_indices) == n_cell_x

    expected_top = collect(n_cell_y:n_cell_y:total_cells)
    @test grid.cell_top_indices == expected_top
    @test length(grid.cell_top_indices) == n_cell_x

    all_boundary_cells = vcat(
        grid.cell_left_indices,
        grid.cell_right_indices,
        grid.cell_bottom_indices,
        grid.cell_top_indices
    )
    @test length(all_boundary_cells) == length(unique(all_boundary_cells))

    @test all(1 .<= all_boundary_cells .<= total_cells)

    left_x_coords = grid.x_coord_cell_flat[grid.cell_left_indices]
    @test all(left_x_coords .≈ grid.h_x * (-0.5))

    right_x_coords = grid.x_coord_cell_flat[grid.cell_right_indices]
    @test all(right_x_coords .≈ grid.h_x * (n_cell_x - 1.5))

    bottom_y_coords = grid.y_coord_cell_flat[grid.cell_bottom_indices]
    @test all(bottom_y_coords .≈ grid.h_y * (-0.5))

    top_y_coords = grid.y_coord_cell_flat[grid.cell_top_indices]
    @test all(top_y_coords .≈ grid.h_y * (n_cell_y - 1.5))
end