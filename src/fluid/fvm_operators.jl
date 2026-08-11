function calculate_midpoint_operators(grid::FVMGrid)

    # [u; v] instead of [vx; vy] is used for velocity/flux-vector
    # notation throughout this function for simplicity

    # extract edge densities
    n_u_x = grid.n_vx_x
    n_u_y = grid.n_vx_y
    n_v_x = grid.n_vy_x
    n_v_y = grid.n_vy_y

    n_u = grid.n_vx
    n_v = grid.n_vy

    h_x = grid.h_x
    h_y = grid.h_y

    # calculate east midpoint u velocities over u control volumes
    u_uew_block = (1.0/(2*sqrt(h_x))) .* sparse(I, n_u_y, n_u_y)
    u_ue = kron(spdiagm(n_u_x, n_u_x, 0 => ones(n_u_x), 1 => ones(n_u_x-1)), u_uew_block)

    # calculate west midpoint u velocities over u control volumes
    u_uw = kron(spdiagm(n_u_x, n_u_x, 0 => ones(n_u_x), -1 => ones(n_u_x-1)), u_uew_block)

    # calculate north midpoint v velocities over u control volumes
    u_un_block = (1.0/(2*sqrt(h_y))) .* spdiagm(n_u_y, n_u_y, 0 => ones(n_u_y), 1 => ones(n_u_y-1))
    u_un = kron(I(n_u_x), u_un_block)

    # calculate south midpoint v velocities over u control volumes
    u_us_block = (1.0/(2*sqrt(h_y))) .* spdiagm(n_u_y, n_u_y, 0 => ones(n_u_y-1), -1 => ones(n_u_y-1))
    u_us = kron(I(n_u_x), u_us_block)

    # calculate north midpoint v velocities over u control volumes
    u_vn_block = (1.0/(2*sqrt(h_y))) .* spdiagm(n_u_y, n_v_y, 1 => ones(n_u_y))
    u_vn = kron(spdiagm(n_u_x, n_v_x, 0 => ones(n_v_x), -1 => ones(n_v_x)), u_vn_block)

    # calculate south midpoint v velocities over u control volumes
    u_vs_block = (1.0/(2*sqrt(h_y))) .* spdiagm(n_u_y, n_v_y, 0 => ones(n_u_y))
    u_vs = kron(spdiagm(n_u_x, n_v_x, 0 => ones(n_v_x), -1 => ones(n_v_x)), u_vs_block)

    # calculate east midpoint v velocities over v control volumes
    v_vew_block = (1.0/(2*sqrt(h_x))) .* sparse(I, n_v_y, n_v_y)
    v_ve = kron(spdiagm(n_v_x, n_v_x, 0 => ones(n_v_x), 1 => ones(n_v_x-1)), v_vew_block)

    # calculate west midpoint v velocities over v control volumes
    v_vw = kron(spdiagm(n_v_x, n_v_x, 0 => ones(n_v_x), -1 => ones(n_v_x-1)), v_vew_block)

    # calculate north midpoint v velocities over v control volumes
    v_vn_block = (1.0/(2*sqrt(h_y))) .* spdiagm(n_v_y, n_v_y, 0 => ones(n_v_y), 1 => ones(n_v_y-1))
    v_vn = kron(I(n_v_x), v_vn_block)

    # calculate south midpoint v velocities over v control volumes
    v_vs_block = (1.0/(2*sqrt(h_y))) .* spdiagm(n_v_y, n_v_y, 0 => ones(n_v_y), -1 => ones(n_v_y-1))
    v_vs = kron(I(n_v_x), v_vs_block)

    # calculate east midpoint u velocities over v control volumes
    v_uew_block = (1.0/(2*sqrt(h_x))) .* spdiagm(n_v_y, n_u_y, 0 => ones(n_u_y), -1 => ones(n_u_y))
    v_ue = kron(spdiagm(n_v_x, n_u_x, 1 => ones(n_v_x)), v_uew_block)

    # calculate west midpoint u velocities over v control volumes
    v_uw = kron(spdiagm(n_v_x, n_u_x, 0 => ones(n_v_x)), v_uew_block)

    # combine directional midpoint operators
    m1 = cat(u_ue, v_vn; dims=(1,2))
    m2 = cat(u_uw, v_vs; dims=(1,2))
    m3 = cat(u_un, v_ve; dims=(1,2))
    m4 = vcat(hcat(spzeros(n_u, n_u), u_vn),
        hcat(v_ue, spzeros(n_v, n_v))
    )
    m5 = cat(u_us, v_vw; dims=(1,2))
    m6 = vcat(hcat(spzeros(n_u, n_u), u_vs),
        hcat(v_uw, spzeros(n_v, n_v))
    )

    # remove boundary rows
    m1[grid.v_boundary_indices, :] .= 0.0
    m2[grid.v_boundary_indices, :] .= 0.0
    m3[grid.v_boundary_indices, :] .= 0.0
    m4[grid.v_boundary_indices, :] .= 0.0
    m5[grid.v_boundary_indices, :] .= 0.0
    m6[grid.v_boundary_indices, :] .= 0.0

    dropzeros!(m1)
    dropzeros!(m2)
    dropzeros!(m3)
    dropzeros!(m4)
    dropzeros!(m5)
    dropzeros!(m6)

    midpoint_operators = (m1, m2, m3, m4, m5, m6)

    return midpoint_operators

end

@testitem "4×3 Grid Midpoint Operators" begin
    using Aquarium
    using SparseArrays

    grid_size = (2, 1) # remember, number of cells = grid_size .+ 2
    grid_dimensions = (5.0, 2.0)

    grid = FVMGrid(grid_size, grid_dimensions)
    m1, m2, m3, m4, m5, m6 = calculate_midpoint_operators(grid)

    @test size(m1) == (grid.n_v, grid.n_v)
    @test size(m2) == (grid.n_v, grid.n_v)
    @test size(m3) == (grid.n_v, grid.n_v)
    @test size(m4) == (grid.n_v, grid.n_v)
    @test size(m5) == (grid.n_v, grid.n_v)
    @test size(m6) == (grid.n_v, grid.n_v)

    @test all(m1[grid.v_boundary_indices, :] .== 0.0)
    @test all(m2[grid.v_boundary_indices, :] .== 0.0)
    @test all(m3[grid.v_boundary_indices, :] .== 0.0)
    @test all(m4[grid.v_boundary_indices, :] .== 0.0)
    @test all(m5[grid.v_boundary_indices, :] .== 0.0)
    @test all(m6[grid.v_boundary_indices, :] .== 0.0)

    @test any(m1[grid.v_interior_indices, :] .!= 0.0)
    @test any(m2[grid.v_interior_indices, :] .!= 0.0)
    @test any(m3[grid.v_interior_indices, :] .!= 0.0)
    @test any(m4[grid.v_interior_indices, :] .!= 0.0)
    @test any(m5[grid.v_interior_indices, :] .!= 0.0)
    @test any(m6[grid.v_interior_indices, :] .!= 0.0)
end

@testitem "3×4 Grid Midpoint Operators" begin
    using Aquarium
    using SparseArrays

    grid_size = (1, 2) # remember, number of cells = grid_size .+ 2
    grid_dimensions = (3.0, 2.0)

    grid = FVMGrid(grid_size, grid_dimensions)
    m1, m2, m3, m4, m5, m6 = calculate_midpoint_operators(grid)

    @test size(m1) == (grid.n_v, grid.n_v)
    @test size(m2) == (grid.n_v, grid.n_v)
    @test size(m3) == (grid.n_v, grid.n_v)
    @test size(m4) == (grid.n_v, grid.n_v)
    @test size(m5) == (grid.n_v, grid.n_v)
    @test size(m6) == (grid.n_v, grid.n_v)

    @test all(m1[grid.v_boundary_indices, :] .== 0.0)
    @test all(m2[grid.v_boundary_indices, :] .== 0.0)
    @test all(m3[grid.v_boundary_indices, :] .== 0.0)
    @test all(m4[grid.v_boundary_indices, :] .== 0.0)
    @test all(m5[grid.v_boundary_indices, :] .== 0.0)
    @test all(m6[grid.v_boundary_indices, :] .== 0.0)

    @test any(m1[grid.v_interior_indices, :] .!= 0.0)
    @test any(m2[grid.v_interior_indices, :] .!= 0.0)
    @test any(m3[grid.v_interior_indices, :] .!= 0.0)
    @test any(m4[grid.v_interior_indices, :] .!= 0.0)
    @test any(m5[grid.v_interior_indices, :] .!= 0.0)
    @test any(m6[grid.v_interior_indices, :] .!= 0.0)
end

function calculate_laplacian_operator(grid::FVMGrid)

    # [u; v] instead of [vx; vy] is used for velocity/flux-vector
    # notation throughout this function for simplicity

    n_u_x = grid.n_vx_x
    n_u_y = grid.n_vx_y
    n_v_x = grid.n_vy_x
    n_v_y = grid.n_vy_y
    n_u = grid.n_vx
    n_v = grid.n_vy
    h_x = grid.h_x
    h_y = grid.h_y

    laplacian_u_x = (1.0/(h_x*h_x)) .* spdiagm(n_u, n_u, -n_u_y => ones(n_u-n_u_y), 0 => -2 .* ones(n_u), n_u_y => ones(n_u-n_u_y))
    laplacian_u_x[1:n_u_y, 1:n_u_y] = -(1.0/(h_x*h_x)) .* sparse(I, n_u_y, n_u_y)
    laplacian_u_x[n_u-n_u_y+1:n_u, n_u-n_u_y+1:n_u] = -(1.0/(h_x*h_x)) .* sparse(I, n_u_y, n_u_y)

    laplacian_u_y_block = (1.0/(h_y*h_y)) .* spdiagm(n_u_y, n_u_y, -1 => ones(n_u_y-1), 0 => -2 .* ones(n_u_y), 1 => ones(n_u_y-1))
    laplacian_u_y_block[1, 1] = -(1.0/(h_y*h_y))
    laplacian_u_y_block[end, end] = -(1.0/(h_y*h_y))
    laplacian_u_y = kron(I(n_u_x), laplacian_u_y_block)

    laplacian_v_x = (1.0/(h_x*h_x)) .* spdiagm(n_v, n_v, -n_v_y => ones(n_v-n_v_y), 0 => -2 .* ones(n_v), n_v_y => ones(n_v-n_v_y))
    laplacian_v_x[1:n_v_y, 1:n_v_y] = -(1.0/(h_x*h_x)) .* sparse(I, n_v_y, n_v_y)
    laplacian_v_x[n_v-n_v_y+1:n_v, n_v-n_v_y+1:n_v] = -(1.0/(h_x*h_x)) .* sparse(I, n_v_y, n_v_y)

    laplacian_v_y_block = (1.0/(h_y*h_y)) .* spdiagm(n_v_y, n_v_y, -1 => ones(n_v_y-1), 0 => -2 .* ones(n_v_y), 1 => ones(n_v_y-1))
    laplacian_v_y_block[1, 1] = -(1.0/(h_y*h_y))
    laplacian_v_y_block[end, end] = -(1.0/(h_y*h_y))
    laplacian_v_y = kron(I(n_v_x), laplacian_v_y_block)

    laplacian_u = laplacian_u_x + laplacian_u_y
    laplacian_v = laplacian_v_x + laplacian_v_y

    laplacian = cat(laplacian_u, laplacian_v, dims=(1, 2))

    # remove boundary rows
    laplacian[grid.v_boundary_indices, :] .= 0.0
    dropzeros!(laplacian)

    return laplacian

end

@testitem "4×3 Grid Laplacian" begin
    using Aquarium
    using LinearAlgebra
    using SparseArrays

    grid_size = (2, 1) # remember, number of cells = grid_size .+ 2
    grid_dimensions = (5.0, 2.0)

    grid = FVMGrid(grid_size, grid_dimensions)
    laplacian = calculate_laplacian_operator(grid)

    @test size(laplacian) == (grid.n_v, grid.n_v)

    @test all(laplacian[grid.v_boundary_indices, :] .== 0.0)

    laplacian_interior = laplacian[grid.v_interior_indices, grid.v_interior_indices]
    @test issymmetric(laplacian_interior)

    @test maximum(eigvals(Matrix(laplacian_interior))) < 1e-10
end

@testitem "3×4 Grid Laplacian" begin
    using Aquarium
    using LinearAlgebra
    using SparseArrays

    grid_size = (1, 2) # remember, number of cells = grid_size .+ 2
    grid_dimensions = (3.0, 2.0)

    grid = FVMGrid(grid_size, grid_dimensions)
    laplacian = calculate_laplacian_operator(grid)

    @test size(laplacian) == (grid.n_v, grid.n_v)

    @test all(laplacian[grid.v_boundary_indices, :] .== 0.0)

    laplacian_interior = laplacian[grid.v_interior_indices, grid.v_interior_indices]
    @test issymmetric(laplacian_interior)

    @test maximum(eigvals(Matrix(laplacian_interior))) < 1e-10
end

function calculate_divergence_operator(grid::FVMGrid)

    # [u; v] instead of [vx; vy] is used for velocity/flux-vector
    # notation throughout this function for simplicity

    n_cell_x = grid.n_cell_x
    n_cell_y = grid.n_cell_y
    n_u_x = grid.n_vx_x
    n_u_y = grid.n_vx_y
    n_v_x = grid.n_vy_x
    n_v_y = grid.n_vy_y
    h_x = grid.h_x
    h_y = grid.h_y

    divergence_u_block = (1.0 / h_x) .* spdiagm(n_cell_y, n_u_y, 0 => ones(n_cell_y))
    divergence_u = kron(diagm(n_cell_x, n_u_x, 0 => -ones(n_cell_x), 1 => ones(n_cell_x)), divergence_u_block)

    divergence_v_block = (1.0 / h_y) .* spdiagm(n_cell_y, n_v_y, 0 => -ones(n_cell_y), 1 => ones(n_cell_y))
    divergence_v = kron(diagm(n_cell_x, n_v_x, 0 => ones(n_cell_x)), divergence_v_block)

    divergence = hcat(divergence_u, divergence_v)

    # Remove rows corresponding to corner cells (which have degenerate constraints)
    # Corner cells are at linear indices: 1, n_cell_y, (n_cell_x-1)*n_cell_y + 1, n_cell_x*n_cell_y
    corner_cell_indices = [
        1,                          # bottom-left corner
        n_cell_y,                   # top-left corner
        (n_cell_x-1)*n_cell_y + 1,  # bottom-right corner
        n_cell_x*n_cell_y           # top-right corner
    ]

    # Remove rows from divergence operator corresponding to corner cells
    all_rows = 1:(n_cell_x*n_cell_y)
    non_corner_rows = setdiff(all_rows, corner_cell_indices)
    divergence = divergence[non_corner_rows, :]

    return divergence

end

@testitem "4×3 Grid Divergence" begin
    using Aquarium
    grid_size = (2, 1) # remember, number of cells = grid_size .+ 2
    grid_dimensions = (11.0, 5.0)

    grid = FVMGrid(grid_size, grid_dimensions)
    divergence = calculate_divergence_operator(grid)

    v = vcat(2 .* ones(grid.n_vx), 1.5 .* ones(grid.n_vy))

    @test size(divergence) == (grid.n_cell-4, grid.n_vx + grid.n_vy)
    @test divergence * v == zeros(grid.n_cell-4)
end

@testitem "3×4 Grid Divergence" begin
    using Aquarium
    grid_size = (1, 2) # remember, number of cells = grid_size .+ 2
    grid_dimensions = (7.0, 3.0)

    grid = FVMGrid(grid_size, grid_dimensions)
    divergence = calculate_divergence_operator(grid)

    v = vcat(2 .* ones(grid.n_vx), 1.5 .* ones(grid.n_vy))

    @test size(divergence) == (grid.n_cell-4, grid.n_vx + grid.n_vy)
    @test divergence * v == zeros(grid.n_cell-4)
end
