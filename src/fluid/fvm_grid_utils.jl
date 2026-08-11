using Interpolations
using ImageFiltering

function create_pressure_grid(fvm_grid::FVMGrid,
    p::AbstractVector
)

    # Pressure vector excludes corner cells (4 corners removed from divergence)
    # We need to pad it back to full grid size before reshaping
    n_cell_x = fvm_grid.n_cell_x
    n_cell_y = fvm_grid.n_cell_y
    total_cells = n_cell_x * n_cell_y

    # Corner cell indices in the full grid (1-indexed, column-major)
    corner_indices = [
        1,                      # bottom-left corner
        n_cell_y,               # top-left corner
        (n_cell_x-1)*n_cell_y + 1,  # bottom-right corner
        n_cell_x*n_cell_y       # top-right corner
    ]

    # Create full pressure vector with zeros at corners
    p_full = zeros(eltype(p), total_cells)

    # Fill non-corner cells with pressure values
    all_indices = 1:total_cells
    non_corner_indices = setdiff(all_indices, corner_indices)
    p_full[non_corner_indices] = p

    # Reshape into grid (column-major: first index is y, second is x)
    p_grid = reshape(p_full, n_cell_y, n_cell_x)

    return p_grid
end

function create_fluid_velocity_grid(fvm_grid::FVMGrid,
    v::AbstractVector
)

    # turn fluid velocity vector into grid
    vx_grid = reshape(v[1:fvm_grid.n_vx],
        fvm_grid.n_vx_y,
        fvm_grid.n_vx_x
    )
    vy_grid = reshape(v[fvm_grid.n_vx+1:end],
        fvm_grid.n_vy_y,
        fvm_grid.n_vy_x
    )

    return vx_grid, vy_grid
end

function create_grid_interpolant(grid_values::AbstractMatrix,
    x_coord_values::AbstractVector,
    y_coord_values::AbstractVector
)

    grid_interpolant = scale(
        interpolate(grid_values', BSpline(Cubic(Line(OnGrid())))),
        x_coord_values,
        y_coord_values
    )

    return grid_interpolant
end

function create_fluid_grid_interpolant(fvm_grid::FVMGrid,
    v::AbstractVector
)

    vx_grid, vy_grid = create_fluid_velocity_grid(fvm_grid, v)

    vx_interpolant = create_grid_interpolant(vx_grid,
        fvm_grid.x_coord_vx_values,
        fvm_grid.y_coord_vx_values
    )

    vy_interpolant = create_grid_interpolant(vy_grid,
        fvm_grid.x_coord_vy_values,
        fvm_grid.y_coord_vy_values
    )

    return vx_interpolant, vy_interpolant
end

function create_pressure_grid_interpolant(fvm_grid::FVMGrid,
    p::AbstractVector
)

    p_grid = create_pressure_grid(fvm_grid, p)
    p_interpolant = create_grid_interpolant(p_grid,
        fvm_grid.x_coord_cell_values,
        fvm_grid.y_coord_cell_values
    )

    return p_interpolant
end

function interpolate_fluid_velocity_grid(fvm_grid::FVMGrid,
    v::AbstractVector,
    x_coord_values::AbstractVector,
    y_coord_values::AbstractVector;
    smooth::Bool=false,
    smooth_sigma::Float64=1.5
)

    # Get velocity grids
    vx_grid, vy_grid = create_fluid_velocity_grid(fvm_grid, v)

    # Apply smoothing to velocity fields if requested
    if smooth
        vx_grid, vy_grid = smooth_velocity_fields(vx_grid, vy_grid; sigma=smooth_sigma)
    end

    # Create interpolants from (possibly smoothed) velocity grids
    vx_interpolant = create_grid_interpolant(vx_grid,
        fvm_grid.x_coord_vx_values,
        fvm_grid.y_coord_vx_values
    )

    vy_interpolant = create_grid_interpolant(vy_grid,
        fvm_grid.x_coord_vy_values,
        fvm_grid.y_coord_vy_values
    )

    vx_grid_interp, vy_grid_interp = interpolate_fluid_velocity_grid(vx_interpolant,
        vy_interpolant,
        x_coord_values,
        y_coord_values
    )

    return vx_grid_interp, vy_grid_interp
end

function interpolate_fluid_velocity_grid(vx_interpolant::AbstractInterpolation,
    vy_interpolant::AbstractInterpolation,
    x_coord_values::AbstractVector,
    y_coord_values::AbstractVector
)

    vx_grid = zeros(length(y_coord_values), length(x_coord_values))
    vy_grid = zeros(length(y_coord_values), length(x_coord_values))

    for i in eachindex(x_coord_values)
        for j in eachindex(y_coord_values)
    
            vx_grid[j, i] = vx_interpolant(x_coord_values[i], y_coord_values[j])
            vy_grid[j, i] = vy_interpolant(x_coord_values[i], y_coord_values[j])
    
        end
    end

    return vx_grid, vy_grid

end

function interpolate_fluid_vorticity_grid(fvm_grid::FVMGrid,
    v::AbstractVector,
    x_coord_values::AbstractVector,
    y_coord_values::AbstractVector;
    smooth::Bool=false,
    smooth_sigma::Float64=1.5
)

    # Get velocity grids
    vx_grid, vy_grid = create_fluid_velocity_grid(fvm_grid, v)

    # Apply smoothing to velocity fields if requested
    if smooth
        vx_grid, vy_grid = smooth_velocity_fields(vx_grid, vy_grid; sigma=smooth_sigma)
    end

    # Create interpolants from (possibly smoothed) velocity grids
    vx_interpolant = create_grid_interpolant(vx_grid,
        fvm_grid.x_coord_vx_values,
        fvm_grid.y_coord_vx_values
    )

    vy_interpolant = create_grid_interpolant(vy_grid,
        fvm_grid.x_coord_vy_values,
        fvm_grid.y_coord_vy_values
    )

    vorticity_grid = interpolate_fluid_vorticity_grid(vx_interpolant, vy_interpolant, x_coord_values, y_coord_values)

    return vorticity_grid

end

function interpolate_fluid_vorticity_grid(vx_interpolant::AbstractInterpolation,
    vy_interpolant::AbstractInterpolation,
    x_coord_values::AbstractVector,
    y_coord_values::AbstractVector
)

    vorticity_grid = zeros(length(y_coord_values), length(x_coord_values))

    for i in eachindex(x_coord_values)
        for j in eachindex(y_coord_values)

            dudy = Interpolations.gradient(vx_interpolant, x_coord_values[i], y_coord_values[j])[2]
            dvdx = Interpolations.gradient(vy_interpolant, x_coord_values[i], y_coord_values[j])[1]

            vorticity_grid[j, i] = dvdx-dudy
        end
    end

    return vorticity_grid

end

function smooth_velocity_fields(vx_grid::AbstractMatrix, vy_grid::AbstractMatrix; sigma::Float64=1.5)
    # Apply Gaussian smoothing to velocity fields using ImageFiltering.jl
    # This is more efficient and accurate than custom implementations

    kernel = ImageFiltering.Kernel.gaussian((sigma, sigma))

    vx_smoothed = imfilter(vx_grid, kernel, "reflect")
    vy_smoothed = imfilter(vy_grid, kernel, "reflect")

    return vx_smoothed, vy_smoothed
end

function interpolate_fluid_q_criterion_grid(fvm_grid::FVMGrid,
    v::AbstractVector,
    x_coord_values::AbstractVector,
    y_coord_values::AbstractVector;
    smooth::Bool=false,
    smooth_sigma::Float64=1.5
)
    # Compute Q-criterion for vortex identification
    # Q = 0.5 * (||Ω||² - ||S||²)
    # where Ω is the vorticity tensor and S is the strain rate tensor
    # In 2D: Q = (∂v/∂x)(∂u/∂y) - (∂u/∂x)(∂v/∂y) - 0.5*((∂u/∂x)² + (∂v/∂y)²)
    # Positive Q indicates regions where rotation dominates strain (vortices)

    # Get velocity grids
    vx_grid, vy_grid = create_fluid_velocity_grid(fvm_grid, v)

    # Apply smoothing to velocity fields if requested
    if smooth
        vx_grid, vy_grid = smooth_velocity_fields(vx_grid, vy_grid; sigma=smooth_sigma)
    end

    # Create interpolants from (possibly smoothed) velocity grids
    vx_interpolant = create_grid_interpolant(vx_grid,
        fvm_grid.x_coord_vx_values,
        fvm_grid.y_coord_vx_values
    )

    vy_interpolant = create_grid_interpolant(vy_grid,
        fvm_grid.x_coord_vy_values,
        fvm_grid.y_coord_vy_values
    )

    q_criterion_grid = interpolate_fluid_q_criterion_grid(vx_interpolant, vy_interpolant, x_coord_values, y_coord_values)

    return q_criterion_grid

end

function interpolate_fluid_q_criterion_grid(vx_interpolant::AbstractInterpolation,
    vy_interpolant::AbstractInterpolation,
    x_coord_values::AbstractVector,
    y_coord_values::AbstractVector
)
    # Compute Q-criterion from velocity interpolants
    # Q = (∂v/∂x)(∂u/∂y) - (∂u/∂x)(∂v/∂y) - 0.5*((∂u/∂x)² + (∂v/∂y)²)

    q_criterion_grid = zeros(length(y_coord_values), length(x_coord_values))

    for i in eachindex(x_coord_values)
        for j in eachindex(y_coord_values)
            # Compute velocity gradients
            # gradient returns [∂/∂x, ∂/∂y]
            dudx = Interpolations.gradient(vx_interpolant, x_coord_values[i], y_coord_values[j])[1]
            dudy = Interpolations.gradient(vx_interpolant, x_coord_values[i], y_coord_values[j])[2]
            dvdx = Interpolations.gradient(vy_interpolant, x_coord_values[i], y_coord_values[j])[1]
            dvdy = Interpolations.gradient(vy_interpolant, x_coord_values[i], y_coord_values[j])[2]

            # Q-criterion: rotation dominance over strain
            # Q = 0.5 * (||Ω||² - ||S||²)
            # In 2D:
            # Ω = [0, 0.5*(∂v/∂x - ∂u/∂y); -0.5*(∂v/∂x - ∂u/∂y), 0]
            # S = [∂u/∂x, 0.5*(∂u/∂y + ∂v/∂x); 0.5*(∂u/∂y + ∂v/∂x), ∂v/∂y]
            # ||Ω||² = 0.5*(∂v/∂x - ∂u/∂y)²
            # ||S||² = (∂u/∂x)² + (∂v/∂y)² + 0.5*(∂u/∂y + ∂v/∂x)²

            omega_squared = 0.5 * (dvdx - dudy)^2
            strain_squared = dudx^2 + dvdy^2 + 0.5 * (dudy + dvdx)^2

            q_criterion_grid[j, i] = 0.5 * (omega_squared - strain_squared)
        end
    end

    return q_criterion_grid

end

function create_trajectory_interpolant(t_hist::AbstractVector,
    value_hist::VecOrMat{<:AbstractVector}
)

    return linear_interpolation(t_hist, value_hist)

end

function stack_states(x_hist::VecOrMat{<:VecOrMat{<:AbstractVector}})

    N = length(x_hist[1])

    return [[x[k] for x in x_hist] for k in 1:N]

end