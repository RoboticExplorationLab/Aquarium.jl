struct Fluid{T<:Real, S<:Real}

    # Differentiable fluid properties
    density::S
    dynamic_viscosity::S
    boundary_velocity::Vector{S}

    # Other fluid properties
    time_step::T
    gravity_constant::T
    cell_mass::S
    cell_area::T
    external_pressure_gradient::Tuple{T,T}

    # Gravitational and pressure-gradient force vectors
    gravitational_acceleration::Vector{T}
    external_pressure_gradient_force::Vector{T}

    # FVM grid properties
    fvm_grid::FVMGrid

    # FVM operators
    constant_boundary_condition_matrix::SparseMatrixCSC{T, Int}
    constant_boundary_condition_vector::Vector{S}
    laplacian::SparseMatrixCSC{T, Int}
    original_divergence::SparseMatrixCSC{T, Int}
    divergence::SparseMatrixCSC{T, Int}
    continuity_vector::Vector{S}
    midpoint_operators::NTuple{6, SparseMatrixCSC{T, Int}}

    # boundary condition properties
    boundary_condition_type::Symbol
    n_boundary_conditions::Int

    # indices pointing to different parts of the state vector
    state_indices::Vector{Int}
    dual_indices::Vector{Int}
    velocity_indices::Vector{Int}
    continuity_dual_indices::Vector{Int}

    # number of fluid states
    n_states::Int
    n_constraints::Int
    n_velocities::Int
    n_continuity_constraints::Int

end

function Fluid(time_step::T;
    density::S,
    dynamic_viscosity::S,
    boundary_velocity::AbstractVector{S},
    grid_size::Tuple{Int,Int},
    grid_dimensions::Tuple{T,T},
    boundary_condition_type::Symbol,
    gravity_constant::T=zero(T),
    external_pressure_gradient::Tuple{T,T}=(zero(T), zero(T))
) where {T<:Real, S<:Real}

    boundary_velocity = collect(boundary_velocity)

    # Calculate grid properties
    fvm_grid = FVMGrid(grid_size, grid_dimensions)

    # Calculate cell mass
    cell_mass = density * fvm_grid.h_x * fvm_grid.h_y

    # Calculate cell area
    cell_area = fvm_grid.h_x * fvm_grid.h_y

    # Calculate FVM operators and boundary conditions
    constant_boundary_condition_matrix = calculate_constant_boundary_condition_operator(
        fvm_grid, boundary_condition_type
    )
    constant_boundary_condition_vector = calculate_constant_boundary_condition_vector(
        fvm_grid,
        boundary_velocity,
        boundary_condition_type
    )

    laplacian = calculate_laplacian_operator(fvm_grid)
    original_divergence = calculate_divergence_operator(fvm_grid)
    
    # For freestream and channel flow boundary conditions, remove divergence constraint at outlet (rightmost) cells
    if boundary_condition_type in (:freestream, :channel_flow_theoretical, :channel_flow_uniform)
        divergence = copy(original_divergence)
        constant_boundary_indices = setdiff(
            fvm_grid.v_boundary_indices,
            vcat(fvm_grid.vx_right_indices, fvm_grid.vy_right_indices)
        )
        continuity_vector = original_divergence * constant_boundary_condition_vector
        divergence[:, constant_boundary_indices] .= 0.0
        dropzeros!(divergence)
    else
        divergence = copy(original_divergence)
        continuity_vector = original_divergence * constant_boundary_condition_vector
        divergence[:, fvm_grid.v_boundary_indices] .= 0.0
        dropzeros!(divergence)
    end
    
    midpoint_operators = calculate_midpoint_operators(fvm_grid)

    # Calculate gravitational and pressure-gradient force vectors - only apply to interior velocities
    gravitational_acceleration = zeros(T, fvm_grid.n_v)
    gravitational_acceleration[fvm_grid.v_interior_indices] .= vcat(
        zeros(T, fvm_grid.n_vx),
        -(gravity_constant) .* ones(T, fvm_grid.n_vy)
    )[fvm_grid.v_interior_indices]

    external_pressure_gradient_force = zeros(T, fvm_grid.n_v)
    external_pressure_gradient_force[fvm_grid.v_interior_indices] .= vcat(
        (-external_pressure_gradient[1]) .* ones(T, fvm_grid.n_vx),
        (-external_pressure_gradient[2]) .* ones(T, fvm_grid.n_vy)
    )[fvm_grid.v_interior_indices]

    # Calculate number of state variables
    n_velocities = fvm_grid.n_v
    n_continuity_constraints = size(divergence, 1)

    # Calculate boundary condition count
    n_boundary_conditions = fvm_grid.n_v_boundary

    # All boundary conditions now embedded in stationarity, no separate BC constraints
    n_constraints = n_continuity_constraints
    n_states = n_velocities + n_constraints

    # Calculate fluid state indices
    state_indices = collect(1 : n_states)
    velocity_indices = collect(1 : n_velocities)
    continuity_dual_indices = collect(n_velocities+1 : n_velocities+n_continuity_constraints)
    dual_indices = collect(n_velocities+1 : n_states)

    return Fluid{T, S}(
        density,
        dynamic_viscosity,
        boundary_velocity,
        time_step,
        gravity_constant,
        cell_mass,
        cell_area,
        external_pressure_gradient,
        gravitational_acceleration,
        external_pressure_gradient_force,
        fvm_grid,
        constant_boundary_condition_matrix,
        constant_boundary_condition_vector,
        laplacian,
        original_divergence,
        divergence,
        continuity_vector,
        midpoint_operators,
        boundary_condition_type,
        n_boundary_conditions,
        state_indices,
        dual_indices,
        velocity_indices,
        continuity_dual_indices,
        n_states,
        n_constraints,
        n_velocities,
        n_continuity_constraints
    )

end

#############################################################################################
## Potential energy
#############################################################################################

function calculate_potential_energy(fluid::Fluid;
    density=fluid.density
)

    cell_mass = density * fluid.fvm_grid.h_x * fluid.fvm_grid.h_y
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat
    
    return sum(cell_mass * fluid.gravity_constant * y_coord_fluid_velocity_y)

end

#############################################################################################
## Kinetic energy
#############################################################################################

function calculate_kinetic_energy(fluid::Fluid,
    fluid_velocity_or_state::AbstractVector)

    # Incase the full fluid state vector is passed, extract fluid velocities
    fluid_velocity = fluid_velocity_or_state[fluid.velocity_indices]

    return 0.5 * sum(fluid.cell_mass .* (fluid_velocity .^ 2))
    
end

#############################################################################################
## Total energy
#############################################################################################

function calculate_total_energy(fluid::Fluid, fluid_velocity_or_state::AbstractVector)
    
    potential_energy = calculate_potential_energy(fluid)
    kinetic_energy = calculate_kinetic_energy(fluid, fluid_velocity_or_state)
    
    return potential_energy + kinetic_energy
end

#############################################################################################
## Nonlinear convective term and jacobian
#############################################################################################

function calculate_convective_term(fluid::Fluid,
    fluid_velocity::AbstractVector
)

    m1, m2, m3, m4, m5, m6 = fluid.midpoint_operators

    N1 = m1*fluid_velocity
    N2 = m2*fluid_velocity
    N3 = m3*fluid_velocity
    N4 = m4*fluid_velocity
    N5 = m5*fluid_velocity
    N6 = m6*fluid_velocity
    
    convective = (N1.*N1 - N2.*N2) + (N3.*N4 - N5.*N6)

    return convective
    
end

function calculate_convective_jacobian(fluid::Fluid,
    v::AbstractVector
)
    @inbounds begin
        m1, m2, m3, m4, m5, m6 = fluid.midpoint_operators
        N1 = m1 * v
        N2 = m2 * v
        N3 = m3 * v
        N4 = m4 * v
        N5 = m5 * v
        N6 = m6 * v

        ∂convective_∂v = (Diagonal(N1) * m1 * 2 - Diagonal(N2) * m2 * 2) +
            (Diagonal(N3) * m4 + Diagonal(N4) * m3) -
            (Diagonal(N5) * m6 + Diagonal(N6) * m5)
    end
    return ∂convective_∂v

end

#############################################################################################
## Conservation-of-mass constraint
#############################################################################################

function calculate_mass_conservation_constraint_residual(fluid::Fluid,
    fluid_velocity_kp1::AbstractVector;
    boundary_velocity::AbstractVector=fluid.boundary_velocity,
    recompute_bc_vector::Bool=false
)

    if recompute_bc_vector
        constant_boundary_condition_vector = calculate_constant_boundary_condition_vector(
            fluid.fvm_grid,
            boundary_velocity,
            fluid.boundary_condition_type
        )
        continuity_vector = fluid.original_divergence * constant_boundary_condition_vector
    else
        continuity_vector = fluid.continuity_vector
    end

    com_residual = fluid.divergence*fluid_velocity_kp1 + continuity_vector
    return com_residual

end
function calculate_mass_conservation_constraint_jacobian(fluid::Fluid)

    ∂constant_boundary_condition_vector_∂boundary_velocity =
        calculate_constant_boundary_condition_vector_jacobian(
            fluid.fvm_grid,
            fluid.boundary_condition_type
        )

    ∂continuity_vector_∂boundary_velocity =
        fluid.original_divergence * ∂constant_boundary_condition_vector_∂boundary_velocity

    return fluid.divergence, ∂continuity_vector_∂boundary_velocity

end

#############################################################################################
## Boundary-condition constraints
#############################################################################################

function calculate_boundary_condition_constraint_residual(fluid::Fluid,
    fluid_velocity_kp1::AbstractVector,
    fluid_velocity_k::AbstractVector;
    boundary_velocity::AbstractVector=fluid.boundary_velocity,
    recompute_bc_vector::Bool=false
)

    # Promote type to handle ForwardDiff Dual numbers from both velocity_kp1 and velocity_k
    T = promote_type(eltype(fluid_velocity_kp1), eltype(fluid_velocity_k), eltype(boundary_velocity))

    boundary_condition_matrix = fluid.constant_boundary_condition_matrix

    if recompute_bc_vector
        boundary_condition_vector = calculate_constant_boundary_condition_vector(
            fluid.fvm_grid,
            boundary_velocity,
            fluid.boundary_condition_type
        )
    else
        boundary_condition_vector = fluid.constant_boundary_condition_vector
    end

    # Compute BC residual: matrix * velocity - vector
    # Create with promoted type T to ensure compatibility with both velocity inputs
    boundary_condition_residual = Vector{T}(boundary_condition_matrix * fluid_velocity_kp1 - boundary_condition_vector)

    # For freestream and channel flow BC, replace outflow residual at right boundary indices
    if fluid.boundary_condition_type in (:freestream, :channel_flow_theoretical, :channel_flow_uniform)

        outflow_residual = calculate_outflow_boundary_condition_residual(
            fluid,
            fluid_velocity_kp1,
            fluid_velocity_k;
            boundary_velocity=boundary_velocity,
        )

        outflow_indices = vcat(
            fluid.fvm_grid.vx_right_indices,
            fluid.fvm_grid.vy_right_indices
        )

        # Now assignment will work because boundary_condition_residual has the promoted type
        boundary_condition_residual[outflow_indices] = outflow_residual

    end

    return boundary_condition_residual

end
function calculate_boundary_condition_constraint_jacobian(fluid::Fluid,
    fluid_velocity_kp1::AbstractVector,
    fluid_velocity_k::AbstractVector
)

    constant_boundary_condition_matrix = fluid.constant_boundary_condition_matrix

    ∂constant_boundary_condition_vector_∂boundary_velocity =
        calculate_constant_boundary_condition_vector_jacobian(
            fluid.fvm_grid,
            fluid.boundary_condition_type
        )

    # Initialize Jacobians
    ∂boundary_condition_∂fluid_velocity_kp1 = constant_boundary_condition_matrix
    ∂boundary_condition_∂fluid_velocity_k = spzeros(fluid.n_velocities, fluid.n_velocities)
    ∂boundary_condition_∂boundary_velocity = -∂constant_boundary_condition_vector_∂boundary_velocity

    # For freestream and channel flow BC, add outflow Jacobian at right boundary indices
    if fluid.boundary_condition_type in (:freestream, :channel_flow_theoretical, :channel_flow_uniform)
        ∂outflow_∂velocity_kp1, ∂outflow_∂velocity_k, ∂outflow_∂boundary_velocity =
            calculate_outflow_boundary_condition_jacobian(
                fluid,
                fluid_velocity_kp1,
                fluid_velocity_k
            )

        # Insert outflow Jacobian rows at the right boundary indices
        outflow_indices = vcat(
            fluid.fvm_grid.vx_right_indices,
            fluid.fvm_grid.vy_right_indices
        )

        ∂boundary_condition_∂fluid_velocity_kp1[outflow_indices, :] = ∂outflow_∂velocity_kp1
        ∂boundary_condition_∂fluid_velocity_k[outflow_indices, :] = ∂outflow_∂velocity_k
        # The outflow BC residual REPLACES the default BC residual at outflow
        # rows (see calculate_boundary_condition_constraint_residual), so its
        # derivative wrt boundary_velocity must also replace — not add to —
        # the default. Overwrite the relevant rows.
        ∂boundary_condition_∂boundary_velocity[outflow_indices, :] = ∂outflow_∂boundary_velocity

    end

    return ∂boundary_condition_∂fluid_velocity_kp1,
        ∂boundary_condition_∂fluid_velocity_k,
        ∂boundary_condition_∂boundary_velocity

end

function calculate_outflow_boundary_condition_residual(fluid::Fluid,
    velocity_kp1::AbstractVector,
    velocity_k::AbstractVector;
    boundary_velocity::AbstractVector=fluid.boundary_velocity,
)

    # Extract fluid properties
    fvm_grid = fluid.fvm_grid
    time_step = fluid.time_step
    n_velocities = fluid.n_velocities

    # Extract grid properties
    h_x = fvm_grid.h_x
    vx_right_indices = fvm_grid.vx_right_indices
    vy_right_indices = fvm_grid.vy_right_indices
    n_outflow = length(vx_right_indices) + length(vy_right_indices)
    outflow_boundary_indices = vcat(vx_right_indices,
        vy_right_indices
    )
    outflow_boundary_adjacent_indices =
        vcat(
            vx_right_indices .- fvm_grid.n_vx_y,
            vy_right_indices .- fvm_grid.n_vy_y
        )

    # Use constant freestream velocity as convection speed for outflow BC
    # This avoids vortex deceleration caused by local velocity variations and max(v, 0) clamping
    freestream_vx = max(boundary_velocity[1], zero(eltype(boundary_velocity)))
    T_vel = promote_type(eltype(velocity_k), eltype(boundary_velocity))
    velocity_local_velocity_x = freestream_vx .* ones(T_vel, n_outflow)

    # Construct outflow boundary condition matrix
    outflow_bc_matrix_1 = sparse(1:n_outflow,
        outflow_boundary_indices,
        ones(n_outflow) + time_step .* velocity_local_velocity_x ./ h_x,
        n_outflow, n_velocities
    )
    outflow_bc_matrix_2 = sparse(1:n_outflow,
        outflow_boundary_adjacent_indices,
        -time_step .* velocity_local_velocity_x ./ h_x,
        n_outflow, n_velocities
    )
    outflow_bc_matrix = outflow_bc_matrix_1 + outflow_bc_matrix_2

    # Calculate residual
    outflow_residual = outflow_bc_matrix*velocity_kp1 -
        velocity_k[outflow_boundary_indices]

    return outflow_residual
end

function calculate_outflow_boundary_condition_jacobian(fluid::Fluid,
    velocity_kp1::AbstractVector,
    velocity_k::AbstractVector
)

    # Extract fluid properties
    fvm_grid = fluid.fvm_grid
    time_step = fluid.time_step
    n_v = fluid.n_velocities

    # Extract grid properties
    h_x = fvm_grid.h_x
    vx_right_indices = fvm_grid.vx_right_indices
    vy_right_indices = fvm_grid.vy_right_indices
    n_outflow = length(vx_right_indices) + length(vy_right_indices)
    outflow_boundary_indices = vcat(vx_right_indices, vy_right_indices)
    outflow_boundary_adjacent_indices = vcat(
        vx_right_indices .- fvm_grid.n_vx_y,
        vy_right_indices .- fvm_grid.n_vy_y
    )
    n_vx_outflow = length(vx_right_indices)

    # Use constant freestream velocity as convection speed for outflow BC
    freestream_vx = max(fluid.boundary_velocity[1], zero(eltype(fluid.boundary_velocity)))
    velocity_local_velocity_x = freestream_vx .* ones(eltype(velocity_k), n_outflow)

    # ∂residual/∂velocity_kp1
    outflow_bc_matrix_1 = sparse(1:n_outflow,
        outflow_boundary_indices,
        ones(n_outflow) + time_step .* velocity_local_velocity_x ./ h_x,
        n_outflow, n_v
    )
    outflow_bc_matrix_2 = sparse(1:n_outflow,
        outflow_boundary_adjacent_indices,
        -time_step .* velocity_local_velocity_x ./ h_x,
        n_outflow, n_v
    )
    ∂outflow_∂velocity_kp1 = outflow_bc_matrix_1 + outflow_bc_matrix_2

    # ∂residual/∂velocity_k
    # Only contribution: -velocity_k[outflow_boundary_indices] → -I on diagonal
    # No ∂(matrix coefficients)/∂velocity_k since convection speed is constant freestream
    ∂outflow_∂velocity_k = sparse(1:n_outflow,
        outflow_boundary_indices,
        -ones(eltype(velocity_k), n_outflow),
        n_outflow, n_v
    )

    # ∂residual/∂boundary_velocity
    # The outflow BC matrices carry a dt*freestream_vx/h_x factor, where
    # freestream_vx = max(boundary_velocity[1], 0). For boundary_velocity[1] > 0
    # the derivative d(freestream_vx)/d(boundary_velocity[1]) = 1, giving a
    # nonzero column for bvx; for bvx <= 0, the max-clamp makes the derivative
    # zero. bvy never enters the outflow BC, so its column is zero.
    # Derivation (for bvx > 0):
    #   residual[i] = (1 + dt*bvx/h_x) * velocity_kp1[bound[i]]
    #               +     (-dt*bvx/h_x) * velocity_kp1[adjacent[i]]
    #               - velocity_k[bound[i]]
    #   ∂residual[i]/∂bvx = (dt/h_x) * (velocity_kp1[bound[i]] - velocity_kp1[adjacent[i]])
    ∂outflow_∂boundary_velocity = spzeros(eltype(velocity_k), n_outflow, 2)
    if fluid.boundary_velocity[1] > 0
        for i in 1:n_outflow
            ∂outflow_∂boundary_velocity[i, 1] = (time_step / h_x) * (
                velocity_kp1[outflow_boundary_indices[i]] -
                velocity_kp1[outflow_boundary_adjacent_indices[i]]
            )
        end
    end

    return ∂outflow_∂velocity_kp1, ∂outflow_∂velocity_k, ∂outflow_∂boundary_velocity

end

#############################################################################################
## Stationarity condition and its Jacobian
# corresponding to conservation-of-momentum part of Navier Stokes
#############################################################################################

function calculate_fluid_stationarity_residual(fluid::Fluid,
    fluid_state_kp1::AbstractVector,
    fluid_state_k::AbstractVector;
    recompute_bc_vector::Bool=false
)

    # Extract components from state_kp1
    fluid_velocity_kp1 = fluid_state_kp1[fluid.velocity_indices]
    continuity_dual_kp1 = fluid_state_kp1[fluid.continuity_dual_indices]

    # Extract components from state_k
    fluid_velocity_k = fluid_state_k[fluid.velocity_indices]

    # Extract variables
    fluid_density = fluid.density
    dynamic_viscosity = fluid.dynamic_viscosity
    boundary_velocity = fluid.boundary_velocity
    time_step = fluid.time_step

    # Calculate density matrix (without cell volume) - only apply to interior velocities
    interior_indices = fluid.fvm_grid.v_interior_indices
    density_matrix = sparse(interior_indices, interior_indices,
                           fill(fluid_density, length(interior_indices)),
                           fluid.fvm_grid.n_v, fluid.fvm_grid.n_v)

    # Extract fluid operators
    laplacian = fluid.laplacian
    divergence = fluid.divergence

    # Calculate gravitational force - only applies to interior velocities (already zeroed at boundaries)
    gravitational_force = fluid_density .* fluid.gravitational_acceleration

    # Calculate boundary condition residual
    boundary_condition_residual = calculate_boundary_condition_constraint_residual(
        fluid,
        fluid_velocity_kp1,
        fluid_velocity_k;
        boundary_velocity=boundary_velocity,
        recompute_bc_vector=recompute_bc_vector
    )

    # calculate momentum component of stationarity residual
    fluid_stationarity_fluid_velocity_kp1 = (density_matrix - ((0.5*time_step*dynamic_viscosity) * laplacian))*fluid_velocity_kp1 +
        (0.5*time_step*fluid_density) * calculate_convective_term(fluid, fluid_velocity_kp1)

    fluid_stationarity_fluid_velocity_k = -(density_matrix + ((0.5*time_step*dynamic_viscosity) * laplacian))*fluid_velocity_k +
        (0.5*time_step*fluid_density) * calculate_convective_term(fluid, fluid_velocity_k)

    fluid_stationarity_residual = fluid_stationarity_fluid_velocity_kp1 +
        fluid_stationarity_fluid_velocity_k +
        boundary_condition_residual +
        divergence'*continuity_dual_kp1 -
        time_step*gravitational_force -
        time_step*fluid.external_pressure_gradient_force

    return fluid_stationarity_residual

end
function calculate_fluid_stationarity_jacobian(fluid::Fluid,
    fluid_state_kp1::AbstractVector,
    fluid_state_k::AbstractVector,
)

    # Extract fluid-velocity components from states
    fluid_velocity_kp1 = fluid_state_kp1[fluid.velocity_indices]
    fluid_velocity_k = fluid_state_k[fluid.velocity_indices]

    # Extract variables
    fluid_density = fluid.density
    dynamic_viscosity = fluid.dynamic_viscosity
    boundary_velocity = fluid.boundary_velocity
    time_step = fluid.time_step

    # Calculate density matrix (without cell volume) - only apply to interior velocities
    interior_indices = fluid.fvm_grid.v_interior_indices
    density_matrix = sparse(interior_indices, interior_indices,
                           fill(fluid_density, length(interior_indices)),
                           fluid.fvm_grid.n_v, fluid.fvm_grid.n_v)

    # Extract operators
    laplacian = fluid.laplacian

    # Calculate base Jacobian for momentum (without BCs)
    ∂momentum_∂fluid_velocity_kp1 = (density_matrix - ((0.5*time_step*dynamic_viscosity) * laplacian)) +
        (0.5*time_step*fluid_density) * calculate_convective_jacobian(fluid, fluid_velocity_kp1)
    ∂momentum_∂fluid_velocity_k = -(density_matrix + ((0.5*time_step*dynamic_viscosity) * laplacian)) +
        (0.5*time_step*fluid_density) * calculate_convective_jacobian(fluid, fluid_velocity_k)

    # Calculate boundary condition Jacobian
    ∂bc_∂fluid_velocity_kp1, ∂bc_∂fluid_velocity_k, ∂bc_∂boundary_velocity = calculate_boundary_condition_constraint_jacobian(
        fluid,
        fluid_velocity_kp1,
        fluid_velocity_k
    )

    # Add boundary condition Jacobian to momentum Jacobian
    ∂stationarity_∂fluid_velocity_kp1 = ∂momentum_∂fluid_velocity_kp1 + ∂bc_∂fluid_velocity_kp1
    ∂stationarity_∂fluid_velocity_k = ∂momentum_∂fluid_velocity_k + ∂bc_∂fluid_velocity_k

    # Calculate Jacobian corresponding to continuity dual (pressure gradient)
    ∂mass_conservation_∂velocity, _ = calculate_mass_conservation_constraint_jacobian(fluid)
    ∂stationarity_∂continuity_dual_kp1 = ∂mass_conservation_∂velocity'

    # Calculate derivatives corresponding to fluid density - only for interior velocities
    # At boundary indices, density doesn't affect the stationarity (BC residual dominates)
    ∂stationarity_kp1_∂fluid_density = zeros(typeof(fluid_density), fluid.fvm_grid.n_v)
    ∂stationarity_kp1_∂fluid_density[interior_indices] = fluid_velocity_kp1[interior_indices] +
        (0.5*time_step) * calculate_convective_term(fluid, fluid_velocity_kp1)[interior_indices]

    ∂stationarity_k_∂fluid_density = zeros(typeof(fluid_density), fluid.fvm_grid.n_v)
    ∂stationarity_k_∂fluid_density[interior_indices] = -fluid_velocity_k[interior_indices] +
        (0.5*time_step) * calculate_convective_term(fluid, fluid_velocity_k)[interior_indices]

    ∂stationarity_∂fluid_density = ∂stationarity_kp1_∂fluid_density +
        ∂stationarity_k_∂fluid_density -
        time_step.*fluid.gravitational_acceleration

    # Calculate derivatives corresponding to dynamic viscosity
    ∂stationarity_∂dynamic_viscosity = -(0.5*time_step) .* (laplacian*fluid_velocity_kp1 +
        laplacian*fluid_velocity_k
    )

    # Combine all contributions to get final fluid state Jacobian using block construction
    ∂stationarity_∂fluid_state_kp1 = [
        ∂stationarity_∂fluid_velocity_kp1 ∂stationarity_∂continuity_dual_kp1
    ]
    ∂stationarity_∂fluid_state_k = [
        ∂stationarity_∂fluid_velocity_k spzeros(fluid.n_velocities, fluid.n_continuity_constraints)
    ]

    # Combine all contributions to get final fluid properties Jacobian using block construction
    # fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    ∂stationarity_∂fluid_properties = [
        ∂stationarity_∂fluid_density ∂stationarity_∂dynamic_viscosity ∂bc_∂boundary_velocity
    ]

    return ∂stationarity_∂fluid_state_kp1,
        ∂stationarity_∂fluid_state_k,
        ∂stationarity_∂fluid_properties

end

#############################################################################################
## Fluid-dynamics residual and Jacobian
#############################################################################################

function calculate_fluid_dynamics_residual(fluid::Fluid,
    fluid_state_kp1::AbstractVector,
    fluid_state_k::AbstractVector;
    recompute_bc_vector::Bool=false
)

    fluid_velocity_kp1 = fluid_state_kp1[fluid.velocity_indices]

    fluid_stationarity_residual = calculate_fluid_stationarity_residual(fluid,
        fluid_state_kp1,
        fluid_state_k;
        recompute_bc_vector=recompute_bc_vector
    )
    com_residual = calculate_mass_conservation_constraint_residual(fluid,
        fluid_velocity_kp1;
        boundary_velocity=fluid.boundary_velocity,
        recompute_bc_vector=recompute_bc_vector
    )

    # All boundary conditions now embedded in stationarity, no separate BC constraints
    fluid_residual = vcat(fluid_stationarity_residual, com_residual)

    return fluid_residual

end

function calculate_fluid_dynamics_jacobian(fluid::Fluid{T},
    fluid_state_kp1::AbstractVector,
    fluid_state_k::AbstractVector,
) where T

    # Extract components from state_kp1
    fluid_velocity_kp1 = fluid_state_kp1[fluid.velocity_indices]

    # Extract components from state_k
    fluid_velocity_k = fluid_state_k[fluid.velocity_indices]

    # extract number of components
    n_fluid_state = fluid.n_states
    n_fluid_velocities = fluid.n_velocities
    n_continuity_constraints = fluid.n_continuity_constraints
    n_boundary_conditions = fluid.n_boundary_conditions

    # Calculate stationarity Jacobian
    ∂stationarity_∂fluid_state_kp1,
    ∂stationarity_∂fluid_state_k,
    ∂stationarity_∂fluid_properties = calculate_fluid_stationarity_jacobian(fluid,
        fluid_state_kp1,
        fluid_state_k,
    )

    # Calculate mass conservation Jacobian using block construction
    ∂com_∂fluid_velocity_kp1, ∂com_∂boundary_velocity = calculate_mass_conservation_constraint_jacobian(fluid)
    ∂com_∂fluid_state_kp1 = [
        ∂com_∂fluid_velocity_kp1 spdiagm(zeros(n_continuity_constraints))
    ]

    # All boundary conditions now embedded in stationarity, no separate BC constraint Jacobian

    # Assemble full dynamics Jacobian using block construction
    ∂dynamics_∂fluid_state_kp1 = [
        ∂stationarity_∂fluid_state_kp1;
        ∂com_∂fluid_state_kp1
    ]
    ∂dynamics_∂fluid_state_k = [
        ∂stationarity_∂fluid_state_k;
        spzeros(n_continuity_constraints, n_fluid_state)
    ]
    ∂dynamics_∂fluid_properties = [
        ∂stationarity_∂fluid_properties;
        spzeros(n_continuity_constraints, 2) ∂com_∂boundary_velocity
    ]

    return ∂dynamics_∂fluid_state_kp1,
        ∂dynamics_∂fluid_state_k,
        ∂dynamics_∂fluid_properties

end

#############################################################################################
## Simulate fluid dynamics
#############################################################################################

function create_freestream_fluid_velocity(fluid::Fluid)

    fluid_velocity_freestream = vcat(
        fluid.boundary_velocity[1] .* ones(fluid.fvm_grid.n_vx),
        fluid.boundary_velocity[2] .* ones(fluid.fvm_grid.n_vy)
    )

    return fluid_velocity_freestream

end

function initialize_fluid_state(fluid::Fluid, fluid_velocity_0::AbstractVector)

    fluid_state_0 = vcat(
        fluid_velocity_0,
        zeros(fluid.n_constraints)
    )

    return fluid_state_0

end

function simulate_fluid(fluid::Fluid,
    fluid_state_0::AbstractVector,
    final_time;
    pivot_type::Symbol=:rcm,
    scaling_type::Symbol=:ruiz,
    solver_type=:gmres,
    preconditioner_type=:ilu,
    lazy::Bool=true,
    n_pardiso_threads::Int=Sys.CPU_THREADS,
    max_newton_iterations::Int=10,
    newton_tolerance::Float64=1e-6,
    ilu_drop_tolerance::Float64=1e-6,
    amg_smoother_type::Symbol=:symmetric_gs,
    gmres_tolerance::Float64=1e-6,
    gmres_memory::Int=50,
    gmres_max_iterations::Int=500,
    dual_regularization=typeof(final_time)(1e-6),
    verbose=false,
    # Objective calculation options
    calculate_objective::Bool=false,
    gradient_method::Symbol=:forward,
    # Objective functions
    calculate_stage_objective::Function = (fluid, time, fluid_state) -> 0.0,
    calculate_terminal_objective::Function = (fluid, time, fluid_state) -> 0.0,
    # Gradients of objectives w.r.t. fluid state and fluid properties
    calculate_stage_objective_gradients::Function = (fluid, time, fluid_state;
        fluid_params::AbstractVector = collect_differentiable_params(fluid),
        rebuild_fluid::Function = p -> inject_differentiable_params(fluid, p),
    ) -> (
        ForwardDiff.gradient(x -> calculate_stage_objective(fluid, time, x), fluid_state),
        ForwardDiff.gradient(fluid_params) do p
            calculate_stage_objective(rebuild_fluid(p), time, fluid_state)
        end,
    ),
    calculate_terminal_objective_gradients::Function = (fluid, time, fluid_state;
        fluid_params::AbstractVector = collect_differentiable_params(fluid),
        rebuild_fluid::Function = p -> inject_differentiable_params(fluid, p),
    ) -> (
        ForwardDiff.gradient(x -> calculate_terminal_objective(fluid, time, x), fluid_state),
        ForwardDiff.gradient(fluid_params) do p
            calculate_terminal_objective(rebuild_fluid(p), time, fluid_state)
        end,
    ),
    initial_fluid_state_fluid_properties_jacobian::AbstractArray = zeros(length(fluid_state_0), 4)
)

    # Validate solver and preconditioner types
    valid_solver_types = (:pardiso, :mumps, :gmres, :backslash)
    valid_preconditioner_types = (:none, :ilu, :ilu0, :pardiso, :amg,
                                   :approx_schur_ilu,
                                   :approx_schur_ilu0,
                                   :approx_schur_partial_amg,
                                   :approx_schur_full_amg)

    if !(solver_type in valid_solver_types)
        solver_type = :gmres
        @warn "Invalid solver_type specified. Using default GMRES instead."
    end

    if solver_type in (:pardiso, :mumps, :backslash)
        preconditioner_type = :none
        lazy=false
        @warn "No preconditioning or lazy with Pardiso, MUMPS or backslash. Setting preconditioner_type to :none."
    elseif !(preconditioner_type in valid_preconditioner_types)
        preconditioner_type = :ilu
        @warn "Invalid preconditioner_type specified. Using default ILU instead."
    end
    
    if verbose
        print("Setting up fluid simulation...")
    end
    
    # determine knot points
    time_step = fluid.time_step
    N_time = Int(final_time/time_step + 1)

    # extract number of fluid states and cells
    n_fluid_state = fluid.n_states

    # initialize trajory
    t_traj = Vector(LinRange(0, final_time, N_time))
    fluid_state_traj = [deepcopy(fluid_state_0) for k = 1:N_time]
    
    # initialize solution vector
    solution_vector = rand(n_fluid_state)

    # Create initial Jacobian for solver initialization
    kkt_rand, _, _ = calculate_fluid_dynamics_jacobian(fluid,
        rand(n_fluid_state),
        rand(n_fluid_state)
    )

    # Create solver once before timestepping - can be reused for all timesteps
    solver = create_solver(kkt_rand, solution_vector, solver_type;
        n_pardiso_threads=n_pardiso_threads,
        gmres_memory=gmres_memory
    )

    # For Schur complement preconditioning, the dimension of the Schur complement is the number of constraints
    schur_dim = fluid.n_continuity_constraints

    # For GMRES with Pardiso preconditioner, create a separate Pardiso solver
    preconditioner = nothing
    preconditioner_solver = nothing

    if solver_type == :gmres && preconditioner_type == :pardiso
        if !PARDISO_LOADED[]
            error("Pardiso is not available. Please use a different preconditioner_type or install Pardiso on Linux/Windows.")
        end
        preconditioner_solver = create_solver(kkt_rand, solution_vector, :pardiso;
            n_pardiso_threads=n_pardiso_threads
        )
        lazy = true # Force lazy mode when using Pardiso preconditioner
    end

    # Initialize objective gradients
    if calculate_objective

        # initialize objective trajectory and cumulated value
        objective_trajectory = zeros(N_time)
        objective_value = 0.0

        # initialize objective gradients trajectory
        objective_gradient_wrt_fluid_properties_traj = [zeros(4) for k = 1:N_time]

        # Initialize with stage 1 contribution
        objective_trajectory[1] = calculate_stage_objective(
            fluid,
            t_traj[1],
            fluid_state_traj[1]
        )

        ∂stage_1_∂x_k, ∂stage_1_∂fluid_properties = calculate_stage_objective_gradients(
            fluid,
            t_traj[1],
            fluid_state_traj[1]
        )

        ∂fluid_state_k_∂fluid_properties = copy(initial_fluid_state_fluid_properties_jacobian)

        ∂objective_∂fluid_properties_gradient = (∂stage_1_∂x_k' * ∂fluid_state_k_∂fluid_properties + ∂stage_1_∂fluid_properties')[:]

        objective_gradient_wrt_fluid_properties_traj[1] = ∂objective_∂fluid_properties_gradient

        # Preallocate matrices for block solve (only fluid properties gradient)
        n_fluid_properties = 4
        if n_fluid_properties > 0
            B_fluid_combined = zeros(n_fluid_state, n_fluid_properties)
            X_fluid_combined = zeros(n_fluid_state, n_fluid_properties)
        end

    end

    if verbose
        println("Finished!")
    end

    # Print solver information
    if verbose
        println("Using solver type: $(solver_type)")
        println("Using preconditioner type: $(preconditioner_type)")
        println("Using pivot type: $(pivot_type)")
        println("Using scaling type: $(scaling_type)")
        println("Lazy preconditioner mode: $(lazy)")
        if solver_type == :pardiso || preconditioner_type == :pardiso
            println("Number of Pardiso threads: $(n_pardiso_threads)")
        end
        if solver_type == :mumps
            println("MUMPS solver initialized")
        end
        if preconditioner_type in (:approx_schur_partial_amg, :approx_schur_full_amg)
            println("AMG smoother type: $(amg_smoother_type)")
        end
    end

    @showprogress desc="Simulating fluid..." for k = 1:N_time-1

        newton_iter = 0
        fluid_state_traj[k+1] = copy(fluid_state_traj[k])

        if verbose
            println("")
            println("Time step: $(k)")
            println("")
            println("Newton iteration: $(newton_iter)")
            println("")
            print("Constructing KKT system...")
        end
        
        residual = calculate_fluid_dynamics_residual(fluid,
            fluid_state_traj[k+1],
            fluid_state_traj[k]
        )

        kkt_matrix, _, _ = calculate_fluid_dynamics_jacobian(fluid,
                fluid_state_traj[k+1],
                fluid_state_traj[k]
            )

        if verbose
            println(" Finished!")
        end

        # Compute scaling factors once (will be reused in Newton iterations)
        left_scale, right_scale = scale_linear_system!(kkt_matrix, residual;
            scaling_type=scaling_type, verbose=verbose
        )

        # Apply dual regularization if specified (using precomputed positions from setup)
        apply_regularization!(kkt_matrix;
            regularization_indices=fluid.dual_indices,
            regularization_value=dual_regularization,
            verbose=verbose
        )

        # Apply pivoting if specified
        permutation, inverse_permutation = pivot_linear_system!(kkt_matrix, residual;
            pivot_type=pivot_type, verbose=verbose
        )

        preconditioner = calculate_preconditioner(kkt_matrix,
            solution_vector, preconditioner_type;
            preconditioner_solver=preconditioner_solver,
            ilu_drop_tolerance=ilu_drop_tolerance,
            amg_smoother_type=amg_smoother_type,
            verbose=verbose,
            schur_dimension=schur_dim
        )

        # Use scaled residual for convergence check
        scaled_residual_norm = maximum(abs.(residual .* left_scale))
        while scaled_residual_norm > newton_tolerance && newton_iter < max_newton_iterations

            newton_iter += 1

            if newton_iter != 1

                if verbose
                    println("")
                    println("Newton iteration: $(newton_iter)")
                    println("")
                    print("Constructing KKT system...")
                end

                kkt_matrix, _, _ = calculate_fluid_dynamics_jacobian(fluid,
                    fluid_state_traj[k+1],
                    fluid_state_traj[k]
                )

                if verbose
                    println(" Finished!")
                end

                if lazy
                    # Reuse the scaling factors computed before the Newton loop
                    scale_linear_system!(kkt_matrix, residual, left_scale, right_scale; verbose=verbose)

                    # Apply dual regularization if specified (using precomputed positions)
                    apply_regularization!(kkt_matrix;
                        regularization_indices=fluid.dual_indices,
                        regularization_value=dual_regularization,
                        verbose=verbose
                    )

                    pivot_linear_system!(kkt_matrix, residual, permutation; verbose=verbose)
                else
                    # Recompute scaling, regularization, and pivoting for non-lazy mode
                    left_scale, right_scale = scale_linear_system!(kkt_matrix, residual;
                        scaling_type=scaling_type, verbose=verbose
                    )

                    # Apply dual regularization if specified (using precomputed positions)
                    apply_regularization!(kkt_matrix;
                        regularization_indices=fluid.dual_indices,
                        regularization_value=dual_regularization,
                        verbose=verbose
                    )

                    permutation, inverse_permutation = pivot_linear_system!(kkt_matrix, residual;
                        pivot_type=pivot_type, verbose=verbose
                    )

                    # Compute preconditioner for non-lazy methods
                    preconditioner = calculate_preconditioner(kkt_matrix,
                        solution_vector, preconditioner_type;
                        preconditioner_solver=preconditioner_solver,
                        ilu_drop_tolerance=ilu_drop_tolerance,
                        amg_smoother_type=amg_smoother_type,
                        verbose=verbose,
                        schur_dimension=schur_dim
                    )
                end

            end

            # Solve linear system
            linear_solve!(solution_vector, kkt_matrix,
                -residual, solver, solver_type;
                preconditioner=preconditioner,
                gmres_tolerance=gmres_tolerance,
                gmres_max_iterations=gmres_max_iterations,
                right_scale=right_scale,
                inverse_permutation=inverse_permutation,
                verbose=verbose
            )

            fluid_state_traj[k+1] .+= solution_vector

            residual = calculate_fluid_dynamics_residual(fluid,
                fluid_state_traj[k+1],
                fluid_state_traj[k]
            )

            # Compute scaled residual norm for convergence check
            scaled_residual_norm = maximum(abs.(residual .* left_scale))

            if verbose
                println("")
                println("\e[1mScaled residual norm: $(scaled_residual_norm)\e[0m")
            end

        end

        if calculate_objective && gradient_method == :forward

            if verbose
                print("Calculating objective gradients using forward mode...")
            end

            kkt_matrix,
            ∂fluid_dynamics_residual_∂fluid_state_k,
            ∂fluid_dynamics_residual_∂fluid_properties = calculate_fluid_dynamics_jacobian(
                fluid,
                fluid_state_traj[k+1],
                fluid_state_traj[k]
            )

            # Reuse the scaling factors computed before the Newton loop
            scale_linear_system_matrix!(kkt_matrix, left_scale, right_scale; verbose=verbose)

            # Apply dual regularization if specified (using precomputed positions)
            apply_regularization!(kkt_matrix;
                regularization_indices=fluid.dual_indices,
                regularization_value=dual_regularization,
                verbose=verbose
            )

            pivot_linear_system_matrix!(kkt_matrix, permutation; verbose=verbose)

            # Implicit function theorem
            gradient_residual_fluid_properties = ∂fluid_dynamics_residual_∂fluid_state_k * ∂fluid_state_k_∂fluid_properties +
                ∂fluid_dynamics_residual_∂fluid_properties

            # Solve linear system for sensitivities using preallocated matrices
            n_fluid_properties = size(∂fluid_state_k_∂fluid_properties, 2)
            if n_fluid_properties > 0

                # Fill preallocated B_fluid_combined
                B_fluid_combined .= -gradient_residual_fluid_properties

                scale_rhs_matrix!(B_fluid_combined, left_scale)
                pivot_rhs_matrix!(B_fluid_combined, permutation)

                block_linear_solve!(X_fluid_combined, kkt_matrix, B_fluid_combined, solver, solver_type;
                    preconditioner=preconditioner,
                    gmres_tolerance=gmres_tolerance,
                    gmres_max_iterations=gmres_max_iterations,
                    gmres_memory=gmres_memory,
                    right_scale=right_scale,
                    inverse_permutation=inverse_permutation,
                    reuse_factorization=true,
                    verbose=false
                )

                # Copy result back to ∂fluid_state_k_∂fluid_properties
                ∂fluid_state_k_∂fluid_properties .= X_fluid_combined
            end

            if k < N_time - 1

                # Compute stage objective value at next timestep k+1
                objective_trajectory[k+1] = calculate_stage_objective(
                    fluid,
                    t_traj[k+1],
                    fluid_state_traj[k+1]
                )

                # Compute stage objective gradients at next timestep k+1
                ∂stage_∂x_kp1, ∂stage_∂p_kp1 = calculate_stage_objective_gradients(
                    fluid,
                    t_traj[k+1],
                    fluid_state_traj[k+1]
                )

                objective_gradient_wrt_fluid_properties_traj[k+1] = (∂stage_∂x_kp1' * ∂fluid_state_k_∂fluid_properties + ∂stage_∂p_kp1')[:]

                # Accumulate gradients via chain rule (using the newly computed sensitivities)
                ∂objective_∂fluid_properties_gradient += objective_gradient_wrt_fluid_properties_traj[k+1]

            else

                # Compute terminal objective value
                objective_trajectory[k+1] = calculate_terminal_objective(
                    fluid,
                    t_traj[k+1],
                    fluid_state_traj[k+1]
                )

                # At final timestep: compute terminal objective gradient
                ∂terminal_∂x_final, ∂terminal_∂p_final = calculate_terminal_objective_gradients(
                    fluid,
                    t_traj[k+1],
                    fluid_state_traj[k+1]
                )

                # Accumulate terminal gradient
                ∂objective_∂fluid_properties_gradient += (∂terminal_∂x_final' * ∂fluid_state_k_∂fluid_properties + ∂terminal_∂p_final')[:]

            end
            
            if verbose
                println("Finished!")
                println("")
                println("\e[1mObjective value: $(objective_value)\e[0m")
                println("\e[1mObjective gradient w.r.t. fluid properties: $(∂objective_∂fluid_properties_gradient)\e[0m")
                println("")
            end

        end

    end

    # Cleanup Pardiso solver if used
    if PARDISO_LOADED[]
        if solver_type == :pardiso
            pardiso_set_phase!(solver, Val(:RELEASE_ALL))
            pardiso_solve!(solver)
        end
        if preconditioner_type == :pardiso
            pardiso_set_phase!(preconditioner_solver, Val(:RELEASE_ALL))
            pardiso_solve!(preconditioner_solver)
        end
    end

    if solver_type == :mumps
        finalize(solver)
        MPI.Finalize()
    end

    fluid_velocity_traj = [fluid_state_traj[k][fluid.velocity_indices] for k = 1:N_time]
    fluid_dual_traj = [fluid_state_traj[k][fluid.continuity_dual_indices] for k = 1:N_time]

    trajectories = Dict(
        :t_traj => t_traj,
        :fluid_state_traj => fluid_state_traj,
        :fluid_velocity_traj => fluid_velocity_traj,
        :fluid_pressure_traj => fluid_dual_traj,
    )

    if calculate_objective

        objective_value = sum(objective_trajectory)

        trajectories[:objective_value] = [objective_value]
        trajectories[:objective_traj] = objective_trajectory
        trajectories[:objective_gradient_wrt_fluid_properties_traj] = objective_gradient_wrt_fluid_properties_traj
        trajectories[:objective_gradient_wrt_fluid_properties] = ∂objective_∂fluid_properties_gradient
    
    end

    return trajectories

end

@testitem "Fluid keyword constructor — basic fields" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity = [1.0, 0.0]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = boundary_velocity,
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
    )

    @test fluid isa Fluid
    @test fluid.time_step == time_step
    @test fluid.density == density
    @test fluid.dynamic_viscosity == dynamic_viscosity
    @test fluid.boundary_velocity == boundary_velocity
    @test fluid.gravity_constant == gravity_constant
    @test fluid.boundary_condition_type == boundary_condition_type
end

@testitem "Fluid keyword constructor — derived quantities" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity = [1.0, 0.0]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = boundary_velocity,
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
    )

    h_x = grid_dimensions[1] / grid_size[1]
    h_y = grid_dimensions[2] / grid_size[2]
    @test fluid.cell_mass ≈ density * h_x * h_y
    @test fluid.cell_area ≈ h_x * h_y
    @test length(fluid.constant_boundary_condition_vector) == fluid.fvm_grid.n_v
    @test fluid.n_states == fluid.n_velocities + fluid.n_constraints
end

@testitem "Wall Boundary Conditions" begin
    using Aquarium
    using LinearAlgebra
    using ForwardDiff

    function create_test_fluid(;
        num_cells_x=10,
        num_cells_y=10,
        length_x=1.0,
        length_y=1.0,
        boundary_condition_type=:wall,
        boundary_velocity=[1.0, 2.0]
    )
        fluid_density = 1.0
        dynamic_viscosity = 0.01
        time_step = 0.01

        fluid = Fluid(
            time_step;
            density = fluid_density,
            dynamic_viscosity = dynamic_viscosity,
            boundary_velocity = boundary_velocity,
            grid_size = (num_cells_x, num_cells_y),
            grid_dimensions = (length_x, length_y),
            boundary_condition_type = boundary_condition_type,
        )

        return fluid
    end

    println("\nTesting Wall Boundary Conditions...")

    # Create fluid with wall boundary conditions
    fluid = create_test_fluid(boundary_condition_type=:wall)
    grid = fluid.fvm_grid

    # Test boundary conditions count
    @test fluid.n_boundary_conditions == grid.n_v_boundary
    println("✓ Wall boundary conditions count verified")

    # Test that boundary condition matrix has correct dimensions (now n_v × n_v)
    @test size(fluid.constant_boundary_condition_matrix) == (grid.n_v, grid.n_v)
    println("✓ Wall boundary matrix has correct dimensions (n_v × n_v)")

    # Construct the boundary condition vector explicitly (now n_v length)
    bc_vector = calculate_constant_boundary_condition_vector(
        grid, fluid.boundary_velocity, fluid.boundary_condition_type
    )
    @test length(bc_vector) == grid.n_v
    println("✓ Wall-boundary vector has correct length (n_v)")

    # Test that all boundary velocities should be zero for wall
    @test all(bc_vector .== 0.0)
    println("✓ Wall-boundary vector has all zeros")

    # Test that boundary condition matrix has identity on diagonal at boundaries
    bc_matrix = fluid.constant_boundary_condition_matrix
    @test all(bc_matrix[grid.v_boundary_indices, grid.v_boundary_indices] .== I(grid.n_v_boundary))
    @test all(bc_matrix[grid.v_interior_indices, :] .== 0.0)
    println("✓ Wall-boundary matrix has identity at boundary indices, zeros at interior")

    # Try applying the boundary conditions to a sample velocity field
    sample_velocity = rand(grid.n_v)
    sample_velocity[grid.vx_left_indices] .= 0.0
    sample_velocity[grid.vx_right_indices] .= 0.0
    sample_velocity[grid.vx_bottom_indices] .= 0.0
    sample_velocity[grid.vx_top_indices] .= 0.0
    sample_velocity[grid.vy_left_indices] .= 0.0
    sample_velocity[grid.vy_right_indices] .= 0.0
    sample_velocity[grid.vy_bottom_indices] .= 0.0
    sample_velocity[grid.vy_top_indices] .= 0.0

    # Apply boundary condition matrix (result is n_v length)
    bc_applied = fluid.constant_boundary_condition_matrix * sample_velocity
    @test length(bc_applied) == grid.n_v
    @test all(bc_applied[grid.v_boundary_indices] .== 0.0)
    @test all(bc_applied[grid.v_interior_indices] .== 0.0)
    println("✓ Wall-boundary conditions applied correctly")

    # Test residual: bc_matrix * velocity - bc_vector
    boundary_residual = calculate_boundary_condition_constraint_residual(
        fluid, sample_velocity, sample_velocity)
    @test length(boundary_residual) == grid.n_v
    @test all(boundary_residual .== 0.0)
    println("✓ Wall-boundary condition residual verified to be zero")

    # Test boundary condition residual jacobians (now n_v × n_v)
    sample_velocity_kp1 = rand(grid.n_v)
    sample_velocity_k = rand(grid.n_v)
    ∂bc_∂velocity_kp1, ∂bc_∂velocity_k, ∂bc_∂boundary_velocity =
        calculate_boundary_condition_constraint_jacobian(fluid,
            sample_velocity_kp1, sample_velocity_k
        )

    @test size(∂bc_∂velocity_kp1) == (grid.n_v, grid.n_v)
    @test size(∂bc_∂velocity_k) == (grid.n_v, grid.n_v)

    @test ∂bc_∂velocity_kp1 == ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, v, sample_velocity_k),
        sample_velocity_kp1
    )
    @test ∂bc_∂velocity_k == ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, v),
        sample_velocity_k
    )
    println("✓ Wall-boundary condition jacobian w.r.t. velocities verified")

    # Test jacobian w.r.t. boundary velocity (still n_v × 2)
    @test size(∂bc_∂boundary_velocity) == (grid.n_v, 2)

    ∂bc_∂bv_fd = ForwardDiff.jacobian(
        bv -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, sample_velocity_k; boundary_velocity=bv, recompute_bc_vector=true),
        fluid.boundary_velocity
    )
    @test ∂bc_∂boundary_velocity ≈ ∂bc_∂bv_fd rtol=1e-8
    println("✓ Wall-boundary condition jacobian w.r.t. boundary velocity verified")

    # For wall BC, derivative w.r.t. boundary velocity should be zero (walls don't move)
    @test all(∂bc_∂boundary_velocity .== 0.0)
    println("✓ Wall-boundary condition has zero derivative w.r.t. boundary velocity")

end

@testitem "Lid Cavity Boundary Conditions" begin
    using Aquarium
    using LinearAlgebra
    using ForwardDiff

    function create_test_fluid(;
        num_cells_x=10,
        num_cells_y=10,
        length_x=1.0,
        length_y=1.0,
        boundary_condition_type=:wall,
        boundary_velocity=[1.0, 2.0]
    )
        fluid_density = 1.0
        dynamic_viscosity = 0.01
        time_step = 0.01

        fluid = Fluid(
            time_step;
            density = fluid_density,
            dynamic_viscosity = dynamic_viscosity,
            boundary_velocity = boundary_velocity,
            grid_size = (num_cells_x, num_cells_y),
            grid_dimensions = (length_x, length_y),
            boundary_condition_type = boundary_condition_type,
        )

        return fluid
    end

    println("\nTesting Lid Cavity Boundary Conditions...")

    # Create fluid with lid cavity boundary conditions
    lid_velocity = [1.0, 0.0]
    fluid = create_test_fluid(
        boundary_condition_type=:lid_cavity,
        boundary_velocity=lid_velocity
    )
    grid = fluid.fvm_grid

    # Test boundary conditions count
    @test fluid.n_boundary_conditions == grid.n_v_boundary
    println("✓ Lid cavity boundary conditions count verified")

    # Test that boundary condition matrix has correct dimensions (now n_v × n_v)
    @test size(fluid.constant_boundary_condition_matrix) == (grid.n_v, grid.n_v)
    println("✓ Lid cavity boundary matrix has correct dimensions (n_v × n_v)")

    # Construct the boundary condition vector explicitly (now n_v length)
    bc_vector = calculate_constant_boundary_condition_vector(
        grid, fluid.boundary_velocity, fluid.boundary_condition_type
    )
    @test length(bc_vector) == grid.n_v
    println("✓ Lid cavity boundary vector has correct length (n_v)")

    # Test that top boundary has lid velocity and others are zero
    @test all(bc_vector[grid.vx_top_indices] .== lid_velocity[1])
    @test all(bc_vector[grid.vy_top_indices] .== 0.0)
    # All other boundaries should be zero
    @test all(bc_vector[grid.vx_left_indices] .== 0.0)
    @test all(bc_vector[grid.vx_right_indices] .== 0.0)
    @test all(bc_vector[grid.vx_bottom_indices] .== 0.0)
    # Interior should be zero
    @test all(bc_vector[grid.v_interior_indices] .== 0.0)
    println("✓ Lid cavity boundary vector has correct values")

    # Test that boundary condition matrix has identity at boundaries
    bc_matrix = fluid.constant_boundary_condition_matrix
    @test all(bc_matrix[grid.v_boundary_indices, grid.v_boundary_indices] .== I(grid.n_v_boundary))
    @test all(bc_matrix[grid.v_interior_indices, :] .== 0.0)
    println("✓ Lid cavity boundary matrix has correct structure")

    # Try applying the boundary conditions to a sample velocity field
    sample_velocity = rand(grid.n_v)
    sample_velocity[grid.vx_left_indices] .= 0.0
    sample_velocity[grid.vx_right_indices] .= 0.0
    sample_velocity[grid.vx_bottom_indices] .= 0.0
    sample_velocity[grid.vx_top_indices] .= lid_velocity[1]
    sample_velocity[grid.vy_left_indices] .= 0.0
    sample_velocity[grid.vy_right_indices] .= 0.0
    sample_velocity[grid.vy_bottom_indices] .= 0.0
    sample_velocity[grid.vy_top_indices] .= 0.0

    # Apply boundary condition matrix (result is n_v length)
    bc_applied = fluid.constant_boundary_condition_matrix * sample_velocity
    @test length(bc_applied) == grid.n_v
    @test all(bc_applied .≈ bc_vector)
    println("✓ Lid cavity boundary conditions applied correctly")

    # Test residual: bc_matrix * velocity - bc_vector
    boundary_residual = calculate_boundary_condition_constraint_residual(
        fluid, sample_velocity, sample_velocity)
    @test length(boundary_residual) == grid.n_v
    @test all(boundary_residual .≈ 0.0)
    println("✓ Lid cavity boundary condition residual verified")

    # Test boundary condition residual jacobians (now n_v × n_v)
    sample_velocity_kp1 = rand(grid.n_v)
    sample_velocity_k = rand(grid.n_v)
    ∂bc_∂velocity_kp1, ∂bc_∂velocity_k, ∂bc_∂boundary_velocity =
        calculate_boundary_condition_constraint_jacobian(fluid,
            sample_velocity_kp1, sample_velocity_k
        )

    @test size(∂bc_∂velocity_kp1) == (grid.n_v, grid.n_v)
    @test size(∂bc_∂velocity_k) == (grid.n_v, grid.n_v)

    @test ∂bc_∂velocity_kp1 ≈ ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, v, sample_velocity_k),
        sample_velocity_kp1
    ) rtol=1e-8
    @test ∂bc_∂velocity_k ≈ ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, v),
        sample_velocity_k
    ) rtol=1e-8
    println("✓ Lid cavity boundary condition jacobian w.r.t. velocities verified")

    # Test jacobian w.r.t. boundary velocity (still n_v × 2)
    @test size(∂bc_∂boundary_velocity) == (grid.n_v, 2)

    ∂bc_∂bv_fd = ForwardDiff.jacobian(
        bv -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, sample_velocity_k; boundary_velocity=bv, recompute_bc_vector=true),
        fluid.boundary_velocity
    )
    @test ∂bc_∂boundary_velocity ≈ ∂bc_∂bv_fd rtol=1e-8
    println("✓ Lid cavity boundary condition jacobian w.r.t. boundary velocity verified")

    # For lid cavity, derivative w.r.t. boundary velocity should be non-zero
    @test any(∂bc_∂boundary_velocity .!= 0.0)
    println("✓ Lid cavity boundary condition has non-zero derivative w.r.t. boundary velocity")

    # Test differentiability of bc_vector w.r.t. boundary velocity (returns n_v × 2)
    bc_vector_grad = ForwardDiff.jacobian(
        bv -> calculate_constant_boundary_condition_vector(grid, bv, :lid_cavity),
        lid_velocity
    )
    @test size(bc_vector_grad) == (grid.n_v, 2)
    @test all(isfinite.(bc_vector_grad))
    println("✓ Lid cavity boundary vector is differentiable w.r.t. boundary velocity")

end
@testitem "Channel Flow Theoretical Boundary Conditions" begin
    using Aquarium
    using LinearAlgebra
    using ForwardDiff

    function create_test_fluid(;
        num_cells_x=10,
        num_cells_y=10,
        length_x=1.0,
        length_y=1.0,
        boundary_condition_type=:wall,
        boundary_velocity=[1.0, 2.0]
    )
        fluid_density = 1.0
        dynamic_viscosity = 0.01
        time_step = 0.01

        fluid = Fluid(
            time_step;
            density = fluid_density,
            dynamic_viscosity = dynamic_viscosity,
            boundary_velocity = boundary_velocity,
            grid_size = (num_cells_x, num_cells_y),
            grid_dimensions = (length_x, length_y),
            boundary_condition_type = boundary_condition_type,
        )

        return fluid
    end


    println("\nTesting Channel Flow Theoretical Boundary Conditions...")

    # Create fluid with channel flow theoretical boundary conditions (parabolic inlet)
    max_velocity = [1.5, 0.0]  # max_velocity[1] is the maximum centerline velocity
    fluid = create_test_fluid(
        boundary_condition_type=:channel_flow_theoretical,
        boundary_velocity=max_velocity,
        num_cells_y=20  # Use more cells in y-direction for better profile resolution
    )
    grid = fluid.fvm_grid

    # Test boundary conditions count
    @test fluid.n_boundary_conditions == grid.n_v_boundary
    println("✓ Channel flow theoretical boundary conditions count verified")

    # Test that boundary condition matrix has correct dimensions (now n_v × n_v)
    @test size(fluid.constant_boundary_condition_matrix) == (grid.n_v, grid.n_v)
    println("✓ Channel flow theoretical boundary matrix has correct dimensions (n_v × n_v)")

    # Construct the boundary condition vector explicitly (now n_v length)
    bc_vector = calculate_constant_boundary_condition_vector(
        grid, fluid.boundary_velocity, fluid.boundary_condition_type
    )
    @test length(bc_vector) == grid.n_v
    println("✓ Channel flow theoretical boundary vector has correct length (n_v)")

    # Test that left boundary has parabolic profile and other boundaries are zero
    L = grid.length_y
    y_coords_left = grid.y_coord_vx_flat[grid.vx_left_indices]
    expected_vx_left = [4.0 * max_velocity[1] * y * (L - y) / (L^2) for y in y_coords_left]

    @test bc_vector[grid.vx_left_indices] ≈ expected_vx_left
    println("✓ Channel flow theoretical left boundary has correct parabolic profile")

    # Check that maximum is at centerline (approximately)
    max_idx = argmax(bc_vector[grid.vx_left_indices])
    y_at_max = y_coords_left[max_idx]
    @test y_at_max ≈ L/2 atol=L/length(grid.vx_left_indices)  # Should be near centerline
    @test bc_vector[grid.vx_left_indices[max_idx]] ≈ max_velocity[1] rtol=0.1
    println("✓ Channel flow theoretical profile maximum is at centerline")

    # All other boundaries should be zero (right=outflow, top/bottom=walls)
    @test all(bc_vector[grid.vx_right_indices] .== 0.0)
    @test all(bc_vector[grid.vx_top_indices] .== 0.0)
    @test all(bc_vector[grid.vx_bottom_indices] .== 0.0)
    @test all(bc_vector[grid.vy_left_indices] .== 0.0)
    @test all(bc_vector[grid.vy_right_indices] .== 0.0)
    @test all(bc_vector[grid.vy_top_indices] .== 0.0)
    @test all(bc_vector[grid.vy_bottom_indices] .== 0.0)
    # Interior should be zero
    @test all(bc_vector[grid.v_interior_indices] .== 0.0)
    println("✓ Channel flow theoretical boundary vector has correct values at all boundaries")

    # Test that boundary condition matrix has identity at boundaries
    bc_matrix = fluid.constant_boundary_condition_matrix
    @test all(bc_matrix[grid.v_boundary_indices, grid.v_boundary_indices] .== I(grid.n_v_boundary))
    @test all(bc_matrix[grid.v_interior_indices, :] .== 0.0)
    println("✓ Channel flow theoretical boundary matrix has correct structure")

    # Try applying the boundary conditions to a sample velocity field
    sample_velocity = rand(grid.n_v)
    # Set boundaries to match theoretical channel flow BC
    sample_velocity[grid.vx_left_indices] .= expected_vx_left
    sample_velocity[grid.vx_right_indices] .= 0.0  # Outflow
    sample_velocity[grid.vx_bottom_indices] .= 0.0
    sample_velocity[grid.vx_top_indices] .= 0.0
    sample_velocity[grid.vy_left_indices] .= 0.0
    sample_velocity[grid.vy_right_indices] .= 0.0
    sample_velocity[grid.vy_bottom_indices] .= 0.0
    sample_velocity[grid.vy_top_indices] .= 0.0

    # Set adjacent to right boundary to zero to simulate outflow condition
    sample_velocity[grid.vx_right_indices .- grid.n_vx_y] .= 0.0
    sample_velocity[grid.vy_right_indices .- grid.n_vy_y] .= 0.0

    # Apply boundary condition matrix (result is n_v length)
    bc_applied = fluid.constant_boundary_condition_matrix * sample_velocity
    @test length(bc_applied) == grid.n_v
    @test all(bc_applied .≈ bc_vector)
    println("✓ Channel flow theoretical boundary conditions applied correctly")

    # Test residual: bc_matrix * velocity - bc_vector (without outflow)
    # Note: For channel flow, outflow BC is handled separately
    boundary_residual = calculate_boundary_condition_constraint_residual(
        fluid, sample_velocity, sample_velocity)
    @test length(boundary_residual) == grid.n_v
    @test maximum(abs.(boundary_residual)) ≈ 0.0 atol=1e-8
    println("✓ Channel flow theoretical boundary condition residual verified")

    # Test boundary condition residual jacobians (now n_v × n_v)
    sample_velocity_kp1 = rand(grid.n_v)
    sample_velocity_k = rand(grid.n_v)
    ∂bc_∂velocity_kp1, ∂bc_∂velocity_k, ∂bc_∂boundary_velocity =
        calculate_boundary_condition_constraint_jacobian(fluid,
            sample_velocity_kp1, sample_velocity_k
        )

    @test size(∂bc_∂velocity_kp1) == (grid.n_v, grid.n_v)
    @test size(∂bc_∂velocity_k) == (grid.n_v, grid.n_v)

    @test ∂bc_∂velocity_kp1 ≈ ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, v, sample_velocity_k),
        sample_velocity_kp1
    ) rtol=1e-8
    @test ∂bc_∂velocity_k ≈ ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, v),
        sample_velocity_k
    ) rtol=1e-8
    println("✓ Channel flow theoretical boundary condition jacobian w.r.t. velocities verified")

    # Test jacobian w.r.t. boundary velocity (still n_v × 2)
    @test size(∂bc_∂boundary_velocity) == (grid.n_v, 2)

    ∂bc_∂bv_fd = ForwardDiff.jacobian(
        bv -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, sample_velocity_k; boundary_velocity=bv, recompute_bc_vector=true),
        fluid.boundary_velocity
    )
    @test ∂bc_∂boundary_velocity ≈ ∂bc_∂bv_fd rtol=1e-8
    println("✓ Channel flow theoretical boundary condition jacobian w.r.t. boundary velocity verified")

    # For channel flow theoretical, derivative w.r.t. boundary velocity should be non-zero (at left boundary)
    @test any(∂bc_∂boundary_velocity .!= 0.0)
    println("✓ Channel flow theoretical boundary condition has non-zero derivative w.r.t. boundary velocity")

    # Test differentiability of bc_vector w.r.t. boundary velocity (returns n_v × 2)
    bc_vector_grad = ForwardDiff.jacobian(
        bv -> calculate_constant_boundary_condition_vector(grid, bv, :channel_flow_theoretical),
        max_velocity
    )
    @test size(bc_vector_grad) == (grid.n_v, 2)
    @test all(isfinite.(bc_vector_grad))
    println("✓ Channel flow theoretical boundary vector is differentiable w.r.t. boundary velocity")

    # Verify the analytical Jacobian formula
    # ∂bc_vector/∂boundary_velocity[1] should be 4*y*(L-y)/L² at left boundary
    expected_grad_col1 = zeros(grid.n_v)
    expected_grad_col1[grid.vx_left_indices] = [4.0 * y * (L - y) / (L^2) for y in y_coords_left]
    @test bc_vector_grad[:, 1] ≈ expected_grad_col1
    @test all(bc_vector_grad[:, 2] .== 0.0)  # No dependence on boundary_velocity[2]
    println("✓ Channel flow theoretical boundary vector Jacobian matches analytical formula")

end

@testitem "Channel Flow Uniform Boundary Conditions" begin
    using Aquarium
    using LinearAlgebra
    using ForwardDiff

    function create_test_fluid(;
        num_cells_x=10,
        num_cells_y=10,
        length_x=1.0,
        length_y=1.0,
        boundary_condition_type=:wall,
        boundary_velocity=[1.0, 2.0]
    )
        fluid_density = 1.0
        dynamic_viscosity = 0.01
        time_step = 0.01

        fluid = Fluid(
            time_step;
            density = fluid_density,
            dynamic_viscosity = dynamic_viscosity,
            boundary_velocity = boundary_velocity,
            grid_size = (num_cells_x, num_cells_y),
            grid_dimensions = (length_x, length_y),
            boundary_condition_type = boundary_condition_type,
        )

        return fluid
    end


    println("\nTesting Channel Flow Uniform Boundary Conditions...")

    # Create fluid with channel flow uniform boundary conditions (uniform inlet)
    inlet_velocity = [2.0, 0.0]  # inlet_velocity[1] is the uniform inlet velocity
    fluid = create_test_fluid(
        boundary_condition_type=:channel_flow_uniform,
        boundary_velocity=inlet_velocity,
        num_cells_y=20  # Use more cells in y-direction for better resolution
    )
    grid = fluid.fvm_grid

    # Test boundary conditions count
    @test fluid.n_boundary_conditions == grid.n_v_boundary
    println("✓ Channel flow uniform boundary conditions count verified")

    # Test that boundary condition matrix has correct dimensions (now n_v × n_v)
    @test size(fluid.constant_boundary_condition_matrix) == (grid.n_v, grid.n_v)
    println("✓ Channel flow uniform boundary matrix has correct dimensions (n_v × n_v)")

    # Construct the boundary condition vector explicitly (now n_v length)
    bc_vector = calculate_constant_boundary_condition_vector(
        grid, fluid.boundary_velocity, fluid.boundary_condition_type
    )
    @test length(bc_vector) == grid.n_v
    println("✓ Channel flow uniform boundary vector has correct length (n_v)")

    # Test that left boundary has uniform profile and other boundaries are zero
    @test all(bc_vector[grid.vx_left_indices] .== inlet_velocity[1])
    println("✓ Channel flow uniform left boundary has correct uniform profile")

    # All other boundaries should be zero (right=outflow, top/bottom=walls)
    @test all(bc_vector[grid.vx_right_indices] .== 0.0)
    @test all(bc_vector[grid.vx_top_indices] .== 0.0)
    @test all(bc_vector[grid.vx_bottom_indices] .== 0.0)
    @test all(bc_vector[grid.vy_left_indices] .== 0.0)
    @test all(bc_vector[grid.vy_right_indices] .== 0.0)
    @test all(bc_vector[grid.vy_top_indices] .== 0.0)
    @test all(bc_vector[grid.vy_bottom_indices] .== 0.0)
    # Interior should be zero
    @test all(bc_vector[grid.v_interior_indices] .== 0.0)
    println("✓ Channel flow uniform boundary vector has correct values at all boundaries")

    # Test that boundary condition matrix has identity at boundaries
    bc_matrix = fluid.constant_boundary_condition_matrix
    @test all(bc_matrix[grid.v_boundary_indices, grid.v_boundary_indices] .== I(grid.n_v_boundary))
    @test all(bc_matrix[grid.v_interior_indices, :] .== 0.0)
    println("✓ Channel flow uniform boundary matrix has correct structure")

    # Try applying the boundary conditions to a sample velocity field
    sample_velocity = rand(grid.n_v)
    # Set boundaries to match uniform channel flow BC
    sample_velocity[grid.vx_left_indices] .= inlet_velocity[1]
    sample_velocity[grid.vx_right_indices] .= 0.0  # Outflow
    sample_velocity[grid.vx_bottom_indices] .= 0.0
    sample_velocity[grid.vx_top_indices] .= 0.0
    sample_velocity[grid.vy_left_indices] .= 0.0
    sample_velocity[grid.vy_right_indices] .= 0.0
    sample_velocity[grid.vy_bottom_indices] .= 0.0
    sample_velocity[grid.vy_top_indices] .= 0.0

    # Set adjacent to right boundary to zero to simulate outflow condition
    sample_velocity[grid.vx_right_indices .- grid.n_vx_y] .= 0.0
    sample_velocity[grid.vy_right_indices .- grid.n_vy_y] .= 0.0

    # Apply boundary condition matrix (result is n_v length)
    bc_applied = fluid.constant_boundary_condition_matrix * sample_velocity
    @test length(bc_applied) == grid.n_v
    @test all(bc_applied .≈ bc_vector)
    println("✓ Channel flow uniform boundary conditions applied correctly")

    # Test residual: bc_matrix * velocity - bc_vector (without outflow)
    # Note: For channel flow, outflow BC is handled separately
    boundary_residual = calculate_boundary_condition_constraint_residual(
        fluid, sample_velocity, sample_velocity)
    @test length(boundary_residual) == grid.n_v
    @test maximum(abs.(boundary_residual)) ≈ 0.0 atol=1e-8
    println("✓ Channel flow uniform boundary condition residual verified")

    # Test boundary condition residual jacobians (now n_v × n_v)
    sample_velocity_kp1 = rand(grid.n_v)
    sample_velocity_k = rand(grid.n_v)
    ∂bc_∂velocity_kp1, ∂bc_∂velocity_k, ∂bc_∂boundary_velocity =
        calculate_boundary_condition_constraint_jacobian(fluid,
            sample_velocity_kp1, sample_velocity_k
        )

    @test size(∂bc_∂velocity_kp1) == (grid.n_v, grid.n_v)
    @test size(∂bc_∂velocity_k) == (grid.n_v, grid.n_v)

    @test ∂bc_∂velocity_kp1 ≈ ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, v, sample_velocity_k),
        sample_velocity_kp1
    ) rtol=1e-8
    @test ∂bc_∂velocity_k ≈ ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, v),
        sample_velocity_k
    ) rtol=1e-8
    println("✓ Channel flow uniform boundary condition jacobian w.r.t. velocities verified")

    # Test jacobian w.r.t. boundary velocity (still n_v × 2)
    @test size(∂bc_∂boundary_velocity) == (grid.n_v, 2)

    ∂bc_∂bv_fd = ForwardDiff.jacobian(
        bv -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, sample_velocity_k; boundary_velocity=bv, recompute_bc_vector=true),
        fluid.boundary_velocity
    )
    @test ∂bc_∂boundary_velocity ≈ ∂bc_∂bv_fd rtol=1e-8
    println("✓ Channel flow uniform boundary condition jacobian w.r.t. boundary velocity verified")

    # For channel flow uniform, derivative w.r.t. boundary velocity should be non-zero (at left boundary)
    @test any(∂bc_∂boundary_velocity .!= 0.0)
    println("✓ Channel flow uniform boundary condition has non-zero derivative w.r.t. boundary velocity")

    # Test differentiability of bc_vector w.r.t. boundary velocity (returns n_v × 2)
    bc_vector_grad = ForwardDiff.jacobian(
        bv -> calculate_constant_boundary_condition_vector(grid, bv, :channel_flow_uniform),
        inlet_velocity
    )
    @test size(bc_vector_grad) == (grid.n_v, 2)
    @test all(isfinite.(bc_vector_grad))
    println("✓ Channel flow uniform boundary vector is differentiable w.r.t. boundary velocity")

    # Verify the analytical Jacobian formula
    # ∂bc_vector/∂boundary_velocity[1] should be 1.0 at left boundary
    expected_grad_col1 = zeros(grid.n_v)
    expected_grad_col1[grid.vx_left_indices] .= 1.0
    @test bc_vector_grad[:, 1] ≈ expected_grad_col1
    @test all(bc_vector_grad[:, 2] .== 0.0)  # No dependence on boundary_velocity[2]
    println("✓ Channel flow uniform boundary vector Jacobian matches analytical formula")

end

@testitem "Freestream Boundary Conditions" begin
    using Aquarium
    using ForwardDiff

    function create_test_fluid(;
        num_cells_x=10,
        num_cells_y=10,
        length_x=1.0,
        length_y=1.0,
        boundary_condition_type=:wall,
        boundary_velocity=[1.0, 2.0]
    )
        fluid_density = 1.0
        dynamic_viscosity = 0.01
        time_step = 0.01

        fluid = Fluid(
            time_step;
            density = fluid_density,
            dynamic_viscosity = dynamic_viscosity,
            boundary_velocity = boundary_velocity,
            grid_size = (num_cells_x, num_cells_y),
            grid_dimensions = (length_x, length_y),
            boundary_condition_type = boundary_condition_type,
        )

        return fluid
    end


    println("\nTesting Freestream Boundary Conditions...")
    
    # Create fluid with freestream boundary conditions
    freestream_velocity = [1.5, 0.0]
    fluid = create_test_fluid(
        boundary_condition_type=:freestream,
        boundary_velocity=freestream_velocity
    )
    grid = fluid.fvm_grid
    
    # Test boundary conditions count
    @test fluid.n_boundary_conditions == grid.n_v_boundary
    n_outflow = length(grid.vx_right_indices) + length(grid.vy_right_indices)
    println("✓ Freestream boundary conditions count verified")

    # For freestream, constant BC matrix is now n_v × n_v (full velocity size)
    @test size(fluid.constant_boundary_condition_matrix) == (grid.n_v, grid.n_v)
    println("✓ Freestream constant BC matrix has correct dimensions (n_v × n_v)")

    # Construct the constant boundary condition vector explicitly (now n_v length)
    bc_vector = calculate_constant_boundary_condition_vector(
        grid, fluid.boundary_velocity, fluid.boundary_condition_type
    )
    @test length(bc_vector) == grid.n_v
    println("✓ Freestream constant BC vector has correct length (n_v)")
    
    # Test that inflow boundaries have freestream velocity (indexed directly)
    @test all(bc_vector[grid.vx_left_indices] .== freestream_velocity[1])
    @test all(bc_vector[grid.vx_bottom_indices] .== freestream_velocity[1])
    @test all(bc_vector[grid.vx_top_indices] .== freestream_velocity[1])
    @test all(bc_vector[grid.vy_left_indices] .== freestream_velocity[2])
    @test all(bc_vector[grid.vy_bottom_indices] .== freestream_velocity[2])
    @test all(bc_vector[grid.vy_top_indices] .== freestream_velocity[2])
    # Outflow boundaries should be zero (not set by constant BC)
    @test all(bc_vector[grid.vx_right_indices] .== 0.0)
    @test all(bc_vector[grid.vy_right_indices] .== 0.0)
    # Interior should be zero
    @test all(bc_vector[grid.v_interior_indices] .== 0.0)
    println("✓ Freestream boundary vector has correct values")

    # Test that constant BC matrix has identity at inflow boundaries, zeros at outflow
    bc_matrix = fluid.constant_boundary_condition_matrix
    inflow_indices = vcat(
        grid.vx_left_indices, grid.vx_bottom_indices, grid.vx_top_indices,
        grid.vy_left_indices, grid.vy_bottom_indices, grid.vy_top_indices
    )
    outflow_indices = vcat(grid.vx_right_indices, grid.vy_right_indices)

    # Inflow boundaries should have identity
    for idx in inflow_indices
        @test bc_matrix[idx, idx] == 1.0
    end
    # Outflow boundaries should have zero rows (handled separately)
    @test all(bc_matrix[outflow_indices, :] .== 0.0)
    # Interior should have zero rows
    @test all(bc_matrix[grid.v_interior_indices, :] .== 0.0)
    println("✓ Freestream constant BC matrix has correct structure")

    # Try applying the boundary conditions to a sample velocity field
    sample_velocity = rand(grid.n_v)
    sample_velocity[grid.vx_left_indices] .= freestream_velocity[1]
    sample_velocity[grid.vx_right_indices] .= 0.0  # Outflow (not constrained)
    sample_velocity[grid.vx_bottom_indices] .= freestream_velocity[1]
    sample_velocity[grid.vx_top_indices] .= freestream_velocity[1]
    sample_velocity[grid.vy_left_indices] .= freestream_velocity[2]
    sample_velocity[grid.vy_right_indices] .= 0.0  # Outflow (not constrained)
    sample_velocity[grid.vy_bottom_indices] .= freestream_velocity[2]
    sample_velocity[grid.vy_top_indices] .= freestream_velocity[2]

    # Apply constant boundary condition matrix (result is n_v length)
    bc_applied = fluid.constant_boundary_condition_matrix * sample_velocity
    @test length(bc_applied) == grid.n_v
    @test all(bc_applied .≈ bc_vector)
    println("✓ Freestream constant BC applied correctly")

    # Test that outflow condition is satisfied when set to freestream
    sample_velocity_uniform = zeros(grid.n_v)
    sample_velocity_uniform[1:grid.n_vx] .= freestream_velocity[1]
    sample_velocity_uniform[(grid.n_vx+1):end] .= freestream_velocity[2]

    outflow_residual = calculate_outflow_boundary_condition_residual(fluid,
        sample_velocity_uniform, sample_velocity_uniform
    )
    @test all(outflow_residual .≈ 0.0)
    println("✓ Freestream outflow BC satisfied when velocity is uniform")

    # Now test outflow-residual jacobians
    ∂outflow_∂velocity_kp1, ∂outflow_∂velocity_k  = 
        calculate_outflow_boundary_condition_jacobian(fluid,
            sample_velocity, sample_velocity
        )
    
    @test ∂outflow_∂velocity_kp1 == ForwardDiff.jacobian(
        v -> calculate_outflow_boundary_condition_residual(fluid, v, sample_velocity),
        sample_velocity
    )
    @test ∂outflow_∂velocity_k == ForwardDiff.jacobian(
        v -> calculate_outflow_boundary_condition_residual(fluid, sample_velocity, v),
        sample_velocity
    )
    println("✓ Freestream outflow boundary condition jacobians verified")

    # Test full boundary condition constraint residual (includes outflow at right boundary)
    # The residual should be n_v length with outflow residual embedded at right boundary indices
    boundary_residual = calculate_boundary_condition_constraint_residual(
        fluid, sample_velocity_uniform, sample_velocity_uniform)
    @test length(boundary_residual) == grid.n_v
    @test all(boundary_residual .≈ 0.0)
    println("✓ Freestream full BC residual computed (includes outflow embedded)")

    # Test full boundary condition constraint jacobians (includes outflow)
    sample_velocity_kp1 = rand(grid.n_v)
    sample_velocity_k = rand(grid.n_v)
    ∂bc_∂velocity_kp1, ∂bc_∂velocity_k, ∂bc_∂boundary_velocity =
        calculate_boundary_condition_constraint_jacobian(fluid,
            sample_velocity_kp1, sample_velocity_k
        )

    # Jacobians should be n_v x n_v (full velocity vector)
    @test size(∂bc_∂velocity_kp1) == (grid.n_v, grid.n_v)
    @test size(∂bc_∂velocity_k) == (grid.n_v, grid.n_v)

    @test ∂bc_∂velocity_kp1 ≈ ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, v, sample_velocity_k),
        sample_velocity_kp1
    ) rtol=1e-8
    @test ∂bc_∂velocity_k ≈ ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, v),
        sample_velocity_k
    ) rtol=1e-8
    println("✓ Freestream full BC jacobian w.r.t. velocities verified (includes outflow)")
    
    # Test jacobian w.r.t. boundary velocity
    ∂bc_∂bv_fd = ForwardDiff.jacobian(
        bv -> calculate_boundary_condition_constraint_residual(
            fluid, sample_velocity_kp1, sample_velocity_k; boundary_velocity=bv, recompute_bc_vector=true),
        fluid.boundary_velocity
    )
    @test ∂bc_∂boundary_velocity ≈ ∂bc_∂bv_fd rtol=1e-8
    println("✓ Freestream boundary condition jacobian w.r.t. boundary velocity verified")
    
    # For freestream BC, derivative w.r.t. boundary velocity should be non-zero
    @test any(∂bc_∂boundary_velocity .!= 0.0)
    println("✓ Freestream boundary condition has non-zero derivative w.r.t. boundary velocity")
    
    # Test differentiability of bc_vector w.r.t. boundary velocity (returns n_v × 2)
    bc_vector_grad = ForwardDiff.jacobian(
        bv -> calculate_constant_boundary_condition_vector(grid, bv, :freestream),
        freestream_velocity
    )
    @test size(bc_vector_grad) == (grid.n_v, 2)
    @test all(isfinite.(bc_vector_grad))
    println("✓ Freestream boundary vector is differentiable w.r.t. boundary velocity")

end

@testitem "Fluid Basic Properties" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Test fluid properties
    @test fluid.density == density
    @test fluid.dynamic_viscosity == dynamic_viscosity
    @test fluid.boundary_velocity ≈ [boundary_velocity_x, boundary_velocity_y]
    println("  ✓ Fluid properties correctly stored")
    
    # Test time and physical constants
    @test fluid.time_step == time_step
    @test fluid.gravity_constant == gravity_constant
    @test fluid.external_pressure_gradient == external_pressure_gradient
    println("  ✓ Time step and constants correctly set")
    
    # Test cell mass
    expected_cell_mass = density * fluid.fvm_grid.h_x * fluid.fvm_grid.h_y
    @test fluid.cell_mass ≈ expected_cell_mass rtol=1e-10
    println("  ✓ Cell mass correctly calculated: $(fluid.cell_mass) kg")
    
    # Test boundary condition properties
    @test fluid.boundary_condition_type == boundary_condition_type
    @test fluid.n_boundary_conditions > 0
    println("  ✓ Boundary condition properties correctly set")
    println("    - Number of boundary conditions: $(fluid.n_boundary_conditions)")
end

@testitem "FVM Grid and State Dimensions" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Test grid properties
    @test fluid.fvm_grid.n_cell_x == grid_size[1] .+ 2 # Test includes ghost cells
    @test fluid.fvm_grid.n_cell_y == grid_size[2] .+ 2 # Test includes ghost cells
    @test fluid.fvm_grid.length_x == grid_dimensions[1]
    @test fluid.fvm_grid.length_y == grid_dimensions[2]
    println("  ✓ FVM grid dimensions correct")
    
    # Test state dimensions (BC now embedded in stationarity, no separate BC constraints)
    @test fluid.n_velocities == fluid.fvm_grid.n_v
    @test fluid.n_continuity_constraints == size(fluid.divergence, 1)
    @test fluid.n_states == fluid.n_velocities + fluid.n_continuity_constraints
    println("  ✓ State dimensions correct:")
    println("    - Velocities: $(fluid.n_velocities)")
    println("    - Continuity constraints: $(fluid.n_continuity_constraints)")
    println("    - Total states: $(fluid.n_states)")
    println("    - (Boundary conditions now embedded in stationarity)")

    # Test index arrays (no boundary_condition_dual_indices anymore)
    @test length(fluid.state_indices) == fluid.n_states
    @test length(fluid.velocity_indices) == fluid.n_velocities
    @test length(fluid.continuity_dual_indices) == fluid.n_continuity_constraints
    @test fluid.velocity_indices[1] == 1
    @test fluid.velocity_indices[end] == fluid.n_velocities
    @test fluid.continuity_dual_indices[1] == fluid.n_velocities + 1
    @test fluid.continuity_dual_indices[end] == fluid.n_velocities + fluid.n_continuity_constraints
    println("  ✓ Index arrays correctly set")
end

@testitem "Gravitational and Pressure Gradient Forces" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Test gravity force
    @test length(fluid.gravitational_acceleration) == fluid.n_velocities

    # Gravity should only affect interior y-velocities (boundary velocities are zeroed)
    n_vx = fluid.fvm_grid.n_vx
    n_vy = fluid.fvm_grid.n_vy
    v_interior_indices = fluid.fvm_grid.v_interior_indices
    v_boundary_indices = fluid.fvm_grid.v_boundary_indices

    # Check that boundary velocities have zero gravity
    @test all(fluid.gravitational_acceleration[v_boundary_indices] .== 0.0)

    # Check that interior x-velocities have zero gravity
    interior_vx_indices = filter(idx -> idx <= n_vx, v_interior_indices)
    @test all(fluid.gravitational_acceleration[interior_vx_indices] .== 0.0)

    # Check that interior y-velocities have gravity
    interior_vy_indices = filter(idx -> idx > n_vx, v_interior_indices)
    @test all(fluid.gravitational_acceleration[interior_vy_indices] .≈ -(gravity_constant))
    println("  ✓ Gravity force correctly constructed")
    println("    - Only affects interior y-velocities with magnitude: $(gravity_constant) m/s²")
    println("    - Boundary velocities have zero gravity")

    # Test external pressure gradient force
    @test length(fluid.external_pressure_gradient_force) == fluid.n_velocities

    # For zero pressure gradient, force should be zero everywhere
    if external_pressure_gradient == (0.0, 0.0)
        @test all(fluid.external_pressure_gradient_force .== 0.0)
        println("  ✓ External pressure gradient force is zero (as expected)")
    else
        # Boundary velocities should have zero pressure gradient force
        @test all(fluid.external_pressure_gradient_force[v_boundary_indices] .== 0.0)
        println("  ✓ Boundary velocities have zero pressure gradient force")
    end
end

@testitem "FVM Operators" begin
    using Aquarium
    using SparseArrays

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Test Laplacian operator
    @test size(fluid.laplacian) == (fluid.n_velocities, fluid.n_velocities)
    @test issparse(fluid.laplacian)
    println("  ✓ Laplacian operator has correct dimensions")

    # Test Divergence operator
    @test size(fluid.divergence) == (fluid.n_continuity_constraints, fluid.n_velocities)
    @test issparse(fluid.divergence)
    println("  ✓ Divergence operator has correct dimensions")

    # Test that divergence operator has boundary columns zeroed out (for constant BCs)
    boundary_indices = fluid.fvm_grid.v_boundary_indices
    if fluid.boundary_condition_type == :freestream
        # For freestream, all boundaries except right boundary are zeroed
        constant_boundary_indices = setdiff(
            boundary_indices,
            vcat(fluid.fvm_grid.vx_right_indices, fluid.fvm_grid.vy_right_indices)
        )
        for idx in constant_boundary_indices
            @test all(fluid.divergence[:, idx] .== 0.0)
        end
        println("  ✓ Divergence operator has constant boundary columns zeroed (freestream)")
    else
        # For other BCs, all boundary columns are zeroed
        for idx in boundary_indices
            @test all(fluid.divergence[:, idx] .== 0.0)
        end
        println("  ✓ Divergence operator has all boundary columns zeroed")
    end

    # Test continuity vector
    @test length(fluid.continuity_vector) == fluid.n_continuity_constraints
    println("  ✓ Continuity vector has correct length")

    # Test that continuity vector is computed correctly from boundary conditions
    # For lid cavity with bottom/top/left/right walls, continuity_vector should account for BC flux
    original_divergence = calculate_divergence_operator(fluid.fvm_grid)
    expected_continuity_vector = original_divergence * fluid.constant_boundary_condition_vector
    @test fluid.continuity_vector ≈ expected_continuity_vector rtol=1e-10
    println("  ✓ Continuity vector correctly computed from boundary conditions")

    # Test compatibility: div' should be gradient (same size as laplacian)
    @test size(fluid.divergence') == (fluid.n_velocities, fluid.n_continuity_constraints)
    println("  ✓ Divergence transpose (gradient) has correct dimensions")

    # Test midpoint operators
    @test length(fluid.midpoint_operators) == 6
    for (i, op) in enumerate(fluid.midpoint_operators)
        @test issparse(op)
        @test size(op, 2) == fluid.n_velocities
    end
    println("  ✓ All 6 midpoint operators constructed correctly")

    # Test boundary condition matrix (now returns n_v × n_v)
    @test size(fluid.constant_boundary_condition_matrix) == (fluid.n_velocities, fluid.n_velocities)
    @test issparse(fluid.constant_boundary_condition_matrix)
    println("  ✓ Boundary condition matrix has correct dimensions (n_v × n_v)")
end

@testitem "Convective Term Calculation" begin
    using Aquarium
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create sample velocity field
    sample_velocity = rand(fluid.n_velocities)
    
    # Calculate convective term
    convective = calculate_convective_term(fluid, sample_velocity)
    
    @test length(convective) == fluid.n_velocities
    println("  ✓ Convective term has correct length")
    
    # Test that convective term is differentiable
    convective_ad = ForwardDiff.jacobian(
        v -> calculate_convective_term(fluid, v),
        sample_velocity
    )
    @test size(convective_ad) == (fluid.n_velocities, fluid.n_velocities)
    println("  ✓ Convective term is differentiable w.r.t. velocity")
end

@testitem "Convective Jacobian" begin
    using Aquarium
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create sample velocity field
    sample_velocity = rand(fluid.n_velocities)
    
    # Calculate convective Jacobian
    convective_jac = calculate_convective_jacobian(fluid, sample_velocity)
    
    @test size(convective_jac) == (fluid.n_velocities, fluid.n_velocities)
    println("  ✓ Convective Jacobian has correct dimensions")
    
    # Verify against ForwardDiff
    convective_jac_fd = ForwardDiff.jacobian(
        v -> calculate_convective_term(fluid, v),
        sample_velocity
    )
    
    @test convective_jac ≈ convective_jac_fd rtol=1e-8
    println("  ✓ Convective Jacobian matches ForwardDiff")
end

@testitem "Mass Conservation Constraint" begin
    using Aquarium
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create sample velocity field
    sample_velocity = rand(fluid.n_velocities)

    # Calculate mass conservation residual
    com_residual = calculate_mass_conservation_constraint_residual(fluid, sample_velocity)

    @test length(com_residual) == fluid.n_continuity_constraints
    println("  ✓ Mass conservation residual has correct length")

    # Test that residual includes contribution from constant boundary conditions
    # The new formulation is: div*v + continuity_vector
    # where continuity_vector = original_div * bc_vector accounts for constant BCs
    expected_residual = fluid.divergence * sample_velocity + fluid.continuity_vector
    @test com_residual ≈ expected_residual rtol=1e-10
    println("  ✓ Mass conservation residual correctly includes continuity_vector")

    # Test that residual changes when we change boundary velocity
    new_boundary_velocity = fluid.boundary_velocity .+ 0.5
    com_residual_modified = calculate_mass_conservation_constraint_residual(
        fluid, sample_velocity;
        boundary_velocity=new_boundary_velocity,
        recompute_bc_vector=true
    )
    com_residual_expected = fluid.divergence * sample_velocity + calculate_divergence_operator(fluid.fvm_grid) *
        calculate_constant_boundary_condition_vector(
            fluid.fvm_grid,
            new_boundary_velocity,
            fluid.boundary_condition_type
        )
    @test com_residual_expected ≈ com_residual_modified
    println("  ✓ Mass conservation residual changes with boundary velocity")

    # Test Jacobian
    com_jacobian, ∂continuity_vector_∂boundary_velocity =
        calculate_mass_conservation_constraint_jacobian(fluid)

    @test size(com_jacobian) == (fluid.n_continuity_constraints, fluid.n_velocities)
    @test com_jacobian == fluid.divergence
    println("  ✓ Mass conservation Jacobian w.r.t. velocity is the divergence operator")

    # Test Jacobian w.r.t. boundary velocity
    @test size(∂continuity_vector_∂boundary_velocity) == (fluid.n_continuity_constraints, 2)
    println("  ✓ Mass conservation Jacobian w.r.t. boundary velocity has correct dimensions")

    # Verify velocity Jacobian against ForwardDiff
    com_jac_fd = ForwardDiff.jacobian(
        v -> calculate_mass_conservation_constraint_residual(fluid, v; recompute_bc_vector=true),
        sample_velocity
    )

    @test com_jacobian ≈ com_jac_fd rtol=1e-8
    println("  ✓ Mass conservation Jacobian w.r.t. velocity matches ForwardDiff")

    # Verify boundary velocity Jacobian against ForwardDiff
    com_jac_bv_fd = ForwardDiff.jacobian(
        bv -> calculate_mass_conservation_constraint_residual(
            fluid, sample_velocity;
            boundary_velocity=bv,
            recompute_bc_vector=true
        ),
        fluid.boundary_velocity
    )

    @test ∂continuity_vector_∂boundary_velocity ≈ com_jac_bv_fd rtol=1e-8
    println("  ✓ Mass conservation Jacobian w.r.t. boundary velocity matches ForwardDiff")

    # Test that for a divergence-free field with matching BCs, residual is small
    # This is hard to construct exactly, but we can verify the structure
    # Just check that the residual is computable and finite
    @test all(isfinite.(com_residual))
    println("  ✓ Mass conservation residual is finite for arbitrary velocity field")
end

@testitem "Recompute BC Vector Flag" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Test that recompute_bc_vector gives same result for same arguments
    sample_velocity = rand(fluid.n_velocities)

    # Test 1: Mass conservation constraint - same boundary velocity should give same result
    com_residual_no_recompute = calculate_mass_conservation_constraint_residual(
        fluid, sample_velocity;
        boundary_velocity=fluid.boundary_velocity,
        recompute_bc_vector=false
    )

    com_residual_with_recompute = calculate_mass_conservation_constraint_residual(
        fluid, sample_velocity;
        boundary_velocity=fluid.boundary_velocity,
        recompute_bc_vector=true
    )

    @test com_residual_no_recompute ≈ com_residual_with_recompute rtol=1e-10
    println("  ✓ Mass conservation: recompute_bc_vector gives same result for same arguments")

    # Test 2: Boundary condition constraint - same boundary velocity should give same result
    velocity_kp1 = rand(fluid.n_velocities)
    velocity_k = rand(fluid.n_velocities)

    bc_residual_no_recompute = calculate_boundary_condition_constraint_residual(
        fluid, velocity_kp1, velocity_k;
        boundary_velocity=fluid.boundary_velocity,
        recompute_bc_vector=false
    )

    bc_residual_with_recompute = calculate_boundary_condition_constraint_residual(
        fluid, velocity_kp1, velocity_k;
        boundary_velocity=fluid.boundary_velocity,
        recompute_bc_vector=true
    )

    @test bc_residual_no_recompute ≈ bc_residual_with_recompute rtol=1e-10
    println("  ✓ Boundary condition: recompute_bc_vector gives same result for same arguments")

    # Test 3: Stationarity residual - same fluid properties should give same result
    fluid_state_kp1 = rand(fluid.n_states)
    fluid_state_k = rand(fluid.n_states)

    stationarity_residual_no_recompute = calculate_fluid_stationarity_residual(
        fluid, fluid_state_kp1, fluid_state_k;
        recompute_bc_vector=false
    )

    stationarity_residual_with_recompute = calculate_fluid_stationarity_residual(
        fluid, fluid_state_kp1, fluid_state_k;
        recompute_bc_vector=true
    )

    @test stationarity_residual_no_recompute ≈ stationarity_residual_with_recompute rtol=1e-10
    println("  ✓ Stationarity: recompute_bc_vector gives same result for same arguments")

    # Test 4: Dynamics residual - same fluid properties should give same result
    dynamics_residual_no_recompute = calculate_fluid_dynamics_residual(
        fluid, fluid_state_kp1, fluid_state_k;
        recompute_bc_vector=false
    )

    dynamics_residual_with_recompute = calculate_fluid_dynamics_residual(
        fluid, fluid_state_kp1, fluid_state_k;
        recompute_bc_vector=true
    )

    @test dynamics_residual_no_recompute ≈ dynamics_residual_with_recompute rtol=1e-10
    println("  ✓ Dynamics: recompute_bc_vector gives same result for same arguments")
end

@testitem "Boundary Condition Constraint" begin
    using Aquarium
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create sample velocity fields
    velocity_kp1 = rand(fluid.n_velocities)
    velocity_k = rand(fluid.n_velocities)
    
    # Calculate boundary condition residual (now returns n_v length)
    bc_residual = calculate_boundary_condition_constraint_residual(
        fluid, velocity_kp1, velocity_k
    )

    @test length(bc_residual) == fluid.n_velocities
    println("  ✓ Boundary condition residual has correct length (n_v)")

    # Test Jacobians (now return n_v × n_v)
    ∂bc_∂velocity_kp1, ∂bc_∂velocity_k, ∂bc_∂boundary_velocity =
        calculate_boundary_condition_constraint_jacobian(
            fluid, velocity_kp1, velocity_k
        )

    @test size(∂bc_∂velocity_kp1) == (fluid.n_velocities, fluid.n_velocities)
    @test size(∂bc_∂velocity_k) == (fluid.n_velocities, fluid.n_velocities)
    println("  ✓ Boundary condition Jacobians have correct dimensions (n_v × n_v)")
    
    # Verify Jacobian w.r.t. velocity_kp1 against ForwardDiff
    bc_jac_kp1_fd = ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(fluid, v, velocity_k; recompute_bc_vector=true),
        velocity_kp1
    )
    
    @test ∂bc_∂velocity_kp1 ≈ bc_jac_kp1_fd rtol=1e-8
    println("  ✓ Boundary condition Jacobian w.r.t. velocity_{k+1} matches ForwardDiff")
    
    # Verify Jacobian w.r.t. velocity_k against ForwardDiff
    bc_jac_k_fd = ForwardDiff.jacobian(
        v -> calculate_boundary_condition_constraint_residual(fluid, velocity_kp1, v; recompute_bc_vector=true),
        velocity_k
    )
    
    @test ∂bc_∂velocity_k ≈ bc_jac_k_fd rtol=1e-8
    println("  ✓ Boundary condition Jacobian w.r.t. velocity_k matches ForwardDiff")
end

@testitem "Fluid Stationarity Residual" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create sample fluid states
    fluid_state_kp1 = rand(fluid.n_states)
    fluid_state_k = rand(fluid.n_states)
    
    # Calculate stationarity residual
    stationarity_residual = calculate_fluid_stationarity_residual(
        fluid, fluid_state_kp1, fluid_state_k
    )
    
    @test length(stationarity_residual) == fluid.n_velocities
    println("  ✓ Stationarity residual has correct length")
    
    # Test with different fluid properties (via inject)
    test_fluid_properties = [density * 1.1, dynamic_viscosity * 1.2,
                             boundary_velocity_x, boundary_velocity_y]
    fluid_modified = inject_differentiable_params(fluid, test_fluid_properties)
    stationarity_residual_modified = calculate_fluid_stationarity_residual(
        fluid_modified, fluid_state_kp1, fluid_state_k
    )

    @test stationarity_residual != stationarity_residual_modified
    println("  ✓ Stationarity residual changes with fluid properties")
    
    # Test that residuals at boundary indices match boundary condition residual
    boundary_condition_residual = calculate_boundary_condition_constraint_residual(
        fluid,
        fluid_state_kp1[fluid.velocity_indices],
        fluid_state_k[fluid.velocity_indices]
    )
    boundary_indices = fluid.fvm_grid.v_boundary_indices
    
    # The boundary condition residual should be included in the stationarity residual
    # at the boundary indices, so check that those components are non-zero when BC residual is non-zero
    @test boundary_condition_residual[boundary_indices] ≈ stationarity_residual[boundary_indices] rtol=1e-10
    println("  ✓ Stationarity residual correctly incorporates boundary condition residual")
end

@testitem "Fluid Stationarity Jacobian" begin
    using Aquarium
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create sample fluid states
    fluid_state_kp1 = rand(fluid.n_states)
    fluid_state_k = rand(fluid.n_states)
    
    # Calculate stationarity Jacobians
    ∂stationarity_∂state_kp1, ∂stationarity_∂state_k, ∂stationarity_∂properties =
        calculate_fluid_stationarity_jacobian(fluid, fluid_state_kp1, fluid_state_k)
    
    # Test dimensions
    @test size(∂stationarity_∂state_kp1) == (fluid.n_velocities, fluid.n_states)
    @test size(∂stationarity_∂state_k) == (fluid.n_velocities, fluid.n_states)
    @test size(∂stationarity_∂properties) == (fluid.n_velocities, 4)
    println("  ✓ Stationarity Jacobians have correct dimensions")
    
    # Verify Jacobian w.r.t. state_kp1 against ForwardDiff
    stationarity_jac_kp1_fd = ForwardDiff.jacobian(
        s -> calculate_fluid_stationarity_residual(fluid, s, fluid_state_k; recompute_bc_vector=true),
        fluid_state_kp1
    )
    
    @test ∂stationarity_∂state_kp1 ≈ stationarity_jac_kp1_fd rtol=1e-8
    println("  ✓ Stationarity Jacobian w.r.t. state_{k+1} matches ForwardDiff")
    
    # Verify Jacobian w.r.t. state_k against ForwardDiff
    stationarity_jac_k_fd = ForwardDiff.jacobian(
        s -> calculate_fluid_stationarity_residual(fluid, fluid_state_kp1, s; recompute_bc_vector=true),
        fluid_state_k
    )
    
    @test ∂stationarity_∂state_k ≈ stationarity_jac_k_fd rtol=1e-8
    println("  ✓ Stationarity Jacobian w.r.t. state_k matches ForwardDiff")
    
    # Verify Jacobian w.r.t. fluid_properties against ForwardDiff (via inject)
    stationarity_jac_props_fd = ForwardDiff.jacobian(
        p -> calculate_fluid_stationarity_residual(
            inject_differentiable_params(fluid, p),
            fluid_state_kp1, fluid_state_k;
            recompute_bc_vector=true),
        collect_differentiable_params(fluid)
    )

    @test ∂stationarity_∂properties ≈ stationarity_jac_props_fd rtol=1e-8
    println("  ✓ Stationarity Jacobian w.r.t. fluid properties matches ForwardDiff")
end

@testitem "Fluid Dynamics Residual" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create sample fluid states
    fluid_state_kp1 = rand(fluid.n_states)
    fluid_state_k = rand(fluid.n_states)
    
    # Calculate dynamics residual
    dynamics_residual = calculate_fluid_dynamics_residual(
        fluid, fluid_state_kp1, fluid_state_k
    )
    
    # Residual should include: stationarity + mass conservation (BC now embedded in stationarity)
    expected_length = fluid.n_velocities + fluid.n_continuity_constraints
    @test length(dynamics_residual) == expected_length
    @test length(dynamics_residual) == fluid.n_states
    println("  ✓ Dynamics residual has correct length: $(length(dynamics_residual))")
    println("    (Boundary conditions embedded in stationarity component)")
    
    # Test with different fluid properties (via inject)
    test_fluid_properties = [density * 1.1, dynamic_viscosity * 1.2,
                             boundary_velocity_x * 1.5, boundary_velocity_y]
    fluid_modified = inject_differentiable_params(fluid, test_fluid_properties)
    dynamics_residual_modified = calculate_fluid_dynamics_residual(
        fluid_modified, fluid_state_kp1, fluid_state_k
    )
    
    @test dynamics_residual != dynamics_residual_modified
    println("  ✓ Dynamics residual changes with fluid properties")
end

@testitem "Fluid Dynamics-Residual Jacobian" begin
    using Aquarium
    using LinearAlgebra
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create sample fluid states
    fluid_state_kp1 = rand(fluid.n_states)
    fluid_state_k = rand(fluid.n_states)
    
    # Calculate dynamics Jacobians
    ∂dynamics_∂state_kp1, ∂dynamics_∂state_k, ∂dynamics_∂properties =
        calculate_fluid_dynamics_jacobian(
            fluid, fluid_state_kp1, fluid_state_k
        )
    
    # Test dimensions
    @test size(∂dynamics_∂state_kp1) == (fluid.n_states, fluid.n_states)
    @test size(∂dynamics_∂state_k) == (fluid.n_states, fluid.n_states)
    @test size(∂dynamics_∂properties) == (fluid.n_states, 4)
    println("  ✓ Dynamics-residual Jacobians have correct dimensions")
    
    # Verify Jacobian w.r.t. state_kp1 against ForwardDiff
    dynamics_jac_kp1_fd = ForwardDiff.jacobian(
        s -> calculate_fluid_dynamics_residual(fluid, s, fluid_state_k; recompute_bc_vector=true),
        fluid_state_kp1
    )
    
    @test Matrix(∂dynamics_∂state_kp1) ≈ dynamics_jac_kp1_fd rtol=1e-7
    println("  ✓ Dynamics-residual Jacobian w.r.t. state_{k+1} matches ForwardDiff")
    
    # Verify Jacobian w.r.t. state_k against ForwardDiff
    dynamics_jac_k_fd = ForwardDiff.jacobian(
        s -> calculate_fluid_dynamics_residual(fluid, fluid_state_kp1, s; recompute_bc_vector=true),
        fluid_state_k
    )
    
    @test Matrix(∂dynamics_∂state_k) ≈ dynamics_jac_k_fd rtol=1e-7
    println("  ✓ Dynamics-residual Jacobian w.r.t. state_k matches ForwardDiff")
    
    # Verify Jacobian w.r.t. fluid_properties against ForwardDiff (via inject)
    dynamics_jac_props_fd = ForwardDiff.jacobian(
        p -> calculate_fluid_dynamics_residual(
            inject_differentiable_params(fluid, p),
            fluid_state_kp1, fluid_state_k;
            recompute_bc_vector=true),
        collect_differentiable_params(fluid)
    )
    
    @test Matrix(∂dynamics_∂properties) ≈ dynamics_jac_props_fd rtol=1e-7
    println("  ✓ Dynamics-residual Jacobian w.r.t. fluid properties matches ForwardDiff")
end

@testitem "Differentiability w.r.t. Fluid Properties" begin
    using Aquarium
    using LinearAlgebra
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create sample fluid states
    fluid_state_kp1 = rand(fluid.n_states)
    fluid_state_k = rand(fluid.n_states)
    
    # Test 1: Verify that dynamics residual is differentiable w.r.t. density (via inject)
    println("  Testing differentiability w.r.t. density...")
    density_grad = ForwardDiff.gradient(
        props -> begin
            residual = calculate_fluid_dynamics_residual(
                inject_differentiable_params(fluid, props),
                fluid_state_kp1, fluid_state_k; recompute_bc_vector=true,
            )
            return sum(abs2, residual)
        end,
        collect_differentiable_params(fluid),
    )
    
    @test all(isfinite.(density_grad))
    @test density_grad[1] != 0.0  # Should affect residual
    println("    ✓ Density gradient is finite and non-zero: $(density_grad[1])")
    
    # Test 2: Verify that dynamics residual is differentiable w.r.t. dynamic viscosity
    println("  Testing differentiability w.r.t. dynamic viscosity...")
    @test all(isfinite.(density_grad))
    @test density_grad[2] != 0.0  # Should affect residual
    println("    ✓ Viscosity gradient is finite and non-zero: $(density_grad[2])")
    
    # Test 3: Verify that dynamics residual is differentiable w.r.t. boundary velocity
    println("  Testing differentiability w.r.t. boundary velocity...")
    @test all(isfinite.(density_grad))
    # Boundary velocity gradient may be zero for wall boundary conditions
    println("    ✓ Boundary velocity gradients are finite: [$(density_grad[3]), $(density_grad[4])]")
    
    # Test 4: Compute full Jacobian w.r.t. fluid properties (via inject)
    println("  Computing full Jacobian w.r.t. fluid properties...")
    jacobian_props = ForwardDiff.jacobian(
        props -> calculate_fluid_dynamics_residual(
            inject_differentiable_params(fluid, props),
            fluid_state_kp1, fluid_state_k; recompute_bc_vector=true,
        ),
        collect_differentiable_params(fluid),
    )
    
    @test size(jacobian_props) == (fluid.n_states, 4)
    @test all(isfinite.(jacobian_props))
    println("    ✓ Full Jacobian w.r.t. fluid properties is finite")
    println("    ✓ Jacobian shape: $(size(jacobian_props))")
    
    # Test 5: Verify consistency between analytical and AD Jacobians
    println("  Verifying analytical Jacobian consistency...")
    _, _, ∂dynamics_∂properties = calculate_fluid_dynamics_jacobian(
        fluid, fluid_state_kp1, fluid_state_k
    )
    
    @test Matrix(∂dynamics_∂properties) ≈ jacobian_props rtol=1e-7
    println("    ✓ Analytical Jacobian matches ForwardDiff Jacobian")
    
    # Test 6: Test gradient through nested function calls (via inject)
    println("  Testing gradient through nested function calls...")
    function loss_function(props)
        residual = calculate_fluid_dynamics_residual(
            inject_differentiable_params(fluid, props),
            fluid_state_kp1, fluid_state_k; recompute_bc_vector=true,
        )
        return 0.5 * sum(abs2, residual)
    end

    grad = ForwardDiff.gradient(loss_function, collect_differentiable_params(fluid))
    @test all(isfinite.(grad))
    @test norm(grad) > 0
    println("    ✓ Gradient through loss function is finite with norm: $(norm(grad))")
end

@testitem "Fluid State Initialization" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create zero state
    zero_state = zeros(fluid.n_states)
    
    @test length(zero_state) == fluid.n_states
    @test all(zero_state .== 0.0)
    println("  ✓ Zero state correctly initialized")
    
    # Create random state
    random_state = rand(fluid.n_states)
    
    @test length(random_state) == fluid.n_states
    @test all(random_state .>= 0.0)
    @test all(random_state .<= 1.0)
    println("  ✓ Random state correctly initialized")
    
    # Test state indexing (no BC duals anymore)
    velocity_part = random_state[fluid.velocity_indices]
    continuity_dual_part = random_state[fluid.continuity_dual_indices]

    @test length(velocity_part) == fluid.n_velocities
    @test length(continuity_dual_part) == fluid.n_continuity_constraints
    println("  ✓ State indexing works correctly")
    println("    (BC duals removed - BCs now embedded in stationarity)")
end

@testitem "Fluid with Different Boundary Conditions" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Test with lid cavity boundary condition
    println("  Testing lid cavity boundary condition...")
    fluid_lid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = :lid_cavity,
        gravity_constant = gravity_constant,
    )

    @test fluid_lid.boundary_condition_type == :lid_cavity
    @test fluid_lid.n_boundary_conditions > 0
    println("    ✓ Lid cavity fluid created successfully")

    # Verify that ALL boundary indices are zeroed in divergence for lid cavity
    for idx in fluid_lid.fvm_grid.v_boundary_indices
        @test all(fluid_lid.divergence[:, idx] .== 0.0)
    end
    println("    ✓ Lid cavity: all boundary columns zeroed in divergence operator")

    # Test with freestream boundary condition
    println("  Testing freestream boundary condition...")
    fluid_freestream = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = :freestream,
        gravity_constant = gravity_constant,
    )

    @test fluid_freestream.boundary_condition_type == :freestream
    @test fluid_freestream.n_boundary_conditions > 0
    println("    ✓ Freestream fluid created successfully")

    # Verify that only constant boundary indices are zeroed (not right boundary)
    constant_boundary_indices = setdiff(
        fluid_freestream.fvm_grid.v_boundary_indices,
        vcat(fluid_freestream.fvm_grid.vx_right_indices,
             fluid_freestream.fvm_grid.vy_right_indices)
    )
    for idx in constant_boundary_indices
        @test all(fluid_freestream.divergence[:, idx] .== 0.0)
    end
    println("    ✓ Freestream: constant boundary columns zeroed in divergence operator")

    # Verify that right boundary indices are NOT zeroed (they're handled by outflow BC)
    right_boundary_indices = vcat(
        fluid_freestream.fvm_grid.vx_right_indices,
        fluid_freestream.fvm_grid.vy_right_indices
    )
    @test any(idx -> any(fluid_freestream.divergence[:, idx] .!= 0.0), right_boundary_indices)
    println("    ✓ Freestream: right boundary columns retained in divergence operator")

    # Test continuity vector for freestream
    original_divergence = calculate_divergence_operator(fluid_freestream.fvm_grid)
    expected_continuity_vector = original_divergence * fluid_freestream.constant_boundary_condition_vector
    @test fluid_freestream.continuity_vector ≈ expected_continuity_vector rtol=1e-10
    println("    ✓ Freestream: continuity vector correctly computed")

    # Test mass conservation residual for freestream with different boundary velocities
    sample_velocity = rand(fluid_freestream.n_velocities)
    com_residual_1 = calculate_mass_conservation_constraint_residual(
        fluid_freestream, sample_velocity
    )

    new_boundary_velocity = [boundary_velocity_x + 0.5, boundary_velocity_y]
    com_residual_2 = calculate_mass_conservation_constraint_residual(
        fluid_freestream, sample_velocity;
        boundary_velocity=new_boundary_velocity,
        recompute_bc_vector=true
    )

    @test com_residual_1 != com_residual_2
    println("    ✓ Freestream: mass conservation residual responds to boundary velocity changes")
end

@testitem "Fluid with External Pressure Gradient" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create fluid with non-zero pressure gradient
    pressure_gradient = (100.0, 50.0)  # Pa/m
    fluid_pg = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = 0.0,  # Turn off gravity to isolate pressure gradient effect
        external_pressure_gradient = pressure_gradient,
    )
    
    @test fluid_pg.external_pressure_gradient == pressure_gradient
    println("  ✓ Pressure gradient correctly set: $pressure_gradient")
    
    # Test that pressure gradient force is non-zero
    @test any(fluid_pg.external_pressure_gradient_force .!= 0.0)

    # Pressure gradient force should only apply to interior velocities
    n_vx = fluid_pg.fvm_grid.n_vx
    v_interior_indices = fluid_pg.fvm_grid.v_interior_indices
    v_boundary_indices = fluid_pg.fvm_grid.v_boundary_indices

    # Boundary velocities should have zero pressure gradient force
    @test all(fluid_pg.external_pressure_gradient_force[v_boundary_indices] .== 0.0)
    println("  ✓ Boundary velocities have zero pressure gradient force")

    # Test x-component of pressure gradient force for interior vx
    expected_fx = -pressure_gradient[1]
    interior_vx_indices = filter(idx -> idx <= n_vx, v_interior_indices)
    @test all(fluid_pg.external_pressure_gradient_force[interior_vx_indices] .≈ expected_fx)
    println("  ✓ X-direction pressure gradient force correct for interior: $expected_fx Pa/m")

    # Test y-component of pressure gradient force for interior vy
    expected_fy = -pressure_gradient[2]
    interior_vy_indices = filter(idx -> idx > n_vx, v_interior_indices)
    @test all(fluid_pg.external_pressure_gradient_force[interior_vy_indices] .≈ expected_fy)
    println("  ✓ Y-direction pressure gradient force correct for interior: $expected_fy Pa/m")
end

@testitem "Sparse Matrix Properties" begin
    using Aquarium
    using SparseArrays

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Test that operators are sparse
    @test issparse(fluid.laplacian)
    @test issparse(fluid.divergence)
    @test issparse(fluid.constant_boundary_condition_matrix)
    println("  ✓ All operators are sparse matrices")
    
    # Test sparsity patterns
    laplacian_nnz = nnz(fluid.laplacian)
    divergence_nnz = nnz(fluid.divergence)
    
    println("  ✓ Laplacian non-zeros: $laplacian_nnz")
    println("  ✓ Divergence non-zeros: $divergence_nnz")
    
    # Laplacian should be relatively sparse for 2D grid
    total_laplacian_entries = fluid.n_velocities * fluid.n_velocities
    sparsity_ratio = laplacian_nnz / total_laplacian_entries
    @test sparsity_ratio < 0.1  # Should be < 10% filled
    println("  ✓ Laplacian sparsity: $(100 * sparsity_ratio)% filled")
end

@testitem "Potential Energy Calculation" begin
    using Aquarium
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Test potential energy calculation with gravity
    potential_energy = calculate_potential_energy(fluid)
    
    # Potential energy should be a scalar or vector depending on implementation
    # For fluid with gravity, PE = m*g*y for each cell
    @test isa(potential_energy, Union{Real, AbstractVector})
    println("  ✓ Potential energy calculated")
    
    # Test with no gravity (should be zero)
    fluid_no_gravity = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = 0.0,
    )
    potential_energy_no_gravity = calculate_potential_energy(fluid_no_gravity)
    
    if isa(potential_energy_no_gravity, AbstractVector)
        @test all(potential_energy_no_gravity .== 0.0)
        println("  ✓ Potential energy is zero with no gravity")
    else
        @test potential_energy_no_gravity == 0.0
        println("  ✓ Potential energy is zero with no gravity")
    end
    
    # Test potential energy with different density
    test_density = density * 1.5
    potential_energy_high_density = calculate_potential_energy(fluid; density=test_density)
    
    # Higher density should give higher potential energy
    if isa(potential_energy, AbstractVector) && isa(potential_energy_high_density, AbstractVector)
        # PE should scale linearly with density
        @test sum(potential_energy_high_density) ≈ sum(potential_energy) * 1.5 rtol=1e-10
        println("  ✓ Potential energy scales linearly with density")
    end
    
    # Test that potential energy is differentiable w.r.t. density
    pe_grad = ForwardDiff.derivative(
        ρ -> begin
            pe = calculate_potential_energy(fluid; density=ρ)
            return isa(pe, AbstractVector) ? sum(pe) : pe
        end,
        density
    )
    
    @test isfinite(pe_grad)
    @test pe_grad != 0.0  # Should be non-zero with gravity
    println("  ✓ Potential energy is differentiable w.r.t. density")
    println("    - Gradient w.r.t. density: $pe_grad")
end

@testitem "Kinetic Energy Calculation" begin
    using Aquarium
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create a sample velocity field
    sample_velocity = rand(fluid.n_velocities)
    
    # Calculate kinetic energy
    kinetic_energy = calculate_kinetic_energy(fluid, sample_velocity)
    
    @test isa(kinetic_energy, Real)
    @test kinetic_energy >= 0.0  # Kinetic energy must be non-negative
    println("  ✓ Kinetic energy is a non-negative scalar: $kinetic_energy J")
    
    # Test with zero velocity (should give zero kinetic energy)
    zero_velocity = zeros(fluid.n_velocities)
    kinetic_energy_zero = calculate_kinetic_energy(fluid, zero_velocity)
    
    @test kinetic_energy_zero ≈ 0.0 atol=1e-12
    println("  ✓ Kinetic energy is zero for zero velocity")
    
    # Test that KE scales with velocity squared
    scaled_velocity = 2.0 .* sample_velocity
    kinetic_energy_scaled = calculate_kinetic_energy(fluid, scaled_velocity)
    
    @test kinetic_energy_scaled ≈ 4.0 * kinetic_energy rtol=1e-10
    println("  ✓ Kinetic energy scales with velocity squared")
    
    # Test with full state vector (should extract velocities automatically)
    full_state = rand(fluid.n_states)
    full_state[fluid.velocity_indices] .= sample_velocity
    kinetic_energy_from_state = calculate_kinetic_energy(fluid, full_state)
    
    @test kinetic_energy_from_state ≈ kinetic_energy rtol=1e-10
    println("  ✓ Kinetic energy correctly extracts velocities from full state")
    
    # Test expected value: KE = 0.5 * sum(m * v^2)
    expected_ke = 0.5 * sum(fluid.cell_mass .* (sample_velocity .^ 2))
    @test kinetic_energy ≈ expected_ke rtol=1e-10
    println("  ✓ Kinetic energy matches expected formula: 0.5 * m * v²")
    
    # Test that kinetic energy is differentiable w.r.t. velocity
    ke_gradient = ForwardDiff.gradient(
        v -> calculate_kinetic_energy(fluid, v),
        sample_velocity
    )
    
    @test length(ke_gradient) == fluid.n_velocities
    @test all(isfinite.(ke_gradient))
    # Gradient should be m*v
    expected_gradient = fluid.cell_mass .* sample_velocity
    @test ke_gradient ≈ expected_gradient rtol=1e-10
    println("  ✓ Kinetic energy gradient w.r.t. velocity is correct: m*v")
end

@testitem "Total Energy Calculation" begin
    using Aquarium
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create a sample velocity field
    sample_velocity = rand(fluid.n_velocities)
    
    # Calculate total energy
    total_energy = calculate_total_energy(fluid, sample_velocity)
    
    @test isa(total_energy, Union{Real, AbstractVector})
    println("  ✓ Total energy calculated")
    
    # Calculate components separately
    potential_energy = calculate_potential_energy(fluid)
    kinetic_energy = calculate_kinetic_energy(fluid, sample_velocity)
    
    # Total energy should equal sum of PE and KE
    if isa(potential_energy, AbstractVector)
        expected_total = sum(potential_energy) + kinetic_energy
    else
        expected_total = potential_energy + kinetic_energy
    end
    
    if isa(total_energy, AbstractVector)
        @test sum(total_energy) ≈ expected_total rtol=1e-10
    else
        @test total_energy ≈ expected_total rtol=1e-10
    end
    println("  ✓ Total energy equals PE + KE")
    
    # Test with full state vector
    full_state = rand(fluid.n_states)
    full_state[fluid.velocity_indices] .= sample_velocity
    total_energy_from_state = calculate_total_energy(fluid, full_state)
    
    if isa(total_energy, AbstractVector) && isa(total_energy_from_state, AbstractVector)
        @test sum(total_energy_from_state) ≈ sum(total_energy) rtol=1e-10
    elseif isa(total_energy, Real) && isa(total_energy_from_state, Real)
        @test total_energy_from_state ≈ total_energy rtol=1e-10
    end
    println("  ✓ Total energy works with full state vector")
    
    # Test that total energy is differentiable w.r.t. velocity
    total_energy_gradient = ForwardDiff.gradient(
        v -> begin
            te = calculate_total_energy(fluid, v)
            return isa(te, AbstractVector) ? sum(te) : te
        end,
        sample_velocity
    )
    
    @test length(total_energy_gradient) == fluid.n_velocities
    @test all(isfinite.(total_energy_gradient))
    # Gradient should equal KE gradient (since PE doesn't depend on velocity)
    expected_gradient = fluid.cell_mass .* sample_velocity
    @test total_energy_gradient ≈ expected_gradient rtol=1e-10
    println("  ✓ Total energy gradient w.r.t. velocity is correct")
    
    # Test energy with zero velocity and gravity
    zero_velocity = zeros(fluid.n_velocities)
    total_energy_zero_velocity = calculate_total_energy(fluid, zero_velocity)
    
    # Should equal potential energy only
    if isa(total_energy_zero_velocity, AbstractVector) && isa(potential_energy, AbstractVector)
        @test sum(total_energy_zero_velocity) ≈ sum(potential_energy) rtol=1e-10
    elseif isa(total_energy_zero_velocity, Real)
        pe_sum = isa(potential_energy, AbstractVector) ? sum(potential_energy) : potential_energy
        @test total_energy_zero_velocity ≈ pe_sum rtol=1e-10
    end
    println("  ✓ Total energy equals PE when velocity is zero")
end

@testitem "Energy Conservation Properties" begin
    using Aquarium
    using LinearAlgebra
    using ForwardDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Test that energy functions work consistently across different fluid configurations
    
    # Create fluid with different properties
    println("  Testing energy calculations with different fluid properties...")
    fluid_dense = Fluid(time_step;
        density = density * 2.0,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
    )
    
    sample_velocity = rand(fluid.n_velocities)
    
    # Kinetic energy should scale with density
    ke_original = calculate_kinetic_energy(fluid, sample_velocity)
    ke_dense = calculate_kinetic_energy(fluid_dense, sample_velocity)
    
    @test ke_dense ≈ 2.0 * ke_original rtol=1e-10
    println("    ✓ Kinetic energy scales with density")
    
    # Potential energy should also scale with density
    pe_original = calculate_potential_energy(fluid)
    pe_dense = calculate_potential_energy(fluid_dense)
    
    pe_original_sum = isa(pe_original, AbstractVector) ? sum(pe_original) : pe_original
    pe_dense_sum = isa(pe_dense, AbstractVector) ? sum(pe_dense) : pe_dense
    
    @test pe_dense_sum ≈ 2.0 * pe_original_sum rtol=1e-10
    println("    ✓ Potential energy scales with density")
    
    # Test energy gradient consistency
    println("  Testing energy gradient consistency...")
    function energy_loss(v)
        te = calculate_total_energy(fluid, v)
        return isa(te, AbstractVector) ? sum(abs2, te) : te^2
    end
    
    energy_grad = ForwardDiff.gradient(energy_loss, sample_velocity)
    @test all(isfinite.(energy_grad))
    @test norm(energy_grad) > 0
    println("    ✓ Energy gradient is finite and non-zero")
    
end

@testitem "Ruiz Scaling on 200x200 Grid" begin
    using Aquarium
    using SparseArrays
    using Statistics

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Create a larger fluid grid for testing Ruiz scaling
    println("  Creating 200x200 fluid grid...")
    large_grid_size = (200, 200)
    large_grid_dimensions = (1.0, 1.0)

    fluid_large = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = large_grid_size,
        grid_dimensions = large_grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )

    println("    ✓ Large fluid created with $(fluid_large.n_states) states")

    # Create sample fluid states
    fluid_state_kp1 = rand(fluid_large.n_states)
    fluid_state_k = rand(fluid_large.n_states)

    # Get the KKT matrix (Jacobian of fluid dynamics)
    println("  Computing KKT matrix...")
    kkt_matrix, _, _ = calculate_fluid_dynamics_jacobian(
        fluid_large, fluid_state_kp1, fluid_state_k
    )

    @test size(kkt_matrix) == (fluid_large.n_states, fluid_large.n_states)
    @test issparse(kkt_matrix)
    println("    ✓ KKT matrix computed: size $(size(kkt_matrix)), nnz=$(nnz(kkt_matrix))")

    # Make a copy for comparison
    kkt_matrix_unscaled = deepcopy(kkt_matrix)

    # Compute condition properties before scaling
    println("  Computing row and column norms before scaling...")
    rows_before = rowvals(kkt_matrix_unscaled)
    vals_before = nonzeros(kkt_matrix_unscaled)
    colptr_before = kkt_matrix_unscaled.colptr

    # Compute row infinity norms before scaling
    row_norms_before = zeros(fluid_large.n_states)
    for col in 1:fluid_large.n_states
        for j in colptr_before[col]:(colptr_before[col+1]-1)
            row = rows_before[j]
            val = abs(vals_before[j])
            row_norms_before[row] = max(row_norms_before[row], val)
        end
    end

    # Compute column infinity norms before scaling
    col_norms_before = zeros(fluid_large.n_states)
    for col in 1:fluid_large.n_states
        local_max = 0.0
        for j in colptr_before[col]:(colptr_before[col+1]-1)
            local_max = max(local_max, abs(vals_before[j]))
        end
        col_norms_before[col] = local_max
    end

    # Filter out zero norms for statistics
    nonzero_row_norms_before = row_norms_before[row_norms_before .> 0]
    nonzero_col_norms_before = col_norms_before[col_norms_before .> 0]

    println("    ✓ Before scaling:")
    println("      - Row norm range: [$(minimum(nonzero_row_norms_before)), $(maximum(nonzero_row_norms_before))]")
    println("      - Col norm range: [$(minimum(nonzero_col_norms_before)), $(maximum(nonzero_col_norms_before))]")
    println("      - Row norm std: $(std(nonzero_row_norms_before))")
    println("      - Col norm std: $(std(nonzero_col_norms_before))")

    # Apply Ruiz scaling
    println("  Applying Ruiz scaling...")
    left_scale, right_scale = calculate_ruiz_scale!(kkt_matrix; max_iterations=10, tolerance=1e-10)

    @test length(left_scale) == fluid_large.n_states
    @test length(right_scale) == fluid_large.n_states
    @test all(isfinite.(left_scale))
    @test all(isfinite.(right_scale))
    @test all(left_scale .> 0)
    @test all(right_scale .> 0)
    println("    ✓ Ruiz scaling factors computed")

    # Compute condition properties after scaling
    println("  Computing row and column norms after scaling...")
    rows_after = rowvals(kkt_matrix)
    vals_after = nonzeros(kkt_matrix)
    colptr_after = kkt_matrix.colptr

    # Compute row infinity norms after scaling
    row_norms_after = zeros(fluid_large.n_states)
    for col in 1:fluid_large.n_states
        for j in colptr_after[col]:(colptr_after[col+1]-1)
            row = rows_after[j]
            val = abs(vals_after[j])
            row_norms_after[row] = max(row_norms_after[row], val)
        end
    end

    # Compute column infinity norms after scaling
    col_norms_after = zeros(fluid_large.n_states)
    for col in 1:fluid_large.n_states
        local_max = 0.0
        for j in colptr_after[col]:(colptr_after[col+1]-1)
            local_max = max(local_max, abs(vals_after[j]))
        end
        col_norms_after[col] = local_max
    end

    # Filter out zero norms for statistics
    nonzero_row_norms_after = row_norms_after[row_norms_after .> 0]
    nonzero_col_norms_after = col_norms_after[col_norms_after .> 0]

    println("    ✓ After scaling:")
    println("      - Row norm range: [$(minimum(nonzero_row_norms_after)), $(maximum(nonzero_row_norms_after))]")
    println("      - Col norm range: [$(minimum(nonzero_col_norms_after)), $(maximum(nonzero_col_norms_after))]")
    println("      - Row norm std: $(std(nonzero_row_norms_after))")
    println("      - Col norm std: $(std(nonzero_col_norms_after))")
    println(" ")

    # Test that Ruiz scaling equilibrates the matrix
    # After Ruiz scaling, row and column norms should be closer to 1
    @test maximum(nonzero_row_norms_after) < 10.0  # Should be reasonably bounded
    @test minimum(nonzero_row_norms_after) > 0.1   # Should be reasonably bounded
    @test maximum(nonzero_col_norms_after) < 10.0  # Should be reasonably bounded
    @test minimum(nonzero_col_norms_after) > 0.1   # Should be reasonably bounded

    # Test that variance is reduced (better equilibration)
    @test std(nonzero_row_norms_after) < std(nonzero_row_norms_before)
    @test std(nonzero_col_norms_after) < std(nonzero_col_norms_before)
    println("    ✓ Ruiz scaling improved equilibration (reduced variance)")

    # Test that the matrix is still the same after applying inverse scaling
    println("  Verifying scaling consistency...")
    # The scaled matrix should satisfy: scaled = Diag(left_scale) * original * Diag(right_scale)
    # So: original = Diag(1/left_scale) * scaled * Diag(1/right_scale)
    left_inv = 1.0 ./ left_scale
    right_inv = 1.0 ./ right_scale

    # Unscale the matrix
    kkt_reconstructed = deepcopy(kkt_matrix)
    rows_recon = rowvals(kkt_reconstructed)
    vals_recon = nonzeros(kkt_reconstructed)
    colptr_recon = kkt_reconstructed.colptr

    # Apply left inverse scaling
    for col in 1:fluid_large.n_states
        for j in colptr_recon[col]:(colptr_recon[col+1]-1)
            row = rows_recon[j]
            vals_recon[j] *= left_inv[row]
        end
    end

    # Apply right inverse scaling
    for col in 1:fluid_large.n_states
        for j in colptr_recon[col]:(colptr_recon[col+1]-1)
            vals_recon[j] *= right_inv[col]
        end
    end

    @test kkt_reconstructed ≈ kkt_matrix_unscaled rtol=1e-10
    println("    ✓ Scaling is reversible (unscaling recovers original matrix)")

    # Test the alternative scale_linear_system! function that takes pre-computed scales
    println("  Testing scale_linear_system! with pre-computed scales...")

    # Create a fresh unscaled matrix
    kkt_matrix_fresh = deepcopy(kkt_matrix_unscaled)
    residual_test = rand(fluid_large.n_states)
    residual_copy = copy(residual_test)

    # Apply scaling using the new function with the previously computed scales
    scale_linear_system!(kkt_matrix_fresh, residual_test, left_scale, right_scale)

    # The result should match what we got from calculate_ruiz_scale!
    @test kkt_matrix_fresh ≈ kkt_matrix rtol=1e-12
    println("    ✓ scale_linear_system! with pre-computed scales produces identical matrix")

    # Test that residual is scaled correctly
    @test residual_test ≈ residual_copy .* left_scale rtol=1e-12
    println("    ✓ Residual scaling is correct")

    # Verify the scaled matrix still has good equilibration
    rows_new = rowvals(kkt_matrix_fresh)
    vals_new = nonzeros(kkt_matrix_fresh)
    colptr_new = kkt_matrix_fresh.colptr

    row_norms_new = zeros(fluid_large.n_states)
    for col in 1:fluid_large.n_states
        for j in colptr_new[col]:(colptr_new[col+1]-1)
            row = rows_new[j]
            val = abs(vals_new[j])
            row_norms_new[row] = max(row_norms_new[row], val)
        end
    end

    nonzero_row_norms_new = row_norms_new[row_norms_new .> 0]
    @test maximum(nonzero_row_norms_new) < 10.0
    @test minimum(nonzero_row_norms_new) > 0.1
    println("    ✓ Matrix equilibration maintained with pre-computed scales")
end

@testitem "Simulation Objective Value and Trajectory" begin
    using Aquarium
    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Setup simulation parameters
    final_time = 0.05  # Short simulation

    # Create a simple fluid for testing
    time_step_test = 0.01
    density_test = 1000.0
    viscosity_test = 0.001
    grid_size_test = (8, 8)
    grid_dimensions_test = (1.0, 1.0)

    fluid_test = Fluid(time_step_test;
        density = density_test,
        dynamic_viscosity = viscosity_test,
        boundary_velocity = [1.0, 0.0],
        grid_size = grid_size_test,
        grid_dimensions = grid_dimensions_test,
        boundary_condition_type = :lid_cavity,
        gravity_constant = 0.0,
    )

    # Initial state (start from rest)
    initial_fluid_state = zeros(fluid_test.n_states)

    # Define stage objective: sum of all states and properties
    function calculate_stage_objective(fluid, time, fluid_state; fluid_properties=[fluid.density, fluid.dynamic_viscosity, fluid.boundary_velocity[1], fluid.boundary_velocity[2]])
        return sum(abs.(fluid_state)) + sum(fluid_properties)
    end

    # Define terminal objective: penalize final state
    function calculate_terminal_objective(fluid, time, fluid_state; fluid_properties=[fluid.density, fluid.dynamic_viscosity, fluid.boundary_velocity[1], fluid.boundary_velocity[2]])
        return 0.5 * sum(abs.(fluid_state)) + sum(fluid_properties)
    end

    # Wrapper to compute total objective for verification
    function simple_objective(fluid, state_traj, time_traj)
        total = 0.0
        for k = 1:(length(time_traj)-1)
            total += calculate_stage_objective(fluid, time_traj[k], state_traj[k])
        end
        total += calculate_terminal_objective(fluid, time_traj[end], state_traj[end])
        return total
    end

    # Run simulation with objective computation
    println("  Running simulation with objective computation...")
    trajectories = simulate_fluid(
        fluid_test,
        initial_fluid_state,
        final_time;
        solver_type=:gmres,
        preconditioner_type=:ilu,
        scaling_type=:ruiz,
        pivot_type=:rcm,
        dual_regularization=1e-6,
        calculate_stage_objective=calculate_stage_objective,
        calculate_terminal_objective=calculate_terminal_objective,
        calculate_objective=true,
        verbose=false
    )

    # Extract objective information
    total_objective = trajectories[:objective_value][1]
    stage_objective_traj = trajectories[:objective_traj][1:end-1]
    terminal_objective = trajectories[:objective_traj][end]

    # Test 1: Total objective is sum of stage and terminal objectives
    expected_total = simple_objective(fluid_test, 
        [vcat(trajectories[:fluid_state_traj][k]) for k = 1:length(trajectories[:t_traj])],
        trajectories[:t_traj])
    @test total_objective ≈ expected_total rtol=1e-10
    println("  ✓ Total objective = sum(stage objectives) + terminal objective")
    println("    Total: $(total_objective)")
    println("    Expected: $(expected_total)")

    # Test 2: Stage objective trajectory has correct length
    n_steps = length(trajectories[:t_traj]) - 1
    @test length(stage_objective_traj) == n_steps
    println("  ✓ Stage objective trajectory has correct length: $(n_steps)")

    # Test 3: All stage objectives are non-negative (for sum objective)
    @test all(s -> s >= 0.0, stage_objective_traj)
    println("  ✓ All stage objectives are non-negative")

    # Test 4: Terminal objective is non-negative (for sum objective)
    @test terminal_objective >= 0.0
    println("  ✓ Terminal objective is non-negative")

    # Test 5: Objective value field exists and is correct
    @test haskey(trajectories, :objective_value)
    @test haskey(trajectories, :objective_traj)
    println("  ✓ Objective data stored in trajectories")
end

@testitem "Gradient Computation" begin
    using Aquarium
    using LinearAlgebra
    using FiniteDiff

    time_step = 0.01
    density = 1000.0
    dynamic_viscosity = 0.001
    boundary_velocity_x = 1.0
    boundary_velocity_y = 0.0
    fluid_properties = [density, dynamic_viscosity, boundary_velocity_x, boundary_velocity_y]
    grid_size = (10, 10)
    grid_dimensions = (1.0, 1.0)
    boundary_condition_type = :lid_cavity
    gravity_constant = 9.81
    external_pressure_gradient = (0.0, 0.0)

    fluid = Fluid(time_step;
        density = density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [boundary_velocity_x, boundary_velocity_y],
        grid_size = grid_size,
        grid_dimensions = grid_dimensions,
        boundary_condition_type = boundary_condition_type,
        gravity_constant = gravity_constant,
        external_pressure_gradient = external_pressure_gradient,
    )


    # Setup simulation parameters
    final_time = 0.1  # Very short simulation for faster testing

    # Create a simple fluid for testing
    time_step_test = 0.01
    density_test = 1000.0
    viscosity_test = 0.001
    fluid_properties_test = [density_test, viscosity_test, 1.0, 0.0]
    grid_size_test = (6, 6)  # Smaller grid for faster testing
    grid_dimensions_test = (1.0, 1.0)

    fluid_test = Fluid(time_step_test;
        density = density_test,
        dynamic_viscosity = viscosity_test,
        boundary_velocity = [1.0, 0.0],
        grid_size = grid_size_test,
        grid_dimensions = grid_dimensions_test,
        boundary_condition_type = :lid_cavity,
        gravity_constant = 0.0,
    )

    # Initial state (start from rest)
    initial_fluid_state = zeros(fluid_test.n_states)

    # Define stage objective: sum of fluid velocities and properties
    function calculate_stage_objective(fluid, time, fluid_state; fluid_properties=[fluid.density, fluid.dynamic_viscosity, fluid.boundary_velocity[1], fluid.boundary_velocity[2]])
        return 0.01 * sum(fluid_state) + 2.0 * sum(fluid_properties)
    end

    # Define terminal objective
    function calculate_terminal_objective(fluid, time, fluid_state; fluid_properties=[fluid.density, fluid.dynamic_viscosity, fluid.boundary_velocity[1], fluid.boundary_velocity[2]])
        return 0.01 * sum(fluid_state) + 2.0 * sum(fluid_properties)
    end

    # Wrapper to compute total objective for finite differences
    function simple_objective(fluid, state_traj, time_traj)
        total = 0.0
        for k = 1:(length(time_traj)-1)
            total += calculate_stage_objective(fluid, time_traj[k], state_traj[k])
        end
        total += calculate_terminal_objective(fluid, time_traj[end], state_traj[end])
        return total
    end

    # Fluid properties Jacobian for initial state (zero for this test)
    ∂initial_fluid_state_∂props = zeros(length(initial_fluid_state), length(fluid_properties_test))

    # Run simulation WITH gradient computation
    println("  Running simulation with gradient computation...")
    trajectories = simulate_fluid(
        fluid_test,
        initial_fluid_state,
        final_time;
        solver_type=:gmres,
        preconditioner_type=:ilu,
        scaling_type=:ruiz,
        pivot_type=:rcm,
        dual_regularization=1e-6,
        calculate_stage_objective=calculate_stage_objective,
        calculate_terminal_objective=calculate_terminal_objective,
        initial_fluid_state_fluid_properties_jacobian=∂initial_fluid_state_∂props,
        calculate_objective=true,
        gradient_method=:forward,
        verbose=false
    )

    # Extract analytical gradients
    ∂J_∂fluid_properties_analytical = trajectories[:objective_gradient_wrt_fluid_properties]

    println("  Computing finite difference gradients...")

    # Define objective function for fluid properties
    function objective_wrt_fluid_properties(props)
        # Create fluid with new properties
        new_fluid = Fluid(time_step_test;
            density = props[1],
            dynamic_viscosity = props[2],
            boundary_velocity = [props[3], props[4]],
            grid_size = grid_size_test,
            grid_dimensions = grid_dimensions_test,
            boundary_condition_type = :lid_cavity,
            gravity_constant = 0.0,
        )

        # Simulate
        traj = simulate_fluid(
            new_fluid,
            initial_fluid_state,
            final_time;
            solver_type=:gmres,
            preconditioner_type=:ilu,
            scaling_type=:ruiz,
            pivot_type=:rcm,
            dual_regularization=1e-6,
            calculate_stage_objective=calculate_stage_objective,
            calculate_terminal_objective=calculate_terminal_objective,
            calculate_objective=false,
            verbose=false
        )

        # Compute objective
        J = simple_objective(new_fluid, traj[:fluid_state_traj], traj[:t_traj])

        return J
    end

    # Compute finite difference gradients
    ∂J_∂fluid_properties_fd = FiniteDiff.finite_difference_gradient(
        objective_wrt_fluid_properties, fluid_properties_test)

    # Compare analytical vs finite difference gradients
    println("  Comparing gradients...")

    # Test gradient w.r.t. fluid properties
    @test ∂J_∂fluid_properties_analytical ≈ ∂J_∂fluid_properties_fd rtol=1e-3
    println("  ✓ Gradient w.r.t. fluid_properties matches finite difference")
    println("    Analytical: $(∂J_∂fluid_properties_analytical)")
    println("    Finite Diff: $(∂J_∂fluid_properties_fd)")
    println("    Relative Error: $(norm(∂J_∂fluid_properties_analytical - ∂J_∂fluid_properties_fd) / norm(∂J_∂fluid_properties_fd))")
end
