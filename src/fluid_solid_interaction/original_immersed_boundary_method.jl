function calculate_original_fsi_kernel(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector{T}
) where T

    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # determine number of eulerian and lagrange coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_nodes = topology.n_boundary_nodes

    kind = topology.discrete_delta_kind
    support = delta_support_radius(kind)

    # calculate delta function values for fluid x- and y-velocities (parallel sparse collection)
    fsi_kernel_I_vec, fsi_kernel_J_vec, fsi_kernel_V_vec = tcollect_sparse(1:n_boundary_nodes, T) do chunk
        local_I, local_J, local_V = Int[], Int[], T[]
        for i in chunk
            for j in 1:n_fluid_velocity_x
                r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_x[j]) / grid_spacing_x
                r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_x[j]) / grid_spacing_y

                (abs(r_x) > support || abs(r_y) > support) && continue

                kernel_x_value = discrete_delta_product(kind, r_x, r_y)

                if kernel_x_value != 0.0
                    push!(local_I, i)
                    push!(local_J, j)
                    push!(local_V, kernel_x_value)
                end
            end
            for j in 1:n_fluid_velocity_y
                r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_y[j]) / grid_spacing_x
                r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_y[j]) / grid_spacing_y

                (abs(r_x) > support || abs(r_y) > support) && continue

                kernel_y_value = discrete_delta_product(kind, r_x, r_y)

                if kernel_y_value != 0.0
                    push!(local_I, i + n_boundary_nodes)
                    push!(local_J, j + n_fluid_velocity_x)
                    push!(local_V, kernel_y_value)
                end
            end
        end
        (local_I, local_J, local_V)
    end

    fsi_kernel = sparse(
        fsi_kernel_I_vec,
        fsi_kernel_J_vec,
        fsi_kernel_V_vec,
        n_boundary_nodes * 2,
        n_fluid_velocity_x + n_fluid_velocity_y
    )

    return fsi_kernel

end

function calculate_original_fsi_kernel_vector_product(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector,
    fluid_velocity::AbstractVector,
)

    T = promote_type(eltype(boundary_configuration), eltype(fluid_velocity))

    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # Extract fluid velocity components
    fluid_velocity_x = fluid_velocity[1:fluid.fvm_grid.n_vx]
    fluid_velocity_y = fluid_velocity[fluid.fvm_grid.n_vx+1:end]

    # extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # determine number of eulerian and lagrange coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_nodes = topology.n_boundary_nodes

    kind = topology.discrete_delta_kind
    support = delta_support_radius(kind)

    # Initialize result vector
    result = zeros(T, n_boundary_nodes * 2)

    # calculate kernel-vector product for fluid x-velocities (parallel direct-write)
    tforeach(1:n_boundary_nodes) do i
        for j in 1:n_fluid_velocity_x
            r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_x[j]) / grid_spacing_x
            r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_x[j]) / grid_spacing_y

            (abs(r_x) > support || abs(r_y) > support) && continue

            kernel_x_value = discrete_delta_product(kind, r_x, r_y)
            result[i] += kernel_x_value * fluid_velocity_x[j]
        end
    end

    # calculate kernel-vector product for fluid y-velocities (parallel direct-write)
    tforeach(1:n_boundary_nodes) do i
        for j in 1:n_fluid_velocity_y
            r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_y[j]) / grid_spacing_x
            r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_y[j]) / grid_spacing_y

            (abs(r_x) > support || abs(r_y) > support) && continue

            kernel_y_value = discrete_delta_product(kind, r_x, r_y)
            result[n_boundary_nodes + i] += kernel_y_value * fluid_velocity_y[j]
        end
    end

    return result

end

function calculate_original_fsi_kernel_vector_product_jacobian(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector,
    fluid_velocity::AbstractVector,
)

    T = promote_type(eltype(boundary_configuration), eltype(fluid_velocity))

    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # Extract fluid velocity components
    fluid_velocity_x = fluid_velocity[1:fluid.fvm_grid.n_vx]
    fluid_velocity_y = fluid_velocity[fluid.fvm_grid.n_vx+1:end]

    # extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # determine number of eulerian and lagrange coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_nodes = topology.n_boundary_nodes

    kind = topology.discrete_delta_kind
    support = delta_support_radius(kind)

    # Initialize result vectors for Jacobians w.r.t. x and y boundary configurations
    jacobian_wrt_x = zeros(T, n_boundary_nodes * 2)
    jacobian_wrt_y = zeros(T, n_boundary_nodes * 2)

    # calculate Jacobian for fluid x-velocities (parallel direct-write)
    tforeach(1:n_boundary_nodes) do i
        for j in 1:n_fluid_velocity_x
            r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_x[j]) / grid_spacing_x
            r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_x[j]) / grid_spacing_y

            (abs(r_x) > support || abs(r_y) > support) && continue

            # Get kernel derivatives w.r.t. normalized distances
            ∂r_x, ∂r_y = discrete_delta_product_derivatives(kind, r_x, r_y)

            # Apply chain rule: ∂kernel/∂boundary_x = (∂kernel/∂r_x) * (∂r_x/∂boundary_x)
            # Note: ∂r_x/∂boundary_x = 1/grid_spacing_x (from r_x definition)
            ∂kernel_∂boundary_x = ∂r_x / grid_spacing_x
            ∂kernel_∂boundary_y = ∂r_y / grid_spacing_y

            # Accumulate Jacobian contributions: ∂(K·v)[i]/∂boundary_config[i] = Σⱼ (∂K[i,j]/∂boundary_config[i]) · v[j]
            jacobian_wrt_x[i] += ∂kernel_∂boundary_x * fluid_velocity_x[j]
            jacobian_wrt_y[i] += ∂kernel_∂boundary_y * fluid_velocity_x[j]
        end
    end

    # calculate Jacobian for fluid y-velocities (parallel direct-write)
    tforeach(1:n_boundary_nodes) do i
        for j in 1:n_fluid_velocity_y
            r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_y[j]) / grid_spacing_x
            r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_y[j]) / grid_spacing_y

            (abs(r_x) > support || abs(r_y) > support) && continue

            # Get kernel derivatives w.r.t. normalized distances
            ∂r_x, ∂r_y = discrete_delta_product_derivatives(kind, r_x, r_y)

            # Apply chain rule
            ∂kernel_∂boundary_x = ∂r_x / grid_spacing_x
            ∂kernel_∂boundary_y = ∂r_y / grid_spacing_y

            # Accumulate Jacobian contributions
            jacobian_wrt_x[n_boundary_nodes + i] += ∂kernel_∂boundary_x * fluid_velocity_y[j]
            jacobian_wrt_y[n_boundary_nodes + i] += ∂kernel_∂boundary_y * fluid_velocity_y[j]
        end
    end

    # Build sparse Jacobian matrix (diagonal blocks)
    # Output is [boundary_vel_x; boundary_vel_y], Input is [boundary_x; boundary_y]
    jacobian_I = Int[]
    jacobian_J = Int[]
    jacobian_V = T[]

    for i in 1:n_boundary_nodes
        # boundary_vel_x[i] depends on boundary_x[i] and boundary_y[i]
        if jacobian_wrt_x[i] != 0
            push!(jacobian_I, i)
            push!(jacobian_J, i)
            push!(jacobian_V, jacobian_wrt_x[i])
        end
        if jacobian_wrt_y[i] != 0
            push!(jacobian_I, i)
            push!(jacobian_J, i + n_boundary_nodes)
            push!(jacobian_V, jacobian_wrt_y[i])
        end
        # boundary_vel_y[i] depends on boundary_x[i] and boundary_y[i]
        if jacobian_wrt_x[i + n_boundary_nodes] != 0
            push!(jacobian_I, i + n_boundary_nodes)
            push!(jacobian_J, i)
            push!(jacobian_V, jacobian_wrt_x[i + n_boundary_nodes])
        end
        if jacobian_wrt_y[i + n_boundary_nodes] != 0
            push!(jacobian_I, i + n_boundary_nodes)
            push!(jacobian_J, i + n_boundary_nodes)
            push!(jacobian_V, jacobian_wrt_y[i + n_boundary_nodes])
        end
    end

    jacobian = sparse(
        jacobian_I,
        jacobian_J,
        jacobian_V,
        2 * n_boundary_nodes,
        2 * n_boundary_nodes
    )

    return jacobian

end

function calculate_original_fsi_vector_kernel_product(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector,
    no_slip_dual::AbstractVector,
)

    T = promote_type(eltype(boundary_configuration), eltype(no_slip_dual))

    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # Extract boundary dual vector components
    no_slip_dual_x = no_slip_dual[1:topology.n_boundary_nodes]
    no_slip_dual_y = no_slip_dual[topology.n_boundary_nodes+1:end]

    # extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # determine number of eulerian and lagrange coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_nodes = topology.n_boundary_nodes

    kind = topology.discrete_delta_kind
    support = delta_support_radius(kind)

    # calculate vector-kernel product for fluid x- and y-velocities (parallel accumulation)
    (result_vx, result_vy) = taccumulate(1:n_boundary_nodes) do chunk
        local_vx = zeros(T, n_fluid_velocity_x)
        local_vy = zeros(T, n_fluid_velocity_y)
        for i in chunk
            for j in 1:n_fluid_velocity_x
                r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_x[j]) / grid_spacing_x
                r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_x[j]) / grid_spacing_y

                (abs(r_x) > support || abs(r_y) > support) && continue

                kernel_x_value = discrete_delta_product(kind, r_x, r_y)
                local_vx[j] += kernel_x_value * no_slip_dual_x[i]
            end
            for j in 1:n_fluid_velocity_y
                r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_y[j]) / grid_spacing_x
                r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_y[j]) / grid_spacing_y

                (abs(r_x) > support || abs(r_y) > support) && continue

                kernel_y_value = discrete_delta_product(kind, r_x, r_y)
                local_vy[j] += kernel_y_value * no_slip_dual_y[i]
            end
        end
        (local_vx, local_vy)
    end

    # Combine x and y results
    result = zeros(T, n_fluid_velocity_x + n_fluid_velocity_y)
    result[1:n_fluid_velocity_x] .= result_vx
    result[n_fluid_velocity_x+1:end] .= result_vy

    return result

end

function calculate_original_fsi_vector_kernel_product_jacobian(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector,
    no_slip_dual::AbstractVector,
)

    T = promote_type(eltype(boundary_configuration), eltype(no_slip_dual))

    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # Extract boundary dual vector components
    no_slip_dual_x = no_slip_dual[1:topology.n_boundary_nodes]
    no_slip_dual_y = no_slip_dual[topology.n_boundary_nodes+1:end]

    # extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # determine number of eulerian and lagrange coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_nodes = topology.n_boundary_nodes

    # Sparse Jacobian construction
    # Rows: [fluid_vel_x; fluid_vel_y], Columns: [boundary_x; boundary_y]

    kind = topology.discrete_delta_kind
    support = delta_support_radius(kind)

    # calculate Jacobian for fluid x- and y-velocities (parallel sparse collection)
    jacobian_I_vec, jacobian_J_vec, jacobian_V_vec = tcollect_sparse(1:n_boundary_nodes, T) do chunk
        local_I, local_J, local_V = Int[], Int[], T[]
        for i in chunk
            for j in 1:n_fluid_velocity_x
                r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_x[j]) / grid_spacing_x
                r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_x[j]) / grid_spacing_y

                (abs(r_x) > support || abs(r_y) > support) && continue

                # Get kernel derivatives w.r.t. normalized distances
                ∂r_x, ∂r_y = discrete_delta_product_derivatives(kind, r_x, r_y)

                # Apply chain rule: ∂kernel/∂boundary_x = (∂kernel/∂r_x) * (∂r_x/∂boundary_x)
                ∂kernel_∂boundary_x = ∂r_x / grid_spacing_x
                ∂kernel_∂boundary_y = ∂r_y / grid_spacing_y

                # Transpose accumulation: ∂(K'·w)[j]/∂boundary_config[i] = (∂K[i,j]/∂boundary_config[i]) · w[i]
                jacobian_val_x = ∂kernel_∂boundary_x * no_slip_dual_x[i]
                jacobian_val_y = ∂kernel_∂boundary_y * no_slip_dual_x[i]

                # Row j (fluid_vel_x[j]), Column i (boundary_x[i])
                if jacobian_val_x != 0
                    push!(local_I, j)
                    push!(local_J, i)
                    push!(local_V, jacobian_val_x)
                end
                # Row j (fluid_vel_x[j]), Column i+n (boundary_y[i])
                if jacobian_val_y != 0
                    push!(local_I, j)
                    push!(local_J, i + n_boundary_nodes)
                    push!(local_V, jacobian_val_y)
                end
            end
            for j in 1:n_fluid_velocity_y
                r_x = (boundary_configuration_x[i] - x_coord_fluid_velocity_y[j]) / grid_spacing_x
                r_y = (boundary_configuration_y[i] - y_coord_fluid_velocity_y[j]) / grid_spacing_y

                (abs(r_x) > support || abs(r_y) > support) && continue

                # Get kernel derivatives w.r.t. normalized distances
                ∂r_x, ∂r_y = discrete_delta_product_derivatives(kind, r_x, r_y)

                # Apply chain rule
                ∂kernel_∂boundary_x = ∂r_x / grid_spacing_x
                ∂kernel_∂boundary_y = ∂r_y / grid_spacing_y

                # Transpose accumulation
                jacobian_val_x = ∂kernel_∂boundary_x * no_slip_dual_y[i]
                jacobian_val_y = ∂kernel_∂boundary_y * no_slip_dual_y[i]

                # Row j+n_vx (fluid_vel_y[j]), Column i (boundary_x[i])
                if jacobian_val_x != 0
                    push!(local_I, j + n_fluid_velocity_x)
                    push!(local_J, i)
                    push!(local_V, jacobian_val_x)
                end
                # Row j+n_vx (fluid_vel_y[j]), Column i+n (boundary_y[i])
                if jacobian_val_y != 0
                    push!(local_I, j + n_fluid_velocity_x)
                    push!(local_J, i + n_boundary_nodes)
                    push!(local_V, jacobian_val_y)
                end
            end
        end
        (local_I, local_J, local_V)
    end

    # Build sparse Jacobian matrix
    jacobian = sparse(
        jacobian_I_vec,
        jacobian_J_vec,
        jacobian_V_vec,
        n_fluid_velocity_x + n_fluid_velocity_y,
        2 * n_boundary_nodes
    )

    return jacobian

end

@testitem "Original FSI kernel matrix" begin
    using AquariumClosed
    using SparseArrays

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(5, 5), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                      ib_method=:original)
    topology = system.topology

    config = zeros(system.n_configurations)
    config[1] = 0.5
    config[2] = 0.5
    boundary_state = calculate_boundary_state(system, vcat(config, zeros(system.n_velocities)))
    boundary_config = boundary_state[topology.boundary_configuration_indices]

    K = calculate_original_fsi_kernel(fluid, topology, boundary_config)

    n_nodes = topology.n_boundary_nodes
    n_v = fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy
    @test size(K) == (2 * n_nodes, n_v)
    @test nnz(K) > 0

    # Partition of unity: row sums ≈ 1 for body well inside grid
    for i in 1:n_nodes
        @test sum(K[i, :]) ≈ 1.0 atol=1e-12
        @test sum(K[n_nodes + i, :]) ≈ 1.0 atol=1e-12
    end

    v = randn(n_v)
    Kv_matrix = K * v
    Kv_direct = calculate_original_fsi_kernel_vector_product(fluid, topology, boundary_config, v)
    @test Kv_matrix ≈ Kv_direct atol=1e-12

    dual = randn(2 * n_nodes)
    Ktd_matrix = K' * dual
    Ktd_direct = calculate_original_fsi_vector_kernel_product(fluid, topology, boundary_config, dual)
    @test Ktd_matrix ≈ Ktd_direct atol=1e-12
end

@testitem "Original FSI kernel Jacobians" begin
    using AquariumClosed
    using ForwardDiff

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(5, 5), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                      ib_method=:original)
    topology = system.topology

    config = zeros(system.n_configurations)
    config[1] = 0.5
    config[2] = 0.5
    boundary_state = calculate_boundary_state(system, vcat(config, zeros(system.n_velocities)))
    boundary_config = boundary_state[topology.boundary_configuration_indices]

    v = randn(fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy)
    dual = randn(2 * topology.n_boundary_nodes)

    J_kv = calculate_original_fsi_kernel_vector_product_jacobian(
        fluid, topology, boundary_config, v)
    J_kv_fd = ForwardDiff.jacobian(
        bc -> calculate_original_fsi_kernel_vector_product(fluid, topology, bc, v),
        boundary_config)
    @test Matrix(J_kv) ≈ J_kv_fd atol=1e-8

    J_vk = calculate_original_fsi_vector_kernel_product_jacobian(
        fluid, topology, boundary_config, dual)
    J_vk_fd = ForwardDiff.jacobian(
        bc -> calculate_original_fsi_vector_kernel_product(fluid, topology, bc, dual),
        boundary_config)
    @test Matrix(J_vk) ≈ J_vk_fd atol=1e-8
end

@testitem "Original FSI kernel matrix (three-point)" begin
    using AquariumClosed
    using SparseArrays

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(5, 5), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )

    system_1pt = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                          ib_method=:original, discrete_delta_kind=:one_point)
    system_3pt = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                          ib_method=:original, discrete_delta_kind=:three_point)

    config = zeros(system_3pt.n_configurations)
    config[1] = 0.5
    config[2] = 0.5
    boundary_state = calculate_boundary_state(system_3pt, vcat(config, zeros(system_3pt.n_velocities)))
    boundary_config = boundary_state[system_3pt.topology.boundary_configuration_indices]

    topology_3pt = system_3pt.topology
    topology_1pt = system_1pt.topology

    K_3pt = calculate_original_fsi_kernel(fluid, topology_3pt, boundary_config)
    K_1pt = calculate_original_fsi_kernel(fluid, topology_1pt, boundary_config)

    n_nodes = topology_3pt.n_boundary_nodes
    n_v = fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy
    @test size(K_3pt) == (2 * n_nodes, n_v)
    @test nnz(K_3pt) > 0

    # Wider stencil: three-point should have at least as many nonzeros
    @test nnz(K_3pt) >= nnz(K_1pt)

    # Partition of unity: row sums ≈ 1 for body well inside grid
    for i in 1:n_nodes
        @test sum(K_3pt[i, :]) ≈ 1.0 atol=1e-12
        @test sum(K_3pt[n_nodes + i, :]) ≈ 1.0 atol=1e-12
    end

    # K*v consistency
    v = randn(n_v)
    Kv_matrix = K_3pt * v
    Kv_direct = calculate_original_fsi_kernel_vector_product(fluid, topology_3pt, boundary_config, v)
    @test Kv_matrix ≈ Kv_direct atol=1e-12

    # K'*d consistency
    dual = randn(2 * n_nodes)
    Ktd_matrix = K_3pt' * dual
    Ktd_direct = calculate_original_fsi_vector_kernel_product(fluid, topology_3pt, boundary_config, dual)
    @test Ktd_matrix ≈ Ktd_direct atol=1e-12
end

@testitem "Original FSI kernel Jacobians (three-point)" begin
    using AquariumClosed
    using ForwardDiff

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(5, 5), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                      ib_method=:original, discrete_delta_kind=:three_point)
    topology = system.topology

    config = zeros(system.n_configurations)
    config[1] = 0.5
    config[2] = 0.5
    boundary_state = calculate_boundary_state(system, vcat(config, zeros(system.n_velocities)))
    boundary_config = boundary_state[topology.boundary_configuration_indices]

    v = randn(fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy)
    dual = randn(2 * topology.n_boundary_nodes)

    J_kv = calculate_original_fsi_kernel_vector_product_jacobian(
        fluid, topology, boundary_config, v)
    J_kv_fd = ForwardDiff.jacobian(
        bc -> calculate_original_fsi_kernel_vector_product(fluid, topology, bc, v),
        boundary_config)
    @test Matrix(J_kv) ≈ J_kv_fd atol=1e-8

    J_vk = calculate_original_fsi_vector_kernel_product_jacobian(
        fluid, topology, boundary_config, dual)
    J_vk_fd = ForwardDiff.jacobian(
        bc -> calculate_original_fsi_vector_kernel_product(fluid, topology, bc, dual),
        boundary_config)
    @test Matrix(J_vk) ≈ J_vk_fd atol=1e-8
end