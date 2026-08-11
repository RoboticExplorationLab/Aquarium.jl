function calculate_weak_form_fsi_kernel(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector{T}
) where T

    # Extract boundary node coordinates
    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    # Extract segment start and end coordinates
    startpoints_x = boundary_configuration_x[topology.boundary_segment_start_nodes]
    endpoints_x = boundary_configuration_x[topology.boundary_segment_end_nodes]
    startpoints_y = boundary_configuration_y[topology.boundary_segment_start_nodes]
    endpoints_y = boundary_configuration_y[topology.boundary_segment_end_nodes]

    # Extract fluid grid coordinates
    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # Extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # Determine number of eulerian and lagrangian coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_segments = topology.n_boundary_segments

    kind = topology.discrete_delta_kind
    support_x = grid_spacing_x * delta_support_radius(kind)
    support_y = grid_spacing_y * delta_support_radius(kind)

    # Calculate delta function integral values for both x and y velocities (single parallel pass)
    fsi_kernel_I_vec, fsi_kernel_J_vec, fsi_kernel_V_vec = tcollect_sparse(1:n_boundary_segments, T) do chunk
        local_I, local_J, local_V = Int[], Int[], T[]
        for i in chunk
            # x-velocity contributions
            for j in 1:n_fluid_velocity_x
                min_dist_x = max(0.0, x_coord_fluid_velocity_x[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_x[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_x[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_x[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                kernel_value = delta_product_segment_integral(
                    grid_spacing_x, grid_spacing_y,
                    x_coord_fluid_velocity_x[j], y_coord_fluid_velocity_x[j],
                    startpoints_x[i], endpoints_x[i],
                    startpoints_y[i], endpoints_y[i],
                    kind,
                )

                if kernel_value != 0.0
                    push!(local_I, i)
                    push!(local_J, j)
                    push!(local_V, kernel_value)
                end
            end

            # y-velocity contributions
            for j in 1:n_fluid_velocity_y
                min_dist_x = max(0.0, x_coord_fluid_velocity_y[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_y[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_y[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_y[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                kernel_value = delta_product_segment_integral(
                    grid_spacing_x, grid_spacing_y,
                    x_coord_fluid_velocity_y[j], y_coord_fluid_velocity_y[j],
                    startpoints_x[i], endpoints_x[i],
                    startpoints_y[i], endpoints_y[i],
                    kind,
                )

                if kernel_value != 0.0
                    push!(local_I, i + n_boundary_segments)
                    push!(local_J, j + n_fluid_velocity_x)
                    push!(local_V, kernel_value)
                end
            end
        end
        (local_I, local_J, local_V)
    end

    fsi_kernel = sparse(
        fsi_kernel_I_vec,
        fsi_kernel_J_vec,
        fsi_kernel_V_vec,
        n_boundary_segments * 2,
        n_fluid_velocity_x + n_fluid_velocity_y
    )

    return fsi_kernel

end

function calculate_weak_form_fsi_kernel_vector_product(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector,
    fluid_velocity::AbstractVector,
)

    T = promote_type(eltype(boundary_configuration), eltype(fluid_velocity))

    # Extract boundary node coordinates
    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    # Extract segment start and end coordinates
    startpoints_x = boundary_configuration_x[topology.boundary_segment_start_nodes]
    endpoints_x = boundary_configuration_x[topology.boundary_segment_end_nodes]
    startpoints_y = boundary_configuration_y[topology.boundary_segment_start_nodes]
    endpoints_y = boundary_configuration_y[topology.boundary_segment_end_nodes]

    # Extract fluid grid coordinates
    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # Extract fluid velocity components
    fluid_velocity_x = fluid_velocity[1:fluid.fvm_grid.n_vx]
    fluid_velocity_y = fluid_velocity[fluid.fvm_grid.n_vx+1:end]

    # Extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # Determine number of eulerian and lagrangian coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_segments = topology.n_boundary_segments

    kind = topology.discrete_delta_kind
    support_x = grid_spacing_x * delta_support_radius(kind)
    support_y = grid_spacing_y * delta_support_radius(kind)

    # Initialize result vectors (direct-write safe: each i writes to disjoint index)
    result_x = zeros(T, n_boundary_segments)
    result_y = zeros(T, n_boundary_segments)

    # Calculate kernel-vector product for fluid x-velocities (task-based parallel, direct-write)
    chunks = Iterators.partition(1:n_boundary_segments, cld(n_boundary_segments, max(1, Threads.nthreads())))
    @sync for chunk in chunks
        Threads.@spawn for i in chunk
            for j in 1:n_fluid_velocity_x
                min_dist_x = max(0.0, x_coord_fluid_velocity_x[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_x[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_x[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_x[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                kernel_value = delta_product_segment_integral(
                    grid_spacing_x, grid_spacing_y,
                    x_coord_fluid_velocity_x[j], y_coord_fluid_velocity_x[j],
                    startpoints_x[i], endpoints_x[i],
                    startpoints_y[i], endpoints_y[i],
                    kind,
                )

                result_x[i] += kernel_value * fluid_velocity_x[j]
            end
        end
    end

    # Calculate kernel-vector product for fluid y-velocities (task-based parallel, direct-write)
    @sync for chunk in chunks
        Threads.@spawn for i in chunk
            for j in 1:n_fluid_velocity_y
                min_dist_x = max(0.0, x_coord_fluid_velocity_y[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_y[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_y[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_y[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                kernel_value = delta_product_segment_integral(
                    grid_spacing_x, grid_spacing_y,
                    x_coord_fluid_velocity_y[j], y_coord_fluid_velocity_y[j],
                    startpoints_x[i], endpoints_x[i],
                    startpoints_y[i], endpoints_y[i],
                    kind,
                )

                result_y[i] += kernel_value * fluid_velocity_y[j]
            end
        end
    end

    return vcat(result_x, result_y)

end

function calculate_weak_form_fsi_kernel_vector_product_jacobian(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector,
    fluid_velocity::AbstractVector,
)

    T = promote_type(eltype(boundary_configuration), eltype(fluid_velocity))

    # Extract boundary node coordinates
    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    # Extract segment start and end coordinates
    startpoints_x = boundary_configuration_x[topology.boundary_segment_start_nodes]
    endpoints_x = boundary_configuration_x[topology.boundary_segment_end_nodes]
    startpoints_y = boundary_configuration_y[topology.boundary_segment_start_nodes]
    endpoints_y = boundary_configuration_y[topology.boundary_segment_end_nodes]

    # Extract fluid grid coordinates
    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # Extract fluid velocity components
    fluid_velocity_x = fluid_velocity[1:fluid.fvm_grid.n_vx]
    fluid_velocity_y = fluid_velocity[fluid.fvm_grid.n_vx+1:end]

    # Extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # Determine number of eulerian and lagrangian coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_segments = topology.n_boundary_segments
    n_boundary_nodes = topology.n_boundary_nodes

    kind = topology.discrete_delta_kind
    support_x = grid_spacing_x * delta_support_radius(kind)
    support_y = grid_spacing_y * delta_support_radius(kind)

    # Calculate Jacobian entries for both x and y segment constraints (single parallel pass)
    jacobian_I_vec, jacobian_J_vec, jacobian_V_vec = tcollect_sparse(1:n_boundary_segments, T) do chunk
        local_I, local_J, local_V = Int[], Int[], T[]
        for i in chunk
            start_node = topology.boundary_segment_start_nodes[i]
            end_node = topology.boundary_segment_end_nodes[i]

            # x-velocity contributions
            for j in 1:n_fluid_velocity_x
                min_dist_x = max(0.0, x_coord_fluid_velocity_x[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_x[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_x[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_x[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                ∂kernel_∂x_startpoint, ∂kernel_∂x_endpoint,
                ∂kernel_∂y_startpoint, ∂kernel_∂y_endpoint =
                    delta_product_segment_integral_derivatives(
                        grid_spacing_x, grid_spacing_y,
                        x_coord_fluid_velocity_x[j], y_coord_fluid_velocity_x[j],
                        startpoints_x[i], endpoints_x[i],
                        startpoints_y[i], endpoints_y[i],
                        kind,
                    )

                push!(local_I, i);                          push!(local_J, start_node);                    push!(local_V, ∂kernel_∂x_startpoint * fluid_velocity_x[j])
                push!(local_I, i);                          push!(local_J, end_node);                      push!(local_V, ∂kernel_∂x_endpoint * fluid_velocity_x[j])
                push!(local_I, i);                          push!(local_J, start_node + n_boundary_nodes); push!(local_V, ∂kernel_∂y_startpoint * fluid_velocity_x[j])
                push!(local_I, i);                          push!(local_J, end_node + n_boundary_nodes);   push!(local_V, ∂kernel_∂y_endpoint * fluid_velocity_x[j])
            end

            # y-velocity contributions
            for j in 1:n_fluid_velocity_y
                min_dist_x = max(0.0, x_coord_fluid_velocity_y[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_y[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_y[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_y[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                ∂kernel_∂x_startpoint, ∂kernel_∂x_endpoint,
                ∂kernel_∂y_startpoint, ∂kernel_∂y_endpoint =
                    delta_product_segment_integral_derivatives(
                        grid_spacing_x, grid_spacing_y,
                        x_coord_fluid_velocity_y[j], y_coord_fluid_velocity_y[j],
                        startpoints_x[i], endpoints_x[i],
                        startpoints_y[i], endpoints_y[i],
                        kind,
                    )

                push!(local_I, i + n_boundary_segments);   push!(local_J, start_node);                    push!(local_V, ∂kernel_∂x_startpoint * fluid_velocity_y[j])
                push!(local_I, i + n_boundary_segments);   push!(local_J, end_node);                      push!(local_V, ∂kernel_∂x_endpoint * fluid_velocity_y[j])
                push!(local_I, i + n_boundary_segments);   push!(local_J, start_node + n_boundary_nodes); push!(local_V, ∂kernel_∂y_startpoint * fluid_velocity_y[j])
                push!(local_I, i + n_boundary_segments);   push!(local_J, end_node + n_boundary_nodes);   push!(local_V, ∂kernel_∂y_endpoint * fluid_velocity_y[j])
            end
        end
        (local_I, local_J, local_V)
    end

    # Build sparse Jacobian matrix
    # Rows: segment constraints (2 * n_boundary_segments)
    # Cols: boundary node coordinates (2 * n_boundary_nodes)
    jacobian = sparse(
        jacobian_I_vec,
        jacobian_J_vec,
        jacobian_V_vec,
        2 * n_boundary_segments,
        2 * n_boundary_nodes
    )

    return jacobian

end

function calculate_weak_form_fsi_vector_kernel_product(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector,
    no_slip_dual::AbstractVector,
)

    T = promote_type(eltype(boundary_configuration), eltype(no_slip_dual))

    # Extract boundary node coordinates
    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    # Extract segment start and end coordinates
    startpoints_x = boundary_configuration_x[topology.boundary_segment_start_nodes]
    endpoints_x = boundary_configuration_x[topology.boundary_segment_end_nodes]
    startpoints_y = boundary_configuration_y[topology.boundary_segment_start_nodes]
    endpoints_y = boundary_configuration_y[topology.boundary_segment_end_nodes]

    # Extract fluid grid coordinates
    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # Extract boundary dual vector components
    no_slip_dual_x = no_slip_dual[1:topology.n_boundary_segments]
    no_slip_dual_y = no_slip_dual[topology.n_boundary_segments+1:end]

    # Extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # Determine number of eulerian and lagrangian coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_segments = topology.n_boundary_segments

    kind = topology.discrete_delta_kind
    support_x = grid_spacing_x * delta_support_radius(kind)
    support_y = grid_spacing_y * delta_support_radius(kind)

    # Calculate vector-kernel product for both x and y velocities (single parallel pass)
    (result_vx, result_vy) = taccumulate(1:n_boundary_segments) do chunk
        local_vx = zeros(T, n_fluid_velocity_x)
        local_vy = zeros(T, n_fluid_velocity_y)
        for i in chunk
            for j in 1:n_fluid_velocity_x
                min_dist_x = max(0.0, x_coord_fluid_velocity_x[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_x[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_x[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_x[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                kernel_value = delta_product_segment_integral(
                    grid_spacing_x, grid_spacing_y,
                    x_coord_fluid_velocity_x[j], y_coord_fluid_velocity_x[j],
                    startpoints_x[i], endpoints_x[i],
                    startpoints_y[i], endpoints_y[i],
                    kind,
                )

                local_vx[j] += kernel_value * no_slip_dual_x[i]
            end

            for j in 1:n_fluid_velocity_y
                min_dist_x = max(0.0, x_coord_fluid_velocity_y[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_y[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_y[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_y[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                kernel_value = delta_product_segment_integral(
                    grid_spacing_x, grid_spacing_y,
                    x_coord_fluid_velocity_y[j], y_coord_fluid_velocity_y[j],
                    startpoints_x[i], endpoints_x[i],
                    startpoints_y[i], endpoints_y[i],
                    kind,
                )

                local_vy[j] += kernel_value * no_slip_dual_y[i]
            end
        end
        (local_vx, local_vy)
    end

    return vcat(result_vx, result_vy)

end

function calculate_weak_form_fsi_vector_kernel_product_jacobian(
    fluid::Fluid,
    topology::SystemTopology,
    boundary_configuration::AbstractVector,
    no_slip_dual::AbstractVector,
)

    T = promote_type(eltype(boundary_configuration), eltype(no_slip_dual))

    # Extract boundary node coordinates
    boundary_configuration_x = boundary_configuration[1:topology.n_boundary_nodes]
    boundary_configuration_y = boundary_configuration[topology.n_boundary_nodes+1:end]

    # Extract segment start and end coordinates
    startpoints_x = boundary_configuration_x[topology.boundary_segment_start_nodes]
    endpoints_x = boundary_configuration_x[topology.boundary_segment_end_nodes]
    startpoints_y = boundary_configuration_y[topology.boundary_segment_start_nodes]
    endpoints_y = boundary_configuration_y[topology.boundary_segment_end_nodes]

    # Extract fluid grid coordinates
    x_coord_fluid_velocity_x = fluid.fvm_grid.x_coord_vx_flat
    y_coord_fluid_velocity_x = fluid.fvm_grid.y_coord_vx_flat

    x_coord_fluid_velocity_y = fluid.fvm_grid.x_coord_vy_flat
    y_coord_fluid_velocity_y = fluid.fvm_grid.y_coord_vy_flat

    # Extract boundary dual vector components
    no_slip_dual_x = no_slip_dual[1:topology.n_boundary_segments]
    no_slip_dual_y = no_slip_dual[topology.n_boundary_segments+1:end]

    # Extract spatial interval steps
    grid_spacing_x = fluid.fvm_grid.h_x
    grid_spacing_y = fluid.fvm_grid.h_y

    # Determine number of eulerian and lagrangian coordinates
    n_fluid_velocity_x = fluid.fvm_grid.n_vx
    n_fluid_velocity_y = fluid.fvm_grid.n_vy
    n_boundary_segments = topology.n_boundary_segments
    n_boundary_nodes = topology.n_boundary_nodes

    kind = topology.discrete_delta_kind
    support_x = grid_spacing_x * delta_support_radius(kind)
    support_y = grid_spacing_y * delta_support_radius(kind)

    # Calculate Jacobian entries for both x and y fluid velocities (single parallel pass)
    jacobian_I_vec, jacobian_J_vec, jacobian_V_vec = tcollect_sparse(1:n_boundary_segments, T) do chunk
        local_I, local_J, local_V = Int[], Int[], T[]
        for i in chunk
            start_node = topology.boundary_segment_start_nodes[i]
            end_node = topology.boundary_segment_end_nodes[i]

            # x-velocity contributions
            for j in 1:n_fluid_velocity_x
                min_dist_x = max(0.0, x_coord_fluid_velocity_x[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_x[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_x[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_x[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                ∂kernel_∂x_startpoint, ∂kernel_∂x_endpoint,
                ∂kernel_∂y_startpoint, ∂kernel_∂y_endpoint =
                    delta_product_segment_integral_derivatives(
                        grid_spacing_x, grid_spacing_y,
                        x_coord_fluid_velocity_x[j], y_coord_fluid_velocity_x[j],
                        startpoints_x[i], endpoints_x[i],
                        startpoints_y[i], endpoints_y[i],
                        kind,
                    )

                push!(local_I, j);                          push!(local_J, start_node);                    push!(local_V, ∂kernel_∂x_startpoint * no_slip_dual_x[i])
                push!(local_I, j);                          push!(local_J, end_node);                      push!(local_V, ∂kernel_∂x_endpoint * no_slip_dual_x[i])
                push!(local_I, j);                          push!(local_J, start_node + n_boundary_nodes); push!(local_V, ∂kernel_∂y_startpoint * no_slip_dual_x[i])
                push!(local_I, j);                          push!(local_J, end_node + n_boundary_nodes);   push!(local_V, ∂kernel_∂y_endpoint * no_slip_dual_x[i])
            end

            # y-velocity contributions
            for j in 1:n_fluid_velocity_y
                min_dist_x = max(0.0, x_coord_fluid_velocity_y[j] - max(startpoints_x[i], endpoints_x[i]),
                                      min(startpoints_x[i], endpoints_x[i]) - x_coord_fluid_velocity_y[j])
                min_dist_y = max(0.0, y_coord_fluid_velocity_y[j] - max(startpoints_y[i], endpoints_y[i]),
                                      min(startpoints_y[i], endpoints_y[i]) - y_coord_fluid_velocity_y[j])

                (min_dist_x > support_x || min_dist_y > support_y) && continue

                ∂kernel_∂x_startpoint, ∂kernel_∂x_endpoint,
                ∂kernel_∂y_startpoint, ∂kernel_∂y_endpoint =
                    delta_product_segment_integral_derivatives(
                        grid_spacing_x, grid_spacing_y,
                        x_coord_fluid_velocity_y[j], y_coord_fluid_velocity_y[j],
                        startpoints_x[i], endpoints_x[i],
                        startpoints_y[i], endpoints_y[i],
                        kind,
                    )

                push!(local_I, j + n_fluid_velocity_x);    push!(local_J, start_node);                    push!(local_V, ∂kernel_∂x_startpoint * no_slip_dual_y[i])
                push!(local_I, j + n_fluid_velocity_x);    push!(local_J, end_node);                      push!(local_V, ∂kernel_∂x_endpoint * no_slip_dual_y[i])
                push!(local_I, j + n_fluid_velocity_x);    push!(local_J, start_node + n_boundary_nodes); push!(local_V, ∂kernel_∂y_startpoint * no_slip_dual_y[i])
                push!(local_I, j + n_fluid_velocity_x);    push!(local_J, end_node + n_boundary_nodes);   push!(local_V, ∂kernel_∂y_endpoint * no_slip_dual_y[i])
            end
        end
        (local_I, local_J, local_V)
    end

    # Build sparse Jacobian matrix
    # Rows: fluid velocities (n_fluid_velocity_x + n_fluid_velocity_y)
    # Cols: boundary node coordinates (2 * n_boundary_nodes)
    jacobian = sparse(
        jacobian_I_vec,
        jacobian_J_vec,
        jacobian_V_vec,
        n_fluid_velocity_x + n_fluid_velocity_y,
        2 * n_boundary_nodes
    )

    return jacobian

end

function calculate_average_velocity_segment(
    topology::SystemTopology,
    boundary_velocity::AbstractVector{T}
) where T

    # Extract boundary node velocities
    boundary_velocity_x = boundary_velocity[1:topology.n_boundary_nodes]
    boundary_velocity_y = boundary_velocity[topology.n_boundary_nodes+1:end]

    # Extract segment start and end velocities
    velocity_x_startpoints = boundary_velocity_x[topology.boundary_segment_start_nodes]
    velocity_x_endpoints = boundary_velocity_x[topology.boundary_segment_end_nodes]
    velocity_y_startpoints = boundary_velocity_y[topology.boundary_segment_start_nodes]
    velocity_y_endpoints = boundary_velocity_y[topology.boundary_segment_end_nodes]

    # Average velocity over each segment (midpoint rule)
    velocity_x_integrated = 0.5 .* (velocity_x_startpoints + velocity_x_endpoints)
    velocity_y_integrated = 0.5 .* (velocity_y_startpoints + velocity_y_endpoints)

    return vcat(velocity_x_integrated, velocity_y_integrated)

end

function calculate_average_velocity_segment_jacobian(
    topology::SystemTopology,
    boundary_velocity::AbstractVector{T}
) where T

    n_boundary_segments = topology.n_boundary_segments
    n_boundary_nodes = topology.n_boundary_nodes
    n_boundary_velocities = (2 * topology.n_boundary_nodes)
    n_no_slip_constraints = topology.n_no_slip_constraints
    
    # Initialize sparse Jacobian matrix
    jacobian_I = Int[]
    jacobian_J = Int[]
    jacobian_V = T[]

    # Populate Jacobian entries
    for i in 1:n_boundary_segments

        start_idx = topology.boundary_segment_start_nodes[i]
        end_idx = topology.boundary_segment_end_nodes[i]

        # x-velocity contributions
        push!(jacobian_I, i)
        push!(jacobian_J, start_idx)
        push!(jacobian_V, 0.5)

        push!(jacobian_I, i)
        push!(jacobian_J, end_idx)
        push!(jacobian_V, 0.5)

        # y-velocity contributions
        push!(jacobian_I, i + n_boundary_segments)
        push!(jacobian_J, start_idx + n_boundary_nodes)
        push!(jacobian_V, 0.5)

        push!(jacobian_I, i + n_boundary_segments)
        push!(jacobian_J, end_idx + n_boundary_nodes)
        push!(jacobian_V, 0.5)
    end

    jacobian = sparse(
        jacobian_I,
        jacobian_J,
        jacobian_V,
        n_no_slip_constraints,
        n_boundary_velocities
    )

    return jacobian

    
end

function delta_product_segment_integral(
    grid_spacing_x, grid_spacing_y,
    fluid_position_x, fluid_position_y,
    boundary_x_startpoint, boundary_x_endpoint,
    boundary_y_startpoint, boundary_y_endpoint,
    kind::Symbol=:one_point,
)
    ## determine discrete-delta-product coefficients: d(ax+b)*d(cx+d)
    a = (boundary_x_startpoint - boundary_x_endpoint)/grid_spacing_x
    b = (fluid_position_x - boundary_x_startpoint)/grid_spacing_x

    c = (boundary_y_startpoint - boundary_y_endpoint)/grid_spacing_y
    d = (fluid_position_y - boundary_y_startpoint)/grid_spacing_y

    if kind === :one_point
        return one_point_delta_product_definite_integral(a, b, c, d)
    elseif kind === :three_point
        return three_point_delta_product_definite_integral(a, b, c, d)
    else
        error("Unknown discrete_delta_kind: $kind")
    end
end

function delta_product_segment_integral_derivatives(
    grid_spacing_x, grid_spacing_y,
    fluid_position_x, fluid_position_y,
    boundary_x_startpoint, boundary_x_endpoint,
    boundary_y_startpoint, boundary_y_endpoint,
    kind::Symbol=:one_point,
)
    a = (boundary_x_startpoint - boundary_x_endpoint)/grid_spacing_x
    b = (fluid_position_x - boundary_x_startpoint)/grid_spacing_x

    c = (boundary_y_startpoint - boundary_y_endpoint)/grid_spacing_y
    d = (fluid_position_y - boundary_y_startpoint)/grid_spacing_y

    if kind === :one_point
        ∂integral_∂a, ∂integral_∂b, ∂integral_∂c, ∂integral_∂d =
            one_point_delta_product_definite_integral_derivatives(a, b, c, d)
    elseif kind === :three_point
        ∂integral_∂a, ∂integral_∂b, ∂integral_∂c, ∂integral_∂d =
            three_point_delta_product_definite_integral_derivatives(a, b, c, d)
    else
        error("Unknown discrete_delta_kind: $kind")
    end

    # Apply chain rule to get derivatives w.r.t. original coordinates
    ∂integral_∂boundary_x_startpoint = ∂integral_∂a / grid_spacing_x -
        ∂integral_∂b / grid_spacing_x
    ∂integral_∂boundary_x_endpoint = -∂integral_∂a / grid_spacing_x
    ∂integral_∂boundary_y_startpoint = ∂integral_∂c / grid_spacing_y -
        ∂integral_∂d / grid_spacing_y
    ∂integral_∂boundary_y_endpoint = -∂integral_∂c / grid_spacing_y

    return ∂integral_∂boundary_x_startpoint,
        ∂integral_∂boundary_x_endpoint,
        ∂integral_∂boundary_y_startpoint,
        ∂integral_∂boundary_y_endpoint
end

@testitem "Weak-form FSI kernel matrix" begin
    using Aquarium
    using SparseArrays

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(5, 5), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                      ib_method=:weak_form)
    topology = system.topology

    config = zeros(system.n_configurations)
    config[1] = 0.5
    config[2] = 0.5
    boundary_state = calculate_boundary_state(system, vcat(config, zeros(system.n_velocities)))
    boundary_config = boundary_state[topology.boundary_configuration_indices]
    boundary_vel = boundary_state[topology.boundary_velocity_indices]

    K = calculate_weak_form_fsi_kernel(fluid, topology, boundary_config)

    n_segments = topology.n_boundary_segments
    n_v = fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy
    @test size(K) == (2 * n_segments, n_v)
    @test nnz(K) > 0

    # Partition of unity: row sums ≈ 1 for body well inside grid
    for i in 1:n_segments
        @test sum(K[i, :]) ≈ 1.0 atol=1e-12
        @test sum(K[n_segments + i, :]) ≈ 1.0 atol=1e-12
    end

    v = randn(n_v)
    Kv_matrix = K * v
    Kv_direct = calculate_weak_form_fsi_kernel_vector_product(fluid, topology, boundary_config, v)
    @test Kv_matrix ≈ Kv_direct atol=1e-10

    dual = randn(2 * n_segments)
    Ktd_matrix = K' * dual
    Ktd_direct = calculate_weak_form_fsi_vector_kernel_product(fluid, topology, boundary_config, dual)
    @test Ktd_matrix ≈ Ktd_direct atol=1e-10

    avg_vel = calculate_average_velocity_segment(topology, boundary_vel)
    @test length(avg_vel) == 2 * n_segments
end

@testitem "Weak-form FSI kernel Jacobians" begin
    using Aquarium
    using ForwardDiff

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(5, 5), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                      ib_method=:weak_form)
    topology = system.topology

    config = zeros(system.n_configurations)
    config[1] = 0.5
    config[2] = 0.5
    boundary_state = calculate_boundary_state(system, vcat(config, zeros(system.n_velocities)))
    boundary_config = boundary_state[topology.boundary_configuration_indices]
    boundary_vel = boundary_state[topology.boundary_velocity_indices]

    v = randn(fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy)
    dual = randn(2 * topology.n_boundary_segments)

    J_kv = calculate_weak_form_fsi_kernel_vector_product_jacobian(
        fluid, topology, boundary_config, v)
    J_kv_fd = ForwardDiff.jacobian(
        bc -> calculate_weak_form_fsi_kernel_vector_product(fluid, topology, bc, v),
        boundary_config)
    @test Matrix(J_kv) ≈ J_kv_fd atol=1e-6

    J_vk = calculate_weak_form_fsi_vector_kernel_product_jacobian(
        fluid, topology, boundary_config, dual)
    J_vk_fd = ForwardDiff.jacobian(
        bc -> calculate_weak_form_fsi_vector_kernel_product(fluid, topology, bc, dual),
        boundary_config)
    @test Matrix(J_vk) ≈ J_vk_fd atol=1e-6

    J_avg = calculate_average_velocity_segment_jacobian(topology, boundary_vel)
    J_avg_fd = ForwardDiff.jacobian(
        bv -> calculate_average_velocity_segment(topology, bv),
        boundary_vel)
    @test Matrix(J_avg) ≈ J_avg_fd atol=1e-12
end

@testitem "Weak-form FSI kernel matrix (three-point)" begin
    using Aquarium
    using SparseArrays

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(5, 5), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )

    system_1pt = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                          ib_method=:weak_form, discrete_delta_kind=:one_point)
    system_3pt = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                          ib_method=:weak_form, discrete_delta_kind=:three_point)

    config = zeros(system_3pt.n_configurations)
    config[1] = 0.5
    config[2] = 0.5
    boundary_state = calculate_boundary_state(system_3pt, vcat(config, zeros(system_3pt.n_velocities)))
    boundary_config = boundary_state[system_3pt.topology.boundary_configuration_indices]

    topology_3pt = system_3pt.topology
    topology_1pt = system_1pt.topology

    K_3pt = calculate_weak_form_fsi_kernel(fluid, topology_3pt, boundary_config)
    K_1pt = calculate_weak_form_fsi_kernel(fluid, topology_1pt, boundary_config)

    n_segments = topology_3pt.n_boundary_segments
    n_v = fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy
    @test size(K_3pt) == (2 * n_segments, n_v)
    @test nnz(K_3pt) > 0

    # Wider stencil: three-point should have at least as many nonzeros
    @test nnz(K_3pt) >= nnz(K_1pt)

    # Partition of unity: row sums ≈ 1 for body well inside grid
    for i in 1:n_segments
        @test sum(K_3pt[i, :]) ≈ 1.0 atol=1e-12
        @test sum(K_3pt[n_segments + i, :]) ≈ 1.0 atol=1e-12
    end

    # K*v consistency
    v = randn(n_v)
    Kv_matrix = K_3pt * v
    Kv_direct = calculate_weak_form_fsi_kernel_vector_product(fluid, topology_3pt, boundary_config, v)
    @test Kv_matrix ≈ Kv_direct atol=1e-10

    # K'*d consistency
    dual = randn(2 * n_segments)
    Ktd_matrix = K_3pt' * dual
    Ktd_direct = calculate_weak_form_fsi_vector_kernel_product(fluid, topology_3pt, boundary_config, dual)
    @test Ktd_matrix ≈ Ktd_direct atol=1e-10
end

@testitem "Weak-form FSI kernel Jacobians (three-point)" begin
    using Aquarium
    using ForwardDiff

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(5, 5), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5, n_boundary_nodes=8,
                      ib_method=:weak_form, discrete_delta_kind=:three_point)
    topology = system.topology

    config = zeros(system.n_configurations)
    config[1] = 0.5
    config[2] = 0.5
    boundary_state = calculate_boundary_state(system, vcat(config, zeros(system.n_velocities)))
    boundary_config = boundary_state[topology.boundary_configuration_indices]

    v = randn(fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy)
    dual = randn(2 * topology.n_boundary_segments)

    J_kv = calculate_weak_form_fsi_kernel_vector_product_jacobian(
        fluid, topology, boundary_config, v)
    J_kv_fd = ForwardDiff.jacobian(
        bc -> calculate_weak_form_fsi_kernel_vector_product(fluid, topology, bc, v),
        boundary_config)
    @test Matrix(J_kv) ≈ J_kv_fd atol=1e-6

    J_vk = calculate_weak_form_fsi_vector_kernel_product_jacobian(
        fluid, topology, boundary_config, dual)
    J_vk_fd = ForwardDiff.jacobian(
        bc -> calculate_weak_form_fsi_vector_kernel_product(fluid, topology, bc, dual),
        boundary_config)
    @test Matrix(J_vk) ≈ J_vk_fd atol=1e-6
end

@testitem "Weak-form long-segment support coverage" begin
    using Aquarium
    using SparseArrays

    # A vertical bar with very few nodes creates segments much longer than the
    # delta support radius.  The bounding-box support check must not skip grid
    # points that lie near the middle of a long segment.

    @testset "one_point delta, ds = 5h" begin
        # Fine grid (h = 0.05), bar with 3 nodes → 2 segments, each spanning 0.25 = 5h
        fluid = Fluid(0.01;
            density=1.0, dynamic_viscosity=0.01,
            boundary_velocity=[0.0, 0.0],
            grid_size=(20, 20), grid_dimensions=(1.0, 1.0),
            boundary_condition_type=:wall,
        )
        system = FreeBar(0.01;
            bar_length=0.5, mass=1.0, moi=1/12,
            n_boundary_nodes=3,
            ib_method=:weak_form, discrete_delta_kind=:one_point,
            gravity=[0.0, 0.0],
        )
        topology = system.topology

        # Place bar vertically at center of domain
        config = zeros(system.n_configurations)
        config[1] = 0.5   # x
        config[2] = 0.5   # y
        config[3] = π/2   # vertical
        body_state = vcat(config, zeros(system.n_velocities))
        boundary_state = calculate_boundary_state(system, body_state)
        boundary_config = boundary_state[topology.boundary_configuration_indices]

        K = calculate_weak_form_fsi_kernel(fluid, topology, boundary_config)

        # Row sums must be ≈ 1 (partition of unity) even for long segments
        n_seg = topology.n_boundary_segments
        for i in 1:n_seg
            @test sum(K[i, :]) ≈ 1.0 atol=1e-10
            @test sum(K[n_seg + i, :]) ≈ 1.0 atol=1e-10
        end

        # K*v and K'*d must match the matrix
        v = randn(fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy)
        Kv_direct = calculate_weak_form_fsi_kernel_vector_product(
            fluid, topology, boundary_config, v)
        @test K * v ≈ Kv_direct atol=1e-10

        dual = randn(2 * n_seg)
        Ktd_direct = calculate_weak_form_fsi_vector_kernel_product(
            fluid, topology, boundary_config, dual)
        @test K' * dual ≈ Ktd_direct atol=1e-10
    end

    @testset "three_point delta, ds = 5h" begin
        fluid = Fluid(0.01;
            density=1.0, dynamic_viscosity=0.01,
            boundary_velocity=[0.0, 0.0],
            grid_size=(20, 20), grid_dimensions=(1.0, 1.0),
            boundary_condition_type=:wall,
        )
        system = FreeBar(0.01;
            bar_length=0.5, mass=1.0, moi=1/12,
            n_boundary_nodes=3,
            ib_method=:weak_form, discrete_delta_kind=:three_point,
            gravity=[0.0, 0.0],
        )
        topology = system.topology

        config = zeros(system.n_configurations)
        config[1] = 0.5
        config[2] = 0.5
        config[3] = π/2
        body_state = vcat(config, zeros(system.n_velocities))
        boundary_state = calculate_boundary_state(system, body_state)
        boundary_config = boundary_state[topology.boundary_configuration_indices]

        K = calculate_weak_form_fsi_kernel(fluid, topology, boundary_config)

        n_seg = topology.n_boundary_segments
        for i in 1:n_seg
            @test sum(K[i, :]) ≈ 1.0 atol=1e-10
            @test sum(K[n_seg + i, :]) ≈ 1.0 atol=1e-10
        end

        v = randn(fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy)
        Kv_direct = calculate_weak_form_fsi_kernel_vector_product(
            fluid, topology, boundary_config, v)
        @test K * v ≈ Kv_direct atol=1e-10

        dual = randn(2 * n_seg)
        Ktd_direct = calculate_weak_form_fsi_vector_kernel_product(
            fluid, topology, boundary_config, dual)
        @test K' * dual ≈ Ktd_direct atol=1e-10
    end

    @testset "three_point delta, ds = 10h (extreme)" begin
        # 2 nodes on a long bar → 1 segment spanning 10h
        fluid = Fluid(0.01;
            density=1.0, dynamic_viscosity=0.01,
            boundary_velocity=[0.0, 0.0],
            grid_size=(20, 20), grid_dimensions=(1.0, 1.0),
            boundary_condition_type=:wall,
        )
        system = FreeBar(0.01;
            bar_length=0.5, mass=1.0, moi=1/12,
            n_boundary_nodes=2,
            ib_method=:weak_form, discrete_delta_kind=:three_point,
            gravity=[0.0, 0.0],
        )
        topology = system.topology

        config = zeros(system.n_configurations)
        config[1] = 0.5
        config[2] = 0.5
        config[3] = π/2
        body_state = vcat(config, zeros(system.n_velocities))
        boundary_state = calculate_boundary_state(system, body_state)
        boundary_config = boundary_state[topology.boundary_configuration_indices]

        K = calculate_weak_form_fsi_kernel(fluid, topology, boundary_config)

        n_seg = topology.n_boundary_segments
        @test n_seg == 1

        # Partition of unity must still hold
        @test sum(K[1, :]) ≈ 1.0 atol=1e-10
        @test sum(K[2, :]) ≈ 1.0 atol=1e-10

        # The kernel must have nonzero entries — with the old endpoint-distance
        # check, grid points in the middle of the segment would be skipped,
        # producing an almost-empty kernel row
        @test nnz(K) > 4

        v = randn(fluid.fvm_grid.n_vx + fluid.fvm_grid.n_vy)
        Kv_direct = calculate_weak_form_fsi_kernel_vector_product(
            fluid, topology, boundary_config, v)
        @test K * v ≈ Kv_direct atol=1e-10

        dual = randn(2 * n_seg)
        Ktd_direct = calculate_weak_form_fsi_vector_kernel_product(
            fluid, topology, boundary_config, dual)
        @test K' * dual ≈ Ktd_direct atol=1e-10
    end
end