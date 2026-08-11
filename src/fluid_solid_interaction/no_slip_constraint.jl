function calculate_no_slip_constraint_residual(fluid::Fluid,
    system::SolidSystem,
    fluid_velocity::AbstractVector,
    system_configuration::AbstractVector,
    system_velocity::AbstractVector;
    is_midpoint_state::Bool=false,
)

    # if empty system, return empty array
    if system.n_bodies == 0
        return zeros(0)
    end

    T = promote_type(eltype(fluid_velocity),
        eltype(system_configuration),
        eltype(system_velocity),
        _system_param_type(system),
    )

    # Construct body state from configuration and velocity
    body_state = zeros(T, system.n_body_states)
    body_state[system.configuration_indices] = system_configuration
    body_state[system.velocity_indices] = system_velocity

    # calculate midpoint
    if is_midpoint_state
        midpoint_body_state = body_state
    else
        midpoint_body_state = calculate_midpoint_state(system, body_state)
    end

    topology = system.topology

    midpoint_boundary_state = calculate_boundary_state(system, midpoint_body_state)
    midpoint_boundary_configuration = midpoint_boundary_state[system.topology.boundary_configuration_indices]
    midpoint_boundary_velocity = midpoint_boundary_state[system.topology.boundary_velocity_indices]

    if system.topology.ib_method == :original
        fsi_kernel_vector_product = calculate_original_fsi_kernel_vector_product(
            fluid, topology, midpoint_boundary_configuration, fluid_velocity
        )
        immersed_boundary_velocity = midpoint_boundary_velocity
    elseif system.topology.ib_method == :weak_form
        fsi_kernel_vector_product = calculate_weak_form_fsi_kernel_vector_product(
            fluid, topology, midpoint_boundary_configuration, fluid_velocity
        )
        immersed_boundary_velocity = calculate_average_velocity_segment(topology, midpoint_boundary_velocity)
    else
        error("Unknown immersed boundary method: $(system.topology.ib_method)")
    end

    no_slip_residual = fsi_kernel_vector_product - immersed_boundary_velocity

    return no_slip_residual

end

function calculate_no_slip_constraint_jacobian(fluid::Fluid,
    system::SolidSystem,
    fluid_velocity::AbstractVector,
    system_configuration::AbstractVector,
    system_velocity::AbstractVector;
    only_velocities::Bool=false,
    is_midpoint_state::Bool=false,
)

    n_params = length(collect_differentiable_params(system))

    if system.n_bodies == 0
        return spzeros(0, fluid.n_velocities),
            spzeros(0, system.n_body_states),
            spzeros(0, n_params)
    end

    topology = system.topology

    T = promote_type(eltype(fluid_velocity),
        eltype(system_configuration),
        eltype(system_velocity),
        _system_param_type(system),
    )

    # Construct body state from configuration and velocity
    body_state = zeros(T, system.n_body_states)
    body_state[system.configuration_indices] = system_configuration
    body_state[system.velocity_indices] = system_velocity


    # calculate midpoint
    if is_midpoint_state

        midpoint_body_state = body_state

        ∂midpoint_boundary_state_∂body_state, ∂midpoint_boundary_state_∂system_params =
            calculate_boundary_state_jacobian(system, body_state)

    else

        midpoint_body_state = calculate_midpoint_state(system, body_state)

        # Compute jacobian through midpoint state using chain rule
        ∂midpoint_body_state_∂body_state = ForwardDiff.jacobian(
            state -> calculate_midpoint_state(system, state),
            body_state
        )

        ∂boundary_state_∂midpoint_body_state, ∂boundary_state_∂system_params =
            calculate_boundary_state_jacobian(system, midpoint_body_state)

        ∂midpoint_boundary_state_∂body_state = ∂boundary_state_∂midpoint_body_state * ∂midpoint_body_state_∂body_state
        ∂midpoint_boundary_state_∂system_params = ∂boundary_state_∂system_params

    end

    midpoint_boundary_state = calculate_boundary_state(system, midpoint_body_state)
    midpoint_boundary_configuration = midpoint_boundary_state[topology.boundary_configuration_indices]
    midpoint_boundary_velocity = midpoint_boundary_state[topology.boundary_velocity_indices]

    ∂midpoint_boundary_configuration_∂body_state =
        ∂midpoint_boundary_state_∂body_state[topology.boundary_configuration_indices, :]
    ∂midpoint_boundary_velocity_∂body_state = 
        ∂midpoint_boundary_state_∂body_state[topology.boundary_velocity_indices, :]

    ∂midpoint_boundary_configuration_∂system_params =
        ∂midpoint_boundary_state_∂system_params[topology.boundary_configuration_indices, :]
    ∂midpoint_boundary_velocity_∂system_params =
        ∂midpoint_boundary_state_∂system_params[topology.boundary_velocity_indices, :]

    n_no_slip_constraints = topology.n_no_slip_constraints
    n_boundary_nodes = topology.n_boundary_nodes

    if only_velocities

        if topology.immersed_boundary_method == :original

            fsi_kernel = calculate_original_fsi_kernel(fluid, topology, midpoint_boundary_configuration)

            ∂immersed_boundary_velocity_∂body_state = zeros(n_no_slip_constraints, system.n_body_states)
            ∂immersed_boundary_velocity_∂body_state[:, system.velocity_indices] =
                ∂midpoint_boundary_velocity_∂body_state[:, system.velocity_indices]

            ∂immersed_boundary_velocity_∂system_params = ∂midpoint_boundary_velocity_∂system_params 

        elseif topology.immersed_boundary_method == :weak_form

            fsi_kernel = calculate_weak_form_fsi_kernel(fluid, topology, midpoint_boundary_configuration)

            ∂immersed_boundary_velocity_∂midpoint_boundary_velocity =
                calculate_average_velocity_segment_jacobian(topology, midpoint_boundary_velocity)

            ∂immersed_boundary_velocity_∂body_state = zeros(n_no_slip_constraints, system.n_body_states)
            ∂immersed_boundary_velocity_∂body_state[:, system.velocity_indices] =
                ∂immersed_boundary_velocity_∂midpoint_boundary_velocity *
                ∂midpoint_boundary_velocity_∂body_state[:, system.velocity_indices]

            ∂immersed_boundary_velocity_∂system_params = ∂immersed_boundary_velocity_∂midpoint_boundary_velocity *
                ∂midpoint_boundary_velocity_∂system_params

        end

        ∂no_slip_∂fluid_velocity = fsi_kernel
        ∂no_slip_∂body_state = -∂immersed_boundary_velocity_∂body_state
        ∂no_slip_∂system_params = -∂immersed_boundary_velocity_∂system_params

    else

        if topology.immersed_boundary_method == :original

            fsi_kernel = calculate_original_fsi_kernel(fluid, topology, midpoint_boundary_configuration)

            ∂fsi_kernel_velocity_product_∂midpoint_boundary_configuration =
                calculate_original_fsi_kernel_vector_product_jacobian(
                    fluid, topology, midpoint_boundary_configuration, fluid_velocity
                )

            ∂fsi_kernel_velocity_product_∂body_state =
                ∂fsi_kernel_velocity_product_∂midpoint_boundary_configuration *
                ∂midpoint_boundary_configuration_∂body_state
            ∂fsi_kernel_velocity_product_∂system_params =
                ∂fsi_kernel_velocity_product_∂midpoint_boundary_configuration *
                ∂midpoint_boundary_configuration_∂system_params

            ∂immersed_boundary_velocity_∂body_state = ∂midpoint_boundary_velocity_∂body_state
            ∂immersed_boundary_velocity_∂system_params = ∂midpoint_boundary_velocity_∂system_params 

        elseif topology.immersed_boundary_method == :weak_form

            fsi_kernel = calculate_weak_form_fsi_kernel(fluid, topology, midpoint_boundary_configuration)

            ∂fsi_kernel_velocity_product_∂midpoint_boundary_configuration =
                calculate_weak_form_fsi_kernel_vector_product_jacobian(
                    fluid, topology, midpoint_boundary_configuration, fluid_velocity
                )

                ∂immersed_boundary_velocity_∂midpoint_boundary_velocity =
                calculate_average_velocity_segment_jacobian(topology, midpoint_boundary_velocity)

            ∂immersed_boundary_velocity_∂body_state = ∂immersed_boundary_velocity_∂midpoint_boundary_velocity *
                ∂midpoint_boundary_velocity_∂body_state

            ∂immersed_boundary_velocity_∂system_params = ∂immersed_boundary_velocity_∂midpoint_boundary_velocity *
                ∂midpoint_boundary_velocity_∂system_params

            ∂fsi_kernel_velocity_product_∂body_state =
                ∂fsi_kernel_velocity_product_∂midpoint_boundary_configuration *
                ∂midpoint_boundary_configuration_∂body_state
            ∂fsi_kernel_velocity_product_∂system_params =
                ∂fsi_kernel_velocity_product_∂midpoint_boundary_configuration *
                ∂midpoint_boundary_configuration_∂system_params
            ∂immersed_boundary_velocity_∂body_state =
                ∂immersed_boundary_velocity_∂midpoint_boundary_velocity *
                ∂midpoint_boundary_velocity_∂body_state
        end

        ∂no_slip_∂body_state = ∂fsi_kernel_velocity_product_∂body_state -
            ∂immersed_boundary_velocity_∂body_state
        ∂no_slip_∂system_params = ∂fsi_kernel_velocity_product_∂system_params -
            ∂immersed_boundary_velocity_∂system_params
        ∂no_slip_∂fluid_velocity = fsi_kernel

    end

    return ∂no_slip_∂fluid_velocity, sparse(∂no_slip_∂body_state), sparse(∂no_slip_∂system_params)

end

function calculate_no_slip_constraint_vjp(fluid::Fluid,
    system::SolidSystem,
    fluid_velocity::AbstractVector,
    system_configuration::AbstractVector,
    system_velocity::AbstractVector,
    no_slip_dual::AbstractVector;
    is_midpoint_state::Bool=false,
)

    if system.n_bodies == 0
        return spzeros(fluid.n_velocities),
            spzeros(system.n_body_states)
    end

    topology = system.topology

    T = promote_type(eltype(fluid_velocity),
        eltype(system_configuration),
        eltype(system_velocity),
        _system_param_type(system),
    )

    # Construct body state from configuration and velocity
    body_state = zeros(T, system.n_body_states)
    body_state[system.configuration_indices] = system_configuration
    body_state[system.velocity_indices] = system_velocity


    # calculate midpoint
    if is_midpoint_state

        midpoint_body_state = body_state

        ∂midpoint_boundary_state_∂body_state, _ =
            calculate_boundary_state_jacobian(system, body_state)

    else

        midpoint_body_state = calculate_midpoint_state(system, body_state)

        # Compute jacobian through midpoint state using chain rule
        ∂midpoint_body_state_∂body_state = ForwardDiff.jacobian(
            state -> calculate_midpoint_state(system, state),
            body_state
        )

        ∂boundary_state_∂midpoint_body_state, _ =
            calculate_boundary_state_jacobian(system, midpoint_body_state)

        ∂midpoint_boundary_state_∂body_state = ∂boundary_state_∂midpoint_body_state * ∂midpoint_body_state_∂body_state

    end

    midpoint_boundary_state = calculate_boundary_state(system, midpoint_body_state)
    midpoint_boundary_configuration = midpoint_boundary_state[topology.boundary_configuration_indices]
    midpoint_boundary_velocity = midpoint_boundary_state[topology.boundary_velocity_indices]

    ∂midpoint_boundary_velocity_∂body_state =
        ∂midpoint_boundary_state_∂body_state[topology.boundary_velocity_indices, :]

    if topology.immersed_boundary_method == :original

        ∂no_slip_∂fluid_velocity_vjp = calculate_original_fsi_vector_kernel_product(
            fluid, topology, midpoint_boundary_configuration, no_slip_dual
        )

        ∂immersed_boundary_velocity_∂body_velocity =
            ∂midpoint_boundary_velocity_∂body_state[:, system.velocity_indices]

        ∂no_slip_∂body_velocity_vjp = -∂immersed_boundary_velocity_∂body_velocity' * no_slip_dual

    elseif topology.immersed_boundary_method == :weak_form

        ∂no_slip_∂fluid_velocity_vjp = calculate_weak_form_fsi_vector_kernel_product(
            fluid, topology, midpoint_boundary_configuration, no_slip_dual
        )

        ∂immersed_boundary_velocity_∂midpoint_boundary_velocity =
            calculate_average_velocity_segment_jacobian(topology, midpoint_boundary_velocity)

        ∂immersed_boundary_velocity_∂body_velocity =
            ∂immersed_boundary_velocity_∂midpoint_boundary_velocity *
            ∂midpoint_boundary_velocity_∂body_state[:, system.velocity_indices]

        ∂no_slip_∂body_velocity_vjp = -∂immersed_boundary_velocity_∂body_velocity' * no_slip_dual

    end

    # Pad with zeros for configuration indices (VJP only depends on velocities in velocity-only mode)
    T = promote_type(eltype(∂no_slip_∂body_velocity_vjp), eltype(no_slip_dual))
    ∂no_slip_∂body_state_vjp = zeros(T, system.n_body_states)
    ∂no_slip_∂body_state_vjp[system.velocity_indices] = ∂no_slip_∂body_velocity_vjp

    return ∂no_slip_∂fluid_velocity_vjp, sparse(∂no_slip_∂body_state_vjp)

end

function calculate_no_slip_constraint_vjp_jacobian(fluid::Fluid,
    system::SolidSystem,
    fluid_velocity::AbstractVector,
    system_configuration::AbstractVector,
    system_velocity::AbstractVector,
    no_slip_dual::AbstractVector;
    is_midpoint_state::Bool=false,
)

    n_params = length(collect_differentiable_params(system))

    if system.n_bodies == 0
        return spzeros(fluid.n_velocities, length(no_slip_dual)),
            spzeros(system.n_velocities, length(no_slip_dual)),
            spzeros(fluid.n_velocities, n_params),
            spzeros(system.n_velocities, n_params),
            spzeros(fluid.n_velocities, system.n_body_states),
            spzeros(system.n_body_states, system.n_body_states)
    end

    topology = system.topology

    T = promote_type(eltype(fluid_velocity),
        eltype(system_configuration),
        eltype(system_velocity),
        _system_param_type(system),
    )

    # Construct body state from configuration and velocity
    body_state = zeros(T, system.n_body_states)
    body_state[system.configuration_indices] = system_configuration
    body_state[system.velocity_indices] = system_velocity

    # Calculate midpoint state
    if is_midpoint_state
        midpoint_body_state = body_state
    else
        midpoint_body_state = calculate_midpoint_state(system, body_state)
    end

    # Calculate boundary state and its jacobians
    boundary_state = calculate_boundary_state(system, midpoint_body_state)
    boundary_configuration = boundary_state[topology.boundary_configuration_indices]
    boundary_velocity = boundary_state[topology.boundary_velocity_indices]

    # Compute ∂boundary_state/∂body_state using analytical jacobian
    if is_midpoint_state
        ∂boundary_state_∂body_state, ∂boundary_state_∂system_params =
            calculate_boundary_state_jacobian(system, body_state)

    else
        # Analytical midpoint Jacobian: q_mid = q - (dt/2)*v, v_mid = v
        n_bs = system.n_body_states
        qi = system.configuration_indices
        vi = system.velocity_indices
        ∂midpoint_body_state_∂body_state = zeros(T, n_bs, n_bs)
        for i in 1:n_bs; ∂midpoint_body_state_∂body_state[i, i] = one(T); end
        for (ci, vi_idx) in zip(qi, vi)
            ∂midpoint_body_state_∂body_state[ci, vi_idx] = -0.5 * system.time_step
        end

        ∂boundary_state_∂midpoint_body_state, ∂boundary_state_∂system_params_direct =
            calculate_boundary_state_jacobian(system, midpoint_body_state)

        ∂boundary_state_∂body_state = ∂boundary_state_∂midpoint_body_state * ∂midpoint_body_state_∂body_state
        ∂boundary_state_∂system_params = ∂boundary_state_∂system_params_direct

    end

    ∂boundary_configuration_∂body_state =
        ∂boundary_state_∂body_state[topology.boundary_configuration_indices, :]
    ∂boundary_velocity_∂body_velocity = ∂boundary_state_∂body_state[
        topology.boundary_velocity_indices, system.velocity_indices
    ]

    ∂boundary_configuration_∂system_params =
        ∂boundary_state_∂system_params[topology.boundary_configuration_indices, :]
    ∂boundary_velocity_∂system_params =
        ∂boundary_state_∂system_params[topology.boundary_velocity_indices, :]

    # Now compute VJP jacobians analytically using chain rule
    if topology.immersed_boundary_method == :original

        # ∂fluid_velocity_vjp_∂no_slip_dual = fsi_kernel^T
        ∂fluid_velocity_vjp_∂no_slip_dual = calculate_original_fsi_kernel(
            fluid, topology, boundary_configuration
        )'

        # ∂body_velocity_vjp_∂no_slip_dual = -∂boundary_velocity_∂body_velocity^T
        ∂body_velocity_vjp_∂no_slip_dual = -∂boundary_velocity_∂body_velocity'

        # ∂fluid_velocity_vjp_∂system_params via chain rule:
        # fluid_vjp = fsi_kernel^T * no_slip_dual
        # ∂(fsi_kernel^T * dual)/∂params = ∂(fsi_kernel^T * dual)/∂boundary_config * ∂boundary_config/∂params
        ∂vector_kernel_product_∂boundary_configuration = calculate_original_fsi_vector_kernel_product_jacobian(
            fluid, topology, boundary_configuration, no_slip_dual
        )
        ∂fluid_velocity_vjp_∂system_params = ∂vector_kernel_product_∂boundary_configuration *
            ∂boundary_configuration_∂system_params

        # Body velocity VJP depends on system_params through ∂boundary_velocity_∂body_velocity
        # Compute ∂(-∂boundary_velocity_∂body_velocity' * dual)/∂params analytically

        ∂Jv_vjp_∂params = calculate_boundary_velocity_vjp_jacobian(
            system, midpoint_body_state, no_slip_dual;
            is_midpoint_state=is_midpoint_state
        )

        ∂body_velocity_vjp_∂system_params = -∂Jv_vjp_∂params

        # ∂fluid_velocity_vjp_∂system_body_state via chain rule:
        # fluid_vjp = fsi_kernel^T * no_slip_dual depends on body state through
        # boundary_configuration (the boundary node positions move with the body).
        ∂fluid_velocity_vjp_∂system_body_state = ∂vector_kernel_product_∂boundary_configuration *
            ∂boundary_configuration_∂body_state

        # Analytical ∂body_velocity_vjp/∂body_state
        ∂body_velocity_vjp_∂body_state = -_boundary_velocity_body_vjp_jacobian(
            system, body_state, no_slip_dual;
            is_midpoint_state=is_midpoint_state)

    elseif topology.immersed_boundary_method == :weak_form

        # ∂fluid_velocity_vjp_∂no_slip_dual = fsi_kernel^T
        ∂fluid_velocity_vjp_∂no_slip_dual = calculate_weak_form_fsi_kernel(
            fluid, topology, boundary_configuration
        )'

        # For weak form, need to include averaging jacobian
        ∂immersed_boundary_velocity_∂boundary_velocity = calculate_average_velocity_segment_jacobian(
            topology, boundary_velocity
        )
        ∂immersed_boundary_velocity_∂body_velocity = ∂immersed_boundary_velocity_∂boundary_velocity *
            ∂boundary_velocity_∂body_velocity

        ∂body_velocity_vjp_∂no_slip_dual = -∂immersed_boundary_velocity_∂body_velocity'

        # ∂fluid_velocity_vjp_∂system_params via chain rule
        # fluid_vjp = fsi_kernel^T * no_slip_dual
        # ∂(fsi_kernel^T * dual)/∂params = ∂(fsi_kernel^T * dual)/∂boundary_config * ∂boundary_config/∂params
        ∂vector_kernel_product_∂boundary_configuration = calculate_weak_form_fsi_vector_kernel_product_jacobian(
            fluid, topology, boundary_configuration, no_slip_dual
        )
        ∂fluid_velocity_vjp_∂system_params = ∂vector_kernel_product_∂boundary_configuration *
            ∂boundary_configuration_∂system_params
        ∂fluid_velocity_vjp_∂system_body_state = ∂vector_kernel_product_∂boundary_configuration *
            ∂boundary_configuration_∂body_state

        # Body velocity VJP depends on system_params through boundary state jacobians
        # Compute analytically: VJP = -(A * J_v)^T * dual = -J_v^T * A^T * dual
        # where A = averaging operator (doesn't depend on params)

        # Determine which state to use for Jacobian evaluation
        midpoint_boundary_state = calculate_boundary_state(system, midpoint_body_state)
        bv = midpoint_boundary_state[topology.boundary_velocity_indices]

        # Apply averaging transpose to dual to get effective dual
        ∂avg_∂bv = calculate_average_velocity_segment_jacobian(topology, bv)
        effective_dual = ∂avg_∂bv' * no_slip_dual

        # Compute analytical mixed derivative with effective dual
        ∂Jv_vjp_∂params = calculate_boundary_velocity_vjp_jacobian(
            system, midpoint_body_state, effective_dual;
            is_midpoint_state=is_midpoint_state
        )

        ∂body_velocity_vjp_∂system_params = -∂Jv_vjp_∂params

        # Analytical ∂body_velocity_vjp/∂body_state: A has constant coefficients,
        # so effective_dual is constant w.r.t. body_state. Only the sin/cos terms
        # in the VJP depend on body_state through the midpoint angle.
        if is_midpoint_state
            ∂body_velocity_vjp_∂body_state = -_boundary_velocity_body_vjp_jacobian(
                system, body_state, effective_dual; is_midpoint_state=true)
        else
            J_at_mid = _boundary_velocity_body_vjp_jacobian(
                system, midpoint_body_state, effective_dual; is_midpoint_state=true)
            ∂body_velocity_vjp_∂body_state = -J_at_mid * ∂midpoint_body_state_∂body_state
        end

    end

    # Pad body_state jacobians with zeros for configuration indices
    # (VJP only depends on velocities in velocity-only mode)
    ∂body_state_vjp_∂no_slip_dual = spzeros(system.n_body_states, size(∂body_velocity_vjp_∂no_slip_dual, 2))
    ∂body_state_vjp_∂no_slip_dual[system.velocity_indices, :] = ∂body_velocity_vjp_∂no_slip_dual

    ∂body_state_vjp_∂system_params = spzeros(system.n_body_states, n_params)
    ∂body_state_vjp_∂system_params[system.velocity_indices, :] = ∂body_velocity_vjp_∂system_params

    ∂body_state_vjp_∂system_body_state = spzeros(system.n_body_states, system.n_body_states)
    ∂body_state_vjp_∂system_body_state[system.velocity_indices, :] = ∂body_velocity_vjp_∂body_state

    return ∂fluid_velocity_vjp_∂no_slip_dual,
        sparse(∂body_state_vjp_∂no_slip_dual),
        sparse(∂fluid_velocity_vjp_∂system_params),
        sparse(∂body_state_vjp_∂system_params),
        sparse(∂fluid_velocity_vjp_∂system_body_state),
        sparse(∂body_state_vjp_∂system_body_state)

end

@testitem "No-slip single body original" begin
    using AquariumClosed
    using ForwardDiff
    using FiniteDiff
    using Random

    Random.seed!(0)

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(10, 10), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = FreeDisc(0.01; radius=0.12, mass=1.0, moi=0.5, n_boundary_nodes=8,
                      ib_method=:original)

    # Non-trivial state: off-center disc with nonzero velocity,
    # nonzero fluid velocity field, random no-slip dual.
    config = [0.47, 0.53, 0.31]
    vel = [0.02, -0.015, 0.03]
    fluid_vel = 0.01 .* randn(fluid.n_velocities)

    for is_midpoint in (true, false)
        residual = calculate_no_slip_constraint_residual(
            fluid, system, fluid_vel, config, vel; is_midpoint_state=is_midpoint)
        @test length(residual) == 2 * system.topology.n_boundary_nodes
        @test all(isfinite, residual)

        J_fluid, J_body, J_params = calculate_no_slip_constraint_jacobian(
            fluid, system, fluid_vel, config, vel; is_midpoint_state=is_midpoint)
        @test size(J_fluid) == (length(residual), fluid.n_velocities)
        @test size(J_body) == (length(residual), system.n_body_states)
        @test size(J_params) == (length(residual), length(collect_differentiable_params(system)))

        # ∂residual/∂fluid_velocity vs ForwardDiff
        J_fluid_fd = ForwardDiff.jacobian(
            fv -> calculate_no_slip_constraint_residual(
                fluid, system, fv, config, vel; is_midpoint_state=is_midpoint),
            fluid_vel)
        @test Matrix(J_fluid) ≈ J_fluid_fd atol=1e-8

        # ∂residual/∂body_state (config + velocity) vs ForwardDiff
        J_body_fd = ForwardDiff.jacobian(
            bs -> begin
                c = bs[system.configuration_indices]
                v = bs[system.velocity_indices]
                calculate_no_slip_constraint_residual(
                    fluid, system, fluid_vel, c, v; is_midpoint_state=is_midpoint)
            end,
            vcat(config, vel))
        @test Matrix(J_body) ≈ J_body_fd atol=1e-8

        # ∂residual/∂system_params via inject_differentiable_params (FiniteDiff
        # to avoid any non-type-stable constructor paths).
        J_params_fd = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(system)) do p
            new_system = inject_differentiable_params(system, p)
            calculate_no_slip_constraint_residual(
                fluid, new_system, fluid_vel, config, vel;
                is_midpoint_state=is_midpoint)
        end
        @test Matrix(J_params) ≈ J_params_fd rtol=1e-5

        # VJP function + all 5 outputs of the VJP jacobian
        dual = 0.1 .* randn(length(residual))
        fluid_vjp, body_vjp = calculate_no_slip_constraint_vjp(
            fluid, system, fluid_vel, config, vel, dual;
            is_midpoint_state=is_midpoint)
        @test length(fluid_vjp) == fluid.n_velocities
        @test length(body_vjp) == system.n_body_states

        J_fv_dual,
        J_bs_dual,
        J_fv_params,
        J_bs_params,
        J_fv_body_state = calculate_no_slip_constraint_vjp_jacobian(
            fluid, system, fluid_vel, config, vel, dual;
            is_midpoint_state=is_midpoint)

        # VJP jacobians wrt the no-slip dual
        J_fv_dual_fd = ForwardDiff.jacobian(
            d -> calculate_no_slip_constraint_vjp(
                fluid, system, fluid_vel, config, vel, d;
                is_midpoint_state=is_midpoint)[1],
            dual)
        @test Matrix(J_fv_dual) ≈ J_fv_dual_fd atol=1e-8

        J_bs_dual_fd = ForwardDiff.jacobian(
            d -> calculate_no_slip_constraint_vjp(
                fluid, system, fluid_vel, config, vel, d;
                is_midpoint_state=is_midpoint)[2],
            dual)
        @test Matrix(J_bs_dual) ≈ J_bs_dual_fd atol=1e-8

        # VJP jacobians wrt system_params (via inject_differentiable_params)
        J_fv_params_fd = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(system)) do p
            new_system = inject_differentiable_params(system, p)
            calculate_no_slip_constraint_vjp(
                fluid, new_system, fluid_vel, config, vel, dual;
                is_midpoint_state=is_midpoint)[1]
        end
        @test Matrix(J_fv_params) ≈ J_fv_params_fd rtol=1e-5

        J_bs_params_fd = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(system)) do p
            new_system = inject_differentiable_params(system, p)
            calculate_no_slip_constraint_vjp(
                fluid, new_system, fluid_vel, config, vel, dual;
                is_midpoint_state=is_midpoint)[2]
        end
        @test Matrix(J_bs_params) ≈ J_bs_params_fd rtol=1e-5

        # VJP jacobian wrt body state (fluid-velocity output only)
        J_fv_body_state_fd = ForwardDiff.jacobian(
            bs -> begin
                c = bs[system.configuration_indices]
                v = bs[system.velocity_indices]
                calculate_no_slip_constraint_vjp(
                    fluid, system, fluid_vel, c, v, dual;
                    is_midpoint_state=is_midpoint)[1]
            end,
            vcat(config, vel))
        @test Matrix(J_fv_body_state) ≈ J_fv_body_state_fd atol=1e-8
    end
end

@testitem "No-slip single body weak form" begin
    using AquariumClosed
    using ForwardDiff
    using FiniteDiff
    using Random

    Random.seed!(1)

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(10, 10), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = FreeDisc(0.01; radius=0.12, mass=1.0, moi=0.5, n_boundary_nodes=8,
                      ib_method=:weak_form)

    config = [0.47, 0.53, 0.31]
    vel = [0.02, -0.015, 0.03]
    fluid_vel = 0.01 .* randn(fluid.n_velocities)

    for is_midpoint in (true, false)
        residual = calculate_no_slip_constraint_residual(
            fluid, system, fluid_vel, config, vel; is_midpoint_state=is_midpoint)
        @test length(residual) == 2 * system.topology.n_boundary_segments
        @test all(isfinite, residual)

        J_fluid, J_body, J_params = calculate_no_slip_constraint_jacobian(
            fluid, system, fluid_vel, config, vel; is_midpoint_state=is_midpoint)
        @test size(J_fluid) == (length(residual), fluid.n_velocities)
        @test size(J_body) == (length(residual), system.n_body_states)
        @test size(J_params) == (length(residual), length(collect_differentiable_params(system)))

        J_fluid_fd = ForwardDiff.jacobian(
            fv -> calculate_no_slip_constraint_residual(
                fluid, system, fv, config, vel; is_midpoint_state=is_midpoint),
            fluid_vel)
        @test Matrix(J_fluid) ≈ J_fluid_fd atol=1e-8

        J_body_fd = ForwardDiff.jacobian(
            bs -> begin
                c = bs[system.configuration_indices]
                v = bs[system.velocity_indices]
                calculate_no_slip_constraint_residual(
                    fluid, system, fluid_vel, c, v; is_midpoint_state=is_midpoint)
            end,
            vcat(config, vel))
        @test Matrix(J_body) ≈ J_body_fd atol=1e-8

        J_params_fd = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(system)) do p
            new_system = inject_differentiable_params(system, p)
            calculate_no_slip_constraint_residual(
                fluid, new_system, fluid_vel, config, vel;
                is_midpoint_state=is_midpoint)
        end
        @test Matrix(J_params) ≈ J_params_fd rtol=1e-5

        dual = 0.1 .* randn(length(residual))
        fluid_vjp, body_vjp = calculate_no_slip_constraint_vjp(
            fluid, system, fluid_vel, config, vel, dual;
            is_midpoint_state=is_midpoint)
        @test length(fluid_vjp) == fluid.n_velocities
        @test length(body_vjp) == system.n_body_states

        J_fv_dual,
        J_bs_dual,
        J_fv_params,
        J_bs_params,
        J_fv_body_state = calculate_no_slip_constraint_vjp_jacobian(
            fluid, system, fluid_vel, config, vel, dual;
            is_midpoint_state=is_midpoint)

        J_fv_dual_fd = ForwardDiff.jacobian(
            d -> calculate_no_slip_constraint_vjp(
                fluid, system, fluid_vel, config, vel, d;
                is_midpoint_state=is_midpoint)[1],
            dual)
        @test Matrix(J_fv_dual) ≈ J_fv_dual_fd atol=1e-8

        J_bs_dual_fd = ForwardDiff.jacobian(
            d -> calculate_no_slip_constraint_vjp(
                fluid, system, fluid_vel, config, vel, d;
                is_midpoint_state=is_midpoint)[2],
            dual)
        @test Matrix(J_bs_dual) ≈ J_bs_dual_fd atol=1e-8

        J_fv_params_fd = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(system)) do p
            new_system = inject_differentiable_params(system, p)
            calculate_no_slip_constraint_vjp(
                fluid, new_system, fluid_vel, config, vel, dual;
                is_midpoint_state=is_midpoint)[1]
        end
        @test Matrix(J_fv_params) ≈ J_fv_params_fd rtol=1e-5

        J_bs_params_fd = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(system)) do p
            new_system = inject_differentiable_params(system, p)
            calculate_no_slip_constraint_vjp(
                fluid, new_system, fluid_vel, config, vel, dual;
                is_midpoint_state=is_midpoint)[2]
        end
        @test Matrix(J_bs_params) ≈ J_bs_params_fd rtol=1e-5

        J_fv_body_state_fd = ForwardDiff.jacobian(
            bs -> begin
                c = bs[system.configuration_indices]
                v = bs[system.velocity_indices]
                calculate_no_slip_constraint_vjp(
                    fluid, system, fluid_vel, c, v, dual;
                    is_midpoint_state=is_midpoint)[1]
            end,
            vcat(config, vel))
        @test Matrix(J_fv_body_state) ≈ J_fv_body_state_fd atol=1e-8
    end
end

@testitem "No-slip multi body" begin
    using AquariumClosed
    using ForwardDiff
    using FiniteDiff
    using Random

    Random.seed!(2)

    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(10, 10), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = DoublePendulum(0.01;
        bar_lengths=[0.3, 0.2], masses=[1.0, 0.5], mois=[0.05, 0.02],
        hinge_position=[0.5, 0.7], n_boundary_nodes_per_link=4,
        ib_method=:original)

    # Non-trivial state: two-link system in a bent configuration with nonzero
    # velocities and nonzero fluid velocity field.
    config = zeros(system.n_configurations)
    config[1] = 0.47; config[2] = 0.48; config[3] = 0.12
    config[4] = 0.53; config[5] = 0.21; config[6] = -0.15
    vel = 0.02 .* randn(system.n_velocities)
    fluid_vel = 0.01 .* randn(fluid.n_velocities)

    for is_midpoint in (true, false)
        residual = calculate_no_slip_constraint_residual(
            fluid, system, fluid_vel, config, vel; is_midpoint_state=is_midpoint)
        @test length(residual) == 2 * system.topology.n_boundary_nodes
        @test all(isfinite, residual)

        J_fluid, J_body, J_params = calculate_no_slip_constraint_jacobian(
            fluid, system, fluid_vel, config, vel; is_midpoint_state=is_midpoint)

        J_fluid_fd = ForwardDiff.jacobian(
            fv -> calculate_no_slip_constraint_residual(
                fluid, system, fv, config, vel; is_midpoint_state=is_midpoint),
            fluid_vel)
        @test Matrix(J_fluid) ≈ J_fluid_fd atol=1e-8

        J_body_fd = ForwardDiff.jacobian(
            bs -> begin
                c = bs[system.configuration_indices]
                v = bs[system.velocity_indices]
                calculate_no_slip_constraint_residual(
                    fluid, system, fluid_vel, c, v; is_midpoint_state=is_midpoint)
            end,
            vcat(config, vel))
        @test Matrix(J_body) ≈ J_body_fd atol=1e-8

        J_params_fd = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(system)) do p
            new_system = inject_differentiable_params(system, p)
            calculate_no_slip_constraint_residual(
                fluid, new_system, fluid_vel, config, vel;
                is_midpoint_state=is_midpoint)
        end
        @test Matrix(J_params) ≈ J_params_fd rtol=1e-5

        dual = 0.1 .* randn(length(residual))

        J_fv_dual,
        J_bs_dual,
        J_fv_params,
        J_bs_params,
        J_fv_body_state = calculate_no_slip_constraint_vjp_jacobian(
            fluid, system, fluid_vel, config, vel, dual;
            is_midpoint_state=is_midpoint)

        J_fv_dual_fd = ForwardDiff.jacobian(
            d -> calculate_no_slip_constraint_vjp(
                fluid, system, fluid_vel, config, vel, d;
                is_midpoint_state=is_midpoint)[1],
            dual)
        @test Matrix(J_fv_dual) ≈ J_fv_dual_fd atol=1e-8

        J_bs_dual_fd = ForwardDiff.jacobian(
            d -> calculate_no_slip_constraint_vjp(
                fluid, system, fluid_vel, config, vel, d;
                is_midpoint_state=is_midpoint)[2],
            dual)
        @test Matrix(J_bs_dual) ≈ J_bs_dual_fd atol=1e-8

        J_fv_params_fd = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(system)) do p
            new_system = inject_differentiable_params(system, p)
            calculate_no_slip_constraint_vjp(
                fluid, new_system, fluid_vel, config, vel, dual;
                is_midpoint_state=is_midpoint)[1]
        end
        @test Matrix(J_fv_params) ≈ J_fv_params_fd rtol=1e-5

        J_bs_params_fd = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(system)) do p
            new_system = inject_differentiable_params(system, p)
            calculate_no_slip_constraint_vjp(
                fluid, new_system, fluid_vel, config, vel, dual;
                is_midpoint_state=is_midpoint)[2]
        end
        @test Matrix(J_bs_params) ≈ J_bs_params_fd rtol=1e-5

        J_fv_body_state_fd = ForwardDiff.jacobian(
            bs -> begin
                c = bs[system.configuration_indices]
                v = bs[system.velocity_indices]
                calculate_no_slip_constraint_vjp(
                    fluid, system, fluid_vel, c, v, dual;
                    is_midpoint_state=is_midpoint)[1]
            end,
            vcat(config, vel))
        @test Matrix(J_fv_body_state) ≈ J_fv_body_state_fd atol=1e-8
    end
end

@testitem "No-slip no body" begin
    using AquariumClosed
    fluid = Fluid(0.01;
        density=1.0, dynamic_viscosity=0.01,
        boundary_velocity=[0.0, 0.0],
        grid_size=(5, 5), grid_dimensions=(1.0, 1.0),
        boundary_condition_type=:wall,
    )
    system = NoSystem()

    residual = calculate_no_slip_constraint_residual(
        fluid, system, zeros(fluid.n_velocities), Float64[], Float64[];
        is_midpoint_state=true)
    @test residual == zeros(0)
end