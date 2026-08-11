#############################################################################################
## Velocity-Based Cost Functions for C-Start Optimization
#############################################################################################
# These functions maximize velocity in a desired direction using linear dot product:
#   Cost: J = -w * (v · d)  where v is COM velocity, d is desired direction
#
# Advantages:
#   - Direct optimization: maximize velocity component along desired direction
#   - Stable gradients independent of current velocity magnitude
#   - Natural physical interpretation: cost value = velocity in cm/s
#   - Well-conditioned optimization landscape for L-BFGS-B
#
# Usage:
#   Pass these functions to simulate_aquarium() with:
#     calculate_stage_objective=calculate_stage_objective
#     calculate_terminal_objective=calculate_terminal_objective
#     calculate_stage_objective_gradients=calculate_stage_objective_gradients
#     calculate_terminal_objective_gradients=calculate_terminal_objective_gradients
#############################################################################################

#############################################################################################
## Helper Function: Compute COM Velocity
#############################################################################################

function calculate_com_velocity(tank, aquarium_state)
    swimmer_state = aquarium_state[tank.swimmer_body_state_indices]
    total_mass = sum(b.mass for b in tank.swimmer.bodies)
    com_vx = zero(eltype(swimmer_state))
    com_vy = zero(eltype(swimmer_state))

    n_links = tank.swimmer.n_bodies
    for i in 1:n_links
        link_vel_indices = tank.swimmer.velocity_indices[(3*(i-1)+1):(3*i)]
        link_velocity = swimmer_state[link_vel_indices]
        link_mass = tank.swimmer.bodies[i].mass

        com_vx += link_mass * link_velocity[1]
        com_vy += link_mass * link_velocity[2]
    end

    return [com_vx / total_mass, com_vy / total_mass]
end

#############################################################################################
## Stage Cost: Accumulate velocity over time
#############################################################################################

function calculate_stage_objective(
    tank, time, aquarium_state, bluff_body_state, swimmer_control,
)
    # Parameters (these would be set in your main script)
    # You can pass these through global variables or closure
    # desired_direction_angle = 0.0  # Set in main script
    # stage_cost_weight = 1.0
    # averaging_window_time = 0.0
    # final_time = 3.0

    # Only accumulate objective in the specified time window
    if time < (final_time - averaging_window_time)
        return zero(eltype(aquarium_state))
    end

    # Compute COM velocity
    com_vel = calculate_com_velocity(tank, aquarium_state)
    vx, vy = com_vel[1], com_vel[2]

    # Desired direction unit vector
    d_x = cos(desired_direction_angle)
    d_y = sin(desired_direction_angle)

    # Velocity component in desired direction
    v_dot_d = vx * d_x + vy * d_y

    # Objective: NEGATIVE of velocity (minimize cost = maximize velocity)
    # Stage cost weight typically includes time_step for proper integration
    return -stage_cost_weight * v_dot_d
end

function calculate_stage_objective_gradients(
    tank, time, aquarium_state, bluff_body_state, swimmer_control,
)
    # Initialize all gradients to zero
    grad_aquarium_state = zeros(eltype(aquarium_state), length(aquarium_state))
    grad_bluff_body_state = zeros(eltype(bluff_body_state), length(bluff_body_state))
    grad_swimmer_control = zeros(eltype(swimmer_control), length(swimmer_control))
    grad_fluid_params = zeros(4)
    grad_swimmer_params = zeros(length(collect_differentiable_params(tank.swimmer)))
    grad_bluff_body_params = zeros(length(collect_differentiable_params(tank.bluff_body)))

    # Only compute gradients within the averaging window
    if time < (final_time - averaging_window_time)
        return grad_aquarium_state, grad_bluff_body_state, grad_swimmer_control,
                grad_fluid_params, grad_swimmer_params, grad_bluff_body_params
    end

    # ==================================================================================
    # ANALYTICAL GRADIENT COMPUTATION
    # ==================================================================================
    # Cost: J = -w * (vx * d_x + vy * d_y)
    #
    # ∂J/∂vx = -w * d_x
    # ∂J/∂vy = -w * d_y
    #
    # COM velocity: vx = (1/M) Σ m_i * vx_i,  vy = (1/M) Σ m_i * vy_i
    # ∂vx/∂(vx_i) = m_i / M,  ∂vy/∂(vy_i) = m_i / M
    #
    # Chain rule: ∂J/∂(vx_i) = (∂J/∂vx) * (∂vx/∂(vx_i)) = -w * d_x * (m_i / M)
    #             ∂J/∂(vy_i) = (∂J/∂vy) * (∂vy/∂(vy_i)) = -w * d_y * (m_i / M)
    # ==================================================================================

    # Desired direction unit vector
    d_x = cos(desired_direction_angle)
    d_y = sin(desired_direction_angle)

    # Derivative of cost w.r.t. COM velocity components
    dJ_dvx = -stage_cost_weight * d_x
    dJ_dvy = -stage_cost_weight * d_y

    # Compute gradients w.r.t. individual link velocities
    total_mass = sum(b.mass for b in tank.swimmer.bodies)

    n_links = tank.swimmer.n_bodies
    for i in 1:n_links
        link_mass = tank.swimmer.bodies[i].mass
        com_weight = link_mass / total_mass

        link_vel_indices = tank.swimmer.velocity_indices[(3*(i-1)+1):(3*i)]
        vx_index_global = tank.swimmer_body_state_indices[link_vel_indices[1]]
        vy_index_global = tank.swimmer_body_state_indices[link_vel_indices[2]]

        grad_aquarium_state[vx_index_global] = dJ_dvx * com_weight
        grad_aquarium_state[vy_index_global] = dJ_dvy * com_weight
    end

    return grad_aquarium_state, grad_bluff_body_state, grad_swimmer_control,
            grad_fluid_params, grad_swimmer_params, grad_bluff_body_params
end

#############################################################################################
## Terminal Cost: Emphasize final velocity
#############################################################################################

function calculate_terminal_objective(
    tank, time, aquarium_state, bluff_body_state,
)
    # Compute COM velocity at terminal time
    com_vel = calculate_com_velocity(tank, aquarium_state)
    vx, vy = com_vel[1], com_vel[2]

    # Desired direction unit vector
    d_x = cos(desired_direction_angle)
    d_y = sin(desired_direction_angle)

    # Velocity component in desired direction
    v_dot_d = vx * d_x + vy * d_y

    # Terminal objective: NEGATIVE of velocity (maximize velocity = minimize negative velocity)
    return -terminal_cost_weight * v_dot_d
end

function calculate_terminal_objective_gradients(
    tank, time, aquarium_state, bluff_body_state,
)
    # Initialize all gradients to zero
    grad_aquarium_state = zeros(eltype(aquarium_state), length(aquarium_state))
    grad_bluff_body_state = zeros(eltype(bluff_body_state), length(bluff_body_state))
    grad_fluid_params = zeros(4)
    grad_swimmer_params = zeros(length(collect_differentiable_params(tank.swimmer)))
    grad_bluff_body_params = zeros(length(collect_differentiable_params(tank.bluff_body)))

    # ==================================================================================
    # Same gradient computation as stage cost (weights may differ)
    # ==================================================================================

    # Desired direction unit vector
    d_x = cos(desired_direction_angle)
    d_y = sin(desired_direction_angle)

    # Derivative of cost w.r.t. COM velocity components
    dJ_dvx = -terminal_cost_weight * d_x
    dJ_dvy = -terminal_cost_weight * d_y

    # Compute gradients w.r.t. individual link velocities
    total_mass = sum(b.mass for b in tank.swimmer.bodies)

    n_links = tank.swimmer.n_bodies
    for i in 1:n_links
        link_mass = tank.swimmer.bodies[i].mass
        com_weight = link_mass / total_mass

        link_vel_indices = tank.swimmer.velocity_indices[(3*(i-1)+1):(3*i)]
        vx_index_global = tank.swimmer_body_state_indices[link_vel_indices[1]]
        vy_index_global = tank.swimmer_body_state_indices[link_vel_indices[2]]

        grad_aquarium_state[vx_index_global] = dJ_dvx * com_weight
        grad_aquarium_state[vy_index_global] = dJ_dvy * com_weight
    end

    return grad_aquarium_state, grad_bluff_body_state, grad_fluid_params,
            grad_swimmer_params, grad_bluff_body_params
end

#############################################################################################
## USAGE NOTES
#############################################################################################
#
# To use these functions in your optimization script:
#
# 1. Set global parameters (or use closures):
#    global desired_direction_angle = deg2rad(0)  # 0° = +x direction
#    global stage_cost_weight = time_step         # Include Δt for proper integration
#    global terminal_cost_weight = 1.0            # Can be larger to emphasize final state
#    global averaging_window_time = 0.5           # Average over last 0.5 seconds
#    global final_time = 3.0
#
# 2. Pass to simulate_aquarium():
#    trajectories = simulate_aquarium(
#        tank, aquarium_state_0, final_time, zeros(0), control_params;
#        calculate_objective=true,
#        calculate_gradient_wrt_control_params=true,
#        calculate_control_input_from_params=calculate_control_input_from_params,
#        calculate_stage_objective=calculate_stage_objective,
#        calculate_terminal_objective=calculate_terminal_objective,
#        calculate_stage_objective_gradients=calculate_stage_objective_gradients,
#        calculate_terminal_objective_gradients=calculate_terminal_objective_gradients,
#        ...
#    )
#
# 3. Interpret results:
#    - Objective value is NEGATIVE of velocity (more negative = faster)
#    - If objective = -50, the fish is moving at ~50 cm/s in desired direction
#    - Optimizer will minimize cost → maximize velocity
#
# 4. Comparison with position-based cost:
#    - Position-based: penalizes deviation from target displacement
#    - Velocity-based: directly optimizes escape speed
#    - Velocity-based is more natural for escape maneuvers
#    - Both can work, but velocity gives more direct control over dynamics
#
#############################################################################################
