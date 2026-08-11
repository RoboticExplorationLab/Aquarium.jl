#############################################################################################
## Cost Functions for C-Start Optimization
#############################################################################################
# Stage cost: none
# Terminal cost: quadratic penalty on COM displacement + orientation
#   J = ½[w_x * ((com_x - target_x) / L)² + w_y * ((com_y - target_y) / L)²
#        + w_orient * (θ_avg - desired_angle)²]
#
# Required globals: target_θ, eel_length,
#                   x_weight, y_weight, orientation_weight, target_x, target_y
#############################################################################################

#############################################################################################
## Stage Cost: None
#############################################################################################

function calculate_stage_objective(
    tank, time, aquarium_state, bluff_body_state, swimmer_control,
)
    return zero(eltype(aquarium_state))
end

function calculate_stage_objective_gradients(
    tank, time, aquarium_state, bluff_body_state, swimmer_control,
)
    grad_aquarium_state = zeros(eltype(aquarium_state), length(aquarium_state))
    grad_bluff_body_state = zeros(eltype(bluff_body_state), length(bluff_body_state))
    grad_swimmer_control = zeros(eltype(swimmer_control), length(swimmer_control))
    grad_fluid_params = zeros(4)
    grad_swimmer_params = zeros(length(collect_differentiable_params(tank.swimmer)))
    grad_bluff_body_params = zeros(length(collect_differentiable_params(tank.bluff_body)))

    return grad_aquarium_state, grad_bluff_body_state, grad_swimmer_control,
            grad_fluid_params, grad_swimmer_params, grad_bluff_body_params
end

#############################################################################################
## Terminal Cost: COM displacement + orientation alignment
#############################################################################################

function calculate_terminal_objective(
    tank, time, aquarium_state, bluff_body_state,
)
    com_pos = calculate_com_position(tank, aquarium_state)

    x_deviation = (com_pos[1] - target_x) / eel_length
    y_deviation = (com_pos[2] - target_y) / eel_length

    # Orientation term: average link angle alignment with desired direction
    n_bodies = tank.swimmer.n_bodies
    θ_avg = zero(eltype(aquarium_state))
    for i in 1:n_bodies
        cfg_indices = tank.swimmer_configuration_indices[body_configuration_indices(i)]
        θ_avg += aquarium_state[cfg_indices[3]]
    end
    θ_avg /= n_bodies

    orient_cost = θ_avg - target_θ

    return 0.5 * (x_weight * x_deviation^2 + y_weight * y_deviation^2 + orientation_weight * orient_cost^2)
end

function calculate_terminal_objective_gradients(
    tank, time, aquarium_state, bluff_body_state,
)
    grad_aquarium_state = zeros(eltype(aquarium_state), length(aquarium_state))
    grad_bluff_body_state = zeros(eltype(bluff_body_state), length(bluff_body_state))
    grad_fluid_params = zeros(4)
    grad_swimmer_params = zeros(length(collect_differentiable_params(tank.swimmer)))
    grad_bluff_body_params = zeros(length(collect_differentiable_params(tank.bluff_body)))

    com_pos = calculate_com_position(tank, aquarium_state)

    x_deviation = (com_pos[1] - target_x) / eel_length
    y_deviation = (com_pos[2] - target_y) / eel_length

    n_bodies = tank.swimmer.n_bodies
    θ_avg = zero(eltype(aquarium_state))
    for i in 1:n_bodies
        cfg_indices = tank.swimmer_configuration_indices[body_configuration_indices(i)]
        θ_avg += aquarium_state[cfg_indices[3]]
    end
    θ_avg /= n_bodies

    x_grad_factor = x_weight * x_deviation / eel_length
    y_grad_factor = y_weight * y_deviation / eel_length

    total_mass = sum(b.mass for b in tank.swimmer.bodies)
    for i in 1:n_bodies
        com_weight = tank.swimmer.bodies[i].mass / total_mass
        cfg_indices = tank.swimmer_configuration_indices[body_configuration_indices(i)]

        grad_aquarium_state[cfg_indices[1]] = x_grad_factor * com_weight
        grad_aquarium_state[cfg_indices[2]] = y_grad_factor * com_weight
    end

    orient_grad = orientation_weight * (θ_avg - target_θ) / n_bodies
    for i in 1:n_bodies
        cfg_indices = tank.swimmer_configuration_indices[body_configuration_indices(i)]
        grad_aquarium_state[cfg_indices[3]] += orient_grad
    end

    return grad_aquarium_state, grad_bluff_body_state, grad_fluid_params,
            grad_swimmer_params, grad_bluff_body_params
end
