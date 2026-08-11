#############################################################################################
## Shared dynamics primitives for every concrete `SolidSystem` subtype.
##
## Organized by concern:
##   1. State-layout helpers (body slices, midpoint rule, rotation, attachment points)
##   2. Energies and constraint residual
##   3. Mass matrix and force primitives (damping, actuators)
##   4. Solid stationarity / dynamics residual and jacobian (tank-facing)
##   5. `initialize_solid_state`
##
## All methods dispatch on `::SolidSystem` unless a concrete split is required
## (e.g., `PassiveSystem` vs `ActuatedSystem` for actuator forces).
#############################################################################################

# --- State layout helpers ----------------------------------------------------------------

body_configuration_indices(i::Int) = (3i - 2):(3i)
body_velocity_indices(i::Int) = (3i - 2):(3i)

function body_configuration(configuration::AbstractVector, i::Int)
    return @view configuration[body_configuration_indices(i)]
end

function body_velocity(velocity::AbstractVector, i::Int)
    return @view velocity[body_velocity_indices(i)]
end

# 2D rotation matrix R(θ) that rotates body-frame vectors into world frame.
function rotation_2d(θ)
    c, s = cos(θ), sin(θ)
    return [c -s; s c]
end

# World-frame position of a named attachment point on a body, given the body's config slice.
function body_attachment_point_world(
    body::RigidBody,
    body_config::AbstractVector,
    role::Symbol,
)
    origin = [body_config[1], body_config[2]]
    θ = body_config[3]
    local_pt = local_attachment_point(body.shape, role)
    shifted = body.com_offset .+ local_pt
    R = rotation_2d(θ)
    return origin .+ R * shifted
end

# Midpoint state used by the variational integrator:
#   q_mid = q - (dt/2) * v
# Works for body-state vectors of length n_body_states and full-state vectors of length
# n_states (the dual slice is passed through unchanged).
function calculate_midpoint_state(system, system_or_body_state::AbstractVector)
    midpoint_velocity = system_or_body_state[system.velocity_indices]
    midpoint_configuration = system_or_body_state[system.configuration_indices] .-
        (0.5 * system.time_step) .* midpoint_velocity
    midpoint_state = copy(system_or_body_state)
    midpoint_state[system.configuration_indices] .= midpoint_configuration
    midpoint_state[system.velocity_indices] .= midpoint_velocity
    return midpoint_state
end

# --- Scalar type helper for differentiable-params flow ------------------------------------

# Extracts the scalar type used by a system's differentiable parameters so PE / gradient
# return types propagate nested `ForwardDiff.Dual` values correctly.
function _system_param_type(system::SolidSystem)
    T = Float64
    for b in system.bodies
        b isa RigidBody || continue
        T = promote_type(T, typeof(b.mass), typeof(b.moi), eltype(b.com_offset))
    end
    for j in system.joints
        T = promote_type(T, typeof(j.stiffness), typeof(j.damping), typeof(j.equilibrium_angle))
        if j isa WorldPinJoint
            T = promote_type(T, eltype(j.world_position))
        end
    end
    return T
end

# --- Energies and constraint residual ----------------------------------------------------

function calculate_kinetic_energy(
    system::SolidSystem,
    system_or_body_state::AbstractVector,
)
    velocity = _extract_velocity(system, system_or_body_state)
    mass_diag = calculate_mass_matrix_diagonal(system)
    T = promote_type(eltype(velocity), eltype(mass_diag))
    ke = zero(T)
    @inbounds for i in eachindex(velocity)
        ke += 0.5 * mass_diag[i] * velocity[i] * velocity[i]
    end
    return ke
end

function calculate_total_energy(
    system::SolidSystem,
    system_or_body_state::AbstractVector,
)
    # The variational integrator stores (q_{k+1}, v_{k+1}) where v_{k+1} is the
    # midpoint velocity: q_{k+1} = q_k + dt * v_{k+1}.  To evaluate energy at a
    # consistent instant, use the midpoint configuration q_{k+1/2} = q_{k+1} - (dt/2)*v
    # paired with v_{k+1}, both at time t_{k+1/2}.
    midpoint_state = calculate_midpoint_state(system, system_or_body_state)
    midpoint_configuration = midpoint_state[system.configuration_indices]
    return calculate_kinetic_energy(system, midpoint_state) +
           calculate_potential_energy(system, midpoint_configuration)
end

function calculate_potential_energy(
    system::SolidSystem,
    configuration::AbstractVector,
)
    T = promote_type(eltype(configuration), _system_param_type(system))
    pe = zero(T)
    g_y = system.gravity[2]

    # Gravitational PE from each body (COM is at body-frame origin under Interpretation P).
    for (i, body) in enumerate(system.bodies)
        cfg = body_configuration(configuration, i)
        y_com = cfg[2]
        pe += -body.mass * g_y * y_com
    end

    # Joint spring PE
    for joint in system.joints
        pe += calculate_joint_potential_energy(joint, configuration, system.bodies)
    end

    return pe
end

# Analytical gradient of potential energy w.r.t. configuration. Avoids nested ForwardDiff.
function calculate_potential_energy_gradient(
    system::SolidSystem,
    configuration::AbstractVector,
)
    T = promote_type(eltype(configuration), _system_param_type(system))
    grad = zeros(T, system.n_configurations)
    g_y = system.gravity[2]

    for (i, body) in enumerate(system.bodies)
        body isa RigidBody || continue
        y_index = 3 * (i - 1) + 2
        grad[y_index] += -body.mass * g_y
    end

    for joint in system.joints
        _add_joint_potential_energy_gradient!(grad, joint, configuration, system.bodies)
    end

    return grad
end

function _add_joint_potential_energy_gradient!(
    grad::AbstractVector,
    joint::PinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    θ_A = configuration[3 * joint.body_id_A]
    θ_B = configuration[3 * joint.body_id_B]
    Δ = θ_B - θ_A - joint.equilibrium_angle
    # PE = (k/2) * Δ^2 → ∂/∂θ_A = -k*Δ, ∂/∂θ_B = +k*Δ
    grad[3 * joint.body_id_A] += -joint.stiffness * Δ
    grad[3 * joint.body_id_B] += joint.stiffness * Δ
    return nothing
end

function _add_joint_potential_energy_gradient!(
    grad::AbstractVector,
    joint::WorldPinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    θ = configuration[3 * joint.body_id]
    Δ = θ - joint.equilibrium_angle
    grad[3 * joint.body_id] += joint.stiffness * Δ
    return nothing
end

function calculate_potential_energy_hessian(
    system::SolidSystem,
    configuration::AbstractVector,
)
    T = promote_type(eltype(configuration), _system_param_type(system))
    H = zeros(T, system.n_configurations, system.n_configurations)
    for joint in system.joints
        _add_joint_potential_energy_hessian!(H, joint, configuration, system.bodies)
    end
    return H
end

function _add_joint_potential_energy_hessian!(
    H::AbstractMatrix,
    joint::PinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    iA = 3 * joint.body_id_A
    iB = 3 * joint.body_id_B
    k = joint.stiffness
    H[iA, iA] += k
    H[iA, iB] += -k
    H[iB, iA] += -k
    H[iB, iB] += k
    return nothing
end

function _add_joint_potential_energy_hessian!(
    H::AbstractMatrix,
    joint::WorldPinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    i = 3 * joint.body_id
    H[i, i] += joint.stiffness
    return nothing
end

function calculate_system_constraint_residual(
    system::SolidSystem,
    configuration::AbstractVector,
)
    T = promote_type(eltype(configuration), _system_param_type(system))
    if isempty(system.joints)
        return T[]
    end
    n_positional = sum(joint_n_constraints(j) for j in system.joints)
    residual = Vector{T}(undef, n_positional)
    offset = 0
    for joint in system.joints
        r = calculate_joint_constraint_residual(joint, configuration, system.bodies)
        n = length(r)
        residual[offset+1:offset+n] .= r
        offset += n
    end
    return residual
end

# --- Prescribed angle constraint (for :prescribed actuation mode) -------------------------

function _prescribed_angle_residual(joint::WorldPinJoint, configuration::AbstractVector, θ_desired)
    θ = configuration[3 * joint.body_id]
    return θ - θ_desired
end

function _prescribed_angle_residual(joint::PinJoint, configuration::AbstractVector, θ_desired)
    θ_A = configuration[3 * joint.body_id_A]
    θ_B = configuration[3 * joint.body_id_B]
    return (θ_B - θ_A) - θ_desired
end

function calculate_prescribed_angle_constraint_residual(
    system::ActuatedSystem,
    configuration::AbstractVector,
    control_k::AbstractVector,
)
    T = promote_type(eltype(configuration), eltype(control_k))
    n = system.n_actuators
    residual = Vector{T}(undef, n)
    for (i, actuator) in enumerate(system.actuators)
        joint = system.joints[actuator.joint_id]
        residual[i] = _prescribed_angle_residual(joint, configuration, control_k[i])
    end
    return residual
end

function _add_prescribed_angle_vjp!(out::AbstractVector, joint::WorldPinJoint, dual_val)
    out[3 * joint.body_id] += dual_val
    return nothing
end

function _add_prescribed_angle_vjp!(out::AbstractVector, joint::PinJoint, dual_val)
    out[3 * joint.body_id_A] += -dual_val
    out[3 * joint.body_id_B] += dual_val
    return nothing
end

function _add_prescribed_angle_constraint_vjp!(
    out::AbstractVector,
    system::ActuatedSystem,
    configuration::AbstractVector,
    angle_duals::AbstractVector,
)
    for (i, actuator) in enumerate(system.actuators)
        joint = system.joints[actuator.joint_id]
        _add_prescribed_angle_vjp!(out, joint, angle_duals[i])
    end
    return nothing
end

# --- Forward constraint Jacobians (∂c/∂q as a matrix) ------------------------------------

function calculate_system_constraint_jacobian(
    system::SolidSystem,
    configuration::AbstractVector,
)
    T = promote_type(eltype(configuration), _system_param_type(system))
    if isempty(system.joints)
        return zeros(T, 0, system.n_configurations)
    end
    n_positional = sum(joint_n_constraints(j) for j in system.joints)
    J = zeros(T, n_positional, system.n_configurations)
    offset = 0
    for joint in system.joints
        n_c = joint_n_constraints(joint)
        _add_joint_constraint_jacobian!(J, joint, configuration, system.bodies, offset)
        offset += n_c
    end
    return J
end

function _add_joint_constraint_jacobian!(
    J::AbstractMatrix,
    joint::WorldPinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
    row_offset::Int,
)
    i = 3 * (joint.body_id - 1)
    body = bodies[joint.body_id]
    θ = configuration[i + 3]

    local_pt = body isa RigidBody ? (body.com_offset .+ local_attachment_point(body.shape, joint.role)) : [zero(θ), zero(θ)]

    s, c = sin(θ), cos(θ)
    dR_v = [-s * local_pt[1] - c * local_pt[2], c * local_pt[1] - s * local_pt[2]]

    J[row_offset + 1, i + 1] += 1
    J[row_offset + 2, i + 2] += 1
    J[row_offset + 1, i + 3] += dR_v[1]
    J[row_offset + 2, i + 3] += dR_v[2]
    return nothing
end

function _add_joint_constraint_jacobian!(
    J::AbstractMatrix,
    joint::PinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
    row_offset::Int,
)
    iA = 3 * (joint.body_id_A - 1)
    iB = 3 * (joint.body_id_B - 1)

    body_A = bodies[joint.body_id_A]
    body_B = bodies[joint.body_id_B]
    θ_A = configuration[iA + 3]
    θ_B = configuration[iB + 3]

    local_A = body_A isa RigidBody ? (body_A.com_offset .+ local_attachment_point(body_A.shape, joint.role_A)) : [zero(θ_A), zero(θ_A)]
    local_B = body_B isa RigidBody ? (body_B.com_offset .+ local_attachment_point(body_B.shape, joint.role_B)) : [zero(θ_B), zero(θ_B)]

    sA, cA = sin(θ_A), cos(θ_A)
    sB, cB = sin(θ_B), cos(θ_B)
    dRA_v = [-sA * local_A[1] - cA * local_A[2], cA * local_A[1] - sA * local_A[2]]
    dRB_v = [-sB * local_B[1] - cB * local_B[2], cB * local_B[1] - sB * local_B[2]]

    J[row_offset + 1, iA + 1] += 1
    J[row_offset + 2, iA + 2] += 1
    J[row_offset + 1, iA + 3] += dRA_v[1]
    J[row_offset + 2, iA + 3] += dRA_v[2]

    J[row_offset + 1, iB + 1] += -1
    J[row_offset + 2, iB + 2] += -1
    J[row_offset + 1, iB + 3] += -dRB_v[1]
    J[row_offset + 2, iB + 3] += -dRB_v[2]
    return nothing
end

function calculate_prescribed_angle_constraint_jacobian(
    system::ActuatedSystem,
    configuration::AbstractVector,
)
    T = promote_type(eltype(configuration), _system_param_type(system))
    n = system.n_actuators
    J = zeros(T, n, system.n_configurations)
    for (i, actuator) in enumerate(system.actuators)
        joint = system.joints[actuator.joint_id]
        _add_prescribed_angle_constraint_jacobian!(J, joint, i)
    end
    return J
end

function _add_prescribed_angle_constraint_jacobian!(J::AbstractMatrix, joint::WorldPinJoint, row::Int)
    J[row, 3 * joint.body_id] += 1
    return nothing
end

function _add_prescribed_angle_constraint_jacobian!(J::AbstractMatrix, joint::PinJoint, row::Int)
    J[row, 3 * joint.body_id_A] += -1
    J[row, 3 * joint.body_id_B] += 1
    return nothing
end

# --- Prescribed mode helper ---------------------------------------------------------------

_is_prescribed(system::ActuatedSystem) = system.actuation_mode == :prescribed
_is_prescribed(::SolidSystem) = false

# --- Mass matrix, damping, actuator force primitives --------------------------------------

# Diagonal mass matrix for a composition-based system: per body [mass, mass, moi] stacked.
function calculate_mass_matrix_diagonal(system::SolidSystem)
    T = promote_type(Float64, [typeof(b.mass) for b in system.bodies if b isa RigidBody]...)
    diag = Vector{T}(undef, system.n_velocities)
    for (i, body) in enumerate(system.bodies)
        body isa RigidBody || continue
        base = 3 * (i - 1)
        diag[base + 1] = body.mass
        diag[base + 2] = body.mass
        diag[base + 3] = body.moi
    end
    return diag
end

# Damping accepts either a velocity vector or a full system/body state vector; the
# `_extract_velocity` helper normalizes the input.
function calculate_damping_force(
    system::SolidSystem,
    system_or_body_state::AbstractVector,
)
    velocity = _extract_velocity(system, system_or_body_state)
    T = promote_type(eltype(velocity), _system_param_type(system))
    force = zeros(T, system.n_velocities)
    for joint in system.joints
        force .+= calculate_joint_damping_force(joint, velocity, system.bodies)
    end
    return force
end

function calculate_damping_force_jacobian(
    system::SolidSystem,
    velocity::AbstractVector,
)
    T = promote_type(eltype(velocity), _system_param_type(system))
    J = zeros(T, system.n_velocities, system.n_velocities)
    for joint in system.joints
        _add_joint_damping_force_jacobian!(J, joint, velocity, system.bodies)
    end
    return J
end

function _add_joint_damping_force_jacobian!(
    J::AbstractMatrix,
    joint::PinJoint,
    velocity::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    iA = 3 * joint.body_id_A
    iB = 3 * joint.body_id_B
    b = joint.damping
    J[iA, iA] += b
    J[iA, iB] += -b
    J[iB, iA] += -b
    J[iB, iB] += b
    return nothing
end

function _add_joint_damping_force_jacobian!(
    J::AbstractMatrix,
    joint::WorldPinJoint,
    velocity::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    i = 3 * joint.body_id
    J[i, i] += -joint.damping
    return nothing
end

function _extract_velocity(system::SolidSystem, system_or_body_state::AbstractVector)
    if length(system_or_body_state) == system.n_velocities
        return system_or_body_state
    elseif length(system_or_body_state) == system.n_body_states
        return system_or_body_state[system.velocity_indices]
    elseif length(system_or_body_state) == system.n_states
        return system_or_body_state[system.velocity_indices]
    else
        error("_extract_velocity: unexpected state length $(length(system_or_body_state))")
    end
end

# PassiveSystem has no actuators → zero force vector.
function calculate_actuator_forces(
    system::PassiveSystem,
    system_or_body_state::AbstractVector,
    control_k::AbstractVector = zeros(0),
)
    T = promote_type(eltype(system_or_body_state), eltype(control_k), Float64)
    return zeros(T, system.n_velocities)
end

# ActuatedSystem delegates to the 2D JointServoMotor-aware force loop.
function calculate_actuator_forces(
    system::ActuatedSystem,
    system_or_body_state::AbstractVector,
    control_k::AbstractVector = zeros(0),
)
    state_slice = if length(system_or_body_state) == system.n_body_states
        system_or_body_state
    elseif length(system_or_body_state) == system.n_states
        system_or_body_state[system.body_state_indices]
    else
        error("calculate_actuator_forces: unexpected state length $(length(system_or_body_state))")
    end
    return calculate_new_actuator_forces(system, state_slice, control_k)
end

# --- Actuator force Jacobian (analytical, with clamp handling) --------------------------

function calculate_actuator_force_jacobians(
    system::ActuatedSystem,
    configuration::AbstractVector,
    velocity::AbstractVector,
    control_k::AbstractVector,
)
    n_vel = system.n_velocities
    n_config = system.n_configurations
    T = promote_type(eltype(configuration), eltype(velocity), eltype(control_k))
    J_q = zeros(T, n_vel, n_config)
    J_v = zeros(T, n_vel, n_vel)
    offset = 0
    for actuator in system.actuators
        n = n_control_inputs_per_actuator(actuator)
        slice = @view control_k[offset+1:offset+n]
        _add_actuator_force_jacobian!(J_q, J_v, actuator, configuration, velocity, slice, system)
        offset += n
    end
    return J_q, J_v
end

function calculate_actuator_force_jacobians(
    system::PassiveSystem,
    configuration::AbstractVector,
    velocity::AbstractVector,
    control_k::AbstractVector = zeros(0),
)
    T = promote_type(eltype(configuration), eltype(velocity), Float64)
    return zeros(T, system.n_velocities, system.n_configurations),
           zeros(T, system.n_velocities, system.n_velocities)
end

function _add_actuator_force_jacobian!(
    J_q::AbstractMatrix,
    J_v::AbstractMatrix,
    motor::JointServoMotor,
    configuration::AbstractVector,
    velocity::AbstractVector,
    control_slice::AbstractVector,
    system::ActuatedSystem,
)
    joint = system.joints[motor.joint_id]
    n_bodies = system.n_bodies

    θ_desired = control_slice[1]
    ω_desired = control_slice[2]
    body_state = [configuration; velocity]
    θ_current, ω_current = current_joint_state(joint, body_state, n_bodies)

    controller = motor.controller
    Kp = controller.Kp
    Kd = controller.Kd

    τ_raw = Kp * (θ_desired - θ_current) + Kd * (ω_desired - ω_current)
    τ_pd = clamp(τ_raw, controller.output_min, controller.output_max)

    pd_saturated = (τ_raw < controller.output_min) || (τ_raw > controller.output_max)
    motor_saturated = (τ_pd < -motor.max_torque) || (τ_pd > motor.max_torque)
    if pd_saturated || motor_saturated
        return nothing
    end

    _apply_joint_torque_jacobian!(J_q, J_v, joint, Kp, Kd)
    return nothing
end

function _apply_joint_torque_jacobian!(
    J_q::AbstractMatrix, J_v::AbstractMatrix,
    joint::WorldPinJoint, Kp, Kd,
)
    i = 3 * joint.body_id
    J_q[i, i] += -Kp
    J_v[i, i] += -Kd
    return nothing
end

function _apply_joint_torque_jacobian!(
    J_q::AbstractMatrix, J_v::AbstractMatrix,
    joint::PinJoint, Kp, Kd,
)
    iA = 3 * joint.body_id_A
    iB = 3 * joint.body_id_B
    J_q[iA, iA] += -Kp
    J_q[iA, iB] += Kp
    J_q[iB, iA] += Kp
    J_q[iB, iB] += -Kp
    J_v[iA, iA] += -Kd
    J_v[iA, iB] += Kd
    J_v[iB, iA] += Kd
    J_v[iB, iB] += -Kd
    return nothing
end

# --- Solid stationarity residual (variational integrator KKT row) -----------------------

function calculate_solid_stationarity_residual(
    solid_system::SolidSystem,
    system_state_kp1::AbstractVector,
    system_state_k::AbstractVector,
    control_k::AbstractVector = zeros(0),
)
    n_body_states = solid_system.n_body_states
    time_step = solid_system.time_step

    configuration_kp1 = system_state_kp1[solid_system.configuration_indices]
    velocity_kp1 = system_state_kp1[solid_system.velocity_indices]
    constraint_dual_kp1 = system_state_kp1[solid_system.dual_indices]

    configuration_k = system_state_k[solid_system.configuration_indices]
    velocity_k = system_state_k[solid_system.velocity_indices]

    midpoint_state_kp1 = calculate_midpoint_state(solid_system, system_state_kp1)
    midpoint_configuration_kp1 = midpoint_state_kp1[solid_system.configuration_indices]
    midpoint_state_k = calculate_midpoint_state(solid_system, system_state_k)
    midpoint_configuration_k = midpoint_state_k[solid_system.configuration_indices]

    mass_matrix_diag = calculate_mass_matrix_diagonal(solid_system)

    # Analytical constraint VJP: (∂residual/∂configuration)^T * dual, computed per joint.
    T_vjp = promote_type(eltype(configuration_k), eltype(constraint_dual_kp1), _system_param_type(solid_system))
    system_constraint_vjp_k = zeros(T_vjp, solid_system.n_configurations)
    if solid_system.n_constraints > 0
        offset = 0
        for joint in solid_system.joints
            n_c = joint_n_constraints(joint)
            slice = @view constraint_dual_kp1[offset+1:offset+n_c]
            _add_joint_constraint_vjp!(system_constraint_vjp_k, joint, configuration_k, slice, solid_system.bodies)
            offset += n_c
        end
        # Prescribed mode: add angle constraint VJP from the angle dual block
        if _is_prescribed(solid_system) && solid_system.n_actuators > 0
            angle_duals = @view constraint_dual_kp1[offset+1:offset+solid_system.n_actuators]
            _add_prescribed_angle_constraint_vjp!(system_constraint_vjp_k, solid_system, configuration_k, angle_duals)
        end
    end

    # Analytical PE gradient avoids nested ForwardDiff (which fails when both system
    # params and states are Dual-wrapped from different tags).
    potential_grad_kp1 = calculate_potential_energy_gradient(solid_system, midpoint_configuration_kp1)
    potential_grad_k = calculate_potential_energy_gradient(solid_system, midpoint_configuration_k)

    # Prescribed mode: skip actuator forces (constraint handles the motion)
    if _is_prescribed(solid_system) || isempty(control_k)
        actuator_force_kp1 = zeros(eltype(velocity_kp1), solid_system.n_velocities)
    else
        actuator_force_kp1 = calculate_actuator_forces(
            solid_system, midpoint_state_kp1, control_k,
        )
    end

    damping_force_kp1 = calculate_damping_force(solid_system, midpoint_state_kp1)
    damping_force_k = calculate_damping_force(solid_system, midpoint_state_k)

    configuration_residual = configuration_kp1 .- configuration_k .- time_step .* velocity_kp1

    velocity_kp1_residual = mass_matrix_diag .* velocity_kp1 .+
        (time_step / 2) .* potential_grad_kp1 .-
        time_step .* actuator_force_kp1 .-
        (time_step / 2) .* damping_force_kp1 .+
        system_constraint_vjp_k
    velocity_k_residual = -mass_matrix_diag .* velocity_k .+
        (time_step / 2) .* potential_grad_k .-
        (time_step / 2) .* damping_force_k

    T_out = promote_type(
        eltype(configuration_residual),
        eltype(velocity_kp1_residual),
        eltype(velocity_k_residual),
    )
    solid_stationarity_residual = zeros(T_out, n_body_states)
    solid_stationarity_residual[solid_system.configuration_indices] = configuration_residual
    solid_stationarity_residual[solid_system.velocity_indices] =
        velocity_kp1_residual .+ velocity_k_residual

    return solid_stationarity_residual
end

function _analytical_stationarity_jacobian_kp1(
    sys::SolidSystem,
    state_kp1::AbstractVector,
    state_k::AbstractVector,
    control_k::AbstractVector,
)
    dt = sys.time_step
    n_bs = sys.n_body_states
    n_s = sys.n_states
    n_q = sys.n_configurations
    n_v = sys.n_velocities
    qi = sys.configuration_indices
    vi = sys.velocity_indices

    midpoint_kp1 = calculate_midpoint_state(sys, state_kp1)
    mid_config = midpoint_kp1[qi]
    mid_vel = midpoint_kp1[vi]
    config_k = state_k[qi]

    T = promote_type(eltype(state_kp1), eltype(state_k), eltype(control_k), _system_param_type(sys))
    ∂ = zeros(T, n_bs, n_s)

    # --- Configuration rows: r_q = q_{k+1} - q_k - dt * v_{k+1} ---
    for (row, col) in zip(1:n_q, qi)
        ∂[row, col] = 1.0
    end
    for (row, col) in zip(1:n_q, vi)
        ∂[row, col] = -dt
    end

    # --- Velocity rows ---
    mass_diag = calculate_mass_matrix_diagonal(sys)
    H_PE = calculate_potential_energy_hessian(sys, mid_config)
    J_damp = calculate_damping_force_jacobian(sys, mid_vel)

    has_actuator = !_is_prescribed(sys) && !isempty(control_k)
    if has_actuator
        J_act_q, J_act_v = calculate_actuator_force_jacobians(sys, mid_config, mid_vel, control_k)
    end

    # ∂r_v/∂q_{k+1} = (dt/2)*H_PE - dt*J_act_q
    dv_dq = (dt / 2) .* H_PE
    if has_actuator
        dv_dq .-= dt .* J_act_q
    end

    # ∂r_v/∂v_{k+1} = M - (dt²/4)*H_PE + (dt²/2)*J_act_q - dt*J_act_v - (dt/2)*J_damp
    dv_dv = -(dt^2 / 4) .* H_PE .- (dt / 2) .* J_damp
    for i in 1:n_v
        dv_dv[i, i] += mass_diag[i]
    end
    if has_actuator
        dv_dv .+= (dt^2 / 2) .* J_act_q .- dt .* J_act_v
    end

    v_rows = n_q+1:n_bs
    for (ri, r) in enumerate(v_rows)
        for (ci, c) in enumerate(qi)
            ∂[r, c] = dv_dq[ri, ci]
        end
        for (ci, c) in enumerate(vi)
            ∂[r, c] = dv_dv[ri, ci]
        end
    end

    # ∂r_v/∂λ_{k+1}: constraint Jacobian transpose evaluated at config_k
    if sys.n_constraints > 0
        di = sys.dual_indices
        J_pos = calculate_system_constraint_jacobian(sys, config_k)
        n_pos = size(J_pos, 1)
        for (ri, r) in enumerate(v_rows)
            for ci in 1:n_pos
                ∂[r, di[ci]] = J_pos[ci, ri]
            end
        end
        if _is_prescribed(sys) && sys.n_actuators > 0
            J_angle = calculate_prescribed_angle_constraint_jacobian(sys, config_k)
            for (ri, r) in enumerate(v_rows)
                for ci in 1:sys.n_actuators
                    ∂[r, di[n_pos + ci]] = J_angle[ci, ri]
                end
            end
        end
    end

    return ∂
end

# Analytical ∂_∂kp1, ForwardDiff for the remaining three blocks.
function calculate_solid_stationarity_jacobian(
    solid_system::SolidSystem,
    system_state_kp1::AbstractVector,
    system_state_k::AbstractVector,
    control_k::AbstractVector = zeros(0),
)
    ∂_∂kp1 = _analytical_stationarity_jacobian_kp1(
        solid_system, system_state_kp1, system_state_k, control_k,
    )
    ∂_∂k = ForwardDiff.jacobian(
        x -> calculate_solid_stationarity_residual(solid_system, system_state_kp1, x, control_k),
        system_state_k,
    )
    ∂_∂u = if isempty(control_k)
        zeros(eltype(∂_∂kp1), solid_system.n_body_states, 0)
    else
        ForwardDiff.jacobian(
            u -> calculate_solid_stationarity_residual(solid_system, system_state_kp1, system_state_k, u),
            control_k,
        )
    end
    p_current = collect_differentiable_params(solid_system)
    ∂_∂p = ForwardDiff.jacobian(
        p -> calculate_solid_stationarity_residual(
            inject_differentiable_params(solid_system, p),
            system_state_kp1, system_state_k, control_k,
        ),
        p_current,
    )
    return ∂_∂kp1, ∂_∂k, ∂_∂u, ∂_∂p
end

# --- Dynamics residual (stationarity + constraint violation) -----------------------------

function calculate_solid_dynamics_residual(
    solid_system::SolidSystem,
    system_state_kp1::AbstractVector,
    system_state_k::AbstractVector,
    control_k::AbstractVector = zeros(0),
)
    stationarity = calculate_solid_stationarity_residual(
        solid_system, system_state_kp1, system_state_k, control_k,
    )

    configuration_kp1 = system_state_kp1[solid_system.configuration_indices]

    positional_residual = if solid_system.n_constraints == 0
        eltype(stationarity)[]
    else
        calculate_system_constraint_residual(solid_system, configuration_kp1)
    end

    # Prescribed mode: append angle constraint residual after positional
    constraint_residual = if _is_prescribed(solid_system) && solid_system.n_actuators > 0
        angle_residual = calculate_prescribed_angle_constraint_residual(
            solid_system, configuration_kp1, control_k)
        vcat(positional_residual, angle_residual)
    else
        positional_residual
    end

    T_out = promote_type(eltype(stationarity), eltype(constraint_residual))
    dynamics_residual = zeros(T_out, solid_system.n_states)
    dynamics_residual[solid_system.body_state_indices] = stationarity
    dynamics_residual[solid_system.dual_indices] = constraint_residual
    return dynamics_residual
end

function calculate_solid_dynamics_jacobian(
    solid_system::SolidSystem,
    system_state_kp1::AbstractVector,
    system_state_k::AbstractVector,
    control_k::AbstractVector = zeros(0),
)
    ∂_∂kp1 = _analytical_dynamics_jacobian_kp1(
        solid_system, system_state_kp1, system_state_k, control_k,
    )
    ∂_∂k = ForwardDiff.jacobian(
        x -> calculate_solid_dynamics_residual(solid_system, system_state_kp1, x, control_k),
        system_state_k,
    )
    ∂_∂u = if isempty(control_k)
        zeros(eltype(∂_∂kp1), solid_system.n_states, 0)
    else
        ForwardDiff.jacobian(
            u -> calculate_solid_dynamics_residual(solid_system, system_state_kp1, system_state_k, u),
            control_k,
        )
    end
    p_current = collect_differentiable_params(solid_system)
    ∂_∂p = ForwardDiff.jacobian(
        p -> calculate_solid_dynamics_residual(
            inject_differentiable_params(solid_system, p),
            system_state_kp1, system_state_k, control_k,
        ),
        p_current,
    )
    return ∂_∂kp1, ∂_∂k, ∂_∂u, ∂_∂p
end

function _analytical_dynamics_jacobian_kp1(
    sys::SolidSystem,
    state_kp1::AbstractVector,
    state_k::AbstractVector,
    control_k::AbstractVector,
)
    n_s = sys.n_states
    bsi = sys.body_state_indices
    di = sys.dual_indices
    qi = sys.configuration_indices

    stat_jac = _analytical_stationarity_jacobian_kp1(sys, state_kp1, state_k, control_k)

    ∂ = zeros(eltype(stat_jac), n_s, n_s)
    ∂[bsi, :] = stat_jac

    config_kp1 = state_kp1[qi]
    if sys.n_constraints > 0
        J_pos = calculate_system_constraint_jacobian(sys, config_kp1)
        n_pos = size(J_pos, 1)
        for row in 1:n_pos
            for (ci, c) in enumerate(qi)
                ∂[di[row], c] = J_pos[row, ci]
            end
        end
        if _is_prescribed(sys) && sys.n_actuators > 0
            J_angle = calculate_prescribed_angle_constraint_jacobian(sys, config_kp1)
            for row in 1:sys.n_actuators
                for (ci, c) in enumerate(qi)
                    ∂[di[n_pos + row], c] = J_angle[row, ci]
                end
            end
        end
    end

    return ∂
end

# --- initialize_solid_state ---------------------------------------------------------------

# Build a full-state vector with the given body state (either [configuration..., velocity...]
# of length n_body_states, or a full vector of length n_states) and zero constraint duals.
# --- Prescribed mode multiplier extraction ------------------------------------------------

function extract_prescribed_angle_torques(system::ActuatedSystem, system_state::AbstractVector)
    if isempty(system.prescribed_angle_dual_indices)
        return eltype(system_state)[]
    end
    return system_state[system.prescribed_angle_dual_indices]
end

extract_prescribed_angle_torques(::SolidSystem, system_state::AbstractVector) = eltype(system_state)[]

function check_torque_feasibility(system::ActuatedSystem, system_state::AbstractVector)
    torques = extract_prescribed_angle_torques(system, system_state)
    if isempty(torques)
        return Bool[]
    end
    return [abs(torques[i]) <= system.actuators[i].max_torque for i in eachindex(torques)]
end

check_torque_feasibility(::SolidSystem, system_state::AbstractVector) = Bool[]

# --- initialize_solid_state ---------------------------------------------------------------

function initialize_solid_state(system::SolidSystem, body_state::AbstractVector)
    if length(body_state) == system.n_states
        return copy(body_state)
    elseif length(body_state) == system.n_body_states
        T = eltype(body_state)
        state = zeros(T, system.n_states)
        state[system.body_state_indices] = body_state
        return state
    else
        error("initialize_solid_state: body_state length $(length(body_state)) " *
              "doesn't match n_body_states=$(system.n_body_states) or n_states=$(system.n_states)")
    end
end


@testitem "Multiplier extraction and torque feasibility" begin
    using AquariumClosed
    @testset "extract: ActuatedPendulum" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=50.0, Kd=5.0, max_torque=2.0,
            actuation_mode=:prescribed)
        state = [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 2.0, 5.5]
        torques = extract_prescribed_angle_torques(ap, state)
        @test length(torques) == 1
        @test torques[1] ≈ 5.5 atol=1e-12
    end

    @testset "extract: RExEel" begin
        n_links = 3
        rex = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(2.0, n_links-1), actuation_mode=:prescribed)
        state = zeros(rex.n_states)
        state[rex.prescribed_angle_dual_indices] .= [7.7, -3.3]
        torques = extract_prescribed_angle_torques(rex, state)
        @test length(torques) == 2
        @test torques[1] ≈ 7.7 atol=1e-12
        @test torques[2] ≈ -3.3 atol=1e-12
    end

    @testset "extract: pd mode returns empty" begin
        ap_pd = ActuatedPendulum(0.01; bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0])
        @test isempty(extract_prescribed_angle_torques(ap_pd, zeros(ap_pd.n_states)))
    end

    @testset "check_torque_feasibility" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0], max_torque=3.0, actuation_mode=:prescribed)
        @test check_torque_feasibility(ap, [0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.5]) == [true]
        @test check_torque_feasibility(ap, [0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.0]) == [false]
        @test check_torque_feasibility(ap, [0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.0]) == [true]
        ap_pd = ActuatedPendulum(0.01; bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0])
        @test check_torque_feasibility(ap_pd, zeros(ap_pd.n_states)) == Bool[]
    end
end

@testitem "Prescribed angle constraint residual and VJP" begin
    using AquariumClosed
    using ForwardDiff

    @testset "WorldPinJoint (ActuatedPendulum)" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=50.0, Kd=5.0, max_torque=2.0,
            actuation_mode=:prescribed)
        configuration = [0.5, 0.0, 0.3]
        r = calculate_prescribed_angle_constraint_residual(ap, configuration, [0.5])
        @test length(r) == 1
        @test r[1] ≈ -0.2 atol=1e-12
        @test calculate_prescribed_angle_constraint_residual(ap, configuration, [0.3])[1] ≈ 0.0 atol=1e-12
    end

    @testset "WorldPinJoint VJP" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0], actuation_mode=:prescribed)
        out = zeros(3)
        _add_prescribed_angle_constraint_vjp!(out, ap, [0.5, 0.0, 0.3], [2.5])
        @test out[1] ≈ 0.0 atol=1e-12
        @test out[2] ≈ 0.0 atol=1e-12
        @test out[3] ≈ 2.5 atol=1e-12
    end

    @testset "PinJoint (RExEel)" begin
        n_links = 3
        rex = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(2.0, n_links-1), actuation_mode=:prescribed)
        configuration = [1.0, 0.0, 0.1,  2.0, 0.0, 0.4,  3.0, 0.0, -0.2]
        r = calculate_prescribed_angle_constraint_residual(rex, configuration, [0.2, -0.5])
        @test length(r) == 2
        @test r[1] ≈ 0.1 atol=1e-12
        @test r[2] ≈ -0.1 atol=1e-12
    end

    @testset "PinJoint VJP" begin
        n_links = 3
        rex = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(2.0, n_links-1), actuation_mode=:prescribed)
        out = zeros(9)
        _add_prescribed_angle_constraint_vjp!(out, rex, zeros(9), [3.0, -1.0])
        @test out[3] ≈ -3.0 atol=1e-12
        @test out[6] ≈ 3.0 + 1.0 atol=1e-12
        @test out[9] ≈ -1.0 atol=1e-12
        @test all(out[[1,2,4,5,7,8]] .≈ 0.0)
    end

    @testset "ForwardDiff compatible" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0], actuation_mode=:prescribed)
        J = ForwardDiff.jacobian(
            q -> calculate_prescribed_angle_constraint_residual(ap, q, [0.5]),
            [0.5, 0.0, 0.3])
        @test size(J) == (1, 3)
        @test J[1, 3] ≈ 1.0 atol=1e-12
        J_u = ForwardDiff.jacobian(
            u -> calculate_prescribed_angle_constraint_residual(ap, [0.5, 0.0, 0.3], u),
            [0.5])
        @test J_u[1, 1] ≈ -1.0 atol=1e-12
    end
end

@testitem "PassiveSystem dynamics primitives" begin
    using AquariumClosed
    # One-body pendulum: length 2, mass 3, COM at body origin, hanging from (0,0).
    body = RigidBody(Bar(2.0); mass=3.0, moi=1.0)
    joint = WorldPinJoint([0.0, 0.0], 1, :root; equilibrium_angle=0.0, stiffness=0.0, damping=0.5)
    sys = PassiveSystem(0.01, [body], Joint[joint]; gravity=[0.0, -9.81])

    @testset "potential energy: gravity only" begin
        # Body COM at height y=1.0 → PE = m * g * h = 3 * 9.81 * 1.0 = 29.43
        configuration = [0.5, 1.0, 0.3]   # (x, y, θ) of body 1
        pe = calculate_potential_energy(sys, configuration)
        @test pe ≈ 3.0 * 9.81 * 1.0 atol=1e-10
    end

    @testset "potential energy: gravity + joint spring" begin
        joint2 = WorldPinJoint([0.0, 0.0], 1, :root; equilibrium_angle=0.0, stiffness=10.0)
        sys2 = PassiveSystem(0.01, [body], Joint[joint2]; gravity=[0.0, -9.81])
        # At y=2, θ=0.5: gravity PE = 3 * 9.81 * 2 = 58.86; spring = 0.5 * 10 * 0.25 = 1.25
        configuration = [0.0, 2.0, 0.5]
        pe = calculate_potential_energy(sys2, configuration)
        @test pe ≈ 3.0 * 9.81 * 2.0 + 1.25 atol=1e-10
    end

    @testset "system constraint residual" begin
        # Satisfying config: body center at (1, 0), angle 0 → root at (0, 0) = anchor
        r = calculate_system_constraint_residual(sys, [1.0, 0.0, 0.0])
        @test length(r) == 2
        @test r ≈ [0.0, 0.0] atol=1e-12

        # Violated
        r2 = calculate_system_constraint_residual(sys, [1.5, 0.2, 0.0])
        @test r2[1] ≈ 0.5 atol=1e-12
        @test r2[2] ≈ 0.2 atol=1e-12
    end

    @testset "system damping force" begin
        # ω = 2.0, damping = 0.5 → torque = -1.0 on body 1
        velocity = [0.0, 0.0, 2.0]
        f = calculate_damping_force(sys, velocity)
        @test length(f) == 3
        @test f[1] == 0.0
        @test f[2] == 0.0
        @test f[3] ≈ -1.0 atol=1e-12
    end

    @testset "two bodies: residual accumulation" begin
        b1 = RigidBody(Bar(2.0); mass=1.0, moi=0.1)
        b2 = RigidBody(Bar(2.0); mass=1.0, moi=0.1)
        jw = WorldPinJoint([0.0, 0.0], 1, :root)
        jp = PinJoint(1, :tip, 2, :root)
        sys2 = PassiveSystem(0.01, [b1, b2], Joint[jw, jp])

        # Body 1 at (1,0), body 2 at (3,0), both θ=0.
        # B1 root = (0,0) ✓. B1 tip = (2,0). B2 root = (2,0) ✓.
        config = [1.0, 0.0, 0.0,  3.0, 0.0, 0.0]
        r = calculate_system_constraint_residual(sys2, config)
        @test length(r) == 4
        @test r ≈ zeros(4) atol=1e-12
    end
end

@testitem "Pendulum dynamics residual and Jacobian" begin
    using AquariumClosed
    using ForwardDiff
    using FiniteDiff
    using Random

    Random.seed!(10)

    system = Pendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
                      hinge_position=[0.0, 0.0], n_boundary_nodes=4,
                      ib_method=:original)

    # Non-trivial states (so FD perturbations meaningfully probe the jacobians)
    state_k = 0.05 .* randn(system.n_states)
    state_kp1 = state_k .+ 0.01 .* randn(system.n_states)

    residual = calculate_solid_dynamics_residual(system, state_kp1, state_k)
    @test length(residual) == system.n_states
    @test all(isfinite, residual)

    J_kp1, J_k, J_u, J_p = calculate_solid_dynamics_jacobian(system, state_kp1, state_k)
    @test size(J_kp1) == (system.n_states, system.n_states)
    @test size(J_k) == (system.n_states, system.n_states)
    @test size(J_u) == (system.n_states, 0)
    @test size(J_p) == (system.n_states, length(collect_differentiable_params(system)))

    # ∂_∂kp1 — the solid dynamics jacobian uses ForwardDiff internally, so FD is the cross-check
    J_kp1_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, s, state_k), state_kp1)
    @test Matrix(J_kp1) ≈ J_kp1_fd rtol=1e-5

    # ∂_∂k
    J_k_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, state_kp1, s), state_k)
    @test Matrix(J_k) ≈ J_k_fd rtol=1e-5

    # ∂_∂params via inject_differentiable_params
    p0 = collect_differentiable_params(system)
    J_p_fd = FiniteDiff.finite_difference_jacobian(p0) do p
        new_system = inject_differentiable_params(system, p)
        calculate_solid_dynamics_residual(new_system, state_kp1, state_k)
    end
    @test Matrix(J_p) ≈ J_p_fd rtol=1e-5
end

@testitem "DoublePendulum dynamics residual and Jacobian" begin
    using AquariumClosed
    using ForwardDiff
    using FiniteDiff
    using Random

    Random.seed!(11)

    system = DoublePendulum(0.01;
        bar_lengths=[1.0, 0.8], masses=[2.0, 1.5], mois=[0.1, 0.08],
        hinge_position=[0.0, 0.0], n_boundary_nodes_per_link=4,
        ib_method=:original)

    state_k = 0.05 .* randn(system.n_states)
    state_kp1 = state_k .+ 0.01 .* randn(system.n_states)

    residual = calculate_solid_dynamics_residual(system, state_kp1, state_k)
    @test length(residual) == system.n_states
    @test all(isfinite, residual)

    J_kp1, J_k, J_u, J_p = calculate_solid_dynamics_jacobian(system, state_kp1, state_k)
    @test size(J_kp1) == (system.n_states, system.n_states)
    @test size(J_k) == (system.n_states, system.n_states)
    @test size(J_u) == (system.n_states, 0)
    @test size(J_p) == (system.n_states, length(collect_differentiable_params(system)))

    J_kp1_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, s, state_k), state_kp1)
    @test Matrix(J_kp1) ≈ J_kp1_fd rtol=1e-5

    J_k_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, state_kp1, s), state_k)
    @test Matrix(J_k) ≈ J_k_fd rtol=1e-5

    p0 = collect_differentiable_params(system)
    J_p_fd = FiniteDiff.finite_difference_jacobian(p0) do p
        new_system = inject_differentiable_params(system, p)
        calculate_solid_dynamics_residual(new_system, state_kp1, state_k)
    end
    @test Matrix(J_p) ≈ J_p_fd rtol=1e-5
end

@testitem "ActuatedPendulum dynamics residual and Jacobian" begin
    using AquariumClosed
    using ForwardDiff
    using FiniteDiff
    using Random

    Random.seed!(12)

    system = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
                              hinge_position=[0.0, 0.0], n_boundary_nodes=4,
                              Kp=20.0, Kd=5.0, max_torque=2.0,
                              ib_method=:original)

    state_k = 0.05 .* randn(system.n_states)
    state_kp1 = state_k .+ 0.01 .* randn(system.n_states)
    control_input = [0.5, 0.1]

    residual = calculate_solid_dynamics_residual(system, state_kp1, state_k, control_input)
    @test length(residual) == system.n_states
    @test all(isfinite, residual)

    J_kp1, J_k, J_u, J_p = calculate_solid_dynamics_jacobian(system, state_kp1, state_k, control_input)
    @test size(J_kp1) == (system.n_states, system.n_states)
    @test size(J_k) == (system.n_states, system.n_states)
    @test size(J_u) == (system.n_states, length(control_input))
    @test size(J_p) == (system.n_states, length(collect_differentiable_params(system)))

    J_kp1_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, s, state_k, control_input),
        state_kp1)
    @test Matrix(J_kp1) ≈ J_kp1_fd rtol=1e-5

    J_k_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, state_kp1, s, control_input),
        state_k)
    @test Matrix(J_k) ≈ J_k_fd rtol=1e-5

    J_u_fd = FiniteDiff.finite_difference_jacobian(
        u -> calculate_solid_dynamics_residual(system, state_kp1, state_k, u),
        control_input)
    @test Matrix(J_u) ≈ J_u_fd rtol=1e-5

    p0 = collect_differentiable_params(system)
    J_p_fd = FiniteDiff.finite_difference_jacobian(p0) do p
        new_system = inject_differentiable_params(system, p)
        calculate_solid_dynamics_residual(new_system, state_kp1, state_k, control_input)
    end
    @test Matrix(J_p) ≈ J_p_fd rtol=1e-5
end

@testitem "Prescribed ActuatedPendulum dynamics residual and Jacobian" begin
    using AquariumClosed
    using ForwardDiff
    using FiniteDiff
    using Random

    Random.seed!(42)

    system = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
                              hinge_position=[0.0, 0.0], n_boundary_nodes=4,
                              Kp=20.0, Kd=5.0, max_torque=2.0,
                              ib_method=:original, actuation_mode=:prescribed)

    @test system.n_states == 9
    @test system.n_control_inputs == 1

    state_k = 0.05 .* randn(system.n_states)
    state_kp1 = state_k .+ 0.01 .* randn(system.n_states)
    control_input = [0.5]

    residual = calculate_solid_dynamics_residual(system, state_kp1, state_k, control_input)
    @test length(residual) == system.n_states
    @test all(isfinite, residual)

    J_kp1, J_k, J_u, J_p = calculate_solid_dynamics_jacobian(system, state_kp1, state_k, control_input)
    @test size(J_kp1) == (system.n_states, system.n_states)
    @test size(J_k) == (system.n_states, system.n_states)
    @test size(J_u) == (system.n_states, 1)
    @test size(J_p) == (system.n_states, length(collect_differentiable_params(system)))

    J_kp1_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, s, state_k, control_input), state_kp1)
    @test Matrix(J_kp1) ≈ J_kp1_fd rtol=1e-5

    J_k_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, state_kp1, s, control_input), state_k)
    @test Matrix(J_k) ≈ J_k_fd rtol=1e-5

    J_u_fd = FiniteDiff.finite_difference_jacobian(
        u -> calculate_solid_dynamics_residual(system, state_kp1, state_k, u), control_input)
    @test Matrix(J_u) ≈ J_u_fd rtol=1e-5
    @test J_u[9, 1] ≈ -1.0 atol=1e-10   # O(1) gradient, not O(Kp)

    p0 = collect_differentiable_params(system)
    J_p_fd = FiniteDiff.finite_difference_jacobian(p0) do p
        new_system = inject_differentiable_params(system, p)
        calculate_solid_dynamics_residual(new_system, state_kp1, state_k, control_input)
    end
    @test Matrix(J_p) ≈ J_p_fd rtol=1e-5
end

@testitem "Prescribed RExEel dynamics residual and Jacobian" begin
    using AquariumClosed
    using ForwardDiff
    using FiniteDiff
    using Random

    Random.seed!(43)

    n_links = 3
    system = RExEel(0.01, n_links;
        bar_lengths=ones(n_links), masses=ones(n_links),
        mois=fill(0.1, n_links),
        Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
        max_torques=fill(2.0, n_links-1), actuation_mode=:prescribed)

    n_joints = n_links - 1
    @test system.n_control_inputs == n_joints
    @test system.n_constraints == 2 * n_joints + n_joints

    state_k = 0.05 .* randn(system.n_states)
    state_kp1 = state_k .+ 0.01 .* randn(system.n_states)
    control_input = 0.1 .* randn(n_joints)

    residual = calculate_solid_dynamics_residual(system, state_kp1, state_k, control_input)
    @test length(residual) == system.n_states
    @test all(isfinite, residual)

    J_kp1, J_k, J_u, J_p = calculate_solid_dynamics_jacobian(system, state_kp1, state_k, control_input)

    J_kp1_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, s, state_k, control_input), state_kp1)
    @test Matrix(J_kp1) ≈ J_kp1_fd rtol=1e-5

    J_k_fd = FiniteDiff.finite_difference_jacobian(
        s -> calculate_solid_dynamics_residual(system, state_kp1, s, control_input), state_k)
    @test Matrix(J_k) ≈ J_k_fd rtol=1e-5

    J_u_fd = FiniteDiff.finite_difference_jacobian(
        u -> calculate_solid_dynamics_residual(system, state_kp1, state_k, u), control_input)
    @test Matrix(J_u) ≈ J_u_fd rtol=1e-5

    p0 = collect_differentiable_params(system)
    J_p_fd = FiniteDiff.finite_difference_jacobian(p0) do p
        new_system = inject_differentiable_params(system, p)
        calculate_solid_dynamics_residual(new_system, state_kp1, state_k, control_input)
    end
    @test Matrix(J_p) ≈ J_p_fd rtol=1e-5
end

@testitem "PE Hessian analytical helpers" begin
    using AquariumClosed
    using ForwardDiff

    @testset "WorldPinJoint (ActuatedPendulum)" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=50.0, Kd=5.0, max_torque=2.0)
        config = [0.5, 0.0, 0.3]
        H = calculate_potential_energy_hessian(ap, config)
        H_fd = ForwardDiff.jacobian(q -> calculate_potential_energy_gradient(ap, q), config)
        @test H ≈ H_fd atol=1e-12
    end

    @testset "PinJoint (RExEel)" begin
        n_links = 3
        rex = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(2.0, n_links-1))
        config = [1.0, 0.0, 0.1, 2.0, 0.0, 0.4, 3.0, 0.0, -0.2]
        H = calculate_potential_energy_hessian(rex, config)
        H_fd = ForwardDiff.jacobian(q -> calculate_potential_energy_gradient(rex, q), config)
        @test H ≈ H_fd atol=1e-12
    end

    @testset "PassiveSystem" begin
        body = RigidBody(Bar(2.0); mass=3.0, moi=1.0)
        joint = WorldPinJoint([0.0, 0.0], 1, :root; stiffness=10.0)
        sys = PassiveSystem(0.01, [body], Joint[joint]; gravity=[0.0, -9.81])
        config = [0.0, 1.0, 0.5]
        H = calculate_potential_energy_hessian(sys, config)
        H_fd = ForwardDiff.jacobian(q -> calculate_potential_energy_gradient(sys, q), config)
        @test H ≈ H_fd atol=1e-12
    end
end

@testitem "Damping force Jacobian analytical helpers" begin
    using AquariumClosed
    using ForwardDiff

    @testset "WorldPinJoint (ActuatedPendulum)" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=50.0, Kd=5.0, max_torque=2.0)
        vel = [0.1, -0.2, 0.8]
        J = calculate_damping_force_jacobian(ap, vel)
        J_fd = ForwardDiff.jacobian(v -> calculate_damping_force(ap, v), vel)
        @test J ≈ J_fd atol=1e-12
    end

    @testset "PinJoint (RExEel)" begin
        n_links = 3
        rex = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(2.0, n_links-1))
        vel = [0.1, -0.1, 0.5, 0.0, 0.2, -0.3, -0.1, 0.0, 0.4]
        J = calculate_damping_force_jacobian(rex, vel)
        J_fd = ForwardDiff.jacobian(v -> calculate_damping_force(rex, v), vel)
        @test J ≈ J_fd atol=1e-12
    end

    @testset "PassiveSystem" begin
        body = RigidBody(Bar(2.0); mass=3.0, moi=1.0)
        joint = WorldPinJoint([0.0, 0.0], 1, :root; damping=0.5)
        sys = PassiveSystem(0.01, [body], Joint[joint]; gravity=[0.0, -9.81])
        vel = [0.0, 0.0, 2.0]
        J = calculate_damping_force_jacobian(sys, vel)
        J_fd = ForwardDiff.jacobian(v -> calculate_damping_force(sys, v), vel)
        @test J ≈ J_fd atol=1e-12
    end
end

@testitem "Actuator force Jacobian analytical helpers" begin
    using AquariumClosed
    using ForwardDiff

    @testset "WorldPinJoint unsaturated (ActuatedPendulum)" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=20.0, Kd=5.0, max_torque=100.0)
        config = [0.5, 0.0, 0.3]
        vel = [0.0, 0.0, 0.1]
        control = [0.5, 0.0]
        J_q, J_v = calculate_actuator_force_jacobians(ap, config, vel, control)
        J_q_fd = ForwardDiff.jacobian(
            q -> calculate_actuator_forces(ap, [q; vel], control), config)
        J_v_fd = ForwardDiff.jacobian(
            v -> calculate_actuator_forces(ap, [config; v], control), vel)
        @test J_q ≈ J_q_fd atol=1e-12
        @test J_v ≈ J_v_fd atol=1e-12
    end

    @testset "WorldPinJoint saturated" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=100.0, Kd=5.0, max_torque=2.0)
        config = [0.5, 0.0, -1.0]
        vel = [0.0, 0.0, 0.0]
        control = [1.0, 0.0]
        J_q, J_v = calculate_actuator_force_jacobians(ap, config, vel, control)
        J_q_fd = ForwardDiff.jacobian(
            q -> calculate_actuator_forces(ap, [q; vel], control), config)
        J_v_fd = ForwardDiff.jacobian(
            v -> calculate_actuator_forces(ap, [config; v], control), vel)
        @test J_q ≈ J_q_fd atol=1e-12
        @test J_v ≈ J_v_fd atol=1e-12
        @test all(J_q .== 0)
        @test all(J_v .== 0)
    end

    @testset "WorldPinJoint boundary (τ == max_torque)" begin
        Kp = 20.0; max_t = 2.0
        θ_desired = 0.0; θ_current = -max_t / Kp
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=Kp, Kd=0.0, max_torque=max_t)
        config = [0.0, 0.0, θ_current]
        vel = zeros(3)
        control = [θ_desired, 0.0]
        J_q, J_v = calculate_actuator_force_jacobians(ap, config, vel, control)
        J_q_fd = ForwardDiff.jacobian(
            q -> calculate_actuator_forces(ap, [q; vel], control), config)
        @test J_q ≈ J_q_fd atol=1e-12
        @test J_q[3, 3] ≈ -Kp atol=1e-12
    end

    @testset "PinJoint (RExEel)" begin
        n_links = 3
        rex = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(100.0, n_links-1))
        config = [1.0, 0.0, 0.1, 2.0, 0.0, 0.4, 3.0, 0.0, -0.2]
        vel = [0.0, 0.0, 0.5, 0.0, 0.0, -0.3, 0.0, 0.0, 0.2]
        control = [0.2, 0.0, -0.5, 0.0]
        J_q, J_v = calculate_actuator_force_jacobians(rex, config, vel, control)
        J_q_fd = ForwardDiff.jacobian(
            q -> calculate_actuator_forces(rex, [q; vel], control), config)
        J_v_fd = ForwardDiff.jacobian(
            v -> calculate_actuator_forces(rex, [config; v], control), vel)
        @test J_q ≈ J_q_fd atol=1e-12
        @test J_v ≈ J_v_fd atol=1e-12
    end

    @testset "PassiveSystem returns zeros" begin
        body = RigidBody(Bar(2.0); mass=3.0, moi=1.0)
        joint = WorldPinJoint([0.0, 0.0], 1, :root; damping=0.5)
        sys = PassiveSystem(0.01, [body], Joint[joint])
        config = [0.0, 1.0, 0.5]
        vel = [0.0, 0.0, 1.0]
        J_q, J_v = calculate_actuator_force_jacobians(sys, config, vel)
        @test all(J_q .== 0)
        @test all(J_v .== 0)
    end
end

@testitem "Forward joint constraint Jacobians" begin
    using AquariumClosed
    using ForwardDiff

    @testset "WorldPinJoint (ActuatedPendulum)" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=50.0, Kd=5.0, max_torque=2.0)
        config = [0.5, 0.0, 0.3]
        J = calculate_system_constraint_jacobian(ap, config)
        J_fd = ForwardDiff.jacobian(q -> calculate_system_constraint_residual(ap, q), config)
        @test J ≈ J_fd atol=1e-12
    end

    @testset "PinJoint (RExEel)" begin
        n_links = 3
        rex = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(2.0, n_links-1))
        config = [1.0, 0.0, 0.1, 2.0, 0.0, 0.4, 3.0, 0.0, -0.2]
        J = calculate_system_constraint_jacobian(rex, config)
        J_fd = ForwardDiff.jacobian(q -> calculate_system_constraint_residual(rex, q), config)
        @test J ≈ J_fd atol=1e-12
    end

    @testset "PassiveSystem two-body" begin
        b1 = RigidBody(Bar(2.0); mass=1.0, moi=0.1)
        b2 = RigidBody(Bar(2.0); mass=1.0, moi=0.1)
        jw = WorldPinJoint([0.0, 0.0], 1, :root)
        jp = PinJoint(1, :tip, 2, :root)
        sys = PassiveSystem(0.01, [b1, b2], Joint[jw, jp])
        config = [1.0, 0.0, 0.3, 3.0, 0.0, -0.1]
        J = calculate_system_constraint_jacobian(sys, config)
        J_fd = ForwardDiff.jacobian(q -> calculate_system_constraint_residual(sys, q), config)
        @test J ≈ J_fd atol=1e-12
    end

    @testset "VJP consistency: J^T * λ matches VJP" begin
        n_links = 3
        rex = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(2.0, n_links-1))
        config = [1.0, 0.0, 0.1, 2.0, 0.0, 0.4, 3.0, 0.0, -0.2]
        J = calculate_system_constraint_jacobian(rex, config)
        λ = [1.0, -0.5, 0.3, -0.2]
        vjp_from_J = J' * λ
        vjp_direct = zeros(9)
        offset = 0
        for joint in rex.joints
            n_c = joint_n_constraints(joint)
            slice = λ[offset+1:offset+n_c]
            _add_joint_constraint_vjp!(vjp_direct, joint, config, slice, rex.bodies)
            offset += n_c
        end
        @test vjp_from_J ≈ vjp_direct atol=1e-12
    end
end

@testitem "Prescribed angle constraint Jacobians" begin
    using AquariumClosed
    using ForwardDiff

    @testset "WorldPinJoint (ActuatedPendulum)" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0], actuation_mode=:prescribed)
        config = [0.5, 0.0, 0.3]
        J = calculate_prescribed_angle_constraint_jacobian(ap, config)
        J_fd = ForwardDiff.jacobian(
            q -> calculate_prescribed_angle_constraint_residual(ap, q, [0.5]), config)
        @test J ≈ J_fd atol=1e-12
    end

    @testset "PinJoint (RExEel)" begin
        n_links = 3
        rex = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(2.0, n_links-1), actuation_mode=:prescribed)
        config = [1.0, 0.0, 0.1, 2.0, 0.0, 0.4, 3.0, 0.0, -0.2]
        J = calculate_prescribed_angle_constraint_jacobian(rex, config)
        J_fd = ForwardDiff.jacobian(
            q -> calculate_prescribed_angle_constraint_residual(rex, q, [0.2, -0.5]), config)
        @test J ≈ J_fd atol=1e-12
    end

    @testset "VJP consistency: J^T * λ matches prescribed VJP" begin
        ap = ActuatedPendulum(0.01; bar_length=1.0, mass=1.0, moi=0.1,
            hinge_position=[0.0, 0.0], actuation_mode=:prescribed)
        config = [0.5, 0.0, 0.3]
        J = calculate_prescribed_angle_constraint_jacobian(ap, config)
        λ = [2.5]
        vjp_from_J = J' * λ
        vjp_direct = zeros(3)
        _add_prescribed_angle_constraint_vjp!(vjp_direct, ap, config, λ)
        @test vjp_from_J ≈ vjp_direct atol=1e-12
    end
end

@testitem "Analytical stationarity Jacobian ∂_∂kp1" begin
    using AquariumClosed
    using ForwardDiff
    using Random

    function fd_stationarity_kp1(sys, state_kp1, state_k, control)
        ForwardDiff.jacobian(
            x -> calculate_solid_stationarity_residual(sys, x, state_k, control),
            state_kp1,
        )
    end

    @testset "PassiveSystem (Pendulum)" begin
        Random.seed!(50)
        sys = Pendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], n_boundary_nodes=4, ib_method=:original)
        state_k = 0.05 .* randn(sys.n_states)
        state_kp1 = state_k .+ 0.01 .* randn(sys.n_states)
        J_kp1, _, _, _ = calculate_solid_stationarity_jacobian(sys, state_kp1, state_k)
        J_fd = fd_stationarity_kp1(sys, state_kp1, state_k, zeros(0))
        @test J_kp1 ≈ J_fd atol=1e-12
    end

    @testset "ActuatedSystem :pd (ActuatedPendulum)" begin
        Random.seed!(51)
        sys = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=20.0, Kd=5.0, max_torque=100.0,
            n_boundary_nodes=4, ib_method=:original)
        state_k = 0.05 .* randn(sys.n_states)
        state_kp1 = state_k .+ 0.01 .* randn(sys.n_states)
        control = [0.5, 0.1]
        J_kp1, _, _, _ = calculate_solid_stationarity_jacobian(sys, state_kp1, state_k, control)
        J_fd = fd_stationarity_kp1(sys, state_kp1, state_k, control)
        @test J_kp1 ≈ J_fd atol=1e-12
    end

    @testset "ActuatedSystem :prescribed (ActuatedPendulum)" begin
        Random.seed!(52)
        sys = ActuatedPendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
            hinge_position=[0.0, 0.0], Kp=20.0, Kd=5.0, max_torque=2.0,
            n_boundary_nodes=4, ib_method=:original, actuation_mode=:prescribed)
        state_k = 0.05 .* randn(sys.n_states)
        state_kp1 = state_k .+ 0.01 .* randn(sys.n_states)
        control = [0.5]
        J_kp1, _, _, _ = calculate_solid_stationarity_jacobian(sys, state_kp1, state_k, control)
        J_fd = fd_stationarity_kp1(sys, state_kp1, state_k, control)
        @test J_kp1 ≈ J_fd atol=1e-12
    end

    @testset "Multi-link :pd (RExEel)" begin
        Random.seed!(53)
        n_links = 3
        sys = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(100.0, n_links-1))
        state_k = 0.05 .* randn(sys.n_states)
        state_kp1 = state_k .+ 0.01 .* randn(sys.n_states)
        control = 0.1 .* randn(2 * (n_links - 1))
        J_kp1, _, _, _ = calculate_solid_stationarity_jacobian(sys, state_kp1, state_k, control)
        J_fd = fd_stationarity_kp1(sys, state_kp1, state_k, control)
        @test J_kp1 ≈ J_fd atol=1e-12
    end

    @testset "Multi-link :prescribed (RExEel)" begin
        Random.seed!(54)
        n_links = 3
        sys = RExEel(0.01, n_links; bar_lengths=ones(n_links), masses=ones(n_links),
            mois=fill(0.1, n_links), Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(2.0, n_links-1), actuation_mode=:prescribed)
        state_k = 0.05 .* randn(sys.n_states)
        state_kp1 = state_k .+ 0.01 .* randn(sys.n_states)
        control = 0.1 .* randn(n_links - 1)
        J_kp1, _, _, _ = calculate_solid_stationarity_jacobian(sys, state_kp1, state_k, control)
        J_fd = fd_stationarity_kp1(sys, state_kp1, state_k, control)
        @test J_kp1 ≈ J_fd atol=1e-12
    end
end
