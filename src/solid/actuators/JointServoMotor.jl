#############################################################################################
## 2D JointServoMotor — a PD-controlled servo that drives a specific joint toward a
## desired (angle, angular velocity) setpoint. Replaces the legacy XC330M288T and
## ServoMotorPD actuator types in the new architecture.
##
## The `joint_id` field indexes into the host ActuatedSystem's `joints::Vector{Joint}`.
## At dynamics time we look up the actual joint struct via `system.joints[joint_id]`,
## which automatically picks up any reconstruction done by `inject_differentiable_params`.
#############################################################################################

struct JointServoMotor{S} <: Actuator
    joint_id::Int
    controller::PDController
    max_torque::S
end

function JointServoMotor(joint_id::Int, controller::PDController;
    max_torque::Real = Inf,
)
    return JointServoMotor{Float64}(joint_id, controller,
        convert(Float64, max_torque))
end

n_control_inputs_per_actuator(::JointServoMotor) = 2    # [θ_desired, ω_desired]

# --- State extraction helpers (dispatch on joint type) --------------------------------------

function current_joint_state(joint::PinJoint, system_state::AbstractVector, n_bodies::Int)
    θ_A = system_state[3 * joint.body_id_A]
    θ_B = system_state[3 * joint.body_id_B]
    ω_A = system_state[3 * n_bodies + 3 * joint.body_id_A]
    ω_B = system_state[3 * n_bodies + 3 * joint.body_id_B]
    return (θ_B - θ_A, ω_B - ω_A)
end

function current_joint_state(joint::WorldPinJoint, system_state::AbstractVector, n_bodies::Int)
    θ = system_state[3 * joint.body_id]
    ω = system_state[3 * n_bodies + 3 * joint.body_id]
    return (θ, ω)
end

function apply_joint_torque(joint::PinJoint, τ, n_bodies::Int, T::Type)
    f = zeros(T, 3 * n_bodies)
    f[3 * joint.body_id_A] = -τ
    f[3 * joint.body_id_B] = τ
    return f
end

function apply_joint_torque(joint::WorldPinJoint, τ, n_bodies::Int, T::Type)
    f = zeros(T, 3 * n_bodies)
    f[3 * joint.body_id] = τ
    return f
end

# --- Main actuator force computation --------------------------------------------------------

function calculate_new_actuator_force(
    motor::JointServoMotor,
    system_state::AbstractVector,
    control_slice::AbstractVector,
    system::ActuatedSystem,
)
    joint = system.joints[motor.joint_id]
    n_bodies = system.n_bodies
    T = promote_type(eltype(system_state), eltype(control_slice))

    θ_desired = control_slice[1]
    ω_desired = control_slice[2]
    θ_current, ω_current = current_joint_state(joint, system_state, n_bodies)

    τ = calculate_control_output(motor.controller, θ_desired, ω_desired, θ_current, ω_current)
    τ = clamp(τ, -motor.max_torque, motor.max_torque)

    return apply_joint_torque(joint, τ, n_bodies, T)
end

function calculate_new_actuator_forces(
    system::ActuatedSystem,
    system_state::AbstractVector,
    control::AbstractVector,
)
    T = promote_type(eltype(system_state), eltype(control))
    force = zeros(T, 3 * system.n_bodies)
    offset = 0
    for actuator in system.actuators
        n = n_control_inputs_per_actuator(actuator)
        slice = @view control[offset+1:offset+n]
        force .+= calculate_new_actuator_force(actuator, system_state, slice, system)
        offset += n
    end
    return force
end

#############################################################################################
## Dynamixel XC330-M288-T gain conversion helper.
## Converts encoder-resolution-based gains into effective Kp/Kd values for PDController.
## Reproduces the conversion logic from the legacy XC330M288T actuator.
#############################################################################################

function xc330m288t_gains(;
    Kp_raw::Real = 1100,
    Kd_raw::Real = 500,
    stall_torque::Real = 9.3e6,          # dyn⋅cm (0.93 Nm)
    encoder_resolution::Int = 4096,
    control_loop_time::Real = 0.001,
)
    pwm_to_torque = stall_torque / 885
    Kp = Kp_raw * (encoder_resolution / 2π) * pwm_to_torque / 128
    Kd = Kd_raw * control_loop_time * (encoder_resolution / 2π) * pwm_to_torque / 16
    return Kp, Kd
end
