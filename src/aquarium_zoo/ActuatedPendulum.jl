#############################################################################################
## ActuatedPendulum constructor function (new composition-based architecture).
##
## Returns an ActuatedSystem composed of one RigidBody{Bar}, one WorldPinJoint,
## and one JointServoMotor driving that joint. Distinct from the legacy positional-args
## constructor above (Julia dispatches by positional arity: 1 vs 3+).
#############################################################################################

function ActuatedPendulum(time_step::Real;
    bar_length::Real = 1.0,
    mass::Real = 1.0,
    moi::Real = 1/12,
    com_offset::AbstractVector = [0.0, 0.0],
    hinge_position::AbstractVector = [0.0, 0.0],
    equilibrium_angle::Real = 0.0,
    stiffness::Real = 0.0,
    damping::Real = 0.0,
    Kp::Real = 100.0,
    Kd::Real = 10.0,
    max_torque::Real = Inf,
    n_boundary_nodes::Int = 16,
    ib_method::Symbol = :weak_form,
    discrete_delta_kind::Symbol = :one_point,
    gravity::AbstractVector = [0.0, -9.81],
    plot_params::Dict{Symbol, Any} = default_plot_params(),
    actuation_mode::Symbol = :pd,
)
    body = RigidBody(Bar(bar_length);
        mass = mass,
        moi = moi,
        com_offset = com_offset,
        n_boundary_nodes = n_boundary_nodes,
        ib_method = ib_method,
        discrete_delta_kind = discrete_delta_kind,
    )
    joint = WorldPinJoint(hinge_position, 1, :root;
        equilibrium_angle = equilibrium_angle,
        stiffness = stiffness,
        damping = damping,
    )
    controller = PDController(Kp, Kd; output_min = -max_torque, output_max = max_torque)
    motor = JointServoMotor(1, controller;
        max_torque = max_torque,
    )
    return ActuatedSystem(time_step, [body], Joint[joint], Actuator[motor];
        gravity = gravity,
        plot_params = plot_params,
        actuation_mode = actuation_mode,
    )
end