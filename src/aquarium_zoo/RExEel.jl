#############################################################################################
## RExEel constructor function (new composition-based architecture).
##
## Returns an ActuatedSystem with `n_links` bar bodies connected by (n_links − 1) PinJoints,
## each driven by a JointServoMotor.
#############################################################################################

function RExEel(time_step::Real, n_links::Int;
    bar_lengths::AbstractVector = ones(n_links),
    masses::AbstractVector = ones(n_links),
    mois::AbstractVector = fill(1/12, n_links),
    com_offsets::AbstractVector = [[0.0, 0.0] for _ in 1:n_links],
    equilibrium_angles::AbstractVector = zeros(n_links - 1),
    stiffnesses::AbstractVector = zeros(n_links - 1),
    dampings::AbstractVector = zeros(n_links - 1),
    Kps::AbstractVector = fill(1100.0, n_links - 1),
    Kds::AbstractVector = fill(500.0, n_links - 1),
    max_torques::AbstractVector = fill(Inf, n_links - 1),
    stall_torques::AbstractVector = fill(9.3e6, n_links - 1),
    n_boundary_nodes_per_link::Union{Int, AbstractVector{Int}} = 16,
    ib_method::Symbol = :weak_form,
    discrete_delta_kind::Symbol = :one_point,
    gravity::AbstractVector = [0.0, -9.81],
    plot_params::Dict{Symbol, Any} = default_plot_params(),
    actuation_mode::Symbol = :prescribed,
)
    n_links >= 2 || error("RExEel: n_links must be >= 2 (need at least one actuated joint)")
    length(bar_lengths)        == n_links     || error("RExEel: bar_lengths must have n_links entries")
    length(masses)             == n_links     || error("RExEel: masses must have n_links entries")
    length(mois)               == n_links     || error("RExEel: mois must have n_links entries")
    length(com_offsets)        == n_links     || error("RExEel: com_offsets must have n_links entries")
    length(equilibrium_angles) == n_links - 1 || error("RExEel: equilibrium_angles must have n_links - 1 entries")
    length(stiffnesses)        == n_links - 1 || error("RExEel: stiffnesses must have n_links - 1 entries")
    length(dampings)           == n_links - 1 || error("RExEel: dampings must have n_links - 1 entries")
    length(Kps)                == n_links - 1 || error("RExEel: Kps must have n_links - 1 entries")
    length(Kds)                == n_links - 1 || error("RExEel: Kds must have n_links - 1 entries")
    length(max_torques)        == n_links - 1 || error("RExEel: max_torques must have n_links - 1 entries")
    length(stall_torques)      == n_links - 1 || error("RExEel: stall_torques must have n_links - 1 entries")

    nbn_per_link = n_boundary_nodes_per_link isa Int ?
        fill(n_boundary_nodes_per_link, n_links) : n_boundary_nodes_per_link
    length(nbn_per_link) == n_links || error("RExEel: n_boundary_nodes_per_link must have n_links entries")

    bodies = [
        RigidBody(Bar(bar_lengths[i]);
            mass = masses[i],
            moi = mois[i],
            com_offset = com_offsets[i],
            n_boundary_nodes = nbn_per_link[i],
            ib_method = ib_method,
            discrete_delta_kind = discrete_delta_kind,
        )
        for i in 1:n_links
    ]

    joints = Joint[
        PinJoint(i, :tip, i + 1, :root;
            equilibrium_angle = equilibrium_angles[i],
            stiffness = stiffnesses[i],
            damping = dampings[i],
        )
        for i in 1:(n_links - 1)
    ]

    # Convert raw encoder gains to effective PD gains (reproduces legacy XC330M288T conversion)
    effective_gains = [xc330m288t_gains(; Kp_raw=Kps[i], Kd_raw=Kds[i], stall_torque=stall_torques[i]) for i in 1:(n_links - 1)]

    actuators = Actuator[
        JointServoMotor(
            i,
            PDController(effective_gains[i][1], effective_gains[i][2];
                output_min = -max_torques[i], output_max = max_torques[i]);
            max_torque = max_torques[i],
        )
        for i in 1:(n_links - 1)
    ]

    return ActuatedSystem(time_step, bodies, joints, actuators;
        gravity = gravity,
        plot_params = plot_params,
        actuation_mode = actuation_mode,
    )
end

# Convert minimal coordinates [x1, y1, θ1, θ2, ..., θ_n] to maximal for RExEel.
function rex_eel_maximal_from_minimal(rex::ActuatedSystem, q_min::AbstractVector, n_links::Int)
    length(q_min) == n_links + 2 ||
        error("rex_eel_maximal_from_minimal: expected q_min of length $(n_links+2), got $(length(q_min))")

    x1, y1 = q_min[1], q_min[2]
    θ1 = q_min[3]

    # Promote element type to handle Dual-typed system params (e.g. from ForwardDiff)
    T = promote_type(eltype(q_min), typeof(rex.bodies[1].shape.length))
    maximal = Vector{T}(undef, 3 * n_links)
    maximal[1], maximal[2], maximal[3] = x1, y1, θ1

    for i in 2:n_links
        prev_body = rex.bodies[i - 1]
        curr_body = rex.bodies[i]
        joint = rex.joints[i - 1]
        joint isa PinJoint || error("rex_eel_maximal_from_minimal: joint $(i-1) must be PinJoint")

        prev_center = [maximal[3 * (i - 2) + 1], maximal[3 * (i - 2) + 2]]
        θ_prev = maximal[3 * (i - 2) + 3]
        R_prev = rotation_2d(θ_prev)

        tip_local = prev_body.com_offset .+ local_attachment_point(prev_body.shape, joint.role_A)
        tip_world = prev_center .+ R_prev * tip_local

        θ_curr = θ_prev + q_min[i + 2]
        R_curr = rotation_2d(θ_curr)
        root_local = curr_body.com_offset .+ local_attachment_point(curr_body.shape, joint.role_B)
        curr_center = tip_world .- R_curr * root_local

        maximal[3 * (i - 1) + 1] = curr_center[1]
        maximal[3 * (i - 1) + 2] = curr_center[2]
        maximal[3 * (i - 1) + 3] = θ_curr
    end

    return maximal
end