#############################################################################################
## DoublePendulum constructor function (new composition-based architecture).
##
## Returns a PassiveSystem with two bar-shaped bodies:
##   - body 1 hinged to world at `hinge_position` (via WorldPinJoint at :root)
##   - body 2 hinged to body 1 (PinJoint from :tip of 1 to :root of 2)
#############################################################################################

function DoublePendulum(time_step::Real;
    bar_lengths::AbstractVector = [1.0, 1.0],
    masses::AbstractVector = [1.0, 1.0],
    mois::AbstractVector = [1/12, 1/12],
    com_offsets::AbstractVector = [[0.0, 0.0], [0.0, 0.0]],
    hinge_position::AbstractVector = [0.0, 0.0],
    equilibrium_angles::AbstractVector = [0.0, 0.0],
    stiffnesses::AbstractVector = [0.0, 0.0],
    dampings::AbstractVector = [0.0, 0.0],
    n_boundary_nodes_per_link::Int = 16,
    ib_method::Symbol = :weak_form,
    discrete_delta_kind::Symbol = :one_point,
    gravity::AbstractVector = [0.0, -9.81],
    plot_params::Dict{Symbol, Any} = default_plot_params(),
)
    length(bar_lengths)       == 2 || error("DoublePendulum: bar_lengths must have 2 entries")
    length(masses)            == 2 || error("DoublePendulum: masses must have 2 entries")
    length(mois)              == 2 || error("DoublePendulum: mois must have 2 entries")
    length(com_offsets)       == 2 || error("DoublePendulum: com_offsets must have 2 entries")
    length(equilibrium_angles) == 2 || error("DoublePendulum: equilibrium_angles must have 2 entries")
    length(stiffnesses)       == 2 || error("DoublePendulum: stiffnesses must have 2 entries")
    length(dampings)          == 2 || error("DoublePendulum: dampings must have 2 entries")

    bodies = [
        RigidBody(Bar(bar_lengths[i]);
            mass = masses[i],
            moi = mois[i],
            com_offset = com_offsets[i],
            n_boundary_nodes = n_boundary_nodes_per_link,
            ib_method = ib_method,
            discrete_delta_kind = discrete_delta_kind,
        )
        for i in 1:2
    ]

    root_joint = WorldPinJoint(hinge_position, 1, :root;
        equilibrium_angle = equilibrium_angles[1],
        stiffness = stiffnesses[1],
        damping = dampings[1],
    )
    link_joint = PinJoint(1, :tip, 2, :root;
        equilibrium_angle = equilibrium_angles[2],
        stiffness = stiffnesses[2],
        damping = dampings[2],
    )

    return PassiveSystem(time_step, bodies, Joint[root_joint, link_joint];
        gravity = gravity,
        plot_params = plot_params,
    )
end

# Convert minimal coordinates [θ1, θ2] (absolute link angles) to maximal coordinates
# [x1, y1, θ1, x2, y2, θ2]. Assumes the chain root is a WorldPinJoint on body 1 and the
# inter-link joint is a PinJoint between body 1's :tip and body 2's :root.
function double_pendulum_maximal_from_minimal(dp::PassiveSystem, q_min::AbstractVector)
    length(q_min) == 2 || error("double_pendulum_maximal_from_minimal: expected q_min of length 2, got $(length(q_min))")
    θ1, θ2 = q_min[1], q_min[2]

    root_joint = dp.joints[1]
    link_joint = dp.joints[2]
    root_joint isa WorldPinJoint || error("double_pendulum_maximal_from_minimal: joints[1] must be WorldPinJoint")
    link_joint isa PinJoint      || error("double_pendulum_maximal_from_minimal: joints[2] must be PinJoint")

    body1 = dp.bodies[1]
    body2 = dp.bodies[2]

    # Body 1: root attaches to world hinge.
    R1 = rotation_2d(θ1)
    root1_local = body1.com_offset .+ local_attachment_point(body1.shape, root_joint.role)
    body1_center = root_joint.world_position .- R1 * root1_local

    # Tip of body 1 in world frame using link_joint.role_A (expected :tip).
    tip1_local = body1.com_offset .+ local_attachment_point(body1.shape, link_joint.role_A)
    tip1_world = body1_center .+ R1 * tip1_local

    # Body 2: its attachment role_B is at tip1_world.
    R2 = rotation_2d(θ2)
    root2_local = body2.com_offset .+ local_attachment_point(body2.shape, link_joint.role_B)
    body2_center = tip1_world .- R2 * root2_local

    return [body1_center[1], body1_center[2], θ1,
            body2_center[1], body2_center[2], θ2]
end