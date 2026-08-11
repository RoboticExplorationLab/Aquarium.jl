#############################################################################################
## Pendulum constructor function (new composition-based architecture).
##
## Returns a PassiveSystem composed of one RigidBody{Bar} hinged to world via a
## WorldPinJoint. Distinct from the legacy positional-args constructor above. Julia
## dispatches between them based on signature: 1 positional arg (kwargs-only) hits this,
## 3 positional args hits the legacy constructor.
#############################################################################################

function Pendulum(time_step::Real;
    bar_length::Real = 1.0,
    mass::Real = 1.0,
    moi::Real = 1/12,
    com_offset::AbstractVector = [0.0, 0.0],
    hinge_position::AbstractVector = [0.0, 0.0],
    equilibrium_angle::Real = 0.0,
    stiffness::Real = 0.0,
    damping::Real = 0.0,
    n_boundary_nodes::Int = 16,
    ib_method::Symbol = :weak_form,
    discrete_delta_kind::Symbol = :one_point,
    gravity::AbstractVector = [0.0, -9.81],
    plot_params::Dict{Symbol, Any} = default_plot_params(),
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
    return PassiveSystem(time_step, [body], Joint[joint];
        gravity = gravity,
        plot_params = plot_params,
    )
end

# Standalone per-system helper: convert minimal coordinates (single angle) to maximal.
# Works for both `Pendulum` (PassiveSystem) and `ActuatedPendulum` (ActuatedSystem)
# since the helper only reads `bodies[1]` / `joints[1]` which are on both.
function pendulum_maximal_from_minimal(pendulum::SolidSystem, q_min::AbstractVector)
    length(q_min) == 1 || error("pendulum_maximal_from_minimal: expected q_min of length 1, got $(length(q_min))")
    θ = q_min[1]
    body = pendulum.bodies[1]
    joint = pendulum.joints[1]
    joint isa WorldPinJoint || error("pendulum_maximal_from_minimal: expected a WorldPinJoint as joint[1]")
    hinge = joint.world_position
    root_local = body.com_offset .+ local_attachment_point(body.shape, joint.role)
    R = rotation_2d(θ)
    body_center = hinge .- R * root_local
    return [body_center[1], body_center[2], θ]
end