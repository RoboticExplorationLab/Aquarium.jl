struct WorldPinJoint{S} <: Joint
    world_position::Vector{S}
    body_id::Int
    role::Symbol
    equilibrium_angle::S
    stiffness::S
    damping::S
end

function WorldPinJoint(world_position::AbstractVector, body_id::Int, role::Symbol;
    equilibrium_angle::Real=0.0,
    stiffness::Real=0.0,
    damping::Real=0.0,
)
    length(world_position) == 2 || error("WorldPinJoint.world_position must be 2D, got length $(length(world_position))")
    S = promote_type(eltype(world_position), typeof(equilibrium_angle), typeof(stiffness), typeof(damping))
    return WorldPinJoint{S}(convert(Vector{S}, world_position), body_id, role,
        convert(S, equilibrium_angle), convert(S, stiffness), convert(S, damping))
end

joint_n_constraints(::WorldPinJoint) = 2

function calculate_joint_constraint_residual(
    joint::WorldPinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    cfg = body_configuration(configuration, joint.body_id)
    world_pt = body_attachment_point_world(bodies[joint.body_id], cfg, joint.role)
    return world_pt .- joint.world_position
end

function calculate_joint_potential_energy(
    joint::WorldPinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    θ = configuration[3 * joint.body_id]
    Δ = θ - joint.equilibrium_angle
    return (joint.stiffness / 2) * Δ^2
end

function calculate_joint_damping_force(
    joint::WorldPinJoint,
    velocity::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    ω = velocity[3 * joint.body_id]
    τ = -joint.damping * ω
    n_bodies = length(bodies)
    T = promote_type(eltype(velocity), typeof(joint.damping))
    force = zeros(T, 3 * n_bodies)
    force[3 * joint.body_id] = τ
    return force
end

# Analytical per-joint constraint VJP. WorldPinJoint residual is
# `body_origin + R(θ) * (com_offset + local) - world_position`.
# ∂/∂(x, y, θ) = [I, ∂R(θ)/∂θ * shifted].
function _add_joint_constraint_vjp!(
    out::AbstractVector,
    joint::WorldPinJoint,
    configuration::AbstractVector,
    dual::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    d = dual
    i = 3 * (joint.body_id - 1)
    body = bodies[joint.body_id]
    θ = configuration[i + 3]

    local_pt = body isa RigidBody ? (body.com_offset .+ local_attachment_point(body.shape, joint.role)) : [zero(θ), zero(θ)]

    s, c = sin(θ), cos(θ)
    dR_v = [-s * local_pt[1] - c * local_pt[2], c * local_pt[1] - s * local_pt[2]]

    out[i + 1] += d[1]
    out[i + 2] += d[2]
    out[i + 3] += d[1] * dR_v[1] + d[2] * dR_v[2]
    return nothing
end
