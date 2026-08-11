struct PinJoint{S} <: Joint
    body_id_A::Int
    role_A::Symbol
    body_id_B::Int
    role_B::Symbol
    equilibrium_angle::S
    stiffness::S
    damping::S
end

function PinJoint(body_id_A::Int, role_A::Symbol, body_id_B::Int, role_B::Symbol;
    equilibrium_angle::Real=0.0,
    stiffness::Real=0.0,
    damping::Real=0.0,
)
    S = promote_type(typeof(equilibrium_angle), typeof(stiffness), typeof(damping))
    return PinJoint{S}(body_id_A, role_A, body_id_B, role_B,
        convert(S, equilibrium_angle), convert(S, stiffness), convert(S, damping))
end

joint_n_constraints(::PinJoint) = 2

function calculate_joint_constraint_residual(
    joint::PinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    cfg_A = body_configuration(configuration, joint.body_id_A)
    cfg_B = body_configuration(configuration, joint.body_id_B)
    world_A = body_attachment_point_world(bodies[joint.body_id_A], cfg_A, joint.role_A)
    world_B = body_attachment_point_world(bodies[joint.body_id_B], cfg_B, joint.role_B)
    return world_A .- world_B
end

function calculate_joint_potential_energy(
    joint::PinJoint,
    configuration::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    θ_A = configuration[3 * joint.body_id_A]
    θ_B = configuration[3 * joint.body_id_B]
    Δ = θ_B - θ_A - joint.equilibrium_angle
    return (joint.stiffness / 2) * Δ^2
end

function calculate_joint_damping_force(
    joint::PinJoint,
    velocity::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    ω_A = velocity[3 * joint.body_id_A]
    ω_B = velocity[3 * joint.body_id_B]
    τ = -joint.damping * (ω_B - ω_A)
    n_bodies = length(bodies)
    T = promote_type(eltype(velocity), typeof(joint.damping))
    force = zeros(T, 3 * n_bodies)
    force[3 * joint.body_id_A] = -τ
    force[3 * joint.body_id_B] = τ
    return force
end

# Analytical contribution to the VJP: (∂residual/∂configuration)^T * dual, added into `out`.
# PinJoint residual is `world_A - world_B` (2D). ∂residual/∂body_A's (x, y, θ) is
# `[I; ∂R(θ_A)/∂θ_A * shifted_A]`, and similarly for body_B with a sign flip.
function _add_joint_constraint_vjp!(
    out::AbstractVector,
    joint::PinJoint,
    configuration::AbstractVector,
    dual::AbstractVector,
    bodies::AbstractVector{<:AbstractRigidBody},
)
    # Dual slice for this joint's 2 constraints
    d = dual   # caller passes the per-joint slice
    iA = 3 * (joint.body_id_A - 1)
    iB = 3 * (joint.body_id_B - 1)

    body_A = bodies[joint.body_id_A]
    body_B = bodies[joint.body_id_B]
    θ_A = configuration[iA + 3]
    θ_B = configuration[iB + 3]

    # shifted = com_offset + local_attachment_point(shape, role)
    local_A = body_A isa RigidBody ? (body_A.com_offset .+ local_attachment_point(body_A.shape, joint.role_A)) : [zero(θ_A), zero(θ_A)]
    local_B = body_B isa RigidBody ? (body_B.com_offset .+ local_attachment_point(body_B.shape, joint.role_B)) : [zero(θ_B), zero(θ_B)]

    # ∂R(θ)/∂θ * v = [[-sin θ, -cos θ], [cos θ, -sin θ]] * v
    sA, cA = sin(θ_A), cos(θ_A)
    sB, cB = sin(θ_B), cos(θ_B)
    dRA_v = [-sA * local_A[1] - cA * local_A[2], cA * local_A[1] - sA * local_A[2]]
    dRB_v = [-sB * local_B[1] - cB * local_B[2], cB * local_B[1] - sB * local_B[2]]

    # body_A contribution: +I for (x, y); +dRA_v for θ_A
    out[iA + 1] += d[1]
    out[iA + 2] += d[2]
    out[iA + 3] += d[1] * dRA_v[1] + d[2] * dRA_v[2]

    # body_B contribution: -I for (x, y); -dRB_v for θ_B
    out[iB + 1] += -d[1]
    out[iB + 2] += -d[2]
    out[iB + 3] += -(d[1] * dRB_v[1] + d[2] * dRB_v[2])

    return nothing
end
