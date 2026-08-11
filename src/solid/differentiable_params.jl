#############################################################################################
## collect_differentiable_params / inject_differentiable_params
##
## Walks a PassiveSystem's body and joint lists in deterministic order and either extracts
## the flagged components' differentiable parameters into a flat `Vector{Float64}`, or
## reconstructs the system with new values pulled from a parameter vector (potentially
## containing `ForwardDiff.Dual` values).
##
## Traversal order: bodies in list order, then joints in list order. Within each body:
## mass, moi, com_offset[1], com_offset[2], then shape-specific params. Within each joint:
## joint-type-specific params (see `collect_joint_diff_params`).
#############################################################################################

# --- Shape-level helpers ------------------------------------------------------------------

collect_shape_diff_params(shape::Bar) = [shape.length]
n_shape_diff_params(::Bar) = 1

function reconstruct_shape(::Bar, params_slice::AbstractVector)
    S = eltype(params_slice)
    return Bar{S}(params_slice[1])
end

collect_shape_diff_params(shape::Disc) = [shape.radius]
n_shape_diff_params(::Disc) = 1

function reconstruct_shape(::Disc, params_slice::AbstractVector)
    S = eltype(params_slice)
    return Disc{S}(params_slice[1])
end

# --- Body-level helpers -------------------------------------------------------------------

function n_body_diff_params(body::RigidBody)
    return 4 + n_shape_diff_params(body.shape)   # mass, moi, com_x, com_y + shape
end

function collect_body_diff_params(body::RigidBody)
    out = [body.mass, body.moi, body.com_offset[1], body.com_offset[2]]
    append!(out, collect_shape_diff_params(body.shape))
    return out
end

function reconstruct_body(body::RigidBody, params_slice::AbstractVector)
    S = eltype(params_slice)
    mass = params_slice[1]
    moi = params_slice[2]
    com_offset = S[params_slice[3], params_slice[4]]
    new_shape = reconstruct_shape(body.shape, @view params_slice[5:end])
    Sh = typeof(new_shape)
    return RigidBody{Sh, S}(
        new_shape, mass, moi, com_offset,
        body.n_boundary_nodes, body.ib_method, body.discrete_delta_kind,
    )
end

# --- Joint-level helpers ------------------------------------------------------------------

n_joint_diff_params(::PinJoint) = 3   # equilibrium_angle, stiffness, damping
collect_joint_diff_params(j::PinJoint) = [j.equilibrium_angle, j.stiffness, j.damping]

function reconstruct_joint(j::PinJoint, params_slice::AbstractVector)
    S = eltype(params_slice)
    return PinJoint{S}(
        j.body_id_A, j.role_A, j.body_id_B, j.role_B,
        params_slice[1], params_slice[2], params_slice[3],
    )
end

n_joint_diff_params(::WorldPinJoint) = 5   # world_x, world_y, equilibrium, stiffness, damping
collect_joint_diff_params(j::WorldPinJoint) = [
    j.world_position[1], j.world_position[2],
    j.equilibrium_angle, j.stiffness, j.damping,
]

function reconstruct_joint(j::WorldPinJoint, params_slice::AbstractVector)
    S = eltype(params_slice)
    world_pos = S[params_slice[1], params_slice[2]]
    return WorldPinJoint{S}(
        world_pos, j.body_id, j.role,
        params_slice[3], params_slice[4], params_slice[5],
    )
end

# --- Top-level collect / inject -----------------------------------------------------------

# Generic counting helper — default counts via the flat vector length. Specific
# systems can override for perf (e.g. NoSystem returns 0 directly).
n_differentiable_params(system::SolidSystem) = length(collect_differentiable_params(system))

# --- Actuator helpers --------------------------------------------------------------------

n_actuator_diff_params(::Actuator) = 0   # default: actuator contributes no gradients

collect_actuator_diff_params(::Actuator) = Float64[]

function reconstruct_actuator(a::Actuator, ::AbstractVector)
    return a
end

# --- Top-level collect / inject -----------------------------------------------------------

function _collect_bodies_joints(bodies::AbstractVector, joints::AbstractVector)
    parts = []
    for body in bodies
        if body isa RigidBody
            push!(parts, collect_body_diff_params(body))
        end
    end
    for joint in joints
        push!(parts, collect_joint_diff_params(joint))
    end
    return isempty(parts) ? Float64[] : vcat(parts...)
end

function collect_differentiable_params(system::PassiveSystem)
    return _collect_bodies_joints(system.bodies, system.joints)
end

function collect_differentiable_params(system::ActuatedSystem)
    base = _collect_bodies_joints(system.bodies, system.joints)
    actuator_parts = [collect_actuator_diff_params(a) for a in system.actuators]
    all_parts = vcat(base, actuator_parts...)
    return all_parts
end

function _inject_bodies_joints(bodies::AbstractVector, joints::AbstractVector,
    params_vec::AbstractVector, idx::Int)
    new_bodies = Vector{AbstractRigidBody}(undef, length(bodies))
    for (i, body) in enumerate(bodies)
        if body isa RigidBody
            n = n_body_diff_params(body)
            new_bodies[i] = reconstruct_body(body, @view params_vec[idx:idx+n-1])
            idx += n
        else
            new_bodies[i] = body
        end
    end

    new_joints = Vector{Joint}(undef, length(joints))
    for (k, joint) in enumerate(joints)
        n = n_joint_diff_params(joint)
        new_joints[k] = reconstruct_joint(joint, @view params_vec[idx:idx+n-1])
        idx += n
    end

    return new_bodies, new_joints, idx
end

function inject_differentiable_params(system::PassiveSystem, params_vec::AbstractVector)
    new_bodies, new_joints, _ = _inject_bodies_joints(
        system.bodies, system.joints, params_vec, 1)

    return PassiveSystem(
        new_bodies, new_joints,
        system.n_bodies, system.n_configurations, system.n_velocities,
        system.n_constraints, system.n_body_states, system.n_states,
        system.state_indices, system.configuration_indices, system.velocity_indices,
        system.body_state_indices, system.dual_indices,
        system.time_step, system.gravity, system.plot_params,
        system.topology,
    )
end

function inject_differentiable_params(system::ActuatedSystem, params_vec::AbstractVector)
    new_bodies, new_joints, idx = _inject_bodies_joints(
        system.bodies, system.joints, params_vec, 1)

    new_actuators = Vector{Actuator}(undef, length(system.actuators))
    for (a_idx, a) in enumerate(system.actuators)
        n = n_actuator_diff_params(a)
        new_actuators[a_idx] = reconstruct_actuator(a, @view params_vec[idx:idx+n-1])
        idx += n
    end

    return ActuatedSystem(
        new_bodies, new_joints, new_actuators,
        system.n_bodies, system.n_configurations, system.n_velocities,
        system.n_constraints, system.n_body_states, system.n_states,
        system.n_actuators, system.n_control_inputs,
        system.state_indices, system.configuration_indices, system.velocity_indices,
        system.body_state_indices, system.dual_indices,
        system.time_step, system.gravity, system.plot_params,
        system.topology,
        system.actuation_mode, system.prescribed_angle_dual_indices,
    )
end


@testitem "Differentiable params" begin
    using Aquarium
    using ForwardDiff

    # One body + one WorldPinJoint + one PinJoint with a second body.
    b1 = RigidBody(Bar(1.5); mass=2.0, moi=0.3, com_offset=[0.1, -0.1])
    b2 = RigidBody(Bar(1.0); mass=1.0, moi=0.2, com_offset=[0.0, 0.0])
    jw = WorldPinJoint([0.5, 1.0], 1, :root;
        equilibrium_angle=0.0, stiffness=3.0, damping=0.4)
    jp = PinJoint(1, :tip, 2, :root;
        equilibrium_angle=0.2, stiffness=1.5, damping=0.1)
    sys = PassiveSystem(0.01, [b1, b2], Joint[jw, jp])

    @testset "collect: ordering and count" begin
        p = collect_differentiable_params(sys)
        # Bodies: per-body 5 values = mass, moi, com_x, com_y, length
        # b1: [2.0, 0.3, 0.1, -0.1, 1.5]
        # b2: [1.0, 0.2, 0.0, 0.0, 1.0]
        # WorldPinJoint jw: [0.5, 1.0, 0.0, 3.0, 0.4]  (world_x, world_y, equil, k, b)
        # PinJoint jp: [0.2, 1.5, 0.1]  (equil, k, b)
        expected = [2.0, 0.3, 0.1, -0.1, 1.5,
                    1.0, 0.2, 0.0,  0.0, 1.0,
                    0.5, 1.0, 0.0,  3.0, 0.4,
                    0.2, 1.5, 0.1]
        @test p == expected
    end

    @testset "inject roundtrip" begin
        p = collect_differentiable_params(sys)
        sys2 = inject_differentiable_params(sys, p)
        @test collect_differentiable_params(sys2) == p

        # Dimensions and indices unchanged
        @test sys2.n_bodies == sys.n_bodies
        @test sys2.n_constraints == sys.n_constraints
        @test sys2.configuration_indices == sys.configuration_indices
    end

    @testset "inject changes values" begin
        p = collect_differentiable_params(sys)
        p_new = copy(p)
        p_new[1] = 99.0          # body 1 mass
        p_new[11] = 7.0          # WorldPinJoint world_x
        sys2 = inject_differentiable_params(sys, p_new)

        @test sys2.bodies[1].mass == 99.0
        @test sys2.bodies[2].mass == 1.0
        jw2 = sys2.joints[1]
        @test jw2.world_position[1] == 7.0
    end

    @testset "ForwardDiff gradient through potential energy" begin
        # Tiny 1-body system where PE depends on params in a closed-form way.
        body = RigidBody(Bar(1.0); mass=2.0, moi=0.1, com_offset=[0.0, 0.0])
        joint = WorldPinJoint([0.0, 0.0], 1, :root;
            equilibrium_angle=0.0, stiffness=0.0, damping=0.0)
        small_sys = PassiveSystem(0.01, [body], Joint[joint])

        config = [0.0, 2.0, 0.0]  # height 2.0

        f = p -> calculate_potential_energy(
            inject_differentiable_params(small_sys, p),
            config
        )
        p = collect_differentiable_params(small_sys)
        grad = ForwardDiff.gradient(f, p)

        # Param order: body [mass, moi, com_x, com_y, length] then joint [wx, wy, equil, k, b]
        # Under Interpretation P: PE = -mass * g_y * config[2] = -2 * (-9.81) * 2 = 39.24
        # Joint PE contribution = (k/2) * (θ - equil)^2 = 0 * 0 / 2 = 0
        @test grad[1] ≈ 19.62 atol=1e-10    # mass
        @test grad[2] ≈ 0.0  atol=1e-10     # moi
        @test grad[3] ≈ 0.0  atol=1e-10     # com_x
        @test grad[4] ≈ 0.0  atol=1e-10     # com_y
        @test grad[5] ≈ 0.0  atol=1e-10     # length
        # Joint params — all zero because k=0 and Δ=0
        @test grad[6]  ≈ 0.0 atol=1e-10     # world_x
        @test grad[7]  ≈ 0.0 atol=1e-10     # world_y
        @test grad[8]  ≈ 0.0 atol=1e-10     # equilibrium_angle
        @test grad[9]  ≈ 0.0 atol=1e-10     # stiffness
        @test grad[10] ≈ 0.0 atol=1e-10     # damping
    end
end
