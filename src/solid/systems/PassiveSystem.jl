struct PassiveSystem <: SolidSystem
    bodies::Vector{AbstractRigidBody}
    joints::Vector{Joint}

    n_bodies::Int
    n_configurations::Int
    n_velocities::Int
    n_constraints::Int
    n_body_states::Int
    n_states::Int

    state_indices::Vector{Int}
    configuration_indices::Vector{Int}
    velocity_indices::Vector{Int}
    body_state_indices::Vector{Int}
    dual_indices::Vector{Int}

    time_step::Float64
    gravity::Vector{Float64}

    plot_params::Dict{Symbol, Any}

    topology::SystemTopology
end

function _validate_bodies_and_joints(
    bodies::AbstractVector{<:AbstractRigidBody},
    joints::AbstractVector{<:Joint},
)
    n_bodies = length(bodies)

    for (i, body) in enumerate(bodies)
        if body isa RigidBody
            body.mass > 0 || error("Body $i has non-positive mass ($(body.mass)); mass must be > 0")
            body.moi > 0 || error("Body $i has non-positive moi ($(body.moi)); moi must be > 0")
        end
    end

    for (k, joint) in enumerate(joints)
        _validate_joint(joint, k, bodies, n_bodies)
    end
    return nothing
end

function _validate_joint(joint::PinJoint, k::Int, bodies, n_bodies::Int)
    1 <= joint.body_id_A <= n_bodies || error("PinJoint $k: body_id_A=$(joint.body_id_A) out of range [1, $n_bodies]")
    1 <= joint.body_id_B <= n_bodies || error("PinJoint $k: body_id_B=$(joint.body_id_B) out of range [1, $n_bodies]")
    body_A = bodies[joint.body_id_A]
    body_B = bodies[joint.body_id_B]
    if body_A isa RigidBody
        joint.role_A in valid_roles(body_A.shape) || error("PinJoint $k: role_A=:$(joint.role_A) not valid for shape $(typeof(body_A.shape)); valid roles: $(valid_roles(body_A.shape))")
    end
    if body_B isa RigidBody
        joint.role_B in valid_roles(body_B.shape) || error("PinJoint $k: role_B=:$(joint.role_B) not valid for shape $(typeof(body_B.shape)); valid roles: $(valid_roles(body_B.shape))")
    end
    return nothing
end

function _validate_joint(joint::WorldPinJoint, k::Int, bodies, n_bodies::Int)
    1 <= joint.body_id <= n_bodies || error("WorldPinJoint $k: body_id=$(joint.body_id) out of range [1, $n_bodies]")
    body = bodies[joint.body_id]
    if body isa RigidBody
        joint.role in valid_roles(body.shape) || error("WorldPinJoint $k: role=:$(joint.role) not valid for shape $(typeof(body.shape)); valid roles: $(valid_roles(body.shape))")
    end
    return nothing
end

function default_plot_params()
    return Dict{Symbol, Any}(
        :bodycolor => :white,
        :linewidth => 1.0,
        :showboundarynodes => false,
        :boundarynodecolor => :white,
        :boundarynodesize => 0.1,
        :showboundaryvelocities => false,
        :arrowcolor => :white,
        :lengthscale => 0.1,
    )
end

function PassiveSystem(
    time_step::Real,
    bodies::AbstractVector{<:AbstractRigidBody},
    joints::AbstractVector{<:Joint};
    gravity::AbstractVector{<:Real} = [0.0, -9.81],
    plot_params::Dict{Symbol, Any} = default_plot_params(),
)
    length(gravity) == 2 || error("PassiveSystem.gravity must be 2D, got length $(length(gravity))")

    _validate_bodies_and_joints(bodies, joints)

    n_bodies = length(bodies)
    n_configurations = 3 * n_bodies
    n_velocities = 3 * n_bodies
    n_constraints = isempty(joints) ? 0 : sum(joint_n_constraints(j) for j in joints)
    n_body_states = n_configurations + n_velocities
    n_states = n_body_states + n_constraints

    state_indices = collect(1:n_states)
    configuration_indices = collect(1:n_configurations)
    velocity_indices = collect((n_configurations+1):(n_configurations+n_velocities))
    body_state_indices = vcat(configuration_indices, velocity_indices)
    dual_indices = collect((n_body_states+1):n_states)

    topology = _compute_system_topology(
        bodies,
        n_configurations, n_velocities, n_body_states, n_states,
        configuration_indices, velocity_indices,
    )

    return PassiveSystem(
        Vector{AbstractRigidBody}(bodies),
        Vector{Joint}(joints),
        n_bodies, n_configurations, n_velocities, n_constraints, n_body_states, n_states,
        state_indices, configuration_indices, velocity_indices, body_state_indices, dual_indices,
        convert(Float64, time_step),
        convert(Vector{Float64}, gravity),
        plot_params,
        topology,
    )
end


@testitem "PassiveSystem" begin
    using Aquarium
    using ForwardDiff

    @testset "empty system" begin
        # Under the new composition-based architecture, an "empty system" is just
        # `PassiveSystem(time_step, RigidBody[], Joint[])`. We deliberately do not
        # shadow the legacy NoSystem(time_step; gravity_constant=...) constructor.
        sys = PassiveSystem(0.01, RigidBody[], Joint[])
        @test sys isa PassiveSystem
        @test sys.n_bodies == 0
        @test sys.n_configurations == 0
        @test sys.n_velocities == 0
        @test sys.n_constraints == 0
        @test sys.n_states == 0
        @test length(sys.bodies) == 0
        @test length(sys.joints) == 0
        @test calculate_potential_energy(sys, Float64[]) == 0.0
        @test isempty(calculate_system_constraint_residual(sys, Float64[]))
    end

    @testset "FreeBar" begin
        sys = FreeBar(0.01; bar_length=2.0, mass=3.0, moi=0.5, n_boundary_nodes=8)
        @test sys isa PassiveSystem
        @test sys.n_bodies == 1
        @test length(sys.joints) == 0
        @test sys.n_constraints == 0
        @test sys.n_configurations == 3
        @test sys.n_velocities == 3
        @test sys.n_states == 6

        body = sys.bodies[1]
        @test body isa RigidBody
        @test body.shape isa Bar
        @test body.shape.length == 2.0
        @test body.mass == 3.0
        @test body.n_boundary_nodes == 8
    end

    @testset "FreeDisc" begin
        sys = FreeDisc(0.01; radius=1.25, mass=4.0, moi=2.5, n_boundary_nodes=12)
        @test sys isa PassiveSystem
        @test sys.n_bodies == 1
        @test length(sys.joints) == 0
        @test sys.n_constraints == 0

        body = sys.bodies[1]
        @test body.shape isa Disc
        @test body.shape.radius == 1.25
        @test body.mass == 4.0
        @test body.n_boundary_nodes == 12

        @testset "Disc boundary nodes" begin
            xs, ys, starts, ends = generate_boundary_nodes(body.shape, 8)
            @test length(xs) == 8
            @test length(ys) == 8
            # All nodes at distance `radius` from origin
            for i in eachindex(xs)
                @test sqrt(xs[i]^2 + ys[i]^2) ≈ 1.25 atol=1e-12
            end
            # Topology wraps: 8 segments (closed polygon)
            @test length(starts) == 8
            @test length(ends) == 8
            @test ends[end] == 1    # wrap back to first
        end

        @testset "Disc attachment roles" begin
            @test local_attachment_point(body.shape, :center) == [0.0, 0.0]
            @test_throws ErrorException local_attachment_point(body.shape, :tip)
        end
    end

    @testset "Eel" begin
        n_links = 4
        eel = Eel(0.01, n_links;
            bar_lengths = [1.0, 1.0, 1.0, 1.0],
            masses = [1.0, 1.0, 1.0, 1.0],
            mois = [0.1, 0.1, 0.1, 0.1],
            equilibrium_angles = zeros(n_links - 1),
            stiffnesses = [5.0, 5.0, 5.0],
            dampings = [0.2, 0.2, 0.2])

        @test eel isa PassiveSystem
        @test eel.n_bodies == n_links
        @test eel.n_configurations == 3 * n_links
        @test eel.n_velocities == 3 * n_links
        @test eel.n_constraints == 2 * (n_links - 1)     # no world attachment — free floating
        @test eel.n_states == 6 * n_links + 2 * (n_links - 1)

        # All joints are PinJoints between consecutive bodies.
        @test length(eel.joints) == n_links - 1
        for (i, j) in enumerate(eel.joints)
            @test j isa PinJoint
            @test j.body_id_A == i
            @test j.body_id_B == i + 1
            @test j.role_A === :tip
            @test j.role_B === :root
            @test j.stiffness == 5.0
        end

        @testset "straight-line config has zero residual" begin
            # All bars along +x, adjacent tips touching. Body i center at (i - 0.5, 0).
            config = Float64[]
            for i in 1:n_links
                append!(config, [i - 0.5, 0.0, 0.0])
            end
            r = calculate_system_constraint_residual(eel, config)
            @test r ≈ zeros(2 * (n_links - 1)) atol=1e-12
        end

        @testset "eel_maximal_from_minimal" begin
            # Minimal layout: [x1, y1, θ1, θ2, θ3, ..., θ_n]
            q_min = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
            q_max = eel_maximal_from_minimal(eel, q_min, n_links)
            @test length(q_max) == 3 * n_links

            # Body 1 center at (0.5, 0, 0)
            @test q_max[1] ≈ 0.5 atol=1e-12
            @test q_max[2] ≈ 0.0 atol=1e-12
            @test q_max[3] ≈ 0.0 atol=1e-12

            # Constraint residual at computed maximal should be zero.
            r = calculate_system_constraint_residual(eel, q_max)
            @test r ≈ zeros(2 * (n_links - 1)) atol=1e-12

            # Bent config
            q_min_bent = [1.0, 0.5, 0.1, 0.3, -0.2, 0.4]
            q_max_bent = eel_maximal_from_minimal(eel, q_min_bent, n_links)
            r2 = calculate_system_constraint_residual(eel, q_max_bent)
            @test r2 ≈ zeros(2 * (n_links - 1)) atol=1e-10
        end
    end

    @testset "DoublePendulum" begin
        dp = DoublePendulum(0.01;
            bar_lengths = [1.0, 1.5],
            masses = [2.0, 1.0],
            mois = [0.2, 0.1],
            hinge_position = [0.0, 0.0],
            equilibrium_angles = [0.0, 0.0],
            stiffnesses = [0.0, 0.0],
            dampings = [0.0, 0.0])

        @test dp isa PassiveSystem
        @test dp.n_bodies == 2
        @test dp.n_configurations == 6
        @test dp.n_velocities == 6
        @test dp.n_constraints == 4    # WorldPinJoint (2) + PinJoint (2)
        @test dp.n_states == 16

        @test dp.bodies[1] isa RigidBody
        @test dp.bodies[1].shape.length == 1.0
        @test dp.bodies[1].mass == 2.0
        @test dp.bodies[2].shape.length == 1.5
        @test dp.bodies[2].mass == 1.0

        @test dp.joints[1] isa WorldPinJoint
        @test dp.joints[1].body_id == 1
        @test dp.joints[1].role === :root
        @test dp.joints[2] isa PinJoint
        @test dp.joints[2].body_id_A == 1 && dp.joints[2].role_A === :tip
        @test dp.joints[2].body_id_B == 2 && dp.joints[2].role_B === :root

        @testset "constraint residual at horizontal config" begin
            # Both links horizontal (θ=0) extending along +x from hinge at origin:
            # body1 center at (0.5, 0), tip at (1.0, 0); body2 center at (1.75, 0), root at (1.0, 0).
            config = [0.5, 0.0, 0.0,  1.75, 0.0, 0.0]
            r = calculate_system_constraint_residual(dp, config)
            @test r ≈ zeros(4) atol=1e-12
        end

        @testset "max-from-min helper" begin
            # θ1 = 0, θ2 = 0: same horizontal configuration as above.
            q = double_pendulum_maximal_from_minimal(dp, [0.0, 0.0])
            @test length(q) == 6
            @test q[1] ≈ 0.5  atol=1e-12
            @test q[2] ≈ 0.0  atol=1e-12
            @test q[3] ≈ 0.0  atol=1e-12
            @test q[4] ≈ 1.75 atol=1e-12
            @test q[5] ≈ 0.0  atol=1e-12
            @test q[6] ≈ 0.0  atol=1e-12

            # Constraint residual at the computed maximal config should be zero.
            r = calculate_system_constraint_residual(dp, q)
            @test r ≈ zeros(4) atol=1e-12

            # Non-trivial angles: also expect zero residual.
            q2 = double_pendulum_maximal_from_minimal(dp, [0.3, -0.5])
            r2 = calculate_system_constraint_residual(dp, q2)
            @test r2 ≈ zeros(4) atol=1e-12
        end
    end

    @testset "Pendulum" begin
        @testset "construction" begin
            pendulum = Pendulum(0.01;
                bar_length=1.0, mass=2.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                stiffness=0.0, damping=0.0)

            @test pendulum isa PassiveSystem
            @test pendulum.n_bodies == 1
            @test pendulum.n_configurations == 3
            @test pendulum.n_velocities == 3
            @test pendulum.n_constraints == 2
            @test pendulum.n_states == 8
            @test pendulum.configuration_indices == [1, 2, 3]
            @test pendulum.dual_indices == [7, 8]
            @test pendulum.time_step == 0.01
            @test pendulum.gravity == [0.0, -9.81]

            body = pendulum.bodies[1]
            @test body isa RigidBody
            @test body.shape isa Bar
            @test body.shape.length == 1.0
            @test body.mass == 2.0
            @test body.moi == 0.1

            joint = pendulum.joints[1]
            @test joint isa WorldPinJoint
            @test joint.world_position == [0.0, 0.0]
            @test joint.body_id == 1
            @test joint.role === :root
        end

        @testset "constraint residual at equilibrium" begin
            pendulum = Pendulum(0.01;
                bar_length=2.0, mass=1.0, moi=0.1,
                hinge_position=[0.0, 0.0])
            # body centered at (1, 0), θ=0 → root at (0, 0) = anchor
            config = [1.0, 0.0, 0.0]
            r = calculate_system_constraint_residual(pendulum, config)
            @test r ≈ [0.0, 0.0] atol=1e-12
        end

        @testset "potential energy at hanging config" begin
            # Pendulum hanging straight down: body at (0, -1), θ = -π/2 (so x-axis of body points down)
            # Actually simpler: horizontal config. Bar length 2, θ=0, center at (1, 0),
            # root at (0, 0) = anchor. COM y = 0, so gravity PE = 0.
            pendulum = Pendulum(0.01;
                bar_length=2.0, mass=3.0, moi=0.1,
                hinge_position=[0.0, 0.0])
            config = [1.0, 0.0, 0.0]
            pe = calculate_potential_energy(pendulum, config)
            @test pe ≈ 0.0 atol=1e-12

            # Lift the body to y=5: PE = m*g*h = 3*9.81*5
            config_up = [1.0, 5.0, 0.0]
            pe_up = calculate_potential_energy(pendulum, config_up)
            @test pe_up ≈ 3.0 * 9.81 * 5.0 atol=1e-10
        end

        @testset "pendulum_maximal_from_minimal" begin
            pendulum = Pendulum(0.01;
                bar_length=2.0, mass=1.0, moi=0.1,
                hinge_position=[0.5, 1.0])
            # At θ = 0: body root attaches at hinge (0.5, 1.0), body extends along +x,
            # so body center is at hinge + (length/2) * [1, 0] = (1.5, 1.0). θ_body = 0.
            q = pendulum_maximal_from_minimal(pendulum, [0.0])
            @test length(q) == 3
            @test q[1] ≈ 1.5 atol=1e-12
            @test q[2] ≈ 1.0 atol=1e-12
            @test q[3] ≈ 0.0 atol=1e-12

            # At θ = -π/2: body hangs straight down → center at hinge + (length/2) * [0, -1] = (0.5, 0.0).
            q_down = pendulum_maximal_from_minimal(pendulum, [-π/2])
            @test q_down[1] ≈ 0.5 atol=1e-12
            @test q_down[2] ≈ 0.0 atol=1e-12
            @test q_down[3] ≈ -π/2 atol=1e-12

            # Constraint residual should be zero at the maximal config
            r = calculate_system_constraint_residual(pendulum, q_down)
            @test r ≈ [0.0, 0.0] atol=1e-12
        end

        @testset "ForwardDiff gradient vs finite-difference" begin
            pendulum = Pendulum(0.01;
                bar_length=2.0, mass=3.0, moi=0.5,
                hinge_position=[0.5, 1.0],
                equilibrium_angle=0.1, stiffness=4.0, damping=0.0)
            config = [1.5, 1.0, 0.3]

            obj = p -> calculate_potential_energy(
                inject_differentiable_params(pendulum, p),
                config,
            )
            p0 = collect_differentiable_params(pendulum)
            grad_fd = ForwardDiff.gradient(obj, p0)

            # Finite-diff baseline
            ε = 1e-6
            grad_num = similar(p0)
            for i in eachindex(p0)
                ei = zeros(length(p0)); ei[i] = ε
                grad_num[i] = (obj(p0 .+ ei) - obj(p0 .- ei)) / (2ε)
            end

            @test grad_fd ≈ grad_num atol=1e-5
        end
    end
end
