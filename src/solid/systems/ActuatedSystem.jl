struct ActuatedSystem <: SolidSystem
    bodies::Vector{AbstractRigidBody}
    joints::Vector{Joint}
    actuators::Vector{Actuator}

    n_bodies::Int
    n_configurations::Int
    n_velocities::Int
    n_constraints::Int
    n_body_states::Int
    n_states::Int
    n_actuators::Int
    n_control_inputs::Int

    state_indices::Vector{Int}
    configuration_indices::Vector{Int}
    velocity_indices::Vector{Int}
    body_state_indices::Vector{Int}
    dual_indices::Vector{Int}

    time_step::Float64
    gravity::Vector{Float64}

    plot_params::Dict{Symbol, Any}

    topology::SystemTopology

    actuation_mode::Symbol
    prescribed_angle_dual_indices::Vector{Int}
end

function ActuatedSystem(
    time_step::Real,
    bodies::AbstractVector{<:AbstractRigidBody},
    joints::AbstractVector{<:Joint},
    actuators::AbstractVector{<:Actuator};
    gravity::AbstractVector{<:Real} = [0.0, -9.81],
    plot_params::Dict{Symbol, Any} = default_plot_params(),
    actuation_mode::Symbol = :pd,
)
    length(gravity) == 2 || error("ActuatedSystem.gravity must be 2D, got length $(length(gravity))")
    actuation_mode in (:pd, :prescribed) || error("ActuatedSystem: actuation_mode must be :pd or :prescribed, got :$actuation_mode")

    _validate_bodies_and_joints(bodies, joints)

    n_bodies = length(bodies)
    n_configurations = 3 * n_bodies
    n_velocities = 3 * n_bodies
    n_positional_constraints = isempty(joints) ? 0 : sum(joint_n_constraints(j) for j in joints)
    n_body_states = n_configurations + n_velocities

    n_actuators = length(actuators)

    n_angle_constraints = actuation_mode == :prescribed ? n_actuators : 0
    n_constraints = n_positional_constraints + n_angle_constraints
    n_states = n_body_states + n_constraints

    n_control_inputs = if actuation_mode == :prescribed
        n_actuators   # 1 DOF per joint (θ_desired only)
    else
        isempty(actuators) ? 0 : sum(n_control_inputs_per_actuator(a) for a in actuators)
    end

    state_indices = collect(1:n_states)
    configuration_indices = collect(1:n_configurations)
    velocity_indices = collect((n_configurations+1):(n_configurations+n_velocities))
    body_state_indices = vcat(configuration_indices, velocity_indices)
    dual_indices = collect((n_body_states+1):n_states)

    prescribed_angle_dual_indices = if actuation_mode == :prescribed
        collect((n_body_states + n_positional_constraints + 1):n_states)
    else
        Int[]
    end

    topology = _compute_system_topology(
        bodies,
        n_configurations, n_velocities, n_body_states, n_states,
        configuration_indices, velocity_indices,
    )

    return ActuatedSystem(
        Vector{AbstractRigidBody}(bodies),
        Vector{Joint}(joints),
        Vector{Actuator}(actuators),
        n_bodies, n_configurations, n_velocities, n_constraints, n_body_states, n_states,
        n_actuators, n_control_inputs,
        state_indices, configuration_indices, velocity_indices, body_state_indices, dual_indices,
        convert(Float64, time_step),
        convert(Vector{Float64}, gravity),
        plot_params,
        topology,
        actuation_mode,
        prescribed_angle_dual_indices,
    )
end

# Declared here, implemented per actuator type elsewhere.
function n_control_inputs_per_actuator end
function calculate_new_actuator_force end


@testitem "ActuatedSystem" begin
    using AquariumClosed
    using ForwardDiff

    @testset "RExEel" begin
        n_links = 4
        rex = RExEel(0.01, n_links;
            bar_lengths = ones(n_links),
            masses = ones(n_links),
            mois = fill(0.1, n_links),
            Kps = fill(50.0, n_links - 1),
            Kds = fill(5.0, n_links - 1),
            max_torques = fill(2.0, n_links - 1),
            actuation_mode = :pd)

        @test rex isa ActuatedSystem
        @test rex.n_bodies == n_links
        @test rex.n_configurations == 3 * n_links
        @test rex.n_constraints == 2 * (n_links - 1)
        @test rex.n_actuators == n_links - 1
        @test rex.n_control_inputs == 2 * (n_links - 1)

        # Raw Kp/Kd are converted via xc330m288t_gains inside RExEel
        Kp_expected, Kd_expected = xc330m288t_gains(; Kp_raw=50.0, Kd_raw=5.0)
        @test all(a isa JointServoMotor for a in rex.actuators)
        for (i, a) in enumerate(rex.actuators)
            @test a.joint_id == i
            @test a.controller.Kp ≈ Kp_expected
            @test a.controller.Kd ≈ Kd_expected
            @test a.max_torque == 2.0
        end

        @testset "actuator forces at rest with zero desired" begin
            config = Float64[]
            for i in 1:n_links
                append!(config, [i - 0.5, 0.0, 0.0])
            end
            velocity = zeros(3 * n_links)
            duals = zeros(2 * (n_links - 1))
            state = vcat(config, velocity, duals)
            control = zeros(2 * (n_links - 1))
            f = calculate_new_actuator_forces(rex, state, control)
            @test length(f) == 3 * n_links
            @test f ≈ zeros(3 * n_links) atol=1e-12
        end

        @testset "rex_eel_maximal_from_minimal" begin
            q_min = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
            q_max = rex_eel_maximal_from_minimal(rex, q_min, n_links)
            @test length(q_max) == 3 * n_links
            r = calculate_system_constraint_residual(rex, q_max)
            @test r ≈ zeros(2 * (n_links - 1)) atol=1e-10

            # Non-trivial
            q_min2 = [1.0, 0.5, 0.2, -0.3, 0.1, 0.4]
            q_max2 = rex_eel_maximal_from_minimal(rex, q_min2, n_links)
            r2 = calculate_system_constraint_residual(rex, q_max2)
            @test r2 ≈ zeros(2 * (n_links - 1)) atol=1e-10
        end
    end

    @testset "prescribed mode: RExEel" begin
        n_links = 4
        rex = RExEel(0.01, n_links;
            bar_lengths = ones(n_links),
            masses = ones(n_links),
            mois = fill(0.1, n_links),
            Kps = fill(50.0, n_links - 1),
            Kds = fill(5.0, n_links - 1),
            max_torques = fill(2.0, n_links - 1),
            actuation_mode=:prescribed)

        @test rex isa ActuatedSystem
        @test rex.actuation_mode == :prescribed
        @test rex.n_bodies == n_links

        n_joints = n_links - 1
        @test rex.n_constraints == 2 * n_joints + n_joints
        @test rex.n_actuators == n_joints
        @test rex.n_control_inputs == n_joints

        n_body_states = 6 * n_links
        @test rex.n_body_states == n_body_states
        @test rex.n_states == n_body_states + 3 * n_joints

        @test rex.prescribed_angle_dual_indices ==
            collect((n_body_states + 2*n_joints + 1):(n_body_states + 3*n_joints))
    end

    @testset "ActuatedPendulum" begin
        @testset "construction" begin
            ap = ActuatedPendulum(0.01;
                bar_length=1.0, mass=2.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=50.0, Kd=5.0, max_torque=2.0)

            @test ap isa ActuatedSystem
            @test ap.n_bodies == 1
            @test ap.n_configurations == 3
            @test ap.n_velocities == 3
            @test ap.n_constraints == 2
            @test ap.n_body_states == 6
            @test ap.n_states == 8
            @test ap.n_actuators == 1
            @test ap.n_control_inputs == 2

            @test ap.bodies[1] isa RigidBody
            @test ap.bodies[1].shape.length == 1.0
            @test ap.bodies[1].mass == 2.0

            @test ap.joints[1] isa WorldPinJoint
            @test ap.actuators[1] isa JointServoMotor
            @test ap.actuators[1].joint_id == 1
            @test ap.actuators[1].controller.Kp == 50.0
            @test ap.actuators[1].controller.Kd == 5.0
            @test ap.actuators[1].max_torque == 2.0
        end

        @testset "actuator force at rest with zero desired" begin
            ap = ActuatedPendulum(0.01;
                bar_length=1.0, mass=1.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=50.0, Kd=5.0)
            # State: x=0.5, y=0, θ=0, velocities all zero → body at rest horizontal
            # Control: [θ_desired=0, ω_desired=0] → no error → zero torque
            system_state = [0.5, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0]
            control = [0.0, 0.0]
            f = calculate_new_actuator_forces(ap, system_state, control)
            @test length(f) == 3    # 3 velocities for 1 body
            @test f ≈ [0.0, 0.0, 0.0] atol=1e-12
        end

        @testset "actuator force with angle error" begin
            ap = ActuatedPendulum(0.01;
                bar_length=1.0, mass=1.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=10.0, Kd=0.0)
            # State: θ=0, ω=0. Control: [θ_des=0.5, ω_des=0] → error = 0.5
            # Torque = Kp * (0.5 - 0) = 5.0
            # Applied to body 1's angular velocity slot (index 3)
            system_state = [0.5, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0]
            control = [0.5, 0.0]
            f = calculate_new_actuator_forces(ap, system_state, control)
            @test f[1] == 0.0
            @test f[2] == 0.0
            @test f[3] ≈ 5.0 atol=1e-12
        end

        @testset "actuator force saturates at max_torque" begin
            ap = ActuatedPendulum(0.01;
                bar_length=1.0, mass=1.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=100.0, Kd=0.0, max_torque=3.0)
            # PD would compute Kp*5 = 500, but also PDController clamps at output_min/max
            # Since we pass output_min/max = ±max_torque to PDController, it saturates to ±3.
            system_state = [0.5, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0]
            control = [5.0, 0.0]
            f = calculate_new_actuator_forces(ap, system_state, control)
            @test f[3] ≈ 3.0 atol=1e-12
        end

        @testset "actuator force with velocity error (Kd)" begin
            ap = ActuatedPendulum(0.01;
                bar_length=1.0, mass=1.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=0.0, Kd=10.0)
            # State: θ=0, ω=2.0. Control: [θ_des=0, ω_des=0] → velocity error = -2.0
            # Torque = Kd * (0 - 2.0) = -20.0
            system_state = [0.5, 0.0, 0.0,  0.0, 0.0, 2.0,  0.0, 0.0]
            control = [0.0, 0.0]
            f = calculate_new_actuator_forces(ap, system_state, control)
            @test f[3] ≈ -20.0 atol=1e-12
        end

        @testset "negative angle error produces negative torque" begin
            ap = ActuatedPendulum(0.01;
                bar_length=1.0, mass=1.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=10.0, Kd=0.0)
            # State: θ=0.5. Control: [θ_des=-0.5] → error = -0.5 - 0.5 = -1.0
            # Torque = 10 * (-1.0) = -10.0
            system_state = [0.5, 0.0, 0.5,  0.0, 0.0, 0.0,  0.0, 0.0]
            control = [-0.5, 0.0]
            f = calculate_new_actuator_forces(ap, system_state, control)
            @test f[3] ≈ -10.0 atol=1e-12
        end

        @testset "negative saturation" begin
            ap = ActuatedPendulum(0.01;
                bar_length=1.0, mass=1.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=100.0, Kd=0.0, max_torque=3.0)
            system_state = [0.5, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0]
            control = [-5.0, 0.0]
            f = calculate_new_actuator_forces(ap, system_state, control)
            @test f[3] ≈ -3.0 atol=1e-12
        end

        @testset "ForwardDiff through torque computation" begin
            ap = ActuatedPendulum(0.01;
                bar_length=1.0, mass=1.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=20.0, Kd=5.0, max_torque=100.0)
            system_state = [0.5, 0.0, 0.1,  0.0, 0.0, 0.3,  0.0, 0.0]
            control = [0.5, 0.0]
            J = ForwardDiff.jacobian(
                s -> calculate_new_actuator_forces(ap, s, control),
                system_state)
            @test size(J) == (3, 8)
            @test all(isfinite.(J))
        end

        @testset "xc330m288t_gains helper reproduces legacy values" begin
            # Legacy XC330M288T defaults:
            #   Kp_raw=1100, Kd_raw=500, stall_torque=9.3e6,
            #   encoder_resolution=4096, control_loop_time=0.001
            # pwm_to_torque = 9.3e6 / 885 ≈ 10508.47
            # Kp = 1100 * (4096/2π) * 10508.47 / 128
            # Kd = 500 * 0.001 * (4096/2π) * 10508.47 / 16
            Kp, Kd = xc330m288t_gains()
            pwm_to_torque = 9.3e6 / 885
            expected_Kp = 1100 * (4096 / 2π) * pwm_to_torque / 128
            expected_Kd = 500 * 0.001 * (4096 / 2π) * pwm_to_torque / 16
            @test Kp ≈ expected_Kp atol=1e-6
            @test Kd ≈ expected_Kd atol=1e-6
        end

        @testset "prescribed mode construction" begin
            ap = ActuatedPendulum(0.01;
                bar_length=1.0, mass=2.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=50.0, Kd=5.0, max_torque=2.0,
                actuation_mode=:prescribed)

            @test ap isa ActuatedSystem
            @test ap.actuation_mode == :prescribed
            @test ap.n_bodies == 1
            @test ap.n_constraints == 3
            @test ap.n_states == 9
            @test ap.n_control_inputs == 1
            @test ap.prescribed_angle_dual_indices == [9]
            @test ap.dual_indices == [7, 8, 9]
        end

        @testset "pd mode backward compat" begin
            ap_pd = ActuatedPendulum(0.01;
                bar_length=1.0, mass=2.0, moi=0.1,
                hinge_position=[0.0, 0.0],
                Kp=50.0, Kd=5.0, max_torque=2.0)

            @test ap_pd.actuation_mode == :pd
            @test ap_pd.n_constraints == 2
            @test ap_pd.n_states == 8
            @test ap_pd.n_control_inputs == 2
            @test ap_pd.prescribed_angle_dual_indices == Int[]
        end

        @testset "ForwardDiff gradient through potential energy" begin
            ap = ActuatedPendulum(0.01;
                bar_length=2.0, mass=3.0, moi=0.5,
                hinge_position=[0.0, 1.0])
            config = [0.0, 2.0, 0.1]

            obj = p -> calculate_potential_energy(
                inject_differentiable_params(ap, p),
                config,
            )
            p0 = collect_differentiable_params(ap)
            grad = ForwardDiff.gradient(obj, p0)

            # Finite-diff baseline
            ε = 1e-6
            grad_num = similar(p0)
            for i in eachindex(p0)
                ei = zeros(length(p0)); ei[i] = ε
                grad_num[i] = (obj(p0 .+ ei) - obj(p0 .- ei)) / (2ε)
            end
            @test grad ≈ grad_num atol=1e-5
        end
    end
end
