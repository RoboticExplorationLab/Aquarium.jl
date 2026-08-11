abstract type Joint end

function joint_n_constraints end
function calculate_joint_constraint_residual end
function calculate_joint_potential_energy end
function calculate_joint_damping_force end


@testitem "Joint construction" begin
    using AquariumClosed
    @testset "PinJoint" begin
        j = PinJoint(1, :tip, 2, :root;
            equilibrium_angle=0.1, stiffness=5.0, damping=0.2)
        @test j.body_id_A == 1
        @test j.role_A === :tip
        @test j.body_id_B == 2
        @test j.role_B === :root
        @test j.equilibrium_angle == 0.1
        @test j.stiffness == 5.0
        @test j.damping == 0.2
        @test joint_n_constraints(j) == 2
        @test j isa Joint
    end

    @testset "WorldPinJoint" begin
        j = WorldPinJoint([0.5, 1.0], 1, :root;
            equilibrium_angle=0.0, stiffness=0.0, damping=0.0)
        @test j.world_position == [0.5, 1.0]
        @test j.body_id == 1
        @test j.role === :root
        @test j.equilibrium_angle == 0.0
        @test j.stiffness == 0.0
        @test j.damping == 0.0
        @test joint_n_constraints(j) == 2
        @test j isa Joint
    end
end

@testitem "Joint residuals and energies" begin
    using AquariumClosed
    # Two bars of length 2, both at θ=0.
    # Body 1 centered at (0,0), tip at (+1, 0).
    # Body 2 centered at (2,0), root at (+1, 0).
    # PinJoint between tip of B1 and root of B2 should be satisfied.
    b1 = RigidBody(Bar(2.0); mass=1.0, moi=0.1)
    b2 = RigidBody(Bar(2.0); mass=1.0, moi=0.1)
    bodies = [b1, b2]

    config_satisfied = [0.0, 0.0, 0.0,   2.0, 0.0, 0.0]    # 3 per body
    velocity_zero    = [0.0, 0.0, 0.0,   0.0, 0.0, 0.0]

    @testset "PinJoint constraint residual — satisfied" begin
        joint = PinJoint(1, :tip, 2, :root)
        r = calculate_joint_constraint_residual(joint, config_satisfied, bodies)
        @test length(r) == 2
        @test r ≈ [0.0, 0.0] atol=1e-12
    end

    @testset "PinJoint constraint residual — violated" begin
        joint = PinJoint(1, :tip, 2, :root)
        bad_config = [0.0, 0.0, 0.0,  3.0, 0.0, 0.0]   # body 2 shifted right
        r = calculate_joint_constraint_residual(joint, bad_config, bodies)
        @test r[1] ≈ -1.0 atol=1e-12  # B1_tip - B2_root = (1, 0) - (2, 0)
        @test r[2] ≈ 0.0  atol=1e-12
    end

    @testset "WorldPinJoint constraint residual — satisfied" begin
        # One body centered at (1, 0); its root is at (0, 0); joint anchored at world (0, 0).
        single_bodies = [b1]
        config = [1.0, 0.0, 0.0]
        joint = WorldPinJoint([0.0, 0.0], 1, :root)
        r = calculate_joint_constraint_residual(joint, config, single_bodies)
        @test length(r) == 2
        @test r ≈ [0.0, 0.0] atol=1e-12
    end

    @testset "WorldPinJoint constraint residual — violated" begin
        single_bodies = [b1]
        joint = WorldPinJoint([0.5, 0.0], 1, :root)   # anchor at (0.5, 0)
        config = [1.0, 0.0, 0.0]                       # root at (0, 0)
        r = calculate_joint_constraint_residual(joint, config, single_bodies)
        @test r[1] ≈ -0.5 atol=1e-12
        @test r[2] ≈ 0.0  atol=1e-12
    end

    @testset "PinJoint spring PE" begin
        # Rest angle 0.3, actual relative angle (θ_B − θ_A) = 0.5, stiffness k = 4.
        # PE = (k/2) * (0.5 - 0.3)^2 = 2 * 0.04 = 0.08
        joint = PinJoint(1, :tip, 2, :root; equilibrium_angle=0.3, stiffness=4.0)
        config = [0.0, 0.0, 0.2,  2.0, 0.0, 0.7]   # θ_A=0.2, θ_B=0.7
        pe = calculate_joint_potential_energy(joint, config, bodies)
        @test pe ≈ 0.08 atol=1e-12
    end

    @testset "WorldPinJoint spring PE" begin
        # Rest angle 0.0, actual angle 0.4, stiffness k = 2.
        # PE = (2/2) * 0.4^2 = 0.16
        joint = WorldPinJoint([0.0, 0.0], 1, :root; equilibrium_angle=0.0, stiffness=2.0)
        single_bodies = [b1]
        config = [1.0, 0.0, 0.4]
        pe = calculate_joint_potential_energy(joint, config, single_bodies)
        @test pe ≈ 0.16 atol=1e-12
    end

    @testset "PinJoint damping force" begin
        # damping coefficient b = 3, ω_A = 0.5, ω_B = 1.5 → relative ω = 1.0
        # Torque on B = -b * ω_rel = -3
        # Torque on A = +3 (action-reaction)
        joint = PinJoint(1, :tip, 2, :root; damping=3.0)
        velocity = [0.0, 0.0, 0.5,  0.0, 0.0, 1.5]
        f = calculate_joint_damping_force(joint, velocity, bodies)
        @test length(f) == 6     # 3 per body × 2 bodies
        # Body A angular velocity slot is index 3; Body B is index 6.
        @test f[3] ≈ 3.0 atol=1e-12     # -(-3) = +3 on A
        @test f[6] ≈ -3.0 atol=1e-12    # -3 on B
        @test f[1] == 0.0 && f[2] == 0.0 && f[4] == 0.0 && f[5] == 0.0
    end

    @testset "WorldPinJoint damping force" begin
        joint = WorldPinJoint([0.0, 0.0], 1, :root; damping=2.0)
        single_bodies = [b1]
        velocity = [0.0, 0.0, 1.5]
        f = calculate_joint_damping_force(joint, velocity, single_bodies)
        @test length(f) == 3
        # Torque = -b * ω = -3.0
        @test f[3] ≈ -3.0 atol=1e-12
    end
end
