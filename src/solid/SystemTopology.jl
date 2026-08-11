#############################################################################################
## SystemTopology
##
## Aggregated count/index information for a solid system's boundary nodes, segments, and
## state layout. Computed once at system construction from the bodies list, then cached as
## `system.topology` on both `PassiveSystem` and `ActuatedSystem`. FSI kernel functions
## (`calculate_*_fsi_kernel*`, `calculate_average_velocity_segment*`) and FSI interface
## functions (`calculate_boundary_state` and friends) take a `SystemTopology` directly
## as their solid-side argument; there is no `SolidBody`-shaped adapter object in between.
##
## Boundary state layout (mirroring the legacy SingleRigidBody convention):
## [boundary_pos_x..., boundary_pos_y..., boundary_vel_x..., boundary_vel_y...]
## So for N total boundary nodes:
##     boundary_configuration_indices = 1 : 2N
##     boundary_velocity_indices      = 2N+1 : 4N
#############################################################################################

struct SystemTopology
    # Aggregated counts
    n_no_slip_constraints::Int
    n_boundary_nodes::Int
    n_boundary_segments::Int
    n_configurations::Int
    n_velocities::Int
    n_body_states::Int
    n_states::Int

    # Per-segment start/end node indices into the concatenated boundary-node vector
    boundary_segment_start_nodes::Vector{Int}
    boundary_segment_end_nodes::Vector{Int}

    # Index vectors carved out of the boundary state
    boundary_configuration_indices::UnitRange{Int}
    boundary_velocity_indices::UnitRange{Int}
    configuration_indices::Vector{Int}
    velocity_indices::Vector{Int}

    # Shared immersed-boundary method across all bodies
    immersed_boundary_method::Symbol
    ib_method::Symbol

    # Shared discrete delta kernel kind across all bodies
    discrete_delta_kind::Symbol
end

function _compute_system_topology(
    bodies::AbstractVector,
    n_configurations::Int,
    n_velocities::Int,
    n_body_states::Int,
    n_states::Int,
    configuration_indices::AbstractVector{Int},
    velocity_indices::AbstractVector{Int},
)
    total_n_nodes = 0
    segs_start = Int[]
    segs_end = Int[]
    ib_method_set = Symbol[]
    delta_kind_set = Symbol[]
    for body in bodies
        body isa RigidBody || continue
        n_nodes = body.n_boundary_nodes
        _, _, body_starts, body_ends = generate_boundary_nodes(body.shape, n_nodes)
        append!(segs_start, body_starts .+ total_n_nodes)
        append!(segs_end, body_ends .+ total_n_nodes)
        total_n_nodes += n_nodes
        push!(ib_method_set, body.ib_method)
        push!(delta_kind_set, body.discrete_delta_kind)
    end
    total_n_segs = length(segs_start)

    unique_ib = unique(ib_method_set)
    if length(unique_ib) > 1
        error("All bodies in a solid system must share the same ib_method; got $(unique_ib)")
    end
    ib_method = isempty(unique_ib) ? :weak_form : unique_ib[1]

    unique_delta = unique(delta_kind_set)
    if length(unique_delta) > 1
        error("All bodies in a solid system must share the same discrete_delta_kind; got $(unique_delta)")
    end
    discrete_delta_kind = isempty(unique_delta) ? :one_point : unique_delta[1]

    n_no_slip = if ib_method === :original
        2 * total_n_nodes
    elseif ib_method === :weak_form
        2 * total_n_segs
    else
        error("Unknown ib_method $(ib_method)")
    end

    boundary_configuration_indices = 1:(2 * total_n_nodes)
    boundary_velocity_indices = (2 * total_n_nodes + 1):(4 * total_n_nodes)

    return SystemTopology(
        n_no_slip,
        total_n_nodes,
        total_n_segs,
        n_configurations,
        n_velocities,
        n_body_states,
        n_states,
        segs_start,
        segs_end,
        boundary_configuration_indices,
        boundary_velocity_indices,
        collect(configuration_indices),
        collect(velocity_indices),
        ib_method,
        ib_method,
        discrete_delta_kind,
    )
end


@testitem "PassiveSystem assembly" begin
    using AquariumClosed
    @testset "single body with world hinge" begin
        body = RigidBody(Bar(1.0); mass=1.0, moi=0.1)
        joint = WorldPinJoint([0.0, 0.0], 1, :root; stiffness=0.0)

        sys = PassiveSystem(0.01, [body], Joint[joint])

        @test sys.n_bodies == 1
        @test sys.n_configurations == 3
        @test sys.n_velocities == 3
        @test sys.n_constraints == 2
        @test sys.n_body_states == 6
        @test sys.n_states == 8

        @test sys.configuration_indices == [1, 2, 3]
        @test sys.velocity_indices == [4, 5, 6]
        @test sys.body_state_indices == [1, 2, 3, 4, 5, 6]
        @test sys.dual_indices == [7, 8]

        @test sys.time_step == 0.01
        @test sys.gravity == [0.0, -9.81]

        @test length(sys.bodies) == 1
        @test length(sys.joints) == 1
    end

    @testset "two bodies with pin joint" begin
        b1 = RigidBody(Bar(1.0); mass=1.0, moi=0.1)
        b2 = RigidBody(Bar(1.0); mass=1.0, moi=0.1)
        jw = WorldPinJoint([0.0, 0.0], 1, :root)
        jp = PinJoint(1, :tip, 2, :root)

        sys = PassiveSystem(0.01, [b1, b2], Joint[jw, jp])

        @test sys.n_bodies == 2
        @test sys.n_configurations == 6
        @test sys.n_velocities == 6
        @test sys.n_constraints == 4  # 2 + 2
        @test sys.n_body_states == 12
        @test sys.n_states == 16
        @test sys.dual_indices == [13, 14, 15, 16]
    end

    @testset "empty system" begin
        sys = PassiveSystem(0.01, RigidBody[], Joint[])
        @test sys.n_bodies == 0
        @test sys.n_configurations == 0
        @test sys.n_constraints == 0
        @test sys.n_states == 0
    end

    @testset "validation" begin
        good_body = RigidBody(Bar(1.0); mass=1.0, moi=0.1)

        # negative mass
        bad_mass_body = RigidBody(Bar(1.0); mass=-1.0, moi=0.1)
        @test_throws ErrorException PassiveSystem(0.01, [bad_mass_body], Joint[])

        # negative MOI
        bad_moi_body = RigidBody(Bar(1.0); mass=1.0, moi=-0.1)
        @test_throws ErrorException PassiveSystem(0.01, [bad_moi_body], Joint[])

        # joint references nonexistent body
        bad_joint = PinJoint(1, :tip, 2, :root)
        @test_throws ErrorException PassiveSystem(0.01, [good_body], Joint[bad_joint])

        # joint role invalid for shape
        bad_role_joint = WorldPinJoint([0.0, 0.0], 1, :bogus)
        @test_throws ErrorException PassiveSystem(0.01, [good_body], Joint[bad_role_joint])

        # world_id = 0 (not allowed in PinJoint body IDs)
        bad_world_pin = PinJoint(0, :tip, 1, :root)
        @test_throws ErrorException PassiveSystem(0.01, [good_body], Joint[bad_world_pin])
    end
end

@testitem "SystemTopology" begin
    using AquariumClosed
    @testset "PassiveSystem with single RigidBody (FreeBar)" begin
        sys = FreeBar(0.01; bar_length=2.0, mass=3.0, moi=0.5, n_boundary_nodes=8)
        topo = sys.topology

        @test topo isa SystemTopology

        # Counts
        @test topo.n_boundary_nodes == 8
        # Bar is an open curve, so 8 nodes → 7 segments
        @test topo.n_boundary_segments == 7
        @test topo.n_configurations == sys.n_configurations
        @test topo.n_velocities == sys.n_velocities
        @test topo.n_body_states == sys.n_body_states
        @test topo.n_states == sys.n_states

        # Layout: config indices go before velocity indices in body state
        @test topo.boundary_configuration_indices == 1:(2 * 8)
        @test topo.boundary_velocity_indices == (2 * 8 + 1):(4 * 8)

        # ib_method agreement
        @test topo.ib_method == sys.bodies[1].ib_method
        @test topo.immersed_boundary_method == sys.bodies[1].ib_method

        # For weak_form: n_no_slip_constraints == 2 * n_boundary_segments
        # For original:  n_no_slip_constraints == 2 * n_boundary_nodes
        expected_n_nsc = topo.ib_method === :weak_form ? 2 * topo.n_boundary_segments : 2 * topo.n_boundary_nodes
        @test topo.n_no_slip_constraints == expected_n_nsc
    end

    @testset "PassiveSystem with Pendulum (2 bodies + joint)" begin
        sys = Pendulum(0.01;
            bar_length = 0.1,
            mass       = 0.1,
            moi        = 1e-4,
            hinge_position = [0.5, 0.5],
            gravity    = [0.0, -98.0],
            n_boundary_nodes = 6,
        )
        topo = sys.topology

        # Only bodies with RigidBody contribute to boundary_nodes, but Pendulum has
        # two bodies: the anchor (Disc, 0 boundary nodes) and the bar (6 nodes).
        # The anchor Disc has 0 boundary nodes in the Pendulum constructor convention.
        total_nodes = sum(b isa RigidBody ? b.n_boundary_nodes : 0 for b in sys.bodies)
        @test topo.n_boundary_nodes == total_nodes
        @test topo.n_boundary_segments > 0

        # Check index layout corresponds to full boundary-state vector of length 4*n_nodes
        @test length(topo.boundary_configuration_indices) == 2 * topo.n_boundary_nodes
        @test length(topo.boundary_velocity_indices) == 2 * topo.n_boundary_nodes
        @test topo.boundary_configuration_indices == 1:(2 * topo.n_boundary_nodes)
        @test topo.boundary_velocity_indices ==
            (2 * topo.n_boundary_nodes + 1):(4 * topo.n_boundary_nodes)
    end

    @testset "ActuatedSystem (ActuatedPendulum) topology" begin
        sys = ActuatedPendulum(0.01;
            bar_length=0.1, mass=0.1, moi=1e-4,
            hinge_position=[0.5, 0.5],
            gravity=[0.0, -98.0],
            n_boundary_nodes=6,
        )
        topo = sys.topology
        @test topo isa SystemTopology
        @test topo.n_boundary_nodes ==
            sum(b isa RigidBody ? b.n_boundary_nodes : 0 for b in sys.bodies)
        @test topo.n_boundary_segments >= 0
        @test topo.ib_method isa Symbol
    end

    @testset "SystemTopology construction errors on inconsistent ib_method" begin
        # Construct two bodies with different ib_methods and try to build a PassiveSystem.
        shape = Bar(1.0)
        body_a = RigidBody(shape; mass=1.0, moi=0.1, com_offset=[0.0, 0.0],
            n_boundary_nodes=4, ib_method=:weak_form)
        body_b = RigidBody(shape; mass=1.0, moi=0.1, com_offset=[0.0, 0.0],
            n_boundary_nodes=4, ib_method=:original)
        @test_throws Exception PassiveSystem(0.01, [body_a, body_b], Joint[])
    end

    @testset "discrete_delta_kind propagation" begin
        # Default is :one_point
        sys_default = FreeBar(0.01; bar_length=1.0, mass=1.0, moi=0.1)
        @test sys_default.topology.discrete_delta_kind === :one_point

        # Explicit :three_point
        sys_3pt = FreeDisc(0.01; radius=0.15, mass=1.0, moi=0.5,
                           discrete_delta_kind=:three_point)
        @test sys_3pt.topology.discrete_delta_kind === :three_point

        # Mismatched discrete_delta_kind errors
        shape = Bar(1.0)
        body_a = RigidBody(shape; mass=1.0, moi=0.1, discrete_delta_kind=:one_point)
        body_b = RigidBody(shape; mass=1.0, moi=0.1, discrete_delta_kind=:three_point)
        @test_throws Exception PassiveSystem(0.01, [body_a, body_b], Joint[])
    end
end
