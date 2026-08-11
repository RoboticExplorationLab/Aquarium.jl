#############################################################################################
## NoSystem singleton + empty_passive_system compatibility helper.
##
## `NoSystem` is a zero-field singleton solid-system type used as an AquariumTank
## placeholder when a tank has no bluff body or no swimmer. Its method set mirrors every
## accessor tank and FSI code calls on a `PassiveSystem`/`ActuatedSystem`, returning an
## appropriate empty/zero value so NoSystem can be plugged into the tank without special-
## casing the `nothing` path.
##
## `empty_passive_system` is the legacy helper that returns an actual empty PassiveSystem.
## It exists while the tank still stores its `bluff_body` / `swimmer` as PassiveSystem
## instances. Slice 8 (#71) retypes those fields to `::SolidSystem` and switches the
## helpers to use `NoSystem()` directly.
#############################################################################################

struct NoSystem <: SolidSystem end

# --- Scalar accessors that tank/FSI code reads on a system ---
Base.getproperty(::NoSystem, name::Symbol) = _no_system_property(name)

function _no_system_property(name::Symbol)
    if name === :time_step
        return 0.0
    elseif name === :gravity
        return [0.0, 0.0]
    elseif name === :n_bodies || name === :n_configurations || name === :n_velocities ||
           name === :n_constraints || name === :n_body_states || name === :n_states
        return 0
    elseif name === :bodies
        return AbstractRigidBody[]
    elseif name === :joints
        return Joint[]
    elseif name === :state_indices || name === :configuration_indices ||
           name === :velocity_indices || name === :body_state_indices ||
           name === :dual_indices
        return Int[]
    elseif name === :topology
        return _zero_system_topology()
    else
        error("NoSystem has no property `$(name)`")
    end
end

Base.propertynames(::NoSystem, private::Bool=false) = (
    :time_step, :gravity, :n_bodies, :n_configurations, :n_velocities,
    :n_constraints, :n_body_states, :n_states, :bodies, :joints,
    :state_indices, :configuration_indices, :velocity_indices,
    :body_state_indices, :dual_indices, :topology,
)

function _zero_system_topology()
    return SystemTopology(
        0,           # n_no_slip_constraints
        0,           # n_boundary_nodes
        0,           # n_boundary_segments
        0,           # n_configurations
        0,           # n_velocities
        0,           # n_body_states
        0,           # n_states
        Int[],       # boundary_segment_start_nodes
        Int[],       # boundary_segment_end_nodes
        1:0,         # boundary_configuration_indices
        1:0,         # boundary_velocity_indices
        Int[],       # configuration_indices
        Int[],       # velocity_indices
        :weak_form,  # immersed_boundary_method (default)
        :weak_form,  # ib_method
        :one_point,  # discrete_delta_kind
    )
end

# --- Differentiable-params contract ---
n_differentiable_params(::NoSystem) = 0
collect_differentiable_params(::NoSystem) = Float64[]
inject_differentiable_params(::NoSystem, ::AbstractVector) = NoSystem()

# --- Legacy empty-PassiveSystem helper ---
# Called from AquariumTank_only_fluid / _only_swimmer / _only_bluff_body and from test
# fixtures that want an empty PassiveSystem bluff body. Slice 8 switches tank code to
# use `NoSystem()` instead; this helper can then be removed.

function empty_passive_system(time_step::Real;
    gravity_constant::Real = 9.81,
    plot_params::Dict{Symbol, Any} = default_plot_params(),
)
    return PassiveSystem(time_step, RigidBody[], Joint[];
        gravity = [0.0, -convert(Float64, gravity_constant)],
        plot_params = plot_params,
    )
end


@testitem "NoSystem singleton" begin
    using AquariumClosed
    @testset "constructor and type" begin
        ns = NoSystem()
        @test ns isa NoSystem
        @test ns isa AquariumClosed.SolidSystem
    end

    @testset "scalar property reads" begin
        ns = NoSystem()
        @test ns.time_step == 0.0
        @test ns.gravity == [0.0, 0.0]
        @test ns.n_bodies == 0
        @test ns.n_configurations == 0
        @test ns.n_velocities == 0
        @test ns.n_constraints == 0
        @test ns.n_body_states == 0
        @test ns.n_states == 0
        @test isempty(ns.bodies)
        @test isempty(ns.joints)
        @test isempty(ns.state_indices)
        @test isempty(ns.configuration_indices)
        @test isempty(ns.velocity_indices)
        @test isempty(ns.body_state_indices)
        @test isempty(ns.dual_indices)
    end

    @testset "topology returns a zero-filled SystemTopology" begin
        ns = NoSystem()
        topo = ns.topology
        @test topo isa SystemTopology
        @test topo.n_no_slip_constraints == 0
        @test topo.n_boundary_nodes == 0
        @test topo.n_boundary_segments == 0
        @test topo.n_states == 0
        @test isempty(topo.boundary_segment_start_nodes)
        @test isempty(topo.boundary_segment_end_nodes)
        @test topo.boundary_configuration_indices == 1:0
        @test topo.boundary_velocity_indices == 1:0
    end

    @testset "differentiable-params contract" begin
        ns = NoSystem()
        @test n_differentiable_params(ns) == 0
        @test collect_differentiable_params(ns) == Float64[]
        @test inject_differentiable_params(ns, Float64[]) isa NoSystem
        @test inject_differentiable_params(ns, [1.0, 2.0]) isa NoSystem  # ignores input
    end
end
