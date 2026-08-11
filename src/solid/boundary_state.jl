#############################################################################################
## Boundary state — solid-side computation of every body's boundary-node positions and
## velocities from a system's body state. The FSI subsystem reads these as input to the
## immersed-boundary kernels, but the math is purely solid-side rigid-body kinematics.
##
## Three functions, all dispatched on `::SolidSystem` so concrete subtypes inherit:
##   - `calculate_boundary_state(system, state)`
##   - `calculate_boundary_state_jacobian(system, state)`
##   - `calculate_boundary_velocity_vjp_jacobian(system, state, dual; is_midpoint_state)`
##
## `NoSystem` overloads at the bottom return empty vectors / zero-sized matrices so a tank
## with no bluff body or no swimmer requires no special-casing on the consumer side.
##
## Boundary state layout (mirroring the legacy SingleRigidBody / MultiRigidBody layout):
##   [pos_x1..pos_xN, pos_y1..pos_yN, vel_x1..vel_xN, vel_y1..vel_yN]
## with nodes from each body concatenated in body-list order.
#############################################################################################

function calculate_boundary_state(
    system::SolidSystem,
    system_or_body_state::AbstractVector,
)
    body_state = if length(system_or_body_state) == system.n_body_states
        system_or_body_state
    elseif length(system_or_body_state) == system.n_states
        system_or_body_state[system.body_state_indices]
    else
        error("calculate_boundary_state: unexpected state length $(length(system_or_body_state))")
    end

    configuration = body_state[system.configuration_indices]
    velocity = body_state[system.velocity_indices]

    total_n_nodes = 0
    for body in system.bodies
        body isa RigidBody && (total_n_nodes += body.n_boundary_nodes)
    end

    T = promote_type(eltype(body_state), _system_param_type(system))
    pos_x = Vector{T}(undef, total_n_nodes)
    pos_y = Vector{T}(undef, total_n_nodes)
    vel_x = Vector{T}(undef, total_n_nodes)
    vel_y = Vector{T}(undef, total_n_nodes)

    node_offset = 0
    for (i, body) in enumerate(system.bodies)
        body isa RigidBody || continue
        n_nodes = body.n_boundary_nodes

        cx = configuration[3 * (i - 1) + 1]
        cy = configuration[3 * (i - 1) + 2]
        θ = configuration[3 * (i - 1) + 3]
        vcx = velocity[3 * (i - 1) + 1]
        vcy = velocity[3 * (i - 1) + 2]
        ω = velocity[3 * (i - 1) + 3]

        cθ, sθ = cos(θ), sin(θ)
        xs_local, ys_local, _, _ = generate_boundary_nodes(body.shape, n_nodes)

        @inbounds for k in 1:n_nodes
            lx = xs_local[k] + body.com_offset[1]
            ly = ys_local[k] + body.com_offset[2]

            rx = cθ * lx - sθ * ly + cx
            ry = sθ * lx + cθ * ly + cy

            vrx = vcx - ω * (ry - cy)
            vry = vcy + ω * (rx - cx)

            pos_x[node_offset + k] = rx
            pos_y[node_offset + k] = ry
            vel_x[node_offset + k] = vrx
            vel_y[node_offset + k] = vry
        end

        node_offset += n_nodes
    end

    return vcat(pos_x, pos_y, vel_x, vel_y)
end

function calculate_boundary_state_jacobian(
    system::SolidSystem,
    system_or_body_state::AbstractVector,
)
    is_body_state_input = if length(system_or_body_state) == system.n_body_states
        true
    elseif length(system_or_body_state) == system.n_states
        false
    else
        error("calculate_boundary_state_jacobian: unexpected state length $(length(system_or_body_state))")
    end

    ∂boundary_state_∂body_state = ForwardDiff.jacobian(
        bs -> calculate_boundary_state(system, bs),
        is_body_state_input ? collect(system_or_body_state) : collect(system_or_body_state[system.body_state_indices]),
    )

    p_current = collect_differentiable_params(system)
    ∂boundary_state_∂system_params = ForwardDiff.jacobian(
        p -> calculate_boundary_state(inject_differentiable_params(system, p), system_or_body_state),
        p_current,
    )

    ∂boundary_state_∂state = if is_body_state_input
        ∂boundary_state_∂body_state
    else
        T = eltype(∂boundary_state_∂body_state)
        full = zeros(T, size(∂boundary_state_∂body_state, 1), system.n_states)
        full[:, system.body_state_indices] .= ∂boundary_state_∂body_state
        full
    end

    return ∂boundary_state_∂state, ∂boundary_state_∂system_params
end

function calculate_boundary_velocity_vjp_jacobian(
    system::SolidSystem,
    system_or_body_state::AbstractVector,
    dual_vector::AbstractVector;
    is_midpoint_state::Bool = true,
)
    p_current = collect_differentiable_params(system)

    body_state = if length(system_or_body_state) == system.n_body_states
        collect(system_or_body_state)
    elseif length(system_or_body_state) == system.n_states
        collect(system_or_body_state[system.body_state_indices])
    else
        error("calculate_boundary_velocity_vjp_jacobian: unexpected state length $(length(system_or_body_state))")
    end

    return ForwardDiff.jacobian(
        p -> _boundary_velocity_body_vjp(
            inject_differentiable_params(system, p),
            body_state,
            dual_vector;
            is_midpoint_state=is_midpoint_state,
        ),
        p_current,
    )
end

# Computes `(∂midpoint_bv/∂body_velocity)^T * dual`, a vector of length `3 * n_bodies`
# (per-body [vcx, vcy, ω] for 2D rigid bodies).
#
# The `is_midpoint_state` flag distinguishes two semantics for the input `body_state`:
#
#   true:  `body_state` IS the midpoint state. The caller asserts no midpoint
#          chain rule is needed. Result is `(∂bv_at_body_state/∂body_velocity)^T * dual`.
#
#   false: `body_state` is already the midpoint state (caller computed it externally),
#          but `midpoint_config` depends on `body_velocity` through
#          `midpoint_config = config - 0.5*dt*velocity`. An extra chain-rule term is
#          needed on the ω row because `R(midpoint_θ)` depends on `ω` via
#          `midpoint_θ = θ - 0.5*dt*ω`. The derivation:
#
#            midpoint_bv[k] = v_com + ω * R'(midpoint_θ) * l[k]
#            ∂midpoint_bv[k]/∂ω = R'(midpoint_θ) * l[k]
#                               + ω * R''(midpoint_θ) * (-0.5*dt) * l[k]
#                               = R'(midpoint_θ) * l[k]
#                               + 0.5*dt*ω * R(midpoint_θ) * l[k]     (since R'' = -R)
#
#          So for the ω row: result[3] = sum_ω + 0.5*dt*ω * Σ_k (dual · R(θ)·l[k]).
function _boundary_velocity_body_vjp(
    system::SolidSystem,
    body_state::AbstractVector,
    boundary_velocity_dual::AbstractVector;
    is_midpoint_state::Bool = true,
)
    configuration = body_state[system.configuration_indices]
    velocity = body_state[system.velocity_indices]
    dt = system.time_step

    n_boundary_nodes_total = 0
    for body in system.bodies
        body isa RigidBody && (n_boundary_nodes_total += body.n_boundary_nodes)
    end

    T = promote_type(
        eltype(configuration), eltype(velocity),
        eltype(boundary_velocity_dual), _system_param_type(system),
    )
    result = zeros(T, 3 * system.n_bodies)

    node_offset = 0
    for (i, body) in enumerate(system.bodies)
        body isa RigidBody || continue
        n_nodes = body.n_boundary_nodes

        θ = configuration[3 * (i - 1) + 3]
        ω = velocity[3 * (i - 1) + 3]
        cθ, sθ = cos(θ), sin(θ)
        xs_local, ys_local, _, _ = generate_boundary_nodes(body.shape, n_nodes)

        sum_vcx = zero(T)
        sum_vcy = zero(T)
        sum_ω = zero(T)
        chain_sum = zero(T)
        for k in 1:n_nodes
            vx_dual = boundary_velocity_dual[node_offset + k]
            vy_dual = boundary_velocity_dual[n_boundary_nodes_total + node_offset + k]

            lx = xs_local[k] + body.com_offset[1]
            ly = ys_local[k] + body.com_offset[2]

            sum_vcx += vx_dual
            sum_vcy += vy_dual
            # Direct term: dual · ∂(ω × R(θ)·l)/∂ω = dual · R'(θ)·l
            sum_ω += vx_dual * (-(sθ * lx + cθ * ly)) + vy_dual * (cθ * lx - sθ * ly)
            # Chain-rule term factor: dual · R(θ)·l (scaled by 0.5*dt*ω below)
            chain_sum += vx_dual * (cθ * lx - sθ * ly) + vy_dual * (sθ * lx + cθ * ly)
        end

        result[3 * (i - 1) + 1] = sum_vcx
        result[3 * (i - 1) + 2] = sum_vcy
        result[3 * (i - 1) + 3] = if is_midpoint_state
            sum_ω
        else
            sum_ω + 0.5 * dt * ω * chain_sum
        end

        node_offset += n_nodes
    end

    return result
end

function _boundary_velocity_body_vjp_jacobian(
    system::SolidSystem,
    body_state::AbstractVector,
    boundary_velocity_dual::AbstractVector;
    is_midpoint_state::Bool = true,
)
    configuration = body_state[system.configuration_indices]
    velocity = body_state[system.velocity_indices]
    dt = system.time_step
    n_v = system.n_velocities
    n_bs = system.n_body_states

    n_boundary_nodes_total = 0
    for body in system.bodies
        body isa RigidBody && (n_boundary_nodes_total += body.n_boundary_nodes)
    end

    T = promote_type(
        eltype(configuration), eltype(velocity),
        eltype(boundary_velocity_dual), _system_param_type(system),
    )
    J = zeros(T, n_v, n_bs)

    node_offset = 0
    for (i, body) in enumerate(system.bodies)
        body isa RigidBody || continue
        n_nodes = body.n_boundary_nodes
        qi = 3 * (i - 1) + 3
        vi = system.n_configurations + 3 * (i - 1) + 3
        row = 3 * (i - 1) + 3

        θ = configuration[3 * (i - 1) + 3]
        ω = velocity[3 * (i - 1) + 3]
        cθ, sθ = cos(θ), sin(θ)
        xs_local, ys_local, _, _ = generate_boundary_nodes(body.shape, n_nodes)

        sum_ω = zero(T)
        chain_sum = zero(T)
        for k in 1:n_nodes
            d_x = boundary_velocity_dual[node_offset + k]
            d_y = boundary_velocity_dual[n_boundary_nodes_total + node_offset + k]
            lx = xs_local[k] + body.com_offset[1]
            ly = ys_local[k] + body.com_offset[2]
            sum_ω += d_x * (-(sθ * lx + cθ * ly)) + d_y * (cθ * lx - sθ * ly)
            chain_sum += d_x * (cθ * lx - sθ * ly) + d_y * (sθ * lx + cθ * ly)
        end

        if is_midpoint_state
            J[row, qi] = -chain_sum
        else
            J[row, qi] = -chain_sum + 0.5 * dt * ω * sum_ω
            J[row, vi] = 0.5 * dt * chain_sum
        end

        node_offset += n_nodes
    end

    return J
end

# --- NoSystem overloads: return empty vectors / zero-sized matrices ---

function calculate_boundary_state(::NoSystem, ::AbstractVector)
    return Float64[]
end

function calculate_boundary_state_jacobian(ns::NoSystem, system_or_body_state::AbstractVector)
    return (zeros(0, length(system_or_body_state)), zeros(0, 0))
end

function calculate_boundary_velocity_vjp_jacobian(
    ::NoSystem,
    ::AbstractVector,
    ::AbstractVector;
    is_midpoint_state::Bool = true,
)
    return zeros(0, 0)
end

@testitem "Boundary state Pendulum" begin
    using Aquarium
    using ForwardDiff

    system = Pendulum(0.01; bar_length=1.0, mass=2.0, moi=0.1,
                      hinge_position=[0.0, 0.0], n_boundary_nodes=6,
                      ib_method=:original)

    # Hanging equilibrium: θ=0 means bar points downward from hinge at origin
    state = zeros(system.n_states)
    bs = calculate_boundary_state(system, state)

    n_nodes = system.topology.n_boundary_nodes
    @test length(bs) == 4 * n_nodes

    pos_x = bs[1:n_nodes]
    pos_y = bs[n_nodes+1:2*n_nodes]
    vel_x = bs[2*n_nodes+1:3*n_nodes]
    vel_y = bs[3*n_nodes+1:4*n_nodes]

    # At rest: all velocities should be zero
    @test all(vel_x .== 0.0)
    @test all(vel_y .== 0.0)

    # Jacobian vs ForwardDiff
    body_state = state[system.body_state_indices]
    J_state, J_params = calculate_boundary_state_jacobian(system, body_state)
    J_fd = ForwardDiff.jacobian(
        bs_in -> calculate_boundary_state(system, bs_in), body_state)
    @test J_state ≈ J_fd atol=1e-10

    # VJP Jacobian
    dual = randn(2 * n_nodes)
    J_vjp = calculate_boundary_velocity_vjp_jacobian(system, body_state, dual)
    @test size(J_vjp, 1) == system.n_velocities
end

@testitem "Boundary state DoublePendulum" begin
    using Aquarium
    using ForwardDiff

    system = DoublePendulum(0.01;
        bar_lengths=[1.0, 0.8], masses=[2.0, 1.5], mois=[0.1, 0.08],
        hinge_position=[0.0, 0.0], n_boundary_nodes_per_link=6,
        ib_method=:original)

    n_nodes = system.topology.n_boundary_nodes
    @test n_nodes == 12  # 6 per body × 2 bodies

    state = zeros(system.n_states)
    bs = calculate_boundary_state(system, state)
    @test length(bs) == 4 * n_nodes

    # Jacobian vs ForwardDiff (multi-body path)
    body_state = state[system.body_state_indices]
    J_state, J_params = calculate_boundary_state_jacobian(system, body_state)
    J_fd = ForwardDiff.jacobian(
        bs_in -> calculate_boundary_state(system, bs_in), body_state)
    @test J_state ≈ J_fd atol=1e-10

    # VJP Jacobian
    dual = randn(2 * n_nodes)
    J_vjp = calculate_boundary_velocity_vjp_jacobian(system, body_state, dual)
    @test size(J_vjp, 1) == system.n_velocities
    n_params = length(collect_differentiable_params(system))
    @test size(J_vjp, 2) == n_params
end

@testitem "Boundary state NoSystem" begin
    using Aquarium
    ns = NoSystem()
    state = Float64[]

    bs = calculate_boundary_state(ns, state)
    @test bs == Float64[]

    J_state, J_params = calculate_boundary_state_jacobian(ns, state)
    @test size(J_state) == (0, 0)
    @test size(J_params) == (0, 0)

    J_vjp = calculate_boundary_velocity_vjp_jacobian(ns, state, Float64[])
    @test size(J_vjp) == (0, 0)
end

@testitem "_boundary_velocity_body_vjp midpoint chain-rule" begin
    using Aquarium
    # Regression test for the bug where `_boundary_velocity_body_vjp` ignored
    # the midpoint chain-rule term when `is_midpoint_state=false`, causing a
    # ~5e-6 discrepancy in the angular-velocity row vs the true body-velocity
    # VJP computed by `calculate_no_slip_constraint_vjp`.
    #
    # Ground truth: `calculate_no_slip_constraint_vjp` constructs
    # `∂midpoint_bv/∂body_state[:, velocity_indices]` via the full chain rule
    # (through `calculate_midpoint_state`) and contracts with the dual. That
    # must match `-_boundary_velocity_body_vjp(system, midpoint_bs, dual;
    # is_midpoint_state=false)` at the velocity indices.
    using Random
    using Aquarium: _boundary_velocity_body_vjp, calculate_no_slip_constraint_vjp,
                          calculate_midpoint_state

    Random.seed!(0)

    @testset "single body (FreeDisc)" begin
        fluid = Fluid(0.01;
            density=1.0, dynamic_viscosity=0.01,
            boundary_velocity=[0.0, 0.0],
            grid_size=(10, 10), grid_dimensions=(1.0, 1.0),
            boundary_condition_type=:wall,
        )
        system = FreeDisc(0.01; radius=0.12, mass=1.0, moi=0.5, n_boundary_nodes=8,
                          ib_method=:original)

        config = [0.47, 0.53, 0.31]
        vel = [0.02, -0.015, 0.03]  # nonzero angular velocity required to trigger the extra term
        fluid_vel = 0.01 .* randn(fluid.n_velocities)
        body_state = vcat(config, vel)
        dual = 0.1 .* randn(2 * system.topology.n_boundary_nodes)

        # is_midpoint_state=false path (the one the dynamics use)
        _, body_vjp = calculate_no_slip_constraint_vjp(
            fluid, system, fluid_vel, config, vel, dual; is_midpoint_state=false)
        midpoint_bs = calculate_midpoint_state(system, body_state)
        helper = _boundary_velocity_body_vjp(
            system, midpoint_bs, dual; is_midpoint_state=false)
        @test body_vjp[system.velocity_indices] ≈ -helper atol=1e-14

        # Prior to the fix, at least one component disagreed by ~5e-6 —
        # assert a tight bound to catch any regression.
        @test maximum(abs, body_vjp[system.velocity_indices] - (-helper)) < 1e-12

        # is_midpoint_state=true path (baseline — no chain rule needed)
        _, body_vjp_mp = calculate_no_slip_constraint_vjp(
            fluid, system, fluid_vel, config, vel, dual; is_midpoint_state=true)
        helper_mp = _boundary_velocity_body_vjp(
            system, body_state, dual; is_midpoint_state=true)
        @test body_vjp_mp[system.velocity_indices] ≈ -helper_mp atol=1e-14
    end

    @testset "multi body (DoublePendulum)" begin
        fluid = Fluid(0.01;
            density=1.0, dynamic_viscosity=0.01,
            boundary_velocity=[0.0, 0.0],
            grid_size=(10, 10), grid_dimensions=(1.0, 1.0),
            boundary_condition_type=:wall,
        )
        system = DoublePendulum(0.01;
            bar_lengths=[0.3, 0.2], masses=[1.0, 0.5], mois=[0.05, 0.02],
            hinge_position=[0.5, 0.7], n_boundary_nodes_per_link=4,
            ib_method=:original)

        config = zeros(system.n_configurations)
        config[1] = 0.47; config[2] = 0.48; config[3] = 0.12
        config[4] = 0.53; config[5] = 0.21; config[6] = -0.15
        vel = 0.02 .* randn(system.n_velocities)  # nonzero ω for both bodies
        fluid_vel = 0.01 .* randn(fluid.n_velocities)
        body_state = vcat(config, vel)
        dual = 0.1 .* randn(2 * system.topology.n_boundary_nodes)

        _, body_vjp = calculate_no_slip_constraint_vjp(
            fluid, system, fluid_vel, config, vel, dual; is_midpoint_state=false)
        midpoint_bs = calculate_midpoint_state(system, body_state)
        helper = _boundary_velocity_body_vjp(
            system, midpoint_bs, dual; is_midpoint_state=false)
        @test body_vjp[system.velocity_indices] ≈ -helper atol=1e-14
        @test maximum(abs, body_vjp[system.velocity_indices] - (-helper)) < 1e-12

        _, body_vjp_mp = calculate_no_slip_constraint_vjp(
            fluid, system, fluid_vel, config, vel, dual; is_midpoint_state=true)
        helper_mp = _boundary_velocity_body_vjp(
            system, body_state, dual; is_midpoint_state=true)
        @test body_vjp_mp[system.velocity_indices] ≈ -helper_mp atol=1e-14
    end
end

@testitem "_boundary_velocity_body_vjp_jacobian analytical" begin
    using Aquarium
    using ForwardDiff
    using Random
    using Aquarium: _boundary_velocity_body_vjp, _boundary_velocity_body_vjp_jacobian

    @testset "FreeDisc midpoint=$mp" for mp in [true, false]
        Random.seed!(70)
        system = FreeDisc(0.01; radius=0.12, mass=1.0, moi=0.5,
            n_boundary_nodes=8, ib_method=:original)
        body_state = 0.05 .* randn(system.n_body_states)
        dual = 0.1 .* randn(2 * system.topology.n_boundary_nodes)
        J = _boundary_velocity_body_vjp_jacobian(system, body_state, dual; is_midpoint_state=mp)
        J_fd = ForwardDiff.jacobian(
            bs -> _boundary_velocity_body_vjp(system, bs, dual; is_midpoint_state=mp),
            body_state)
        @test J ≈ J_fd atol=1e-12
    end

    @testset "RExEel midpoint=$mp" for mp in [true, false]
        Random.seed!(71)
        n_links = 3
        system = RExEel(0.01, n_links; bar_lengths=fill(0.1, n_links),
            masses=fill(1.0, n_links), mois=fill(0.01, n_links),
            Kps=fill(50.0, n_links-1), Kds=fill(5.0, n_links-1),
            max_torques=fill(Inf, n_links-1),
            n_boundary_nodes_per_link=fill(3, n_links),
            ib_method=:weak_form, discrete_delta_kind=:three_point)
        body_state = 0.05 .* randn(system.n_body_states)
        dual = 0.1 .* randn(2 * system.topology.n_boundary_nodes)
        J = _boundary_velocity_body_vjp_jacobian(system, body_state, dual; is_midpoint_state=mp)
        J_fd = ForwardDiff.jacobian(
            bs -> _boundary_velocity_body_vjp(system, bs, dual; is_midpoint_state=mp),
            body_state)
        @test J ≈ J_fd atol=1e-12
    end
end
