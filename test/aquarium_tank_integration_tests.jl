@testmodule TankSetup begin

using Aquarium
using ForwardDiff
using LinearAlgebra
using Random

# Physical parameters matching the old AquariumTank_test.jl
const time_step = 0.01
const gravity_constant = 98.0

# ---------- Fluid ----------
const fluid = Fluid(time_step;
    density=1.0,
    dynamic_viscosity=0.01,
    boundary_velocity=[0.01, 0.0],
    grid_size=(20, 20),
    grid_dimensions=(1.0, 1.0),
    boundary_condition_type=:freestream,
    gravity_constant=gravity_constant,
)

# ---------- FreeDisc bluff body ----------
# Pre-compute mass/moi from old params: density=1.0, diameter=0.1 → radius=0.05
const disc_radius = 0.05
const disc_density = 1.0
const disc_mass = π * disc_density * disc_radius^2
const disc_moi = 0.5 * disc_mass * disc_radius^2

const bluff_body = FreeDisc(time_step;
    radius=disc_radius,
    mass=disc_mass,
    moi=disc_moi,
    n_boundary_nodes=6,
    ib_method=:weak_form,
    discrete_delta_kind=:three_point,
    gravity=[0.0, -gravity_constant],
)

# ---------- RExEel swimmer ----------
const n_links = 3
const bar_lengths = fill(0.1, n_links)
const masses = fill(2.0, n_links)
const mois = (1 / 12) .* masses .* (bar_lengths .^ 2)

# Reverse-engineer raw gains so xc330m288t_gains yields effective Kp=Kd=100.0
const stall_torque = 9.3e6
const encoder_resolution = 4096
const control_loop_time = 0.001
const pwm_to_torque = stall_torque / 885
const Kp_raw = 100.0 / ((encoder_resolution / (2π)) * pwm_to_torque / 128)
const Kd_raw = 100.0 / (control_loop_time * (encoder_resolution / (2π)) * pwm_to_torque / 16)

# Verify effective gains
const Kp_eff, Kd_eff = xc330m288t_gains(Kp_raw=Kp_raw, Kd_raw=Kd_raw)
@assert Kp_eff ≈ 100.0 "Expected Kp_eff ≈ 100.0, got $Kp_eff"
@assert Kd_eff ≈ 100.0 "Expected Kd_eff ≈ 100.0, got $Kd_eff"

const swimmer = RExEel(time_step, n_links;
    bar_lengths=bar_lengths,
    masses=masses,
    mois=mois,
    Kps=fill(Kp_raw, n_links - 1),
    Kds=fill(Kd_raw, n_links - 1),
    max_torques=fill(Inf, n_links - 1),
    n_boundary_nodes_per_link=fill(3, n_links),
    ib_method=:weak_form,
    discrete_delta_kind=:three_point,
    gravity=[0.0, -gravity_constant],
    actuation_mode=:pd,
)

# ---------- AquariumTank ----------
const tank = AquariumTank(fluid, bluff_body, swimmer)

# ---------- Initial states ----------
Random.seed!(42)

# Minimal coordinates: [x1, y1, θ1, θ_rel_2, θ_rel_3]
const q_min_config = [0.5, 0.5, 0.0, 0.0, 0.0]
const v_min = 0.01 .* randn(n_links + 2)

# Maximal config via kinematic chain
const maximal_config = rex_eel_maximal_from_minimal(swimmer, q_min_config, n_links)

# Velocity conversion: Jacobian of config conversion applied to minimal velocity
const config_jacobian = ForwardDiff.jacobian(
    q -> rex_eel_maximal_from_minimal(swimmer, q, n_links), q_min_config
)
const maximal_velocity = config_jacobian * v_min

# Full swimmer body state: [config; velocity]
const swimmer_initial_body_state = vcat(maximal_config, maximal_velocity)

# Fluid initial velocity (uniform boundary velocity)
const fluid_initial_velocity = repeat(fluid.boundary_velocity, fluid.n_velocities ÷ 2)

# Full aquarium initial state
const aquarium_state_0 = initialize_aquarium_state(tank, fluid_initial_velocity, swimmer_initial_body_state)

# Bluff body state trajectory (stationary near center)
const bluff_body_state_0 = [0.47, 0.53, 0.31, 0.02, -0.015, 0.03]

# Control params (2 joints × 2 inputs = 4)
Random.seed!(42)
const swimmer_control_params = deg2rad.(5.0 .* randn(swimmer.n_control_inputs))

end  # TankSetup


# Gradient test setup — reuses 20×20 TankSetup physics, adds shared objective functions.
@testmodule GradientSetup begin

using Aquarium
using ForwardDiff
using LinearAlgebra
using Random

# Reuse the same 20×20 physical setup as TankSetup
const time_step = 0.01
const gravity_constant = 98.0

const fluid = Fluid(time_step;
    density=1.0, dynamic_viscosity=0.01,
    boundary_velocity=[0.01, 0.0],
    grid_size=(20, 20), grid_dimensions=(1.0, 1.0),
    boundary_condition_type=:freestream,
    gravity_constant=gravity_constant,
)

const disc_radius = 0.05
const disc_density = 1.0
const disc_mass = π * disc_density * disc_radius^2
const disc_moi = 0.5 * disc_mass * disc_radius^2

const bluff_body = FreeDisc(time_step;
    radius=disc_radius, mass=disc_mass, moi=disc_moi,
    n_boundary_nodes=6, ib_method=:weak_form,
    discrete_delta_kind=:three_point,
    gravity=[0.0, -gravity_constant],
)

const n_links = 3
const bar_lengths = fill(0.1, n_links)
const masses = fill(2.0, n_links)
const mois = (1 / 12) .* masses .* (bar_lengths .^ 2)

const stall_torque = 9.3e6
const encoder_resolution = 4096
const control_loop_time = 0.001
const pwm_to_torque = stall_torque / 885
const Kp_raw = 100.0 / ((encoder_resolution / (2π)) * pwm_to_torque / 128)
const Kd_raw = 100.0 / (control_loop_time * (encoder_resolution / (2π)) * pwm_to_torque / 16)

const swimmer = RExEel(time_step, n_links;
    bar_lengths=bar_lengths, masses=masses, mois=mois,
    Kps=fill(Kp_raw, n_links - 1), Kds=fill(Kd_raw, n_links - 1),
    max_torques=fill(Inf, n_links - 1),
    n_boundary_nodes_per_link=fill(3, n_links),
    ib_method=:weak_form, discrete_delta_kind=:three_point,
    gravity=[0.0, -gravity_constant],
    actuation_mode=:pd,
)

const tank = AquariumTank(fluid, bluff_body, swimmer)

Random.seed!(42)

const q_min_config = [0.5, 0.5, 0.0, 0.0, 0.0]
const v_min = 0.01 .* randn(n_links + 2)

const maximal_config = rex_eel_maximal_from_minimal(swimmer, q_min_config, n_links)
const config_jacobian = ForwardDiff.jacobian(
    q -> rex_eel_maximal_from_minimal(swimmer, q, n_links), q_min_config)
const maximal_velocity = config_jacobian * v_min
const swimmer_initial_body_state = vcat(maximal_config, maximal_velocity)

const fluid_initial_velocity = repeat(fluid.boundary_velocity, fluid.n_velocities ÷ 2)
const aquarium_state_0 = initialize_aquarium_state(tank, fluid_initial_velocity, swimmer_initial_body_state)

const bluff_body_state_0 = [0.47, 0.53, 0.31, 0.02, -0.015, 0.03]

Random.seed!(42)
const swimmer_control_params = deg2rad.(5.0 .* randn(swimmer.n_control_inputs))

# Shared objective functions for all gradient tests
const calc_stage = (tank, t, x, bb, u) ->
    sum(x) + sum(bb) + sum(u) +
    t * (tank.fluid.density + tank.swimmer.bodies[1].mass + tank.bluff_body.bodies[1].mass)
const calc_terminal = (tank, t, x, bb) -> sum(x .^ 2) + sum(bb .^ 2)
const calc_bb_state = (bluff_body, t, params; bluff_body_params=collect_differentiable_params(bluff_body)) -> params

const n_steps = 3
const final_time = n_steps * time_step

end  # GradientSetup


# =========================================================================================
# Slice 1: AquariumTank Construction
# =========================================================================================

@testitem "AquariumTank Construction" setup=[TankSetup] begin
    using Aquarium
    using LinearAlgebra

    tank = TankSetup.tank
    fluid = TankSetup.fluid
    bluff_body = TankSetup.bluff_body
    swimmer = TankSetup.swimmer

    # Field identity
    @test tank.fluid === fluid
    @test tank.bluff_body === bluff_body
    @test tank.swimmer === swimmer

    # Time step consistency
    @test tank.time_step == fluid.time_step
    @test tank.time_step == bluff_body.time_step
    @test tank.time_step == swimmer.time_step

    # Constraint counting
    @test tank.n_bluff_body_no_slip_constraints == bluff_body.topology.n_no_slip_constraints
    @test tank.n_swimmer_no_slip_constraints == swimmer.topology.n_no_slip_constraints
    @test tank.n_no_slip_constraints == tank.n_bluff_body_no_slip_constraints + tank.n_swimmer_no_slip_constraints

    # State counting
    @test tank.n_fluid_velocities == fluid.n_velocities
    @test tank.n_swimmer_body_states == swimmer.n_body_states
    @test tank.n_fluid_constraints == fluid.n_constraints
    @test tank.n_swimmer_constraints == swimmer.n_constraints
    @test tank.n_states == (tank.n_fluid_velocities + tank.n_swimmer_body_states +
                            tank.n_fluid_constraints + tank.n_swimmer_constraints +
                            tank.n_bluff_body_no_slip_duals + tank.n_swimmer_no_slip_duals)

    # State index partitioning: non-overlapping and correct ordering
    all_indices = vcat(
        tank.fluid_velocity_indices,
        tank.swimmer_body_state_indices,
        tank.fluid_dual_indices,
        tank.swimmer_dual_indices,
        tank.bluff_body_no_slip_dual_indices,
        tank.swimmer_no_slip_dual_indices,
    )
    @test sort(all_indices) == collect(1:tank.n_states)
    @test length(unique(all_indices)) == tank.n_states

    # Correct ordering: fluid velocity < swimmer body state < fluid dual < swimmer dual < BB no-slip < swimmer no-slip
    if !isempty(tank.fluid_velocity_indices) && !isempty(tank.swimmer_body_state_indices)
        @test maximum(tank.fluid_velocity_indices) < minimum(tank.swimmer_body_state_indices)
    end
    if !isempty(tank.swimmer_body_state_indices) && !isempty(tank.fluid_dual_indices)
        @test maximum(tank.swimmer_body_state_indices) < minimum(tank.fluid_dual_indices)
    end
end


# =========================================================================================
# Slice 2: State Extraction
# =========================================================================================

@testitem "State Extraction" setup=[TankSetup] begin
    using Aquarium

    tank = TankSetup.tank
    aquarium_state = TankSetup.aquarium_state_0

    # All 8 extract functions return correct sizes
    @test length(extract_fluid_state(tank, aquarium_state)) == tank.n_fluid_states
    @test length(extract_swimmer_state(tank, aquarium_state)) == tank.n_swimmer_states
    @test length(extract_fluid_velocity(tank, aquarium_state)) == tank.n_fluid_velocities
    @test length(extract_swimmer_body_state(tank, aquarium_state)) == tank.n_swimmer_body_states
    @test length(extract_fluid_dual(tank, aquarium_state)) == tank.n_fluid_constraints
    @test length(extract_swimmer_dual(tank, aquarium_state)) == tank.n_swimmer_constraints
    @test length(extract_bluff_body_no_slip_dual(tank, aquarium_state)) == tank.n_bluff_body_no_slip_duals
    @test length(extract_swimmer_no_slip_dual(tank, aquarium_state)) == tank.n_swimmer_no_slip_duals

    # Extracted values match direct indexing
    @test extract_fluid_velocity(tank, aquarium_state) == aquarium_state[tank.fluid_velocity_indices]
    @test extract_swimmer_body_state(tank, aquarium_state) == aquarium_state[tank.swimmer_body_state_indices]
    @test extract_fluid_dual(tank, aquarium_state) == aquarium_state[tank.fluid_dual_indices]
    @test extract_swimmer_dual(tank, aquarium_state) == aquarium_state[tank.swimmer_dual_indices]
    @test extract_bluff_body_no_slip_dual(tank, aquarium_state) == aquarium_state[tank.bluff_body_no_slip_dual_indices]
    @test extract_swimmer_no_slip_dual(tank, aquarium_state) == aquarium_state[tank.swimmer_no_slip_dual_indices]
end


# =========================================================================================
# Slice 2: Initialize Aquarium State
# =========================================================================================

@testitem "Initialize Aquarium State" setup=[TankSetup] begin
    using Aquarium

    tank = TankSetup.tank
    fluid_vel = TankSetup.fluid_initial_velocity
    swimmer_bs = TankSetup.swimmer_initial_body_state
    aquarium_state = TankSetup.aquarium_state_0

    # Correct total size
    @test length(aquarium_state) == tank.n_states

    # Fluid velocities placed correctly
    @test extract_fluid_velocity(tank, aquarium_state) ≈ fluid_vel[1:tank.n_fluid_velocities]

    # Swimmer body state placed correctly
    @test extract_swimmer_body_state(tank, aquarium_state) ≈ swimmer_bs

    # All dual variables are zero
    @test all(extract_fluid_dual(tank, aquarium_state) .== 0.0)
    @test all(extract_swimmer_dual(tank, aquarium_state) .== 0.0)
    @test all(extract_bluff_body_no_slip_dual(tank, aquarium_state) .== 0.0)
    @test all(extract_swimmer_no_slip_dual(tank, aquarium_state) .== 0.0)
end


# =========================================================================================
# Slice 2: NoSystem Configurations
# =========================================================================================

@testitem "NoSystem Configurations" setup=[TankSetup] begin
    using Aquarium

    fluid = TankSetup.fluid

    @testset "NoSystem as bluff body" begin
        tank_no_bb = AquariumTank(fluid, NoSystem(), TankSetup.swimmer)
        @test tank_no_bb.n_bluff_body_no_slip_constraints == 0
        @test isempty(tank_no_bb.bluff_body_no_slip_dual_indices)

        # Residual evaluates without error
        state = zeros(tank_no_bb.n_states)
        bb_state = zeros(0)
        r = calculate_aquarium_dynamics_residual(tank_no_bb, state, state, bb_state)
        @test length(r) == tank_no_bb.n_states
        @test all(isfinite, r)
    end

    @testset "NoSystem as swimmer" begin
        tank_no_sw = AquariumTank(fluid, TankSetup.bluff_body, NoSystem())
        @test tank_no_sw.n_swimmer_no_slip_constraints == 0
        @test tank_no_sw.n_swimmer_body_states == 0
        @test isempty(tank_no_sw.swimmer_body_state_indices)
    end

    @testset "Both NoSystem" begin
        tank_fluid_only = AquariumTank(fluid, NoSystem(), NoSystem())
        @test tank_fluid_only.n_bluff_body_no_slip_constraints == 0
        @test tank_fluid_only.n_swimmer_no_slip_constraints == 0
        @test tank_fluid_only.n_swimmer_body_states == 0
        # State is fluid-only
        @test tank_fluid_only.n_states == tank_fluid_only.n_fluid_velocities + tank_fluid_only.n_fluid_constraints
    end

    @testset "Initialization with NoSystem" begin
        tank_no_sw = AquariumTank(fluid, TankSetup.bluff_body, NoSystem())
        fluid_vel = TankSetup.fluid_initial_velocity
        state = initialize_aquarium_state(tank_no_sw, fluid_vel)
        @test length(state) == tank_no_sw.n_states
        @test all(isfinite, state)
    end
end


# =========================================================================================
# Slice 3: Dynamics Residual Evaluation
# =========================================================================================

@testitem "Dynamics Residual" setup=[TankSetup] begin
    using Aquarium
    using Random

    tank = TankSetup.tank
    aquarium_state_0 = TankSetup.aquarium_state_0
    bb_state = TankSetup.bluff_body_state_0
    control = TankSetup.swimmer_control_params

    # Create a second state for midpoint evaluation
    Random.seed!(123)
    aquarium_state_1 = aquarium_state_0 .+ 0.001 .* randn(tank.n_states)

    # Endpoint residual (no control)
    r_endpoint = calculate_aquarium_dynamics_residual(
        tank, aquarium_state_1, aquarium_state_0, bb_state)
    @test length(r_endpoint) == tank.n_states
    @test all(isfinite, r_endpoint)

    # Endpoint residual with control
    r_with_control = calculate_aquarium_dynamics_residual(
        tank, aquarium_state_1, aquarium_state_0, bb_state, control)
    @test length(r_with_control) == tank.n_states
    @test all(isfinite, r_with_control)

    # Midpoint residual
    r_midpoint = calculate_aquarium_dynamics_residual(
        tank, aquarium_state_1, aquarium_state_0, bb_state, control;
        is_midpoint_state_bluff_body=true)
    @test length(r_midpoint) == tank.n_states
    @test all(isfinite, r_midpoint)

    # Endpoint and midpoint residuals should differ
    @test r_endpoint != r_with_control
    @test r_with_control != r_midpoint
end


# =========================================================================================
# Slice 3: Dynamics Jacobian Structure
# =========================================================================================

@testitem "Dynamics Jacobian Structure" setup=[TankSetup] begin
    using Aquarium
    using Random

    tank = TankSetup.tank
    fluid = TankSetup.fluid
    bluff_body = TankSetup.bluff_body
    swimmer = TankSetup.swimmer
    aquarium_state_0 = TankSetup.aquarium_state_0
    bb_state = TankSetup.bluff_body_state_0
    control = TankSetup.swimmer_control_params

    Random.seed!(123)
    aquarium_state_1 = aquarium_state_0 .+ 0.001 .* randn(tank.n_states)

    (∂D_∂xkp1, ∂D_∂xk, ∂D_∂uk, ∂D_∂fluid_props, ∂D_∂sw_params,
     ∂D_∂bb_params, ∂D_∂bb_state_kp1) = calculate_aquarium_dynamics_jacobian(
        tank, aquarium_state_1, aquarium_state_0, bb_state, control)

    n = tank.n_states

    # State Jacobians
    @test size(∂D_∂xkp1) == (n, n)
    @test size(∂D_∂xk) == (n, n)

    # Control Jacobian
    @test size(∂D_∂uk) == (n, length(control))

    # Fluid properties Jacobian (4 params: density, viscosity, bv_x, bv_y)
    @test size(∂D_∂fluid_props) == (n, length(collect_differentiable_params(fluid)))
    @test size(∂D_∂fluid_props, 2) == 4

    # Swimmer params Jacobian (21 params: 5 per body × 3 + 3 per joint × 2)
    @test size(∂D_∂sw_params) == (n, length(collect_differentiable_params(swimmer)))

    # Bluff body params Jacobian (5 params: mass, moi, com_x, com_y, radius)
    @test size(∂D_∂bb_params) == (n, length(collect_differentiable_params(bluff_body)))

    # Bluff body state Jacobian
    @test size(∂D_∂bb_state_kp1) == (n, bluff_body.n_body_states)
end


# =========================================================================================
# Slice 3: Jacobian vs FiniteDiff
# =========================================================================================

@testitem "Jacobian vs ForwardDiff and FiniteDiff" setup=[TankSetup] begin
    using Aquarium
    using ForwardDiff
    using FiniteDiff
    using Random
    using LinearAlgebra

    tank = TankSetup.tank
    fluid = TankSetup.fluid
    bluff_body = TankSetup.bluff_body
    swimmer = TankSetup.swimmer
    aquarium_state_0 = TankSetup.aquarium_state_0
    bb_state = TankSetup.bluff_body_state_0
    control = TankSetup.swimmer_control_params

    Random.seed!(123)
    aquarium_state_1 = aquarium_state_0 .+ 0.001 .* randn(tank.n_states)

    (∂D_∂xkp1, ∂D_∂xk, ∂D_∂uk, ∂D_∂fluid_props, ∂D_∂sw_params,
     ∂D_∂bb_params, ∂D_∂bb_state_kp1) = calculate_aquarium_dynamics_jacobian(
        tank, aquarium_state_1, aquarium_state_0, bb_state, control)

    # --- State/control Jacobians: ForwardDiff (exact AD) ---

    # ∂D/∂x_kp1 — the full KKT matrix
    ad_∂D_∂xkp1 = ForwardDiff.jacobian(
        x -> calculate_aquarium_dynamics_residual(tank, x, aquarium_state_0, bb_state, control),
        aquarium_state_1)
    @test Matrix(∂D_∂xkp1) ≈ ad_∂D_∂xkp1 atol=1e-6

    # ∂D/∂x_k
    ad_∂D_∂xk = ForwardDiff.jacobian(
        x -> calculate_aquarium_dynamics_residual(tank, aquarium_state_1, x, bb_state, control),
        aquarium_state_0)
    @test Matrix(∂D_∂xk) ≈ ad_∂D_∂xk atol=1e-10

    # ∂D/∂u_k
    ad_∂D_∂uk = ForwardDiff.jacobian(
        u -> calculate_aquarium_dynamics_residual(tank, aquarium_state_1, aquarium_state_0, bb_state, u),
        control)
    @test Matrix(∂D_∂uk) ≈ ad_∂D_∂uk atol=1e-10

    # ∂D/∂bluff_body_state_kp1
    ad_∂D_∂bb_state = ForwardDiff.jacobian(
        bb -> calculate_aquarium_dynamics_residual(tank, aquarium_state_1, aquarium_state_0, bb, control),
        bb_state)
    @test Matrix(∂D_∂bb_state_kp1) ≈ ad_∂D_∂bb_state atol=1e-10

    # --- Parameter Jacobians: FiniteDiff (inject_differentiable_params has Float64 barriers) ---

    # ∂D/∂fluid_props
    fd_∂D_∂fluid_props = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(fluid)) do p
        new_fluid = inject_differentiable_params(fluid, p)
        new_tank = rebuild_tank_with_fluid(tank, new_fluid)
        calculate_aquarium_dynamics_residual(new_tank, aquarium_state_1, aquarium_state_0, bb_state, control)
    end
    @test Matrix(∂D_∂fluid_props) ≈ fd_∂D_∂fluid_props rtol=1e-4

    # ∂D/∂swimmer_params
    fd_∂D_∂sw_params = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(swimmer)) do p
        new_sw = inject_differentiable_params(swimmer, p)
        new_tank = rebuild_tank_with_swimmer(tank, new_sw)
        calculate_aquarium_dynamics_residual(new_tank, aquarium_state_1, aquarium_state_0, bb_state, control)
    end
    @test Matrix(∂D_∂sw_params) ≈ fd_∂D_∂sw_params rtol=1e-4

    # ∂D/∂bluff_body_params
    fd_∂D_∂bb_params = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(bluff_body)) do p
        new_bb = inject_differentiable_params(bluff_body, p)
        new_tank = rebuild_tank_with_bluff_body(tank, new_bb)
        calculate_aquarium_dynamics_residual(new_tank, aquarium_state_1, aquarium_state_0, bb_state, control)
    end
    @test Matrix(∂D_∂bb_params) ≈ fd_∂D_∂bb_params rtol=1e-4
end


# =========================================================================================
# Slice 4: Simulation + Objective Computation
# =========================================================================================

@testitem "Simulation Objective Computation" setup=[TankSetup] begin
    using Aquarium

    tank = TankSetup.tank
    aquarium_state_0 = TankSetup.aquarium_state_0
    bb_state_0 = TankSetup.bluff_body_state_0
    control_params = TankSetup.swimmer_control_params

    n_steps = 3
    final_time = n_steps * tank.time_step

    # Objective functions that read params from tank struct tree.
    # Access fields directly (not collect_differentiable_params) so Dual types propagate.
    calc_stage = (tank, t, x, bb, u) ->
        sum(x) + sum(bb) + sum(u) +
        t * (tank.fluid.density +
             tank.swimmer.bodies[1].mass +
             tank.bluff_body.bodies[1].mass)

    calc_terminal = (tank, t, x, bb) ->
        sum(x .^ 2) + sum(bb .^ 2)

    # Bluff body state function (constant position)
    calc_bb_state = (bluff_body, t, params; bluff_body_params=collect_differentiable_params(bluff_body)) -> params

    trajectories = simulate_aquarium(
        tank, aquarium_state_0, final_time,
        bb_state_0,
        control_params;
        calculate_objective=true,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    # Simulation completes and returns correct keys
    @test haskey(trajectories, :time_traj)
    @test haskey(trajectories, :aquarium_state_traj)
    @test haskey(trajectories, :objective_value)
    @test haskey(trajectories, :objective_traj)

    # Trajectory dimensions
    @test length(trajectories[:time_traj]) == n_steps + 1
    @test length(trajectories[:aquarium_state_traj]) == n_steps + 1
    @test all(length(s) == tank.n_states for s in trajectories[:aquarium_state_traj])

    # Objective value matches manual sum of stage + terminal
    obj_traj = trajectories[:objective_traj]
    @test length(obj_traj) == n_steps + 1
    @test trajectories[:objective_value][1] ≈ sum(obj_traj)

    # Manual recomputation: stages 1..N-1, terminal at N
    time_traj = trajectories[:time_traj]
    state_traj = trajectories[:aquarium_state_traj]
    bb_traj = trajectories[:bluff_body_state_traj]
    ctrl_traj = trajectories[:control_traj]

    manual_obj = sum(k -> calc_stage(tank, time_traj[k], state_traj[k], bb_traj[k], ctrl_traj[k]), 1:n_steps)
    manual_obj += calc_terminal(tank, time_traj[end], state_traj[end], bb_traj[end])

    @test trajectories[:objective_value][1] ≈ manual_obj
end


# =========================================================================================
# Slice 5: Gradient wrt Fluid Properties
# =========================================================================================

@testitem "Gradient wrt Fluid Properties" setup=[GradientSetup] begin
    using Aquarium
    using FiniteDiff

    G = GradientSetup
    tank = G.tank;  fluid = G.fluid
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    swimmer_bs = G.swimmer_initial_body_state
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal;  calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    # Use a fixed initial state (zero fluid velocity) that does NOT depend on fluid params.
    # This avoids the initial-state Jacobian complication and isolates the dynamics gradient.
    x0 = initialize_aquarium_state(tank, zeros(fluid.n_velocities), swimmer_bs)

    trajectories = simulate_aquarium(
        tank, x0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_gradient_wrt_fluid_properties=true,
        calculate_gradient_wrt_swimmer_params=false,
        calculate_gradient_wrt_bluff_body_params=false,
        calculate_gradient_wrt_control_params=false,
        calculate_gradient_wrt_bluff_body_state_params=false,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    analytical_grad = trajectories[:objective_gradient_wrt_fluid_properties]
    @test length(analytical_grad) == 4
    @test all(isfinite, analytical_grad)

    # Finite-difference validation — same fixed initial state for each perturbation
    fd_grad = FiniteDiff.finite_difference_gradient(collect_differentiable_params(fluid)) do p
        nf = inject_differentiable_params(fluid, p)
        nt = rebuild_tank_with_fluid(tank, nf)
        traj = simulate_aquarium(
            nt, x0, final_time, bb_state_0, control_params;
            max_newton_iterations=50,
            calculate_objective=true,
            calculate_gradient_wrt_fluid_properties=false,
            calculate_gradient_wrt_swimmer_params=false,
            calculate_gradient_wrt_bluff_body_params=false,
            calculate_gradient_wrt_control_params=false,
            calculate_gradient_wrt_bluff_body_state_params=false,
            calculate_stage_objective=calc_stage,
            calculate_terminal_objective=calc_terminal,
            calculate_bluff_body_state_from_params=calc_bb_state,
        )
        traj[:objective_value][1]
    end

    @test analytical_grad ≈ fd_grad rtol=0.05
end


# =========================================================================================
# Slice 5: Gradient wrt Bluff Body State Params
# =========================================================================================

@testitem "Gradient wrt Bluff Body State Params" setup=[GradientSetup] begin
    using Aquarium
    using FiniteDiff

    G = GradientSetup
    tank = G.tank
    aquarium_state_0 = G.aquarium_state_0
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal
    final_time = G.final_time

    # Custom bluff body state function: prescribed sinusoidal motion
    calc_bb_state = (bluff_body, t, params; bluff_body_params=collect_differentiable_params(bluff_body)) ->
        [params[1] + 0.01*sin(t), params[2] + 0.01*cos(t), params[3],
         params[4], params[5], params[6]]

    trajectories = simulate_aquarium(
        tank, aquarium_state_0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_gradient_wrt_fluid_properties=false,
        calculate_gradient_wrt_swimmer_params=false,
        calculate_gradient_wrt_bluff_body_params=false,
        calculate_gradient_wrt_control_params=false,
        calculate_gradient_wrt_bluff_body_state_params=true,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    analytical_grad = trajectories[:objective_gradient_wrt_bluff_body_state_params]
    @test length(analytical_grad) == length(bb_state_0)
    @test all(isfinite, analytical_grad)

    # Finite-difference validation
    fd_grad = FiniteDiff.finite_difference_gradient(bb_state_0) do p
        traj = simulate_aquarium(
            tank, aquarium_state_0, final_time, p, control_params;
            max_newton_iterations=50,
            calculate_objective=true,
            calculate_gradient_wrt_fluid_properties=false,
            calculate_gradient_wrt_swimmer_params=false,
            calculate_gradient_wrt_bluff_body_params=false,
            calculate_gradient_wrt_control_params=false,
            calculate_gradient_wrt_bluff_body_state_params=false,
            calculate_stage_objective=calc_stage,
            calculate_terminal_objective=calc_terminal,
            calculate_bluff_body_state_from_params=calc_bb_state,
        )
        traj[:objective_value][1]
    end

    @test analytical_grad ≈ fd_grad rtol=0.05
end


# =========================================================================================
# Slice 5: Gradient wrt Control Params
# =========================================================================================

@testitem "Gradient wrt Control Params" setup=[GradientSetup] begin
    using Aquarium
    using FiniteDiff

    G = GradientSetup
    tank = G.tank
    aquarium_state_0 = G.aquarium_state_0
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal;  calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    trajectories = simulate_aquarium(
        tank, aquarium_state_0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_gradient_wrt_fluid_properties=false,
        calculate_gradient_wrt_swimmer_params=false,
        calculate_gradient_wrt_bluff_body_params=false,
        calculate_gradient_wrt_control_params=true,
        calculate_gradient_wrt_bluff_body_state_params=false,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    analytical_grad = trajectories[:objective_gradient_wrt_control_params]
    @test length(analytical_grad) == length(control_params)
    @test all(isfinite, analytical_grad)

    # Finite-difference validation
    fd_grad = FiniteDiff.finite_difference_gradient(control_params) do p
        traj = simulate_aquarium(
            tank, aquarium_state_0, final_time, bb_state_0, p;
            max_newton_iterations=50,
            calculate_objective=true,
            calculate_gradient_wrt_fluid_properties=false,
            calculate_gradient_wrt_swimmer_params=false,
            calculate_gradient_wrt_bluff_body_params=false,
            calculate_gradient_wrt_control_params=false,
            calculate_gradient_wrt_bluff_body_state_params=false,
            calculate_stage_objective=calc_stage,
            calculate_terminal_objective=calc_terminal,
            calculate_bluff_body_state_from_params=calc_bb_state,
        )
        traj[:objective_value][1]
    end

    @test analytical_grad ≈ fd_grad rtol=0.05
end


# =========================================================================================
# Slice 6: Gradient wrt Swimmer Params
# =========================================================================================

@testitem "Gradient wrt Swimmer Params" setup=[GradientSetup] begin
    using Aquarium
    using FiniteDiff
    using ForwardDiff

    G = GradientSetup
    tank = G.tank;  fluid = G.fluid;  swimmer = G.swimmer
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    q_min_config = G.q_min_config
    v_min = G.v_min
    n_links = G.n_links
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal;  calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    # Initial-state Jacobian wrt swimmer params
    sw_bs = G.swimmer_initial_body_state
    initial_state_swimmer_jac = ForwardDiff.jacobian(collect_differentiable_params(swimmer)) do p
        new_sw = inject_differentiable_params(swimmer, p)
        mc = rex_eel_maximal_from_minimal(new_sw, q_min_config, n_links)
        J = ForwardDiff.jacobian(q -> rex_eel_maximal_from_minimal(new_sw, q, n_links), q_min_config)
        mv = J * v_min
        new_bs = vcat(mc, mv)
        initialize_aquarium_state(tank, zeros(fluid.n_velocities), new_bs)
    end

    x0 = initialize_aquarium_state(tank, zeros(fluid.n_velocities), sw_bs)

    trajectories = simulate_aquarium(
        tank, x0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_gradient_wrt_fluid_properties=false,
        calculate_gradient_wrt_swimmer_params=true,
        calculate_gradient_wrt_bluff_body_params=false,
        calculate_gradient_wrt_control_params=false,
        calculate_gradient_wrt_bluff_body_state_params=false,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
        initial_aquarium_state_swimmer_params_jacobian=initial_state_swimmer_jac,
    )

    analytical_grad = trajectories[:objective_gradient_wrt_swimmer_params]
    n_sw_params = length(collect_differentiable_params(swimmer))
    @test length(analytical_grad) == n_sw_params
    @test all(isfinite, analytical_grad)

    # Finite-difference validation
    fd_grad = FiniteDiff.finite_difference_gradient(collect_differentiable_params(swimmer)) do p
        new_sw = inject_differentiable_params(swimmer, p)
        new_tank = rebuild_tank_with_swimmer(tank, new_sw)
        mc = rex_eel_maximal_from_minimal(new_sw, q_min_config, n_links)
        J = ForwardDiff.jacobian(q -> rex_eel_maximal_from_minimal(new_sw, q, n_links), q_min_config)
        mv = J * v_min
        new_sw_bs = vcat(mc, mv)
        new_x0 = initialize_aquarium_state(new_tank, zeros(fluid.n_velocities), new_sw_bs)
        traj = simulate_aquarium(
            new_tank, new_x0, final_time, bb_state_0, control_params;
            max_newton_iterations=50,
            calculate_objective=true,
            calculate_gradient_wrt_fluid_properties=false,
            calculate_gradient_wrt_swimmer_params=false,
            calculate_gradient_wrt_bluff_body_params=false,
            calculate_gradient_wrt_control_params=false,
            calculate_gradient_wrt_bluff_body_state_params=false,
            calculate_stage_objective=calc_stage,
            calculate_terminal_objective=calc_terminal,
            calculate_bluff_body_state_from_params=calc_bb_state,
        )
        traj[:objective_value][1]
    end

    @test analytical_grad ≈ fd_grad rtol=0.05
end


# =========================================================================================
# Slice 6: Individual Gradient Flags
# =========================================================================================

@testitem "Individual Gradient Flags" setup=[GradientSetup] begin
    using Aquarium

    G = GradientSetup
    tank = G.tank
    aquarium_state_0 = G.aquarium_state_0
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal;  calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    # Reference: all gradient flags enabled
    ref = simulate_aquarium(
        tank, aquarium_state_0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    flags = [
        :calculate_gradient_wrt_fluid_properties,
        :calculate_gradient_wrt_swimmer_params,
        :calculate_gradient_wrt_bluff_body_params,
        :calculate_gradient_wrt_control_params,
        :calculate_gradient_wrt_bluff_body_state_params,
    ]

    result_keys = [
        :objective_gradient_wrt_fluid_properties,
        :objective_gradient_wrt_swimmer_params,
        :objective_gradient_wrt_bluff_body_params,
        :objective_gradient_wrt_control_params,
        :objective_gradient_wrt_bluff_body_state_params,
    ]

    @testset "Flag: $(flags[i])" for i in eachindex(flags)
        kwargs = Dict{Symbol,Any}(
            :max_newton_iterations => 50,
            :calculate_objective => true,
            :calculate_stage_objective => calc_stage,
            :calculate_terminal_objective => calc_terminal,
            :calculate_bluff_body_state_from_params => calc_bb_state,
        )
        for f in flags
            kwargs[f] = false
        end
        kwargs[flags[i]] = true

        traj = simulate_aquarium(
            tank, aquarium_state_0, final_time, bb_state_0, control_params;
            kwargs...
        )

        @test traj[result_keys[i]] ≈ ref[result_keys[i]] rtol=1e-3  # GMRES block-solve noise
    end
end


# =========================================================================================
# Slice 6: Swimmer State Dynamics Jacobians
# =========================================================================================

@testitem "Swimmer State Dynamics Jacobians" setup=[GradientSetup] begin
    using Aquarium
    using FiniteDiff
    using LinearAlgebra

    G = GradientSetup
    tank = G.tank;  swimmer = G.swimmer
    aquarium_state_0 = G.aquarium_state_0
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    trajectories = simulate_aquarium(
        tank, aquarium_state_0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=false,
        compute_swimmer_dynamics_jacobian=true,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    A_traj = trajectories[:dynamics_jacobian_wrt_state_traj]
    B_traj = trajectories[:dynamics_jacobian_wrt_control_traj]

    n_body = swimmer.n_body_states
    n_ctrl = swimmer.n_control_inputs
    n_steps = G.n_steps

    @test length(A_traj) == n_steps + 1
    @test length(B_traj) == n_steps + 1

    @test size(A_traj[1]) == (n_body, n_body)
    @test size(B_traj[1]) == (n_body, n_ctrl)

    # A₁ = I, B₁ = 0
    @test A_traj[1] ≈ Matrix(I, n_body, n_body)
    @test B_traj[1] ≈ zeros(n_body, n_ctrl)

    # k=2: finite values and correct dimensions
    @test size(A_traj[2]) == (n_body, n_body)
    @test size(B_traj[2]) == (n_body, n_ctrl)
    @test all(isfinite, A_traj[2])
    @test all(isfinite, B_traj[2])

    # Validate A₂ and B₂ against FiniteDiff at step 1→2
    x_k = trajectories[:aquarium_state_traj][1]
    bb_kp1 = trajectories[:bluff_body_state_traj][2]
    u_k = trajectories[:control_traj][1]
    body_idx = tank.swimmer_body_state_indices

    # One-step Newton solve: given (aquarium_state_k, bb_kp1, u_k) → aquarium_state_{k+1}
    function solve_one_step(tank, x_k, bb_kp1, u_k)
        x_kp1 = copy(x_k)
        for _ in 1:50
            r = calculate_aquarium_dynamics_residual(tank, x_kp1, x_k, bb_kp1, u_k)
            norm(r) < 1e-10 && break
            J = calculate_aquarium_dynamics_jacobian(tank, x_kp1, x_k, bb_kp1, u_k)[1]
            x_kp1 .-= Matrix(J) \ r
        end
        return x_kp1
    end

    # A: ∂(body_state_{k+1})/∂(body_state_k) via FiniteDiff
    fd_A = FiniteDiff.finite_difference_jacobian(x_k[body_idx]) do bs_k
        xk_perturbed = copy(x_k)
        xk_perturbed[body_idx] .= bs_k
        x_kp1 = solve_one_step(tank, xk_perturbed, bb_kp1, u_k)
        x_kp1[body_idx]
    end
    @test A_traj[2] ≈ fd_A rtol=1e-3

    # B: ∂(body_state_{k+1})/∂(control_k) via FiniteDiff
    fd_B = FiniteDiff.finite_difference_jacobian(u_k) do uk_perturbed
        x_kp1 = solve_one_step(tank, x_k, bb_kp1, uk_perturbed)
        x_kp1[body_idx]
    end
    @test B_traj[2] ≈ fd_B rtol=1e-3
end


# =========================================================================================
# Torque saturation differentiability test
# =========================================================================================

@testitem "Torque Saturation Differentiability" setup=[TankSetup] begin
    using Aquarium
    using ForwardDiff
    using LinearAlgebra

    # Build swimmer with a low max_torque so PD output saturates
    n_links = TankSetup.n_links
    low_max_torque = 1.0  # PD with Kp=100 will far exceed this

    swimmer_sat = RExEel(TankSetup.time_step, n_links;
        bar_lengths=TankSetup.bar_lengths,
        masses=TankSetup.masses,
        mois=TankSetup.mois,
        Kps=fill(TankSetup.Kp_raw, n_links - 1),
        Kds=fill(TankSetup.Kd_raw, n_links - 1),
        max_torques=fill(low_max_torque, n_links - 1),
        n_boundary_nodes_per_link=fill(3, n_links),
        discrete_delta_kind=:three_point,
        gravity=[0.0, -TankSetup.gravity_constant],
        actuation_mode=:pd,
    )

    # Use a state with non-zero joint angles so PD output exceeds max_torque
    q_min_angled = [0.5, 0.5, 0.0, deg2rad(30.0), deg2rad(-30.0)]
    mc = rex_eel_maximal_from_minimal(swimmer_sat, q_min_angled, n_links)
    sw_state = vcat(mc, zeros(3 * n_links))
    full_state = initialize_solid_state(swimmer_sat, sw_state)
    ctrl = zeros(swimmer_sat.n_control_inputs)

    force = calculate_actuator_forces(swimmer_sat, full_state, ctrl)
    @test any(abs.(force) .≈ low_max_torque)  # at least one actuator saturated

    # ForwardDiff of saturated actuator force w.r.t. body state:
    # saturated entries should have zero derivative (clamp kills gradient)
    jac = ForwardDiff.jacobian(
        x -> calculate_actuator_forces(swimmer_sat, x, ctrl), full_state)
    saturated_rows = findall(abs.(force) .≈ low_max_torque)
    @test all(jac[saturated_rows, :] .== 0.0)

    # Now build swimmer with Inf max_torque — same state should have non-zero derivatives
    swimmer_unsat = RExEel(TankSetup.time_step, n_links;
        bar_lengths=TankSetup.bar_lengths,
        masses=TankSetup.masses,
        mois=TankSetup.mois,
        Kps=fill(TankSetup.Kp_raw, n_links - 1),
        Kds=fill(TankSetup.Kd_raw, n_links - 1),
        max_torques=fill(Inf, n_links - 1),
        n_boundary_nodes_per_link=fill(3, n_links),
        discrete_delta_kind=:three_point,
        gravity=[0.0, -TankSetup.gravity_constant],
        actuation_mode=:pd,
    )

    full_state_unsat = initialize_solid_state(swimmer_unsat, sw_state)
    jac_unsat = ForwardDiff.jacobian(
        x -> calculate_actuator_forces(swimmer_unsat, x, ctrl), full_state_unsat)
    @test any(jac_unsat .!= 0.0)  # unsaturated has non-zero derivatives
end


# #########################################################################################
# Prescribed-mode FSI integration tests
#
# Mirror the PD-mode TankSetup / GradientSetup structure with a second swimmer
# instance.  Validates that the FSI path works end-to-end with a different
# initial state / control parameterisation.
# #########################################################################################

@testmodule PrescribedTankSetup begin

using Aquarium
using ForwardDiff
using LinearAlgebra
using Random

# Physical parameters — identical to TankSetup
const time_step = 0.01
const gravity_constant = 98.0

# ---------- Fluid ----------
const fluid = Fluid(time_step;
    density=1.0,
    dynamic_viscosity=0.01,
    boundary_velocity=[0.01, 0.0],
    grid_size=(20, 20),
    grid_dimensions=(1.0, 1.0),
    boundary_condition_type=:freestream,
    gravity_constant=gravity_constant,
)

# ---------- FreeDisc bluff body ----------
const disc_radius = 0.05
const disc_density = 1.0
const disc_mass = π * disc_density * disc_radius^2
const disc_moi = 0.5 * disc_mass * disc_radius^2

const bluff_body = FreeDisc(time_step;
    radius=disc_radius,
    mass=disc_mass,
    moi=disc_moi,
    n_boundary_nodes=6,
    ib_method=:weak_form,
    discrete_delta_kind=:three_point,
    gravity=[0.0, -gravity_constant],
)

# ---------- RExEel swimmer ----------
const n_links = 3
const bar_lengths = fill(0.1, n_links)
const masses = fill(2.0, n_links)
const mois = (1 / 12) .* masses .* (bar_lengths .^ 2)

const stall_torque = 9.3e6
const encoder_resolution = 4096
const control_loop_time = 0.001
const pwm_to_torque = stall_torque / 885
const Kp_raw = 100.0 / ((encoder_resolution / (2π)) * pwm_to_torque / 128)
const Kd_raw = 100.0 / (control_loop_time * (encoder_resolution / (2π)) * pwm_to_torque / 16)

const swimmer = RExEel(time_step, n_links;
    bar_lengths=bar_lengths,
    masses=masses,
    mois=mois,
    Kps=fill(Kp_raw, n_links - 1),
    Kds=fill(Kd_raw, n_links - 1),
    max_torques=fill(Inf, n_links - 1),
    n_boundary_nodes_per_link=fill(3, n_links),
    ib_method=:weak_form,
    discrete_delta_kind=:three_point,
    gravity=[0.0, -gravity_constant],
    actuation_mode=:prescribed,
)

# ---------- AquariumTank ----------
const tank = AquariumTank(fluid, bluff_body, swimmer)

# ---------- Initial states ----------
Random.seed!(42)

const q_min_config = [0.5, 0.5, 0.0, 0.0, 0.0]
const v_min = 0.01 .* randn(n_links + 2)

const maximal_config = rex_eel_maximal_from_minimal(swimmer, q_min_config, n_links)

const config_jacobian = ForwardDiff.jacobian(
    q -> rex_eel_maximal_from_minimal(swimmer, q, n_links), q_min_config
)
const maximal_velocity = config_jacobian * v_min

const swimmer_initial_body_state = vcat(maximal_config, maximal_velocity)

const fluid_initial_velocity = repeat(fluid.boundary_velocity, fluid.n_velocities ÷ 2)

const aquarium_state_0 = initialize_aquarium_state(tank, fluid_initial_velocity, swimmer_initial_body_state)

const bluff_body_state_0 = [0.47, 0.53, 0.31, 0.02, -0.015, 0.03]

# Control params: n_joints values (θ_desired only) in prescribed mode
Random.seed!(42)
const swimmer_control_params = deg2rad.(5.0 .* randn(swimmer.n_control_inputs))

end  # PrescribedTankSetup


# =========================================================================================
# Prescribed Slice 1: AquariumTank Construction
# =========================================================================================

@testitem "Prescribed: AquariumTank Construction" setup=[PrescribedTankSetup] begin
    using Aquarium
    using LinearAlgebra

    tank = PrescribedTankSetup.tank
    fluid = PrescribedTankSetup.fluid
    bluff_body = PrescribedTankSetup.bluff_body
    swimmer = PrescribedTankSetup.swimmer

    # Field identity
    @test tank.fluid === fluid
    @test tank.bluff_body === bluff_body
    @test tank.swimmer === swimmer

    # Time step consistency
    @test tank.time_step == fluid.time_step
    @test tank.time_step == bluff_body.time_step
    @test tank.time_step == swimmer.time_step

    # Prescribed mode: 1 control input per actuator (θ_desired only)
    @test swimmer.actuation_mode == :prescribed
    @test swimmer.n_control_inputs == swimmer.n_actuators

    # Constraint counting — positional (2 per joint) + angle (1 per joint)
    n_joints = swimmer.n_actuators
    @test swimmer.n_constraints == 2 * n_joints + n_joints
    @test tank.n_bluff_body_no_slip_constraints == bluff_body.topology.n_no_slip_constraints
    @test tank.n_swimmer_no_slip_constraints == swimmer.topology.n_no_slip_constraints
    @test tank.n_no_slip_constraints == tank.n_bluff_body_no_slip_constraints + tank.n_swimmer_no_slip_constraints

    # State counting
    @test tank.n_fluid_velocities == fluid.n_velocities
    @test tank.n_swimmer_body_states == swimmer.n_body_states
    @test tank.n_fluid_constraints == fluid.n_constraints
    @test tank.n_swimmer_constraints == swimmer.n_constraints
    @test tank.n_states == (tank.n_fluid_velocities + tank.n_swimmer_body_states +
                            tank.n_fluid_constraints + tank.n_swimmer_constraints +
                            tank.n_bluff_body_no_slip_duals + tank.n_swimmer_no_slip_duals)

    # State index partitioning: non-overlapping and correct ordering
    all_indices = vcat(
        tank.fluid_velocity_indices,
        tank.swimmer_body_state_indices,
        tank.fluid_dual_indices,
        tank.swimmer_dual_indices,
        tank.bluff_body_no_slip_dual_indices,
        tank.swimmer_no_slip_dual_indices,
    )
    @test sort(all_indices) == collect(1:tank.n_states)
    @test length(unique(all_indices)) == tank.n_states

    # Correct ordering: fluid velocity < swimmer body state < fluid dual < swimmer dual < BB no-slip < swimmer no-slip
    if !isempty(tank.fluid_velocity_indices) && !isempty(tank.swimmer_body_state_indices)
        @test maximum(tank.fluid_velocity_indices) < minimum(tank.swimmer_body_state_indices)
    end
    if !isempty(tank.swimmer_body_state_indices) && !isempty(tank.fluid_dual_indices)
        @test maximum(tank.swimmer_body_state_indices) < minimum(tank.fluid_dual_indices)
    end
end


# =========================================================================================
# Prescribed Slice 2: State Extraction
# =========================================================================================

@testitem "Prescribed: State Extraction" setup=[PrescribedTankSetup] begin
    using Aquarium

    tank = PrescribedTankSetup.tank
    aquarium_state = PrescribedTankSetup.aquarium_state_0

    # All 8 extract functions return correct sizes
    @test length(extract_fluid_state(tank, aquarium_state)) == tank.n_fluid_states
    @test length(extract_swimmer_state(tank, aquarium_state)) == tank.n_swimmer_states
    @test length(extract_fluid_velocity(tank, aquarium_state)) == tank.n_fluid_velocities
    @test length(extract_swimmer_body_state(tank, aquarium_state)) == tank.n_swimmer_body_states
    @test length(extract_fluid_dual(tank, aquarium_state)) == tank.n_fluid_constraints
    @test length(extract_swimmer_dual(tank, aquarium_state)) == tank.n_swimmer_constraints
    @test length(extract_bluff_body_no_slip_dual(tank, aquarium_state)) == tank.n_bluff_body_no_slip_duals
    @test length(extract_swimmer_no_slip_dual(tank, aquarium_state)) == tank.n_swimmer_no_slip_duals

    # Extracted values match direct indexing
    @test extract_fluid_velocity(tank, aquarium_state) == aquarium_state[tank.fluid_velocity_indices]
    @test extract_swimmer_body_state(tank, aquarium_state) == aquarium_state[tank.swimmer_body_state_indices]
    @test extract_fluid_dual(tank, aquarium_state) == aquarium_state[tank.fluid_dual_indices]
    @test extract_swimmer_dual(tank, aquarium_state) == aquarium_state[tank.swimmer_dual_indices]
    @test extract_bluff_body_no_slip_dual(tank, aquarium_state) == aquarium_state[tank.bluff_body_no_slip_dual_indices]
    @test extract_swimmer_no_slip_dual(tank, aquarium_state) == aquarium_state[tank.swimmer_no_slip_dual_indices]
end


# =========================================================================================
# Prescribed Slice 2: Initialize Aquarium State
# =========================================================================================

@testitem "Prescribed: Initialize Aquarium State" setup=[PrescribedTankSetup] begin
    using Aquarium

    tank = PrescribedTankSetup.tank
    fluid_vel = PrescribedTankSetup.fluid_initial_velocity
    swimmer_bs = PrescribedTankSetup.swimmer_initial_body_state
    aquarium_state = PrescribedTankSetup.aquarium_state_0

    # Correct total size
    @test length(aquarium_state) == tank.n_states

    # Fluid velocities placed correctly
    @test extract_fluid_velocity(tank, aquarium_state) ≈ fluid_vel[1:tank.n_fluid_velocities]

    # Swimmer body state placed correctly
    @test extract_swimmer_body_state(tank, aquarium_state) ≈ swimmer_bs

    # All dual variables are zero (including angle constraint duals)
    @test all(extract_fluid_dual(tank, aquarium_state) .== 0.0)
    @test all(extract_swimmer_dual(tank, aquarium_state) .== 0.0)
    @test all(extract_bluff_body_no_slip_dual(tank, aquarium_state) .== 0.0)
    @test all(extract_swimmer_no_slip_dual(tank, aquarium_state) .== 0.0)
end


# =========================================================================================
# Prescribed Slice 3: Dynamics Residual Evaluation
# =========================================================================================

@testitem "Prescribed: Dynamics Residual" setup=[PrescribedTankSetup] begin
    using Aquarium
    using Random

    tank = PrescribedTankSetup.tank
    aquarium_state_0 = PrescribedTankSetup.aquarium_state_0
    bb_state = PrescribedTankSetup.bluff_body_state_0
    control = PrescribedTankSetup.swimmer_control_params

    # Create a second state for midpoint evaluation
    Random.seed!(123)
    aquarium_state_1 = aquarium_state_0 .+ 0.001 .* randn(tank.n_states)

    # Endpoint residual with zero control (prescribed mode requires explicit control vector)
    zero_control = zeros(tank.swimmer.n_control_inputs)
    r_zero_control = calculate_aquarium_dynamics_residual(
        tank, aquarium_state_1, aquarium_state_0, bb_state, zero_control)
    @test length(r_zero_control) == tank.n_states
    @test all(isfinite, r_zero_control)

    # Endpoint residual with non-zero control
    r_with_control = calculate_aquarium_dynamics_residual(
        tank, aquarium_state_1, aquarium_state_0, bb_state, control)
    @test length(r_with_control) == tank.n_states
    @test all(isfinite, r_with_control)

    # Midpoint residual
    r_midpoint = calculate_aquarium_dynamics_residual(
        tank, aquarium_state_1, aquarium_state_0, bb_state, control;
        is_midpoint_state_bluff_body=true)
    @test length(r_midpoint) == tank.n_states
    @test all(isfinite, r_midpoint)

    # Zero-control and non-zero-control residuals should differ
    @test r_zero_control != r_with_control
    @test r_with_control != r_midpoint
end


# =========================================================================================
# Prescribed Slice 3: Dynamics Jacobian Structure
# =========================================================================================

@testitem "Prescribed: Dynamics Jacobian Structure" setup=[PrescribedTankSetup] begin
    using Aquarium
    using Random

    tank = PrescribedTankSetup.tank
    fluid = PrescribedTankSetup.fluid
    bluff_body = PrescribedTankSetup.bluff_body
    swimmer = PrescribedTankSetup.swimmer
    aquarium_state_0 = PrescribedTankSetup.aquarium_state_0
    bb_state = PrescribedTankSetup.bluff_body_state_0
    control = PrescribedTankSetup.swimmer_control_params

    Random.seed!(123)
    aquarium_state_1 = aquarium_state_0 .+ 0.001 .* randn(tank.n_states)

    (∂D_∂xkp1, ∂D_∂xk, ∂D_∂uk, ∂D_∂fluid_props, ∂D_∂sw_params,
     ∂D_∂bb_params, ∂D_∂bb_state_kp1) = calculate_aquarium_dynamics_jacobian(
        tank, aquarium_state_1, aquarium_state_0, bb_state, control)

    n = tank.n_states

    # State Jacobians
    @test size(∂D_∂xkp1) == (n, n)
    @test size(∂D_∂xk) == (n, n)

    # Control Jacobian — prescribed mode: n_joints columns (NOT 2*n_joints)
    n_joints = swimmer.n_actuators
    @test size(∂D_∂uk) == (n, n_joints)
    @test size(∂D_∂uk) == (n, swimmer.n_control_inputs)
    @test size(∂D_∂uk) == (n, length(control))

    # Fluid properties Jacobian (4 params: density, viscosity, bv_x, bv_y)
    @test size(∂D_∂fluid_props) == (n, length(collect_differentiable_params(fluid)))
    @test size(∂D_∂fluid_props, 2) == 4

    # Swimmer params Jacobian (21 params: 5 per body × 3 + 3 per joint × 2)
    @test size(∂D_∂sw_params) == (n, length(collect_differentiable_params(swimmer)))

    # Bluff body params Jacobian (5 params: mass, moi, com_x, com_y, radius)
    @test size(∂D_∂bb_params) == (n, length(collect_differentiable_params(bluff_body)))

    # Bluff body state Jacobian
    @test size(∂D_∂bb_state_kp1) == (n, bluff_body.n_body_states)
end


# =========================================================================================
# Prescribed Slice 4: Jacobian vs ForwardDiff and FiniteDiff
# =========================================================================================

@testitem "Prescribed: Jacobian vs ForwardDiff and FiniteDiff" setup=[PrescribedTankSetup] begin
    using Aquarium
    using ForwardDiff
    using FiniteDiff
    using Random
    using LinearAlgebra

    tank = PrescribedTankSetup.tank
    fluid = PrescribedTankSetup.fluid
    bluff_body = PrescribedTankSetup.bluff_body
    swimmer = PrescribedTankSetup.swimmer
    aquarium_state_0 = PrescribedTankSetup.aquarium_state_0
    bb_state = PrescribedTankSetup.bluff_body_state_0
    control = PrescribedTankSetup.swimmer_control_params

    Random.seed!(123)
    aquarium_state_1 = aquarium_state_0 .+ 0.001 .* randn(tank.n_states)

    (∂D_∂xkp1, ∂D_∂xk, ∂D_∂uk, ∂D_∂fluid_props, ∂D_∂sw_params,
     ∂D_∂bb_params, ∂D_∂bb_state_kp1) = calculate_aquarium_dynamics_jacobian(
        tank, aquarium_state_1, aquarium_state_0, bb_state, control)

    # --- State/control Jacobians: ForwardDiff (exact AD) ---

    # ∂D/∂x_kp1 — the full KKT matrix
    ad_∂D_∂xkp1 = ForwardDiff.jacobian(
        x -> calculate_aquarium_dynamics_residual(tank, x, aquarium_state_0, bb_state, control),
        aquarium_state_1)
    @test Matrix(∂D_∂xkp1) ≈ ad_∂D_∂xkp1 atol=1e-6

    # ∂D/∂x_k
    ad_∂D_∂xk = ForwardDiff.jacobian(
        x -> calculate_aquarium_dynamics_residual(tank, aquarium_state_1, x, bb_state, control),
        aquarium_state_0)
    @test Matrix(∂D_∂xk) ≈ ad_∂D_∂xk atol=1e-10

    # ∂D/∂u_k
    ad_∂D_∂uk = ForwardDiff.jacobian(
        u -> calculate_aquarium_dynamics_residual(tank, aquarium_state_1, aquarium_state_0, bb_state, u),
        control)
    @test Matrix(∂D_∂uk) ≈ ad_∂D_∂uk atol=1e-10

    # ∂D/∂bluff_body_state_kp1
    ad_∂D_∂bb_state = ForwardDiff.jacobian(
        bb -> calculate_aquarium_dynamics_residual(tank, aquarium_state_1, aquarium_state_0, bb, control),
        bb_state)
    @test Matrix(∂D_∂bb_state_kp1) ≈ ad_∂D_∂bb_state atol=1e-10

    # --- Parameter Jacobians: FiniteDiff (inject_differentiable_params has Float64 barriers) ---

    # ∂D/∂fluid_props
    fd_∂D_∂fluid_props = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(fluid)) do p
        new_fluid = inject_differentiable_params(fluid, p)
        new_tank = rebuild_tank_with_fluid(tank, new_fluid)
        calculate_aquarium_dynamics_residual(new_tank, aquarium_state_1, aquarium_state_0, bb_state, control)
    end
    @test Matrix(∂D_∂fluid_props) ≈ fd_∂D_∂fluid_props rtol=1e-4

    # ∂D/∂swimmer_params
    fd_∂D_∂sw_params = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(swimmer)) do p
        new_sw = inject_differentiable_params(swimmer, p)
        new_tank = rebuild_tank_with_swimmer(tank, new_sw)
        calculate_aquarium_dynamics_residual(new_tank, aquarium_state_1, aquarium_state_0, bb_state, control)
    end
    @test Matrix(∂D_∂sw_params) ≈ fd_∂D_∂sw_params rtol=1e-4

    # ∂D/∂bluff_body_params
    fd_∂D_∂bb_params = FiniteDiff.finite_difference_jacobian(collect_differentiable_params(bluff_body)) do p
        new_bb = inject_differentiable_params(bluff_body, p)
        new_tank = rebuild_tank_with_bluff_body(tank, new_bb)
        calculate_aquarium_dynamics_residual(new_tank, aquarium_state_1, aquarium_state_0, bb_state, control)
    end
    @test Matrix(∂D_∂bb_params) ≈ fd_∂D_∂bb_params rtol=1e-4
end


# #########################################################################################
# Prescribed-mode gradient test setup
# #########################################################################################

@testmodule PrescribedGradientSetup begin

using Aquarium
using ForwardDiff
using LinearAlgebra
using Random

# Reuse the same 20×20 physical setup — prescribed mode
const time_step = 0.01
const gravity_constant = 98.0

const fluid = Fluid(time_step;
    density=1.0, dynamic_viscosity=0.01,
    boundary_velocity=[0.01, 0.0],
    grid_size=(20, 20), grid_dimensions=(1.0, 1.0),
    boundary_condition_type=:freestream,
    gravity_constant=gravity_constant,
)

const disc_radius = 0.05
const disc_density = 1.0
const disc_mass = π * disc_density * disc_radius^2
const disc_moi = 0.5 * disc_mass * disc_radius^2

const bluff_body = FreeDisc(time_step;
    radius=disc_radius, mass=disc_mass, moi=disc_moi,
    n_boundary_nodes=6, ib_method=:weak_form,
    discrete_delta_kind=:three_point,
    gravity=[0.0, -gravity_constant],
)

const n_links = 3
const bar_lengths = fill(0.1, n_links)
const masses = fill(2.0, n_links)
const mois = (1 / 12) .* masses .* (bar_lengths .^ 2)

const stall_torque = 9.3e6
const encoder_resolution = 4096
const control_loop_time = 0.001
const pwm_to_torque = stall_torque / 885
const Kp_raw = 100.0 / ((encoder_resolution / (2π)) * pwm_to_torque / 128)
const Kd_raw = 100.0 / (control_loop_time * (encoder_resolution / (2π)) * pwm_to_torque / 16)

const swimmer = RExEel(time_step, n_links;
    bar_lengths=bar_lengths, masses=masses, mois=mois,
    Kps=fill(Kp_raw, n_links - 1), Kds=fill(Kd_raw, n_links - 1),
    max_torques=fill(Inf, n_links - 1),
    n_boundary_nodes_per_link=fill(3, n_links),
    ib_method=:weak_form, discrete_delta_kind=:three_point,
    gravity=[0.0, -gravity_constant],
    actuation_mode=:prescribed,
)

const tank = AquariumTank(fluid, bluff_body, swimmer)

Random.seed!(42)

const q_min_config = [0.5, 0.5, 0.0, 0.0, 0.0]
const v_min = 0.01 .* randn(n_links + 2)

const maximal_config = rex_eel_maximal_from_minimal(swimmer, q_min_config, n_links)
const config_jacobian = ForwardDiff.jacobian(
    q -> rex_eel_maximal_from_minimal(swimmer, q, n_links), q_min_config)
const maximal_velocity = config_jacobian * v_min
const swimmer_initial_body_state = vcat(maximal_config, maximal_velocity)

const fluid_initial_velocity = repeat(fluid.boundary_velocity, fluid.n_velocities ÷ 2)
const aquarium_state_0 = initialize_aquarium_state(tank, fluid_initial_velocity, swimmer_initial_body_state)

const bluff_body_state_0 = [0.47, 0.53, 0.31, 0.02, -0.015, 0.03]

Random.seed!(42)
const swimmer_control_params = deg2rad.(5.0 .* randn(swimmer.n_control_inputs))

# Shared objective functions for all gradient tests
const calc_stage = (tank, t, x, bb, u) ->
    sum(x) + sum(bb) + sum(u) +
    t * (tank.fluid.density + tank.swimmer.bodies[1].mass + tank.bluff_body.bodies[1].mass)
const calc_terminal = (tank, t, x, bb) -> sum(x .^ 2) + sum(bb .^ 2)
const calc_bb_state = (bluff_body, t, params; bluff_body_params=collect_differentiable_params(bluff_body)) -> params

const n_steps = 3
const final_time = n_steps * time_step

end  # PrescribedGradientSetup


# =========================================================================================
# Prescribed Slice 5: Simulation + Objective Computation
# =========================================================================================

@testitem "Prescribed: Simulation Objective Computation" setup=[PrescribedGradientSetup] begin
    using Aquarium

    G = PrescribedGradientSetup
    tank = G.tank
    aquarium_state_0 = G.aquarium_state_0
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal;  calc_bb_state = G.calc_bb_state
    final_time = G.final_time
    n_steps = G.n_steps

    trajectories = simulate_aquarium(
        tank, aquarium_state_0, final_time,
        bb_state_0,
        control_params;
        calculate_objective=true,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    # Simulation completes and returns correct keys
    @test haskey(trajectories, :time_traj)
    @test haskey(trajectories, :aquarium_state_traj)
    @test haskey(trajectories, :objective_value)
    @test haskey(trajectories, :objective_traj)

    # Trajectory dimensions
    @test length(trajectories[:time_traj]) == n_steps + 1
    @test length(trajectories[:aquarium_state_traj]) == n_steps + 1
    @test all(length(s) == tank.n_states for s in trajectories[:aquarium_state_traj])

    # Objective value matches manual sum of stage + terminal
    obj_traj = trajectories[:objective_traj]
    @test length(obj_traj) == n_steps + 1
    @test trajectories[:objective_value][1] ≈ sum(obj_traj)

    # Manual recomputation: stages 1..N-1, terminal at N
    time_traj = trajectories[:time_traj]
    state_traj = trajectories[:aquarium_state_traj]
    bb_traj = trajectories[:bluff_body_state_traj]
    ctrl_traj = trajectories[:control_traj]

    manual_obj = sum(k -> calc_stage(tank, time_traj[k], state_traj[k], bb_traj[k], ctrl_traj[k]), 1:n_steps)
    manual_obj += calc_terminal(tank, time_traj[end], state_traj[end], bb_traj[end])

    @test trajectories[:objective_value][1] ≈ manual_obj
end


# =========================================================================================
# Prescribed Slice 6: Gradient wrt Control Params
# =========================================================================================

@testitem "Prescribed: Gradient wrt Control Params" setup=[PrescribedGradientSetup] begin
    using Aquarium
    using FiniteDiff

    G = PrescribedGradientSetup
    tank = G.tank
    aquarium_state_0 = G.aquarium_state_0
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal;  calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    trajectories = simulate_aquarium(
        tank, aquarium_state_0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_gradient_wrt_fluid_properties=false,
        calculate_gradient_wrt_swimmer_params=false,
        calculate_gradient_wrt_bluff_body_params=false,
        calculate_gradient_wrt_control_params=true,
        calculate_gradient_wrt_bluff_body_state_params=false,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    analytical_grad = trajectories[:objective_gradient_wrt_control_params]
    @test length(analytical_grad) == length(control_params)
    @test all(isfinite, analytical_grad)

    # Finite-difference validation
    fd_grad = FiniteDiff.finite_difference_gradient(control_params) do p
        traj = simulate_aquarium(
            tank, aquarium_state_0, final_time, bb_state_0, p;
            max_newton_iterations=50,
            calculate_objective=true,
            calculate_gradient_wrt_fluid_properties=false,
            calculate_gradient_wrt_swimmer_params=false,
            calculate_gradient_wrt_bluff_body_params=false,
            calculate_gradient_wrt_control_params=false,
            calculate_gradient_wrt_bluff_body_state_params=false,
            calculate_stage_objective=calc_stage,
            calculate_terminal_objective=calc_terminal,
            calculate_bluff_body_state_from_params=calc_bb_state,
        )
        traj[:objective_value][1]
    end

    @test analytical_grad ≈ fd_grad rtol=0.05
end


# =========================================================================================
# Prescribed Slice 7: Gradient wrt Fluid Properties
# =========================================================================================

@testitem "Prescribed: Gradient wrt Fluid Properties" setup=[PrescribedGradientSetup] begin
    using Aquarium
    using FiniteDiff

    G = PrescribedGradientSetup
    tank = G.tank;  fluid = G.fluid
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    swimmer_bs = G.swimmer_initial_body_state
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal;  calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    # Use a fixed initial state (zero fluid velocity) that does NOT depend on fluid params.
    x0 = initialize_aquarium_state(tank, zeros(fluid.n_velocities), swimmer_bs)

    trajectories = simulate_aquarium(
        tank, x0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_gradient_wrt_fluid_properties=true,
        calculate_gradient_wrt_swimmer_params=false,
        calculate_gradient_wrt_bluff_body_params=false,
        calculate_gradient_wrt_control_params=false,
        calculate_gradient_wrt_bluff_body_state_params=false,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    analytical_grad = trajectories[:objective_gradient_wrt_fluid_properties]
    @test length(analytical_grad) == 4
    @test all(isfinite, analytical_grad)

    # Finite-difference validation — same fixed initial state for each perturbation
    fd_grad = FiniteDiff.finite_difference_gradient(collect_differentiable_params(fluid)) do p
        nf = inject_differentiable_params(fluid, p)
        nt = rebuild_tank_with_fluid(tank, nf)
        traj = simulate_aquarium(
            nt, x0, final_time, bb_state_0, control_params;
            max_newton_iterations=50,
            calculate_objective=true,
            calculate_gradient_wrt_fluid_properties=false,
            calculate_gradient_wrt_swimmer_params=false,
            calculate_gradient_wrt_bluff_body_params=false,
            calculate_gradient_wrt_control_params=false,
            calculate_gradient_wrt_bluff_body_state_params=false,
            calculate_stage_objective=calc_stage,
            calculate_terminal_objective=calc_terminal,
            calculate_bluff_body_state_from_params=calc_bb_state,
        )
        traj[:objective_value][1]
    end

    @test analytical_grad ≈ fd_grad rtol=0.05
end


# =========================================================================================
# Prescribed Slice 8: Gradient wrt Swimmer Params
# =========================================================================================

@testitem "Prescribed: Gradient wrt Swimmer Params" setup=[PrescribedGradientSetup] begin
    using Aquarium
    using FiniteDiff
    using ForwardDiff

    G = PrescribedGradientSetup
    tank = G.tank;  fluid = G.fluid;  swimmer = G.swimmer
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    q_min_config = G.q_min_config
    v_min = G.v_min
    n_links = G.n_links
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal;  calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    # Initial-state Jacobian wrt swimmer params
    sw_bs = G.swimmer_initial_body_state
    initial_state_swimmer_jac = ForwardDiff.jacobian(collect_differentiable_params(swimmer)) do p
        new_sw = inject_differentiable_params(swimmer, p)
        mc = rex_eel_maximal_from_minimal(new_sw, q_min_config, n_links)
        J = ForwardDiff.jacobian(q -> rex_eel_maximal_from_minimal(new_sw, q, n_links), q_min_config)
        mv = J * v_min
        new_bs = vcat(mc, mv)
        initialize_aquarium_state(tank, zeros(fluid.n_velocities), new_bs)
    end

    x0 = initialize_aquarium_state(tank, zeros(fluid.n_velocities), sw_bs)

    trajectories = simulate_aquarium(
        tank, x0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_gradient_wrt_fluid_properties=false,
        calculate_gradient_wrt_swimmer_params=true,
        calculate_gradient_wrt_bluff_body_params=false,
        calculate_gradient_wrt_control_params=false,
        calculate_gradient_wrt_bluff_body_state_params=false,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
        initial_aquarium_state_swimmer_params_jacobian=initial_state_swimmer_jac,
    )

    analytical_grad = trajectories[:objective_gradient_wrt_swimmer_params]
    n_sw_params = length(collect_differentiable_params(swimmer))
    @test length(analytical_grad) == n_sw_params
    @test all(isfinite, analytical_grad)

    # Finite-difference validation
    fd_grad = FiniteDiff.finite_difference_gradient(collect_differentiable_params(swimmer)) do p
        new_sw = inject_differentiable_params(swimmer, p)
        new_tank = rebuild_tank_with_swimmer(tank, new_sw)
        mc = rex_eel_maximal_from_minimal(new_sw, q_min_config, n_links)
        J = ForwardDiff.jacobian(q -> rex_eel_maximal_from_minimal(new_sw, q, n_links), q_min_config)
        mv = J * v_min
        new_sw_bs = vcat(mc, mv)
        new_x0 = initialize_aquarium_state(new_tank, zeros(fluid.n_velocities), new_sw_bs)
        traj = simulate_aquarium(
            new_tank, new_x0, final_time, bb_state_0, control_params;
            max_newton_iterations=50,
            calculate_objective=true,
            calculate_gradient_wrt_fluid_properties=false,
            calculate_gradient_wrt_swimmer_params=false,
            calculate_gradient_wrt_bluff_body_params=false,
            calculate_gradient_wrt_control_params=false,
            calculate_gradient_wrt_bluff_body_state_params=false,
            calculate_stage_objective=calc_stage,
            calculate_terminal_objective=calc_terminal,
            calculate_bluff_body_state_from_params=calc_bb_state,
        )
        traj[:objective_value][1]
    end

    @test analytical_grad ≈ fd_grad rtol=0.05
end


# =========================================================================================
# Prescribed Slice 9: Gradient wrt Bluff Body State Params
# =========================================================================================

@testitem "Prescribed: Gradient wrt Bluff Body State Params" setup=[PrescribedGradientSetup] begin
    using Aquarium
    using FiniteDiff

    G = PrescribedGradientSetup
    tank = G.tank
    aquarium_state_0 = G.aquarium_state_0
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal
    final_time = G.final_time

    # Custom bluff body state function: prescribed sinusoidal motion
    calc_bb_state = (bluff_body, t, params; bluff_body_params=collect_differentiable_params(bluff_body)) ->
        [params[1] + 0.01*sin(t), params[2] + 0.01*cos(t), params[3],
         params[4], params[5], params[6]]

    trajectories = simulate_aquarium(
        tank, aquarium_state_0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_gradient_wrt_fluid_properties=false,
        calculate_gradient_wrt_swimmer_params=false,
        calculate_gradient_wrt_bluff_body_params=false,
        calculate_gradient_wrt_control_params=false,
        calculate_gradient_wrt_bluff_body_state_params=true,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    analytical_grad = trajectories[:objective_gradient_wrt_bluff_body_state_params]
    @test length(analytical_grad) == length(bb_state_0)
    @test all(isfinite, analytical_grad)

    # Finite-difference validation
    fd_grad = FiniteDiff.finite_difference_gradient(bb_state_0) do p
        traj = simulate_aquarium(
            tank, aquarium_state_0, final_time, p, control_params;
            max_newton_iterations=50,
            calculate_objective=true,
            calculate_gradient_wrt_fluid_properties=false,
            calculate_gradient_wrt_swimmer_params=false,
            calculate_gradient_wrt_bluff_body_params=false,
            calculate_gradient_wrt_control_params=false,
            calculate_gradient_wrt_bluff_body_state_params=false,
            calculate_stage_objective=calc_stage,
            calculate_terminal_objective=calc_terminal,
            calculate_bluff_body_state_from_params=calc_bb_state,
        )
        traj[:objective_value][1]
    end

    @test analytical_grad ≈ fd_grad rtol=0.05
end


# =========================================================================================
# Prescribed Slice 10: Individual Gradient Flags
# =========================================================================================

@testitem "Prescribed: Individual Gradient Flags" setup=[PrescribedGradientSetup] begin
    using Aquarium

    G = PrescribedGradientSetup
    tank = G.tank
    aquarium_state_0 = G.aquarium_state_0
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    calc_stage = G.calc_stage;  calc_terminal = G.calc_terminal;  calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    # Reference: all gradient flags enabled
    ref = simulate_aquarium(
        tank, aquarium_state_0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=true,
        calculate_stage_objective=calc_stage,
        calculate_terminal_objective=calc_terminal,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    flags = [
        :calculate_gradient_wrt_fluid_properties,
        :calculate_gradient_wrt_swimmer_params,
        :calculate_gradient_wrt_bluff_body_params,
        :calculate_gradient_wrt_control_params,
        :calculate_gradient_wrt_bluff_body_state_params,
    ]

    result_keys = [
        :objective_gradient_wrt_fluid_properties,
        :objective_gradient_wrt_swimmer_params,
        :objective_gradient_wrt_bluff_body_params,
        :objective_gradient_wrt_control_params,
        :objective_gradient_wrt_bluff_body_state_params,
    ]

    @testset "Flag: $(flags[i])" for i in eachindex(flags)
        kwargs = Dict{Symbol,Any}(
            :max_newton_iterations => 50,
            :calculate_objective => true,
            :calculate_stage_objective => calc_stage,
            :calculate_terminal_objective => calc_terminal,
            :calculate_bluff_body_state_from_params => calc_bb_state,
        )
        for f in flags
            kwargs[f] = false
        end
        kwargs[flags[i]] = true

        traj = simulate_aquarium(
            tank, aquarium_state_0, final_time, bb_state_0, control_params;
            kwargs...
        )

        @test traj[result_keys[i]] ≈ ref[result_keys[i]] rtol=1e-3  # GMRES block-solve noise
    end
end


# =========================================================================================
# Prescribed Slice 10: Swimmer State Dynamics Jacobians
# =========================================================================================

@testitem "Prescribed: Swimmer State Dynamics Jacobians" setup=[PrescribedGradientSetup] begin
    using Aquarium
    using FiniteDiff
    using LinearAlgebra

    G = PrescribedGradientSetup
    tank = G.tank;  swimmer = G.swimmer
    aquarium_state_0 = G.aquarium_state_0
    bb_state_0 = G.bluff_body_state_0
    control_params = G.swimmer_control_params
    calc_bb_state = G.calc_bb_state
    final_time = G.final_time

    trajectories = simulate_aquarium(
        tank, aquarium_state_0, final_time, bb_state_0, control_params;
        max_newton_iterations=50,
        calculate_objective=false,
        compute_swimmer_dynamics_jacobian=true,
        calculate_bluff_body_state_from_params=calc_bb_state,
    )

    A_traj = trajectories[:dynamics_jacobian_wrt_state_traj]
    B_traj = trajectories[:dynamics_jacobian_wrt_control_traj]

    n_body = swimmer.n_body_states
    n_ctrl = swimmer.n_control_inputs
    n_steps = G.n_steps

    @test length(A_traj) == n_steps + 1
    @test length(B_traj) == n_steps + 1

    @test size(A_traj[1]) == (n_body, n_body)
    @test size(B_traj[1]) == (n_body, n_ctrl)

    # A₁ = I, B₁ = 0
    @test A_traj[1] ≈ Matrix(I, n_body, n_body)
    @test B_traj[1] ≈ zeros(n_body, n_ctrl)

    # k=2: finite values and correct dimensions
    @test size(A_traj[2]) == (n_body, n_body)
    @test size(B_traj[2]) == (n_body, n_ctrl)
    @test all(isfinite, A_traj[2])
    @test all(isfinite, B_traj[2])

    # Validate A₂ and B₂ against FiniteDiff at step 1→2
    x_k = trajectories[:aquarium_state_traj][1]
    bb_kp1 = trajectories[:bluff_body_state_traj][2]
    u_k = trajectories[:control_traj][1]
    body_idx = tank.swimmer_body_state_indices

    # One-step Newton solve: given (aquarium_state_k, bb_kp1, u_k) → aquarium_state_{k+1}
    function solve_one_step(tank, x_k, bb_kp1, u_k)
        x_kp1 = copy(x_k)
        for _ in 1:50
            r = calculate_aquarium_dynamics_residual(tank, x_kp1, x_k, bb_kp1, u_k)
            norm(r) < 1e-10 && break
            J = calculate_aquarium_dynamics_jacobian(tank, x_kp1, x_k, bb_kp1, u_k)[1]
            x_kp1 .-= Matrix(J) \ r
        end
        return x_kp1
    end

    # A: ∂(body_state_{k+1})/∂(body_state_k) via FiniteDiff
    fd_A = FiniteDiff.finite_difference_jacobian(x_k[body_idx]) do bs_k
        xk_perturbed = copy(x_k)
        xk_perturbed[body_idx] .= bs_k
        x_kp1 = solve_one_step(tank, xk_perturbed, bb_kp1, u_k)
        x_kp1[body_idx]
    end
    @test A_traj[2] ≈ fd_A rtol=1e-3

    # B: ∂(body_state_{k+1})/∂(control_k) via FiniteDiff
    fd_B = FiniteDiff.finite_difference_jacobian(u_k) do uk_perturbed
        x_kp1 = solve_one_step(tank, x_k, bb_kp1, uk_perturbed)
        x_kp1[body_idx]
    end
    @test B_traj[2] ≈ fd_B rtol=1e-3
end
