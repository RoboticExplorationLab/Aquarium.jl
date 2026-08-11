import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using AquariumClosed
using AquariumClosed.LinearAlgebra
using AquariumClosed.ForwardDiff
using AquariumClosed.CairoMakie
using FiniteDiff
using Colors
using JLD2
using Test
using LBFGSB
using Dates
using Random

vis_dir = joinpath(AquariumClosed.VIS_DIR, "rexeel_c_start_optimization")
mkpath(vis_dir)

#############################################################################################
## Plot params
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 800)
logocolors = Colors.JULIA_LOGO_COLORS

#############################################################################################
## Define fluid domain (4ft x 4ft tank with wall boundaries)
#############################################################################################

# time properties
time_step = 0.01
final_time = 3.0  # Total C-start duration
N_time = Int(final_time/time_step) + 1

# fluid properties (water)
fluid_density = 1.0  # g/cm³
dynamic_viscosity = 0.01  # g/(cm*s) - water at room temperature

# fish tank dimensions
length_x = 122.
length_y = 122.

# fluid grid
num_cells_x = 122
num_cells_y = 122

# boundary conditions - wall boundaries (no flow in/out)
boundary_condition_type = :wall

#############################################################################################
## Create fluid environment
#############################################################################################

fluid_env = Fluid(
    time_step;
    density = fluid_density,
    dynamic_viscosity = dynamic_viscosity,
    boundary_velocity = [0.0, 0.0],
    grid_size = (num_cells_x, num_cells_y),
    grid_dimensions = (length_x, length_y),
    boundary_condition_type = boundary_condition_type,
)

println("\nFluid environment created:")
println("  Domain: $(length_x) cm × $(length_y) cm")
println("  Grid: $(num_cells_x) × $(num_cells_y) cells")
println("  Boundary conditions: $(boundary_condition_type)")
println("  Fluid density: $(fluid_density) g/cm³")
println("  Dynamic viscosity: $(dynamic_viscosity) g/(cm*s)")
println()

#############################################################################################
## Define 6-link RExEel (swimmer)
#############################################################################################

# eel properties
n_links = 6
# Link lengths for 3-link eel (using same as manipulator)
link_lengths = [12.0, 9.8 .* ones(n_links-1)...]  # cm per link
height = 9.35  # cm
masses_per_link = [192, 140 .* ones(n_links-1)...] ./ height # g per link
moi_per_link = [2435.99, 1483.49 .* ones(n_links-1)...] ./ height  # g·cm²
gravity_constant = 0.0

eel_length = sum(link_lengths)

# boundary properties - compute per-link boundary nodes based on link length
n_boundary_nodes = floor.(Int, link_lengths ./ fluid_env.fvm_grid.h_x)

# Starting position at center of tank, oriented vertically downward (head down, tail up)
start_x = 50.0
start_y = length_y / 2
start_θ = -π/2  # vertical orientation pointing downward (-90 degrees)

# PD gains for each actuated joint (legacy: XC330M288T(Kp=2500, Kd=500, max_torque=...))
max_torque_per_joint = 2 * 9.3e6 / height
Kps_rex         = fill(2500.0, n_links - 1)
Kds_rex         = fill(500.0,  n_links - 1)
max_torques_rex = fill(max_torque_per_joint, n_links - 1)

rexeel = RExEel(time_step, n_links;
    bar_lengths = link_lengths,
    masses = masses_per_link,
    mois = moi_per_link,
    Kps = Kps_rex,
    Kds = Kds_rex,
    max_torques = max_torques_rex,
    n_boundary_nodes_per_link = n_boundary_nodes,
    ib_method = :weak_form,
    discrete_delta_kind = :three_point,
    gravity = [0.0, -gravity_constant],
    actuation_mode=:prescribed,
)

rexeel.plot_params[:bodycolor] = logocolors[3]
rexeel.plot_params[:linewidth] = 4.0
rexeel.plot_params[:showboundaryvelocities] = false
rexeel.plot_params[:arrowcolor] = logocolors[1]
rexeel.plot_params[:lengthscale] = 1.0
rexeel.plot_params[:showboundarynodes] = false
rexeel.plot_params[:boundarynodesize] = 10.0
rexeel.plot_params[:boundarynodecolor] = logocolors[2]

println("RExEel Configuration:")
for i in 1:n_links
    println("  Link $i: length=$(link_lengths[i])cm, mass=$(masses_per_link[i])g, moi=$(moi_per_link[i])g/cm²")
end
println("  Total length: $(sum(link_lengths))cm")
println("  Total mass: $(sum(masses_per_link))g")
println("  Starting at center: ($start_x, $start_y) with vertical orientation $(rad2deg(start_θ))°")
println()

#############################################################################################
## Create AquariumTank with RExEel as swimmer (no bluff body)
#############################################################################################

tank = AquariumTank_only_swimmer(fluid_env, rexeel)

println("AquariumTank created:")
println("  Fluid states: ", tank.n_fluid_states)
println("  Swimmer states: ", tank.n_swimmer_states)
println("  No-slip constraints: ", tank.n_no_slip_constraints)
println("  Total aquarium states: ", tank.n_states)
println()

#############################################################################################
## Optimization Setup
#############################################################################################

# Fixed timing parameters (not optimized)
T_prep = 1.0   # Preparatory stroke duration [s]
T_prop = 1.0   # Propulsive phase duration [s]

# Desired direction angle for velocity maximization
# -π/2 = downward (same as initial orientation), 0 = +x direction, π/2 = +y direction
desired_direction_angle = deg2rad(0)

# Quadratic penalty objective: J = (target_x - disp_x)² + (target_y - disp_y)²
# where target_x = 0.7*cos(desired_angle), target_y = 0.7*sin(desired_angle)
# Gradients: ∂J/∂disp_x = -2(target_x - disp_x), ∂J/∂disp_y = -2(target_y - disp_y)
# Target distance: 0.7 body lengths in the desired direction
# Simpler convex landscape without dot product coupling

# Number of joints
n_joints = n_links - 1

# Initial guess for optimization parameters (n_joints + 1 total)
# [B_prep (n_joints values, one per joint), φ]
# Random.seed!(42)  # Set random seed for reproducibility
# B_prep_init = deg2rad.(rand(n_joints) .* 30.0)  # Random angles between 0 and 30 degrees [rad]
# B_prep_init = deg2rad.(30 .* ones(n_joints))

# B_prep_init = deg2rad.([10, 10, 10, 10, 10])  # Increasing bend from head to tail [rad]
# φ_init = -pi/3 # 1.11 - 2*pi*0.3    # Initial phase offset [rad]
# ψ_tail_init = 2π * 0.6              # Total phase lag head-to-tail [rad] (wavelength = 0.7)

B_prep_init = deg2rad.([17.175683961250282, 17.77561203549385, 19.40557132265484, 23.105955851639145, 36.0])  # Example - update with actual values
ψ_tail_init = 3.582384186876536  # Example - update with actual values
φ_init = -1.157910829319428  # Example - update with actual values

# Fixed parameters (not optimized)
K_prop_fixed = deg2rad(25.0)  # Undulation amplitude [rad]

x0 = [B_prep_init..., ψ_tail_init, φ_init]

# Smoothness regularization: penalize non-smooth bending patterns between adjacent joints
# This prevents numerically unstable asymmetric configurations without biasing toward any particular bend magnitude
# Unlike absolute-value Tikhonov, this scales naturally and won't dominate as objective → 0
tikhonov_lambda_smooth = 0.0  # Regularization strength for smoothness penalty

# Parameter bounds
lower_bounds = [deg2rad.([10, 10, 10, 10, 10])..., 2π * 0.5, -π/2]
upper_bounds = [deg2rad.([20, 20, 25, 25, 36])..., 2π, 0.0]

println("Optimization Setup:")
println("  Number of optimization parameters: $(length(x0))")
println("  Fixed timing: T_prep=$(T_prep)s, T_prop=$(T_prop)s")
println("  Fixed K_prop: $(rad2deg(K_prop_fixed))°")
println("  Desired direction angle: $(rad2deg(desired_direction_angle))°")
println("  Initial B_prep: $(rad2deg.(B_prep_init))°")
println("  Initial ψ_tail: $(ψ_tail_init) rad (wavelength = $(ψ_tail_init/2π))")
println("  Initial φ: $(φ_init) rad")
println()

#############################################################################################
## Define control inputs for C-start escape maneuver
#############################################################################################

function calculate_control_input_from_params(solid_system, t, control_params)
    # Joint angle limit
    joint_angle_limit = deg2rad(45.0)

    # Smooth clamp using softplus
    function smooth_clamp(x, limit; k=20.0)
        softplus(z) = log(1 + exp(z))
        return x - softplus(k * (x - limit)) / k + softplus(k * (-x - limit)) / k
    end

    # Smooth ramp: cubic interpolation with zero derivatives at endpoints
    function smooth_ramp(t, t_start, t_end, val_start, val_end)
        τ = (t - t_start) / (t_end - t_start)
        ramp_val = val_start + (val_end - val_start) * (3τ^2 - 2τ^3)
        return ifelse(t <= t_start, val_start, ifelse(t >= t_end, val_end, ramp_val))
    end

    n_joints = length(solid_system.joints)

    # Extract parameters: [B_prep (n_joints), K_prop, ψ_tail, φ, T_prep, T_prop]
    B_prep = control_params[1:n_joints]
    K_prop = control_params[n_joints + 1]
    ψ_tail = control_params[n_joints + 2]
    φ = control_params[n_joints + 3]       # Initial phase offset [rad]
    T_prep = control_params[n_joints + 4]
    T_prop = control_params[n_joints + 5]

    f = 1.0  # Undulation frequency [Hz]

    # Preallocate output
    T = promote_type(eltype(B_prep), typeof(K_prop), typeof(t))
    θ_joints = zeros(T, n_joints)

    for i in 1:n_joints
        s_i = i / n_joints  # Normalized position along body (0=head, 1=tail)

        # =======================================================================
        # Phase lag (ψ): ramps during prep (paper's approach)
        # Reaches full value at T_prep, creating traveling wave from the start
        # =======================================================================
        ψ_factor = smooth_ramp(t, 0.0, T_prep, 0.0, 1.0)
        ψ_i = s_i * ψ_tail * ψ_factor

        # =======================================================================
        # C-bend (B): ramps up during prep, then decays during propulsive stroke
        # Paper: "By the end of this cycle, at t = T_prep + T_prop, the baseline
        # curvature has returned to zero"
        # =======================================================================
        B_up = smooth_ramp(t, 0.0, T_prep, 0.0, 1.0)

        # B decays during propulsive stroke, reaching zero at T_prep + T_prop
        B_down = smooth_ramp(t, T_prep, T_prep + 2*T_prop, 1.0, 0.0)

        # Combined: ramp up during prep, then ramp down during prop
        B_envelope = B_up * B_down

        B_i = B_prep[i] * B_envelope

        # =======================================================================
        # Undulation amplitude (K): ramps during prep (paper's approach)
        # Both B and K reach full values at T_prep
        # =======================================================================
        K_factor = smooth_ramp(t, 0.0, T_prep, 0.0, 1.0)
        K_i = K_prop * K_factor

        # =======================================================================
        # Combined curvature: κ = B + K * sin(2πft - ψ + φ)
        # During prep: sin(arg) ≈ 1, so κ ≈ B + K (static C-bend)
        # After prep: sinusoid oscillates, creating propulsive wave
        # =======================================================================
        θ_raw = B_i + K_i * sin(2π * f * t - ψ_i + φ)

        # Apply smooth joint limits
        θ_joints[i] = smooth_clamp(θ_raw, joint_angle_limit)
    end

    # Prescribed mode: return only desired joint angles [θ₁, θ₂, ..., θ_n]
    return θ_joints
end

#############################################################################################
## Define initial RExEel state
#############################################################################################

# Start with eel in straight configuration at rest (C-start begins from straight line)
q_min = zeros(n_links + 2)
q_min[1] = start_x  # x position of head
q_min[2] = start_y  # y position of head
q_min[3] = start_θ  # vertical orientation

maximal_config = rex_eel_maximal_from_minimal(rexeel, q_min, n_links)
maximal_velocity = zeros(3 * n_links)
initial_body_state = vcat(maximal_config, maximal_velocity)

swimmer_initial_state = initialize_solid_state(rexeel, initial_body_state)

# test positional joint constraints satisfied
positional_residual = calculate_system_constraint_residual(rexeel,
    swimmer_initial_state[rexeel.configuration_indices]
)
@test positional_residual ≈ zeros(length(positional_residual)) atol=1e-8

println("Initial state check passed: all constraints satisfied")
println()

#############################################################################################
## Initialize aquarium state
#############################################################################################

# Initialize fluid at rest (zero velocity everywhere)
fluid_initial_velocity = zeros(fluid_env.n_velocities)

aquarium_state_0 = initialize_aquarium_state(
    tank,
    fluid_initial_velocity,
    swimmer_initial_state[rexeel.body_state_indices]
)

#############################################################################################
## Define objective functions
#############################################################################################

# COM position for direction objective
function calculate_com_position(tank, aquarium_state)
    total_mass = sum(b.mass for b in tank.swimmer.bodies)
    com_x = zero(eltype(aquarium_state))
    com_y = zero(eltype(aquarium_state))

    for i in 1:tank.swimmer.n_bodies
        cfg_indices = tank.swimmer_configuration_indices[body_configuration_indices(i)]
        link_mass = tank.swimmer.bodies[i].mass
        com_x += link_mass * aquarium_state[cfg_indices[1]]
        com_y += link_mass * aquarium_state[cfg_indices[2]]
    end

    return [com_x / total_mass, com_y / total_mass]
end

initial_com_position = calculate_com_position(tank, aquarium_state_0)

include("displacement_cost_functions.jl")

x_weight = 1.0
y_weight = 1.0
orientation_weight = 0.1
target_x = 86.0
target_y = 62.0
target_objective = 1.6
target_θ = deg2rad(-180.0)

global desired_direction_angle = desired_direction_angle
global eel_length = eel_length
global initial_com_position = initial_com_position
global x_weight = x_weight
global y_weight = y_weight
global orientation_weight = orientation_weight
global target_x = target_x
global target_y = target_y
global target_objective = target_objective
global target_θ = target_θ

println("Cost functions loaded:")
println("  Stage cost: none")
println("  Terminal cost: ((com_x - $(target_x))/L)² + ((com_y - $(target_y))/L)² + (1 - cos(θ_avg - $(rad2deg(desired_direction_angle))°))² at t=$(final_time)s")
println()

#############################################################################################
## Test analytical gradients against ForwardDiff
#############################################################################################

println("Testing analytical gradients against ForwardDiff...")
println()

# Create a random aquarium state for testing
test_aquarium_state = randn(length(aquarium_state_0))
test_bluff_body_state = zeros(0)
test_swimmer_control = zeros(2 * n_joints)
test_time = final_time - 0.1  # Within averaging window

# Test stage cost gradients
println("Testing stage cost gradients:")
grad_analytical = calculate_stage_objective_gradients(
    tank, test_time, test_aquarium_state, test_bluff_body_state, test_swimmer_control
)[1]
grad_fd_aquarium = ForwardDiff.gradient(
    x -> calculate_stage_objective(tank, test_time, x, test_bluff_body_state, test_swimmer_control),
    test_aquarium_state
)

@test grad_fd_aquarium ≈ grad_analytical rtol=1e-6
println("  ✓ Stage cost gradients match!")
println()

# Test terminal cost gradients
println("Testing terminal cost gradients:")
grad_analytical_term = calculate_terminal_objective_gradients(
    tank, final_time, test_aquarium_state, test_bluff_body_state
)[1]
grad_fd_aquarium_term = ForwardDiff.gradient(
    x -> calculate_terminal_objective(tank, final_time, x, test_bluff_body_state),
    test_aquarium_state
)

@test grad_fd_aquarium_term ≈ grad_analytical_term rtol=1e-6
println("  ✓ Terminal cost gradients match!")
println()

println("All gradient tests passed!")
println()

#############################################################################################
## Define optimization objective function
#############################################################################################

# Counter for optimization iterations
iteration_count = Ref(0)

# Global log file handle (will be set before optimization)
log_file_handle = Ref{Union{IOStream, Nothing}}(nothing)

# Cache for storing last simulation results
mutable struct SimulationCache
    x::Vector{Float64}           # Cached parameter values
    obj_value::Float64           # Cached objective value
    gradient::Vector{Float64}    # Cached gradient
    trajectories::Any            # Cached full trajectories (for final visualization)
    is_valid::Bool               # Whether cache contains valid data
end

# Initialize empty cache
sim_cache = SimulationCache(Float64[], 0.0, Float64[], nothing, false)

#############################################################################################
## Simulation wrapper (no parameter normalization)
#############################################################################################

function run_simulation_with_gradients(opt_params)
    # opt_params are already in physical space (no normalization)

    # opt_params = [B_prep (n_joints), ψ_tail, φ]
    # control_params order: [B_prep (n_joints), K_prop, ψ_tail, φ, T_prep, T_prop]
    B_prep_opt = opt_params[1:n_joints]
    ψ_tail_opt = opt_params[n_joints + 1]
    φ_opt = opt_params[n_joints + 2]
    control_params = [B_prep_opt..., K_prop_fixed, ψ_tail_opt, φ_opt, T_prep, T_prop]

    # Run simulation with gradient computation
    # Relax tolerances to reduce numerical noise in gradients
    trajectories = simulate_aquarium(
        tank,
        aquarium_state_0,
        final_time,
        zeros(0),  # No bluff body
        control_params;
        is_midpoint_bluff_body=false,
        pivot_type=:metis,
        scaling_type=:ruiz,
        solver_type=:gmres,
        preconditioner_type=:ilu,
        lazy=true,
        ilu_drop_tolerance=1e-2,
        newton_tolerance=1e-6,
        gmres_tolerance=1e-6,
        dual_regularization=1e-6,
        # primal_regularization=1e-6,
        max_newton_iterations=20,
        gmres_memory=100,
        gmres_max_iterations=1000,
        calculate_objective=true,
        calculate_gradient_wrt_control_params=true,
        calculate_gradient_wrt_fluid_properties=false,
        calculate_gradient_wrt_swimmer_params=false,
        calculate_gradient_wrt_bluff_body_params=false,
        calculate_gradient_wrt_bluff_body_state_params=false,
        calculate_control_input_from_params=calculate_control_input_from_params,
        calculate_stage_objective=calculate_stage_objective,
        calculate_terminal_objective=calculate_terminal_objective,
        calculate_stage_objective_gradients=calculate_stage_objective_gradients,
        calculate_terminal_objective_gradients=calculate_terminal_objective_gradients,
        verbose=false
    )

    # Extract objective value and gradient
    obj_value = trajectories[:objective_value][1]
    full_gradient = trajectories[:objective_gradient_wrt_control_params]

    # Extract gradients for optimization parameters only
    # full_gradient order: [B_prep (n_joints), K_prop, ψ_tail, φ, T_prep, T_prop]
    # We optimize: [B_prep (n_joints), ψ_tail, φ]
    # So extract indices 1:n_joints for B_prep, skip n_joints+1 (K_prop),
    # get n_joints+2 (ψ_tail) and n_joints+3 (φ)
    gradient = [full_gradient[1:n_joints]..., full_gradient[n_joints + 2], full_gradient[n_joints + 3]]

    return obj_value, gradient, trajectories, opt_params
end

# Objective function for LBFGSB: returns scalar
# x is in physical space (no normalization)
function f_objective(x)
    iteration_count[] += 1

    # Run simulation and cache results (x already in physical space)
    obj, grad, traj, _ = run_simulation_with_gradients(x)

    # Smoothness regularization: penalize differences between adjacent joint bending
    # Only applied to B_prep parameters (first n_joints elements)
    B_prep_current = x[1:n_joints]
    smoothness_term = 0.0
    smoothness_grad = zeros(length(x))

    for i in 1:(n_joints-1)
        diff = B_prep_current[i+1] - B_prep_current[i]
        smoothness_term += tikhonov_lambda_smooth * diff^2

        # Gradient: ∂/∂B_i [(B_{i+1} - B_i)²] = -2(B_{i+1} - B_i)
        # Gradient: ∂/∂B_{i+1} [(B_{i+1} - B_i)²] = 2(B_{i+1} - B_i)
        smoothness_grad[i] -= 2 * tikhonov_lambda_smooth * diff
        smoothness_grad[i+1] += 2 * tikhonov_lambda_smooth * diff
    end

    # Add regularization to objective and gradient
    obj_regularized = obj + smoothness_term
    grad_regularized = grad .+ smoothness_grad

    # Store in cache for gradient function to use
    sim_cache.x = copy(x)
    sim_cache.obj_value = obj_regularized
    sim_cache.gradient = grad_regularized
    sim_cache.trajectories = traj
    sim_cache.is_valid = true

    # Print progress (to both console and log file)
    grad_norm = norm(grad_regularized)
    msg1 = "  Objective (raw): $(obj), Smoothness penalty: $(smoothness_term), Total: $(obj_regularized)"
    msg2 = "  Parameters: B_prep=$(rad2deg.(x[1:n_joints]))°, ψ_tail=$(x[n_joints+1]) rad (wavelength=$(x[n_joints+1]/2π)), φ=$(x[n_joints+2]) rad"
    msg3 = "  Gradient: $(grad_regularized)"
    msg4 = "  Gradient magnitude: $(grad_norm)"

    println(stderr, " ")
    println(stderr, msg1)
    println(stderr, msg2)
    println(stderr, msg3)
    println(stderr, msg4)
    println(stderr, " ")

    # Also write to log file if available
    if log_file_handle[] !== nothing
        println(log_file_handle[], msg1)
        println(log_file_handle[], msg2)
        println(log_file_handle[], msg3)
        println(log_file_handle[], msg4)
        flush(log_file_handle[])
    end

    return obj_regularized
end

# Gradient function for LBFGSB: modifies z in-place
# x is in physical space (no normalization)
function g_gradient!(z, x)
    # Check if cache is valid and matches current x
    if sim_cache.is_valid && sim_cache.x == x
        # Use cached gradient
        z .= sim_cache.gradient
    else
        # Cache miss - need to run simulation (shouldn't happen in normal LBFGSB usage)
        println("  [Cache miss - running simulation for gradient]")
        _, grad, traj, _ = run_simulation_with_gradients(x)
        z .= grad

        # Update cache
        sim_cache.x = copy(x)
        sim_cache.obj_value = 0.0  # Not computed here
        sim_cache.gradient = grad
        sim_cache.trajectories = traj
        sim_cache.is_valid = true
    end
    return nothing
end

#############################################################################################
## Gradient verification with finite differences
#############################################################################################

# println("Verifying analytical gradients against finite differences...")

# # Compute finite difference gradient using FiniteDiff package
# fd_gradient = FiniteDiff.finite_difference_gradient(f_objective, x0)
# initial_grad = sim_cache.gradient

# # Compare gradients
# abs_error = abs.(initial_grad - fd_gradient)

# # Check if gradients match within tolerance
# gradient_match = all(abs_error .< 1e-4)
# if gradient_match
#     println("  ✓ Gradients match within tolerance!")
# else
#     println("  ⚠ Warning: Gradients may not match - check relative/absolute errors above")
# end
# println()

#############################################################################################
## Run optimization
#############################################################################################

println("="^80)
println("Starting C-Start Trajectory Optimization")
println("="^80)
println()

# Run optimization
println("Starting optimization iterations...")

# No normalization - use physical parameters directly
println("Optimization parameters (physical space):")
println("  Initial guess: $(x0)")
println("  Lower bounds: $(lower_bounds)")
println("  Upper bounds: $(upper_bounds)")
println("  Smoothness regularization lambda: $(tikhonov_lambda_smooth)")
println()

# Create log file for optimization output
log_file = joinpath(vis_dir, "optimization_log_$(rad2deg(desired_direction_angle))_deg_physical.txt")
log_file_handle[] = open(log_file, "w")

println(log_file_handle[], "C-Start Trajectory Optimization Log (physical parameters, smoothness regularization)")
println(log_file_handle[], "="^80)
println(log_file_handle[], "Optimization started at: $(now())")
println(log_file_handle[], "Parameter bounds: lb=$(lower_bounds), ub=$(upper_bounds)")
println(log_file_handle[], "Smoothness regularization lambda: $(tikhonov_lambda_smooth)")
println(log_file_handle[], "Regularization type: Penalizes differences between adjacent joint bending angles")
println(log_file_handle[], "="^80)
flush(log_file_handle[])

# Redirect stdout to log file to capture LBFGSB output
original_stdout = stdout
redirect_stdout(log_file_handle[])

try
    # Run optimization - LBFGSB output will go to log file
    global fout, xout
    fout, xout = lbfgsb(f_objective, g_gradient!, x0;
        lb=lower_bounds,
        ub=upper_bounds,
        m=3,            # Number of limited memory corrections
        factr=1e7,       # Convergence tolerance
        pgtol=1e-6,     # Projected gradient tolerance
        iprint=101,     # Print every iteration: -1=none, 0=final, 1=every iter, >1=more detail
        maxiter=50,     # Maximum iterations
        maxfun=200      # Maximum function evaluations
    )
finally
    # Restore stdout
    redirect_stdout(original_stdout)
end

# Close log file
println(log_file_handle[], "="^80)
println(log_file_handle[], "Optimization complete at: $(now())")
println(log_file_handle[], "="^80)
close(log_file_handle[])
log_file_handle[] = nothing

println("Optimization log saved to: $log_file")

# xout is already in physical space (no denormalization needed)

println()
println("="^80)
println("Optimization Complete!")
println("="^80)
println()

B_prep_opt = xout[1:n_joints]
ψ_tail_opt = xout[n_joints + 1]
φ_opt = xout[n_joints + 2]

println("Optimized Parameters:")
println("  B_prep: $(rad2deg.(B_prep_opt))°")
println("  ψ_tail: $(ψ_tail_opt) rad (wavelength = $(ψ_tail_opt/2π))")
println("  φ: $(φ_opt) rad")
println("  K_prop (fixed): $(rad2deg(K_prop_fixed))°")
println("  Final objective: $(fout)")
println()