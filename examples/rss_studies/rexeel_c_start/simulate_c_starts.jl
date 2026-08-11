import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))

using AquariumClosed
using AquariumClosed.LinearAlgebra
using AquariumClosed.ForwardDiff
using BenchmarkTools
using JLD2
using Test

#############################################################################################
## Configuration
#############################################################################################

# Paths
output_dir = expanduser("~/aquariumCLOSED/data/rexeel_c_start/")

# Create output directory if it doesn't exist
mkpath(output_dir)

# Parameter sets to simulate
parameter_set_names = ["initial", "optimal"]

println("="^80)
println("C-Start Simulation - Initial vs Optimal Parameters")
println("="^80)
println("Simulating parameter sets: ", parameter_set_names)
println()

#############################################################################################
## Define simulation parameters (constant across all trials)
#############################################################################################

# time properties
time_step = 1/60
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
## Create fluid environment (constant)
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

println("Fluid environment created:")
println("  Domain: $(length_x) cm × $(length_y) cm")
println("  Grid: $(num_cells_x) × $(num_cells_y) cells")
println("  Boundary conditions: $(boundary_condition_type)")
println("  Time step: $(time_step) s")
println()

#############################################################################################
## Define 6-link RExEel (swimmer) - constant
#############################################################################################

# eel properties
n_links = 6
link_lengths = [12.0, 9.8 .* ones(n_links-1)...]  # cm
height = 9.35  # cm
masses_per_link = [192, 140 .* ones(n_links-1)...] ./ height # g per link
moi_per_link = [2435.99, 1483.49 .* ones(n_links-1)...] ./ height  # g·cm²
gravity_constant = 0.0

# boundary properties - compute per-link boundary nodes based on link length
n_boundary_nodes = floor.(Int, link_lengths ./ fluid_env.fvm_grid.h_x)

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
    stall_torques = max_torques_rex,
    n_boundary_nodes_per_link = n_boundary_nodes,
    ib_method = :weak_form,
    discrete_delta_kind = :three_point,
    gravity = [0.0, -gravity_constant],
    actuation_mode=:prescribed,
)

println("RExEel Configuration:")
for i in 1:n_links
    println("  Link $i: length=$(link_lengths[i])cm, mass=$(masses_per_link[i])g, moi=$(moi_per_link[i])g/cm²")
end
println("  Total length: $(sum(link_lengths))cm")
println("  Total mass: $(sum(masses_per_link))g")
println()

#############################################################################################
## Create AquariumTank (constant)
#############################################################################################

tank = AquariumTank_only_swimmer(fluid_env, rexeel)

println("AquariumTank created:")
println("  Fluid states: ", tank.n_fluid_states)
println("  Swimmer states: ", tank.n_swimmer_states)
println("  No-slip constraints: ", tank.n_no_slip_constraints)
println("  Total aquarium states: ", tank.n_states)
println()

#############################################################################################
## Define control inputs for C-start escape maneuver
#############################################################################################

n_joints = n_links - 1

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

        # Phase lag (ψ): ramps during prep
        ψ_factor = smooth_ramp(t, 0.0, T_prep, 0.0, 1.0)
        ψ_i = s_i * ψ_tail * ψ_factor

        # C-bend (B): ramps up during prep, then decays during propulsive stroke
        B_up = smooth_ramp(t, 0.0, T_prep, 0.0, 1.0)

        # B decays during propulsive stroke, reaching zero at T_prep + T_prop
        B_down = smooth_ramp(t, T_prep, T_prep + 2*T_prop, 1.0, 0.0)

        # Combined: ramp up during prep, then ramp down during prop
        B_envelope = B_up * B_down

        B_i = B_prep[i] * B_envelope

        # Undulation amplitude (K): ramps during prep
        K_factor = smooth_ramp(t, 0.0, T_prep, 0.0, 1.0)
        K_i = K_prop * K_factor

        # Combined curvature: κ = B + K * sin(2πft - ψ + φ)
        θ_raw = B_i + K_i * sin(2π * f * t - ψ_i + φ)

        # Apply smooth joint limits
        θ_joints[i] = smooth_clamp(θ_raw, joint_angle_limit)
    end

    # Prescribed mode: return only desired joint angles [θ₁, θ₂, ..., θ_n]
    return θ_joints
end

#############################################################################################
## Define parameter sets
#############################################################################################

# Fixed timing parameters
T_prep = 1.0   # Preparatory stroke duration [s]
T_prop = 1.0   # Propulsive phase duration [s]

# Fixed parameter (constant across initial and optimal)
K_prop_fixed = deg2rad(25.0)  # Undulation amplitude [rad]

# Initial parameters (from optimization script)
B_prep_initial = deg2rad.([10, 10, 10, 10, 10])
ψ_tail_initial = 2π * 0.6      # Total phase lag head-to-tail [rad]
φ_initial = -pi/3

# Optimal parameters (these should be updated with actual optimization results)
# For now, using example values - replace with actual optimized values
B_prep_optimal = deg2rad.([17, 18, 20, 24, 36.0])  # Example - update with actual values
ψ_tail_optimal = 2π * 0.57  # Example - update with actual values
φ_optimal = -1.16  # Example - update with actual values

# Store parameter sets in a dictionary
parameter_sets = Dict(
    "initial" => [B_prep_initial..., K_prop_fixed, ψ_tail_initial, φ_initial, T_prep, T_prop],
    "optimal" => [B_prep_optimal..., K_prop_fixed, ψ_tail_optimal, φ_optimal, T_prep, T_prop],
)

println("Control configuration:")
println("  Fixed parameters:")
println("    K_prop: $(rad2deg(K_prop_fixed))°")
println("    T_prep: $(T_prep) s")
println("    T_prop: $(T_prop) s")
println()
println("  Initial parameters:")
println("    B_prep: $(rad2deg.(B_prep_initial))°")
println("    ψ_tail: $(ψ_tail_initial) rad (wavelength = $(ψ_tail_initial/2π))")
println("    φ: $(φ_initial) rad")
println()
println("  Optimal parameters:")
println("    B_prep: $(rad2deg.(B_prep_optimal))°")
println("    ψ_tail: $(ψ_tail_optimal) rad (wavelength = $(ψ_tail_optimal/2π))")
println("    φ: $(φ_optimal) rad")
println()

#############################################################################################
## Process each parameter set
#############################################################################################

for (set_idx, param_set_name) in enumerate(parameter_set_names)

    println("="^80)
    println("Simulation $set_idx / $(length(parameter_set_names)): $(param_set_name) parameters")
    println("="^80)

    # Get control parameters for this set
    control_params = parameter_sets[param_set_name]

    #############################################################################
    ## Initialize state with robot at starting position
    #############################################################################

    # Starting position (same as optimization script)
    start_x = 50.0
    start_y = length_y / 2.0
    start_θ = -π/2  # vertical orientation pointing downward

    println("  Initial Conditions:")
    println("    Position: ($(round(start_x, digits=2)), $(round(start_y, digits=2))) cm")
    println("    Orientation: $(round(rad2deg(start_θ), digits=2))°")

    # Start with eel in straight configuration at rest
    q_min = zeros(n_links + 2)
    q_min[1] = start_x  # x position of head
    q_min[2] = start_y  # y position of head
    q_min[3] = start_θ  # orientation
    # Joint angles start at zero (straight configuration)

    maximal_config = rex_eel_maximal_from_minimal(rexeel, q_min, n_links)
    maximal_velocity = zeros(3 * n_links)
    initial_body_state = vcat(maximal_config, maximal_velocity)

    swimmer_initial_state = initialize_solid_state(rexeel, initial_body_state)

    # Test positional joint constraints satisfied
    positional_residual = calculate_system_constraint_residual(rexeel,
        swimmer_initial_state[rexeel.configuration_indices]
    )
    @test positional_residual ≈ zeros(length(positional_residual)) atol=1e-8

    println("  ✓ Initial state validated")

    # Initialize fluid at rest (zero velocity everywhere)
    fluid_initial_velocity = zeros(fluid_env.n_velocities)

    aquarium_state_0 = initialize_aquarium_state(
        tank,
        fluid_initial_velocity,
        swimmer_initial_state[rexeel.body_state_indices]
    )

    #############################################################################
    ## Simulate aquarium dynamics
    #############################################################################

    println("  Starting simulation...")

    trajectories_ref = Ref{Any}(nothing)
    bench = @benchmark $(trajectories_ref)[] = simulate_aquarium(
        $tank,
        $aquarium_state_0,
        $final_time,
        $(zeros(0)),  # No bluff body
        $control_params;
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
        max_newton_iterations=20,
        gmres_memory=100,
        gmres_max_iterations=1000,
        calculate_control_input_from_params=$calculate_control_input_from_params,
        verbose=false
    ) samples=1 evals=1
    trajectories = trajectories_ref[]

    println("  ✓ Simulation complete")
    println()
    println("  === Rollout Solve Benchmark ===")
    display(bench)
    println()

    #############################################################################
    ## Extract and analyze results
    #############################################################################

    # Extract trajectories
    time_traj = trajectories[:time_traj]
    swimmer_state_traj = trajectories[:swimmer_state_traj]
    N_time_sim = length(time_traj)

    # Extract swimmer configuration trajectories
    swimmer_configuration_traj = [swimmer_state_traj[k][rexeel.configuration_indices] for k in 1:N_time_sim]

    # Calculate center of mass trajectory
    com_x = zeros(N_time_sim)
    com_y = zeros(N_time_sim)

    for t in 1:N_time_sim
        config = swimmer_configuration_traj[t]
        total_mass = sum(masses_per_link)

        for i in 1:n_links
            x_i = config[3*(i-1) + 1]
            y_i = config[3*(i-1) + 2]
            com_x[t] += masses_per_link[i] * x_i / total_mass
            com_y[t] += masses_per_link[i] * y_i / total_mass
        end
    end

    # Extract COM velocity
    com_vx = zeros(N_time_sim)
    com_vy = zeros(N_time_sim)

    for t in 1:N_time_sim
        state = swimmer_state_traj[t]
        total_mass = sum(masses_per_link)

        for i in 1:n_links
            vel_indices = (3i - 2):(3i)
            vx_i = state[rexeel.body_state_indices[rexeel.velocity_indices[vel_indices[1]]]]
            vy_i = state[rexeel.body_state_indices[rexeel.velocity_indices[vel_indices[2]]]]
            com_vx[t] += masses_per_link[i] * vx_i / total_mass
            com_vy[t] += masses_per_link[i] * vy_i / total_mass
        end
    end

    com_speed = sqrt.(com_vx.^2 + com_vy.^2)

    # Calculate total displacement
    total_displacement = sqrt((com_x[end] - com_x[1])^2 + (com_y[end] - com_y[1])^2)

    #############################################################################
    ## Print summary statistics
    #############################################################################

    println("  Results:")
    println("    Forward (X) displacement: $(round(com_x[end] - com_x[1], digits=2)) cm")
    println("    Lateral (Y) displacement: $(round(com_y[end] - com_y[1], digits=2)) cm")
    println("    Total displacement: $(round(total_displacement, digits=2)) cm")
    println("    Max COM speed: $(round(maximum(com_speed), digits=2)) cm/s")
    println("    Final velocity: ($(round(com_vx[end], digits=2)), $(round(com_vy[end], digits=2))) cm/s")

    #############################################################################
    ## Save simulation results
    #############################################################################

    # Determine output filename
    save_file = joinpath(output_dir, "$(param_set_name)/$(param_set_name)_simulation.jld2")

    # Create subdirectory if needed
    mkpath(dirname(save_file))

    # Extract B_prep, ψ_tail, and φ from control_params
    B_prep = control_params[1:n_joints]
    K_prop = control_params[n_joints + 1]
    ψ_tail = control_params[n_joints + 2]
    φ = control_params[n_joints + 3]

    jldsave(save_file;
        trajectories,
        parameter_set_name = param_set_name,
        control_params,
        B_prep = B_prep,
        K_prop = K_prop,
        ψ_tail = ψ_tail,
        φ = φ,
        T_prep = T_prep,
        T_prop = T_prop,
        initial_conditions = Dict(
            "x" => start_x,
            "y" => start_y,
            "theta" => start_θ
        ),
        summary_stats = Dict(
            "forward_displacement" => com_x[end] - com_x[1],
            "lateral_displacement" => com_y[end] - com_y[1],
            "total_displacement" => total_displacement,
            "max_speed" => maximum(com_speed),
            "final_vx" => com_vx[end],
            "final_vy" => com_vy[end],
            "final_x" => com_x[end],
            "final_y" => com_y[end]
        )
    )

    println("  ✓ Saved to: $(param_set_name)/$(param_set_name)_simulation.jld2")
    println()

end  # End of loop over parameter sets

#############################################################################################
## Summary
#############################################################################################

println("="^80)
println("C-START SIMULATION COMPLETE")
println("="^80)

# Count processed files
simulated_files = []
for param_set_name in parameter_set_names
    file_path = joinpath(output_dir, "$(param_set_name)/$(param_set_name)_simulation.jld2")
    if isfile(file_path)
        push!(simulated_files, param_set_name)
    end
end

println("Successfully simulated: $(length(simulated_files)) parameter sets")
println("Parameter sets tested: ", parameter_set_names)
println("Output directory: $output_dir")
println("="^80)
