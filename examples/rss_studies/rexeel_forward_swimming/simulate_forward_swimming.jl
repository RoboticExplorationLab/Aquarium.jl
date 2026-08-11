import Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))

using AquariumClosed
using AquariumClosed.LinearAlgebra
using AquariumClosed.ForwardDiff
using JLD2
using Test

#############################################################################################
## Configuration
#############################################################################################

# Paths
output_dir = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/")

# Create output directory if it doesn't exist
mkpath(output_dir)

# Amplitudes to simulate (in degrees)
amplitudes = [10, 20, 30, 40]

println("="^80)
println("Forward Swimming Simulation - Amplitude Study")
println("="^80)
println("Simulating amplitudes: ", amplitudes, "°")
println()

#############################################################################################
## Define simulation parameters (constant across all trials)
#############################################################################################

# time properties - fixed at 60 Hz for 4 seconds
time_step = 1.0 / 60.0  # 60 Hz (0.01667 s)
final_time = 4.0
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
println("  Time step: $(time_step) s (60 Hz)")
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
    n_boundary_nodes_per_link = n_boundary_nodes,
    ib_method = :weak_form,
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
## Define control inputs (constant)
#############################################################################################

n_joints_ctrl = n_links - 1

# Phase parameters for traveling wave (constant across all amplitudes)
ψ_tail = 2π * 0.7  # Total phase lag head-to-tail [rad]

# Frequency (constant across all amplitudes)
f = 0.5  # Undulation frequency [Hz]

# Ramp-up time (constant across all amplitudes)
T_ramp = 0.5/f  # Time to reach full amplitude [s]

function calculate_control_input_from_params(solid_system, t, control_params)

    # Smooth ramp: cubic interpolation with zero derivatives at endpoints
    function smooth_ramp(t, t_start, t_end, val_start, val_end)
        τ = (t - t_start) / (t_end - t_start)
        ramp_val = val_start + (val_end - val_start) * (3τ^2 - 2τ^3)
        return ifelse(t <= t_start, val_start, ifelse(t >= t_end, val_end, ramp_val))
    end

    n_joints = length(solid_system.joints)

    # Extract parameters: [K_prop, ψ_tail, f, T_ramp]
    K_prop = control_params[1]
    ψ_tail = control_params[2]
    f = control_params[3]
    T_ramp = control_params[4]

    # Preallocate output
    T = promote_type(typeof(K_prop), typeof(t))
    θ_joints = zeros(T, n_joints)

    for i in 1:n_joints
        s_i = i / n_joints  # Normalized position along body (0=head, 1=tail)

        # Phase lag (ψ): constant phase shift for traveling wave
        ψ_i = s_i * ψ_tail

        # Undulation amplitude (K): ramps up during T_ramp period
        K_factor = smooth_ramp(t, 0.0, T_ramp, 0.0, 1.0)
        K_i = K_prop * K_factor

        # Traveling wave: θ = K * sin(2πft - ψ)
        θ_joints[i] = K_i * sin(2π * f * t - ψ_i)
    end

    # Prescribed mode: return only desired joint angles [θ₁, θ₂, ..., θ_n]
    return θ_joints
end

println("Control configuration:")
println("  Frequency: $(f) Hz")
println("  Phase lag: $(ψ_tail) rad")
println("  Ramp time: $(T_ramp) s")
println()

#############################################################################################
## Process each amplitude
#############################################################################################

for (amp_idx, amplitude_deg) in enumerate(amplitudes)

    println("="^80)
    println("Simulation $amp_idx / $(length(amplitudes)): Amplitude = $(amplitude_deg)°")
    println("="^80)

    # Convert amplitude to radians
    K_prop = deg2rad(amplitude_deg)
    control_params = [K_prop, ψ_tail, f, T_ramp]

    #############################################################################
    ## Initialize state with robot head at center of environment
    #############################################################################

    # Center position of the environment
    x_center = length_x / 2.0
    y_center = length_y / 2.0
    θ_initial = -pi/2

    println("  Initial Conditions:")
    println("    Position: ($(round(x_center, digits=2)), $(round(y_center, digits=2))) cm (center)")
    println("    Orientation: $(round(rad2deg(θ_initial), digits=2))°")
    println("    Amplitude: $(amplitude_deg)°")

    # Start with eel in straight configuration at center position
    q_min = zeros(n_links + 2)
    q_min[1] = x_center  # x position of head at center
    q_min[2] = y_center  # y position of head at center
    q_min[3] = θ_initial  # orientation
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
        max_newton_iterations=20,
        gmres_memory=100,
        gmres_max_iterations=1000,
        calculate_control_input_from_params=calculate_control_input_from_params,
        verbose=false
    )

    println("  ✓ Simulation complete")

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
            vel_indices = rexeel.velocity_indices[(3*(i-1)+1):(3*i)]
            vx_i = state[rexeel.body_state_indices[vel_indices[1]]]
            vy_i = state[rexeel.body_state_indices[vel_indices[2]]]
            com_vx[t] += masses_per_link[i] * vx_i / total_mass
            com_vy[t] += masses_per_link[i] * vy_i / total_mass
        end
    end

    com_speed = sqrt.(com_vx.^2 + com_vy.^2)

    #############################################################################
    ## Print summary statistics
    #############################################################################

    println("  Results:")
    println("    Forward (X) displacement: $(round(com_x[end] - com_x[1], digits=2)) cm")
    println("    Lateral (Y) displacement: $(round(com_y[end] - com_y[1], digits=2)) cm")
    println("    Max COM speed: $(round(maximum(com_speed), digits=2)) cm/s")
    println("    Final forward velocity: $(round(com_vx[end], digits=2)) cm/s")

    #############################################################################
    ## Save simulation results
    #############################################################################

    # Determine output filename
    save_file = joinpath(output_dir, "$(amplitude_deg)deg/$(amplitude_deg)deg_simulation.jld2")

    jldsave(save_file;
        trajectories,
        amplitude_deg,
        control_params,
        initial_conditions = Dict(
            "x" => x_center,
            "y" => y_center,
            "theta" => θ_initial
        ),
        summary_stats = Dict(
            "forward_displacement" => com_x[end] - com_x[1],
            "lateral_displacement" => com_y[end] - com_y[1],
            "max_speed" => maximum(com_speed),
            "final_vx" => com_vx[end],
            "final_vy" => com_vy[end],
            "final_x" => com_x[end],
            "final_y" => com_y[end]
        )
    )

    println("  ✓ Saved to: $(amplitude_deg)deg/$(amplitude_deg)deg_simulation.jld2")
    println()

end  # End of loop over amplitudes

#############################################################################################
## Summary
#############################################################################################

println("="^80)
println("AMPLITUDE STUDY COMPLETE")
println("="^80)

# Count processed files
simulated_files = filter(f -> endswith(f, "_simulation.jld2"), readdir(output_dir))
println("Successfully simulated: $(length(simulated_files)) amplitudes")
println("Amplitudes tested: ", amplitudes, "°")
println("Output directory: $output_dir")
println("="^80)