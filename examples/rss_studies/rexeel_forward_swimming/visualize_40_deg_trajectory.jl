import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using AquariumClosed
using AquariumClosed.CairoMakie
using JLD2
using LinearAlgebra
using Statistics
using Colors
using PGFPlotsX
using CSV
using DataFrames
using Interpolations
using VideoIO
using Printf

#############################################################################################
## Configuration
#############################################################################################

# Data paths
hardware_tracking_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/40deg/40deg_hardware_trajectories.csv")
simulation_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/40deg/40deg_simulation.jld2")
motor_angles_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/40deg/40deg_hardware_motor_angles.csv")
video_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/40deg/Camo 录像 2026-01-27 01-11-15.mov")

# Genesis simulation paths
genesis_tracking_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/genesis_simulation_trajectories.csv")
genesis_video_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/40deg/40deg_genesis_simulation.mp4")

# Output directory
output_dir = expanduser("~/aquariumCLOSED/visualization/rss_figures/40deg_forward_swimming")
mkpath(output_dir)

println("="^80)
println("RExEel 40deg Trial 1: Head Link Position Analysis")
println("="^80)
println()

#############################################################################################
## Load Data
#############################################################################################

println("Loading data...")

# Load hardware tracking data (CSV with multiple videos)
hw_data_df = CSV.read(hardware_tracking_path, DataFrame)

# Organize trajectories and timestamps by video name in dictionaries
hw_trajectories_by_video = Dict{String, Vector{Vector{Float64}}}()
hw_timestamps_by_video = Dict{String, Vector{Float64}}()

for video_name in unique(hw_data_df.video)
    # Filter data for this video
    video_data = hw_data_df[hw_data_df.video .== video_name, :]
    
    # Sort by frame number to ensure correct order
    sort!(video_data, :frame)
    
    # Extract trajectory as vector of [x, y, theta] vectors
    traj = [[row.x_cm, row.y_cm, deg2rad(row.robot_yaw)] for row in eachrow(video_data)]
    
    # Create timestamps as range (30fps video: 120 frames over 4 seconds)
    timestamps = range(0.0, (length(traj)-1)/30.0, length=length(traj))
    
    hw_trajectories_by_video[video_name] = traj
    hw_timestamps_by_video[video_name] = timestamps
end

println("  ✓ Hardware tracking loaded: $(length(hw_trajectories_by_video)) videos")
for (video_name, traj) in hw_trajectories_by_video
    println("    - $video_name: $(length(traj)) frames")
end

# Select the trajectory for the current analysis (matches the video file we're using)
# current_video_name = "Camo 录像 2026-01-27 01-11-15"
current_video_name = "Camo 录像 2026-01-27 01-11-15"
hw_traj = hw_trajectories_by_video[current_video_name]

println("  ✓ Selected video for analysis: $current_video_name ($(length(hw_traj)) frames)")

# Load genesis simulation tracking data (K40 only)
genesis_data_df = CSV.read(genesis_tracking_path, DataFrame)
genesis_video_name = "Simulation_K40"

# Filter for K40 only
genesis_data = genesis_data_df[genesis_data_df.video .== genesis_video_name, :]
sort!(genesis_data, :frame)

# Extract trajectory as vector of [x, y, theta] vectors
genesis_traj = [[row.x_cm, row.y_cm, deg2rad(row.robot_yaw)] for row in eachrow(genesis_data)]

# Create timestamps
genesis_timestamps = 0.0:1/60:4.0-1/60

println("  ✓ Genesis simulation (K40) loaded: $(length(genesis_traj)) frames")

# Load simulation data
sim_data = load(simulation_path)
trajectories = sim_data["trajectories"]
time_traj_sim = trajectories[:time_traj]
swimmer_state_traj = trajectories[:swimmer_state_traj]
aquarium_state_traj = trajectories[:aquarium_state_traj]

println("  ✓ Simulation loaded: $(length(swimmer_state_traj)) timesteps")

# Load hardware motor angles
motor_angles_df = CSV.read(motor_angles_path, DataFrame; comment="#")

println("  ✓ Hardware motor angles loaded: $(size(motor_angles_df, 1)) rows")

# Extract motor angles (5 joints)
time_traj_motors = range(0.0, motor_angles_df.time_s[end-1], length=size(motor_angles_df, 1))
actual_motor_angles = zeros(length(time_traj_motors), 5)
for i in 1:5
    actual_motor_angles[:, i] = deg2rad.(motor_angles_df[:, Symbol("actual_$i")])
end

println("  ✓ Motor angles extracted: 5 joints, $(length(time_traj_motors)) timesteps")

# Interpolate hardware motor angles to simulation time points
println("  Interpolating hardware motor angles to simulation time...")
actual_motor_angles_interp = zeros(length(time_traj_sim), 5)
for joint in 1:5
    # Create cubic spline interpolation for this joint
    itp = CubicSplineInterpolation(time_traj_motors, actual_motor_angles[:, joint], extrapolation_bc=Line())
    
    # Evaluate at simulation time points
    for i in 1:length(time_traj_sim)
        actual_motor_angles_interp[i, joint] = itp(time_traj_sim[i])
    end
end

println("  ✓ Motor angles interpolated to $(length(time_traj_sim)) simulation timesteps")

# Interpolate hardware head trajectories (ArUco tracking) to simulation time points
println("  Interpolating hardware head trajectories to simulation time...")
hw_trajectories_interp_by_video = Dict{String, Vector{Vector{Float64}}}()

for (video_name, traj) in hw_trajectories_by_video
    hw_timestamps = hw_timestamps_by_video[video_name]
    hw_timestamps = range(0.0, hw_timestamps[end-1], length=length(hw_timestamps))

    # Extract x, y, theta trajectories
    x_traj = [traj[i][1] for i in 1:length(traj)]
    y_traj = [traj[i][2] for i in 1:length(traj)]
    theta_traj = [traj[i][3] for i in 1:length(traj)]
    
    # Create interpolators for each component
    itp_x = CubicSplineInterpolation(hw_timestamps, x_traj, extrapolation_bc=Line())
    itp_y = CubicSplineInterpolation(hw_timestamps, y_traj, extrapolation_bc=Line())
    itp_theta = CubicSplineInterpolation(hw_timestamps, theta_traj, extrapolation_bc=Line())
    
    # Interpolate to simulation time points
    traj_interp = []
    for t_sim in time_traj_sim
        x_interp = itp_x(t_sim)
        y_interp = itp_y(t_sim)
        theta_interp = itp_theta(t_sim)
        push!(traj_interp, [x_interp, y_interp, theta_interp])
    end
    
    hw_trajectories_interp_by_video[video_name] = traj_interp
end

# Select current video for single-video analysis
hw_traj_interp = hw_trajectories_interp_by_video[current_video_name]

println("  ✓ Hardware head trajectories interpolated for $(length(hw_trajectories_interp_by_video)) videos to $(length(hw_traj_interp)) timesteps")

# Interpolate genesis head trajectory to simulation time points
println("  Interpolating genesis head trajectory to simulation time...")

genesis_timestamps_range = range(0.0, genesis_timestamps[end], length=length(genesis_timestamps))

# Extract x, y, theta trajectories
x_traj = [genesis_traj[i][1] for i in 1:length(genesis_traj)]
y_traj = [genesis_traj[i][2] for i in 1:length(genesis_traj)]
theta_traj = [genesis_traj[i][3] for i in 1:length(genesis_traj)]

# Create interpolators for each component
itp_x = CubicSplineInterpolation(genesis_timestamps_range, x_traj, extrapolation_bc=Line())
itp_y = CubicSplineInterpolation(genesis_timestamps_range, y_traj, extrapolation_bc=Line())
itp_theta = CubicSplineInterpolation(genesis_timestamps_range, theta_traj, extrapolation_bc=Line())

# Interpolate to simulation time points
genesis_traj_interp = []
for t_sim in time_traj_sim
    x_interp = itp_x(t_sim)
    y_interp = itp_y(t_sim)
    theta_interp = itp_theta(t_sim)
    push!(genesis_traj_interp, [x_interp, y_interp, theta_interp])
end

println("  ✓ Genesis head trajectory interpolated to $(length(genesis_traj_interp)) timesteps")
println()

#############################################################################################
## Plot params
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 800)
logocolors = Colors.JULIA_LOGO_COLORS

# Define colors
simulation_color = RGB(0.0, 0.7294, 0.3451)  # jj_green
simulation_color_opaque = RGBA(0.0, 0.7294, 0.3451, 1.0)  # jj_green fully opaque
hardware_color = RGB(0.933, 0.227, 0.275)  # jj_red
hardware_color_opaque = RGBA(0.933, 0.227, 0.275, 1.0)  # jj_red fully opaque
genesis_color = RGB(0.9451,0.6745,0.09020);  # jj_orange
genesis_color_opaque = RGBA(0.9451,0.6745,0.09020, 1.0);  # jj_orange fully opaque
println("colors defined")

#############################################################################################
## Define fluid domain (4ft x 4ft tank with wall boundaries)
#############################################################################################

# time properties
time_step = 1/60
final_time = 4.0  # Longer simulation for steady-state swimming
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
# Link lengths for 6-link eel
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

rexeel.plot_params[:bodycolor] = :black
rexeel.plot_params[:linewidth] = 10.0
rexeel.plot_params[:showboundaryvelocities] = false
rexeel.plot_params[:lengthscale] = 1.0
rexeel.plot_params[:showboundarynodes] = false
rexeel.plot_params[:boundarynodesize] = 10.0

tank = AquariumTank_only_swimmer(fluid_env, rexeel)

#############################################################################################
## Extract Maximal Config Trajectory of Sim
#############################################################################################

println("Extracting maximal configuration trajectories...")

sim_maximal_config_traj = [swimmer_state_traj[i][rexeel.configuration_indices] for i in 1:length(swimmer_state_traj)]


#############################################################################################
## Extract Motor Angles from Simulation Trajectory
#############################################################################################

println("Extracting motor angles from simulation trajectory...")

# Compute motor angles from maximal configuration trajectory
# Motor angle φ_i = θ_{i+1} - θ_i (relative angle between consecutive links)
sim_motor_angles_traj = []

for config in sim_maximal_config_traj
    motor_angles = zeros(n_links - 1)  # 5 joints for 6 links
    
    for joint_idx in 1:(n_links - 1)
        # Link indices (body_i and body_{i+1})
        link_i_idx = joint_idx
        link_ip1_idx = joint_idx + 1
        
        # Extract absolute angles θ_i and θ_{i+1}
        θ_i = config[3 * link_i_idx]        # θ of link i (3rd element of link's config)
        θ_ip1 = config[3 * link_ip1_idx]    # θ of link i+1
        
        # Compute relative angle φ_i = θ_{i+1} - θ_i
        motor_angles[joint_idx] = θ_ip1 - θ_i
    end
    
    push!(sim_motor_angles_traj, motor_angles)
end

println("  ✓ Extracted motor angles: $(length(sim_motor_angles_traj)) timesteps, $(n_links-1) joints")
println()

#############################################################################################
## Reconstruct Hardware Maximal Configurations
#############################################################################################

# Reconstruct hardware maximal configurations for ALL videos using interpolated head trajectories
hw_maximal_config_traj_by_video = Dict{String, Vector{Vector{Float64}}}()

for (video_name, traj_interp) in hw_trajectories_interp_by_video
    println("  Processing video: $video_name")
    
    hw_maximal_config_traj = []
    
    for i in 1:length(traj_interp)
        # Get head link pose from interpolated ArUco tracking (already at simulation timesteps)
        x_head = traj_interp[i][1]
        y_head = traj_interp[i][2]
        θ_head = -traj_interp[i][3]
        
        # Construct minimal configuration: [x0, y0, θ0, φ1, φ2, ..., φ_{n-1}]
        # x0, y0, θ0 are the head link pose
        # φi are the relative joint angles (motor angles)
        n_minimal_coords = n_links + 2  # n_links angles (including head) + x, y positions
        minimal_config = zeros(n_minimal_coords)
        minimal_config[1] = x_head
        minimal_config[2] = y_head 
        minimal_config[3] = θ_head
        minimal_config[4:end] = actual_motor_angles_interp[i, :]  # relative joint angles φ1, ..., φ5 (already interpolated)
        
        # Convert minimal configuration to maximal configuration using RExEel's function
        maximal_config = rex_eel_maximal_from_minimal(rexeel, minimal_config, n_links)
        push!(hw_maximal_config_traj, maximal_config)
    end
    
    hw_maximal_config_traj_by_video[video_name] = hw_maximal_config_traj
    println("    ✓ Reconstructed $(length(hw_maximal_config_traj)) configs (interpolated)")
end

# Select current video for single-video analysis
hw_maximal_config_traj = hw_maximal_config_traj_by_video[current_video_name]

println("  ✓ Configuration dimension: $(length(hw_maximal_config_traj[1])) (6 links × 3 DOFs)")

# Reconstruct genesis maximal configurations
println("  Reconstructing genesis maximal configurations...")

genesis_maximal_config_traj = []

for i in 1:length(genesis_traj_interp)
    # Get head link pose from interpolated genesis tracking (already at simulation timesteps)
    x_head = -genesis_traj_interp[i][1] + 122
    y_head = genesis_traj_interp[i][2]
    θ_head = -genesis_traj_interp[i][3]
    
    # For genesis simulation, use simulation motor angles (since it's a simulation)
    # Construct minimal configuration: [x0, y0, θ0, φ1, φ2, ..., φ_{n-1}]
    n_minimal_coords = n_links + 2
    minimal_config = zeros(n_minimal_coords)
    minimal_config[1] = x_head
    minimal_config[2] = y_head
    minimal_config[3] = θ_head
    minimal_config[4:end] = sim_motor_angles_traj[i]  # Use simulation motor angles for genesis
    
    # Convert minimal configuration to maximal configuration
    maximal_config = rex_eel_maximal_from_minimal(rexeel, minimal_config, n_links)
    push!(genesis_maximal_config_traj, maximal_config)
end

println("  ✓ Genesis configuration dimension: $(length(genesis_maximal_config_traj[1])) (6 links × 3 DOFs)")
println("  ✓ Reconstructed $(length(genesis_maximal_config_traj)) configs")
println()

#############################################################################################
## Compute Center of Mass Trajectories
#############################################################################################

println("Computing center of mass trajectories...")

# Extract masses for each link
total_mass = sum(masses_per_link)

# Compute simulation COM trajectory
sim_com_traj = []
for config in sim_maximal_config_traj
    x_com = 0.0
    y_com = 0.0
    
    for link in 1:n_links
        x_i = config[3*(link-1) + 1]
        y_i = config[3*(link-1) + 2]
        x_com += masses_per_link[link] * x_i
        y_com += masses_per_link[link] * y_i
    end
    
    x_com /= total_mass
    y_com /= total_mass
    
    push!(sim_com_traj, [x_com, y_com])
end

# Compute hardware COM trajectories for ALL videos (configs already interpolated)
hw_com_traj_by_video = Dict{String, Vector{Vector{Float64}}}()

for (video_name, maximal_configs) in hw_maximal_config_traj_by_video
    hw_com_traj = []
    
    for config in maximal_configs
        x_com = 0.0
        y_com = 0.0
        
        for link in 1:n_links
            x_i = config[3*(link-1) + 1]
            y_i = config[3*(link-1) + 2]
            x_com += masses_per_link[link] * x_i
            y_com += masses_per_link[link] * y_i
        end
        
        x_com /= total_mass
        y_com /= total_mass
        
        push!(hw_com_traj, [x_com, y_com])
    end
    
    hw_com_traj_by_video[video_name] = hw_com_traj
end

# Select current video for single-video analysis
hw_com_traj = hw_com_traj_by_video[current_video_name]

println("  ✓ Simulation COM trajectory: $(length(sim_com_traj)) positions")
println("  ✓ Hardware COM trajectories computed for $(length(hw_com_traj_by_video)) videos (interpolated)")
for (video_name, traj) in hw_com_traj_by_video
    println("    - $video_name: $(length(traj)) positions")
end

# Compute genesis COM trajectory
genesis_com_traj = []

for config in genesis_maximal_config_traj
    x_com = 0.0
    y_com = 0.0
    
    for link in 1:n_links
        x_i = config[3*(link-1) + 1]
        y_i = config[3*(link-1) + 2]
        x_com += masses_per_link[link] * x_i
        y_com += masses_per_link[link] * y_i
    end
    
    x_com /= total_mass
    y_com /= total_mass
    
    push!(genesis_com_traj, [x_com, y_com])
end

println("  ✓ Genesis COM trajectory computed: $(length(genesis_com_traj)) positions")
println()

# Extract X and Y components for plotting
sim_com_x = [sim_com_traj[i][1] for i in 1:length(sim_com_traj)]
sim_com_y = [sim_com_traj[i][2] for i in 1:length(sim_com_traj)]
hw_com_x = [hw_com_traj[i][1] for i in 1:length(hw_com_traj)]
hw_com_y = [hw_com_traj[i][2] for i in 1:length(hw_com_traj)]

# For tank visualizations (same as hw_com_x/y)
hw_com_x_tank = hw_com_x
hw_com_y_tank = hw_com_y
sim_com_x_tank = sim_com_x
sim_com_y_tank = sim_com_y

#############################################################################################
## Compute Position Differences
#############################################################################################

println("Computing position differences...")

# Both trajectories now have the same length (hardware interpolated to simulation time)
N = length(sim_com_traj)

println("  Using all timesteps: $N")

# Unbias COM trajectories w.r.t. initial conditions
# Shift both trajectories so they start at the origin
hw_com_initial = hw_com_traj[1]
sim_com_initial = sim_com_traj[1]

hw_com_unbiased = [[hw_com_traj[i][1] - hw_com_initial[1], hw_com_traj[i][2] - hw_com_initial[2]] for i in 1:N]
sim_com_unbiased = [[sim_com_traj[i][1] - sim_com_initial[1], sim_com_traj[i][2] - sim_com_initial[2]] for i in 1:N]

# Compute errors for unbiased COM trajectories
com_x_errors = [hw_com_unbiased[i][1] - sim_com_unbiased[i][1] for i in 1:N]
com_y_errors = [hw_com_unbiased[i][2] - sim_com_unbiased[i][2] for i in 1:N]
com_position_errors = [sqrt(com_x_errors[i]^2 + com_y_errors[i]^2) for i in 1:N]

# Compute RMSE for COM trajectories
com_rmse_x = sqrt(mean(com_x_errors.^2))
com_rmse_y = sqrt(mean(com_y_errors.^2))
com_rmse_position = sqrt(mean(com_position_errors.^2))

# Compute head position errors (using head link configurations)
# Extract head positions from maximal configurations (already computed earlier)
sim_head_x_vals = [sim_maximal_config_traj[i][1] for i in 1:length(sim_maximal_config_traj)]
sim_head_y_vals = [sim_maximal_config_traj[i][2] for i in 1:length(sim_maximal_config_traj)]
hw_head_x_vals = [hw_maximal_config_traj[i][1] for i in 1:length(hw_maximal_config_traj)]
hw_head_y_vals = [hw_maximal_config_traj[i][2] for i in 1:length(hw_maximal_config_traj)]

# Unbias head trajectories w.r.t. initial conditions
hw_head_initial_x = hw_head_x_vals[1]
hw_head_initial_y = hw_head_y_vals[1]
sim_head_initial_x = sim_head_x_vals[1]
sim_head_initial_y = sim_head_y_vals[1]

hw_head_x_unbiased = [hw_head_x_vals[i] - hw_head_initial_x for i in 1:N]
hw_head_y_unbiased = [hw_head_y_vals[i] - hw_head_initial_y for i in 1:N]
sim_head_x_unbiased = [sim_head_x_vals[i] - sim_head_initial_x for i in 1:N]
sim_head_y_unbiased = [sim_head_y_vals[i] - sim_head_initial_y for i in 1:N]

# Compute head position errors
head_x_errors = [hw_head_x_unbiased[i] - sim_head_x_unbiased[i] for i in 1:N]
head_y_errors = [hw_head_y_unbiased[i] - sim_head_y_unbiased[i] for i in 1:N]
head_position_errors = [sqrt(head_x_errors[i]^2 + head_y_errors[i]^2) for i in 1:N]

# Compute RMSE for head trajectories
head_rmse_x = sqrt(mean(head_x_errors.^2))
head_rmse_y = sqrt(mean(head_y_errors.^2))
head_rmse_position = sqrt(mean(head_position_errors.^2))

# Summary statistics
mean_head_pos_error = mean(head_position_errors)
max_head_pos_error = maximum(head_position_errors)
mean_com_pos_error = mean(com_position_errors)
max_com_pos_error = maximum(com_position_errors)

println("\n  Hardware Center of Mass (Unbiased) RMSE:")
println("    X RMSE: $(round(com_rmse_x, digits=3)) cm")
println("    Y RMSE: $(round(com_rmse_y, digits=3)) cm")
println("    Position RMSE: $(round(com_rmse_position, digits=3)) cm")
println("    Mean position error: $(round(mean_com_pos_error, digits=3)) cm")
println("    Max position error: $(round(max_com_pos_error, digits=3)) cm")
println("\n  Hardware Head Link Position (Unbiased) RMSE:")
println("    X RMSE: $(round(head_rmse_x, digits=3)) cm")
println("    Y RMSE: $(round(head_rmse_y, digits=3)) cm")
println("    Position RMSE: $(round(head_rmse_position, digits=3)) cm")
println("    Mean position error: $(round(mean_head_pos_error, digits=3)) cm")
println("    Max position error: $(round(max_head_pos_error, digits=3)) cm")

# Compute genesis errors
genesis_com_initial = genesis_com_traj[1]
genesis_com_unbiased = [[genesis_com_traj[i][1] - genesis_com_initial[1], genesis_com_traj[i][2] - genesis_com_initial[2]] for i in 1:N]

genesis_com_x_errors = [genesis_com_unbiased[i][1] - sim_com_unbiased[i][1] for i in 1:N]
genesis_com_y_errors = [genesis_com_unbiased[i][2] - sim_com_unbiased[i][2] for i in 1:N]
genesis_com_position_errors = [sqrt(genesis_com_x_errors[i]^2 + genesis_com_y_errors[i]^2) for i in 1:N]

genesis_com_rmse_x = sqrt(mean(genesis_com_x_errors.^2))
genesis_com_rmse_y = sqrt(mean(genesis_com_y_errors.^2))
genesis_com_rmse_position = sqrt(mean(genesis_com_position_errors.^2))
genesis_mean_com_pos_error = mean(genesis_com_position_errors)
genesis_max_com_pos_error = maximum(genesis_com_position_errors)

# Genesis head errors
genesis_head_x_vals = [genesis_maximal_config_traj[i][1] for i in 1:length(genesis_maximal_config_traj)]
genesis_head_y_vals = [genesis_maximal_config_traj[i][2] for i in 1:length(genesis_maximal_config_traj)]

genesis_head_initial_x = genesis_head_x_vals[1]
genesis_head_initial_y = genesis_head_y_vals[1]

genesis_head_x_unbiased = [genesis_head_x_vals[i] - genesis_head_initial_x for i in 1:N]
genesis_head_y_unbiased = [genesis_head_y_vals[i] - genesis_head_initial_y for i in 1:N]

genesis_head_x_errors = [genesis_head_x_unbiased[i] - sim_head_x_unbiased[i] for i in 1:N]
genesis_head_y_errors = [genesis_head_y_unbiased[i] - sim_head_y_unbiased[i] for i in 1:N]
genesis_head_position_errors = [sqrt(genesis_head_x_errors[i]^2 + genesis_head_y_errors[i]^2) for i in 1:N]

genesis_head_rmse_x = sqrt(mean(genesis_head_x_errors.^2))
genesis_head_rmse_y = sqrt(mean(genesis_head_y_errors.^2))
genesis_head_rmse_position = sqrt(mean(genesis_head_position_errors.^2))
genesis_mean_head_pos_error = mean(genesis_head_position_errors)
genesis_max_head_pos_error = maximum(genesis_head_position_errors)

println("\n  Genesis Center of Mass (Unbiased) RMSE:")
println("    X RMSE: $(round(genesis_com_rmse_x, digits=3)) cm")
println("    Y RMSE: $(round(genesis_com_rmse_y, digits=3)) cm")
println("    Position RMSE: $(round(genesis_com_rmse_position, digits=3)) cm")
println("    Mean position error: $(round(genesis_mean_com_pos_error, digits=3)) cm")
println("    Max position error: $(round(genesis_max_com_pos_error, digits=3)) cm")
println("\n  Genesis Head Link Position (Unbiased) RMSE:")
println("    X RMSE: $(round(genesis_head_rmse_x, digits=3)) cm")
println("    Y RMSE: $(round(genesis_head_rmse_y, digits=3)) cm")
println("    Position RMSE: $(round(genesis_head_rmse_position, digits=3)) cm")
println("    Mean position error: $(round(genesis_mean_head_pos_error, digits=3)) cm")
println("    Max position error: $(round(genesis_max_head_pos_error, digits=3)) cm")
println()

#############################################################################################
## Identify Outlier Video by Computing Errors for All Videos
#############################################################################################

println("Analyzing errors for all videos to identify outliers...")

video_error_stats = Dict{String, Dict{String, Float64}}()

for (video_name, hw_com_traj_video) in hw_com_traj_by_video
    # Get corresponding hardware maximal config trajectory
    hw_maximal_config_video = hw_maximal_config_traj_by_video[video_name]
    
    # Unbias COM trajectories
    hw_com_initial_video = hw_com_traj_video[1]
    hw_com_unbiased_video = [[hw_com_traj_video[i][1] - hw_com_initial_video[1], 
                               hw_com_traj_video[i][2] - hw_com_initial_video[2]] for i in 1:N]
    
    # COM errors
    com_x_errors_video = [hw_com_unbiased_video[i][1] - sim_com_unbiased[i][1] for i in 1:N]
    com_y_errors_video = [hw_com_unbiased_video[i][2] - sim_com_unbiased[i][2] for i in 1:N]
    com_position_errors_video = [sqrt(com_x_errors_video[i]^2 + com_y_errors_video[i]^2) for i in 1:N]
    
    # Head errors
    hw_head_x_vals_video = [hw_maximal_config_video[i][1] for i in 1:N]
    hw_head_y_vals_video = [hw_maximal_config_video[i][2] for i in 1:N]
    
    hw_head_initial_x_video = hw_head_x_vals_video[1]
    hw_head_initial_y_video = hw_head_y_vals_video[1]
    
    hw_head_x_unbiased_video = [hw_head_x_vals_video[i] - hw_head_initial_x_video for i in 1:N]
    hw_head_y_unbiased_video = [hw_head_y_vals_video[i] - hw_head_initial_y_video for i in 1:N]
    
    head_x_errors_video = [hw_head_x_unbiased_video[i] - sim_head_x_unbiased[i] for i in 1:N]
    head_y_errors_video = [hw_head_y_unbiased_video[i] - sim_head_y_unbiased[i] for i in 1:N]
    head_position_errors_video = [sqrt(head_x_errors_video[i]^2 + head_y_errors_video[i]^2) for i in 1:N]
    
    # Store statistics
    video_error_stats[video_name] = Dict(
        "com_rmse" => sqrt(mean(com_position_errors_video.^2)),
        "com_mean" => mean(com_position_errors_video),
        "com_max" => maximum(com_position_errors_video),
        "head_rmse" => sqrt(mean(head_position_errors_video.^2)),
        "head_mean" => mean(head_position_errors_video),
        "head_max" => maximum(head_position_errors_video)
    )
end

# Print error statistics for each video
println("\nError statistics by video:")
println("-"^100)
println(@sprintf("%-40s │ COM RMSE │ COM Mean │ COM Max │ Head RMSE │ Head Mean │ Head Max", "Video Name"))
println("-"^100)

for (video_name, stats) in sort(collect(video_error_stats), by=x->x[2]["com_rmse"], rev=true)
    println(@sprintf("%-40s │  %6.3f  │  %6.3f  │ %6.3f │   %6.3f  │   %6.3f  │  %6.3f",
        video_name,
        stats["com_rmse"],
        stats["com_mean"],
        stats["com_max"],
        stats["head_rmse"],
        stats["head_mean"],
        stats["head_max"]))
end
println("-"^100)

# Identify outlier (video with highest COM RMSE)
outlier_video = argmax(video_name -> video_error_stats[video_name]["com_rmse"], keys(video_error_stats))
outlier_com_rmse = video_error_stats[outlier_video]["com_rmse"]
outlier_head_rmse = video_error_stats[outlier_video]["head_rmse"]

println("\n⚠ OUTLIER IDENTIFIED:")
println("  Video: $outlier_video")
println("  COM RMSE: $(round(outlier_com_rmse, digits=3)) cm")
println("  Head RMSE: $(round(outlier_head_rmse, digits=3)) cm")
println()

#############################################################################################
## COM Trajectory Comparison Plots
#############################################################################################

println("Creating comparison plots...")

# Figure 2: COM X position vs time
fig2, ax2 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    fontsize=18,
    resolution=(1000, 600),
    xlabel="Time (s)",
    ylabel="X Displacement (cm)",
    use_data_aspect=false
)

# Plot all hardware video trajectories as opaque lines
for (video_name, com_traj) in hw_com_traj_by_video
    com_x = [com_traj[i][1] for i in 1:length(com_traj)]
    lines!(ax2, time_traj_sim, com_x, color=hardware_color_opaque, linewidth=2)
end

# Plot genesis simulation trajectory
genesis_com_x = [genesis_com_traj[i][1] for i in 1:length(genesis_com_traj)]
lines!(ax2, time_traj_sim, genesis_com_x, color=genesis_color_opaque, linewidth=2)

# Overlay current hardware and simulation trajectories
lines!(ax2, time_traj_sim, sim_com_x, color=simulation_color, linewidth=3, label="Simulation COM")
axislegend(ax2, position=:lt)

display(fig2)
println("  ✓ Displayed: COM X position vs time (all videos)")

# Create tikz plot for COM X comparison
lineopts = @pgf {no_marks, "very thick"}

# Build plot with all hardware trajectories
com_x_plots = []
for (video_name, com_traj) in hw_com_traj_by_video
    com_x = [com_traj[i][1] for i in 1:length(com_traj)]
    push!(com_x_plots, PlotInc(@pgf({no_marks, "thick", color=hardware_color_opaque}),
        Coordinates(time_traj_sim, com_x)))
end

# Add genesis simulation trajectory
genesis_com_x = [genesis_com_traj[i][1] for i in 1:length(genesis_com_traj)]
push!(com_x_plots, PlotInc(@pgf({no_marks, "thick", color=genesis_color_opaque}),
    Coordinates(time_traj_sim, genesis_com_x)))

# Add simulation trajectory
push!(com_x_plots, PlotInc(@pgf({no_marks, "very thick", color=simulation_color}),
    Coordinates(time_traj_sim, sim_com_x)))

com_x_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "X Displacement (cm)",
        legend_pos = "north east",
        legend_cell_align="left",
    },
    com_x_plots...,
    PGFPlotsX.Legend(["Hardware" for _ in 1:length(hw_com_traj_by_video)]..., "Genesis", "Simulation COM")
)

tikz_filename = joinpath(output_dir, "com_x_comparison.tikz")
pgfsave(tikz_filename, com_x_plot, include_preamble=false)
println("  ✓ Saved: com_x_comparison.tikz")

# Figure 3: COM Y position vs time
fig3, ax3 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    fontsize=18,
    resolution=(1000, 600),
    xlabel="Time (s)",
    ylabel="Y Displacement (cm)",
    use_data_aspect=false
)

# Plot all hardware video trajectories as opaque lines
for (video_name, com_traj) in hw_com_traj_by_video
    com_y = [com_traj[i][2] for i in 1:length(com_traj)]
    lines!(ax3, time_traj_sim, com_y, color=hardware_color_opaque, linewidth=2)
end

# Plot genesis simulation trajectory
genesis_com_y = [genesis_com_traj[i][2] for i in 1:length(genesis_com_traj)]
lines!(ax3, time_traj_sim, genesis_com_y, color=genesis_color_opaque, linewidth=2)

# Overlay current hardware and simulation trajectories
lines!(ax3, time_traj_sim, sim_com_y, color=simulation_color, linewidth=3, label="Simulation COM")
axislegend(ax3, position=:lt)

display(fig3)
println("  ✓ Displayed: COM Y position vs time (all videos)")

# Create tikz plot for COM Y comparison

# Build plot with all hardware trajectories
com_y_plots = []
for (video_name, com_traj) in hw_com_traj_by_video
    com_y = [com_traj[i][2] for i in 1:length(com_traj)]
    push!(com_y_plots, PlotInc(@pgf({no_marks, "thick", color=hardware_color_opaque}),
        Coordinates(time_traj_sim, com_y)))
end

# Add genesis simulation trajectory
genesis_com_y = [genesis_com_traj[i][2] for i in 1:length(genesis_com_traj)]
push!(com_y_plots, PlotInc(@pgf({no_marks, "thick", color=genesis_color_opaque}),
    Coordinates(time_traj_sim, genesis_com_y)))

# Add simulation trajectory
push!(com_y_plots, PlotInc(@pgf({no_marks, "very thick", color=simulation_color}),
    Coordinates(time_traj_sim, sim_com_y)))

com_y_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Y Displacement (cm)",
        legend_pos = "north east",
        legend_cell_align="left",
    },
    com_y_plots...,
    PGFPlotsX.Legend(["Hardware" for _ in 1:length(hw_com_traj_by_video)]..., "Genesis", "Simulation COM")
)

tikz_filename = joinpath(output_dir, "com_y_comparison.tikz")
pgfsave(tikz_filename, com_y_plot, include_preamble=false)
println("  ✓ Saved: com_y_comparison.tikz")

#############################################################################################
## Head Configuration Comparison Plots
#############################################################################################

println()
println("Creating head configuration comparison plots...")

# Extract head link configurations (link 1: indices 1, 2, 3 for x, y, theta)
# Hardware configs already interpolated to simulation time
sim_head_x = [sim_maximal_config_traj[i][1] for i in 1:length(sim_maximal_config_traj)]
sim_head_y = [sim_maximal_config_traj[i][2] for i in 1:length(sim_maximal_config_traj)]
sim_head_theta = [sim_maximal_config_traj[i][3] for i in 1:length(sim_maximal_config_traj)]

hw_head_x = [hw_maximal_config_traj[i][1] for i in 1:length(hw_maximal_config_traj)]
hw_head_y = [hw_maximal_config_traj[i][2] for i in 1:length(hw_maximal_config_traj)]
hw_head_theta = [hw_maximal_config_traj[i][3] for i in 1:length(hw_maximal_config_traj)]

# Plot 1: Head X position
fig_head_x, ax_head_x = create_aquarium_figure(;
    backgroundcolor=:transparent,
    fontsize=18,
    resolution=(1000, 600),
    xlabel="Time (s)",
    ylabel="Head X Position (cm)",
    use_data_aspect=false
)

# Plot all hardware video trajectories as opaque lines
for (video_name, maximal_configs) in hw_maximal_config_traj_by_video
    head_x = [maximal_configs[i][1] for i in 1:length(maximal_configs)]
    lines!(ax_head_x, time_traj_sim, head_x, color=hardware_color_opaque, linewidth=2)
end

# Plot genesis simulation trajectory
genesis_head_x = [genesis_maximal_config_traj[i][1] for i in 1:length(genesis_maximal_config_traj)]
lines!(ax_head_x, time_traj_sim, genesis_head_x, color=genesis_color_opaque, linewidth=2)

# Overlay simulation trajectory
lines!(ax_head_x, time_traj_sim, sim_head_x, 
       color=simulation_color, linewidth=3, label="Simulation")
axislegend(ax_head_x, position=:lt)
display(fig_head_x)
println("  ✓ Displayed: Head X position vs time (all videos)")

# Create tikz plot for head X comparison

# Build plot with all hardware trajectories
head_x_plots = []
for (video_name, maximal_configs) in hw_maximal_config_traj_by_video
    head_x = [maximal_configs[i][1] for i in 1:length(maximal_configs)]
    push!(head_x_plots, PlotInc(@pgf({no_marks, "thick", color=hardware_color_opaque}),
        Coordinates(time_traj_sim, head_x)))
end

# Add genesis simulation trajectory
genesis_head_x = [genesis_maximal_config_traj[i][1] for i in 1:length(genesis_maximal_config_traj)]
push!(head_x_plots, PlotInc(@pgf({no_marks, "thick", color=genesis_color_opaque}),
    Coordinates(time_traj_sim, genesis_head_x)))

# Add simulation trajectory
push!(head_x_plots, PlotInc(@pgf({no_marks, "very thick", color=simulation_color}),
    Coordinates(time_traj_sim, sim_head_x)))

head_x_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Head X Position (cm)",
        legend_pos = "north east",
        legend_cell_align="left",
    },
    head_x_plots...,
    PGFPlotsX.Legend(["Hardware" for _ in 1:length(hw_maximal_config_traj_by_video)]..., "Genesis", "Simulation")
)
tikz_filename = joinpath(output_dir, "head_x_comparison.tikz")
pgfsave(tikz_filename, head_x_plot, include_preamble=false)
println("  ✓ Saved: head_x_comparison.tikz")

# Create tikz plot for head Y comparison

# Build plot with all hardware trajectories
head_y_plots = []
for (video_name, maximal_configs) in hw_maximal_config_traj_by_video
    head_y = [maximal_configs[i][2] for i in 1:length(maximal_configs)]
    push!(head_y_plots, PlotInc(@pgf({no_marks, "thick", color=hardware_color_opaque}),
        Coordinates(time_traj_sim, head_y)))
end

# Add genesis simulation trajectory
genesis_head_y = [genesis_maximal_config_traj[i][2] for i in 1:length(genesis_maximal_config_traj)]
push!(head_y_plots, PlotInc(@pgf({no_marks, "thick", color=genesis_color_opaque}),
    Coordinates(time_traj_sim, genesis_head_y)))

# Add simulation trajectory
push!(head_y_plots, PlotInc(@pgf({no_marks, "very thick", color=simulation_color}),
    Coordinates(time_traj_sim, sim_head_y)))

head_y_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Head Y Position (cm)",
        legend_pos = "north east",
        legend_cell_align="left",
    },
    head_y_plots...,
    PGFPlotsX.Legend(["Hardware" for _ in 1:length(hw_maximal_config_traj_by_video)]..., "Genesis", "Simulation")
)

tikz_filename = joinpath(output_dir, "head_y_comparison.tikz")
pgfsave(tikz_filename, head_y_plot, include_preamble=false)
println("  ✓ Saved: head_y_comparison.tikz")
tikz_filename = joinpath(output_dir, "head_x_comparison.tikz")
pgfsave(tikz_filename, head_x_plot, include_preamble=false)
println("  ✓ Saved: head_x_comparison.tikz")

# Plot 2: Head Y position
fig_head_y, ax_head_y = create_aquarium_figure(;
    backgroundcolor=:transparent,
    fontsize=18,
    resolution=(1000, 600),
    xlabel="Time (s)",
    ylabel="Head Y Position (cm)",
    use_data_aspect=false
)

# Plot all hardware video trajectories as opaque lines
for (video_name, maximal_configs) in hw_maximal_config_traj_by_video
    head_y = [maximal_configs[i][2] for i in 1:length(maximal_configs)]
    lines!(ax_head_y, time_traj_sim, head_y, color=hardware_color_opaque, linewidth=2)
end

# Plot genesis simulation trajectory
genesis_head_y = [genesis_maximal_config_traj[i][2] for i in 1:length(genesis_maximal_config_traj)]
lines!(ax_head_y, time_traj_sim, genesis_head_y, color=genesis_color_opaque, linewidth=2)

# Overlay simulation trajectory
lines!(ax_head_y, time_traj_sim, sim_head_y, 
       color=simulation_color, linewidth=3, label="Simulation")
axislegend(ax_head_y, position=:lt)

# Create tikz plot for head angle comparison

# Build plot with all hardware trajectories
head_theta_plots = []
for (video_name, maximal_configs) in hw_maximal_config_traj_by_video
    head_theta = [maximal_configs[i][3] for i in 1:length(maximal_configs)]
    push!(head_theta_plots, PlotInc(@pgf({no_marks, "thick", color=hardware_color_opaque}),
        Coordinates(time_traj_sim, head_theta)))
end

# Add genesis simulation trajectory
genesis_head_theta = [genesis_maximal_config_traj[i][3] for i in 1:length(genesis_maximal_config_traj)]
push!(head_theta_plots, PlotInc(@pgf({no_marks, "thick", color=genesis_color_opaque}),
    Coordinates(time_traj_sim, genesis_head_theta)))

# Add simulation trajectory
push!(head_theta_plots, PlotInc(@pgf({no_marks, "very thick", color=simulation_color}),
    Coordinates(time_traj_sim, sim_head_theta)))

head_theta_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "Time (s)",
        ylabel = "Head Angle (rad)",
        legend_pos = "north east",
        legend_cell_align="left",
    },
    head_theta_plots...,
    PGFPlotsX.Legend(["Hardware" for _ in 1:length(hw_maximal_config_traj_by_video)]..., "Genesis", "Simulation")
)

tikz_filename = joinpath(output_dir, "head_theta_comparison.tikz")
pgfsave(tikz_filename, head_theta_plot, include_preamble=false)
println("  ✓ Saved: head_theta_comparison.tikz")
display(fig_head_y)

# Plot 3: Head angle (theta)
fig_head_theta, ax_head_theta = create_aquarium_figure(;
    backgroundcolor=:transparent,
    fontsize=18,
    resolution=(1000, 600),
    xlabel="Time (s)",
    ylabel="Head Angle (rad)",
    use_data_aspect=false
)

# Plot all hardware video trajectories as opaque lines
for (video_name, maximal_configs) in hw_maximal_config_traj_by_video
    head_theta = [maximal_configs[i][3] for i in 1:length(maximal_configs)]
    lines!(ax_head_theta, time_traj_sim, head_theta, color=hardware_color_opaque, linewidth=2)
end

# Plot genesis simulation trajectory
genesis_head_theta = [genesis_maximal_config_traj[i][3] for i in 1:length(genesis_maximal_config_traj)]
lines!(ax_head_theta, time_traj_sim, genesis_head_theta, color=genesis_color_opaque, linewidth=2)

# Overlay simulation trajectory
lines!(ax_head_theta, time_traj_sim, sim_head_theta, 
       color=simulation_color, linewidth=3, label="Simulation")
axislegend(ax_head_theta, position=:lt)
display(fig_head_theta)
println("  ✓ Displayed: Head angle (theta) vs time (all videos)")

#############################################################################################
## Motor Angle Comparison Plots
#############################################################################################

println()
println("Creating motor angle comparison plots...")

# Create individual plots for each motor
for motor_idx in 1:5
    fig_motor, ax_motor = create_aquarium_figure(;
        backgroundcolor=:transparent,
        fontsize=18,
        resolution=(1000, 600),
        xlabel="Time (s)",
        ylabel="Motor Angle (rad)",
        use_data_aspect=false
    )
    
    # Plot hardware motor angle
    lines!(ax_motor, time_traj_sim, actual_motor_angles_interp[:, motor_idx], 
           color=hardware_color, linewidth=3, label="Hardware Motor $motor_idx")
    
    # Plot simulation motor angle (extracted from maximal configuration)
    sim_motor_angles = [sim_motor_angles_traj[i][motor_idx] for i in 1:length(sim_motor_angles_traj)]
    lines!(ax_motor, time_traj_sim, sim_motor_angles, 
           color=simulation_color, linewidth=3, label="Simulation Motor $motor_idx")
    
    axislegend(ax_motor, position=:lt)
    
    display(fig_motor)
    save(joinpath(output_dir, "motor_angle_$(motor_idx)_comparison.png"), fig_motor)
    println("  ✓ Saved: motor_angle_$(motor_idx)_comparison.png")
    
    # Create tikz plot for motor angle comparison
    lineopts = @pgf {no_marks, "very thick"}
    motor_plot = @pgf PGFPlotsX.Axis(
        {
            xmajorgrids,
            ymajorgrids,
            xlabel = "Time (s)",
            ylabel = "Motor Angle (rad)",
            legend_pos = "north east",
            legend_cell_align="left",
        },
        PlotInc({lineopts..., color=hardware_color},
            Coordinates(time_traj_motors, actual_motor_angles[:, motor_idx])),
        PlotInc({lineopts..., color=simulation_color},
            Coordinates(time_traj_sim, sim_motor_angles)),
        PGFPlotsX.Legend(["Hardware Motor $motor_idx", "Simulation Motor $motor_idx"])
    )
    
    tikz_filename = joinpath(output_dir, "motor_angle_$(motor_idx)_comparison.tikz")
    pgfsave(tikz_filename, motor_plot, include_preamble=false)
    println("  ✓ Saved: motor_angle_$(motor_idx)_comparison.tikz")
end

#############################################################################################
## Visualize COM Trajectories in Physical Tank
#############################################################################################

# Version 1: Transparent background, solid line
fig5, ax5 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

# Plot hardware COM trajectory (not unbiased, actual positions in tank)
hw_com_x_tank = [hw_com_traj[i][1] for i in 1:length(hw_com_traj)]
hw_com_y_tank = [hw_com_traj[i][2] for i in 1:length(hw_com_traj)]

lines!(ax5, hw_com_x_tank, hw_com_y_tank, color=hardware_color, linewidth=10)

# Mark start and end points
scatter!(ax5, [hw_com_x_tank[1]], [hw_com_y_tank[1]], color=hardware_color, markersize=20, marker=:circle)
scatter!(ax5, [hw_com_x_tank[end]], [hw_com_y_tank[end]], color=hardware_color, markersize=20, marker=:square)

display(fig5)

# Version 2: Opaque background, dashed line
fig6, ax6 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

lines!(ax6, hw_com_x_tank, hw_com_y_tank, color=hardware_color_opaque, linewidth=10, linestyle=:dash)
display(fig6)

# Simulation COM Trajectory Visualizations
println()
println("Creating simulation COM trajectory visualizations...")

# Version 1: Simulation COM - Transparent background, solid line
fig7, ax7 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

# Plot simulation COM trajectory (actual positions in tank)
lines!(ax7, sim_com_x_tank, sim_com_y_tank, color=simulation_color, linewidth=10)

# Mark start and end points
scatter!(ax7, [sim_com_x_tank[1]], [sim_com_y_tank[1]], color=simulation_color, markersize=20, marker=:circle)
scatter!(ax7, [sim_com_x_tank[end]], [sim_com_y_tank[end]], color=simulation_color, markersize=20, marker=:square)

display(fig7)

# Version 2: Simulation COM - Opaque background, dashed line
fig8, ax8 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

lines!(ax8, sim_com_x_tank, sim_com_y_tank, color=simulation_color_opaque, linewidth=10, linestyle=:dash)
display(fig8)

# Genesis COM Trajectory Visualizations
println()
println("Creating genesis COM trajectory visualizations...")

# Version 1: Genesis COM - Transparent background, solid line
fig_genesis_com, ax_genesis_com = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

# Plot genesis COM trajectory (actual positions in tank)
genesis_com_x_tank = [genesis_com_traj[i][1] for i in 1:length(genesis_com_traj)]
genesis_com_y_tank = [genesis_com_traj[i][2] for i in 1:length(genesis_com_traj)]

lines!(ax_genesis_com, genesis_com_x_tank, genesis_com_y_tank, color=genesis_color, linewidth=10)

# Mark start and end points
scatter!(ax_genesis_com, [genesis_com_x_tank[1]], [genesis_com_y_tank[1]], color=genesis_color, markersize=20, marker=:circle)
scatter!(ax_genesis_com, [genesis_com_x_tank[end]], [genesis_com_y_tank[end]], color=genesis_color, markersize=20, marker=:square)

display(fig_genesis_com)

# Version 2: Genesis COM - Opaque background, dashed line
fig_genesis_com_dash, ax_genesis_com_dash = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

lines!(ax_genesis_com_dash, genesis_com_x_tank, genesis_com_y_tank, color=genesis_color_opaque, linewidth=10, linestyle=:dash)
display(fig_genesis_com_dash)

#############################################################################################
## Visualize Head Trajectories in Physical Tank
#############################################################################################

println()
println("Creating head link trajectory visualizations...")

# Hardware Head Trajectory - Version 1: Transparent background, solid line
fig9, ax9 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

# Plot hardware head trajectory (actual positions in tank)
lines!(ax9, hw_head_x, hw_head_y, color=hardware_color, linewidth=10)

# Mark start and end points
scatter!(ax9, [hw_head_x[1]], [hw_head_y[1]], color=hardware_color, markersize=20, marker=:circle)
scatter!(ax9, [hw_head_x[end]], [hw_head_y[end]], color=hardware_color, markersize=20, marker=:square)

display(fig9)

# Hardware Head Trajectory - Version 2: Opaque background, dashed line
fig10, ax10 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

lines!(ax10, hw_head_x, hw_head_y, color=hardware_color_opaque, linewidth=10, linestyle=:dash)
display(fig10)

# Simulation Head Trajectory - Version 1: Transparent background, solid line
fig11, ax11 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

# Plot simulation head trajectory (actual positions in tank)
lines!(ax11, sim_head_x, sim_head_y, color=simulation_color, linewidth=10)

# Mark start and end points
scatter!(ax11, [sim_head_x[1]], [sim_head_y[1]], color=simulation_color, markersize=20, marker=:circle)
scatter!(ax11, [sim_head_x[end]], [sim_head_y[end]], color=simulation_color, markersize=20, marker=:square)

display(fig11)

# Simulation Head Trajectory - Version 2: Opaque background, dashed line
fig12, ax12 = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

lines!(ax12, sim_head_x, sim_head_y, color=simulation_color_opaque, linewidth=10, linestyle=:dash)
display(fig12)

# Genesis Head Trajectory Visualizations
println()
println("Creating genesis head trajectory visualizations...")

# Version 1: Genesis Head - Transparent background, solid line
fig_genesis_head, ax_genesis_head = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

# Plot genesis head trajectory (actual positions in tank)
genesis_head_x_tank = [genesis_maximal_config_traj[i][1] for i in 1:length(genesis_maximal_config_traj)]
genesis_head_y_tank = [genesis_maximal_config_traj[i][2] for i in 1:length(genesis_maximal_config_traj)]

lines!(ax_genesis_head, genesis_head_x_tank, genesis_head_y_tank, color=genesis_color, linewidth=10)

# Mark start and end points
scatter!(ax_genesis_head, [genesis_head_x_tank[1]], [genesis_head_y_tank[1]], color=genesis_color, markersize=20, marker=:circle)
scatter!(ax_genesis_head, [genesis_head_x_tank[end]], [genesis_head_y_tank[end]], color=genesis_color, markersize=20, marker=:square)

display(fig_genesis_head)

# Version 2: Genesis Head - Opaque background, dashed line
fig_genesis_head_dash, ax_genesis_head_dash = create_aquarium_figure(;
    backgroundcolor=:transparent,
    resolution=resolution,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    spinevisible=false,
    ticksvisible=false,
    xlim=(0, 122),
    ylim=(0, 122),
    use_data_aspect=true
)

lines!(ax_genesis_head_dash, genesis_head_x_tank, genesis_head_y_tank, color=genesis_color_opaque, linewidth=10, linestyle=:dash)
display(fig_genesis_head_dash)

#############################################################################################
## Visualize Trajectories at Specific Time Points
#############################################################################################

println()
println("Creating time-specific trajectory visualizations...")

time_points = [0.0, 1.0, 2.0, 3.0, 4.0]

for t in time_points
    # Find the frame index closest to this time
    frame_idx = argmin(abs.(time_traj_sim .- t))
    actual_time = time_traj_sim[frame_idx]
    
    println("  Creating visualizations for t=$(t)s (frame $frame_idx, actual t=$(round(actual_time, digits=3))s)")
    
    # Hardware trajectory up to this time
    fig_hw_t, ax_hw_t = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_hw_t, hw_com_x_tank[1:frame_idx], hw_com_y_tank[1:frame_idx], color=hardware_color, linewidth=10)
    scatter!(ax_hw_t, [hw_com_x_tank[1]], [hw_com_y_tank[1]], color=hardware_color, markersize=20, marker=:circle)
    scatter!(ax_hw_t, [hw_com_x_tank[frame_idx]], [hw_com_y_tank[frame_idx]], color=hardware_color, markersize=20, marker=:square)
    
    display(fig_hw_t)
    save(joinpath(output_dir, "hardware_traj_t$(Int(t))s.png"), fig_hw_t)
    
    # Simulation trajectory up to this time
    fig_sim_t, ax_sim_t = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_sim_t, sim_com_x_tank[1:frame_idx], sim_com_y_tank[1:frame_idx], color=simulation_color, linewidth=10)
    scatter!(ax_sim_t, [sim_com_x_tank[1]], [sim_com_y_tank[1]], color=simulation_color, markersize=20, marker=:circle)
    scatter!(ax_sim_t, [sim_com_x_tank[frame_idx]], [sim_com_y_tank[frame_idx]], color=simulation_color, markersize=20, marker=:square)
    
    display(fig_sim_t)
    save(joinpath(output_dir, "simulation_traj_t$(Int(t))s.png"), fig_sim_t)
    
    # Genesis trajectory up to this time
    fig_genesis_t, ax_genesis_t = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_genesis_t, genesis_com_x_tank[1:frame_idx], genesis_com_y_tank[1:frame_idx], color=genesis_color, linewidth=10)
    scatter!(ax_genesis_t, [genesis_com_x_tank[1]], [genesis_com_y_tank[1]], color=genesis_color, markersize=20, marker=:circle)
    scatter!(ax_genesis_t, [genesis_com_x_tank[frame_idx]], [genesis_com_y_tank[frame_idx]], color=genesis_color, markersize=20, marker=:square)
    
    display(fig_genesis_t)
    save(joinpath(output_dir, "genesis_traj_t$(Int(t))s.png"), fig_genesis_t)
    
    # Create opaque/dashed versions
    # Hardware opaque/dashed
    fig_hw_t_dashed, ax_hw_t_dashed = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_hw_t_dashed, hw_com_x_tank[1:frame_idx], hw_com_y_tank[1:frame_idx], 
           color=hardware_color_opaque, linewidth=10, linestyle=:dash)
    scatter!(ax_hw_t_dashed, [hw_com_x_tank[1]], [hw_com_y_tank[1]], 
             color=hardware_color_opaque, markersize=20, marker=:circle)
    scatter!(ax_hw_t_dashed, [hw_com_x_tank[frame_idx]], [hw_com_y_tank[frame_idx]], 
             color=hardware_color_opaque, markersize=20, marker=:square)
    
    save(joinpath(output_dir, "hardware_traj_t$(Int(t))s_opaque_dashed.png"), fig_hw_t_dashed)
    
    # Simulation opaque/dashed
    fig_sim_t_dashed, ax_sim_t_dashed = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_sim_t_dashed, sim_com_x_tank[1:frame_idx], sim_com_y_tank[1:frame_idx], 
           color=simulation_color_opaque, linewidth=10, linestyle=:dash)
    scatter!(ax_sim_t_dashed, [sim_com_x_tank[1]], [sim_com_y_tank[1]], 
             color=simulation_color_opaque, markersize=20, marker=:circle)
    scatter!(ax_sim_t_dashed, [sim_com_x_tank[frame_idx]], [sim_com_y_tank[frame_idx]], 
             color=simulation_color_opaque, markersize=20, marker=:square)
    
    save(joinpath(output_dir, "simulation_traj_t$(Int(t))s_opaque_dashed.png"), fig_sim_t_dashed)
    
    # Genesis opaque/dashed
    fig_genesis_t_dashed, ax_genesis_t_dashed = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_genesis_t_dashed, genesis_com_x_tank[1:frame_idx], genesis_com_y_tank[1:frame_idx], 
           color=genesis_color_opaque, linewidth=10, linestyle=:dash)
    scatter!(ax_genesis_t_dashed, [genesis_com_x_tank[1]], [genesis_com_y_tank[1]], 
             color=genesis_color_opaque, markersize=20, marker=:circle)
    scatter!(ax_genesis_t_dashed, [genesis_com_x_tank[frame_idx]], [genesis_com_y_tank[frame_idx]], 
             color=genesis_color_opaque, markersize=20, marker=:square)
    
    save(joinpath(output_dir, "genesis_traj_t$(Int(t))s_opaque_dashed.png"), fig_genesis_t_dashed)
    
    # Hardware head trajectory up to this time
    fig_hw_head_t, ax_hw_head_t = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_hw_head_t, hw_head_x[1:frame_idx], hw_head_y[1:frame_idx], color=hardware_color, linewidth=10)
    scatter!(ax_hw_head_t, [hw_head_x[1]], [hw_head_y[1]], color=hardware_color, markersize=20, marker=:circle)
    scatter!(ax_hw_head_t, [hw_head_x[frame_idx]], [hw_head_y[frame_idx]], color=hardware_color, markersize=20, marker=:square)
    
    display(fig_hw_head_t)
    save(joinpath(output_dir, "hardware_head_traj_t$(Int(t))s.png"), fig_hw_head_t)
    
    # Simulation head trajectory up to this time
    fig_sim_head_t, ax_sim_head_t = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_sim_head_t, sim_head_x[1:frame_idx], sim_head_y[1:frame_idx], color=simulation_color, linewidth=10)
    scatter!(ax_sim_head_t, [sim_head_x[1]], [sim_head_y[1]], color=simulation_color, markersize=20, marker=:circle)
    scatter!(ax_sim_head_t, [sim_head_x[frame_idx]], [sim_head_y[frame_idx]], color=simulation_color, markersize=20, marker=:square)
    
    display(fig_sim_head_t)
    save(joinpath(output_dir, "simulation_head_traj_t$(Int(t))s.png"), fig_sim_head_t)
    
    # Hardware head opaque/dashed
    fig_hw_head_t_dashed, ax_hw_head_t_dashed = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_hw_head_t_dashed, hw_head_x[1:frame_idx], hw_head_y[1:frame_idx], 
           color=hardware_color_opaque, linewidth=10, linestyle=:dash)
    scatter!(ax_hw_head_t_dashed, [hw_head_x[1]], [hw_head_y[1]], 
             color=hardware_color_opaque, markersize=20, marker=:circle)
    scatter!(ax_hw_head_t_dashed, [hw_head_x[frame_idx]], [hw_head_y[frame_idx]], 
             color=hardware_color_opaque, markersize=20, marker=:square)
    
    save(joinpath(output_dir, "hardware_head_traj_t$(Int(t))s_opaque_dashed.png"), fig_hw_head_t_dashed)
    
    # Simulation head opaque/dashed
    fig_sim_head_t_dashed, ax_sim_head_t_dashed = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_sim_head_t_dashed, sim_head_x[1:frame_idx], sim_head_y[1:frame_idx], 
           color=simulation_color_opaque, linewidth=10, linestyle=:dash)
    scatter!(ax_sim_head_t_dashed, [sim_head_x[1]], [sim_head_y[1]], 
             color=simulation_color_opaque, markersize=20, marker=:circle)
    scatter!(ax_sim_head_t_dashed, [sim_head_x[frame_idx]], [sim_head_y[frame_idx]], 
             color=simulation_color_opaque, markersize=20, marker=:square)
    
    save(joinpath(output_dir, "simulation_head_traj_t$(Int(t))s_opaque_dashed.png"), fig_sim_head_t_dashed)
    
    # Genesis head trajectory up to this time
    fig_genesis_head_t, ax_genesis_head_t = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_genesis_head_t, genesis_head_x_tank[1:frame_idx], genesis_head_y_tank[1:frame_idx], color=genesis_color, linewidth=10)
    scatter!(ax_genesis_head_t, [genesis_head_x_tank[1]], [genesis_head_y_tank[1]], color=genesis_color, markersize=20, marker=:circle)
    scatter!(ax_genesis_head_t, [genesis_head_x_tank[frame_idx]], [genesis_head_y_tank[frame_idx]], color=genesis_color, markersize=20, marker=:square)
    
    display(fig_genesis_head_t)
    save(joinpath(output_dir, "genesis_head_traj_t$(Int(t))s.png"), fig_genesis_head_t)
    
    # Genesis head opaque/dashed
    fig_genesis_head_t_dashed, ax_genesis_head_t_dashed = create_aquarium_figure(;
        backgroundcolor=:transparent,
        resolution=resolution,
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        xlim=(0, 122),
        ylim=(0, 122),
        use_data_aspect=true
    )
    
    lines!(ax_genesis_head_t_dashed, genesis_head_x_tank[1:frame_idx], genesis_head_y_tank[1:frame_idx], 
           color=genesis_color_opaque, linewidth=10, linestyle=:dash)
    scatter!(ax_genesis_head_t_dashed, [genesis_head_x_tank[1]], [genesis_head_y_tank[1]], 
             color=genesis_color_opaque, markersize=20, marker=:circle)
    scatter!(ax_genesis_head_t_dashed, [genesis_head_x_tank[frame_idx]], [genesis_head_y_tank[frame_idx]], 
             color=genesis_color_opaque, markersize=20, marker=:square)
    
    save(joinpath(output_dir, "genesis_head_traj_t$(Int(t))s_opaque_dashed.png"), fig_genesis_head_t_dashed)
    
    println("    ✓ Saved: hardware_traj_t$(Int(t))s.png, simulation_traj_t$(Int(t))s.png, and genesis_traj_t$(Int(t))s.png")
    println("    ✓ Saved: hardware_traj_t$(Int(t))s_opaque_dashed.png, simulation_traj_t$(Int(t))s_opaque_dashed.png, and genesis_traj_t$(Int(t))s_opaque_dashed.png")
    println("    ✓ Saved: hardware_head_traj_t$(Int(t))s.png, simulation_head_traj_t$(Int(t))s.png, and genesis_head_traj_t$(Int(t))s.png")
    println("    ✓ Saved: hardware_head_traj_t$(Int(t))s_opaque_dashed.png, simulation_head_traj_t$(Int(t))s_opaque_dashed.png, and genesis_head_traj_t$(Int(t))s_opaque_dashed.png")
end

#############################################################################################
## Extract Video Frames at Key Time Points
#############################################################################################

println()
println("Extracting video frames at key time points...")

# Open video file
video = VideoIO.openvideo(video_path)

# Get video properties
video_fps = VideoIO.framerate(video)
total_frames = VideoIO.counttotalframes(video)
video_duration = total_frames / video_fps

println("  Video properties:")
println("    FPS: $(round(video_fps, digits=2))")
println("    Total frames: $total_frames")
println("    Duration: $(round(video_duration, digits=2))s")

# Get motion start and end frames from tracking data
# These correspond to t=0s and t=4s in the trajectory data
# For the new CSV format, we use the actual frame range from the data
motion_start_frame = Int(hw_data_df[hw_data_df.video .== current_video_name, :frame][1])
motion_end_frame = Int(hw_data_df[hw_data_df.video .== current_video_name, :frame][end])

println("    Motion start frame (t=0s): $motion_start_frame")
println("    Motion end frame (t=4s): $motion_end_frame")
println("    Trajectory frames: $(length(hw_traj))")

# Time points to extract (same as trajectory visualizations)
time_points = [0.0, 1.0, 2.0, 3.0, 4.0]
frames = [2, 30, 60, 90, 120]
num_trajectory_frames = length(hw_traj)

for t in time_points
    # Calculate frame number by interpolating between motion_start_frame and motion_end_frame
    # based on the trajectory data (241 frames over 4 seconds)
    frame_number = frames[time_points .== t][1]
    # Seek to specific frame
    VideoIO.seek(video, (frame_number - 1) / video_fps)  # Seek by time in seconds
    
    # Read frame - returns raw image data
    img = read(video)
    display(img)
    
    # Convert to proper image format if needed
    # VideoIO.read returns an Array, we need to ensure it's in the right format
    if ndims(img) == 3 && size(img, 3) == 3
        # RGB image - reinterpret as RGB colorant array
        img_rgb = colorview(RGB, permutedims(img, (3, 1, 2)) ./ 255.0)
    else
        # Already in correct format
        img_rgb = img
    end
    
    # Save frame as PNG
    frame_filename = joinpath(output_dir, "hardware_video_frame_t$(Int(t))s.png")
    save(frame_filename, img_rgb)
    
    println("    ✓ Saved: hardware_video_frame_t$(Int(t))s.png")
end

# Close video
close(video)

println("  ✓ Video frame extraction complete")

#############################################################################################
## Extract Genesis Video Frames at Key Time Points
#############################################################################################

println()
println("Extracting genesis video frames at key time points...")

# Open genesis video file
genesis_video = VideoIO.openvideo(genesis_video_path)

# Get video properties
genesis_video_fps = VideoIO.framerate(genesis_video)
genesis_total_frames = VideoIO.counttotalframes(genesis_video)
genesis_video_duration = genesis_total_frames / genesis_video_fps

println("  Genesis video properties:")
println("    FPS: $(round(genesis_video_fps, digits=2))")
println("    Total frames: $genesis_total_frames")
println("    Duration: $(round(genesis_video_duration, digits=2))s")

# Get motion start and end frames from genesis tracking data
genesis_motion_start_frame = 1
genesis_motion_end_frame = genesis_total_frames

println("    Motion start frame (t=0s): $genesis_motion_start_frame")
println("    Motion end frame (t=8s): $genesis_motion_end_frame")
println("    Trajectory frames: $(length(genesis_traj))")

# Time points to extract (same as other visualizations)
time_points = [0.0, 1.0, 2.0, 3.0, 4.0]
frames = [2, 60, 120, 180, 240]
num_genesis_trajectory_frames = length(genesis_traj)

for t in time_points

    frame_number = frames[time_points .== t][1]
    
    println("  Extracting genesis frame at t=$(t)s (video frame $frame_number)...")
    
    # Seek to specific frame
    VideoIO.seek(genesis_video, (frame_number - 1) / genesis_video_fps)
    
    # Read frame
    img = read(genesis_video)
    
    # Convert to proper image format if needed
    if ndims(img) == 3 && size(img, 3) == 3
        # RGB image - reinterpret as RGB colorant array
        img_rgb = colorview(RGB, permutedims(img, (3, 1, 2)) ./ 255.0)
    else
        # Already in correct format
        img_rgb = img
    end
    
    # Display frame
    display(img_rgb)
    
    # Save frame as PNG
    frame_filename = joinpath(output_dir, "genesis_video_frame_t$(Int(t))s.png")
    save(frame_filename, img_rgb)
    
    println("    ✓ Saved: genesis_video_frame_t$(Int(t))s.png")
end

# Close video
close(genesis_video)

println("  ✓ Genesis video frame extraction complete")

#############################################################################################
## Create Vorticity Field Visualizations at Key Time Points
#############################################################################################

println()
println("Creating vorticity field visualizations...")

# Extract fluid velocity trajectory from aquarium state
fluid_velocity_traj = [extract_fluid_velocity(tank, aquarium_state_traj[k]) for k in 1:length(aquarium_state_traj)]

# Time points for vorticity visualization
time_points = [0.0, 1.0, 2.0, 3.0, 4.0]

for t in time_points
    # Find closest frame in simulation
    frame_idx = argmin(abs.(time_traj_sim .- t))
    actual_t = time_traj_sim[frame_idx]
    
    println("  Creating vorticity field for t=$(t)s (frame $frame_idx, actual t=$(round(actual_t, digits=1))s)")
    
    # Create figure with white background
    fig, ax = create_aquarium_figure(;
        backgroundcolor=:white,
        fontsize=fontsize,
        xlabel="X (cm)", 
        ylabel="Y (cm)",
        xlim=(0.0, length_x), 
        ylim=(0.0, length_y),
        resolution=resolution,
        spinevisible=false,
        ticksvisible=false,
        use_data_aspect=true
    )

    if t == 0.0

        # Add swimmer outline only for t=0s
        plot_solid_systems!(fig, ax,
            [rexeel],
            [swimmer_state_traj[frame_idx]]
        )

    else
    
        # Plot vorticity field with red-blue colormap
        plot_vorticity_field!(fig, ax,
            fluid_env,
            nothing, rexeel,
            fluid_velocity_traj[frame_idx],
            [], swimmer_state_traj[frame_idx];
            colormap=:PuOr,
            density=100,
            threshold_percentage=1.0,
            smooth=true,
            smooth_sigma=4.0
        )
        display(fig)

    end

    # Save figure
    vorticity_filename = joinpath(output_dir, "vorticity_field_t$(Int(t))s.png")
    save(vorticity_filename, fig)
    
    println("    ✓ Saved: vorticity_field_t$(Int(t))s.png")
end

println("  ✓ Vorticity field visualization complete")

#############################################################################################
## Save Results
#############################################################################################

println()
println("Saving analysis results...")

# Save to JLD2
jldsave(joinpath(output_dir, "trajectory_analysis.jld2");
    hw_com_traj,
    hw_com_traj_by_video,
    sim_com_traj,
    genesis_com_traj,
    hw_com_unbiased,
    sim_com_unbiased,
    genesis_com_unbiased,
    hw_head_x_unbiased,
    hw_head_y_unbiased,
    sim_head_x_unbiased,
    sim_head_y_unbiased,
    genesis_head_x_unbiased,
    genesis_head_y_unbiased,
    time_traj_motors,
    time_traj_sim,
    com_x_errors,
    com_y_errors,
    com_position_errors,
    genesis_com_x_errors,
    genesis_com_y_errors,
    genesis_com_position_errors,
    head_x_errors,
    head_y_errors,
    head_position_errors,
    genesis_head_x_errors,
    genesis_head_y_errors,
    genesis_head_position_errors,
    summary_stats = Dict(
        "hw_com_rmse_x_cm" => com_rmse_x,
        "hw_com_rmse_y_cm" => com_rmse_y,
        "hw_com_rmse_position_cm" => com_rmse_position,
        "hw_mean_com_position_error_cm" => mean_com_pos_error,
        "hw_max_com_position_error_cm" => max_com_pos_error,
        "hw_mean_com_x_error_cm" => mean(com_x_errors),
        "hw_std_com_x_error_cm" => std(com_x_errors),
        "hw_mean_com_y_error_cm" => mean(com_y_errors),
        "hw_std_com_y_error_cm" => std(com_y_errors),
        "hw_head_rmse_x_cm" => head_rmse_x,
        "hw_head_rmse_y_cm" => head_rmse_y,
        "hw_head_rmse_position_cm" => head_rmse_position,
        "hw_mean_head_position_error_cm" => mean_head_pos_error,
        "hw_max_head_position_error_cm" => max_head_pos_error,
        "hw_mean_head_x_error_cm" => mean(head_x_errors),
        "hw_std_head_x_error_cm" => std(head_x_errors),
        "hw_mean_head_y_error_cm" => mean(head_y_errors),
        "hw_std_head_y_error_cm" => std(head_y_errors),
        "genesis_com_rmse_x_cm" => genesis_com_rmse_x,
        "genesis_com_rmse_y_cm" => genesis_com_rmse_y,
        "genesis_com_rmse_position_cm" => genesis_com_rmse_position,
        "genesis_mean_com_position_error_cm" => genesis_mean_com_pos_error,
        "genesis_max_com_position_error_cm" => genesis_max_com_pos_error,
        "genesis_mean_com_x_error_cm" => mean(genesis_com_x_errors),
        "genesis_std_com_x_error_cm" => std(genesis_com_x_errors),
        "genesis_mean_com_y_error_cm" => mean(genesis_com_y_errors),
        "genesis_std_com_y_error_cm" => std(genesis_com_y_errors),
        "genesis_head_rmse_x_cm" => genesis_head_rmse_x,
        "genesis_head_rmse_y_cm" => genesis_head_rmse_y,
        "genesis_head_rmse_position_cm" => genesis_head_rmse_position,
        "genesis_mean_head_position_error_cm" => genesis_mean_head_pos_error,
        "genesis_max_head_position_error_cm" => genesis_max_head_pos_error,
        "genesis_mean_head_x_error_cm" => mean(genesis_head_x_errors),
        "genesis_std_head_x_error_cm" => std(genesis_head_x_errors),
        "genesis_mean_head_y_error_cm" => mean(genesis_head_y_errors),
        "genesis_std_head_y_error_cm" => std(genesis_head_y_errors)
    )
)

println("  ✓ Saved: trajectory_analysis.jld2 (includes all $(length(hw_com_traj_by_video)) video trajectories)")

#############################################################################################
## Summary
#############################################################################################

println()
println("="^80)
println("ANALYSIS COMPLETE")
println("="^80)
println()
println("Hardware Center of Mass Trajectory Statistics (Unbiased):")
println("  X RMSE: $(round(com_rmse_x, digits=3)) cm")
println("  Y RMSE: $(round(com_rmse_y, digits=3)) cm")
println("  Position RMSE: $(round(com_rmse_position, digits=3)) cm")
println("  Mean position error: $(round(mean_com_pos_error, digits=3)) cm")
println("  Max position error: $(round(max_com_pos_error, digits=3)) cm")
println()
println("Hardware Head Link Position Statistics (Unbiased):")
println("  X RMSE: $(round(head_rmse_x, digits=3)) cm")
println("  Y RMSE: $(round(head_rmse_y, digits=3)) cm")
println("  Position RMSE: $(round(head_rmse_position, digits=3)) cm")
println("  Mean position error: $(round(mean_head_pos_error, digits=3)) cm")
println("  Max position error: $(round(max_head_pos_error, digits=3)) cm")
println()
println("Genesis Center of Mass Trajectory Statistics (Unbiased):")
println("  X RMSE: $(round(genesis_com_rmse_x, digits=3)) cm")
println("  Y RMSE: $(round(genesis_com_rmse_y, digits=3)) cm")
println("  Position RMSE: $(round(genesis_com_rmse_position, digits=3)) cm")
println("  Mean position error: $(round(genesis_mean_com_pos_error, digits=3)) cm")
println("  Max position error: $(round(genesis_max_com_pos_error, digits=3)) cm")
println()
println("Genesis Head Link Position Statistics (Unbiased):")
println("  X RMSE: $(round(genesis_head_rmse_x, digits=3)) cm")
println("  Y RMSE: $(round(genesis_head_rmse_y, digits=3)) cm")
println("  Position RMSE: $(round(genesis_head_rmse_position, digits=3)) cm")
println("  Mean position error: $(round(genesis_mean_head_pos_error, digits=3)) cm")
println("  Max position error: $(round(genesis_max_head_pos_error, digits=3)) cm")
println()
println("="^80)