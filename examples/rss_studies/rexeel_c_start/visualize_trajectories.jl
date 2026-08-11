import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using AquariumClosed
using AquariumClosed.CairoMakie
using JLD2
using LinearAlgebra
using Statistics
using Colors
using PGFPlotsX
# using CSV
# using DataFrames
# using Interpolations
# using VideoIO
using Printf

#############################################################################################
## Configuration
#############################################################################################

# Data paths
data_dir = expanduser("~/aquariumCLOSED/data/rexeel_c_start/")
# hardware_tracking_path = expanduser("~/aquariumCLOSED/data/rexeel_c_start/hardware_trajectories.csv")
# genesis_tracking_path = expanduser("~/aquariumCLOSED/data/rexeel_c_start/genesis_trajectories.csv")
simulation_dir = expanduser("~/aquariumCLOSED/data/rexeel_c_start/")

# Output directory
output_dir = expanduser("~/aquariumCLOSED/visualization/rss_figures/c_start")
mkpath(output_dir)

# Parameter sets to visualize (must match simulate_c_starts.jl)
parameter_set_names = ["initial", "optimal"]

# # Map hardware trial names to parameter set names
# trial_to_param_map = Dict(
#     "Camo 录像 2026-01-29 00-19-03" => "optimal",
#     "Camo 录像 2026-01-29 01-06-22" => "initial",
# )

# trial_to_param_map_genesis = Dict(
#     "Simulation_cstart_optimal" => "optimal",
#     "Simulation_cstart_original" => "initial",
# )

println("="^80)
println("RExEel C-Start: Head Trajectory Analysis")
println("="^80)
println()

#############################################################################################
## Load Data
#############################################################################################

println("Loading data...")

# # Load hardware tracking data (single CSV with multiple trials)
# hw_data_df = CSV.read(hardware_tracking_path, DataFrame)
#
# # Organize trajectories and timestamps by trial name, then map to parameter sets
# hw_trajectories_by_param = Dict{String, Vector{Vector{Float64}}}()
# hw_timestamps_by_param = Dict{String, Vector{Float64}}()
#
# for trial_name in unique(hw_data_df.video)
#     trial_data = hw_data_df[hw_data_df.video .== trial_name, :]
#     sort!(trial_data, :frame)
#     traj = [[row.x_cm, row.y_cm, deg2rad(row.robot_yaw)] for row in eachrow(trial_data)]
#     theta_vals = [traj[i][3] for i in 1:length(traj)]
#     theta_unwrapped = copy(theta_vals)
#     for i in 2:length(theta_unwrapped)
#         diff = theta_unwrapped[i] - theta_unwrapped[i-1]
#         if diff > π
#             theta_unwrapped[i:end] .-= 2π
#         elseif diff < -π
#             theta_unwrapped[i:end] .+= 2π
#         end
#     end
#     traj = [[traj[i][1], traj[i][2], theta_unwrapped[i]] for i in 1:length(traj)]
#     timestamps = range(0.0, (length(traj)-1)/30.0, length=length(traj))
#     if haskey(trial_to_param_map, trial_name)
#         param_set_name = trial_to_param_map[trial_name]
#         hw_trajectories_by_param[param_set_name] = traj
#         hw_timestamps_by_param[param_set_name] = timestamps
#         println("  ✓ Hardware tracking loaded: $trial_name → $param_set_name ($(length(traj)) frames)")
#     else
#         println("  ⚠ Warning: Unknown trial name '$trial_name' (not in trial_to_param_map)")
#     end
# end

# # Load genesis tracking data (single CSV with multiple trials)
# genesis_data_df = CSV.read(genesis_tracking_path, DataFrame)
#
# genesis_trajectories_by_param = Dict{String, Vector{Vector{Float64}}}()
# genesis_timestamps_by_param = Dict{String, Vector{Float64}}()
#
# for trial_name in unique(genesis_data_df.video)
#     trial_data = genesis_data_df[genesis_data_df.video .== trial_name, :]
#     sort!(trial_data, :frame)
#     traj = [[row.x_cm, row.y_cm, deg2rad(row.robot_yaw)] for row in eachrow(trial_data)]
#     timestamps = range(0.0, (length(traj)-1)/60.0, length=length(traj))
#     if haskey(trial_to_param_map_genesis, trial_name)
#         param_set_name = trial_to_param_map_genesis[trial_name]
#         genesis_trajectories_by_param[param_set_name] = traj
#         genesis_timestamps_by_param[param_set_name] = timestamps
#     end
# end

# Load simulation data for each parameter set
sim_data_by_param = Dict{String, Dict}()
time_traj_sim = nothing  # Will be set from first simulation

for param_set_name in parameter_set_names
    sim_path = joinpath(simulation_dir, param_set_name, "$(param_set_name)_simulation.jld2")

    if !isfile(sim_path)
        println("  ⚠ Warning: Simulation file not found: $sim_path")
        continue
    end

    sim_data = load(sim_path)
    sim_data_by_param[param_set_name] = sim_data

    # Set time trajectory from first simulation
    if time_traj_sim === nothing
        global time_traj_sim = sim_data["trajectories"][:time_traj]
    end

    println("  ✓ Loaded simulation: $param_set_name ($(length(sim_data["trajectories"][:time_traj])) timesteps)")
end

if time_traj_sim === nothing
    error("No simulation data loaded. Please run simulate_c_starts.jl first.")
end

println()

#############################################################################################
## Plot parameters
#############################################################################################

background_color = :transparent
fontsize = 18
resolution = (1000, 600)

# Define colors
simulation_color = RGB(0.0, 0.7294, 0.3451)  # jj_green
simulation_color_opaque = RGBA(0.0, 0.7294, 0.3451, 1.0)
# hardware_color = RGB(0.933, 0.227, 0.275)  # jj_red
# hardware_color_opaque = RGBA(0.933, 0.227, 0.275, 1.0)
# genesis_color = RGB(0.9451, 0.6745, 0.09020)  # jj_orange
# genesis_color_opaque = RGBA(0.9451, 0.6745, 0.09020, 1.0)

println("Colors defined")

#############################################################################################
## Define fluid domain and RExEel (for reconstruction)
#############################################################################################

# time properties
time_step = 1/60
final_time = 3.0
N_time = Int(final_time/time_step) + 1

# fluid properties (water)
fluid_density = 1.0  # g/cm³
dynamic_viscosity = 0.01  # g/(cm*s)

# fish tank dimensions
length_x = 122.
length_y = 122.

# fluid grid
num_cells_x = 122
num_cells_y = 122

# boundary conditions
boundary_condition_type = :wall

# Create fluid environment
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
println()

#############################################################################################
## Define 6-link RExEel (swimmer)
#############################################################################################

# eel properties
n_links = 6
link_lengths = [12.0, 9.8 .* ones(n_links-1)...]  # cm
height = 9.35  # cm
masses_per_link = [192, 140 .* ones(n_links-1)...] ./ height # g per link
moi_per_link = [2435.99, 1483.49 .* ones(n_links-1)...] ./ height  # g·cm²
gravity_constant = 0.0

# boundary properties
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

println("RExEel configuration: 6 links")
println()

# Create AquariumTank (needed for vorticity field extraction)
tank = AquariumTank_only_swimmer(fluid_env, rexeel)

println("AquariumTank created")
println()

#############################################################################################
## Extract Simulation Head Trajectories
#############################################################################################

println("Extracting simulation head trajectories...")

sim_head_traj_by_param = Dict{String, Vector{Vector{Float64}}}()

for (param_set_name, sim_data) in sim_data_by_param
    trajectories = sim_data["trajectories"]
    swimmer_state_traj = trajectories[:swimmer_state_traj]

    # Extract head positions (first link, indices 1, 2, 3 for x, y, theta)
    head_traj = []
    for state in swimmer_state_traj
        config = state[rexeel.configuration_indices]
        x_head = config[1]
        y_head = config[2]
        theta_head = config[3]
        push!(head_traj, [x_head, y_head, theta_head])
    end

    sim_head_traj_by_param[param_set_name] = head_traj
    println("  ✓ $param_set_name: $(length(head_traj)) timesteps")
end

println()

#############################################################################################
## Export Simulation Head Trajectories to CSV (Hardware Format)
#############################################################################################

# println("Exporting simulation head trajectories to CSV...")
#
# csv_output_dir = joinpath(data_dir, "simulation_trajectories")
# mkpath(csv_output_dir)
#
# for (param_set_name, head_traj) in sim_head_traj_by_param
#     df = DataFrame(
#         video = String[], frame = Int[], timestamp = Float64[],
#         x_cm = Float64[], y_cm = Float64[], robot_yaw = Float64[], interpolated = Int[])
#     trial_name = "Simulation_cstart_$(param_set_name)"
#     for (i, (state, t)) in enumerate(zip(head_traj, time_traj_sim))
#         push!(df, (video=trial_name, frame=i-1, timestamp=t,
#             x_cm=state[1], y_cm=state[2], robot_yaw=rad2deg(state[3]), interpolated=0))
#     end
#     CSV.write(joinpath(csv_output_dir, "$(param_set_name)_simulation_trajectory.csv"), df)
# end
#
# combined_df = DataFrame(
#     video = String[], frame = Int[], timestamp = Float64[],
#     x_cm = Float64[], y_cm = Float64[], robot_yaw = Float64[], interpolated = Int[])
# for (param_set_name, head_traj) in sim_head_traj_by_param
#     trial_name = "Simulation_cstart_$(param_set_name)"
#     for (i, (state, t)) in enumerate(zip(head_traj, time_traj_sim))
#         push!(combined_df, (video=trial_name, frame=i-1, timestamp=t,
#             x_cm=state[1], y_cm=state[2], robot_yaw=rad2deg(state[3]), interpolated=0))
#     end
# end
# CSV.write(joinpath(data_dir, "simulation_trajectories.csv"), combined_df)
#
# println()

# #############################################################################################
# ## Interpolate Hardware Head Trajectories to Simulation Time
# #############################################################################################
#
# println("Interpolating hardware head trajectories to simulation time...")
#
# hw_head_traj_interp_by_param = Dict{String, Vector{Vector{Float64}}}()
#
# for (param_set_name, traj) in hw_trajectories_by_param
#     hw_timestamps = hw_timestamps_by_param[param_set_name]
#     hw_timestamps = range(0.0, min(hw_timestamps[end], time_traj_sim[end]), length=length(hw_timestamps))
#     x_traj = [traj[i][1] for i in 1:length(traj)]
#     y_traj = [traj[i][2] for i in 1:length(traj)]
#     theta_traj = [-traj[i][3] for i in 1:length(traj)]
#     itp_x = CubicSplineInterpolation(hw_timestamps, x_traj, extrapolation_bc=Line())
#     itp_y = CubicSplineInterpolation(hw_timestamps, y_traj, extrapolation_bc=Line())
#     itp_theta = CubicSplineInterpolation(hw_timestamps, theta_traj, extrapolation_bc=Line())
#     traj_interp = []
#     for t_sim in time_traj_sim
#         x_interp = itp_x(t_sim)
#         y_interp = itp_y(t_sim)
#         theta_interp = itp_theta(t_sim)
#         push!(traj_interp, [x_interp, y_interp, theta_interp])
#     end
#     hw_head_traj_interp_by_param[param_set_name] = traj_interp
#     println("  ✓ $param_set_name: interpolated to $(length(traj_interp)) timesteps")
# end
#
# println()

#############################################################################################
## Interpolate Genesis Head Trajectories to Simulation Time
#############################################################################################

# println("Interpolating genesis head trajectories to simulation time...")
#
# genesis_head_traj_interp_by_param = Dict{String, Vector{Vector{Float64}}}()
#
# for (param_set_name, traj) in genesis_trajectories_by_param
#     genesis_timestamps = genesis_timestamps_by_param[param_set_name]
#     genesis_timestamps = range(0.0, min(genesis_timestamps[end], time_traj_sim[end]), length=length(genesis_timestamps))
#     x_traj = [traj[i][1]-11 for i in 1:length(traj)]
#     y_traj = [traj[i][2] for i in 1:length(traj)]
#     theta_traj = [-traj[i][3] for i in 1:length(traj)]
#     itp_x = CubicSplineInterpolation(genesis_timestamps, x_traj, extrapolation_bc=Line())
#     itp_y = CubicSplineInterpolation(genesis_timestamps, y_traj, extrapolation_bc=Line())
#     itp_theta = CubicSplineInterpolation(genesis_timestamps, theta_traj, extrapolation_bc=Line())
#     traj_interp = []
#     for t_sim in time_traj_sim
#         x_interp = itp_x(t_sim)
#         y_interp = itp_y(t_sim)
#         theta_interp = itp_theta(t_sim)
#         push!(traj_interp, [x_interp, y_interp, theta_interp])
#     end
#     genesis_head_traj_interp_by_param[param_set_name] = traj_interp
# end
#
# println()

#############################################################################################
## Compute Position Differences (RMSE) for each parameter set
#############################################################################################

# println("Computing position differences (RMSE)...")
#
# println()
# println("="^80)
# println("RMSE Analysis: Hardware vs Simulation")
# println("="^80)
#
# for param_set_name in parameter_set_names
#     if !haskey(sim_head_traj_by_param, param_set_name)
#         continue
#     end
#     if !haskey(hw_head_traj_interp_by_param, param_set_name)
#         println("  ⚠ Warning: No hardware data for parameter set: $param_set_name")
#         continue
#     end
#     sim_traj = sim_head_traj_by_param[param_set_name]
#     hw_traj = hw_head_traj_interp_by_param[param_set_name]
#     N = min(length(sim_traj), length(hw_traj))
#     sim_initial = sim_traj[1]
#     hw_initial = hw_traj[1]
#     x_errors = [(hw_traj[i][1] - hw_initial[1]) - (sim_traj[i][1] - sim_initial[1]) for i in 1:N]
#     y_errors = [(hw_traj[i][2] - hw_initial[2]) - (sim_traj[i][2] - sim_initial[2]) for i in 1:N]
#     theta_errors = [(hw_traj[i][3] - hw_initial[3]) - (sim_traj[i][3] - sim_initial[3]) for i in 1:N]
#     position_errors = [sqrt(x_errors[i]^2 + y_errors[i]^2) for i in 1:N]
#     rmse_x = sqrt(mean(x_errors.^2))
#     rmse_y = sqrt(mean(y_errors.^2))
#     rmse_theta = sqrt(mean(theta_errors.^2))
#     rmse_position = sqrt(mean(position_errors.^2))
#     println("$param_set_name:")
#     println("  RMSE X: $(round(rmse_x, digits=3)) cm")
#     println("  RMSE Y: $(round(rmse_y, digits=3)) cm")
#     println("  RMSE Theta: $(round(rad2deg(rmse_theta), digits=3))°")
#     println("  RMSE Position: $(round(rmse_position, digits=3)) cm")
#     println()
# end
#
# println("="^80)
# println("RMSE Analysis: Genesis vs Hardware")
# println("="^80)
#
# for param_set_name in parameter_set_names
#     if !haskey(hw_head_traj_interp_by_param, param_set_name)
#         println("  ⚠ Warning: No hardware data for parameter set: $param_set_name")
#         continue
#     end
#     if !haskey(genesis_head_traj_interp_by_param, param_set_name)
#         println("  ⚠ Warning: No genesis data for parameter set: $param_set_name")
#         continue
#     end
#     hw_traj = hw_head_traj_interp_by_param[param_set_name]
#     genesis_traj = genesis_head_traj_interp_by_param[param_set_name]
#     N = min(length(hw_traj), length(genesis_traj))
#     hw_initial = hw_traj[1]
#     genesis_initial = genesis_traj[1]
#     x_errors = [(genesis_traj[i][1] - genesis_initial[1]) - (hw_traj[i][1] - hw_initial[1]) for i in 1:N]
#     y_errors = [(genesis_traj[i][2] - genesis_initial[2]) - (hw_traj[i][2] - hw_initial[2]) for i in 1:N]
#     theta_errors = [(genesis_traj[i][3] - genesis_initial[3]) - (hw_traj[i][3] - hw_initial[3]) for i in 1:N]
#     position_errors = [sqrt(x_errors[i]^2 + y_errors[i]^2) for i in 1:N]
#     rmse_x = sqrt(mean(x_errors.^2))
#     rmse_y = sqrt(mean(y_errors.^2))
#     rmse_theta = sqrt(mean(theta_errors.^2))
#     rmse_position = sqrt(mean(position_errors.^2))
#     println("$param_set_name:")
#     println("  RMSE X: $(round(rmse_x, digits=3)) cm")
#     println("  RMSE Y: $(round(rmse_y, digits=3)) cm")
#     println("  RMSE Theta: $(round(rad2deg(rmse_theta), digits=3))°")
#     println("  RMSE Position: $(round(rmse_position, digits=3)) cm")
#     println()
# end
#
# println()

#############################################################################################
## Extract Simulation COM Trajectories
#############################################################################################

println("Extracting simulation COM trajectories...")

sim_com_traj_by_param = Dict{String, Vector{Vector{Float64}}}()

for (param_set_name, sim_data) in sim_data_by_param
    trajectories = sim_data["trajectories"]
    swimmer_state_traj = trajectories[:swimmer_state_traj]

    # Extract COM positions for each timestep
    com_traj = []
    for state in swimmer_state_traj
        config = state[rexeel.configuration_indices]

        # Compute COM from link positions
        total_mass = sum(masses_per_link)
        com_x = 0.0
        com_y = 0.0

        for i in 1:n_links
            link_x = config[3*(i-1) + 1]
            link_y = config[3*(i-1) + 2]
            com_x += masses_per_link[i] * link_x
            com_y += masses_per_link[i] * link_y
        end

        com_x /= total_mass
        com_y /= total_mass

        push!(com_traj, [com_x, com_y])
    end

    sim_com_traj_by_param[param_set_name] = com_traj
    println("  ✓ $param_set_name: $(length(com_traj)) timesteps")
end

println()

#############################################################################################
## Compute Trajectory Metrics (Heading, Velocity, Distance)
#############################################################################################

println("="^80)
println("Trajectory Metrics Analysis")
println("="^80)
println()

trajectory_metrics = Dict()

for param_set_name in parameter_set_names
    if !haskey(sim_head_traj_by_param, param_set_name)
        continue
    end

    metrics = Dict()

    # Helper function to compute metrics for a COM trajectory (2D: x, y only)
    function compute_com_trajectory_metrics(traj, timestamps)
        # Calculate final velocity using finite differences (average over last few frames)
        n_frames = min(10, length(traj) - 1)  # Use last 10 frames or fewer
        final_velocities = []
        for i in (length(traj) - n_frames):(length(traj) - 1)
            dt = timestamps[i+1] - timestamps[i]
            dx = traj[i+1][1] - traj[i][1]
            dy = traj[i+1][2] - traj[i][2]
            vx = dx / dt
            vy = dy / dt
            push!(final_velocities, [vx, vy])
        end

        # Average final velocity
        avg_final_vx = mean([v[1] for v in final_velocities])
        avg_final_vy = mean([v[2] for v in final_velocities])

        # Final velocity magnitude
        final_vel_mag = sqrt(avg_final_vx^2 + avg_final_vy^2)

        # Final velocity direction (heading angle)
        final_heading = atan(avg_final_vy, avg_final_vx)  # radians

        # Calculate displacement vector from start to end
        displacement_x = traj[end][1] - traj[1][1]
        displacement_y = traj[end][2] - traj[1][2]

        # Project displacement onto final heading direction
        # This gives displacement in the direction the COM is ultimately heading
        net_displacement_in_heading = displacement_x * cos(final_heading) + displacement_y * sin(final_heading)

        return Dict(
            "final_velocity_magnitude" => final_vel_mag,
            "final_heading_rad" => final_heading,
            "final_heading_deg" => rad2deg(final_heading),
            "net_displacement_in_heading" => net_displacement_in_heading,
            "final_vx" => avg_final_vx,
            "final_vy" => avg_final_vy
        )
    end

    # Compute metrics for simulation using COM trajectory
    if haskey(sim_com_traj_by_param, param_set_name)
        sim_metrics = compute_com_trajectory_metrics(
            sim_com_traj_by_param[param_set_name],
            time_traj_sim
        )
        metrics["simulation"] = sim_metrics
    end

    # For hardware and genesis, we need COM data - check if available
    # Note: If hardware/genesis CSVs don't have COM data, these will be skipped
    # and only simulation metrics will be shown
    # println("  Note: Hardware and Genesis COM data not yet implemented - only simulation COM velocity will be reported")

    # Placeholder for future COM data from hardware/genesis
    # if haskey(hw_com_trajectories_by_param, param_set_name)
    #     hw_metrics = compute_com_trajectory_metrics(
    #         hw_com_trajectories_by_param[param_set_name],
    #         hw_timestamps_by_param[param_set_name]
    #     )
    #     metrics["hardware"] = hw_metrics
    # end

    # if haskey(genesis_com_trajectories_by_param, param_set_name)
    #     genesis_metrics = compute_com_trajectory_metrics(
    #         genesis_com_trajectories_by_param[param_set_name],
    #         genesis_timestamps_by_param[param_set_name]
    #     )
    #     metrics["genesis"] = genesis_metrics
    # end

    trajectory_metrics[param_set_name] = metrics
end

println()

#############################################################################################
## Create plots for each parameter set
#############################################################################################

for param_set_name in parameter_set_names
    if !haskey(sim_head_traj_by_param, param_set_name)
        println("  ⚠ Skipping $param_set_name (no simulation data)")
        continue
    end

    # if !haskey(hw_head_traj_interp_by_param, param_set_name)
    #     println("  ⚠ Skipping $param_set_name (no hardware data)")
    #     continue
    # end

    # if !haskey(genesis_head_traj_interp_by_param, param_set_name)
    #     println("  ⚠ Skipping $param_set_name (no genesis data)")
    #     continue
    # end

    println("="^80)
    println("Creating plots for: $param_set_name")
    println("="^80)

    # Get trajectories
    sim_traj = sim_head_traj_by_param[param_set_name]
    # hw_traj = hw_head_traj_interp_by_param[param_set_name]
    # genesis_traj = genesis_head_traj_interp_by_param[param_set_name]

    # Extract components
    sim_head_x = [sim_traj[i][1] for i in 1:length(sim_traj)]
    sim_head_y = [sim_traj[i][2] for i in 1:length(sim_traj)]
    sim_head_theta = [sim_traj[i][3] for i in 1:length(sim_traj)]

    # hw_head_x = [hw_traj[i][1] for i in 1:length(hw_traj)]
    # hw_head_y = [hw_traj[i][2] for i in 1:length(hw_traj)]
    # hw_head_theta = [hw_traj[i][3] for i in 1:length(hw_traj)]

    # genesis_head_x = [genesis_traj[i][1] for i in 1:length(genesis_traj)]
    # genesis_head_y = [genesis_traj[i][2] for i in 1:length(genesis_traj)]
    # genesis_head_theta = [genesis_traj[i][3] for i in 1:length(genesis_traj)]

    #####################################################################################
    ## Plot 1: Head X Position
    #####################################################################################

    println("  Creating Head X position plot...")

    fig_head_x, ax_head_x = create_aquarium_figure(;
        backgroundcolor=background_color,
        fontsize=fontsize,
        resolution=resolution,
        xlabel="Time (s)",
        ylabel="Head X Position (cm)",
        use_data_aspect=false
    )

    # # Plot hardware trajectory
    # lines!(ax_head_x, time_traj_sim, hw_head_x, color=hardware_color_opaque, linewidth=2, label="Hardware")

    # Plot genesis trajectory
    # lines!(ax_head_x, time_traj_sim, genesis_head_x, color=genesis_color_opaque, linewidth=2, label="Genesis")

    # Plot simulation trajectory
    lines!(ax_head_x, time_traj_sim, sim_head_x, color=simulation_color_opaque, linewidth=3, label="Simulation")

    axislegend(ax_head_x, position=:lt)
    display(fig_head_x)

    # Save as PNG
    save(joinpath(output_dir, "$(param_set_name)_head_x_comparison.png"), fig_head_x)
    println("    ✓ Saved: $(param_set_name)_head_x_comparison.png")

    # Create TikZ plot
    head_x_plot = @pgf PGFPlotsX.Axis(
        {
            xmajorgrids,
            ymajorgrids,
            xlabel = "Time (s)",
            ylabel = "Head X Position (cm)",
            legend_pos = "north east",
            legend_cell_align = "left",
        },
        # PlotInc(@pgf({no_marks, "thick", color=hardware_color_opaque}),
        #     Coordinates(time_traj_sim, hw_head_x)),
        # PlotInc(@pgf({no_marks, "thick", color=genesis_color_opaque}),
        #     Coordinates(time_traj_sim, genesis_head_x)),
        PlotInc(@pgf({no_marks, "very thick", color=simulation_color_opaque}),
            Coordinates(time_traj_sim, sim_head_x)),
        PGFPlotsX.Legend(["Simulation"])
    )

    tikz_filename = joinpath(output_dir, "$(param_set_name)_head_x_comparison.tikz")
    pgfsave(tikz_filename, head_x_plot, include_preamble=false)
    println("    ✓ Saved: $(param_set_name)_head_x_comparison.tikz")

    #####################################################################################
    ## Plot 2: Head Y Position
    #####################################################################################

    println("  Creating Head Y position plot...")

    fig_head_y, ax_head_y = create_aquarium_figure(;
        backgroundcolor=background_color,
        fontsize=fontsize,
        resolution=resolution,
        xlabel="Time (s)",
        ylabel="Head Y Position (cm)",
        use_data_aspect=false
    )

    # # Plot hardware trajectory
    # lines!(ax_head_y, time_traj_sim, hw_head_y, color=hardware_color_opaque, linewidth=2, label="Hardware")

    # Plot genesis trajectory
    # lines!(ax_head_y, time_traj_sim, genesis_head_y, color=genesis_color_opaque, linewidth=2, label="Genesis")

    # Plot simulation trajectory
    lines!(ax_head_y, time_traj_sim, sim_head_y, color=simulation_color_opaque, linewidth=3, label="Simulation")

    axislegend(ax_head_y, position=:lt)
    display(fig_head_y)

    # Save as PNG
    save(joinpath(output_dir, "$(param_set_name)_head_y_comparison.png"), fig_head_y)
    println("    ✓ Saved: $(param_set_name)_head_y_comparison.png")

    # Create TikZ plot
    head_y_plot = @pgf PGFPlotsX.Axis(
        {
            xmajorgrids,
            ymajorgrids,
            xlabel = "Time (s)",
            ylabel = "Head Y Position (cm)",
            legend_pos = "north east",
            legend_cell_align = "left",
        },
        # PlotInc(@pgf({no_marks, "thick", color=hardware_color_opaque}),
        #     Coordinates(time_traj_sim, hw_head_y)),
        # PlotInc(@pgf({no_marks, "thick", color=genesis_color_opaque}),
        #     Coordinates(time_traj_sim, genesis_head_y)),
        PlotInc(@pgf({no_marks, "very thick", color=simulation_color_opaque}),
            Coordinates(time_traj_sim, sim_head_y)),
        PGFPlotsX.Legend(["Simulation"])
    )

    tikz_filename = joinpath(output_dir, "$(param_set_name)_head_y_comparison.tikz")
    pgfsave(tikz_filename, head_y_plot, include_preamble=false)
    println("    ✓ Saved: $(param_set_name)_head_y_comparison.tikz")

    #####################################################################################
    ## Plot 3: Head Angle (Theta)
    #####################################################################################

    println("  Creating Head theta plot...")

    fig_head_theta, ax_head_theta = create_aquarium_figure(;
        backgroundcolor=background_color,
        fontsize=fontsize,
        resolution=resolution,
        xlabel="Time (s)",
        ylabel="Head Angle (rad)",
        use_data_aspect=false
    )

    # # Plot hardware trajectory
    # lines!(ax_head_theta, time_traj_sim, hw_head_theta, color=hardware_color_opaque, linewidth=2, label="Hardware")

    # Plot genesis trajectory
    # lines!(ax_head_theta, time_traj_sim, genesis_head_theta, color=genesis_color_opaque, linewidth=2, label="Genesis")

    # Plot simulation trajectory
    lines!(ax_head_theta, time_traj_sim, sim_head_theta, color=simulation_color_opaque, linewidth=3, label="Simulation")

    axislegend(ax_head_theta, position=:lt)
    display(fig_head_theta)

    # Save as PNG
    save(joinpath(output_dir, "$(param_set_name)_head_theta_comparison.png"), fig_head_theta)
    println("    ✓ Saved: $(param_set_name)_head_theta_comparison.png")

    # Create TikZ plot
    head_theta_plot = @pgf PGFPlotsX.Axis(
        {
            xmajorgrids,
            ymajorgrids,
            xlabel = "Time (s)",
            ylabel = "Head Angle (rad)",
            legend_pos = "north east",
            legend_cell_align = "left",
        },
        # PlotInc(@pgf({no_marks, "thick", color=hardware_color_opaque}),
        #     Coordinates(time_traj_sim, hw_head_theta)),
        # PlotInc(@pgf({no_marks, "thick", color=genesis_color_opaque}),
        #     Coordinates(time_traj_sim, genesis_head_theta)),
        PlotInc(@pgf({no_marks, "very thick", color=simulation_color_opaque}),
            Coordinates(time_traj_sim, sim_head_theta)),
        PGFPlotsX.Legend(["Simulation"])
    )

    tikz_filename = joinpath(output_dir, "$(param_set_name)_head_theta_comparison.tikz")
    pgfsave(tikz_filename, head_theta_plot, include_preamble=false)
    println("    ✓ Saved: $(param_set_name)_head_theta_comparison.tikz")

    #####################################################################################
    ## Plot 4: Head Trajectory in XY Plane (Tank View)
    #####################################################################################

    println("  Creating Head XY trajectory plot...")

    fig_head_xy, ax_head_xy = create_aquarium_figure(;
        backgroundcolor=background_color,
        fontsize=fontsize,
        resolution=(800, 800),
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        use_data_aspect=true
    )

    # Set axis limits to tank dimensions
    xlims!(ax_head_xy, 0, length_x)
    ylims!(ax_head_xy, 0, length_y)

    # If showing optimal trajectories, also plot initial trajectories as dotted lines
    if param_set_name == "optimal"
        # Get initial trajectories
        if haskey(sim_head_traj_by_param, "initial")
            initial_sim_traj = sim_head_traj_by_param["initial"]
            initial_sim_head_x = [initial_sim_traj[i][1] for i in 1:length(initial_sim_traj)]
            initial_sim_head_y = [initial_sim_traj[i][2] for i in 1:length(initial_sim_traj)]
            lines!(ax_head_xy, initial_sim_head_x, initial_sim_head_y, color=simulation_color_opaque, linewidth=5, linestyle=:dot, label="Simulation (Initial)")
        end

        # if haskey(hw_head_traj_interp_by_param, "initial")
        #     initial_hw_traj = hw_head_traj_interp_by_param["initial"]
        #     initial_hw_head_x = [initial_hw_traj[i][1] for i in 1:length(initial_hw_traj)]
        #     initial_hw_head_y = [initial_hw_traj[i][2] for i in 1:length(initial_hw_traj)]
        #     lines!(ax_head_xy, initial_hw_head_x, initial_hw_head_y, color=hardware_color_opaque, linewidth=3, linestyle=:dot, label="Hardware (Initial)")
        # end

        # if haskey(genesis_head_traj_interp_by_param, "initial")
        #     initial_genesis_traj = genesis_head_traj_interp_by_param["initial"]
        #     initial_genesis_head_x = [initial_genesis_traj[i][1] for i in 1:length(initial_genesis_traj)]
        #     initial_genesis_head_y = [initial_genesis_traj[i][2] for i in 1:length(initial_genesis_traj)]
        #     lines!(ax_head_xy, initial_genesis_head_x, initial_genesis_head_y, color=genesis_color_opaque, linewidth=3, linestyle=:dot, label="Genesis (Initial)")
        # end
    end

    # # Plot hardware trajectory
    # lines!(ax_head_xy, hw_head_x, hw_head_y, color=hardware_color_opaque, linewidth=5, label="Hardware")

    # # Plot genesis trajectory
    # lines!(ax_head_xy, genesis_head_x, genesis_head_y, color=genesis_color_opaque, linewidth=5, label="Genesis")

    # Plot simulation trajectory
    lines!(ax_head_xy, sim_head_x, sim_head_y, color=simulation_color_opaque, linewidth=7, label="Simulation")

    axislegend(ax_head_xy, position=:lt)
    display(fig_head_xy)

    # Save as PNG
    save(joinpath(output_dir, "$(param_set_name)_head_xy_trajectory.png"), fig_head_xy)
    println("    ✓ Saved: $(param_set_name)_head_xy_trajectory.png")

    # Create TikZ plot (also with initial trajectories if optimal)
    if param_set_name == "optimal"
        # Build plot with initial trajectories
        plot_increments = []
        legend_entries = []

        # Add initial trajectories first (dotted)
        # if haskey(hw_head_traj_interp_by_param, "initial")
        #     initial_hw_traj = hw_head_traj_interp_by_param["initial"]
        #     initial_hw_head_x = [initial_hw_traj[i][1] for i in 1:length(initial_hw_traj)]
        #     initial_hw_head_y = [initial_hw_traj[i][2] for i in 1:length(initial_hw_traj)]
        #     push!(plot_increments, PlotInc(@pgf({no_marks, "thick", "dotted", color=hardware_color}),
        #         Coordinates(initial_hw_head_x, initial_hw_head_y)))
        #     push!(legend_entries, "Hardware (Initial)")
        # end

        # if haskey(genesis_head_traj_interp_by_param, "initial")
        #     initial_genesis_traj = genesis_head_traj_interp_by_param["initial"]
        #     initial_genesis_head_x = [initial_genesis_traj[i][1] for i in 1:length(initial_genesis_traj)]
        #     initial_genesis_head_y = [initial_genesis_traj[i][2] for i in 1:length(initial_genesis_traj)]
        #     push!(plot_increments, PlotInc(@pgf({no_marks, "thick", "dotted", color=genesis_color}),
        #         Coordinates(initial_genesis_head_x, initial_genesis_head_y)))
        #     push!(legend_entries, "Genesis (Initial)")
        # end

        if haskey(sim_head_traj_by_param, "initial")
            initial_sim_traj = sim_head_traj_by_param["initial"]
            initial_sim_head_x = [initial_sim_traj[i][1] for i in 1:length(initial_sim_traj)]
            initial_sim_head_y = [initial_sim_traj[i][2] for i in 1:length(initial_sim_traj)]
            push!(plot_increments, PlotInc(@pgf({no_marks, "thick", "dotted", color=simulation_color}),
                Coordinates(initial_sim_head_x, initial_sim_head_y)))
            push!(legend_entries, "Simulation (Initial)")
        end

        # Add optimal trajectories (solid)
        # push!(plot_increments, PlotInc(@pgf({no_marks, "very thick", color=hardware_color}),
        #     Coordinates(hw_head_x, hw_head_y)))
        # push!(legend_entries, "Hardware")

        # push!(plot_increments, PlotInc(@pgf({no_marks, "very thick", color=genesis_color}),
        #     Coordinates(genesis_head_x, genesis_head_y)))
        # push!(legend_entries, "Genesis")

        push!(plot_increments, PlotInc(@pgf({no_marks, "very thick", color=simulation_color}),
            Coordinates(sim_head_x, sim_head_y)))
        push!(legend_entries, "Simulation")

        head_xy_plot = @pgf PGFPlotsX.Axis(
            {
                xmajorgrids,
                ymajorgrids,
                xlabel = "X Position (cm)",
                ylabel = "Y Position (cm)",
                legend_pos = "north west",
                legend_cell_align = "left",
                xmin = 0,
                xmax = length_x,
                ymin = 0,
                ymax = length_y,
                axis_equal,
            },
            plot_increments...,
            PGFPlotsX.Legend(legend_entries)
        )
    else
        # Original plot without initial trajectories
        head_xy_plot = @pgf PGFPlotsX.Axis(
            {
                xmajorgrids,
                ymajorgrids,
                xlabel = "X Position (cm)",
                ylabel = "Y Position (cm)",
                legend_pos = "north west",
                legend_cell_align = "left",
                xmin = 0,
                xmax = length_x,
                ymin = 0,
                ymax = length_y,
                axis_equal,
            },
            # PlotInc(@pgf({no_marks, "very thick", color=hardware_color}),
            #     Coordinates(hw_head_x, hw_head_y)),
            # PlotInc(@pgf({no_marks, "very thick", color=genesis_color}),
            #     Coordinates(genesis_head_x, genesis_head_y)),
            PlotInc(@pgf({no_marks, "very thick", color=simulation_color}),
                Coordinates(sim_head_x, sim_head_y)),
            PGFPlotsX.Legend(["Simulation"])
        )
    end

    tikz_filename = joinpath(output_dir, "$(param_set_name)_head_xy_trajectory.tikz")
    pgfsave(tikz_filename, head_xy_plot, include_preamble=false)
    println("    ✓ Saved: $(param_set_name)_head_xy_trajectory.tikz")

    # #####################################################################################
    # ## Plot 5: Hardware Head Trajectory - Version 1 (Solid, Non-opaque)
    # #####################################################################################
    #
    # println("  Creating Hardware head trajectory (solid, non-opaque)...")
    # fig_hw_head_v1, ax_hw_head_v1 = create_aquarium_figure(;
    #     backgroundcolor=background_color, fontsize=fontsize, resolution=(800, 800),
    #     xlabel="X Position (cm)", ylabel="Y Position (cm)",
    #     spinevisible=false, ticksvisible=false, use_data_aspect=true)
    # xlims!(ax_hw_head_v1, 0, length_x)
    # ylims!(ax_hw_head_v1, 0, length_y)
    # lines!(ax_hw_head_v1, hw_head_x, hw_head_y, color=hardware_color, linewidth=10)
    # scatter!(ax_hw_head_v1, [hw_head_x[1]], [hw_head_y[1]], color=hardware_color, markersize=20, marker=:circle)
    # scatter!(ax_hw_head_v1, [hw_head_x[end]], [hw_head_y[end]], color=hardware_color, markersize=20, marker=:square)
    # display(fig_hw_head_v1)
    # save(joinpath(output_dir, "$(param_set_name)_hardware_head_trajectory.png"), fig_hw_head_v1)
    #
    # #####################################################################################
    # ## Plot 6: Hardware Head Trajectory - Version 2 (Dashed, Opaque)
    # #####################################################################################
    #
    # println("  Creating Hardware head trajectory (dashed, opaque)...")
    # fig_hw_head_v2, ax_hw_head_v2 = create_aquarium_figure(;
    #     backgroundcolor=background_color, fontsize=fontsize, resolution=(800, 800),
    #     xlabel="X Position (cm)", ylabel="Y Position (cm)",
    #     spinevisible=false, ticksvisible=false, use_data_aspect=true)
    # xlims!(ax_hw_head_v2, 0, length_x)
    # ylims!(ax_hw_head_v2, 0, length_y)
    # lines!(ax_hw_head_v2, hw_head_x, hw_head_y, color=hardware_color_opaque, linewidth=10, linestyle=:dash)
    # display(fig_hw_head_v2)
    # save(joinpath(output_dir, "$(param_set_name)_hardware_head_trajectory_dashed.png"), fig_hw_head_v2)

    # #####################################################################################
    # ## Plot 7: Genesis Head Trajectory - Version 1 (Solid, Non-opaque)
    # #####################################################################################
    #
    # println("  Creating Genesis head trajectory (solid, non-opaque)...")
    # fig_gen_head_v1, ax_gen_head_v1 = create_aquarium_figure(;
    #     backgroundcolor=background_color, fontsize=fontsize, resolution=(800, 800),
    #     xlabel="X Position (cm)", ylabel="Y Position (cm)",
    #     spinevisible=false, ticksvisible=false, use_data_aspect=true)
    # xlims!(ax_gen_head_v1, 0, length_x)
    # ylims!(ax_gen_head_v1, 0, length_y)
    # lines!(ax_gen_head_v1, genesis_head_x, genesis_head_y, color=genesis_color, linewidth=10)
    # scatter!(ax_gen_head_v1, [genesis_head_x[1]], [genesis_head_y[1]], color=genesis_color, markersize=20, marker=:circle)
    # scatter!(ax_gen_head_v1, [genesis_head_x[end]], [genesis_head_y[end]], color=genesis_color, markersize=20, marker=:square)
    # display(fig_gen_head_v1)
    # save(joinpath(output_dir, "$(param_set_name)_genesis_head_trajectory.png"), fig_gen_head_v1)
    #
    # #####################################################################################
    # ## Plot 8: Genesis Head Trajectory - Version 2 (Dashed, Opaque)
    # #####################################################################################
    #
    # println("  Creating Genesis head trajectory (dashed, opaque)...")
    # fig_gen_head_v2, ax_gen_head_v2 = create_aquarium_figure(;
    #     backgroundcolor=background_color, fontsize=fontsize, resolution=(800, 800),
    #     xlabel="X Position (cm)", ylabel="Y Position (cm)",
    #     spinevisible=false, ticksvisible=false, use_data_aspect=true)
    # xlims!(ax_gen_head_v2, 0, length_x)
    # ylims!(ax_gen_head_v2, 0, length_y)
    # lines!(ax_gen_head_v2, genesis_head_x, genesis_head_y, color=genesis_color_opaque, linewidth=10, linestyle=:dash)
    # display(fig_gen_head_v2)
    # save(joinpath(output_dir, "$(param_set_name)_genesis_head_trajectory_dashed.png"), fig_gen_head_v2)

    #####################################################################################
    ## Plot 9: Simulation Head Trajectory - Version 1 (Solid, Non-opaque)
    #####################################################################################

    println("  Creating Simulation head trajectory (solid, non-opaque)...")

    fig_sim_head_v1, ax_sim_head_v1 = create_aquarium_figure(;
        backgroundcolor=background_color,
        fontsize=fontsize,
        resolution=(800, 800),
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        use_data_aspect=true
    )

    # Set axis limits to tank dimensions
    xlims!(ax_sim_head_v1, 0, length_x)
    ylims!(ax_sim_head_v1, 0, length_y)

    # Plot simulation trajectory (solid, non-opaque)
    lines!(ax_sim_head_v1, sim_head_x, sim_head_y, color=simulation_color, linewidth=10)

    # Mark start and end points
    scatter!(ax_sim_head_v1, [sim_head_x[1]], [sim_head_y[1]], color=simulation_color, markersize=20, marker=:circle)
    scatter!(ax_sim_head_v1, [sim_head_x[end]], [sim_head_y[end]], color=simulation_color, markersize=20, marker=:square)

    display(fig_sim_head_v1)

    # Save as PNG
    save(joinpath(output_dir, "$(param_set_name)_simulation_head_trajectory.png"), fig_sim_head_v1)
    println("    ✓ Saved: $(param_set_name)_simulation_head_trajectory.png")

    #####################################################################################
    ## Plot 10: Simulation Head Trajectory - Version 2 (Dashed, Opaque)
    #####################################################################################

    println("  Creating Simulation head trajectory (dashed, opaque)...")

    fig_sim_head_v2, ax_sim_head_v2 = create_aquarium_figure(;
        backgroundcolor=background_color,
        fontsize=fontsize,
        resolution=(800, 800),
        xlabel="X Position (cm)",
        ylabel="Y Position (cm)",
        spinevisible=false,
        ticksvisible=false,
        use_data_aspect=true
    )

    # Set axis limits to tank dimensions
    xlims!(ax_sim_head_v2, 0, length_x)
    ylims!(ax_sim_head_v2, 0, length_y)

    # Plot simulation trajectory (dashed, opaque)
    lines!(ax_sim_head_v2, sim_head_x, sim_head_y, color=simulation_color_opaque, linewidth=10, linestyle=:dash)

    display(fig_sim_head_v2)

    # Save as PNG
    save(joinpath(output_dir, "$(param_set_name)_simulation_head_trajectory_dashed.png"), fig_sim_head_v2)
    println("    ✓ Saved: $(param_set_name)_simulation_head_trajectory_dashed.png")

    #####################################################################################
    ## Create Time-Specific Trajectory Visualizations
    #####################################################################################

    println()
    println("  Creating time-specific trajectory visualizations for $param_set_name...")

    # Time points for trajectory snapshots (same as video frame extraction)
    time_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for t in time_points
        # Find the frame index closest to this time
        frame_idx = argmin(abs.(time_traj_sim .- t))
        actual_time = time_traj_sim[frame_idx]

        println("    Creating trajectory visualizations for t=$(t)s (frame $frame_idx, actual t=$(round(actual_time, digits=3))s)")

        # # Hardware head trajectory up to this time (solid, non-opaque)
        # fig_hw_head_t, ax_hw_head_t = create_aquarium_figure(;
        #     backgroundcolor=background_color, fontsize=fontsize, resolution=(800, 800),
        #     xlabel="X Position (cm)", ylabel="Y Position (cm)",
        #     spinevisible=false, ticksvisible=false, use_data_aspect=true)
        # xlims!(ax_hw_head_t, 0, length_x)
        # ylims!(ax_hw_head_t, 0, length_y)
        # lines!(ax_hw_head_t, hw_head_x[1:frame_idx], hw_head_y[1:frame_idx], color=hardware_color, linewidth=10)
        # scatter!(ax_hw_head_t, [hw_head_x[1]], [hw_head_y[1]], color=hardware_color, markersize=20, marker=:circle)
        # scatter!(ax_hw_head_t, [hw_head_x[frame_idx]], [hw_head_y[frame_idx]], color=hardware_color, markersize=20, marker=:square)
        # save(joinpath(output_dir, "$(param_set_name)_hardware_head_traj_t$(replace(string(t), "." => "p"))s.png"), fig_hw_head_t)
        #
        # # Hardware head trajectory (dashed, opaque)
        # fig_hw_head_t_dashed, ax_hw_head_t_dashed = create_aquarium_figure(;
        #     backgroundcolor=background_color, fontsize=fontsize, resolution=(800, 800),
        #     xlabel="X Position (cm)", ylabel="Y Position (cm)",
        #     spinevisible=false, ticksvisible=false, use_data_aspect=true)
        # xlims!(ax_hw_head_t_dashed, 0, length_x)
        # ylims!(ax_hw_head_t_dashed, 0, length_y)
        # lines!(ax_hw_head_t_dashed, hw_head_x[1:frame_idx], hw_head_y[1:frame_idx],
        #        color=hardware_color_opaque, linewidth=10, linestyle=:dash)
        # save(joinpath(output_dir, "$(param_set_name)_hardware_head_traj_t$(replace(string(t), "." => "p"))s_dashed.png"), fig_hw_head_t_dashed)

        # # Genesis head trajectory up to this time (solid, non-opaque)
        # fig_genesis_head_t, ax_genesis_head_t = create_aquarium_figure(;
        #     backgroundcolor=background_color, fontsize=fontsize, resolution=(800, 800),
        #     xlabel="X Position (cm)", ylabel="Y Position (cm)",
        #     spinevisible=false, ticksvisible=false, use_data_aspect=true)
        # xlims!(ax_genesis_head_t, 0, length_x)
        # ylims!(ax_genesis_head_t, 0, length_y)
        # lines!(ax_genesis_head_t, genesis_head_x[1:frame_idx], genesis_head_y[1:frame_idx], color=genesis_color, linewidth=10)
        # scatter!(ax_genesis_head_t, [genesis_head_x[1]], [genesis_head_y[1]], color=genesis_color, markersize=20, marker=:circle)
        # scatter!(ax_genesis_head_t, [genesis_head_x[frame_idx]], [genesis_head_y[frame_idx]], color=genesis_color, markersize=20, marker=:square)
        # save(joinpath(output_dir, "$(param_set_name)_genesis_head_traj_t$(replace(string(t), "." => "p"))s.png"), fig_genesis_head_t)
        #
        # # Genesis head trajectory (dashed, opaque)
        # fig_genesis_head_t_dashed, ax_genesis_head_t_dashed = create_aquarium_figure(;
        #     backgroundcolor=background_color, fontsize=fontsize, resolution=(800, 800),
        #     xlabel="X Position (cm)", ylabel="Y Position (cm)",
        #     spinevisible=false, ticksvisible=false, use_data_aspect=true)
        # xlims!(ax_genesis_head_t_dashed, 0, length_x)
        # ylims!(ax_genesis_head_t_dashed, 0, length_y)
        # lines!(ax_genesis_head_t_dashed, genesis_head_x[1:frame_idx], genesis_head_y[1:frame_idx],
        #        color=genesis_color_opaque, linewidth=10, linestyle=:dash)
        # save(joinpath(output_dir, "$(param_set_name)_genesis_head_traj_t$(replace(string(t), "." => "p"))s_dashed.png"), fig_genesis_head_t_dashed)

        # Simulation head trajectory up to this time (solid, non-opaque)
        fig_sim_head_t, ax_sim_head_t = create_aquarium_figure(;
            backgroundcolor=background_color,
            fontsize=fontsize,
            resolution=(800, 800),
            xlabel="X Position (cm)",
            ylabel="Y Position (cm)",
            spinevisible=false,
            ticksvisible=false,
            use_data_aspect=true
        )

        xlims!(ax_sim_head_t, 0, length_x)
        ylims!(ax_sim_head_t, 0, length_y)

        lines!(ax_sim_head_t, sim_head_x[1:frame_idx], sim_head_y[1:frame_idx], color=simulation_color, linewidth=10)
        scatter!(ax_sim_head_t, [sim_head_x[1]], [sim_head_y[1]], color=simulation_color, markersize=20, marker=:circle)
        scatter!(ax_sim_head_t, [sim_head_x[frame_idx]], [sim_head_y[frame_idx]], color=simulation_color, markersize=20, marker=:square)

        save(joinpath(output_dir, "$(param_set_name)_simulation_head_traj_t$(replace(string(t), "." => "p"))s.png"), fig_sim_head_t)

        # Simulation head trajectory (dashed, opaque)
        fig_sim_head_t_dashed, ax_sim_head_t_dashed = create_aquarium_figure(;
            backgroundcolor=background_color,
            fontsize=fontsize,
            resolution=(800, 800),
            xlabel="X Position (cm)",
            ylabel="Y Position (cm)",
            spinevisible=false,
            ticksvisible=false,
            use_data_aspect=true
        )

        xlims!(ax_sim_head_t_dashed, 0, length_x)
        ylims!(ax_sim_head_t_dashed, 0, length_y)

        lines!(ax_sim_head_t_dashed, sim_head_x[1:frame_idx], sim_head_y[1:frame_idx],
               color=simulation_color_opaque, linewidth=10, linestyle=:dash)

        save(joinpath(output_dir, "$(param_set_name)_simulation_head_traj_t$(replace(string(t), "." => "p"))s_dashed.png"), fig_sim_head_t_dashed)

        println("      ✓ Saved trajectory visualizations for t=$(t)s")
    end

    println("    ✓ Time-specific trajectory visualizations complete")

    # #####################################################################################
    # ## Extract Hardware Video Frames at Key Time Points
    # #####################################################################################
    #
    # println()
    # println("  Extracting hardware video frames for $param_set_name...")
    # trial_name = nothing
    # for (t_name, p_name) in trial_to_param_map
    #     if p_name == param_set_name
    #         trial_name = t_name
    #         break
    #     end
    # end
    # if trial_name === nothing
    #     println("    ⚠ Warning: No trial name mapping found for $param_set_name")
    # else
    #     hardware_video_path = joinpath(data_dir, "$(trial_name).mov")
    #     if isfile(hardware_video_path)
    #         video = VideoIO.openvideo(hardware_video_path)
    #         video_fps = VideoIO.framerate(video)
    #         total_frames = VideoIO.counttotalframes(video)
    #         hw_trial_data = hw_data_df[hw_data_df.video .== trial_name, :]
    #         sort!(hw_trial_data, :frame)
    #         motion_start_frame = Int(hw_trial_data.frame[1])
    #         time_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    #         for t in time_points
    #             frame_number = Int(round(t * video_fps)) + motion_start_frame
    #             if param_set_name == "optimal"
    #                 VideoIO.seek(video, (frame_number - 5) / video_fps)
    #             else
    #                 VideoIO.seek(video, (frame_number - 1) / video_fps)
    #             end
    #             img = read(video)
    #             if ndims(img) == 3 && size(img, 3) == 3
    #                 img_rgb = colorview(RGB, permutedims(img, (3, 1, 2)) ./ 255.0)
    #             else
    #                 img_rgb = img
    #             end
    #             frame_filename = joinpath(output_dir, "$(param_set_name)_hardware_video_frame_t$(replace(string(t), "." => "p"))s.png")
    #             save(frame_filename, img_rgb)
    #         end
    #         close(video)
    #     else
    #         println("    ⚠ Warning: Hardware video not found: $hardware_video_path")
    #     end
    # end

    # #####################################################################################
    # ## Extract Genesis Video Frames at Key Time Points
    # #####################################################################################
    #
    # println("  Extracting genesis video frames for $param_set_name...")
    # genesis_trial_name = nothing
    # for (t_name, p_name) in trial_to_param_map_genesis
    #     if p_name == param_set_name
    #         genesis_trial_name = t_name
    #         break
    #     end
    # end
    # if genesis_trial_name === nothing
    #     println("    ⚠ Warning: No genesis trial name mapping found for $param_set_name")
    # else
    #     genesis_video_path = joinpath(data_dir, "$(genesis_trial_name).mp4")
    #     if isfile(genesis_video_path)
    #         genesis_video = VideoIO.openvideo(genesis_video_path)
    #         genesis_video_fps = VideoIO.framerate(genesis_video)
    #         genesis_trial_data = genesis_data_df[genesis_data_df.video .== genesis_trial_name, :]
    #         sort!(genesis_trial_data, :frame)
    #         genesis_motion_start_frame = Int(genesis_trial_data.frame[1])
    #         time_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    #         for t in time_points
    #             frame_number = Int(round(t * genesis_video_fps)) + genesis_motion_start_frame
    #             VideoIO.seek(genesis_video, (frame_number - 1) / genesis_video_fps)
    #             img = read(genesis_video)
    #             if ndims(img) == 3 && size(img, 3) == 3
    #                 img_rgb = colorview(RGB, permutedims(img, (3, 1, 2)) ./ 255.0)
    #             else
    #                 img_rgb = img
    #             end
    #             frame_filename = joinpath(output_dir, "$(param_set_name)_genesis_video_frame_t$(replace(string(t), "." => "p"))s.png")
    #             save(frame_filename, img_rgb)
    #         end
    #         close(genesis_video)
    #     end
    # end

    ######################################################################################
    ## Create Vorticity Field Visualizations at Key Time Points
    #####################################################################################

    println()
    println("  Creating vorticity field visualizations for $param_set_name...")

    # Get trajectories for this parameter set
    trajectories = sim_data_by_param[param_set_name]["trajectories"]
    aquarium_state_traj = trajectories[:aquarium_state_traj]
    swimmer_state_traj = trajectories[:swimmer_state_traj]
    time_traj = trajectories[:time_traj]

    # Extract fluid velocity trajectory from aquarium state
    fluid_velocity_traj = [extract_fluid_velocity(tank, aquarium_state_traj[k]) for k in 1:length(aquarium_state_traj)]

    # Time points for vorticity visualization
    time_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for t in time_points
        # Find closest frame in simulation
        frame_idx = argmin(abs.(time_traj .- t))
        actual_t = time_traj[frame_idx]

        println("    Creating vorticity field for t=$(t)s (frame $frame_idx, actual t=$(round(actual_t, digits=2))s)")

        # Create figure with white background
        fig_vort, ax_vort = create_aquarium_figure(;
            backgroundcolor=:white,
            fontsize=fontsize,
            xlabel="X (cm)",
            ylabel="Y (cm)",
            xlim=(0.0, length_x),
            ylim=(0.0, length_y),
            resolution=(800, 800),
            spinevisible=false,
            ticksvisible=false,
            use_data_aspect=true
        )

        if t == 0.0
            # Add swimmer outline only for t=0s
            plot_solid_systems!(fig_vort, ax_vort,
                [rexeel],
                [swimmer_state_traj[frame_idx]]
            )
        else
            # Plot vorticity field with red-blue colormap
            plot_vorticity_field!(fig_vort, ax_vort,
                fluid_env,
                nothing, rexeel,
                fluid_velocity_traj[frame_idx],
                [], swimmer_state_traj[frame_idx];
                colormap=:oslo,
                density=100,
                threshold_percentage=1.0,
                smooth=true,
                smooth_sigma=4.0
            )
        end

        display(fig_vort)

        # Save figure
        vorticity_filename = joinpath(output_dir, "$(param_set_name)_vorticity_field_t$(replace(string(t), "." => "p"))s.png")
        save(vorticity_filename, fig_vort)

        println("      ✓ Saved: $(param_set_name)_vorticity_field_t$(replace(string(t), "." => "p"))s.png")
    end

    println("    ✓ Vorticity field visualization complete")

    ######################################################################################
    ## Animate Vorticity Field
    #####################################################################################

    println()
    println("  Creating vorticity field animation for $param_set_name...")

    fig_anim, ax_anim = create_aquarium_figure(;
        backgroundcolor=:white,
        fontsize=fontsize,
        xlabel="X (cm)",
        ylabel="Y (cm)",
        xlim=(0.0, length_x),
        ylim=(0.0, length_y),
        resolution=(800, 800),
        spinevisible=false,
        ticksvisible=false,
        use_data_aspect=true
    )

    anim_save_path = joinpath(output_dir, "$(param_set_name)_vorticity_animation.mp4")
    animate_vorticity_field(fig_anim, ax_anim,
        fluid_env,
        nothing, rexeel,
        time_traj,
        fluid_velocity_traj,
        [[]], swimmer_state_traj,
        anim_save_path;
        colormap=:oslo,
        density=100,
        framerate=20,
        timescale=1.0,
        threshold_percentage=1.0,
        smooth=true,
        smooth_sigma=4.0
    )

    println("    ✓ Saved: $(param_set_name)_vorticity_animation.mp4")

end

#############################################################################################
## Summary
#############################################################################################

println("="^80)
println("C-START VISUALIZATION COMPLETE")
println("="^80)
println("Output directory: $output_dir")
println("Generated plots for each parameter set (simulation only):")
for param_set_name in parameter_set_names
    if haskey(sim_head_traj_by_param, param_set_name)
        println("  $param_set_name:")
        println("    Simulation trajectories:")
        println("      - $(param_set_name)_simulation_head_trajectory.png (solid)")
        println("      - $(param_set_name)_simulation_head_trajectory_dashed.png")
        println("    Vorticity fields (t=0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0s):")
        println("      - $(param_set_name)_vorticity_field_t*.png (7 fields)")
    end
end
println()
println("Note: Hardware and genesis comparison plots are commented out.")
println("      Uncomment the relevant sections to generate them when data is available.")
println("="^80)

#############################################################################################
## Trajectory Metrics Summary
#############################################################################################

println()
println()
println("="^80)
println("TRAJECTORY METRICS SUMMARY")
println("="^80)
println()

# Print trajectory metrics in a nice table
for param_set_name in parameter_set_names
    if !haskey(trajectory_metrics, param_set_name)
        continue
    end

    println("Parameter Set: $param_set_name")
    println("-"^80)
    println(@sprintf("%-15s │ %12s │ %12s │ %12s",
        "Source", "Final Vel", "Heading", "Net Disp."))
    println(@sprintf("%-15s │ %12s │ %12s │ %12s",
        "", "(cm/s)", "(deg)", "(cm)"))
    println("-"^80)

    metrics = trajectory_metrics[param_set_name]

    for (source_key, source_name) in [("simulation", "Simulation")]
        if haskey(metrics, source_key)
            m = metrics[source_key]
            println(@sprintf("%-15s │ %12.3f │ %12.2f │ %12.3f",
                source_name,
                m["final_velocity_magnitude"],
                m["final_heading_deg"],
                m["net_displacement_in_heading"]))
        end
    end
    println()
end

println("="^80)