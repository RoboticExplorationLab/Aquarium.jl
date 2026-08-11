import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using AquariumClosed
using AquariumClosed.CairoMakie
using JLD2
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using Interpolations
using Printf
using Colors
using PGFPlotsX

println("="^80)
println("RExEel Forward Swimming: Sim-to-Real Trajectory Error Analysis")
println("="^80)
println()

# Create output directory
output_dir = expanduser("~/aquariumCLOSED/visualization/rss_figures/")
mkpath(output_dir)

# Open log file
log_path = joinpath(output_dir, "sim_to_real_comparison_results.txt")
log_file = open(log_path, "w")

println("Log file created: $log_path")
println()

# Amplitude cases to analyze
amplitude_cases = ["10deg", "20deg", "30deg", "40deg"]

# Store results for each case
results = Dict{String, Dict{String, Any}}()

#############################################################################################
## Define fluid environment (same for all cases)
#############################################################################################

# time properties
time_step = 1/60
final_time = 4.0
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

fluid_env = Fluid(
    time_step;
    density = fluid_density,
    dynamic_viscosity = dynamic_viscosity,
    boundary_velocity = [0.0, 0.0],
    grid_size = (num_cells_x, num_cells_y),
    grid_dimensions = (length_x, length_y),
    boundary_condition_type = boundary_condition_type,
)

#############################################################################################
## Define 6-link RExEel (same for all cases)
#############################################################################################

n_links = 6
link_lengths = [12.0, 9.8 .* ones(n_links-1)...]  # cm
height = 9.35  # cm
masses_per_link = [192, 140 .* ones(n_links-1)...] ./ height # g per link
moi_per_link = [2435.99, 1483.49 .* ones(n_links-1)...] ./ height  # g·cm²
gravity_constant = 0.0

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

total_mass = sum(masses_per_link)

#############################################################################################
## Loop over all amplitude cases
#############################################################################################

for amplitude in amplitude_cases
    println("\n" * "="^80)
    println("Processing: $amplitude")
    println("="^80)
    
    # Data paths
    hardware_tracking_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/$amplitude/$(amplitude)_hardware_trajectories.csv")
    simulation_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/$amplitude/$(amplitude)_simulation.jld2")
    motor_angles_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/$amplitude/$(amplitude)_hardware_motor_angles.csv")
    genesis_tracking_path = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/genesis_simulation_trajectories.csv")
    
    println("\nLoading data...")
    
    #############################################################################################
    ## Load Data
    #############################################################################################
    
    # Load hardware tracking data
    hw_data_df = CSV.read(hardware_tracking_path, DataFrame)
    
    # Organize trajectories and timestamps by video name
    hw_trajectories_by_video = Dict{String, Vector{Vector{Float64}}}()
    hw_timestamps_by_video = Dict{String, Vector{Float64}}()
    
    for video_name in unique(hw_data_df.video)
        video_data = hw_data_df[hw_data_df.video .== video_name, :]
        sort!(video_data, :frame)
        
        traj = [[row.x_cm, row.y_cm, deg2rad(row.robot_yaw)] for row in eachrow(video_data)]
        timestamps = range(0.0, (length(traj)-1)/30.0, length=length(traj))
        
        hw_trajectories_by_video[video_name] = traj
        hw_timestamps_by_video[video_name] = timestamps
    end
    
    println("  ✓ Hardware tracking loaded: $(length(hw_trajectories_by_video)) videos")
    
    # Load genesis simulation tracking data for all amplitudes
    genesis_data_df = CSV.read(genesis_tracking_path, DataFrame)
    
    # Map amplitude to Genesis video name
    amplitude_to_genesis = Dict(
        "10deg" => "Simulation_K10",
        "20deg" => "Simulation_K20",
        "30deg" => "Simulation_K30",
        "40deg" => "Simulation_K40"
    )
    genesis_video_name = amplitude_to_genesis[amplitude]
    
    # Filter for the appropriate amplitude
    genesis_data = genesis_data_df[genesis_data_df.video .== genesis_video_name, :]
    sort!(genesis_data, :frame)
    
    # Extract trajectory as vector of [x, y, theta] vectors
    genesis_traj = [[row.x_cm, row.y_cm, deg2rad(row.robot_yaw)] for row in eachrow(genesis_data)]
    
    # Create timestamps (60fps simulation)
    genesis_timestamps = range(0.0, (length(genesis_traj)-1)/60.0, length=length(genesis_traj))
    
    println("  ✓ Genesis simulation ($genesis_video_name) loaded: $(length(genesis_traj)) frames")
    
    # Load simulation data
    sim_data = load(simulation_path)
    trajectories = sim_data["trajectories"]
    time_traj_sim = trajectories[:time_traj]
    swimmer_state_traj = trajectories[:swimmer_state_traj]
    motor_angle_sim_traj = trajectories[:control_traj]
    
    println("  ✓ Simulation loaded: $(length(swimmer_state_traj)) timesteps")
    
    # Load hardware motor angles
    motor_angles_df = CSV.read(motor_angles_path, DataFrame; comment="#")
    println("  ✓ Hardware motor angles loaded: $(size(motor_angles_df, 1)) rows")
    
    # Extract and interpolate motor angles
    time_traj_motors = range(0.0, motor_angles_df.time_s[end-1], length=size(motor_angles_df, 1))
    actual_motor_angles = zeros(length(time_traj_motors), 5)
    for i in 1:5
        actual_motor_angles[:, i] = deg2rad.(motor_angles_df[:, Symbol("actual_$i")])
    end
    
    actual_motor_angles_interp = zeros(length(time_traj_sim), 5)
    for joint in 1:5
        itp = CubicSplineInterpolation(time_traj_motors, actual_motor_angles[:, joint], extrapolation_bc=Line())
        for i in 1:length(time_traj_sim)
            actual_motor_angles_interp[i, joint] = itp(time_traj_sim[i])
        end
    end
    
    #############################################################################################
    ## Interpolate hardware trajectories to simulation time
    #############################################################################################
    
    hw_trajectories_interp_by_video = Dict{String, Vector{Vector{Float64}}}()
    
    for (video_name, traj) in hw_trajectories_by_video
        hw_timestamps = hw_timestamps_by_video[video_name]
        hw_timestamps = range(0.0, hw_timestamps[end-1], length=length(hw_timestamps))
        
        x_traj = [traj[i][1] for i in 1:length(traj)]
        y_traj = [traj[i][2] for i in 1:length(traj)]
        theta_traj = [traj[i][3] for i in 1:length(traj)]
        
        itp_x = CubicSplineInterpolation(hw_timestamps, x_traj, extrapolation_bc=Line())
        itp_y = CubicSplineInterpolation(hw_timestamps, y_traj, extrapolation_bc=Line())
        itp_theta = CubicSplineInterpolation(hw_timestamps, theta_traj, extrapolation_bc=Line())
        
        traj_interp = []
        for t_sim in time_traj_sim
            push!(traj_interp, [itp_x(t_sim), itp_y(t_sim), itp_theta(t_sim)])
        end
        
        hw_trajectories_interp_by_video[video_name] = traj_interp
    end
    
    # Interpolate genesis trajectory to simulation time
    genesis_traj_interp = []
    
    x_traj_genesis = [genesis_traj[i][1] for i in 1:length(genesis_traj)]
    y_traj_genesis = [genesis_traj[i][2] for i in 1:length(genesis_traj)]
    theta_traj_genesis = [genesis_traj[i][3] for i in 1:length(genesis_traj)]
    
    itp_x_genesis = CubicSplineInterpolation(genesis_timestamps, x_traj_genesis, extrapolation_bc=Line())
    itp_y_genesis = CubicSplineInterpolation(genesis_timestamps, y_traj_genesis, extrapolation_bc=Line())
    itp_theta_genesis = CubicSplineInterpolation(genesis_timestamps, theta_traj_genesis, extrapolation_bc=Line())
    
    for t_sim in time_traj_sim
        push!(genesis_traj_interp, [itp_x_genesis(t_sim), itp_y_genesis(t_sim), itp_theta_genesis(t_sim)])
    end
    
    println("  ✓ Genesis trajectory interpolated to $(length(genesis_traj_interp)) simulation timesteps")
    
    #############################################################################################
    ## Extract simulation maximal configuration trajectory
    #############################################################################################
    
    sim_maximal_config_traj = [swimmer_state_traj[i][rexeel.configuration_indices] for i in 1:length(swimmer_state_traj)]
    
    #############################################################################################
    ## Reconstruct hardware maximal configurations
    #############################################################################################
    
    hw_maximal_config_traj_by_video = Dict{String, Vector{Vector{Float64}}}()
    
    for (video_name, traj_interp) in hw_trajectories_interp_by_video
        hw_maximal_config_traj = []
        
        for i in 1:length(traj_interp)
            x_head = traj_interp[i][1]
            y_head = traj_interp[i][2]
            θ_head = -traj_interp[i][3]
            
            n_minimal_coords = n_links + 2
            minimal_config = zeros(n_minimal_coords)
            minimal_config[1] = x_head
            minimal_config[2] = y_head 
            minimal_config[3] = θ_head
            minimal_config[4:end] = actual_motor_angles_interp[i, :]
            
            maximal_config = rex_eel_maximal_from_minimal(rexeel, minimal_config, n_links)
            push!(hw_maximal_config_traj, maximal_config)
        end
        
        hw_maximal_config_traj_by_video[video_name] = hw_maximal_config_traj
    end
    
    # Reconstruct genesis maximal configurations
    genesis_maximal_config_traj = []
    
    for i in 1:length(genesis_traj_interp)
        x_head = genesis_traj_interp[i][1]
        y_head = genesis_traj_interp[i][2]
        θ_head = -genesis_traj_interp[i][3]
        
        n_minimal_coords = n_links + 2
        minimal_config = zeros(n_minimal_coords)
        minimal_config[1] = x_head
        minimal_config[2] = y_head 
        minimal_config[3] = θ_head
        minimal_config[4:end] = actual_motor_angles_interp[i, :]
        
        maximal_config = rex_eel_maximal_from_minimal(rexeel, minimal_config, n_links)
        push!(genesis_maximal_config_traj, maximal_config)
    end
    
    println("  ✓ Genesis maximal configurations reconstructed: $(length(genesis_maximal_config_traj)) timesteps")
    
    #############################################################################################
    ## Compute COM trajectories for all videos
    #############################################################################################
    
    # Simulation COM
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
    
    # Hardware COM for all videos
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
    
    # Genesis COM
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
    
    println("  ✓ Genesis COM trajectory computed: $(length(genesis_com_traj)) timesteps")
    
    #############################################################################################
    ## Compute errors for each video
    #############################################################################################

    video_errors = Dict{String, Dict{String, Any}}()

    for (video_name, hw_com_traj) in hw_com_traj_by_video
        N = length(sim_com_traj)

        # Unbias COM trajectories w.r.t. initial conditions
        hw_com_initial = hw_com_traj[1]
        sim_com_initial = sim_com_traj[1]

        hw_com_unbiased = [[hw_com_traj[i][1] - hw_com_initial[1], hw_com_traj[i][2] - hw_com_initial[2]] for i in 1:N]
        sim_com_unbiased = [[sim_com_traj[i][1] - sim_com_initial[1], sim_com_traj[i][2] - sim_com_initial[2]] for i in 1:N]

        # COM errors
        com_x_errors = [hw_com_unbiased[i][1] - sim_com_unbiased[i][1] for i in 1:N]
        com_y_errors = [hw_com_unbiased[i][2] - sim_com_unbiased[i][2] for i in 1:N]
        com_position_errors = [sqrt(com_x_errors[i]^2 + com_y_errors[i]^2) for i in 1:N]

        # Head errors
        hw_maximal_config_traj = hw_maximal_config_traj_by_video[video_name]

        hw_head_x_vals = [hw_maximal_config_traj[i][1] for i in 1:N]
        hw_head_y_vals = [hw_maximal_config_traj[i][2] for i in 1:N]
        hw_head_theta_vals = [hw_maximal_config_traj[i][3] for i in 1:N]
        sim_head_x_vals = [sim_maximal_config_traj[i][1] for i in 1:N]
        sim_head_y_vals = [sim_maximal_config_traj[i][2] for i in 1:N]
        sim_head_theta_vals = [sim_maximal_config_traj[i][3] for i in 1:N]

        hw_head_initial_x = hw_head_x_vals[1]
        hw_head_initial_y = hw_head_y_vals[1]
        hw_head_initial_theta = hw_head_theta_vals[1]
        sim_head_initial_x = sim_head_x_vals[1]
        sim_head_initial_y = sim_head_y_vals[1]
        sim_head_initial_theta = sim_head_theta_vals[1]

        hw_head_x_unbiased = [hw_head_x_vals[i] - hw_head_initial_x for i in 1:N]
        hw_head_y_unbiased = [hw_head_y_vals[i] - hw_head_initial_y for i in 1:N]
        hw_head_theta_unbiased = [hw_head_theta_vals[i] - hw_head_initial_theta for i in 1:N]
        sim_head_x_unbiased = [sim_head_x_vals[i] - sim_head_initial_x for i in 1:N]
        sim_head_y_unbiased = [sim_head_y_vals[i] - sim_head_initial_y for i in 1:N]
        sim_head_theta_unbiased = [sim_head_theta_vals[i] - sim_head_initial_theta for i in 1:N]

        head_x_errors = [hw_head_x_unbiased[i] - sim_head_x_unbiased[i] for i in 1:N]
        head_y_errors = [hw_head_y_unbiased[i] - sim_head_y_unbiased[i] for i in 1:N]
        head_theta_errors = [hw_head_theta_unbiased[i] - sim_head_theta_unbiased[i] for i in 1:N]
        head_position_errors = [sqrt(head_x_errors[i]^2 + head_y_errors[i]^2) for i in 1:N]

        # Store errors for this video
        video_errors[video_name] = Dict(
            "com_x_errors" => com_x_errors,
            "com_y_errors" => com_y_errors,
            "com_position_errors" => com_position_errors,
            "head_x_errors" => head_x_errors,
            "head_y_errors" => head_y_errors,
            "head_theta_errors" => head_theta_errors,
            "head_position_errors" => head_position_errors
        )
    end
    
    #############################################################################################
    ## Compute genesis errors w.r.t. hardware
    #############################################################################################

    N = length(sim_com_traj)

    # Compute genesis errors against each hardware video
    genesis_video_errors = Dict{String, Dict{String, Any}}()

        for (video_name, hw_com_traj) in hw_com_traj_by_video
            # Unbias COM trajectories w.r.t. initial conditions
            genesis_com_initial = genesis_com_traj[1]
            hw_com_initial = hw_com_traj[1]

            genesis_com_unbiased = [[genesis_com_traj[i][1] - genesis_com_initial[1], genesis_com_traj[i][2] - genesis_com_initial[2]] for i in 1:N]
            hw_com_unbiased = [[hw_com_traj[i][1] - hw_com_initial[1], hw_com_traj[i][2] - hw_com_initial[2]] for i in 1:N]

            # Genesis COM errors vs hardware
            com_x_errors = [genesis_com_unbiased[i][1] - hw_com_unbiased[i][1] for i in 1:N]
            com_y_errors = [genesis_com_unbiased[i][2] - hw_com_unbiased[i][2] for i in 1:N]
            com_position_errors = [sqrt(com_x_errors[i]^2 + com_y_errors[i]^2) for i in 1:N]

            # Genesis head errors vs hardware
            hw_maximal_config_traj = hw_maximal_config_traj_by_video[video_name]

            genesis_head_x_vals = [genesis_maximal_config_traj[i][1] for i in 1:N]
            genesis_head_y_vals = [genesis_maximal_config_traj[i][2] for i in 1:N]
            genesis_head_theta_vals = [genesis_maximal_config_traj[i][3] for i in 1:N]
            hw_head_x_vals = [hw_maximal_config_traj[i][1] for i in 1:N]
            hw_head_y_vals = [hw_maximal_config_traj[i][2] for i in 1:N]
            hw_head_theta_vals = [hw_maximal_config_traj[i][3] for i in 1:N]

            genesis_head_initial_x = genesis_head_x_vals[1]
            genesis_head_initial_y = genesis_head_y_vals[1]
            genesis_head_initial_theta = genesis_head_theta_vals[1]
            hw_head_initial_x = hw_head_x_vals[1]
            hw_head_initial_y = hw_head_y_vals[1]
            hw_head_initial_theta = hw_head_theta_vals[1]

            genesis_head_x_unbiased = [genesis_head_x_vals[i] - genesis_head_initial_x for i in 1:N]
            genesis_head_y_unbiased = [genesis_head_y_vals[i] - genesis_head_initial_y for i in 1:N]
            genesis_head_theta_unbiased = [genesis_head_theta_vals[i] - genesis_head_initial_theta for i in 1:N]
            hw_head_x_unbiased = [hw_head_x_vals[i] - hw_head_initial_x for i in 1:N]
            hw_head_y_unbiased = [hw_head_y_vals[i] - hw_head_initial_y for i in 1:N]
            hw_head_theta_unbiased = [hw_head_theta_vals[i] - hw_head_initial_theta for i in 1:N]

            head_x_errors = [genesis_head_x_unbiased[i] - hw_head_x_unbiased[i] for i in 1:N]
            head_y_errors = [genesis_head_y_unbiased[i] - hw_head_y_unbiased[i] for i in 1:N]
            head_theta_errors = [genesis_head_theta_unbiased[i] - hw_head_theta_unbiased[i] for i in 1:N]
            head_position_errors = [sqrt(head_x_errors[i]^2 + head_y_errors[i]^2) for i in 1:N]

            # Store errors for this video
            genesis_video_errors[video_name] = Dict(
                "com_x_errors" => com_x_errors,
                "com_y_errors" => com_y_errors,
                "com_position_errors" => com_position_errors,
                "head_x_errors" => head_x_errors,
                "head_y_errors" => head_y_errors,
                "head_theta_errors" => head_theta_errors,
                "head_position_errors" => head_position_errors
            )
        end

        println("  ✓ Genesis vs. hardware errors computed for $(length(genesis_video_errors)) videos")
    
    #############################################################################################
    ## Identify and filter out outlier videos
    #############################################################################################
    
    println("\n  Analyzing errors by video to identify outliers...")
    
    # Compute per-video statistics
    video_stats = Dict{String, Dict{String, Float64}}()
    for (video_name, errors) in video_errors
        video_stats[video_name] = Dict(
            "com_rmse" => sqrt(mean(errors["com_position_errors"].^2)),
            "head_rmse" => sqrt(mean(errors["head_position_errors"].^2))
        )
    end
    
    # Print per-video statistics
    println("\n  Per-video error statistics:")
    println("  " * "-"^80)
    println(@sprintf("  %-35s │ COM RMSE │ Head RMSE", "Video Name"))
    println("  " * "-"^80)
    for (video_name, stats) in sort(collect(video_stats), by=x->x[2]["com_rmse"], rev=true)
        println(@sprintf("  %-35s │  %6.3f  │   %6.3f", 
            video_name, stats["com_rmse"], stats["head_rmse"]))
    end
    println("  " * "-"^80)
    
    # Identify outlier using IQR method on COM RMSE
    com_rmse_values = [stats["com_rmse"] for stats in values(video_stats)]
    Q1 = quantile(com_rmse_values, 0.25)
    Q3 = quantile(com_rmse_values, 0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 2.5* IQR
    
    outlier_videos = [video_name for (video_name, stats) in video_stats if stats["com_rmse"] > outlier_threshold]
    
    if !isempty(outlier_videos)
        println("\n  ⚠ OUTLIERS DETECTED (COM RMSE > $(round(outlier_threshold, digits=3)) cm):")
        for outlier_video in outlier_videos
            println("    - $outlier_video: COM RMSE = $(round(video_stats[outlier_video]["com_rmse"], digits=3)) cm")
        end
        println("  Note: Outliers are displayed but NOT filtered from statistics.")
    else
        println("\n  ✓ No outliers detected.")
    end
    
    #############################################################################################
    ## Aggregate statistics across all videos
    #############################################################################################

    # Collect all errors from all videos (no filtering)
    all_com_x_errors = vcat([video_errors[v]["com_x_errors"] for v in keys(video_errors)]...)
    all_com_y_errors = vcat([video_errors[v]["com_y_errors"] for v in keys(video_errors)]...)
    all_com_position_errors = vcat([video_errors[v]["com_position_errors"] for v in keys(video_errors)]...)
    all_head_x_errors = vcat([video_errors[v]["head_x_errors"] for v in keys(video_errors)]...)
    all_head_y_errors = vcat([video_errors[v]["head_y_errors"] for v in keys(video_errors)]...)
    all_head_theta_errors = vcat([video_errors[v]["head_theta_errors"] for v in keys(video_errors)]...)
    all_head_position_errors = vcat([video_errors[v]["head_position_errors"] for v in keys(video_errors)]...)

    # Compute per-video RMSE values for cross-trial statistics
    per_video_com_x_rmse = [sqrt(mean(video_errors[v]["com_x_errors"].^2)) for v in keys(video_errors)]
    per_video_com_y_rmse = [sqrt(mean(video_errors[v]["com_y_errors"].^2)) for v in keys(video_errors)]
    per_video_com_position_rmse = [sqrt(mean(video_errors[v]["com_position_errors"].^2)) for v in keys(video_errors)]
    per_video_head_x_rmse = [sqrt(mean(video_errors[v]["head_x_errors"].^2)) for v in keys(video_errors)]
    per_video_head_y_rmse = [sqrt(mean(video_errors[v]["head_y_errors"].^2)) for v in keys(video_errors)]
    per_video_head_theta_rmse = [sqrt(mean(video_errors[v]["head_theta_errors"].^2)) for v in keys(video_errors)]
    per_video_head_position_rmse = [sqrt(mean(video_errors[v]["head_position_errors"].^2)) for v in keys(video_errors)]

    # Compute statistics
    com_stats = Dict(
        "x_rmse" => sqrt(mean(all_com_x_errors.^2)),
        "x_mean" => mean(abs.(all_com_x_errors)),
        "x_median" => median(abs.(all_com_x_errors)),
        "x_std" => std(per_video_com_x_rmse),  # Std of RMSE across videos
        "x_95th" => quantile(abs.(all_com_x_errors), 0.95),
        "y_rmse" => sqrt(mean(all_com_y_errors.^2)),
        "y_mean" => mean(abs.(all_com_y_errors)),
        "y_median" => median(abs.(all_com_y_errors)),
        "y_std" => std(per_video_com_y_rmse),  # Std of RMSE across videos
        "y_95th" => quantile(abs.(all_com_y_errors), 0.95),
        "position_rmse" => sqrt(mean(all_com_position_errors.^2)),
        "position_mean" => mean(all_com_position_errors),
        "position_median" => median(all_com_position_errors),
        "position_std" => std(per_video_com_position_rmse),  # Std of RMSE across videos
        "position_95th" => quantile(all_com_position_errors, 0.95)
    )

    head_stats = Dict(
        "x_rmse" => sqrt(mean(all_head_x_errors.^2)),
        "x_mean" => mean(abs.(all_head_x_errors)),
        "x_median" => median(abs.(all_head_x_errors)),
        "x_std" => std(per_video_head_x_rmse),  # Std of RMSE across videos
        "x_95th" => quantile(abs.(all_head_x_errors), 0.95),
        "y_rmse" => sqrt(mean(all_head_y_errors.^2)),
        "y_mean" => mean(abs.(all_head_y_errors)),
        "y_median" => median(abs.(all_head_y_errors)),
        "y_std" => std(per_video_head_y_rmse),  # Std of RMSE across videos
        "y_95th" => quantile(abs.(all_head_y_errors), 0.95),
        "theta_rmse" => sqrt(mean(all_head_theta_errors.^2)),
        "theta_mean" => mean(abs.(all_head_theta_errors)),
        "theta_median" => median(abs.(all_head_theta_errors)),
        "theta_std" => std(per_video_head_theta_rmse),  # Std of RMSE across videos
        "theta_95th" => quantile(abs.(all_head_theta_errors), 0.95),
        "position_rmse" => sqrt(mean(all_head_position_errors.^2)),
        "position_mean" => mean(all_head_position_errors),
        "position_median" => median(all_head_position_errors),
        "position_std" => std(per_video_head_position_rmse),  # Std of RMSE across videos
        "position_95th" => quantile(all_head_position_errors, 0.95)
    )
    
    results[amplitude] = Dict(
        "com_stats" => com_stats,
        "head_stats" => head_stats,
        "num_videos" => length(video_errors),
        "num_samples" => length(all_com_position_errors),
        "outliers" => !isempty(outlier_videos) ? outlier_videos : String[]
    )
    
    # Add genesis vs. hardware statistics
    # Collect all genesis errors from all hardware videos
    all_genesis_com_x_errors = vcat([genesis_video_errors[v]["com_x_errors"] for v in keys(genesis_video_errors)]...)
        all_genesis_com_y_errors = vcat([genesis_video_errors[v]["com_y_errors"] for v in keys(genesis_video_errors)]...)
        all_genesis_com_position_errors = vcat([genesis_video_errors[v]["com_position_errors"] for v in keys(genesis_video_errors)]...)
        all_genesis_head_x_errors = vcat([genesis_video_errors[v]["head_x_errors"] for v in keys(genesis_video_errors)]...)
        all_genesis_head_y_errors = vcat([genesis_video_errors[v]["head_y_errors"] for v in keys(genesis_video_errors)]...)
        all_genesis_head_theta_errors = vcat([genesis_video_errors[v]["head_theta_errors"] for v in keys(genesis_video_errors)]...)
        all_genesis_head_position_errors = vcat([genesis_video_errors[v]["head_position_errors"] for v in keys(genesis_video_errors)]...)

        # Compute per-video RMSE values for genesis cross-trial statistics
        per_video_genesis_com_x_rmse = [sqrt(mean(genesis_video_errors[v]["com_x_errors"].^2)) for v in keys(genesis_video_errors)]
        per_video_genesis_com_y_rmse = [sqrt(mean(genesis_video_errors[v]["com_y_errors"].^2)) for v in keys(genesis_video_errors)]
        per_video_genesis_com_position_rmse = [sqrt(mean(genesis_video_errors[v]["com_position_errors"].^2)) for v in keys(genesis_video_errors)]
        per_video_genesis_head_x_rmse = [sqrt(mean(genesis_video_errors[v]["head_x_errors"].^2)) for v in keys(genesis_video_errors)]
        per_video_genesis_head_y_rmse = [sqrt(mean(genesis_video_errors[v]["head_y_errors"].^2)) for v in keys(genesis_video_errors)]
        per_video_genesis_head_theta_rmse = [sqrt(mean(genesis_video_errors[v]["head_theta_errors"].^2)) for v in keys(genesis_video_errors)]
        per_video_genesis_head_position_rmse = [sqrt(mean(genesis_video_errors[v]["head_position_errors"].^2)) for v in keys(genesis_video_errors)]

        genesis_com_stats = Dict(
            "x_rmse" => sqrt(mean(all_genesis_com_x_errors.^2)),
            "x_mean" => mean(abs.(all_genesis_com_x_errors)),
            "x_median" => median(abs.(all_genesis_com_x_errors)),
            "x_std" => std(per_video_genesis_com_x_rmse),  # Std of RMSE across videos
            "x_95th" => quantile(abs.(all_genesis_com_x_errors), 0.95),
            "y_rmse" => sqrt(mean(all_genesis_com_y_errors.^2)),
            "y_mean" => mean(abs.(all_genesis_com_y_errors)),
            "y_median" => median(abs.(all_genesis_com_y_errors)),
            "y_std" => std(per_video_genesis_com_y_rmse),  # Std of RMSE across videos
            "y_95th" => quantile(abs.(all_genesis_com_y_errors), 0.95),
            "position_rmse" => sqrt(mean(all_genesis_com_position_errors.^2)),
            "position_mean" => mean(all_genesis_com_position_errors),
            "position_median" => median(all_genesis_com_position_errors),
            "position_std" => std(per_video_genesis_com_position_rmse),  # Std of RMSE across videos
            "position_95th" => quantile(all_genesis_com_position_errors, 0.95)
        )

        genesis_head_stats = Dict(
            "x_rmse" => sqrt(mean(all_genesis_head_x_errors.^2)),
            "x_mean" => mean(abs.(all_genesis_head_x_errors)),
            "x_median" => median(abs.(all_genesis_head_x_errors)),
            "x_std" => std(per_video_genesis_head_x_rmse),  # Std of RMSE across videos
            "x_95th" => quantile(abs.(all_genesis_head_x_errors), 0.95),
            "y_rmse" => sqrt(mean(all_genesis_head_y_errors.^2)),
            "y_mean" => mean(abs.(all_genesis_head_y_errors)),
            "y_median" => median(abs.(all_genesis_head_y_errors)),
            "y_std" => std(per_video_genesis_head_y_rmse),  # Std of RMSE across videos
            "y_95th" => quantile(abs.(all_genesis_head_y_errors), 0.95),
            "theta_rmse" => sqrt(mean(all_genesis_head_theta_errors.^2)),
            "theta_mean" => mean(abs.(all_genesis_head_theta_errors)),
            "theta_median" => median(abs.(all_genesis_head_theta_errors)),
            "theta_std" => std(per_video_genesis_head_theta_rmse),  # Std of RMSE across videos
            "theta_95th" => quantile(abs.(all_genesis_head_theta_errors), 0.95),
            "position_rmse" => sqrt(mean(all_genesis_head_position_errors.^2)),
            "position_mean" => mean(all_genesis_head_position_errors),
            "position_median" => median(all_genesis_head_position_errors),
            "position_std" => std(per_video_genesis_head_position_rmse),  # Std of RMSE across videos
            "position_95th" => quantile(all_genesis_head_position_errors, 0.95)
        )
    
    results[amplitude]["genesis_com_stats"] = genesis_com_stats
    results[amplitude]["genesis_head_stats"] = genesis_head_stats
    results[amplitude]["genesis_num_samples"] = length(all_genesis_com_position_errors)

    # Calculate percent improvement (positive = simulation is better)
    percent_improvements = Dict(
        "com_x" => (genesis_com_stats["x_rmse"] - com_stats["x_rmse"]) / genesis_com_stats["x_rmse"] * 100,
        "com_y" => (genesis_com_stats["y_rmse"] - com_stats["y_rmse"]) / genesis_com_stats["y_rmse"] * 100,
        "com_position" => (genesis_com_stats["position_rmse"] - com_stats["position_rmse"]) / genesis_com_stats["position_rmse"] * 100,
        "head_x" => (genesis_head_stats["x_rmse"] - head_stats["x_rmse"]) / genesis_head_stats["x_rmse"] * 100,
        "head_y" => (genesis_head_stats["y_rmse"] - head_stats["y_rmse"]) / genesis_head_stats["y_rmse"] * 100,
        "head_theta" => (genesis_head_stats["theta_rmse"] - head_stats["theta_rmse"]) / genesis_head_stats["theta_rmse"] * 100,
        "head_position" => (genesis_head_stats["position_rmse"] - head_stats["position_rmse"]) / genesis_head_stats["position_rmse"] * 100
    )

    results[amplitude]["percent_improvements"] = percent_improvements

    #############################################################################################
    ## Print results for this amplitude
    #############################################################################################

    println("\n" * "-"^80)
    println("Results for $amplitude ($(results[amplitude]["num_videos"]) videos, $(results[amplitude]["num_samples"]) total samples)")
    if !isempty(results[amplitude]["outliers"])
        println("  Outliers detected (not filtered): $(join(results[amplitude]["outliers"], ", "))")
    end
    println("-"^80)
    
    println("\nCenter of Mass (COM) Trajectory Errors:")
    println("  X-direction:")
    println("    RMSE:        $(round(com_stats["x_rmse"], digits=3)) cm")
    println("    Mean:        $(round(com_stats["x_mean"], digits=3)) cm")
    println("    Median:      $(round(com_stats["x_median"], digits=3)) cm")
    println("    Std Dev:     $(round(com_stats["x_std"], digits=3)) cm")
    println("    95th %ile:   $(round(com_stats["x_95th"], digits=3)) cm")
    
    println("\n  Y-direction:")
    println("    RMSE:        $(round(com_stats["y_rmse"], digits=3)) cm")
    println("    Mean:        $(round(com_stats["y_mean"], digits=3)) cm")
    println("    Median:      $(round(com_stats["y_median"], digits=3)) cm")
    println("    Std Dev:     $(round(com_stats["y_std"], digits=3)) cm")
    println("    95th %ile:   $(round(com_stats["y_95th"], digits=3)) cm")
    
    println("\n  Position (Euclidean):")
    println("    RMSE:        $(round(com_stats["position_rmse"], digits=3)) cm")
    println("    Mean:        $(round(com_stats["position_mean"], digits=3)) cm")
    println("    Median:      $(round(com_stats["position_median"], digits=3)) cm")
    println("    Std Dev:     $(round(com_stats["position_std"], digits=3)) cm")
    println("    95th %ile:   $(round(com_stats["position_95th"], digits=3)) cm")
    
    println("\n\nHead Link Trajectory Errors:")
    println("  X-direction:")
    println("    RMSE:        $(round(head_stats["x_rmse"], digits=3)) cm")
    println("    Mean:        $(round(head_stats["x_mean"], digits=3)) cm")
    println("    Median:      $(round(head_stats["x_median"], digits=3)) cm")
    println("    Std Dev:     $(round(head_stats["x_std"], digits=3)) cm")
    println("    95th %ile:   $(round(head_stats["x_95th"], digits=3)) cm")

    println("\n  Y-direction:")
    println("    RMSE:        $(round(head_stats["y_rmse"], digits=3)) cm")
    println("    Mean:        $(round(head_stats["y_mean"], digits=3)) cm")
    println("    Median:      $(round(head_stats["y_median"], digits=3)) cm")
    println("    Std Dev:     $(round(head_stats["y_std"], digits=3)) cm")
    println("    95th %ile:   $(round(head_stats["y_95th"], digits=3)) cm")

    println("\n  Theta (angle):")
    println("    RMSE:        $(round(rad2deg(head_stats["theta_rmse"]), digits=3))°")
    println("    Mean:        $(round(rad2deg(head_stats["theta_mean"]), digits=3))°")
    println("    Median:      $(round(rad2deg(head_stats["theta_median"]), digits=3))°")
    println("    Std Dev:     $(round(rad2deg(head_stats["theta_std"]), digits=3))°")
    println("    95th %ile:   $(round(rad2deg(head_stats["theta_95th"]), digits=3))°")

    println("\n  Position (Euclidean):")
    println("    RMSE:        $(round(head_stats["position_rmse"], digits=3)) cm")
    println("    Mean:        $(round(head_stats["position_mean"], digits=3)) cm")
    println("    Median:      $(round(head_stats["position_median"], digits=3)) cm")
    println("    Std Dev:     $(round(head_stats["position_std"], digits=3)) cm")
    println("    95th %ile:   $(round(head_stats["position_95th"], digits=3)) cm")

    println("\n\n% Improvement over Baseline (Genesis):")
    println("  (positive = Unified Multiphysics is better)")
    println("\n  Center of Mass:")
    println("    X RMSE:       $(round(percent_improvements["com_x"], digits=2))%")
    println("    Y RMSE:       $(round(percent_improvements["com_y"], digits=2))%")
    println("    Position RMSE: $(round(percent_improvements["com_position"], digits=2))%")
    println("\n  Head Link:")
    println("    X RMSE:       $(round(percent_improvements["head_x"], digits=2))%")
    println("    Y RMSE:       $(round(percent_improvements["head_y"], digits=2))%")
    println("    Theta RMSE:   $(round(percent_improvements["head_theta"], digits=2))%")
    println("    Position RMSE: $(round(percent_improvements["head_position"], digits=2))%")
end

#############################################################################################
## Summary table across all amplitudes
#############################################################################################

println("\n\n" * "="^80)
println("SUMMARY: Trajectory Error Analysis Across All Amplitudes")
println("="^80)

println(log_file, "\n" * "="^80)
println(log_file, "SUMMARY: Trajectory Error Analysis Across All Amplitudes")
println(log_file, "="^80)

println("\nUnified Multiphysics RMSE by Component:")
println("-"^80)
println("Amplitude │  COM X  │  COM Y  │  Head X │  Head Y │ Head θ (deg) │ Videos │ Samples")
println("-"^80)

println(log_file, "\nUnified Multiphysics RMSE by Component:")
println(log_file, "-"^80)
println(log_file, "Amplitude │  COM X  │  COM Y  │  Head X │  Head Y │ Head θ (deg) │ Videos │ Samples")
println(log_file, "-"^80)
for amp in amplitude_cases
    com_stats = results[amp]["com_stats"]
    head_stats = results[amp]["head_stats"]
    n_samples = results[amp]["num_samples"]
    n_videos = results[amp]["num_videos"]
    line = @sprintf("%-9s │ %7.3f │ %7.3f │ %7.3f │ %7.3f │    %7.3f   │   %2d   │ %7d\n",
        amp,
        com_stats["x_rmse"],
        com_stats["y_rmse"],
        head_stats["x_rmse"],
        head_stats["y_rmse"],
        rad2deg(head_stats["theta_rmse"]),
        n_videos,
        n_samples)
    print(line)
    print(log_file, line)
end

# Genesis vs. Hardware comparison tables
println("\n\nBaseline (Genesis) RMSE by Component:")
println("-"^80)
println("Amplitude │  COM X  │  COM Y  │  Head X │  Head Y │ Head θ (deg) │ Videos │ Samples")
println("-"^80)

println(log_file, "\n\nBaseline (Genesis) RMSE by Component:")
println(log_file, "-"^80)
println(log_file, "Amplitude │  COM X  │  COM Y  │  Head X │  Head Y │ Head θ (deg) │ Videos │ Samples")
println(log_file, "-"^80)
for amp in amplitude_cases
    if haskey(results[amp], "genesis_com_stats")
        genesis_com_stats = results[amp]["genesis_com_stats"]
        genesis_head_stats = results[amp]["genesis_head_stats"]
        genesis_samples = results[amp]["genesis_num_samples"]
        n_videos = results[amp]["num_videos"]
        line = @sprintf("%-9s │ %7.3f │ %7.3f │ %7.3f │ %7.3f │    %7.3f   │   %2d   │ %7d\n",
            amp,
            genesis_com_stats["x_rmse"],
            genesis_com_stats["y_rmse"],
            genesis_head_stats["x_rmse"],
            genesis_head_stats["y_rmse"],
            rad2deg(genesis_head_stats["theta_rmse"]),
            n_videos,
            genesis_samples)
        print(line)
        print(log_file, line)
    end
end

println("\n\n% Improvement: Unified Multiphysics vs Baseline (Genesis)")
println("-"^90)
println("Amplitude │ COM X │ COM Y │ COM Pos │ Head X │ Head Y │ Head θ │ Head Pos")
println("-"^90)

println(log_file, "\n\n% Improvement: Unified Multiphysics vs Baseline (Genesis)")
println(log_file, "-"^90)
println(log_file, "Amplitude │ COM X │ COM Y │ COM Pos │ Head X │ Head Y │ Head θ │ Head Pos")
println(log_file, "-"^90)

for amp in amplitude_cases
    improvements = results[amp]["percent_improvements"]
    line = @sprintf("%-9s │ %5.1f │ %5.1f │  %5.1f  │ %6.1f │ %6.1f │ %6.1f │  %6.1f\n",
        amp,
        improvements["com_x"],
        improvements["com_y"],
        improvements["com_position"],
        improvements["head_x"],
        improvements["head_y"],
        improvements["head_theta"],
        improvements["head_position"])
    print(line)
    print(log_file, line)
end
println("Note: Positive values indicate Unified Multiphysics has lower RMSE (better performance)")

println(log_file, "Note: Positive values indicate Unified Multiphysics has lower RMSE (better performance)")

println("\n" * "="^80)
println("Analysis complete!")
println("="^80)

println(log_file, "\n" * "="^80)
println(log_file, "Analysis complete!")
println(log_file, "="^80)

close(log_file)
println("\nResults saved to: $log_path")

#############################################################################################
## Create Visualization: Grouped Bar Charts with Error Bars
#############################################################################################

println("\n" * "="^80)
println("Creating grouped bar charts for component-wise error comparison...")
println("="^80)

# Define colors
simulation_color = RGB(0.0, 0.7294, 0.3451)  # jj_green
genesis_color = RGB(0.9451, 0.6745, 0.09020)  # jj_orange

# Plot parameters
background_color = :transparent
fontsize = 18
resolution = (1000, 600)

# For each component (x, y, theta), create a grouped bar chart
components = [
    ("x", "X Position Error", "cm"),
    ("y", "Y Position Error", "cm"),
    ("theta", "Theta (Angle) Error", "deg")
]

for (component_key, component_label, unit) in components
    println("\n  Creating bar chart for $component_label...")

    # Extract RMSE and std for simulation and genesis
    sim_rmse_values = Float64[]
    sim_std_values = Float64[]
    genesis_rmse_values = Float64[]
    genesis_std_values = Float64[]

    for amp in amplitude_cases
        if component_key == "theta"
            # Convert theta from radians to degrees
            push!(sim_rmse_values, rad2deg(results[amp]["head_stats"]["$(component_key)_rmse"]))
            push!(sim_std_values, rad2deg(results[amp]["head_stats"]["$(component_key)_std"]))
            push!(genesis_rmse_values, rad2deg(results[amp]["genesis_head_stats"]["$(component_key)_rmse"]))
            push!(genesis_std_values, rad2deg(results[amp]["genesis_head_stats"]["$(component_key)_std"]))
        else
            push!(sim_rmse_values, results[amp]["head_stats"]["$(component_key)_rmse"])
            push!(sim_std_values, results[amp]["head_stats"]["$(component_key)_std"])
            push!(genesis_rmse_values, results[amp]["genesis_head_stats"]["$(component_key)_rmse"])
            push!(genesis_std_values, results[amp]["genesis_head_stats"]["$(component_key)_std"])
        end
    end

    # Create figure with CairoMakie
    fig, ax = create_aquarium_figure(;
        backgroundcolor=background_color,
        fontsize=fontsize,
        resolution=resolution,
        axiscolor=:black,
        xlabel="Amplitude",
        ylabel="$component_label ($unit)",
        use_data_aspect=false
    )

    # X positions for bars
    x_positions = 1:length(amplitude_cases)
    bar_width = 0.35

    # Plot simulation bars
    barplot!(ax, x_positions .- bar_width/2, sim_rmse_values,
        width=bar_width,
        color=simulation_color,
        label="Simulation")

    # Plot genesis bars
    barplot!(ax, x_positions .+ bar_width/2, genesis_rmse_values,
        width=bar_width,
        color=genesis_color,
        label="Genesis")

    # Add error bars for simulation
    errorbars!(ax, x_positions .- bar_width/2, sim_rmse_values, sim_std_values,
        color=:blue,
        linewidth=2,
        whiskerwidth=10)

    # Add error bars for genesis
    errorbars!(ax, x_positions .+ bar_width/2, genesis_rmse_values, genesis_std_values,
        color=:purple,
        linewidth=2,
        whiskerwidth=10)

    # Set x-axis labels
    ax.xticks = (x_positions, amplitude_cases)

    # Add legend
    axislegend(ax, position=:lt)

    display(fig)

    # Save as PNG
    png_filename = joinpath(output_dir, "$(component_key)_error_comparison.png")
    save(png_filename, fig)
    println("    ✓ Saved: $(component_key)_error_comparison.png")

    # Create TikZ plot using PGFPlotsX
    # Create DataFrame for easier table handling
    # Convert amplitude strings to numeric values for x-axis
    amplitude_numeric = [10, 20, 30, 40]

    sim_df = DataFrame(
        x = amplitude_numeric,
        y = sim_rmse_values,
        error = sim_std_values
    )

    genesis_df = DataFrame(
        x = amplitude_numeric,
        y = genesis_rmse_values,
        error = genesis_std_values
    )

    # Build bar chart - conditionally include legend and ylabel
    if component_key == "x"
        bar_chart = @pgf PGFPlotsX.Axis(
            {
                ybar,
                bar_width = "$(bar_width)cm",
                ylabel = "RMSE",
                xlabel = "Amplitude (deg)",
                xtick = amplitude_numeric,
                ymin = 0,
                legend_pos = "north west",
                legend_cell_align = "left",
                ymajorgrids,
            },
            PlotInc(@pgf({fill = simulation_color, "error bars/.cd", y_dir="both", y_explicit}),
                Table({x = "x", y = "y", "y error" = "error"}, sim_df)),
            PlotInc(@pgf({fill = genesis_color, "error bars/.cd", y_dir="both", y_explicit}),
                Table({x = "x", y = "y", "y error" = "error"}, genesis_df)),
            PGFPlotsX.Legend(["Unified Multiphysics", "Baseline"])
        )
    else
        bar_chart = @pgf PGFPlotsX.Axis(
            {
                ybar,
                bar_width = "$(bar_width)cm",
                xlabel = "Amplitude (deg)",
                xtick = amplitude_numeric,
                ymin = 0,
                ymajorgrids,
            },
            PlotInc(@pgf({fill = simulation_color, "error bars/.cd", y_dir="both", y_explicit}),
                Table({x = "x", y = "y", "y error" = "error"}, sim_df)),
            PlotInc(@pgf({fill = genesis_color, "error bars/.cd", y_dir="both", y_explicit}),
                Table({x = "x", y = "y", "y error" = "error"}, genesis_df))
        )
    end

    tikz_filename = joinpath(output_dir, "$(component_key)_error_comparison.tikz")
    pgfsave(tikz_filename, bar_chart, include_preamble=false)
    println("    ✓ Saved: $(component_key)_error_comparison.tikz")
end

println("\n" * "="^80)
println("Visualization complete!")
println("Generated grouped bar charts for X, Y, and Theta components")
println("Output directory: $output_dir")
println("="^80)