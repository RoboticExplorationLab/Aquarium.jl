import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using VideoIO
using JLD2
using ArucoTracking
using Images
using FileIO
using Interpolations
using SmoothingSplines
using Plots

const cv2 = ArucoTracking.cv2
const aruco = ArucoTracking.aruco

#############################################################################################
## Configuration
#############################################################################################

# amplitude
amp = 40  # degrees

# Paths
calibration_path = joinpath(@__DIR__, "..", "camera_calibration", "calibration_data.jld2")
data_dir = expanduser("~/aquariumCLOSED/data/rexeel_forward_swimming/$(amp)deg")

# Create output directory if it doesn't exist
mkpath(output_dir)

# AprilTag parameters
marker_length = 4.0  # Physical marker size in cm (adjust as needed)
robot_marker_id = 1  # Robot's AprilTag ID
origin_marker_id = 11  # Origin/reference AprilTag ID
zero_angle_id = 13  # Zero-angle AprilTag ID

# Find all MP4 videos in the directory
video_files = filter(f -> endswith(f, ".mp4"), readdir(data_dir))
sort!(video_files)  # Sort for consistent ordering

println("="^70)
println("AprilTag Tracking - Rexeel Forward Swimming ($(amp)deg)")
println("="^70)
println("Found $(length(video_files)) videos to process")
println()

#############################################################################################
## Load calibration data
#############################################################################################

println("Loading calibration data...")
if !isfile(calibration_path)
    error("Calibration file not found: $calibration_path")
end

calibration = load(calibration_path)
mtx = calibration["mtx"]
dist = calibration["dist"]
println("  ✓ Calibration loaded successfully")
println("  Camera matrix size: $(size(mtx))")
println()

#############################################################################################
## Setup AprilTag detector
#############################################################################################

println("Setting up AprilTag detector...")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
detector_params = cv2.aruco.DetectorParameters()

# Optimize detection parameters for better reliability
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 23
detector_params.adaptiveThreshWinSizeStep = 10
detector_params.adaptiveThreshConstant = 7

# Corner refinement for sub-pixel accuracy
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector_params.cornerRefinementWinSize = 5
detector_params.cornerRefinementMaxIterations = 30
detector_params.cornerRefinementMinAccuracy = 0.1

# Increase detection robustness
detector_params.minMarkerPerimeterRate = 0.03
detector_params.maxMarkerPerimeterRate = 4.0
detector_params.polygonalApproxAccuracyRate = 0.03

# Error correction
detector_params.errorCorrectionRate = 0.6

detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
ids_to_track = [robot_marker_id, origin_marker_id, zero_angle_id]
println("  ✓ Detector initialized")
println("  Dictionary: DICT_APRILTAG_25h9 (tag25h9 family)")
println("  Robot marker ID: $robot_marker_id")
println("  Origin marker ID: $origin_marker_id")
println("  Zero-angle marker ID: $zero_angle_id")
println("  Marker length: $(marker_length) cm")
println()

#############################################################################################
## Process each video
#############################################################################################

for (video_idx, video_name) in enumerate(video_files[3:3])

    video_idx = 3  # For testing, process only the first video
    video_name = video_files[video_idx]

    println("="^70)
    println("Processing video $video_idx / $(length(video_files)): $video_name")
    println("="^70)

    video_path = joinpath(data_dir, video_name)

    # Determine output filename
    base_name = replace(video_name, "_video.mp4" => "")
    output_path = joinpath(data_dir, "$(base_name)_trajectories.jld2")

    # Skip if already processed
    # if isfile(output_path)
    #     println("  ⊗ Already processed, skipping...")
    #     println()
    #     continue
    # end

    #############################################################################
    ## Load video
    #############################################################################

    println("Loading video...")
    if !isfile(video_path)
        @warn "Video file not found: $video_path"
        continue
    end
    
    video_reader = VideoIO.openvideo(video_path)
    img_traj = Vector{Matrix{RGB{N0f8}}}()

    frame_count = 0
    for frame in video_reader
        push!(img_traj, frame)
        frame_count += 1
        if frame_count % 100 == 0
            print("\r  Frames loaded: $frame_count")
        end
    end
    println("\r  ✓ Loaded $frame_count frames")
    println("  Frame size: $(size(img_traj[1]))")

    #############################################################################
    ## Detect markers
    #############################################################################

    println("Detecting markers across all frames...")

    time_traj = 0:1/60:(frame_count-1)/60
    marker_configs_traj = get_marker_configurations_trajectory(
        img_traj,
        detector,
        marker_length,
        mtx;
        ids=ids_to_track
    )

    println("  ✓ Marker detection complete")
    detected_ids = collect(keys(marker_configs_traj[2]))
    println("  Detected marker IDs: $detected_ids")

    #############################################################################
    ## Transform trajectory
    #############################################################################

    println("Transforming robot trajectory to origin reference frame...")

    # Extract trajectories for origin and robot markers
    origin = [marker_configs_traj[1][marker_configs_traj[2] .== origin_marker_id][1][100][1:2]...
                marker_configs_traj[1][marker_configs_traj[2] .== zero_angle_id][1][100][4]
            ]

    robot_traj = marker_configs_traj[1][marker_configs_traj[2] .== robot_marker_id][1]
    num_frames = length(robot_traj)

    # Find first valid frame (non-zero detection) and pad leading zeros
    first_valid_frame = findfirst(cfg -> !all(cfg .≈ 0.0), robot_traj)

    if first_valid_frame === nothing
        @warn "No valid robot marker detections found in entire video"
        @warn "Skipping this video"
        println()
        continue
    end

    if first_valid_frame > 1
        println("  Padding $(first_valid_frame - 1) leading zero frames with first valid detection")
        first_valid_config = robot_traj[first_valid_frame]
        # Replace all leading zeros with the first valid configuration
        for i in 1:(first_valid_frame - 1)
            robot_traj[i] = first_valid_config
        end
    end

    # Transform robot trajectory relative to origin
    robot_head_sim_frame_traj = [[-robot_traj[i][2]+origin[2], robot_traj[i][1]-origin[1], -robot_traj[i][4]+origin[3] - pi/2] for i in 1:num_frames]

    println("  ✓ Robot trajectory transformed to origin reference frame")

    #############################################################################
    ## Fit Smoothing Splines to Trajectories
    #############################################################################

    println("Fitting smoothing splines to trajectories...")

    # Smoothing parameter (adjust as needed)
    λ = 0.05  # Lower values = closer to data, higher values = smoother

    # Convert time_traj to Vector{Float64} for SmoothingSplines
    time_vec = collect(Float64, time_traj)

    # Extract x, y, θ components from robot_traj
    robot_x = Float64[cfg[1] for cfg in robot_traj]
    robot_y = Float64[cfg[2] for cfg in robot_traj]
    robot_θ = Float64[cfg[4] for cfg in robot_traj]

    # Fit smoothing splines for robot_traj components
    robot_x_spl = fit(SmoothingSpline, time_vec, robot_x, λ)
    robot_y_spl = fit(SmoothingSpline, time_vec, robot_y, λ)
    robot_θ_spl = fit(SmoothingSpline, time_vec, robot_θ, λ)

    # Extract x, y, θ components from robot_head_sim_frame_traj
    robot_sim_x = Float64[cfg[1] for cfg in robot_head_sim_frame_traj]
    robot_sim_y = Float64[cfg[2] for cfg in robot_head_sim_frame_traj]
    robot_sim_θ = Float64[cfg[3] for cfg in robot_head_sim_frame_traj]

    # Fit smoothing splines for robot_head_sim_frame_traj components
    robot_sim_x_spl = fit(SmoothingSpline, time_vec, robot_sim_x, λ)
    robot_sim_y_spl = fit(SmoothingSpline, time_vec, robot_sim_y, λ)
    robot_sim_θ_spl = fit(SmoothingSpline, time_vec, robot_sim_θ, λ)

    println("  ✓ Smoothing splines fitted (λ = $λ)")

    #############################################################################
    ## Generate Smoothed Trajectories
    #############################################################################

    println("Generating smoothed trajectories...")

    # Get smoothed values at original time points
    robot_x_smooth = predict(robot_x_spl)
    robot_y_smooth = predict(robot_y_spl)
    robot_θ_smooth = predict(robot_θ_spl)

    # Reconstruct smoothed robot_traj
    robot_traj_smooth = [[robot_x_smooth[i], robot_y_smooth[i], robot_traj[i][3], robot_θ_smooth[i]] for i in 1:num_frames]

    # Get smoothed values for sim frame trajectory
    robot_sim_x_smooth = predict(robot_sim_x_spl)
    robot_sim_y_smooth = predict(robot_sim_y_spl)
    robot_sim_θ_smooth = predict(robot_sim_θ_spl)

    # Reconstruct smoothed robot_head_sim_frame_traj
    robot_head_sim_frame_traj_smooth = [[robot_sim_x_smooth[i], robot_sim_y_smooth[i], robot_sim_θ_smooth[i]] for i in 1:num_frames]

    println("  ✓ Smoothed trajectories generated")
    println("  Time range: $(time_traj[1]) to $(time_traj[end]) seconds")

    #############################################################################
    ## Plot original and smoothed trajectories
    #############################################################################

    println("Plotting trajectories...")

    # Create subplots for x and y
    p1 = Plots.plot(collect(time_traj), robot_sim_x,
              label="Original X",
              linewidth=1,
              alpha=0.6,
              xlabel="Time (s)",
              ylabel="X Position (cm)",
              title="Robot X Position in Sim Frame")
    Plots.plot!(p1, collect(time_traj), robot_sim_x_smooth,
          label="Smoothed X",
          linewidth=2,
          color=:red)

    p2 = Plots.plot(collect(time_traj), robot_sim_y,
              label="Original Y",
              linewidth=1,
              alpha=0.6,
              xlabel="Time (s)",
              ylabel="Y Position (cm)",
              title="Robot Y Position in Sim Frame")
    Plots.plot!(p2, collect(time_traj), robot_sim_y_smooth,
          label="Smoothed Y",
          linewidth=2,
          color=:red)

    # Combine into single figure
    plot_combined = Plots.plot(p1, p2, layout=(2,1), size=(800, 600))
    Plots.display(plot_combined)

    println("  ✓ Trajectories plotted")
    println("  Continue with this video? (y/n, default=y): ")
    user_input = readline()

    if lowercase(strip(user_input)) == "n"
        println("  Skipping this video...")
        println()
        continue
    end

    #############################################################################
    ## Find frame of trajectory start
    #############################################################################

    println("Detecting motion onset frame...")

    # Create cubic spline interpolants that support derivatives
    robot_sim_x_interp = CubicSplineInterpolation(time_traj, robot_sim_x_smooth)
    robot_sim_y_interp = CubicSplineInterpolation(time_traj, robot_sim_y_smooth)

    # Compute velocities using Interpolations.jl gradient at each time point
    # vx = [gradient(robot_sim_x_interp, t)[1] for t in time_traj]
    vy = [Interpolations.gradient(robot_sim_y_interp, t)[1] for t in time_traj]
    speed = vy # sqrt.(vx.^2 .+ vy.^2)  # Speed magnitude in cm/s

    # Motion detection parameters
    motion_threshold = 0.3  # cm/s (sensitive detection for gradual starts)
    min_consecutive_frames = 3  # Require sustained motion to avoid noise

    # Find first frame where speed exceeds threshold for min_consecutive_frames
    motion_start_frame = nothing
    for i in 1:(length(speed) - min_consecutive_frames + 1)
        if all(speed[i:i+min_consecutive_frames-1] .> motion_threshold)
            motion_start_frame = i
            break
        end
    end

    motion_start_frame = 45

    if !isnothing(motion_start_frame)
        motion_start_time = time_traj[motion_start_frame]
        println("  ✓ Motion start detected at frame $motion_start_frame (t = $(round(motion_start_time, digits=3)) s)")
        println("  Speed at onset: $(round(speed[motion_start_frame], digits=2)) cm/s")
    else
        @warn "  No clear motion onset detected (max speed: $(round(maximum(speed), digits=2)) cm/s)"
        motion_start_frame = 1
        motion_start_time = time_traj[1]
    end

    # Motion end frame is 4 seconds (240 frames at 60 Hz) after start
    motion_duration_frames = 240  # 4 seconds at 60 Hz
    motion_end_frame = motion_start_frame + motion_duration_frames

    # Check if trajectory is long enough
    if motion_end_frame > num_frames
        @warn "Trajectory too short: need $(motion_end_frame) frames, have $(num_frames) frames."
        @warn "Skipping this video (needs at least 4 seconds of motion after onset)"
        println()
        continue
    end

    motion_end_time = time_traj[motion_end_frame]
    println("  ✓ Motion end set at frame $motion_end_frame (t = $(round(motion_end_time, digits=3)) s)")
    println("  Motion duration: 4.0 s (240 frames)")

    #############################################################################
    ## Save results
    #############################################################################

    println("Saving results...")
    
    # Extract only the motion segment
    frame_range = motion_start_frame:motion_end_frame
    
    save(output_path, Dict(
        "robot_head_sim_frame_traj" => robot_head_sim_frame_traj[frame_range],
        "robot_head_sim_frame_traj_smooth" => robot_head_sim_frame_traj_smooth[frame_range],
        "robot_sim_x_spl" => robot_sim_x_spl,
        "robot_sim_y_spl" => robot_sim_y_spl,
        "robot_sim_θ_spl" => robot_sim_θ_spl,
        "time_traj" => time_traj[frame_range],
        "robot_marker_id" => robot_marker_id,
        "origin_marker_id" => origin_marker_id,
        "origin" => origin,
        "robot_traj" => robot_traj[frame_range],
        "robot_traj_smooth" => robot_traj_smooth[frame_range],
        "robot_x_spl" => robot_x_spl,
        "robot_y_spl" => robot_y_spl,
        "robot_θ_spl" => robot_θ_spl,
        "marker_configs_traj" => marker_configs_traj,
        "video_path" => video_path,
        "video_name" => video_name,
        "num_frames" => length(frame_range),
        "marker_length" => marker_length,
        "calibration_path" => calibration_path,
        "λ" => λ,
        "motion_start_frame" => motion_start_frame,
        "motion_start_time" => motion_start_time,
        "motion_end_frame" => motion_end_frame,
        "motion_end_time" => motion_end_time,
        "speed_trajectory" => speed[frame_range],
        "motion_threshold" => 0.3
    ))
    println("  ✓ Results saved to: $output_path")
    println("  ✓ Exported frames: $(length(frame_range)) (from frame $motion_start_frame to $motion_end_frame)")
    println()

end

#############################################################################################
## Summary
#############################################################################################

println("="^70)
println("BATCH TRACKING COMPLETE")
println("="^70)

# Count processed files
processed_files = filter(f -> endswith(f, "_trajectories.jld2"), readdir(output_dir))
println("Successfully processed: $(length(processed_files)) / $(length(video_files)) videos")
println("Output directory: $output_dir")
println("="^70)
