include(joinpath(@__DIR__, "..", "common.jl"))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.ForwardDiff
using Aquarium.CairoMakie
using Colors
using JLD2
using Test
using Random

vis_dir = visualization_dir("rexeel_c_start")

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
final_time = 3.0  # Total C-start duration: T_prep + T_prop = 1.0 + 2.0 = 3.0s
N_time = Int(final_time/time_step) + 1

# fluid properties (water)
fluid_density = 1.0  # g/cm³
dynamic_viscosity = 0.01 .* 1.05  # g/(cm*s) - water at room temperature

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
## Define 3-link RExEel (swimmer)
#############################################################################################

# eel properties
n_links = 6
# Link lengths for 3-link eel (using same as manipulator)
link_lengths = [12.0, 9.8 .* ones(n_links-1)...]  # cm per link
height = 9.35  # cm
masses_per_link = [192, 140 .* ones(n_links-1)...] ./ height .* 1.05 # g per link
moi_per_link = [2435.99, 1483.49 .* ones(n_links-1)...] ./ height .* 1.05 # g·cm²
gravity_constant = 0.0

# boundary properties - compute per-link boundary nodes based on link length
n_boundary_nodes = floor.(Int, link_lengths ./ fluid_env.fvm_grid.h_x)

# Starting position at center of tank, oriented vertically downward (head down, tail up)
start_x = 50.0
start_y = length_y / 2
start_θ = -π/2  # vertical orientation pointing downward (-90 degrees)

# PD gains for each actuated joint (raw encoder gains, converted internally by RExEel)
max_torque_per_joint = 2 * 9.3e6 / height
Kps_rex         = fill(2500.0, n_links - 1)
Kds_rex         = fill(500.0,  n_links - 1)
max_torques_rex = fill(max_torque_per_joint, n_links - 1)
stall_torques_rex = fill(max_torque_per_joint, n_links - 1)

rexeel = RExEel(time_step, n_links;
    bar_lengths = link_lengths,
    masses = masses_per_link,
    mois = moi_per_link,
    Kps = Kps_rex,
    Kds = Kds_rex,
    max_torques = max_torques_rex,
    stall_torques = stall_torques_rex,
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
## Define control inputs for C-start escape maneuver
#############################################################################################

# C-start kinematics inspired by Gazzola et al. (2012) "C-start: optimal start of larval fish"
#
# Phases:
# - Phase 1 (0 → T_prep): Preparatory stroke - body bends into C-shape
# - Phase 2 (T_prep → T_prep + T_prop): Propulsive stroke - traveling wave
#
# Control parameters:
# - B_prep: C-bend amplitude per joint during prep phase [rad] (head to tail)
# - K_prop: Undulation amplitude during propulsive phase [rad] (constant for all joints)
# - ψ_tail: Phase lag head-to-tail for traveling wave [rad]
# - φ: Initial phase offset [rad] (paper's optimal: 0.83 for 3D, 1.11 for 2D)
# - T_prep: Preparatory stroke duration [s]
# - T_prop: Propulsive phase duration [s]

# Number of joints (n_links - 1 = 4 joints for 5-link eel)
n_joints_ctrl = n_links - 1

# B_prep = deg2rad.([36, 36, 36, 36, 36]) # [rad] per joint
B_prep = deg2rad.([17.175683961250282, 17.77561203549385, 19.40557132265484, 23.105955851639145, 36.0])  # Example - update with actual values

# Undulation amplitude during propulsive phase (constant for all joints)
K_prop = deg2rad(25.0)  # [rad]

# Phase parameters
# ψ_tail = 2π * 0.7 # Total phase lag head-to-tail for traveling wave [rad]
# φ = -pi/4 # 1.11 - 2*pi*0.3  # Adjusted for T_prep=1.0s (originally 0.83 for T_prep=0.7s)
ψ_tail = 3.582384186876536    # Total phase lag head-to-tail for traveling wave [rad]
φ = -1.157910829319428 # 1.11 - 2*pi*0.3  # Adjusted for T_prep=1.0s (originally 0.83 for T_prep=0.7s)

# Timing
T_prep = 1.0   # Preparatory stroke duration [s]
T_prop = 1.0   # Propulsive phase duration [s]

control_params = [B_prep..., K_prop, ψ_tail, φ, T_prep, T_prop]

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

        # Phase lag (ψ): ramps during prep, creating traveling wave from the start
        ψ_factor = smooth_ramp(t, 0.0, T_prep, 0.0, 1.0)
        ψ_i = s_i * ψ_tail * ψ_factor

        # C-bend (B): ramps up during prep, then decays during propulsive stroke
        B_up = smooth_ramp(t, 0.0, T_prep, 0.0, 1.0)
        B_down = smooth_ramp(t, T_prep, T_prep + 2*T_prop, 1.0, 0.0)
        B_i = B_prep[i] * B_up * B_down

        # Undulation amplitude (K): ramps during prep
        K_i = K_prop * smooth_ramp(t, 0.0, T_prep, 0.0, 1.0)

        # Combined curvature: κ = B + K * sin(2πft - ψ + φ)
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
q_min[1] = start_x
q_min[2] = start_y
q_min[3] = start_θ
# Joint angles (4..end) start at zero for straight configuration

maximal_config = rex_eel_maximal_from_minimal(rexeel, q_min, n_links)
maximal_velocity = zeros(3 * n_links)
initial_body_state = vcat(maximal_config, maximal_velocity)

swimmer_initial_state = initialize_solid_state(rexeel, initial_body_state)

# test positional joint constraints satisfied
positional_residual = calculate_system_constraint_residual(
    rexeel,
    swimmer_initial_state[rexeel.configuration_indices],
)
@test positional_residual ≈ zeros(length(positional_residual)) atol=1e-8

println("Initial state check passed: all constraints satisfied")
println()

#############################################################################################
## Visualize control inputs over time
#############################################################################################

println("Generating control input visualization...")

# Generate time vector
t_viz = 0:0.01:final_time
n_viz = length(t_viz)

# Compute control inputs at each time step
control_viz = [calculate_control_input_from_params(rexeel, t, control_params) for t in t_viz]

# Extract joint angles (prescribed mode: control vector is θ directly)
n_joints = n_links - 1
θ_viz = hcat(control_viz...)'  # N_time × n_joints

# Plot desired joint angles
fig_ctrl, ax_ctrl = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel="Time (s)",
    ylabel="Desired Joint Angle (deg)",
    use_data_aspect=false
)

for i in 1:n_joints
    lines!(ax_ctrl, collect(t_viz), rad2deg.(θ_viz[:, i]), linewidth=2, label="Joint $i")
end

# Add phase markers
vlines!(ax_ctrl, [T_prep], color=:gray, linestyle=:dash, linewidth=1)
vlines!(ax_ctrl, [T_prep + T_prop], color=:gray, linestyle=:dash, linewidth=1)

axislegend(ax_ctrl, position=:rt)
display(fig_ctrl)
save(joinpath(vis_dir, "control_inputs.png"), fig_ctrl)
println("Control input plot saved.")

#############################################################################################
## Simulate solid system only (no fluid)
#############################################################################################

println("\nSimulating solid system (no fluid)...")

# Build explicit control trajectory for the solid-only simulation
solid_time_traj_precomputed = collect(0.0:time_step:final_time)
solid_control_trajectory = [
    calculate_control_input_from_params(rexeel, t, control_params)
    for t in solid_time_traj_precomputed[2:end]
]

solid_trajectories = simulate_solid_system(rexeel,
    swimmer_initial_state,
    final_time;
    control_trajectory = solid_control_trajectory,
    verbose = false,
)

solid_time_traj = solid_trajectories[:time_traj]
solid_configuration_traj = solid_trajectories[:configuration_traj]
solid_velocity_traj = solid_trajectories[:velocity_traj]
solid_state_traj = solid_trajectories[:system_state_traj]

solid_control_traj = [calculate_control_input_from_params(rexeel, t, control_params) for t in solid_time_traj]

solid_midpoint_state_traj = calculate_midpoint_state_trajectory(
    rexeel,
    solid_state_traj
)

println("Solid system simulation complete!")

# Plot actual vs desired joint angles for solid-only simulation
fig_solid_joints, ax_solid_joints = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel="Time (s)",
    ylabel="Joint Angle (deg)",
    use_data_aspect=false
)

# Extract actual joint angles from solid simulation
n_solid_time = length(solid_time_traj)
actual_solid_φ = zeros(n_solid_time, n_joints)

for k in 1:n_solid_time
    config = solid_configuration_traj[k]
    for i in 1:n_joints
        θ_i = config[3*i]      # Current link angle
        θ_ip1 = config[3*(i+1)]  # Next link angle
        actual_solid_φ[k, i] = θ_ip1 - θ_i
    end
end

# Plot actual joint angles (solid lines)
colors = [logocolors[1], logocolors[2], logocolors[3], logocolors[4], logocolors[1], logocolors[2]]
for i in 1:n_joints
    lines!(ax_solid_joints, solid_time_traj, rad2deg.(actual_solid_φ[:, i]),
        color=colors[i], linewidth=2, label="Actual φ$i")
end

# Plot desired joint angles (dashed lines)
desired_solid_φ = hcat([solid_control_traj[k] for k in 1:n_solid_time]...)'
for i in 1:n_joints
    lines!(ax_solid_joints, solid_time_traj, rad2deg.(desired_solid_φ[:, i]),
        color=colors[i], linewidth=2, linestyle=:dash, label="Desired φ$i")
end

# Add phase markers
vlines!(ax_solid_joints, [T_prep], color=:gray, linestyle=:dash, linewidth=1)
vlines!(ax_solid_joints, [T_prep + T_prop], color=:gray, linestyle=:dash, linewidth=1)

axislegend(ax_solid_joints, position=:rt)
display(fig_solid_joints)
save(joinpath(vis_dir, "solid_joint_angles.png"), fig_solid_joints)
println("Solid joint angles plot saved.")

# Animate solid system
fig_solid, ax_solid = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel="X (cm)",
    ylabel="Y (cm)",
    xlim=(0.0, length_x),
    ylim=(0.0, length_y),
    resolution=resolution,
    use_data_aspect=true
)

plot_solid_systems!(fig_solid, ax_solid, [rexeel], [solid_midpoint_state_traj[end]])
display(fig_solid)

clear_aquarium_axis!(ax_solid)
save_path_solid = joinpath(vis_dir, "solid_animation.mp4")
animate_solid_systems(fig_solid, ax_solid,
    [rexeel],
    solid_time_traj,
    [solid_midpoint_state_traj],
    save_path_solid;
    framerate=20,
    timescale=1.0,
)
println("Solid animation saved to: $save_path_solid")

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
## Simulate aquarium dynamics
#############################################################################################

println("Starting simulation...")
println()

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
    ilu_drop_tolerance=1e-3,
    newton_tolerance=1e-6,
    gmres_tolerance=1e-6,
    dual_regularization=1e-6,
    max_newton_iterations=20,
    gmres_memory=100,
    gmres_max_iterations=1000,
    calculate_control_input_from_params=calculate_control_input_from_params,
    verbose=false
)

println("\nSimulation complete!")

# Save simulation data
save_file = data_file("rexeel_c_start.jld2")
jldsave(save_file; trajectories)
println("Results saved to: ", save_file)
println()

#############################################################################################
## Load and process simulation data
#############################################################################################

# Load simulation data

load_file = data_file("rexeel_c_start.jld2")
data = load(load_file)
trajectories = data["trajectories"]

# Extract trajectories
time_traj = trajectories[:time_traj]
aquarium_state_traj = trajectories[:aquarium_state_traj]
fluid_state_traj = trajectories[:fluid_state_traj]
swimmer_state_traj = trajectories[:swimmer_state_traj]
control_traj = trajectories[:control_traj]

# Extract fluid velocity and swimmer configuration trajectories
fluid_velocity_traj = [extract_fluid_velocity(tank, aquarium_state_traj[k]) for k in 1:N_time]
swimmer_configuration_traj = [swimmer_state_traj[k][rexeel.configuration_indices] for k in 1:N_time]

#############################################################################################
## Plot joint angles over time
#############################################################################################

# Extract actual joint angles from configuration trajectory
n_joints = n_links - 1
actual_φ = zeros(N_time, n_joints)

for t in 1:N_time
    config = swimmer_configuration_traj[t]
    for i in 1:n_joints
        θ_i = config[3*i]        # Current link angle
        θ_ip1 = config[3*(i+1)]  # Next link angle
        actual_φ[t, i] = θ_ip1 - θ_i
    end
end

# Extract desired control angles
time_control = time_traj[2:end]
desired_φ = hcat([control_traj[t] for t in 1:N_time-1]...)'

# Create joint angle plot
joint_fig, ax_joint = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel="Time (s)",
    ylabel="Joint Angle (deg)",
    use_data_aspect=false
)

# Colors for each joint
colors = distinguishable_colors(n_joints, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

# Plot actual and desired joint angles
for i in 1:n_joints
    lines!(ax_joint, time_traj, rad2deg.(actual_φ[:, i]),
        color=colors[i], linewidth=2, label="Actual φ$i")
    lines!(ax_joint, time_control, rad2deg.(desired_φ[:, i]),
        color=colors[i], linewidth=2, linestyle=:dash, label="Desired φ$i")
end

axislegend(ax_joint, position=:rt)
display(joint_fig)
save(joinpath(vis_dir, "joint_angles.png"), joint_fig)

#############################################################################################
## Plot center of mass trajectory
#############################################################################################

# Extract center of mass position over time
com_x = zeros(N_time)
com_y = zeros(N_time)

for t in 1:N_time
    config = swimmer_configuration_traj[t]

    # Calculate center of mass (weighted average by mass)
    total_mass = sum(masses_per_link)
    com_x_weighted = 0.0
    com_y_weighted = 0.0

    # Loop through all links
    for i in 1:n_links
        xi = config[3*(i-1) + 1]  # x position of link i
        yi = config[3*(i-1) + 2]  # y position of link i
        com_x_weighted += masses_per_link[i] * xi
        com_y_weighted += masses_per_link[i] * yi
    end

    com_x[t] = com_x_weighted / total_mass
    com_y[t] = com_y_weighted / total_mass
end

# Plot COM trajectory
com_fig, ax_com = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel="X Position (cm)",
    ylabel="Y Position (cm)",
    xlim=(0.0, length_x),
    ylim=(0.0, length_y),
    resolution=resolution,
    use_data_aspect=true
)

lines!(ax_com, com_x, com_y, color=logocolors[1], linewidth=2, label="Center of Mass")
scatter!(ax_com, [com_x[1]], [com_y[1]], color=logocolors[3], markersize=15, label="Start")
scatter!(ax_com, [com_x[end]], [com_y[end]], color=logocolors[2], markersize=15, label="End")
axislegend(ax_com, position=:lb)
display(com_fig)
save(joinpath(vis_dir, "com_trajectory.png"), com_fig)

# Print displacement
println("Vertical displacement: $(com_y[end] - com_y[1]) cm")
println("Horizontal displacement: $(com_x[end] - com_x[1]) cm")
println("Direction: $(atan((com_y[end] - com_y[1]), (com_x[end] - com_x[1])) * 180 / π) degrees")
println()

#############################################################################################
## Plot center of mass velocity trajectories
#############################################################################################

# Extract center of mass velocity over time
com_vx = zeros(N_time)
com_vy = zeros(N_time)

for t in 1:N_time
    # Extract velocities from swimmer state
    velocities = swimmer_state_traj[t][rexeel.velocity_indices]

    # Calculate center of mass velocity (weighted average by mass)
    total_mass = sum(masses_per_link)
    com_vx_weighted = 0.0
    com_vy_weighted = 0.0

    # Loop through all links
    for i in 1:n_links
        vxi = velocities[3*(i-1) + 1]  # x velocity of link i
        vyi = velocities[3*(i-1) + 2]  # y velocity of link i
        com_vx_weighted += masses_per_link[i] * vxi
        com_vy_weighted += masses_per_link[i] * vyi
    end

    com_vx[t] = com_vx_weighted / total_mass
    com_vy[t] = com_vy_weighted / total_mass
end

# Calculate velocity magnitude
com_v_mag = sqrt.(com_vx.^2 .+ com_vy.^2)

# Plot COM velocity components
com_vel_fig, ax_com_vel = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel="Time (s)",
    ylabel="COM Velocity (cm/s)",
    use_data_aspect=false
)

lines!(ax_com_vel, time_traj, com_vx, color=logocolors[1], linewidth=2, label="Vₓ")
lines!(ax_com_vel, time_traj, com_vy, color=logocolors[2], linewidth=2, label="Vᵧ")
lines!(ax_com_vel, time_traj, com_v_mag, color=logocolors[4], linewidth=2, label="|V|")

# Add phase markers
vlines!(ax_com_vel, [T_prep], color=:gray, linestyle=:dash, linewidth=1)
vlines!(ax_com_vel, [T_prep + T_prop], color=:gray, linestyle=:dash, linewidth=1)

axislegend(ax_com_vel, position=:lt)
display(com_vel_fig)
save(joinpath(vis_dir, "com_velocity.png"), com_vel_fig)

# Print velocity statistics
println("Peak velocity magnitude: $(maximum(com_v_mag)) cm/s")
println("Final velocity magnitude: $(com_v_mag[end]) cm/s")
println("Final velocity components: vx=$(com_vx[end]) cm/s, vy=$(com_vy[end]) cm/s")
println()

#############################################################################################
## Plot swimmer orientation over time
#############################################################################################

# For a multi-link swimmer, we compute four orientation measures:
#   1. Head link angle (θ₁)
#   2. Mass-weighted mean angle (Σ mᵢθᵢ / Σ mᵢ)
#   3. Head-to-tail chord angle: atan2(y_tail - y_head, x_tail - x_head)
#   4. COM velocity heading: atan2(vy, vx)

head_angle = zeros(N_time)
mean_angle = zeros(N_time)
chord_angle = zeros(N_time)
velocity_heading = zeros(N_time)

total_mass = sum(masses_per_link)

for t in 1:N_time
    config = swimmer_configuration_traj[t]

    # Head link angle
    head_angle[t] = config[3]

    # Mass-weighted mean angle
    weighted_sum = 0.0
    for i in 1:n_links
        θ_i = config[3*i]
        weighted_sum += masses_per_link[i] * θ_i
    end
    mean_angle[t] = weighted_sum / total_mass

    # Head-to-tail chord angle
    x_head = config[1]
    y_head = config[2]
    x_tail = config[3*(n_links-1) + 1]
    y_tail = config[3*(n_links-1) + 2]
    chord_angle[t] = atan(y_tail - y_head, x_tail - x_head)

    # COM velocity heading
    velocity_heading[t] = atan(com_vy[t], com_vx[t])
end

# Unwrap angles to avoid discontinuities at ±π
function unwrap!(angles)
    for i in 2:length(angles)
        d = angles[i] - angles[i-1]
        if d > π
            angles[i:end] .-= 2π
        elseif d < -π
            angles[i:end] .+= 2π
        end
    end
end

unwrap!(head_angle)
unwrap!(mean_angle)
unwrap!(chord_angle)
unwrap!(velocity_heading)

# Plot orientation
orient_fig, ax_orient = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel="Time (s)",
    ylabel="Orientation (deg)",
    use_data_aspect=false
)

lines!(ax_orient, time_traj, rad2deg.(head_angle),
    color=logocolors[1], linewidth=2, label="Head link (θ₁)")
lines!(ax_orient, time_traj, rad2deg.(mean_angle),
    color=logocolors[2], linewidth=2, label="Mass-weighted mean")
lines!(ax_orient, time_traj, rad2deg.(chord_angle),
    color=logocolors[3], linewidth=2, label="Head-to-tail chord")
lines!(ax_orient, time_traj, rad2deg.(velocity_heading),
    color=logocolors[4], linewidth=2, linestyle=:dash, label="COM velocity heading")

vlines!(ax_orient, [T_prep], color=:gray, linestyle=:dash, linewidth=1)
vlines!(ax_orient, [T_prep + T_prop], color=:gray, linestyle=:dash, linewidth=1)

axislegend(ax_orient, position=:rt)
display(orient_fig)
save(joinpath(vis_dir, "swimmer_orientation.png"), orient_fig)

println("Orientation plot saved.")
println("  Initial head angle: $(rad2deg(head_angle[1]))°")
println("  Final head angle: $(rad2deg(head_angle[end]))°")
println("  Total head rotation: $(rad2deg(head_angle[end] - head_angle[1]))°")
println()

#############################################################################################
## Plot velocity field
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "X (cm)", ylabel = "Y (cm)",
    xlim = (0.0, length_x), ylim = (0.0, length_y),
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true,
    use_data_aspect=true
)

plot_velocity_field!(fig, ax,
    fluid_env,
    nothing, rexeel,
    fluid_velocity_traj[end],
    [], swimmer_state_traj[end];
    x_density=61,
    y_density=61,
    tipwidth=0.5,
    tiplength=0.2,
    tipcolor=:white,
    shaftwidth=0.2,
    shaftlength=1.0,
    shaftcolor=:white,
    lengthscale=1.0,
    normalize_velocity=true,
    smooth=true,
    smooth_sigma=3.0
)
display(fig)
save(joinpath(vis_dir, "rexeel_velocity_final.png"), fig)

# Animate velocity field
save_path = joinpath(vis_dir, "rexeel_velocity_animation.mp4")
animate_velocity_field(fig, ax,
    fluid_env,
    nothing, rexeel,
    time_traj,
    fluid_velocity_traj,
    [[]], swimmer_state_traj,
    save_path;
    x_density=61,
    y_density=61,
    tipwidth=0.5,
    tiplength=0.2,
    tipcolor=:white,
    shaftwidth=0.2,
    shaftlength=1.0,
    shaftcolor=:white,
    lengthscale=1.0,
    normalize_velocity=true,
    framerate=20,
    timescale=1.0,
    smooth=true,
    smooth_sigma=2.0
)

#############################################################################################
## Plot vorticity field
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "X (cm)", ylabel = "Y (cm)",
    xlim = (0.0, length_x), ylim = (0.0, length_y),
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true,
    use_data_aspect=true
)

plot_vorticity_field!(fig, ax,
    fluid_env,
    nothing, rexeel,
    fluid_velocity_traj[end],
    [], swimmer_state_traj[end];
    density=20,
    threshold_percentage=1.0,
    smooth=true,
    smooth_sigma=4.0
)

display(fig)
save(joinpath(vis_dir, "rexeel_vorticity_final.png"), fig)

# Animate vorticity field
save_path = joinpath(vis_dir, "rexeel_vorticity_animation.mp4")
animate_vorticity_field(fig, ax,
    fluid_env,
    nothing, rexeel,
    time_traj,
    fluid_velocity_traj,
    [[]], swimmer_state_traj,
    save_path;
    density=100,
    framerate=20,
    timescale=1.0,
    threshold_percentage=1.0,
    smooth=true,
    smooth_sigma=4.0
)

println("Visualization complete!")
println("Results saved to: ", vis_dir)

com_x_N = com_x[end]
com_y_N = com_y[end]
eel_length = sum(link_lengths)
mean_angle_N = mean_angle[end]

obj_1 = (1.6 - (com_x_N-50)/eel_length + cos(0 - mean_angle_N))^2

obj_2 = ((86 - com_x_N)/eel_length)^2 + (1-cos(-pi - mean_angle_N))^2 + ((com_y_N-61)/eel_length)^2