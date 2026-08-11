import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using AquariumClosed
using AquariumClosed.LinearAlgebra
using AquariumClosed.ForwardDiff
using AquariumClosed.CairoMakie
using Colors
using JLD2
using Test

vis_dir = joinpath(AquariumClosed.VIS_DIR, "rexeel_forward_swimming")
mkpath(vis_dir)

#############################################################################################
## Plot params
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 800)
logocolors = Colors.JULIA_LOGO_COLORS

blue_sim = RGB(0.0, 0.565, 0.914)  # jj_blue
green_sim = RGB(0.0, 0.7294, 0.3451) # jj_green

#############################################################################################
## Define fluid domain (4ft x 4ft tank with wall boundaries)
#############################################################################################

# time properties
time_step = 0.02
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

# Starting position - eel starts on left side of tank, oriented horizontally (swimming right)
start_x = length_x / 2
start_y = length_y / 2
start_θ = -π/2  # horizontal orientation pointing right

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

rexeel.plot_params[:bodycolor] = green_sim
rexeel.plot_params[:linewidth] = 10.0
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
println("  Starting position: ($start_x, $start_y) with horizontal orientation $(rad2deg(start_θ))°")
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
## Define control inputs for forward swimming (traveling wave)
#############################################################################################

# Forward swimming using traveling wave propulsion
# Based on the propulsive phase from C-start, but without preparatory C-bend
#
# Control parameters:
# - K_prop: Undulation amplitude [rad] (constant for all joints)
# - ψ_tail: Phase lag head-to-tail for traveling wave [rad]
# - f: Undulation frequency [Hz]
# - T_ramp: Ramp-up time to reach full amplitude [s]

n_joints_ctrl = n_links - 1

# Undulation amplitude during swimming (constant for all joints)
K_prop = deg2rad(40.0)  # [rad] - slightly larger amplitude for stronger propulsion

# Phase parameters for traveling wave
ψ_tail = 2π * 0.7  # Total phase lag head-to-tail [rad] - one full wavelength

# Frequency
f = 0.5  # Undulation frequency [Hz]

# Ramp-up time
T_ramp = 0.5/f  # Time to reach full amplitude [s]

control_params = [K_prop, ψ_tail, f, T_ramp]

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

        # Phase lag: constant phase shift for traveling wave
        ψ_i = s_i * ψ_tail

        # Undulation amplitude: ramps up during T_ramp period
        K_i = K_prop * smooth_ramp(t, 0.0, T_ramp, 0.0, 1.0)

        # Traveling wave: θ = K * sin(2πft - ψ)
        θ_joints[i] = K_i * sin(2π * f * t - ψ_i)
    end

    # Prescribed mode: return only desired joint angles [θ₁, θ₂, ..., θ_n]
    return θ_joints
end

#############################################################################################
## Define initial RExEel state
#############################################################################################

# Start with eel in straight configuration at rest
q_min = zeros(n_links + 2)
q_min[1] = start_x
q_min[2] = start_y
q_min[3] = start_θ
# Joint angles (4..end) stay at 0 for straight configuration

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

# Add ramp-up marker
vlines!(ax_ctrl, [T_ramp], color=:gray, linestyle=:dash, linewidth=1)

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

# Add ramp-up marker
vlines!(ax_solid_joints, [T_ramp], color=:gray, linestyle=:dash, linewidth=1)

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

println("\nSimulation complete!")

# Save simulation data
save_file = joinpath(AquariumClosed.DATA_DIR, "rexeel_forward_swimming.jld2")
jldsave(save_file; trajectories)
println("Results saved to: ", save_file)
println()

#############################################################################################
## Load and process simulation data
#############################################################################################

# Load simulation data
load_file = joinpath(AquariumClosed.DATA_DIR, "rexeel_forward_swimming.jld2")
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
    total_mass = sum(masses_per_link)

    # Calculate center of mass (weighted average by mass for all links)
    for i in 1:n_links
        x_i = config[3*(i-1) + 1]
        y_i = config[3*(i-1) + 2]
        com_x[t] += masses_per_link[i] * x_i / total_mass
        com_y[t] += masses_per_link[i] * y_i / total_mass
    end
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
println("Forward (X) displacement: $(com_x[end] - com_x[1]) cm")
println("Lateral (Y) displacement: $(com_y[end] - com_y[1]) cm")
println()

# Extract COM velocity from analytical velocities
com_vx = zeros(N_time)
com_vy = zeros(N_time)

for t in 1:N_time
    state = swimmer_state_traj[t]
    total_mass = sum(masses_per_link)

    # Sum mass-weighted velocities for each link
    for i in 1:n_links
        vx_i = state[rexeel.velocity_indices[3*(i-1) + 1]]
        vy_i = state[rexeel.velocity_indices[3*(i-1) + 2]]
        com_vx[t] += masses_per_link[i] * vx_i / total_mass
        com_vy[t] += masses_per_link[i] * vy_i / total_mass
    end
end

com_speed = sqrt.(com_vx.^2 + com_vy.^2)

# Plot COM velocity over time
vel_fig, ax_vel = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel="Time (s)",
    ylabel="COM Velocity (cm/s)",
    use_data_aspect=false
)

lines!(ax_vel, time_traj, com_vx, color=logocolors[1], linewidth=2, label="vₓ (forward)")
lines!(ax_vel, time_traj, com_vy, color=logocolors[2], linewidth=2, label="vᵧ (lateral)")
lines!(ax_vel, time_traj, com_speed, color=logocolors[3], linewidth=2, label="speed")

# Add ramp-up marker
vlines!(ax_vel, [T_ramp], color=:gray, linestyle=:dash, linewidth=1)

axislegend(ax_vel, position=:rt)
display(vel_fig)
save(joinpath(vis_dir, "com_velocity.png"), vel_fig)

println("Max COM speed: $(maximum(com_speed)) cm/s")
println("Final forward velocity: $(com_vx[end]) cm/s")
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
    x_density=30,
    y_density=30,
    tipwidth=0.5,
    tiplength=0.5,
    tipcolor=blue_sim,
    shaftwidth=0.2,
    shaftlength=1.0,
    shaftcolor=blue_sim,
    lengthscale=2.0,
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