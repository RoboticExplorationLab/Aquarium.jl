import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.ForwardDiff
using Aquarium.CairoMakie
using Colors
using JLD2
using Test

vis_dir = joinpath(Aquarium.VIS_DIR, "rexeel")
mkpath(vis_dir)

#############################################################################################
## Plot params
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 600)
logocolors = Colors.JULIA_LOGO_COLORS

#############################################################################################
## Define 3-link RExEel
#############################################################################################

# time properties
time_step = 0.01
final_time = 10.0
N_time = Int(final_time/time_step) + 1

# eel properties
n_links = 3
link_lengths = [0.1, 0.1, 0.1]
masses_per_link = [0.2, 0.2, 0.2]
mois_per_link = [(1/12) * m * L^2 for (m, L) in zip(masses_per_link, link_lengths)]
gravity_constant = 0.0

# boundary properties
n_boundary_nodes_per_link = 8

# PD gains for each actuated joint
Kps = fill(100.0, n_links - 1)
Kds = fill(100.0, n_links - 1)

rexeel = RExEel(time_step, n_links;
    bar_lengths = link_lengths,
    masses = masses_per_link,
    mois = mois_per_link,
    Kps = Kps,
    Kds = Kds,
    n_boundary_nodes_per_link = n_boundary_nodes_per_link,
    gravity = [0.0, -gravity_constant],
    actuation_mode=:prescribed,
)

rexeel.plot_params[:bodycolor] = logocolors[3]
rexeel.plot_params[:linewidth] = 10.0
rexeel.plot_params[:showboundaryvelocities] = true
rexeel.plot_params[:arrowcolor] = logocolors[1]
rexeel.plot_params[:lengthscale] = 1.0
rexeel.plot_params[:showboundarynodes] = true
rexeel.plot_params[:boundarynodesize] = 20.0
rexeel.plot_params[:boundarynodecolor] = logocolors[2]

println("\nRExEel Configuration:")
for i in 1:n_links
    println("  Link $i: length=$(link_lengths[i])m, mass=$(masses_per_link[i])kg")
end
println("  Total length: $(sum(link_lengths))m")
println("  Total mass: $(sum(masses_per_link))kg")
println()

#############################################################################################
## Define initial RExEel state
#############################################################################################

q_min = zeros(n_links + 2)
q_min[1] = -sum(link_lengths)/2 + link_lengths[1]/2  # x position of head

maximal_config = rex_eel_maximal_from_minimal(rexeel, q_min, n_links)
maximal_velocity = zeros(3 * n_links)

initial_body_state = vcat(maximal_config, maximal_velocity)
full_system_state_0 = initialize_solid_state(rexeel, initial_body_state)

# test positional joint constraints satisfied
positional_residual = calculate_system_constraint_residual(
    rexeel,
    full_system_state_0[rexeel.configuration_indices],
)
@test positional_residual ≈ zeros(length(positional_residual)) atol=1e-8

#############################################################################################
## Define control inputs
#############################################################################################

control_params = [deg2rad(45), deg2rad(45), 0.5, 0.5]  # Initial joint angles (both 45 degrees)

function calculate_control_input_from_params(solid_system, t, control_params)

    amplitude1 = control_params[1]
    amplitude2 = control_params[2]
    frequency1 = control_params[3]
    frequency2 = control_params[4]

    θ1 = -amplitude1 * sin(2π * frequency1 * t)
    θ2 = amplitude2 * sin(2π * frequency2 * t)

    # Prescribed mode: return only desired joint angles [θ₁, θ₂]
    return [θ1, θ2]

end

# control_params = [deg2rad(45), 0, deg2rad(45), 0]  # [θ_des_1, ω_des_1, θ_des_2, ω_des_2]

# function calculate_control_input_from_params(solid_system, t, control_params)

#     return control_params

# end

#############################################################################################
## Simulate with Aquarium dynamics (variational integrator)
#############################################################################################

# Build explicit control trajectory (new simulate_solid_system expects a vector of controls,
# one per time step; old API accepted a user fn + control_params which is now inlined here).
time_traj_precomputed = collect(0.0:time_step:final_time)
control_trajectory_sim = [
    calculate_control_input_from_params(rexeel, t, control_params)
    for t in time_traj_precomputed[2:end]
]

trajectories = simulate_solid_system(rexeel,
    full_system_state_0,
    final_time;
    control_trajectory = control_trajectory_sim,
    verbose = false,
)

time_traj = trajectories[:time_traj]
configuration_traj = trajectories[:configuration_traj]
velocity_traj = trajectories[:velocity_traj]
state_traj = trajectories[:system_state_traj]

control_trajectory = [calculate_control_input_from_params(rexeel, t, control_params) for t in time_traj]

midpoint_state_traj = calculate_midpoint_state_trajectory(
    rexeel,
    state_traj
)

#############################################################################################
## Visualize RExEel trajectory
#############################################################################################

# Determine plot limits based on total eel length
total_length = sum(link_lengths)
plot_lim = 0.6 * total_length

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel = "X (m)", ylabel = "Y (m)",
    xlim=(-plot_lim, plot_lim),
    ylim=(-plot_lim, plot_lim),
    use_data_aspect=true
)
plot_solid_systems!(fig, ax, [rexeel], [midpoint_state_traj[end]])
display(fig)

clear_aquarium_axis!(ax)
save_path = joinpath(vis_dir, "rexeel_animation.mp4")
animate_solid_systems(fig, ax,
    [rexeel],
    time_traj,
    [midpoint_state_traj],
    save_path;
    framerate=20,
    timescale=1.0,
)

#############################################################################################
## Plot joint angles over time
#############################################################################################

# Extract actual joint angles from configuration trajectory
actual_φ1 = zeros(N_time)
actual_φ2 = zeros(N_time)

for t in 1:N_time
    config = configuration_traj[t]
    # Extract link angles
    θ1 = config[3]  # First link angle
    θ2 = config[6]  # Second link angle
    θ3 = config[9]  # Third link angle

    # Calculate relative angles
    actual_φ1[t] = θ2 - θ1  # Joint 1: relative angle between link 1 and 2
    actual_φ2[t] = θ3 - θ2  # Joint 2: relative angle between link 2 and 3
end

# Extract desired control angles (prescribed mode: control vector is θ directly)
desired_φ1 = [control_trajectory[t][1] for t in 1:N_time-1]
desired_φ2 = [control_trajectory[t][2] for t in 1:N_time-1]

# Create time vector for control (N_time-1 points)
time_control = time_traj[1:end-1]

# Create joint angle plot
joint_fig, ax_joint = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel="Time (s)",
    ylabel="Joint Angle (deg)",
    use_data_aspect=false
)

# Plot actual joint angles (solid lines)
lines!(ax_joint, time_traj, rad2deg.(actual_φ1),
    color=logocolors[3], linewidth=2, label="Actual φ₁")
lines!(ax_joint, time_traj, rad2deg.(actual_φ2),
    color=logocolors[2], linewidth=2, label="Actual φ₂")

# Plot desired control angles (dashed lines)
lines!(ax_joint, time_control, rad2deg.(desired_φ1),
    color=logocolors[3], linewidth=2, linestyle=:dash, label="Desired φ₁")
lines!(ax_joint, time_control, rad2deg.(desired_φ2),
    color=logocolors[2], linewidth=2, linestyle=:dash, label="Desired φ₂")

axislegend(ax_joint, position=:rt)
display(joint_fig)