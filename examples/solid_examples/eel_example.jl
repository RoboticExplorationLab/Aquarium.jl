include(joinpath(@__DIR__, "..", "common.jl"))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.ForwardDiff
using Aquarium.CairoMakie
using Colors
using JLD2
using Test

vis_dir = visualization_dir("eel")

#############################################################################################
## Plot params
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 600)
logocolors = Colors.JULIA_LOGO_COLORS

#############################################################################################
## Define eel (multi-link pendulum with joint springs)
#############################################################################################

# time properties
time_step = 0.01
final_time = 10.0
N_time = Int(final_time/time_step) + 1

# eel properties
n_links = 3
link_lengths = [0.5, 0.5, 0.5]
masses_per_link = [2.0, 2.0, 2.0]
mois_per_link = [(1/12) * m * L^2 for (m, L) in zip(masses_per_link, link_lengths)]
gravity_constant = 0.0

# Joint spring and damping properties (one per joint)
joint_stiffnesses = fill(0.1, n_links - 1)  # N⋅m/rad
joint_dampings = fill(0.005, n_links - 1)   # N⋅m⋅s/rad

# boundary properties
n_boundary_nodes_per_link = 5

eel = Eel(time_step, n_links;
    bar_lengths = link_lengths,
    masses = masses_per_link,
    mois = mois_per_link,
    stiffnesses = joint_stiffnesses,
    dampings = joint_dampings,
    n_boundary_nodes_per_link = n_boundary_nodes_per_link,
    gravity = [0.0, -gravity_constant],
)

eel.plot_params[:bodycolor] = logocolors[3]
eel.plot_params[:linewidth] = 10.0
eel.plot_params[:showboundaryvelocities] = true
eel.plot_params[:arrowcolor] = logocolors[1]
eel.plot_params[:lengthscale] = 1.0
eel.plot_params[:showboundarynodes] = true
eel.plot_params[:boundarynodesize] = 20.0
eel.plot_params[:boundarynodecolor] = logocolors[2]

println("\nEel Configuration:")
for i in 1:n_links
    println("  Link $i: length=$(link_lengths[i])m, mass=$(masses_per_link[i])kg")
end
println("  Total length: $(sum(link_lengths))m")
println("  Total mass: $(sum(masses_per_link))kg")
println("  Joint stiffnesses: $joint_stiffnesses N⋅m/rad")
println("  Joint dampings: $joint_dampings N⋅m⋅s/rad")
println()

#############################################################################################
## Define initial eel state using minimal coordinates
#############################################################################################

# Minimal coordinates: [x1, y1, θ1, θ2, ..., θ_n]
# First three entries are (x, y, θ) of body 1's center; later entries are absolute
# angles of bodies 2..n.
q_min = zeros(n_links + 2)
q_min[1] = 0.0                  # x1
q_min[2] = 0.0                  # y1
q_min[3] = deg2rad(20.0)        # θ1
q_min[4] = deg2rad(-20.0)       # θ2 (absolute)
q_min[5] = deg2rad(20.0)        # θ3 (absolute)

maximal_config = eel_maximal_from_minimal(eel, q_min, n_links)

# Zero initial velocity
maximal_velocity = zeros(3 * n_links)

initial_body_state = vcat(maximal_config, maximal_velocity)
full_system_state_0 = initialize_solid_state(eel, initial_body_state)

# Test that constraints are satisfied
constraint_residual = calculate_system_constraint_residual(
    eel,
    full_system_state_0[eel.configuration_indices],
)

println("Number of links: ", n_links)
println("Number of constraints: ", eel.n_constraints)
println("Constraint residual norm: ", norm(constraint_residual))
@test norm(constraint_residual) < 1e-6

#############################################################################################
## Simulate with Aquarium dynamics (variational integrator)
#############################################################################################

println("\nSimulating eel dynamics...")

trajectories = simulate_solid_system(eel,
    full_system_state_0,
    final_time; verbose=false
)
time_traj = trajectories[:time_traj]
configuration_traj = trajectories[:configuration_traj]
velocity_traj = trajectories[:velocity_traj]
state_traj = trajectories[:system_state_traj]

midpoint_state_traj = calculate_midpoint_state_trajectory(
    eel,
    state_traj
)

N = length(time_traj)

println("Simulation complete!")

#############################################################################################
## Animate motion
#############################################################################################

# Create animation with plot limits based on total eel length
total_length = sum(link_lengths)
plot_lim_x = 1.2 * total_length
plot_lim_y = 0.6 * total_length

aquarium_fig, aquarium_ax = create_aquarium_figure(resolution=resolution,
    backgroundcolor=background_color,
    xlabel = "X (m)", ylabel = "Y (m)",
    xlim=(-0.5, plot_lim_x),
    ylim=(-plot_lim_y, plot_lim_y),
    use_data_aspect=true,
    fontsize=fontsize
)
plot_solid_systems!(aquarium_fig, aquarium_ax, [eel], [midpoint_state_traj[end]])
maybe_display(aquarium_fig)

# Animate (subsample for faster rendering)
clear_aquarium_axis!(aquarium_ax)
save_path = joinpath(vis_dir, "eel_animation.mp4")
animate_if_enabled(animate_solid_systems, aquarium_fig, aquarium_ax,
    [eel],
    time_traj,
    [midpoint_state_traj],
    save_path;
    framerate=20,
    timescale=1.0,
)