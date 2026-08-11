import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.ForwardDiff
using Aquarium.CairoMakie
using Colors
using JLD2
using Test

vis_dir = joinpath(Aquarium.VIS_DIR, "double_pendulum")
mkpath(vis_dir)

#############################################################################################
## Plot params
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 600)
logocolors = Colors.JULIA_LOGO_COLORS

#############################################################################################
## Define double pendulum
#############################################################################################

# time properties
time_step = 0.01
final_time = 10.0
N_time = Int(final_time/time_step) + 1

# double pendulum properties
link_length = 0.5  # length of each link
mass = 5.0  # mass of each link
moi = (1/12) * mass * link_length^2
gravity_constant = 9.81

# boundary properties
n_boundary_nodes = 5

# hinge properties
hinge_position = [0.0, 0.0]

double_pendulum = DoublePendulum(time_step;
    bar_lengths = [link_length, link_length],
    masses = [mass, mass],
    mois = [moi, moi],
    hinge_position = hinge_position,
    n_boundary_nodes_per_link = n_boundary_nodes,
    gravity = [0.0, -gravity_constant],
)

double_pendulum.plot_params[:bodycolor] = logocolors[3]
double_pendulum.plot_params[:linewidth] = 10.0
double_pendulum.plot_params[:showboundaryvelocities] = true
double_pendulum.plot_params[:arrowcolor] = logocolors[1]
double_pendulum.plot_params[:lengthscale] = 0.02
double_pendulum.plot_params[:showboundarynodes] = true
double_pendulum.plot_params[:boundarynodesize] = 20.0
double_pendulum.plot_params[:boundarynodecolor] = logocolors[2]

#############################################################################################
## Define initial double pendulum state
#############################################################################################

# Initial absolute angles for both links
θ1_0 = deg2rad(45)
θ2_0 = deg2rad(45)

maximal_config_0 = double_pendulum_maximal_from_minimal(double_pendulum, [θ1_0, θ2_0])

# Initial velocities (starting from rest)
maximal_velocity_0 = zeros(6)

initial_body_state = vcat(maximal_config_0, maximal_velocity_0)
full_system_state_0 = initialize_solid_state(double_pendulum, initial_body_state)

# Test that hinge constraints are satisfied
@test calculate_system_constraint_residual(
    double_pendulum,
    full_system_state_0[double_pendulum.configuration_indices]
) ≈ zeros(double_pendulum.n_constraints) atol=1e-10

#############################################################################################
## Simulate with Aquarium dynamics (variational integrator)
#############################################################################################

trajectories = simulate_solid_system(
    double_pendulum,
    full_system_state_0,
    final_time;
    verbose=false
)

time_traj = trajectories[:time_traj]
configuration_traj = trajectories[:configuration_traj]
velocity_traj = trajectories[:velocity_traj]
state_traj = trajectories[:system_state_traj]

midpoint_state_traj = calculate_midpoint_state_trajectory(
    double_pendulum,
    state_traj
)

#############################################################################################
## Calculate and plot total energy trajectory
#############################################################################################

# Calculate total energy at each time step using the built-in function
total_energy_traj = [calculate_total_energy(double_pendulum, state) for state in state_traj]

# Plot energy trajectory
energy_fig, energy_ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=(800, 600),
    xlabel="Time (s)",
    ylabel="Energy (J)"
)

lines!(energy_ax, time_traj, total_energy_traj, label="Total Energy", linewidth=2)
display(energy_fig)

# Save energy plot
energy_save_path = joinpath(vis_dir, "double_pendulum_energy.png")
save(energy_save_path, energy_fig)
println("Energy plot saved to: $energy_save_path")

# Calculate energy drift
initial_energy = total_energy_traj[1]
final_energy = total_energy_traj[end]
energy_drift = final_energy - initial_energy
relative_energy_drift = energy_drift / initial_energy * 100

println("Initial Energy: $initial_energy J")
println("Final Energy: $final_energy J")
println("Energy Drift: $energy_drift J ($(relative_energy_drift)%)")

#############################################################################################
## Visualize double pendulum trajectory
#############################################################################################

# Set up plot limits to show full range of motion
xlim_range = 2.5 * link_length
ylim_range = 2.5 * link_length

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel="X",
    ylabel="Y",
    xlim=(-xlim_range, xlim_range),
    ylim=(-ylim_range, ylim_range),
    use_data_aspect=true
)

# Plot final state
plot_solid_systems!(fig, ax, [double_pendulum], [midpoint_state_traj[2]])

# Create animation
clear_aquarium_axis!(ax)
save_path = joinpath(vis_dir, "double_pendulum_animation.mp4")
animate_solid_systems(
    fig,
    ax,
    [double_pendulum],
    time_traj,
    [midpoint_state_traj],
    save_path;
    framerate=20,
    timescale=1.0,
)