import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.ForwardDiff
using Aquarium.CairoMakie
using Colors
using JLD2
using Test

vis_dir = joinpath(Aquarium.VIS_DIR, "pendulum")
mkpath(vis_dir)

#############################################################################################
## Plot params
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 600)
logocolors = Colors.JULIA_LOGO_COLORS

#############################################################################################
## Define pendulum
#############################################################################################

# time properties
time_step = 0.01
final_time = 10.0
N_time = Int(final_time/time_step) + 1

# pendulum properties
pendulum_length = 0.5
mass = 5.0
moi = (1/12) * mass * pendulum_length^2    # thin-rod MOI about COM
gravity_constant = 9.81

# boundary properties
n_boundary_nodes = 5

# hinge position
hinge_position = [0.0, 0.0]

# spring and damping properties (equilibrium at θ=0, hanging straight down)
spring_stiffness = 5.0      # N⋅m/rad
damping_coefficient = 1.0   # N⋅m⋅s/rad

plot_params = Dict{Symbol, Any}(
    :bodycolor => logocolors[3],
    :linewidth => 10.0,
    :showboundarynodes => true,
    :boundarynodecolor => logocolors[2],
    :boundarynodesize => 20.0,
    :showboundaryvelocities => true,
    :arrowcolor => logocolors[1],
    :lengthscale => 0.1,
)

pendulum = Pendulum(time_step;
    bar_length = pendulum_length,
    mass = mass,
    moi = moi,
    hinge_position = hinge_position,
    equilibrium_angle = 0.0,
    stiffness = spring_stiffness,
    damping = damping_coefficient,
    n_boundary_nodes = n_boundary_nodes,
    gravity = [0.0, -gravity_constant],
    plot_params = plot_params,
)

#############################################################################################
## Define initial pendulum state
#############################################################################################

θ_0 = deg2rad(-45)
initial_configuration = pendulum_maximal_from_minimal(pendulum, [θ_0])
initial_body_state = vcat(initial_configuration, zeros(3))
full_system_state_0 = initialize_solid_state(pendulum, initial_body_state)

# test hinge constraint satisfied
@test calculate_system_constraint_residual(pendulum,
    full_system_state_0[pendulum.configuration_indices]
) ≈ zeros(pendulum.n_constraints)

#############################################################################################
## Simulate with Aquarium dynamics (variational integrator)
#############################################################################################

trajectories = simulate_solid_system(pendulum,
    full_system_state_0,
    final_time; verbose=false
)
time_traj = trajectories[:time_traj]
configuration_traj = trajectories[:configuration_traj]
velocity_traj = trajectories[:velocity_traj]
state_traj = trajectories[:system_state_traj]

midpoint_state_traj = calculate_midpoint_state_trajectory(
    pendulum,
    state_traj
)

#############################################################################################
## Visualize pendulum trajectory
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    resolution=resolution,
    xlabel = "X", ylabel = "Y",
    xlim=(-pendulum_length, pendulum_length),
    ylim=(-1.1*pendulum_length, 0.1*pendulum_length),
    use_data_aspect=true
)
plot_solid_systems!(fig, ax, [pendulum], [midpoint_state_traj[end]])

clear_aquarium_axis!(ax)
save_path = joinpath(vis_dir, "pendulum_animation.mp4")
animate_solid_systems(fig, ax,
    [pendulum],
    time_traj,
    [midpoint_state_traj],
    save_path;
    framerate=20,
    timescale=1.0,
)