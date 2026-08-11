include(joinpath(@__DIR__, "..", "common.jl"))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.ForwardDiff
using Aquarium.CairoMakie
using Colors
using JLD2
using Test

vis_dir = visualization_dir("actuated_pendulum")

#############################################################################################
## Plot params
#############################################################################################

background_color = :transparent
fontsize = 18
resolution = (800, 600)
logocolors = Colors.JULIA_LOGO_COLORS

#############################################################################################
## Define pendulum (new composition-based API)
##
## Under the new architecture, `ActuatedPendulum` is a constructor function that returns
## an `ActuatedSystem` composed of one `RigidBody{Bar}`, one `WorldPinJoint` for the
## hinge, and one `JointServoMotor` driving that joint via a `PDController`. All
## previously-configured `system_params` now live as direct struct fields on the returned
## system; gradients flow via `collect_differentiable_params` and
## `inject_differentiable_params`.
#############################################################################################

time_step = 0.01
final_time = 10.0
N_time = Int(final_time / time_step) + 1

# pendulum geometry / inertia
pendulum_length = 0.5
mass = 1.0
moi = (1 / 12) * mass * pendulum_length^2
gravity_constant = 9.81

# boundary properties
n_boundary_nodes = 5

# hinge position (differentiable, tracked via `WorldPinJoint.world_position`)
hinge_position = [0.0, 0.0]

pendulum = ActuatedPendulum(time_step;
    bar_length = pendulum_length,
    mass = mass,
    moi = moi,
    hinge_position = hinge_position,
    Kp = 100.0,
    Kd = 100.0,
    n_boundary_nodes = n_boundary_nodes,
    gravity = [0.0, -gravity_constant],
)

# Plot params are shared across all new systems via `system.plot_params::Dict{Symbol,Any}`.
pendulum.plot_params[:bodycolor] = logocolors[3]
pendulum.plot_params[:linewidth] = 10.0
pendulum.plot_params[:showboundaryvelocities] = true
pendulum.plot_params[:arrowcolor] = logocolors[1]
pendulum.plot_params[:lengthscale] = 1.0
pendulum.plot_params[:showboundarynodes] = true
pendulum.plot_params[:boundarynodesize] = 20.0
pendulum.plot_params[:boundarynodecolor] = logocolors[2]

#############################################################################################
## Define initial pendulum state
##
## The legacy `calculate_maximal_state_from_minimal` is gone; the new API exposes one
## converter per morphology. For a single-body pendulum, `pendulum_maximal_from_minimal`
## takes the single absolute-angle `[θ]` and returns the maximal `(x, y, θ)` configuration
## of the body center. Velocity is assembled manually from the rigid-body formula
## `v_center = ω × (center − hinge)`.
#############################################################################################

θ_0 = deg2rad(-45)
ω_0 = 0.0

max_config = pendulum_maximal_from_minimal(pendulum, [θ_0])

# Body-frame velocity from angular velocity: at the body center (which equals the COM
# under Interpretation P), v_center = ω × (center − hinge) in 2D.
body = pendulum.bodies[1]
center_minus_hinge = max_config[1:2] .- hinge_position
max_velocity = [-ω_0 * center_minus_hinge[2], ω_0 * center_minus_hinge[1], ω_0]

initial_body_state = vcat(max_config, max_velocity)
full_system_state_0 = initialize_solid_state(pendulum, initial_body_state)

# test hinge constraint satisfied
@test calculate_system_constraint_residual(
    pendulum,
    full_system_state_0[pendulum.configuration_indices],
) ≈ zeros(pendulum.n_constraints) atol = 1e-12

#############################################################################################
## Define control inputs
#############################################################################################

control_params = [deg2rad(45), 0.0]
control_trajectory = [control_params for _ in 1:(N_time - 1)]

#############################################################################################
## Simulate with Aquarium dynamics (variational integrator)
#############################################################################################

trajectories = simulate_solid_system(
    pendulum,
    full_system_state_0,
    final_time;
    control_trajectory = control_trajectory,
    verbose = false,
)
time_traj = trajectories[:time_traj]
configuration_traj = trajectories[:configuration_traj]
velocity_traj = trajectories[:velocity_traj]
state_traj = trajectories[:system_state_traj]

midpoint_state_traj = calculate_midpoint_state_trajectory(pendulum, state_traj)

#############################################################################################
## Visualize pendulum trajectory
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor = background_color,
    fontsize = fontsize,
    resolution = resolution,
    xlabel = "X", ylabel = "Y",
    xlim = (-1.1 * pendulum_length, 1.1 * pendulum_length),
    ylim = (-1.1 * pendulum_length, 1.1 * pendulum_length),
    use_data_aspect = true,
)
plot_solid_systems!(fig, ax, [pendulum], [midpoint_state_traj[end]])
maybe_display(fig)

clear_aquarium_axis!(ax)
save_path = joinpath(vis_dir, "actuated_pendulum_animation.mp4")
animate_if_enabled(animate_solid_systems, fig, ax,
    [pendulum],
    time_traj,
    [midpoint_state_traj],
    save_path;
    framerate = 20,
    timescale = 1.0,
)

#############################################################################################
## Plot pendulum angle over time
#############################################################################################

actual_angles = [configuration_traj[k][3] for k in 1:N_time]
desired_angles = [control_params[1] for k in 1:N_time-1]
time_control = time_traj[2:end]

angle_fig, ax_angle = create_aquarium_figure(;
    backgroundcolor = background_color,
    fontsize = fontsize,
    resolution = resolution,
    xlabel = "Time (s)",
    ylabel = "Angle (deg)",
    use_data_aspect = false,
)

lines!(ax_angle, time_traj, rad2deg.(actual_angles),
    color = logocolors[3], linewidth = 2, label = "Actual θ")

lines!(ax_angle, time_control, rad2deg.(desired_angles),
    color = logocolors[2], linewidth = 2, linestyle = :dash, label = "Target θ")

axislegend(ax_angle, position = :rt)
maybe_display(angle_fig)
