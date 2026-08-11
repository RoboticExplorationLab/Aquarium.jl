import Pkg
Pkg.activate(joinpath(@__DIR__,".."))
Pkg.develop(path=joinpath(@__DIR__,"..",".."))
Pkg.instantiate()

using AquariumClosed
using AquariumClosed.LinearAlgebra
using AquariumClosed.CairoMakie
using Pardiso
using Colors
using JLD2
using Test

vis_dir = joinpath(AquariumClosed.VIS_DIR, "lid_cavity_flow")
mkpath(vis_dir)

#############################################################################################
## Plot parameters
#############################################################################################

background_color=:transparent
fontsize=18
resolution=(800, 600)
logocolors = Colors.JULIA_LOGO_COLORS

#############################################################################################
## Define fluid environment
#############################################################################################

# time step
time_step = 10.0
final_time = 500.0
N = Int(final_time/time_step + 1)

# fluid properties
fluid_density = 1.0 # g/cm³
dynamic_viscosity = 0.01 # g/(cm·s)

# fluid grid
length_x = 10.0
length_y = 10.0

num_cells_x = 100
num_cells_y = 100

# boundary conditions
boundary_velocity = [1.0, 0.0]
boundary_condition_type = :lid_cavity

# calculate Reynolds number
reynolds_number = (fluid_density * boundary_velocity[1] * length_x) / dynamic_viscosity
println("Reynolds number: ", reynolds_number)

#############################################################################################
## Create fluid environment
#############################################################################################

fluid_env = Fluid(
    time_step;
    density = fluid_density,
    dynamic_viscosity = dynamic_viscosity,
    boundary_velocity = boundary_velocity,
    grid_size = (num_cells_x, num_cells_y),
    grid_dimensions = (length_x, length_y),
    boundary_condition_type = boundary_condition_type,
)

#############################################################################################
## Simulate
#############################################################################################

# initial conditions
fluid_state_0 = initialize_fluid_state(fluid_env, zeros(fluid_env.n_velocities))

# simulate using fluid only
fluid_sim_data = simulate_fluid(fluid_env, fluid_state_0, final_time;
    max_newton_iterations=10, solver_type=:gmres, scaling_type=:ruiz,
    preconditioner_type=:ilu, pivot_type=:metis,
    ilu_drop_tolerance=1e-2, gmres_tolerance=1e-8,
    newton_tolerance=1e-6, dual_regularization=1e-8, 
    lazy=true, verbose=false
)

save_file = joinpath(AquariumClosed.DATA_DIR, "lid_cavity_flow.jld2")
jldsave(save_file;
    fluid_sim_data
);

#############################################################################################
## Import sim results
#############################################################################################

data = load(joinpath(AquariumClosed.DATA_DIR, "lid_cavity_flow.jld2"))
fluid_sim_data = data["fluid_sim_data"]
t_traj = fluid_sim_data[:t_traj]
fluid_velocity_traj = fluid_sim_data[:fluid_velocity_traj]
fluid_pressure_traj = fluid_sim_data[:fluid_pressure_traj]

#############################################################################################
## test if boundary conditions are satisfied
#############################################################################################

fluid_velocity_end = fluid_velocity_traj[end]
vx_fluid_left = fluid_velocity_end[fluid_env.fvm_grid.vx_left_indices]
vx_fluid_right = fluid_velocity_end[fluid_env.fvm_grid.vx_right_indices]
vx_fluid_top = fluid_velocity_end[fluid_env.fvm_grid.vx_top_indices]
vx_fluid_bottom = fluid_velocity_end[fluid_env.fvm_grid.vx_bottom_indices]

vy_fluid_left = fluid_velocity_end[fluid_env.fvm_grid.vy_left_indices]
vy_fluid_right = fluid_velocity_end[fluid_env.fvm_grid.vy_right_indices]
vy_fluid_top = fluid_velocity_end[fluid_env.fvm_grid.vy_top_indices]
vy_fluid_bottom = fluid_velocity_end[fluid_env.fvm_grid.vy_bottom_indices]

@test all(vx_fluid_top .≈ boundary_velocity[1])
@test norm(vy_fluid_top, Inf) .≈ 0.0 atol = 1e-8
@test norm(vx_fluid_left, Inf) .≈ 0.0 atol = 1e-8
@test norm(vx_fluid_right, Inf) .≈ 0.0 atol = 1e-8
@test norm(vx_fluid_bottom, Inf) .≈ 0.0 atol = 1e-8
@test norm(vy_fluid_left, Inf) .≈ 0.0 atol = 1e-8
@test norm(vy_fluid_right, Inf) .≈ 0.0 atol = 1e-8
@test norm(vy_fluid_bottom, Inf) .≈ 0.0 atol = 1e-8

#############################################################################################
## Plot streamlines
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "X (cm)", ylabel = "Y (cm)",
    xlim = (0.0, length_x), ylim = (0.0, length_y),
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true
)
plot_streamlines!(fig, ax,
    fluid_env,
    nothing, nothing,
    fluid_velocity_traj[end],
    [], [];
    density=50
)
display(fig)

save_path = joinpath(vis_dir, "lid_cavity_streamlines_animation.mp4")
animate_streamlines(fig, ax,
    fluid_env,
    nothing, nothing,
    t_traj,
    fluid_velocity_traj,
    [[]], [[]],
    save_path;
    density=50,
    framerate=20,
    timescale=0.01,
)

#############################################################################################
## Plot vorticity
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "X (cm)", ylabel = "Y (cm)",
    xlim = (0.0, length_x), ylim = (0.0, length_y),
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true
)
plot_vorticity_field!(fig, ax,
    fluid_env,
    nothing, nothing,
    fluid_velocity_traj[end],
    [], [];
    density=100,
    min_threshold=-5.0,
    max_threshold=5.0
)
display(fig)

save_path = joinpath(vis_dir, "lid_cavity_vorticity_animation.mp4")
animate_vorticity_field(fig, ax,
    fluid_env,
    nothing, nothing,
    t_traj,
    fluid_velocity_traj,
    [[]], [[]],
    save_path;
    density=100,
    min_threshold=-10.0,
    max_threshold=10.0,
    framerate=20,
    timescale=0.01,
)

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
    ticksvisible=true
)
plot_velocity_field!(fig, ax,
    fluid_env,
    nothing, nothing,
    fluid_velocity_traj[end],
    [], [];
    x_density=25,
    y_density=25,
    tipwidth=0.5,
    tiplength=0.6,
    shaftwidth=0.1,
    lengthscale=1.0,
    tipcolor=logocolors.blue,
    shaftcolor=logocolors.blue,
)
display(fig)

save_path = joinpath(vis_dir, "lid_cavity_velocity_field_animation.mp4")
animate_velocity_field(fig, ax,
    fluid_env,
    nothing, nothing,
    t_traj,
    fluid_velocity_traj,
    [[]], [[]],
    save_path;
    x_density=25,
    y_density=25,
    tipwidth=0.5,
    tiplength=0.6,
    shaftwidth=0.1,
    lengthscale=1.0,
    tipcolor=logocolors.blue,
    shaftcolor=logocolors.blue,
    framerate=20,
    timescale=0.01,
)

#############################################################################################
## Plot pressure field
#############################################################################################

fig, ax = create_aquarium_figure(;
    backgroundcolor=background_color,
    fontsize=fontsize,
    xlabel = "X (cm)", ylabel = "Y (cm)",
    xlim = (0.0, length_x), ylim = (0.0, length_y),
    resolution=resolution,
    spinevisible=true,
    ticksvisible=true
)
plot_pressure_field!(fig, ax,
    fluid_env,
    nothing, nothing,
    fluid_pressure_traj[end],
    [], [];

)
display(fig)

save_path = joinpath(vis_dir, "lid_cavity_pressure_field_animation.mp4")
animate_pressure_field(fig, ax,
    fluid_env,
    nothing, nothing,
    t_traj,
    fluid_pressure_traj,
    [[]], [[]],
    save_path;
    framerate=20,
    timescale=0.01,
)