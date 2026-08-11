include(joinpath(@__DIR__, "..", "common.jl"))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.CairoMakie
using Colors
using JLD2
using Test
using PGFPlotsX

vis_dir = visualization_dir("poiseuille_flow")

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
dt = 0.1
tf = 1.0
N = Int(tf/dt + 1)

# fluid properties (vegetable oil)
fluid_density = 0.92 # g/cm³
dynamic_viscosity = 0.8 # P
pressure_grad = (0.0, 0.0) # g/(cm²·s²)

# fluid grid
length_x = 3.0
length_y = 1.0

# boundary conditions
boundary_velocity = [1.0, 0.0]
boundary_condition_type = :channel_flow_theoretical

#############################################################################################
## Create fluid environment
#############################################################################################

fluid_env_30x10 = Fluid(
    dt;
    density = fluid_density,
    dynamic_viscosity = dynamic_viscosity,
    boundary_velocity = boundary_velocity,
    grid_size = (30, 10),
    grid_dimensions = (length_x, length_y),
    boundary_condition_type = boundary_condition_type,
    external_pressure_gradient=pressure_grad
);
fluid_env_150x50 = Fluid(
    dt;
    density = fluid_density,
    dynamic_viscosity = dynamic_viscosity,
    boundary_velocity = boundary_velocity,
    grid_size = (150, 50),
    grid_dimensions = (length_x, length_y),
    boundary_condition_type = boundary_condition_type,
    external_pressure_gradient=pressure_grad
);
fluid_env_300x100 = Fluid(
    dt;
    density = fluid_density,
    dynamic_viscosity = dynamic_viscosity,
    boundary_velocity = boundary_velocity,
    grid_size = (300, 100),
    grid_dimensions = (length_x, length_y),
    boundary_condition_type = boundary_condition_type,
    external_pressure_gradient=pressure_grad
);

#############################################################################################
## Simulate
#############################################################################################

# initial conditions
fluid_velocity_0_30x10 = initialize_fluid_state(fluid_env_30x10, zeros(fluid_env_30x10.fvm_grid.n_v))
fluid_velocity_0_150x50 = initialize_fluid_state(fluid_env_150x50, zeros(fluid_env_150x50.fvm_grid.n_v))
fluid_velocity_0_300x100 = initialize_fluid_state(fluid_env_300x100, zeros(fluid_env_300x100.fvm_grid.n_v))

# simulate using fluid only
fluid_sim_data_30x10 = simulate_fluid(fluid_env_30x10, fluid_velocity_0_30x10, tf;
    max_newton_iterations=10, newton_tolerance=1e-6,
    preconditioner_type=:ilu, gmres_tolerance=1e-8,
    ilu_drop_tolerance=1e-2, dual_regularization=1e-6, verbose=false
)
fluid_sim_data_150x50 = simulate_fluid(fluid_env_150x50, fluid_velocity_0_150x50, tf;
    max_newton_iterations=10, newton_tolerance=1e-6,
    preconditioner_type=:ilu, gmres_tolerance=1e-8,
    ilu_drop_tolerance=1e-2, dual_regularization=1e-6, verbose=false
)
fluid_sim_data_300x100 = simulate_fluid(fluid_env_300x100, fluid_velocity_0_300x100, tf;
    max_newton_iterations=10, newton_tolerance=1e-6,
    preconditioner_type=:ilu, gmres_tolerance=1e-8,
    ilu_drop_tolerance=1e-2, dual_regularization=1e-6, verbose=false
)

save_file = data_file("poiseuille_flow.jld2")
jldsave(save_file;
    fluid_sim_data_30x10,
    fluid_sim_data_150x50,
    fluid_sim_data_300x100
);

#############################################################################################
## Import sim results
#############################################################################################

data = load(data_file("poiseuille_flow.jld2"))
fluid_sim_data_30x10 = data["fluid_sim_data_30x10"]
fluid_sim_data_150x50 = data["fluid_sim_data_150x50"]
fluid_sim_data_300x100 = data["fluid_sim_data_300x100"]

t_traj_30x10 = fluid_sim_data_30x10[:t_traj]
fluid_velocity_traj_30x10 = fluid_sim_data_30x10[:fluid_velocity_traj]
fluid_pressure_traj_30x10 = fluid_sim_data_30x10[:fluid_pressure_traj]

t_traj_150x50 = fluid_sim_data_150x50[:t_traj]
fluid_velocity_traj_150x50 = fluid_sim_data_150x50[:fluid_velocity_traj]
fluid_pressure_traj_150x50 = fluid_sim_data_150x50[:fluid_pressure_traj]

t_traj_300x100 = fluid_sim_data_300x100[:t_traj]
fluid_velocity_traj_300x100 = fluid_sim_data_300x100[:fluid_velocity_traj]
fluid_pressure_traj_300x100 = fluid_sim_data_300x100[:fluid_pressure_traj]

#############################################################################################
## test if boundary conditions are satisfied
#############################################################################################

fluid_velocity_end = fluid_velocity_traj_300x100[end]
vx_fluid_left = fluid_velocity_end[fluid_env_300x100.fvm_grid.vx_left_indices]
vx_fluid_right = fluid_velocity_end[fluid_env_300x100.fvm_grid.vx_right_indices]
vx_fluid_top = fluid_velocity_end[fluid_env_300x100.fvm_grid.vx_top_indices]
vx_fluid_bottom = fluid_velocity_end[fluid_env_300x100.fvm_grid.vx_bottom_indices]

vy_fluid_left = fluid_velocity_end[fluid_env_300x100.fvm_grid.vy_left_indices]
vy_fluid_right = fluid_velocity_end[fluid_env_300x100.fvm_grid.vy_right_indices]
vy_fluid_top = fluid_velocity_end[fluid_env_300x100.fvm_grid.vy_top_indices]
vy_fluid_bottom = fluid_velocity_end[fluid_env_300x100.fvm_grid.vy_bottom_indices]

@test norm(vx_fluid_top, Inf) < 1e-6
@test norm(vx_fluid_bottom, Inf) < 1e-6
@test norm(vy_fluid_left, Inf) < 1e-6
@test norm(vy_fluid_top, Inf) < 1e-6
@test norm(vy_fluid_bottom, Inf) < 1e-6

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
    fluid_env_300x100,
    nothing, nothing,
    fluid_velocity_traj_300x100[end],
    [], []
)
display(fig)

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
    fluid_env_300x100,
    nothing, nothing,
    fluid_velocity_traj_300x100[end],
    [], [];
    density=100,
    min_threshold=-10.0,
    max_threshold=10.0
)
display(fig)

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
    fluid_env_300x100,
    nothing, nothing,
    fluid_velocity_traj_300x100[end],
    [], [];
    x_density=4,
    y_density=20,
    tipwidth=0.05,
    tiplength=0.06,
    shaftwidth=0.02,
    lengthscale=0.5,
    tipcolor=logocolors.blue,
    shaftcolor=logocolors.blue,
)
display(fig)

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
    fluid_env_300x100,
    nothing, nothing,
    fluid_pressure_traj_300x100[end],
    [], [];

)
display(fig)

#############################################################################################
## plot energy over time
#############################################################################################

# calculate kinetic energy
fluid_energy_traj = [calculate_total_energy(
    fluid_env_300x100,
    fluid_velocity_traj_300x100[i]
) for i in 1:N]

# plot kinetic energy
fig, ax = create_aquarium_figure(;
    fontsize = 24,
    xlabel = "Time (s)", ylabel = "Kinetic Energy (erg)",
    use_data_aspect=false
)
lines!(ax, 0:dt:tf, fluid_energy_traj)
display(fig)

#############################################################################################
## plot pressure norms over time
#############################################################################################

# calculate kinetic energy
fluid_pressure_norm_traj = [norm(fluid_pressure_traj_300x100[i]) for i in 1:N-1]

# plot kinetic energy on log scale
fig, ax = create_aquarium_figure(;
    fontsize = 24,
    xlabel = "Time (s)", ylabel = "Pressure Norm (dyn/cm²)",
    use_data_aspect=false
)
lines!(ax, dt:dt:tf, fluid_pressure_norm_traj, label="Pressure (Fluid)")
display(fig)

#############################################################################################
## plot velocity profiles
#############################################################################################

fluid_velocity_x_theoretical(y) = 4.0 * boundary_velocity[1] * y * (length_y - y) / (length_y^2)

fluid_velocity_x_grid_30x10, _ = create_fluid_velocity_grid(fluid_env_30x10.fvm_grid,
    fluid_velocity_traj_30x10[end]
)
fluid_velocity_x_grid_150x50, _ = create_fluid_velocity_grid(fluid_env_150x50.fvm_grid,
    fluid_velocity_traj_150x50[end]
)
fluid_velocity_x_grid_300x100, _ = create_fluid_velocity_grid(fluid_env_300x100.fvm_grid,
    fluid_velocity_traj_300x100[end]
)

fluid_velocity_x_true = fluid_velocity_x_theoretical.(LinRange(0, length_y, 300))
fluid_velocity_x_profile_30x10 = fluid_velocity_x_grid_30x10[:, end]
fluid_velocity_x_profile_150x50 = fluid_velocity_x_grid_150x50[:, end]
fluid_velocity_x_profile_300x100 = fluid_velocity_x_grid_300x100[:, end]

fig, ax = create_aquarium_figure(; backgroundcolor=:white,
    axiscolor = :black, fontsize = 24,
    xlabel = "Velocity (cm/s)", ylabel = "y (cm)",
    use_data_aspect=false
)
lines!(ax, fluid_velocity_x_true,
    LinRange(0, length_y, 300),
    color=:orange, label="Ground Truth"
)
scatter!(ax, fluid_velocity_x_profile_30x10[2:end-1], 
    fluid_env_30x10.fvm_grid.y_coord_vx_values[2:end-1],
    color=:green, label="30x10 Grid"
)
scatter!(ax, fluid_velocity_x_profile_150x50[2:end-1],
    fluid_env_150x50.fvm_grid.y_coord_vx_values[2:end-1],
    color=:blue, label="150x50 Grid"
)
scatter!(ax, fluid_velocity_x_profile_300x100[2:end-1],
    fluid_env_300x100.fvm_grid.y_coord_vx_values[2:end-1],
    color=:red, label="300x100 Grid"
)

xlims!(ax, -0.01, 1.1*boundary_velocity[1])

axislegend(ax, backgroundcolor=:transparent, labelcolor=:black, framecolor=:black)
display(fig)

#############################################################################################
## Make tikz plot of velocity profiles
#############################################################################################

ground_truth_color = colorant"black"
color1 = RGBA(214/255, 172/255, 23/255, 1.0)
color2 = RGBA(178/255, 104/255, 218/255, 1.0)
color3 = RGBA(0/255, 186/255, 88/255, 1.0)

lineopts = @pgf {no_marks, "very thick"}
y_history_plot = @pgf PGFPlotsX.Axis(
    {
        xmajorgrids,
        ymajorgrids,
        xlabel = "\$v^{f}\$ (cm/s)",
        ylabel = "Y (cm)",
        legend_pos = "north east",
        legend_columns=2,
        legend_cell_align="left",
        xmin = -1,
        xmax = 15,
        ymin = -0.1,
        ymax = 1.25,
        
    },
    PlotInc({lineopts..., color=ground_truth_color},
        Coordinates(fluid_velocity_x_true,
        LinRange(0, length_y, 300))),
    PlotInc({color=color1, mark="star"},
        Coordinates(fluid_velocity_x_profile_30x10[2:end-1], 
        fluid_env_30x10.fvm_grid.y_coord_vx_values[2:end-1])),
    PlotInc({mark="square", color=color2},
        Coordinates(fluid_velocity_x_profile_150x50[2:end-1],
        fluid_env_150x50.fvm_grid.y_coord_vx_values[2:end-1])),
    PlotInc({mark="o", color=color3},
        Coordinates(fluid_velocity_x_profile_300x100[2:end-1],
        fluid_env_300x100.fvm_grid.y_coord_vx_values[2:end-1])),

    PGFPlotsX.Legend(["Ground Truth", "30x10 Grid", "150x50 Grid", "300x100 Grid"])
)

filename = joinpath(vis_dir, "poiseuille_flow_convergence_study.tikz");
pgfsave(filename, y_history_plot, include_preamble=false);