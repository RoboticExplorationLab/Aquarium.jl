import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Aquarium
using Aquarium.LinearAlgebra
using Aquarium.CairoMakie
using Colors
using JLD2

vis_dir = joinpath(Aquarium.VIS_DIR, "case_studies/immersed_boundary_comparison")
data_dir = joinpath(Aquarium.DATA_DIR, "case_studies/immersed_boundary_comparison")
mkpath(vis_dir)
mkpath(data_dir)

#############################################################################################
## Parameters
#############################################################################################

# Bar
bar_length = 1.0
n_nodes_fine = 11   # ds = 1.0/10 = 0.1 = 2h  (well-resolved)
n_nodes_coarse = 6    # ds = 1.0/3  ≈ 0.33 = 6.7h (under-resolved)

# Flow
U_freestream = 1.0
fluid_density = 1.0
Re = 100.0               # Re = U * bar_length / nu
dynamic_viscosity = fluid_density * U_freestream * bar_length / Re   # = 0.01

# Grid (h = 0.05, fine enough for Re=100 flow)
length_x = 5.0 * bar_length   # 5.0 — streamwise
length_y = 2.0 * bar_length   # 2.0 — transverse
h = 0.05
num_cells_x = round(Int, length_x / h)   # 100
num_cells_y = round(Int, length_y / h)   # 40

# Time
dt = 0.05
final_time = 10.0  # ~10 convective times (L_final / U)

# Bar placement (stationary, vertical)
bar_x = 1.0         # 1/5 of domain from inlet
bar_y = length_y / 2  # centered transversely

#############################################################################################
## Helper functions
#############################################################################################

function create_fluid()
    Fluid(
        dt;
        density = fluid_density,
        dynamic_viscosity = dynamic_viscosity,
        boundary_velocity = [U_freestream, 0.0],
        grid_size = (num_cells_x, num_cells_y),
        grid_dimensions = (length_x, length_y),
        boundary_condition_type = :freestream,
    )
end

function create_bar(; n_nodes, ib_method)
    plot_params = Dict{Symbol, Any}(
        :bodycolor => RGB(0.9451, 0.6745, 0.09020),
        :linewidth => ib_method == :original ? 0 : 5.0,
        :showboundarynodes => true,
        :boundarynodecolor => RGB(0.0, 0.7294, 0.3451),
        :boundarynodesize => 16.0,
    )
    FreeBar(
        dt;
        bar_length = bar_length,
        mass = 1.0,
        moi = 1 / 12,
        n_boundary_nodes = n_nodes,
        ib_method = ib_method,
        discrete_delta_kind = :three_point,
        gravity = [0.0, 0.0],
        plot_params = plot_params,
    )
end

#############################################################################################
## Run simulations
#############################################################################################

cases = [
    (ib_method = :original,  n_nodes = n_nodes_fine,   label = "original_fine"),
    (ib_method = :original,  n_nodes = n_nodes_coarse, label = "original_coarse"),
    (ib_method = :weak_form, n_nodes = n_nodes_fine,   label = "weak_form_fine"),
    (ib_method = :weak_form, n_nodes = n_nodes_coarse, label = "weak_form_coarse"),
]

for case in cases
    println("\n" * "="^80)
    println("Running: $(case.label)  (ib=$(case.ib_method), n=$(case.n_nodes))")
    println("="^80)

    # Build tank
    fluid_env = create_fluid()
    bluff_body = create_bar(; n_nodes = case.n_nodes, ib_method = case.ib_method)
    tank = AquariumTank_only_bluff_body(fluid_env, bluff_body)

    # Bluff body state: stationary vertical bar
    bluff_body_state = [bar_x, bar_y, π / 2, 0.0, 0.0, 0.0]

    # Initialize fluid to freestream
    fluid_initial_vx = U_freestream * ones(fluid_env.fvm_grid.n_vx)
    fluid_initial_vy = zeros(fluid_env.fvm_grid.n_vy)
    fluid_initial_velocity = vcat(fluid_initial_vx, fluid_initial_vy)
    aquarium_state_0 = initialize_aquarium_state(tank, fluid_initial_velocity)

    # Simulate
    trajectories = simulate_aquarium(
        tank,
        aquarium_state_0,
        final_time,
        bluff_body_state;
        is_midpoint_bluff_body = false,
        pivot_type = :rcm,
        scaling_type = :ruiz,
        solver_type = :gmres,
        preconditioner_type = :ilu,
        lazy = false,
        ilu_drop_tolerance = 1e-2,
        newton_tolerance = 1e-6,
        gmres_tolerance = 1e-8,
        dual_regularization = 1e-8,
        max_newton_iterations = 10,
        gmres_memory = 100,
        gmres_max_iterations = 1000,
        verbose = false,
    )

    # Save
    save_file = joinpath(data_dir, "$(case.label).jld2")
    jldsave(save_file; tank, trajectories, bluff_body_state)
    println("Saved: $save_file")
end

#############################################################################################
## Streamline animations
#############################################################################################

for case in cases
    println("\nAnimating: $(case.label)")

    # Load
    save_file = joinpath(data_dir, "$(case.label).jld2")
    data = load(save_file)
    tank = data["tank"]
    trajectories = data["trajectories"]
    bluff_body_state = data["bluff_body_state"]

    fluid_env = tank.fluid
    bluff_body = tank.bluff_body

    time_traj = trajectories[:time_traj]
    aquarium_state_traj = trajectories[:aquarium_state_traj]
    bluff_body_state_traj = trajectories[:bluff_body_state_traj]
    N = length(time_traj)

    fluid_velocity_traj = [extract_fluid_velocity(tank, aquarium_state_traj[i]) for i in 1:N]

    # Create figure (no axes, labels, ticks; resolution matches 5:2 domain)
    fig, ax = create_aquarium_figure(;
        spinevisible = false,
        ticksvisible = false,
        resolution = (1000, 400),
        xlim = (0.0, length_x),
        ylim = (0.0, length_y),
    )

    # Animate (10s video: timescale = final_time / 10)
    save_path = joinpath(vis_dir, "$(case.label)_streamlines.mp4")
    animate_streamlines(
        fig, ax,
        fluid_env,
        bluff_body, nothing,
        time_traj,
        fluid_velocity_traj,
        bluff_body_state_traj, [[]],
        save_path;
        colormap = [RGB(0.0, 0.565, 0.914), RGB(0.0, 0.565, 0.914)],
        density = 20,
        framerate = 30,
        timescale = final_time / 10.0,
    )
    println("Saved: $save_path")

    # Final-state SVG: white background, black axes/labels, Times New Roman
    # Flip segment/node colors relative to the animation.
    bluff_body.plot_params[:bodycolor], bluff_body.plot_params[:boundarynodecolor] =
        bluff_body.plot_params[:boundarynodecolor], bluff_body.plot_params[:bodycolor]

    fig_svg, ax_svg = create_aquarium_figure(;
        backgroundcolor = :transparent,
        axiscolor = :black,
        font = "Times New Roman",
        fontsize = 28,
        xlabel = "x (cm)",
        ylabel = "y (cm)",
        xlim = (0.0, length_x),
        ylim = (0.0, length_y),
        resolution = (1000, 400),
    )
    plot_streamlines!(
        fig_svg, ax_svg,
        fluid_env,
        bluff_body, nothing,
        fluid_velocity_traj[end],
        bluff_body_state_traj[end], [];
        colormap = [RGB(0.0, 0.565, 0.914), RGB(0.0, 0.565, 0.914)],
        density = 20,
    )
    svg_path = joinpath(vis_dir, "$(case.label)_final.svg")
    save(svg_path, fig_svg)
    println("Saved: $svg_path")
end

println("\n" * "="^80)
println("All done! Animations saved to: $vis_dir")
println("="^80)
