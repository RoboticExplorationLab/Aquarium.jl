function animate_velocity_field(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    t_traj::AbstractVector,
    v_fluid_traj::VecOrMat{<:AbstractVector},
    x_bluff_body_traj::VecOrMat{<:AbstractVector},
    x_swimmer_traj::VecOrMat{<:AbstractVector},
    save_path::String;
    x_density=fluid.fvm_grid.n_cell_x-1,
    y_density=fluid.fvm_grid.n_cell_y-1,
    tipwidth=0.5,
    tiplength=0.2,
    tipcolor=:white,
    shaftwidth=0.2,
    shaftlength=1.0,
    shaftcolor=:white,
    lengthscale=1.0,
    normalize_velocity=false,
    framerate=30,
    timescale=1.0,
    smooth=false,
    smooth_sigma=1.5
)

    # create time trajectory of animation
    dt_animation = 1/(framerate*timescale)
    t_animation_traj = t_traj[1]:dt_animation:t_traj[end]

    p = Progress(length(t_animation_traj), 1, "Creating animation...")

    # create trajectory interpolants
    v_fluid_traj_interpolant = create_trajectory_interpolant(t_traj, v_fluid_traj)

    # Create trajectory interpolants for bodies if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body_traj)
        x_bluff_body_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_bluff_body_traj)
    end

    if !isnothing(swimmer) && !isempty(x_swimmer_traj)
        x_swimmer_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_swimmer_traj)
    end

    record(aquarium_fig, save_path, t_animation_traj, framerate=framerate) do tk

        # Clear previous plots
        empty!(aquarium_ax)

        # Get fluid velocity at current time
        v_fluid_k = v_fluid_traj_interpolant(tk)

        # Get body states if present
        x_bluff_body_k = (!isnothing(bluff_body) && !isempty(x_bluff_body_traj)) ?
            x_bluff_body_trajectory_interpolant(tk) : []
        x_swimmer_k = (!isnothing(swimmer) && !isempty(x_swimmer_traj)) ?
            x_swimmer_trajectory_interpolant(tk) : []

        # Use the existing plot function
        plot_velocity_field!(aquarium_fig, aquarium_ax, fluid, bluff_body, swimmer,
            v_fluid_k, x_bluff_body_k, x_swimmer_k;
            x_density=x_density, y_density=y_density,
            tipwidth=tipwidth, tiplength=tiplength,
            tipcolor=tipcolor, shaftwidth=shaftwidth,
            shaftlength=shaftlength, shaftcolor=shaftcolor,
            lengthscale=lengthscale, normalize_velocity=normalize_velocity,
            smooth=smooth, smooth_sigma=smooth_sigma
        )

        next!(p)

    end

end

function animate_pressure_field(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    t_traj::AbstractVector,
    p_fluid_traj::VecOrMat{<:AbstractVector},
    x_bluff_body_traj::VecOrMat{<:AbstractVector},
    x_swimmer_traj::VecOrMat{<:AbstractVector},
    save_path::String;
    colormap=:berlin,
    density=10,
    framerate=30,
    timescale=1.0,
)

    # create time trajectory of animation
    dt_animation = 1/(framerate*timescale)
    t_animation_traj = t_traj[1]:dt_animation:t_traj[end]

    p = Progress(length(t_animation_traj), 1, "Creating animation...")

    # create trajectory interpolants
    p_fluid_traj_interpolant = create_trajectory_interpolant(t_traj, p_fluid_traj)

    # Create trajectory interpolants for bodies if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body_traj)
        x_bluff_body_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_bluff_body_traj)
    end

    if !isnothing(swimmer) && !isempty(x_swimmer_traj)
        x_swimmer_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_swimmer_traj)
    end

    record(aquarium_fig, save_path, t_animation_traj, framerate=framerate) do tk

        # Clear previous plots
        empty!(aquarium_ax)

        # Get fluid pressure at current time
        p_fluid_k = p_fluid_traj_interpolant(tk)

        # Get body states if present
        x_bluff_body_k = (!isnothing(bluff_body) && !isempty(x_bluff_body_traj)) ?
            x_bluff_body_trajectory_interpolant(tk) : []
        x_swimmer_k = (!isnothing(swimmer) && !isempty(x_swimmer_traj)) ?
            x_swimmer_trajectory_interpolant(tk) : []

        # Use the existing plot function
        plot_pressure_field!(aquarium_fig, aquarium_ax, fluid, bluff_body, swimmer,
            p_fluid_k, x_bluff_body_k, x_swimmer_k;
            colormap=colormap, density=density
        )

        next!(p)

    end

end

function animate_vorticity_field(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    t_traj::AbstractVector,
    v_fluid_traj::VecOrMat{<:AbstractVector},
    x_bluff_body_traj::VecOrMat{<:AbstractVector},
    x_swimmer_traj::VecOrMat{<:AbstractVector},
    save_path::String;
    colormap=:berlin,
    x_density=fluid.fvm_grid.n_cell_x-1,
    y_density=fluid.fvm_grid.n_cell_y-1,
    density=10,
    threshold_percentage=0.1,  # Use 10% of max vorticity as threshold
    min_threshold=nothing,     # Optional: manually override threshold
    max_threshold=nothing,     # Optional: manually override threshold
    framerate=30,
    timescale=1.0,
    smooth=false,
    smooth_sigma=1.5
)

    # create time trajectory of animation
    dt_animation = 1/(framerate*timescale)
    t_animation_traj = t_traj[1]:dt_animation:t_traj[end]

    p = Progress(length(t_animation_traj), 1, "Creating animation...")

    # create trajectory interpolants
    v_fluid_traj_interpolant = create_trajectory_interpolant(t_traj, v_fluid_traj)

    # Create trajectory interpolants for bodies if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body_traj)
        x_bluff_body_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_bluff_body_traj)
    end

    if !isnothing(swimmer) && !isempty(x_swimmer_traj)
        x_swimmer_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_swimmer_traj)
    end

    # Compute percentage-based threshold across entire trajectory if not manually specified
    if isnothing(min_threshold) || isnothing(max_threshold)
        fvm_grid = fluid.fvm_grid
        x_plot_cell_values = Vector(LinRange(0.0, fvm_grid.length_x, x_density))
        y_plot_cell_values = Vector(LinRange(0.0, fvm_grid.length_y, y_density))
        
        max_vorticity_traj = 0.0
        for v_fluid in v_fluid_traj
            vorticity_grid_temp = interpolate_fluid_vorticity_grid(
                fvm_grid, v_fluid, x_plot_cell_values, y_plot_cell_values;
                smooth=smooth, smooth_sigma=smooth_sigma
            )
            max_vorticity_traj = max(max_vorticity_traj, maximum(abs.(vorticity_grid_temp)))
        end
        
        threshold = threshold_percentage * max_vorticity_traj
        min_threshold = isnothing(min_threshold) ? -threshold : min_threshold
        max_threshold = isnothing(max_threshold) ? threshold : max_threshold
    end

    record(aquarium_fig, save_path, t_animation_traj, framerate=framerate) do tk

        # Clear previous plots
        empty!(aquarium_ax)

        # Get fluid velocity at current time
        v_fluid_k = v_fluid_traj_interpolant(tk)

        # Get body states if present
        x_bluff_body_k = (!isnothing(bluff_body) && !isempty(x_bluff_body_traj)) ?
            x_bluff_body_trajectory_interpolant(tk) : []
        x_swimmer_k = (!isnothing(swimmer) && !isempty(x_swimmer_traj)) ?
            x_swimmer_trajectory_interpolant(tk) : []

        # Use the existing plot function
        plot_vorticity_field!(aquarium_fig, aquarium_ax, fluid, bluff_body, swimmer,
            v_fluid_k, x_bluff_body_k, x_swimmer_k;
            colormap=colormap, x_density=x_density, y_density=y_density,
            density=density, min_threshold=min_threshold, max_threshold=max_threshold,
            smooth=smooth, smooth_sigma=smooth_sigma
        )

        next!(p)

    end

end


function animate_q_criterion_field(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    t_traj::AbstractVector,
    v_fluid_traj::VecOrMat{<:AbstractVector},
    x_bluff_body_traj::VecOrMat{<:AbstractVector},
    x_swimmer_traj::VecOrMat{<:AbstractVector},
    save_path::String;
    colormap=:RdBu,
    x_density=fluid.fvm_grid.n_cell_x-1,
    y_density=fluid.fvm_grid.n_cell_y-1,
    density=10,
    threshold_percentage=0.1,  # Use 10% of max Q-criterion as threshold
    min_threshold=nothing,     # Optional: manually override threshold
    max_threshold=nothing,     # Optional: manually override threshold
    framerate=30,
    timescale=1.0,
    smooth=false,
    smooth_sigma=1.5
)

    # create time trajectory of animation
    dt_animation = 1/(framerate*timescale)
    t_animation_traj = t_traj[1]:dt_animation:t_traj[end]

    p = Progress(length(t_animation_traj), 1, "Creating animation...")

    # create trajectory interpolants
    v_fluid_traj_interpolant = create_trajectory_interpolant(t_traj, v_fluid_traj)

    # Create trajectory interpolants for bodies if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body_traj)
        x_bluff_body_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_bluff_body_traj)
    end

    if !isnothing(swimmer) && !isempty(x_swimmer_traj)
        x_swimmer_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_swimmer_traj)
    end

    # Compute percentage-based threshold across entire trajectory if not manually specified
    if isnothing(min_threshold) || isnothing(max_threshold)
        fvm_grid = fluid.fvm_grid
        x_plot_cell_values = Vector(LinRange(0.0, fvm_grid.length_x, x_density))
        y_plot_cell_values = Vector(LinRange(0.0, fvm_grid.length_y, y_density))
        
        max_q_traj = 0.0
        for v_fluid in v_fluid_traj
            q_criterion_grid_temp = interpolate_fluid_q_criterion_grid(
                fvm_grid, v_fluid, x_plot_cell_values, y_plot_cell_values;
                smooth=smooth, smooth_sigma=smooth_sigma
            )
            max_q_traj = max(max_q_traj, maximum(abs.(q_criterion_grid_temp)))
        end
        
        threshold = threshold_percentage * max_q_traj
        min_threshold = isnothing(min_threshold) ? -threshold : min_threshold
        max_threshold = isnothing(max_threshold) ? threshold : max_threshold
    end

    record(aquarium_fig, save_path, t_animation_traj, framerate=framerate) do tk

        # Clear previous plots
        empty!(aquarium_ax)

        # Get fluid velocity at current time
        v_fluid_k = v_fluid_traj_interpolant(tk)

        # Get body states if present
        x_bluff_body_k = (!isnothing(bluff_body) && !isempty(x_bluff_body_traj)) ?
            x_bluff_body_trajectory_interpolant(tk) : []
        x_swimmer_k = (!isnothing(swimmer) && !isempty(x_swimmer_traj)) ?
            x_swimmer_trajectory_interpolant(tk) : []

        # Use the existing plot function
        plot_q_criterion_field!(aquarium_fig, aquarium_ax, fluid, bluff_body, swimmer,
            v_fluid_k, x_bluff_body_k, x_swimmer_k;
            colormap=colormap, x_density=x_density, y_density=y_density,
            density=density, min_threshold=min_threshold, max_threshold=max_threshold,
            smooth=smooth, smooth_sigma=smooth_sigma
        )

        next!(p)

    end

end


function animate_streamlines(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    t_traj::AbstractVector,
    v_fluid_traj::VecOrMat{<:AbstractVector},
    x_bluff_body_traj::VecOrMat{<:AbstractVector},
    x_swimmer_traj::VecOrMat{<:AbstractVector},
    save_path::String;
    colormap=:imola,
    x_density=fluid.fvm_grid.n_cell_x-1,
    y_density=fluid.fvm_grid.n_cell_y-1,
    density=10.0,
    linewidth=1.5,
    arrowsize=5,
    framerate=30,
    timescale=1.0,
)

    # create time trajectory of animation
    dt_animation = 1/(framerate*timescale)
    t_animation_traj = t_traj[1]:dt_animation:t_traj[end]

    p = Progress(length(t_animation_traj), 1, "Creating animation...")

    # create trajectory interpolants
    v_fluid_traj_interpolant = create_trajectory_interpolant(t_traj, v_fluid_traj)

    # Create trajectory interpolants for bodies if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body_traj)
        x_bluff_body_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_bluff_body_traj)
    end

    if !isnothing(swimmer) && !isempty(x_swimmer_traj)
        x_swimmer_trajectory_interpolant = create_trajectory_interpolant(t_traj, x_swimmer_traj)
    end

    record(aquarium_fig, save_path, t_animation_traj, framerate=framerate) do tk

        # Clear previous plots
        empty!(aquarium_ax)

        # Get fluid velocity at current time
        v_fluid_k = v_fluid_traj_interpolant(tk)

        # Get body states if present
        x_bluff_body_k = (!isnothing(bluff_body) && !isempty(x_bluff_body_traj)) ?
            x_bluff_body_trajectory_interpolant(tk) : []
        x_swimmer_k = (!isnothing(swimmer) && !isempty(x_swimmer_traj)) ?
            x_swimmer_trajectory_interpolant(tk) : []

        # Use the existing plot function
        plot_streamlines!(aquarium_fig, aquarium_ax, fluid, bluff_body, swimmer,
            v_fluid_k, x_bluff_body_k, x_swimmer_k;
            colormap=colormap, x_density=x_density, y_density=y_density,
            density=density, linewidth=linewidth, arrowsize=arrowsize
        )

        next!(p)

    end

end

function animate_solid_systems(aquarium_fig::Figure,
    aquarium_ax::Axis,
    systems::AbstractVector{<:SolidSystem},
    t_traj::AbstractVector,
    xr_list_traj::AbstractVector{<:VecOrMat{<:AbstractVector}},
    save_path::String;
    framerate=30,
    timescale=1.0,
)

    # create time trajectory of animation
    dt_animation = 1/(framerate*timescale)
    t_animation_traj = t_traj[1]:dt_animation:t_traj[end]

    p = Progress(length(t_animation_traj), 1, "Creating animation...")

    # create trajectory interpolants for each body
    xr_interpolants = [create_trajectory_interpolant(t_traj, xr_list_traj[i]) for i in eachindex(systems)]

    record(aquarium_fig, save_path, t_animation_traj, framerate=framerate) do tk

        # Clear previous plots
        for i in length(aquarium_ax.scene):-1:1
            delete!(aquarium_ax, aquarium_ax.scene[i])
        end

        # plot systems onto the provided axis
        if !isempty(systems)
            for i in eachindex(systems)
                system_i = systems[i]
                xr_i = xr_interpolants[i](tk)
                plot_solid_system!(aquarium_ax, system_i, xr_i)
            end
        end

        next!(p)

    end
    
end