############################
## create Aquarium figure
############################

function create_aquarium_figure(; backgroundcolor=:transparent,
    fontsize=18,
    font = "Times New Roman",
    xlabel = "x",
    ylabel = "y",
    axiscolor = :white,
    spinevisible = true,
    ticksvisible = true,
    resolution=(800, 600),
    xlim=nothing,
    ylim=nothing,
    scale=identity,
    use_data_aspect=true
)

    # plot velocity field
    set_theme!(backgroundcolor = backgroundcolor,
        fontsize=fontsize,
        Axis = (xticksvisible = ticksvisible,
            yticksvisible = ticksvisible,
            xticklabelsvisible = ticksvisible,
            yticklabelsvisible = ticksvisible,
            xlabelvisible = ticksvisible,
            ylabelvisible = ticksvisible,
            bottomspinevisible = spinevisible,
            leftspinevisible = spinevisible,
            rightspinevisible = spinevisible,
            topspinevisible = spinevisible,
            xgridvisible= false,
            ygridvisible= false,
            xticklabelcolor = axiscolor,
            yticklabelcolor = axiscolor,
            xtickcolor = axiscolor,
            ytickcolor = axiscolor,
            xlabelcolor = axiscolor,
            ylabelcolor = axiscolor,
            bottomspinecolor = axiscolor,
            leftspinecolor = axiscolor,
            rightspinecolor = axiscolor,
            topspinecolor = axiscolor,
            backgroundcolor = backgroundcolor
    ))

    fig = Figure(size = resolution,
        fonts = (; regular=font, weird=font)
    )
    ax = Axis(fig[1,1], xlabel=xlabel, ylabel=ylabel,
        xlabelfont=:regular, ylabelfont=:regular,
        yscale=scale
    )

    if xlim !== nothing
        xlims!(ax, xlim[1], xlim[2])
    end
    if ylim !== nothing
        ylims!(ax, ylim[1], ylim[2])
    end

    if use_data_aspect
        ax.aspect = DataAspect()
    end

    return fig, ax

end

function clear_aquarium_axis!(aquarium_ax::Axis)
    
    for i in length(aquarium_ax.scene):-1:1
        delete!(aquarium_ax, aquarium_ax.scene[i])
    end

    return nothing

end

###############################
## plot Aquarium environment
###############################

function plot_velocity_field!(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    v_fluid::AbstractVector,
    x_bluff_body::AbstractVector,
    x_swimmer::AbstractVector;
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
    smooth=false,
    smooth_sigma=1.5
)

    # extract fvm_grid from fluid
    fvm_grid = fluid.fvm_grid

    # determine grid points for plotting
    x_plot_cell_values = Vector(
        LinRange(0.0, fvm_grid.length_x, x_density)
    )
    y_plot_cell_values = Vector(
        LinRange(0.0, fvm_grid.length_y, y_density)
    )

    v_fluid_x_grid, v_fluid_y_grid = interpolate_fluid_velocity_grid(
        fvm_grid, v_fluid, x_plot_cell_values, y_plot_cell_values;
        smooth=smooth, smooth_sigma=smooth_sigma
    )

    if normalize_velocity

        mag = sqrt.(v_fluid_x_grid.^2 .+ v_fluid_y_grid.^2)
        v_fluid_x_grid ./= mag
        v_fluid_y_grid ./= mag

    end

    arrows2d!(aquarium_ax, x_plot_cell_values,
        y_plot_cell_values, v_fluid_x_grid', v_fluid_y_grid',
        tipwidth=tipwidth, tiplength=tiplength,
        shaftwidth=shaftwidth, shaftlength=shaftlength,
        lengthscale=lengthscale,
        tipcolor=tipcolor, shaftcolor=shaftcolor,
    )

    # Plot bluff body if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body)
        plot_solid_system!(aquarium_ax, bluff_body, x_bluff_body)
    end

    # Plot swimmer if present
    if !isnothing(swimmer) && !isempty(x_swimmer)
        plot_solid_system!(aquarium_ax, swimmer, x_swimmer)
    end

    return v_fluid_x_grid, v_fluid_y_grid

end

function plot_pressure_field!(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    pf::AbstractVector,
    x_bluff_body::AbstractVector,
    x_swimmer::AbstractVector;
    colormap=:berlin,
    density=10,
)

    fvm_grid = fluid.fvm_grid
    pf_grid = create_pressure_grid(fvm_grid, pf)

    # plot pressure contours onto the provided axis
    contourf!(aquarium_ax,
        fvm_grid.x_coord_cell_values,
        fvm_grid.y_coord_cell_values,
        pf_grid',
        levels=density, colormap=colormap,
        extendlow = :auto, extendhigh = :auto
    )

    # Plot bluff body if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body)
        plot_solid_system!(aquarium_ax, bluff_body, x_bluff_body)
    end

    # Plot swimmer if present
    if !isnothing(swimmer) && !isempty(x_swimmer)
        plot_solid_system!(aquarium_ax, swimmer, x_swimmer)
    end

    return pf_grid

end

function plot_vorticity_field!(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    v_fluid::AbstractVector,
    x_bluff_body::AbstractVector,
    x_swimmer::AbstractVector;
    colormap=:berlin,
    x_density=fluid.fvm_grid.n_cell_x-1,
    y_density=fluid.fvm_grid.n_cell_y-1,
    density=10,
    threshold_percentage=0.1,  # Use 10% of max vorticity as threshold
    min_threshold=nothing,     # Optional: manually override threshold
    max_threshold=nothing,     # Optional: manually override threshold
    smooth=false,
    smooth_sigma=1.5
)

    fvm_grid = fluid.fvm_grid

    # make meshgrid
    x_plot_cell_values = Vector(
        LinRange(0.0, fvm_grid.length_x, x_density)
    )
    y_plot_cell_values = Vector(
        LinRange(0.0, fvm_grid.length_y, y_density)
    )

    # calculate vorticity (with optional smoothing of velocity fields)
    vorticity_grid = interpolate_fluid_vorticity_grid(
        fvm_grid, v_fluid, x_plot_cell_values, y_plot_cell_values;
        smooth=smooth, smooth_sigma=smooth_sigma
    )

    # Compute percentage-based threshold if not manually specified
    if isnothing(min_threshold) || isnothing(max_threshold)
        max_vorticity = maximum(abs.(vorticity_grid))
        threshold = threshold_percentage * max_vorticity
        min_threshold = isnothing(min_threshold) ? -threshold : min_threshold
        max_threshold = isnothing(max_threshold) ? threshold : max_threshold
    end

    # plot vorticity contours onto the provided axis
    contourf!(aquarium_ax,
        x_plot_cell_values,
        y_plot_cell_values,
        vorticity_grid',
        levels=range(min_threshold, max_threshold, density),
        colormap=colormap,
        extendlow = :auto,
        extendhigh = :auto,
    )

    # Plot bluff body if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body)
        plot_solid_system!(aquarium_ax, bluff_body, x_bluff_body)
    end

    # Plot swimmer if present
    if !isnothing(swimmer) && !isempty(x_swimmer)
        plot_solid_system!(aquarium_ax, swimmer, x_swimmer)
    end

    return vorticity_grid

end


function plot_q_criterion_field!(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    v_fluid::AbstractVector,
    x_bluff_body::AbstractVector,
    x_swimmer::AbstractVector;
    colormap=:berlin,
    x_density=fluid.fvm_grid.n_cell_x-1,
    y_density=fluid.fvm_grid.n_cell_y-1,
    density=10,
    threshold_percentage=0.1,  # Use 10% of max Q-criterion as threshold
    min_threshold=nothing,     # Optional: manually override threshold
    max_threshold=nothing,     # Optional: manually override threshold
    smooth=false,
    smooth_sigma=1.5
)

    fvm_grid = fluid.fvm_grid

    # make meshgrid
    x_plot_cell_values = Vector(
        LinRange(0.0, fvm_grid.length_x, x_density)
    )
    y_plot_cell_values = Vector(
        LinRange(0.0, fvm_grid.length_y, y_density)
    )

    # calculate Q-criterion (with optional smoothing of velocity fields)
    q_criterion_grid = interpolate_fluid_q_criterion_grid(
        fvm_grid, v_fluid, x_plot_cell_values, y_plot_cell_values;
        smooth=smooth, smooth_sigma=smooth_sigma
    )

    # Compute percentage-based threshold if not manually specified
    if isnothing(min_threshold) || isnothing(max_threshold)
        max_q = maximum(abs.(q_criterion_grid))
        threshold = threshold_percentage * max_q
        min_threshold = isnothing(min_threshold) ? -threshold : min_threshold
        max_threshold = isnothing(max_threshold) ? threshold : max_threshold
    end

    # plot Q-criterion contours onto the provided axis
    contourf!(aquarium_ax,
        x_plot_cell_values,
        y_plot_cell_values,
        q_criterion_grid',
        levels=range(min_threshold, max_threshold, density),
        colormap=colormap,
        extendlow = :auto,
        extendhigh = :auto,
    )

    # Plot bluff body if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body)
        plot_solid_system!(aquarium_ax, bluff_body, x_bluff_body)
    end

    # Plot swimmer if present
    if !isnothing(swimmer) && !isempty(x_swimmer)
        plot_solid_system!(aquarium_ax, swimmer, x_swimmer)
    end

    return q_criterion_grid

end


function plot_streamlines!(aquarium_fig::Figure,
    aquarium_ax::Axis,
    fluid::Fluid,
    bluff_body::Union{Nothing, <:SolidSystem},
    swimmer::Union{Nothing, <:SolidSystem},
    v_fluid::AbstractVector,
    x_bluff_body::AbstractVector,
    x_swimmer::AbstractVector;
    colormap=:imola,
    x_density=fluid.fvm_grid.n_cell_x-1,
    y_density=fluid.fvm_grid.n_cell_y-1,
    density=10.0,
    linewidth=1.5,
    arrowsize=5
)

    fvm_grid = fluid.fvm_grid

    # create coordinates for plotting
    x_plot_cell_values = Vector(
        LinRange(fvm_grid.h_x, fvm_grid.length_x-fvm_grid.h_x, x_density)
    )
    y_plot_cell_values = Vector(
        LinRange(fvm_grid.h_y, fvm_grid.length_y-fvm_grid.h_y, y_density)
    )

    # create function that interpolates fluid velocity at given coordinates
    v_fluid_x_interpolant, v_fluid_y_interpolant = create_fluid_grid_interpolant(fvm_grid, v_fluid)

    function interpolate_fluid_velocity(x,y)
        Point2f(v_fluid_x_interpolant(x,y), v_fluid_y_interpolant(x,y))
    end

    # plot streamlines onto the provided axis
    streamplot!(aquarium_ax, interpolate_fluid_velocity,
        x_plot_cell_values, y_plot_cell_values,
        gridsize=(density, density, density),
        arrow_size=arrowsize, colormap=colormap,
        linewidth=linewidth
    )

    # Plot bluff body if present
    if !isnothing(bluff_body) && !isempty(x_bluff_body)
        plot_solid_system!(aquarium_ax, bluff_body, x_bluff_body)
    end

    # Plot swimmer if present
    if !isnothing(swimmer) && !isempty(x_swimmer)
        plot_solid_system!(aquarium_ax, swimmer, x_swimmer)
    end

    return nothing

end

function plot_solid_systems!(aquarium_fig::Figure,
    aquarium_ax::Axis,
    systems::AbstractVector{<:SolidSystem},
    xr_list::AbstractVector{<:AbstractVector},
)

    # plot bodies onto the provided axis
    if !isempty(systems)
        for i in eachindex(systems)
            systems_i = systems[i]
            xr_i = xr_list[i]
            plot_solid_system!(aquarium_ax, systems_i, xr_i)
        end
    end

    return nothing

end