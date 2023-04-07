"""
Extra utility functions for CFD and FSI
"""

function animate_velocityfield(model::FSIModel, boundary::ImmersedBoundary,
    T_hist::AbstractVector, x_rollout::VecOrMat{<:AbstractVector},
    u_hist::VecOrMat{<:AbstractVector}, anime_file::String;
    background_color=:transparent, obj_color=:black, arrowcolor=:blue,
    x_lim=(0, nothing), y_lim=(0, nothing), fontsize=18, framerate=60,
    timescale=1.0, og_scale::Bool=true, display_live=true, lengthscale=0.5,
    arrowsize=10, normalize_arrow=false, density=1, resolution=(800, 600),
    kwargs...)

    # find frame indices
    total_frames = (T_hist[end]*timescale*framerate)
    video_dt = T_hist[end]/total_frames
    N = length(T_hist)

    # define dt
    if model.normalize
        dt = model.dt * (model.ref_L / model.ref_u) 
    else
        dt = model.dt 
    end

    if boundary.normalize && og_scale
        boundary = Aquarium.unnormalize(boundary, model.ref_L)
        x_rollout = map(x -> Aquarium.unnormalize(boundary, x, 
        model.ref_L, model.ref_u), x_rollout)
    end

    # determine what knot points to use
    if video_dt <= dt
        frames = 1:N
    else
        factor = Int(floor(video_dt/dt))
        frames = 1:factor:N
    end

    total_frames = length(frames)
    p = Progress(total_frames, 1, "Creating animation...")

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    x_u_new = LinRange(x_u[1], x_u[end], round(Int, density*length(x_u)))
    y_u_new = LinRange(y_u[1], y_u[end], round(Int, density*length(y_u)))

    function velocityfield_u(r)

        u = u_hist[r]
        u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

        u_grid_new = zeros(length(y_u_new), length(x_u_new))
        v_grid_new = zeros(length(y_u_new), length(x_u_new))

        f_u=scale(interpolate(u_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)
        f_v=scale(interpolate(v_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)

        for i in eachindex(x_u_new)
            for j in eachindex(y_u_new)
        
                u_grid_new[j, i] = f_u(x_u_new[i], y_u_new[j])
                v_grid_new[j, i] = f_v(x_u_new[i], y_u_new[j])
                    
            end
        end

        if normalize_arrow

            mag = sqrt.(u_grid_new.^2 .+ v_grid_new.^2)
            u_grid_new ./= mag
    
        end

        return u_grid_new

    end
    function velocityfield_v(r)

        u = u_hist[r]
        u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

        u_grid_new = zeros(length(y_u_new), length(x_u_new))
        v_grid_new = zeros(length(y_u_new), length(x_u_new))

        f_u=scale(interpolate(u_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)
        f_v=scale(interpolate(v_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)

        for i in eachindex(x_u_new)
            for j in eachindex(y_u_new)
        
                u_grid_new[j, i] = f_u(x_u_new[i], y_u_new[j])
                v_grid_new[j, i] = f_v(x_u_new[i], y_u_new[j])
                    
            end
        end

        if normalize_arrow

            mag = sqrt.(u_grid_new.^2 .+ v_grid_new.^2)
            v_grid_new ./= mag
    
        end

        return v_grid_new

    end

    # Declare a Makie Observable for storing velocity arrays
    knot_point = Observable(1)
    vor_obs_u = @lift(velocityfield_u($knot_point))
    vor_obs_v = @lift(velocityfield_v($knot_point))

    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = background_color,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    arrows!(ax, x_u_new, y_u_new, vor_obs_u, vor_obs_v, arrowsize=arrowsize,
        lengthscale=lengthscale, arrowcolor=arrowcolor, linecolor=arrowcolor
    )

    if boundary.normalize && og_scale
        boundary = Aquarium.unnormalize(boundary, model.ref_L)
        x = Aquarium.unnormalize(boundary, x, 
        model.ref_L, model.ref_u)
    end
    boundary_plot = plot_boundary!(boundary, x_rollout[1]; color=obj_color, kwargs...)

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    if display_live
        display(fig)
    end

    record(fig, anime_file, frames, framerate=framerate) do i

        knot_point[] = i

        if length(boundary_plot) > 1
            for subplot in boundary_plot
                delete!(ax, subplot)
            end
        else
            delete!(ax, boundary_plot)
        end
        boundary_plot = plot_boundary!(boundary, x_rollout[i]; color=obj_color, kwargs...)

        if display_live
            display(fig)
        end

        next!(p)

    end
    
end

function animate_vorticity(model::FSIModel, boundary::ImmersedBoundary,
    T_hist::AbstractVector, x_rollout::VecOrMat{<:AbstractVector},
    u_hist::VecOrMat{<:AbstractVector}, anime_file::String;
    levels=50.0, level_perc=0.3, colormap=:bwr, obj_color=:black,
    x_lim=(0, nothing), y_lim=(0, nothing), fontsize=18, framerate=60,
    timescale=1.0, og_scale::Bool=true, resolution=(800, 600),
    display_live=true, kwargs...)

    # find frame indices
    total_frames = (T_hist[end]*timescale*framerate)
    video_dt = T_hist[end]/total_frames
    N = length(T_hist)

    # define dt
    if model.normalize
        dt = model.dt * (model.ref_L / model.ref_u) 
    else
        dt = model.dt 
    end

    if boundary.normalize && og_scale
        boundary = Aquarium.unnormalize(boundary, model.ref_L)
        x_rollout = map(x -> Aquarium.unnormalize(boundary, x, 
        model.ref_L, model.ref_u), x_rollout)
    end

    # determine what knot points to use
    if video_dt <= dt
        frames = 1:N
    else
        factor = Int(floor(video_dt/dt))
        frames = 1:factor:N
    end

    total_frames = length(frames)
    p = Progress(total_frames, 1, "Creating animation...")

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    function vorticity(r)

        u = u_hist[r]
        u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

        f_u=scale(interpolate(u_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)
        f_v=scale(interpolate(v_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)

        vor_grid = deepcopy(u_grid)

        for i in 1:size(u_grid)[2]
            for j in 1:size(u_grid)[1]
        
                dudy = Interpolations.gradient(f_u, x_u[i], y_u[j])[2]
                dvdx = Interpolations.gradient(f_v, x_u[i], y_u[j])[1]
                
                vor_grid[j, i] = dvdx-dudy
        
            end
        end

        return vor_grid'

    end

    # Declare a Makie Observable for storing velocity arrays
    knot_point = Observable(1)
    vor_obs = @lift(vorticity($knot_point))

    max_mag = maximum(abs.(vorticity(length(x_rollout))))
    min_level = -level_perc*max_mag
    max_level = level_perc*max_mag

    # plot vorticity contours
    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = Makie.ColorSchemes.eval(colormap)[end÷2],
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    contourf!(ax, x_u, y_u, vor_obs, levels=range(min_level, max_level, levels),
        colormap=colormap, extendlow = :auto, extendhigh = :auto
    )

    if boundary.normalize && og_scale
        boundary = Aquarium.unnormalize(boundary, model.ref_L)
        x = Aquarium.unnormalize(boundary, x, 
        model.ref_L, model.ref_u)
    end
    boundary_plot = plot_boundary!(boundary, x_rollout[1]; color=obj_color, kwargs...)

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    if display_live
        display(fig)
    end

    record(fig, anime_file, frames, framerate=framerate) do i

        knot_point[] = i

        if length(boundary_plot) > 1
            for subplot in boundary_plot
                delete!(ax, subplot)
            end
        else
            delete!(ax, boundary_plot)
        end
        boundary_plot = plot_boundary!(boundary, x_rollout[i]; color=obj_color, kwargs...)

        if display_live
            display(fig)
        end

        next!(p)

    end
    
end

function animate_vorticity(model::CFDModel, T_hist::AbstractVector,
    u_hist::VecOrMat{<:AbstractVector}, anime_file::String;
    levels=50.0, colormap=:bwr, level_perc=0.3,
    x_lim=(0, nothing), y_lim=(0, nothing), fontsize=18,
    framerate=60, timescale=1.0, og_scale::Bool=true,
    resolution=(800, 600), display_live=true)

    # find frame indices
    total_frames = (T_hist[end]*timescale*framerate)
    video_dt = T_hist[end]/total_frames
    N = length(T_hist)

    # check if unnormalized
    if model.normalize && og_scale
        dt = model.dt * (model.ref_L / model.ref_u) 
        model = unnormalize(model)
    else
        dt = model.dt 
    end    

    # determine what knot points to use
    if video_dt <= dt
        frames = 1:N
    else
        factor = Int(floor(video_dt/dt))
        frames = 1:factor:N
    end

    total_frames = length(frames)
    p = Progress(total_frames, 1, "Creating animation...")

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    # make vorticity interpolant function
    function vorticity(r)

        u = u_hist[r]

        u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

        f_u=scale(interpolate(u_grid',BSpline(Cubic(Reflect(OnGrid())))), x_u, y_u)
        f_v=scale(interpolate(v_grid',BSpline(Cubic(Reflect(OnGrid())))), x_u, y_u)

        vor_grid = deepcopy(u_grid)

        for i in 1:size(u_grid)[1]
            for j in 1:size(u_grid)[2]

                dudy = Interpolations.gradient(f_u, x_u[i], y_u[j])[2]
                dvdx = Interpolations.gradient(f_v, x_u[i], y_u[j])[1]
                
                vor_grid[i, j] = dvdx-dudy

            end
        end

        return vor_grid

    end

    # Declare a Makie Observable for storing velocity arrays
    knot_point = Observable(1)
    vor_obs = @lift(vorticity($knot_point))

    max_mag = maximum(abs.(vorticity(length(x_rollout))))
    min_level = -level_perc*max_mag
    max_level = level_perc*max_mag

    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = Makie.ColorSchemes.eval(colormap)[end÷2],
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    contourf!(ax, x_u, y_u, vor_obs,
        levels=range(min_level, max_level, levels),
        colormap=colormap, extendlow = :auto, extendhigh = :auto
    )

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    if display_live
        display(fig)
    end

    record(fig, anime_file, frames, framerate=framerate) do i

        knot_point[] = i

        if display_live
            display(fig)
        end

        next!(p)

    end
    
end

function animate_streamlines(model::FSIModel, boundary::ImmersedBoundary,
    T_hist::AbstractVector, x_rollout::VecOrMat{<:AbstractVector},
    u_hist::VecOrMat{<:AbstractVector}, anime_file::String;
    density=50.0, linewidth=1.5, colormap=:bwr, obj_color=:black, background_color=:transparent,
    fontsize=18, x_lim=(0, nothing), y_lim=(0, nothing), framerate=60, timescale=1.0,
    og_scale::Bool=true, display_live=true, resolution=(800, 600), kwargs...)

    # find frame indices
    total_frames = (T_hist[end]*timescale*framerate)
    video_dt = T_hist[end]/total_frames
    N = length(T_hist)

    # define dt
    if model.normalize
        dt = model.dt * (model.ref_L / model.ref_u) 
    else
        dt = model.dt 
    end

    if boundary.normalize && og_scale
        boundary = Aquarium.unnormalize(boundary, model.ref_L)
        x_rollout = map(x -> Aquarium.unnormalize(boundary, x, 
        model.ref_L, model.ref_u), x_rollout)
    end

    # determine what knot points to use
    if video_dt <= dt
        frames = 1:N
    else
        factor = Int(floor(video_dt/dt))
        frames = 1:factor:N
    end

    total_frames = length(frames)
    p = Progress(total_frames, 1, "Creating animation...")

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    function streamlines(i)

        u = u_hist[i]
    
        return u_interpolate(fluidgrid(model, u; og_scale=og_scale), [x_u, y_u])
    
    end
    
    # Declare a Makie Observable for storing velocity arrays
    knot_point = Observable(1)
    uvn = @lift(streamlines($knot_point))

    # plot streamlines
    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
            backgroundcolor = background_color,
            xgridcolor = :transparent,
            ygridcolor = :transparent,
        )
    )
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    streamplot!(ax, uvn, x_u, y_u, color=:blue,
        arrow_size=0.02, gridsize=(density, density, density),
        colormap=colormap, linewidth=linewidth
    )

    if boundary.normalize && og_scale
        boundary = Aquarium.unnormalize(boundary, model.ref_L)
        x_rollout = map(x -> Aquarium.unnormalize(boundary, x, 
        model.ref_L, model.ref_u), x_rollout)
    end

    boundary_plot = plot_boundary!(boundary, x_rollout[1]; color=obj_color, kwargs...)

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    if display_live
        display(fig)
    end

    record(fig, anime_file, frames, framerate=framerate) do i
        
        knot_point[] = i

        if length(boundary_plot) > 1
            for subplot in boundary_plot
                delete!(ax, subplot)
            end
        else
            delete!(ax, boundary_plot)
        end
        boundary_plot = plot_boundary!(boundary, x_rollout[i]; color=obj_color, kwargs...)

        if display_live
            display(fig)
        end

        next!(p)

    end
    
end

function animate_streamlines(model::CFDModel, T_hist::AbstractVector,
    u_hist::VecOrMat{<:AbstractVector}, anime_file::String;
    density=50.0, linewidth=1.5, colormap=:bwr, background_color=:transparent,
    fontsize=18, x_lim=(0, nothing), y_lim=(0, nothing), framerate=60, 
    timescale=1.0, og_scale::Bool=true, resolution=(800, 600), display_live=true)

    # find frame indices
    total_frames = (T_hist[end]*timescale*framerate)
    video_dt = T_hist[end]/total_frames
    N = length(T_hist)

    # define dt
    if model.normalize
        dt = model.dt * (model.ref_L / model.ref_u) 
    else
        dt = model.dt 
    end

    # determine what knot points to use
    if video_dt <= dt
        frames = 1:N
    else
        factor = Int(floor(video_dt/dt))
        frames = 1:factor:N
    end

    total_frames = length(frames)
    p = Progress(total_frames, 1, "Creating animation...")

    # make meshgrid
    function streamlines(i)

        u = u_hist[i]
    
        return u_interpolate(fluidgrid(model, u; og_scale=og_scale), [x_u, y_u])
    
    end
    
    # Declare a Makie Observable for storing velocity arrays
    knot_point = Observable(1)
    uvn = @lift(streamlines($knot_point))

    # plot streamlines
    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = background_color,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    streamplot!(ax, uvn, x_u, y_u, arrow_size=0.02, gridsize=(density, density, density),
        axis = (xlabel = "x", ylabel = "y"), colormap=colormap, linewidth=linewidth
    )

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    if display_live
        display(fig)
    end

    record(fig, anime_file, frames, framerate=framerate) do i

        knot_point[] = i

        if display_live
            display(fig)
        end

        next!(p)

    end
    
end

function plot_vorticity(model::FSIModel, boundary::ImmersedBoundary,
    x::AbstractVector, u::AbstractVector; levels=100, colormap=:bwr,
    level_perc=1.0, obj_color=:black, fontsize=18, x_lim=(0, nothing),
    y_lim=(0, nothing), resolution=(800, 600), og_scale::Bool=true,
    kwargs...)

    # check if averaged over p cv
    if length(u) != 2 * model.ne_x * model.ne_y
        u = average(model, u)
    end

    # check if unnormalized
    if model.normalize && og_scale
        model = unnormalize(model)
        u = u .* model.ref_u
    end

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    # average u over p cv
    u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

    f_u=scale(interpolate(u_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)
    f_v=scale(interpolate(v_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)

    vor_grid = deepcopy(u_grid)

    for i in 1:size(u_grid)[2]
        for j in 1:size(u_grid)[1]
    
            dudy = Interpolations.gradient(f_u, x_u[i], y_u[j])[2]
            dvdx = Interpolations.gradient(f_v, x_u[i], y_u[j])[1]
            
            vor_grid[j, i] = dvdx-dudy
    
        end
    end

    max_mag = maximum(abs.(vor_grid))
    min_level = -level_perc*max_mag
    max_level = level_perc*max_mag

    # plot vorticity contours
    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = Makie.ColorSchemes.eval(colormap)[end÷2],
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    contourf!(ax, x_u, y_u, vor_grid', levels=range(min_level, max_level, levels),
        colormap=colormap, extendlow = :auto, extendhigh = :auto
    )

    if boundary.normalize && og_scale
        boundary = Aquarium.unnormalize(boundary, model.ref_L)
        x = Aquarium.unnormalize(boundary, x, 
        model.ref_L, model.ref_u)
    end
    plot_boundary!(boundary, x; color=obj_color, kwargs...)

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    return fig
    
end

function plot_vorticity(model::CFDModel, u::AbstractVector;
    levels=100, colormap=:bwr, level_perc=1.0, fontsize=18,
    x_lim=(0, nothing), y_lim=(0, nothing),
    resolution=(800, 600), og_scale::Bool=true)

    # check if averaged over p cv
    if length(u) != 2 * model.ne_x * model.ne_y
        u = average(model, u)
    end

    # check if unnormalized
    if model.normalize && og_scale
        model = unnormalize(model)
        u = u .* model.ref_u
    end

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    # average u over p cv
    u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

    f_u=scale(interpolate(u_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)
    f_v=scale(interpolate(v_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)

    vor_grid = deepcopy(u_grid)

    for i in 1:size(u_grid)[1]
        for j in 1:size(u_grid)[2]

            dudy = Interpolations.gradient(f_u, x_u[i], y_u[j])[2]
            dvdx = Interpolations.gradient(f_v, x_u[i], y_u[j])[1]
            
            vor_grid[i, j] = dvdx-dudy

        end
    end

    max_mag = maximum(abs.(vor_grid))
    min_level = -level_perc*max_mag
    max_level = level_perc*max_mag

    # plot vorticity contours
    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = background_color,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    contourf!(ax, x_u, y_u, vor_grid,
        levels=range(min_level, max_level, levels),
        colormap=colormap, color=contour_color
    )

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    return fig
    
end

function plot_velocityfield(model::FSIModel, boundary::ImmersedBoundary,
    x::AbstractVector, u::AbstractVector; arrowcolor="strength",
    background_color=:transparent, obj_color=:black, fontsize=18, x_lim=(0, nothing),
    y_lim=(0, nothing), og_scale::Bool=true, lengthscale=0.5, arrowsize=10,
    resolution=(800, 600), normalize_arrow=false, density=1, kwargs...)

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    x_u_new = LinRange(x_u[1], x_u[end], round(Int, density*length(x_u)))
    y_u_new = LinRange(y_u[1], y_u[end], round(Int, density*length(y_u)))

    # average u over p cv
    u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

    f_u=scale(interpolate(u_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)
    f_v=scale(interpolate(v_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)

    u_grid_new = zeros(length(y_u_new), length(x_u_new))
    v_grid_new = zeros(length(y_u_new), length(x_u_new))

    for i in eachindex(x_u_new)
        for j in eachindex(y_u_new)
    
            u_grid_new[j, i] = f_u(x_u_new[i], y_u_new[j])
            v_grid_new[j, i] = f_v(x_u_new[i], y_u_new[j])
                
        end
    end

    if arrowcolor == "strength"

        arrowcolor = vec(sqrt.(u_grid_new.^2 .+ v_grid_new.^2))

    end

    if normalize_arrow

        mag = sqrt.(u_grid_new.^2 .+ v_grid_new.^2)
        u_grid_new ./= mag
        v_grid_new ./= mag

    end

    # plot streamlines
    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = background_color,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    arrows!(ax, x_u_new, y_u_new, u_grid_new, v_grid_new, arrowsize=arrowsize,
        lengthscale=lengthscale, arrowcolor=arrowcolor, linecolor=arrowcolor
    )

    if model.normalize && og_scale
        boundary = Aquarium.unnormalize(boundary, model.ref_L)
        x = Aquarium.unnormalize(boundary, x, 
        model.ref_L, model.ref_u)
    end
    plot_boundary!(boundary, x; color=obj_color, kwargs...)

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    return fig
    
end

function plot_velocityfield(model::CFDModel, u::AbstractVector;
    arrowcolor="strength", background_color=:transparent,
    fontsize=18, x_lim=(0, nothing), y_lim=(0, nothing),
    og_scale::Bool=true,lengthscale=0.5, arrowsize=10,
    resolution=(800, 600), density=1)

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    x_u_new = LinRange(x_u[1], x_u[end], round(Int, density*length(x_u)))
    y_u_new = LinRange(y_u[1], y_u[end], round(Int, density*length(y_u)))

    # average u over p cv
    u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

    f_u=scale(interpolate(u_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)
    f_v=scale(interpolate(v_grid',BSpline(Quadratic(Reflect(OnGrid())))), x_u, y_u)

    u_grid_new = zeros(length(y_u_new), length(x_u_new))
    v_grid_new = zeros(length(y_u_new), length(x_u_new))

    for i in eachindex(x_u_new)
        for j in eachindex(y_u_new)
    
            u_grid_new[j, i] = f_u(x_u_new[i], y_u_new[j])
            v_grid_new[j, i] = f_v(x_u_new[i], y_u_new[j])
                
        end
    end

    if arrowcolor == "strength"

        arrowcolor = vec(sqrt.(u_grid_new.^2 .+ v_grid_new.^2))

    end

    # plot streamlines
    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = background_color,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    arrows!(ax, x_u_new, y_u_new, u_grid_new, v_grid_new, arrowsize=arrowsize,
        lengthscale=lengthscale, arrowcolor=arrowcolor, linecolor=arrowcolor
    )

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    return fig
    
end

function plot_streamlines(model::FSIModel, boundary::ImmersedBoundary,
    x::AbstractVector, u::AbstractVector; density=50.0, linewidth=1.5,
    colormap=:bwr, obj_color=:black, background_color=:transparent,
    fontsize=18, x_lim=(0, nothing), y_lim=(0, nothing),
    resolution=(800, 600), og_scale::Bool=true, kwargs...)

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    # average u over p cv
    u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

    u_int(uv) = u_interpolate(uv, [x_u, y_u])

    ## Declare a Makie Observable for storing velocity arrays
    uvn=Observable([u_grid, v_grid])

    # plot streamlines
    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = background_color,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    streamplot!(ax, lift(a -> u_int(to_value(a)),uvn),
        x_u, y_u, color=:blue, arrow_size=0.02,
        gridsize=(density, density, density),
        colormap=colormap, linewidth=linewidth
    )

    if model.normalize && og_scale
        boundary = Aquarium.unnormalize(boundary, model.ref_L)
        x = Aquarium.unnormalize(boundary, x, 
        model.ref_L, model.ref_u)
    end
    plot_boundary!(boundary, x; color=obj_color, kwargs...)

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    return fig
    
end

function plot_streamlines(model::CFDModel, u::AbstractVector;
    density=50.0, linewidth=1.5, colormap=:bwr,
    background_color=:transparent, fontsize=18,
    x_lim=(0, nothing), y_lim=(0, nothing),
    resolution=(800, 600), og_scale::Bool=true)

    # make meshgrid
    x_u, y_u = fluidcoord(model; og_scale=og_scale)

    # average u over p cv
    u_grid, v_grid = fluidgrid(model, u; og_scale=og_scale)

    u_int(uv) = u_interpolate(uv, [x_u, y_u])

    ## Declare a Makie Observable for storing velocity arrays
    uvn=Observable([u_grid, v_grid])

    # plot streamlines
    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = background_color,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    streamplot!(ax, lift(a -> u_int(to_value(a)),uvn),
        x_u, y_u, color=:blue, arrow_size=0.02,
        gridsize=(density, density, density),
        colormap=colormap, linewidth=linewidth
    )

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    return fig
    
end

function u_interpolate(uv, xs)
    
    f_u=scale(interpolate(uv[1]',BSpline(Quadratic(Reflect(OnCell())))), xs[1], xs[2])
    f_v=scale(interpolate(uv[2]',BSpline(Quadratic(Reflect(OnCell())))), xs[1], xs[2])

    function velocity(x,y)
        Point2f0(f_u(x,y),f_v(x,y))
    end

    return velocity
end

function fluidgrid(model::Union{CFDModel, FSIModel}, u::AbstractVector;
    og_scale::Bool=true)

    # check if averaged over p cv
    if length(u) != 2 * model.ne_x * model.ne_y
        u = average(model, u)
    end

    # check if unnormalized
    if model.normalize && og_scale
        model = unnormalize(model)
        u = u .* model.ref_u
    end

    # turn fluid vel vector into grid
    u_grid = reshape(u[1:(model.ne_y*model.ne_x)], model.ne_y, model.ne_x)
    v_grid = reshape(u[(model.ne_y*model.ne_x)+1:end], model.ne_y, model.ne_x)

    return u_grid, v_grid
end

function fluidcoord(model::Union{CFDModel, FSIModel}; og_scale::Bool=true)

    if model.normalize && og_scale
        model = unnormalize(model)
    end

    x_coord = LinRange(model.h_x/2, model.L_x-model.h_x/2, model.ne_x)
    y_coord = LinRange(model.h_y/2, model.L_y-model.h_y/2, model.ne_y)

    return x_coord, y_coord
end

function meshgrid(model::Union{CFDModel, FSIModel}; og_scale::Bool=true)

    if model.normalize && og_scale
        model = unnormalize(model)
    end

    # make meshgrid
    x_grid = (model.x_coord_p)' .* ones(length(model.y_coord_p))
    y_grid = ones(length(model.x_coord_p))' .* model.y_coord_p

    return x_grid, y_grid
end

function average(model::Union{CFDModel, FSIModel}, u::AbstractVector)

    u_avg = Vector(model.FVM_ops.cv_avg[1] * u + model.FVM_ops.cv_avg[2])

    return u_avg

end

function stack_states(x_rollout::VecOrMat{<:VecOrMat{<:AbstractVector}})

    N = length(x_rollout[1])
    
    return [[x[k] for x in x_rollout] for k in 1:N]

end