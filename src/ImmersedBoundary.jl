"""
Immersed Boundary Model
"""

abstract type ImmersedBoundary end

function simulate(boundary::ImmersedBoundary, x0::AbstractVector,
    f_b_rollout::VecOrMat{<:AbstractVector}, fext::AbstractVector=[0, 0];
    t0=0.0, tf=5.0, dt=0.01, max_iter=10, tol=1e-6, verbose=false)
    
    # build time history vectors
    N = Int((tf-t0)÷dt + 1)
    f_b_hist = [zeros(length(f_b_rollout[1])) for _ in 1:length(f_b_rollout)+1]
    f_b_hist[2:end] = copy(f_b_rollout)
    x_hist = [copy(x0) for _ in 1:N]
    t_hist = t0:dt:tf

    @showprogress "Simulating..." for ind in 2:N

        t = t_hist[ind]

        if verbose
            println("\n")
            @show t
        end

        if ind > length(f_b_rollout)
            
            fn_b = f_b_rollout[end]
            fk_b = f_b_rollout[end]
            push!(f_b_hist, f_b_rollout[end])
        
        elseif ind >= 3
            
            fn_b = f_b_rollout[ind-1]
            fk_b = f_b_rollout[ind-2]
            
        end

        # integrate over dynamics
        discrete_dynamics!(boundary, x_hist[ind], f_b_hist[ind], x_hist[ind-1], fext;
            dt=dt, max_iter=max_iter, tol=tol, verbose=verbose)

    end

    return t_hist, x_hist, f_b_hist[1:length(t_hist)]
end

function animate_boundary(model::ImmersedBoundary, anime_file::String, T::AbstractVector, X::VecOrMat{<:AbstractVector};
    x_lim=(0.0, nothing), y_lim=(0.0, nothing), lengthscale=0.5, framerate=60, timescale=1.0, show_vel=true,
    obj_color=:black, background_color=:white, fontsize=18, resolution=(800, 600), kwargs...)

    dt = T[2]-T[1]
    valid_ind = 1:Int(minimum([length(T), length(X)]))
    T = T[valid_ind]
    X = X[valid_ind]

    total_frames = (T[end]*timescale*framerate)
    video_dt = T[end]/total_frames
    N = length(T)

    if video_dt <= dt
        frames = 1:N
    else
        factor = Int(floor(video_dt/dt))
        frames = 1:factor:N
    end

    X = X[frames]

    x = X[1]
    state_b = Aquarium.boundary_state(model, x)
    x_b = state_b[1:end÷2]
    u_b = state_b[end÷2+1:end]

    x = x_b[1:model.nodes]
    y = x_b[model.nodes+1:end]
    u = u_b[1:model.nodes]
    v = u_b[model.nodes+1:end]

    total_frames = length(X)
    p = Progress(total_frames, 1, "Creating animation...")

    set_theme!(font = "Times New Roman", fontsize=fontsize, Axis = (
        backgroundcolor = background_color,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
    ))
    fig = Figure(resolution = resolution)
    ax = Axis(fig[1,1], xlabel = "x", ylabel = "y")
    Aquarium.plot_boundary!(model, X[1]; color=obj_color, kwargs...)

    if show_vel
        arrows!(x, y, u, v, color=:red, lengthscale=lengthscale)
    end

    ax.aspect = DataAspect()
    xlims!(ax, x_lim)
    ylims!(ax, y_lim)

    record(fig, anime_file, 1:total_frames, framerate=framerate) do i

        empty!(ax)
        x = X[i]

        state_b = Aquarium.boundary_state(model, x)
        x_b = state_b[1:end÷2]
        u_b = state_b[end÷2+1:end]
        
        x = x_b[1:model.nodes]
        y = x_b[model.nodes+1:end]
        u = u_b[1:model.nodes]
        v = u_b[model.nodes+1:end]

        Aquarium.plot_boundary!(model, X[i]; color=obj_color, kwargs...)
        
        if show_vel
            arrows!(x, y, u, v, color=:red, lengthscale=lengthscale)
        end

        ax.aspect = DataAspect()
        xlims!(ax, x_lim)
        ylims!(ax, y_lim)

        next!(p)

    end

end